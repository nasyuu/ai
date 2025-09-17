import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import grpc

import interface.grpc.grpc_module_pb2 as pb2
import interface.grpc.grpc_module_pb2_grpc as pb2_grpc

# 配置
SERVER_ADDRESS = ""  # "ip:port"
IMAGE_DIR = ""  # 待推理图片目录
OUTPUT_DIR = ""  # *.json 输出目录
TASK_ID = ""  # 任务ID
STREAM_NAME = ""  # 流名称
MAX_WORKERS = 1  # 默认串行处理
DEFAULT_CALL_TIMEOUT = 60  # gRPC deadline
RETRY_MAX = 3  # 推理重试
RETRY_BASE_SEC = 2  # 线性退避

# 线程安全 fail.log
_log_lock = threading.Lock()


def record_fail(msg: str):
    """
    记录失败信息到 fail.log
    :param msg: 失败信息
    """
    ts = time.strftime("%F %T")
    with _log_lock:
        with open("fail.log", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")


def create_grpc_stub(address, max_msg=100 * 1024 * 1024):
    """
    创建 gRPC 通道和 Stub
    :param address: gRPC 服务器地址，格式为 "ip:port"
    :param max_msg: 最大消息长度，默认 100MB
    :return: (channel, stub)
    """
    channel = grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_receive_message_length", max_msg),
            ("grpc.max_send_message_length", max_msg),
        ],
    )
    stub = pb2_grpc.StandardGrpcServiceStub(channel)
    return channel, stub


def infer_grpc(
    stub, image_bytes, file_name, timeout=DEFAULT_CALL_TIMEOUT, retries=RETRY_MAX
):
    """
    调用 gRPC 推理接口
    :param stub: gRPC stub
    :param image_bytes: 图片二进制数据
    :param file_name: 文件名（用于日志）
    :param timeout: 超时时间（秒）
    :param retries: 重试次数
    :return: 推理结果
    """
    # 构建请求
    request = pb2.Request(
        stream_name=STREAM_NAME,
        image_data=image_bytes,
    )

    # 设置metadata
    metadata = [("taskid", TASK_ID)]  # task_id 通过 header 传递

    for i in range(retries):
        try:
            # gRPC 调用
            response = stub.standardInfer(request, timeout=timeout, metadata=metadata)
            # 验证响应内容是否为合法的JSON
            try:
                json.loads(response.message)
                return response
            except json.JSONDecodeError:
                raise RuntimeError("响应格式错误：无法解析JSON数据")

        except grpc.RpcError as e:
            # 处理gRPC调用相关的错误
            if i == retries - 1:
                record_fail(f"[ERROR] {file_name} gRPC网络调用失败 | {e}")
                raise
            logging.warning(f"[WARN] gRPC调用失败，准备重试: {e}")
        except Exception as e:
            # 处理其他错误
            if i == retries - 1:
                record_fail(f"[ERROR] {file_name} 处理失败 | {e}")
                raise
            logging.warning(f"[WARN] 处理失败，准备重试: {e}")
            wait = RETRY_BASE_SEC * (i + 1)
            logging.warning(
                f"[WARN] {file_name} 第 {i + 1}/{retries} 次失败，{wait}s 后重试 | {e}"
            )
            time.sleep(wait)


def process_single_image(stub, img_dir, img_file, out_dir):
    """
    处理单张图片
    :param stub: gRPC stub
    :param img_dir: 图片目录
    :param img_file: 图片文件名
    :param out_dir: 输出目录
    :return: 是否成功
    """
    img_path = os.path.join(img_dir, img_file)
    try:
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        # 获取gRPC响应
        response = infer_grpc(stub, image_bytes, img_file)

        # 确保输出目录存在
        os.makedirs(out_dir, exist_ok=True)

        # 解析响应数据
        resp_data = json.loads(response.message)
        out_path = os.path.join(out_dir, f"{os.path.splitext(img_file)[0]}.json")

        # 检查业务响应码
        if response.code == 1:
            # 业务成功，保存message中的内容
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(resp_data, f, ensure_ascii=False, indent=2)
            logging.info(f"✅ 成功处理图片 {img_file}")
            return True
        else:
            # 业务失败，仅记录错误日志，不写入文件
            error_msg = resp_data.get("message", "未知错误")
            record_fail(f"业务处理失败 {img_file} | 原因: {error_msg}")
            return False
        return True
    except Exception as e:
        record_fail(f"处理 {img_file} 失败: {e}")
        return False


def save_all_images_multithread(img_dir, out_dir, stub, max_workers=MAX_WORKERS):
    """
    多线程处理所有图片
    :param img_dir: 图片目录
    :param out_dir: 输出目录
    :param stub: gRPC stub
    :param max_workers: 最大线程数
    :return: None
    """
    os.makedirs(out_dir, exist_ok=True)
    img_files = [
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total = len(img_files)
    if total == 0:
        logging.warning("未找到任何图片")
        return

    logging.info(f"开始处理 {total} 张图片...")

    success = failed = 0
    processed = 0

    if max_workers == 1:
        # 串行处理：一个接一个执行，不使用线程池
        logging.info("使用串行模式处理...")
        for img_file in img_files:
            processed += 1
            try:
                if process_single_image(stub, img_dir, img_file, out_dir):
                    success += 1
                    logging.info(f"✅ 成功处理图片 {img_file}")
                else:
                    failed += 1
                    logging.warning(f"❌ 处理失败 {img_file}")
            except Exception as e:
                failed += 1
                record_fail(f"处理 {img_file} 失败: {e}")
                logging.error(f"❌ 处理异常 {img_file}: {e}")
            # 每张完成后记录进度
            logging.info(f"进度: {processed}/{total} | 成功 {success} | 失败 {failed}")
    else:
        # 并行处理：使用线程池
        logging.info(f"使用并行模式处理（{max_workers} 个工作线程）...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_image, stub, img_dir, img_file, out_dir
                ): img_file
                for img_file in img_files
            }

            for future in as_completed(futures):
                img_file = futures[future]
                processed += 1
                try:
                    if future.result():
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    record_fail(f"处理 {img_file} 失败: {e}")
                # 每张完成后记录进度
                logging.info(
                    f"进度: {processed}/{total} | 成功 {success} | 失败 {failed}"
                )

    logging.info(f"处理完成: 成功 {success} 张，失败 {failed} 张")


if __name__ == "__main__":
    chan, stub = create_grpc_stub(SERVER_ADDRESS)
    save_all_images_multithread(IMAGE_DIR, OUTPUT_DIR, stub)
    chan.close()
