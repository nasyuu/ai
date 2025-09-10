from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import grpc

# 由你的 proto 生成的模块；保持与原工程一致的命名
from clients.img_infer.grpc import grpc_module_pb2 as pb2
from clients.img_infer.grpc import grpc_module_pb2_grpc as pb2_grpc
from utils.exception import (
    InferenceError,
    IOErrorEx,
    SerializationError,
    ensure,
    from_grpc_rpc_error,
)
from utils.logger import get_logger, record_fail

log = get_logger("grpc.standard")


@dataclass
class GrpcConfig:
    server: str  # "ip:port"
    stream_name: str  # 流名称
    task_id: str = ""  # 任务ID（通过 metadata 传递）
    max_workers: int = 1  # 1=串行；>1=并行
    deadline_sec: int = 60  # 每次调用的超时
    retry_max: int = 3  # 最大重试次数
    retry_base_sec: float = 2.0  # 线性退避基准（2,4,6...）
    max_message_bytes: int = 100 * 1024 * 1024  # 100MB

    def validate(self):
        ensure(
            bool(self.server), message="grpc server 不能为空", code="config.grpc.server"
        )
        ensure(
            bool(self.stream_name),
            message="stream_name 不能为空",
            code="config.grpc.stream",
        )
        ensure(
            self.max_workers >= 1,
            message="max_workers 必须 ≥1",
            code="config.grpc.workers",
        )


def create_stub(
    cfg: GrpcConfig,
) -> Tuple[grpc.Channel, pb2_grpc.StandardGrpcServiceStub]:
    """
    创建 gRPC 通道与 Stub
    """
    cfg.validate()
    channel = grpc.insecure_channel(
        cfg.server,
        options=[
            ("grpc.max_receive_message_length", cfg.max_message_bytes),
            ("grpc.max_send_message_length", cfg.max_message_bytes),
        ],
    )
    stub = pb2_grpc.StandardGrpcServiceStub(channel)
    log.info("gRPC 通道已建立 → %s", cfg.server)
    return channel, stub


# 线程安全 fail.log
_log_lock = threading.Lock()


def _record_fail_line(msg: str):
    with _log_lock:
        record_fail(msg)


def infer_image(
    stub: pb2_grpc.StandardGrpcServiceStub,
    cfg: GrpcConfig,
    image_bytes: bytes,
    file_name: str,
) -> dict:
    """
    单张图片推理。返回 message 解析出的 dict。
    - gRPC 调用异常：转为 GRPCError / InferenceError
    - message 不是合法 JSON：抛 SerializationError
    - 业务 code!=1：记录 fail.log，抛 InferenceError(retriable=False)
    """
    request = pb2.Request(
        stream_name=cfg.stream_name,
        image_data=image_bytes,
    )

    metadata = []
    if cfg.task_id:
        metadata.append(("taskid", cfg.task_id))

    last_err: Optional[Exception] = None

    for i in range(cfg.retry_max):
        try:
            resp = stub.standardInfer(
                request, timeout=cfg.deadline_sec, metadata=metadata or None
            )

            # 解析 message 为 JSON
            try:
                msg_data = json.loads(resp.message)
            except Exception as e:
                raise SerializationError(
                    "响应 message 不是合法 JSON", cause=e, file=file_name
                )

            # 业务成功：code==1 → 返回 message（dict 或 JSON字符串解析后）
            if getattr(resp, "code", None) == 1:
                return msg_data

            # 业务失败：记录并抛出不可重试
            biz_msg = (
                msg_data.get("message", "未知错误")
                if isinstance(msg_data, dict)
                else "未知错误"
            )
            err_txt = f"[BUSINESS_FAIL] file={file_name} | code={getattr(resp, 'code', None)} | message={biz_msg}"
            _record_fail_line(err_txt)
            raise InferenceError(err_txt, retriable=False, file=file_name)

        except grpc.RpcError as e:
            ae = from_grpc_rpc_error(e)
            last_err = ae
            # 可重试则退避
            if ae.retriable and i < cfg.retry_max - 1:
                wait = cfg.retry_base_sec * (i + 1)
                log.warning(
                    "gRPC 调用失败（可重试）[%s]，%ss 后重试... | file=%s",
                    ae,
                    wait,
                    file_name,
                )
                time.sleep(wait)
                continue
            _record_fail_line(f"[RPC_ERROR] file={file_name} | {ae}")
            raise ae

        except SerializationError as e:
            # 序列化错误不可重试
            _record_fail_line(f"[SERDE_ERROR] file={file_name} | {e}")
            raise

        except InferenceError as e:
            # 业务错误是否可重试取决于上面设置
            last_err = e
            if e.retriable and i < cfg.retry_max - 1:
                wait = cfg.retry_base_sec * (i + 1)
                log.warning(
                    "推理异常（可重试），%ss 后重试... | file=%s | %s",
                    wait,
                    file_name,
                    e,
                )
                time.sleep(wait)
                continue
            _record_fail_line(f"[INFER_ERROR] file={file_name} | {e}")
            raise

        except Exception as e:
            last_err = InferenceError("未知推理异常", cause=e, file=file_name)
            if i < cfg.retry_max - 1:
                wait = cfg.retry_base_sec * (i + 1)
                log.warning(
                    "未知异常（可重试），%ss 后重试... | file=%s | %s",
                    wait,
                    file_name,
                    e,
                )
                time.sleep(wait)
                continue
            _record_fail_line(f"[UNKNOWN] file={file_name} | {e}")
            raise last_err

    # 正常不会到这里
    raise last_err or InferenceError("推理失败", file=file_name)


def _save_json(obj: dict, out_path: Path, file_name: str):
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _record_fail_line(f"[WRITE_ERROR] file={file_name} | {e}")
        raise IOErrorEx("保存结果失败", cause=e, path=str(out_path), file=file_name)


def infer_dir(
    cfg: GrpcConfig,
    img_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[int, int]:
    """
    遍历目录推理。返回 (success, failed)。
    - 串行：max_workers==1
    - 并行：ThreadPoolExecutor
    """
    cfg.validate()
    img_dir = str(img_dir)
    out_dir = str(out_dir)

    files = [
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    total = len(files)
    if total == 0:
        log.warning("未找到任何图片: %s", img_dir)
        return 0, 0

    channel, stub = create_stub(cfg)
    try:
        ok = fail = 0

        def run_one(fname: str) -> bool:
            fpath = os.path.join(img_dir, fname)
            try:
                with open(fpath, "rb") as fr:
                    image_bytes = fr.read()
                msg = infer_image(stub, cfg, image_bytes, fname)
                outp = Path(out_dir) / (Path(fname).stem + ".json")
                _save_json(msg, outp, fname)
                log.info("✅ 成功: %s", fname)
                return True
            except Exception as e:
                log.error("❌ 失败: %s | %s", fname, e)
                return False

        if cfg.max_workers == 1:
            log.info("使用串行模式处理，共 %d 张", total)
            for i, fn in enumerate(files, 1):
                ok += 1 if run_one(fn) else 0
                fail = i - ok
                if progress:
                    progress(i, total)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            log.info("使用并行模式处理（%d 个线程），共 %d 张", cfg.max_workers, total)
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
                futs = {ex.submit(run_one, fn): fn for fn in files}
                for i, fut in enumerate(as_completed(futs), 1):
                    res = fut.result()
                    ok += 1 if res else 0
                    fail = i - ok
                    if progress:
                        progress(i, total)

        log.info("处理完成：成功 %d，失败 %d", ok, fail)
        return ok, fail
    finally:
        channel.close()
        log.info("gRPC 通道已关闭")
