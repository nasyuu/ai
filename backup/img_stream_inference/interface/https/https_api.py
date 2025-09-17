import base64
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional
from urllib.parse import urlparse

import cv2
import requests
import urllib3


def create_safe_tqdm():
    """在无控制台或缺少 tqdm 时提供安全替代"""
    try:
        from tqdm import tqdm

        if getattr(sys, "frozen", False):

            def _silent(iterable, **kw):
                kw.pop("desc", None)
                kw.pop("position", None)
                try:
                    return tqdm(iterable, disable=True, **kw)
                except Exception:
                    return iterable

            return _silent
        else:
            return tqdm
    except ImportError:
        return lambda iterable, **kw: iterable


safe_tqdm = create_safe_tqdm()


IMG_STREAM_URL = ""  # 推理接口
IMAGE_PATH = ""  # 待推理图片目录
RAW_JSON_DIR = ""  # 原始响应输出目录
STREAM_NAME = ""  # 流名称

AK, SK = "", ""
TENANT_ID = 1002
JPEG_QUALITY = 80
MAX_WORKERS = 1  # 默认串行处理
TOKEN_EXPIRE_SECONDS = 3600
REQUEST_RETRY_MAX = 3
RETRY_SLEEP_BASE = 2

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_log_lock = threading.Lock()
token_lock = threading.Lock()
token_info = {"token": None, "acquired_time": 0.0}

session = requests.Session()


def _normalize_url(url: str) -> str:
    """若缺少 http/https 前缀则默认补 https://"""
    if url and not urlparse(url).scheme:
        return f"https://{url.lstrip('/')}"
    return url


def get_token_url(api_url: str | None = None) -> str:
    """构造 token 接口 URL"""
    url = _normalize_url(api_url or IMG_STREAM_URL)
    parsed = urlparse(url)
    ip = parsed.hostname or ""
    port = parsed.port or (38443 if parsed.scheme == "https" else 80)
    return f"https://{ip}:{port}/gam/v1/auth/tokens"


def get_token_data(api_url: str | None = None, access_key: str | None = None) -> dict:
    """构造 token 请求体"""
    url = _normalize_url(api_url or IMG_STREAM_URL)
    ak = access_key or AK
    parsed = urlparse(url)
    ip = parsed.hostname or ""
    return {
        "identity": {"methods": ["basic"], "access": {"key": ak}},
        "clientIp": ip,
        "tenantId": TENANT_ID,
    }


def record_fail(msg: str):
    """记录失败信息到 fail.log"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _log_lock:
        with open("fail.log", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")


def frame2base64(frame, quality: int = JPEG_QUALITY) -> str:
    """OpenCV 帧 → Base64 JPEG 字符串"""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buf).decode("utf-8")


def request_with_retry(
    url: str,
    headers: Dict[str, str],
    data: str,
    retries: int = REQUEST_RETRY_MAX,
    timeout: int = 15,
    **kw,
):
    """POST 请求（带重试）"""
    for i in range(retries):
        try:
            return session.post(
                url, headers=headers, data=data, verify=False, timeout=timeout, **kw
            )
        except Exception as e:
            if i == retries - 1:
                record_fail(f"请求重试耗尽: {url} | {e}")
                raise
            time.sleep(RETRY_SLEEP_BASE * (i + 1))
            logging.warning(f"[WARN] 第{i + 1}/{retries}次请求失败，将重试: {e}")
    raise RuntimeError(f"请求 {url} 失败")


def _refresh_token(api_url=None, ak=None, sk=None) -> str:
    access_key = ak or AK
    secret_key = sk or SK
    auth = base64.b64encode(f"{access_key}:{secret_key}".encode()).decode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {auth}",
    }
    resp = request_with_retry(
        get_token_url(api_url), headers, json.dumps(get_token_data(api_url, access_key))
    )
    resp.raise_for_status()
    token = resp.headers.get("X-Subject-Token")
    if not token:
        raise RuntimeError("Token not found in response headers.")
    token_info.update({"token": token, "acquired_time": time.time()})
    logging.info("[INFO] Token 获取成功")
    return token


def get_token_if_expired(api_url=None, ak=None, sk=None) -> str:
    now = time.time()
    if (
        token_info["token"]
        and now - token_info["acquired_time"] <= TOKEN_EXPIRE_SECONDS
    ):
        return token_info["token"]
    with token_lock:
        now = time.time()
        if (
            token_info["token"]
            and now - token_info["acquired_time"] <= TOKEN_EXPIRE_SECONDS
        ):
            return token_info["token"]
        try:
            return _refresh_token(api_url, ak, sk)
        except Exception as e:
            token_info.update({"token": None, "acquired_time": 0})
            record_fail(f"Token fetch failed: {e}")
            raise


def infer_image_save_raw_json(
    img_path: str,
    out_dir: str = RAW_JSON_DIR,
    max_retries: int = 3,
    config: dict | None = None,
) -> None:
    """推理单张图片 → 保存原始 JSON"""
    if config:
        api_url = config.get("img_stream_url", IMG_STREAM_URL)
        stream_name = config.get("stream_name", STREAM_NAME)
        ak = config.get("access_key", AK)
        sk = config.get("secret_key", SK)
    else:
        api_url, stream_name, ak, sk = IMG_STREAM_URL, STREAM_NAME, AK, SK

    frame = cv2.imread(img_path)
    if frame is None:
        record_fail(f"无法读取图片: {img_path}")
        raise RuntimeError("imread fail")

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}.json")

    for attempt in range(1, max_retries + 1):
        try:
            token = get_token_if_expired(api_url, ak, sk)
            payload = {
                "stream_name": stream_name,
                "image_base64": f"data:image/jpeg;base64,{frame2base64(frame)}",
            }
            headers = {"Content-Type": "application/json", "Authorization": token}
            resp = request_with_retry(api_url, headers, json.dumps(payload))
            if resp.status_code == 200:
                resp_data = resp.json()

                # 检查业务响应码
                if resp_data.get("code") == 1:
                    message = resp_data.get("message", "")
                    if isinstance(message, dict):
                        message_data = message
                    else:
                        # 尝试将message解析为JSON
                        message_data = (
                            json.loads(message) if isinstance(message, str) else {}
                        )

                    # 业务成功，保存message中的内容
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(message_data, f, indent=2, ensure_ascii=False)
                    logging.info(f"[OK] 结果已保存: {out_path}")
                    return
                else:
                    # 业务失败，仅记录错误日志，不写入文件
                    error_msg = resp_data.get("message", "未知错误")
                    record_fail(f"业务处理失败: {img_path} | 原因: {error_msg}")
                    return
            else:
                record_fail(f"推理失败: {img_path} | {resp.status_code} | {resp.text}")
        except Exception as e:
            record_fail(f"[{os.path.basename(img_path)}] 推理异常: {e}")

        if attempt < max_retries:
            time.sleep(RETRY_SLEEP_BASE)

    record_fail(f"最终推理失败: {img_path}")


def infer_dir_to_jsons(
    image_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    config: dict | None = None,
    max_workers: int = MAX_WORKERS,
) -> None:
    """批量推理目录下所有图片"""
    image_dir = image_dir or IMAGE_PATH
    out_dir = out_dir or RAW_JSON_DIR
    os.makedirs(out_dir, exist_ok=True)

    img_list = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not img_list:
        logging.warning("目录无图像文件")
        return

    try:
        if config:
            get_token_if_expired(
                api_url=config.get("img_stream_url", IMG_STREAM_URL),
                ak=config.get("access_key", AK),
                sk=config.get("secret_key", SK),
            )
        else:
            get_token_if_expired()
    except Exception as e:
        logging.error(f"[ERR] 获取 token 失败，结束任务: {e}")
        return

    logging.info(f"开始处理 {len(img_list)} 张图片...")

    if max_workers == 1:
        # 串行处理：一个接一个执行，不使用线程池
        logging.info("使用串行模式处理...")
        success = failed = 0
        for i, img_path in enumerate(img_list):
            img_name = os.path.basename(img_path)
            try:
                infer_image_save_raw_json(img_path, out_dir, 3, config or {})
                success += 1
                logging.info(f"✅ 成功处理图片 {img_name}")
            except Exception as e:
                failed += 1
                logging.error(f"❌ 处理异常 {img_name}: {e}")
            # 每张完成后记录进度
            logging.info(
                f"进度: {i + 1}/{len(img_list)} | 成功 {success} | 失败 {failed}"
            )
        logging.info(f"串行处理完成: 成功 {success} 张，失败 {failed} 张")
    else:
        # 并行处理：使用线程池
        logging.info(f"使用并行模式处理（{max_workers} 个工作线程）...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    infer_image_save_raw_json, p, out_dir, 3, config or {}
                ): p
                for p in img_list
            }
            for future in safe_tqdm(
                as_completed(futures), total=len(futures), desc="推理进度"
            ):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"[ERR] 线程异常: {futures[future]} | {e}")


if __name__ == "__main__":
    infer_dir_to_jsons()
