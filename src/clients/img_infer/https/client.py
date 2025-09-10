from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import requests
import urllib3

from clients.auth import BasicTokenConfig, BasicTokenProvider, TokenProvider
from utils.exception import (
    InferenceError,
    IOErrorEx,
    SerializationError,
    ensure,
)
from utils.logger import get_logger, record_fail

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
log = get_logger("https.client")


# --------------------
# 配置
# --------------------
@dataclass
class HttpsConfig:
    img_stream_url: str
    stream_name: str
    access_key: str
    secret_key: str
    max_workers: int = 1
    deadline_sec: int = 15
    retry_max: int = 3
    retry_base_sec: float = 2.0
    jpeg_quality: int = 80

    def validate(self):
        ensure(bool(self.img_stream_url), "img_stream_url 不能为空", code="https.url")
        ensure(bool(self.stream_name), "stream_name 不能为空", code="https.stream")
        ensure(bool(self.access_key), "access_key 不能为空", code="https.ak")
        ensure(bool(self.secret_key), "secret_key 不能为空", code="https.sk")


# 线程安全 fail.log
_log_lock = threading.Lock()


def _record_fail_line(msg: str):
    with _log_lock:
        record_fail(msg)


# --------------------
# 工具
# --------------------
_session = requests.Session()


def _frame_to_base64_jpeg(frame, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise IOErrorEx("OpenCV 编码失败")
    return base64.b64encode(buf).decode("utf-8")


def _post_with_retry(
    url: str,
    headers: Dict[str, str],
    data: str,
    timeout: int,
    retry_max: int,
    retry_base: float,
):
    for i in range(retry_max):
        try:
            return _session.post(
                url, headers=headers, data=data, timeout=timeout, verify=False
            )
        except Exception as e:
            if i < retry_max - 1:
                wait = retry_base * (i + 1)
                log.warning("POST 失败（重试中）%s s 后重试... | %s", wait, e)
                time.sleep(wait)
                continue
            _record_fail_line(f"[REQUEST_ERROR] {url} | {e}")
            raise


# --------------------
# 单图推理
# --------------------
def infer_image_bytes(
    cfg: HttpsConfig,
    token_provider: TokenProvider,
    image_bytes: bytes,
    file_name: str,
) -> dict:
    """
    将二进制图片做推理，返回业务 message 解析后的 dict（仅 code==1 才返回）。
    """
    cfg.validate()

    # 构造 payload（保持与你原来的服务字段一致）
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "stream_name": cfg.stream_name,
        "image_base64": f"data:image/jpeg;base64,{img_b64}",
    }

    for i in range(cfg.retry_max):
        try:
            token = token_provider.get_token()
            headers = {"Content-Type": "application/json", "Authorization": token}
            resp = _post_with_retry(
                url=cfg.img_stream_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=cfg.deadline_sec,
                retry_max=1,  # 单次 HTTP 不再内部重试，交给外层 for-loop 控制
                retry_base=cfg.retry_base_sec,
            )

            if resp.status_code != 200:
                _record_fail_line(
                    f"[HTTP_{resp.status_code}] {file_name} | {resp.text[:1000]}"
                )
                raise InferenceError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                    retriable=True,
                    file=file_name,
                )

            # 返回 JSON
            try:
                resp_data = resp.json()
            except Exception as e:
                _record_fail_line(f"[SERDE_ERROR] {file_name} | 响应不是 JSON | {e}")
                raise SerializationError("响应不是合法 JSON", cause=e, file=file_name)

            code = resp_data.get("code")
            msg = resp_data.get("message", {})

            if code == 1:
                # message 可能是 dict / str(JSON)
                if isinstance(msg, dict):
                    return msg
                try:
                    return json.loads(msg)
                except Exception as e:
                    _record_fail_line(
                        f"[SERDE_ERROR] {file_name} | message 解析失败 | {e}"
                    )
                    raise SerializationError(
                        "message 不是合法 JSON", cause=e, file=file_name
                    )
            else:
                biz_msg = (
                    msg
                    if isinstance(msg, str)
                    else (msg.get("message") if isinstance(msg, dict) else str(msg))
                )
                _record_fail_line(f"[BUSINESS_FAIL] {file_name} | {biz_msg}")
                raise InferenceError(
                    f"业务失败: {biz_msg}", retriable=False, file=file_name
                )

        except (SerializationError, InferenceError) as e:
            # 按 retriable 控制是否重试
            if getattr(e, "retriable", False) and i < cfg.retry_max - 1:
                wait = cfg.retry_base_sec * (i + 1)
                log.warning("推理异常（可重试），%s s 后重试... | %s", wait, e)
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            # 其他异常默认可重试
            if i < cfg.retry_max - 1:
                wait = cfg.retry_base_sec * (i + 1)
                log.warning("未知异常（可重试），%s s 后重试... | %s", wait, e)
                time.sleep(wait)
                continue
            _record_fail_line(f"[UNKNOWN] {file_name} | {e}")
            raise


def infer_image_path(
    cfg: HttpsConfig,
    token_provider: TokenProvider,
    img_path: str | os.PathLike,
) -> dict:
    p = str(img_path)
    frame = cv2.imread(p)
    if frame is None:
        _record_fail_line(f"[READ_ERROR] 无法读取图片: {p}")
        raise IOErrorEx("无法读取图片", path=p, file=os.path.basename(p))

    ok, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)]
    )
    if not ok:
        raise IOErrorEx("OpenCV 编码失败", path=p, file=os.path.basename(p))
    return infer_image_bytes(cfg, token_provider, bytes(buf), os.path.basename(p))


# --------------------
# 目录批量推理（串/并行）
# --------------------
def infer_dir(
    cfg: HttpsConfig,
    out_dir: str | os.PathLike,
    image_dir: str | os.PathLike,
    token_provider: Optional[TokenProvider] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[int, int]:
    """
    遍历目录推理，仅在 code==1 时保存 message(JSON) → *.json
    返回 (success, failed)
    """
    cfg.validate()
    out_dir = str(out_dir)
    image_dir = str(image_dir)

    files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    total = len(files)
    if total == 0:
        log.warning("目录无图像文件: %s", image_dir)
        return 0, 0

    # 默认使用 BasicTokenProvider
    token_provider = token_provider or BasicTokenProvider(
        BasicTokenConfig(
            endpoint=cfg.img_stream_url,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
        )
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def run_one(fname: str) -> bool:
        try:
            msg = infer_image_path(cfg, token_provider, Path(image_dir) / fname)
            outp = Path(out_dir) / (Path(fname).stem + ".json")
            with outp.open("w", encoding="utf-8") as f:
                json.dump(msg, f, ensure_ascii=False, indent=2)
            log.info("✅ 成功: %s", fname)
            return True
        except Exception as e:
            log.error("❌ 失败: %s | %s", fname, e)
            return False

    ok = fail = 0
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
