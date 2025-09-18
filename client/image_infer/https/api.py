"""HTTPS 推理客户端。"""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import requests
import urllib3

from client.auth import HTTPSAuthConfig, HTTPSAuthenticator
from utils.exceptions import ExternalServiceError, ValidationError
from utils.logger import get_logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

__all__ = [
    "HTTPSClientConfig",
    "infer_image_save_raw_json",
    "infer_dir_to_jsons",
]

logger = get_logger(__name__)
_authenticator = HTTPSAuthenticator(logger=logger)
_session = requests.Session()

_FAIL_LOG_PATH = Path("fail.log")
_REQUEST_RETRY_MAX = 3
_RETRY_SLEEP_BASE = 2.0
_JPEG_QUALITY = 80

_log_lock = threading.Lock()


@dataclass(slots=True)
class HTTPSClientConfig:
    """HTTPS 推理所需的核心配置。"""

    img_stream_url: str
    stream_name: str
    access_key: str
    secret_key: str
    tenant_id: int = 1002
    raw_responses_dir: Path = Path("https/responses")
    max_workers: int = 1
    token_endpoint: str = "/gam/v1/auth/tokens"
    verify_ssl: bool = False
    request_timeout: int = 15

    @classmethod
    def from_mapping(cls, payload: dict) -> "HTTPSClientConfig":
        required = ["img_stream_url", "stream_name", "access_key", "secret_key"]
        for key in required:
            if not payload.get(key):
                raise ValidationError(f"配置项 {key} 缺失或为空")
        return cls(
            img_stream_url=payload["img_stream_url"],
            stream_name=payload["stream_name"],
            access_key=payload["access_key"],
            secret_key=payload["secret_key"],
            tenant_id=payload.get("tenant_id", 1002),
            raw_responses_dir=Path(payload.get("raw_responses_dir", "https/responses")),
            max_workers=int(payload.get("max_workers", 1)),
            token_endpoint=payload.get("token_endpoint", "/gam/v1/auth/tokens"),
            verify_ssl=bool(payload.get("verify_ssl", False)),
            request_timeout=int(payload.get("request_timeout", 15)),
        )

    def auth_config(self) -> HTTPSAuthConfig:
        return HTTPSAuthConfig(
            base_url=self.img_stream_url,
            access_key=self.access_key,
            secret_key=self.secret_key,
            tenant_id=self.tenant_id,
            token_endpoint=self.token_endpoint,
            verify=self.verify_ssl,
            timeout=self.request_timeout,
        )

    def request_url(self) -> str:
        return self.auth_config().normalised_base()


def _create_safe_tqdm():
    try:
        from tqdm import tqdm

        if getattr(sys, "frozen", False):

            def silent(iterable, **kwargs):
                kwargs.pop("desc", None)
                kwargs.pop("position", None)
                try:
                    return tqdm(iterable, disable=True, **kwargs)
                except Exception:  # noqa: BLE001
                    return iterable

            return silent
        return tqdm
    except Exception:  # noqa: BLE001
        return lambda iterable, **_: iterable


def _record_fail(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _log_lock:
        _FAIL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _FAIL_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")


safe_tqdm = _create_safe_tqdm()


def frame_to_base64(frame, quality: int = _JPEG_QUALITY) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ExternalServiceError("图像编码失败", service="cv2")
    return base64.b64encode(buffer).decode("utf-8")


def _request_with_retry(
    url: str,
    headers: dict[str, str],
    data: str,
    *,
    retries: int = _REQUEST_RETRY_MAX,
    timeout: int = 15,
    verify: bool = False,
) -> requests.Response:
    last_error: Optional[Exception] = None
    sleep = _RETRY_SLEEP_BASE
    for attempt in range(1, retries + 1):
        try:
            response = _session.post(
                url,
                headers=headers,
                data=data,
                verify=verify,
                timeout=timeout,
            )
            return response
        except requests.RequestException as exc:  # type: ignore[union-attr]
            last_error = exc
            logger.warning("第%s次请求失败，将重试: %s", attempt, exc, exc_info=False)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("第%s次请求失败，将重试: %s", attempt, exc, exc_info=False)

        if attempt < retries:
            time.sleep(sleep)
            sleep *= 2

    raise ExternalServiceError(
        "请求失败", details={"url": url}, service="https"
    ) from last_error


def infer_image_save_raw_json(
    img_path: str,
    config: HTTPSClientConfig,
    *,
    max_retries: int = _REQUEST_RETRY_MAX,
) -> None:
    frame = cv2.imread(img_path)
    if frame is None:
        _record_fail(f"无法读取图片: {img_path}")
        raise ValidationError("无法载入图像", details={"path": img_path})

    output_dir = config.raw_responses_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(img_path).stem
    out_path = output_dir / f"{base_name}.json"

    for attempt in range(1, max_retries + 1):
        try:
            token = _authenticator.get_token(config.auth_config())
            payload = {
                "stream_name": config.stream_name,
                "image_base64": f"data:image/jpeg;base64,{frame_to_base64(frame)}",
            }
            headers = {"Content-Type": "application/json", "Authorization": token}
            response = _request_with_retry(
                config.request_url(),
                headers,
                json.dumps(payload),
                timeout=config.request_timeout,
                verify=config.verify_ssl,
            )
            if response.status_code != 200:
                _record_fail(
                    f"推理失败: {img_path} | {response.status_code} | {response.text}"
                )
                break

            body = response.json()
            if body.get("code") == 1:
                message = body.get("message", {})
                if isinstance(message, str):
                    try:
                        message = json.loads(message)
                    except json.JSONDecodeError:
                        message = {"message": message}
                with out_path.open("w", encoding="utf-8") as handle:
                    json.dump(message, handle, indent=2, ensure_ascii=False)
                logger.info("结果已保存: %s", out_path)
                return

            error_msg = body.get("message", "未知错误")
            _record_fail(f"业务处理失败: {img_path} | {error_msg}")
            return
        except Exception as exc:  # noqa: BLE001
            _record_fail(f"[{Path(img_path).name}] 推理异常: {exc}")

        if attempt < max_retries:
            time.sleep(_RETRY_SLEEP_BASE)

    _record_fail(f"最终推理失败: {img_path}")


def infer_dir_to_jsons(
    image_dir: str | Path,
    config: HTTPSClientConfig,
) -> None:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValidationError("图片目录不存在", details={"path": str(image_dir)})

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        logger.warning("目录无图像文件: %s", image_dir)
        return

    try:
        _authenticator.get_token(config.auth_config())
    except ExternalServiceError as exc:
        logger.error("获取 token 失败: %s", exc)
        return

    logger.info("开始处理 %s 张图片", len(images))

    if config.max_workers == 1:
        success = failed = 0
        for idx, path in enumerate(images, start=1):
            try:
                infer_image_save_raw_json(str(path), config)
                success += 1
                logger.info("成功处理: %s", path.name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.error("处理失败 %s: %s", path.name, exc)
            logger.info(
                "进度: %s/%s | 成功 %s | 失败 %s", idx, len(images), success, failed
            )
        logger.info("串行处理完成: 成功 %s, 失败 %s", success, failed)
        return

    logger.info("使用并行模式，工作线程: %s", config.max_workers)
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_map = {
            executor.submit(infer_image_save_raw_json, str(path), config): path
            for path in images
        }
        for future in safe_tqdm(
            as_completed(future_map), total=len(future_map), desc="推理进度"
        ):
            target = future_map[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("线程处理失败 %s: %s", target, exc)
