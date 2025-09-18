"""gRPC 推理客户端。"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import grpc

from utils.exceptions import ExternalServiceError, ValidationError
from utils.logger import get_logger

from . import grpc_module_pb2 as pb2
from . import grpc_module_pb2_grpc as pb2_grpc

__all__ = [
    "GRPCClientConfig",
    "create_stub",
    "infer_image_save_raw_json",
    "infer_dir_to_jsons",
]

logger = get_logger(__name__)

_FAIL_LOG_PATH = Path("fail.log")
_LOG_LOCK = threading.Lock()


@dataclass(slots=True)
class GRPCClientConfig:
    """gRPC 推理核心配置。"""

    server_address: str
    stream_name: str
    task_id: str
    output_dir: Path = Path("grpc/responses")
    max_workers: int = 1
    call_timeout: int = 60
    retry_max: int = 3
    retry_backoff: float = 2.0
    max_message_bytes: int = 100 * 1024 * 1024

    @classmethod
    def from_mapping(cls, payload: dict) -> "GRPCClientConfig":
        required = ["server_address", "stream_name", "task_id"]
        for key in required:
            if not payload.get(key):
                raise ValidationError(f"配置项 {key} 缺失或为空")
        return cls(
            server_address=payload["server_address"],
            stream_name=payload["stream_name"],
            task_id=payload["task_id"],
            output_dir=Path(payload.get("output_dir", "grpc/responses")),
            max_workers=int(payload.get("max_workers", 1)),
            call_timeout=int(payload.get("call_timeout", 60)),
            retry_max=int(payload.get("retry_max", 3)),
            retry_backoff=float(payload.get("retry_backoff", 2.0)),
            max_message_bytes=int(payload.get("max_message_bytes", 100 * 1024 * 1024)),
        )


def _record_fail(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _LOG_LOCK:
        _FAIL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _FAIL_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")


def create_stub(
    config: GRPCClientConfig,
) -> tuple[grpc.Channel, pb2_grpc.StandardGrpcServiceStub]:
    if not config.server_address:
        raise ValidationError("server_address 不能为空")
    options = [
        ("grpc.max_receive_message_length", config.max_message_bytes),
        ("grpc.max_send_message_length", config.max_message_bytes),
    ]
    channel = grpc.insecure_channel(config.server_address, options=options)
    stub = pb2_grpc.StandardGrpcServiceStub(channel)
    return channel, stub


def _infer_bytes(
    stub: pb2_grpc.StandardGrpcServiceStub,
    config: GRPCClientConfig,
    image_bytes: bytes,
    file_name: str,
) -> pb2.Response:
    request = pb2.Request(stream_name=config.stream_name, image_data=image_bytes)
    metadata = [("taskid", config.task_id)]

    last_error: Optional[Exception] = None
    backoff = config.retry_backoff
    for attempt in range(1, config.retry_max + 1):
        try:
            response = stub.standardInfer(
                request,
                timeout=config.call_timeout,
                metadata=metadata,
            )
            try:
                json.loads(response.message)
            except json.JSONDecodeError as exc:  # noqa: PERF203
                raise ExternalServiceError(
                    "响应格式错误：无法解析 JSON 数据",
                    details={"file": file_name},
                    service="grpc",
                ) from exc
            return response
        except grpc.RpcError as exc:  # type: ignore[union-attr]
            last_error = exc
            logger.warning(
                "gRPC 调用失败，将重试 (%s/%s): %s",
                attempt,
                config.retry_max,
                exc,
                exc_info=False,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "处理失败，将重试 (%s/%s): %s",
                attempt,
                config.retry_max,
                exc,
                exc_info=False,
            )

        if attempt < config.retry_max:
            time.sleep(backoff)
            backoff += config.retry_backoff

    raise ExternalServiceError(
        "gRPC 推理失败",
        details={"file": file_name},
        service="grpc",
    ) from last_error


def infer_image_save_raw_json(
    image_path: str | Path,
    config: GRPCClientConfig,
    stub: Optional[pb2_grpc.StandardGrpcServiceStub] = None,
) -> bool:
    path = Path(image_path)
    if not path.exists():
        raise ValidationError("图片不存在", details={"path": str(path)})

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with path.open("rb") as handle:
        image_bytes = handle.read()

    owns_stub = False
    channel: Optional[grpc.Channel] = None

    if stub is None:
        channel, stub = create_stub(config)
        owns_stub = True

    try:
        response = _infer_bytes(stub, config, image_bytes, path.name)
        payload = json.loads(response.message)
        out_path = output_dir / f"{path.stem}.json"

        if response.code == 1:
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            logger.info("成功处理图片 %s", path.name)
            return True

        error_msg = (
            payload.get("message", "未知错误") if isinstance(payload, dict) else payload
        )
        _record_fail(f"业务处理失败 {path.name} | 原因: {error_msg}")
        return False
    except Exception as exc:  # noqa: BLE001
        _record_fail(f"处理 {path.name} 失败: {exc}")
        logger.error("处理图片 %s 失败: %s", path.name, exc)
        return False
    finally:
        if owns_stub and channel is not None:
            channel.close()


def infer_dir_to_jsons(
    image_dir: str | Path,
    config: GRPCClientConfig,
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

    channel, stub = create_stub(config)
    try:
        logger.info("开始处理 %s 张图片", len(images))

        if config.max_workers == 1:
            success = failed = 0
            for index, path in enumerate(images, start=1):
                if infer_image_save_raw_json(path, config, stub):
                    success += 1
                else:
                    failed += 1
                logger.info(
                    "进度: %s/%s | 成功 %s | 失败 %s",
                    index,
                    len(images),
                    success,
                    failed,
                )
            logger.info("串行处理完成: 成功 %s, 失败 %s", success, failed)
            return

        logger.info("使用并行模式，工作线程: %s", config.max_workers)
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(infer_image_save_raw_json, path, config, stub): path
                for path in images
            }
            success = failed = 0
            completed = 0
            for future in as_completed(futures):
                completed += 1
                target = futures[future]
                try:
                    if future.result():
                        success += 1
                    else:
                        failed += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    _record_fail(f"处理 {target.name} 失败: {exc}")
                logger.info(
                    "进度: %s/%s | 成功 %s | 失败 %s",
                    completed,
                    len(images),
                    success,
                    failed,
                )
            logger.info("并行处理完成: 成功 %s, 失败 %s", success, failed)
    finally:
        channel.close()
