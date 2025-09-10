# src/utils/exception.py
from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional, Type

# 建议：项目内优先使用 utils.logger 中的工具
try:
    from utils.logger import get_logger, record_fail
except Exception:  # 降级可运行
    import logging

    def get_logger(name="ai"):
        return logging.getLogger(name)

    def record_fail(msg: str, log_dir=None):
        pass


__all__ = [
    # 核心类型
    "AppError",
    "RetryableError",
    "NonRetryableError",
    # 具体类别
    "ConfigError",
    "IOErrorEx",
    "SerializationError",
    "ValidationError",
    "HTTPError",
    "GRPCError",
    "AuthError",
    "InferenceError",
    "ConversionError",
    "EvaluationError",
    # 工具/工厂
    "wrap_ex",
    "reraise_as",
    "from_http_response",
    "from_grpc_rpc_error",
    "raise_if_none",
    "ensure",
]


@dataclass
class AppError(Exception):
    """
    项目统一异常：含错误码、消息、上下文，可序列化。
    code: 机器友好错误码（大类.子类.细节）
    retriable: 是否建议重试
    ctx: 附带上下文（文件名、请求ID、图像名、配置项等）
    """

    code: str
    message: str
    retriable: bool = False
    cause: Optional[BaseException] = field(default=None, repr=False)
    ctx: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 确保 Exception 正常初始化
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.cause:
            d["cause"] = repr(self.cause)
            d["trace"] = "".join(
                traceback.format_exception(None, self.cause, self.cause.__traceback__)
            )
        return d

    def __str__(self):
        return json.dumps(
            {
                "code": self.code,
                "message": self.message,
                "retriable": self.retriable,
                "ctx": self.ctx,
            },
            ensure_ascii=False,
        )

    # 便捷日志
    def log(self, level: str = "error"):
        log = get_logger("utils.exception")
        payload = f"{self.code} | {self.message} | ctx={self.ctx}"
        if level == "warning":
            log.warning(payload)
        elif level == "info":
            log.info(payload)
        else:
            log.error(payload)
        # 业务失败可选记录 fail.log
        if not self.retriable:
            record_fail(payload)


class RetryableError(AppError):
    """建议重试的异常（网络抖动、超时、限流等）"""

    def __init__(
        self, code: str, message: str, cause: Optional[BaseException] = None, **ctx
    ):
        super().__init__(
            code=code, message=message, retriable=True, cause=cause, ctx=ctx
        )


class NonRetryableError(AppError):
    """不建议重试的异常（参数错误、权限、数据不符合规范等）"""

    def __init__(
        self, code: str, message: str, cause: Optional[BaseException] = None, **ctx
    ):
        super().__init__(
            code=code, message=message, retriable=False, cause=cause, ctx=ctx
        )


class ConfigError(NonRetryableError):
    """配置缺失/非法"""

    def __init__(self, message: str, **ctx):
        super().__init__("config.invalid", message, **ctx)


class IOErrorEx(RetryableError):
    """文件/目录/磁盘等 I/O 异常"""

    def __init__(self, message: str, cause: Optional[BaseException] = None, **ctx):
        super().__init__("io.error", message, cause, **ctx)


class SerializationError(NonRetryableError):
    """JSON/Proto/LabelMe 等序列化解析异常"""

    def __init__(self, message: str, cause: Optional[BaseException] = None, **ctx):
        super().__init__("serde.error", message, cause, **ctx)


class ValidationError(NonRetryableError):
    """数据内容不满足约束"""

    def __init__(self, message: str, **ctx):
        super().__init__("data.invalid", message, **ctx)


class HTTPError(RetryableError):
    """HTTP 请求异常（默认可重试；按状态码可切换）"""

    def __init__(
        self,
        status: int,
        message: str,
        cause: Optional[BaseException] = None,
        retriable: Optional[bool] = None,
        **ctx,
    ):
        if retriable is None:
            retriable = status >= 500 or status in (408, 429)
        super().__init__(
            "net.http", f"[{status}] {message}", cause, status=status, **ctx
        )


class GRPCError(RetryableError):
    """gRPC 异常（默认按 StatusCode 识别）"""

    def __init__(
        self,
        status_code: str,
        message: str,
        cause: Optional[BaseException] = None,
        retriable: Optional[bool] = None,
        **ctx,
    ):
        if retriable is None:
            retriable = status_code in {
                "UNAVAILABLE",
                "DEADLINE_EXCEEDED",
                "RESOURCE_EXHAUSTED",
                "ABORTED",
            }
        super().__init__(
            "net.grpc",
            f"[{status_code}] {message}",
            cause,
            status_code=status_code,
            **ctx,
        )


class AuthError(NonRetryableError):
    """认证/鉴权失败（AK/SK、Token 失效、权限不足等）"""

    def __init__(self, message: str, **ctx):
        super().__init__("auth.failed", message, **ctx)


class InferenceError(RetryableError):
    """推理阶段错误（服务不可用/超时/响应异常等）"""

    def __init__(
        self,
        message: str,
        cause: Optional[BaseException] = None,
        retriable: bool = True,
        **ctx,
    ):
        super().__init__("infer.error", message, cause, **ctx)
        self.retriable = retriable


class ConversionError(NonRetryableError):
    """格式转换错误（RLE→LabelMe、字段缺失/非法）"""

    def __init__(self, message: str, cause: Optional[BaseException] = None, **ctx):
        super().__init__("convert.error", message, cause, **ctx)


class EvaluationError(NonRetryableError):
    """评估阶段错误（指标计算、输入不齐等）"""

    def __init__(self, message: str, cause: Optional[BaseException] = None, **ctx):
        super().__init__("eval.error", message, cause, **ctx)


def from_http_response(
    resp, body_text: Optional[str] = None, ctx: Optional[dict] = None
) -> AppError:
    """
    由 requests.Response 生成统一 HTTPError/AuthError。
    """
    try:
        status = int(getattr(resp, "status_code", 0))
    except Exception:
        status = 0

    # 解析 body
    text = body_text
    if text is None:
        try:
            text = resp.text  # type: ignore[attr-defined]
        except Exception:
            text = ""

    msg = text or "HTTP request failed"
    # 尝试抽取 message 字段
    try:
        data = resp.json()  # type: ignore[attr-defined]
        msg = data.get("message") or data.get("error") or msg
        # 若业务返回 code 领域信息，附加到 ctx
        if isinstance(ctx, dict) and "biz_code" not in ctx:
            biz_code = data.get("code")
            if biz_code is not None:
                ctx["biz_code"] = biz_code
    except Exception:
        pass

    # 401/403 归为 AuthError
    if status in (401, 403):
        return AuthError(f"{msg}", status=status, **(ctx or {}))
    return HTTPError(status=status, message=msg, retriable=None, **(ctx or {}))


def from_grpc_rpc_error(err) -> AppError:
    """
    由 grpc.RpcError 生成 GRPCError/AuthError。
    """
    # 兼容无 grpc 依赖的环境
    code_name = getattr(getattr(err, "code", lambda: None)(), "name", "UNKNOWN")
    details = getattr(err, "details", lambda: str(err))()
    message = details or repr(err)

    # 常见：UNAUTHENTICATED -> AuthError
    if code_name in {"UNAUTHENTICATED", "PERMISSION_DENIED"}:
        return AuthError(message, status_code=code_name)
    return GRPCError(status_code=code_name, message=message, cause=err, retriable=None)


def wrap_ex(
    fn: Callable[..., Any],
    *,
    reraise: bool = True,
    default: Any = None,
    map_to: Optional[Type[AppError]] = None,
    **ctx,
):
    """
    运行函数并把非 AppError 统一包装；可选择返回默认值或抛出。
    - map_to: 指定将未知异常映射为的 AppError 子类（默认 InferenceError）
    """
    try:
        return fn()
    except AppError as ae:
        ae.ctx.update(ctx)
        ae.log("error")
        if reraise:
            raise
        return default
    except Exception as e:
        err_cls: Type[AppError] = map_to or InferenceError
        ae = err_cls(message=str(e), cause=e, **ctx)  # type: ignore[arg-type]
        ae.log("error")
        if reraise:
            raise ae
        return default


def reraise_as(err: BaseException, target: Type[AppError], **ctx) -> AppError:
    """
    将任意异常转换为目标 AppError 并抛出。
    """
    if isinstance(err, AppError):
        err.ctx.update(ctx)
        raise err
    raise target(message=str(err), cause=err, **ctx)


def raise_if_none(
    value: Any, *, name: str, code: str = "data.missing", retriable: bool = False, **ctx
):
    if value is None:
        if retriable:
            raise RetryableError(code, f"{name} is None", **ctx)
        raise NonRetryableError(code, f"{name} is None", **ctx)
    return value


def ensure(
    cond: bool,
    *,
    message: str,
    code: str = "check.failed",
    retriable: bool = False,
    **ctx,
):
    if not cond:
        if retriable:
            raise RetryableError(code, message, **ctx)
        raise NonRetryableError(code, message, **ctx)
