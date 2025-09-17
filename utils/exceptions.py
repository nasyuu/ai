"""
异常处理模块

为图像流推理与评估工具提供统一的异常定义和处理机制。
"""


class BaseInferenceException(Exception):
    """推理工具基础异常类"""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}

    def __str__(self):
        error_info = f"[{self.error_code}] {self.message}"
        if self.details:
            error_info += f" | Details: {self.details}"
        return error_info


class ConfigurationError(BaseInferenceException):
    """配置相关异常"""

    def __init__(self, message: str, config_key: str = None, expected_type: str = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if expected_type:
            details["expected_type"] = expected_type

        super().__init__(message=message, error_code="CONFIG_ERROR", details=details)


class FileOperationError(BaseInferenceException):
    """文件操作相关异常"""

    def __init__(self, message: str, file_path: str = None, operation: str = None):
        details = {}
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation

        super().__init__(message=message, error_code="FILE_ERROR", details=details)


class InferenceAPIError(BaseInferenceException):
    """推理API相关异常"""

    def __init__(
        self,
        message: str,
        api_type: str = None,
        status_code: int = None,
        response_data: dict = None,
    ):
        details = {}
        if api_type:
            details["api_type"] = api_type
        if status_code:
            details["status_code"] = status_code
        if response_data:
            details["response_data"] = response_data

        super().__init__(message=message, error_code="API_ERROR", details=details)


class ModelEvaluationError(BaseInferenceException):
    """模型评估相关异常"""

    def __init__(self, message: str, eval_type: str = None, metric: str = None):
        details = {}
        if eval_type:
            details["eval_type"] = eval_type
        if metric:
            details["metric"] = metric

        super().__init__(message=message, error_code="EVAL_ERROR", details=details)


class DataFormatError(BaseInferenceException):
    """数据格式相关异常"""

    def __init__(
        self,
        message: str,
        data_type: str = None,
        expected_format: str = None,
        actual_format: str = None,
    ):
        details = {}
        if data_type:
            details["data_type"] = data_type
        if expected_format:
            details["expected_format"] = expected_format
        if actual_format:
            details["actual_format"] = actual_format

        super().__init__(message=message, error_code="FORMAT_ERROR", details=details)


class ImageProcessingError(BaseInferenceException):
    """图像处理相关异常"""

    def __init__(self, message: str, image_path: str = None, operation: str = None):
        details = {}
        if image_path:
            details["image_path"] = image_path
        if operation:
            details["operation"] = operation

        super().__init__(message=message, error_code="IMAGE_ERROR", details=details)


class NetworkError(BaseInferenceException):
    """网络连接相关异常"""

    def __init__(self, message: str, endpoint: str = None, timeout: int = None):
        details = {}
        if endpoint:
            details["endpoint"] = endpoint
        if timeout:
            details["timeout"] = timeout

        super().__init__(message=message, error_code="NETWORK_ERROR", details=details)


class ValidationError(BaseInferenceException):
    """数据验证相关异常"""

    def __init__(
        self, message: str, field: str = None, value=None, constraint: str = None
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        if constraint:
            details["constraint"] = constraint

        super().__init__(
            message=message, error_code="VALIDATION_ERROR", details=details
        )


class PipelineError(BaseInferenceException):
    """流水线执行相关异常"""

    def __init__(self, message: str, stage: str = None, step: str = None):
        details = {}
        if stage:
            details["stage"] = stage
        if step:
            details["step"] = step

        super().__init__(message=message, error_code="PIPELINE_ERROR", details=details)


class ThreadExecutionError(BaseInferenceException):
    """线程执行相关异常"""

    def __init__(self, message: str, thread_name: str = None, worker_id: int = None):
        details = {}
        if thread_name:
            details["thread_name"] = thread_name
        if worker_id is not None:
            details["worker_id"] = worker_id

        super().__init__(message=message, error_code="THREAD_ERROR", details=details)


# 异常处理装饰器
def handle_exceptions(logger=None, reraise=True, fallback_result=None):
    """
    异常处理装饰器

    Args:
        logger: 日志记录器，如果提供则记录异常信息
        reraise: 是否重新抛出异常，默认True
        fallback_result: 如果不重新抛出异常，返回的默认值
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseInferenceException as e:
                if logger:
                    logger.error(f"推理工具异常: {e}")
                if reraise:
                    raise
                return fallback_result
            except Exception as e:
                if logger:
                    logger.error(f"未知异常: {e}", exc_info=True)
                if reraise:
                    # 将普通异常包装为推理工具异常
                    raise BaseInferenceException(
                        message=f"未知错误: {str(e)}",
                        error_code="UNKNOWN_ERROR",
                        details={"original_exception": type(e).__name__},
                    )
                return fallback_result

        return wrapper

    return decorator


def format_exception_for_user(exception: BaseInferenceException) -> str:
    """
    格式化异常信息以供用户友好显示

    Args:
        exception: 推理工具异常实例

    Returns:
        str: 格式化的异常信息
    """
    if not isinstance(exception, BaseInferenceException):
        return f"系统错误: {str(exception)}"

    error_mapping = {
        "CONFIG_ERROR": "配置错误",
        "FILE_ERROR": "文件操作错误",
        "API_ERROR": "接口调用错误",
        "EVAL_ERROR": "模型评估错误",
        "FORMAT_ERROR": "数据格式错误",
        "IMAGE_ERROR": "图像处理错误",
        "NETWORK_ERROR": "网络连接错误",
        "VALIDATION_ERROR": "数据验证错误",
        "PIPELINE_ERROR": "流水线执行错误",
        "THREAD_ERROR": "线程执行错误",
        "UNKNOWN_ERROR": "未知错误",
    }

    error_type = error_mapping.get(exception.error_code, "系统错误")
    return f"{error_type}: {exception.message}"
