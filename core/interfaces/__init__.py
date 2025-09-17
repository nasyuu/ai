"""
接口抽象层

定义统一的推理接口抽象，支持多种协议（HTTPS、gRPC等）的统一访问。
为后续功能扩充提供可插拔的接口架构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.exceptions import InferenceAPIError, ValidationError
from utils.logger import get_logger

logger = get_logger("interfaces")


class InferenceProtocol(Enum):
    """推理协议类型"""

    HTTPS = "https"
    GRPC = "grpc"
    GRPC_STANDARD = "grpc_standard"
    WEBSOCKET = "websocket"  # 预留


class InferenceStatus(Enum):
    """推理状态"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class InferenceRequest:
    """推理请求"""

    image_path: Union[str, Path]
    image_data: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)

        if self.metadata is None:
            self.metadata = {}


@dataclass
class InferenceResponse:
    """推理响应"""

    request_id: str
    status: InferenceStatus
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self.status == InferenceStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        """是否失败"""
        return self.status in [InferenceStatus.FAILED, InferenceStatus.TIMEOUT]


@dataclass
class ConnectionConfig:
    """连接配置"""

    endpoint: str
    timeout: int = 60
    retry_count: int = 3
    retry_delay: float = 1.0
    max_concurrent: int = 1

    # 认证配置
    auth_type: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    token: Optional[str] = None

    # 协议特定配置
    protocol_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.protocol_config is None:
            self.protocol_config = {}


class BaseInferenceClient(ABC):
    """推理客户端基类"""

    def __init__(self, protocol: InferenceProtocol, config: ConnectionConfig):
        self.protocol = protocol
        self.config = config
        self._connected = False
        self._client_info = {}

    @abstractmethod
    async def connect(self) -> bool:
        """建立连接"""
        pass

    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass

    @abstractmethod
    async def infer_single(self, request: InferenceRequest) -> InferenceResponse:
        """单张图片推理"""
        pass

    async def infer_batch(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """批量推理（默认实现为串行）"""
        responses = []
        for request in requests:
            try:
                response = await self.infer_single(request)
                responses.append(response)
            except Exception as e:
                error_response = InferenceResponse(
                    request_id=request.request_id or "unknown",
                    status=InferenceStatus.FAILED,
                    error_message=str(e),
                )
                responses.append(error_response)
        return responses

    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置"""
        pass

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected

    @property
    def client_info(self) -> Dict[str, Any]:
        """客户端信息"""
        return self._client_info.copy()


class InferenceClientFactory:
    """推理客户端工厂"""

    _client_classes = {}

    @classmethod
    def register_client(cls, protocol: InferenceProtocol, client_class: type):
        """注册客户端类"""
        cls._client_classes[protocol] = client_class
        logger.info(f"注册推理客户端: {protocol.value} -> {client_class.__name__}")

    @classmethod
    def create_client(
        cls, protocol: InferenceProtocol, config: ConnectionConfig
    ) -> BaseInferenceClient:
        """创建客户端实例"""
        if protocol not in cls._client_classes:
            raise InferenceAPIError(
                f"不支持的推理协议: {protocol.value}", api_type=protocol.value
            )

        client_class = cls._client_classes[protocol]
        return client_class(protocol, config)

    @classmethod
    def get_supported_protocols(cls) -> List[InferenceProtocol]:
        """获取支持的协议列表"""
        return list(cls._client_classes.keys())


class InferenceSessionManager:
    """推理会话管理器"""

    def __init__(self):
        self._sessions: Dict[str, BaseInferenceClient] = {}
        self._session_configs: Dict[str, ConnectionConfig] = {}

    def create_session(
        self, session_id: str, protocol: InferenceProtocol, config: ConnectionConfig
    ) -> BaseInferenceClient:
        """创建推理会话"""
        if session_id in self._sessions:
            raise ValidationError(f"会话ID已存在: {session_id}")

        client = InferenceClientFactory.create_client(protocol, config)
        self._sessions[session_id] = client
        self._session_configs[session_id] = config

        logger.info(f"创建推理会话: {session_id} ({protocol.value})")
        return client

    def get_session(self, session_id: str) -> Optional[BaseInferenceClient]:
        """获取推理会话"""
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> bool:
        """删除推理会话"""
        if session_id not in self._sessions:
            return False

        client = self._sessions[session_id]
        if client.is_connected:
            # 异步断开连接需要在事件循环中处理
            logger.warning(f"会话{session_id}仍在连接状态，建议先断开连接")

        del self._sessions[session_id]
        del self._session_configs[session_id]

        logger.info(f"删除推理会话: {session_id}")
        return True

    def list_sessions(self) -> List[str]:
        """列出所有会话ID"""
        return list(self._sessions.keys())

    async def connect_all(self) -> Dict[str, bool]:
        """连接所有会话"""
        results = {}
        for session_id, client in self._sessions.items():
            try:
                success = await client.connect()
                results[session_id] = success
                if success:
                    logger.info(f"会话{session_id}连接成功")
                else:
                    logger.error(f"会话{session_id}连接失败")
            except Exception as e:
                results[session_id] = False
                logger.error(f"会话{session_id}连接异常: {e}")
        return results

    async def disconnect_all(self):
        """断开所有会话"""
        for session_id, client in self._sessions.items():
            try:
                if client.is_connected:
                    await client.disconnect()
                    logger.info(f"会话{session_id}已断开")
            except Exception as e:
                logger.error(f"会话{session_id}断开异常: {e}")


class InferenceMiddleware(ABC):
    """推理中间件基类"""

    @abstractmethod
    async def before_request(self, request: InferenceRequest) -> InferenceRequest:
        """请求前处理"""
        pass

    @abstractmethod
    async def after_response(
        self, request: InferenceRequest, response: InferenceResponse
    ) -> InferenceResponse:
        """响应后处理"""
        pass

    @abstractmethod
    async def on_error(
        self, request: InferenceRequest, error: Exception
    ) -> Optional[InferenceResponse]:
        """错误处理"""
        pass


class LoggingMiddleware(InferenceMiddleware):
    """日志中间件"""

    def __init__(self, enable_debug: bool = False):
        self.enable_debug = enable_debug

    async def before_request(self, request: InferenceRequest) -> InferenceRequest:
        if self.enable_debug:
            logger.debug(f"推理请求: {request.request_id} - {request.image_path}")
        return request

    async def after_response(
        self, request: InferenceRequest, response: InferenceResponse
    ) -> InferenceResponse:
        if response.is_success:
            logger.info(
                f"推理成功: {response.request_id} - 耗时{response.processing_time:.2f}s"
            )
        else:
            logger.error(f"推理失败: {response.request_id} - {response.error_message}")
        return response

    async def on_error(
        self, request: InferenceRequest, error: Exception
    ) -> Optional[InferenceResponse]:
        logger.error(f"推理异常: {request.request_id} - {error}")
        return None


class RetryMiddleware(InferenceMiddleware):
    """重试中间件"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def before_request(self, request: InferenceRequest) -> InferenceRequest:
        return request

    async def after_response(
        self, request: InferenceRequest, response: InferenceResponse
    ) -> InferenceResponse:
        return response

    async def on_error(
        self, request: InferenceRequest, error: Exception
    ) -> Optional[InferenceResponse]:
        # 重试逻辑可以在这里实现
        return None


class MiddlewareManager:
    """中间件管理器"""

    def __init__(self):
        self._middlewares: List[InferenceMiddleware] = []

    def add_middleware(self, middleware: InferenceMiddleware):
        """添加中间件"""
        self._middlewares.append(middleware)
        logger.info(f"添加中间件: {middleware.__class__.__name__}")

    def remove_middleware(self, middleware: InferenceMiddleware):
        """移除中间件"""
        if middleware in self._middlewares:
            self._middlewares.remove(middleware)
            logger.info(f"移除中间件: {middleware.__class__.__name__}")

    async def process_request(self, request: InferenceRequest) -> InferenceRequest:
        """处理请求"""
        for middleware in self._middlewares:
            request = await middleware.before_request(request)
        return request

    async def process_response(
        self, request: InferenceRequest, response: InferenceResponse
    ) -> InferenceResponse:
        """处理响应"""
        for middleware in reversed(self._middlewares):
            response = await middleware.after_response(request, response)
        return response

    async def handle_error(
        self, request: InferenceRequest, error: Exception
    ) -> Optional[InferenceResponse]:
        """处理错误"""
        for middleware in self._middlewares:
            result = await middleware.on_error(request, error)
            if result is not None:
                return result
        return None


# 全局会话管理器
session_manager = InferenceSessionManager()

# 全局中间件管理器
middleware_manager = MiddlewareManager()

# 默认添加日志中间件
middleware_manager.add_middleware(LoggingMiddleware())


def get_session_manager() -> InferenceSessionManager:
    """获取会话管理器"""
    return session_manager


def get_middleware_manager() -> MiddlewareManager:
    """获取中间件管理器"""
    return middleware_manager
