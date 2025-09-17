"""
Token认证模块

从backup项目中提取HTTPS Token认证机制。
"""

import base64
import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
import urllib3

from utils.exceptions import InferenceAPIError, ValidationError
from utils.logger import get_logger

logger = get_logger("token_auth")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class TokenConfig:
    """Token认证配置"""

    access_key: str
    secret_key: str
    tenant_id: int = 1002
    token_expire_seconds: int = 3600
    retry_count: int = 3
    retry_delay: float = 1.0
    token_url_template: str = "https://{host}:{port}/gam/v1/auth/tokens"

    def validate(self) -> bool:
        """验证配置"""
        return bool(self.access_key and self.secret_key)


@dataclass
class TokenInfo:
    """Token信息"""

    token: Optional[str] = None
    acquired_time: float = 0.0
    expires_at: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Token是否有效"""
        return bool(self.token and time.time() < self.expires_at)

    @property
    def is_expired(self) -> bool:
        """Token是否过期"""
        return time.time() >= self.expires_at

    def clear(self):
        """清除Token信息"""
        self.token = None
        self.acquired_time = 0.0
        self.expires_at = 0.0


class TokenAuthenticator:
    """Token认证器"""

    def __init__(self, config: TokenConfig):
        if not config.validate():
            raise ValidationError("Token配置验证失败：access_key和secret_key不能为空")

        self.config = config
        self._token_info = TokenInfo()
        self._lock = threading.RLock()
        self._session = requests.Session()

    def _build_token_url(self, endpoint: str) -> str:
        """构建token获取URL"""
        parsed = urlparse(endpoint)
        host = parsed.hostname or ""
        port = parsed.port or (38443 if parsed.scheme == "https" else 80)

        return self.config.token_url_template.format(host=host, port=port)

    def _build_token_request(self, endpoint: str) -> tuple[str, Dict[str, str], str]:
        """构建token请求"""
        token_url = self._build_token_url(endpoint)

        # Basic认证头
        auth_string = f"{self.config.access_key}:{self.config.secret_key}"
        auth_b64 = base64.b64encode(auth_string.encode()).decode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth_b64}",
        }

        # 请求体
        parsed = urlparse(endpoint)
        client_ip = parsed.hostname or ""
        payload = {
            "identity": {
                "methods": ["basic"],
                "access": {"key": self.config.access_key},
            },
            "clientIp": client_ip,
            "tenantId": self.config.tenant_id,
        }

        return token_url, headers, json.dumps(payload)

    def _fetch_token(self, endpoint: str) -> bool:
        """获取新token"""
        try:
            token_url, headers, data = self._build_token_request(endpoint)

            last_error = None
            for attempt in range(self.config.retry_count):
                try:
                    response = self._session.post(
                        token_url, headers=headers, data=data, verify=False, timeout=30
                    )
                    response.raise_for_status()

                    # 从响应头获取token
                    token = response.headers.get("X-Subject-Token")
                    if not token:
                        raise InferenceAPIError("响应中未找到X-Subject-Token头")

                    # 更新token信息
                    current_time = time.time()
                    self._token_info.token = token
                    self._token_info.acquired_time = current_time
                    self._token_info.expires_at = (
                        current_time + self.config.token_expire_seconds
                    )

                    logger.info(f"Token获取成功: {endpoint}")
                    return True

                except Exception as e:
                    last_error = e
                    if attempt < self.config.retry_count - 1:
                        logger.warning(f"Token获取失败，第{attempt + 1}次重试: {e}")
                        time.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        logger.error(f"Token获取最终失败: {e}")

            raise InferenceAPIError(f"Token获取失败: {last_error}")

        except Exception as e:
            logger.error(f"Token认证异常: {e}")
            self._token_info.clear()
            raise

    def get_valid_token(self, endpoint: str) -> str:
        """获取有效的token（自动刷新过期token）"""
        with self._lock:
            if self._token_info.is_valid:
                return self._token_info.token

            if self._token_info.token:
                logger.info("Token已过期，重新获取")
            else:
                logger.info("首次获取Token")

            self._fetch_token(endpoint)

            if not self._token_info.is_valid:
                raise InferenceAPIError("获取有效Token失败")

            return self._token_info.token

    def get_auth_headers(self, endpoint: str) -> Dict[str, str]:
        """获取HTTP认证头"""
        token = self.get_valid_token(endpoint)
        return {"Authorization": token}

    def get_auth_metadata(self, endpoint: str) -> Dict[str, str]:
        """获取gRPC认证元数据"""
        token = self.get_valid_token(endpoint)
        return {"authorization": token}

    def clear_token(self):
        """清除token（强制重新获取）"""
        with self._lock:
            self._token_info.clear()
            logger.info("Token已清除")

    def is_authenticated(self) -> bool:
        """检查是否已认证"""
        with self._lock:
            return self._token_info.is_valid


# 兼容性函数（模拟backup项目中的原有函数）
def get_token_if_expired(
    endpoint: str, access_key: str, secret_key: str, tenant_id: int = 1002
) -> str:
    """
    兼容backup项目的token获取函数

    Args:
        endpoint: 推理接口地址
        access_key: 访问密钥
        secret_key: 秘密密钥
        tenant_id: 租户ID

    Returns:
        str: 有效的token
    """
    config = TokenConfig(
        access_key=access_key, secret_key=secret_key, tenant_id=tenant_id
    )
    authenticator = TokenAuthenticator(config)
    return authenticator.get_valid_token(endpoint)


def setup_token_auth(
    access_key: str, secret_key: str, tenant_id: int = 1002, **config_kwargs
) -> TokenAuthenticator:
    """
    设置Token认证的便捷函数

    Args:
        access_key: 访问密钥
        secret_key: 秘密密钥
        tenant_id: 租户ID
        **config_kwargs: 其他配置参数

    Returns:
        TokenAuthenticator: 认证器实例
    """
    config = TokenConfig(
        access_key=access_key,
        secret_key=secret_key,
        tenant_id=tenant_id,
        **config_kwargs,
    )
    return TokenAuthenticator(config)
