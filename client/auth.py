"""认证与令牌管理工具。"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests

from utils.exceptions import ExternalServiceError, ValidationError

__all__ = [
    "HTTPSAuthConfig",
    "HTTPSAuthenticator",
]


def _normalize_url(url: str) -> str:
    if not url:
        raise ValidationError("认证地址不能为空")
    parsed = urlparse(url)
    if parsed.scheme:
        return url
    return f"https://{url.lstrip('/')}"


@dataclass(frozen=True)
class HTTPSAuthConfig:
    """HTTPS 鉴权配置。"""

    base_url: str
    access_key: str
    secret_key: str
    tenant_id: int = 1002
    token_endpoint: str = "/gam/v1/auth/tokens"
    verify: bool = False
    timeout: int = 15

    def normalised_base(self) -> str:
        return _normalize_url(self.base_url)

    def token_url(self) -> str:
        parsed = urlparse(self.normalised_base())
        scheme = parsed.scheme or "https"
        host = parsed.hostname or ""
        port = parsed.port or (38443 if scheme == "https" else 80)
        return f"{scheme}://{host}:{port}{self.token_endpoint}"

    def client_ip(self) -> str:
        return urlparse(self.normalised_base()).hostname or ""


class HTTPSAuthenticator:
    """负责获取与缓存 HTTPS API 所需 token。"""

    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        logger: Optional[logging.Logger] = None,
        token_ttl: int = 3600,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self._session = session or requests.Session()
        self._logger = logger or logging.getLogger(__name__)
        self._token_ttl = token_ttl
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._token: Optional[str] = None
        self._acquired_at: float = 0.0
        self._config_key: Optional[tuple[str, str, int]] = None
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            self._token = None
            self._acquired_at = 0.0
            self._config_key = None

    def get_token(self, config: HTTPSAuthConfig, *, force_refresh: bool = False) -> str:
        key = (config.token_url(), config.access_key, config.tenant_id)
        if not force_refresh and self._is_token_valid(key):
            return self._token  # type: ignore[return-value]

        with self._lock:
            if not force_refresh and self._is_token_valid(key):
                return self._token  # type: ignore[return-value]
            token = self._request_token(config)
            self._token = token
            self._acquired_at = time.time()
            self._config_key = key
            return token

    def _is_token_valid(self, key: tuple[str, str, int]) -> bool:
        if self._token is None or self._config_key != key:
            return False
        return (time.time() - self._acquired_at) < self._token_ttl

    def _request_token(self, config: HTTPSAuthConfig) -> str:
        payload = {
            "identity": {"methods": ["basic"], "access": {"key": config.access_key}},
            "clientIp": config.client_ip(),
            "tenantId": config.tenant_id,
        }
        basic = base64.b64encode(
            f"{config.access_key}:{config.secret_key}".encode("utf-8")
        ).decode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {basic}",
        }

        last_error: Optional[Exception] = None
        backoff = self._retry_backoff
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.post(
                    config.token_url(),
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=config.timeout,
                    verify=config.verify,
                )
                response.raise_for_status()
                token = response.headers.get("X-Subject-Token")
                if not token:
                    raise ExternalServiceError(
                        "认证响应缺少 token",
                        details={"url": config.token_url()},
                    )
                self._logger.info("Token 获取成功")
                return token
            except requests.RequestException as exc:  # type: ignore[union-attr]
                last_error = exc
                self._logger.warning(
                    "第%s次请求 token 失败: %s", attempt, exc, exc_info=False
                )
            except ExternalServiceError as exc:
                raise exc
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._logger.warning(
                    "第%s次请求 token 失败: %s", attempt, exc, exc_info=False
                )

            if attempt < self._max_retries:
                time.sleep(backoff)
                backoff *= 2

        raise ExternalServiceError(
            "获取 token 失败",
            details={"url": config.token_url()},
            service="auth",
        ) from last_error

    def auth_header(self, config: HTTPSAuthConfig) -> dict[str, str]:
        token = self.get_token(config)
        return {"Authorization": token}
