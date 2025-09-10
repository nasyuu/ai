from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Protocol
from urllib.parse import urlparse

import requests
import urllib3

from utils.exception import AuthError, IOErrorEx, ensure
from utils.logger import get_logger, record_fail

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
log = get_logger("client.auth")


class TokenProvider(Protocol):
    """统一令牌获取接口，便于不同协议（HTTPS/gRPC）公用。"""

    def get_token(self) -> str:
        """获取（或刷新）可用 token。"""
        ...


@dataclass
class BasicTokenConfig:
    """
    通用的 ak/sk 认证配置。
    endpoint: 业务推理地址，用于推导 token 接口和 clientIp
    token_path: 认证接口路径，默认 /gam/v1/auth/tokens
    expire_seconds: 认为 token 过期的时长（服务端也常为1h）
    """

    endpoint: str
    access_key: str
    secret_key: str
    tenant_id: int = 1002
    token_path: str = "/gam/v1/auth/tokens"
    expire_seconds: int = 3600
    timeout: int = 15

    def validate(self):
        ensure(bool(self.endpoint), "endpoint 不能为空", code="auth.endpoint")
        ensure(bool(self.access_key), "access_key 不能为空", code="auth.ak")
        ensure(bool(self.secret_key), "secret_key 不能为空", code="auth.sk")


class BasicTokenProvider(TokenProvider):
    """
    使用 ak/sk 换取 X-Subject-Token 的 Provider。
    - 线程安全缓存
    - 过期自动刷新
    - 请求异常/IO 异常结构化抛出
    """

    def __init__(
        self, cfg: BasicTokenConfig, session: Optional[requests.Session] = None
    ):
        cfg.validate()
        self.cfg = cfg
        self._session = session or requests.Session()
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._acquired_at: float = 0.0

    # ---------- 内部工具 ----------

    def _normalize_url(self, url: str) -> str:
        if not url:
            return url
        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://{url.lstrip('/')}"
        return url

    def _build_token_url(self) -> str:
        base = self._normalize_url(self.cfg.endpoint)
        parsed = urlparse(base)
        host = parsed.hostname or ""
        port = parsed.port or (38443 if parsed.scheme == "https" else 80)
        return f"{parsed.scheme}://{host}:{port}{self.cfg.token_path}"

    def _build_token_body(self) -> Dict:
        base = self._normalize_url(self.cfg.endpoint)
        parsed = urlparse(base)
        ip = parsed.hostname or ""
        return {
            "identity": {"methods": ["basic"], "access": {"key": self.cfg.access_key}},
            "clientIp": ip,
            "tenantId": self.cfg.tenant_id,
        }

    def _refresh(self) -> str:
        url = self._build_token_url()
        body = json.dumps(self._build_token_body())
        auth = base64.b64encode(
            f"{self.cfg.access_key}:{self.cfg.secret_key}".encode("utf-8")
        ).decode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Basic {auth}"}

        try:
            resp = self._session.post(
                url, headers=headers, data=body, verify=False, timeout=self.cfg.timeout
            )
        except Exception as e:
            record_fail(f"[AUTH_IO] 请求失败 {url} | {e}")
            raise IOErrorEx("认证请求失败", cause=e, path=url)

        if not (200 <= resp.status_code < 300):
            txt = resp.text[:2000] if resp.text else ""
            record_fail(f"[AUTH_HTTP] {resp.status_code} {url} | {txt}")
            raise AuthError(f"认证失败: HTTP {resp.status_code} | {txt}")

        token = resp.headers.get("X-Subject-Token")
        if not token:
            record_fail("[AUTH_PARSE] 响应缺少 X-Subject-Token")
            raise AuthError("认证失败：缺少 X-Subject-Token")

        self._token = token
        self._acquired_at = time.time()
        log.info("Token 获取成功")
        return token

    # ---------- 对外接口 ----------

    def get_token(self) -> str:
        # 未过期直接返回
        if self._token and (time.time() - self._acquired_at) <= self.cfg.expire_seconds:
            return self._token

        with self._lock:
            if (
                self._token
                and (time.time() - self._acquired_at) <= self.cfg.expire_seconds
            ):
                return self._token
            return self._refresh()
