"""
认证模块使用示例

展示如何使用统一认证模块进行各种类型的认证。
"""

from client.auth import (
    AuthCredentials,
    AuthType,
    create_api_key_auth,
    create_basic_auth,
    create_no_auth,
    create_token_auth,
    get_auth_manager,
)
from utils.logger import setup_logger

logger = setup_logger("auth_example")


def example_token_auth():
    """Token认证示例（类似backup项目中的HTTPS认证）"""
    logger.info("=== Token认证示例 ===")

    # 创建认证管理器
    auth_manager = get_auth_manager()

    # 创建Token认证凭据
    credentials = create_token_auth(
        access_key="your_access_key", secret_key="your_secret_key", tenant_id=1002
    )

    # 注册认证器
    auth_manager.register_auth("https_api", credentials)

    # 模拟推理接口地址
    endpoint = "https://192.168.1.100:8080/api/inference"

    try:
        # 执行认证
        auth_result = auth_manager.authenticate("https_api", endpoint)

        if auth_result.is_valid:
            logger.info("Token认证成功")

            # 获取认证头（用于HTTP请求）
            headers = auth_manager.get_auth_headers("https_api")
            logger.info(f"HTTP认证头: {headers}")

            # 获取认证元数据（用于gRPC请求）
            metadata = auth_manager.get_auth_metadata("https_api")
            logger.info(f"gRPC认证元数据: {metadata}")

        else:
            logger.error(f"Token认证失败: {auth_result.error_message}")

    except Exception as e:
        logger.error(f"认证异常: {e}")


def example_basic_auth():
    """Basic认证示例"""
    logger.info("=== Basic认证示例 ===")

    auth_manager = get_auth_manager()

    # 创建Basic认证凭据
    credentials = create_basic_auth("username", "password")

    # 注册认证器
    auth_manager.register_auth("basic_api", credentials)

    endpoint = "https://api.example.com"

    try:
        # 确保认证有效（会自动处理认证流程）
        auth_result = auth_manager.ensure_authenticated("basic_api", endpoint)

        if auth_result.is_valid:
            logger.info("Basic认证成功")
            headers = auth_manager.get_auth_headers("basic_api")
            logger.info(f"认证头: {headers}")
        else:
            logger.error(f"Basic认证失败: {auth_result.error_message}")

    except Exception as e:
        logger.error(f"认证异常: {e}")


def example_api_key_auth():
    """API Key认证示例"""
    logger.info("=== API Key认证示例 ===")

    auth_manager = get_auth_manager()

    # 创建API Key认证凭据
    credentials = create_api_key_auth("your-api-key-here")

    # 注册认证器（可以指定自定义头名称）
    auth_manager.register_auth(
        "api_key_service", credentials, header_name="X-Custom-API-Key"
    )

    endpoint = "https://api.service.com"

    try:
        auth_result = auth_manager.ensure_authenticated("api_key_service", endpoint)

        if auth_result.is_valid:
            logger.info("API Key认证成功")
            headers = auth_manager.get_auth_headers("api_key_service")
            logger.info(f"认证头: {headers}")
        else:
            logger.error(f"API Key认证失败: {auth_result.error_message}")

    except Exception as e:
        logger.error(f"认证异常: {e}")


def example_no_auth():
    """无认证示例（用于gRPC等不需要认证的接口）"""
    logger.info("=== 无认证示例 ===")

    auth_manager = get_auth_manager()

    # 创建无认证凭据
    credentials = create_no_auth()

    # 注册认证器
    auth_manager.register_auth("grpc_service", credentials)

    endpoint = "grpc://192.168.1.100:9090"

    try:
        auth_result = auth_manager.ensure_authenticated("grpc_service", endpoint)

        if auth_result.is_valid:
            logger.info("无认证模式")
            headers = auth_manager.get_auth_headers("grpc_service")
            metadata = auth_manager.get_auth_metadata("grpc_service")
            logger.info(f"认证头: {headers}")
            logger.info(f"认证元数据: {metadata}")
        else:
            logger.error("无认证模式失败（不应该发生）")

    except Exception as e:
        logger.error(f"认证异常: {e}")


def example_auth_management():
    """认证管理示例"""
    logger.info("=== 认证管理示例 ===")

    auth_manager = get_auth_manager()

    # 注册多个认证器
    auth_manager.register_auth("service1", create_token_auth("ak1", "sk1"))
    auth_manager.register_auth("service2", create_basic_auth("user", "pass"))
    auth_manager.register_auth("service3", create_api_key_auth("api-key"))
    auth_manager.register_auth("service4", create_no_auth())

    # 列出所有认证器
    authenticators = auth_manager.list_authenticators()
    logger.info(f"已注册的认证器: {authenticators}")

    # 清除特定认证
    auth_manager.clear_auth("service1")
    logger.info("已清除service1的认证状态")

    # 移除认证器
    auth_manager.remove_authenticator("service2")
    logger.info("已移除service2认证器")

    # 再次列出认证器
    authenticators = auth_manager.list_authenticators()
    logger.info(f"当前认证器: {authenticators}")


def example_integration_with_requests():
    """与requests库集成示例"""
    logger.info("=== 与requests集成示例 ===")

    auth_manager = get_auth_manager()

    # 注册Token认证
    credentials = create_token_auth("test_ak", "test_sk")
    auth_manager.register_auth("inference_api", credentials)

    endpoint = "https://api.inference.com/predict"

    try:
        # 确保认证有效
        auth_result = auth_manager.ensure_authenticated("inference_api", endpoint)

        if auth_result.is_valid:
            # 获取认证头
            auth_headers = auth_manager.get_auth_headers("inference_api")

            # 准备请求
            headers = {
                "Content-Type": "application/json",
                **auth_headers,  # 合并认证头
            }

            # 发送请求（这里只是示例，不会真正发送）
            logger.info(f"准备发送请求到: {endpoint}")
            logger.info(f"请求头: {headers}")
            logger.info("请求数据: [图像数据]")

            # 实际使用时的代码示例：
            # import requests
            # data = {"image_base64": "base64_encoded_image_data"}
            # response = requests.post(endpoint, headers=headers, json=data)
            # if response.status_code == 200:
            #     result = response.json()
            #     logger.info(f"推理结果: {result}")

        else:
            logger.error(f"认证失败，无法发送请求: {auth_result.error_message}")

    except Exception as e:
        logger.error(f"请求异常: {e}")


def example_integration_with_grpc():
    """与gRPC集成示例"""
    logger.info("=== 与gRPC集成示例 ===")

    auth_manager = get_auth_manager()

    # 对于需要认证的gRPC服务，可以使用Token或API Key认证
    credentials = create_token_auth("grpc_ak", "grpc_sk")
    auth_manager.register_auth("grpc_inference", credentials)

    endpoint = "grpc://192.168.1.100:9090"

    try:
        # 确保认证有效
        auth_result = auth_manager.ensure_authenticated("grpc_inference", endpoint)

        if auth_result.is_valid:
            # 获取认证元数据
            auth_metadata = auth_manager.get_auth_metadata("grpc_inference")

            # 转换为gRPC metadata格式
            metadata = [(k, v) for k, v in auth_metadata.items()]

            logger.info(f"gRPC认证元数据: {metadata}")

            # 在gRPC调用中使用
            # response = stub.Inference(request, metadata=metadata)

        else:
            logger.error(f"gRPC认证失败: {auth_result.error_message}")

    except Exception as e:
        logger.error(f"gRPC认证异常: {e}")


def example_custom_authenticator():
    """自定义认证器示例"""
    logger.info("=== 自定义认证器示例 ===")

    from client.auth import (
        AuthenticatorFactory,
        AuthResult,
        AuthStatus,
        BaseAuthenticator,
    )

    class CustomAuthenticator(BaseAuthenticator):
        """自定义认证器示例"""

        def authenticate(self, endpoint: str) -> AuthResult:
            # 实现自定义认证逻辑
            logger.info(f"执行自定义认证: {endpoint}")

            # 模拟认证成功
            return AuthResult(
                status=AuthStatus.AUTHENTICATED,
                headers={"X-Custom-Auth": "custom_token_value"},
                metadata={"custom-auth": "custom_token_value"},
            )

        def refresh_auth(self, endpoint: str) -> AuthResult:
            return self.authenticate(endpoint)

        def get_auth_headers(self) -> dict:
            if self._auth_result and self._auth_result.is_valid:
                return self._auth_result.headers.copy()
            return {}

        def get_auth_metadata(self) -> dict:
            if self._auth_result and self._auth_result.is_valid:
                return self._auth_result.metadata.copy()
            return {}

    # 注册自定义认证器
    AuthenticatorFactory.register_authenticator(AuthType.CUSTOM, CustomAuthenticator)

    # 使用自定义认证器
    auth_manager = get_auth_manager()

    credentials = AuthCredentials(auth_type=AuthType.CUSTOM)
    auth_manager.register_auth("custom_service", credentials)

    endpoint = "https://custom.api.com"

    try:
        auth_result = auth_manager.ensure_authenticated("custom_service", endpoint)

        if auth_result.is_valid:
            logger.info("自定义认证成功")
            headers = auth_manager.get_auth_headers("custom_service")
            logger.info(f"自定义认证头: {headers}")
        else:
            logger.error(f"自定义认证失败: {auth_result.error_message}")

    except Exception as e:
        logger.error(f"自定义认证异常: {e}")


def main():
    """运行所有示例"""
    logger.info("开始认证模块示例演示")

    try:
        # 运行各种认证示例
        example_token_auth()
        print()

        example_basic_auth()
        print()

        example_api_key_auth()
        print()

        example_no_auth()
        print()

        example_auth_management()
        print()

        example_integration_with_requests()
        print()

        example_integration_with_grpc()
        print()

        example_custom_authenticator()
        print()

        logger.info("所有示例演示完成")

    except Exception as e:
        logger.error(f"示例运行异常: {e}")


if __name__ == "__main__":
    main()
