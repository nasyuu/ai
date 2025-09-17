"""
配置管理模块

提供统一的配置管理功能，支持多种配置源（文件、环境变量、命令行参数）
和动态配置更新，为后续功能扩充提供灵活的配置基础。
"""

import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from utils.exceptions import ConfigurationError, ValidationError
from utils.logger import get_logger

logger = get_logger("config")

T = TypeVar("T")


class ConfigSource(Enum):
    """配置源类型"""

    FILE = "file"
    ENV = "environment"
    ARGS = "arguments"
    DEFAULT = "default"


@dataclass
class ConfigItem:
    """配置项定义"""

    key: str
    value_type: Type
    default_value: Any = None
    description: str = ""
    required: bool = False
    validator: Optional[callable] = None
    env_var: Optional[str] = None
    source: ConfigSource = ConfigSource.DEFAULT
    choices: Optional[List[Any]] = None


class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate_positive_int(value: Any) -> bool:
        """验证正整数"""
        return isinstance(value, int) and value > 0

    @staticmethod
    def validate_positive_float(value: Any) -> bool:
        """验证正浮点数"""
        return isinstance(value, (int, float)) and value > 0

    @staticmethod
    def validate_url(value: Any) -> bool:
        """验证URL格式"""
        if not isinstance(value, str):
            return False
        return value.startswith(("http://", "https://")) or ":" in value

    @staticmethod
    def validate_path(value: Any) -> bool:
        """验证路径存在"""
        if not isinstance(value, str):
            return False
        return Path(value).exists()

    @staticmethod
    def validate_dir_path(value: Any) -> bool:
        """验证目录路径"""
        if not isinstance(value, str):
            return False
        path = Path(value)
        return path.exists() and path.is_dir()

    @staticmethod
    def validate_file_path(value: Any) -> bool:
        """验证文件路径"""
        if not isinstance(value, str):
            return False
        path = Path(value)
        return path.exists() and path.is_file()

    @staticmethod
    def validate_choice(choices: List[Any]):
        """验证值在指定选择范围内"""

        def validator(value: Any) -> bool:
            return value in choices

        return validator


class BaseConfigSchema(ABC):
    """配置模式基类"""

    @abstractmethod
    def get_config_items(self) -> List[ConfigItem]:
        """获取配置项定义"""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置完整性"""
        pass


class InferenceConfigSchema(BaseConfigSchema):
    """推理配置模式"""

    def get_config_items(self) -> List[ConfigItem]:
        return [
            # 通用配置
            ConfigItem(
                key="inference_type",
                value_type=str,
                default_value="https",
                description="推理接口类型",
                required=True,
                choices=["https", "grpc", "grpc_standard"],
            ),
            ConfigItem(
                key="max_workers",
                value_type=int,
                default_value=1,
                description="最大并发线程数",
                validator=ConfigValidator.validate_positive_int,
            ),
            ConfigItem(
                key="timeout",
                value_type=int,
                default_value=60,
                description="请求超时时间（秒）",
                validator=ConfigValidator.validate_positive_int,
            ),
            ConfigItem(
                key="retry_max",
                value_type=int,
                default_value=3,
                description="最大重试次数",
                validator=ConfigValidator.validate_positive_int,
            ),
            # HTTPS配置
            ConfigItem(
                key="https_url",
                value_type=str,
                default_value="",
                description="HTTPS推理接口URL",
                env_var="INFERENCE_HTTPS_URL",
            ),
            ConfigItem(
                key="access_key",
                value_type=str,
                default_value="",
                description="访问密钥",
                env_var="INFERENCE_ACCESS_KEY",
            ),
            ConfigItem(
                key="secret_key",
                value_type=str,
                default_value="",
                description="秘密密钥",
                env_var="INFERENCE_SECRET_KEY",
            ),
            # gRPC配置
            ConfigItem(
                key="grpc_address",
                value_type=str,
                default_value="",
                description="gRPC服务器地址",
                env_var="INFERENCE_GRPC_ADDRESS",
            ),
            ConfigItem(
                key="task_id",
                value_type=str,
                default_value="",
                description="任务ID",
                env_var="INFERENCE_TASK_ID",
            ),
            ConfigItem(
                key="stream_name",
                value_type=str,
                default_value="",
                description="流名称",
                env_var="INFERENCE_STREAM_NAME",
            ),
            # 路径配置
            ConfigItem(
                key="images_dir",
                value_type=str,
                default_value="",
                description="输入图片目录",
                required=True,
            ),
            ConfigItem(
                key="output_dir",
                value_type=str,
                default_value="output",
                description="输出结果目录",
            ),
            ConfigItem(
                key="gt_jsons_dir",
                value_type=str,
                default_value="",
                description="真值标注目录",
            ),
        ]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证推理配置"""
        inference_type = config.get("inference_type")

        if inference_type == "https":
            if not config.get("https_url"):
                raise ValidationError("HTTPS模式下必须提供https_url")
        elif inference_type in ["grpc", "grpc_standard"]:
            if not config.get("grpc_address"):
                raise ValidationError("gRPC模式下必须提供grpc_address")

        return True


class EvaluationConfigSchema(BaseConfigSchema):
    """评估配置模式"""

    def get_config_items(self) -> List[ConfigItem]:
        return [
            ConfigItem(
                key="eval_enabled",
                value_type=bool,
                default_value=True,
                description="是否启用评估",
            ),
            ConfigItem(
                key="iou_threshold",
                value_type=float,
                default_value=0.5,
                description="IoU阈值",
                validator=lambda x: 0 < x <= 1,
            ),
            ConfigItem(
                key="confidence_threshold",
                value_type=float,
                default_value=0.5,
                description="置信度阈值",
                validator=lambda x: 0 < x <= 1,
            ),
            ConfigItem(
                key="eval_metrics",
                value_type=list,
                default_value=["precision", "recall", "f1", "accuracy"],
                description="评估指标",
            ),
            ConfigItem(
                key="visualization_enabled",
                value_type=bool,
                default_value=True,
                description="是否启用可视化",
            ),
            ConfigItem(
                key="save_reports",
                value_type=bool,
                default_value=True,
                description="是否保存评估报告",
            ),
        ]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证评估配置"""
        if config.get("eval_enabled") and not config.get("gt_jsons_dir"):
            raise ValidationError("启用评估时必须提供真值标注目录")
        return True


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._config_items: Dict[str, ConfigItem] = {}
        self._schemas: List[BaseConfigSchema] = []
        self._lock = threading.RLock()
        self._watchers: List[callable] = []

    def register_schema(self, schema: BaseConfigSchema):
        """注册配置模式"""
        with self._lock:
            self._schemas.append(schema)
            for item in schema.get_config_items():
                self._config_items[item.key] = item
                # 设置默认值
                if item.key not in self._config:
                    self._config[item.key] = item.default_value

    def load_from_file(self, file_path: Union[str, Path], format: str = "auto"):
        """从文件加载配置"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise ConfigurationError(f"配置文件不存在: {file_path}")

        # 自动检测格式
        if format == "auto":
            if file_path.suffix.lower() == ".json":
                format = "json"
            elif file_path.suffix.lower() in [".yml", ".yaml"]:
                format = "yaml"
            else:
                format = "json"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if format == "json":
                    data = json.load(f)
                elif format == "yaml":
                    try:
                        import yaml

                        data = yaml.safe_load(f)
                    except ImportError:
                        raise ConfigurationError(
                            "YAML格式需要安装PyYAML: pip install PyYAML"
                        )
                else:
                    raise ConfigurationError(f"不支持的配置文件格式: {format}")

            self._update_config(data, ConfigSource.FILE)
            logger.info(f"从文件加载配置: {file_path}")

        except Exception as e:
            raise ConfigurationError(f"加载配置文件失败: {e}", str(file_path))

    def load_from_env(self):
        """从环境变量加载配置"""
        env_config = {}
        for item in self._config_items.values():
            if item.env_var and item.env_var in os.environ:
                value = os.environ[item.env_var]
                # 类型转换
                try:
                    if item.value_type is bool:
                        value = value.lower() in ("true", "1", "yes", "on")
                    elif item.value_type is int:
                        value = int(value)
                    elif item.value_type is float:
                        value = float(value)
                    elif item.value_type is list:
                        value = value.split(",")
                    env_config[item.key] = value
                except ValueError:
                    logger.warning(f"环境变量{item.env_var}类型转换失败，跳过")

        if env_config:
            self._update_config(env_config, ConfigSource.ENV)
            logger.info(f"从环境变量加载配置: {len(env_config)}项")

    def set(self, key: str, value: Any, validate: bool = True):
        """设置配置项"""
        with self._lock:
            if validate and key in self._config_items:
                self._validate_item(key, value)

            old_value = self._config.get(key)
            self._config[key] = value

            # 通知观察者
            if old_value != value:
                self._notify_watchers(key, value, old_value)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        with self._lock:
            return self._config.get(key, default)

    def get_typed(self, key: str, value_type: Type[T], default: T = None) -> T:
        """获取指定类型的配置项"""
        value = self.get(key, default)
        if value is not None and not isinstance(value, value_type):
            try:
                value = value_type(value)
            except (ValueError, TypeError):
                raise ConfigurationError(
                    f"配置项{key}类型转换失败", key, str(value_type)
                )
        return value

    def get_section(self, prefix: str) -> Dict[str, Any]:
        """获取配置段"""
        with self._lock:
            return {
                k[len(prefix) + 1 :]: v
                for k, v in self._config.items()
                if k.startswith(prefix + ".")
            }

    def validate(self) -> bool:
        """验证所有配置"""
        with self._lock:
            # 验证必需项
            for item in self._config_items.values():
                if item.required and self._config.get(item.key) is None:
                    raise ValidationError(f"必需配置项缺失: {item.key}")

            # 验证各个模式
            for schema in self._schemas:
                schema.validate_config(self._config)

            logger.info("配置验证通过")
            return True

    def add_watcher(self, callback: callable):
        """添加配置变更观察者"""
        self._watchers.append(callback)

    def remove_watcher(self, callback: callable):
        """移除配置变更观察者"""
        if callback in self._watchers:
            self._watchers.remove(callback)

    def save_to_file(self, file_path: Union[str, Path], format: str = "json"):
        """保存配置到文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                if format == "json":
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
                elif format == "yaml":
                    try:
                        import yaml

                        yaml.dump(
                            self._config,
                            f,
                            default_flow_style=False,
                            allow_unicode=True,
                        )
                    except ImportError:
                        raise ConfigurationError(
                            "YAML格式需要安装PyYAML: pip install PyYAML"
                        )
                else:
                    raise ConfigurationError(f"不支持的保存格式: {format}")

            logger.info(f"配置已保存到: {file_path}")

        except Exception as e:
            raise ConfigurationError(f"保存配置文件失败: {e}", str(file_path))

    def _update_config(self, data: Dict[str, Any], source: ConfigSource):
        """更新配置"""
        with self._lock:
            for key, value in data.items():
                if key in self._config_items:
                    self._config_items[key].source = source

                old_value = self._config.get(key)
                self._config[key] = value

                if old_value != value:
                    self._notify_watchers(key, value, old_value)

    def _validate_item(self, key: str, value: Any):
        """验证单个配置项"""
        item = self._config_items.get(key)
        if not item:
            return

        # 类型检查
        if value is not None and not isinstance(value, item.value_type):
            if item.value_type is bool and isinstance(value, str):
                # 特殊处理布尔类型的字符串
                value = value.lower() in ("true", "1", "yes", "on")
            else:
                try:
                    value = item.value_type(value)
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"配置项{key}类型不匹配", key, value, str(item.value_type)
                    )

        # 选择检查
        if item.choices and value not in item.choices:
            raise ValidationError(
                f"配置项{key}值不在允许范围内", key, value, str(item.choices)
            )

        # 自定义验证器
        if item.validator and not item.validator(value):
            raise ValidationError(f"配置项{key}验证失败", key, value)

    def _notify_watchers(self, key: str, new_value: Any, old_value: Any):
        """通知观察者配置变更"""
        for watcher in self._watchers:
            try:
                watcher(key, new_value, old_value)
            except Exception as e:
                logger.error(f"配置变更通知失败: {e}")


# 全局配置管理器实例
config_manager = ConfigManager()

# 注册默认配置模式
config_manager.register_schema(InferenceConfigSchema())
config_manager.register_schema(EvaluationConfigSchema())


def get_config() -> ConfigManager:
    """获取配置管理器实例"""
    return config_manager


def init_config(config_file: Optional[str] = None):
    """初始化配置"""
    # 加载环境变量
    config_manager.load_from_env()

    # 加载配置文件
    if config_file:
        config_manager.load_from_file(config_file)
    else:
        # 查找默认配置文件
        default_files = ["config.json", "config.yaml", "config.yml"]
        for file in default_files:
            if Path(file).exists():
                config_manager.load_from_file(file)
                break

    # 验证配置
    config_manager.validate()

    logger.info("配置初始化完成")
