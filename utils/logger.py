"""
日志模块

为图像流推理与评估工具提供统一的日志配置和管理功能。
支持文件日志、控制台日志、不同级别的日志输出等。
"""

import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional


class LogLevel(Enum):
    """日志级别枚举"""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LoggerConfig:
    """日志配置类"""

    def __init__(
        self,
        name: str = "inference_tool",
        level: LogLevel = LogLevel.INFO,
        log_dir: str = "logs",
        enable_console: bool = True,
        enable_file: bool = True,
        file_prefix: str = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        format_string: str = None,
    ):
        """
        初始化日志配置

        Args:
            name: 日志器名称
            level: 日志级别
            log_dir: 日志文件目录
            enable_console: 是否启用控制台输出
            enable_file: 是否启用文件输出
            file_prefix: 日志文件前缀
            max_file_size: 日志文件最大大小（字节）
            backup_count: 日志文件备份数量
            format_string: 自定义日志格式
        """
        self.name = name
        self.level = level
        self.log_dir = Path(log_dir)
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.file_prefix = file_prefix or name
        self.max_file_size = max_file_size
        self.backup_count = backup_count

        # 默认日志格式
        if format_string is None:
            self.format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            self.format_string = format_string


class InferenceLogger:
    """推理工具日志器"""

    _loggers: Dict[str, logging.Logger] = {}
    _default_config: Optional[LoggerConfig] = None

    @classmethod
    def set_default_config(cls, config: LoggerConfig):
        """设置默认日志配置"""
        cls._default_config = config

    @classmethod
    def get_logger(
        cls, name: str = None, config: LoggerConfig = None
    ) -> logging.Logger:
        """
        获取日志器实例

        Args:
            name: 日志器名称，如果为None则使用默认配置的名称
            config: 日志配置，如果为None则使用默认配置

        Returns:
            logging.Logger: 配置好的日志器实例
        """
        # 使用默认配置
        if config is None:
            config = cls._default_config or LoggerConfig()

        # 使用配置中的名称
        if name is None:
            name = config.name

        # 如果已存在同名日志器，直接返回
        if name in cls._loggers:
            return cls._loggers[name]

        # 创建新的日志器
        logger = cls._create_logger(name, config)
        cls._loggers[name] = logger

        return logger

    @classmethod
    def _create_logger(cls, name: str, config: LoggerConfig) -> logging.Logger:
        """
        创建日志器

        Args:
            name: 日志器名称
            config: 日志配置

        Returns:
            logging.Logger: 配置好的日志器
        """
        logger = logging.getLogger(name)
        logger.setLevel(config.level.value)

        # 防止重复添加处理器
        if logger.handlers:
            logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter(config.format_string)

        # 添加控制台处理器
        if config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(config.level.value)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 添加文件处理器
        if config.enable_file:
            file_handler = cls._create_file_handler(config, formatter)
            logger.addHandler(file_handler)

        # 防止日志传播到父日志器
        logger.propagate = False

        return logger

    @classmethod
    def _create_file_handler(cls, config: LoggerConfig, formatter: logging.Formatter):
        """
        创建文件处理器

        Args:
            config: 日志配置
            formatter: 日志格式化器

        Returns:
            logging.Handler: 文件处理器
        """
        from logging.handlers import RotatingFileHandler

        # 确保日志目录存在
        config.log_dir.mkdir(parents=True, exist_ok=True)

        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{config.file_prefix}_{timestamp}.log"
        log_path = config.log_dir / log_filename

        # 创建轮转文件处理器
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(config.level.value)
        file_handler.setFormatter(formatter)

        return file_handler

    @classmethod
    def setup_for_executable(cls, name: str = "inference_tool", log_dir: str = "logs"):
        """
        为可执行文件设置日志（仅文件输出，不输出到控制台）

        Args:
            name: 日志器名称
            log_dir: 日志目录
        """
        config = LoggerConfig(
            name=name,
            log_dir=log_dir,
            enable_console=False,  # 可执行文件不输出到控制台
            enable_file=True,
        )
        cls.set_default_config(config)
        return cls.get_logger()

    @classmethod
    def setup_for_development(cls, name: str = "inference_tool", log_dir: str = "logs"):
        """
        为开发环境设置日志（同时输出到控制台和文件）

        Args:
            name: 日志器名称
            log_dir: 日志目录
        """
        config = LoggerConfig(
            name=name,
            log_dir=log_dir,
            enable_console=True,  # 开发环境输出到控制台
            enable_file=True,
        )
        cls.set_default_config(config)
        return cls.get_logger()

    @classmethod
    def setup_gui_logger(
        cls, name: str = "gui", log_dir: str = "logs", callback=None
    ) -> logging.Logger:
        """
        为GUI应用设置日志器

        Args:
            name: 日志器名称
            log_dir: 日志目录
            callback: GUI回调函数，用于将日志显示在界面上

        Returns:
            logging.Logger: 配置好的日志器
        """
        config = LoggerConfig(
            name=name,
            log_dir=log_dir,
            enable_console=False,  # GUI不输出到控制台
            enable_file=True,
        )

        logger = cls._create_logger(name, config)

        # 如果提供了回调函数，添加GUI处理器
        if callback:
            gui_handler = GuiLogHandler(callback)
            gui_handler.setLevel(config.level.value)
            formatter = logging.Formatter(config.format_string)
            gui_handler.setFormatter(formatter)
            logger.addHandler(gui_handler)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def cleanup(cls):
        """清理所有日志器"""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()


class GuiLogHandler(logging.Handler):
    """GUI日志处理器，将日志发送到GUI回调函数"""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        try:
            msg = self.format(record)
            self.callback(msg)
        except Exception:
            # 避免在日志处理中产生异常
            pass


class ProgressLogger:
    """进度日志器，用于记录和回调进度信息"""

    def __init__(self, logger: logging.Logger, progress_callback=None):
        self.logger = logger
        self.progress_callback = progress_callback
        self.current_step = ""
        self.total_steps = 0
        self.completed_steps = 0

    def start(self, total_steps: int, initial_message: str = "开始执行"):
        """开始进度跟踪"""
        self.total_steps = total_steps
        self.completed_steps = 0
        self.logger.info(f"{initial_message} - 总共{total_steps}步")
        self._update_progress(initial_message, 0)

    def step(self, step_name: str, message: str = None):
        """执行一个步骤"""
        self.current_step = step_name
        self.completed_steps += 1

        if message:
            self.logger.info(
                f"[步骤 {self.completed_steps}/{self.total_steps}] {step_name}: {message}"
            )
        else:
            self.logger.info(
                f"[步骤 {self.completed_steps}/{self.total_steps}] {step_name}"
            )

        progress = (
            (self.completed_steps / self.total_steps) * 100
            if self.total_steps > 0
            else 0
        )
        self._update_progress(step_name, progress)

    def complete(self, final_message: str = "执行完成"):
        """完成进度跟踪"""
        self.logger.info(final_message)
        self._update_progress(final_message, 100)

    def _update_progress(self, step_name: str, progress: float):
        """更新进度回调"""
        if self.progress_callback:
            self.progress_callback(step_name, progress)


# 便捷函数
def setup_logger(
    name: str = "inference_tool",
    level: LogLevel = LogLevel.INFO,
    log_dir: str = "logs",
    enable_console: bool = None,
    enable_file: bool = True,
) -> logging.Logger:
    """
    快速设置日志器的便捷函数

    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志目录
        enable_console: 是否启用控制台输出，None时自动判断（exe为False，开发为True）
        enable_file: 是否启用文件输出

    Returns:
        logging.Logger: 配置好的日志器
    """
    # 自动判断是否为可执行文件
    if enable_console is None:
        enable_console = not getattr(sys, "frozen", False)

    config = LoggerConfig(
        name=name,
        level=level,
        log_dir=log_dir,
        enable_console=enable_console,
        enable_file=enable_file,
    )

    InferenceLogger.set_default_config(config)
    return InferenceLogger.get_logger()


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志器的便捷函数

    Args:
        name: 日志器名称，None时使用默认名称

    Returns:
        logging.Logger: 日志器实例
    """
    return InferenceLogger.get_logger(name)
