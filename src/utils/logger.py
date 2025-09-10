from __future__ import annotations

import io
import logging
import logging.handlers
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

__all__ = [
    "get_logger",
    "configure_logging",
    "set_level",
    "record_fail",
    "log_time",
    "safe_tqdm",
]

_PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
    if Path(__file__).resolve().parts[-3:]
    else Path.cwd()
)
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "logs"
_FAIL_LOG_NAME = "fail.log"

_init_lock = threading.Lock()
_inited = False
_fail_lock = threading.Lock()


class _Color:
    RESET = "\033[0m"
    GREY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"


def _supports_color(stream) -> bool:
    try:
        return stream.isatty() and os.name != "nt"
    except Exception:
        return False


class ColorFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.DEBUG: _Color.GREY,
        logging.INFO: _Color.GREEN,
        logging.WARNING: _Color.YELLOW,
        logging.ERROR: _Color.RED,
        logging.CRITICAL: _Color.RED,
    }

    def __init__(self, fmt: str, datefmt: Optional[str] = None, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self.use_color:
            color = self.LEVEL_MAP.get(record.levelno, _Color.BLUE)
            return f"{color}{msg}{_Color.RESET}"
        return msg


@dataclass
class LogConfig:
    name: str = "ai"
    level: int = logging.INFO
    log_dir: Path = _DEFAULT_LOG_DIR
    # 文件策略：每天一个文件，保留 14 天
    when: str = "midnight"
    backup_count: int = 14
    # 控制台输出：在打包 exe 时默认关闭
    console: Optional[bool] = None  # None=自动: 控制台/EXE 自适应
    # 单次运行专属文件（带时间戳），用于排障
    session_file: bool = True


def configure_logging(cfg: LogConfig = LogConfig()) -> logging.Logger:
    """
    配置全局日志。只会初始化一次，多次调用直接返回 Logger。
    """
    global _inited
    with _init_lock:
        if _inited:
            return logging.getLogger(cfg.name)

        # 目录
        cfg.log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(cfg.name)
        logger.setLevel(cfg.level)
        logger.propagate = False  # 避免重复输出

        # ---- 控制台 Handler
        if cfg.console is None:
            # 打包为 exe 时默认不打控制台；否则输出到控制台
            is_exe = getattr(sys, "frozen", False)
            enable_console = not is_exe
        else:
            enable_console = bool(cfg.console)

        if enable_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(cfg.level)
            fmt = "%(asctime)s  %(levelname)s  %(message)s"
            ch.setFormatter(
                ColorFormatter(fmt, "%H:%M:%S", use_color=_supports_color(sys.stdout))
            )
            logger.addHandler(ch)

        # ---- 主文件（按天滚动）
        fh = logging.handlers.TimedRotatingFileHandler(
            filename=str(cfg.log_dir / "app.log"),
            when=cfg.when,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        fh.setLevel(cfg.level)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)s  [%(process)d:%(threadName)s] "
                "%(name)s - %(filename)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

        # ---- 会话文件（一次运行一个）
        if cfg.session_file:
            ts = time.strftime("%Y%m%d_%H%M%S")
            sh = logging.FileHandler(
                cfg.log_dir / f"session_{ts}.log", encoding="utf-8"
            )
            sh.setLevel(cfg.level)
            sh.setFormatter(
                logging.Formatter(
                    "%(asctime)s  %(levelname)s  %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(sh)

        # 降噪第三方库
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("grpc").setLevel(logging.WARNING)

        _inited = True
        logger.debug("Logger initialized.")
        return logger


def get_logger(name: str = "ai") -> logging.Logger:
    """
    获取一个子 logger。确保已调用 configure_logging()。
    """
    if not _inited:
        configure_logging()
    return logging.getLogger(name)


def set_level(level: int) -> None:
    """动态调整日志等级。"""
    root = get_logger()
    root.setLevel(level)
    for h in root.handlers:
        h.setLevel(level)


def record_fail(msg: str, log_dir: Path | None = None) -> None:
    """
    写入 fail.log，一行一条，带时间戳。
    供“业务失败/不可重试失败”等场景快速定位。
    """
    d = log_dir or _DEFAULT_LOG_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / _FAIL_LOG_NAME
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with _fail_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


@contextmanager
def log_time(
    title: str, logger: Optional[logging.Logger] = None, level: int = logging.INFO
):
    """
    用法:
        with log_time("infer_dir"):
            run()
    """
    lg = logger or get_logger()
    start = time.perf_counter()
    try:
        yield
    finally:
        cost = (time.perf_counter() - start) * 1000.0
        lg.log(level, f"{title} finished in {cost:.2f} ms")


def _create_safe_tqdm() -> Callable:
    try:
        from tqdm import tqdm  # type: ignore

        # exe 环境中默认禁用进度条，避免花屏
        is_exe = getattr(sys, "frozen", False)

        def _safe(iterable, **kw):
            if is_exe:
                kw = {k: v for k, v in kw.items() if k not in {"desc", "position"}}
                return tqdm(iterable, disable=True, **kw)
            return tqdm(iterable, **kw)

        return _safe
    except Exception:
        # 无 tqdm 依赖时退化为原可迭代对象
        return lambda iterable, **kw: iterable


safe_tqdm: Callable = _create_safe_tqdm()


def install_exception_hook(logger: Optional[logging.Logger] = None) -> None:
    """
    捕获未处理异常并写入日志；如需启用，在应用入口调用一次。
    """
    lg = logger or get_logger()

    def _hook(exc_type, exc, tb):
        import traceback

        buf = io.StringIO()
        traceback.print_exception(exc_type, exc, tb, file=buf)
        lg.critical("Uncaught exception:\n%s", buf.getvalue())
        # 仍然沿用默认钩子行为（打印/退出）
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook
