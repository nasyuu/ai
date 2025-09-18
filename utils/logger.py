"""Project-wide logging utilities."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "configure_logging",
    "get_logger",
    "reset_logging",
]

_DEFAULT_LOG_FILENAME = "app.log"
_LOGGER_CONFIGURED = False


def _resolve_log_dir(log_dir: str | os.PathLike[str] | None) -> Path:
    """Resolve and create the directory that will store log files."""
    if log_dir is not None:
        path = Path(log_dir).expanduser()
    else:
        env_dir = os.getenv("APP_LOG_DIR") or os.getenv("LOG_DIR")
        if env_dir:
            path = Path(env_dir).expanduser()
        else:
            path = Path(__file__).resolve().parent.parent / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_file_handler(
    log_path: Path,
    *,
    max_bytes: int,
    backup_count: int,
    fmt: str,
) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(fmt))
    return handler


def configure_logging(
    *,
    level: int = logging.INFO,
    log_dir: str | os.PathLike[str] | None = None,
    log_file: str | os.PathLike[str] = _DEFAULT_LOG_FILENAME,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    console: bool | None = None,
    console_fmt: str = "%(levelname)s | %(name)s | %(message)s",
    file_fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    extra_handlers: Optional[list[logging.Handler]] = None,
    force: bool = False,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure the root logger for the application.

    Parameters mirror the defaults used by the previous monolithic project while
    adding flexibility for future modules.
    """

    global _LOGGER_CONFIGURED

    if _LOGGER_CONFIGURED and not force:
        root = logging.getLogger()
        root.setLevel(level)
        return root

    root = logging.getLogger()
    is_frozen = bool(getattr(sys, "frozen", False))
    use_console = console if console is not None else not is_frozen

    if force or _LOGGER_CONFIGURED:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    root.setLevel(level)

    log_dir_path = _resolve_log_dir(log_dir)
    log_path = (log_dir_path / log_file).resolve()

    file_handler = _build_file_handler(
        log_path,
        max_bytes=max_bytes,
        backup_count=backup_count,
        fmt=file_fmt,
    )
    root.addHandler(file_handler)

    if use_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(console_fmt))
        root.addHandler(console_handler)

    if extra_handlers:
        for handler in extra_handlers:
            root.addHandler(handler)

    if capture_warnings:
        logging.captureWarnings(True)

    _LOGGER_CONFIGURED = True
    return root


def get_logger(name: Optional[str] = None, **configure_kwargs: Any) -> logging.Logger:
    """Return a logger instance, configuring the system on first use."""
    if not _LOGGER_CONFIGURED:
        configure_logging(**configure_kwargs)
    return logging.getLogger(name)


def reset_logging() -> None:
    """Remove all handlers from the root logger and mark the system unconfigured."""
    global _LOGGER_CONFIGURED
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    _LOGGER_CONFIGURED = False
    logging.captureWarnings(False)
