# src/ui/controllers/pipeline_controller.py
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

from utils.exception import AppError

from pipeline.main import PipelineConfig, request_stop, set_progress_callback
from pipeline.main import run as run_pipeline
from utils.logger import get_logger

log = get_logger("ui.controller")


class PipelineController:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._on_progress: Optional[Callable[[str, int], None]] = None
        self._on_done: Optional[Callable[[bool], None]] = None

    # 由 View 注册回调
    def bind_progress(self, cb: Callable[[str, int], None]):
        self._on_progress = cb
        set_progress_callback(cb)

    def bind_done(self, cb: Callable[[bool], None]):
        self._on_done = cb

    def validate(self, cfg: PipelineConfig) -> tuple[bool, str]:
        # 轻量校验，重校验交给 pipeline._validate
        if not cfg.images_dir or not Path(cfg.images_dir).exists():
            return False, "请选择有效的图片目录"
        if not cfg.gt_jsons_dir or not Path(cfg.gt_jsons_dir).exists():
            return False, "请选择有效的真值标注目录"

        if cfg.inference_type == "https":
            h = cfg.https
            if not (
                h
                and h.img_stream_url
                and h.stream_name
                and h.access_key
                and h.secret_key
            ):
                return False, "HTTPS 配置不完整"
        else:
            g = cfg.grpc
            if not (g and g.grpc_server and g.task_id and g.stream_name):
                return False, "gRPC 标准配置不完整"

        return True, "OK"

    def start(self, cfg: PipelineConfig):
        ok, msg = self.validate(cfg)
        if not ok:
            raise AppError(msg)

        if self._thread and self._thread.is_alive():
            raise AppError("已有任务在运行")

        def _worker():
            try:
                success = run_pipeline(cfg)
            except Exception as e:
                log.exception("Pipeline 运行异常: %s", e)
                success = False
            finally:
                if self._on_done:
                    self._on_done(success)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self):
        request_stop()
