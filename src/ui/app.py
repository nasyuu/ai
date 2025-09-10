# src/ui/app.py
from __future__ import annotations

import tkinter as tk

from ui.controllers.pipeline_controller import PipelineController
from ui.views.main_window import MainWindow
from utils.logger import get_logger

log = get_logger("ui.app")


def main():
    root = tk.Tk()
    root.title("🚀 小模型评估工具")
    # 统一在 View 里管理宽高/样式
    controller = PipelineController()
    MainWindow(root, controller).run()


if __name__ == "__main__":
    main()
