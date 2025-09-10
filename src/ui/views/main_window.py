# src/ui/views/main_window.py
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from pipeline.main import PipelineConfig
from ui.models.config_model import GUIState
from ui.widgets.log_console import TkLogHandler
from utils.logger import get_logger

log = get_logger("ui.view")


class MainWindow:
    def __init__(self, root: tk.Tk, controller):
        self.root = root
        self.controller = controller
        self.state = GUIState()

        self._build_ui()
        self._bind_controller()

    def run(self):
        self.root.mainloop()

    # -------- UI ----------
    def _build_ui(self):
        self.root.geometry("820x760")
        self.root.minsize(820, 640)

        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        frm.columnconfigure(1, weight=1)

        # 推理方式
        ttk.Label(frm, text="推理方式").grid(row=0, column=0, sticky="w")
        self.infer_var = tk.StringVar(value=self.state.inference_type)
        ttk.Radiobutton(
            frm,
            text="HTTPS",
            variable=self.infer_var,
            value="https",
            command=self._switch_mode,
        ).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(
            frm,
            text="gRPC(standard)",
            variable=self.infer_var,
            value="grpc_standard",
            command=self._switch_mode,
        ).grid(row=0, column=1, sticky="e")

        # 图片/标注目录
        ttk.Label(frm, text="图片目录").grid(row=1, column=0, sticky="w", pady=4)
        self.images_var = tk.StringVar()
        row1 = ttk.Frame(frm)
        row1.grid(row=1, column=1, sticky="ew")
        row1.columnconfigure(0, weight=1)
        ttk.Entry(row1, textvariable=self.images_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            row1, text="浏览", command=lambda: self._pick_dir(self.images_var)
        ).grid(row=0, column=1, padx=6)

        ttk.Label(frm, text="标注目录").grid(row=2, column=0, sticky="w", pady=4)
        self.gt_var = tk.StringVar()
        row2 = ttk.Frame(frm)
        row2.grid(row=2, column=1, sticky="ew")
        row2.columnconfigure(0, weight=1)
        ttk.Entry(row2, textvariable=self.gt_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(row2, text="浏览", command=lambda: self._pick_dir(self.gt_var)).grid(
            row=0, column=1, padx=6
        )

        # 并发
        ttk.Label(frm, text="处理线程数").grid(row=3, column=0, sticky="w", pady=4)
        self.workers_var = tk.IntVar(value=1)
        ttk.Combobox(
            frm,
            textvariable=self.workers_var,
            values=[1, 2, 4, 8],
            width=8,
            state="readonly",
        ).grid(row=3, column=1, sticky="w")

        # HTTPS配置
        self.https_frame = ttk.LabelFrame(frm, text="HTTPS配置", padding=8)
        self.https_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)
        self.https_frame.columnconfigure(1, weight=1)
        self.https_url = tk.StringVar()
        self.https_stream = tk.StringVar()
        self.https_ak = tk.StringVar()
        self.https_sk = tk.StringVar()
        ttk.Label(self.https_frame, text="URL").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.https_frame, textvariable=self.https_url).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Label(self.https_frame, text="Stream").grid(row=1, column=0, sticky="w")
        ttk.Entry(self.https_frame, textvariable=self.https_stream).grid(
            row=1, column=1, sticky="ew"
        )
        ttk.Label(self.https_frame, text="AK").grid(row=2, column=0, sticky="w")
        ttk.Entry(self.https_frame, textvariable=self.https_ak).grid(
            row=2, column=1, sticky="ew"
        )
        ttk.Label(self.https_frame, text="SK").grid(row=3, column=0, sticky="w")
        ttk.Entry(self.https_frame, textvariable=self.https_sk, show="*").grid(
            row=3, column=1, sticky="ew"
        )

        # gRPC配置
        self.grpc_frame = ttk.LabelFrame(frm, text="gRPC standard 配置", padding=8)
        self.grpc_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=8)
        self.grpc_frame.columnconfigure(1, weight=1)
        self.grpc_server = tk.StringVar()
        self.grpc_task = tk.StringVar()
        self.grpc_stream_name = tk.StringVar()
        ttk.Label(self.grpc_frame, text="Server").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.grpc_frame, textvariable=self.grpc_server).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Label(self.grpc_frame, text="TaskID").grid(row=1, column=0, sticky="w")
        ttk.Entry(self.grpc_frame, textvariable=self.grpc_task).grid(
            row=1, column=1, sticky="ew"
        )
        ttk.Label(self.grpc_frame, text="Stream").grid(row=2, column=0, sticky="w")
        ttk.Entry(self.grpc_frame, textvariable=self.grpc_stream_name).grid(
            row=2, column=1, sticky="ew"
        )

        # 评估/可视化配置
        cfg_frame = ttk.LabelFrame(frm, text="评估与可视化", padding=8)
        cfg_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=8)
        cfg_frame.columnconfigure(0, weight=1)
        self.iou_var = tk.DoubleVar(value=0.5)
        row = ttk.Frame(cfg_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="IoU阈值").pack(side=tk.LEFT)
        ttk.Scale(
            row,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.iou_var,
            length=220,
        ).pack(side=tk.LEFT, padx=8)
        self.viz_stats = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg_frame, text="统计模式可视化", variable=self.viz_stats).pack(
            anchor="w"
        )

        # 步骤选择（与之前一致：推理必选）
        steps = ttk.LabelFrame(frm, text="执行步骤", padding=8)
        steps.grid(row=7, column=0, columnspan=2, sticky="ew", pady=8)
        self.chk_infer = tk.BooleanVar(value=True)
        self.chk_conv = tk.BooleanVar(value=True)
        self.chk_eval = tk.BooleanVar(value=True)
        self.chk_viz = tk.BooleanVar(value=True)
        self.chk_seg_eval = tk.BooleanVar(value=False)
        self.chk_seg_viz = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            steps, text="执行推理 (必选)", variable=self.chk_infer, state=tk.DISABLED
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(steps, text="格式转换", variable=self.chk_conv).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Checkbutton(
            steps, text="检测评估", variable=self.chk_eval, command=self._sync_mutex
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            steps, text="检测可视化", variable=self.chk_viz, command=self._sync_mutex
        ).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(
            steps,
            text="语义分割评估",
            variable=self.chk_seg_eval,
            command=self._sync_mutex,
        ).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(
            steps,
            text="语义分割可视化",
            variable=self.chk_seg_viz,
            command=self._sync_mutex,
        ).grid(row=2, column=1, sticky="w")

        # 控制栏
        bar = ttk.Frame(frm)
        bar.grid(row=8, column=0, columnspan=2, sticky="ew", pady=6)
        self.btn_run = ttk.Button(bar, text="▶ 开始执行", command=self._on_run)
        self.btn_stop = ttk.Button(
            bar, text="⏹ 停止", command=self._on_stop, state=tk.DISABLED
        )
        self.btn_run.pack(side=tk.LEFT)
        self.btn_stop.pack(side=tk.LEFT, padx=8)

        # 进度
        prog = ttk.Frame(frm)
        prog.grid(row=9, column=0, columnspan=2, sticky="ew")
        ttk.Label(prog, text="进度").pack(side=tk.LEFT)
        self.prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(prog, variable=self.prog_var, maximum=100).pack(
            side=tk.LEFT, fill="x", expand=True, padx=8
        )
        self.prog_label = ttk.Label(prog, text="0%")
        self.prog_label.pack(side=tk.LEFT)

        # 日志控件
        log_frame = ttk.LabelFrame(frm, text="日志", padding=4)
        log_frame.grid(row=10, column=0, columnspan=2, sticky="nsew", pady=6)
        frm.rowconfigure(10, weight=1)
        import tkinter.scrolledtext as st

        self.txt = st.ScrolledText(log_frame, height=12)
        self.txt.pack(fill=tk.BOTH, expand=True)
        # 绑定 logging 到 Text
        import logging

        handler = TkLogHandler(self.txt)
        logging.getLogger().addHandler(handler)

        self._switch_mode()
        self._sync_mutex()

    # -------- 绑定控制器回调 ----------
    def _bind_controller(self):
        self.controller.bind_progress(self._on_progress)
        self.controller.bind_done(self._on_done)

    # -------- 事件 ----------
    def _pick_dir(self, var):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _switch_mode(self):
        if self.infer_var.get() == "https":
            self.https_frame.grid()
            self.grpc_frame.grid_remove()
        else:
            self.grpc_frame.grid()
            self.https_frame.grid_remove()

    def _sync_mutex(self):
        # 检测与分割互斥：若选择分割*任一*功能，则禁用/取消检测功能 & 关闭“格式转换”
        use_seg = self.chk_seg_eval.get() or self.chk_seg_viz.get()
        use_det = self.chk_eval.get() or self.chk_viz.get()
        if use_seg and use_det:
            # 优先保留分割，关闭检测
            self.chk_eval.set(False)
            self.chk_viz.set(False)
        if self.chk_eval.get() or self.chk_viz.get():
            self.chk_conv.set(True)  # 检测线需要转换
            # 也可：把转换复选框禁用；这里保留可切换，根据你偏好
        if use_seg:
            self.chk_conv.set(False)  # 分割线不需要转换

    def _collect_state(self) -> GUIState:
        s = GUIState()
        s.inference_type = self.infer_var.get()
        s.images_dir = self.images_var.get().strip()
        s.gt_jsons_dir = self.gt_var.get().strip()
        s.global_workers = int(self.workers_var.get())

        s.iou_thr = float(self.iou_var.get())
        s.viz_mode_stats = bool(self.viz_stats.get())

        s.run_infer = True
        s.run_conv = bool(self.chk_conv.get())
        s.run_eval = bool(self.chk_eval.get())
        s.run_viz = bool(self.chk_viz.get())
        s.seg_eval = bool(self.chk_seg_eval.get())
        s.seg_viz = bool(self.chk_seg_viz.get())
        s.seg_enabled = s.seg_eval or s.seg_viz

        if s.inference_type == "https":
            s.https_url = self.https_url.get().strip()
            s.https_stream = self.https_stream.get().strip()
            s.https_ak = self.https_ak.get().strip()
            s.https_sk = self.https_sk.get().strip()
        else:
            s.grpc_server = self.grpc_server.get().strip()
            s.grpc_task_id = self.grpc_task.get().strip()
            s.grpc_stream_name = self.grpc_stream_name.get().strip()
        return s

    def _on_run(self):
        # UI 状态 → PipelineConfig
        s = self._collect_state()
        cfg: PipelineConfig = s.to_pipeline_config()

        # 对齐“时间戳根目录”的路径（你原先 GUI 的做法是每次运行构造新根目录）
        # 这里示例：保持默认；你也可以在此处按需追加时间戳
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        try:
            self.controller.start(cfg)
        except Exception as e:
            log.error("启动失败: %s", e)
            self.btn_run.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def _on_stop(self):
        self.controller.stop()
        self.btn_stop.config(state=tk.DISABLED)

    # ---- 控制器回调到 UI ----
    def _on_progress(self, step: str, p: int):
        try:
            self.prog_var.set(p)
            self.prog_label.config(text=f"{p}%  {step}")
        except Exception:
            pass

    def _on_done(self, success: bool):
        self.btn_run.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if success:
            log.info("🎉 执行完成")
        else:
            log.error("❌ 执行失败")
