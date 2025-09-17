import logging
import os
import re
import sys
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

# 添加项目根目录到Python路径，确保能正确导入其他模块
sys.path.append(str(Path(__file__).parent))


class PipelineGUI:
    # 常量定义
    WINDOW_WIDTH = 600
    WINDOW_MIN_HEIGHT = 1000
    WINDOW_MAX_HEIGHT = 1500
    WINDOW_INIT_HEIGHT = 1250

    # 版本信息
    VERSION = "v2.3"
    AUTHOR = "👨‍💻 z30055758"
    CHANGELOG_URL = "https://github.com/nasyuu/tool/releases"

    # 目录常量
    LOG_DIR = "logs"

    def __init__(self):
        # 创建主窗口
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title(f"🚀 目标检测类小模型评估工具 - {self.VERSION}")
        self.root.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_INIT_HEIGHT}")
        self.root.configure(bg="#f8f9fa")  # 设置背景色为浅灰色
        self.root.minsize(self.WINDOW_WIDTH, self.WINDOW_MIN_HEIGHT)
        self.root.maxsize(self.WINDOW_WIDTH, self.WINDOW_MAX_HEIGHT)

        # 初始化推理类型状态（避免第一次切换时的闪烁）
        self._last_inference_type = None

        # 设置日志（必须在 _setup_window 之前）
        self.setup_gui_logging()

        # 初始化字体和颜色配置
        self._initialize_styles()

        # 创建界面
        self.create_widgets()

        # 添加窗口大小变化监听器
        self.root.bind("<Configure>", self._on_window_configure)

        # 初始化目录路径（使用时间戳）
        self.update_directory_paths()

        # 初始化步骤依赖关系
        self.on_step_change()

        # 设置输入框自动清理功能
        self._setup_input_cleaning()

        self.root.update_idletasks()

        # 确保HTTPS配置在启动时显示（延迟执行确保所有组件都已创建）
        self.root.after(1, self._ensure_initial_config_display)

        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _ensure_initial_config_display(self):
        """确保初始配置面板正确显示"""
        try:
            # 强制重置并显示默认的HTTPS配置
            current_type = self.inference_type.get()
            if current_type == "https":
                self.https_config_frame.grid(row=0, column=0, sticky="we", pady=(0, 10))
            elif current_type == "grpc_standard":
                self.grpc_standard_config_frame.grid(
                    row=0, column=0, sticky="we", pady=(0, 10)
                )
        except AttributeError:
            # 如果组件还未完全初始化，稍后再试
            self.root.after(10, self._ensure_initial_config_display)

    def _initialize_styles(self):
        # 定义字体样式 - 使用系统推荐字体确保跨平台兼容性
        self.fonts = {
            "title": ("Segoe UI", 16, "bold"),  # 标题字体
            "subtitle": ("Segoe UI", 12, "bold"),  # 副标题字体
            "label": ("Segoe UI", 10),  # 标签字体
            "small": ("Segoe UI", 9),  # 小字体（用于说明文本）
            "button": ("Segoe UI", 10, "bold"),  # 按钮字体
            "status": ("Segoe UI", 10, "bold"),  # 状态字体
            "log": ("Consolas", 10),  # 日志字体（等宽字体）
        }

        # 定义颜色主题 - 使用现代化的颜色方案
        self.colors = {
            "primary": "#2563eb",  # 蓝色 - 主要按钮和重要元素
            "success": "#059669",  # 绿色 - 成功状态和完成提示
            "warning": "#d97706",  # 橙色 - 警告状态和注意事项
            "danger": "#dc2626",  # 红色 - 错误状态和失败提示
            "secondary": "#64748b",  # 灰色 - 次要信息和辅助元素
            "info": "#0891b2",  # 青色 - 信息状态和提示
            "text_primary": "#1f2937",  # 主要文字颜色（深灰色）
            "text_secondary": "#6b7280",  # 次要文字颜色（中灰色）
        }

        # === 核心配置变量 ===
        self.inference_type = tk.StringVar(
            value="https"
        )  # 推理类型：https/grpc_standard
        self.images_dir = tk.StringVar()  # 待推理图片目录
        self.gt_jsons_dir = tk.StringVar()  # 真值标注文件目录

        # === 全局线程数配置 ===
        # 统一控制所有处理步骤（推理、转换、评估、可视化）的并发度
        self.global_workers = tk.IntVar(value=1)  # 默认串行处理，避免资源竞争

        # === HTTPS API推理配置 ===
        self.https_url = tk.StringVar()  # API服务器URL
        self.https_stream = tk.StringVar()  # 流名称
        self.https_access_key = tk.StringVar()  # 访问密钥
        self.https_secret_key = tk.StringVar()  # 秘密密钥
        self.https_raw_dir = tk.StringVar(value="https/responses")  # 原始响应保存目录
        self.https_pred_dir = tk.StringVar(value="https/pred_jsons")  # 预测结果保存目录

        # === 标准gRPC推理配置 ===
        # 支持更多自定义参数的gRPC推理方式
        self.grpc_standard_server = tk.StringVar()  # 服务器地址
        self.grpc_standard_task_id = tk.StringVar()  # 任务ID（用于区分不同任务）
        self.grpc_standard_stream_name = tk.StringVar()  # 流名称（数据流标识）
        self.grpc_standard_raw_dir = tk.StringVar(
            value="grpc_standard/responses"
        )  # 原始响应保存目录（内部使用）
        self.grpc_standard_pred_dir = tk.StringVar(
            value="grpc_standard/pred_jsons"
        )  # 预测结果保存目录（内部使用）

        # === 模型评估配置 ===
        self.iou_threshold = tk.DoubleVar(value=0.5)  # IoU阈值（用于判断检测框匹配）
        self.iou_threshold_display = tk.StringVar(
            value="0.50"
        )  # IoU阈值显示字符串（格式化用）
        self.viz_mode = tk.BooleanVar(
            value=True
        )  # 可视化模式：True=统计模式，False=标签颜色模式

        # === 语义分割评估配置 ===
        # 可选的高级功能，用于像素级分割任务评估
        self.semseg_iou_threshold = tk.DoubleVar(
            value=0.8
        )  # 语义分割IoU阈值（通常比目标检测更严格）
        self.semseg_iou_threshold_display = tk.StringVar(value="0.80")  # 阈值显示字符串

        # === 流程步骤控制变量 ===
        # 用户可以选择性地执行或跳过某些步骤
        self.run_inference = tk.BooleanVar(value=True)  # 执行推理：调用模型获取预测结果
        self.run_conversion = tk.BooleanVar(
            value=True
        )  # 格式转换：将原始响应转为LabelMe格式
        self.run_evaluation = tk.BooleanVar(
            value=True
        )  # 模型评估：计算精度、召回率、mAP等指标
        self.run_visualization = tk.BooleanVar(
            value=True
        )  # 结果可视化：生成对比图和统计图表
        self.run_semseg_evaluation = tk.BooleanVar(
            value=False
        )  # 语义分割评估（默认关闭）
        self.run_semseg_visualization = tk.BooleanVar(
            value=False
        )  # 语义分割可视化（默认关闭）

        # === 运行时状态变量 ===
        self.is_running = False  # 是否正在执行任务（用于防止重复启动）
        self.current_step = 0  # 当前执行的步骤索引
        self.total_steps = 0

    def _setup_input_cleaning(self):
        """为输入框设置自动清理功能"""
        # 需要清理的字符串变量列表
        string_vars_to_clean = [
            self.https_url,
            self.https_stream,
            self.https_access_key,
            self.https_secret_key,
            self.grpc_standard_server,
            self.grpc_standard_task_id,
            self.grpc_standard_stream_name,
        ]

        # 为每个变量添加清理回调
        for var in string_vars_to_clean:
            var.trace("w", lambda *args, v=var: self._on_text_change(v))

    def get_current_base_dir(self):
        # 获取当前时间戳（时分秒格式）
        timestamp = datetime.now().strftime("%H%M%S")

        if self.inference_type.get() == "https":
            return f"https_{timestamp}"
        else:  # grpc_standard
            return f"grpc_standard_{timestamp}"

    def get_current_base_dir_display(self):
        if self.inference_type.get() == "https":
            return "https"
        else:  # grpc_standard
            return "grpc_standard"

    def get_eval_output_path(self):
        base_dir = self.get_current_base_dir()
        return f"{base_dir}/reports/evaluation_report.csv"

    def get_viz_output_path(self):
        base_dir = self.get_current_base_dir()
        return f"{base_dir}/reports/visualization_results"

    def get_semseg_eval_output_path(self):
        base_dir = self.get_current_base_dir()
        return f"{base_dir}/reports/semseg_eval.csv"

    def get_semseg_viz_output_path(self):
        base_dir = self.get_current_base_dir()
        return f"{base_dir}/reports/semseg_vis_masks"

    def _clean_input_text(self, text):
        """清理输入文本中的多余空白字符"""
        if not text:
            return text
        # 去除前后空白字符和换行符
        cleaned = text.strip()
        # 替换内部的换行符和制表符为空格
        cleaned = cleaned.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        # 压缩多个连续空格为单个空格
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _validate_input(
        self, field_name: str, value: str, required: bool = True
    ) -> bool:
        """验证输入字段的有效性"""
        if required and not value.strip():
            self.logger.warning(f"{field_name} 不能为空")
            return False
        return True

    def _on_text_change(self, var, *args):
        try:
            current_value = var.get()
            cleaned_value = self._clean_input_text(current_value)
            if cleaned_value != current_value:
                var.set(cleaned_value)
                self.logger.info("已自动清理输入文本中的多余空白字符")
        except Exception:
            # 忽略清理过程中的任何错误
            pass

    def setup_gui_logging(self):
        """设置GUI专用的日志系统"""
        log_dir = Path(self.LOG_DIR)
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # 创建GUI专用的logger
        self.logger = logging.getLogger("app")
        self.logger.setLevel(logging.INFO)

        # 清除已有的handlers
        self.logger.handlers.clear()

        # 文件handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

    def create_widgets(self):
        # ==== 外层容器 ====
        container = ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky="nsew")

        # 根窗口网格拉伸
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # container 网格：0列放画布，1列放滚动条
        container.columnconfigure(0, weight=1)  # 画布列可拉伸
        container.columnconfigure(1, weight=0)  # 滚动条列不拉伸
        container.rowconfigure(0, weight=1)  # ✅ 关键：这一行让画布有高度

        # ==== 画布 + 右侧滚动条 ====
        self.canvas = tk.Canvas(container, highlightthickness=0, bg="#f8f9fa")
        self.vbar = ttk.Scrollbar(
            container, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")

        # ==== 可滚动的主内容区（原 main_frame）====
        self.main_frame = ttk.Frame(self.canvas, padding="8")
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.main_frame, anchor="nw"
        )

        # 让画布根据内容自适应滚动区域
        def _on_frame_configure(event):
            self.canvas.after_idle(self._update_scroll_region)

        self.main_frame.bind("<Configure>", _on_frame_configure)

        # 让内部 frame 的宽度跟随画布变化（避免横向滚动）
        def _on_canvas_configure(event):
            # 确保内容宽度适配画布宽度
            canvas_width = event.width
            self.canvas.itemconfig(self.window_id, width=canvas_width)
            # 同时更新滚动区域
            self.canvas.after_idle(self._update_scroll_region)

        self.canvas.bind("<Configure>", _on_canvas_configure)

        # ==== 鼠标滚轮支持（跨平台）====
        def _on_mousewheel(event):
            # 检查是否需要滚动（内容高度大于画布高度）
            if self.canvas.winfo_reqheight() > 0:
                if hasattr(event, "delta"):
                    # Windows
                    delta = -int(event.delta / 120)
                else:
                    # macOS/Linux
                    delta = -event.delta
                self.canvas.yview_scroll(delta, "units")

        # 绑定鼠标滚轮事件
        def _bind_mousewheel(event):
            self.root.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/Linux
            self.root.bind_all(
                "<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units")
            )  # Linux
            self.root.bind_all(
                "<Button-5>", lambda e: self.canvas.yview_scroll(1, "units")
            )  # Linux

        def _unbind_mousewheel(event):
            self.root.unbind_all("<MouseWheel>")
            self.root.unbind_all("<Button-4>")
            self.root.unbind_all("<Button-5>")

        self.canvas.bind("<Enter>", _bind_mousewheel)
        self.canvas.bind("<Leave>", _unbind_mousewheel)

        # ==== 以下保持你的原逻辑，parent 改为 main_frame ====
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        self.create_header_section(self.main_frame)  # 第0行：标题区域
        self.create_config_section(self.main_frame)  # 第1行：配置设置
        self.create_steps_control_section(self.main_frame)  # 第2行：步骤控制与语义分割
        self.create_progress_section(self.main_frame)  # 第3行：执行进度

        # 首次渲染后，主动更新一次 scrollregion，确保能滚到底
        self.root.after_idle(self._update_scroll_region)

    def _update_scroll_region(self):
        """更新滚动区域，确保滚动条正常显示和工作"""
        try:
            # 强制更新布局
            self.main_frame.update_idletasks()
            # 获取所有子组件的边界框
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=bbox)

                # 检查是否需要显示滚动条
                canvas_height = self.canvas.winfo_height()
                content_height = bbox[3] - bbox[1]

                # 总是显示滚动条，让系统自动决定是否需要激活
                # 这样可以避免滚动条显示/隐藏时的布局跳跃问题
                self.vbar.grid(row=0, column=1, sticky="ns")

                # 如果内容高度小于画布高度，确保滚动位置在顶部
                if content_height <= canvas_height and canvas_height > 1:
                    self.canvas.yview_moveto(0)
            else:
                # 如果没有内容，设置一个最小滚动区域
                self.canvas.configure(scrollregion=(0, 0, 0, 0))
                self.vbar.grid(row=0, column=1, sticky="ns")
        except (tk.TclError, AttributeError):
            # 如果组件还未初始化完成，稍后再试
            self.root.after(100, self._update_scroll_region)

    def _on_window_configure(self, event):
        """处理窗口大小变化事件"""
        # 只处理根窗口的配置变化事件，忽略子组件的事件
        if event.widget == self.root:
            # 延迟更新滚动区域，确保所有组件都已重新布局
            self.root.after_idle(self._update_scroll_region)

    def create_header_section(self, parent):
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 10))
        header_frame.columnconfigure(0, weight=1)

        # 创建一个居中的内容框架
        content_frame = ttk.Frame(header_frame)
        content_frame.grid(row=0, column=0, sticky="we")

        # 获取工作目录
        current_path = os.getcwd()
        display_name = os.path.basename(current_path)

        # 目录信息区域 - 分组显示
        dirs_container = tk.Frame(content_frame, bg="#f8f9fa")
        dirs_container.pack(fill=tk.X, pady=(5, 0))

        # 工作环境区域 - 带版本信息
        workspace_frame = tk.Frame(dirs_container, bg="#f8f9fa")
        workspace_frame.pack(fill=tk.X, pady=(0, 6))

        # 左侧工作环境信息
        workspace_left = tk.Frame(workspace_frame, bg="#f8f9fa")
        workspace_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        workspace_title = tk.Label(
            workspace_left,
            text="📁 工作环境:",
            font=self.fonts["small"],
            fg=self.colors["text_secondary"],
            bg="#f8f9fa",
        )
        workspace_title.pack(side=tk.LEFT)

        workspace_path = tk.Label(
            workspace_left,
            text=display_name,
            font=self.fonts["small"],
            fg=self.colors["primary"],
            bg="#f8f9fa",
            cursor="hand2",
        )
        workspace_path.pack(side=tk.LEFT, padx=(5, 0))

        # 点击复制路径
        workspace_path.bind("<Button-1>", lambda e: self._copy_path(current_path))

        # 右上角版本信息（可点击查看更新日志）
        version_frame = tk.Frame(workspace_frame, bg="#f8f9fa")
        version_frame.pack(side=tk.RIGHT, anchor=tk.N)

        # 分离版本号和作者信息，版本号单独可点击
        version_label = tk.Label(
            version_frame,
            text=self.VERSION,
            font=self.fonts["small"],
            fg=self.colors["primary"],  # 使用蓝色表示可点击
            bg="#f8f9fa",
            cursor="hand2",  # 设置鼠标悬停时的手型光标
        )
        version_label.pack(side=tk.LEFT, anchor=tk.E)

        # 作者信息（不可点击）
        author_info = tk.Label(
            version_frame,
            text=f" | {self.AUTHOR}",
            font=self.fonts["small"],
            fg=self.colors["text_secondary"],
            bg="#f8f9fa",
        )
        author_info.pack(side=tk.LEFT, anchor=tk.E)

        # 为版本号添加点击事件处理器，用于打开更新日志
        version_label.bind("<Button-1>", self._open_changelog)

        # 为版本号添加悬停效果
        def on_enter(event):
            version_label.config(
                fg=self.colors["info"], font=self.fonts["small"] + ("underline",)
            )

        def on_leave(event):
            version_label.config(fg=self.colors["primary"], font=self.fonts["small"])

        version_label.bind("<Enter>", on_enter)
        version_label.bind("<Leave>", on_leave)

        # 响应目录说明
        def get_current_response_dir():
            if self.inference_type.get() == "https":
                return self.https_raw_dir.get()
            else:  # grpc_standard
                return self.grpc_standard_raw_dir.get()

        def get_current_pred_dir():
            if self.inference_type.get() == "https":
                return self.https_pred_dir.get()
            else:  # grpc_standard
                return self.grpc_standard_pred_dir.get()

        # 数据流向区域
        flow_frame = tk.Frame(dirs_container, bg="#f8f9fa")
        flow_frame.pack(fill=tk.X)

        flow_label = tk.Label(
            flow_frame,
            text="📊 数据流向:",
            font=self.fonts["small"],
            fg=self.colors["text_secondary"],
            bg="#f8f9fa",
        )
        flow_label.pack(side=tk.LEFT)

        # 创建动态更新的流向显示
        self.flow_display = tk.Label(
            flow_frame,
            text="图片 → 推理响应 → 预测标注 → 对比评估 → 可视化报告",
            font=self.fonts["small"],
            fg=self.colors["info"],
            bg="#f8f9fa",
        )
        self.flow_display.pack(side=tk.LEFT, padx=(5, 0))

        # 目录显示区域
        dirs_flow_frame = tk.Frame(dirs_container, bg="#f8f9fa")
        dirs_flow_frame.pack(fill=tk.X, pady=(2, 0))

        dirs_path_label = tk.Label(
            dirs_flow_frame,
            text="📂 目录:",
            font=self.fonts["small"],
            fg=self.colors["text_secondary"],
            bg="#f8f9fa",
        )
        dirs_path_label.pack(side=tk.LEFT)

        # 动态目录显示
        self.dirs_display = tk.Label(
            dirs_flow_frame,
            text=f"images/ → {get_current_response_dir()}/ → {get_current_pred_dir()}/ → {self.get_current_base_dir()}/reports/",
            font=self.fonts["small"],
            fg=self.colors["secondary"],
            bg="#f8f9fa",
        )
        self.dirs_display.pack(side=tk.LEFT, padx=(5, 0))

    def create_config_section(self, parent):
        config_frame = ttk.LabelFrame(parent, text="⚙️ 配置设置", padding="10")
        config_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)

        # 推理类型选择
        ttk.Label(config_frame, text="推理方式:", font=self.fonts["label"]).grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )

        inference_frame = ttk.Frame(config_frame)
        inference_frame.grid(row=0, column=1, sticky="we", pady=(0, 10))

        https_radio = ttk.Radiobutton(
            inference_frame,
            text="HTTPS API",
            variable=self.inference_type,
            value="https",
            command=self.on_inference_type_change,
        )
        https_radio.pack(side=tk.LEFT, padx=(0, 20))

        grpc_standard_radio = ttk.Radiobutton(
            inference_frame,
            text="标准 gRPC",
            variable=self.inference_type,
            value="grpc_standard",
            command=self.on_inference_type_change,
        )
        grpc_standard_radio.pack(side=tk.LEFT)

        # 通用输入目录配置
        ttk.Label(config_frame, text="图片目录:", font=self.fonts["label"]).grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=(5, 5)
        )

        images_frame = ttk.Frame(config_frame)
        images_frame.grid(row=1, column=1, sticky="we", pady=(5, 5))
        images_frame.columnconfigure(0, weight=1)

        ttk.Entry(images_frame, textvariable=self.images_dir, width=50).grid(
            row=0, column=0, sticky="we", padx=(0, 5)
        )
        ttk.Button(
            images_frame,
            text="浏览",
            command=lambda: self.browse_directory(self.images_dir),
        ).grid(row=0, column=1)

        ttk.Label(config_frame, text="标注目录:", font=self.fonts["label"]).grid(
            row=2, column=0, sticky="w", padx=(0, 10), pady=(5, 5)
        )

        gt_frame = ttk.Frame(config_frame)
        gt_frame.grid(row=2, column=1, sticky="we", pady=(5, 5))
        gt_frame.columnconfigure(0, weight=1)

        ttk.Entry(gt_frame, textvariable=self.gt_jsons_dir, width=50).grid(
            row=0, column=0, sticky="we", padx=(0, 5)
        )
        ttk.Button(
            gt_frame,
            text="浏览",
            command=lambda: self.browse_directory(self.gt_jsons_dir),
        ).grid(row=0, column=1)

        # 全局线程数配置
        ttk.Label(config_frame, text="处理线程数:", font=self.fonts["label"]).grid(
            row=3, column=0, sticky="w", padx=(0, 10), pady=(5, 10)
        )

        workers_frame = ttk.Frame(config_frame)
        workers_frame.grid(row=3, column=1, sticky="we", pady=(5, 10))
        workers_frame.columnconfigure(0, weight=1)
        workers_frame.columnconfigure(1, weight=0)  # 说明文字不需要扩展

        ttk.Combobox(
            workers_frame,
            textvariable=self.global_workers,
            values=[1, 2, 4, 8],
            state="readonly",
        ).grid(row=0, column=0, sticky="we", padx=(0, 5))

        # 添加线程数说明
        ttk.Label(
            workers_frame,
            text="(影响所有处理步骤)",
            font=self.fonts["small"],
            foreground=self.colors["text_secondary"],
        ).grid(row=0, column=1, sticky="w", padx=(5, 0))

        # 动态配置区域
        self.dynamic_config_frame = ttk.Frame(config_frame)
        self.dynamic_config_frame.grid(
            row=4, column=0, columnspan=2, sticky="we", pady=(10, 0)
        )
        self.dynamic_config_frame.columnconfigure(0, weight=1)

        # 创建HTTPS和gRPC配置框架（但不立即显示）
        self.create_https_config_frame()
        self.create_grpc_standard_config_frame()  # 添加标准gRPC配置框架
        self.create_eval_config_frame()
        self.create_viz_config_frame()
        self.create_semseg_config_frame()

        # 初始化配置面板 - 默认显示HTTPS配置
        self._last_inference_type = None  # 确保初始化时会触发显示
        self.on_inference_type_change()

    def create_https_config_frame(self):
        self.https_config_frame = ttk.LabelFrame(
            self.dynamic_config_frame, text="🌐 HTTPS 接口配置", padding="10"
        )
        self.https_config_frame.columnconfigure(1, weight=1)

        ttk.Label(
            self.https_config_frame, text="API地址:", font=self.fonts["label"]
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        https_url_entry = ttk.Entry(
            self.https_config_frame, textvariable=self.https_url, width=50
        )
        https_url_entry.grid(row=0, column=1, sticky="we")

        ttk.Label(
            self.https_config_frame, text="流名称:", font=self.fonts["label"]
        ).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        https_stream_entry = ttk.Entry(
            self.https_config_frame, textvariable=self.https_stream, width=50
        )
        https_stream_entry.grid(row=1, column=1, sticky="we", pady=(5, 0))

        ttk.Label(
            self.https_config_frame, text="ak密钥:", font=self.fonts["label"]
        ).grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        https_access_key_entry = ttk.Entry(
            self.https_config_frame, textvariable=self.https_access_key, width=50
        )
        https_access_key_entry.grid(row=2, column=1, sticky="we", pady=(5, 0))

        ttk.Label(
            self.https_config_frame, text="sk密钥:", font=self.fonts["label"]
        ).grid(row=3, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        https_secret_key_entry = ttk.Entry(
            self.https_config_frame,
            textvariable=self.https_secret_key,
            show="*",
            width=50,
        )
        https_secret_key_entry.grid(row=3, column=1, sticky="we", pady=(5, 0))

    def create_grpc_standard_config_frame(self):
        self.grpc_standard_config_frame = ttk.LabelFrame(
            self.dynamic_config_frame, text="⚡ 标准gRPC 服务配置", padding="10"
        )
        self.grpc_standard_config_frame.columnconfigure(1, weight=1)

        # 服务器地址配置
        ttk.Label(
            self.grpc_standard_config_frame,
            text="服务器地址:",
            font=self.fonts["label"],
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        grpc_standard_server_entry = ttk.Entry(
            self.grpc_standard_config_frame,
            textvariable=self.grpc_standard_server,
            width=50,
        )
        grpc_standard_server_entry.grid(row=0, column=1, sticky="we")

        # 任务ID配置
        ttk.Label(
            self.grpc_standard_config_frame, text="服务ID:", font=self.fonts["label"]
        ).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        grpc_standard_task_id_entry = ttk.Entry(
            self.grpc_standard_config_frame,
            textvariable=self.grpc_standard_task_id,
            width=50,
        )
        grpc_standard_task_id_entry.grid(row=1, column=1, sticky="we", pady=(5, 0))

        # 流名称配置
        ttk.Label(
            self.grpc_standard_config_frame, text="流名称:", font=self.fonts["label"]
        ).grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        ttk.Entry(
            self.grpc_standard_config_frame,
            textvariable=self.grpc_standard_stream_name,
            width=50,
        ).grid(row=2, column=1, sticky="we", pady=(5, 0))

    def create_eval_config_frame(self):
        self.eval_config_frame = ttk.LabelFrame(
            self.dynamic_config_frame, text="📊 评估参数配置", padding="10"
        )
        self.eval_config_frame.columnconfigure(1, weight=1)

        ttk.Label(
            self.eval_config_frame, text="IoU阈值:", font=self.fonts["label"]
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        iou_scale = ttk.Scale(
            self.eval_config_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.iou_threshold,
            length=200,
            command=self.on_iou_threshold_change,
        )
        iou_scale.grid(row=0, column=1, sticky="we")

        iou_value_label = tk.Label(
            self.eval_config_frame,
            textvariable=self.iou_threshold_display,
            font=self.fonts["label"],
            fg=self.colors["primary"],
        )
        iou_value_label.grid(row=0, column=2, padx=(10, 0))

    def create_viz_config_frame(self):
        self.viz_config_frame = ttk.LabelFrame(
            self.dynamic_config_frame, text="🎨 可视化配置", padding="10"
        )
        self.viz_config_frame.columnconfigure(1, weight=1)

        ttk.Label(
            self.viz_config_frame, text="可视化模式:", font=self.fonts["label"]
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))

        viz_mode_frame = ttk.Frame(self.viz_config_frame)
        viz_mode_frame.grid(row=0, column=1, sticky="w")

        ttk.Radiobutton(
            viz_mode_frame, text="📈 统计模式", variable=self.viz_mode, value=True
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(
            viz_mode_frame, text="🎨 标签颜色模式", variable=self.viz_mode, value=False
        ).pack(side=tk.LEFT)

    def create_semseg_config_frame(self):
        self.semseg_config_frame = ttk.LabelFrame(
            self.dynamic_config_frame, text="🔬 语义分割配置", padding="12"
        )
        self.semseg_config_frame.columnconfigure(1, weight=1)

        # 说明信息
        info_label = tk.Label(
            self.semseg_config_frame,
            text="💡 语义分割功能使用默认配置，无需手动设置\n"
            "📊 评估报告：{协议}/reports/semseg_eval.csv\n"
            "🎨 可视化输出：{协议}/reports/semseg_vis_masks\n"
            "⚡ 使用全局线程数配置\n"
            "🚫 不保存差分图片",
            font=self.fonts["small"],
            fg=self.colors["text_secondary"],
            justify=tk.LEFT,
        )
        info_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

    def update_directory_paths(self):
        """根据当前推理类型更新目录路径"""
        base_dir = self.get_current_base_dir()

        if self.inference_type.get() == "https":
            self.https_raw_dir.set(f"{base_dir}/responses")
            self.https_pred_dir.set(f"{base_dir}/pred_jsons")
        else:  # grpc_standard
            self.grpc_standard_raw_dir.set(f"{base_dir}/responses")
            self.grpc_standard_pred_dir.set(f"{base_dir}/pred_jsons")

    def on_inference_type_change(self):
        # 防止重复点击同一选项时触发界面更新
        current_type = self.inference_type.get()
        if (
            hasattr(self, "_last_inference_type")
            and self._last_inference_type == current_type
            and self._last_inference_type is not None  # 允许初始化时的显示
        ):
            return
        self._last_inference_type = current_type

        # 隐藏所有配置框架（使用try-except防止初始化时的错误）
        try:
            self.https_config_frame.grid_remove()
            self.grpc_standard_config_frame.grid_remove()
        except AttributeError:
            # 如果配置框架还没有创建，忽略错误
            pass

        # 显示对应的配置框架
        try:
            if current_type == "https":
                self.https_config_frame.grid(row=0, column=0, sticky="we", pady=(0, 10))
            else:  # grpc_standard
                self.grpc_standard_config_frame.grid(
                    row=0, column=0, sticky="we", pady=(0, 10)
                )
        except AttributeError:
            # 如果配置框架还没有创建，忽略错误
            pass

        # 更新目录路径（使用时间戳）
        try:
            self.update_directory_paths()
        except AttributeError:
            pass

        # 更新动态显示内容
        self.update_dynamic_displays()

        # 根据当前选择状态决定是否显示评估和可视化配置
        self.update_eval_viz_config_display()

        # 语义分割配置只在启用时显示
        self.update_semseg_config_display()

    def update_dynamic_displays(self):
        try:
            # 使用不带时间戳的基础目录进行GUI显示
            base_dir_display = self.get_current_base_dir_display()

            # 更新目录显示（显示简化的路径格式）
            self.dirs_display.config(
                text=f"images/ → {base_dir_display}/responses/ → {base_dir_display}/pred_jsons/ → {base_dir_display}/reports/"
            )

            # 根据当前选择的步骤更新流向显示
            self.update_flow_display()

        except Exception:
            # 如果动态显示组件还未创建，忽略错误
            pass

    def update_flow_display(self):
        try:
            has_semseg = (
                self.run_semseg_evaluation.get() or self.run_semseg_visualization.get()
            )
            has_detection = self.run_evaluation.get() or self.run_visualization.get()

            if has_semseg:
                # 语义分割流程
                flow_text = "图片 → 推理响应 → 语义分割评估/可视化"
            elif has_detection:
                # 目标检测流程
                flow_text = "图片 → 推理响应 → 预测标注 → 对比评估 → 可视化报告"
            else:
                # 基础流程
                flow_text = "图片 → 推理响应 → 预测标注 → 对比评估 → 可视化报告"

            self.flow_display.config(text=flow_text)

        except Exception:
            # 如果流向显示组件还未创建，忽略错误
            pass

    def update_semseg_config_display(self):
        # 语义分割配置使用默认值，不需要显示配置框架
        self.semseg_config_frame.grid_remove()

    def on_semseg_evaluation_change(self):
        if self.run_semseg_evaluation.get():
            # 选择了语义分割评估，禁用目标检测功能
            self.run_evaluation.set(False)
            self.run_visualization.set(False)
            self.evaluation_checkbox.config(state=tk.DISABLED)
            self.visualization_checkbox.config(state=tk.DISABLED)
            self.logger.info("✅ 已选择语义分割评估，禁用目标检测功能")
        else:
            # 取消选择语义分割评估时，检查是否需要恢复目标检测选项
            if not self.run_semseg_visualization.get():
                # 如果语义分割可视化也没选中，恢复目标检测选项
                self.evaluation_checkbox.config(state=tk.NORMAL)
                self.visualization_checkbox.config(state=tk.NORMAL)
                self.logger.info("✅ 已取消语义分割评估，恢复目标检测选项")
        self.on_step_change()

    def on_semseg_visualization_change(self):
        if self.run_semseg_visualization.get():
            # 选择了语义分割可视化，禁用目标检测功能
            self.run_evaluation.set(False)
            self.run_visualization.set(False)
            self.evaluation_checkbox.config(state=tk.DISABLED)
            self.visualization_checkbox.config(state=tk.DISABLED)
            self.logger.info("✅ 已选择语义分割可视化，禁用目标检测功能")
        else:
            # 取消选择语义分割可视化时，检查是否需要恢复目标检测选项
            if not self.run_semseg_evaluation.get():
                # 如果语义分割评估也没选中，恢复目标检测选项
                self.evaluation_checkbox.config(state=tk.NORMAL)
                self.visualization_checkbox.config(state=tk.NORMAL)
                self.logger.info("✅ 已取消语义分割可视化，恢复目标检测选项")
        self.on_step_change()

    def on_detection_evaluation_change(self):
        if self.run_evaluation.get():
            # 选择了目标检测评估，禁用语义分割功能
            self.run_semseg_evaluation.set(False)
            self.run_semseg_visualization.set(False)
            self.semseg_evaluation_checkbox.config(state=tk.DISABLED)
            self.semseg_visualization_checkbox.config(state=tk.DISABLED)
            self.logger.info("✅ 已选择目标检测评估，禁用语义分割功能")
        else:
            # 取消选择目标检测评估时，检查是否需要恢复语义分割选项
            if not self.run_visualization.get():
                # 如果目标检测可视化也没选中，恢复语义分割选项
                self.semseg_evaluation_checkbox.config(state=tk.NORMAL)
                self.semseg_visualization_checkbox.config(state=tk.NORMAL)
                self.logger.info("✅ 已取消目标检测评估，恢复语义分割选项")
        self.on_step_change()

    def on_detection_visualization_change(self):
        if self.run_visualization.get():
            # 选择了目标检测可视化，禁用语义分割功能
            self.run_semseg_evaluation.set(False)
            self.run_semseg_visualization.set(False)
            self.semseg_evaluation_checkbox.config(state=tk.DISABLED)
            self.semseg_visualization_checkbox.config(state=tk.DISABLED)
            self.logger.info("✅ 已选择目标检测可视化，禁用语义分割功能")
        else:
            # 取消选择目标检测可视化时，检查是否需要恢复语义分割选项
            if not self.run_evaluation.get():
                # 如果目标检测评估也没选中，恢复语义分割选项
                self.semseg_evaluation_checkbox.config(state=tk.NORMAL)
                self.semseg_visualization_checkbox.config(state=tk.NORMAL)
                self.logger.info("✅ 已取消目标检测可视化，恢复语义分割选项")
        self.on_step_change()

    def browse_directory(self, var):
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)

    def start_pipeline(self):
        self.logger.info("🚀 开始执行推理流水线")

        # 每次运行时更新目录路径，生成新的时间戳
        self.update_directory_paths()

        # 从GUI收集配置，创建独立的配置字典
        gui_config = {
            "inference_type": self.inference_type.get(),
            "images_dir": self.images_dir.get(),
            "gt_jsons_dir": self.gt_jsons_dir.get(),
            "https_config": {
                "img_stream_url": self.https_url.get(),
                "stream_name": self.https_stream.get(),
                "access_key": self.https_access_key.get(),
                "secret_key": self.https_secret_key.get(),
                "raw_responses_dir": self.https_raw_dir.get(),
                "pred_jsons_dir": self.https_pred_dir.get(),
                "max_workers": self.global_workers.get(),  # 使用全局线程数
            },
            "grpc_standard_config": {
                "grpc_server": self.grpc_standard_server.get(),
                "task_id": self.grpc_standard_task_id.get(),  # 任务ID
                "stream_name": self.grpc_standard_stream_name.get(),  # 流名称
                "raw_responses_dir": self.grpc_standard_raw_dir.get(),  # 使用默认值
                "pred_jsons_dir": self.grpc_standard_pred_dir.get(),
                "max_workers": self.global_workers.get(),  # 使用全局线程数
            },
            "eval_config": {
                "iou_threshold": self.iou_threshold.get(),
                "eval_output_file": self.get_eval_output_path(),
                "viz_output_dir": self.get_viz_output_path(),
                "viz_mode": self.viz_mode.get(),
                "max_workers": self.global_workers.get(),  # 使用全局线程数
            },
            "semseg_config": {
                "enabled": (
                    self.run_semseg_evaluation.get()
                    or self.run_semseg_visualization.get()
                ),
                "eval_output_file": self.get_semseg_eval_output_path(),
                "viz_output_dir": self.get_semseg_viz_output_path(),
                "save_diff_png": False,  # 固定为False，不保存差分图
                "max_workers": self.global_workers.get(),  # 使用全局线程数
                "iou_threshold": self.semseg_iou_threshold.get(),
            },
            "steps": {
                "run_inference": self.run_inference.get(),
                "run_conversion": self.run_conversion.get(),
                "run_evaluation": self.run_evaluation.get(),
                "run_visualization": self.run_visualization.get(),
                "run_semseg_evaluation": self.run_semseg_evaluation.get(),
                "run_semseg_visualization": self.run_semseg_visualization.get(),
            },
        }

        self.logger.info("✅ GUI配置收集完成")

        # 验证配置
        if not self._validate_config(gui_config):
            self.logger.error("❌ 配置验证失败，请检查输入参数")
            return

        # 调用pipeline执行，传入GUI配置
        try:
            # 导入pipeline模块
            import pipeline

            # 将GUI配置转换为pipeline格式
            pipeline_config = {
                "INFERENCE_TYPE": gui_config["inference_type"],
                "IMAGES_DIR": gui_config["images_dir"],
                "GT_JSONS_DIR": gui_config["gt_jsons_dir"],
                "HTTPS_CONFIG": gui_config["https_config"],
                "GRPC_STANDARD_CONFIG": gui_config["grpc_standard_config"],
                "EVAL_CONFIG": gui_config["eval_config"],
                "SEMSEG_CONFIG": gui_config["semseg_config"],
                "STEPS": gui_config["steps"],
            }

            self.logger.info("🔧 开始执行pipeline...")

            # 设置进度回调函数
            def progress_callback(step_name, progress):
                self.root.after(0, self._update_progress, step_name, progress)

            pipeline.set_progress_callback(progress_callback)

            # 在后台线程中执行pipeline，避免GUI卡死
            def run_pipeline():
                try:
                    # 禁用开始按钮，启用停止按钮
                    self.root.after(
                        0,
                        lambda: [
                            self.run_button.config(state=tk.DISABLED),
                            self.stop_button.config(state=tk.NORMAL),
                        ],
                    )

                    # 执行pipeline
                    success = pipeline.run_inference_pipeline(pipeline_config)

                    # 执行完成后的UI更新
                    if success:
                        self.root.after(0, self._update_progress, "执行完成", 100)
                        self.root.after(
                            0, lambda: self.logger.info("🎉 Pipeline执行成功！")
                        )
                        # 显示执行结果
                        self.root.after(0, lambda: self._show_results(gui_config))
                    else:
                        self.root.after(0, self._update_progress, "执行失败", 0)
                        self.root.after(
                            0, lambda: self.logger.error("❌ Pipeline执行失败！")
                        )

                except Exception as ex:
                    error_msg = str(ex)
                    self.root.after(
                        0,
                        lambda msg=error_msg: self.logger.error(
                            f"❌ Pipeline执行异常: {msg}"
                        ),
                    )
                finally:
                    # 恢复按钮状态
                    self.root.after(
                        0,
                        lambda: [
                            self.run_button.config(state=tk.NORMAL),
                            self.stop_button.config(state=tk.DISABLED),
                        ],
                    )

            # 启动后台线程
            pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
            pipeline_thread.start()

        except Exception as e:
            self.logger.error(f"❌ 启动Pipeline失败: {e}")
            import traceback

            self.logger.error(f"详细错误: {traceback.format_exc()}")

    def stop_pipeline(self):
        self.logger.info("⚠️ 用户请求停止执行")

    def _update_progress(self, step_name, progress):
        try:
            # 更新当前步骤标签
            self.current_step_label.config(text=f"🔄 {step_name}")

            # 更新进度条
            self.progress_var.set(progress)

            # 更新进度百分比标签
            self.progress_label.config(text=f"{progress}%")

            # 根据进度设置状态标签
            if progress == 0:
                self.status_label.config(text="🚀 开始执行...", fg=self.colors["info"])
            elif progress == 100:
                self.status_label.config(
                    text="✅ 执行完成！", fg=self.colors["success"]
                )
            else:
                self.status_label.config(
                    text=f"⏳ 正在执行 {step_name}...", fg=self.colors["warning"]
                )

        except Exception as e:
            self.logger.error(f"更新进度失败: {e}")

    def _validate_config(self, config):
        try:
            # === 验证必填目录 ===
            if not config["images_dir"] or not os.path.exists(config["images_dir"]):
                self.logger.error("❌ 请选择有效的输入图片目录")
                return False

            if not config["gt_jsons_dir"] or not os.path.exists(config["gt_jsons_dir"]):
                self.logger.error("❌ 请选择有效的真值标注目录")
                return False

            # === 验证推理接口配置 ===
            if config["inference_type"] == "https":
                https_config = config["https_config"]
                if not https_config["img_stream_url"]:
                    self.logger.error("❌ 请填写HTTPS API地址")
                    return False
                if not https_config["stream_name"]:
                    self.logger.error("❌ 请填写流名称")
                    return False
                if not https_config["access_key"]:
                    self.logger.error("❌ 请填写访问密钥")
                    return False
                if not https_config["secret_key"]:
                    self.logger.error("❌ 请填写秘密密钥")
                    return False
            elif config["inference_type"] == "grpc_standard":
                grpc_standard_config = config["grpc_standard_config"]
                if not grpc_standard_config["grpc_server"]:
                    self.logger.error("❌ 请填写标准gRPC服务器地址")
                    return False
                if not grpc_standard_config["task_id"]:
                    self.logger.error("❌ 请填写任务ID")
                    return False
                if not grpc_standard_config["stream_name"]:
                    self.logger.error("❌ 请填写流名称")
                    return False

            # === 验证数值参数范围 ===
            iou = config["eval_config"]["iou_threshold"]
            if not (0.0 <= iou <= 1.0):
                self.logger.error("❌ IoU阈值必须在0.0-1.0之间")
                return False

            # === 验证步骤依赖关系 ===
            steps = config["steps"]
            has_detection = steps["run_evaluation"] or steps["run_visualization"]
            has_semseg = (
                steps["run_semseg_evaluation"] or steps["run_semseg_visualization"]
            )

            # 检查是否选择了任何功能
            if not (
                has_detection
                or has_semseg
                or steps["run_conversion"]
                or steps["run_inference"]
            ):
                self.logger.error("❌ 请至少选择一个功能步骤")
                return False

            # === 检查步骤依赖关系 ===
            # 检查目标检测路线的依赖
            if has_detection:
                if not steps["run_inference"]:
                    self.logger.error("❌ 目标检测功能需要执行推理步骤")
                    return False
                if not steps["run_conversion"]:
                    self.logger.error("❌ 目标检测功能需要格式转换步骤")
                    return False

            # 检查语义分割路线的依赖
            if has_semseg:
                if not steps["run_inference"]:
                    self.logger.error("❌ 语义分割功能需要执行推理步骤")
                    return False
                # 语义分割不需要格式转换，如果用户选择了会发出警告但不阻止执行
                if steps["run_conversion"]:
                    self.logger.warning("⚠️ 语义分割功能不需要格式转换，将自动跳过")

            # === 检查功能互斥性 ===
            if has_detection and has_semseg:
                self.logger.error("❌ 目标检测和语义分割功能不能同时选择")
                return False

            self.logger.info("✅ 配置验证通过")
            return True

        except Exception as e:
            self.logger.error(f"❌ 配置验证异常: {e}")
            return False

    def _show_results(self, config):
        try:
            # 获取输出目录
            if config["inference_type"] == "https":
                raw_dir = config["https_config"]["raw_responses_dir"]
                pred_dir = config["https_config"]["pred_jsons_dir"]
            else:  # grpc_standard
                raw_dir = config["grpc_standard_config"]["raw_responses_dir"]
                pred_dir = config["grpc_standard_config"]["pred_jsons_dir"]

            eval_file = config["eval_config"]["eval_output_file"]
            viz_dir = config["eval_config"]["viz_output_dir"]

            # 检查输出文件
            results = []
            if os.path.exists(raw_dir):
                count = len([f for f in os.listdir(raw_dir) if f.endswith(".json")])
                results.append(f"📁 原始响应: {raw_dir} ({count}个文件)")

            if os.path.exists(pred_dir):
                count = len([f for f in os.listdir(pred_dir) if f.endswith(".json")])
                results.append(f"📁 预测结果: {pred_dir} ({count}个文件)")

            if os.path.exists(eval_file):
                results.append(f"📊 评估报告: {eval_file}")

            if os.path.exists(viz_dir):
                correct_dir = os.path.join(viz_dir, "correct")
                error_dir = os.path.join(viz_dir, "error")
                correct_count = (
                    len(os.listdir(correct_dir)) if os.path.exists(correct_dir) else 0
                )
                error_count = (
                    len(os.listdir(error_dir)) if os.path.exists(error_dir) else 0
                )
                results.append(f"🎨 可视化图片: {viz_dir}")
                results.append(f"   ├── 正确预测: {correct_count}张")
                results.append(f"   └── 错误预测: {error_count}张")

            if results:
                self.logger.info("📋 执行结果:")
                for result in results:
                    self.logger.info(f"   {result}")
            else:
                self.logger.warning("⚠️ 未找到输出文件")

        except Exception as e:
            self.logger.error(f"❌ 结果检查异常: {e}")

    def log_message(self, message, level="INFO"):
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def _copy_path(self, path):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(path)
            self.logger.info(f"已复制到剪贴板: {path}")
        except Exception:
            self.logger.warning("复制失败")

    def _open_changelog(self, event):
        """打开更新日志页面"""
        try:
            webbrowser.open(self.CHANGELOG_URL)
            self.logger.info("已打开更新日志页面")
        except Exception as e:
            self.logger.error(f"打开更新日志失败: {e}")
            # 如果无法打开浏览器，可以提供备用信息
            import tkinter.messagebox as msgbox

            msgbox.showinfo(
                "更新日志",
                "无法打开浏览器。\n"
                "请手动访问以下链接查看更新日志：\n"
                f"{self.CHANGELOG_URL}",
            )

    def create_steps_control_section(self, parent):
        # 左侧：基础步骤控制
        steps_frame = ttk.LabelFrame(parent, text="📋 执行步骤", padding="12")
        steps_frame.grid(
            row=2, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 15), padx=(0, 8)
        )

        # 执行推理（必选，不可取消）
        self.inference_checkbox = ttk.Checkbutton(
            steps_frame,
            text="🚀 执行推理 (必选)",
            variable=self.run_inference,
            command=self.on_step_change,
        )
        self.inference_checkbox.pack(anchor=tk.W, pady=3)
        # 设置为选中且禁用，放在创建Checkbutton之后立即设置
        self.inference_checkbox.state(
            ["selected", "disabled"]
        )  # 使用ttk state来同时设置选中和禁用
        self.run_inference.set(True)  # 同时设置变量

        # 格式转换
        self.conversion_checkbox = ttk.Checkbutton(
            steps_frame,
            text="🔄 格式转换",
            variable=self.run_conversion,
            command=self.on_step_change,
        )
        self.conversion_checkbox.pack(anchor=tk.W, pady=3)

        # 目标检测标题
        detection_title = tk.Label(
            steps_frame,
            text="🎯 目标检测评估 (需要格式转换)",
            font=self.fonts["subtitle"],
            fg=self.colors["primary"],
        )
        detection_title.pack(anchor=tk.W, pady=(8, 5))

        # 模型评估
        self.evaluation_checkbox = ttk.Checkbutton(
            steps_frame,
            text="📊 目标检测IoU评估",
            variable=self.run_evaluation,
            command=self.on_detection_evaluation_change,
        )
        self.evaluation_checkbox.pack(anchor=tk.W, pady=2)

        # 结果可视化
        self.visualization_checkbox = ttk.Checkbutton(
            steps_frame,
            text="🎨 目标检测bbox可视化",
            variable=self.run_visualization,
            command=self.on_detection_visualization_change,
        )
        self.visualization_checkbox.pack(anchor=tk.W, pady=2)

        # 添加分隔线和说明，让左侧内容更丰富
        separator_left = ttk.Separator(steps_frame, orient="horizontal")
        separator_left.pack(fill=tk.X, pady=(15, 10))

        # 流程说明标题
        flow_title = tk.Label(
            steps_frame,
            text="📋 执行流程说明",
            font=self.fonts["subtitle"],
            fg=self.colors["info"],
        )
        flow_title.pack(anchor=tk.W, pady=(0, 5))

        # 流程步骤说明
        flow_steps = [
            "1️⃣ 执行推理：调用AI模型处理图片",
            "2️⃣ 格式转换：RLE → LabelMe格式",
            "3️⃣ 模型评估：计算IoU准确率",
            "4️⃣ 结果可视化：生成对比图片",
        ]

        for step in flow_steps:
            step_label = tk.Label(
                steps_frame,
                text=step,
                font=self.fonts["small"],
                fg=self.colors["text_secondary"],
            )
            step_label.pack(anchor=tk.W, pady=1)

        # 添加提示信息
        tip_label = tk.Label(
            steps_frame,
            text="💡 提示：推理步骤为必选项，其他步骤可选择",
            font=self.fonts["small"],
            fg=self.colors["info"],
            wraplength=280,
        )
        tip_label.pack(anchor=tk.W, pady=(10, 0))

        # 右侧：语义分割评估
        semseg_frame = ttk.LabelFrame(
            parent, text="🔬 语义分割评估 (Belt/Coal类别)", padding="12"
        )
        semseg_frame.grid(
            row=2, column=1, sticky=(tk.W, tk.E, tk.N), pady=(0, 15), padx=(8, 0)
        )

        # 语义分割评估
        self.semseg_evaluation_checkbox = ttk.Checkbutton(
            semseg_frame,
            text="📊 语义分割IoU评估",
            variable=self.run_semseg_evaluation,
            command=self.on_semseg_evaluation_change,
        )
        self.semseg_evaluation_checkbox.pack(anchor=tk.W, pady=2)

        # 语义分割可视化
        self.semseg_visualization_checkbox = ttk.Checkbutton(
            semseg_frame,
            text="🎨 语义分割mask可视化",
            variable=self.run_semseg_visualization,
            command=self.on_semseg_visualization_change,
        )
        self.semseg_visualization_checkbox.pack(anchor=tk.W, pady=2)

        # 分隔线
        separator = ttk.Separator(semseg_frame, orient="horizontal")
        separator.pack(fill=tk.X, pady=(10, 8))

        # IoU阈值配置标题
        iou_title = tk.Label(
            semseg_frame,
            text="📊 IoU统计配置",
            font=self.fonts["subtitle"],
            fg=self.colors["info"],
        )
        iou_title.pack(anchor=tk.W, pady=(0, 5))

        # IoU阈值配置
        iou_threshold_frame = ttk.Frame(semseg_frame)
        iou_threshold_frame.pack(fill=tk.X, pady=(0, 5))

        # IoU阈值标签
        ttk.Label(iou_threshold_frame, text="IoU阈值:", font=self.fonts["label"]).pack(
            side=tk.LEFT
        )

        # IoU阈值滑动条
        iou_scale = ttk.Scale(
            iou_threshold_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.semseg_iou_threshold,
            length=120,
            command=self.on_semseg_iou_threshold_change,
        )
        iou_scale.pack(side=tk.LEFT, padx=(5, 0))

        # IoU阈值显示标签
        iou_display_label = tk.Label(
            iou_threshold_frame,
            textvariable=self.semseg_iou_threshold_display,
            font=self.fonts["label"],
            fg=self.colors["primary"],
            width=6,
        )
        iou_display_label.pack(side=tk.LEFT, padx=(5, 0))

        # IoU阈值说明
        iou_info_label = tk.Label(
            semseg_frame,
            text="统计Belt/Coal IoU ≥ 此值的样本占比",
            font=self.fonts["small"],
            fg=self.colors["text_secondary"],
        )
        iou_info_label.pack(anchor=tk.W, pady=(0, 10))

        # 第二个分隔线
        separator_right = ttk.Separator(semseg_frame, orient="horizontal")
        separator_right.pack(fill=tk.X, pady=(5, 10))

        # 对比说明标题
        compare_title = tk.Label(
            semseg_frame,
            text="🔄 功能对比",
            font=self.fonts["subtitle"],
            fg=self.colors["info"],
        )
        compare_title.pack(anchor=tk.W, pady=(0, 5))

        # 功能对比说明
        compare_info = [
            "🎯 目标检测：bbox格式，需要格式转换",
            "🔬 语义分割：RLE格式，直接使用推理响应",
            "📊 评估方式：IoU计算 + 统计分析",
            "🎨 可视化：mask覆盖 + 差分对比",
        ]

        for info in compare_info:
            info_label = tk.Label(
                semseg_frame,
                text=info,
                font=self.fonts["small"],
                fg=self.colors["text_secondary"],
            )
            info_label.pack(anchor=tk.W, pady=1)

        # 重要提示
        warning_label = tk.Label(
            semseg_frame,
            text="⚠️ 重要：目标检测评估与语义分割评估互斥",
            font=self.fonts["small"],
            fg=self.colors["warning"],
            wraplength=280,
        )
        warning_label.pack(anchor=tk.W, pady=(10, 0))

        # 下方：控制面板（水平排列按钮）
        control_frame = ttk.LabelFrame(
            parent,
            text="🎮 控制面板",
            padding="12",
        )
        control_frame.grid(
            row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15)
        )
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)

        # 执行按钮
        self.run_button = ttk.Button(
            control_frame,
            text="▶ 开始执行",
            command=self.start_pipeline,
        )
        self.run_button.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 4))

        # 停止按钮
        self.stop_button = ttk.Button(
            control_frame,
            text="⏹ 停止执行",
            command=self.stop_pipeline,
            state=tk.DISABLED,
        )
        self.stop_button.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(4, 0))

        # 添加快捷操作说明
        shortcut_label = tk.Label(
            control_frame,
            text="💡 快捷键：Ctrl+Enter 开始执行",
            font=self.fonts["small"],
            fg=self.colors["text_secondary"],
        )
        shortcut_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(12, 0))

        # 绑定快捷键
        self.root.bind("<Control-Return>", lambda e: self.start_pipeline())

    def on_step_change(self):
        # 检查当前选择状态
        has_semseg = (
            self.run_semseg_evaluation.get() or self.run_semseg_visualization.get()
        )
        has_detection = self.run_evaluation.get() or self.run_visualization.get()

        # === 互斥逻辑处理 ===
        # 语义分割和目标检测不能同时选择
        if has_semseg and has_detection:
            # 如果同时选择了，优先保持语义分割，清除目标检测
            # 这个策略可以根据用户需求调整
            self.run_evaluation.set(False)
            self.run_visualization.set(False)
            has_detection = False
            self.logger.info("🔄 检测到冲突，已自动切换到语义分割模式")

        # === 推理步骤管理 ===
        # 推理步骤始终保持选中和禁用状态（所有功能都需要推理）
        self.run_inference.set(True)
        self.inference_checkbox.state(["selected", "disabled"])

        # === 根据选择的功能模式配置界面 ===
        if has_semseg:
            # 语义分割模式配置
            self._configure_semseg_mode()
        elif has_detection:
            # 目标检测模式配置
            self._configure_detection_mode()
        else:
            # 自由模式配置（用户可以自由选择步骤）
            self._configure_free_mode()

        # === 更新界面显示 ===
        self.update_step_display()  # 更新步骤显示状态
        self.update_eval_viz_config_display()  # 更新评估和可视化配置显示
        self.update_semseg_config_display()  # 更新语义分割配置显示
        self.update_flow_display()  # 更新流向显示

    def _configure_semseg_mode(self):
        # 禁用目标检测选项
        self.evaluation_checkbox.config(state=tk.DISABLED)
        self.visualization_checkbox.config(state=tk.DISABLED)
        # 确保推理步骤选中
        self.run_inference.set(True)
        # 语义分割不需要格式转换，强制禁用并取消选择
        self.run_conversion.set(False)
        self.conversion_checkbox.config(state=tk.DISABLED)
        self.logger.info("🔬 语义分割模式：推理 → 直接评估/可视化")

    def _configure_detection_mode(self):
        # 禁用语义分割选项
        self.semseg_evaluation_checkbox.config(state=tk.DISABLED)
        self.semseg_visualization_checkbox.config(state=tk.DISABLED)
        # 强制启用格式转换和推理
        self.run_conversion.set(True)  # 强制启用格式转换
        self.conversion_checkbox.config(state=tk.DISABLED)  # 禁用用户修改
        self.run_inference.set(True)
        self.logger.info("🎯 目标检测模式：推理 → 格式转换 → 评估/可视化")

    def _configure_free_mode(self):
        # 恢复所有选项的可用状态
        self.evaluation_checkbox.config(state=tk.NORMAL)
        self.visualization_checkbox.config(state=tk.NORMAL)
        self.semseg_evaluation_checkbox.config(state=tk.NORMAL)
        self.semseg_visualization_checkbox.config(state=tk.NORMAL)
        # 格式转换恢复为可选状态，推理保持禁用
        self.conversion_checkbox.config(state=tk.NORMAL)
        # 推理保持选中且禁用
        self.run_inference.set(True)
        self.inference_checkbox.state(["disabled"])
        self.logger.info("💡 自由模式：可选择是否执行格式转换")

    def update_step_display(self):
        # 检查当前选择状态
        has_semseg = (
            self.run_semseg_evaluation.get() or self.run_semseg_visualization.get()
        )
        has_detection = self.run_evaluation.get() or self.run_visualization.get()

        # 更新复选框的文本显示，让用户了解依赖关系
        if has_semseg:
            # 语义分割模式：格式转换不需要
            self.conversion_checkbox.config(text="🔄 格式转换 (语义分割不需要)")
        elif has_detection:
            # 目标检测模式：格式转换必需
            self.conversion_checkbox.config(text="🔄 格式转换 (必需)")
        else:
            # 没有选择任何评估：格式转换可选
            self.conversion_checkbox.config(text="🔄 格式转换")

        # 确保执行推理始终选中
        self.run_inference.set(True)

    def create_progress_section(self, parent):
        progress_frame = ttk.LabelFrame(parent, text="📈 执行进度", padding="12")
        progress_frame.grid(
            row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15)
        )
        progress_frame.columnconfigure(0, weight=1)

        # 当前步骤标签
        self.current_step_label = tk.Label(
            progress_frame,
            text="✅ 准备就绪",
            font=self.fonts["subtitle"],
            fg=self.colors["info"],
        )
        self.current_step_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 8))

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        # 进度百分比标签
        self.progress_label = tk.Label(
            progress_frame,
            text="0%",
            font=self.fonts["label"],
            fg=self.colors["primary"],
        )
        self.progress_label.grid(row=1, column=1, padx=(15, 0))

        # 状态标签
        self.status_label = tk.Label(
            progress_frame,
            text="💡 点击'开始执行'来运行流水线",
            font=self.fonts["status"],
            fg=self.colors["info"],
        )
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

    def update_eval_viz_config_display(self):
        # 如果选择了模型评估或结果可视化，显示IoU阈值配置
        if self.run_evaluation.get() or self.run_visualization.get():
            self.eval_config_frame.grid(
                row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10)
            )
        else:
            self.eval_config_frame.grid_remove()

        # 如果选择了结果可视化，显示可视化配置
        if self.run_visualization.get():
            self.viz_config_frame.grid(
                row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10)
            )
        else:
            self.viz_config_frame.grid_remove()

    def update_viz_config_display(self):
        # 这个方法保持向后兼容，但实际调用新的方法
        self.update_eval_viz_config_display()

    def on_iou_threshold_change(self, value):
        # 将值格式化为两位小数
        formatted_value = f"{float(value):.2f}"
        self.iou_threshold_display.set(formatted_value)

    def on_semseg_iou_threshold_change(self, value):
        # 将值格式化为两位小数
        formatted_value = f"{float(value):.2f}"
        self.semseg_iou_threshold_display.set(formatted_value)


def main():
    # 创建GUI应用实例并运行
    app = PipelineGUI()
    app.root.mainloop()


if __name__ == "__main__":
    main()
