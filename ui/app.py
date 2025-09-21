"""图形化界面：组合推理、转换与评估流程。"""

from __future__ import annotations

import logging
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.image_infer.grpc import GRPCClientConfig
from client.image_infer.grpc import infer_dir_to_jsons as grpc_infer
from client.image_infer.https import (
    HTTPSClientConfig,
)
from client.image_infer.https import (
    infer_dir_to_jsons as https_infer,
)
from core.converters.labelme import LabelmeConverter, LabelmeConverterConfig
from eval.object_detection import (
    DetectionEvalConfig,
    DetectionEvaluator,
    DetectionVisualizer,
    DetectionVizConfig,
)
from eval.semantic_segmentation import (
    SemSegEvalConfig,
    SemSegEvaluator,
    SemSegVisualizer,
    SemSegVizConfig,
)
from utils.exceptions import AIError, ValidationError
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class HTTPSFields:
    url: tk.StringVar
    stream: tk.StringVar
    access_key: tk.StringVar
    secret_key: tk.StringVar
    workers: tk.IntVar
    raw_dir: tk.StringVar


@dataclass(slots=True)
class GRPCFields:
    server: tk.StringVar
    stream: tk.StringVar
    task_id: tk.StringVar
    workers: tk.IntVar
    output_dir: tk.StringVar


@dataclass(slots=True)
class GeneralPaths:
    images: tk.StringVar
    gt_labelme: tk.StringVar
    raw_json_dir: tk.StringVar
    converted_dir: tk.StringVar
    detection_report: tk.StringVar
    detection_viz: tk.StringVar
    detection_viz_correct: tk.StringVar
    detection_viz_error: tk.StringVar
    semseg_report: tk.StringVar
    semseg_pred_dir: tk.StringVar
    semseg_viz: tk.StringVar


class TextHandler(logging.Handler):
    """把日志写入 Tk 文本控件。"""

    def __init__(self, widget: ScrolledText) -> None:
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.widget.after(0, self._append, msg)

    def _append(self, message: str) -> None:
        self.widget.configure(state=tk.NORMAL)
        self.widget.insert(tk.END, message + "\n")
        self.widget.configure(state=tk.DISABLED)
        self.widget.see(tk.END)


class PipelineGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI Pipeline GUI")
        self.root.geometry("900x900")

        self.is_running = False
        self._stop_requested = False

        self._build_variables()
        self._build_layout()
        self._setup_logging()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_variables(self) -> None:
        self.https_vars = HTTPSFields(
            url=tk.StringVar(value="https://"),
            stream=tk.StringVar(),
            access_key=tk.StringVar(),
            secret_key=tk.StringVar(),
            workers=tk.IntVar(value=1),
            raw_dir=tk.StringVar(value=str(Path("outputs/https/responses"))),
        )
        self.grpc_vars = GRPCFields(
            server=tk.StringVar(),
            stream=tk.StringVar(),
            task_id=tk.StringVar(),
            workers=tk.IntVar(value=1),
            output_dir=tk.StringVar(value=str(Path("outputs/grpc/responses"))),
        )
        self.paths = GeneralPaths(
            images=tk.StringVar(value=str(Path("data/images"))),
            gt_labelme=tk.StringVar(value=str(Path("data/labelme"))),
            raw_json_dir=tk.StringVar(value=str(Path("outputs/https/responses"))),
            converted_dir=tk.StringVar(value=str(Path("outputs/labelme_pred"))),
            detection_report=tk.StringVar(
                value=str(Path("outputs/detection/eval.csv"))
            ),
            detection_viz=tk.StringVar(value=str(Path("outputs/detection/viz"))),
            detection_viz_correct=tk.StringVar(
                value=str(Path("outputs/detection/viz/correct"))
            ),
            detection_viz_error=tk.StringVar(
                value=str(Path("outputs/detection/viz/error"))
            ),
            semseg_report=tk.StringVar(value=str(Path("outputs/semseg/eval.csv"))),
            semseg_pred_dir=tk.StringVar(value=str(Path("outputs/https/responses"))),
            semseg_viz=tk.StringVar(value=str(Path("outputs/semseg/viz"))),
        )

        self.run_https = tk.BooleanVar(value=False)
        self.run_grpc = tk.BooleanVar(value=False)
        self.run_convert = tk.BooleanVar(value=True)
        self.run_det_eval = tk.BooleanVar(value=True)
        self.run_det_viz = tk.BooleanVar(value=False)
        self.run_semseg_eval = tk.BooleanVar(value=False)
        self.run_semseg_viz = tk.BooleanVar(value=False)
        self.iou_threshold = tk.DoubleVar(value=0.5)
        self.semseg_iou_threshold = tk.DoubleVar(value=0.8)
        self.convert_workers = tk.IntVar(value=4)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        # 通用路径
        general_frame = ttk.LabelFrame(container, text="通用路径配置", padding=10)
        general_frame.pack(fill=tk.X, pady=5)
        self._add_entry(general_frame, "图片目录", self.paths.images, row=0)
        self._add_entry(general_frame, "GT LabelMe 目录", self.paths.gt_labelme, row=1)
        self._add_entry(general_frame, "转换输出目录", self.paths.converted_dir, row=2)

        # HTTPS 配置
        https_frame = ttk.LabelFrame(container, text="HTTPS 推理", padding=10)
        https_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            https_frame,
            text="执行 HTTPS 推理",
            variable=self.run_https,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        self._add_entry(https_frame, "推理 URL", self.https_vars.url, row=1)
        self._add_entry(https_frame, "流名称", self.https_vars.stream, row=2)
        self._add_entry(https_frame, "Access Key", self.https_vars.access_key, row=3)
        self._add_entry(
            https_frame, "Secret Key", self.https_vars.secret_key, row=4, show="*"
        )
        self._add_entry(https_frame, "并发数", self.https_vars.workers, row=5)
        self._add_entry(https_frame, "输出目录", self.https_vars.raw_dir, row=6)

        # gRPC 配置
        grpc_frame = ttk.LabelFrame(container, text="gRPC 推理", padding=10)
        grpc_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            grpc_frame,
            text="执行 gRPC 推理",
            variable=self.run_grpc,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        self._add_entry(grpc_frame, "服务器地址", self.grpc_vars.server, row=1)
        self._add_entry(grpc_frame, "流名称", self.grpc_vars.stream, row=2)
        self._add_entry(grpc_frame, "Task ID", self.grpc_vars.task_id, row=3)
        self._add_entry(grpc_frame, "并发数", self.grpc_vars.workers, row=4)
        self._add_entry(grpc_frame, "输出目录", self.grpc_vars.output_dir, row=5)

        # 转换
        convert_frame = ttk.LabelFrame(container, text="推理结果转换", padding=10)
        convert_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            convert_frame,
            text="执行推理结果转换 (LabelMe)",
            variable=self.run_convert,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        self._add_entry(
            convert_frame, "原始推理 JSON 目录", self.paths.raw_json_dir, row=1
        )
        self._add_entry(
            convert_frame, "输出 LabelMe 目录", self.paths.converted_dir, row=2
        )
        self._add_entry(convert_frame, "并行线程数", self.convert_workers, row=3)

        # 目标检测评估/可视化
        det_frame = ttk.LabelFrame(container, text="目标检测评估", padding=10)
        det_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            det_frame,
            text="执行检测评估",
            variable=self.run_det_eval,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        self._add_entry(det_frame, "评估输出 CSV", self.paths.detection_report, row=1)
        self._add_entry(det_frame, "IoU 阈值", self.iou_threshold, row=2)

        ttk.Checkbutton(
            det_frame,
            text="生成检测可视化",
            variable=self.run_det_viz,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self._add_entry(det_frame, "可视化输出目录", self.paths.detection_viz, row=4)
        self._add_entry(
            det_frame, "正确案例目录", self.paths.detection_viz_correct, row=5
        )
        self._add_entry(
            det_frame, "错误案例目录", self.paths.detection_viz_error, row=6
        )

        # 语义分割
        semseg_frame = ttk.LabelFrame(container, text="语义分割评估", padding=10)
        semseg_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            semseg_frame,
            text="执行语义分割评估",
            variable=self.run_semseg_eval,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        self._add_entry(
            semseg_frame, "预测 JSON 目录", self.paths.semseg_pred_dir, row=1
        )
        self._add_entry(semseg_frame, "评估输出 CSV", self.paths.semseg_report, row=2)
        self._add_entry(semseg_frame, "IoU 阈值", self.semseg_iou_threshold, row=3)

        ttk.Checkbutton(
            semseg_frame,
            text="生成语义分割可视化",
            variable=self.run_semseg_viz,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self._add_entry(semseg_frame, "可视化输出目录", self.paths.semseg_viz, row=5)

        # 控制按钮
        control_frame = ttk.Frame(container)
        control_frame.pack(fill=tk.X, pady=10)
        self.run_button = ttk.Button(
            control_frame, text="开始执行", command=self.start_pipeline
        )
        self.run_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(
            control_frame, text="停止", command=self.stop_pipeline, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # 日志展示
        log_frame = ttk.LabelFrame(container, text="执行日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_view = ScrolledText(log_frame, height=15, state=tk.DISABLED)
        self.log_view.pack(fill=tk.BOTH, expand=True)

    def _add_entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        *,
        row: int,
        show: str | None = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=variable, show=show)
        entry.grid(row=row, column=1, sticky="we", padx=5, pady=2)
        parent.grid_columnconfigure(1, weight=1)

    def _setup_logging(self) -> None:
        handler = TextHandler(self.log_view)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # 运行逻辑
    # ------------------------------------------------------------------
    def start_pipeline(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._stop_requested = False
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        thread = threading.Thread(target=self._run_pipeline_thread, daemon=True)
        thread.start()

    def stop_pipeline(self) -> None:
        self._stop_requested = True
        logger.warning("收到停止请求，当前步骤执行完毕后停止。")

    def _run_pipeline_thread(self) -> None:
        try:
            steps = [
                (self.run_https.get(), self._step_https_infer, "HTTPS 推理"),
                (self.run_grpc.get(), self._step_grpc_infer, "gRPC 推理"),
                (self.run_convert.get(), self._step_convert_labelme, "结果转换"),
                (self.run_det_eval.get(), self._step_detection_eval, "目标检测评估"),
                (self.run_det_viz.get(), self._step_detection_viz, "目标检测可视化"),
                (self.run_semseg_eval.get(), self._step_semseg_eval, "语义分割评估"),
                (self.run_semseg_viz.get(), self._step_semseg_viz, "语义分割可视化"),
            ]
            for enabled, func, name in steps:
                if self._stop_requested:
                    logger.warning("执行已中止。")
                    break
                if not enabled:
                    continue
                logger.info("开始执行步骤: %s", name)
                func()
                logger.info("完成步骤: %s", name)
        except ValidationError as exc:
            logger.error("参数错误: %s", exc)
        except AIError as exc:
            logger.error("执行失败: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("未捕获的异常: %s", exc)
        finally:
            self.root.after(0, self._on_pipeline_finished)

    def _on_pipeline_finished(self) -> None:
        self.is_running = False
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        logger.info("全部任务完成。")

    # ------------------------------------------------------------------
    # 各步骤实现
    # ------------------------------------------------------------------
    def _step_https_infer(self) -> None:
        config = HTTPSClientConfig(
            img_stream_url=self.https_vars.url.get().strip(),
            stream_name=self.https_vars.stream.get().strip(),
            access_key=self.https_vars.access_key.get().strip(),
            secret_key=self.https_vars.secret_key.get().strip(),
            raw_responses_dir=Path(self.https_vars.raw_dir.get()),
            max_workers=max(1, self.https_vars.workers.get()),
        )
        images_dir = Path(self.paths.images.get())
        https_infer(images_dir, config)
        raw_dir = Path(self.https_vars.raw_dir.get()).resolve()
        self.paths.raw_json_dir.set(str(raw_dir))
        self.paths.semseg_pred_dir.set(str(raw_dir))

    def _step_grpc_infer(self) -> None:
        config = GRPCClientConfig(
            server_address=self.grpc_vars.server.get().strip(),
            stream_name=self.grpc_vars.stream.get().strip(),
            task_id=self.grpc_vars.task_id.get().strip(),
            output_dir=Path(self.grpc_vars.output_dir.get()),
            max_workers=max(1, self.grpc_vars.workers.get()),
        )
        images_dir = Path(self.paths.images.get())
        grpc_infer(images_dir, config)
        raw_dir = Path(self.grpc_vars.output_dir.get()).resolve()
        self.paths.raw_json_dir.set(str(raw_dir))

    def _step_convert_labelme(self) -> None:
        converter = LabelmeConverter(
            LabelmeConverterConfig(
                raw_json_dir=Path(self.paths.raw_json_dir.get()),
                image_dir=Path(self.paths.images.get()),
                output_dir=Path(self.paths.converted_dir.get()),
                max_workers=max(1, self.convert_workers.get()),
            )
        )
        converter.convert_directory()

    def _step_detection_eval(self) -> None:
        eval_config = DetectionEvalConfig(
            gt_dir=Path(self.paths.gt_labelme.get()),
            pred_dir=Path(self.paths.converted_dir.get()),
            output_csv=Path(self.paths.detection_report.get()),
            iou_threshold=self.iou_threshold.get(),
        )
        DetectionEvaluator(eval_config).evaluate()

    def _step_detection_viz(self) -> None:
        viz_config = DetectionVizConfig(
            gt_dir=Path(self.paths.gt_labelme.get()),
            pred_dir=Path(self.paths.converted_dir.get()),
            images_dir=Path(self.paths.images.get()),
            output_correct_dir=self._resolve_detection_correct_dir(),
            output_error_dir=self._resolve_detection_error_dir(),
            match_method="hungarian",
            iou_threshold=self.iou_threshold.get(),
            max_workers=4,
        )
        DetectionVisualizer(viz_config).visualize_directory()

    def _step_semseg_eval(self) -> None:
        eval_config = SemSegEvalConfig(
            pred_dir=Path(self.paths.semseg_pred_dir.get()),
            gt_dir=Path(self.paths.gt_labelme.get()),
            output_csv=Path(self.paths.semseg_report.get()),
            iou_threshold=self.semseg_iou_threshold.get(),
        )
        SemSegEvaluator(eval_config).evaluate()

    def _step_semseg_viz(self) -> None:
        viz_config = SemSegVizConfig(
            pred_dir=Path(self.paths.semseg_pred_dir.get()),
            image_dir=Path(self.paths.images.get()),
            output_dir=Path(self.paths.semseg_viz.get()),
        )
        SemSegVisualizer(viz_config).visualize_directory()

    def _resolve_detection_correct_dir(self) -> Path:
        base = Path(self.paths.detection_viz.get())
        value = self.paths.detection_viz_correct.get().strip()
        if value:
            return Path(value)
        correct = base / "correct"
        self.paths.detection_viz_correct.set(str(correct))
        return correct

    def _resolve_detection_error_dir(self) -> Path:
        base = Path(self.paths.detection_viz.get())
        value = self.paths.detection_viz_error.get().strip()
        if value:
            return Path(value)
        error = base / "error"
        self.paths.detection_viz_error.set(str(error))
        return error

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    gui = PipelineGUI()
    gui.run()


if __name__ == "__main__":
    main()
