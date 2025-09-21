from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from client.image_infer.grpc import GRPCClientConfig
from client.image_infer.grpc import infer_dir_to_jsons as grpc_infer
from client.image_infer.https import HTTPSClientConfig
from client.image_infer.https import infer_dir_to_jsons as https_infer
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

logger = logging.getLogger(__name__)


@dataclass
class PipelinePaths:
    images: str
    gt_labelme: str
    raw_json_dir: str
    converted_dir: str
    detection_report: str
    detection_viz: str
    detection_viz_correct: str
    detection_viz_error: str
    semseg_report: str
    semseg_pred_dir: str
    semseg_viz: str


@dataclass
class HTTPSSettings:
    url: str
    stream: str
    access_key: str
    secret_key: str
    workers: int
    raw_dir: str


@dataclass
class GRPCSettings:
    server: str
    stream: str
    task_id: str
    workers: int
    output_dir: str


@dataclass
class PipelineSettings:
    run_https: bool
    run_grpc: bool
    run_convert: bool
    run_det_eval: bool
    run_det_viz: bool
    run_semseg_eval: bool
    run_semseg_viz: bool
    https: HTTPSSettings
    grpc: GRPCSettings
    paths: PipelinePaths
    convert_workers: int
    iou_threshold: float
    semseg_iou_threshold: float


DEFAULT_STATE: dict[str, Any] = {
    "images": "data/images",
    "gt_labelme": "data/labelme",
    "converted_dir": "outputs/labelme_pred",
    "run_https": False,
    "https_url": "https://",
    "https_stream": "",
    "https_access_key": "",
    "https_secret_key": "",
    "https_workers": 1,
    "https_output_dir": "outputs/https/responses",
    "run_grpc": False,
    "grpc_server": "",
    "grpc_stream": "",
    "grpc_task_id": "",
    "grpc_workers": 1,
    "grpc_output_dir": "outputs/grpc/responses",
    "run_convert": True,
    "raw_json_dir": "outputs/https/responses",
    "convert_workers": 4,
    "run_det_eval": True,
    "detection_report": "outputs/detection/eval.csv",
    "iou_threshold": 0.5,
    "run_det_viz": False,
    "detection_viz": "outputs/detection/viz",
    "detection_viz_correct": "outputs/detection/viz/correct",
    "detection_viz_error": "outputs/detection/viz/error",
    "run_semseg_eval": False,
    "semseg_pred_dir": "outputs/https/responses",
    "semseg_report": "outputs/semseg/eval.csv",
    "semseg_iou_threshold": 0.8,
    "run_semseg_viz": False,
    "semseg_viz": "outputs/semseg/viz",
}


class StreamlitLogHandler(logging.Handler):
    """Mirror log records into a Streamlit placeholder."""

    def __init__(self, placeholder: Any) -> None:
        super().__init__()
        self.placeholder = placeholder
        self._buffer: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self._buffer.append(message)
        self.placeholder.text("\n".join(self._buffer[-400:]))


def init_session_state() -> None:
    for key, value in DEFAULT_STATE.items():
        st.session_state.setdefault(key, value)


def _expand(path_str: str) -> Path:
    return Path(path_str).expanduser()


def _resolve_detection_correct_dir(paths: PipelinePaths) -> Path:
    base = _expand(paths.detection_viz)
    value = paths.detection_viz_correct.strip()
    if value:
        return _expand(value)
    correct = base / "correct"
    paths.detection_viz_correct = str(correct)
    return correct


def _resolve_detection_error_dir(paths: PipelinePaths) -> Path:
    base = _expand(paths.detection_viz)
    value = paths.detection_viz_error.strip()
    if value:
        return _expand(value)
    error = base / "error"
    paths.detection_viz_error = str(error)
    return error


def step_https(settings: PipelineSettings) -> None:
    config = HTTPSClientConfig(
        img_stream_url=settings.https.url.strip(),
        stream_name=settings.https.stream.strip(),
        access_key=settings.https.access_key.strip(),
        secret_key=settings.https.secret_key.strip(),
        raw_responses_dir=_expand(settings.https.raw_dir),
        max_workers=max(1, settings.https.workers),
    )
    images_dir = _expand(settings.paths.images)
    https_infer(images_dir, config)
    raw_dir = _expand(settings.https.raw_dir).resolve()
    settings.paths.raw_json_dir = str(raw_dir)
    settings.paths.semseg_pred_dir = str(raw_dir)


def step_grpc(settings: PipelineSettings) -> None:
    config = GRPCClientConfig(
        server_address=settings.grpc.server.strip(),
        stream_name=settings.grpc.stream.strip(),
        task_id=settings.grpc.task_id.strip(),
        output_dir=_expand(settings.grpc.output_dir),
        max_workers=max(1, settings.grpc.workers),
    )
    images_dir = _expand(settings.paths.images)
    grpc_infer(images_dir, config)
    raw_dir = _expand(settings.grpc.output_dir).resolve()
    settings.paths.raw_json_dir = str(raw_dir)


def step_convert_labelme(settings: PipelineSettings) -> None:
    converter = LabelmeConverter(
        LabelmeConverterConfig(
            raw_json_dir=_expand(settings.paths.raw_json_dir),
            image_dir=_expand(settings.paths.images),
            output_dir=_expand(settings.paths.converted_dir),
            max_workers=max(1, settings.convert_workers),
        )
    )
    converter.convert_directory()


def step_detection_eval(settings: PipelineSettings) -> None:
    eval_config = DetectionEvalConfig(
        gt_dir=_expand(settings.paths.gt_labelme),
        pred_dir=_expand(settings.paths.converted_dir),
        output_csv=_expand(settings.paths.detection_report),
        iou_threshold=settings.iou_threshold,
    )
    DetectionEvaluator(eval_config).evaluate()


def step_detection_viz(settings: PipelineSettings) -> None:
    viz_config = DetectionVizConfig(
        gt_dir=_expand(settings.paths.gt_labelme),
        pred_dir=_expand(settings.paths.converted_dir),
        images_dir=_expand(settings.paths.images),
        output_correct_dir=_resolve_detection_correct_dir(settings.paths),
        output_error_dir=_resolve_detection_error_dir(settings.paths),
        match_method="hungarian",
        iou_threshold=settings.iou_threshold,
        max_workers=4,
    )
    DetectionVisualizer(viz_config).visualize_directory()


def step_semseg_eval(settings: PipelineSettings) -> None:
    eval_config = SemSegEvalConfig(
        pred_dir=_expand(settings.paths.semseg_pred_dir),
        gt_dir=_expand(settings.paths.gt_labelme),
        output_csv=_expand(settings.paths.semseg_report),
        iou_threshold=settings.semseg_iou_threshold,
    )
    SemSegEvaluator(eval_config).evaluate()


def step_semseg_viz(settings: PipelineSettings) -> None:
    viz_config = SemSegVizConfig(
        pred_dir=_expand(settings.paths.semseg_pred_dir),
        image_dir=_expand(settings.paths.images),
        output_dir=_expand(settings.paths.semseg_viz),
    )
    SemSegVisualizer(viz_config).visualize_directory()


STEP_SEQUENCE: list[tuple[str, Callable[[PipelineSettings], None], str]] = [
    ("HTTPS 推理", step_https, "run_https"),
    ("gRPC 推理", step_grpc, "run_grpc"),
    ("结果转换", step_convert_labelme, "run_convert"),
    ("目标检测评估", step_detection_eval, "run_det_eval"),
    ("目标检测可视化", step_detection_viz, "run_det_viz"),
    ("语义分割评估", step_semseg_eval, "run_semseg_eval"),
    ("语义分割可视化", step_semseg_viz, "run_semseg_viz"),
]


def build_settings_from_session() -> PipelineSettings:
    paths = PipelinePaths(
        images=st.session_state["images"].strip(),
        gt_labelme=st.session_state["gt_labelme"].strip(),
        raw_json_dir=st.session_state["raw_json_dir"].strip(),
        converted_dir=st.session_state["converted_dir"].strip(),
        detection_report=st.session_state["detection_report"].strip(),
        detection_viz=st.session_state["detection_viz"].strip(),
        detection_viz_correct=st.session_state["detection_viz_correct"].strip(),
        detection_viz_error=st.session_state["detection_viz_error"].strip(),
        semseg_report=st.session_state["semseg_report"].strip(),
        semseg_pred_dir=st.session_state["semseg_pred_dir"].strip(),
        semseg_viz=st.session_state["semseg_viz"].strip(),
    )
    https = HTTPSSettings(
        url=st.session_state["https_url"].strip(),
        stream=st.session_state["https_stream"].strip(),
        access_key=st.session_state["https_access_key"].strip(),
        secret_key=st.session_state["https_secret_key"].strip(),
        workers=int(st.session_state["https_workers"]),
        raw_dir=st.session_state["https_output_dir"].strip(),
    )
    grpc = GRPCSettings(
        server=st.session_state["grpc_server"].strip(),
        stream=st.session_state["grpc_stream"].strip(),
        task_id=st.session_state["grpc_task_id"].strip(),
        workers=int(st.session_state["grpc_workers"]),
        output_dir=st.session_state["grpc_output_dir"].strip(),
    )
    return PipelineSettings(
        run_https=bool(st.session_state["run_https"]),
        run_grpc=bool(st.session_state["run_grpc"]),
        run_convert=bool(st.session_state["run_convert"]),
        run_det_eval=bool(st.session_state["run_det_eval"]),
        run_det_viz=bool(st.session_state["run_det_viz"]),
        run_semseg_eval=bool(st.session_state["run_semseg_eval"]),
        run_semseg_viz=bool(st.session_state["run_semseg_viz"]),
        https=https,
        grpc=grpc,
        paths=paths,
        convert_workers=max(1, int(st.session_state["convert_workers"])),
        iou_threshold=float(st.session_state["iou_threshold"]),
        semseg_iou_threshold=float(st.session_state["semseg_iou_threshold"]),
    )


def sync_paths_to_session(paths: PipelinePaths) -> None:
    st.session_state["images"] = paths.images
    st.session_state["gt_labelme"] = paths.gt_labelme
    st.session_state["raw_json_dir"] = paths.raw_json_dir
    st.session_state["converted_dir"] = paths.converted_dir
    st.session_state["detection_report"] = paths.detection_report
    st.session_state["detection_viz"] = paths.detection_viz
    st.session_state["detection_viz_correct"] = paths.detection_viz_correct
    st.session_state["detection_viz_error"] = paths.detection_viz_error
    st.session_state["semseg_report"] = paths.semseg_report
    st.session_state["semseg_pred_dir"] = paths.semseg_pred_dir
    st.session_state["semseg_viz"] = paths.semseg_viz


def run_pipeline(
    settings: PipelineSettings,
    status_placeholder: Any,
    progress_placeholder: Any,
) -> None:
    active_steps = [
        (name, func) for name, func, flag in STEP_SEQUENCE if getattr(settings, flag)
    ]
    if not active_steps:
        status_placeholder.info("请选择至少一个步骤后再执行。")
        progress_placeholder.empty()
        return

    progress_bar = progress_placeholder.progress(0.0)
    status_messages: list[str] = []

    def render_status() -> None:
        status_placeholder.markdown(
            "\n".join(f"- {message}" for message in status_messages),
            unsafe_allow_html=False,
        )

    total = len(active_steps)
    for idx, (name, func) in enumerate(active_steps, start=1):
        status_messages.append(f"🟡 {name} 执行中…")
        render_status()
        logger.info("开始执行步骤: %s", name)
        try:
            func(settings)
        except Exception as exc:  # noqa: BLE001
            logger.exception("步骤失败: %s", name)
            status_messages[-1] = f"❌ {name} 失败: {exc}"
            render_status()
            progress_bar.progress((idx - 1) / total)
            raise
        else:
            logger.info("完成步骤: %s", name)
            status_messages[-1] = f"✅ {name} 完成"
            render_status()
            progress_bar.progress(idx / total)

    progress_bar.progress(1.0)


def render_configuration_controls() -> None:
    st.subheader("通用路径配置")
    st.text_input("图片目录", key="images")
    st.text_input("GT LabelMe 目录", key="gt_labelme")
    st.text_input("转换输出目录", key="converted_dir")

    st.markdown("---")
    st.subheader("HTTPS 推理")
    st.checkbox("执行 HTTPS 推理", key="run_https", value=st.session_state["run_https"])
    st.text_input("推理 URL", key="https_url")
    st.text_input("流名称", key="https_stream")
    st.text_input("Access Key", key="https_access_key")
    st.text_input("Secret Key", key="https_secret_key", type="password")
    st.number_input(
        "并发数",
        min_value=1,
        max_value=64,
        value=st.session_state["https_workers"],
        key="https_workers",
    )
    st.text_input("输出目录", key="https_output_dir")

    st.markdown("---")
    st.subheader("gRPC 推理")
    st.checkbox("执行 gRPC 推理", key="run_grpc", value=st.session_state["run_grpc"])
    st.text_input("服务器地址", key="grpc_server")
    st.text_input("流名称", key="grpc_stream")
    st.text_input("Task ID", key="grpc_task_id")
    st.number_input(
        "并发数",
        min_value=1,
        max_value=64,
        value=st.session_state["grpc_workers"],
        key="grpc_workers",
    )
    st.text_input("输出目录", key="grpc_output_dir")

    st.markdown("---")
    st.subheader("推理结果转换")
    st.checkbox(
        "执行推理结果转换 (LabelMe)",
        key="run_convert",
        value=st.session_state["run_convert"],
    )
    st.text_input("原始推理 JSON 目录", key="raw_json_dir")
    st.number_input(
        "并行线程数",
        min_value=1,
        max_value=64,
        value=st.session_state["convert_workers"],
        key="convert_workers",
    )

    st.markdown("---")
    st.subheader("目标检测评估")
    st.checkbox(
        "执行检测评估", key="run_det_eval", value=st.session_state["run_det_eval"]
    )
    st.text_input("评估输出 CSV", key="detection_report")
    st.number_input(
        "IoU 阈值",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=float(st.session_state["iou_threshold"]),
        key="iou_threshold",
    )
    st.checkbox(
        "生成检测可视化",
        key="run_det_viz",
        value=st.session_state["run_det_viz"],
    )
    st.text_input("可视化输出目录", key="detection_viz")
    st.text_input("正确案例目录", key="detection_viz_correct")
    st.text_input("错误案例目录", key="detection_viz_error")

    st.markdown("---")
    st.subheader("语义分割评估")
    st.checkbox(
        "执行语义分割评估",
        key="run_semseg_eval",
        value=st.session_state["run_semseg_eval"],
    )
    st.text_input("预测 JSON 目录", key="semseg_pred_dir")
    st.text_input("评估输出 CSV", key="semseg_report")
    st.number_input(
        "IoU 阈值 (语义分割)",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=float(st.session_state["semseg_iou_threshold"]),
        key="semseg_iou_threshold",
    )
    st.checkbox(
        "生成语义分割可视化",
        key="run_semseg_viz",
        value=st.session_state["run_semseg_viz"],
    )
    st.text_input("可视化输出目录", key="semseg_viz")


def main() -> None:
    st.set_page_config(page_title="AI Pipeline 控制台", page_icon="🧪", layout="wide")
    init_session_state()

    st.title("AI Pipeline 控制台")
    st.caption("使用 Streamlit 管理推理、转换与评估流程")

    render_configuration_controls()

    execute = st.button("开始执行", type="primary")
    run_area = st.container()

    if execute:
        settings = build_settings_from_session()
        with run_area:
            status_col, log_col = st.columns([2, 3])
            with status_col:
                st.subheader("步骤状态")
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
            with log_col:
                st.subheader("执行日志")
                log_placeholder = st.empty()

            handler = StreamlitLogHandler(log_placeholder)
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )

            root_logger = logging.getLogger()
            previous_level = root_logger.level
            if previous_level > logging.INFO:
                root_logger.setLevel(logging.INFO)
            root_logger.addHandler(handler)

            try:
                run_pipeline(settings, status_placeholder, progress_placeholder)
            except ValidationError as exc:
                st.error(f"参数错误: {exc}")
            except AIError as exc:
                st.error(f"执行失败: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"未捕获的异常: {exc}")
            else:
                st.success("全部任务完成。")
            finally:
                sync_paths_to_session(settings.paths)
                root_logger.removeHandler(handler)
                handler.close()
                root_logger.setLevel(previous_level)


if __name__ == "__main__":
    main()
