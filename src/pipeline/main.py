from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from clients.img_infer.grpc.standard import infer_dir as grpc_stream

# ---- 业务模块 ----
from clients.img_infer.https.client import infer_dir as https_stream
from core.labelme.converter import batch_convert_dir as convert_preds_to_labelme
from utils.exception import AppError

from eval.detection.report import DetectionEvalConfig as DetEvalConfig
from eval.detection.report import evaluate_dir_to_csv
from eval.detection.visualize import VizConfig as DetVizConfig
from eval.detection.visualize import visualize_dir as det_visualize
from eval.segmentation import (
    SegEvalConfig,
    SegVizConfig,
)
from eval.segmentation import (
    evaluate_dir_to_csv as seg_evaluate_dir_to_csv,
)
from eval.segmentation import (
    visualize_dir as seg_visualize_dir,
)
from utils.logger import get_logger

log = get_logger("pipeline")


# ============ 配置结构 ============


@dataclass
class HTTPSConfig:
    img_stream_url: str
    stream_name: str
    access_key: str
    secret_key: str
    raw_responses_dir: str = "https/responses"
    pred_jsons_dir: str = "https/pred_jsons"
    max_workers: int = 1


@dataclass
class GRPCStandardConfig:
    grpc_server: str
    task_id: str
    stream_name: str
    raw_responses_dir: str = "grpc_standard/responses"
    pred_jsons_dir: str = "grpc_standard/pred_jsons"
    max_workers: int = 1


@dataclass
class DetEvalRuntime:
    iou_threshold: float = 0.5
    eval_output_file: str = "reports/evaluation_report.csv"
    viz_output_dir: str = "reports/visualization_results"
    viz_mode_stats: bool = True
    max_workers: int = 1


@dataclass
class SegEvalRuntime:
    enabled: bool = False
    iou_threshold: float = 0.8
    eval_output_file: str = "reports/semseg_eval.csv"
    viz_output_dir: str = "reports/semseg_vis_masks"
    max_workers: int = 1  # 目前评估内部串行/轻并行，留作扩展


@dataclass
class StepSwitch:
    run_inference: bool = True  # 必选
    run_conversion: bool = True  # 目标检测需要；语义分割不需要
    run_evaluation: bool = True  # 目标检测评估
    run_visualization: bool = True  # 目标检测可视化
    run_seg_evaluation: bool = False  # 语义分割评估（与检测互斥）
    run_seg_visualization: bool = False  # 语义分割可视化（与检测互斥）


@dataclass
class PipelineConfig:
    inference_type: str  # "https" | "grpc_standard"
    images_dir: str
    gt_jsons_dir: str
    https: Optional[HTTPSConfig] = None
    grpc: Optional[GRPCStandardConfig] = None
    det: DetEvalRuntime = field(default_factory=DetEvalRuntime)
    seg: SegEvalRuntime = field(default_factory=SegEvalRuntime)
    steps: StepSwitch = field(default_factory=StepSwitch)


# ============ 运行控制 ============

_progress_cb: Optional[Callable[[str, int], None]] = None
_stop_flag: bool = False


def set_progress_callback(cb: Optional[Callable[[str, int], None]]):
    global _progress_cb
    _progress_cb = cb


def request_stop():
    global _stop_flag
    _stop_flag = True


def _stopped() -> bool:
    return _stop_flag


def _progress(step: str, p: int):
    if _progress_cb:
        try:
            _progress_cb(step, p)
        except Exception:
            pass


# ============ 工具函数 ============


def _ensure_dirs(cfg: PipelineConfig):
    # 推理输出目录 & 评估可视化目录
    if cfg.inference_type == "https":
        assert cfg.https, "HTTPS 配置缺失"
        Path(cfg.https.raw_responses_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.https.pred_jsons_dir).mkdir(parents=True, exist_ok=True)
    elif cfg.inference_type == "grpc_standard":
        assert cfg.grpc, "gRPC 标准 配置缺失"
        Path(cfg.grpc.raw_responses_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.grpc.pred_jsons_dir).mkdir(parents=True, exist_ok=True)

    Path(cfg.det.viz_output_dir, "correct").mkdir(parents=True, exist_ok=True)
    Path(cfg.det.viz_output_dir, "error").mkdir(parents=True, exist_ok=True)
    Path(cfg.det.eval_output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.seg.eval_output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.seg.viz_output_dir).mkdir(parents=True, exist_ok=True)


def _validate(cfg: PipelineConfig) -> None:
    images = Path(cfg.images_dir)
    gt = Path(cfg.gt_jsons_dir)
    if not images.exists() or not images.is_dir():
        raise AppError(f"输入图片目录无效: {images}")
    if not gt.exists() or not gt.is_dir():
        raise AppError(f"真值标注目录无效: {gt}")

    det = cfg.det
    if not (0.0 < det.iou_threshold <= 1.0):
        raise AppError(f"IoU 阈值无效: {det.iou_threshold}")

    # 互斥检查：检测 与 分割
    s = cfg.steps
    use_det = s.run_evaluation or s.run_visualization
    use_seg = s.run_seg_evaluation or s.run_seg_visualization
    if use_det and use_seg:
        raise AppError("目标检测与语义分割功能不能同时启用")

    # 推理类型配置
    if cfg.inference_type == "https":
        if not cfg.https:
            raise AppError("HTTPS 配置未提供")
        h = cfg.https
        if not all([h.img_stream_url, h.stream_name, h.access_key, h.secret_key]):
            raise AppError("HTTPS 接口配置不完整（url/stream/ak/sk）")
    elif cfg.inference_type == "grpc_standard":
        if not cfg.grpc:
            raise AppError("gRPC 标准 配置未提供")
        g = cfg.grpc
        if not all([g.grpc_server, g.task_id, g.stream_name]):
            raise AppError("gRPC 标准 配置不完整（server/task_id/stream_name）")
    else:
        raise AppError(f"不支持的推理类型: {cfg.inference_type}")


def _dynamic_report_paths(cfg: PipelineConfig) -> tuple[str, str, str, str]:
    """
    按推理类型拼“基底目录”，用于 GUI 里每次运行一个时间戳目录时，外部先把
    cfg.https/grpc 的 raw/pred 目录设置好即可。这里只负责把报告与可视化也对齐到同一根。
    """
    if cfg.inference_type == "https":
        base = Path(cfg.https.raw_responses_dir).parent.parent  # …/https_XXXX
    else:
        base = Path(cfg.grpc.raw_responses_dir).parent.parent  # …/grpc_standard_XXXX
    eval_csv = (base / "reports" / "evaluation_report.csv").as_posix()
    det_viz = (base / "reports" / "visualization_results").as_posix()
    seg_csv = (base / "reports" / "semseg_eval.csv").as_posix()
    seg_vis = (base / "reports" / "semseg_vis_masks").as_posix()
    return eval_csv, det_viz, seg_csv, seg_vis


# ============ 主流程 ============


def run(cfg: PipelineConfig) -> bool:
    """
    返回 True 表示全流程成功；False 表示中途失败（或被请求停止）。
    """
    global _stop_flag
    _stop_flag = False

    t0 = time.time()
    log.info("=" * 60)
    log.info("🚀 启动推理流水线（%s）", cfg.inference_type)
    log.info("📁 images=%s", cfg.images_dir)
    log.info("📁 gt_jsons=%s", cfg.gt_jsons_dir)

    try:
        _validate(cfg)

        # 用动态路径对齐输出（评估/可视化）到同一根目录（与GUI时间戳一致）
        det_eval_csv, det_viz_dir, seg_eval_csv, seg_viz_dir = _dynamic_report_paths(
            cfg
        )
        cfg.det.eval_output_file = det_eval_csv
        cfg.det.viz_output_dir = det_viz_dir
        cfg.seg.eval_output_file = seg_eval_csv
        cfg.seg.viz_output_dir = seg_viz_dir

        _ensure_dirs(cfg)

        # ========== Step 1: 推理 ==========
        if cfg.steps.run_inference:
            _progress("步骤1: 推理", 20)
            if cfg.inference_type == "https":
                h = cfg.https
                https_stream.infer_dir_to_jsons(
                    image_dir=cfg.images_dir,
                    out_dir=h.raw_responses_dir,
                    config={
                        "img_stream_url": h.img_stream_url,
                        "stream_name": h.stream_name,
                        "access_key": h.access_key,
                        "secret_key": h.secret_key,
                    },
                    max_workers=h.max_workers,
                )
            else:  # grpc_standard
                g = cfg.grpc
                grpc_stream.infer_dir_to_jsons(
                    image_dir=cfg.images_dir,
                    out_dir=g.raw_responses_dir,
                    grpc_server=g.grpc_server,
                    stream_name=g.stream_name,
                    task_id=g.task_id,
                    max_workers=g.max_workers,
                )
            log.info("✅ 推理完成")
            if _stopped():
                return False

        # ========== Step 2: 转 LabelMe（仅目标检测线） ==========
        if cfg.steps.run_conversion:
            _progress("步骤2: 格式转换", 40)
            # 选择原始响应目录 → 转成 labelme 预测目录
            if cfg.inference_type == "https":
                raw_dir = cfg.https.raw_responses_dir
                pred_dir = cfg.https.pred_jsons_dir
            else:
                raw_dir = cfg.grpc.raw_responses_dir
                pred_dir = cfg.grpc.pred_jsons_dir

            convert_preds_to_labelme(
                raw_dir=raw_dir,
                img_dir=cfg.images_dir,
                out_dir=pred_dir,
                max_workers=(
                    cfg.https.max_workers
                    if cfg.inference_type == "https"
                    else cfg.grpc.max_workers
                ),
            )
            log.info("✅ 格式转换完成")
            if _stopped():
                return False

        # ========== 目标检测评估 ==========
        if cfg.steps.run_evaluation:
            _progress("步骤3: 目标检测评估", 60)
            # 预测目录选用 labelme 转换后的目录
            pred_dir = (
                cfg.https.pred_jsons_dir
                if cfg.inference_type == "https"
                else cfg.grpc.pred_jsons_dir
            )
            evaluate_dir_to_csv(
                DetEvalConfig(
                    gt_dir=cfg.gt_jsons_dir,
                    pred_dir=pred_dir,
                    iou_thr=cfg.det.iou_threshold,
                    out_csv=cfg.det.eval_output_file,
                )
            )
            log.info("✅ 目标检测评估完成")
            if _stopped():
                return False

        # ========== 目标检测可视化 ==========
        if cfg.steps.run_visualization:
            _progress("步骤4: 目标检测可视化", 75)
            pred_dir = (
                cfg.https.pred_jsons_dir
                if cfg.inference_type == "https"
                else cfg.grpc.pred_jsons_dir
            )
            det_visualize(
                DetVizConfig(
                    gt_dir=cfg.gt_jsons_dir,
                    pred_dir=pred_dir,
                    image_dir=cfg.images_dir,
                    iou_thr=cfg.det.iou_threshold,
                    out_dir=cfg.det.viz_output_dir,
                    stats_mode=cfg.det.viz_mode_stats,
                    max_workers=cfg.det.max_workers,
                )
            )
            log.info("✅ 目标检测可视化完成")
            if _stopped():
                return False

        # ========== 语义分割评估（直接用原始响应目录） ==========
        if cfg.steps.run_seg_evaluation and cfg.seg.enabled:
            _progress("步骤5: 语义分割评估", 90)
            raw_dir = (
                cfg.https.raw_responses_dir
                if cfg.inference_type == "https"
                else cfg.grpc.raw_responses_dir
            )
            seg_evaluate_dir_to_csv(
                SegEvalConfig(
                    pred_dir=raw_dir,
                    gt_dir=cfg.gt_jsons_dir,
                    out_csv=cfg.seg.eval_output_file,
                    iou_threshold=cfg.seg.iou_threshold,
                )
            )
            log.info("✅ 语义分割评估完成")
            if _stopped():
                return False

        # ========== 语义分割可视化（直接用原始响应目录） ==========
        if cfg.steps.run_seg_visualization and cfg.seg.enabled:
            _progress("步骤6: 语义分割可视化", 95)
            raw_dir = (
                cfg.https.raw_responses_dir
                if cfg.inference_type == "https"
                else cfg.grpc.raw_responses_dir
            )
            seg_visualize_dir(
                SegVizConfig(
                    pred_dir=raw_dir,
                    image_dir=cfg.images_dir,
                    out_dir=cfg.seg.viz_output_dir,
                    alpha=0.45,
                )
            )
            log.info("✅ 语义分割可视化完成")
            if _stopped():
                return False

        _progress("执行完成", 100)
        log.info("=" * 60)
        log.info("🎉 流水线执行成功，耗时 %.2fs", time.time() - t0)
        log.info("📄 评估报告: %s", cfg.det.eval_output_file)
        log.info("🖼 可视化目录: %s", cfg.det.viz_output_dir)
        if cfg.seg.enabled:
            log.info("📄 SemSeg评估: %s", cfg.seg.eval_output_file)
            log.info("🖼 SemSeg可视化: %s", cfg.seg.viz_output_dir)
        log.info("=" * 60)
        return True

    except AppError as e:
        _progress("执行失败", 100)
        log.error("❌ 配置/流程错误: %s", e)
        return False
    except Exception as e:
        _progress("执行失败", 100)
        log.exception("❌ 未预期异常: %s", e)
        return False
