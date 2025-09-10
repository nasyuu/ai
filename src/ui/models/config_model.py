# src/ui/models/config_model.py
from __future__ import annotations

from dataclasses import dataclass

from pipeline.main import (
    DetEvalRuntime,
    GRPCStandardConfig,
    HTTPSConfig,
    PipelineConfig,
    SegEvalRuntime,
    StepSwitch,
)


@dataclass
class GUIState:
    # 基本
    inference_type: str = "https"
    images_dir: str = ""
    gt_jsons_dir: str = ""
    # 线程并发
    global_workers: int = 1
    # https
    https_url: str = ""
    https_stream: str = ""
    https_ak: str = ""
    https_sk: str = ""
    https_raw_dir: str = "https/responses"
    https_pred_dir: str = "https/pred_jsons"
    # grpc standard
    grpc_server: str = ""
    grpc_task_id: str = ""
    grpc_stream_name: str = ""
    grpc_raw_dir: str = "grpc_standard/responses"
    grpc_pred_dir: str = "grpc_standard/pred_jsons"
    # 检测评估/可视化
    iou_thr: float = 0.5
    viz_mode_stats: bool = True
    # 语义分割
    seg_enabled: bool = False
    seg_iou_thr: float = 0.8
    seg_viz: bool = False
    seg_eval: bool = False
    # 步骤
    run_infer: bool = True
    run_conv: bool = True
    run_eval: bool = True
    run_viz: bool = True

    def to_pipeline_config(self) -> PipelineConfig:
        https = None
        grpc = None
        if self.inference_type == "https":
            https = HTTPSConfig(
                img_stream_url=self.https_url,
                stream_name=self.https_stream,
                access_key=self.https_ak,
                secret_key=self.https_sk,
                raw_responses_dir=self.https_raw_dir,
                pred_jsons_dir=self.https_pred_dir,
                max_workers=self.global_workers,
            )
        else:
            grpc = GRPCStandardConfig(
                grpc_server=self.grpc_server,
                task_id=self.grpc_task_id,
                stream_name=self.grpc_stream_name,
                raw_responses_dir=self.grpc_raw_dir,
                pred_jsons_dir=self.grpc_pred_dir,
                max_workers=self.global_workers,
            )

        steps = StepSwitch(
            run_inference=self.run_infer,
            run_conversion=self.run_conv,
            run_evaluation=self.run_eval,
            run_visualization=self.run_viz,
            run_seg_evaluation=self.seg_eval,
            run_seg_visualization=self.seg_viz,
        )

        return PipelineConfig(
            inference_type=self.inference_type,
            images_dir=self.images_dir,
            gt_jsons_dir=self.gt_jsons_dir,
            https=https,
            grpc=grpc,
            det=DetEvalRuntime(
                iou_threshold=self.iou_thr,
                eval_output_file="reports/evaluation_report.csv",
                viz_output_dir="reports/visualization_results",
                viz_mode_stats=self.viz_mode_stats,
                max_workers=self.global_workers,
            ),
            seg=SegEvalRuntime(
                enabled=self.seg_enabled,
                iou_threshold=self.seg_iou_thr,
                eval_output_file="reports/semseg_eval.csv",
                viz_output_dir="reports/semseg_vis_masks",
                max_workers=self.global_workers,
            ),
            steps=steps,
        )
