"""目标检测评估与可视化流水线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from utils.exceptions import ValidationError
from utils.logger import get_logger

from .metrics import DetectionEvalConfig, DetectionEvaluator
from .visualizer import DetectionVisualizer, DetectionVizConfig

__all__ = [
    "DetectionPipelineConfig",
    "DetectionPipeline",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class DetectionPipelineConfig:
    """组合评估与可视化的配置容器。"""

    evaluate: bool = True
    visualize: bool = False
    eval_config: Optional[DetectionEvalConfig] = None
    viz_config: Optional[DetectionVizConfig] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "DetectionPipelineConfig":
        evaluate = bool(payload.get("evaluate", True))
        visualize = bool(payload.get("visualize", False))

        eval_cfg = None
        if evaluate and "eval_config" in payload:
            eval_cfg = DetectionEvalConfig.from_mapping(
                payload["eval_config"]  # type: ignore[arg-type]
            )

        viz_cfg = None
        if visualize and "viz_config" in payload:
            viz_cfg = DetectionVizConfig.from_mapping(
                payload["viz_config"]  # type: ignore[arg-type]
            )

        return cls(
            evaluate=evaluate,
            visualize=visualize,
            eval_config=eval_cfg,
            viz_config=viz_cfg,
        )


class DetectionPipeline:
    """封装目标检测评估与可视化的顺序调用。"""

    def __init__(self, config: DetectionPipelineConfig) -> None:
        self.config = config
        self.logger = logger
        if config.evaluate and not config.eval_config:
            raise ValidationError("缺少评估配置")
        if config.visualize and not config.viz_config:
            raise ValidationError("缺少可视化配置")

        self.evaluator = (
            DetectionEvaluator(config.eval_config)
            if config.evaluate and config.eval_config
            else None
        )
        self.visualizer = (
            DetectionVisualizer(config.viz_config)
            if config.visualize and config.viz_config
            else None
        )

    def run(self) -> None:
        if self.evaluator:
            self.logger.info("开始执行目标检测评估")
            self.evaluator.evaluate()
        else:
            self.logger.debug("评估步骤已关闭")

        if self.visualizer:
            self.logger.info("开始执行目标检测可视化")
            self.visualizer.visualize_directory()
        else:
            self.logger.debug("可视化步骤已关闭")
