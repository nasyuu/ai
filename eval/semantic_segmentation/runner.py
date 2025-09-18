"""语义分割评估与可视化流水线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from utils.exceptions import ValidationError
from utils.logger import get_logger

from .metrics import SemSegEvalConfig, SemSegEvaluator
from .visualizer import SemSegVisualizer, SemSegVizConfig

__all__ = [
    "SemSegPipelineConfig",
    "SemSegPipeline",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class SemSegPipelineConfig:
    """组合语义分割评估与可视化的配置容器。"""

    evaluate: bool = True
    visualize: bool = False
    eval_config: Optional[SemSegEvalConfig] = None
    viz_config: Optional[SemSegVizConfig] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "SemSegPipelineConfig":
        evaluate = bool(payload.get("evaluate", True))
        visualize = bool(payload.get("visualize", False))

        eval_cfg = None
        if evaluate and "eval_config" in payload:
            eval_cfg = SemSegEvalConfig.from_mapping(
                payload["eval_config"]  # type: ignore[arg-type]
            )

        viz_cfg = None
        if visualize and "viz_config" in payload:
            viz_cfg = SemSegVizConfig.from_mapping(
                payload["viz_config"]  # type: ignore[arg-type]
            )

        return cls(
            evaluate=evaluate,
            visualize=visualize,
            eval_config=eval_cfg,
            viz_config=viz_cfg,
        )


class SemSegPipeline:
    """串联语义分割评估与可视化。"""

    def __init__(self, config: SemSegPipelineConfig) -> None:
        self.config = config
        self.logger = logger
        if config.evaluate and not config.eval_config:
            raise ValidationError("缺少语义分割评估配置")
        if config.visualize and not config.viz_config:
            raise ValidationError("缺少语义分割可视化配置")

        self.evaluator = (
            SemSegEvaluator(config.eval_config)
            if config.evaluate and config.eval_config
            else None
        )
        self.visualizer = (
            SemSegVisualizer(config.viz_config)
            if config.visualize and config.viz_config
            else None
        )

    def run(self) -> None:
        if self.evaluator:
            self.logger.info("开始执行语义分割评估")
            self.evaluator.evaluate()
        else:
            self.logger.debug("语义分割评估已禁用")

        if self.visualizer:
            self.logger.info("开始执行语义分割可视化")
            self.visualizer.visualize_directory()
        else:
            self.logger.debug("语义分割可视化已禁用")
