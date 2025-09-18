"""目标检测评估模块。"""

from .matcher import BoundingBox, MatchResult, assign_boxes, get_iou
from .metrics import DetectionEvalConfig, DetectionEvaluator, parse_labelme_shapes
from .runner import DetectionPipeline, DetectionPipelineConfig
from .visualizer import DetectionVisualizer, DetectionVizConfig

__all__ = [
    "BoundingBox",
    "MatchResult",
    "assign_boxes",
    "get_iou",
    "DetectionEvalConfig",
    "DetectionEvaluator",
    "DetectionVizConfig",
    "DetectionVisualizer",
    "DetectionPipelineConfig",
    "DetectionPipeline",
    "parse_labelme_shapes",
]
