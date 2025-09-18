"""目标检测评估报告生成。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from utils.exceptions import ValidationError
from utils.logger import get_logger

from .matcher import BoundingBox, assign_boxes

__all__ = [
    "DetectionEvalConfig",
    "DetectionEvaluator",
    "parse_labelme_shapes",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class DetectionEvalConfig:
    """评估流程所需的输入输出配置。"""

    gt_dir: Path
    pred_dir: Path
    output_csv: Path
    iou_threshold: float = 0.5
    match_method: str = "hungarian"
    use_label_for_metrics: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "DetectionEvalConfig":
        try:
            gt_dir = Path(str(payload["gt_dir"]))
            pred_dir = Path(str(payload["pred_dir"]))
            output_csv = Path(str(payload["output_csv"]))
        except KeyError as exc:  # noqa: PERF203
            raise ValidationError(f"配置项 {exc.args[0]} 缺失") from exc

        return cls(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_csv=output_csv,
            iou_threshold=float(payload.get("iou_threshold", 0.5)),
            match_method=str(payload.get("match_method", "hungarian")),
            use_label_for_metrics=bool(payload.get("use_label_for_metrics", False)),
        )


def parse_labelme_shapes(json_path: Path) -> list[BoundingBox]:
    try:
        with json_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "无法解析 LabelMe JSON", details={"path": str(json_path)}
        ) from exc

    boxes: list[BoundingBox] = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", ""))
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")
        if shape_type == "rectangle" and len(points) == 2:
            (x1, y1), (x2, y2) = points
            bbox = [x1, y1, x2, y2]
        else:
            bbox = points
        score = shape.get("attributes", {}).get("score")
        boxes.append(
            BoundingBox(
                label=label,
                bbox=bbox,
                score=float(score) if score is not None else None,
            )
        )
    return boxes


def _group_by_label(boxes: Iterable[BoundingBox]) -> dict[str, list[BoundingBox]]:
    grouped: dict[str, list[BoundingBox]] = {}
    for box in boxes:
        grouped.setdefault(box.label, []).append(box)
    return grouped


class DetectionEvaluator:
    """封装目标检测评估逻辑。"""

    def __init__(self, config: DetectionEvalConfig) -> None:
        self.config = config
        self.logger = logger

    def evaluate(self) -> pd.DataFrame:
        gt_dir = self.config.gt_dir
        pred_dir = self.config.pred_dir

        if not gt_dir.exists():
            raise ValidationError("真实标注目录不存在", details={"path": str(gt_dir)})
        if not pred_dir.exists():
            raise ValidationError("预测标注目录不存在", details={"path": str(pred_dir)})

        labels: set[str] = set()
        tp: dict[str, int] = {}
        fn: dict[str, int] = {}
        fp: dict[str, int] = {}
        total_images = 0
        fully_correct = 0

        for gt_json in sorted(gt_dir.glob("*.json")):
            total_images += 1
            pred_json = pred_dir / gt_json.name

            gt_boxes = parse_labelme_shapes(gt_json)
            pred_boxes = parse_labelme_shapes(pred_json) if pred_json.exists() else []

            grouped_gt = _group_by_label(gt_boxes)
            grouped_pred = _group_by_label(pred_boxes)
            image_labels = set(grouped_gt) | set(grouped_pred)
            labels.update(image_labels)

            for label in image_labels:
                matches = assign_boxes(
                    grouped_gt.get(label, []),
                    grouped_pred.get(label, []),
                    iou_threshold=self.config.iou_threshold,
                    use_label=False,
                    method=self.config.match_method,
                )
                tp[label] = tp.get(label, 0) + matches.true_positives()
                fn[label] = fn.get(label, 0) + matches.false_negatives()
                fp[label] = fp.get(label, 0) + matches.false_positives()

            overall = assign_boxes(
                gt_boxes,
                pred_boxes,
                iou_threshold=self.config.iou_threshold,
                use_label=self.config.use_label_for_metrics,
                method=self.config.match_method,
            )
            if not overall.unmatched_gt and not overall.unmatched_pred:
                fully_correct += 1

        rows: list[dict[str, object]] = []
        for label in sorted(labels):
            tp_count = tp.get(label, 0)
            fn_count = fn.get(label, 0)
            fp_count = fp.get(label, 0)

            precision: object
            recall: object
            if tp_count + fp_count == 0:
                precision = "N/A"
            else:
                precision = round(tp_count / (tp_count + fp_count), 4)

            if tp_count + fn_count == 0:
                recall = "N/A"
            else:
                recall = round(tp_count / (tp_count + fn_count), 4)

            rows.append(
                {
                    "label": label,
                    "true_positive": tp_count,
                    "false_negative": fn_count,
                    "false_positive": fp_count,
                    "precision": precision,
                    "recall": recall,
                }
            )

        rows.append(
            {
                "label": "image_level_accuracy",
                "true_positive": "",
                "false_negative": "",
                "false_positive": "",
                "precision": (
                    f"{fully_correct}/{total_images} = {fully_correct / total_images:.2%}"
                    if total_images
                    else "0"
                ),
                "recall": "",
            }
        )

        df = pd.DataFrame(rows)
        self._write_output(df)
        return df

    def _write_output(self, dataframe: pd.DataFrame) -> None:
        output_path = self.config.output_csv
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False, encoding="utf-8")
        self.logger.info("评估报告已生成: %s", output_path)
