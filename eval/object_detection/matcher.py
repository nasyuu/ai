"""目标检测框匹配与 IoU 计算工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

__all__ = [
    "BoundingBox",
    "MatchResult",
    "assign_boxes",
    "get_iou",
]

_EPS = 1e-6


@dataclass(slots=True)
class BoundingBox:
    """描述一个检测框的数据结构。"""

    label: str
    bbox: Sequence[float | Sequence[float]]
    score: float | None = None


@dataclass(slots=True)
class MatchResult:
    """匹配结果容器。"""

    matches: list[tuple[int, int]]
    unmatched_gt: list[int]
    unmatched_pred: list[int]

    def true_positives(self) -> int:
        return len(self.matches)

    def false_negatives(self) -> int:
        return len(self.unmatched_gt)

    def false_positives(self) -> int:
        return len(self.unmatched_pred)


def _list_to_poly(points: Sequence[float | Sequence[float]]) -> Polygon:
    if len(points) == 4 and isinstance(points[0], (int, float)):
        x1, y1, x2, y2 = points  # type: ignore[assignment]
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        coords = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    elif len(points) == 2 and isinstance(points[0], (list, tuple)):
        (x1, y1), (x2, y2) = points  # type: ignore[misc]
        coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    elif isinstance(points[0], (list, tuple)):
        coords = [(float(x), float(y)) for x, y in points]  # type: ignore[assignment]
    else:
        flat = list(map(float, points))
        coords = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    return Polygon(coords)


def get_iou(
    box_a: Sequence[float | Sequence[float]], box_b: Sequence[float | Sequence[float]]
) -> float:
    poly_a, poly_b = _list_to_poly(box_a), _list_to_poly(box_b)

    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0

    intersection = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return float(intersection / (union + _EPS))


def _sort_preds(pred_boxes: Sequence[BoundingBox]) -> list[int]:
    if pred_boxes and pred_boxes[0].score is not None:
        return sorted(
            range(len(pred_boxes)),
            key=lambda i: pred_boxes[i].score or 0.0,
            reverse=True,
        )
    return list(range(len(pred_boxes)))


def _match_greedy(
    gt_boxes: Sequence[BoundingBox],
    pred_boxes: Sequence[BoundingBox],
    *,
    iou_threshold: float,
    use_label: bool,
) -> MatchResult:
    matches: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    for pred_index in _sort_preds(pred_boxes):
        pred = pred_boxes[pred_index]
        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in used_gt:
                continue
            if use_label and pred.label != gt.label:
                continue
            if get_iou(pred.bbox, gt.bbox) >= iou_threshold:
                matches.append((gt_index, pred_index))
                used_gt.add(gt_index)
                used_pred.add(pred_index)
                break

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in used_pred]
    return MatchResult(matches, unmatched_gt, unmatched_pred)


def _match_hungarian(
    gt_boxes: Sequence[BoundingBox],
    pred_boxes: Sequence[BoundingBox],
    *,
    iou_threshold: float,
    use_label: bool,
) -> MatchResult:
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    if n_gt == 0 or n_pred == 0:
        return MatchResult([], list(range(n_gt)), list(range(n_pred)))

    cost = np.ones((n_gt, n_pred), dtype=np.float32)
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            if use_label and gt.label != pred.label:
                continue
            iou = get_iou(gt.bbox, pred.bbox)
            if iou >= iou_threshold:
                cost[gi, pi] = 1 - iou

    gt_idx, pred_idx = linear_sum_assignment(cost)
    matches = [(gi, pi) for gi, pi in zip(gt_idx, pred_idx) if cost[gi, pi] < 1]
    matched_gt = {gi for gi, _ in matches}
    matched_pred = {pi for _, pi in matches}
    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    return MatchResult(matches, unmatched_gt, unmatched_pred)


def assign_boxes(
    gt_boxes: Sequence[Mapping[str, object] | BoundingBox],
    pred_boxes: Sequence[Mapping[str, object] | BoundingBox],
    *,
    iou_threshold: float = 0.5,
    use_label: bool = True,
    method: str = "hungarian",
) -> MatchResult:
    def to_box(item: Mapping[str, object] | BoundingBox) -> BoundingBox:
        if isinstance(item, BoundingBox):
            return item
        return BoundingBox(
            label=str(item.get("label", "")),
            bbox=item.get("bbox", []),
            score=float(item["score"])
            if "score" in item and item["score"] is not None
            else None,
        )

    gt_processed = [to_box(b) for b in gt_boxes]
    pred_processed = [to_box(b) for b in pred_boxes]

    if method == "greedy":
        return _match_greedy(
            gt_processed,
            pred_processed,
            iou_threshold=iou_threshold,
            use_label=use_label,
        )
    if method == "hungarian":
        return _match_hungarian(
            gt_processed,
            pred_processed,
            iou_threshold=iou_threshold,
            use_label=use_label,
        )
    raise ValueError("method must be 'greedy' or 'hungarian'")
