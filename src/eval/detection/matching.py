from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

EPS = 1e-6


def _list_to_poly(pts: Sequence) -> Polygon:
    """
    将多种bbox表达统一为Polygon：
    - [x1,y1,x2,y2]
    - [[x1,y1],[x2,y2]]
    - list of (x,y)
    - 扁平数组 [x1,y1,x2,y2,...]
    """
    if len(pts) == 4 and isinstance(pts[0], (int, float)):
        x1, y1, x2, y2 = pts
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    elif len(pts) == 2 and isinstance(pts[0], (list, tuple)):
        (x1, y1), (x2, y2) = pts
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    elif isinstance(pts[0], (list, tuple)):
        pts = [(float(x), float(y)) for x, y in pts]
    else:
        pts = [(float(pts[i]), float(pts[i + 1])) for i in range(0, len(pts), 2)]
    return Polygon(pts)


def get_iou(boxA, boxB) -> float:
    polyA, polyB = _list_to_poly(boxA), _list_to_poly(boxB)
    if not polyA.is_valid:
        polyA = polyA.buffer(0)
    if not polyB.is_valid:
        polyB = polyB.buffer(0)
    if not polyA.is_valid or not polyB.is_valid:
        return 0.0
    inter = polyA.intersection(polyB).area
    union = polyA.union(polyB).area
    return float(inter / (union + EPS))


def _sort_preds(pred_boxes: List[Dict]) -> List[int]:
    if pred_boxes and "score" in pred_boxes[0]:
        return sorted(
            range(len(pred_boxes)), key=lambda i: pred_boxes[i]["score"], reverse=True
        )
    return list(range(len(pred_boxes)))


def match_greedy(
    gt_boxes: List[Dict],
    pred_boxes: List[Dict],
    iou_threshold: float = 0.5,
    use_label: bool = True,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    matched, used_gt, used_pred = [], set(), set()
    for pi in _sort_preds(pred_boxes):
        p = pred_boxes[pi]
        for gi, g in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            if (not use_label or p["label"] == g["label"]) and get_iou(
                p["bbox"], g["bbox"]
            ) >= iou_threshold:
                matched.append((gi, pi))
                used_gt.add(gi)
                used_pred.add(pi)
                break
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in used_pred]
    return matched, unmatched_gt, unmatched_pred


def match_hungarian(
    gt_boxes: List[Dict],
    pred_boxes: List[Dict],
    iou_threshold: float = 0.5,
    use_label: bool = True,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    if n_gt == 0 or n_pred == 0:
        return [], list(range(n_gt)), list(range(n_pred))

    cost = np.ones((n_gt, n_pred), dtype=np.float32)
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            if use_label and g["label"] != p["label"]:
                continue
            iou = get_iou(g["bbox"], p["bbox"])
            if iou >= iou_threshold:
                cost[gi, pi] = 1 - iou

    gt_idx, pred_idx = linear_sum_assignment(cost)
    matched = [(gi, pi) for gi, pi in zip(gt_idx, pred_idx) if cost[gi, pi] < 1]
    matched_gt = {gi for gi, _ in matched}
    matched_pred = {pi for _, pi in matched}
    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    return matched, unmatched_gt, unmatched_pred


def assign_boxes(
    gt_boxes: List[Dict],
    pred_boxes: List[Dict],
    iou_thr: float = 0.5,
    use_label: bool = True,
    method: str = "hungarian",
) -> Dict[str, List]:
    if method == "greedy":
        m, ug, up = match_greedy(gt_boxes, pred_boxes, iou_thr, use_label)
    elif method == "hungarian":
        m, ug, up = match_hungarian(gt_boxes, pred_boxes, iou_thr, use_label)
    else:
        raise ValueError("method must be 'greedy' or 'hungarian'")
    return {"matches": m, "unmatched_gt": ug, "unmatched_pred": up}
