from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

EPS = 1e-6


def _list_to_poly(pts):
    """
    将输入的坐标列表转换为 Shapely Polygon 对象。
    :param pts: 输入坐标列表
    :return: Shapely Polygon 对象
    """
    # 四元组 - 统一按 [x1, y1, x2, y2] 格式处理
    if len(pts) == 4 and isinstance(pts[0], (int, float)):
        x1, y1, x2, y2 = pts
        # 确保坐标顺序正确：左上到右下
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    # 两点矩形
    elif len(pts) == 2 and isinstance(pts[0], (list, tuple)):
        (x1, y1), (x2, y2) = pts
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    # 多边形 (list of list/tuple)
    elif isinstance(pts[0], (list, tuple)):
        pts = [(float(x), float(y)) for x, y in pts]

    # 多边形 (扁平数组)
    else:
        pts = [(float(pts[i]), float(pts[i + 1])) for i in range(0, len(pts), 2)]

    return Polygon(pts)


def get_iou(boxA, boxB) -> float:
    """
    计算两个边界框的交并比 (IoU)。
    :param boxA: 第一个边界框
    :param boxB: 第二个边界框
    :return: 交并比 (IoU) 值
    """
    polyA, polyB = _list_to_poly(boxA), _list_to_poly(boxB)

    if not polyA.is_valid:
        polyA = polyA.buffer(0)
    if not polyB.is_valid:
        polyB = polyB.buffer(0)
    if not polyA.is_valid or not polyB.is_valid:
        return 0.0

    inter = polyA.intersection(polyB).area
    union = polyA.union(polyB).area
    return inter / (union + EPS)


def _sort_preds(pred_boxes: List[Dict]) -> List[int]:
    """
    根据预测框的置信度分数对预测框进行排序。
    :param pred_boxes: 预测框列表，每个框是一个字典，包含 "bbox" 和 "score" 键
    :return: 按照置信度分数降序排列的索引列表
    """
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
    """
    贪婪匹配算法：为每个预测框找到第一个满足 IoU 阈值的真实框。
    :param gt_boxes: 真实框列表，每个框是一个字典，包含 "bbox" 和 "label" 键
    :param pred_boxes: 预测框列表，每个框是一个字典，包含 "bbox" 和 "label" 键
    :param iou_threshold: IoU 阈值，默认 0.5
    :param use_label: 是否考虑标签匹配，默认 True
    :return: 匹配结果 (匹配对列表、未匹配的真实框索引、未匹配的预测框索引)
    """
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
    """
    匈牙利算法匹配：使用线性分配算法为真实框和预测框找到最佳匹配。
    :param gt_boxes: 真实框列表，每个框是一个字典，包含 "bbox" 和 "label" 键
    :param pred_boxes: 预测框列表，每个框是一个字典，包含 "bbox" 和 "label" 键
    :param iou_threshold: IoU 阈值，默认 0.5
    :param use_label: 是否考虑标签匹配，默认 True
    :return: 匹配结果 (匹配对列表、未匹配的真实框索引、未匹配的预测框索引)
    """
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
    """
    根据指定的匹配方法（贪婪或匈牙利算法）为真实框和预测框分配匹配。
    :param gt_boxes: 真实框列表，每个框是一个字典，包含 "bbox" 和 "label" 键
    :param pred_boxes: 预测框列表，每个框是一个字典，包含 "bbox" 和 "label" 键
    :param iou_thr: IoU 阈值，默认 0.5
    :param use_label: 是否考虑标签匹配，默认 True
    :param method: 匹配方法，"greedy" 或 "hungarian"，默认 "hungarian"
    :return: 匹配结果字典，包含 "matches"、"unmatched_gt" 和 "unmatched_pred"
    """
    if method == "greedy":
        m, ug, up = match_greedy(gt_boxes, pred_boxes, iou_thr, use_label)
    elif method == "hungarian":
        m, ug, up = match_hungarian(gt_boxes, pred_boxes, iou_thr, use_label)
    else:
        raise ValueError("method must be 'greedy' or 'hungarian'")
    return {"matches": m, "unmatched_gt": ug, "unmatched_pred": up}
