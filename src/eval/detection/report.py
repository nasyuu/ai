from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils.logger import get_logger
from .matching import assign_boxes

log = get_logger("eval.detection.report")


def _parse_labelme(json_path: Path) -> List[Dict]:
    """读取LabelMe，返回 [{'label': str, 'bbox': list, 'score': Optional[float]}]"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    objs: List[Dict] = []
    for shp in data.get("shapes", []):
        label, pts = shp["label"], shp["points"]
        stype = shp.get("shape_type", "polygon")
        if stype == "rectangle" and len(pts) == 2:      # 两点矩形 -> 四元组
            (x1, y1), (x2, y2) = pts
            bbox = [x1, y1, x2, y2]
        else:
            bbox = pts
        obj = {"label": label, "bbox": bbox}
        # 兼容属性里可能携带score
        score = (
            shp.get("attributes", {}).get("score")
            if isinstance(shp.get("attributes"), dict)
            else None
        )
        if score is not None:
            obj["score"] = float(score)
        objs.append(obj)
    return objs


def _group_by_label(objs: List[Dict]) -> Dict[str, List]:
    by: Dict[str, List] = {}
    for o in objs:
        by.setdefault(o["label"], []).append(o["bbox"])
    return by


@dataclass
class DetectionEvalConfig:
    gt_dir: str
    pred_dir: str
    iou_thr: float = 0.5
    out_csv: str = "reports/evaluation_report.csv"
    # 未来可扩展：method='hungarian' / 'greedy'，是否使用label 等


def evaluate_dir_to_csv(cfg: DetectionEvalConfig) -> pd.DataFrame:
    """
    逐文件对齐 GT/PRED（同名json），输出每类 TP/FN/FP 与 per-image 完全正确率。
    """
    gt_dir = Path(cfg.gt_dir)
    pred_dir = Path(cfg.pred_dir)

    tp: Dict[str, int] = {}
    fn: Dict[str, int]  = {}
    fp: Dict[str, int] = {}
    labels: set[str] = set()
    total_imgs, fully_correct = 0, 0

    for gt_json in sorted(gt_dir.glob("*.json")):
        fname = gt_json.name
        pred_json = pred_dir / fname
        total_imgs += 1

        gt_objs = _parse_labelme(gt_json)
        pred_objs = _parse_labelme(pred_json) if pred_json.exists() else []

        gt_map, pred_map = _group_by_label(gt_objs), _group_by_label(pred_objs)
        img_labels = set(gt_map) | set(pred_map)
        labels.update(img_labels)

        for lb in img_labels:
            res = assign_boxes(
                [{"label": lb, "bbox": b} for b in gt_map.get(lb, [])],
                [{"label": lb, "bbox": b} for b in pred_map.get(lb, [])],
                iou_thr=cfg.iou_thr,
                method="hungarian",
                use_label=False,
            )
            tp[lb] = tp.get(lb, 0) + len(res["matches"])
            fn[lb] = fn.get(lb, 0) + len(res["unmatched_gt"])
            fp[lb] = fp.get(lb, 0) + len(res["unmatched_pred"])

        # 完全正确（图片级）
        img_res = assign_boxes(gt_objs, pred_objs, iou_thr=cfg.iou_thr, method="hungarian")
        if not img_res["unmatched_gt"] and not img_res["unmatched_pred"]:
            fully_correct += 1

    # 汇总
    rows: List[Dict] = []
    for lb in sorted(labels):
        TP, FN, FP = tp.get(lb, 0), fn.get(lb, 0), fp.get(lb, 0)
        prec = "N/A (无预测)" if TP + FP == 0 else round(TP / (TP + FP), 4)
        rec = "N/A (无GT)"   if TP + FN == 0 else round(TP / (TP + FN), 4)
        rows.append(dict(类别=lb, TP=TP, FN=FN, FP=FP, 精确率_Precision=prec, 召回率_Recall=rec))

    rows.append(
        dict(
            类别="图片级完全正确率",
            TP="", FN="", FP="",
            精确率_Precision=(f"{fully_correct}/{total_imgs} = {fully_correct / total_imgs:.2%}" if total_imgs else "0"),
            召回率_Recall="",
        )
    )

    df = pd.DataFrame(rows)
    out_path = Path(cfg.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info("√ 评估完成 → %s", out_path.as_posix())
    return df
