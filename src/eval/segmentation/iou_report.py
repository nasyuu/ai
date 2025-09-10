from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from utils.logger import get_logger

from .utils import (
    LABELS_ALIAS_LC,
    LABELS_CONFIG,
    load_gt_mask,
    load_pred_mask,
)

log = get_logger("eval.segmentation.iou_report")


@dataclass
class SegEvalConfig:
    pred_dir: str  # 推理JSON目录（直接用原始响应）
    gt_dir: str  # LabelMe目录
    out_csv: str = "reports/semseg_eval.csv"
    iou_threshold: float = 0.8  # 统计阈值
    num_decimals: int = 5


def _compute_iou(pred_bin, gt_bin) -> float:
    inter = np.logical_and(pred_bin, gt_bin).sum()
    uni = np.logical_or(pred_bin, gt_bin).sum()
    return float(inter / uni) if uni else 0.0


def _process_one(name: str, pred_dir: Path, gt_dir: Path):
    base = Path(name).stem
    pred_path = pred_dir / name
    gt_path = gt_dir / name
    if not gt_path.exists():
        return base, {}, "未找到 GT"

    try:
        pred_mask, idx_dict = load_pred_mask(pred_path)
        H, W = pred_mask.shape
        ious = {}
        for key in LABELS_CONFIG:
            gt_mask = load_gt_mask(gt_path, key, (H, W))
            alias_set = LABELS_ALIAS_LC[key]
            idxs = [idx for label, idx in idx_dict.items() if label in alias_set]
            pred_bin = (
                np.isin(pred_mask, idxs).astype(np.uint8)
                if idxs
                else np.zeros_like(gt_mask)
            )
            ious[key] = _compute_iou(pred_bin, gt_mask)
        return base, ious, ""
    except ValueError as e:
        remark = "空result" if "result 为空" in str(e) else f"异常: {e}"
        log.warning("[%s] %s", base, remark)
        return base, {}, remark
    except Exception as e:
        log.exception("[%s] 处理异常：%s", base, e)
        return base, {}, f"异常: {e}"


def evaluate_dir_to_csv(cfg: SegEvalConfig):
    pred_dir = Path(cfg.pred_dir)
    gt_dir = Path(cfg.gt_dir)
    out_csv = Path(cfg.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    names = sorted([f.name for f in pred_dir.iterdir() if f.suffix.lower() == ".json"])
    results = []
    for n in names:
        results.append(_process_one(n, pred_dir, gt_dir))

    # 组装CSV
    header = ["图片名"] + [f"{k}_IoU" for k in LABELS_CONFIG] + ["备注"]
    rows: List[List] = [header]

    stats = {k: {"total": 0, "above": 0, "zero": 0, "sum": 0.0} for k in LABELS_CONFIG}

    for base, ious, remark in sorted(results, key=lambda x: x[0]):
        row = [base]
        for k in LABELS_CONFIG:
            v = float(ious.get(k, 0.0))
            row.append(round(v, cfg.num_decimals))
            stats[k]["total"] += 1
            stats[k]["sum"] += v
            if v >= cfg.iou_threshold:
                stats[k]["above"] += 1
            if v == 0.0:
                stats[k]["zero"] += 1
        row.append(remark)
        rows.append(row)

    # 统计
    rows.append([])
    rows.append([f"IoU统计 (阈值: {cfg.iou_threshold})"])
    for k in LABELS_CONFIG:
        total = max(1, stats[k]["total"])
        above = stats[k]["above"]
        zero = stats[k]["zero"]
        mean_iou = stats[k]["sum"] / total
        rows.append(
            [
                f"{k}_IoU ≥ {cfg.iou_threshold} 占比:",
                f"{above}/{total} ({above / total:.2%})",
            ]
        )
        rows.append([f"{k}_IoU 平均值 (mIoU):", round(mean_iou, cfg.num_decimals)])
        if k == "Coal":
            rows.append([f"{k}_IoU = 0 数量:", f"{zero}"])

    rows.append([])
    rows.append(["Overall 宏平均 mIoU（Mean over classes）:"])
    overall = np.mean(
        [stats[k]["sum"] / max(1, stats[k]["total"]) for k in LABELS_CONFIG]
    )
    rows.append(["mIoU", round(float(overall), cfg.num_decimals)])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    log.info("√ 语义分割评估完成 → %s", out_csv.as_posix())
