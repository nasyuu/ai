import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ss_utils import LABELS_ALIAS_LC, LABELS_CONFIG, load_gt_mask, load_pred_mask

PRED_JSON_DIR = ""  # 预测 JSON 目录
GT_JSON_DIR = ""  # LabelMe JSON 目录
OUTPUT_CSV = "semseg_eval.csv"

IOU_THRESHOLD = 0.8  # 达标阈值
NUM_FMT_DECIMALS = 5
MAX_WORKERS = max(4, os.cpu_count() or 4)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)


def compute_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float(inter / union) if union else 0.0


def process_one(name: str):
    base = Path(name).stem
    pred_path = Path(PRED_JSON_DIR) / name
    gt_path = Path(GT_JSON_DIR) / name
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
            iou = compute_iou(pred_bin, gt_mask)
            ious[key] = iou

        return base, ious, ""

    except ValueError as e:
        remark = "空result" if "result 为空" in str(e) else f"异常: {e}"
        logging.warning(f"[{base}] {remark}")
        return base, {}, remark
    except Exception as e:
        logging.exception(f"[{base}] 处理异常：{e}")
        return base, {}, f"异常: {e}"


def main():
    if not PRED_JSON_DIR or not GT_JSON_DIR:
        logging.error("请先在代码顶部设置 PRED_JSON_DIR 与 GT_JSON_DIR")
        return

    names = sorted([f for f in os.listdir(PRED_JSON_DIR) if f.endswith(".json")])

    results = []
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futs = [pool.submit(process_one, n) for n in names]
        for fut in as_completed(futs):
            results.append(fut.result())

    # 准备 CSV 数据
    csv_data = []
    header = ["图片名"] + [f"{k}_IoU" for k in LABELS_CONFIG] + ["备注"]
    csv_data.append(header)

    stats = {k: {"total": 0, "above": 0, "zero": 0, "sum": 0.0} for k in LABELS_CONFIG}

    for base, ious, remark in sorted(results, key=lambda x: x[0]):
        row = [base]
        for k in LABELS_CONFIG:
            v = float(ious.get(k, 0.0))
            row.append(round(v, NUM_FMT_DECIMALS))
            stats[k]["total"] += 1
            stats[k]["sum"] += v
            if v >= IOU_THRESHOLD:
                stats[k]["above"] += 1
            if v == 0.0:
                stats[k]["zero"] += 1
        row.append(remark)
        csv_data.append(row)

    # 添加统计信息
    csv_data.append([])  # 空行
    csv_data.append([f"IoU统计 (阈值: {IOU_THRESHOLD})"])
    for k in LABELS_CONFIG:
        total = max(1, stats[k]["total"])
        above = stats[k]["above"]
        zero = stats[k]["zero"]
        mean_iou = stats[k]["sum"] / total
        csv_data.append(
            [
                f"{k}_IoU ≥ {IOU_THRESHOLD} 占比:",
                f"{above}/{total} ({above / total:.2%})",
            ]
        )
        csv_data.append([f"{k}_IoU 平均值 (mIoU):", round(mean_iou, NUM_FMT_DECIMALS)])
        if k == "Coal":
            csv_data.append([f"{k}_IoU = 0 数量:", f"{zero}"])

    # 宏平均（每类 mIoU 的平均）
    csv_data.append([])  # 空行
    csv_data.append(["Overall 宏平均 mIoU（Mean over classes）:"])
    overall = np.mean(
        [stats[k]["sum"] / max(1, stats[k]["total"]) for k in LABELS_CONFIG]
    )
    csv_data.append(["mIoU", round(float(overall), NUM_FMT_DECIMALS)])

    # 保存为 CSV 文件
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    logging.info("√ 评估完成 → %s", OUTPUT_CSV)

    for k in LABELS_CONFIG:
        t = max(1, stats[k]["total"])
        logging.info(
            "%s ≥%.2f: %d/%d (%.2f%%), mIoU=%.5f, =0 %d",
            k,
            IOU_THRESHOLD,
            stats[k]["above"],
            t,
            100.0 * stats[k]["above"] / t,
            stats[k]["sum"] / t,
            stats[k]["zero"],
        )


if __name__ == "__main__":
    main()
