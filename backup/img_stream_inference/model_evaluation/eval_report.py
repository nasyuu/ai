import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from model_evaluation.matcher import assign_boxes


def parse_shapes(json_path) -> List[Dict]:
    """
    解析 LabelMe JSON 文件，提取标注对象和边界框。
    :param json_path: LabelMe JSON 文件路径
    :return: 标注对象列表，每个对象包含标签和边界框
    :rtype: List[Dict]
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    objs: List[Dict] = []
    for shp in data.get("shapes", []):
        label, pts = shp["label"], shp["points"]
        stype = shp.get("shape_type", "polygon")
        if stype == "rectangle" and len(pts) == 2:  # 两点 → 四元组
            (x1, y1), (x2, y2) = pts
            bbox = [x1, y1, x2, y2]
        else:
            bbox = pts
        obj = {"label": label, "bbox": bbox}
        score = shp.get("attributes", {}).get("score")
        if score is not None:
            obj["score"] = float(score)
        objs.append(obj)
    return objs


def group_by_label(objs: List[Dict]) -> Dict[str, List]:
    """
    按标签对标注对象进行分组。
    :param objs: 标注对象列表
    :return: 按标签分组的字典，键为标签名，值为边界框列表
    :rtype: Dict[str, List]
    """
    by: Dict[str, List] = {}
    for o in objs:
        by.setdefault(o["label"], []).append(o["bbox"])
    return by


def evaluate(
    gt_dir,
    pred_dir,
    iou_thr: float = 0.5,
    out_csv: str = "评估汇总报告.csv",
) -> pd.DataFrame:
    """
    评估预测结果与真实标注的匹配情况，并生成汇总报告。
    :param gt_dir: 真实标注 JSON 文件所在目录
    :param pred_dir: 预测结果 JSON 文件所在目录
    :param iou_thr: IoU 阈值，用于判断匹配
    :param out_csv: 输出的 CSV 汇总报告文件名
    :return: 包含评估结果的 DataFrame
    """
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)

    tp, fn, fp = {}, {}, {}
    labels: set[str] = set()
    total_imgs, fully_correct = 0, 0

    for gt_json in sorted(gt_dir.glob("*.json")):
        fname = gt_json.name
        pred_json = pred_dir / fname
        total_imgs += 1

        gt_objs = parse_shapes(gt_json)
        pred_objs = parse_shapes(pred_json) if pred_json.exists() else []

        gt_map, pred_map = group_by_label(gt_objs), group_by_label(pred_objs)
        img_labels = set(gt_map) | set(pred_map)
        labels.update(img_labels)

        # 框级逐类别
        for lb in img_labels:
            res = assign_boxes(
                [{"label": lb, "bbox": b} for b in gt_map.get(lb, [])],
                [{"label": lb, "bbox": b} for b in pred_map.get(lb, [])],
                iou_thr=iou_thr,
                method="hungarian",
                use_label=False,
            )
            tp[lb] = tp.get(lb, 0) + len(res["matches"])
            fn[lb] = fn.get(lb, 0) + len(res["unmatched_gt"])
            fp[lb] = fp.get(lb, 0) + len(res["unmatched_pred"])

        # 图片级“完全正确”
        img_res = assign_boxes(gt_objs, pred_objs, iou_thr=iou_thr, method="hungarian")
        if not img_res["unmatched_gt"] and not img_res["unmatched_pred"]:
            fully_correct += 1

    # 汇总 DataFrame
    rows: List[Dict] = []
    for lb in sorted(labels):
        TP, FN, FP = tp.get(lb, 0), fn.get(lb, 0), fp.get(lb, 0)

        # 改进精确率计算：当无预测时显示N/A
        if TP + FP == 0:
            prec = "N/A (无预测)"
        else:
            prec = round(TP / (TP + FP), 4)

        # 改进召回率计算：当无GT时显示N/A
        if TP + FN == 0:
            rec = "N/A (无GT)"
        else:
            rec = round(TP / (TP + FN), 4)

        rows.append(
            dict(类别=lb, TP=TP, FN=FN, FP=FP, 精确率_Precision=prec, 召回率_Recall=rec)
        )

    rows.append(
        dict(
            类别="图片级完全正确率",
            TP="",
            FN="",
            FP="",
            精确率_Precision=(
                f"{fully_correct}/{total_imgs} = {fully_correct / total_imgs:.2%}"
                if total_imgs
                else "0"
            ),
            召回率_Recall="",
        )
    )

    df = pd.DataFrame(rows)

    output_path = Path(out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False, encoding="utf-8")

    import logging

    logging.info(f"√ 评估完成 → {out_csv}")
    return df


if __name__ == "__main__":
    evaluate(
        gt_dir="",
        pred_dir="",
        iou_thr=0.5,
    )
