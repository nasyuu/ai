from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from utils.logger import get_logger

from .matching import assign_boxes

log = get_logger("eval.detection.visualize")


class VizMode(str, Enum):
    Stats = "stats"  # 画TP/FP/FN + 数字
    Labels = "labels"  # 仅按类别颜色画预测框


@dataclass
class VizConfig:
    gt_dir: str
    pred_dir: str
    image_dir: str
    out_dir_correct: str
    out_dir_error: str
    iou_thr: float = 0.5
    mode: VizMode = VizMode.Stats
    max_workers: int = 1


def _parse_objs(json_file: Path) -> List[Dict]:
    data = json.loads(json_file.read_text(encoding="utf-8"))
    objs = []
    for shp in data.get("shapes", []):
        label = shp["label"]
        pts = shp["points"]
        stype = shp.get("shape_type", "polygon")
        if stype == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            bbox = [x1, y1, x2, y2]
        else:
            bbox = pts
        objs.append({"label": label, "bbox": bbox})
    return objs


def _find_image(image_dir: Path, base: str):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"):
        p = image_dir / f"{base}{ext}"
        if p.exists():
            return str(p)
    return None


def _top_left(pts):
    if len(pts) == 4 and isinstance(pts[0], (int, float)):
        return int(pts[0]), int(pts[1])
    if len(pts) == 2 and isinstance(pts[0], (list, tuple)):
        xs, ys = zip(*pts)
    else:
        xs, ys = zip(
            *(
                pts
                if isinstance(pts[0], (list, tuple))
                else [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
            )
        )
    return int(min(xs)), int(min(ys))


def _draw_geom(img, pts, color, thick):
    if len(pts) == 4 and isinstance(pts[0], (int, float)):
        x1, y1, x2, y2 = map(int, pts)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    elif len(pts) == 2 and isinstance(pts[0], (list, tuple)):
        (x1, y1), (x2, y2) = pts
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
    else:
        pts_int = np.array(
            [
                (int(x), int(y))
                for x, y in (
                    pts
                    if isinstance(pts[0], (list, tuple))
                    else [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
                )
            ]
        )
        cv2.polylines(img, [pts_int], isClosed=True, color=color, thickness=thick)


def _get_fixed_colors(labels):
    vivid = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
    ]
    return {lb: vivid[i % len(vivid)] for i, lb in enumerate(sorted(labels))}


def _draw_stats(img_path, gt, pred, matches, save_path):
    img = cv2.imread(img_path)
    if img is None:
        log.error("无法读取图片: %s", img_path)
        return
    TP_COLOR, FP_COLOR, FN_COLOR = (0, 255, 0), (0, 0, 255), (0, 255, 255)
    matched_gt = {gi for gi, _ in matches}
    matched_pred = {pi for _, pi in matches}
    for gi, _ in matches:
        _draw_geom(img, gt[gi]["bbox"], TP_COLOR, 3)
    for i, g in enumerate(gt):
        if i not in matched_gt:
            _draw_geom(img, g["bbox"], FN_COLOR, 3)
    for i, p in enumerate(pred):
        if i not in matched_pred:
            _draw_geom(img, p["bbox"], FP_COLOR, 3)
    tp, fn, fp = len(matches), len(gt) - len(matches), len(pred) - len(matches)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x0, y0, gap = 15, 45, 55
    cv2.putText(img, f"TP: {tp}", (x0, y0), font, 1.2, TP_COLOR, 3)
    cv2.putText(img, f"FP: {fp}", (x0, y0 + gap), font, 1.2, FP_COLOR, 3)
    cv2.putText(img, f"FN: {fn}", (x0, y0 + 2 * gap), font, 1.2, FN_COLOR, 3)
    cv2.imwrite(save_path, img)


def _draw_labels(img_path, gt, pred, matches, save_path, colors):
    img = cv2.imread(img_path)
    if img is None:
        log.error("无法读取图片: %s", img_path)
        return
    matched_pred = {pi for _, pi in matches}
    for i, p in enumerate(pred):
        is_tp = i in matched_pred
        color = colors.get(p["label"], (0, 255, 0) if is_tp else (255, 0, 0))
        thick = 3 if is_tp else 1
        _draw_geom(img, p["bbox"], color, thick)
        x, y = _top_left(p["bbox"])
        cv2.putText(
            img,
            p["label"],
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(save_path, img)


def visualize_dir(cfg: VizConfig):
    gt_dir = Path(cfg.gt_dir)
    pred_dir = Path(cfg.pred_dir)
    img_dir = Path(cfg.image_dir)

    out_ok = Path(cfg.out_dir_correct)
    out_ok.mkdir(parents=True, exist_ok=True)
    out_err = Path(cfg.out_dir_error)
    out_err.mkdir(parents=True, exist_ok=True)

    fns = [p.name for p in pred_dir.glob("*.json")]
    if not fns:
        log.warning("预测目录中没有JSON：%s", pred_dir.as_posix())
        return

    # 收集配色（仅Labels模式）
    colors = None
    if cfg.mode == VizMode.Labels:
        labels = set()
        for name in fns:
            gp, pp = gt_dir / name, pred_dir / name
            try:
                if gp.exists():
                    labels.update([o["label"] for o in _parse_objs(gp)])
                if pp.exists():
                    labels.update([o["label"] for o in _parse_objs(pp)])
            except Exception:
                pass
        colors = _get_fixed_colors(labels)
        log.info("收集到 %d 个类别用于配色", len(labels))

    ok = err = skip = 0
    for fname in fns:
        base = Path(fname).stem
        gp, pp = gt_dir / fname, pred_dir / fname
        ip = _find_image(img_dir, base)
        missing = []
        if not gp.exists():
            missing.append(f"GT: {gp}")
        if not pp.exists():
            missing.append(f"PRED: {pp}")
        if not ip:
            missing.append(f"IMG: {base}.*")
        if missing:
            skip += 1
            for m in missing:
                log.warning("缺失文件: %s", m)
            continue

        try:
            gt = _parse_objs(gp)
            pred = _parse_objs(pp)
            res = assign_boxes(gt, pred, iou_thr=cfg.iou_thr, method="hungarian")
            matches = res["matches"]
            all_ok = (not res["unmatched_gt"]) and (not res["unmatched_pred"])
            out_dir = out_ok if all_ok else out_err
            save_path = (out_dir / f"{base}.jpg").as_posix()

            if cfg.mode == VizMode.Stats:
                _draw_stats(ip, gt, pred, matches, save_path)
            else:
                _draw_labels(ip, gt, pred, matches, save_path, colors)

            ok += 1 if all_ok else 0
            err += 0 if all_ok else 1
        except Exception as e:
            log.error("处理 %s 失败：%s", base, e)
            skip += 1

    log.info("可视化完成：正确=%d, 错误=%d, 跳过=%d", ok, err, skip)
