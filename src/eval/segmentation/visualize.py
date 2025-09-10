from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger

from .utils import (
    COLOR_MAP,
    LABELS_ALIAS_LC,
    find_image,
    load_pred_mask,
    overlay_mask,
)

log = get_logger("eval.segmentation.visualize")


@dataclass
class SegVizConfig:
    pred_dir: str  # 预测JSON目录（原始响应）
    image_dir: str  # 原图目录
    out_dir: str  # 输出目录
    alpha: float = 0.45


def _visualize_one(pred_json: Path, image_dir: Path, out_dir: Path, alpha: float):
    base = pred_json.stem

    img_path = find_image(image_dir, base)
    if img_path is None:
        log.warning("[WARN] 找不到原图: %s", base)
        return

    img = cv2.imread(str(img_path))
    if img is None:
        log.warning("[WARN] 图片无法读取: %s", img_path)
        return

    try:
        mask_full, idx_dict = load_pred_mask(pred_json)
    except ValueError as e:
        if "result 为空" in str(e):
            out_path = out_dir / f"{base}_vis.jpg"
            cv2.imwrite(str(out_path), img)  # 空result写原图
            log.info("[空result] 保存原图副本 → %s", out_path.name)
            return
        log.error("[ERROR] 解析预测失败 %s: %s", pred_json, e)
        return
    except Exception as e:
        log.error("[ERROR] 解析预测失败 %s: %s", pred_json, e)
        return

    H, W = mask_full.shape
    img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    vis = img_resized.copy()

    for canonical_key, alias_set in LABELS_ALIAS_LC.items():
        idxs = [idx for label, idx in idx_dict.items() if label in alias_set]
        if not idxs:
            continue
        bin_mask = np.isin(mask_full, idxs).astype(np.uint8)
        color = COLOR_MAP.get(canonical_key, (255, 255, 255))
        vis = overlay_mask(vis, bin_mask, color, alpha)

    out_path = out_dir / f"{base}_vis.jpg"
    cv2.imwrite(str(out_path), vis)
    log.info("√ %s → %s", base, out_path.name)


def visualize_dir(cfg: SegVizConfig):
    pred_root = Path(cfg.pred_dir)
    img_root = Path(cfg.image_dir)
    vis_root = Path(cfg.out_dir)
    vis_root.mkdir(parents=True, exist_ok=True)

    if not pred_root.exists():
        log.error("预测目录不存在：%s", pred_root.as_posix())
        return
    if not img_root.exists():
        log.error("图像目录不存在：%s", img_root.as_posix())
        return

    json_files = [p for p in pred_root.iterdir() if p.suffix.lower() == ".json"]
    if not json_files:
        log.warning("未在预测目录中找到任何 .json 文件")
        return

    for p in json_files:
        _visualize_one(p, img_root, vis_root, cfg.alpha)
