import logging
from pathlib import Path

import cv2
import numpy as np
from ss_utils import (
    COLOR_MAP,
    LABELS_ALIAS_LC,
    find_image,
    load_pred_mask,
    overlay_mask,
)

PRED_JSON_DIR = "pred_jsons"  # 预测 JSON 目录
IMAGE_DIR = "images"  # 原图目录
VIS_DIR = "vis_masks"  # 输出目录
ALPHA = 0.45  # 叠加透明度


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)


def visualize_one(pred_json_path: Path, image_dir: Path, vis_dir: Path):
    base = pred_json_path.stem

    img_path = find_image(image_dir, base)
    if img_path is None:
        logging.warning(f"[WARN] 找不到原图: {base}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        logging.warning(f"[WARN] 图片无法读取: {img_path}")
        return

    try:
        mask_full, idx_dict = load_pred_mask(pred_json_path)
    except ValueError as e:
        # result 为空 → 保存原图副本，避免缺文件
        if "result 为空" in str(e):
            out_path = vis_dir / f"{base}_vis.jpg"
            cv2.imwrite(str(out_path), img)
            logging.info(f"[空result] 保存原图副本 → {out_path.name}")
            return
        logging.error(f"[ERROR] 解析预测失败 {pred_json_path}: {e}")
        return
    except Exception as e:
        logging.error(f"[ERROR] 解析预测失败 {pred_json_path}: {e}")
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
        vis = overlay_mask(vis, bin_mask, color, ALPHA)

    out_path = vis_dir / f"{base}_vis.jpg"
    cv2.imwrite(str(out_path), vis)
    logging.info(f"√ {base}  → {out_path.name}")


def main():
    pred_root = Path(PRED_JSON_DIR)
    img_root = Path(IMAGE_DIR)
    vis_root = Path(VIS_DIR)
    vis_root.mkdir(exist_ok=True)

    if not pred_root.exists():
        logging.error(f"预测目录不存在：{pred_root}")
        return
    if not img_root.exists():
        logging.error(f"图像目录不存在：{img_root}")
        return

    json_files = [p for p in pred_root.iterdir() if p.suffix.lower() == ".json"]
    if not json_files:
        logging.warning("未在预测目录中找到任何 .json 文件")
    for p in json_files:
        visualize_one(p, img_root, vis_root)


if __name__ == "__main__":
    main()
