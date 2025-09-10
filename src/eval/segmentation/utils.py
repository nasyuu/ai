from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger("eval.segmentation.utils")

# ------- 类别与别名（可按需扩展/外部注入） -------
LABELS_CONFIG: Dict[str, List[str]] = {
    "Belt": ["belt", "pidai", "Belt"],
    "Coal": ["coal", "meiliu", "Coal"],
}
LABELS_ALIAS_LC: Dict[str, set] = {
    key: set(a.lower() for a in aliases) for key, aliases in LABELS_CONFIG.items()
}
# 可视化颜色（BGR）
COLOR_MAP = {
    "Belt": (0, 255, 0),
    "Coal": (0, 0, 255),
}

# 严格/容错：True=严格校验尺寸；False=自动对齐（推断后裁剪/补零）
STRICT_RLE_MODE = False


@dataclass(frozen=True)
class SegAliases:
    """如果未来要从外部传入别名映射，可用此dataclass承载。"""

    labels_config: Dict[str, List[str]] = field(default_factory=lambda: LABELS_CONFIG.copy())
    alias_lc: Dict[str, set] = field(default_factory=lambda: LABELS_ALIAS_LC.copy())


def rle_decode(
    rle_raw: Union[dict, List[List[int]]], shape_: Tuple[int, int]
) -> np.ndarray:
    """
    RLE 解码：支持 [values, lengths] 或 {"values":[...], "lengths":[...]}。
    - 严格模式：sum(lengths) 必须等于 H*W；
    - 容错模式：不等则尝试以单边为准推断真实尺寸并裁剪/补零对齐到 (H,W)。
    """
    if isinstance(rle_raw, list) and len(rle_raw) == 2:
        values, lengths = rle_raw
    elif isinstance(rle_raw, dict):
        values, lengths = rle_raw.get("values", []), rle_raw.get("lengths", [])
    else:
        raise ValueError("rle_dict 应为 [values, lengths] 或 {values,lengths}")

    H, W = int(shape_[0]), int(shape_[1])
    values = np.asarray(values, dtype=np.int32)
    lengths = np.asarray(lengths, dtype=np.int32)

    if values.size == 0 and lengths.size == 0:
        return np.zeros((H, W), np.uint8)

    if values.ndim != 1 or lengths.ndim != 1 or values.size != lengths.size:
        raise ValueError("RLE 格式错误：values/lengths 维度或长度不匹配")

    total = int(lengths.sum())
    expected = H * W
    flat = np.repeat(values, lengths)

    if total == expected:
        return flat.reshape((H, W)).astype(np.uint8)

    if STRICT_RLE_MODE:
        raise ValueError(
            f"RLE 长度与尺寸不匹配：sum(lengths)={total}, 期望={expected} ({H}x{W})"
        )

    # —— 容错：推断并对齐 ——
    if W > 0 and total % W == 0:
        H2 = total // W
        mask = flat.reshape((H2, W)).astype(np.uint8)
        if H2 != H:
            if H2 > H:
                mask = mask[:H, :]
            else:
                mask = np.vstack([mask, np.zeros((H - H2, W), mask.dtype)])
        return mask

    if H > 0 and total % H == 0:
        W2 = total // H
        mask = flat.reshape((H, W2)).astype(np.uint8)
        if W2 != W:
            if W2 > W:
                mask = mask[:, :W]
            else:
                mask = np.hstack([mask, np.zeros((H, W - W2), mask.dtype)])
        return mask

    raise ValueError(
        f"RLE 解码长度 {total} 与期望 {expected}（{H}x{W}）不匹配，且无法按单边推断修正；"
        f"请检查导出 size 与 RLE 是否一致。"
    )


def load_pred_mask(json_path: Path):
    """
    新结构：
    {
      "result": [
        {
          "idx_dict": {...},               # label->index
          "rle_dict": [values,lengths] 或 {"values","lengths"},
          "size": [H, W]
        }
      ]
    }
    返回: (mask: HxW uint8, idx_dict_lower: Dict[str,int])
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rlist = data.get("result", [])
    if not isinstance(rlist, list) or not rlist:
        raise ValueError("result 为空")

    res = rlist[0]
    size = res.get("size")
    if not size or len(size) != 2:
        raise KeyError("size 缺失或非法，应为 [H, W]")
    H, W = int(size[0]), int(size[1])

    rle_raw = res.get("rle_dict", [[], []])  # 允许空
    mask = rle_decode(rle_raw, (H, W))

    raw_idx = res.get("idx_dict", {})
    if not isinstance(raw_idx, dict):
        raise KeyError("idx_dict 缺失或非法")
    idx_dict = {str(k).lower(): int(v) for k, v in raw_idx.items()}
    return mask, idx_dict


def _fill_labelme_shape(mask: np.ndarray, shp: dict):
    pts = np.asarray(shp.get("points", []), dtype=np.float32)
    if pts.size == 0:
        return
    stype = str(shp.get("shape_type", "")).lower()
    if stype in ("polygon", "", None):
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    elif stype == "rectangle" and pts.shape[0] >= 2:
        p1, p2 = pts[0], pts[1]
        x1, y1 = map(int, np.floor(p1))
        x2, y2 = map(int, np.floor(p2))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
    elif stype == "circle" and pts.shape[0] >= 2:
        c, b = pts[0], pts[1]
        r = int(np.linalg.norm(c - b))
        cv2.circle(mask, (int(c[0]), int(c[1])), r, 1, -1)
    elif stype in ("linestrip", "line") and pts.shape[0] >= 2:
        for i in range(len(pts) - 1):
            p1 = tuple(int(x) for x in pts[i])
            p2 = tuple(int(x) for x in pts[i + 1])
            cv2.line(mask, p1, p2, 1, 3)
    else:
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)


def _build_label_mask(shapes: list, aliases_lc: set, H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    for shp in shapes:
        if str(shp.get("label", "")).lower() in aliases_lc:
            _fill_labelme_shape(m, shp)
    return m


def load_gt_mask(
    labelme_json: Path, canonical_key: str, shape_: Tuple[int, int]
) -> np.ndarray:
    """
    以预测尺寸 shape_ 栅格化 LabelMe GT；业务规则：Belt 减去 Coal。
    """
    H, W = shape_
    shapes = json.loads(labelme_json.read_text(encoding="utf-8")).get("shapes", [])
    tgt = _build_label_mask(shapes, LABELS_ALIAS_LC[canonical_key], H, W)
    if canonical_key == "Belt":
        coal = _build_label_mask(shapes, LABELS_ALIAS_LC["Coal"], H, W)
        tgt[coal > 0] = 0
    return tgt


def overlay_mask(img: np.ndarray, bin_mask: np.ndarray, color, alpha=0.45):
    if bin_mask.max() == 0:
        return img
    color_layer = np.zeros_like(img)
    color_layer[bin_mask > 0] = color
    return cv2.addWeighted(color_layer, alpha, img, 1 - alpha, 0)


def find_image(image_dir: Path, base: str):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"):
        p = image_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None
