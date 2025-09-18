"""语义分割标签与掩码处理工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from utils.exceptions import ValidationError

__all__ = [
    "SemSegLabelConfig",
    "decode_rle",
    "load_prediction_mask",
    "load_ground_truth_mask",
    "overlay_mask",
    "find_image",
]

_DEFAULT_ALIASES = {
    "Belt": ["belt", "pidai", "Belt"],
    "Coal": ["coal", "meiliu", "Coal"],
}
_DEFAULT_COLORS = {
    "Belt": (0, 255, 0),
    "Coal": (0, 0, 255),
}


@dataclass(slots=True)
class SemSegLabelConfig:
    """管理语义分割标签别名、颜色及 RLE 规则。"""

    aliases: Dict[str, List[str]] = field(
        default_factory=lambda: dict(_DEFAULT_ALIASES)
    )
    colors: Dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(_DEFAULT_COLORS)
    )
    strict_rle: bool = False

    @classmethod
    def default(cls) -> "SemSegLabelConfig":
        return cls()

    def alias_lower_map(self) -> dict[str, set[str]]:
        return {
            key: {alias.lower() for alias in values}
            for key, values in self.aliases.items()
        }

    def color_for(self, label: str) -> tuple[int, int, int]:
        return self.colors.get(label, (255, 255, 255))


def decode_rle(
    rle_raw: Mapping[str, Sequence[int]] | Sequence[Sequence[int]],
    shape: tuple[int, int],
    *,
    strict: bool,
) -> np.ndarray:
    if isinstance(rle_raw, Sequence) and len(rle_raw) == 2:
        values, lengths = rle_raw  # type: ignore[assignment]
    elif isinstance(rle_raw, Mapping):
        values = rle_raw.get("values", [])
        lengths = rle_raw.get("lengths", [])
    else:
        raise ValidationError("RLE 数据格式错误")

    h, w = map(int, shape)
    values_arr = np.asarray(values, dtype=np.int32)
    lengths_arr = np.asarray(lengths, dtype=np.int32)

    if values_arr.size == 0 and lengths_arr.size == 0:
        return np.zeros((h, w), np.uint8)

    if (
        values_arr.ndim != 1
        or lengths_arr.ndim != 1
        or values_arr.size != lengths_arr.size
    ):
        raise ValidationError("RLE 的 values 与 lengths 长度或维度不匹配")

    total = int(lengths_arr.sum())
    expected = h * w

    flat = np.repeat(values_arr, lengths_arr)
    if total == expected:
        return flat.reshape((h, w)).astype(np.uint8)

    if strict:
        raise ValidationError(
            "RLE 长度与尺寸不符",
            details={"total": total, "expected": expected, "shape": shape},
        )

    if w > 0 and total % w == 0:
        h2 = total // w
        mask = flat.reshape((h2, w)).astype(np.uint8)
        if h2 > h:
            mask = mask[:h, :]
        elif h2 < h:
            pad = np.zeros((h - h2, w), dtype=mask.dtype)
            mask = np.vstack([mask, pad])
        return mask

    if h > 0 and total % h == 0:
        w2 = total // h
        mask = flat.reshape((h, w2)).astype(np.uint8)
        if w2 > w:
            mask = mask[:, :w]
        elif w2 < w:
            pad = np.zeros((h, w - w2), dtype=mask.dtype)
            mask = np.hstack([mask, pad])
        return mask

    raise ValidationError(
        "RLE 长度与尺寸不符，且无法推断真实尺寸",
        details={"total": total, "expected": expected, "shape": shape},
    )


def load_prediction_mask(
    json_path: Path,
    *,
    label_config: SemSegLabelConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "无法读取预测 JSON", details={"path": str(json_path)}
        ) from exc

    result = data.get("result", [])
    if not isinstance(result, list) or not result:
        raise ValidationError("result 为空", details={"path": str(json_path)})

    entry = result[0]
    size = entry.get("size")
    if not size or len(size) != 2:
        raise ValidationError(
            "预测 JSON 缺少 size 字段", details={"path": str(json_path)}
        )
    h, w = int(size[0]), int(size[1])

    mask = decode_rle(
        entry.get("rle_dict", [[], []]),
        (h, w),
        strict=label_config.strict_rle,
    )

    raw_idx = entry.get("idx_dict", {})
    if not isinstance(raw_idx, Mapping):
        raise ValidationError(
            "idx_dict 缺失或类型错误", details={"path": str(json_path)}
        )

    lower_idx = {str(key).lower(): int(value) for key, value in raw_idx.items()}
    return mask, lower_idx


def _fill_labelme_shape(mask: np.ndarray, shape_entry: Mapping[str, object]) -> None:
    points = np.asarray(shape_entry.get("points", []), dtype=np.float32)
    if points.size == 0:
        return
    shape_type = str(shape_entry.get("shape_type", "")).lower()
    if shape_type in ("polygon", "", None):
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        return
    if shape_type == "rectangle" and points.shape[0] >= 2:
        (x1, y1), (x2, y2) = points[:2]
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
        return
    if shape_type == "circle" and points.shape[0] >= 2:
        center, boundary = points[:2]
        radius = int(np.linalg.norm(center - boundary))
        cv2.circle(mask, (int(center[0]), int(center[1])), radius, 1, -1)
        return
    if shape_type in ("linestrip", "line") and points.shape[0] >= 2:
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(
                mask,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                1,
                3,
            )
        return
    cv2.fillPoly(mask, [points.astype(np.int32)], 1)


def _build_label_mask(
    shapes: Iterable[Mapping[str, object]],
    aliases_lower: set[str],
    shape: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape_entry in shapes:
        label = str(shape_entry.get("label", "")).lower()
        if label in aliases_lower:
            _fill_labelme_shape(mask, shape_entry)
    return mask


def load_ground_truth_mask(
    labelme_path: Path,
    canonical_label: str,
    *,
    label_config: SemSegLabelConfig,
    shape: tuple[int, int],
) -> np.ndarray:
    try:
        with labelme_path.open(encoding="utf-8") as handle:
            shapes = json.load(handle).get("shapes", [])
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "无法解析 LabelMe JSON", details={"path": str(labelme_path)}
        ) from exc

    aliases_map = label_config.alias_lower_map()
    if canonical_label not in aliases_map:
        raise ValidationError(
            "标签配置缺少指定键",
            details={"label": canonical_label},
        )

    mask = _build_label_mask(shapes, aliases_map[canonical_label], shape)

    if canonical_label == "Belt" and "Coal" in aliases_map:
        coal_mask = _build_label_mask(shapes, aliases_map["Coal"], shape)
        mask[coal_mask > 0] = 0

    return mask


def overlay_mask(
    image: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    if binary_mask.max() == 0:
        return image
    color_layer = np.zeros_like(image)
    color_layer[binary_mask > 0] = color
    return cv2.addWeighted(color_layer, alpha, image, 1 - alpha, 0)


def find_image(
    image_dir: Path, stem: str, *, extensions: Optional[Sequence[str]] = None
) -> Optional[Path]:
    exts = extensions or (".jpg", ".jpeg", ".png", ".bmp")
    for ext in exts:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None
