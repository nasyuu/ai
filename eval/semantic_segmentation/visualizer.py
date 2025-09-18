"""语义分割结果可视化。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Mapping, Optional

import cv2
import numpy as np

from utils.exceptions import ValidationError
from utils.logger import get_logger

from .labels import (
    SemSegLabelConfig,
    find_image,
    load_prediction_mask,
    overlay_mask,
)

__all__ = [
    "SemSegVizConfig",
    "SemSegVisualizer",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class SemSegVizConfig:
    """语义分割可视化配置。"""

    pred_dir: Path
    image_dir: Path
    output_dir: Path
    label_config: SemSegLabelConfig = dataclass_field(default_factory=SemSegLabelConfig.default)
    alpha: float = 0.45
    max_workers: int = 4
    save_empty_as_original: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "SemSegVizConfig":
        try:
            pred_dir = Path(str(payload["pred_dir"]))
            image_dir = Path(str(payload["image_dir"]))
            output_dir = Path(str(payload["output_dir"]))
        except KeyError as exc:  # noqa: PERF203
            raise ValidationError(f"配置项 {exc.args[0]} 缺失") from exc

        label_config_payload = payload.get("label_config")
        if isinstance(label_config_payload, Mapping):
            aliases = {
                str(key): [str(alias) for alias in value]
                for key, value in label_config_payload.get("aliases", {}).items()
            }
            colors = {
                str(key): tuple(map(int, value))
                for key, value in label_config_payload.get("colors", {}).items()
            }
            strict_rle = bool(label_config_payload.get("strict_rle", False))
            label_config = SemSegLabelConfig(
                aliases=aliases or SemSegLabelConfig.default().aliases,
                colors=colors or SemSegLabelConfig.default().colors,
                strict_rle=strict_rle,
            )
        else:
            label_config = SemSegLabelConfig.default()

        return cls(
            pred_dir=pred_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            label_config=label_config,
            alpha=float(payload.get("alpha", 0.45)),
            max_workers=int(payload.get("max_workers", 4)),
            save_empty_as_original=bool(payload.get("save_empty_as_original", True)),
        )


class SemSegVisualizer:
    """负责绘制语义分割预测掩码。"""

    def __init__(self, config: SemSegVizConfig) -> None:
        self.config = config
        self.logger = logger
        self.alias_map = config.label_config.alias_lower_map()

    def visualize_directory(self) -> None:
        if not self.config.pred_dir.exists():
            raise ValidationError(
                "预测目录不存在", details={"path": str(self.config.pred_dir)}
            )
        if not self.config.image_dir.exists():
            raise ValidationError(
                "图片目录不存在", details={"path": str(self.config.image_dir)}
            )

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        pred_files = sorted(self.config.pred_dir.glob("*.json"))
        if not pred_files:
            self.logger.warning("预测目录未找到 JSON 文件: %s", self.config.pred_dir)
            return

        success = skipped = failed = 0
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._visualize_single, path): path
                for path in pred_files
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    if result is True:
                        success += 1
                    elif result is False:
                        failed += 1
                    else:
                        skipped += 1
                except Exception as exc:  # noqa: BLE001
                    self.logger.error("可视化异常 %s: %s", path.name, exc)
                    failed += 1

        self.logger.info(
            "语义分割可视化完成: 成功 %s, 失败 %s, 跳过 %s",
            success,
            failed,
            skipped,
        )
        if success:
            self.logger.info("输出目录: %s", output_dir)

    def _visualize_single(self, pred_json: Path) -> Optional[bool]:
        stem = pred_json.stem
        image_path = find_image(self.config.image_dir, stem)
        if image_path is None:
            self.logger.warning("未找到配对原图: %s", stem)
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.warning("原图无法读取: %s", image_path)
            return False

        try:
            mask, idx_dict = load_prediction_mask(
                pred_json, label_config=self.config.label_config
            )
        except ValidationError as exc:
            if "result 为空" in str(exc) and self.config.save_empty_as_original:
                return self._save_original(image, stem)
            self.logger.warning("解析预测失败 %s: %s", pred_json.name, exc)
            return False

        height, width = mask.shape
        image_resized = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_LINEAR
        )

        visualization = image_resized.copy()
        for canonical, aliases in self.alias_map.items():
            indices = [idx for label, idx in idx_dict.items() if label in aliases]
            if not indices:
                continue
            binary_mask = np.isin(mask, indices)
            if not binary_mask.any():
                continue
            color = self.config.label_config.color_for(canonical)
            visualization = overlay_mask(
                visualization,
                binary_mask.astype(np.uint8),
                color,
                alpha=self.config.alpha,
            )

        output_path = self.config.output_dir / f"{stem}_vis.jpg"
        cv2.imwrite(str(output_path), visualization)
        return True

    def _save_original(self, image: np.ndarray, stem: str) -> bool:
        output_path = self.config.output_dir / f"{stem}_vis.jpg"
        cv2.imwrite(str(output_path), image)
        self.logger.info("空 result，已保存原图: %s", output_path.name)
        return True
