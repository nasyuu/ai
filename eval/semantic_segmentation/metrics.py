"""语义分割评估与统计。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from utils.exceptions import ValidationError
from utils.logger import get_logger

from .labels import SemSegLabelConfig, load_ground_truth_mask, load_prediction_mask

__all__ = [
    "SemSegEvalConfig",
    "SemSegEvaluator",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class SemSegEvalConfig:
    """语义分割评估所需配置。"""

    pred_dir: Path
    gt_dir: Path
    output_csv: Path
    label_config: SemSegLabelConfig = dataclass_field(default_factory=SemSegLabelConfig.default)
    iou_threshold: float = 0.8
    decimals: int = 5
    max_workers: int = 4

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "SemSegEvalConfig":
        try:
            pred_dir = Path(str(payload["pred_dir"]))
            gt_dir = Path(str(payload["gt_dir"]))
            output_csv = Path(str(payload["output_csv"]))
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
            gt_dir=gt_dir,
            output_csv=output_csv,
            label_config=label_config,
            iou_threshold=float(payload.get("iou_threshold", 0.8)),
            decimals=int(payload.get("decimals", 5)),
            max_workers=int(payload.get("max_workers", 4)),
        )


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union else 0.0


class SemSegEvaluator:
    """执行语义分割 IoU 评估并生成报告。"""

    def __init__(self, config: SemSegEvalConfig) -> None:
        self.config = config
        self.logger = logger
        self.alias_map = config.label_config.alias_lower_map()

    def evaluate(self) -> pd.DataFrame:
        if not self.config.pred_dir.exists():
            raise ValidationError(
                "预测目录不存在", details={"path": str(self.config.pred_dir)}
            )
        if not self.config.gt_dir.exists():
            raise ValidationError(
                "标注目录不存在", details={"path": str(self.config.gt_dir)}
            )

        pred_files = sorted(self.config.pred_dir.glob("*.json"))
        if not pred_files:
            self.logger.warning("预测目录未找到 JSON 文件: %s", self.config.pred_dir)

        results: list[tuple[str, dict[str, float], str]] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_single, path): path
                for path in pred_files
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except ValidationError as exc:
                    path = futures[future]
                    self.logger.warning("评估失败 %s: %s", path.name, exc)
                    results.append((path.stem, {}, str(exc)))
                except Exception as exc:  # noqa: BLE001
                    path = futures[future]
                    self.logger.error("评估异常 %s: %s", path.name, exc)
                    results.append((path.stem, {}, f"异常: {exc}"))

        dataframe = self._build_dataframe(results)
        self._write_output(dataframe)
        return dataframe

    def _evaluate_single(self, pred_json: Path) -> tuple[str, dict[str, float], str]:
        stem = pred_json.stem
        gt_json = self.config.gt_dir / pred_json.name
        if not gt_json.exists():
            return stem, {}, "缺少对应 GT"

        try:
            pred_mask, idx_dict = load_prediction_mask(
                pred_json, label_config=self.config.label_config
            )
        except ValidationError as exc:
            return stem, {}, str(exc)

        height, width = pred_mask.shape
        ious: dict[str, float] = {}

        for canonical in self.alias_map:
            gt_mask = load_ground_truth_mask(
                gt_json,
                canonical,
                label_config=self.config.label_config,
                shape=(height, width),
            )
            aliases = self.alias_map[canonical]
            prediction_indices = [
                idx for label, idx in idx_dict.items() if label in aliases
            ]
            if prediction_indices:
                pred_binary = np.isin(pred_mask, prediction_indices)
            else:
                pred_binary = np.zeros_like(gt_mask, dtype=bool)
            ious[canonical] = _compute_iou(pred_binary, gt_mask.astype(bool))

        return stem, ious, ""

    def _build_dataframe(
        self, entries: Iterable[tuple[str, Mapping[str, float], str]]
    ) -> pd.DataFrame:
        labels = list(self.alias_map.keys())
        rows = []
        stats = {
            label: {"total": 0, "above": 0, "zero": 0, "sum": 0.0} for label in labels
        }

        for image_name, iou_map, remark in sorted(entries, key=lambda item: item[0]):
            row = {"image": image_name, "remark": remark}
            for label in labels:
                value = float(iou_map.get(label, 0.0))
                row[f"{label}_iou"] = round(value, self.config.decimals)
                stats[label]["total"] += 1
                stats[label]["sum"] += value
                if value >= self.config.iou_threshold:
                    stats[label]["above"] += 1
                if value == 0.0:
                    stats[label]["zero"] += 1
            rows.append(row)

        df = pd.DataFrame(rows)
        summary_rows = []
        for label in labels:
            total = max(1, stats[label]["total"])
            above = stats[label]["above"]
            zero = stats[label]["zero"]
            mean = stats[label]["sum"] / total
            summary_rows.append(
                {
                    "image": f"{label}_summary",
                    "remark": (
                        f"≥{self.config.iou_threshold}: {above}/{total} ({above / total:.2%}), "
                        f"mIoU={mean:.{self.config.decimals}f}, zero={zero}"
                    ),
                }
            )

        if labels:
            overall = np.mean(
                [
                    stats[label]["sum"] / max(1, stats[label]["total"])
                    for label in labels
                ]
            )
            summary_rows.append(
                {
                    "image": "macro_mIoU",
                    "remark": f"{overall:.{self.config.decimals}f}",
                }
            )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            df = pd.concat([df, pd.DataFrame([{}]), summary_df], ignore_index=True)

        return df

    def _write_output(self, dataframe: pd.DataFrame) -> None:
        path = self.config.output_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False, encoding="utf-8")
        self.logger.info("语义分割评估报告已生成: %s", path)
