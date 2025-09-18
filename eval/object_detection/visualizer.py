"""目标检测结果可视化工具。"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

from utils.exceptions import ValidationError
from utils.logger import get_logger

from .matcher import BoundingBox, MatchResult, assign_boxes
from .metrics import parse_labelme_shapes

__all__ = [
    "DetectionVizConfig",
    "DetectionVisualizer",
]

logger = get_logger(__name__)


def _create_safe_tqdm():
    try:
        from tqdm import tqdm

        if getattr(sys, "frozen", False):

            def silent(iterable, **kwargs):
                kwargs.pop("desc", None)
                kwargs.pop("position", None)
                try:
                    return tqdm(iterable, disable=True, **kwargs)
                except Exception:  # noqa: BLE001
                    return iterable

            return silent
        return tqdm
    except Exception:  # noqa: BLE001
        return lambda iterable, **_: iterable


safe_tqdm = _create_safe_tqdm()


SequenceLike = Sequence[float] | Sequence[Sequence[float]]


@dataclass(slots=True)
class DetectionVizConfig:
    """控制可视化行为的配置。"""

    gt_dir: Path
    pred_dir: Path
    output_correct_dir: Path
    output_error_dir: Path
    images_dir: Optional[Path] = None
    iou_threshold: float = 0.5
    stats_mode: bool = False
    max_workers: int = 4
    match_method: str = "hungarian"
    background_alpha: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "DetectionVizConfig":
        try:
            gt_dir = Path(str(payload["gt_dir"]))
            pred_dir = Path(str(payload["pred_dir"]))
            correct_dir = Path(str(payload["output_correct_dir"]))
            error_dir = Path(str(payload["output_error_dir"]))
        except KeyError as exc:  # noqa: PERF203
            raise ValidationError(f"配置项 {exc.args[0]} 缺失") from exc

        images_dir = (
            Path(str(payload["images_dir"])) if payload.get("images_dir") else None
        )
        return cls(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_correct_dir=correct_dir,
            output_error_dir=error_dir,
            images_dir=images_dir,
            iou_threshold=float(payload.get("iou_threshold", 0.5)),
            stats_mode=bool(payload.get("stats_mode", False)),
            max_workers=int(payload.get("max_workers", 4)),
            match_method=str(payload.get("match_method", "hungarian")),
            background_alpha=float(payload.get("background_alpha", 0.0)),
        )


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValidationError("图片读取失败", details={"path": str(path)})
    return image


def _top_left(points: Iterable[Iterable[float]]) -> tuple[int, int]:
    xs, ys = zip(*((float(x), float(y)) for x, y in points))
    return int(min(xs)), int(min(ys))


def _iter_points(bbox: SequenceLike) -> list[tuple[float, float]]:
    if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
        x1, y1, x2, y2 = map(float, bbox)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if len(bbox) == 2 and isinstance(bbox[0], (list, tuple)):
        (x1, y1), (x2, y2) = bbox  # type: ignore[misc]
        return [
            (float(x1), float(y1)),
            (float(x2), float(y1)),
            (float(x2), float(y2)),
            (float(x1), float(y2)),
        ]
    if isinstance(bbox[0], (list, tuple)):
        return [(float(x), float(y)) for x, y in bbox]  # type: ignore[return-value]
    flat = list(map(float, bbox))
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def _draw_geometry(
    image: np.ndarray, bbox: SequenceLike, color: tuple[int, int, int], thickness: int
) -> None:
    points = _iter_points(bbox)
    if len(points) == 4 and all(
        points[i][0] == points[(i + 1) % 4][0] or points[i][1] == points[(i + 1) % 4][1]
        for i in range(4)
    ):
        x_vals = [int(p[0]) for p in points]
        y_vals = [int(p[1]) for p in points]
        cv2.rectangle(
            image,
            (min(x_vals), min(y_vals)),
            (max(x_vals), max(y_vals)),
            color,
            thickness,
        )
    else:
        pts_int = np.array([(int(x), int(y)) for x, y in points])
        cv2.polylines(image, [pts_int], isClosed=True, color=color, thickness=thickness)


def _put_text_no_overlap(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    *,
    occupied: list[tuple[int, int, int, int]],
    background_alpha: float,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.8,
    thickness: int = 2,
    y_step: int = 20,
) -> None:
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position

    def intersects(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    while any(
        intersects((x, y - height, width, height + baseline), occ) for occ in occupied
    ):
        y += y_step

    if background_alpha > 0:
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (x, y - height - baseline),
            (x + width, y + baseline),
            color=(0, 0, 0),
            thickness=-1,
        )
        cv2.addWeighted(
            overlay, background_alpha, image, 1 - background_alpha, 0, image
        )

    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    occupied.append((x, y - height, width, height + baseline))


def _find_image_file(stem: str, config: DetectionVizConfig) -> Optional[Path]:
    candidates: list[Path] = []
    if config.images_dir:
        candidates.append(config.images_dir)
    gt_dir = config.gt_dir
    candidates.extend(
        [
            gt_dir,
            gt_dir.parent,
            gt_dir.parent / "images",
            gt_dir.parent / "imgs",
        ]
    )

    for directory in candidates:
        if not directory:
            continue
        for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            path = directory / f"{stem}{suffix}"
            if path.exists():
                return path
    return None


def _collect_label_colors(
    files: Iterable[Path], config: DetectionVizConfig
) -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {}
    labels: set[str] = set()
    for file in files:
        try:
            labels.update(box.label for box in parse_labelme_shapes(file))
        except ValidationError as exc:
            logger.warning("读取标签失败 %s: %s", file.name, exc)
        pred_path = config.pred_dir / file.name
        if pred_path.exists():
            try:
                labels.update(box.label for box in parse_labelme_shapes(pred_path))
            except ValidationError as exc:
                logger.warning("读取预测标签失败 %s: %s", file.name, exc)
    palette = [
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
    for idx, label in enumerate(sorted(labels)):
        colors[label] = palette[idx % len(palette)]
    return colors


class DetectionVisualizer:
    """负责绘制目标检测预测结果。"""

    def __init__(self, config: DetectionVizConfig) -> None:
        self.config = config
        self.logger = logger

    def visualize_directory(self) -> None:
        if not self.config.pred_dir.exists():
            raise ValidationError(
                "预测结果目录不存在", details={"path": str(self.config.pred_dir)}
            )
        if not self.config.gt_dir.exists():
            raise ValidationError(
                "真实标注目录不存在", details={"path": str(self.config.gt_dir)}
            )

        gt_files = sorted(self.config.gt_dir.glob("*.json"))
        if not gt_files:
            self.logger.warning("真实标注目录中没有 JSON 文件: %s", self.config.gt_dir)
            return

        colors = None
        if not self.config.stats_mode:
            colors = _collect_label_colors(gt_files, self.config)
            self.logger.info("共收集 %s 个标签", len(colors))

        self.config.output_correct_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_error_dir.mkdir(parents=True, exist_ok=True)

        total = len(gt_files)
        success = failed = skipped = 0
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_single, file, colors): file
                for file in gt_files
            }
            for future in safe_tqdm(
                as_completed(futures), total=total, desc="可视化进度"
            ):
                file = futures[future]
                try:
                    result = future.result()
                    if result is True:
                        success += 1
                    elif result is False:
                        failed += 1
                    else:
                        skipped += 1
                except Exception as exc:  # noqa: BLE001
                    self.logger.error("处理 %s 时发生错误: %s", file.name, exc)
                    skipped += 1

        self.logger.info(
            "可视化完成: 成功 %s, 失败 %s, 跳过 %s", success, failed, skipped
        )
        if success or failed:
            self.logger.info("正确输出目录: %s", self.config.output_correct_dir)
            self.logger.info("错误输出目录: %s", self.config.output_error_dir)

    def _process_single(
        self,
        gt_file: Path,
        colors: Optional[dict[str, tuple[int, int, int]]],
    ) -> Optional[bool]:
        stem = gt_file.stem
        pred_file = self.config.pred_dir / gt_file.name
        if not pred_file.exists():
            self.logger.warning("缺少预测文件: %s", pred_file)
            return None

        image_path = _find_image_file(stem, self.config)
        if image_path is None:
            self.logger.warning("未找到图片文件: %s", stem)
            return None

        try:
            gt_boxes = parse_labelme_shapes(gt_file)
            pred_boxes = parse_labelme_shapes(pred_file)
        except ValidationError as exc:
            self.logger.warning("解析 JSON 失败 %s: %s", gt_file.name, exc)
            return None

        match_result = assign_boxes(
            gt_boxes,
            pred_boxes,
            iou_threshold=self.config.iou_threshold,
            use_label=not self.config.stats_mode,
            method=self.config.match_method,
        )

        output_dir = (
            self.config.output_correct_dir
            if not match_result.unmatched_gt and not match_result.unmatched_pred
            else self.config.output_error_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{stem}.jpg"

        if self.config.stats_mode:
            self._draw_with_stats(
                image_path, gt_boxes, pred_boxes, match_result, output_path
            )
        else:
            self._draw_with_labels(
                image_path,
                pred_boxes,
                match_result,
                output_path,
                colors or {},
            )

        return output_dir is self.config.output_correct_dir

    def _draw_with_labels(
        self,
        image_path: Path,
        predictions: list[BoundingBox],
        match_result: MatchResult,
        output_path: Path,
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        image = _load_image(image_path)
        matched_pred = {pi for _, pi in match_result.matches}
        occupied: list[tuple[int, int, int, int]] = []
        for index, box in enumerate(predictions):
            is_tp = index in matched_pred
            color = colors.get(box.label, (0, 255, 0) if is_tp else (255, 0, 0))
            thickness = 3 if is_tp else 1
            _draw_geometry(image, box.bbox, color, thickness)
            top_left = _top_left(_iter_points(box.bbox))
            _put_text_no_overlap(
                image,
                box.label,
                (max(5, top_left[0]), max(20, top_left[1] - 10)),
                color,
                occupied=occupied,
                background_alpha=self.config.background_alpha,
            )
        cv2.imwrite(str(output_path), image)

    def _draw_with_stats(
        self,
        image_path: Path,
        gt_boxes: list[BoundingBox],
        pred_boxes: list[BoundingBox],
        match_result: MatchResult,
        output_path: Path,
    ) -> None:
        image = _load_image(image_path)
        tp_color = (0, 255, 0)
        fp_color = (0, 0, 255)
        fn_color = (0, 255, 255)
        matched_gt = {gi for gi, _ in match_result.matches}
        matched_pred = {pi for _, pi in match_result.matches}

        for gi, _ in match_result.matches:
            _draw_geometry(image, gt_boxes[gi].bbox, tp_color, 3)
        for index, box in enumerate(gt_boxes):
            if index not in matched_gt:
                _draw_geometry(image, box.bbox, fn_color, 3)
        for index, box in enumerate(pred_boxes):
            if index not in matched_pred:
                _draw_geometry(image, box.bbox, fp_color, 3)

        tp = match_result.true_positives()
        fn = match_result.false_negatives()
        fp = match_result.false_positives()
        font = cv2.FONT_HERSHEY_SIMPLEX
        x0, y0, gap = 15, 45, 55
        cv2.putText(image, f"TP: {tp}", (x0, y0), font, 1.2, tp_color, 3)
        cv2.putText(image, f"FP: {fp}", (x0, y0 + gap), font, 1.2, fp_color, 3)
        cv2.putText(image, f"FN: {fn}", (x0, y0 + 2 * gap), font, 1.2, fn_color, 3)
        cv2.imwrite(str(output_path), image)
