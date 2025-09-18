"""LabelMe 转换工具。"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import atan2
from pathlib import Path
from typing import Mapping, Optional

import cv2
from shapely.geometry import Polygon

from utils.exceptions import ExternalServiceError, ValidationError
from utils.logger import get_logger

__all__ = [
    "LabelmeConverterConfig",
    "LabelmeConverter",
    "build_image_index",
    "convert_file",
]

logger = get_logger(__name__)

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
_FAIL_LOG_PATH = Path("fail.log")
_LOG_LOCK = threading.Lock()


@dataclass(slots=True)
class LabelmeConverterConfig:
    """LabelMe 转换所需路径及并发配置。"""

    raw_json_dir: Path
    image_dir: Path
    output_dir: Path
    max_workers: int = 1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "LabelmeConverterConfig":
        try:
            raw_dir = Path(str(payload["raw_json_dir"]))
            img_dir = Path(str(payload["image_dir"]))
            out_dir = Path(str(payload["output_dir"]))
        except KeyError as exc:  # noqa: PERF203
            raise ValidationError(f"配置项 {exc.args[0]} 缺失") from exc

        max_workers = int(payload.get("max_workers", 1))
        return cls(raw_dir, img_dir, out_dir, max_workers=max_workers)


def _record_fail(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _LOG_LOCK:
        _FAIL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _FAIL_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")


def build_image_index(image_dir: Path) -> dict[str, Path]:
    if not image_dir.exists():
        raise ValidationError("图片目录不存在", details={"path": str(image_dir)})
    return {
        path.stem: path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def _parse_coord(coord: str) -> list[list[float]]:
    nums = list(map(float, _NUM_RE.findall(coord)))
    return [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]


def _is_axis_rect(points: list[list[float]], tol: float = 1e-6) -> bool:
    if len(points) != 4:
        return False
    poly = Polygon(points)
    if not poly.is_valid or poly.area < tol:
        return False
    env = poly.envelope
    return abs(poly.area - env.area) <= tol and poly.equals_exact(env, tol)


def _sort_clockwise(points: list[list[float]]) -> list[list[float]]:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: atan2(p[1] - cy, p[0] - cx))


def _labelme_template(image_name: str, height: int, width: int) -> dict[str, object]:
    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_name,
        "imageHeight": height,
        "imageWidth": width,
        "imageData": None,
    }


def _load_image(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        _record_fail(f"图片读取失败: {path}")
        raise ExternalServiceError(
            "图片读取失败", details={"path": str(path)}, service="cv2"
        )
    height, width = image.shape[:2]
    return height, width


def _convert_shapes(
    response: Mapping[str, object], image_path: Path
) -> list[dict[str, object]]:
    shapes: list[dict[str, object]] = []
    result = response.get("result")
    if not isinstance(result, list):
        return shapes

    for entry in result:
        if not isinstance(entry, Mapping):
            continue
        raw_coord = str(entry.get("coord", "")).strip()
        if not raw_coord:
            continue
        points = _parse_coord(raw_coord)
        if len(points) < 4:
            _record_fail(f"点数 <4: {raw_coord} | {image_path}")
            continue

        label = str(entry.get("name", "unknown"))
        confidence = entry.get("confidence", "-")

        if len(points) == 4 and _is_axis_rect(points):
            xs, ys = zip(*points)
            shape_points = [[min(xs), min(ys)], [max(xs), max(ys)]]
            shape_type = "rectangle"
        else:
            shape_points = _sort_clockwise(points)
            shape_type = "polygon"

        shapes.append(
            {
                "label": label,
                "points": shape_points,
                "group_id": None,
                "shape_type": shape_type,
                "flags": {},
                "description": f"conf={confidence}",
            }
        )
    return shapes


def convert_file(
    raw_json_path: str | Path,
    *,
    image_index: Mapping[str, Path],
    output_dir: Path,
) -> bool:
    raw_path = Path(raw_json_path)
    if not raw_path.exists():
        raise ValidationError("推理 JSON 不存在", details={"path": str(raw_path)})

    stem = raw_path.stem
    image_path = image_index.get(stem)
    if image_path is None:
        _record_fail(f"找不到图片: {stem}")
        return False

    try:
        with raw_path.open("r", encoding="utf-8") as handle:
            response = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        _record_fail(f"JSON 解析失败: {raw_path}: {exc}")
        return False

    try:
        height, width = _load_image(image_path)
    except ExternalServiceError:
        return False

    labelme = _labelme_template(image_path.name, height, width)
    shapes = _convert_shapes(response, image_path)
    if not shapes:
        labelme["flags"]["no_objects"] = True
        _record_fail(f"无标注: {image_path}")
    labelme["shapes"] = shapes

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{stem}.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(labelme, handle, ensure_ascii=False, indent=2)
    logger.info("LabelMe 已生成: %s", target)
    return True


class LabelmeConverter:
    """负责批量执行推理结果到 LabelMe 的转换。"""

    def __init__(self, config: LabelmeConverterConfig) -> None:
        self.config = config
        self.logger = logger

    def convert_single(
        self,
        raw_json_path: str | Path,
        *,
        image_index: Optional[Mapping[str, Path]] = None,
    ) -> bool:
        index = image_index or build_image_index(self.config.image_dir)
        return convert_file(
            raw_json_path, image_index=index, output_dir=self.config.output_dir
        )

    def convert_directory(self) -> None:
        raw_dir = self.config.raw_json_dir
        files = sorted(raw_dir.glob("*.json"))
        if not files:
            self.logger.warning("推理结果目录为空: %s", raw_dir)
            return

        index = build_image_index(self.config.image_dir)
        if not index:
            self.logger.warning("图片目录无可用文件: %s", self.config.image_dir)
            return

        self.logger.info("开始转换 %s 个推理结果", len(files))

        if self.config.max_workers == 1:
            success = failed = 0
            for raw_path in files:
                if convert_file(
                    raw_path, image_index=index, output_dir=self.config.output_dir
                ):
                    success += 1
                else:
                    failed += 1
            self.logger.info("转换完成: 成功 %s, 失败 %s", success, failed)
            return

        self.logger.info("使用并行模式，工作线程: %s", self.config.max_workers)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    convert_file,
                    raw_path,
                    image_index=index,
                    output_dir=self.config.output_dir,
                ): raw_path
                for raw_path in files
            }
            success = failed = 0
            for future in as_completed(futures):
                try:
                    if future.result():
                        success += 1
                    else:
                        failed += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    raw_path = futures[future]
                    _record_fail(f"处理 {raw_path.name} 失败: {exc}")
            self.logger.info("并行转换完成: 成功 %s, 失败 %s", success, failed)
