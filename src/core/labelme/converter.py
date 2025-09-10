from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import atan2
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from shapely.geometry import Polygon

from utils.logger import get_logger, record_fail

log = get_logger("labelme.converter")
_log_lock = threading.Lock()


@dataclass
class ConvertConfig:
    raw_json_dir: str  # 原始响应目录（HTTPS 或 gRPC）
    image_dir: str  # 原图目录（文件名需与 JSON 同 stem）
    out_dir: str  # LabelMe JSON 输出目录
    max_workers: int = 1  # 1=串行；>1=并行
    rect_tol: float = 1e-6  # 轴对齐矩形判定容差
    allow_empty: bool = True  # 没有shape时仍写LabelMe（并在flags标记）


_num_re = re.compile(r"[-+]?\d*\.?\d+")


def parse_coord(coord: str) -> List[List[float]]:
    """解析 'x,y x,y ...' 顶点串 -> [[x,y], ...]"""
    nums = list(map(float, _num_re.findall(coord)))
    return [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]


def is_axis_rect(pts: List[List[float]], tol: float = 1e-6) -> bool:
    """
    使用 Shapely 判断 4点是否为轴对齐矩形：
    面积≈外接矩形面积，且几何形状几乎重合
    """
    if len(pts) != 4:
        return False
    poly = Polygon(pts)
    if not poly.is_valid or poly.area < tol:
        return False
    env = poly.envelope
    return abs(poly.area - env.area) <= tol and poly.equals_exact(env, tol)


def sort_clockwise(pts: List[List[float]]) -> List[List[float]]:
    """按质心顺时针排序"""
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: atan2(p[1] - cy, p[0] - cx))


def lm_template(img_name: str, h: int, w: int) -> Dict:
    """LabelMe 5.x 模板"""
    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": img_name,
        "imageHeight": h,
        "imageWidth": w,
        "imageData": None,
    }


def _safe_read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        record_fail(f"[JSON_PARSE] {path.name} 解析失败 | {e}")
        raise


def _extract_objects(resp: dict) -> List[dict]:
    """
    只支持“新响应结构”：
      {"alarm": 1, "infer_id": "...", "result": [
          {"name": "...", "coord": "x,y x,y ...", "confidence": 0.xx, ...}, ...
      ]}
    """
    result = resp.get("result", [])
    if not isinstance(result, list):
        return []
    return result


def _to_labelme(
    img_path: Path, objs: List[dict], rect_tol: float, allow_empty: bool
) -> Dict:
    img = cv2.imread(str(img_path))
    if img is None:
        record_fail(f"[IMG_READ] 读取失败: {img_path}")
        raise ValueError(f"无法读取图片: {img_path}")
    h, w = img.shape[:2]

    lm = lm_template(img_path.name, h, w)
    for obj in objs:
        raw = str(obj.get("coord", "")).strip()
        if not raw:
            continue
        pts = parse_coord(raw)
        if len(pts) < 4:
            record_fail(f"[COORD] 点数<4: {img_path.name} | {raw}")
            continue

        label = obj.get("name", "unknown")
        conf = obj.get("confidence", "-")

        if len(pts) == 4 and is_axis_rect(pts, tol=rect_tol):
            xs, ys = zip(*pts)
            points, stype = [[min(xs), min(ys)], [max(xs), max(ys)]], "rectangle"
        else:
            points, stype = sort_clockwise(pts), "polygon"

        lm["shapes"].append(
            dict(
                label=label,
                points=points,
                group_id=None,
                shape_type=stype,
                flags={},
                description=f"conf={conf}",
            )
        )

    if not lm["shapes"]:
        if allow_empty:
            lm["flags"]["no_objects"] = True
        else:
            # 不允许空 -> 抛错以便外层计入失败
            raise ValueError("无可转换的对象")
    return lm


def save_labelme(lm: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False, indent=2)


def build_image_index(img_dir: str | os.PathLike) -> Dict[str, str]:
    """构建 {stem: full_path}"""
    mapping: Dict[str, str] = {}
    p = Path(img_dir)
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"):
        for fp in p.glob(f"*{ext}"):
            mapping[fp.stem] = str(fp)
    return mapping


def convert_one_file(
    json_path: str | os.PathLike,
    image_index: Dict[str, str],
    out_dir: str | os.PathLike,
    rect_tol: float = 1e-6,
    allow_empty: bool = True,
) -> bool:
    """单文件转换：raw.json -> labelme.json"""
    jp = Path(json_path)
    stem = jp.stem
    img_path = image_index.get(stem)
    if not img_path:
        record_fail(f"[MISS_IMG] 找不到图片: {stem}")
        return False

    try:
        resp = _safe_read_json(jp)
        objs = _extract_objects(resp)
        lm = _to_labelme(Path(img_path), objs, rect_tol, allow_empty)
        save_labelme(lm, Path(out_dir) / f"{stem}.json")
        log.info("OK  -> %s", stem)
        return True
    except Exception as e:
        record_fail(f"[CONVERT_ERR] {jp.name} | {e}")
        log.exception("转换失败: %s | %s", jp.name, e)
        return False


def batch_convert_dir(cfg: ConvertConfig) -> Tuple[int, int]:
    """
    批量遍历 raw_json_dir 下 *.json，转换为 LabelMe 到 out_dir。
    返回 (success, failed)
    """
    raw_dir = Path(cfg.raw_json_dir)
    out_dir = Path(cfg.out_dir)
    files = sorted([p for p in raw_dir.glob("*.json")])

    if not files:
        log.warning("raw_json 目录为空: %s", raw_dir)
        return 0, 0

    img_idx = build_image_index(cfg.image_dir)
    if not img_idx:
        log.warning("image 目录无图片: %s", cfg.image_dir)
        return 0, len(files)

    ok = fail = 0
    if cfg.max_workers == 1:
        log.info("使用串行模式, 共 %d 个", len(files))
        for p in files:
            ok += (
                1
                if convert_one_file(
                    p,
                    img_idx,
                    out_dir,
                    rect_tol=cfg.rect_tol,
                    allow_empty=cfg.allow_empty,
                )
                else 0
            )
        fail = len(files) - ok
    else:
        log.info("使用并行模式（%d 线程）, 共 %d 个", cfg.max_workers, len(files))
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futs = {
                ex.submit(
                    convert_one_file,
                    p,
                    img_idx,
                    out_dir,
                    cfg.rect_tol,
                    cfg.allow_empty,
                ): p
                for p in files
            }
            for fut in as_completed(futs):
                ok += 1 if fut.result() else 0
        fail = len(files) - ok

    log.info("转换完成：成功 %d，失败 %d", ok, fail)
    return ok, fail
