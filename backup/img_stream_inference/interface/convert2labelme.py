import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import atan2
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from shapely.geometry import Polygon

RAW_JSON_DIR = ""  # 推理 *.json 文件所在目录
IMAGE_DIR = ""  # 原图所在目录（文件名需与 JSON 同 stem）
LABELME_OUTPUT_DIR = ""  # 输出的 LabelMe JSON 目录
MAX_WORKERS = 1  # 默认串行处理

_log_lock = threading.Lock()


def record_fail(msg: str):
    """
    追加写 fail.log，一行一条，带时间戳
    :param msg: 失败信息
    :return: None
    """
    with _log_lock:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("fail.log", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")


_num_re = re.compile(r"[-+]?\d*\.?\d+")


def parse_coord(coord: str) -> List[List[float]]:
    """
    解析形如 "x,y x,y ..." 的顶点串
    :param coord: 顶点串
    :return: 顶点列表
    """
    nums = list(map(float, _num_re.findall(coord)))
    return [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]


def is_axis_rect(pts: List[List[float]], tol: float = 1e-6) -> bool:
    """
    使用 Shapely 判断 4 点是否为 **轴对齐矩形**
    原理：多边形面积≈外接矩形面积，且几何形状几乎重合
    :param pts: 顶点列表
    :param tol: 浮点误差容忍
    :return: 是否为轴对齐矩形
    """
    if len(pts) != 4:
        return False
    poly = Polygon(pts)
    # 非法或面积过小直接否掉
    if not poly.is_valid or poly.area < tol:
        return False
    env = poly.envelope
    return abs(poly.area - env.area) <= tol and poly.equals_exact(env, tol)


def sort_clockwise(pts: List[List[float]]) -> List[List[float]]:
    """
    按质心为参考将顶点顺时针排序，避免 LabelMe 中乱线条
    :param pts: 顶点列表
    :return: 顺时针排序后的顶点列表
    """
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: atan2(p[1] - cy, p[0] - cx))


def lm_template(img_name: str, h: int, w: int) -> Dict:
    """返回符合 LabelMe 5.x 版本的空模板"""
    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": img_name,
        "imageHeight": h,
        "imageWidth": w,
        "imageData": None,
    }


def save_labelme(img_path: str, resp: Dict, out_dir: str):
    """
    根据一张图片和对应推理 JSON（resp），写出 LabelMe JSON
    :param img_path: 图片路径
    :param resp: 推理 JSON
    :param out_dir: 输出目录
    :return: None
    """
    img = cv2.imread(img_path)
    if img is None:
        record_fail(f"图片读取失败: {img_path}")
        return

    h, w = img.shape[:2]
    lm = lm_template(os.path.basename(img_path), h, w)

    # 遍历推理结果 - 只支持新格式
    # 新格式：{"alamr": 1, "infer_id": "xxx", "result": [...]}
    result_list = resp.get("result", [])

    for obj in result_list:
        raw = (obj.get("coord") or "").strip()
        if not raw:
            continue
        pts = parse_coord(raw)

        if len(pts) < 4:
            record_fail(f"点数 <4: {raw} | {img_path}")
            continue

        label = obj.get("name", "unknown")
        conf = obj.get("confidence", "-")

        if len(pts) == 4 and is_axis_rect(pts):
            # 两点式 rectangle
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
        lm["flags"]["no_objects"] = True
        record_fail(f"无标注: {img_path}")

    os.makedirs(out_dir, exist_ok=True)
    dst_json = os.path.join(out_dir, Path(img_path).stem + ".json")
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False, indent=2)
    logging.info(f"[OK] {dst_json}")


def build_img_index(img_dir: str) -> Dict[str, str]:
    """
    构建 {stem: full_path} 字典
    :param img_dir: 图片目录
    :return: {stem: full_path} 字典
    """
    return {
        p.stem: str(p)
        for p in Path(img_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def handle_one(raw_json: str, img_idx: Dict[str, str], out_dir: str):
    """
    处理单个推理 JSON → 调用 save_labelme
    :param raw_json: 推理 JSON 文件名
    :param img_idx: {stem: full_path} 字典
    :param out_dir: 输出目录
    :return: None
    """
    stem = Path(raw_json).stem
    img_path = img_idx.get(stem)
    if not img_path:
        record_fail(f"找不到图片: {stem}")
        return
    try:
        with open(raw_json, "r", encoding="utf-8") as f:
            resp = json.load(f)
    except Exception as e:
        record_fail(f"JSON 解析失败: {raw_json}: {e}")
        return
    save_labelme(img_path, resp, out_dir)


def batch_convert(
    raw_dir: Optional[str] = None,
    img_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
):
    """
    批量遍历 raw_dir 下 *.json，使用线程池并行转换
    :param raw_dir: 推理 JSON 目录
    :param img_dir: 图片目录
    :param out_dir: 输出目录
    :return: None
    """
    if raw_dir is None:
        raw_dir = RAW_JSON_DIR
    if img_dir is None:
        img_dir = IMAGE_DIR
    if out_dir is None:
        out_dir = LABELME_OUTPUT_DIR

    files = [str(p) for p in Path(raw_dir).glob("*.json")]
    if not files:
        logging.warning("raw_json 目录为空")
        return

    img_idx = build_img_index(img_dir)
    if not img_idx:
        logging.warning("image 目录无图片")
        return

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = [ex.submit(handle_one, p, img_idx, out_dir) for p in files]
        for fut in as_completed(futs):
            fut.result()  # 抛出异常便于调试


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        img_idx = build_img_index(IMAGE_DIR)
        handle_one(sys.argv[1], img_idx, LABELME_OUTPUT_DIR)

    else:
        batch_convert()
