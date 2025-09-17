import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from .matcher import assign_boxes


def _setup_tqdm():
    try:
        from tqdm import tqdm

        is_exe = getattr(sys, "frozen", False)
        if is_exe:
            # exe环境下禁用tqdm输出
            def exe_tqdm(iterable, **kwargs):
                try:
                    return tqdm(iterable, disable=True, **kwargs)
                except:
                    return iterable

            return exe_tqdm
        else:
            return tqdm
    except ImportError:
        # 如果没有tqdm，使用简单的迭代器
        def fallback_tqdm(iterable, **kwargs):
            return iterable

        return fallback_tqdm


safe_tqdm = _setup_tqdm()


GT_FOLDER = ""  # 真实标注（LabelMe JSON）目录
PRED_FOLDER = ""  # 预测结果（LabelMe JSON）目录
IMAGE_FOLDER = ""  # 若图片与 JSON 不同目录，指定图片目录
CORRECT_DIR = ""
ERROR_DIR = ""

IOU_THRESHOLD = 0.5  # IoU 阈值
STATS_MODE = True  # True: 统计视图   False: 仅预测框视图
MAX_WORKERS = 1  # 默认串行处理

BG_ALPHA = 0.0  # 底色透明度[0.0-1.0]


def get_fixed_colors(labels):
    """
    获取固定颜色
    :param labels: 标签列表
    :return: 颜色字典
    """
    vivid = [
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
    return {lb: vivid[i % len(vivid)] for i, lb in enumerate(sorted(labels))}


def _put_text_no_overlap(
    img,
    text,
    x,
    y,
    color,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.8,
    thickness=2,
    occupied=None,
    y_step=20,
):
    """
    绘制文本，避免重叠
    :param img: 图像
    :param text: 文本
    :param x: x坐标
    :param y: y坐标
    """
    if occupied is None:
        occupied = []
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)

    def intersects(x1, y1, w1, h1, x2, y2, w2, h2):
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    y_curr = y
    while any(
        intersects(x, y_curr - h, w, h + baseline, ox, oy, ow, oh)
        for ox, oy, ow, oh in occupied
    ):
        y_curr += y_step

    if BG_ALPHA > 0:
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (x, y_curr - h - baseline),
            (x + w, y_curr + baseline),
            color=(0, 0, 0),
            thickness=-1,
        )
        cv2.addWeighted(overlay, BG_ALPHA, img, 1 - BG_ALPHA, 0, img)

    cv2.putText(img, text, (x, y_curr), font, scale, color, thickness, cv2.LINE_AA)

    occupied.append((x, y_curr - h, w, h + baseline))
    return occupied


def json_to_objs(json_file):
    """
    将 JSON 文件转换为对象列表
    :param json_file: JSON 文件路径
    :return: 对象列表
    """
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    objs = []
    for shp in data.get("shapes", []):
        label = shp["label"]
        pts = shp["points"]
        stype = shp.get("shape_type", "polygon")

        if stype == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            bbox = [x1, y1, x2, y2]
        else:
            bbox = pts
        objs.append({"label": label, "bbox": bbox})
    return objs


def find_image_file(base):
    """
    查找图片文件
    :param base: 文件名
    :return: 图片文件路径
    """
    # 如果IMAGE_FOLDER已设置，优先在该目录查找
    if IMAGE_FOLDER and os.path.exists(IMAGE_FOLDER):
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            cand = os.path.join(IMAGE_FOLDER, base + ext)
            if os.path.exists(cand):
                logging.debug(f"找到图片文件: {cand}")
                return cand

    # 如果IMAGE_FOLDER为空或未找到，在GT_FOLDER同目录查找
    if GT_FOLDER and os.path.exists(GT_FOLDER):
        search_dirs = [
            GT_FOLDER,  # GT目录本身
            os.path.dirname(GT_FOLDER),  # GT目录的父目录
            os.path.join(
                os.path.dirname(GT_FOLDER), "images"
            ),  # 父目录下的images文件夹
            os.path.join(os.path.dirname(GT_FOLDER), "imgs"),  # 父目录下的imgs文件夹
        ]

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                    cand = os.path.join(search_dir, base + ext)
                    if os.path.exists(cand):
                        logging.debug(f"找到图片文件: {cand}")
                        return cand

    # 最后在当前目录查找
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        cand = base + ext
        if os.path.exists(cand):
            logging.debug(f"找到图片文件: {cand}")
            return cand

    logging.warning(f"未找到图片文件: {base}")
    return None


def _top_left(pts):
    """
    获取左上角坐标
    :param pts: 顶点列表
    :return: 左上角坐标
    """
    if len(pts) == 4 and isinstance(pts[0], (int, float)):
        x1, y1 = pts[0], pts[1]
    else:
        if len(pts) == 2 and isinstance(pts[0], (list, tuple)):
            xs, ys = zip(*pts)
        else:
            xs, ys = zip(
                *(
                    pts
                    if isinstance(pts[0], (list, tuple))
                    else [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
                )
            )
        x1, y1 = min(xs), min(ys)
    return int(x1), int(y1)


def _draw_geom(img, pts, color, thick):
    """
    绘制几何图形
    :param img: 图像
    :param pts: 顶点列表
    :param color: 颜色
    :param thick: 厚度
    """
    if len(pts) == 4 and isinstance(pts[0], (int, float)):
        x1, y1, x2, y2 = map(int, pts)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    elif len(pts) == 2 and isinstance(pts[0], (list, tuple)):
        (x1, y1), (x2, y2) = pts
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
    else:
        pts_int = np.array(
            [
                (int(x), int(y))
                for x, y in (
                    pts
                    if isinstance(pts[0], (list, tuple))
                    else [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
                )
            ]
        )
        cv2.polylines(img, [pts_int], isClosed=True, color=color, thickness=thick)


def draw_boxes_label_color(img_path, gt, pred, matches, save_path, colors):
    """
    绘制预测框和标签
    :param img_path: 图片路径
    :param gt: 真实标注
    :param pred: 预测结果
    :param matches: 匹配结果
    :param save_path: 保存路径
    :param colors: 颜色
    """
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"无法读取图片: {img_path}")
        return

    matched_pred = {pi for _, pi in matches}
    occupied = []

    for i, p in enumerate(pred):
        is_tp = i in matched_pred
        color = colors.get(p["label"], (0, 255, 0) if is_tp else (255, 0, 0))
        thick = 3 if is_tp else 1
        _draw_geom(img, p["bbox"], color, thick)

        x, y = _top_left(p["bbox"])
        occupied = _put_text_no_overlap(
            img, p["label"], x, max(20, y - 10), color, occupied=occupied
        )

    cv2.imwrite(save_path, img)


def draw_boxes_with_stats(img_path, gt, pred, matches, save_path):
    """
    绘制预测框和标签
    :param img_path: 图片路径
    :param gt: 真实标注
    :param pred: 预测结果
    :param matches: 匹配结果
    :param save_path: 保存路径
    """
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"无法读取图片: {img_path}")
        return

    TP_COLOR, FP_COLOR, FN_COLOR = (0, 255, 0), (0, 0, 255), (0, 255, 255)
    matched_gt = {gi for gi, _ in matches}
    matched_pred = {pi for _, pi in matches}

    # 绘框
    for gi, _ in matches:
        _draw_geom(img, gt[gi]["bbox"], TP_COLOR, 3)  # TP
    for i, g in enumerate(gt):
        if i in matched_gt:
            continue
        _draw_geom(img, g["bbox"], FN_COLOR, 3)  # FN
    for i, p in enumerate(pred):
        if i in matched_pred:
            continue
        _draw_geom(img, p["bbox"], FP_COLOR, 3)  # FP

    # 数字统计
    tp, fn, fp = len(matches), len(gt) - len(matches), len(pred) - len(matches)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x0, y0, gap = 15, 45, 55
    cv2.putText(img, f"TP: {tp}", (x0, y0), font, 1.2, TP_COLOR, 3)
    cv2.putText(img, f"FP: {fp}", (x0, y0 + gap), font, 1.2, FP_COLOR, 3)
    cv2.putText(img, f"FN: {fn}", (x0, y0 + 2 * gap), font, 1.2, FN_COLOR, 3)

    cv2.imwrite(save_path, img)


def process_single(fname, stats_mode, colors):
    """
    处理单个文件
    :param fname: 文件名
    :param stats_mode: 统计模式
    :param colors: 颜色
    :return: 是否成功
    """
    base = os.path.splitext(fname)[0]

    # 构建文件路径
    gt_path = os.path.join(GT_FOLDER, fname)
    pred_path = os.path.join(PRED_FOLDER, fname)

    # 检查JSON文件是否存在
    missing_files = []
    if not os.path.exists(gt_path):
        missing_files.append(f"GT文件: {gt_path}")
    if not os.path.exists(pred_path):
        missing_files.append(f"预测文件: {pred_path}")

    # 查找图片文件
    img_path = find_image_file(base)
    if not img_path:
        missing_files.append(f"图片文件: {base}.*")

    # 如果有文件缺失，记录详细信息并跳过
    if missing_files:
        logging.warning(f"处理 {base} 时缺失文件:")
        for missing in missing_files:
            logging.warning(f"  - {missing}")
        return None

    try:
        # 读取JSON文件
        gt = json_to_objs(gt_path)
        pred = json_to_objs(pred_path)

        # 执行匹配
        res = assign_boxes(gt, pred, iou_thr=IOU_THRESHOLD, method="hungarian")
        matches = res["matches"]

        # 判断是否完全匹配
        all_matched = not res["unmatched_gt"] and not res["unmatched_pred"]
        out_dir = CORRECT_DIR if all_matched else ERROR_DIR

        # 确保输出目录存在
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, base + ".jpg")

        # 执行绘制
        if stats_mode:
            draw_boxes_with_stats(img_path, gt, pred, matches, out_path)
        else:
            draw_boxes_label_color(img_path, gt, pred, matches, out_path, colors)

        logging.debug(f"成功处理: {base} -> {out_path}")
        return all_matched

    except Exception as e:
        logging.error(f"处理 {base} 时发生错误: {e}")
        return None


def visualize_all(gt_folder, pred_folder, iou_thr=0.5, stats_mode=False, max_workers=8):
    """
    可视化所有文件
    :param gt_folder: 真实标注目录
    :param pred_folder: 预测结果目录
    :param iou_thr: IoU 阈值
    :param stats_mode: 统计模式
    :param max_workers: 最大工作线程数
    """
    global GT_FOLDER, PRED_FOLDER, IOU_THRESHOLD
    GT_FOLDER, PRED_FOLDER, IOU_THRESHOLD = gt_folder, pred_folder, iou_thr

    # 检查输入目录
    if not os.path.exists(pred_folder):
        logging.error(f"预测结果目录不存在: {pred_folder}")
        return
    if not os.path.exists(gt_folder):
        logging.error(f"真实标注目录不存在: {gt_folder}")
        return

    # 获取所有JSON文件
    fns = [f for f in os.listdir(pred_folder) if f.endswith(".json")]
    if not fns:
        logging.warning(f"预测结果目录中没有JSON文件: {pred_folder}")
        return

    logging.info(f"找到 {len(fns)} 个预测文件")

    # 非统计模式先收集类别配色
    colors = None
    if not stats_mode:
        labels = set()
        for f in fns:
            gt_p, pred_p = os.path.join(gt_folder, f), os.path.join(pred_folder, f)
            try:
                if os.path.exists(gt_p):
                    labels.update(o["label"] for o in json_to_objs(gt_p))
                if os.path.exists(pred_p):
                    labels.update(o["label"] for o in json_to_objs(pred_p))
            except Exception as e:
                logging.warning(f"收集标签时出错 {f}: {e}")
        colors = get_fixed_colors(labels)
        logging.info(f"收集到 {len(labels)} 个类别: {sorted(labels)}")

    # 确保输出目录存在
    if not CORRECT_DIR or not ERROR_DIR:
        logging.error("输出目录未设置")
        return

    os.makedirs(CORRECT_DIR, exist_ok=True)
    os.makedirs(ERROR_DIR, exist_ok=True)

    # 并行处理
    ok, err, skip = 0, 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_single, f, stats_mode, colors): f for f in fns}
        for fut in safe_tqdm(
            as_completed(futures), total=len(futures), desc="可视化进度", ncols=100
        ):
            try:
                res = fut.result()
                if res is True:
                    ok += 1
                elif res is False:
                    err += 1
                else:  # res is None
                    skip += 1
            except Exception as e:
                logging.error(f"处理文件时出错: {e}")
                skip += 1

    logging.info(f"可视化完成！正确: {ok}, 错误: {err}, 跳过: {skip}")
    if ok > 0 or err > 0:
        logging.info("结果已保存至:")
        logging.info(f"  正确预测: {CORRECT_DIR}")
        logging.info(f"  错误预测: {ERROR_DIR}")
    else:
        logging.warning("没有成功处理任何文件，请检查文件路径和格式")


if __name__ == "__main__":
    visualize_all(
        GT_FOLDER,
        PRED_FOLDER,
        iou_thr=IOU_THRESHOLD,
        stats_mode=STATS_MODE,
        max_workers=MAX_WORKERS,
    )
