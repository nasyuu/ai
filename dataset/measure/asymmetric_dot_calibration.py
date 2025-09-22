# -*- coding: utf-8 -*-

import csv
import glob
import json
from pathlib import Path

import cv2
import numpy as np

# =====================【参数区 · 请补充/确认】=====================
PATTERN_SIZE = None
SPACING_VALUE_MM = None  # TODO: 圆心-圆心物理间距（mm），例如 10.0
IMAGE_GLOB = "./images/*.jpg"
OUT_DIR = "./_vis"
SAVE_JSON = "./_vis/telecentric_2d_calib_result.json"
SAVE_9PT_JSON = "./_vis/nine_point_result.json"
SAVE_9PT_CSV = "./_vis/nine_point_result.csv"
SAVE_PER_IMAGE_9PT = "./_vis/per_image_nine_points.json"

USE_CLAHE = False
INVERT_TRY = True
SUBPIX_WIN = (5, 5)
SUBPIX_TERM = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)


def create_blob_detector():
    p = cv2.SimpleBlobDetector_Params()
    p.filterByColor = False
    p.filterByArea = True
    p.minArea = 30
    p.maxArea = 1e8
    p.filterByCircularity = True
    p.minCircularity = 0.6
    p.filterByConvexity = False
    p.filterByInertia = False
    p.minThreshold = 10
    p.maxThreshold = 220
    p.thresholdStep = 10
    return cv2.SimpleBlobDetector_create(p)


def _preprocess(gray):
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def _infer_pattern_size_from_images(image_glob, detector, invert_try):
    paths = sorted(glob.glob(image_glob))
    if not paths:
        raise FileNotFoundError(f"未找到图像用于推断 PATTERN_SIZE：{image_glob}")

    sample = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError(f"读取图像失败，无法推断 PATTERN_SIZE：{paths[0]}")

    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    processed = _preprocess(gray)
    search_images = [processed]
    if invert_try:
        search_images.append(cv2.bitwise_not(processed))

    def _try_find(patterns):
        for cols, rows in patterns:
            if cols < 3 or rows < 3:
                continue
            for img in search_images:
                ok, _ = cv2.findCirclesGrid(
                    img,
                    (cols, rows),
                    flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                    blobDetector=detector,
                )
                if ok:
                    return cols, rows
        return None

    keypoints = []
    for img in search_images:
        keypoints = detector.detect(img)
        if keypoints:
            break

    if keypoints:
        count = len(keypoints)
        candidates = set()
        for rows in range(3, count + 1):
            if count % rows == 0:
                cols = count // rows
                if cols >= 3:
                    candidates.add((cols, rows))
                    candidates.add((rows, cols))
        ordered = sorted(candidates, key=lambda x: (-min(x), -max(x)))
        match = _try_find(ordered)
        if match:
            return match

    brute_force = [(cols, rows) for rows in range(3, 21) for cols in range(3, 21)]
    match = _try_find(brute_force)
    if match:
        return match

    raise RuntimeError("自动推断 PATTERN_SIZE 失败，请手动填写 PATTERN_SIZE。")


def _find_asym_circles(gray, pattern_size, detector):
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID
    return cv2.findCirclesGrid(gray, pattern_size, flags=flags, blobDetector=detector)


def _grid_object_points_mm(pattern_size, spacing_mm):
    cols, rows = pattern_size
    obj = []
    for i in range(rows):
        for j in range(cols):
            x = (2 * j + (i % 2)) * (spacing_mm / 2.0)
            y = i * spacing_mm
            obj.append([x, y])
    return np.asarray(obj, dtype=np.float32)  # (N,2)


def _neighbor_pixel_spacings(centers, cols, rows):
    pts = centers.reshape(-1, 2)
    horiz, vert = [], []
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if j + 1 < cols:
                horiz.append(np.linalg.norm(pts[idx + 1] - pts[idx]))
            if i + 1 < rows:
                vert.append(np.linalg.norm(pts[idx + cols] - pts[idx]))
    return np.array(horiz), np.array(vert)


def _affine_partial_fit(model_mm, pixels):
    A, inliers = cv2.estimateAffinePartial2D(
        model_mm.reshape(-1, 1, 2),
        pixels.reshape(-1, 1, 2),
        method=cv2.LMEDS,
        ransacReprojThreshold=3.0,
    )
    if A is None:
        return None, None, None
    s1 = np.hypot(A[0, 0], A[1, 0])
    s2 = np.hypot(A[0, 1], A[1, 1])
    scale_px_per_mm = (s1 + s2) / 2.0
    return A, inliers, scale_px_per_mm


def _select_nine_indices(cols, rows):
    """固定选 9 个格点索引 = [r0, r_mid, r_last] × [c0, c_mid, c_last]"""
    r_sel = [0, rows // 2, rows - 1]
    c_sel = [0, cols // 2, cols - 1]
    idxs = []
    for ri in r_sel:
        for ci in c_sel:
            idxs.append(ri * cols + ci)
    return idxs, r_sel, c_sel  # 扁平索引 + 行列选择


def _export_nine_points_csv(json_items, csv_path):
    headers = ["order", "row", "col", "pixel_x", "pixel_y", "world_x_mm", "world_y_mm"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for it in json_items:
            w.writerow(
                [
                    it["order"],
                    it["row"],
                    it["col"],
                    f"{it['pixel'][0]:.4f}",
                    f"{it['pixel'][1]:.4f}",
                    f"{it['world_mm'][0]:.6f}",
                    f"{it['world_mm'][1]:.6f}",
                ]
            )


def main():
    # 参数检查
    if SPACING_VALUE_MM is None or SPACING_VALUE_MM <= 0:
        raise ValueError("请填写 SPACING_VALUE_MM（相邻圆心物理间距，单位mm，正数）")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = create_blob_detector()

    pattern_size = PATTERN_SIZE
    if pattern_size is None:
        pattern_size = _infer_pattern_size_from_images(IMAGE_GLOB, detector, INVERT_TRY)
        globals()["PATTERN_SIZE"] = pattern_size

    cols, rows = pattern_size
    if cols < 3 or rows < 3:
        raise ValueError("PATTERN_SIZE 至少 3x3。")

    img_list = sorted(glob.glob(IMAGE_GLOB))
    if not img_list:
        raise FileNotFoundError(f"未找到图像：{IMAGE_GLOB}")

    model_mm = _grid_object_points_mm(pattern_size, SPACING_VALUE_MM)  # (N,2)
    nine_flat_idxs, r_sel, c_sel = _select_nine_indices(cols, rows)

    per_image = []  # 每张图的九点&比例
    mm_per_px_ratio_list = []
    mm_per_px_affine_list = []

    for path in img_list:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[读取失败] {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = _preprocess(gray)

        ok, centers = _find_asym_circles(g, pattern_size, detector)
        if (not ok) and INVERT_TRY:
            ok, centers = _find_asym_circles(cv2.bitwise_not(g), pattern_size, detector)
        if not ok:
            print(f"[未检测到] {path}")
            continue

        centers = cv2.cornerSubPix(gray, centers, SUBPIX_WIN, (-1, -1), SUBPIX_TERM)
        pts_px = centers.reshape(-1, 2)

        # —— 比例（邻近间距）法 —— #
        horiz, vert = _neighbor_pixel_spacings(centers, cols, rows)

        def rmean(x):
            if len(x) == 0:
                return np.nan
            q1, q9 = np.quantile(x, 0.05), np.quantile(x, 0.95)
            x = x[(x >= q1) & (x <= q9)]
            return float(np.mean(x))

        px_per_spacing = np.nanmean([rmean(horiz), rmean(vert)])
        mm_per_px_ratio = SPACING_VALUE_MM / px_per_spacing
        mm_per_px_ratio_list.append(mm_per_px_ratio)

        # —— 相似变换拟合 —— #
        A, inliers, scale_px_per_mm = _affine_partial_fit(model_mm, pts_px)
        if A is not None and scale_px_per_mm and scale_px_per_mm > 0:
            mm_per_px_aff = 1.0 / float(scale_px_per_mm)
            mm_per_px_affine_list.append(mm_per_px_aff)
        else:
            mm_per_px_aff = None

        # —— 九点提取（像素 & 物理）—— #
        nine_px = pts_px[nine_flat_idxs]  # (9,2)
        nine_mm = model_mm[nine_flat_idxs]  # (9,2)

        # 可视化
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, centers, True)
        for k, (x, y) in enumerate(nine_px):
            cv2.circle(vis, (int(round(x)), int(round(y))), 7, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{k + 1}",
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        outp = out_dir / f"{Path(path).stem}_det_9pt.png"
        cv2.imwrite(str(outp), vis)

        per_image.append(
            {
                "image": path,
                "mm_per_px_ratio": round(mm_per_px_ratio, 8),
                "mm_per_px_affine": round(mm_per_px_aff, 8) if mm_per_px_aff else None,
                "nine_points": [
                    {
                        "order": i + 1,
                        "row": r_sel[i // 3],
                        "col": c_sel[i % 3],
                        "pixel": [float(nine_px[i, 0]), float(nine_px[i, 1])],
                        "world_mm": [float(nine_mm[i, 0]), float(nine_mm[i, 1])],
                    }
                    for i in range(9)
                ],
                "det_vis": str(outp),
            }
        )

    if not per_image:
        raise RuntimeError("所有图片均未成功检测到圆点。")

    # —— 选择 mm/px 比例：优先 affine 的均值，否则用 ratio 的均值 —— #
    def robust_avg(x):
        x = [v for v in x if v is not None and not np.isnan(v)]
        x = np.array(x, float)
        if len(x) == 0:
            return None, None
        q1, q9 = np.quantile(x, 0.1), np.quantile(x, 0.9)
        x = x[(x >= q1) & (x <= q9)]
        return float(np.mean(x)), float(np.std(x))

    aff_mean, aff_std = robust_avg(mm_per_px_affine_list)
    ratio_mean, ratio_std = robust_avg(mm_per_px_ratio_list)

    if aff_mean is not None:
        mm_per_px = aff_mean
        method = "affine_similarity_fit"
        spread = aff_std or 0.0
    else:
        mm_per_px = ratio_mean
        method = "neighbor_ratio"
        spread = ratio_std or 0.0

    # —— 计算“最终九点”（跨图像像素均值）—— #
    # 以 (row,col) 的顺序稳定聚合
    agg_px = [[] for _ in range(9)]
    world_mm_template = None
    for item in per_image:
        if world_mm_template is None:
            world_mm_template = [pt["world_mm"] for pt in item["nine_points"]]
        for i, pt in enumerate(item["nine_points"]):
            agg_px[i].append(pt["pixel"])
    final_nine = []
    for i in range(9):
        arr = np.array(agg_px[i], float)  # (M,2)
        mean_xy = np.mean(arr, axis=0)  # (2,)
        r = [0, rows // 2, rows - 1][i // 3]
        c = [0, cols // 2, cols - 1][i % 3]
        final_nine.append(
            {
                "order": i + 1,
                "row": r,
                "col": c,
                "pixel": [float(mean_xy[0]), float(mean_xy[1])],
                "world_mm": [
                    float(world_mm_template[i][0]),
                    float(world_mm_template[i][1]),
                ],
            }
        )

    # —— 主结果打印 —— #
    print("\n====== 九点标定结果（远心镜头二维测量）======")
    print(f"像素→毫米：{mm_per_px:.8f} mm/px   方法：{method}   样本Std≈{spread:.2e}")
    print("九点（order,row,col, pixel(x,y) -> world(X,Y,mm)）：")
    for p in final_nine:
        print(
            f"{p['order']:2d}  r={p['row']}, c={p['col']}  "
            f"px=({p['pixel'][0]:.2f},{p['pixel'][1]:.2f})  "
            f"mm=({p['world_mm'][0]:.3f},{p['world_mm'][1]:.3f})"
        )

    # —— 保存通用标定 JSON —— #
    Path(SAVE_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pattern_size": {"cols": cols, "rows": rows},
                "spacing_mm": float(SPACING_VALUE_MM),
                "mm_per_px": float(mm_per_px),
                "method": method,
                "std_of_mm_per_px": float(spread),
                "images": per_image,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # —— 保存九点最终结果（JSON & CSV）—— #
    with open(SAVE_9PT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mm_per_px": float(mm_per_px),
                "pattern_size": {"cols": cols, "rows": rows},
                "spacing_mm": float(SPACING_VALUE_MM),
                "rows_selected": [0, rows // 2, rows - 1],
                "cols_selected": [0, cols // 2, cols - 1],
                "nine_points": final_nine,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    _export_nine_points_csv(final_nine, SAVE_9PT_CSV)

    # —— 保存每张图的九点（便于溯源） —— #
    with open(SAVE_PER_IMAGE_9PT, "w", encoding="utf-8") as f:
        json.dump(per_image, f, ensure_ascii=False, indent=2)

    print(f"\n[已保存] 总体结果：{SAVE_JSON}")
    print(f"[已保存] 九点最终结果（JSON）：{SAVE_9PT_JSON}")
    print(f"[已保存] 九点最终结果（CSV）：{SAVE_9PT_CSV}")
    print(f"[已保存] 每图九点详情：{SAVE_PER_IMAGE_9PT}")


if __name__ == "__main__":
    main()
