# -*- coding: utf-8 -*-
"""
通用 FOV / mm-per-px 计算器
适配 Hikrobot MV-CE200-10GM/10GC (Sony IMX183, 1", 5472x3648, 像元 2.4 µm)
"""

# ===== 相机参数 =====
RES_W, RES_H = 5472, 3648  # 分辨率
PIXEL_UM = 2.4  # 像元大小 (µm)

# ===== 镜头焦距 (mm) 和 工作距离 WD (mm) 的候选列表 =====
FOCAL_LENGTHS = [16, 35]  # 可以加别的焦距
WORK_DISTANCES = [200, 250, 300, 350]  # mm


def calc_fov(f_mm, wd_mm, res_w, res_h, pixel_um):
    """计算 FOV 和 mm/px"""
    # 传感器物理尺寸 (mm)
    sensor_w_mm = res_w * pixel_um * 1e-3
    sensor_h_mm = res_h * pixel_um * 1e-3

    # 视野 (mm)
    fov_w = wd_mm * sensor_w_mm / f_mm
    fov_h = wd_mm * sensor_h_mm / f_mm

    # 分辨率 (mm/px)
    mm_per_px_w = fov_w / res_w
    mm_per_px_h = fov_h / res_h

    return fov_w, fov_h, mm_per_px_w, mm_per_px_h


def main():
    print("=== FOV / 分辨率计算结果 ===\n")
    header = f"{'焦距(mm)':>8} {'WD(mm)':>8} {'FOV宽(mm)':>12} {'FOV高(mm)':>12} {'µm/px(横)':>12} {'µm/px(纵)':>12}"
    print(header)
    print("-" * len(header))

    for f_mm in FOCAL_LENGTHS:
        for wd in WORK_DISTANCES:
            fov_w, fov_h, mm_px_w, mm_px_h = calc_fov(f_mm, wd, RES_W, RES_H, PIXEL_UM)
            print(
                f"{f_mm:8.1f} {wd:8.1f} {fov_w:12.2f} {fov_h:12.2f} "
                f"{mm_px_w * 1000:12.2f} {mm_px_h * 1000:12.2f}"
            )


if __name__ == "__main__":
    main()
