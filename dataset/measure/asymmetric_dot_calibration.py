# -*- coding: utf-8 -*-
"""
精确 FOV / mm-per-px 计算器（无近似）
适配任意相机；示例：Hikrobot MV-CE200-10GM/10GC (Sony IMX183, 1", 5472x3648, 像元 2.4 µm)

公式（全部精确）:
  S_w = p * N_w,  S_h = p * N_h                     # 传感器尺寸 (mm)
  M   = f / (u - f)                                  # 放大率 (u 为主平面到物面的物距)
  FOV_w = S_w * (u - f) / f                          # 物面宽视野 (mm)
  FOV_h = S_h * (u - f) / f                          # 物面高视野 (mm)
  mm/px = p * (u - f) / f                            # 每像素物理长度 (mm/px)

注意：
- 如果你只有“镜头前端到物体”的机械 WD，请用 principal_plane_offset_mm 把它换算为 u：
    u = wd_mech_mm + principal_plane_offset_mm
- 当 u <= f（物距小于等于焦距）时公式无意义，会抛异常。
"""

# ===== 相机参数 =====
RES_W, RES_H = 5472, 3648     # 分辨率（像素）
PIXEL_UM = 2.4                 # 像元大小 (µm) —— 会转换为 mm

# ===== 镜头焦距 (mm) 和 机械工作距离 WD (mm) 列表 =====
FOCAL_LENGTHS = [16, 35]
WORK_DISTANCES = [200, 250, 300, 350]  # 机械 WD（镜头前端到物体），如已知主平面偏移请在下方设置

# ===== 主平面相对前端的偏移 (mm) =====
# 正数表示主平面在镜头外壳前端的“后方”（向传感器方向），
# 常见十几到几十毫米；未知时先设 0，会稍低估 FOV。
PRINCIPAL_PLANE_OFFSET_MM = 0.0


def calc_fov_exact(f_mm, wd_mech_mm, res_w, res_h, pixel_um, principal_plane_offset_mm=0.0):
    """精确计算 FOV 和 mm/px

    参数：
        f_mm: 焦距 f (mm)
        wd_mech_mm: 机械 WD（前端到物体）(mm)
        res_w, res_h: 分辨率
        pixel_um: 像元 (µm)
        principal_plane_offset_mm: 主平面相对前端偏移 (mm)

    返回：
        fov_w, fov_h, mm_per_px_w, mm_per_px_h  （单位 mm / mm / mm/px / mm/px）
    """
    # 传感器物理尺寸 (mm)
    pixel_mm = pixel_um * 1e-3
    sensor_w_mm = res_w * pixel_mm
    sensor_h_mm = res_h * pixel_mm

    # 由机械 WD 换算理论物距 u（主平面→物体）
    u_mm = wd_mech_mm + principal_plane_offset_mm
    if u_mm <= f_mm:
        raise ValueError(f"物距 u 必须大于焦距 f：u={u_mm:.3f} mm, f={f_mm:.3f} mm")

    # 精确放大率与 FOV
    scale = (u_mm - f_mm) / f_mm                  # == 1/M
    fov_w = sensor_w_mm * scale
    fov_h = sensor_h_mm * scale

    # 精确 mm/px（方像元两轴相同）
    mm_per_px = pixel_mm * scale
    return fov_w, fov_h, mm_per_px, mm_per_px


def main():
    print("=== 精确 FOV / 分辨率计算结果 ===\n")
    header = f"{'焦距(mm)':>8} {'WD(mech,mm)':>12} {'u(主平面,mm)':>12} {'FOV宽(mm)':>12} {'FOV高(mm)':>12} {'µm/px(横)':>12} {'µm/px(纵)':>12}"
    print(header)
    print("-" * len(header))

    for f_mm in FOCAL_LENGTHS:
        for wd in WORK_DISTANCES:
            # 精确
            fov_w, fov_h, mm_px_w, mm_px_h = calc_fov_exact(
                f_mm, wd, RES_W, RES_H, PIXEL_UM, PRINCIPAL_PLANE_OFFSET_MM
            )
            u_eff = wd + PRINCIPAL_PLANE_OFFSET_MM
            print(
                f"{f_mm:8.1f} {wd:12.1f} {u_eff:12.1f} {fov_w:12.2f} {fov_h:12.2f} {mm_px_w * 1000:12.2f} {mm_px_h * 1000:12.2f}"
            )


if __name__ == "__main__":
    main()