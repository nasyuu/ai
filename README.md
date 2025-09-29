# 普通镜头下 FOV、放大率与像元、分辨率、焦距、工作距离的关系

## 1) 参数定义

- **像元大小 $p$**：传感器单个像素的物理尺寸。常用单位 µm（需除以 1000 才是 mm）。
- **分辨率 $N_w, N_h$**：输出图像的宽、高像素数。
- **传感器尺寸 $S_w, S_h$**：传感器的物理宽、高（mm）
  
  $$
  S_w = p \cdot N_w,\quad S_h = p \cdot N_h
  $$

- **焦距 $f$**：镜头等效焦距（mm）。
- **物距 $u$**：镜头主平面到被测平面距离（mm），工程中常称工作距离 WD。
- **像距 $v$**：镜头主平面到成像面的距离（mm）。
- **放大率 $M$**：成像比例（无量纲）
  
  $$
  M = \frac{L_{\text{像}}}{L_{\text{物}}} = \frac{v}{u}
  $$

- **FOV（Field of View）**：相机在被测平面上的可见范围（宽/高，单位 mm）。

---

## 2) 从薄透镜公式推导放大率

薄透镜成像关系：
$$
\frac{1}{f}=\frac{1}{u}+\frac{1}{v}
\;\;\Rightarrow\;\;
v=\frac{fu}{u-f}
$$

放大率：
$$
M=\frac{v}{u}=\frac{f}{u-f}
$$

当 $u \gg f$（物距远大于焦距）时的近似：
$$
M \approx \frac{f}{u}
$$

---

## 3) FOV（视场）公式

物面尺寸 = 像面尺寸 ÷ 放大率，因此
$$
FOV_w=\frac{S_w}{M},\qquad FOV_h=\frac{S_h}{M}
$$

代入 $S_w=pN_w$、$M=\dfrac{f}{u-f}$：
$$
\boxed{
FOV_w = pN_w\cdot\frac{u-f}{f},\qquad
FOV_h = pN_h\cdot\frac{u-f}{f}
}
$$

当 $u \gg f$ 的近似：
$$
FOV_w \approx pN_w\cdot\frac{u}{f},\qquad
FOV_h \approx pN_h\cdot\frac{u}{f}
$$

对角线：
$$
FOV_d=\sqrt{FOV_w^2+FOV_h^2}
$$

---

## 4) 每像素对应的实际长度（mm/px）

$$
\boxed{
\text{mm/px} = \frac{FOV_w}{N_w} = \frac{p}{M} = p\cdot\frac{u-f}{f}
}
$$

近似式：
$$
\text{mm/px} \approx p\cdot\frac{u}{f}
$$

直觉：$p$ 越大 → 每像素实际长度越大；$f$ 越大 → 每像素实际长度越小；$u$ 越大 → 每像素实际长度越大。

---

## 5) 常用反推公式

**已知 FOV，求所需工作距离 $u$（按宽方向）**
$$
u = f\!\left(\frac{FOV_w}{S_w}+1\right)
$$

**已知 FOV 和工作距离，求焦距 $f$**
$$
f = \frac{u\cdot S_w}{S_w+FOV_w}
$$

**已知期望 mm/px，求 $u$ 或 $f$**
$$
\text{mm/px} = p\cdot\frac{u-f}{f}
\;\Rightarrow\;
u = f\!\left(\frac{\text{mm/px}}{p}+1\right),\qquad
f = \frac{u\,p}{p+\text{mm/px}}
$$

---

## 6) 与远心镜头的区别（补充）

远心镜头放大率 $M$ 在其标称工作距离范围内近似不变：
$$
FOV=\frac{\text{Sensor Size}}{M},\qquad
\text{mm/px}=\frac{p}{M}
$$

---

## 7) 工程注意

- **WD 基准**：理论上 $u$ 是到主平面；若用“前端到物体”的机械 WD，$u\gg f$ 时误差通常很小。  
- **畸变**：广角镜头边缘畸变会让实际 FOV 与几何值偏离，精确测量需标定修正。  
- **ROI/缩放**：只用局部 ROI 时，把 $N_w,N_h$ 换成实际使用的像素数。  
- **对焦**：调焦会改变有效焦距与主平面位置，最终对焦后再做标定。

---

## ✅ 一页总公式

$$
\begin{aligned}
& S_w = pN_w,\quad S_h = pN_h \\
& M = \frac{f}{u-f}\;\;(\approx \tfrac{f}{u}) \\
& FOV_{w,h}=S_{w,h}\cdot\frac{u-f}{f} \\
& \text{mm/px}=p\cdot\frac{u-f}{f}\;\;(\approx p\cdot\tfrac{u}{f})
\end{aligned}
$$

> 关键直觉：FOV 与像元、分辨率、工作距离成正比，与焦距成反比；每像素实际长度 = $p/M$。远心镜头用固定 $M$，FOV 基本不随 $u$ 变。