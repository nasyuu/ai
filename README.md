# 普通镜头下 FOV、放大率与像元、分辨率、焦距、工作距离的关系

## 1. 参数定义

- **像元大小 \(p\)**  
  相机传感器单个像素的物理尺寸。常用单位 **µm**，需要除以 1000 转换成 **mm**。

- **分辨率 \(N_w, N_h\)**  
  输出图像的宽、高像素数。

- **传感器尺寸 \(S_w, S_h\)**  
  传感器的物理宽、高（mm）。
  \[
  S_w = p \cdot N_w,\quad S_h = p \cdot N_h
  \]

- **焦距 \(f\)**  
  镜头的等效焦距（mm）。

- **物距 \(u\)**  
  镜头主平面到被测平面的距离（mm）。工程中常称 **工作距离 WD**。

- **像距 \(v\)**  
  镜头主平面到成像面的距离（mm）。

- **放大率 \(M\)**  
  物体在传感器上的成像比例（无量纲）：
  \[
  M = \frac{L_\text{像}}{L_\text{物}} = \frac{v}{u}
  \]

- **FOV（Field of View）**  
  相机在被测平面上的可见范围（宽、高，单位 mm）。

---

## 2. 从薄透镜公式推导放大率

薄透镜成像关系：
\[
\frac{1}{f} = \frac{1}{u} + \frac{1}{v}
\quad\Rightarrow\quad
v = \frac{fu}{u-f}
\]

放大率：
\[
M = \frac{v}{u} = \frac{f}{u - f}
\]

> 当 \(u \gg f\)（物距远大于焦距）时，可近似：
\[
M \approx \frac{f}{u}
\]

---

## 3. FOV（视场）公式

物面尺寸 = 像面尺寸 ÷ 放大率，因此：
\[
FOV_w = \frac{S_w}{M},\quad
FOV_h = \frac{S_h}{M}
\]

代入 \(S_w = p N_w\)、\(M = \dfrac{f}{u - f}\)：
\[
\boxed{
FOV_w = p N_w \cdot \frac{u - f}{f},\quad
FOV_h = p N_h \cdot \frac{u - f}{f}
}
\]

> 当 \(u \gg f\) 时的近似：
\[
FOV_w \approx p N_w \cdot \frac{u}{f},\quad
FOV_h \approx p N_h \cdot \frac{u}{f}
\]

对角线：
\[
FOV_d = \sqrt{FOV_w^2 + FOV_h^2}
\]

---

## 4. 每像素对应的实际长度（mm/px）

\[
\boxed{
\text{mm/px} = \frac{FOV_w}{N_w} = \frac{p}{M} = p \cdot \frac{u - f}{f}
}
\]

> 近似：\(\text{mm/px} \approx p \cdot \dfrac{u}{f}\)

- \(p\) 越大 → 每像素对应的实际长度越大（分辨率降低）。  
- \(f\) 越大 → 每像素实际长度越小（分辨率提高）。  
- \(u\) 越大 → 每像素实际长度越大（视野变大）。

---

## 5. 常用反推公式

- **已知 FOV，求所需工作距离 \(u\)**  
\[
u = f\left(\frac{FOV_w}{S_w}+1\right)
\]

- **已知 FOV 和工作距离，求焦距 \(f\)**  
\[
f = \frac{u \cdot S_w}{S_w + FOV_w}
\]

- **已知期望 mm/px，求物距或焦距**  
\[
\text{mm/px} = p \cdot \frac{u - f}{f}
\quad\Rightarrow\quad
u = f\left(\frac{\text{mm/px}}{p}+1\right),
\quad
f = \frac{u \cdot p}{p+\text{mm/px}}
\]

---

## 6. 与远心镜头的区别

- **普通镜头**：放大率 \(M = f/(u-f)\)，随工作距离变化。  
- **远心镜头**：厂家设计放大率 \(M\) 基本固定，在标称工作距离范围内几乎不随 \(u\) 变化。

\[
FOV = \frac{\text{Sensor Size}}{M},\quad
\text{mm/px} = \frac{p}{M}
\]

---

## 7. 工程使用注意

- **WD 基准**：理论上是主平面到物体；实际用镜头前端到物体误差通常可接受（尤其是 WD ≫ f 时）。  
- **畸变**：广角镜头边缘畸变大，实际 FOV 与公式有差别；精确测量需标定修正。  
- **ROI/缩放**：若只取局部 ROI，用 ROI 像素数代替全分辨率。  
- **对焦**：调焦会改变有效焦距和主平面位置，最终对焦后再用标定数据。

---

### ✅ 总结公式

\[
\begin{cases}
S_w = p N_w,\quad S_h = p N_h\\[4pt]
M = \dfrac{f}{u - f} \;(\approx\dfrac{f}{u})\\[6pt]
FOV_{w,h}=S_{w,h}\cdot\dfrac{u - f}{f}\\[6pt]
\text{mm/px}=p\cdot\dfrac{u - f}{f}\;(\approx p\cdot\dfrac{u}{f})
\end{cases}
\]

> **关键直觉**  
> - FOV ∝ 像元大小、分辨率、工作距离；∝ 1/焦距  
> - 每像素实际长度 = \(p/M\)：放大率越大像素越细、FOV 越小  
> - 远心镜头：用厂家给的固定 \(M\)，FOV 与工作距离几乎无关