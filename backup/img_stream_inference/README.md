# 图像流推理与评估工具 (img-stream-inference)

一个面向**小型视觉模型批量评估**与**推理流水线管理**的统一工具套件，支持：

- 多协议推理接口：HTTPS / gRPC / 标准 gRPC (header 传 taskId)
- 批量图片推理（串行或受控并行）
- 推理结果 → LabelMe JSON 格式转换
- 目标检测评估（IoU / Precision / Recall / 全图正确率）
- 结果可视化（统计模式 / 标签配色模式）
- 语义分割（可选：评估 + 掩膜可视化）
- 统一线程数配置 + 时间戳隔离输出目录
- GUI 图形化操作 + 独立 `pipeline.py` 无界面批处理

---
## ✨ 设计目标
| 目标 | 说明 |
|------|------|
| 统一性 | 线程控制、目录结构、配置格式统一管理 |
| 可追溯 | 每次执行生成独立时间戳目录：`https_HHMMSS/` 等 |
| 稳定性 | 默认串行（max_workers=1），避免 I/O / 服务器瓶颈 |
| 安全性 | 所有文本输入粘贴自动清理空白字符 |
| 可扩展 | 新增推理后端或评估类型最少改动 |
| 直观性 | GUI 引导式操作 + 进度 / 日志实时输出 |

---
## 📂 目录结构（核心）
```
img_stream_inference/
├── app.py                     # GUI 主程序（推荐入口）
├── pipeline.py                # 流水线核心（可脚本调用）
├── interface/
│   ├── https/https_api.py     # HTTPS 推理客户端
│   ├── grpc/grpc_api.py       # 标准 gRPC（带 taskId, streamName）
│   └── __init__.py
├── grpc_client/grpc_api.py    # 普通 gRPC 推理客户端
├── model_evaluation/
│   ├── eval_report.py         # 检测结果统计 & Excel 报告
│   ├── matcher.py             # 框匹配 (Hungarian)
│   └── vision.py              # 可视化工具
├── semantic_segmentation/     # 语义分割评估/可视化
├── interface/convert2labelme.py / response2labelme.py  # 格式转换
├── pyproject.toml / uv.lock   # 依赖与环境
└── logs/                      # 运行日志
```

---
## 🧠 核心组件说明
| 模块 | 作用 | 关键点 |
|------|------|--------|
| `app.py` | GUI 界面 | 统一配置、线程控制、实时日志 / 进度回调 |
| `pipeline.py` | 无界面流水线 | 可独立脚本调用；封装步骤编排 |
| `https_api.py` | HTTPS 推理 | Token 缓存 + 重试 + 失败日志 |
| `grpc_client/grpc_api.py` | gRPC 推理 | 支持串行/并行；失败重试；超时控制 |
| `interface/grpc/grpc_api.py` | 标准 gRPC | header 传递 taskId / streamName |
| `convert2labelme.py` | 结果格式转换 | 统一转换为 LabelMe JSON 结构 |
| `eval_report.py` | 检测评估 | TP/FP/FN 统计 + 全图正确率 + Excel 输出 |
| `vision.py` | 可视化 | 两种模式：统计模式 / 标签颜色模式 |
| `semantic_segmentation/` | 分割评估 | IoU / 掩膜输出（可选） |

---
## 🔄 处理流程 (检测)
```
Images → 推理 (HTTPS / gRPC / 标准gRPC)
       → 原始响应(JSON) → LabelMe 转换
       → 评估 (IoU/Precision/Recall/全图正确率 + Excel)
       → 可视化 (统计对比 / 颜色展示)
```
语义分割路线：
```
Images → 推理 → (跳过格式转换) → 分割评估 → 掩膜可视化
```

---
## 🧵 线程与执行策略
| 线程数 (global_workers) | 行为 |
|-------------------------|------|
| 1 | 串行，稳定性优先，避免服务器/磁盘争用 |
| >1 | 线程池并行（仅推理/转换/评估内部支持）|

所有模块使用**统一线程数**，由 GUI 或配置字典传入：
```python
{"https_config": {"max_workers": global_workers}, ...}
```

---
## 🕒 输出目录策略
每次执行生成独立时间戳根目录：
```
https_143022/
  ├── responses/         # 原始接口响应
  ├── pred_jsons/        # 统一预测(JSON)
  └── reports/
       ├── evaluation_report.xlsx
       └── visualization_results/
```
其他类型：`grpc_143022/`, `grpc_standard_143022/`。

---
## 🚀 快速开始
### 1. 安装依赖
使用 [uv](https://github.com/astral-sh/uv)：
```bash
uv sync
```
或使用 pip：
```bash
pip install -e .
```

### 2. 启动 GUI
```bash
uv run app.py
```

### 3. 命令行批处理 (调用 pipeline)
```python
import pipeline
config = {
    "INFERENCE_TYPE": "https",
    "IMAGES_DIR": "samples/images",
    "GT_JSONS_DIR": "samples/gt",
    "HTTPS_CONFIG": {
        "img_stream_url": "https://x.x.x.x:38443/api/v1/stream",
        "stream_name": "demo",
        "access_key": "YOUR_AK",
        "secret_key": "YOUR_SK",
        "raw_responses_dir": "https/responses",
        "pred_jsons_dir": "https/pred_jsons",
        "max_workers": 1
    },
    "EVAL_CONFIG": {"iou_threshold": 0.5},
    "SEMSEG_CONFIG": {"enabled": False},
    "STEPS": {
        "run_inference": True,
        "run_conversion": True,
        "run_evaluation": True,
        "run_visualization": True,
        "run_semseg_evaluation": False,
        "run_semseg_visualization": False,
    }
}
from pipeline import run_inference_pipeline, set_progress_callback
set_progress_callback(lambda step, p: print(f"[{step}] {p}%"))
run_inference_pipeline(config)
```

---
## ⚙️ GUI 关键功能
| 功能 | 说明 |
|------|------|
| 智能粘贴 | 自动清理多余空格/换行 |
| 步骤联动 | 目标检测与语义分割互斥，推理固定开启 |
| 进度反馈 | 每阶段推送到进度条与日志框 |
| 模式提示 | 日志告知当前路线：检测 / 分割 / 自由 |
| 串行优先 | 默认线程=1，保证稳定性 |

---
## 🧪 评估指标 (目标检测)
| 指标 | 说明 |
|------|------|
| TP / FP / FN | 按类别聚合统计 |
| Precision | TP / (TP + FP)；当无预测 → `N/A (无预测)` |
| Recall | TP / (TP + FN)；当无GT → `N/A (无GT)` |
| 图片级完全正确率 | 所有GT均匹配且无多余预测 |

输出：`evaluation_report.xlsx`

---
## 🖼 可视化模式
| 模式 | 说明 |
|------|------|
| 统计模式 | 突出 TP / FP / FN 分类差异 |
| 标签颜色模式 | 为不同类别分配一致颜色 |

---
## 🛡 稳定性与容错
- 所有网络调用带重试（指数或线性退避）
- 失败写入 `fail.log`
- JSON响应解析失败 → 明确日志提示
- 目录自动创建
- 语义分割与检测互斥保证流程清晰

---
## 🔌 扩展指南
新增推理后端示例步骤：
1. 在 `interface/` 下创建新模块，如 `interface/triton/triton_api.py`
2. 提供统一方法：`save_all_images_multithread(img_dir, out_dir, ..., max_workers)`
3. 在 GUI 与 `pipeline.py` 中增加 `INFERENCE_TYPE` 分支
4. 复用目录策略 + 线程控制

---
## 🧷 常见问题 (FAQ)
| 问题 | 解决方案 |
|------|----------|
| Excel 为空 / 统计为0 | 检查预测JSON是否与GT文件名一致 |
| JSON 无法解析 | 查看 `responses/` 原始响应是否为合法JSON |
| 结果目录被覆盖 | 每次运行自动生成新时间戳，确保未手动改写 |
| 线程>1速度不提升 | 可能受限于：网络带宽 / 服务QPS / I/O 瓶颈 |
| 语义分割无法勾选格式转换 | 该流程无需LabelMe转换，已自动禁用 |

---
## 🗺 后续可改进方向
- [ ] 增加 mAP@0.5:0.95 统计
- [ ] 支持 COCO / YOLO 标注格式输入
- [ ] 增加 Web 版本（FastAPI + 前端）
- [ ] 增量执行：复用已存在的推理结果
- [ ] 语义分割掩膜差分图支持
- [ ] GPU 资源使用监控

---
## 🧾 许可证
当前未指定 License，可根据需要添加（推荐 MIT / Apache-2.0）。

---
## 🙋 支持
若需接入新模型 / 指标拓展，可在 `issues` 中提出或直接扩展对应模块。

> 编写与注释已优化，欢迎继续迭代。祝使用顺利！
