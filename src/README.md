# 🤖 AI项目使用说明

## 快速启动

### 方式一：启动图形界面 (推荐)
```bash
python main.py gui
# 或简单地
python main.py
```

### 方式二：命令行方式
```bash
python main.py pipeline
```

### 获取帮助
```bash
python main.py help
```

## 项目结构

```
src/
├── main.py                 # 项目启动入口
├── ui/                     # 图形界面模块
│   ├── app.py             # GUI应用主入口
│   ├── controllers/       # 控制器
│   ├── models/            # 数据模型
│   ├── views/             # 视图组件
│   └── widgets/           # 自定义组件
├── pipeline/              # 流水线处理模块
│   └── main.py           # 流水线主逻辑
├── clients/               # 推理客户端
│   ├── auth.py           # 认证相关
│   └── img_infer/        # 图像推理客户端
│       ├── grpc/         # gRPC客户端
│       └── https/        # HTTP客户端
├── eval/                  # 评估模块
│   ├── detection/        # 目标检测评估
│   └── segmentation/     # 分割评估
├── core/                  # 核心功能
│   └── labelme/          # Labelme格式转换
└── utils/                 # 工具模块
    ├── logger.py         # 日志工具
    └── exception.py      # 异常处理
```

## 功能说明

### GUI界面功能
- 🎯 模型推理pipeline配置
- 📊 评估结果可视化
- 📝 日志实时显示
- ⚙️ 参数配置界面

### Pipeline功能
- 🔄 HTTPS/gRPC推理客户端
- 📋 批量图像处理
- 🏷️ Labelme格式转换
- 📈 检测和分割评估
- 🖼️ 结果可视化

## 环境要求

- Python 3.13+
- 已安装的依赖包（见 pyproject.toml）
- tkinter（GUI界面需要）

## 开发模式

如果要直接调用模块功能：

```python
# 导入并使用pipeline
from pipeline.main import run, PipelineConfig

# 导入并使用评估工具
from eval.detection.report import evaluate_dir_to_csv
from eval.segmentation import evaluate_dir_to_csv as seg_eval

# 导入并使用客户端
from clients.img_infer.https.client import infer_dir
from clients.img_infer.grpc.standard import infer_dir as grpc_infer
```

## 故障排除

### 常见问题

1. **tkinter导入错误**
   - 确保系统已安装tkinter（通常随Python一起安装）
   - macOS: `brew install python-tk`
   - Ubuntu: `sudo apt-get install python3-tk`

2. **模块导入错误**
   - 确保在src目录下运行命令
   - 检查Python环境是否正确激活

3. **依赖包缺失**
   - 运行 `uv sync` 或 `pip install -e .` 安装依赖

### 获取帮助
- 运行 `python main.py help` 查看可用命令
- 查看日志输出了解详细错误信息
