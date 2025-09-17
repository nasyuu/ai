# AI 图像推理与评估工具 - 可扩展架构设计

## 项目概述

基于backup项目重构设计的下一代图像流推理与评估工具，采用模块化、可扩展的架构设计，支持多种推理协议、评估方法和数据格式。

## 架构特点

### 🏗️ 模块化设计
- 每个功能模块独立封装，降低耦合度
- 基于接口和抽象类的设计，支持插件式扩展
- 统一的错误处理和日志系统

### 🔧 可扩展性
- 插件化的推理客户端架构
- 可注册的数据转换器
- 中间件机制支持功能增强
- 配置驱动的功能启用/禁用

### 🛡️ 健壮性
- 统一的异常处理体系
- 完善的数据验证机制
- 多级配置系统（默认值、配置文件、环境变量）
- 线程安全的操作

## 核心模块

### 1. 基础设施层 (`utils/`)

#### 异常处理模块 (`utils/exceptions.py`)
```python
# 统一的异常体系
BaseInferenceException
  ├── ConfigurationError      # 配置错误
  ├── FileOperationError      # 文件操作错误
  ├── InferenceAPIError       # 推理API错误
  ├── ModelEvaluationError    # 模型评估错误
  ├── DataFormatError         # 数据格式错误
  ├── ImageProcessingError    # 图像处理错误
  ├── NetworkError            # 网络错误
  ├── ValidationError         # 验证错误
  ├── PipelineError           # 流水线错误
  └── ThreadExecutionError    # 线程执行错误

# 异常处理装饰器
@handle_exceptions(logger=logger, reraise=True)
def your_function():
    pass
```

#### 日志模块 (`utils/logger.py`)
```python
# 统一的日志管理
InferenceLogger
  ├── 文件日志 + 控制台日志
  ├── 环境自适应（开发/生产）
  ├── GUI日志处理器
  ├── 进度跟踪日志器
  └── 日志轮转管理

# 使用示例
logger = setup_logger("my_module", level=LogLevel.INFO)
progress_logger = ProgressLogger(logger, progress_callback)
```

### 2. 配置管理层 (`core/config/`)

#### 配置管理模块 (`core/config/__init__.py`)
```python
# 多层次配置系统
ConfigManager
  ├── 多配置源支持（文件、环境变量、默认值）
  ├── 配置验证器
  ├── 动态配置更新
  ├── 配置观察者模式
  └── 类型安全的配置获取

# 配置模式
InferenceConfigSchema    # 推理配置
EvaluationConfigSchema   # 评估配置

# 使用示例
config = get_config()
config.set("max_workers", 4)
inference_url = config.get("https_url")
```

### 3. 接口抽象层 (`core/interfaces/`)

#### 统一推理接口 (`core/interfaces/__init__.py`)
```python
# 协议抽象
BaseInferenceClient
  ├── HTTPSClient        # HTTPS推理客户端
  ├── GRPCClient         # gRPC推理客户端
  ├── GRPCStandardClient # 标准gRPC客户端
  └── WebSocketClient    # WebSocket客户端（预留）

# 会话管理
InferenceSessionManager
  ├── 多会话管理
  ├── 连接池管理
  └── 会话状态追踪

# 中间件系统
MiddlewareManager
  ├── LoggingMiddleware    # 日志中间件
  ├── RetryMiddleware      # 重试中间件
  ├── CacheMiddleware      # 缓存中间件（预留）
  └── MetricsMiddleware    # 监控中间件（预留）
```

### 4. 数据处理层 (`core/data/`)

#### 数据转换模块 (`core/data/__init__.py`)
```python
# 数据格式转换
DataConverterFactory
  ├── LabelMeConverter     # LabelMe格式
  ├── CocoConverter        # COCO格式
  ├── YoloConverter        # YOLO格式（预留）
  └── PascalVOCConverter   # Pascal VOC格式（预留）

# 数据处理器
DataProcessor
  ├── 格式转换
  ├── 批量处理
  ├── 数据验证
  ├── 标注合并
  └── 置信度过滤
```

## 扩展点设计

### 1. 推理协议扩展
```python
# 注册新的推理客户端
class CustomInferenceClient(BaseInferenceClient):
    async def connect(self): pass
    async def infer_single(self, request): pass

InferenceClientFactory.register_client(
    InferenceProtocol.CUSTOM,
    CustomInferenceClient
)
```

### 2. 数据格式扩展
```python
# 注册新的数据转换器
class CustomConverter(BaseDataConverter):
    def load(self, file_path): pass
    def save(self, annotation, file_path): pass

DataConverterFactory.register_converter(
    AnnotationFormat.CUSTOM,
    CustomConverter
)
```

### 3. 中间件扩展
```python
# 添加自定义中间件
class CustomMiddleware(InferenceMiddleware):
    async def before_request(self, request): pass
    async def after_response(self, request, response): pass

middleware_manager.add_middleware(CustomMiddleware())
```

### 4. 配置模式扩展
```python
# 注册新的配置模式
class CustomConfigSchema(BaseConfigSchema):
    def get_config_items(self): pass
    def validate_config(self, config): pass

config_manager.register_schema(CustomConfigSchema())
```

## 后续扩展规划

### 阶段1：核心功能实现
- [ ] HTTPS/gRPC推理客户端实现
- [ ] LabelMe/COCO格式转换器完善
- [ ] 基础评估指标实现
- [ ] 流水线管理器

### 阶段2：高级功能
- [ ] WebSocket实时推理支持
- [ ] 分布式推理支持
- [ ] 模型性能分析
- [ ] 自动化测试框架

### 阶段3：企业级特性
- [ ] 用户权限管理
- [ ] 审计日志
- [ ] API网关集成
- [ ] 监控告警系统

### 阶段4：AI增强
- [ ] 智能参数调优
- [ ] 异常检测
- [ ] 自动标注质量评估
- [ ] 模型推荐系统

## 设计原则

### SOLID原则
- **单一职责**：每个模块专注单一功能
- **开放封闭**：对扩展开放，对修改封闭
- **里氏替换**：子类可以替换父类
- **接口隔离**：细粒度的接口设计
- **依赖倒置**：依赖抽象而非具体实现

### 其他原则
- **配置优于编码**：通过配置控制行为
- **约定优于配置**：提供合理的默认值
- **失败快速**：尽早发现和报告错误
- **渐进式增强**：支持功能的逐步启用

## 使用示例

### 基础使用
```python
from core.config import init_config, get_config
from core.interfaces import get_session_manager
from core.data import get_data_processor

# 初始化配置
init_config("config.json")

# 创建推理会话
session_manager = get_session_manager()
client = session_manager.create_session(
    "main_session",
    InferenceProtocol.HTTPS,
    connection_config
)

# 数据格式转换
data_processor = get_data_processor()
data_processor.convert_format(
    "input.json", "output.json",
    AnnotationFormat.LABELME,
    AnnotationFormat.COCO
)
```

### 高级扩展
```python
# 添加自定义中间件
class TimingMiddleware(InferenceMiddleware):
    async def before_request(self, request):
        request.metadata["start_time"] = time.time()
        return request

    async def after_response(self, request, response):
        duration = time.time() - request.metadata["start_time"]
        logger.info(f"推理耗时: {duration:.2f}s")
        return response

middleware_manager.add_middleware(TimingMiddleware())

# 配置观察者
def on_config_change(key, new_value, old_value):
    logger.info(f"配置变更: {key} = {new_value}")

config_manager.add_watcher(on_config_change)
```

这个架构设计为后续功能扩充提供了强大的基础，支持渐进式开发和插件化扩展。
