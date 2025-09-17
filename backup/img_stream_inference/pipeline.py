import logging
import os
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path


# 配置日志系统
def setup_logging():
    """设置日志系统"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    is_exe = getattr(sys, "frozen", False)

    if is_exe:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
    return logging.getLogger(__name__)


logger = setup_logging()

# 全局进度回调和停止标志
progress_callback = None
stop_requested = False


def set_progress_callback(callback):
    """设置进度回调函数"""
    global progress_callback
    progress_callback = callback


def request_stop():
    """请求停止执行"""
    global stop_requested
    stop_requested = True


def reset_stop_flag():
    """重置停止标志"""
    global stop_requested
    stop_requested = False


def update_progress(step_name, progress):
    """更新进度"""
    if progress_callback:
        progress_callback(step_name, progress)


def check_stop():
    """检查是否需要停止"""
    return stop_requested


INFERENCE_TYPE = "https"  # 'https' 或 'grpc_standard'
IMAGES_DIR = ""  # 输入图片目录
GT_JSONS_DIR = ""  # 真值标注目录


def get_dynamic_report_paths(inference_type):
    """根据推理类型生成动态报告路径"""
    base_path = f"{inference_type}/reports"
    return {
        "eval_output_file": f"{base_path}/evaluation_report.csv",
        "viz_output_dir": f"{base_path}/visualization_results",
        "semseg_eval_output_file": f"{base_path}/semseg_eval.csv",
        "semseg_viz_output_dir": f"{base_path}/semseg_vis_masks",
    }


def update_config_with_dynamic_paths(config, inference_type):
    """使用动态路径更新配置"""
    paths = get_dynamic_report_paths(inference_type)

    # 更新评估配置
    if "EVAL_CONFIG" in config:
        config["EVAL_CONFIG"]["eval_output_file"] = paths["eval_output_file"]
        config["EVAL_CONFIG"]["viz_output_dir"] = paths["viz_output_dir"]

    # 更新语义分割配置
    if "SEMSEG_CONFIG" in config:
        config["SEMSEG_CONFIG"]["eval_output_file"] = paths["semseg_eval_output_file"]
        config["SEMSEG_CONFIG"]["viz_output_dir"] = paths["semseg_viz_output_dir"]

    return config


# HTTPS 配置 (当 INFERENCE_TYPE='https' 时使用)
HTTPS_CONFIG = {
    "img_stream_url": "",
    "stream_name": "",
    "access_key": "",
    "secret_key": "",
    "raw_responses_dir": "https/responses",
    "pred_jsons_dir": "https/pred_jsons",
    "max_workers": 1,  # 默认串行处理
}

# 标准 gRPC 配置 (当 INFERENCE_TYPE='grpc_standard' 时使用)
GRPC_STANDARD_CONFIG = {
    "grpc_server": "",
    "task_id": "",  # 任务ID
    "stream_name": "",  # 流名称
    "raw_responses_dir": "grpc_standard/responses",
    "pred_jsons_dir": "grpc_standard/pred_jsons",
    "max_workers": 1,  # 默认串行处理
}

# 评估和可视化配置
EVAL_CONFIG = {
    "iou_threshold": 0.5,
    "eval_output_file": "reports/evaluation_report.csv",  # 会被动态路径覆盖
    "viz_output_dir": "reports/visualization_results",  # 会被动态路径覆盖
    "viz_mode": True,  # True: 统计模式(显示TP/FP/FN) 或 False: 标签颜色模式
}

# 语义分割配置
SEMSEG_CONFIG = {
    "enabled": False,  # 是否启用语义分割评估
    "eval_output_file": "reports/semseg_eval.csv",  # 会被动态路径覆盖
    "viz_output_dir": "reports/semseg_vis_masks",  # 会被动态路径覆盖
    "save_diff_png": False,  # 是否保存差分PNG
    "max_workers": 1,  # 默认串行处理
    "iou_threshold": 0.8,  # IoU统计阈值
}

#  流程控制 - 可以跳过某些步骤
STEPS = {
    "run_inference": True,  # 执行推理
    "run_conversion": True,  # 格式转换
    "run_evaluation": True,  # 模型评估
    "run_visualization": True,  # 结果可视化
    "run_semseg_evaluation": False,  # 语义分割评估
    "run_semseg_visualization": False,  # 语义分割可视化
}

# 向后兼容的CONFIG变量
CONFIG = HTTPS_CONFIG  # 默认使用HTTPS配置


def get_config():
    """获取当前配置"""
    if INFERENCE_TYPE == "https":
        return HTTPS_CONFIG
    elif INFERENCE_TYPE == "grpc_standard":
        return GRPC_STANDARD_CONFIG
    else:
        raise ValueError(f"不支持的推理类型: {INFERENCE_TYPE}")


def validate_config():
    """验证配置是否完整"""
    errors = []

    # 检查基础目录配置
    if not IMAGES_DIR or not os.path.exists(IMAGES_DIR):
        errors.append(f"输入图片目录无效: {IMAGES_DIR}")

    if not GT_JSONS_DIR or not os.path.exists(GT_JSONS_DIR):
        errors.append(f"真值标注目录无效: {GT_JSONS_DIR}")

    # 检查推理类型特定配置
    if INFERENCE_TYPE == "https":
        if not HTTPS_CONFIG["img_stream_url"]:
            errors.append("HTTPS推理URL未设置")
        if not HTTPS_CONFIG["stream_name"]:
            errors.append("HTTPS流名称未设置")
        if not HTTPS_CONFIG["access_key"]:
            errors.append("HTTPS访问密钥未设置")
        if not HTTPS_CONFIG["secret_key"]:
            errors.append("HTTPS密钥未设置")
    elif INFERENCE_TYPE == "grpc_standard":
        if not GRPC_STANDARD_CONFIG["grpc_server"]:
            errors.append("标准gRPC服务器地址未设置")
        if not GRPC_STANDARD_CONFIG["task_id"]:
            errors.append("标准gRPC任务ID未设置")
        if not GRPC_STANDARD_CONFIG["stream_name"]:
            errors.append("标准gRPC流名称未设置")

    # 检查IoU阈值
    if not (0 < EVAL_CONFIG["iou_threshold"] <= 1):
        errors.append(f"IoU阈值无效: {EVAL_CONFIG['iou_threshold']}")

    if errors:
        error_msg = "配置验证失败:\n" + "\n".join(f"  - {error}" for error in errors)
        logger.error(error_msg)
        return False, error_msg

    return True, "配置验证通过"


def create_dirs():
    """创建必要的目录"""
    config = get_config()
    dirs = [
        config["raw_responses_dir"],
        config["pred_jsons_dir"],
        EVAL_CONFIG["viz_output_dir"],
    ]

    # 添加评估报告文件的父目录
    eval_output_parent = Path(EVAL_CONFIG["eval_output_file"]).parent
    if str(eval_output_parent) != ".":  # 不是当前目录
        dirs.append(str(eval_output_parent))

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"✓ 目录: {dir_path}")


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = Exception("未知错误")
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} 第{attempt + 1}次尝试失败: {e}, {delay}秒后重试...",
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} 所有{max_retries}次尝试都失败了")
            raise last_exception

        return wrapper

    return decorator


@retry_on_failure(max_retries=2, delay=2.0)
def step_1_https_inference():
    """步骤1: HTTPS推理"""
    logger.info("🌐 步骤1: HTTPS推理...")

    try:
        # 不修改全局变量，直接传递配置
        from interface.https import https_api

        # 构造配置字典
        config = {
            "img_stream_url": HTTPS_CONFIG["img_stream_url"],
            "stream_name": HTTPS_CONFIG["stream_name"],
            "access_key": HTTPS_CONFIG["access_key"],
            "secret_key": HTTPS_CONFIG["secret_key"],
        }

        # 执行推理，传入配置
        https_api.infer_dir_to_jsons(
            image_dir=IMAGES_DIR,
            out_dir=HTTPS_CONFIG["raw_responses_dir"],
            config=config,
            max_workers=HTTPS_CONFIG["max_workers"],
        )

        logger.info("✅ HTTPS推理完成")
        return True

    except Exception as e:
        logger.error(f"❌ HTTPS推理失败: {e}")
        return False


@retry_on_failure(max_retries=2, delay=1.0)
def step_2_https_conversion():
    """步骤2: HTTPS格式转换"""
    logger.info("🔄 步骤2: HTTPS格式转换...")

    try:
        from interface import convert2labelme

        # 设置参数
        convert2labelme.RAW_JSON_DIR = HTTPS_CONFIG["raw_responses_dir"]
        convert2labelme.IMAGE_DIR = IMAGES_DIR
        convert2labelme.LABELME_OUTPUT_DIR = HTTPS_CONFIG["pred_jsons_dir"]
        convert2labelme.MAX_WORKERS = HTTPS_CONFIG["max_workers"]

        # 执行转换
        convert2labelme.batch_convert()

        logger.info("✅ HTTPS格式转换完成")
        return True

    except Exception as e:
        logger.error(f"❌ HTTPS格式转换失败: {e}")
        return False


@retry_on_failure(max_retries=2, delay=2.0)
def step_1_grpc_standard_inference():
    """步骤1: 标准gRPC推理"""
    logger.info("⚡ 步骤1: 标准gRPC推理...")

    try:
        from interface.grpc import grpc_api

        # 设置参数
        grpc_api.SERVER_ADDRESS = GRPC_STANDARD_CONFIG["grpc_server"]
        grpc_api.TASK_ID = GRPC_STANDARD_CONFIG["task_id"]  # 设置任务ID
        grpc_api.STREAM_NAME = GRPC_STANDARD_CONFIG["stream_name"]  # 设置流名称
        grpc_api.IMAGE_DIR = IMAGES_DIR
        grpc_api.OUTPUT_DIR = GRPC_STANDARD_CONFIG["raw_responses_dir"]
        grpc_api.MAX_WORKERS = GRPC_STANDARD_CONFIG["max_workers"]

        # 执行推理
        channel, stub = grpc_api.create_grpc_stub(GRPC_STANDARD_CONFIG["grpc_server"])
        try:
            grpc_api.save_all_images_multithread(
                IMAGES_DIR,
                GRPC_STANDARD_CONFIG["raw_responses_dir"],
                stub,
                GRPC_STANDARD_CONFIG["max_workers"],
            )
        finally:
            channel.close()

        logger.info("✅ 标准gRPC推理完成")
        return True

    except Exception as e:
        logger.error(f"❌ 标准gRPC推理失败: {e}")
        return False


@retry_on_failure(max_retries=2, delay=1.0)
def step_2_grpc_standard_conversion():
    """步骤2: 标准gRPC格式转换"""
    logger.info("🔄 步骤2: 标准gRPC格式转换...")

    try:
        from interface import convert2labelme

        # 设置参数
        convert2labelme.RAW_JSON_DIR = GRPC_STANDARD_CONFIG["raw_responses_dir"]
        convert2labelme.IMAGE_DIR = IMAGES_DIR
        convert2labelme.LABELME_OUTPUT_DIR = GRPC_STANDARD_CONFIG["pred_jsons_dir"]
        convert2labelme.MAX_WORKERS = GRPC_STANDARD_CONFIG["max_workers"]

        # 执行转换
        convert2labelme.batch_convert()

        logger.info("✅ 标准gRPC格式转换完成")
        return True

    except Exception as e:
        logger.error(f"❌ 标准gRPC格式转换失败: {e}")
        return False


@retry_on_failure(max_retries=1, delay=1.0)
def step_3_evaluation():
    """步骤3: 模型评估"""
    logger.info("📊 步骤3: 模型评估...")

    try:
        from model_evaluation import eval_report

        config = get_config()

        # 执行评估
        eval_report.evaluate(
            gt_dir=GT_JSONS_DIR,
            pred_dir=config["pred_jsons_dir"],
            iou_thr=EVAL_CONFIG["iou_threshold"],
            out_csv=EVAL_CONFIG["eval_output_file"],
        )

        logger.info(f"✅ 模型评估完成，报告: {EVAL_CONFIG['eval_output_file']}")
        return True

    except Exception as e:
        logger.error(f"❌ 模型评估失败: {e}")
        return False


@retry_on_failure(max_retries=1, delay=1.0)
def step_4_visualization():
    """步骤4: 结果可视化"""
    logger.info("🎨 步骤4: 结果可视化...")

    try:
        import os

        from model_evaluation import vision

        config = get_config()

        # 设置必要的全局变量
        vision.IMAGE_FOLDER = IMAGES_DIR
        vision.GT_FOLDER = GT_JSONS_DIR
        vision.PRED_FOLDER = config["pred_jsons_dir"]
        vision.IOU_THRESHOLD = EVAL_CONFIG["iou_threshold"]

        # 设置输出目录
        output_base = EVAL_CONFIG["viz_output_dir"]
        vision.CORRECT_DIR = os.path.join(output_base, "correct")
        vision.ERROR_DIR = os.path.join(output_base, "error")

        # 确保输出目录存在
        os.makedirs(vision.CORRECT_DIR, exist_ok=True)
        os.makedirs(vision.ERROR_DIR, exist_ok=True)

        # 执行可视化
        vision.visualize_all(
            gt_folder=GT_JSONS_DIR,
            pred_folder=config["pred_jsons_dir"],
            iou_thr=EVAL_CONFIG["iou_threshold"],
            stats_mode=EVAL_CONFIG["viz_mode"],
            max_workers=EVAL_CONFIG["max_workers"],
        )

        logger.info(f"✅ 结果可视化完成，输出: {EVAL_CONFIG['viz_output_dir']}")
        return True

    except Exception as e:
        logger.error(f"❌ 结果可视化失败: {e}")
        return False


@retry_on_failure(max_retries=1, delay=1.0)
def step_5_semseg_evaluation():
    """步骤5: 语义分割IoU评估"""
    logger.info("🔬 步骤5: 语义分割IoU评估...")

    try:
        if not SEMSEG_CONFIG["enabled"]:
            logger.info("⚠️ 语义分割评估未启用，跳过")
            return True

        # 动态导入语义分割模块
        import sys
        from pathlib import Path

        semseg_path = Path(__file__).parent / "semantic_segmentation"
        if str(semseg_path) not in sys.path:
            sys.path.append(str(semseg_path))

        from semantic_segmentation import ss_IoU

        config = get_config()
        raw_responses_dir = config["raw_responses_dir"]  # 直接使用原始推理响应
        eval_output = SEMSEG_CONFIG["eval_output_file"]

        logger.info("🔬 开始语义分割IoU评估")
        logger.info(f"   推理响应目录: {raw_responses_dir}")
        logger.info(f"   真值目录: {GT_JSONS_DIR}")
        logger.info(f"   输出文件: {eval_output}")

        if check_stop():
            logger.warning("⚠️ 用户请求停止执行")
            return False

        # 设置语义分割模块的配置
        ss_IoU.PRED_JSON_DIR = raw_responses_dir
        ss_IoU.GT_JSON_DIR = GT_JSONS_DIR
        ss_IoU.OUTPUT_XLSX = eval_output
        ss_IoU.SAVE_DIFF_PNG = SEMSEG_CONFIG["save_diff_png"]
        ss_IoU.MAX_WORKERS = SEMSEG_CONFIG["max_workers"]
        ss_IoU.IOU_THRESHOLD = SEMSEG_CONFIG["iou_threshold"]

        # 执行语义分割评估
        ss_IoU.main()

        logger.info("✅ 语义分割IoU评估完成")
        return True

    except Exception as e:
        logger.error(f"❌ 语义分割评估失败: {e}")
        import traceback

        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


@retry_on_failure(max_retries=1, delay=1.0)
def step_6_semseg_visualization():
    """步骤6: 语义分割可视化"""
    logger.info("🎨 步骤6: 语义分割可视化...")

    try:
        if not SEMSEG_CONFIG["enabled"]:
            logger.info("⚠️ 语义分割可视化未启用，跳过")
            return True

        # 动态导入语义分割模块
        import sys
        from pathlib import Path

        semseg_path = Path(__file__).parent / "semantic_segmentation"
        if str(semseg_path) not in sys.path:
            sys.path.append(str(semseg_path))

        from semantic_segmentation import ss_visualization

        config = get_config()
        raw_responses_dir = config["raw_responses_dir"]  # 直接使用原始推理响应
        viz_dir = SEMSEG_CONFIG["viz_output_dir"]

        logger.info("🎨 开始语义分割可视化")
        logger.info(f"   推理响应目录: {raw_responses_dir}")
        logger.info(f"   图像目录: {IMAGES_DIR}")
        logger.info(f"   输出目录: {viz_dir}")

        if check_stop():
            logger.warning("⚠️ 用户请求停止执行")
            return False

        # 设置语义分割可视化模块的配置
        ss_visualization.PRED_JSON_DIR = raw_responses_dir
        ss_visualization.IMAGE_DIR = IMAGES_DIR
        ss_visualization.VIS_DIR = viz_dir

        # 执行语义分割可视化
        ss_visualization.main()

        logger.info("✅ 语义分割可视化完成")
        return True

    except Exception as e:
        logger.error(f"❌ 语义分割可视化失败: {e}")
        import traceback

        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def main():
    start_time = time.time()

    # 重置停止标志
    reset_stop_flag()

    logger.info("=" * 60)
    logger.info("🚀 开始一键执行推理流水线")
    logger.info(f"📋 配置: {INFERENCE_TYPE.upper()} 模式")
    logger.info("=" * 60)

    update_progress("初始化", 0)

    # 验证配置
    is_valid, validation_msg = validate_config()
    if not is_valid:
        logger.error("❌ 配置验证失败，停止执行")
        update_progress("配置错误", 0)
        return False

    logger.info("✅ 配置验证通过")

    # 创建必要的目录
    create_dirs()
    logger.info("📁 目录配置:")
    logger.info(f"   输入图片: {IMAGES_DIR}")
    logger.info(f"   真值标注: {GT_JSONS_DIR}")
    config = get_config()
    logger.info(f"   原始响应: {config['raw_responses_dir']}")
    logger.info(f"   预测结果: {config['pred_jsons_dir']}")
    logger.info(f"   评估报告: {EVAL_CONFIG['eval_output_file']}")
    logger.info(f"   可视化图: {EVAL_CONFIG['viz_output_dir']}")

    # 执行步骤
    success = True

    def log_step_progress(step_name: str, success: bool):
        """记录步骤进度"""
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(f"   {step_name}: {status}")

    # 步骤1: 推理
    if STEPS["run_inference"]:
        if check_stop():
            logger.info("⚠️ 收到停止请求，终止执行")
            return False

        update_progress("步骤1: 推理", 25)

        if INFERENCE_TYPE == "https":
            result = step_1_https_inference()
        elif INFERENCE_TYPE == "grpc_standard":
            result = step_1_grpc_standard_inference()
        else:
            logger.error(f"❌ 不支持的推理类型: {INFERENCE_TYPE}")
            result = False

        log_step_progress("推理", result)
        success = success and result
        if not result:
            logger.error("推理步骤失败，停止执行")
            return False

    # 步骤2: 格式转换
    if STEPS["run_conversion"]:
        if check_stop():
            logger.info("⚠️ 收到停止请求，终止执行")
            return False

        update_progress("步骤2: 格式转换", 50)

        if INFERENCE_TYPE == "https":
            result = step_2_https_conversion()
        elif INFERENCE_TYPE == "grpc_standard":
            result = step_2_grpc_standard_conversion()
        else:
            logger.error(f"❌ 不支持的推理类型: {INFERENCE_TYPE}")
            result = False

        log_step_progress("格式转换", result)
        success = success and result
        if not result:
            logger.error("格式转换步骤失败，停止执行")
            return False

    # 步骤3: 模型评估
    if STEPS["run_evaluation"]:
        if check_stop():
            logger.info("⚠️ 收到停止请求，终止执行")
            return False

        update_progress("步骤3: 模型评估", 75)

        result = step_3_evaluation()
        log_step_progress("模型评估", result)
        success = success and result
        if not result:
            logger.error("模型评估步骤失败，停止执行")
            return False

    # 步骤4: 结果可视化
    if STEPS["run_visualization"]:
        if check_stop():
            logger.info("⚠️ 收到停止请求，终止执行")
            return False

        update_progress("步骤4: 结果可视化", 85)

        result = step_4_visualization()
        log_step_progress("结果可视化", result)
        success = success and result
        if not result:
            logger.error("结果可视化步骤失败，停止执行")
            return False

    # 步骤5: 语义分割IoU评估
    if STEPS["run_semseg_evaluation"]:
        if check_stop():
            logger.info("⚠️ 收到停止请求，终止执行")
            return False

        update_progress("步骤5: 语义分割评估", 90)

        result = step_5_semseg_evaluation()
        log_step_progress("语义分割评估", result)
        success = success and result
        if not result:
            logger.error("语义分割评估步骤失败，停止执行")
            return False

    # 步骤6: 语义分割可视化
    if STEPS["run_semseg_visualization"]:
        if check_stop():
            logger.info("⚠️ 收到停止请求，终止执行")
            return False

        update_progress("步骤6: 语义分割可视化", 95)

        result = step_6_semseg_visualization()
        log_step_progress("语义分割可视化", result)
        success = success and result
        if not result:
            logger.error("语义分割可视化步骤失败，停止执行")
            return False

    # 总结
    total_time = time.time() - start_time

    # 设置最终进度
    if success:
        update_progress("执行完成", 100)
    else:
        update_progress("执行失败", 100)

    logger.info("=" * 60)
    if success:
        logger.info("🎉 流水线执行成功！")
    else:
        logger.error("❌ 流水线执行失败！")
    logger.info(f"⏱️  总耗时: {total_time:.2f}秒")
    logger.info("📁 输出文件:")
    logger.info(f"   ├── {config['raw_responses_dir']}/          # 原始推理响应")
    logger.info(f"   ├── {config['pred_jsons_dir']}/            # LabelMe预测结果")
    logger.info(f"   ├── {EVAL_CONFIG['eval_output_file']}      # 评估报告")
    logger.info(f"   └── {EVAL_CONFIG['viz_output_dir']}/        # 可视化图片")
    logger.info("       ├── correct/     # 正确预测的图片")
    logger.info("       └── error/       # 错误预测的图片")
    if not success:
        logger.error("请检查错误信息并修复配置。")
    logger.info("=" * 60)

    return success


def run_inference_pipeline(custom_config=None):
    # 如果传入了自定义配置，临时使用它
    if custom_config:
        # 保存原始配置
        original_config = {}

        # 安全的配置项列表
        safe_config_keys = [
            "INFERENCE_TYPE",
            "IMAGES_DIR",
            "GT_JSONS_DIR",
            "HTTPS_CONFIG",
            "GRPC_CONFIG",
            "GRPC_STANDARD_CONFIG",
            "EVAL_CONFIG",
            "SEMSEG_CONFIG",
            "STEPS",
        ]

        for key, value in custom_config.items():
            if key in safe_config_keys and key in globals():
                original_config[key] = globals()[key]
                globals()[key] = value
                logger.debug(f"临时设置配置: {key} = {value}")
            else:
                logger.warning(f"忽略不安全的配置项: {key}")

        try:
            # 执行主流程
            logger.info("使用自定义配置执行流水线")
            result = main()
        except Exception as e:
            logger.error(f"执行流水线时发生异常: {e}")
            result = False
        finally:
            # 恢复原始配置
            for key, value in original_config.items():
                globals()[key] = value
                logger.debug(f"恢复配置: {key}")
            logger.info("配置已恢复")

        return result
    else:
        # 使用默认配置，但更新路径为动态路径
        logger.info("使用默认配置执行流水线，应用动态报告路径")

        # 保存原始配置
        original_eval_config = EVAL_CONFIG.copy()
        original_semseg_config = SEMSEG_CONFIG.copy()

        try:
            # 根据当前推理类型更新路径
            paths = get_dynamic_report_paths(INFERENCE_TYPE)
            globals()["EVAL_CONFIG"]["eval_output_file"] = paths["eval_output_file"]
            globals()["EVAL_CONFIG"]["viz_output_dir"] = paths["viz_output_dir"]
            globals()["SEMSEG_CONFIG"]["eval_output_file"] = paths[
                "semseg_eval_output_file"
            ]
            globals()["SEMSEG_CONFIG"]["viz_output_dir"] = paths[
                "semseg_viz_output_dir"
            ]

            logger.info(f"报告路径设置为: {INFERENCE_TYPE}/reports/")
            result = main()

        except Exception as e:
            logger.error(f"执行流水线时发生异常: {e}")
            result = False
        finally:
            # 恢复原始配置
            globals()["EVAL_CONFIG"] = original_eval_config
            globals()["SEMSEG_CONFIG"] = original_semseg_config

        return result


if __name__ == "__main__":
    main()
