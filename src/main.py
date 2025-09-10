#!/usr/bin/env python3
"""
AI项目启动入口
支持启动GUI界面或运行pipeline流水线
"""
import argparse
import sys
from pathlib import Path


def start_gui():
    """启动图形界面"""
    try:
        from ui.app import main as ui_main
        print("🚀 启动图形界面...")
        ui_main()
    except ImportError as e:
        print(f"❌ 无法启动GUI界面，缺少依赖: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ GUI启动失败: {e}")
        sys.exit(1)


def run_pipeline():
    """运行pipeline流水线"""
    try:
        from pipeline.main import run, PipelineConfig
        print("🔧 运行pipeline流水线...")
        # 这里可以添加默认配置或从配置文件读取
        print("提示: 请使用GUI界面或直接导入pipeline.main模块来配置和运行流水线")
    except ImportError as e:
        print(f"❌ 无法运行pipeline，缺少依赖: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline运行失败: {e}")
        sys.exit(1)


def show_help():
    """显示帮助信息"""
    help_text = """
🤖 AI项目启动器

可用功能:
  gui      启动图形界面 (推荐)
  pipeline 运行pipeline流水线
  help     显示此帮助信息

示例:
  python main.py gui       # 启动GUI界面
  python main.py pipeline  # 运行pipeline
  python main.py           # 默认启动GUI界面

项目结构:
  - ui/           图形界面模块
  - pipeline/     流水线处理模块
  - clients/      推理客户端模块
  - eval/         评估模块
  - core/         核心功能模块
  - utils/        工具模块
    """
    print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description="AI项目启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'command',
        nargs='?',
        default='gui',
        choices=['gui', 'pipeline', 'help'],
        help='要执行的命令 (默认: gui)'
    )

    args = parser.parse_args()

    if args.command == 'gui':
        start_gui()
    elif args.command == 'pipeline':
        run_pipeline()
    elif args.command == 'help':
        show_help()
    else:
        show_help()


if __name__ == "__main__":
    main()
