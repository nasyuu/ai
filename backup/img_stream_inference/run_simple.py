#!/usr/bin/env python3
"""
简约版 AI 视觉评估平台启动脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app_gradio_simple import PipelineGradioAppSimple

    if __name__ == "__main__":
        print("🎨 启动 AI 视觉评估平台 - 黑白大气版...")
        print("🌐 访问地址: http://localhost:7860")
        print("📱 支持移动端访问")
        print("✨ 优雅 • 简约 • 专业")
        print("-" * 50)

        app = PipelineGradioAppSimple()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
        )

except ImportError as e:
    print(f"导入失败: {e}")
    print("请确保所有依赖已正确安装")
    sys.exit(1)
except Exception as e:
    print(f"启动失败: {e}")
    sys.exit(1)
