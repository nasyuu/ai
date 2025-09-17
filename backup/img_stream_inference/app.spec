# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, collect_dynamic_libs
)
import os

# ========= 基本参数 =========
entry_script = 'app.py'    # 入口脚本文件名（改成你的主文件）
app_name     = '目标检测类小模型评估工具'      # 应用名称（输出目录/EXE 名称）

# 如果希望给 EXE 设置“文件图标”（仅资源管理器显示），把 ico_path 改成你的 ico 路径；
# 如果不需要文件图标，保持 None 或删除 EXE(icon=...) 参数。
ico_path = None  # 例如: r'assets\app.ico'

# ========= 安全收集工具函数（避免第三方包缺失时报错） =========
def safe_collect_submodules(pkg):
    try:
        return collect_submodules(pkg)
    except Exception:
        return []

def safe_collect_data(pkg, **kw):
    try:
        return collect_data_files(pkg, **kw)
    except Exception:
        return []

def safe_collect_dylibs(pkg):
    try:
        return collect_dynamic_libs(pkg)
    except Exception:
        return []

# ========= 隐式导入（hiddenimports）=========
hiddenimports = []
# gRPC / protobuf
hiddenimports += safe_collect_submodules('grpc')
hiddenimports += safe_collect_submodules('grpc._cython')
hiddenimports += safe_collect_submodules('grpc_tools')
hiddenimports += safe_collect_submodules('google')
hiddenimports += safe_collect_submodules('google.protobuf')

# 科学 & 数据
hiddenimports += safe_collect_submodules('numpy')
hiddenimports += safe_collect_submodules('scipy')
hiddenimports += safe_collect_submodules('pandas')

# 几何
hiddenimports += safe_collect_submodules('shapely')

# 图像 / 进度条 / 网络
hiddenimports += safe_collect_submodules('PIL')       # Pillow
hiddenimports += safe_collect_submodules('tqdm')
hiddenimports += safe_collect_submodules('requests')

# OpenCV
hiddenimports += safe_collect_submodules('cv2')

# ========= 数据文件（datas）=========
datas = []

# ========= 二进制依赖（binaries）=========
binaries = []
# numpy/scipy 的底层动态库
binaries += safe_collect_dylibs('numpy')
binaries += safe_collect_dylibs('scipy')
# shapely 的 GEOS 动态库
binaries += safe_collect_dylibs('shapely')
# OpenCV 动态库
binaries += safe_collect_dylibs('cv2')

# ========= 分析（Analysis）=========
a = Analysis(
    [entry_script],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,   # True 可略提速启动，False 体积更小一点
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ========= 生成 EXE（one-folder，不用临时目录）=========
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # 建议关闭，避免某些 DLL 被压缩后不稳定
    upx_exclude=[],
    runtime_tmpdir=None, # ⭐ one-folder 运行时不解包到 Temp
    console=False,       # GUI 程序关闭控制台；若要调试可临时改 True
    icon=ico_path,       # 可为 None；仅影响“文件图标”，与任务栏无关
)

