# 一些常用的脚本

## Streamlit 管道界面

仓库现在包含一个基于 Streamlit 的可视化界面，覆盖原先 Tkinter GUI 的全部功能：HTTPS/gRPC 推理、LabelMe 转换、检测与分割评估以及可视化。

### 启动界面

```bash
streamlit run ui/streamlit_app.py
```

根据本地环境，也可以使用 `uv run streamlit ...` 或 `poetry run streamlit ...` 来启动。

### 使用提示

- 配置项按照流程分块排列，默认值沿用原 GUI。
- 点击 `开始执行` 后，左侧显示步骤进度，右侧实时滚动日志。
- 若仅执行部分流程，取消勾选对应步骤即可。
- 日志同步写入原有日志系统，仍可在 `logs/` 下查阅。

保留的 `ui/app.py` (Tkinter) 仍可使用，但后续推荐迁移到 Streamlit 界面。
