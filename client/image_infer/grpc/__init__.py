"""gRPC 推理接口模块。"""

from .api import (
    GRPCClientConfig,
    create_stub,
    infer_dir_to_jsons,
    infer_image_save_raw_json,
)

__all__ = [
    "GRPCClientConfig",
    "create_stub",
    "infer_dir_to_jsons",
    "infer_image_save_raw_json",
]
