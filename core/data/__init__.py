"""
数据处理和转换模块

提供统一的数据格式处理、转换和验证功能。
支持多种标注格式（LabelMe、COCO、YOLO等）的互相转换。
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.exceptions import DataFormatError, ValidationError
from utils.logger import get_logger

logger = get_logger("data")


class AnnotationFormat(Enum):
    """标注格式类型"""

    LABELME = "labelme"
    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"
    CUSTOM = "custom"


class TaskType(Enum):
    """任务类型"""

    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    CLASSIFICATION = "classification"


@dataclass
class BoundingBox:
    """边界框"""

    x: float
    y: float
    width: float
    height: float

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """转换为 (x1, y1, x2, y2) 格式"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_center(self) -> Tuple[float, float]:
        """获取中心点坐标"""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def area(self) -> float:
        """计算面积"""
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        """计算IoU"""
        x1_max = max(self.x, other.x)
        y1_max = max(self.y, other.y)
        x2_min = min(self.x + self.width, other.x + other.width)
        y2_min = min(self.y + self.height, other.y + other.height)

        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0

        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        union = self.area() + other.area() - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class Detection:
    """检测结果"""

    bbox: BoundingBox
    label: str
    confidence: float = 1.0
    attributes: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class Segmentation:
    """分割结果"""

    points: List[Tuple[float, float]]
    label: str
    confidence: float = 1.0
    attributes: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class ImageAnnotation:
    """图像标注"""

    image_path: Union[str, Path]
    image_width: int
    image_height: int
    detections: List[Detection] = None
    segmentations: List[Segmentation] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)
        if self.detections is None:
            self.detections = []
        if self.segmentations is None:
            self.segmentations = []
        if self.metadata is None:
            self.metadata = {}


class BaseDataConverter(ABC):
    """数据转换器基类"""

    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> ImageAnnotation:
        """加载标注文件"""
        pass

    @abstractmethod
    def save(self, annotation: ImageAnnotation, file_path: Union[str, Path]):
        """保存标注文件"""
        pass

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """验证数据格式"""
        pass

    @property
    @abstractmethod
    def format_type(self) -> AnnotationFormat:
        """格式类型"""
        pass


class LabelMeConverter(BaseDataConverter):
    """LabelMe格式转换器"""

    @property
    def format_type(self) -> AnnotationFormat:
        return AnnotationFormat.LABELME

    def load(self, file_path: Union[str, Path]) -> ImageAnnotation:
        """加载LabelMe JSON文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise DataFormatError(f"文件不存在: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataFormatError(f"JSON格式错误: {e}", file_path=str(file_path))

        if not self.validate(data):
            raise DataFormatError("LabelMe格式验证失败", file_path=str(file_path))

        # 提取图像信息
        image_path = data.get("imagePath", file_path.stem + ".jpg")
        if not Path(image_path).is_absolute():
            image_path = file_path.parent / image_path

        image_width = data.get("imageWidth", 0)
        image_height = data.get("imageHeight", 0)

        # 创建标注对象
        annotation = ImageAnnotation(
            image_path=image_path, image_width=image_width, image_height=image_height
        )

        # 解析shapes
        for shape in data.get("shapes", []):
            shape_type = shape.get("shape_type", "")
            label = shape.get("label", "")
            points = shape.get("points", [])

            if shape_type == "rectangle" and len(points) == 2:
                # 矩形检测框
                x1, y1 = points[0]
                x2, y2 = points[1]
                bbox = BoundingBox(
                    x=min(x1, x2),
                    y=min(y1, y2),
                    width=abs(x2 - x1),
                    height=abs(y2 - y1),
                )
                detection = Detection(bbox=bbox, label=label)
                annotation.detections.append(detection)

            elif shape_type == "polygon":
                # 多边形分割
                segmentation = Segmentation(
                    points=[(x, y) for x, y in points], label=label
                )
                annotation.segmentations.append(segmentation)

        return annotation

    def save(self, annotation: ImageAnnotation, file_path: Union[str, Path]):
        """保存为LabelMe JSON格式"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 构建LabelMe格式数据
        data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": annotation.image_path.name,
            "imageData": None,
            "imageHeight": annotation.image_height,
            "imageWidth": annotation.image_width,
        }

        # 添加检测框
        for detection in annotation.detections:
            shape = {
                "label": detection.label,
                "points": [
                    [detection.bbox.x, detection.bbox.y],
                    [
                        detection.bbox.x + detection.bbox.width,
                        detection.bbox.y + detection.bbox.height,
                    ],
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
            data["shapes"].append(shape)

        # 添加分割
        for segmentation in annotation.segmentations:
            shape = {
                "label": segmentation.label,
                "points": segmentation.points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
            data["shapes"].append(shape)

        # 保存文件
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise DataFormatError(f"保存LabelMe文件失败: {e}", file_path=str(file_path))

    def validate(self, data: Dict[str, Any]) -> bool:
        """验证LabelMe格式"""
        required_fields = ["imagePath", "imageHeight", "imageWidth"]
        for field in required_fields:
            if field not in data:
                logger.warning(f"缺少必需字段: {field}")
                return False

        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            logger.warning("shapes字段必须是列表")
            return False

        return True


class CocoConverter(BaseDataConverter):
    """COCO格式转换器"""

    @property
    def format_type(self) -> AnnotationFormat:
        return AnnotationFormat.COCO

    def load(self, file_path: Union[str, Path]) -> ImageAnnotation:
        """加载COCO JSON文件（简化实现）"""
        # TODO: 实现COCO格式加载
        raise NotImplementedError("COCO格式转换器待实现")

    def save(self, annotation: ImageAnnotation, file_path: Union[str, Path]):
        """保存为COCO JSON格式（简化实现）"""
        # TODO: 实现COCO格式保存
        raise NotImplementedError("COCO格式转换器待实现")

    def validate(self, data: Dict[str, Any]) -> bool:
        """验证COCO格式"""
        # TODO: 实现COCO格式验证
        return True


class DataConverterFactory:
    """数据转换器工厂"""

    _converters = {
        AnnotationFormat.LABELME: LabelMeConverter,
        AnnotationFormat.COCO: CocoConverter,
    }

    @classmethod
    def create_converter(cls, format_type: AnnotationFormat) -> BaseDataConverter:
        """创建转换器"""
        if format_type not in cls._converters:
            raise DataFormatError(f"不支持的数据格式: {format_type.value}")

        converter_class = cls._converters[format_type]
        return converter_class()

    @classmethod
    def register_converter(cls, format_type: AnnotationFormat, converter_class: type):
        """注册转换器"""
        cls._converters[format_type] = converter_class
        logger.info(
            f"注册数据转换器: {format_type.value} -> {converter_class.__name__}"
        )

    @classmethod
    def get_supported_formats(cls) -> List[AnnotationFormat]:
        """获取支持的格式列表"""
        return list(cls._converters.keys())


class DataProcessor:
    """数据处理器"""

    def __init__(self):
        self.converters = {}
        for format_type in DataConverterFactory.get_supported_formats():
            self.converters[format_type] = DataConverterFactory.create_converter(
                format_type
            )

    def convert_format(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        source_format: AnnotationFormat,
        target_format: AnnotationFormat,
    ):
        """格式转换"""
        if source_format not in self.converters:
            raise DataFormatError(f"不支持的源格式: {source_format.value}")

        if target_format not in self.converters:
            raise DataFormatError(f"不支持的目标格式: {target_format.value}")

        # 加载源文件
        source_converter = self.converters[source_format]
        annotation = source_converter.load(input_file)

        # 保存为目标格式
        target_converter = self.converters[target_format]
        target_converter.save(annotation, output_file)

        logger.info(f"格式转换完成: {source_format.value} -> {target_format.value}")

    def batch_convert(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        source_format: AnnotationFormat,
        target_format: AnnotationFormat,
        pattern: str = "*.json",
    ):
        """批量格式转换"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.glob(pattern))
        if not files:
            logger.warning(f"未找到匹配的文件: {input_dir / pattern}")
            return

        for file_path in files:
            try:
                output_file = output_dir / file_path.name
                self.convert_format(
                    file_path, output_file, source_format, target_format
                )
            except Exception as e:
                logger.error(f"转换文件失败 {file_path}: {e}")

        logger.info(f"批量转换完成: {len(files)}个文件")

    def validate_annotation(self, annotation: ImageAnnotation) -> List[str]:
        """验证标注数据"""
        issues = []

        # 检查图像路径
        if not annotation.image_path.exists():
            issues.append(f"图像文件不存在: {annotation.image_path}")

        # 检查图像尺寸
        if annotation.image_width <= 0 or annotation.image_height <= 0:
            issues.append("图像尺寸无效")

        # 检查检测框
        for i, detection in enumerate(annotation.detections):
            bbox = detection.bbox
            if bbox.x < 0 or bbox.y < 0:
                issues.append(f"检测框{i}坐标为负数")
            if bbox.x + bbox.width > annotation.image_width:
                issues.append(f"检测框{i}超出图像宽度")
            if bbox.y + bbox.height > annotation.image_height:
                issues.append(f"检测框{i}超出图像高度")
            if bbox.width <= 0 or bbox.height <= 0:
                issues.append(f"检测框{i}尺寸无效")
            if not detection.label.strip():
                issues.append(f"检测框{i}标签为空")

        # 检查分割
        for i, segmentation in enumerate(annotation.segmentations):
            if len(segmentation.points) < 3:
                issues.append(f"分割{i}点数不足")
            if not segmentation.label.strip():
                issues.append(f"分割{i}标签为空")

        return issues

    def merge_annotations(self, annotations: List[ImageAnnotation]) -> ImageAnnotation:
        """合并多个标注"""
        if not annotations:
            raise ValidationError("标注列表为空")

        base_annotation = annotations[0]
        merged = ImageAnnotation(
            image_path=base_annotation.image_path,
            image_width=base_annotation.image_width,
            image_height=base_annotation.image_height,
        )

        for annotation in annotations:
            # 验证图像一致性
            if annotation.image_path != base_annotation.image_path:
                raise ValidationError("标注对应的图像不一致")

            merged.detections.extend(annotation.detections)
            merged.segmentations.extend(annotation.segmentations)

        return merged

    def filter_by_confidence(
        self, annotation: ImageAnnotation, threshold: float = 0.5
    ) -> ImageAnnotation:
        """根据置信度过滤标注"""
        filtered = ImageAnnotation(
            image_path=annotation.image_path,
            image_width=annotation.image_width,
            image_height=annotation.image_height,
        )

        # 过滤检测框
        filtered.detections = [
            det for det in annotation.detections if det.confidence >= threshold
        ]

        # 过滤分割
        filtered.segmentations = [
            seg for seg in annotation.segmentations if seg.confidence >= threshold
        ]

        return filtered


# 创建全局数据处理器实例
data_processor = DataProcessor()


def get_data_processor() -> DataProcessor:
    """获取数据处理器实例"""
    return data_processor


def create_converter(format_type: AnnotationFormat) -> BaseDataConverter:
    """创建数据转换器的便捷函数"""
    return DataConverterFactory.create_converter(format_type)
