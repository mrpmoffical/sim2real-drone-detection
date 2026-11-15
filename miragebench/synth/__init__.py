"""Sentetik veri üretimi: faktörler, geometri, varlıklar ve Blender üretici.

geometry.py ve factors.py saf-Python/numpy olduğu için bpy olmadan da import
edilebilir ve test edilebilir. blender_generator ve assets yalnız bpy ortamında
çalışır; import hatası vermemesi için bpy'yi opsiyonel tutarlar.
"""

from .factors import FactorSwitches, SampledScene, SceneRanges, SceneSampler
from .geometry import (
    CameraIntrinsics,
    bbox_from_points,
    distance_for_target_px,
    euler_to_matrix,
    project_points,
    to_yolo,
)

__all__ = [
    "CameraIntrinsics",
    "FactorSwitches",
    "SampledScene",
    "SceneRanges",
    "SceneSampler",
    "bbox_from_points",
    "distance_for_target_px",
    "euler_to_matrix",
    "project_points",
    "to_yolo",
]
