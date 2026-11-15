"""Kamera projeksiyonu ve bounding box hesabı (Blender'dan bağımsız, saf numpy).

Sentetik verinin değeri etiketlerin KUSURSUZ olmasındadır. Bbox'ı render edilmiş
pikselleri eşikleyerek değil, 3B model köşelerini (veya örneklenmiş yüzey
noktalarını) kamera matrisiyle görüntü düzlemine yansıtıp onların eksen-hizalı
sınırını alarak üretiriz. Bu, kısmi saydamlık, gölge ve motion blur'dan etkilenmez.

Blender konvansiyonu:
  - Kamera -Z yönüne bakar, +Y yukarı, +X sağ (kamera yerel uzayı).
  - Görüntüde +Y aşağı olduğu için projeksiyonda y ekseni ters çevrilir.

Pinhole model: sensör genişliği (mm) ve odak uzaklığından (mm) yatay görüş
açısı türetilir; Blender'ın varsayılan sensör uyumu 'HORIZONTAL' kabul edilir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    focal_mm: float
    sensor_width_mm: float = 36.0  # Blender varsayılan

    def focal_px(self) -> tuple[float, float]:
        """Odak uzaklığı piksel cinsinden (fx, fy). Kare piksel varsayımı."""
        fx = self.width * self.focal_mm / self.sensor_width_mm
        fy = fx  # kare piksel
        return fx, fy

    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.sensor_width_mm / (2.0 * self.focal_mm)))

    def K(self) -> np.ndarray:
        fx, fy = self.focal_px()
        cx, cy = self.width / 2.0, self.height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def euler_to_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) -> 3x3 dönme matrisi. Nesneyi dünyada döndürür."""
    y, p, r = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    return Rz @ Ry @ Rx


def project_points(
    points_world: np.ndarray,
    cam_location: np.ndarray,
    cam_rotation: np.ndarray,
    intr: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Dünya noktalarını görüntü düzlemine yansıtır.

    Dönüş: (uv [N,2] piksel, in_front [N] bool — kameranın önünde mi).
    cam_rotation: kamera YEREL->DÜNYA dönme matrisi (Blender obj.matrix_world'ün 3x3'ü).
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    # Dünya -> kamera yerel: R^T (p - t)
    rel = (pts - cam_location.reshape(1, 3)) @ cam_rotation  # = (R^T (p-t))^T
    # Blender kamerası -Z'ye bakar: önde olmak için z < 0.
    z_cam = rel[:, 2]
    in_front = z_cam < -1e-9
    fx, fy = intr.focal_px()
    cx, cy = intr.width / 2.0, intr.height / 2.0
    # Perspektif bölme; -z ile (çünkü ileri -Z).
    denom = -z_cam
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    u = cx + fx * (rel[:, 0] / denom)
    v = cy - fy * (rel[:, 1] / denom)  # +Y yukarı -> görüntüde ters
    uv = np.stack([u, v], axis=1)
    return uv, in_front


def bbox_from_points(
    uv: np.ndarray,
    in_front: np.ndarray,
    img_w: int,
    img_h: int,
    min_side_px: int = 2,
) -> tuple[float, float, float, float] | None:
    """Yansıtılmış noktaların eksen-hizalı sınırını görüntüye kırparak bbox verir.

    Dönüş: (x1, y1, x2, y2) piksel, veya None (nesne tamamen kadraj dışı/arkada).
    """
    if in_front.sum() == 0:
        return None
    pts = uv[in_front]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    # Görüntüye kırp
    x1c, y1c = max(0.0, x1), max(0.0, y1)
    x2c, y2c = min(float(img_w), x2), min(float(img_h), y2)
    if x2c - x1c < min_side_px or y2c - y1c < min_side_px:
        return None
    return (x1c, y1c, x2c, y2c)


def to_yolo(
    bbox_xyxy: tuple[float, float, float, float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """(x1,y1,x2,y2) piksel -> YOLO (xc,yc,w,h) normalize [0,1]."""
    x1, y1, x2, y2 = bbox_xyxy
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (xc, yc, w, h)


def distance_for_target_px(
    object_extent_m: float, target_px: float, intr: CameraIntrinsics
) -> float:
    """Bir nesnenin görüntüde ~target_px uzunlukta görünmesi için gereken mesafe.

    Küçük-nesne rejimini (ör. 16 px) kontrollü örneklemek için kullanılır.
    projelen_px ≈ f_px * extent_m / distance_m  ->  distance = f_px * extent / target_px
    """
    fx, _ = intr.focal_px()
    return fx * object_extent_m / max(1e-6, target_px)
