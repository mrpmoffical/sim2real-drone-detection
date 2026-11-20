"""Veri seti kaydının format-bağımsız iç temsili.

Tüm adaptörler (COCO/VOC/YOLO) girişlerini `ImageRecord` dizisine çevirir; hat
bundan sonra formatı bilmez. Kutular YOLO konvansiyonunda normalize saklanır
(xc,yc,w,h ∈ [0,1]); piksel dönüşümü için görüntü boyutu kayıtta tutulur.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Box:
    cls: int
    xc: float
    yc: float
    w: float
    h: float

    def area_px(self, img_w: int, img_h: int) -> float:
        return (self.w * img_w) * (self.h * img_h)

    def to_xyxy_px(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        x1 = (self.xc - self.w / 2) * img_w
        y1 = (self.yc - self.h / 2) * img_h
        x2 = (self.xc + self.w / 2) * img_w
        y2 = (self.yc + self.h / 2) * img_h
        return (x1, y1, x2, y2)

    def is_valid(self) -> bool:
        return (
            0.0 <= self.xc <= 1.0 and 0.0 <= self.yc <= 1.0
            and 0.0 < self.w <= 1.0 and 0.0 < self.h <= 1.0
        )


@dataclass
class ImageRecord:
    image_path: Path
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    source: str = ""          # adaptör/köken etiketi (ör. 'real', 'synth')
    group: str | None = None  # grup-farkında split için (ör. video kimliği)

    def size_bucket_counts(self, small_max: int = 32, medium_max: int = 96) -> dict[str, int]:
        c = {"small": 0, "medium": 0, "large": 0}
        for b in self.boxes:
            a = b.area_px(self.width, self.height)
            if a < small_max * small_max:
                c["small"] += 1
            elif a < medium_max * medium_max:
                c["medium"] += 1
            else:
                c["large"] += 1
        return c


def infer_group(image_path: Path, group_regex: str) -> str:
    """Dosya adından grup (video/çekim) kimliği çıkarır; eşleşmezse dosya kökü.

    Grup-farkında split için kritik: aynı videodan kareler train/test'e
    bölünmemeli (docs/RESEARCH.md §5)."""
    stem = image_path.stem
    m = re.match(group_regex, stem)
    if m and m.groupdict().get("group"):
        return m.group("group")
    # Alternatif: üst dizin adını grup kabul et (klasör-başına-video yapısı)
    parent = image_path.parent.name
    return parent if parent else stem
