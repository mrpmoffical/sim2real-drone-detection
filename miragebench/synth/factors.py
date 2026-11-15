"""Sentetik üretim faktörleri — domain randomization'ın kontrol edilebilir eksenleri.

Her faktör ablasyonda tek tek KAPATILABİLİR (docs/RESEARCH.md §4). Bir faktörü
kapatmak, onu tek bir "nominal" değere sabitlemek demektir (çeşitlilik = 0).
Bu modül Blender'dan bağımsızdır (saf Python); hem gerçek render hem de GPU'suz
test (sahne parametre örnekleme) aynı mantığı kullanır.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class FactorSwitches:
    """Ablasyon anahtarları. True = faktör randomize (tam çeşitlilik)."""

    bg_variation: bool = True       # arka plan HDRI / görüntü çeşitliliği
    light_random: bool = True       # ışık yönü, şiddeti, rengi
    pose_scale_random: bool = True  # drone yönelimi + kameraya uzaklık (ölçek)
    texture_realism: bool = True    # yüksek foto-gerçekçi doku vs düz/düşük

    @classmethod
    def from_name(cls, name: str) -> FactorSwitches:
        """Ablasyon kol adından anahtar üretir, ör. 'no_light_random'."""
        s = cls()
        mapping = {
            "no_bg_variation": "bg_variation",
            "no_light_random": "light_random",
            "no_pose_scale_random": "pose_scale_random",
            "low_texture_realism": "texture_realism",
        }
        if name in mapping:
            setattr(s, mapping[name], False)
        elif name not in ("full", "synth_only"):
            raise ValueError(f"bilinmeyen ablasyon faktörü: {name}")
        return s


@dataclass
class SceneRanges:
    """Örnekleme aralıkları. Faktör kapatılınca ilgili aralık nominale çöker."""

    # Kamera-drone mesafesi (metre). Küçük nesne rejimi için geniş üst sınır.
    distance_m: tuple[float, float] = (3.0, 120.0)
    # Drone yönelimi (derece): yaw/pitch/roll.
    yaw_deg: tuple[float, float] = (0.0, 360.0)
    pitch_deg: tuple[float, float] = (-40.0, 40.0)
    roll_deg: tuple[float, float] = (-30.0, 30.0)
    # Işık
    sun_elevation_deg: tuple[float, float] = (5.0, 85.0)
    sun_azimuth_deg: tuple[float, float] = (0.0, 360.0)
    sun_strength: tuple[float, float] = (0.5, 6.0)
    sun_warmth: tuple[float, float] = (0.0, 1.0)  # 0 soğuk .. 1 sıcak
    # Kamera odak uzaklığı (mm) — ölçek çeşitliliğini artırır
    focal_mm: tuple[float, float] = (24.0, 85.0)

    # Nominal (faktör kapalıyken kullanılacak sabit) değerler
    nominal_distance_m: float = 30.0
    nominal_focal_mm: float = 50.0


@dataclass
class SampledScene:
    """Tek bir render için örneklenmiş parametreler. Blender'a bu verilir."""

    distance_m: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    sun_elevation_deg: float
    sun_azimuth_deg: float
    sun_strength: float
    sun_warmth: float
    focal_mm: float
    background_id: int | None
    texture_realism: bool
    asset_name: str
    paint_rgb: tuple[int, int, int] | None = None
    extras: dict = field(default_factory=dict)


class SceneSampler:
    """Faktör anahtarlarına göre sahne parametreleri üretir (deterministik, seed'li)."""

    def __init__(
        self,
        switches: FactorSwitches,
        ranges: SceneRanges | None = None,
        n_backgrounds: int = 50,
        assets: list[str] | None = None,
        seed: int = 0,
    ):
        self.sw = switches
        self.r = ranges or SceneRanges()
        self.n_backgrounds = max(1, n_backgrounds)
        self.assets = assets or ["quadcopter_generic"]
        self.rng = random.Random(seed)

    def _u(self, lo_hi: tuple[float, float], enabled: bool, nominal: float) -> float:
        if not enabled:
            return nominal
        lo, hi = lo_hi
        return self.rng.uniform(lo, hi)

    def sample(self) -> SampledScene:
        r, sw = self.r, self.sw

        distance = self._u(r.distance_m, sw.pose_scale_random, r.nominal_distance_m)
        focal = self._u(r.focal_mm, sw.pose_scale_random, r.nominal_focal_mm)

        if sw.pose_scale_random:
            yaw = self.rng.uniform(*r.yaw_deg)
            pitch = self.rng.uniform(*r.pitch_deg)
            roll = self.rng.uniform(*r.roll_deg)
        else:
            yaw = pitch = roll = 0.0

        if sw.light_random:
            sun_el = self.rng.uniform(*r.sun_elevation_deg)
            sun_az = self.rng.uniform(*r.sun_azimuth_deg)
            sun_str = self.rng.uniform(*r.sun_strength)
            sun_warm = self.rng.uniform(*r.sun_warmth)
        else:
            sun_el, sun_az, sun_str, sun_warm = 45.0, 180.0, 3.0, 0.5

        bg_id = self.rng.randrange(self.n_backgrounds) if sw.bg_variation else 0
        asset = self.rng.choice(self.assets) if len(self.assets) > 1 else self.assets[0]

        return SampledScene(
            distance_m=distance,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            sun_elevation_deg=sun_el,
            sun_azimuth_deg=sun_az,
            sun_strength=sun_str,
            sun_warmth=sun_warm,
            focal_mm=focal,
            background_id=bg_id,
            texture_realism=sw.texture_realism,
            asset_name=asset,
        )

    def sun_direction(self, scene: SampledScene) -> tuple[float, float, float]:
        """Güneş elevasyon/azimut -> birim yön vektörü (Blender dünya uzayı)."""
        el = math.radians(scene.sun_elevation_deg)
        az = math.radians(scene.sun_azimuth_deg)
        x = math.cos(el) * math.cos(az)
        y = math.cos(el) * math.sin(az)
        z = math.sin(el)
        return (x, y, z)
