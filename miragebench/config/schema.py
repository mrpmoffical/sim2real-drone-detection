"""Deney yapılandırması için pydantic şemaları.

Buradaki modeller her deney kolunun (`configs/arms/*.yaml`) ve temel config'in
(`configs/base.yaml`) yapısal olarak doğru olmasını garanti eder. Hatalı bir
alan, deney saatlerce sürdükten sonra değil, yükleme anında yakalanır.

Tasarım ilkesi (bkz. docs/RESEARCH.md §7): model/eğitim hiperparametreleri
`base` içinde sabittir; kollar yalnızca `data` bileşimini değiştirir. Şema bunu
yapısal olarak teşvik eder — kol config'i `base`i yeniden tanımlamak zorunda
kalmasın diye `data` ve `meta` alanlarına odaklanır.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SizeBuckets(BaseModel):
    """COCO tarzı alan eşikleri (piksel²). small < small_max² <= medium < medium_max²."""

    small_max: int = 32
    medium_max: int = 96

    @model_validator(mode="after")
    def _ordered(self) -> SizeBuckets:
        if not 0 < self.small_max < self.medium_max:
            raise ValueError("size eşikleri 0 < small_max < medium_max olmalı")
        return self


class ModelConfig(BaseModel):
    """Dedektör mimarisi ve girdi — kollar arası SABİT tutulur."""

    arch: str = "yolo11s"
    imgsz: int = 1280  # küçük nesne için yüksek çözünürlük
    # SAHI dilimli çıkarım (uzak/küçük hedefler). Yalnız değerlendirme/çıkarımda.
    sahi_enabled: bool = False
    sahi_slice: int = 640
    sahi_overlap: float = 0.2


class TrainConfig(BaseModel):
    """Eğitim bütçesi ve hiperparametreler — kollar arası SABİT tutulur.

    `total_steps` kollar arası eşitlenmelidir; aksi halde 'daha çok veri' ile
    'daha çok eğitim' karışır (docs/RESEARCH.md §7).
    """

    epochs: int = 100
    batch: int = 16
    optimizer: Literal["SGD", "Adam", "AdamW", "auto"] = "auto"
    lr0: float = 0.01
    patience: int = 20  # erken durdurma
    # Artırma politikası (Albumentations/Ultralytics) — SABİT. Ablasyon sentetik
    # ÜRETİMİNDE yapılır, eğitim artırmasında değil; ikisi karışmamalı.
    augment_preset: str = "small_object_v1"
    total_steps: int | None = Field(
        default=None,
        description="Eşitlenmiş optimizer adımı bütçesi; None ise epochs*len(data) kullanılır",
    )


class DataSource(str, Enum):
    REAL = "real"
    SYNTH = "synth"


class DataSpec(BaseModel):
    """Bir kolun eğitim verisi bileşimi — manipüle edilen TEK değişken."""

    # Ön-eğitim aşaması (opsiyonel). synth_pretrain kollarında burası synth olur.
    pretrain: DataSource | None = None
    pretrain_count: int | None = None  # None = tümü

    # Ana (ince ayar veya tek aşama) eğitim verisi.
    train_source: Literal["real", "synth", "mixed"] = "real"
    real_count: int | None = None  # None = mevcut tüm gerçek eğitim verisi
    synth_count: int | None = None

    @model_validator(mode="after")
    def _coherent(self) -> DataSpec:
        if self.train_source == "real" and self.synth_count:
            raise ValueError("train_source=real iken synth_count verilemez")
        if self.train_source == "synth" and self.real_count:
            raise ValueError("train_source=synth iken real_count verilemez")
        if self.pretrain is not None and self.pretrain_count is not None and self.pretrain_count <= 0:
            raise ValueError("pretrain_count > 0 olmalı")
        return self


class SplitConfig(BaseModel):
    """Grup-farkında bölme ve sızıntı önleme (docs/RESEARCH.md §5)."""

    train: float = 0.80
    val: float = 0.15
    test: float = 0.05
    group_aware: bool = True
    # Grup kimliğinin dosya adından çıkarılma deseni (ör. video_id).
    group_regex: str = r"^(?P<group>.*?)(?:_frame|_f|_)\d+"
    seed: int = 0

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitConfig:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split oranları 1.0'a toplanmalı (şu an {total})")
        return self


class BaseConfig(BaseModel):
    """Tüm kollarca paylaşılan sabit temel (configs/base.yaml)."""

    project: str = "miragebench"
    classes: list[str] = Field(default_factory=lambda: ["drone"])
    size_buckets: SizeBuckets = Field(default_factory=SizeBuckets)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    # Belirsizlik ölçümü için tohumlar (docs/RESEARCH.md §6).
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])


class ArmConfig(BaseModel):
    """Tek bir deney kolu. `base`i miras alır, yalnız `data` ve `meta`yı özelleştirir."""

    name: str
    description: str = ""
    data: DataSpec
    # Yalnız bu kol için geçersiz kılınabilecek alanlar (dikkatli kullanılmalı —
    # çok fazla override, kontrollü deneyi bozar). Boş bırakmak tercih edilir.
    overrides: dict = Field(default_factory=dict)
    hypothesis_refs: list[str] = Field(default_factory=list)  # ör. ["H1","H2"]

    @model_validator(mode="after")
    def _named(self) -> ArmConfig:
        if not self.name or " " in self.name:
            raise ValueError("kol adı boşluksuz ve dolu olmalı (dosya/dizin adı olarak kullanılır)")
        return self


def load_base(path: str | Path) -> BaseConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return BaseConfig.model_validate(raw)


def load_arm(path: str | Path) -> ArmConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ArmConfig.model_validate(raw)


def merged_config(base: BaseConfig, arm: ArmConfig) -> dict:
    """Kolun override'larını base üzerine uygular; etkin (effective) config döndürür.

    Hangi alanların override edildiğini `_overridden` altında kaydeder; rapor
    bunu kontrollü-deney ihlali uyarısı olarak gösterebilir.
    """
    eff = base.model_dump()
    overridden: list[str] = []
    for key, val in arm.overrides.items():
        if key not in eff:
            raise ValueError(f"override anahtarı base'de yok: {key}")
        if isinstance(eff[key], dict) and isinstance(val, dict):
            eff[key].update(val)
        else:
            eff[key] = val
        overridden.append(key)
    eff["_arm"] = arm.model_dump()
    eff["_overridden"] = overridden
    return eff
