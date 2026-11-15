"""Ultralytics YOLO eğitim/çıkarım sarmalayıcısı.

Bu modül GPU ve `ultralytics` gerektirir; RTX 4070'li dağıtım makinesinde çalışır.
Burada (GPU'suz) import edilebilir ama çağrıldığında net bir hata verir. Deney
mantığı (adım-bütçesi eşitleme, ön-eğitim→ince-ayar sırası) GPU'dan bağımsız
olduğu için ayrı tutulmuş ve `tests/test_step_budget.py` ile test edilir.

Kritik kontrol (docs/RESEARCH.md §7): tüm kollar aynı TOPLAM optimizer adımı
bütçesinde eğitilir. Az görüntülü kolda epoch sayısı otomatik artırılır ki
'daha çok veri' etkisi 'daha çok eğitim' etkisiyle karışmasın.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


def epochs_for_budget(n_images: int, batch: int, total_steps: int,
                      min_epochs: int = 10, max_epochs: int = 600) -> int:
    """Toplam optimizer adımını sabit tutacak epoch sayısı.

    steps_per_epoch = ceil(n_images / batch); epochs = total_steps / steps_per_epoch.
    Uç değerler makul sınırlara kırpılır (çok küçük sette sonsuz epoch'u önler)."""
    if n_images <= 0:
        raise ValueError("n_images > 0 olmalı")
    steps_per_epoch = max(1, math.ceil(n_images / batch))
    ep = round(total_steps / steps_per_epoch)
    return int(max(min_epochs, min(max_epochs, ep)))


def count_images(data_split_dir: str | Path) -> int:
    d = Path(data_split_dir)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in exts)


@dataclass
class TrainResult:
    weights_path: Path
    epochs: int
    n_train_images: int


def _require_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except ImportError as e:
        raise RuntimeError(
            "Bu adım `ultralytics` ve bir GPU gerektirir. "
            "Kurun: pip install 'miragebench[train]' ve RTX 4070'li makinede çalıştırın."
        ) from e


def train_stage(
    data_yaml: str | Path,
    arch: str,
    imgsz: int,
    batch: int,
    total_steps: int,
    project_dir: str | Path,
    name: str,
    seed: int,
    lr0: float = 0.01,
    optimizer: str = "auto",
    patience: int = 20,
    init_weights: str | Path | None = None,
    augment_preset: str = "small_object_v1",
) -> TrainResult:
    """Tek eğitim aşaması. init_weights verilirse ince ayar (transfer) yapılır."""
    YOLO = _require_ultralytics()
    data_yaml = Path(data_yaml)
    train_dir = data_yaml.parent / "images" / "train"
    n_images = count_images(train_dir)
    epochs = epochs_for_budget(n_images, batch, total_steps)

    model = YOLO(str(init_weights) if init_weights else f"{arch}.pt")
    aug = _augment_kwargs(augment_preset)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        optimizer=optimizer,
        patience=patience,
        seed=seed,
        project=str(project_dir),
        name=name,
        exist_ok=True,
        verbose=False,
        **aug,
    )
    weights = Path(project_dir) / name / "weights" / "best.pt"
    return TrainResult(weights, epochs, n_images)


def _augment_kwargs(preset: str) -> dict:
    """Eğitim artırma politikası — kollar arası SABİT. Küçük nesne için ayarlı.

    Not: Sentetik üretim ablasyonu (bg/light/pose/texture) bu eğitim
    artırmasından BAĞIMSIZDIR; ikisi karışmamalı (docs/RESEARCH.md §4)."""
    if preset == "small_object_v1":
        return {
            "mosaic": 1.0,        # küçük nesne için mozaik faydalı
            "close_mosaic": 10,   # son 10 epoch mozaik kapalı
            "scale": 0.5,
            "fliplr": 0.5,
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
            "translate": 0.1,
            "copy_paste": 0.3,    # küçük nesne kopyala-yapıştır
        }
    if preset == "none":
        return {"mosaic": 0.0, "scale": 0.0, "fliplr": 0.0, "copy_paste": 0.0,
                "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "translate": 0.0}
    raise ValueError(f"bilinmeyen augment_preset: {preset}")


def predict_to_txt(
    weights: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    imgsz: int,
    conf: float = 0.001,          # düşük eşik: PR eğrisi için tüm tespitler
    iou: float = 0.6,
    sahi_enabled: bool = False,
    sahi_slice: int = 640,
    sahi_overlap: float = 0.2,
    class_names: list[str] | None = None,
) -> Path:
    """Test görüntülerinde tahmin üretir, 'cls xc yc w h conf' YOLO txt yazar.

    sahi_enabled ise dilimli çıkarım (SAHI) kullanılır — uzak/küçük hedefler için.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if sahi_enabled:
        return _predict_sahi(weights, images_dir, out_dir, imgsz, conf, iou,
                             sahi_slice, sahi_overlap, class_names)
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for img in sorted(Path(images_dir).rglob("*")):
        if img.suffix.lower() not in exts:
            continue
        r = model.predict(str(img), imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        lines = []
        for b in r.boxes:
            cls = int(b.cls.item())
            xc, yc, w, h = b.xywhn[0].tolist()
            cf = float(b.conf.item())
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {cf:.6f}")
        (out_dir / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return out_dir


def _predict_sahi(weights, images_dir, out_dir, imgsz, conf, iou,
                  slice_px, overlap, class_names) -> Path:
    try:
        from sahi import AutoDetectionModel  # type: ignore
        from sahi.predict import get_sliced_prediction  # type: ignore
    except ImportError as e:
        raise RuntimeError("SAHI gerekli: pip install sahi") from e
    from PIL import Image
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(weights), confidence_threshold=conf)
    out_dir = Path(out_dir)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for img in sorted(Path(images_dir).rglob("*")):
        if img.suffix.lower() not in exts:
            continue
        with Image.open(img) as im:
            iw, ih = im.width, im.height
        res = get_sliced_prediction(
            str(img), model, slice_height=slice_px, slice_width=slice_px,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap, verbose=0)
        lines = []
        for p in res.object_prediction_list:
            x1, y1, x2, y2 = p.bbox.to_xyxy()
            xc = (x1 + x2) / 2 / iw
            yc = (y1 + y2) / 2 / ih
            w = (x2 - x1) / iw
            h = (y2 - y1) / ih
            lines.append(f"{p.category.id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {p.score.value:.6f}")
        (out_dir / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return out_dir
