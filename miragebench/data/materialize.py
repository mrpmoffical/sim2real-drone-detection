"""ImageRecord split'lerini Ultralytics YOLO düzenine yazar + data.yaml üretir.

Görüntüler kopyalanmaz; sembolik bağ (symlink) kurulur (disk tasarrufu, hız).
Etiketler normalize YOLO formatında yeniden yazılır (kaynak format ne olursa olsun).
Bir kolun eğitim verisi 'real + synth' karışımıysa, iki kaynak aynı train
bölütünde birleştirilir; val/test HER ZAMAN yalnız gerçek kalır (domain gap'i
gerçek dağılımda ölçmek için).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .records import ImageRecord


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def _write_record(rec: ImageRecord, split: str, root: Path, idx: int, prefix: str) -> None:
    stem = f"{prefix}_{idx:07d}"
    img_dst = root / "images" / split / f"{stem}{rec.image_path.suffix.lower()}"
    lbl_dst = root / "labels" / split / f"{stem}.txt"
    _link_or_copy(rec.image_path, img_dst)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{b.cls} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}" for b in rec.boxes]
    lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def materialize(
    out_root: str | Path,
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    test_records: list[ImageRecord],
    class_names: list[str],
    synth_train_records: list[ImageRecord] | None = None,
) -> Path:
    """YOLO veri setini diske yazar ve data.yaml yolunu döndürür."""
    out_root = Path(out_root)
    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (out_root / sub / sp).mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(train_records):
        _write_record(r, "train", out_root, i, "real")
    if synth_train_records:
        for i, r in enumerate(synth_train_records):
            _write_record(r, "train", out_root, i, "synth")
    for i, r in enumerate(val_records):
        _write_record(r, "val", out_root, i, "real")
    for i, r in enumerate(test_records):
        _write_record(r, "test", out_root, i, "real")

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False))
    return yaml_path
