"""Değerlendirme için GT ve tahmin yükleme (YOLO txt düzeni).

GT: materialize edilmiş veri setinin bir bölütü (labels/<split>/*.txt, normalize).
Tahmin: dedektörün ürettiği <stem>.txt satırları 'cls xc yc w h conf' (normalize).
Ultralytics `save_txt=True, save_conf=True` tam bu biçimi üretir.

Normalize kutular piksel xyxy'ye çevrilir; bunun için görüntü boyutu gerekir,
görüntü başlığından okunur.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .metrics import Detection, GroundTruth

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _img_size_for(label_path: Path, images_dir: Path) -> tuple[int, int]:
    for ext in IMG_EXTS:
        cand = images_dir / f"{label_path.stem}{ext}"
        if cand.exists():
            with Image.open(cand) as im:
                return im.width, im.height
    raise FileNotFoundError(f"{label_path.stem} için görüntü bulunamadı ({images_dir})")


def _norm_to_xyxy(xc, yc, w, h, iw, ih):
    x1 = (xc - w / 2) * iw
    y1 = (yc - h / 2) * ih
    x2 = (xc + w / 2) * iw
    y2 = (yc + h / 2) * ih
    return (x1, y1, x2, y2)


def load_ground_truth(labels_dir: str | Path, images_dir: str | Path) -> list[GroundTruth]:
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    gts: list[GroundTruth] = []
    for lbl in sorted(labels_dir.glob("*.txt")):
        iw, ih = _img_size_for(lbl, images_dir)
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(float(p[0]))
            xc, yc, w, h = (float(v) for v in p[1:5])
            gts.append(GroundTruth(lbl.stem, cls, _norm_to_xyxy(xc, yc, w, h, iw, ih)))
    return gts


def load_predictions(preds_dir: str | Path, images_dir: str | Path) -> list[Detection]:
    preds_dir = Path(preds_dir)
    images_dir = Path(images_dir)
    dets: list[Detection] = []
    for lbl in sorted(preds_dir.glob("*.txt")):
        iw, ih = _img_size_for(lbl, images_dir)
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 6:
                # conf yoksa 1.0 varsay (uyarı: PR eğrisi anlamsızlaşır)
                if len(p) == 5:
                    cls = int(float(p[0]))
                    xc, yc, w, h = (float(v) for v in p[1:5])
                    dets.append(Detection(lbl.stem, cls, 1.0, _norm_to_xyxy(xc, yc, w, h, iw, ih)))
                continue
            cls = int(float(p[0]))
            xc, yc, w, h = (float(v) for v in p[1:5])
            conf = float(p[5])
            dets.append(Detection(lbl.stem, cls, conf, _norm_to_xyxy(xc, yc, w, h, iw, ih)))
    return dets
