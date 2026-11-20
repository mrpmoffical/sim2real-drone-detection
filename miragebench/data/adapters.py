"""Gerçek veri seti adaptörleri: çeşitli etiket formatlarını ImageRecord'a çevirir.

Desteklenen formatlar:
  - YOLO        : images/ + labels/ (her .txt: cls xc yc w h, normalize)
  - COCO        : tek annotations.json (bbox [x,y,w,h] piksel, xywh)
  - Pascal VOC  : her görüntü için .xml (bndbox xmin,ymin,xmax,ymax piksel)

Açık drone setleri (Anti-UAV, MAV-VID, Drone-vs-Bird) bu üç formattan birinde ya
da ona kolayca indirgenebilir biçimde gelir. Yeni bir format için yalnızca yeni
bir `iter_*` fonksiyonu eklemek yeterlidir; hattın gerisi değişmez.

Görüntü boyutu etiket dosyalarında her zaman bulunmadığından, gerektiğinde
görüntü başlığından (header) okunur — tüm pikselleri decode etmeden.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from .records import Box, ImageRecord, infer_group

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.width, im.height


def _label_for_image(img: Path, lbl_dir: Path) -> Path:
    parts = img.parts
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        new_parts = list(parts)
        new_parts[idx] = "labels"
        return Path(*new_parts).with_suffix(".txt")
    return lbl_dir / f"{img.stem}.txt"


def iter_yolo(
    root: str | Path,
    class_map: dict[int, int] | None = None,
    group_regex: str = r"^(?P<group>.*?)(?:_frame|_f|_)\d+",
    source: str = "yolo",
) -> Iterator[ImageRecord]:
    """YOLO düzeni: <root>/images/*.jpg + <root>/labels/*.txt.

    class_map verilirse kaynak sınıf id'leri hedefe yeniden eşlenir (ör. çok
    sınıflı seti tek 'drone' sınıfına indirgemek için {k: 0})."""
    root = Path(root)
    img_dir = root / "images" if (root / "images").is_dir() else root
    lbl_dir = root / "labels" if (root / "labels").is_dir() else root
    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = _label_for_image(img, lbl_dir)
        w, h = _image_size(img)
        boxes: list[Box] = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c = int(float(parts[0]))
                if class_map is not None:
                    if c not in class_map:
                        continue
                    c = class_map[c]
                b = Box(c, *(float(x) for x in parts[1:]))
                if b.is_valid():
                    boxes.append(b)
        yield ImageRecord(img, w, h, boxes, source, infer_group(img, group_regex))


def iter_coco(
    annotations_json: str | Path,
    images_root: str | Path,
    class_map: dict[int, int] | None = None,
    group_regex: str = r"^(?P<group>.*?)(?:_frame|_f|_)\d+",
    source: str = "coco",
) -> Iterator[ImageRecord]:
    """COCO: bbox [x,y,w,h] piksel (sol-üst). images_root görüntü dosyalarının kökü."""
    annotations_json = Path(annotations_json)
    images_root = Path(images_root)
    data = json.loads(annotations_json.read_text())
    images = {im["id"]: im for im in data["images"]}
    anns_by_img: dict[int, list] = {}
    for a in data.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)
    for img_id, im in images.items():
        w, h = int(im["width"]), int(im["height"])
        path = images_root / im["file_name"]
        boxes: list[Box] = []
        for a in anns_by_img.get(img_id, []):
            if a.get("iscrowd", 0) == 1:
                continue
            c = int(a["category_id"])
            if class_map is not None:
                if c not in class_map:
                    continue
                c = class_map[c]
            x, y, bw, bh = a["bbox"]
            b = Box(c, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)
            if b.is_valid():
                boxes.append(b)
        yield ImageRecord(path, w, h, boxes, source, infer_group(path, group_regex))


def iter_voc(
    root: str | Path,
    class_to_id: dict[str, int],
    group_regex: str = r"^(?P<group>.*?)(?:_frame|_f|_)\d+",
    source: str = "voc",
) -> Iterator[ImageRecord]:
    """Pascal VOC: <root> içinde eşleşen .xml ve görüntü dosyaları.

    class_to_id: VOC sınıf adı -> hedef sınıf id (ör. {'drone':0,'uav':0})."""
    root = Path(root)
    for xml in sorted(root.rglob("*.xml")):
        tree = ET.parse(xml)
        r = tree.getroot()
        size = r.find("size")
        if size is None:
            continue
        w = int(size.findtext("width", "0"))
        h = int(size.findtext("height", "0"))
        fname = r.findtext("filename", f"{xml.stem}.jpg")
        img_path = xml.parent / fname
        if not img_path.exists():
            cand = [xml.parent / f"{xml.stem}{e}" for e in IMG_EXTS]
            img_path = next((c for c in cand if c.exists()), img_path)
        if (w == 0 or h == 0) and img_path.exists():
            w, h = _image_size(img_path)
        boxes: list[Box] = []
        for obj in r.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name not in class_to_id:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
            b = Box(class_to_id[name], (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h,
                    (xmax - xmin) / w, (ymax - ymin) / h)
            if b.is_valid():
                boxes.append(b)
        yield ImageRecord(img_path, w, h, boxes, source, infer_group(img_path, group_regex))


ADAPTERS = {"yolo": iter_yolo, "coco": iter_coco, "voc": iter_voc}
