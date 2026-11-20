"""DUT Anti-UAV (Pascal VOC XML) -> YOLO düzeni dönüştürücü."""
from __future__ import annotations
import argparse, os, sys
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ("train", "val", "test")

def _find_dir(base, name):
    if (base / name).is_dir():
        return base / name
    for p in base.rglob(name):
        if p.is_dir():
            return p
    return None

def _image_for(xml_path, img_dir):
    try:
        fname = ET.parse(xml_path).getroot().findtext("filename")
    except ET.ParseError:
        fname = None
    cands = []
    if fname:
        cands.append(img_dir / fname)
    cands += [img_dir / f"{xml_path.stem}{e}" for e in IMG_EXTS]
    for c in cands:
        if c.exists():
            return c
    return None

def _boxes(xml_path, w, h):
    out = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        b = obj.find("bndbox")
        if b is None:
            continue
        try:
            xmin=float(b.findtext("xmin","0")); ymin=float(b.findtext("ymin","0"))
            xmax=float(b.findtext("xmax","0")); ymax=float(b.findtext("ymax","0"))
        except ValueError:
            continue
        if xmax<=xmin or ymax<=ymin:
            continue
        xc=(xmin+xmax)/2/w; yc=(ymin+ymax)/2/h; bw=(xmax-xmin)/w; bh=(ymax-ymin)/h
        if 0<bw<=1 and 0<bh<=1 and 0<=xc<=1 and 0<=yc<=1:
            out.append((xc,yc,bw,bh))
    return out

def _xml_size(xml_path):
    size = ET.parse(xml_path).getroot().find("size")
    if size is None:
        return None
    try:
        w=int(float(size.findtext("width","0"))); h=int(float(size.findtext("height","0")))
    except ValueError:
        return None
    return (w,h) if w>0 and h>0 else None

def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil; shutil.copy2(src, dst)

def convert_split(raw, out, split):
    base = raw / split
    if not base.is_dir():
        return {"split":split,"status":"yok","images":0,"boxes":0}
    xml_dir=_find_dir(base,"xml"); img_dir=_find_dir(base,"img")
    if xml_dir is None or img_dir is None:
        return {"split":split,"status":"xml/img bulunamadı","images":0,"boxes":0}
    n_img=n_box=0
    for xml in sorted(xml_dir.glob("*.xml")):
        img=_image_for(xml,img_dir)
        if img is None:
            continue
        size=_xml_size(xml)
        if size is None:
            with Image.open(img) as im: size=(im.width,im.height)
        w,h=size
        bs=_boxes(xml,w,h)
        stem=img.stem
        _link(img, out/"images"/split/f"{stem}{img.suffix.lower()}")
        lbl=out/"labels"/split/f"{stem}.txt"; lbl.parent.mkdir(parents=True,exist_ok=True)
        lbl.write_text("".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n" for a,b,c,d in bs))
        n_img+=1; n_box+=len(bs)
    return {"split":split,"status":"ok","images":n_img,"boxes":n_box,
            "xml_dir":str(xml_dir),"img_dir":str(img_dir)}

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw",default="data/dut_raw"); ap.add_argument("--out",default="data/real")
    a=ap.parse_args(argv)
    raw=Path(a.raw); out=Path(a.out)
    if not raw.is_dir():
        print(f"HATA: {raw} yok",file=sys.stderr); return 1
    stats=[convert_split(raw,out,s) for s in SPLITS]
    import yaml
    out.mkdir(parents=True,exist_ok=True)
    (out/"data.yaml").write_text(yaml.safe_dump(
        {"path":str(out.resolve()),"train":"images/train","val":"images/val",
         "test":"images/test","names":{0:"drone"}}, allow_unicode=True, sort_keys=False))
    print("=== DUT -> YOLO özeti ===")
    for s in stats: print(s)
    ti=sum(s["images"] for s in stats); tb=sum(s["boxes"] for s in stats)
    print(f"TOPLAM: {ti} görüntü, {tb} kutu -> {out/'data.yaml'}")
    return 0 if ti>0 else 1

if __name__=="__main__":
    raise SystemExit(main())
