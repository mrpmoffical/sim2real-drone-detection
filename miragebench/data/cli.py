"""`mb-ingest`: gerçek veri setini okur, grup-farkında böler, YOLO düzenine yazar.

Örnek:
  mb-ingest --format yolo --root /data/anti_uav --out data/real --merge-classes
  mb-ingest --format coco --ann ann.json --images imgs/ --out data/real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import load_base
from . import adapters as A
from .materialize import materialize
from .split import group_aware_split, split_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MirageBench gerçek veri ingest")
    p.add_argument("--format", required=True, choices=["yolo", "coco", "voc"])
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="configs/base.yaml")
    # yolo/voc
    p.add_argument("--root", help="yolo/voc kök dizini")
    # coco
    p.add_argument("--ann", help="coco annotations.json")
    p.add_argument("--images", help="coco görüntü kökü")
    # sınıf eşleme
    p.add_argument("--merge-classes", action="store_true",
                   help="tüm kaynak sınıfları tek 'drone' (0) sınıfına indir")
    p.add_argument("--voc-names", nargs="*", default=["drone", "uav"],
                   help="VOC'ta drone kabul edilecek sınıf adları")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = load_base(args.base)
    gr = base.split.group_regex

    if args.format == "yolo":
        cmap = {k: 0 for k in range(100)} if args.merge_classes else None
        recs = list(A.iter_yolo(args.root, class_map=cmap, group_regex=gr))
    elif args.format == "coco":
        cmap = {k: 0 for k in range(1000)} if args.merge_classes else None
        recs = list(A.iter_coco(args.ann, args.images, class_map=cmap, group_regex=gr))
    else:  # voc
        c2i = {n.lower(): 0 for n in args.voc_names}
        recs = list(A.iter_voc(args.root, c2i, group_regex=gr))

    if not recs:
        print("UYARI: hiç kayıt bulunamadı. Yol ve format doğru mu?", file=sys.stderr)
        return 1

    sp = group_aware_split(recs, base.split.train, base.split.val, base.split.test,
                           seed=base.split.seed)
    rep = split_report(sp)
    dy = materialize(args.out, sp["train"], sp["val"], sp["test"], base.classes)

    (Path(args.out) / "split_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"ingest ok: {len(recs)} kayıt | {rep['counts']} | leak_free={rep['leak_free']}")
    print(f"boyut dağılımı: {rep['size_dist']}")
    print(f"data.yaml -> {dy}")
    if not rep["leak_free"]:
        print(f"DİKKAT: grup sızıntısı var: {rep['group_leak'][:5]} ...", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
