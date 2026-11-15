"""`mb-eval`: bir tahmin dizinini GT'ye karşı değerlendirir, boyut-katmanlı mAP yazar.

Örnek:
  mb-eval --gt data/real/labels/test --images data/real/images/test \
          --preds runs/real_only_250/preds --classes 1 --out results/real_only_250.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_ground_truth, load_predictions
from .metrics import IOU_50_95, evaluate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MirageBench boyut-katmanlı mAP değerlendirme")
    p.add_argument("--gt", required=True, help="GT etiket dizini (labels/<split>)")
    p.add_argument("--images", required=True, help="görüntü dizini (images/<split>)")
    p.add_argument("--preds", required=True, help="tahmin txt dizini (conf içeren)")
    p.add_argument("--classes", type=int, default=1, help="sınıf sayısı")
    p.add_argument("--out", default=None, help="JSON çıktı yolu")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gts = load_ground_truth(args.gt, args.images)
    dets = load_predictions(args.preds, args.images)
    res = evaluate(dets, gts, list(range(args.classes)), iou_thrs=IOU_50_95)
    res["n_gt"] = len(gts)
    res["n_det"] = len(dets)

    print(f"mAP@50={res['map50']:.4f}  mAP@50-95={res['map50_95']:.4f}")
    print(f"  small={res['map50_small']:.4f}  medium={res['map50_medium']:.4f}  "
          f"large={res['map50_large']:.4f}  (mAP@50, boyut katmanı)")
    print(f"  GT={len(gts)} tespit={len(dets)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
