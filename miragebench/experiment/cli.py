"""`mb-run`: bir veya daha çok kolu, bir veya daha çok tohumla çalıştırır.

Örnek:
  mb-run --arm configs/arms/synth_pretrain_real_250.yaml \
         --real-root data/real --synth full=data/synth/full \
         --seeds 0 1 2 --results results/

Yalnızca veri hazırlığını denemek (GPU olmadan) için --dry-run: materyalize eder,
eğitim/çıkarımı atlar, kompozisyon özetini yazar.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..config import load_arm, load_base
from .runner import prepare_arm_data, run_arm_seed


def _parse_synth(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--synth biçimi faktör=dizin olmalı: {it}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MirageBench deney koşucusu")
    p.add_argument("--arm", required=True, nargs="+", help="kol YAML dosyaları")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--real-root", required=True)
    p.add_argument("--synth", nargs="*", default=[], help="faktör=dizin çiftleri")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="override; verilmezse base.seeds")
    p.add_argument("--work", default="work")
    p.add_argument("--results", default="results")
    p.add_argument("--dry-run", action="store_true",
                   help="sadece veri hazırlığı; eğitim/çıkarım atlanır (GPU'suz)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = load_base(args.base)
    synth_roots = _parse_synth(args.synth)
    seeds = args.seeds if args.seeds is not None else base.seeds

    for arm_path in args.arm:
        arm = load_arm(arm_path)
        for seed in seeds:
            if args.dry_run:
                prep = prepare_arm_data(base, arm, args.real_root, synth_roots, args.work, seed)
                print(json.dumps({
                    "arm": arm.name, "seed": seed, "stage": prep.stage,
                    "n_real_train": prep.n_real_train, "n_synth_train": prep.n_synth_train,
                    "data_yaml": str(prep.data_yaml),
                    "pretrain_yaml": str(prep.pretrain_yaml) if prep.pretrain_yaml else None,
                }, ensure_ascii=False))
                continue
            try:
                rec = run_arm_seed(base, arm, args.real_root, synth_roots,
                                   args.work, args.results, seed)
            except RuntimeError as e:
                print(f"HATA ({arm.name} seed{seed}): {e}", file=sys.stderr)
                return 2
            mt = rec["metrics"]
            print(f"{arm.name} seed{seed}: mAP@50={mt['map50']:.4f} "
                  f"mAP@50-95={mt['map50_95']:.4f} small={mt['map50_small']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
