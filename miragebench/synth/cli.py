"""`mb-synth` komut satırı girişi.

İki çalışma biçimi:
  1. bpy modülü kuruluysa doğrudan: `python -m miragebench.synth.cli --n 1000 --out data/synth/full`
  2. Blender çalıştırılabiliriyle:
     `blender --background --python -m miragebench.synth.cli -- --n 1000 --out ...`
     (argümanlar '--' sonrasına yazılır)

Ablasyon kolları için --factor: full | no_bg_variation | no_light_random |
no_pose_scale_random | low_texture_realism
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MirageBench sentetik drone üretici")
    p.add_argument("--out", required=True, help="çıktı dizini")
    p.add_argument("--n", type=int, default=1000, help="üretilecek örnek sayısı")
    p.add_argument("--factor", default="full",
                   help="domain-randomization ablasyon anahtarı")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--engine", default="BLENDER_EEVEE", choices=["BLENDER_EEVEE", "CYCLES"])
    p.add_argument("--samples", type=int, default=32, help="Cycles örnek sayısı")
    p.add_argument("--bg-dir", default=None, help="HDRI/arka plan görüntü dizini")
    p.add_argument("--assets", nargs="*", default=None,
                   help="varlık adları; 'asset:/yol/model.glb' biçimi dosya yükler")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--class-id", type=int, default=0)
    return p


def _split_argv(argv: list[str]) -> list[str]:
    # Blender 'blender ... -- <bizim argümanlar>' biçiminde çağırır.
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return argv


def main(argv: list[str] | None = None) -> int:
    argv = _split_argv(argv if argv is not None else sys.argv[1:])
    args = build_parser().parse_args(argv)

    try:
        from .blender_generator import generate
    except Exception as e:
        print(f"HATA: Blender üretici yüklenemedi ({e}). bpy gerekli.", file=sys.stderr)
        return 2

    stats = generate(
        out_dir=Path(args.out),
        n=args.n,
        factor_name=args.factor,
        width=args.width,
        height=args.height,
        engine=args.engine,
        samples=args.samples,
        bg_dir=args.bg_dir,
        assets=args.assets,
        seed=args.seed,
        class_id=args.class_id,
    )
    print(f"Üretildi: {stats['rendered']}/{stats['requested']} "
          f"(atlanan {stats['skipped']}) faktör={stats['factor']} "
          f"boyut={stats['buckets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
