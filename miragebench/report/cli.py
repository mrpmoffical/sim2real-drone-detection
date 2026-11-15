"""`mb-report`: sonuç JSON'larını toplar, tablo + hipotez kararları + grafikler üretir.

Örnek:
  mb-report --results results/ --out docs/report
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .aggregate import (
    aggregate_by_arm,
    data_efficiency_curve,
    hypothesis_decisions,
    load_results,
    markdown_table,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MirageBench rapor üretici")
    p.add_argument("--results", required=True, help="kol×tohum JSON dizini")
    p.add_argument("--out", default="docs/report")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_results(args.results)
    if not records:
        print("sonuç bulunamadı:", args.results)
        return 1
    agg = aggregate_by_arm(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n_seeds", "n_real_train", "n_synth_train",
                    "map50_mean", "map50_std", "map50_95_mean",
                    "map50_small_mean", "map50_medium_mean", "map50_large_mean"])
        for arm in sorted(agg):
            e = agg[arm]
            m = e["metrics"]
            w.writerow([arm, e["n_seeds"], e["n_real_train"], e["n_synth_train"],
                        m["map50"]["mean"], m["map50"]["std"], m["map50_95"]["mean"],
                        m["map50_small"]["mean"], m["map50_medium"]["mean"], m["map50_large"]["mean"]])

    decisions = hypothesis_decisions(agg)
    (out / "hypothesis_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2))
    (out / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))

    # Markdown rapor
    md = ["# MirageBench — Sonuçlar\n", "## Kol özeti\n", markdown_table(agg), "\n## Hipotez kararları\n"]
    for d in decisions:
        md.append(f"- **{d['hypothesis']}**: {d['verdict']} — {d['detail']}")
    md.append("\n## Veri verimliliği (mAP@50)\n")
    for name, pts in data_efficiency_curve(agg).items():
        if pts:
            md.append(f"- {name}: " + ", ".join(f"N={n}:{m:.3f}" for n, m, _ in pts))
    report_md = "\n".join(md) + "\n"
    (out / "report.md").write_text(report_md)

    if not args.no_plots:
        import contextlib
        try:
            from .plots import plot_ablation, plot_data_efficiency, plot_size_contribution
            plot_data_efficiency(agg, out / "figures" / "data_efficiency.png")
            plot_size_contribution(agg, out / "figures" / "size_contribution.png")
            with contextlib.suppress(RuntimeError):  # ablasyon kolu yoksa atla
                plot_ablation(agg, out / "figures" / "ablation.png")
            print("grafikler ->", out / "figures")
        except RuntimeError as e:
            print(f"grafik atlandı: {e}")

    print(f"rapor -> {out/'report.md'}")
    for d in decisions:
        print(f"  {d['hypothesis']}: {d['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
