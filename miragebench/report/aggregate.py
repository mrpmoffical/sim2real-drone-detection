"""Sonuç toplama: kol × tohum JSON'larından özet tablo ve hipotez kararları.

Girdi: results/ içinde `<arm>_seed<k>.json` dosyaları (runner çıktısı).
Çıktı: kol başına ortalama ± std metrikler, veri-verimliliği eğrisi noktaları,
ablasyon sıralaması ve docs/RESEARCH.md §6'daki karar kurallarına göre hipotez
sonuçları.

Hiçbir yorum katmaz; yalnızca önceden tanımlı kuralları uygular.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

METRIC_KEYS = ["map50", "map50_95", "map50_small", "map50_medium", "map50_large"]


def load_results(results_dir: str | Path) -> list[dict]:
    out = []
    for f in sorted(Path(results_dir).glob("*_seed*.json")):
        out.append(json.loads(f.read_text()))
    return out


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return (m, math.sqrt(var))


def aggregate_by_arm(records: list[dict]) -> dict[str, dict]:
    """Kol başına metrik ortalaması ± std ve meta."""
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)
    agg: dict[str, dict] = {}
    for arm, recs in by_arm.items():
        entry: dict = {"n_seeds": len(recs),
                       "n_real_train": recs[0].get("n_real_train"),
                       "n_synth_train": recs[0].get("n_synth_train"),
                       "hypothesis_refs": recs[0].get("hypothesis_refs", []),
                       "metrics": {}}
        for k in METRIC_KEYS:
            vals = [r["metrics"][k] for r in recs if k in r["metrics"]]
            m, s = _mean_std(vals)
            entry["metrics"][k] = {"mean": round(m, 4), "std": round(s, 4), "vals": vals}
        agg[arm] = entry
    return agg


def _real_count_from_name(arm: str) -> int | None:
    m = re.search(r"real_(\d+)$", arm)
    return int(m.group(1)) if m else None


def data_efficiency_curve(agg: dict[str, dict], metric: str = "map50") -> dict[str, list[tuple[int, float, float]]]:
    """real_only ve synth_pretrain eğrileri: (N gerçek görüntü, mean, std) noktaları."""
    curves: dict[str, list[tuple[int, float, float]]] = {"real_only": [], "synth_pretrain": []}
    for arm, e in agg.items():
        n = e.get("n_real_train")
        if arm.startswith("real_only_"):
            key = "real_only"
        elif arm.startswith("synth_pretrain_real_"):
            key = "synth_pretrain"
        else:
            continue
        mv = e["metrics"][metric]
        if n is not None:
            curves[key].append((n, mv["mean"], mv["std"]))
    for k in curves:
        curves[k].sort(key=lambda t: t[0])
    return curves


def ablation_ranking(
    agg: dict[str, dict], metric: str = "map50", reference: str = "synth_pretrain_real_250"
) -> list[dict]:
    """Ablasyon: her faktörü KAPATMANIN referansa göre metrik düşüşü.

    Daha büyük düşüş = daha önemli faktör. Referans tam-randomizasyon (full) kolu.
    """
    ref = agg.get(reference)
    if ref is None:
        return []
    ref_m = ref["metrics"][metric]["mean"]
    rows = []
    for arm, e in agg.items():
        if not arm.startswith("ablation_"):
            continue
        factor = arm[len("ablation_"):]
        m = e["metrics"][metric]["mean"]
        rows.append({"factor": factor, "map": round(m, 4),
                     "drop_vs_ref": round(ref_m - m, 4)})
    rows.sort(key=lambda r: r["drop_vs_ref"], reverse=True)
    return rows


def hypothesis_decisions(agg: dict[str, dict]) -> list[dict]:
    """docs/RESEARCH.md §6 karar kurallarını uygular. Yorum yok, sadece kural."""
    out = []

    # H1: synth_pretrain_real_N, real_only_N'i N<=250'de mAP@50'de >=3 puan geçerse destek
    def get(arm, metric="map50"):
        return agg.get(arm, {}).get("metrics", {}).get(metric, {}).get("mean")

    h1_pairs = []
    for arm in agg:
        n = _real_count_from_name(arm)
        if arm.startswith("synth_pretrain_real_") and n is not None and n <= 250:
            sp = get(arm)
            ro = get(f"real_only_{n}")
            if sp is not None and ro is not None:
                h1_pairs.append((n, sp, ro, (sp - ro) * 100))
    if h1_pairs:
        best = max(h1_pairs, key=lambda t: t[3])
        verdict = "desteklendi" if any(d >= 3.0 for *_x, d in h1_pairs) else "desteklenmedi"
        out.append({"hypothesis": "H1", "verdict": verdict,
                    "detail": f"N<=250'de en büyük kazanım {best[3]:.1f} puan (N={best[0]})",
                    "pairs": [{"N": n, "synth_pretrain": round(sp, 4),
                               "real_only": round(ro, 4), "gain_pts": round(d, 2)}
                              for n, sp, ro, d in h1_pairs]})
    else:
        out.append({"hypothesis": "H1", "verdict": "sonuçsuz", "detail": "eşleşen N bulunamadı"})

    # H2: synth_only, real_only'nin belirgin altında mı (büyük domain gap)
    so = get("synth_only")
    ro_all = [get(a) for a in agg if a.startswith("real_only_")]
    ro_all = [x for x in ro_all if x is not None]
    if so is not None and ro_all:
        best_real = max(ro_all)
        gap = (best_real - so) * 100
        verdict = "desteklendi" if gap > 0 else "desteklenmedi"
        out.append({"hypothesis": "H2", "verdict": verdict,
                    "detail": f"synth_only mAP@50={so:.3f}, en iyi real_only={best_real:.3f}, "
                              f"açık={gap:.1f} puan"})
    else:
        out.append({"hypothesis": "H2", "verdict": "sonuçsuz", "detail": "synth_only veya real_only yok"})

    # H3: ablasyonda bg/light, texture'dan daha mı önemli
    rank = ablation_ranking(agg)
    if rank:
        top = rank[0]["factor"]
        tex_drop = next((r["drop_vs_ref"] for r in rank if "texture" in r["factor"]), None)
        light_bg = [r for r in rank if "light" in r["factor"] or "bg" in r["factor"]]
        verdict = "sonuçsuz"
        if light_bg and tex_drop is not None:
            max_lb = max(r["drop_vs_ref"] for r in light_bg)
            verdict = "desteklendi" if max_lb > tex_drop else "desteklenmedi"
        out.append({"hypothesis": "H3", "verdict": verdict,
                    "detail": f"en önemli faktör '{top}'; sıralama={[(r['factor'], r['drop_vs_ref']) for r in rank]}"})
    else:
        out.append({"hypothesis": "H3", "verdict": "sonuçsuz", "detail": "ablasyon kolu yok"})

    # H4: synth katkısı en küçük boyutta en yüksek mi
    # katkı = synth_pretrain - real_only, boyut başına
    h4_arm = next((a for a in agg if a.startswith("synth_pretrain_real_")), None)
    if h4_arm:
        n = _real_count_from_name(h4_arm)
        ro_arm = f"real_only_{n}"
        if ro_arm in agg:
            gains = {}
            for sz in ["small", "medium", "large"]:
                sp = get(h4_arm, f"map50_{sz}")
                ro = get(ro_arm, f"map50_{sz}")
                if sp is not None and ro is not None:
                    gains[sz] = round((sp - ro) * 100, 2)
            if gains:
                top_sz = max(gains, key=gains.get)
                verdict = "desteklendi" if top_sz == "small" else "desteklenmedi"
                out.append({"hypothesis": "H4", "verdict": verdict,
                            "detail": f"boyut başına kazanım (puan)={gains}; en yüksek={top_sz}"})
            else:
                out.append({"hypothesis": "H4", "verdict": "sonuçsuz", "detail": "boyut metrikleri eksik"})
        else:
            out.append({"hypothesis": "H4", "verdict": "sonuçsuz", "detail": f"{ro_arm} yok"})
    else:
        out.append({"hypothesis": "H4", "verdict": "sonuçsuz", "detail": "synth_pretrain kolu yok"})

    return out


def markdown_table(agg: dict[str, dict]) -> str:
    """Kol özet tablosunu markdown olarak döndürür."""
    lines = ["| Kol | tohum | gerçek | sentetik | mAP@50 | mAP@50-95 | small | medium | large |",
             "|-----|------:|-------:|---------:|-------:|----------:|------:|-------:|------:|"]
    for arm in sorted(agg):
        e = agg[arm]
        m = e["metrics"]
        def cell(k, m=m):
            return f"{m[k]['mean']:.3f}±{m[k]['std']:.3f}"
        lines.append(f"| {arm} | {e['n_seeds']} | {e['n_real_train']} | {e['n_synth_train']} | "
                     f"{cell('map50')} | {cell('map50_95')} | {cell('map50_small')} | "
                     f"{cell('map50_medium')} | {cell('map50_large')} |")
    return "\n".join(lines)
