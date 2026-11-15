"""Grafikler: veri-verimliliği eğrisi, boyut-katmanlı katkı, ablasyon sıralaması.

matplotlib opsiyoneldir; yoksa fonksiyonlar açık bir hata verir ve rapor metin
(markdown/JSON) kısmı yine de üretilir.
"""

from __future__ import annotations

from pathlib import Path

from .aggregate import ablation_ranking, data_efficiency_curve


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise RuntimeError("Grafikler için matplotlib gerekli: pip install 'miragebench[report]'") from e


def plot_data_efficiency(agg: dict, out_path: str | Path, metric: str = "map50") -> Path:
    plt = _require_mpl()
    curves = data_efficiency_curve(agg, metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, pts in curves.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        label = {"real_only": "Gerçek-only", "synth_pretrain": "Sentetik ön-eğitim + gerçek"}[name]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Gerçek eğitim görüntüsü sayısı (log)")
    ax.set_ylabel(f"{metric} (gerçek test seti)")
    ax.set_title("Veri verimliliği: sentetik ön-eğitimin katkısı")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_size_contribution(agg: dict, out_path: str | Path,
                           sp_arm: str = "synth_pretrain_real_250",
                           ro_arm: str = "real_only_250") -> Path:
    """Boyut katmanına göre sentetik ön-eğitimin kazanımı (puan)."""
    plt = _require_mpl()
    sizes = ["small", "medium", "large"]
    gains = []
    for sz in sizes:
        sp = agg.get(sp_arm, {}).get("metrics", {}).get(f"map50_{sz}", {}).get("mean")
        ro = agg.get(ro_arm, {}).get("metrics", {}).get(f"map50_{sz}", {}).get("mean")
        gains.append((sp - ro) * 100 if sp is not None and ro is not None else 0.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#c44", "#4a7", "#47c"]
    ax.bar(sizes, gains, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("mAP@50 kazanımı (puan)")
    ax.set_title("Sentetik ön-eğitim katkısı — boyut katmanına göre")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_ablation(agg: dict, out_path: str | Path, metric: str = "map50") -> Path:
    plt = _require_mpl()
    rows = ablation_ranking(agg, metric)
    if not rows:
        raise RuntimeError("ablasyon kolu bulunamadı")
    factors = [r["factor"] for r in rows]
    drops = [r["drop_vs_ref"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(factors, drops, color="#a55")
    ax.set_xlabel(f"{metric} düşüşü (puan) — faktör kapatılınca")
    ax.set_title("Domain-randomization faktör önemi (ablasyon)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
