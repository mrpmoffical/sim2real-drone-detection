import json

import pytest
from PIL import Image

from miragebench.config import load_arm, load_base
from miragebench.experiment.runner import prepare_arm_data
from miragebench.experiment.train_wrapper import epochs_for_budget
from miragebench.report.aggregate import (
    aggregate_by_arm,
    data_efficiency_curve,
    hypothesis_decisions,
)


def test_epochs_for_budget_monotonic():
    # az veri -> çok epoch, çok veri -> az epoch
    e_small = epochs_for_budget(50, 16, 40000)
    e_large = epochs_for_budget(5000, 16, 40000)
    assert e_small > e_large
    assert e_large >= 10


def test_epochs_clipped():
    assert epochs_for_budget(1, 16, 10**9) <= 600
    assert epochs_for_budget(10**7, 16, 1) >= 10


def _make_real(root):
    for sp, n in [("train", 40), ("val", 8), ("test", 8)]:
        (root / "images" / sp).mkdir(parents=True)
        (root / "labels" / sp).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (1280, 720), (110, 110, 110)).save(
                root / "images" / sp / f"vid{i % 4}_frame{i:03d}.jpg")
            (root / "labels" / sp / f"vid{i % 4}_frame{i:03d}.txt").write_text("0 0.5 0.5 0.04 0.04\n")


def _make_synth(root, n=60):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (1280, 720), (130, 130, 140)).save(root / "images" / f"synth_{i:05d}.png")
        (root / "labels" / f"synth_{i:05d}.txt").write_text("0 0.5 0.5 0.03 0.03\n")


@pytest.mark.parametrize("arm_file,exp_stage,exp_real,exp_synth", [
    ("real_only_250.yaml", "single", 40, 0),
    ("synth_only.yaml", "single", 0, 60),
    ("synth_pretrain_real_250.yaml", "pretrain_finetune", 40, 60),
    ("ablation_no_light_random.yaml", "pretrain_finetune", 40, 60),
])
def test_prepare_arm_data_composition(tmp_path, arm_file, exp_stage, exp_real, exp_synth):
    _make_real(tmp_path / "real")
    _make_synth(tmp_path / "synth" / "full")
    _make_synth(tmp_path / "synth" / "no_light_random")
    base = load_base("configs/base.yaml")
    arm = load_arm(f"configs/arms/{arm_file}")
    prep = prepare_arm_data(
        base, arm, tmp_path / "real",
        {"full": tmp_path / "synth" / "full",
         "no_light_random": tmp_path / "synth" / "no_light_random"},
        tmp_path / "work", seed=0)
    assert prep.stage == exp_stage
    assert prep.n_real_train == exp_real
    assert prep.n_synth_train == exp_synth
    assert prep.data_yaml.exists()


def test_ablation_uses_correct_synth_dir(tmp_path):
    # ablasyon kolu 'no_light_random' sentetik dizinini kullanmalı, 'full' değil
    _make_real(tmp_path / "real")
    _make_synth(tmp_path / "synth" / "full", n=10)
    _make_synth(tmp_path / "synth" / "no_light_random", n=25)
    base = load_base("configs/base.yaml")
    arm = load_arm("configs/arms/ablation_no_light_random.yaml")
    prep = prepare_arm_data(
        base, arm, tmp_path / "real",
        {"full": tmp_path / "synth" / "full",
         "no_light_random": tmp_path / "synth" / "no_light_random"},
        tmp_path / "work", seed=0)
    assert prep.n_synth_train == 25  # no_light_random dizininden


def _fabricate_results(rdir):
    rdir.mkdir(parents=True)

    def metrics(m50, small, med, large):
        return {"map50": m50, "map50_95": m50 * 0.65, "map50_small": small,
                "map50_medium": med, "map50_large": large}

    def write(arm, seed, nreal, nsynth, m, hyp):
        (rdir / f"{arm}_seed{seed}.json").write_text(json.dumps(
            {"arm": arm, "seed": seed, "stage": "x", "n_real_train": nreal,
             "n_synth_train": nsynth, "hypothesis_refs": hyp, "metrics": m}))

    curve = {50: 0.30, 100: 0.42, 250: 0.55, 1000: 0.70}
    for N, ro in curve.items():
        sp = min(0.72, ro + (0.12 if N <= 100 else 0.02))
        for seed in (0, 1, 2):
            write(f"real_only_{N}", seed, N, 0, metrics(ro, ro * 0.5, ro * 0.9, ro * 1.05), ["H1"])
            write(f"synth_pretrain_real_{N}", seed, N, 2000,
                  metrics(sp, ro * 0.5 + 0.18, ro * 0.9 + 0.05, ro * 1.05 + 0.01), ["H1", "H4"])
    for seed in (0, 1, 2):
        write("synth_only", seed, 0, 2000, metrics(0.18, 0.10, 0.16, 0.22), ["H2"])
    for fac, val in [("no_light_random", 0.46), ("low_texture_realism", 0.55)]:
        for seed in (0, 1, 2):
            write(f"ablation_{fac}", seed, 250, 2000, metrics(val, val * 0.6, val * 0.9, val * 1.05), ["H3"])


def test_hypothesis_decisions(tmp_path):
    from miragebench.report.aggregate import load_results
    _fabricate_results(tmp_path / "results")
    agg = aggregate_by_arm(load_results(tmp_path / "results"))
    decisions = {d["hypothesis"]: d["verdict"] for d in hypothesis_decisions(agg)}
    # Fabrike veri desenine göre dördü de desteklenmeli
    assert decisions["H1"] == "desteklendi"
    assert decisions["H2"] == "desteklendi"
    assert decisions["H3"] == "desteklendi"
    assert decisions["H4"] == "desteklendi"


def test_data_efficiency_curve_sorted(tmp_path):
    from miragebench.report.aggregate import load_results
    _fabricate_results(tmp_path / "results")
    agg = aggregate_by_arm(load_results(tmp_path / "results"))
    curves = data_efficiency_curve(agg)
    ns = [n for n, _, _ in curves["real_only"]]
    assert ns == sorted(ns)
    assert len(curves["synth_pretrain"]) > 0
