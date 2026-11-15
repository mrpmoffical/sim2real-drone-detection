"""Deney kolu koşucusu: config -> veri hazırlığı -> eğitim -> çıkarım -> değerlendirme.

Bir kolu bir tohum (seed) için uçtan uca çalıştırır ve sonucu JSON yazar. Veri
hazırlığı (hangi görüntüler, hangi bütçe, ön-eğitim/ince-ayar ayrımı) GPU'dan
bağımsızdır ve `prepare_arm_data` olarak ayrılmıştır; `tests/` bunu GPU'suz test eder.

Beklenen girdi düzeni:
  real_root/   : mb-ingest çıktısı (images/{train,val,test}, labels/...)
  synth_roots  : {faktör_adı: dizin}. En az 'full' olmalı. Ablasyon kolları
                 kendi faktör dizinini kullanır (ör. 'no_light_random').
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import ArmConfig, BaseConfig
from ..data.adapters import iter_yolo
from ..data.materialize import materialize
from ..data.records import ImageRecord
from ..data.split import stratified_budget_subsample


@dataclass
class PreparedData:
    data_yaml: Path
    pretrain_yaml: Path | None
    n_real_train: int
    n_synth_train: int
    stage: str  # 'single' | 'pretrain_finetune'


def _ablation_factor(arm_name: str) -> str:
    if arm_name.startswith("ablation_"):
        return arm_name[len("ablation_"):]
    return "full"


def prepare_arm_data(
    base: BaseConfig,
    arm: ArmConfig,
    real_root: str | Path,
    synth_roots: dict[str, str | Path],
    work_dir: str | Path,
    seed: int,
) -> PreparedData:
    """Kolun veri kompozisyonunu diske materyalize eder (GPU'suz).

    Kurallar:
      - val ve test HER ZAMAN gerçek (real_root'tan), değişmez.
      - real_count verilmişse train grup-korumalı değil, tohum-belirli alt örneklenir.
      - synth gerekiyorsa ilgili faktör dizininden alınır.
    """
    real_root = Path(real_root)
    work_dir = Path(work_dir)
    gr = base.split.group_regex

    real_train = list(iter_yolo(real_root / "images" / "train",
                                group_regex=gr, source="real")) if (real_root / "images" / "train").is_dir() else []
    # labels train ile eşleştir: iter_yolo images/ ve labels/ yan yana bekler;
    # materialize düzeninde labels kardeş dizinde, bu yüzden root'u images üstü veririz
    real_train = list(iter_yolo(real_root, group_regex=gr, source="real"))
    # Yukarıdaki iter_yolo tüm split'leri karıştırmasın diye train'e kısıtla:
    real_train = [r for r in real_train if "/images/train/" in str(r.image_path).replace("\\", "/")]

    real_val = _read_split(real_root, "val", gr)
    real_test = _read_split(real_root, "test", gr)

    d = arm.data
    # Gerçek eğitim alt örneklemi
    if d.real_count is not None:
        real_train = stratified_budget_subsample(
            real_train, d.real_count, seed=seed,
            small_max=base.size_buckets.small_max, medium_max=base.size_buckets.medium_max)

    # Sentetik kayıtlar
    def synth_records(count: int | None) -> list[ImageRecord]:
        factor = _ablation_factor(arm.name)
        root = synth_roots.get(factor) or synth_roots.get("full")
        if root is None:
            raise ValueError(f"sentetik dizin yok (faktör={factor})")
        recs = list(iter_yolo(root, group_regex=gr, source=f"synth_{factor}"))
        if count is not None:
            recs = stratified_budget_subsample(recs, count, seed=seed)
        return recs

    arm_dir = work_dir / f"{arm.name}_seed{seed}"

    # Kompozisyon
    if d.pretrain is not None:
        # İki aşama: ön-eğitim (synth) -> ince-ayar (real)
        synth_pre = synth_records(d.pretrain_count)
        pre_yaml = materialize(arm_dir / "pretrain", synth_pre, real_val, real_test, base.classes)
        ft_yaml = materialize(arm_dir / "finetune", real_train, real_val, real_test, base.classes)
        return PreparedData(ft_yaml, pre_yaml, len(real_train), len(synth_pre), "pretrain_finetune")

    if d.train_source == "synth":
        synth_tr = synth_records(d.synth_count)
        y = materialize(arm_dir / "single", [], real_val, real_test, base.classes,
                        synth_train_records=synth_tr)
        return PreparedData(y, None, 0, len(synth_tr), "single")

    if d.train_source == "mixed":
        synth_tr = synth_records(d.synth_count)
        y = materialize(arm_dir / "single", real_train, real_val, real_test, base.classes,
                        synth_train_records=synth_tr)
        return PreparedData(y, None, len(real_train), len(synth_tr), "single")

    # real
    y = materialize(arm_dir / "single", real_train, real_val, real_test, base.classes)
    return PreparedData(y, None, len(real_train), 0, "single")


def _read_split(real_root: Path, split: str, gr: str) -> list[ImageRecord]:
    recs = list(iter_yolo(real_root, group_regex=gr, source="real"))
    key = f"/images/{split}/"
    return [r for r in recs if key in str(r.image_path).replace("\\", "/")]


def run_arm_seed(
    base: BaseConfig,
    arm: ArmConfig,
    real_root: str | Path,
    synth_roots: dict[str, str | Path],
    work_dir: str | Path,
    results_dir: str | Path,
    seed: int,
) -> dict:
    """Bir kolu bir tohumla uçtan uca çalıştırır (GPU gerektirir) ve sonucu döndürür."""
    from ..eval.io import load_ground_truth, load_predictions
    from ..eval.metrics import IOU_50_95, evaluate
    from .train_wrapper import predict_to_txt, train_stage

    prepared = prepare_arm_data(base, arm, real_root, synth_roots, work_dir, seed)
    arm_dir = Path(work_dir) / f"{arm.name}_seed{seed}"
    m, t = base.model, base.train

    init_w = None
    if prepared.stage == "pretrain_finetune":
        pre = train_stage(prepared.pretrain_yaml, m.arch, m.imgsz, t.batch, t.total_steps,
                          arm_dir / "runs", "pretrain", seed, t.lr0, t.optimizer, t.patience,
                          augment_preset=t.augment_preset)
        init_w = pre.weights_path

    fit = train_stage(prepared.data_yaml, m.arch, m.imgsz, t.batch, t.total_steps,
                      arm_dir / "runs", "fit", seed, t.lr0, t.optimizer, t.patience,
                      init_weights=init_w, augment_preset=t.augment_preset)

    data_root = prepared.data_yaml.parent
    test_images = data_root / "images" / "test"
    test_labels = data_root / "labels" / "test"
    preds_dir = arm_dir / "preds"
    predict_to_txt(fit.weights_path, test_images, preds_dir, m.imgsz,
                   sahi_enabled=m.sahi_enabled, sahi_slice=m.sahi_slice,
                   sahi_overlap=m.sahi_overlap, class_names=base.classes)

    gts = load_ground_truth(test_labels, test_images)
    dets = load_predictions(preds_dir, test_images)
    res = evaluate(dets, gts, list(range(len(base.classes))), iou_thrs=IOU_50_95)

    record = {
        "arm": arm.name, "seed": seed, "stage": prepared.stage,
        "n_real_train": prepared.n_real_train, "n_synth_train": prepared.n_synth_train,
        "hypothesis_refs": arm.hypothesis_refs,
        "epochs_fit": fit.epochs,
        "metrics": res, "n_gt": len(gts), "n_det": len(dets),
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{arm.name}_seed{seed}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2))
    return record
