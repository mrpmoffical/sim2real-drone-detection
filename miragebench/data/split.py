"""Grup-farkında bölme ve veri-bütçesi alt örnekleme.

İki sızıntı/yanlılık kaynağını kontrol eder:
  1. Grup sızıntısı: aynı video/çekimden kareler train ve test'e dağılmamalı.
     Split GRUP düzeyinde yapılır; bir grubun tüm kareleri aynı bölüte gider.
  2. Bütçe yanlılığı: 'N gerçek görüntü' kolları için alt örnekleme
     DETERMİNİSTİK (seed'li) ve mümkünse grup/boyut dağılımını koruyacak
     biçimde yapılır; aksi halde kollar arası fark veriden değil örneklem
     şansından gelir.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .records import ImageRecord


def group_aware_split(
    records: list[ImageRecord],
    train: float,
    val: float,
    test: float,
    seed: int = 0,
) -> dict[str, list[ImageRecord]]:
    """Kayıtları gruplara göre train/val/test'e böler. Bir grup tek bölüte düşer.

    Gruplar görüntü sayısına göre büyükten küçüğe sıralanıp hedef oranlara en az
    sapmayla dağıtılır (greedy). Bu, az sayıda büyük grupta oranların aşırı
    kaymasını engeller."""
    by_group: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        by_group[r.group or r.image_path.stem].append(r)

    groups = sorted(by_group.items(), key=lambda kv: len(kv[1]), reverse=True)
    total = len(records)
    targets = {"train": train * total, "val": val * total, "test": test * total}
    assigned: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    rng = random.Random(seed)
    # Eşit büyüklükteki gruplarda deterministik ama toh-bağımlı sıra
    rng.shuffle(groups)
    groups.sort(key=lambda kv: len(kv[1]), reverse=True)

    for _gname, recs in groups:
        # Hedefe göre en çok 'açığı' olan bölüte ata
        deficit = {s: targets[s] - counts[s] for s in assigned}
        # test ve val küçük olduğundan, oranı korumak için açık/oran ölçütü
        pick = max(assigned, key=lambda s: deficit[s] / max(targets[s], 1e-9))
        assigned[pick].extend(recs)
        counts[pick] += len(recs)
    return assigned


def stratified_budget_subsample(
    records: list[ImageRecord],
    n: int,
    seed: int = 0,
    small_max: int = 32,
    medium_max: int = 96,
) -> list[ImageRecord]:
    """n görüntülük deterministik alt örneklem; boyut-katman dağılımını korumaya çalışır.

    Her görüntü 'baskın' boyut katmanına (en küçük nesnesi) göre tabakalanır ve
    her tabakadan orantılı çekilir. n >= len(records) ise hepsi döner."""
    if n >= len(records):
        return list(records)
    rng = random.Random(seed)

    def dominant_bucket(r: ImageRecord) -> str:
        c = r.size_bucket_counts(small_max, medium_max)
        if c["small"] > 0:
            return "small"
        if c["medium"] > 0:
            return "medium"
        if c["large"] > 0:
            return "large"
        return "empty"

    strata: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        strata[dominant_bucket(r)].append(r)

    chosen: list[ImageRecord] = []
    total = len(records)
    for bucket, recs in strata.items():
        rng.shuffle(recs)
        k = round(n * len(recs) / total)
        chosen.extend(recs[:k])
    # yuvarlama düzeltmesi
    rng.shuffle(chosen)
    if len(chosen) > n:
        chosen = chosen[:n]
    elif len(chosen) < n:
        chosen_ids = {id(r) for r in chosen}
        remaining = [r for r in records if id(r) not in chosen_ids]
        rng.shuffle(remaining)
        chosen.extend(remaining[: n - len(chosen)])
    return chosen


def split_report(assigned: dict[str, list[ImageRecord]]) -> dict:
    """Split'in sağlık raporu: grup sızıntısı var mı, oranlar, boyut dağılımı."""
    rep: dict = {"counts": {}, "group_leak": [], "size_dist": {}}
    group_to_splits: dict[str, set] = defaultdict(set)
    for sp, recs in assigned.items():
        rep["counts"][sp] = len(recs)
        dist = {"small": 0, "medium": 0, "large": 0}
        for r in recs:
            group_to_splits[r.group or r.image_path.stem].add(sp)
            c = r.size_bucket_counts()
            for k in dist:
                dist[k] += c[k]
        rep["size_dist"][sp] = dist
    # Bir grup birden çok bölütteyse sızıntı var (olmamalı)
    rep["group_leak"] = [g for g, s in group_to_splits.items() if len(s) > 1]
    rep["leak_free"] = len(rep["group_leak"]) == 0
    return rep
