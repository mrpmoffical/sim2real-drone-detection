"""COCO protokolüne sadık mAP — boyut katmanlı (small/medium/large).

Neden sıfırdan: çalışmanın birincil çıktısı bu metrik olduğundan, davranışını tam
kontrol etmek ve birim testleriyle doğrulamak gerekir. Uygulama COCOeval'ın
mantığını izler:

  - Tespitler güven skoruna göre azalan sırada eşlenir (greedy).
  - Her IoU eşiği için her tespit, aynı görüntüde IoU >= eşik olan, henüz
    eşlenmemiş en yüksek IoU'lu GT'ye eşlenir.
  - AP, 101 nokta (0.00..1.00) recall ızgarasında interpolasyonlu precision'ın
    ortalamasıdır.
  - mAP@50 = AP@IoU0.50; mAP@[.50:.05:.95] = 10 eşik ortalaması.

Boyut katmanlama (COCO 'ignore' mantığı):
  - Verilen alan aralığının DIŞINDAki GT'ler 'ignore' işaretlenir: onlara eşlenen
    tespitler ne TP ne FP sayılır (çıkarılır); kendileri FN sayılmaz.
  - Eşleşmeyen VE alan aralığı dışında olan tespitler de 'ignore' (FP sayılmaz).
  Böylece ör. 'small' AP'si yalnız küçük nesnelerin tespit başarısını ölçer.

Kutu biçimi: xyxy piksel. GT ve tahminler görüntü (image_id) bazında gruplanır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO alan aralıkları (piksel²). all = tüm boyutlar.
AREA_RANGES = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0 * 32.0),
    "medium": (32.0 * 32.0, 96.0 * 96.0),
    "large": (96.0 * 96.0, 1e10),
}

IOU_50 = [0.5]
IOU_50_95 = [round(0.5 + 0.05 * i, 2) for i in range(10)]


@dataclass
class Detection:
    image_id: str
    cls: int
    score: float
    xyxy: tuple[float, float, float, float]


@dataclass
class GroundTruth:
    image_id: str
    cls: int
    xyxy: tuple[float, float, float, float]


def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou_matrix(dets: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """dets [N,4], gts [M,4] xyxy -> IoU [N,M]."""
    if len(dets) == 0 or len(gts) == 0:
        return np.zeros((len(dets), len(gts)), dtype=np.float64)
    d = dets[:, None, :]
    g = gts[None, :, :]
    xx1 = np.maximum(d[..., 0], g[..., 0])
    yy1 = np.maximum(d[..., 1], g[..., 1])
    xx2 = np.minimum(d[..., 2], g[..., 2])
    yy2 = np.minimum(d[..., 3], g[..., 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    union = area_d + area_g - inter
    return np.where(union > 0, inter / union, 0.0)


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-nokta interpolasyonlu AP (COCO)."""
    rec_grid = np.linspace(0.0, 1.0, 101)
    # precision'ı recall'a göre monoton azalmayan yap (sağdan maksimum)
    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]
    # her ızgara noktası için recall >= nokta olan en yüksek precision
    p_interp = np.zeros_like(rec_grid)
    for i, r in enumerate(rec_grid):
        mask = recalls >= r
        p_interp[i] = precisions[mask].max() if mask.any() else 0.0
    return float(p_interp.mean())


def _match_single(
    dets: list[Detection], gts: list[GroundTruth], iou_thr: float, area_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Tek sınıf, tek IoU eşiği, tek alan aralığı için eşleme.

    Dönüş: (tp[Ndet], fp[Ndet] — güven sırasında), npos (ignore olmayan GT sayısı).
    'ignore' edilen tespitler tp=fp=0 ile işaretlenir (çıkarılmış sayılır).
    """
    lo, hi = area_range
    # GT ignore bayrakları: alan aralığı dışındakiler ignore
    gt_boxes = np.array([g.xyxy for g in gts], dtype=np.float64).reshape(-1, 4)
    gt_ignore = np.array([not (lo <= box_area(g.xyxy) < hi) for g in gts], dtype=bool)
    npos = int((~gt_ignore).sum())

    # Tespitleri güvene göre azalan sırala
    order = sorted(range(len(dets)), key=lambda i: dets[i].score, reverse=True)
    tp = np.zeros(len(dets), dtype=np.float64)
    fp = np.zeros(len(dets), dtype=np.float64)
    gt_matched = np.zeros(len(gts), dtype=bool)

    det_boxes = np.array([d.xyxy for d in dets], dtype=np.float64).reshape(-1, 4)
    ious = iou_matrix(det_boxes, gt_boxes) if len(gts) else np.zeros((len(dets), 0))

    for rank_i in order:
        # En iyi eşleşmeyi bul: önce ignore-olmayan eşleşmiş-olmayan GT'ler arasında
        best_iou = iou_thr
        best_gt = -1
        best_is_ignore = False
        for j in range(len(gts)):
            if gt_matched[j] and not gt_ignore[j]:
                continue
            iou = ious[rank_i, j]
            if iou < best_iou:
                continue
            # ignore olmayan eşleşmeyi tercih et; COCO: ignore GT'ye ancak daha iyi
            # ignore-olmayan yoksa eşlenir
            if best_gt >= 0 and not best_is_ignore and gt_ignore[j]:
                continue
            best_iou = iou
            best_gt = j
            best_is_ignore = gt_ignore[j]
        if best_gt >= 0:
            if best_is_ignore:
                # ignore GT'ye eşlendi: ne TP ne FP (çıkar)
                tp[rank_i] = 0
                fp[rank_i] = 0
                gt_matched[best_gt] = True  # bir kez kullanılsın
            else:
                tp[rank_i] = 1
                gt_matched[best_gt] = True
        else:
            # Eşleşmedi. Alan aralığı dışındaysa ignore (FP sayma), değilse FP.
            d_area = box_area(dets[rank_i].xyxy)
            if lo <= d_area < hi:
                fp[rank_i] = 1
            else:
                fp[rank_i] = 0  # ignore

    # Güven sırasına göre döndür (PR eğrisi için)
    tp_sorted = tp[order]
    fp_sorted = fp[order]
    return tp_sorted, fp_sorted, npos


@dataclass
class APResult:
    ap: float
    precision: float
    recall: float
    npos: int
    ndet: int


def average_precision(
    dets: list[Detection], gts: list[GroundTruth], iou_thr: float, area: str = "all"
) -> APResult:
    """Tek sınıf AP (tüm görüntüler birlikte, görüntü-içi eşleme ile)."""
    area_range = AREA_RANGES[area]
    # Görüntü bazında eşle, sonra güven sırasında birleştir
    by_img_d: dict[str, list[Detection]] = {}
    by_img_g: dict[str, list[GroundTruth]] = {}
    for d in dets:
        by_img_d.setdefault(d.image_id, []).append(d)
    for g in gts:
        by_img_g.setdefault(g.image_id, []).append(g)
    all_imgs = set(by_img_d) | set(by_img_g)

    tp_all, fp_all, score_all = [], [], []
    npos = 0
    for img in all_imgs:
        di = by_img_d.get(img, [])
        gi = by_img_g.get(img, [])
        tp, fp, n = _match_single(di, gi, iou_thr, area_range)
        npos += n
        # eşleme güven sırasında döndü; skorları da aynı sırada topla
        order = sorted(range(len(di)), key=lambda i: di[i].score, reverse=True)
        tp_all.append(tp)
        fp_all.append(fp)
        score_all.append(np.array([di[i].score for i in order]))

    if npos == 0:
        return APResult(0.0, 0.0, 0.0, 0, len(dets))
    if not tp_all or sum(len(t) for t in tp_all) == 0:
        return APResult(0.0, 0.0, 0.0, npos, 0)

    tp = np.concatenate(tp_all)
    fp = np.concatenate(fp_all)
    scores = np.concatenate(score_all)
    order = np.argsort(-scores)
    tp, fp = tp[order], fp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / npos
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    ap = _ap_from_pr(recalls, precisions)
    return APResult(
        ap=ap,
        precision=float(precisions[-1]),
        recall=float(recalls[-1]),
        npos=npos,
        ndet=int(tp.sum() + fp.sum()),
    )


def evaluate(
    dets: list[Detection],
    gts: list[GroundTruth],
    classes: list[int],
    iou_thrs: list[float] | None = None,
    areas: tuple[str, ...] = ("all", "small", "medium", "large"),
) -> dict:
    """Tam değerlendirme: sınıf × IoU × alan üzerinden mAP ve boyut-katmanlı AP.

    Dönüş sözlüğü (anahtarlar):
      map50, map50_95            : tüm boyutlar
      map50_small/medium/large   : boyut-katmanlı mAP@50
      per_class                  : sınıf bazında ayrıntı
    """
    iou_thrs = iou_thrs or IOU_50_95
    out: dict = {"per_class": {}, "iou_thrs": iou_thrs}

    # sınıf filtreli alt kümeler
    def sub(items, c):
        return [x for x in items if x.cls == c]

    # her alan ve IoU için sınıf-ortalaması
    area_iou_ap: dict[tuple[str, float], list[float]] = {}
    for c in classes:
        dc, gc = sub(dets, c), sub(gts, c)
        out["per_class"][c] = {}
        for area in areas:
            aps = []
            for thr in iou_thrs:
                res = average_precision(dc, gc, thr, area)
                area_iou_ap.setdefault((area, thr), []).append(res.ap)
                aps.append(res.ap)
            out["per_class"][c][area] = {
                "ap50": round(float(aps[0]), 4),
                "ap50_95": round(float(np.mean(aps)), 4),
            }

    def mean_ap(area, thr=None):
        if thr is not None:
            return float(np.mean(area_iou_ap[(area, thr)]))
        vals = [np.mean(area_iou_ap[(area, t)]) for t in iou_thrs]
        return float(np.mean(vals))

    out["map50"] = round(mean_ap("all", iou_thrs[0]), 4)
    out["map50_95"] = round(mean_ap("all"), 4)
    for area in ("small", "medium", "large"):
        out[f"map50_{area}"] = round(mean_ap(area, iou_thrs[0]), 4)
        out[f"map50_95_{area}"] = round(mean_ap(area), 4)
    return out
