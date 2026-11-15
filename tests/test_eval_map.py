import numpy as np
import pytest

from miragebench.eval.metrics import (
    Detection,
    GroundTruth,
    average_precision,
    evaluate,
    iou_matrix,
)


def test_iou_half_overlap():
    d = np.array([[0, 0, 10, 10]], float)
    g = np.array([[5, 0, 15, 10]], float)
    assert abs(iou_matrix(d, g)[0, 0] - (50 / 150)) < 1e-9


def test_perfect_detection_ap1():
    dets = [Detection("im1", 0, 0.9, (0, 0, 10, 10))]
    gts = [GroundTruth("im1", 0, (0, 0, 10, 10))]
    assert average_precision(dets, gts, 0.5).ap == pytest.approx(1.0)


def test_missed_gt_lowers_ap():
    dets = [Detection("im1", 0, 0.9, (0, 0, 10, 10))]
    gts = [GroundTruth("im1", 0, (0, 0, 10, 10)), GroundTruth("im1", 0, (50, 50, 60, 60))]
    # tek TP, 2 GT -> COCO 101-nokta ~0.505
    assert average_precision(dets, gts, 0.5).ap == pytest.approx(0.505, abs=0.01)


def test_duplicate_detection_is_fp():
    dets = [Detection("im1", 0, 0.9, (0, 0, 10, 10)), Detection("im1", 0, 0.8, (1, 1, 11, 11))]
    gts = [GroundTruth("im1", 0, (0, 0, 10, 10))]
    r = average_precision(dets, gts, 0.5)
    assert r.recall == pytest.approx(1.0)  # 1 TP 1 FP, recall tam


def test_size_stratification_isolates_buckets():
    small_gt = GroundTruth("im1", 0, (0, 0, 20, 20))        # alan 400 -> small
    large_gt = GroundTruth("im1", 0, (200, 200, 320, 320))  # alan 14400 -> large
    gts = [small_gt, large_gt]
    dets = [Detection("im1", 0, 0.9, (0, 0, 20, 20))]       # sadece küçüğü bul
    assert average_precision(dets, gts, 0.5, "small").ap == pytest.approx(1.0)
    assert average_precision(dets, gts, 0.5, "large").ap == pytest.approx(0.0)


def test_evaluate_keys_present():
    gts = [GroundTruth("im1", 0, (0, 0, 20, 20))]
    dets = [Detection("im1", 0, 0.9, (0, 0, 20, 20))]
    res = evaluate(dets, gts, [0])
    for k in ("map50", "map50_95", "map50_small", "map50_medium", "map50_large"):
        assert k in res


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_cross_check_pycocotools(seed):
    """Referans pycocotools ile sayısal uyum (kuruluysa)."""
    pytest.importorskip("pycocotools")
    import contextlib
    import io as _io
    import json
    import tempfile

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(seed)
    gts, dets = [], []
    images, anns, dts = [], [], []
    ann_id = 1
    for im in range(15):
        images.append({"id": im, "width": 1280, "height": 720})
        for _ in range(int(rng.integers(0, 4))):
            side = float(rng.choice([15, 30, 60, 120, 200]))
            x = float(rng.uniform(0, 1280 - side))
            y = float(rng.uniform(0, 720 - side))
            gts.append(GroundTruth(str(im), 0, (x, y, x + side, y + side)))
            anns.append({"id": ann_id, "image_id": im, "category_id": 1,
                         "bbox": [x, y, side, side], "area": side * side, "iscrowd": 0})
            ann_id += 1
            if rng.random() < 0.8:
                j = rng.normal(0, 3, 4)
                db = (x + j[0], y + j[1], x + side + j[2], y + side + j[3])
                cf = float(rng.uniform(0.3, 0.99))
                dets.append(Detection(str(im), 0, cf, db))
                dts.append({"image_id": im, "category_id": 1,
                            "bbox": [db[0], db[1], db[2] - db[0], db[3] - db[1]], "score": cf})

    mine = evaluate(dets, gts, [0])
    gt_json = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "drone"}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt_json, f)
        gtp = f.name
    coco_gt = COCO(gtp)
    if not dts:
        pytest.skip("tespit üretilmedi")
    coco_dt = coco_gt.loadRes(dts)
    with contextlib.redirect_stdout(_io.StringIO()):
        E = COCOeval(coco_gt, coco_dt, "bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
    assert mine["map50_95"] == pytest.approx(E.stats[0], abs=1e-3)
    assert mine["map50"] == pytest.approx(E.stats[1], abs=1e-3)
