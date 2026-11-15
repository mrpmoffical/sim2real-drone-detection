import json

from PIL import Image

from miragebench.data import (
    group_aware_split,
    iter_coco,
    iter_voc,
    iter_yolo,
    materialize,
    split_report,
    stratified_budget_subsample,
)


def _make_yolo(root, n_vid=6, n_frame=10):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for vid in range(n_vid):
        for fr in range(n_frame):
            name = f"vid{vid:02d}_frame{fr:03d}"
            Image.new("RGB", (640, 480), (120, 120, 120)).save(root / "images" / f"{name}.jpg")
            w = 0.02 + 0.08 * (fr / n_frame)
            (root / "labels" / f"{name}.txt").write_text(f"0 0.5 0.5 {w:.4f} {w:.4f}\n")


def test_yolo_adapter_and_groups(tmp_path):
    _make_yolo(tmp_path / "y")
    recs = list(iter_yolo(tmp_path / "y"))
    assert len(recs) == 60
    assert {r.group for r in recs} == {f"vid{v:02d}" for v in range(6)}


def test_group_aware_split_no_leak(tmp_path):
    _make_yolo(tmp_path / "y")
    recs = list(iter_yolo(tmp_path / "y"))
    sp = group_aware_split(recs, 0.6, 0.2, 0.2, seed=0)
    rep = split_report(sp)
    assert rep["leak_free"]
    assert rep["group_leak"] == []
    assert sum(rep["counts"].values()) == 60


def test_subsample_deterministic(tmp_path):
    _make_yolo(tmp_path / "y")
    recs = list(iter_yolo(tmp_path / "y"))
    a = {r.image_path for r in stratified_budget_subsample(recs, 15, seed=1)}
    b = {r.image_path for r in stratified_budget_subsample(recs, 15, seed=1)}
    c = {r.image_path for r in stratified_budget_subsample(recs, 15, seed=2)}
    assert a == b
    assert a != c
    assert len(a) == 15


def test_subsample_n_ge_len_returns_all(tmp_path):
    _make_yolo(tmp_path / "y")
    recs = list(iter_yolo(tmp_path / "y"))
    assert len(stratified_budget_subsample(recs, 999, seed=0)) == len(recs)


def test_coco_adapter(tmp_path):
    croot = tmp_path / "c"
    (croot / "imgs").mkdir(parents=True)
    Image.new("RGB", (800, 600), (100, 100, 100)).save(croot / "imgs" / "shotA_frame001.jpg")
    coco = {"images": [{"id": 1, "file_name": "shotA_frame001.jpg", "width": 800, "height": 600}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [100, 80, 40, 30], "iscrowd": 0}],
            "categories": [{"id": 3, "name": "uav"}]}
    (croot / "ann.json").write_text(json.dumps(coco))
    r = next(iter(iter_coco(croot / "ann.json", croot / "imgs", class_map={3: 0})))
    b = r.boxes[0]
    assert b.cls == 0
    assert b.xc == (100 + 20) / 800
    assert b.w == 40 / 800


def test_voc_adapter(tmp_path):
    vroot = tmp_path / "v"
    vroot.mkdir()
    Image.new("RGB", (1000, 500), (90, 90, 90)).save(vroot / "clipB_f0001.png")
    (vroot / "clipB_f0001.xml").write_text(
        "<annotation><filename>clipB_f0001.png</filename>"
        "<size><width>1000</width><height>500</height></size>"
        "<object><name>drone</name><bndbox><xmin>400</xmin><ymin>200</ymin>"
        "<xmax>450</xmax><ymax>260</ymax></bndbox></object></annotation>")
    r = next(iter(iter_voc(vroot, {"drone": 0})))
    b = r.boxes[0]
    assert b.xc == 425 / 1000
    assert b.h == 60 / 500


def test_materialize_writes_dataset(tmp_path):
    _make_yolo(tmp_path / "y")
    recs = list(iter_yolo(tmp_path / "y"))
    sp = group_aware_split(recs, 0.6, 0.2, 0.2, seed=0)
    dy = materialize(tmp_path / "out", sp["train"], sp["val"], sp["test"], ["drone"])
    assert dy.exists()
    import yaml
    data = yaml.safe_load(dy.read_text())
    assert data["names"] == {0: "drone"}
    n_img = len(list((tmp_path / "out" / "images" / "train").glob("*")))
    n_lbl = len(list((tmp_path / "out" / "labels" / "train").glob("*.txt")))
    assert n_img == n_lbl == len(sp["train"])
