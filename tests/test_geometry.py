import numpy as np

from miragebench.synth.geometry import (
    CameraIntrinsics,
    bbox_from_points,
    distance_for_target_px,
    project_points,
    to_yolo,
)

CAM_LOC = np.zeros(3)
R_IDENTITY = np.eye(3)  # kamera -Z'ye bakar


def test_center_projection():
    intr = CameraIntrinsics(1280, 720, 50)
    uv, inf = project_points(np.array([[0, 0, -10.0]]), CAM_LOC, R_IDENTITY, intr)
    assert inf[0]
    assert abs(uv[0, 0] - 640) < 1e-6
    assert abs(uv[0, 1] - 360) < 1e-6


def test_behind_camera_not_in_front():
    intr = CameraIntrinsics(1280, 720, 50)
    _, inf = project_points(np.array([[0, 0, 10.0]]), CAM_LOC, R_IDENTITY, intr)
    assert not inf[0]


def test_up_right_direction():
    intr = CameraIntrinsics(1280, 720, 50)
    uv, _ = project_points(np.array([[1, 1, -10.0]]), CAM_LOC, R_IDENTITY, intr)
    assert uv[0, 0] > 640  # +X sağ
    assert uv[0, 1] < 360  # +Y yukarı -> görüntüde yukarı


def test_known_box_width_matches_analytic():
    intr = CameraIntrinsics(1280, 720, 50)
    fx, _ = intr.focal_px()
    exp_w = fx * 0.5 / 30
    corners = np.array([[x, y, -30.0] for x in (-0.25, 0.25) for y in (-0.25, 0.25)])
    uv, inf = project_points(corners, CAM_LOC, R_IDENTITY, intr)
    bb = bbox_from_points(uv, inf, 1280, 720)
    assert bb is not None
    assert abs((bb[2] - bb[0]) - exp_w) < 1e-3


def test_distance_for_target_px_roundtrip():
    intr = CameraIntrinsics(1280, 720, 50)
    fx, _ = intr.focal_px()
    exp_w = fx * 0.5 / 30
    d = distance_for_target_px(0.5, exp_w, intr)
    assert abs(d - 30.0) < 1e-6


def test_hfov():
    intr = CameraIntrinsics(1280, 720, 50, 36)
    assert abs(intr.hfov_deg() - 39.6) < 0.2


def test_to_yolo_center():
    xc, yc, w, h = to_yolo((100, 100, 200, 300), 1000, 1000)
    assert abs(xc - 0.15) < 1e-9 and abs(yc - 0.20) < 1e-9
    assert abs(w - 0.10) < 1e-9 and abs(h - 0.20) < 1e-9


def test_offscreen_returns_none():
    intr = CameraIntrinsics(1280, 720, 50)
    # tamamen arkada
    uv, inf = project_points(np.array([[0, 0, 10.0]]), CAM_LOC, R_IDENTITY, intr)
    assert bbox_from_points(uv, inf, 1280, 720) is None
