"""Blender entegrasyon testi: bir kare render edip bbox'ın drone'u sardığını doğrular.

bpy yoksa atlanır. Yavaştır (GPU'suz yazılım render ~1 dk), bu yüzden 'slow' ve
'blender' ile işaretli."""

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")  # bpy yoksa tüm modül atlanır


@pytest.mark.blender
@pytest.mark.slow
def test_render_one_bbox_bounds_drone(tmp_path):
    import cv2

    from miragebench.synth import blender_generator as bg
    from miragebench.synth.factors import FactorSwitches, SampledScene, SceneSampler
    from miragebench.synth.geometry import CameraIntrinsics

    sampler = SceneSampler(FactorSwitches(), seed=1)
    sp = SampledScene(distance_m=6.0, yaw_deg=35, pitch_deg=10, roll_deg=5,
                      sun_elevation_deg=55, sun_azimuth_deg=120, sun_strength=3.0,
                      sun_warmth=0.5, focal_mm=50.0, background_id=3, texture_realism=True,
                      asset_name="quadcopter_generic", paint_rgb=(200, 40, 40))
    sp.extras["allow_offset"] = False
    intr = CameraIntrinsics(320, 240, 50.0)
    meta = bg.render_one(tmp_path, 0, sp, sampler, intr, None, "BLENDER_EEVEE", 8, 0, (32, 96))
    assert meta is not None

    img = cv2.imread(str(tmp_path / "images" / "synth_0000000.png"))
    x1, y1, x2, y2 = map(int, meta["bbox_xyxy_px"])
    bg_patch = np.concatenate([img[0:8, 0:8].reshape(-1, 3), img[0:8, -8:].reshape(-1, 3)])
    bg_col = bg_patch.mean(0)
    inside = img[y1:y2, x1:x2].reshape(-1, 3).astype(float)
    fg_in = (np.linalg.norm(inside - bg_col, axis=1) > 25).mean()
    # kutu dışı halka arka plan olmalı
    m = 5
    ring = img[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m].astype(float)
    mask = np.ones(ring.shape[:2], bool)
    mask[m:m + (y2 - y1), m:m + (x2 - x1)] = False
    fg_ring = (np.linalg.norm(ring[mask] - bg_col, axis=1) > 25).mean()
    assert fg_in > 0.15, f"kutu içinde drone pikseli az: {fg_in}"
    assert fg_ring < 0.1, f"kutu dışına drone taşmış: {fg_ring}"


@pytest.mark.blender
def test_procedural_quadcopter_builds():
    from miragebench.synth.assets import build_procedural_quadcopter
    bpy.ops.wm.read_factory_settings(use_empty=True)
    drone = build_procedural_quadcopter("q", size_m=0.45)
    assert drone is not None
    assert len(drone.data.vertices) > 0
