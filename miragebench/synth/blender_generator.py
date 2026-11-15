"""Blender içi sentetik drone görüntüsü üretici.

Yalnızca bpy ortamında çalışır (ya `pip install bpy` modülüyle doğrudan, ya da
`blender --background --python ...` ile). Üretilen her örnek için:
  - görüntü (PNG)
  - YOLO etiketi (.txt)
  - meta JSON (örneklenmiş faktörler, mesafe, bbox, boyut katmanı)

Bbox, render edilmiş pikselleri eşikleyerek DEĞİL, drone mesh köşelerini dünya
uzayında kamera matrisiyle görüntüye yansıtıp (geometry.py) onların sınırını
alarak hesaplanır. Böylece gölge/blur/saydamlık bbox'ı bozmaz.

Not: Bu modül bpy'ye bağımlı olduğu için GPU'suz ortamda birim testleri
`geometry.py` ve `factors.py` üzerinden yapılır; tam render entegrasyonu
`tests/test_blender_smoke.py` ile (bpy varsa) çalıştırılır.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .factors import FactorSwitches, SampledScene, SceneRanges, SceneSampler
from .geometry import CameraIntrinsics, bbox_from_points, project_points, to_yolo

try:
    import bpy  # type: ignore
    from mathutils import Euler, Vector  # type: ignore
except ImportError:
    bpy = None  # type: ignore

from . import assets as _assets


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _setup_world_background(scene_params: SampledScene, bg_dir: Path | None) -> None:
    """Dünya arka planını ayarlar. bg_dir verilirse HDRI/görüntü, yoksa düz gri gradyan.

    Arka plan çeşitliliği (bg_variation) kapalıysa background_id=0 sabittir, bu da
    tek bir sabit arka plana yol açar (SceneSampler bunu garantiler)."""
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if bg is None:
        return
    images = sorted(bg_dir.glob("*.hdr")) + sorted(bg_dir.glob("*.exr")) + \
        sorted(bg_dir.glob("*.jpg")) + sorted(bg_dir.glob("*.png")) if bg_dir else []
    if images and scene_params.background_id is not None:
        img_path = images[scene_params.background_id % len(images)]
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(img_path))
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        scene_params.extras["background"] = img_path.name
    else:
        # Prosedürel gök: background_id'den türetilmiş deterministik gri/mavi ton
        bid = scene_params.background_id or 0
        tone = 0.35 + 0.5 * ((bid * 0.137) % 1.0)
        bg.inputs["Color"].default_value = (tone * 0.7, tone * 0.8, tone, 1.0)
        scene_params.extras["background"] = f"procedural_tone_{tone:.2f}"
    bg.inputs["Strength"].default_value = 1.0


def _setup_sun(scene_params: SampledScene, sampler: SceneSampler) -> None:
    dx, dy, dz = sampler.sun_direction(scene_params)
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 0))
    sun = bpy.context.object
    # Yönü güneş vektöründen türet (sun -Z'ye ışır; vektöre bakacak şekilde döndür)
    direction = Vector((dx, dy, dz))
    sun.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    sun.data.energy = scene_params.sun_strength
    # Renk sıcaklığı: warmth 0->mavimsi, 1->sarımsı
    w = scene_params.sun_warmth
    sun.data.color = (0.8 + 0.2 * w, 0.85 + 0.1 * w, 1.0 - 0.3 * w)


def _setup_camera(intr: CameraIntrinsics):
    bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(math.pi / 2, 0, 0))
    cam = bpy.context.object
    cam.data.lens = intr.focal_mm
    cam.data.sensor_width = intr.sensor_width_mm
    cam.data.sensor_fit = "HORIZONTAL"
    bpy.context.scene.camera = cam
    return cam


def _place_drone(drone, scene_params: SampledScene) -> None:
    """Drone'u kameranın önüne, örneklenmiş mesafeye ve yönelime yerleştirir.

    Kamera (0,0,0)'da ve +Y'ye bakar (rotation_euler x=90°). Dolayısıyla drone'u
    +Y ekseninde distance_m kadar ileriye koyarız; yatayda/dikeyde küçük jitter
    ekleyerek nesnenin her zaman tam merkezde olmasını engelleriz (aksi halde
    dedektör 'drone hep ortada' kolaycılığını öğrenir)."""
    d = scene_params.distance_m
    sampler_rng = scene_params.extras.get("_rng")
    jx = jz = 0.0
    if scene_params.extras.get("allow_offset", True):
        # kadraj içinde kalacak küçük açısal kayma
        import random as _r
        rng = sampler_rng or _r.Random(int(d * 1000))
        jx = rng.uniform(-0.25, 0.25) * d * math.tan(math.radians(18))
        jz = rng.uniform(-0.18, 0.18) * d * math.tan(math.radians(18))
    drone.location = (jx, d, jz)
    drone.rotation_euler = Euler(
        (math.radians(scene_params.pitch_deg),
         math.radians(scene_params.roll_deg),
         math.radians(scene_params.yaw_deg)),
        "XYZ",
    )
    bpy.context.view_layer.update()


def _world_vertices(drone) -> np.ndarray:
    """Drone mesh köşelerinin dünya-uzayı koordinatları [N,3]."""
    mw = drone.matrix_world
    verts = drone.data.vertices
    out = np.empty((len(verts), 3), dtype=np.float64)
    for i, v in enumerate(verts):
        co = mw @ v.co
        out[i] = (co.x, co.y, co.z)
    return out


def _camera_pose(cam) -> tuple[np.ndarray, np.ndarray]:
    """Kamera konumu (t) ve YEREL->DÜNYA dönme matrisi (R) numpy olarak."""
    mw = cam.matrix_world
    t = np.array([mw.translation.x, mw.translation.y, mw.translation.z], float)
    R = np.array([[mw[r][c] for c in range(3)] for r in range(3)], float)
    return t, R


def size_bucket(area_px: float, small_max: int, medium_max: int) -> str:
    if area_px < small_max * small_max:
        return "small"
    if area_px < medium_max * medium_max:
        return "medium"
    return "large"


def render_one(
    out_dir: Path,
    index: int,
    scene_params: SampledScene,
    sampler: SceneSampler,
    intr: CameraIntrinsics,
    bg_dir: Path | None,
    engine: str,
    samples: int,
    class_id: int,
    size_buckets: tuple[int, int],
) -> dict | None:
    """Tek bir örnek render eder ve etiketini yazar. Dönüş: meta dict veya None (kadraj dışı)."""
    _reset_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = intr.width
    scene.render.resolution_y = intr.height
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = samples
    scene.render.film_transparent = False

    _setup_world_background(scene_params, bg_dir)
    _setup_sun(scene_params, sampler)
    cam = _setup_camera(intr)

    if scene_params.asset_name.startswith("asset:"):
        drone = _assets.load_asset(scene_params.asset_name.split(":", 1)[1], scene_params.asset_name)
    else:
        drone = _assets.build_procedural_quadcopter(scene_params.asset_name)
    if scene_params.paint_rgb:
        _assets.apply_body_material(drone, scene_params.paint_rgb, scene_params.texture_realism)

    _place_drone(drone, scene_params)

    # Bbox'ı mesh köşelerinden hesapla (tek doğruluk kaynağı: geometry.py)
    t, R = _camera_pose(cam)
    verts = _world_vertices(drone)
    uv, in_front = project_points(verts, t, R, intr)
    bb = bbox_from_points(uv, in_front, intr.width, intr.height)
    if bb is None:
        return None  # drone kadraj dışı; bu örneği at (etiketsiz arka plan ayrı üretilir)

    area_px = (bb[2] - bb[0]) * (bb[3] - bb[1])
    bucket = size_bucket(area_px, *size_buckets)

    stem = f"synth_{index:07d}"
    img_path = out_dir / "images" / f"{stem}.png"
    lbl_path = out_dir / "labels" / f"{stem}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)

    scene.render.filepath = str(img_path)
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)

    xc, yc, w, h = to_yolo(bb, intr.width, intr.height)
    lbl_path.write_text(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    meta = {
        "stem": stem,
        "bbox_xyxy_px": [round(v, 2) for v in bb],
        "area_px": round(area_px, 1),
        "size_bucket": bucket,
        "distance_m": round(scene_params.distance_m, 3),
        "yaw_deg": round(scene_params.yaw_deg, 2),
        "pitch_deg": round(scene_params.pitch_deg, 2),
        "roll_deg": round(scene_params.roll_deg, 2),
        "sun_elevation_deg": round(scene_params.sun_elevation_deg, 2),
        "sun_strength": round(scene_params.sun_strength, 3),
        "focal_mm": round(scene_params.focal_mm, 2),
        "texture_realism": scene_params.texture_realism,
        "background": scene_params.extras.get("background"),
        "asset": scene_params.asset_name,
    }
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False))
    return meta


def generate(
    out_dir: str | Path,
    n: int,
    factor_name: str = "full",
    width: int = 1280,
    height: int = 720,
    engine: str = "BLENDER_EEVEE",
    samples: int = 32,
    bg_dir: str | Path | None = None,
    assets: list[str] | None = None,
    paint_rgb: tuple[int, int, int] | None = (60, 60, 65),
    class_id: int = 0,
    size_buckets: tuple[int, int] = (32, 96),
    seed: int = 0,
    manifest_name: str = "manifest.jsonl",
) -> dict:
    """n sentetik örnek üretir. factor_name ablasyon anahtarını belirler.

    Dönüş: özet istatistik (üretilen, atlanan, boyut katman dağılımı)."""
    if bpy is None:
        raise RuntimeError("generate() yalnızca bpy ortamında çalışır. "
                           "`pip install bpy` veya `blender --background --python`.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    switches = FactorSwitches.from_name(factor_name)
    sampler = SceneSampler(
        switches=switches,
        ranges=SceneRanges(),
        n_backgrounds=50,
        assets=assets or ["quadcopter_generic"],
        seed=seed,
    )

    manifest = (out_dir / manifest_name).open("w", encoding="utf-8")
    stats = {"requested": n, "rendered": 0, "skipped": 0,
             "buckets": {"small": 0, "medium": 0, "large": 0}, "factor": factor_name}
    produced = 0
    attempts = 0
    max_attempts = n * 4  # kadraj dışı örnekleri telafi için
    while produced < n and attempts < max_attempts:
        attempts += 1
        sp = sampler.sample()
        sp.paint_rgb = paint_rgb
        intr = CameraIntrinsics(width=width, height=height, focal_mm=sp.focal_mm)
        meta = render_one(out_dir, produced, sp, sampler, intr, bg_dir_path(bg_dir),
                          engine, samples, class_id, size_buckets)
        if meta is None:
            stats["skipped"] += 1
            continue
        produced += 1
        stats["rendered"] += 1
        stats["buckets"][meta["size_bucket"]] += 1
        manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")
    manifest.close()
    (out_dir / "generation_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def bg_dir_path(bg_dir: str | Path | None) -> Path | None:
    if bg_dir is None:
        return None
    p = Path(bg_dir)
    return p if p.is_dir() else None
