"""Drone 3B varlıkları: dosyadan yükleme veya prosedürel üretim.

Repo HİÇBİR harici varlık olmadan çalışmalı (portföy için önemli: klonlayan
herkes tek komutla üretebilsin). Bu yüzden varlık dosyası verilmezse plausible
bir quadcopter silueti prosedürel olarak kurulur. Gerçek .obj/.glb/.fbx
modelleri `assets/drone_models/` içine konup config'ten gösterilebilir; bunlar
domain çeşitliliğini (ve dolayısıyla transfer başarısını) artırır.

Bu modül yalnızca bpy ortamında (Blender içi) çağrılır.
"""

from __future__ import annotations

import math
from pathlib import Path

try:
    import bpy  # type: ignore
except ImportError:  # bpy yoksa modül import edilebilir ama fonksiyonlar çağrılamaz
    bpy = None  # type: ignore


def _require_bpy() -> None:
    if bpy is None:
        raise RuntimeError("Bu fonksiyon yalnızca Blender (bpy) ortamında çalışır.")


def build_procedural_quadcopter(name: str = "quadcopter_generic", size_m: float = 0.45):
    """Merkez gövde + 4 kol + 4 rotor diski olan basit bir quadcopter kurar.

    size_m: en uzun açıklık (köşegen motor-motor) yaklaşık bu değerdir. Gerçek
    küçük drone'lar ~0.2–0.5 m aralığında; mesafe-ölçek kestirimi buna dayanır.
    Dönüş: tüm parçaları birleştiren tek bir Blender nesnesi.
    """
    _require_bpy()
    half = size_m / 2.0
    arm_len = half * 0.9
    objs = []

    # Gövde (yassı kutu)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    body = bpy.context.object
    body.scale = (size_m * 0.28, size_m * 0.28, size_m * 0.10)
    objs.append(body)

    # 4 kol + rotor (X düzeninde)
    for i in range(4):
        ang = math.radians(45 + i * 90)
        dx, dy = math.cos(ang) * arm_len, math.sin(ang) * arm_len
        # kol (ince çubuk)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(dx / 2, dy / 2, 0))
        arm = bpy.context.object
        arm.scale = (abs(arm_len) * 0.5 + size_m * 0.05, size_m * 0.03, size_m * 0.03)
        arm.rotation_euler = (0, 0, ang)
        objs.append(arm)
        # motor (küçük silindir)
        bpy.ops.mesh.primitive_cylinder_add(radius=size_m * 0.05, depth=size_m * 0.06,
                                             location=(dx, dy, 0))
        objs.append(bpy.context.object)
        # rotor diski (ince disk)
        bpy.ops.mesh.primitive_cylinder_add(radius=size_m * 0.16, depth=size_m * 0.01,
                                            location=(dx, dy, size_m * 0.04))
        objs.append(bpy.context.object)

    # Birleştir
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    drone = bpy.context.object
    drone.name = name
    # Orijini geometri merkezine al (dönüşler düzgün olsun)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    drone.location = (0, 0, 0)
    return drone


def load_asset(path: str | Path, name: str = "drone_asset"):
    """Harici 3B model yükler (.obj/.glb/.gltf/.fbx) ve tek nesne döndürür."""
    _require_bpy()
    path = Path(path)
    ext = path.suffix.lower()
    before = set(bpy.context.scene.objects)
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise ValueError(f"desteklenmeyen varlık uzantısı: {ext}")
    new = [o for o in bpy.context.scene.objects if o not in before and o.type == "MESH"]
    if not new:
        raise RuntimeError(f"{path} içinden mesh yüklenemedi")
    for o in new:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    drone = bpy.context.object
    drone.name = name
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    drone.location = (0, 0, 0)
    return drone


def apply_body_material(obj, rgb: tuple[int, int, int], realistic: bool = True) -> None:
    """Gövdeye renk/malzeme atar. realistic=False düz (emisyonsuz, düşük detay) =
    doku-gerçekçiliği ablasyonu için."""
    _require_bpy()
    mat = bpy.data.materials.new(name=f"{obj.name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    r, g, b = (c / 255.0 for c in rgb)
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
        if realistic:
            # hafif metalik/pürüzlü gövde
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = 0.3
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.5
        else:
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = 0.0
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 1.0
    obj.data.materials.clear()
    obj.data.materials.append(mat)
