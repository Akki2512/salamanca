"""Everyday-looking items for the simulator's home scene (copied into our simulator copy's aalto_task folder).

The simulator's objects are 45 mm blocks, one flat colour each. Here every colour + shape pair that is an item
(remote_hands/home.py) gets a body with the item's real outline plus extra visual-only parts in other colours
(cap, label, lid, handle). The body keeps the 45 mm width and the same resting height the grasp was tuned for;
the body alone collides (convex hull), so the physics is as before.

Mesh frame: metres, origin at the centre of the 45 mm block the item stands in (base at z = -S/2).
Mesh +y points away from the robot when an item is placed with yaw 0.
"""
from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

S = 0.045
H = S / 2
N = 48


def _cyl(r, z0, z1, sections=N):
    m = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=sections)
    m.apply_translation([0, 0, (z0 + z1) / 2])
    return m


def _rev(profile, sections=N):
    """Solid of revolution about z from [(radius, z), ...] going bottom to top."""
    return trimesh.creation.revolve(np.asarray(profile, float), sections=sections)


def _box(sx, sy, z0, z1, x=0.0, y=0.0):
    m = trimesh.creation.box(extents=[sx, sy, z1 - z0])
    m.apply_translation([x, y, (z0 + z1) / 2])
    return m


def _band(r, z0, z1, thick=0.0006):
    return trimesh.creation.annulus(r_min=r - 0.002, r_max=r + thick, height=z1 - z0).apply_translation([0, 0, (z0 + z1) / 2])


def _box_band(s, z0, z1, thick=0.0006):
    return _box(s + 2 * thick, s + 2 * thick, z0, z1)


def pill_bottle():
    body = _rev([(0, -H), (H - 0.001, -H), (H, -H + 0.001), (H, 0.006), (0.017, 0.010), (0.017, 0.011), (0, 0.011)])
    cap = _rev([(0, 0.0105), (0.0185, 0.0105), (0.0185, 0.0235), (0.0175, 0.0245), (0, 0.0245)], sections=24)
    ribs = [_box(0.0012, 0.0012, 0.0115, 0.0232, x=0.0187 * np.cos(a), y=0.0187 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 20, endpoint=False)]
    label = _band(H, -0.016, 0.001)
    stripe = _band(H + 0.0003, -0.0075, -0.0045)
    return body, [(cap, (0.96, 0.96, 0.94), 0.5, 0.0), (trimesh.util.concatenate(ribs), (0.90, 0.90, 0.88), 0.6, 0.0),
                  (label, (0.10, 0.50, 0.56), 0.6, 0.0), (stripe, (0.97, 0.97, 0.97), 0.6, 0.0)]


def soda_can():
    body = _rev([(0, -H), (0.0185, -H), (H, -H + 0.004), (H, 0.0175), (0.0195, H - 0.001), (0.0195, H), (0, H)])
    top = _rev([(0, H - 0.0004), (0.0200, H - 0.0004), (0.0200, H + 0.0006), (0.0182, H + 0.0006), (0.0182, H + 0.0001), (0, H + 0.0001)])
    tab = _box(0.006, 0.011, H + 0.0001, H + 0.0011, y=0.004)
    foot = _rev([(0, -H - 0.0001), (0.0188, -H - 0.0001), (H + 0.0004, -H + 0.004), (H + 0.0004, -H + 0.0045), (0, -H + 0.0045)])
    wave = _band(H, -0.006, 0.003)
    return body, [(top, (0.80, 0.81, 0.83), 0.25, 0.9), (tab, (0.70, 0.71, 0.73), 0.3, 0.9),
                  (foot, (0.80, 0.81, 0.83), 0.25, 0.9), (wave, (0.97, 0.97, 0.97), 0.5, 0.0)]


def cup():
    body = _rev([(0, -H), (0.0195, -H), (H, -H + 0.006), (H, H), (0, H)])
    coffee = _cyl(0.0195, H - 0.0003, H + 0.0004)
    rim = trimesh.creation.annulus(r_min=0.0195, r_max=H + 0.0002, height=0.0012).apply_translation([0, 0, H])
    ring = trimesh.creation.torus(major_radius=0.012, minor_radius=0.0035, major_sections=24, minor_sections=10)
    ring.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))   # stand it up (in the y-z plane)
    ring.apply_translation([0, H + 0.006, 0.0])                                           # on the far side
    return body, [(coffee, (0.20, 0.11, 0.05), 0.3, 0.0), (rim, (0.93, 0.94, 0.97), 0.4, 0.0),
                  (ring, (0.12, 0.27, 0.78), 0.7, 0.0)]


def tea_box():
    body = _box(S, S, -H, H)
    lid = _box(S + 0.0016, S + 0.0016, H - 0.008, H + 0.0006)
    label = _box_band(S, -0.014, 0.004)
    leaf = _cyl(0.008, H + 0.0006, H + 0.0012, sections=24)
    return body, [(lid, (0.80, 0.62, 0.05), 0.6, 0.0), (label, (0.13, 0.36, 0.20), 0.7, 0.0),
                  (leaf, (0.13, 0.36, 0.20), 0.7, 0.0)]


def juice_box():
    w, z_sh, z_top = H, 0.013, H + 0.005          # walls up to the shoulder, then a gable roof with its ridge along x
    v = np.array([[-w, -w, -H], [w, -w, -H], [w, w, -H], [-w, w, -H], [-w, -w, z_sh], [w, -w, z_sh], [w, w, z_sh],
                  [-w, w, z_sh], [-w, 0, z_top], [w, 0, z_top]])
    body = trimesh.convex.convex_hull(v)
    fin = _box(S, 0.0016, z_top - 0.001, z_top + 0.004)                    # the sealed strip on top
    label = _box_band(S, -0.012, 0.004)
    fruit = _cyl(0.009, -0.0005, 0.0005, sections=24)
    fruit.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    fruit.apply_translation([0, -w - 0.0009, -0.004])                      # an orange on the side facing the robot
    cap = _cyl(0.0055, 0.0, 0.005, sections=20)
    cap.apply_transform(trimesh.transformations.rotation_matrix(-np.arctan2(z_top - z_sh, w), [1, 0, 0]))
    cap.apply_translation([0.009, 0.011, (z_sh + z_top) / 2 + 0.001])
    return body, [(fin, (0.93, 0.93, 0.90), 0.6, 0.0), (label, (0.97, 0.97, 0.95), 0.6, 0.0),
                  (fruit, (0.95, 0.55, 0.08), 0.6, 0.0), (cap, (0.95, 0.95, 0.95), 0.4, 0.0)]


ITEMS = {"white_cylinder": pill_bottle, "red_cylinder": soda_can, "blue_cylinder": cup,
         "yellow_cube": tea_box, "green_cube": juice_box}


@configclass
class ItemMeshCfg(sim_utils.MeshCfg):
    """A rigid item: `item` = "<colour>_<shape>". Anything not in ITEMS falls back to the plain block STL at `path`."""
    func: object = None
    item: str = ""
    path: str = ""

    def __post_init__(self):
        self.func = spawn_item


def _add_visual(stage, path, mesh, rgb, roughness, metallic):
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr([Gf.Vec3f(float(a), float(b), float(c)) for a, b, c in mesh.vertices])
    g.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    g.CreateFaceVertexIndicesAttr([int(i) for i in np.asarray(mesh.faces).reshape(-1)])
    g.CreateSubdivisionSchemeAttr("none")
    g.CreateDoubleSidedAttr(True)
    mat = UsdShade.Material.Define(stage, path + "_material")
    sh = UsdShade.Shader.Define(stage, path + "_material/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgb]))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)


@clone
def spawn_item(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
    extras = []
    if cfg.item in ITEMS:
        body, extras = ITEMS[cfg.item]()
    else:
        body = trimesh.load(cfg.path, force="mesh")
        body.apply_scale(0.001)
        lo, hi = body.bounds
        body.apply_translation([0.0, 0.0, -(lo[2] + hi[2]) / 2])
    prim = _spawn_mesh_geom_from_mesh(prim_path, cfg, body, translation, orientation, **kwargs)
    stage = prim.GetStage()
    for k, (mesh, rgb, rough, metal) in enumerate(extras):
        _add_visual(stage, f"{prim_path}/geometry/part{k}", mesh, rgb, rough, metal)
    return prim


if __name__ == "__main__":      # offline check of the meshes (no simulator needed)
    for name, fn in ITEMS.items():
        body, extras = fn()
        lo, hi = body.bounds
        print(f"{name:15s} body {np.round(hi - lo, 4).tolist()} base z {lo[2]:+.4f} top z {hi[2]:+.4f} "
              f"watertight {body.is_watertight} parts {len(extras)}")
