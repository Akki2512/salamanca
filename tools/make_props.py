"""The small things modelled for the demo room: the brass call bell and the glass of water.

    .venv/bin/python tools/make_props.py

Writes OBJ files (cm, z up) to unreal/RemoteHandsBedroom/SourceAssets/Props, which build_bedroom.py imports.
The bell stands on z = 0. The glass's origin is where it is gripped, 5.5 cm above its base (home.py grip_z).
The sweater is made by tools/make_sweater.py (Blender)."""
from pathlib import Path

import numpy as np
import trimesh

OUT = Path(__file__).resolve().parents[1] / "unreal" / "RemoteHandsBedroom" / "SourceAssets" / "Props"
OUT.mkdir(parents=True, exist_ok=True)
rev = lambda prof, n=64: trimesh.creation.revolve(np.asarray(prof, float), sections=n)

# a desk call bell: dark base plate, polished dome, plunger on top
base = rev([(0, 0), (5.6, 0), (5.8, 0.25), (5.8, 0.9), (5.3, 1.2), (0, 1.2)])
dome = rev([(0, 1.2)] + [(5.1 * np.cos(a), 1.2 + 4.3 * np.sin(a)) for a in np.linspace(0, np.pi / 2 * 0.93, 22)] + [(0.9, 5.55), (0, 5.55)])
stem = rev([(0, 5.5), (0.75, 5.5), (0.75, 6.6), (0, 6.6)], 24)
knob = rev([(0, 6.5), (1.25, 6.5), (1.45, 6.8), (1.25, 7.25), (0, 7.35)], 32)
GRIP = 5.5
glass = rev([(0, 0), (3.0, 0), (3.45, 11.0), (3.25, 11.0), (2.85, 0.8), (0, 0.8)], 48)
water = rev([(0, 0.85), (2.83, 0.85), (3.12, 8.0), (0, 8.0)], 48)
for m in (glass, water):
    m.apply_translation((0, 0, -GRIP))

# a curtain panel: a sheet 1 m wide and 1 m tall hanging in folds (scaled in the level), with texture coordinates
def curtain(path, folds=7, nx=84, nz=2, depth=3.2):
    v, vt, f = [], [], []
    for j in range(nz):
        for i in range(nx):
            u = i / (nx - 1)
            v.append((u * 100.0 - 50.0, depth * np.sin(u * folds * 2 * np.pi) + 0.8 * np.sin(u * 23.0), j / (nz - 1) * 100.0))
            vt.append((u * 2.4, j / (nz - 1)))               # the cloth is longer than the panel is wide: it is gathered
    for j in range(nz - 1):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i + 1, j * nx + i + 2, (j + 1) * nx + i + 2, (j + 1) * nx + i + 1
            f += [(a, b, c), (a, c, d)]
    with open(path, "w") as out:
        out.write("".join("v %.3f %.3f %.3f\n" % p for p in v) + "".join("vt %.4f %.4f\n" % t for t in vt)
                  + "".join("f %d/%d %d/%d %d/%d\n" % (a, a, b, b, c, c) for a, b, c in f))
    print("curtain", len(v), "points")


curtain(OUT / "curtain.obj")
for name, mesh in (("call_bell_base", base), ("call_bell_dome", dome), ("call_bell_button", trimesh.util.concatenate([stem, knob])),
                   ("glass", glass), ("glass_water", water)):
    mesh.fix_normals()
    mesh.export(OUT / f"{name}.obj", include_normals=False, include_texture=False)
    print(name, np.round(mesh.bounds, 2).tolist())
