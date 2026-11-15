"""Turn the G1's URDF + STL parts into what Unreal needs: one OBJ per link and a JSON description of the tree.

    .venv/bin/python tools/export_g1_for_unreal.py

Output: unreal/RemoteHandsBedroom/SourceAssets/G1/<link>.obj and g1.json. OBJ vertices are written in the
link's own frame in centimetres with the URDF's axes (x forward, y left, z up). Unreal's OBJ importer mirrors y
on import (measured on UE 5.8), which is exactly the conversion to Unreal's left-handed frame.
g1.json: for every link its parent, the joint that moves it, and its pose relative to the parent at joint angle 0,
already in Unreal's convention (cm, y mirrored).
"""
import json, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parents[1]
G1 = HERE.parent / "aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1"
OUT = HERE / "unreal/RemoteHandsBedroom/SourceAssets/G1"
OUT.mkdir(parents=True, exist_ok=True)


def origin(el):
    o = el.find("origin") if el is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return np.array(xyz), R.from_euler("xyz", rpy)        # URDF rpy = fixed-axis roll, pitch, yaw


root = ET.parse(G1 / "g1_body29_hand14.urdf").getroot()
joints = {j.find("child").get("link"): j for j in root.findall("joint")}
links = []
for l in root.findall("link"):
    name = l.get("name")
    j = joints.get(name)
    p, q = origin(j) if j is not None else (np.zeros(3), R.identity())
    x, y, z, w = q.as_quat()
    entry = {"name": name, "parent": j.find("parent").get("link") if j is not None else None,
             "joint": j.get("name") if j is not None else None, "joint_type": j.get("type") if j is not None else None,
             # Unreal: cm, y mirrored. Mirroring y turns the quaternion (x, y, z, w) into (-x, y, -z, w).
             "pos_cm": [round(p[0] * 100, 4), round(-p[1] * 100, 4), round(p[2] * 100, 4)],
             "quat_xyzw": [round(-x, 7), round(y, 7), round(-z, 7), round(w, 7)], "mesh": None, "color": None}
    v = l.find("visual")
    m = v.find("geometry/mesh") if v is not None else None
    if m is not None:
        mesh = trimesh.load(G1 / m.get("filename"), force="mesh")
        vp, vq = origin(v)
        mesh.apply_transform(np.vstack([np.c_[vq.as_matrix(), vp], [0, 0, 0, 1]]))
        mesh.merge_vertices()
        mesh.apply_scale(100.0)      # Unreal's OBJ import reads numbers as centimetres and mirrors y itself (measured)
        mesh.export(OUT / f"{name}.obj", include_normals=False, include_texture=False)
        c = v.find("material/color")
        entry["mesh"] = f"{name}.obj"
        entry["color"] = "dark" if c is not None and float(c.get("rgba").split()[0]) < 0.4 else "light"
        entry["bounds_cm"] = np.round(mesh.bounds, 2).tolist()
    links.append(entry)
json.dump({"links": links}, open(OUT / "g1.json", "w"), indent=1)
print(len(links), "links,", sum(1 for e in links if e["mesh"]), "meshes ->", OUT)
