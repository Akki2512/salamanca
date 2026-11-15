"""Brings the room's surface textures from the team's browser room (zain-shp/junction-2026, branch
all/pass-me-that, projects/pass-me-that/room/look/assets/bedroom) into the Unreal project's SourceAssets, in the
shape Scripts/build_bedroom.py's surface_material() reads: textures/<name>/<name>_{color,normal,roughness}.jpg.

    .venv/bin/python tools/import_room_textures.py <path to .../look/assets/bedroom>

Their normal maps are for three.js (OpenGL: green up); Unreal wants DirectX (green down), so green is inverted.
A set without a roughness map gets a flat one. Then run Scripts/import_polyhaven.py and build_bedroom.py.
"""
import sys
from pathlib import Path

from PIL import Image, ImageOps

SRC = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "unreal/RemoteHandsBedroom/SourceAssets/PolyHaven/textures"
# our name: (colour, normal or None, roughness file or a flat value 0..1)
SETS = {"room_oak": ("surf_oak_color.jpg", "surf_oak_normal.jpg", "surf_oak_rough.jpg"),
        "room_plaster": ("surf_plaster_color.jpg", "surf_plaster_normal.jpg", "surf_plaster_rough.jpg"),
        "room_rug": ("surf_rug_color.jpg", "surf_rug_normal.jpg", 0.95),
        "room_knit": ("surf_knit_color.jpg", "surf_knit_normal.jpg", 1.0),
        "room_picture": ("surf_picture.jpg", None, 0.85),
        "room_ceiling": ("surf_ceiling_color.jpg", None, 0.95)}
# Fabric sets (fabric/<name>_detail.jpg is a grey weave, _orm.jpg holds roughness in its green channel): the weave
# is coloured here. our name: (fabric name, colour 0..1)
FABRICS = {"room_linen": ("linen", (0.80, 0.74, 0.62))}
for name, (col, nor, rough) in SETS.items():
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    c = Image.open(SRC / col).convert("RGB")
    c.save(d / f"{name}_color.jpg", quality=92)
    if nor:
        r, g, b = Image.open(SRC / nor).convert("RGB").split()
        n = Image.merge("RGB", (r, ImageOps.invert(g), b))
    else:
        n = Image.new("RGB", c.size, (128, 128, 255))            # flat
    n.save(d / f"{name}_normal.jpg", quality=92)
    (Image.open(SRC / rough).convert("RGB") if isinstance(rough, str) else Image.new("RGB", (64, 64), (int(rough * 255),) * 3)) \
        .save(d / f"{name}_roughness.jpg", quality=92)
    print(name, c.size)
for name, (fab, rgb) in FABRICS.items():
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    weave = Image.open(SRC / "fabric" / f"{fab}_detail.jpg").convert("L")
    mean = sum(weave.resize((32, 32)).getdata()) / 1024.0
    c = Image.merge("RGB", [weave.point(lambda v, k=k: max(0, min(255, int(v / mean * k * 255)))) for k in rgb])
    c.save(d / f"{name}_color.jpg", quality=92)
    r, g, b = Image.open(SRC / "fabric" / f"{fab}_normal.jpg").convert("RGB").split()
    Image.merge("RGB", (r, ImageOps.invert(g), b)).save(d / f"{name}_normal.jpg", quality=92)
    rough = Image.open(SRC / "fabric" / f"{fab}_orm.jpg").convert("RGB").split()[1]
    Image.merge("RGB", (rough, rough, rough)).save(d / f"{name}_roughness.jpg", quality=92)
    print(name, c.size)
