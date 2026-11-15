"""Download CC0 models and textures from Poly Haven (polyhaven.com) into the Unreal project's SourceAssets.

    .venv/bin/python tools/fetch_polyhaven.py                # everything listed below
Models come as glTF + textures at 2k; surface materials as 2k PNG/JPG maps (colour, normal, roughness).
Licence: CC0 (public domain). Files already present are skipped.
"""
import json, subprocess, sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "unreal/RemoteHandsBedroom/SourceAssets/PolyHaven"
MODELS = ["GothicBed_01", "painted_wooden_nightstand", "side_table_01", "side_table_tall_01", "modern_arm_chair_01",
          "drawer_cabinet", "Television_01", "desk_lamp_arm_01", "modern_ceiling_lamp_01", "alarm_clock_01", "jug_01",
          "tea_set_01", "medical_box", "book_encyclopedia_set_01", "decorative_book_set_01", "classic_laptop", "boombox",
          "wicker_basket_01", "cardboard_box_01", "trashbag", "throw_pillows_01", "potted_plant_04", "potted_plant_01",
          "standing_picture_frame_01", "hanging_picture_frame_01", "wooden_bookshelf_worn", "plastic_bottle_gallon",
          "WoodenTable_03", "wine_bottles_01", "wheelchair_01",
          # the demo room's furnishing (15 Nov)
          "sofa_03", "coffee_table_round_01", "ornate_mirror_01", "wall_clock", "vintage_grandfather_clock_01",
          "vintage_wooden_drawer_01", "ceramic_vase_03", "fancy_picture_frame_01", "hanging_picture_frame_02",
          "mantel_clock_01", "Ottoman_01", "brass_vase_01", "wicker_basket_02", "vintage_suitcase"]
HDRIS = ["garden_nook"]             # what is seen through the window, and the daylight
TEXTURES = sys.argv[2:] if len(sys.argv) > 2 and sys.argv[1] == "--textures" else []
UA = "remote-hands-asset-fetch"


def get(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    subprocess.run(["curl", "-sS", "-L", "--fail", "-m", "300", "-A", UA, "-o", str(dest), url], check=True)
    return dest.stat().st_size


def api(path):
    r = subprocess.run(["curl", "-sS", "--fail", "-m", "60", "-A", UA, "https://api.polyhaven.com/" + path],
                       check=True, capture_output=True)
    return json.loads(r.stdout)


total = 0
if not TEXTURES:
    for m in MODELS:
        f = api("files/" + m).get("gltf")
        if not f:
            print(f"{m}: no glTF version, skipped", flush=True)
            continue
        res = "2k" if "2k" in f else sorted(f)[0]
        g = f[res]["gltf"]
        n = get(g["url"], OUT / "models" / m / (m + ".gltf"))
        for rel, inc in g.get("include", {}).items():
            n += get(inc["url"], OUT / "models" / m / rel)
        total += n
        print(f"{m}: {res}, {n / 1e6:.1f} MB new", flush=True)
for t in TEXTURES:
    f = api("files/" + t)
    n = 0
    for key, name in (("Diffuse", "color"), ("nor_dx", "normal"), ("Rough", "roughness")):
        if key in f:
            fmt = "jpg" if "jpg" in f[key]["2k"] else "png"
            n += get(f[key]["2k"][fmt]["url"], OUT / "textures" / t / f"{t}_{name}.{fmt}")
    total += n
    print(f"texture {t}: {n / 1e6:.1f} MB new ({[k for k in f if k in ('Diffuse', 'nor_dx', 'Rough')]})", flush=True)
if not TEXTURES:
    for h in HDRIS:
        n = get(api("files/" + h)["hdri"]["4k"]["hdr"]["url"], OUT / "hdris" / (h + ".hdr"))
        total += n
        print(f"hdri {h}: {n / 1e6:.1f} MB new", flush=True)
print(f"total new: {total / 1e6:.0f} MB -> {OUT}")
