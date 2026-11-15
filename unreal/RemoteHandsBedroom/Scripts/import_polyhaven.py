"""Imports everything under SourceAssets/PolyHaven into /Game/PolyHaven and writes what it got to
SourceAssets/PolyHaven/imported.json (static meshes per model with their sizes).

    UnrealEditor-Cmd RemoteHandsBedroom.uproject -ExecutePythonScript=Scripts/import_polyhaven.py -unattended -nullrhi
"""
import json, os, traceback
import unreal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "SourceAssets", "PolyHaven")
report = {"models": {}, "textures": {}}
try:
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    for kind, ext in (("models", ".gltf"), ("textures", None)):
        for name in sorted(os.listdir(os.path.join(SRC, kind))):
            folder = os.path.join(SRC, kind, name)
            if not os.path.isdir(folder):
                continue
            dest = "/Game/PolyHaven/%s/%s" % (kind.capitalize(), name)
            files = [f for f in sorted(os.listdir(folder)) if (f.endswith(ext) if ext else f.lower().endswith((".jpg", ".png")))]
            if not unreal.EditorAssetLibrary.does_directory_have_assets(dest):
                tasks = []
                for f in files:
                    t = unreal.AssetImportTask()
                    t.filename = os.path.join(folder, f)
                    t.destination_path = dest
                    t.automated, t.replace_existing, t.save = True, True, True
                    tasks.append(t)
                tools.import_asset_tasks(tasks)
            entry = {"meshes": [], "other": {}}
            for path in unreal.EditorAssetLibrary.list_assets(dest, recursive=True):
                a = unreal.load_asset(path)
                if isinstance(a, unreal.StaticMesh):
                    b = a.get_bounding_box()
                    entry["meshes"].append({"path": path.split(".")[0], "min": [round(v, 1) for v in b.min.to_tuple()],
                                            "max": [round(v, 1) for v in b.max.to_tuple()],
                                            "materials": [str(m.material_interface.get_name()) if m.material_interface else None
                                                          for m in a.static_materials]})
                elif a is not None:
                    c = a.get_class().get_name()
                    entry["other"].setdefault(c, []).append(path.split(".")[0])
            report[kind][name] = entry
    # sky panoramas (.hdr): each becomes a cube texture, used for the sky light and for the view out of the window
    report["hdris"] = {}
    hd = os.path.join(SRC, "hdris")
    for f in sorted(os.listdir(hd)) if os.path.isdir(hd) else []:
        if f.endswith(".hdr"):
            path = "/Game/PolyHaven/HDRI/" + f[:-4]
            if not unreal.EditorAssetLibrary.does_asset_exist(path):
                t = unreal.AssetImportTask()
                t.filename, t.destination_path = os.path.join(hd, f), "/Game/PolyHaven/HDRI"
                t.automated, t.replace_existing, t.save = True, True, True
                tools.import_asset_tasks([t])
            a = unreal.load_asset(path)
            report["hdris"][f[:-4]] = {"path": path, "class": a.get_class().get_name() if a else None}
    unreal.EditorAssetLibrary.save_directory("/Game/PolyHaven", only_if_is_dirty=True, recursive=True)
except Exception:
    report["error"] = traceback.format_exc()
json.dump(report, open(os.path.join(SRC, "imported.json"), "w"), indent=1)
