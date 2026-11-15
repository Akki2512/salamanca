"""Run ON THE POD: put the home scene into our own copy of the simulator repo (never the shared one).

    python /workspace/aalto/remote-hands/home/apply_to_sim_copy.py [/workspace/aalto/repo_home]

Copies the scene file, the table texture and home_items.py into the copy and makes its scene_cfg.py spawn
objects through home_items.ItemMeshCfg. Safe to run again.
"""
import shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
copy = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/aalto/repo_home").resolve()
if copy == Path("/workspace/aalto/repo") or not (copy / "aaltonomous/sim/aalto_task/scene_cfg.py").exists():
    sys.exit(f"refusing: {copy} is the shared repo or not a simulator copy")
shutil.copy(HERE / "scene.yaml", copy / "aaltonomous/scene/scene.yaml")
shutil.copy(HERE / "table_texture.png", copy / "aaltonomous/scene/assets/mat_texture.png")
shutil.copy(HERE / "home_items.py", copy / "aaltonomous/sim/aalto_task/home_items.py")
p = copy / "aaltonomous/sim/aalto_task/scene_cfg.py"
s = p.read_text()
if "home_items" not in s:
    a = "from .layout import ARM_HOME, PELVIS_W, assign, object_names, parse_name\n"
    b = '''            spawn=StlMeshCfg(
                path=str(ASSETS / f"object_{shape}.stl"),'''
    assert a in s and b in s, "scene_cfg.py is not as expected"
    s = s.replace(a, a + "from .home_items import ItemMeshCfg\n")
    s = s.replace(b, '''            spawn=ItemMeshCfg(
                item=f"{color}_{shape}", path=str(ASSETS / f"object_{shape}.stl"),''')
    p.write_text(s)
print("home scene applied to", copy)
