"""Scripted pick and place on the Isaac sim through the robot node; reports speed, tracking and safety.

On the pod (2nd sim instance):
    export REMOTE_HANDS_SIM=instance AALTO_REPO=/workspace/aalto/repo AALTO_DDS_DOMAIN=7 AALTO_CAM_PORT_BASE=56555 AALTO_SCENE_PORT=56580 AALTO_RGBD_PORT=56560
    python tools/sim_trial.py --seconds 30 --out /workspace/aalto/rh_trial
"""
import argparse, json, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="sim"); ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--out", default="rh_trial"); ap.add_argument("--scale", type=float, default=0.75)
a = ap.parse_args()
logging.basicConfig(level=logging.WARNING)
print("clock:", "set after init")
logging.getLogger("aaltonomous.safety").setLevel(logging.ERROR)
from remote_hands.robot_node import HZ, RobotNode
from remote_hands.sources import SyntheticSource

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
t_init = time.monotonic()
node = RobotNode(backend=a.backend, realtime=True, scale=a.scale, log_path=str(out / "telemetry.jsonl"))
print("pacing on:", "sim clock" if node.env.clock is not None else "WALL time (no sim clock)")
print("init %.1f s; layout: %s" % (time.monotonic() - t_init, [(o.get("color"), o.get("shape")) for o in node.env.layout] if isinstance(node.env.layout, list) else type(node.env.layout)))
src = SyntheticSource()
rows, w0, t_drive0 = [], time.monotonic(), None
n = int(a.seconds * HZ)
for k in range(n):
    if node.state != "homing" and t_drive0 is None:
        t_drive0 = node.t
    if t_drive0 is not None:
        node.push(src.at(node.t - t_drive0, k))
    w = time.monotonic(); tel = node.tick(); dt_wall = time.monotonic() - w
    meas = node.kin.grasp_pose("right", np.asarray(node.obs["arm_q"], float))[0]
    cmd = node.grasp_points(node.arm_cmd)["right"]
    tgt = tel["target"].get("right")
    rows.append((tel["t"], dt_wall, tel["state"], None if tgt is None else float(np.linalg.norm(cmd - np.array(tgt))),
                 float(np.linalg.norm(meas - cmd)), tel["safety_mode"], tel["table_guard"], tel["t"] - (t_drive0 or 0)))
wall = time.monotonic() - w0
sim_t = rows[-1][0]
dts = np.array([r[1] for r in rows])
print("ticks %d | sim time %.1f s | wall %.1f s | real-time factor %.2f | tick wall median %.0f ms p95 %.0f ms"
      % (n, sim_t, wall, sim_t / wall, np.median(dts) * 1e3, np.percentile(dts, 95) * 1e3))
print("homing took %.1f sim s" % (t_drive0 or -1))
for label, ts in (("over object (end of wait)", 4.5), ("over bin (end of wait)", 8.4)):
    r = min((r for r in rows if r[2] == "driving"), key=lambda r: abs(r[7] - ts), default=None)
    if r:
        print("%-28s command-to-target %.1f mm | measured-to-command %.1f mm" % (label, r[3] * 1e3, r[4] * 1e3))
drv = [r for r in rows if r[2] == "driving"]
print("measured-to-command while driving: median %.1f mm, max %.1f mm" % (np.median([r[4] for r in drv]) * 1e3, max(r[4] for r in drv) * 1e3))
print("table guard ticks:", sum(r[6] for r in rows), "| safety:", node.env.filter.summary())
try:
    from PIL import Image
    for cam in ("head_rgb", "right_wrist_rgb"):
        Image.fromarray(node.obs[cam]).save(out / f"{cam}.jpg", quality=85)
    print("camera mean brightness:", {c: round(float(node.obs[c].mean()), 1) for c in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")})
except Exception as e:
    print("image save failed:", e)
try:
    from aaltonomous.sim.scene_client import SceneClient
    st = SceneClient("127.0.0.1").state()
    print("object record example:", json.dumps(st["objects"][0])[:300])
    print("objects on the counter:", [(o.get("name"), [round(v, 3) for v in o.get("pos", o.get("p", []))[:3]]) for o in st["objects"] if o.get("pos", o.get("p", [0, 0, -1]))[2] > -0.1])
except Exception as e:
    print("scene state failed:", repr(e)[:200])
node.close()
