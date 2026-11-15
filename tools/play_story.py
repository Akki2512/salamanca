"""Plays the whole story in the node with no suit: point + trigger and joystick driving only."""
import sys, math, json, logging
import numpy as np
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("aaltonomous.safety").setLevel(logging.ERROR); logging.getLogger("remote_hands.node").setLevel(logging.WARNING)
from remote_hands.robot_node import RobotNode, EYE
from remote_hands import home, unreal_stream
node = RobotNode(backend="mock", sync_ik=True)
out = open(sys.argv[1], "w") if len(sys.argv) > 1 else None
def tick(n=1, **kw):
    for _ in range(n):
        if "drive" in kw: node.drive(*kw["drive"])
        if "aim" in kw: node.point(*kw["aim"])
        tel = node.tick()
        if out: out.write(json.dumps(unreal_stream.packet(node, tel)) + "\n")
    return tel
def aim_at(name):
    o = next(o for o in node._objects() if home.name_of(o["color"], o["shape"]) == name)
    it = home.BY_NAME[name]
    v = np.asarray(o["pos"], float) + np.asarray(it.get("aim", (0, 0, 0))) - EYE
    return float(np.degrees(np.arctan2(v[1], v[0]))), float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))
def press(name=None):
    a = aim_at(name) if name else (0.0, 60.0)
    tick(8, aim=a); node.point(a[0], a[1], True); return tick(3, aim=a)
def walk_to(x, y, yaw=None, tol=0.08):
    for _ in range(3000):
        b = node.walker.base
        d = np.array([x, y]) - b[:2]; dist = float(np.linalg.norm(d))
        if dist < tol: break
        e = (math.atan2(d[1], d[0]) - b[2] + math.pi) % (2 * math.pi) - math.pi
        tick(drive=(0.7 * max(0.0, math.cos(e)) * min(1, dist / 0.3 + 0.2), 0.0, float(np.clip(2.5 * e, -1.5, 1.5))))
    if yaw is not None:
        for _ in range(600):
            e = (yaw - node.walker.base[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(e) < 0.04: break
            tick(drive=(0.0, 0.0, float(np.clip(2.5 * e, -1.5, 1.5))))
    tick(20)
while node.state == "homing": tick()
tick(30)
S = lambda: node.scene.state()
walk_to(-0.2, 0.6, math.atan2(0.75, -0.35)); press("sweater")
for _ in range(900):
    tick()
    if node.scene.reach is None: break
print("1", S(), "assist", node.assist.state, "base", np.round(node.walker.base, 2).tolist(), "sweater", [np.round(o["pos"], 2).tolist() for o in node.env.backend.objects if o["color"] == "red"])
walk_to(-0.5, -0.05, -math.pi / 2); press(); tick(60); print("2 sweater on bed:", [np.round(o["pos"], 2).tolist() for o in node.env.backend.objects if o["color"] == "red"])
tick(90); print("3", S())
walk_to(-0.55, 0.45, 0.6); print("   glass hint from 0.9 m away, not facing it:", S()["glass"]); press()    # no pointing: near the table is enough
for _ in range(600):
    tel = tick()
    if node.scene.reach is None and tel["assist"] and tel["assist"]["state"] == "holding": break
print("4 glass:", tel["assist"]["state"] if tel["assist"] else None, S()["phase"], "coughs", S()["cough"], "rings", S()["bell"])
c = node.scene.chair; f = np.array([math.cos(c[2]), math.sin(c[2])])
stand = c[:2] + 1.15 * f + 0.35 * np.array([-f[1], f[0]])       # near him, not lined up, the hand nowhere near his lap
walk_to(-1.0, 0.8); walk_to(stand[0], stand[1], c[2] + math.pi + 0.5)
print("   glass hint beside him:", S()["glass"])
print("   handover in robot frame:", np.round(node.walker.to_pelvis(home.HANDOVER), 2).tolist(), "hand", np.round(node.grasp_points(node.arm_cmd)["right"], 2).tolist())
press(); tick(60); print("5", S(), "glass", [np.round(o["pos"], 2).tolist() for o in node.env.backend.objects if o["color"] == "white"])
side = c[:2] + 1.2 * np.array([-f[1], f[0]]); behind = c[:2] - 0.8 * f
left = np.array([-f[1], f[0]])                                        # round his left side, then in behind the chair
for k_f, k_l in ((0.35, 0.80), (-0.45, 0.85)):
    walk_to(*(c[:2] + k_f * f + k_l * left), tol=0.12)
walk_to(behind[0], behind[1], c[2])
print("   can push:", S()["can_push"], "robot at", np.round(node.walker.base, 2).tolist(), "wanted", np.round(behind, 2).tolist(), "via", np.round(side, 2).tolist()); press(); tick(40); print("6", S())
walk_to(-1.2, 0.9); walk_to(-0.6, 0.35, -math.pi / 2); tick(30); print("   chair", S()["chair"])
press(); tick(60); print("7", S())
tick(60)
node.reset_scene(); tick(120)
print("8 after reset:", S(), "base", np.round(node.walker.base, 2).tolist(), "state", node.state,
      "items", {home.name_of(o["color"], o["shape"]): np.round(o["pos"], 2).tolist() for o in node.env.backend.objects})
walk_to(-0.2, 0.6, math.atan2(0.75, -0.35)); press("sweater")
for _ in range(900):
    tick()
    if node.scene.reach is None: break
print("9 second run, sweater:", node.assist.state)
node.close()
