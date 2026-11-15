"""Replay a stretch of a Rokoko Studio recording (hips, legs and arms) through the robot node offline and
save the Unreal packets, for rendering with unreal/RemoteHandsBedroom/Scripts/render_clip.py.

    .venv/bin/python tools/replay_walk.py recordings/suit_104834.rkkudp --start 636 --seconds 20 --out recordings/walk_packets.jsonl
"""
import argparse, json, logging, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lz4.frame
import numpy as np
from remote_hands import unreal_stream
from remote_hands.messages import WristPose
from remote_hands.robot_node import HZ, RobotNode
from remote_hands.rokoko import RokokoStudioSource as Studio

ap = argparse.ArgumentParser()
ap.add_argument("recording")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--out", default="recordings/walk_packets.jsonl")
ap.add_argument("--base", default="0,0,0", help="where the robot starts: x,y (m, home frame) and heading (deg)")
a = ap.parse_args()
logging.disable(logging.WARNING)

frames, t_first = [], None
with open(a.recording, "rb") as f:
    while True:
        h = f.read(12)
        if len(h) < 12:
            break
        t, n = struct.unpack("<dI", h)
        data = f.read(n)
        t_first = t if t_first is None else t_first
        if t - t_first < a.start:
            continue
        if t - t_first > a.start + a.seconds:
            break
        if data[:4] == b"\x04\x22\x4d\x18":
            data = lz4.frame.decompress(data)
        try:
            fr = json.loads(data)
            frames.append((t - t_first - a.start, Studio.wrists_from_frame(fr), Studio.body_from_frame(fr)))
        except (ValueError, KeyError, IndexError):
            pass
print(len(frames), "frames")
node = RobotNode(backend="mock", sync_ik=True)
while node.state == "homing":
    node.tick()
bx, by, bdeg = (float(v) for v in a.base.split(","))
node.walker.base[:] = [bx, by, np.radians(bdeg)]
out, k, base_path = open(a.out, "w"), 0, []
for i in range(int(a.seconds * HZ)):
    while k + 1 < len(frames) and frames[k + 1][0] <= i / HZ:
        k += 1
    _, w, body = frames[k]
    node.push(WristPose(seq=i, t=i / HZ, left=w["left"], right=w["right"], clutch=True, source="replay", body=body))
    tel = node.tick()
    out.write(json.dumps({"at": round(i / HZ, 4), "packet": unreal_stream.packet(node, tel)}) + "\n")
    base_path.append(tel["base"])
b = np.array(base_path)
s = node.env.filter.summary()
print("base x %.2f..%.2f  y %.2f..%.2f  heading %.0f..%.0f deg  height change %.3f..%.3f m" % (
    b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max(), np.degrees(b[:, 2].min()), np.degrees(b[:, 2].max()),
    b[:, 3].min(), b[:, 3].max()))
print("path length %.1f m | clipped %d | safety %s | robot leg %.3f m" % (
    np.linalg.norm(np.diff(b[:, :2], axis=0), axis=1).sum(), s["clipped_steps"], s["mode"], node.walker.leg_len))
legs = [json.loads(l)["packet"]["ext"]["joints"] for l in open(a.out)]
for j in ("left_hip_pitch_joint", "left_knee_joint", "right_hip_roll_joint"):
    v = np.array([x[j] for x in legs])
    print("  %-22s %.2f..%.2f rad" % (j, v.min(), v.max()))
node.close()
