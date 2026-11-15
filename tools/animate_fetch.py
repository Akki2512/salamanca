"""An authored, smooth pick-up for the camera: the robot reaches for an item, takes it and holds it out
towards the bed, without letting go. No suit and no person driving: this script is the "worker".

    .venv/bin/python tools/animate_fetch.py ["water jug"] [--node ws://127.0.0.1:8765] [--present x,y,z]

It talks to a running robot node exactly as the suit bridge does (wrist positions, clutch), so joint limits,
reach limits and the grasp are the node's. Record with tools/record_packets.py, render with
unreal/RemoteHandsBedroom/Scripts/render_clip.py.
"""
import argparse, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from websockets.sync.client import connect
from remote_hands.messages import WristPose
from remote_hands.sources import REST

ap = argparse.ArgumentParser()
ap.add_argument("item", nargs="?", default="water jug")
ap.add_argument("--node", default="ws://127.0.0.1:8765")
ap.add_argument("--present", default="0.34,-0.33,0.27", help="where the item is held out at the end (robot frame, m)")
a = ap.parse_args()
present = np.array([float(v) for v in a.present.split(",")])
tel = {}
node = connect(a.node, max_size=2 ** 24)


def listen():
    for raw in node:
        m = json.loads(raw)
        if m.get("type") == "telemetry":
            tel.update(m)


threading.Thread(target=listen, daemon=True).start()
while not tel:
    time.sleep(0.05)
rid = "anim-%d" % int(time.time() * 1000)
node.send(json.dumps({"type": "target", "request_id": rid, "name": a.item, "destination": "hold"}))
while (tel.get("task_target") or {}).get("request_id") != rid or not (tel.get("assist") or {}).get("object"):
    time.sleep(0.05)
obj = np.array(tel["assist"]["object"])
node.send(json.dumps({"type": "clutch", "on": True}))
seq, scale, side = 0, 0.75, "right"


def send(goal, r0):
    """One wrist message that asks the robot's hand to be at `goal` (robot frame)."""
    global seq
    human = {s: np.array(REST[s], float) for s in ("left", "right")}
    human[side] = REST[side] + (goal - r0) / scale
    node.send(WristPose(seq=seq, t=seq / 30, left=tuple(human["left"]), right=tuple(human["right"]), grip=(0.0, 0.0),
                        clutch=True, source="animation").to_json())
    seq += 1
    time.sleep(max(0.0, 1 / 30 - (time.time() - send.last)))    # 30 messages a second by the clock
    send.last = time.time()


send.last = 0.0
while tel.get("state") != "driving":            # engage with the hand at rest: nothing moves
    send(np.zeros(3), np.zeros(3))
r0 = np.array(tel["hands"][side])
ease = lambda u: u * u * (3 - 2 * u)            # start and stop gently


def glide(p0, p1, seconds, until=None):
    """Move the asked-for hand position from p0 to p1 in `seconds` of real time."""
    t0 = time.time()
    while True:
        u = min(1.0, (time.time() - t0) / seconds)
        if until and until():
            return
        send(p0 + (p1 - p0) * ease(u), r0)
        if u >= 1.0:
            return


state = lambda: (tel.get("assist") or {}).get("state")
hover = np.array([obj[0] - 0.02, obj[1], obj[2] + 0.10])
print("item at", obj.round(3).tolist(), "| presenting at", present.round(3).tolist(), flush=True)
glide(r0, r0, 1.0)                                              # a beat at rest
glide(r0, hover, 1.8)                                           # reach over the handle
glide(hover, obj + [0.0, 0.0, 0.03], 1.2, until=lambda: state() in ("approach", "descend", "closing", "holding"))
t0 = time.time()
while state() != "holding" and time.time() - t0 < 8:            # the node lowers the hand and closes it
    send(obj + [0.0, 0.0, 0.03], r0)
print("grip:", state(), "hand at", tel["hands"][side], flush=True)
glide(obj, obj, 0.4)
lift = obj + [0.0, 0.0, 0.07]
glide(obj, lift, 1.1)                                           # lift it clear
glide(lift, present, 2.6)                                       # carry it round to the bed
glide(present, present, 3.0)                                    # and hold it there
print("end:", state(), "hand at", tel["hands"][side], "clipped", tel["clipped_steps"], "safety", tel["safety_mode"], flush=True)
node.send(json.dumps({"type": "clutch", "on": False}))
time.sleep(0.3)
