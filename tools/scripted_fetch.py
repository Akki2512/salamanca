"""A scripted remote worker, for checking a whole fetch without the suit: sends a caller request, accepts it,
then moves a hand roughly (a few cm off, like the suit) to the item, up, and towards the person, and lets grasp
assist do the rest. Works against any robot node (mock or the pod simulator through the tunnel).

    .venv/bin/python tools/scripted_fetch.py "could I have some water" [--node ws://127.0.0.1:8765] [--server ws://127.0.0.1:8000]
"""
import argparse, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from websockets.sync.client import connect
from remote_hands.messages import WristPose
from remote_hands.sources import REST

ap = argparse.ArgumentParser()
ap.add_argument("text")
ap.add_argument("--node", default="ws://127.0.0.1:8765")
ap.add_argument("--server", default="ws://127.0.0.1:8000")
ap.add_argument("--session", default="default")
ap.add_argument("--direct", default=None, help='skip the caller and Nemotron: "water jug"')
ap.add_argument("--miss", type=float, default=0.03, help="how far off the worker aims, metres")
ap.add_argument("--timeout", type=float, default=600.0)
a = ap.parse_args()
t0 = time.time()
say = lambda *x: print(f"[{time.time() - t0:6.1f} s]", *x, flush=True)

tel = {}
node = connect(a.node, max_size=2 ** 24)
def listen():
    for raw in node:
        m = json.loads(raw)
        if m.get("type") == "telemetry":
            tel.update(m)
threading.Thread(target=listen, daemon=True).start()

done = {"state": None}
if a.direct:
    rid = "direct-%d" % int(time.time() * 1000)
    node.send(json.dumps({"type": "target", "request_id": rid, "name": a.direct}))
    caller = helper = None
    while (tel.get("task_target") or {}).get("request_id") != rid or (tel.get("assist") or {}).get("state") != "free":
        time.sleep(0.05)                    # wait for the node to take THIS request (not the previous one's leftovers)
else:
    caller = connect(f"{a.server}/ws/caller?session={a.session}")
    helper = connect(f"{a.server}/ws/helper?session={a.session}", max_size=2 ** 24)
    caller.send(json.dumps({"type": "request", "text": a.text}))
def caller_listen():
    for raw in caller:
        m = json.loads(raw)
        if m["type"] == "status":
            say("caller hears:", m["state"], "-", m["say"])
            if m["state"] == "offered":
                helper.send(json.dumps({"type": "status", "state": "accepted"}))
            if m["state"] in ("done", "failed"):
                done["state"] = m["state"]
if caller:
    threading.Thread(target=caller_listen, daemon=True).start()

while not (tel.get("assist") and tel["assist"].get("object")) and done["state"] is None and time.time() - t0 < 200:
    time.sleep(0.1)
if not tel.get("assist") or not tel["assist"].get("object"):
    say("no target reached the robot:", done["state"]); sys.exit(1)
obj, drop = np.array(tel["assist"]["object"]), np.array(tel["assist"]["drop"])
side = "left" if obj[1] > 0 else "right"
if abs(obj[1]) < 0.05:
    side = "right"
say(f"target at {obj.round(3).tolist()}, hand-over point {drop.round(3).tolist()}, using the {side} hand")
node.send(json.dumps({"type": "clutch", "on": True}))
r0 = np.array(tel["hands"][side]); scale = 0.75
miss = np.array([a.miss, -a.miss, 0.0])
goal, seq, last, t_sim0, start_sim = r0.copy(), 0, None, tel["t"], tel["t"]
# Engage with the hand at rest first: the robot ties its hand to wherever the worker's hand is at that moment.
while tel.get("state") != "driving" and time.time() - t0 < 320:
    node.send(WristPose(seq=seq, t=seq / 30, left=tuple(REST["left"]), right=tuple(REST["right"]),
                        grip=(0.0, 0.0), clutch=True, source="scripted").to_json())
    seq += 1
    time.sleep(1 / 30)
r0 = np.array(tel["hands"][side])
say("engaged; robot hand at", r0.round(3).tolist())
while done["state"] is None and time.time() - t0 < a.timeout:
    st = tel["assist"]["state"] if tel.get("assist") else "done"
    if a.direct and st == "done":
        done["state"] = "done"
    if seq % 300 == 0:
        say(f"   ... robot time {tel['t'] - start_sim:5.1f} s, hand at {tel['hands'][side]}, lag {tel.get('lag_cm')} cm, ik {tel['ik_ok'][side]}, guard {tel['table_guard']}")
    if st != last:
        say(f"assist: {st}   (robot time {tel['t'] - start_sim:5.1f} s, hand at {tel['hands'][side]}, clipped {tel['clipped_steps']}, safety {tel['safety_mode']})")
        last = st
    if st in ("free", "approach", "descend", "closing"):
        goal = np.r_[obj[:2], 0.07] + miss
    elif st == "holding":
        h = np.array(tel["hands"][side])
        goal = np.r_[h[:2], 0.20] if h[2] < 0.15 and np.linalg.norm(h[:2] - drop[:2]) > 0.12 else np.r_[drop[:2], 0.18] + [-0.03, 0.03, 0]
    human = {s: np.array(REST[s], float) for s in ("left", "right")}
    human[side] = REST[side] + (goal - r0) / scale
    node.send(WristPose(seq=seq, t=seq / 30, left=tuple(human["left"]), right=tuple(human["right"]),
                        grip=(0.0, 0.0), clutch=True, source="scripted").to_json())
    seq += 1
    time.sleep(1 / 30)
time.sleep(1.0)
objs = [o for o in tel.get("objects", [])]
say("result:", done["state"], "| robot time", round(tel["t"] - start_sim, 1), "s | wall", round(time.time() - t0, 1), "s | clipped", tel["clipped_steps"], "| safety", tel["safety_mode"])
for o in objs:
    say("  object", o["color"], o["shape"], o["pos"])
node.send(json.dumps({"type": "clutch", "on": False}))
