"""End-to-end check of a running session server (local or Modal), acting as wearer, helper and robot.

    .venv/bin/python tools/check_deployment.py                       # the Modal deployment, session from .env
    .venv/bin/python tools/check_deployment.py ws://127.0.0.1:8000 default
"""
import os
import base64, io, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import remote_hands  # noqa: F401  loads .env
from PIL import Image
from websockets.sync.client import connect

H = sys.argv[1] if len(sys.argv) > 1 else os.environ["REMOTE_HANDS_DEPLOY_URL"]     # the wss:// URL `modal deploy` printed
S = sys.argv[2] if len(sys.argv) > 2 else os.environ["REMOTE_HANDS_DEPLOY_SESSION"]
IMG = Path(__file__).resolve().parents[2] / "sim-snap" / "head_camera.jpg"


def until(ws, pred, t=60):
    end = time.time() + t
    while time.time() < end:
        m = json.loads(ws.recv(timeout=t))
        if pred(m):
            return m
    raise TimeoutError

if H.startswith("wss://"):
    try:
        with connect(f"{H}/ws/wearer?session=not-the-session", open_timeout=30) as w:
            w.recv(timeout=10)
        print("a wrong session name was ACCEPTED: the deployment is open to anyone")
    except Exception as e:
        print("wrong session name rejected:", type(e).__name__)
buf = io.BytesIO(); Image.open(IMG).convert("RGB").save(buf, "JPEG", quality=85)
with connect(f"{H}/ws/helper?session={S}", max_size=2 ** 24, open_timeout=30) as helper, \
     connect(f"{H}/ws/robot?session={S}", open_timeout=30) as robot, \
     connect(f"{H}/ws/wearer?session={S}", max_size=2 ** 24, open_timeout=30) as wearer:
    t0 = time.time()
    wearer.send(json.dumps({"type": "request", "text": "can you pass me the yellow cube",
                            "frame_jpeg_b64": base64.b64encode(buf.getvalue()).decode()}))
    task = until(helper, lambda m: m["type"] == "task" or m.get("state") == "failed")
    if task["type"] != "task":
        sys.exit("request FAILED after %.1f s: %s" % (time.time() - t0, task.get("say")))
    print("helper got the task after %.1f s:" % (time.time() - t0), task["status"], task["target"], task["destination"],
          "| extractor", task.get("latency_s"), "s,", task.get("attempts"), "attempt(s)", task.get("attempt_errors"), "| model", task["model"], "| visible", [(v["color"], v["shape"]) for v in task["visible"]])
    print("wearer hears:", until(wearer, lambda m: m.get("state") in ("offered", "failed"))["say"])
    helper.send(json.dumps({"type": "status", "state": "accepted"}))
    print("robot got:", until(robot, lambda m: m["type"] == "target"))
    helper.send(json.dumps({"type": "status", "state": "done"}))
    print("wearer hears:", until(wearer, lambda m: m.get("state") == "done")["say"])
