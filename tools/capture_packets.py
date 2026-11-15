"""Save the robot_state packets of one fetch at three moments (carrying, handing over, done) for rendering.

    .venv/bin/python tools/capture_packets.py OUT_DIR [--port 7777]     (run it, then start a fetch)
"""
import json, socket, sys, time
out, port = sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 else 7777
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", port)); s.settimeout(1.0)
got, t0 = {}, time.time()
while time.time() - t0 < 60 and len(got) < 3:
    try:
        d, _ = s.recvfrom(65535)
    except socket.timeout:
        continue
    m = json.loads(d); t = m.get("task") or {}
    side = t.get("side")
    if t.get("state") == "holding" and side and m["hands"][side][2] > 0.14 and "carrying" not in got:
        got["carrying"] = m
    if t.get("state") == "releasing" and "handing_over" not in got:
        got["handing_over"] = m
    if t.get("state") == "done" and "done" not in got:
        got["done"] = m
for k, m in got.items():
    json.dump(m, open(f"{out}/packet_{k}.json", "w"))
print("captured", sorted(got))
