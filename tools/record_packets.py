"""Record the robot node's Unreal packets to a file, for rendering a clip afterwards.

    .venv/bin/python tools/record_packets.py OUT.jsonl [--port 7795] [--seconds 120]
Start the node with a second endpoint: --unreal-udp 127.0.0.1:7777,127.0.0.1:7795
Each line: {"at": seconds since the first packet, "packet": {...}}. Stops 3 s after a fetch reaches "done".
"""
import json, socket, sys, time
out = sys.argv[1]
port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 7795
limit = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 120.0
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", port)); s.settimeout(1.0)
t0 = done_at = None; n = 0
with open(out, "w") as f:
    while True:
        try:
            d, _ = s.recvfrom(65535)
        except socket.timeout:
            if t0 and time.time() - t0 > limit: break
            continue
        now = time.time(); t0 = t0 or now
        m = json.loads(d)
        f.write(json.dumps({"at": round(now - t0, 4), "packet": m}) + "\n"); n += 1
        if (m.get("task") or {}).get("state") == "done" and done_at is None: done_at = now
        if (done_at and now - done_at > 3.0) or now - t0 > limit: break
print("recorded", n, "packets,", round(time.time() - t0, 1), "s ->", out)
