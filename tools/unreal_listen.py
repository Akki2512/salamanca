"""Stand-in for Unreal: prints what the robot node sends on --unreal-udp, twice a second.

    .venv/bin/python tools/unreal_listen.py [--port 7777]
"""
import argparse, json, socket, time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=7777)
ap.add_argument("--seconds", type=float, default=None)
a = ap.parse_args()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", a.port))
s.settimeout(1.0)
t0, last, n, size = time.time(), 0.0, 0, 0
while a.seconds is None or time.time() - t0 < a.seconds:
    try:
        d, _ = s.recvfrom(65535)
    except socket.timeout:
        continue
    n += 1; size = len(d)
    if time.time() - last >= 0.5:
        last = time.time()
        m = json.loads(d)
        held = [f'{i["name"]} in {i["held_by"]} hand' for i in m["items"] if i["held_by"]]
        print(f'seq {m["seq"]} t {m["t"]:.1f} | {n / (time.time() - t0):.0f} packets/s, {size} bytes | right hand {m["hands"]["right"]} '
              f'| task {m["task"]} | {held or "nothing held"}', flush=True)
