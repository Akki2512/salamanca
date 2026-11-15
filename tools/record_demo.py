"""Records one short episode on the LIVE robot node without a suit: reset, walk to the sweater, pick it up off the
floor, carry it to the bed, put it down. For checking the recorder, the labels and the exporter end to end.

    .venv/bin/python tools/record_demo.py [ws://127.0.0.1:8765]      (the node must run with --record DIR)
"""
import asyncio, json, math, sys

import websockets

EYE = (0.0537, 0.0175, 0.4739)      # robot_node.EYE: the head camera in the robot's own frame
URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"


async def main():
    async with websockets.connect(URL, max_size=None) as ws:
        tel = {}

        async def pump(seconds=0.0, until=None, drive=None, aim=None, limit=40.0):
            """Read telemetry (keeping `tel` fresh) for `seconds`, or until `until(tel)` is true; repeat drive / aim."""
            nonlocal tel
            t0 = asyncio.get_event_loop().time()
            while True:
                raw = await ws.recv()
                if len(raw) > 20000:
                    await ws.send('{"type":"pov_ack"}')
                    continue
                m = json.loads(raw)
                if m.get("type") != "telemetry":
                    continue
                tel = m
                if drive:
                    await ws.send(json.dumps({"type": "drive", **drive(tel)}))
                if aim:
                    await ws.send(json.dumps({"type": "point", "yaw": aim[0], "pitch": aim[1]}))
                now = asyncio.get_event_loop().time() - t0
                if (until and until(tel)) or (not until and now >= seconds) or now > limit:
                    return

        def go(x, y):
            def f(t):
                bx, by, yaw = t["base"][:3]
                dx, dy = x - bx, y - by
                e = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
                return {"vx": 0.6 * max(0.0, math.cos(e)) * min(1.0, math.hypot(dx, dy) / 0.3 + 0.2), "vy": 0.0, "wz": max(-1.4, min(1.4, 2.5 * e))}
            return f, (lambda t: math.hypot(x - t["base"][0], y - t["base"][1]) < 0.10)

        await ws.send(json.dumps({"type": "clutch", "on": False}))
        await ws.send(json.dumps({"type": "scene", "cue": "reset"}))
        await pump(until=lambda t: t["state"] == "idle" and abs(t["base"][0]) < 0.01, limit=15)
        await ws.send(json.dumps({"type": "record", "on": True, "task": "pick the sweater up off the floor and put it on the bed"}))
        await pump(1.0)
        print("recording:", tel.get("recording"))
        d, done = go(-0.2, 0.6)
        await pump(until=done, drive=d)
        await pump(0.8)
        sw = next(o for o in tel["objects"] if o["item"] == "sweater")       # in the robot's own frame
        v = [sw["pos"][i] - EYE[i] for i in range(3)]
        aim = (math.degrees(math.atan2(v[1], v[0])), math.degrees(math.asin(v[2] / math.sqrt(sum(c * c for c in v)))))
        await pump(0.4, aim=aim)
        await ws.send(json.dumps({"type": "point", "yaw": aim[0], "pitch": aim[1], "press": True}))
        await pump(1.0)
        await pump(until=lambda t: not (t["scene"] or {}).get("reach"), limit=40)
        print("after the pick:", (tel.get("assist") or {}).get("state"), "| scene", {k: tel["scene"][k] for k in ("phase", "reach")})
        d, done = go(-0.5, -0.05)
        await pump(until=done, drive=d)
        await pump(until=lambda t: abs((t["base"][2] + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi) < 0.08,
                   drive=lambda t: {"vx": 0.0, "vy": 0.0, "wz": max(-1.4, min(1.4, 2.5 * ((-math.pi / 2 - t["base"][2] + math.pi) % (2 * math.pi) - math.pi)))}, limit=10)
        await pump(0.8)
        await ws.send(json.dumps({"type": "point", "yaw": 0.0, "pitch": 60.0, "press": True}))     # let go
        await pump(2.5)
        print("at the end:", tel.get("recording"), "| sweater", [o["pos"] for o in tel["objects"] if o["item"] == "sweater"])
        await ws.send(json.dumps({"type": "record", "on": False, "success": True}))
        await pump(1.0)
        print("stopped:", tel.get("recording"))


asyncio.run(main())
