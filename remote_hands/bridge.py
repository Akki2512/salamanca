"""Mocap bridge: runs on the helper's laptop, reads a mocap source, sends wrist poses to the robot node.

    python -m remote_hands.bridge --source synthetic --url ws://127.0.0.1:8765

Sends at the source's rate in wall time. If the connection drops it reconnects; the robot node holds meanwhile.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

from .sources import SOURCES

log = logging.getLogger("remote_hands.bridge")


async def run(source, url, seconds=None, print_every=1.0, start_open=False):
    import websockets

    t0 = time.monotonic()
    while seconds is None or time.monotonic() - t0 < seconds:
        try:
            async with websockets.connect(url) as ws:
                log.info("connected to %s", url)

                async def show():
                    last = 0.0
                    async for raw in ws:
                        tel = json.loads(raw)
                        if tel.get("type") == "telemetry" and tel["t"] - last >= print_every:
                            last = tel["t"]
                            log.info("robot t=%.1f %s safety=%s clipped=%d", tel["t"], tel["state"],
                                     tel["safety_mode"], tel["clipped_steps"])
                reader = asyncio.ensure_future(show())
                if start_open:      # a real suit: the robot follows only after the helper switches the clutch on
                    await ws.send(json.dumps({"type": "clutch", "on": False}))
                seq = 0
                try:
                    while seconds is None or time.monotonic() - t0 < seconds:
                        t = time.monotonic() - t0
                        await ws.send(source.at(t, seq).to_json())
                        seq += 1
                        await asyncio.sleep(max(0.0, (seq / source.hz) - (time.monotonic() - t0)))
                finally:
                    reader.cancel()
        except (OSError, websockets.ConnectionClosed) as e:
            log.warning("connection lost (%s); retrying in 1 s", e)
            await asyncio.sleep(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="synthetic", choices=sorted(SOURCES))
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--file", default="recordings/suit_104834_wrists.npz", help="replay: the recording")
    ap.add_argument("--start", type=float, default=0.0, help="replay: seconds into the recording")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=None, help="rokoko: the helper's height in metres (arm lengths)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        real = args.source in ("rokoko", "studio")
        if args.source == "replay":
            source = SOURCES["replay"](args.file, start=args.start, end=args.end, speed=args.speed)
            log.info("replaying %.0f s of %s", source.period, args.file)
        else:
            source = SOURCES[args.source](height=args.height) if real else SOURCES[args.source]()
        if hasattr(source, "calibrate"):
            import threading

            def keys():             # Enter in this terminal = calibrate again
                while True:
                    try:
                        input()
                    except EOFError:
                        return
                    source.calibrate()
            threading.Thread(target=keys, daemon=True).start()
            log.info("Enter here = calibrate; then switch the clutch on the helper page (space bar)")
        asyncio.run(run(source, args.url, args.seconds, start_open=real))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
