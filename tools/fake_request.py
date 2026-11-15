"""Stand-in for the phone app: send one photo + sentence as the wearer and print what would be spoken.

    .venv/bin/python tools/fake_request.py ../dbg/s1_rgb.png "pass me the blue one" [--url ws://127.0.0.1:8000] [--session default]
"""
import argparse, asyncio, base64, io, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import websockets
from PIL import Image


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image"); ap.add_argument("text")
    ap.add_argument("--url", default="ws://127.0.0.1:8000"); ap.add_argument("--session", default="default")
    ap.add_argument("--log-frame", action="store_true", help="opt in to saving the frame in the session log")
    a = ap.parse_args()
    buf = io.BytesIO(); Image.open(a.image).convert("RGB").save(buf, "JPEG", quality=85)
    async with websockets.connect(f"{a.url}/ws/wearer?session={a.session}", max_size=2 ** 24) as ws:
        await ws.send(json.dumps({"type": "request", "text": a.text, "log_frame": a.log_frame,
                                  "frame_jpeg_b64": base64.b64encode(buf.getvalue()).decode()}))
        async for raw in ws:
            m = json.loads(raw)
            if m["type"] == "status":
                print(f'[{m["state"]}] {m["say"] or ""}')
                if m["state"] in ("done", "failed"):
                    return

asyncio.run(main())
