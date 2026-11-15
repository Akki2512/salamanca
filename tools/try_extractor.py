"""Call the extractor on an image file:  python tools/try_extractor.py IMAGE "sentence" ["sentence" ...]"""
import asyncio, io, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
import remote_hands  # loads .env
from remote_hands.extractor import extract

for line in (Path(__file__).resolve().parents[1] / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())


async def main():
    img = Image.open(sys.argv[1]).convert("RGB")
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=85)
    for text in sys.argv[2:]:
        try:
            t = await extract(buf.getvalue(), text, img.width, img.height)
            tg = t["target"]
            print(f'{text!r}\n   -> {t["status"]} {tg and (tg["color"], tg["shape"], tg["box_xyxy"])} dest={t["destination"]} '
                  f'conf={t["confidence"]} {t["latency_s"]}s x{t["attempts"]}\n      say: {t["say"]}\n      visible: {[(v["color"], v["shape"]) for v in t["visible"]]}')
        except Exception as e:
            print(f"{text!r}\n   -> ERROR {e}")

asyncio.run(main())
