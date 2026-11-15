"""Draw the boxes of remote_hands/labels.py on an episode's robot_eyes frames, to check them by eye.

    .venv/bin/python tools/draw_labels.py <episode_dir> [--every N] [--out DIR]

Saves the drawn frames as <out>/000000.jpg ... (default <episode_dir>/labels/drawn). Frames without a head
direction are saved with "unlabelled" written on them. The boxes are worked out afresh from the step data.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_hands import labels  # noqa: E402

COLORS = {"water jug": (80, 160, 255), "alarm clock": (80, 220, 110), "glass of water": (240, 240, 240),
          "sweater": (255, 80, 80), "man in wheelchair": (255, 200, 60), "wheelchair": (255, 200, 60)}


def draw(im, bxs, labelled):
    d = ImageDraw.Draw(im)
    if not labelled:
        d.text((10, 10), "unlabelled (no head direction)", fill=(255, 255, 0))
    for b in bxs:
        c = COLORS.get(b["name"], (255, 0, 255))
        x0, y0, x1, y1 = b["box"]
        d.rectangle([x0, y0, x1, y1], outline=c, width=3)
        u, v = b["center"]
        d.line([u - 6, v, u + 6, v], fill=c, width=2)
        d.line([u, v - 6, u, v + 6], fill=c, width=2)
        text = "%s %.2f m%s%s" % (b["name"], b["depth"], " (%s hand)" % b["held_by"] if b["held_by"] else "",
                                  "" if b["visible"] >= 0.995 else " %d%%" % round(100 * b["visible"]))
        ty = y0 - 14 if y0 >= 14 else y1 + 2
        d.rectangle([x0, ty, x0 + 6.5 * len(text) + 4, ty + 12], fill=(0, 0, 0))
        d.text((x0 + 2, ty), text, fill=c)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("episode_dir")
    ap.add_argument("--every", type=int, default=1, help="draw every Nth frame")
    ap.add_argument("--out", default=None, help="where to save (default <episode_dir>/labels/drawn)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.episode_dir) / "labels" / "drawn"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for j, (meta, step, jpg) in enumerate(labels.frames(a.episode_dir)):
        if j % max(a.every, 1) or not jpg.exists():
            continue
        im = Image.open(jpg).convert("RGB")
        labelled = step is not None and labels.camera(step, meta) is not None
        draw(im, labels.boxes(step, meta, *im.size) if labelled else [], labelled).save(out / jpg.name, quality=90)
        n += 1
    print("drew %d frames into %s" % (n, out))


if __name__ == "__main__":
    main()
