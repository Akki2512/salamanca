"""The title in Departure Mono, the pixel face of the Junction 2026 poster, to match the question card.

    python tools/cut_title_pixel.py <output .mp4> ["by {}"] [--still seconds]
White capitals typed onto black, as on the question card; LA (lanthanum, 57) and CA (calcium, 20) arrive as flat
element tiles; the by-line is typed underneath. 1920x1080, 24 fps, with a tick per key and a thud per tile.
"""
import subprocess, sys, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1920, 1080, 24, 48000
LENGTH = 6.5
HERE = Path(__file__).resolve().parent.parent
DM = str(HERE / "unreal/renders/opening/fonts/departure_mono/DepartureMono-1.500/DepartureMono-Regular.otf")
big, small, line = ImageFont.truetype(DM, 187), ImageFont.truetype(DM, 22), ImageFont.truetype(DM, 55)     # 17x, 2x and 5x the 11-pixel grid
WHITE, TEAL, AMBER = (255, 255, 255), (16, 128, 116), (204, 122, 38)
out = Path(sys.argv[1])
BY = next((a for a in sys.argv[2:] if not a.startswith("--") and not a.replace(".", "").isdigit()), "")
# the name as it is typed: plain letters, or a tile (symbol, number, mass, name, colour)
KEYS = ["S", "A", ("LA", 57, "138.91", "LANTHANUM", TEAL), "M", "A", "N", ("CA", 20, "40.078", "CALCIUM", AMBER)]
CELL = big.getlength("M")
PAD, TOP = 22, 372                                          # room round a tile's letters; top of the title line
widths = [CELL if isinstance(k, str) else 2 * CELL + 2 * PAD for k in KEYS]
X0 = W / 2 - sum(widths) / 2
START, STEP, BY_AT, BY_STEP = 0.45, 0.17, 2.5, 0.09
key_time = lambda j: START + STEP * j


def frame(i):
    t = i / FPS
    im = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(im)
    x, typed = X0, 0
    for j, (k, w) in enumerate(zip(KEYS, widths)):
        if t >= key_time(j):
            typed = j + 1
            if isinstance(k, str):
                d.text((round(x), TOP), k, font=big, fill=WHITE)
            else:
                s, num, mass, name, col = k
                box = [round(x) + 6, TOP - 34, round(x + w) - 6, TOP + 246]
                d.rectangle(box, fill=col, outline=WHITE, width=6)
                d.text((round(x + PAD), TOP), s, font=big, fill=WHITE)
                d.text((box[0] + 18, box[1] + 14), mass, font=small, fill=WHITE)
                d.text((box[2] - 18, box[1] + 14), str(num), font=small, fill=WHITE, anchor="ra")
                d.text(((box[0] + box[2]) / 2, box[3] - 16), name, font=small, fill=WHITE, anchor="ms")
        x += w
    cx, cy, ch, cw = X0 + sum(widths[:typed]) + 8, TOP + 22, 150, CELL * 0.7           # the cursor: after the last key typed
    shown = 0
    if BY:
        shown = int(max(0, min(len(BY), (t - BY_AT) / BY_STEP)))
        bx = W / 2 - line.getlength("M") * len(BY) / 2
        d.text((round(bx), 700), BY[:shown], font=line, fill=WHITE)
        if t >= BY_AT - 0.3:
            cx, cy, ch, cw = bx + line.getlength("M") * shown + 4, 706, 46, line.getlength("M") * 0.75
    busy = t < key_time(len(KEYS) - 1) + 0.3 or (BY and BY_AT - 0.3 <= t < BY_AT + BY_STEP * len(BY) + 0.2)
    if busy or int(t * 2.5) % 2 == 0:
        d.rectangle([round(cx), round(cy), round(cx + cw), round(cy + ch)], fill=WHITE)
    fade = min(1.0, max(0.0, (LENGTH - t) / 0.5))
    return (np.asarray(im, dtype=np.float32) * fade).astype(np.uint8)


def sound():
    t = np.arange(int(SR * LENGTH)) / SR
    x = np.zeros_like(t); rng = np.random.default_rng(2)
    ticks = [key_time(j) for j in range(len(KEYS))] + ([BY_AT + BY_STEP * (j + 1) for j in range(len(BY)) if BY[j] != " "] if BY else [])
    for at in ticks:                                         # a small dry click per key
        a = int(SR * at); n = int(SR * 0.03); u = np.arange(n) / SR
        x[a:a + n] += (rng.uniform(-1, 1, n) * 0.5 + np.sin(2 * np.pi * 1900 * u)) * np.exp(-u / 0.004) * 0.22
    for j, k in enumerate(KEYS):                             # a low thud under each tile
        if not isinstance(k, str):
            u = np.clip(t - key_time(j), 0, None)
            x += (t >= key_time(j)) * np.sin(2 * np.pi * (44 * u + 34 * (1 - np.exp(-u * 12)) / 12)) * np.exp(-u / 0.42)
    x[-int(SR * 0.5):] *= np.linspace(1, 0, int(SR * 0.5))
    return 0.85 * x / np.abs(x).max()


if "--still" in sys.argv:
    Image.fromarray(frame(int(float(sys.argv[sys.argv.index("--still") + 1]) * FPS))).save(out); sys.exit()
wav = out.with_suffix(".wav")
with wave.open(str(wav), "wb") as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes((sound() * 32767).astype(np.int16).tobytes())
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav),
                      "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                      "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
for i in range(int(LENGTH * FPS)):
    p.stdin.write(frame(i).tobytes())
p.stdin.close(); p.wait(); wav.unlink()
print("wrote", out, LENGTH, "s")
