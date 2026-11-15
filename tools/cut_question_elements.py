"""The question card in the element-tile style of the title (tools/cut_title_elements.py).

    python tools/cut_question_elements.py <output .mp4> [--still seconds]
"Can robots help our loved ones / when we can't?" with the Ca of "Can" and of "can't" as calcium tiles: the same
two letters carry the question and its answer. Same ground, haze, face and colours as the title. 1920x1080, 24 fps.
"""
import math, subprocess, sys, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SR = 1920, 1080, 24, 48000
LENGTH = 10.0
BG, WHITE = (6, 9, 11), (238, 242, 244)
AV = "/System/Library/Fonts/Avenir Next.ttc"
SIZE, TILE, GAP = 104, 132, 10
big, sym, small = ImageFont.truetype(AV, SIZE, index=2), ImageFont.truetype(AV, 74, index=0), ImageFont.truetype(AV, 17, index=5)
out = Path(sys.argv[1])
smooth = lambda k: (lambda k: k * k * (3 - 2 * k))(min(1.0, max(0.0, k)))
rng = np.random.default_rng(3)
TEAL, AMBER = ((22, 150, 136), (10, 84, 80)), ((226, 148, 52), (150, 78, 30))
# each line: its baseline, when it starts, and its pieces: text, or a tile (None, colours)
LINES = [(480, 0.6, [(None, TEAL), "n robots help our loved ones"]),
         (680, 3.2, ["when we ", (None, AMBER), "n’t?"])]
haze = [Image.fromarray((rng.random((27, 48)) * 255).astype(np.uint8)).resize((2600, 1460), Image.BICUBIC).filter(ImageFilter.GaussianBlur(95)) for _ in range(2)]


def tile(colours, k):
    top, bottom = colours
    t_ = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0)); d = ImageDraw.Draw(t_)
    for y in range(TILE):
        d.line([0, y, TILE, y], fill=tuple(int(a + (b - a) * y / TILE) for a, b in zip(top, bottom)) + (255,))
    d.rectangle([0, 0, TILE - 1, TILE - 1], outline=WHITE + (255,), width=3)
    d.text((TILE / 2, TILE * 0.74), "Ca", font=sym, fill=WHITE + (255,), anchor="ms")
    d.text((TILE - 10, 8), "20", font=small, fill=WHITE + (255,), anchor="ra")
    d.text((10, 8), "40.078", font=small, fill=WHITE + (230,), anchor="la")
    z = 1 + 0.5 * (1 - smooth(k)) ** 2
    t_ = t_.resize((int(TILE * z), int(TILE * z)), Image.LANCZOS)
    t_.putalpha(Image.eval(t_.getchannel("A"), lambda v: int(v * smooth(k * 2))))
    return t_


def frame(i):
    t = i / FPS
    base = Image.new("RGB", (W, H), BG)
    for j, hz in enumerate(haze):
        ox, oy = int(340 + 300 * math.sin(0.11 * t + 2 * j)), int(190 + 160 * math.cos(0.09 * t + j))
        m = Image.eval(hz.crop((ox, oy, ox + W, oy + H)), lambda v: int(max(0, v - 105) * 1.6))
        base.paste(Image.new("RGB", (W, H), ((18, 120, 110), (150, 90, 30))[j]), (0, 0), m)
    art = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(art)
    flash = 0.0
    for baseline, start, pieces in LINES:
        widths = [big.getlength(p) if isinstance(p, str) else TILE + 2 * GAP for p in pieces]
        x = W / 2 - sum(widths) / 2
        for p, w in zip(pieces, widths):
            if isinstance(p, str):
                d.text((x, baseline), p, font=big, fill=WHITE + (int(255 * smooth((t - start - 0.25) / 1.0)),), anchor="ls")
            else:
                k = (t - start) / 0.28
                if k > 0:
                    tl = tile(p[1], min(1.0, k))
                    art.alpha_composite(tl, (int(x + GAP + TILE / 2 - tl.width / 2), int(baseline - TILE * 0.74 + TILE / 2 - tl.height / 2)))
                    d = ImageDraw.Draw(art)
                    flash = max(flash, 1 - abs(k - 1) / 0.9 if 0.4 < k < 1.9 else 0)
            x += w
    glow = Image.eval(art.filter(ImageFilter.GaussianBlur(22)), lambda v: int(v * 0.35))
    pic = Image.alpha_composite(Image.alpha_composite(base.convert("RGBA"), glow), art).convert("RGB")
    a = np.asarray(pic, dtype=np.float32) * (1 + 0.18 * flash)
    return np.clip(a * smooth(t / 0.4) * (1 - smooth((t - 8.3) / 1.4)), 0, 255).astype(np.uint8)


def pluck(freq, length, bright):
    n = int(SR / freq); buf = rng.uniform(-1, 1, n)
    buf = np.convolve(buf, np.ones(5) / 5, "same") * (1 - bright) + buf * bright
    y = np.empty(int(SR * length))
    for i in range(len(y)):
        y[i] = buf[i % n]
        buf[i % n] = 0.998 * 0.5 * (buf[i % n] + buf[(i + 1) % n])
    return y


def sound():
    """Bare and dry: a low drone, a thud under each tile, and a few slow notes on a low string with a slide of pitch."""
    t = np.arange(int(SR * LENGTH)) / SR
    x = 0.16 * np.sin(2 * np.pi * 55 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * t)) * np.clip(t / 1.5, 0, 1)
    for at in (0.6 + 0.28, 3.2 + 0.28):
        u = np.clip(t - at, 0, None)
        x += (t >= at) * np.sin(2 * np.pi * (42 * u + 34 * (1 - np.exp(-u * 12)) / 12)) * np.exp(-u / 0.42)
    for at, f in ((1.6, 73.42), (2.5, 87.31), (4.3, 73.42), (5.2, 65.41), (6.4, 55.0)):          # D2 F2 D2 C2 A1
        y = pluck(f, 3.5, 0.7); a = int(SR * at); n = min(len(y), len(x) - a)
        x[a:a + n] += 0.6 * y[:n]
    for dly, g in ((0.11, 0.3), (0.23, 0.2), (0.37, 0.12)):
        s = int(SR * dly); x[s:] += g * x[:-s]
    x[-int(SR * 1.6):] *= np.linspace(1, 0, int(SR * 1.6))
    return 0.85 * x / np.abs(x).max()


if "--still" in sys.argv:
    Image.fromarray(frame(int(float(sys.argv[sys.argv.index("--still") + 1]) * FPS))).save(out); sys.exit()
wav = out.with_suffix(".wav")
with wave.open(str(wav), "wb") as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes((sound() * 32767).astype(np.int16).tobytes())
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav),
                      "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                      "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
for i in range(int(LENGTH * FPS)):
    p.stdin.write(frame(i).tobytes())
p.stdin.close(); p.wait(); wav.unlink()
print("wrote", out, LENGTH, "s")
