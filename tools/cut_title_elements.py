"""The title with two of its syllables as periodic-table tiles: Sa [La] man [Ca].

    python tools/cut_title_elements.py <output .mp4> [--still seconds]
Lanthanum (57) and calcium (20) are real elements whose symbols fall inside the name. The look is this project's
own (its dark ground, the salamander's teal and amber, Avenir Next); only the idea of element tiles in a title is
borrowed. Made here frame by frame, 1920x1080 at 24 fps, with its sound.
"""
import math, os, subprocess, sys, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SR = 1920, 1080, 24, 48000
LENGTH = 6.5
BG, WHITE = (6, 9, 11), (238, 242, 244)
AV = "/System/Library/Fonts/Avenir Next.ttc"
big, sym, small = ImageFont.truetype(AV, 210, index=2), ImageFont.truetype(AV, 138, index=0), ImageFont.truetype(AV, 30, index=5)
out = Path(sys.argv[1])
smooth = lambda k: (lambda k: k * k * (3 - 2 * k))(min(1.0, max(0.0, k)))
rng = np.random.default_rng(3)
TILE, GAP, BASE = 250, 16, 610
# the name in pieces: plain letters, or a tile (symbol, atomic number, atomic mass, name, colours top and bottom, when it lands)
PIECES = [("Sa",), ("La", 57, "138.91", "Lanthanum", (22, 150, 136), (10, 84, 80), 1.25), ("man",),
          ("Ca", 20, "40.078", "Calcium", (226, 148, 52), (150, 78, 30), 1.7)]
widths = [big.getlength(p[0]) if len(p) == 1 else TILE for p in PIECES]
x0 = W / 2 - (sum(widths) + GAP * (len(PIECES) - 1)) / 2
XS = [x0 + sum(widths[:i]) + GAP * i for i in range(len(PIECES))]

# a slow haze behind everything: soft blobs of noise, larger than the picture, sliding past
haze = [Image.fromarray((rng.random((27, 48)) * 255).astype(np.uint8)).resize((2600, 1460), Image.BICUBIC).filter(ImageFilter.GaussianBlur(95)) for _ in range(2)]


def tile(piece, k):
    """One element tile, k from 0 (not yet) to 1 (landed): it arrives a little too large and settles."""
    s_, num, mass, name, top, bottom, _ = piece
    t_ = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0)); d = ImageDraw.Draw(t_)
    for y in range(TILE):
        c = tuple(int(a + (b - a) * y / TILE) for a, b in zip(top, bottom))
        d.line([0, y, TILE, y], fill=c + (255,))
    d.rectangle([0, 0, TILE - 1, TILE - 1], outline=WHITE + (255,), width=4)
    d.text((TILE / 2, TILE * 0.70), s_, font=sym, fill=WHITE + (255,), anchor="ms")
    d.text((TILE - 18, 16), str(num), font=small, fill=WHITE + (255,), anchor="ra")
    d.text((18, 16), mass, font=small, fill=WHITE + (230,), anchor="la")
    d.text((TILE / 2, TILE - 20), name.upper(), font=ImageFont.truetype(AV, 22, index=5), fill=WHITE + (230,), anchor="ms")
    z = 1 + 0.5 * (1 - smooth(k)) ** 2
    t_ = t_.resize((int(TILE * z), int(TILE * z)), Image.LANCZOS)
    t_.putalpha(Image.eval(t_.getchannel("A"), lambda v: int(v * smooth(k * 2))))
    return t_


def frame(i):
    t = i / FPS
    base = Image.new("RGB", (W, H), BG)
    for j, hz in enumerate(haze):                                        # the haze, teal one way and amber the other
        ox, oy = int(340 + 300 * math.sin(0.11 * t + 2 * j)), int(190 + 160 * math.cos(0.09 * t + j))
        m = Image.eval(hz.crop((ox, oy, ox + W, oy + H)), lambda v: int(max(0, v - 105) * 1.6))
        base.paste(Image.new("RGB", (W, H), ((18, 120, 110), (150, 90, 30))[j]), (0, 0), m)
    art = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(art)
    flash = 0.0
    for piece, x, w in zip(PIECES, XS, widths):
        if len(piece) == 1:                                              # plain letters come up one piece after another
            k = smooth((t - (0.45 if piece[0] == "Sa" else 0.8)) / 0.5)
            d.text((x, BASE), piece[0], font=big, fill=WHITE + (int(255 * k),), anchor="ls")
        else:
            k = (t - piece[6]) / 0.28
            if k > 0:
                tl = tile(piece, min(1.0, k))
                art.alpha_composite(tl, (int(x + TILE / 2 - tl.width / 2), int(BASE - TILE * 0.70 + TILE / 2 - tl.height / 2)))
                d = ImageDraw.Draw(art)
                flash = max(flash, 1 - abs(k - 1) / 0.9 if 0.4 < k < 1.9 else 0)
    rule = smooth((t - 2.2) / 0.8)                                       # a hairline and the atomic numbers' sum: nothing, just a line
    if rule:
        d.line([W / 2 - 420 * rule, BASE + 110, W / 2 + 420 * rule, BASE + 110], fill=WHITE + (150,), width=2)
    by = os.environ.get("RH_BYLINE")                                    # e.g. RH_BYLINE="by {}": the team, under the line
    if by:
        d.text((W / 2, BASE + 200), by, font=ImageFont.truetype(AV, 64, index=5), fill=WHITE + (int(235 * smooth((t - 2.7) / 0.8)),), anchor="ms")
    glow = Image.eval(art.filter(ImageFilter.GaussianBlur(26)), lambda v: int(v * 0.4))
    pic = Image.alpha_composite(Image.alpha_composite(base.convert("RGBA"), glow), art).convert("RGB")
    a = np.asarray(pic, dtype=np.float32) * (1 + 0.22 * flash)
    return np.clip(a * smooth(t / 0.3) * (1 - smooth((t - (LENGTH - 0.9)) / 0.8)), 0, 255).astype(np.uint8)


def sound():
    t = np.arange(int(SR * LENGTH)) / SR
    x = 0.10 * np.sin(2 * np.pi * 55 * t) * np.sin(2 * np.pi * 0.21 * t + 1) ** 2 * smooth_arr(t)                    # a low drone under it all
    for at, f in ((1.25 + 0.28, 46), (1.7 + 0.28, 39)):                                                              # a thud as each tile lands
        u = np.clip(t - at, 0, None); on = t >= at
        x += on * (np.sin(2 * np.pi * (f * u + 34 * (1 - np.exp(-u * 12)) / 12)) * np.exp(-u / 0.42)
                   + 0.25 * rng.uniform(-1, 1, len(t)) * np.exp(-u / 0.02))
    u = np.clip(t - 2.0, 0, None)                                                                                    # and a thin high tone that rings on
    x += (t >= 2.0) * 0.12 * (np.sin(2 * np.pi * 880 * u) + 0.6 * np.sin(2 * np.pi * 1318.5 * u)) * (1 - np.exp(-u / 0.2)) * np.exp(-u / 1.8)
    x[-int(SR * 0.9):] *= np.linspace(1, 0, int(SR * 0.9))
    return 0.85 * x / np.abs(x).max()


smooth_arr = lambda t: np.clip(t / 0.8, 0, 1)
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
