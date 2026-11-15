"""The title in the manner of the question card before it: a renaissance emblem, drawn in ink on black.

    python tools/cut_title_renaissance.py <output .mp4> [--still seconds]
A salamander is drawn by an unseen pen, nose to curled tail (the salamander was a renaissance device: Francis I of
France took one for his emblem); the name is then written under it in Aquiline Two, the hand of the question
card, and a pen flourish closes it. A lute plays a short cadence. Made here frame by frame, 1920x1080 at 24 fps.
"""
import math, subprocess, sys, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SR, SS = 1920, 1080, 24, 48000, 2
LENGTH = 7.5
INK, EMBER = np.array([233, 221, 196.0]), np.array([214, 120, 62.0])       # old paper; and the warm end of the ink
HERE = Path(__file__).resolve().parent.parent
font = ImageFont.truetype(str(HERE / "unreal/renders/opening/fonts/aquiline_two/AquilineTwo.ttf"), 250 * SS)
out = Path(sys.argv[1])
smooth = lambda k: (lambda k: k * k * (3 - 2 * k))(min(1.0, max(0.0, k)))
rng = np.random.default_rng(5)

# ---- the salamander, once: a line from the nose back along the spine, turning as it goes, the tail curling under
M = 260
S = np.linspace(0, 1, M)
turn = 0.42 * np.sin(2 * np.pi * (1.05 * S + 0.05)) + 5.6 * np.clip(S - 0.62, 0, None) ** 1.7     # heading, radians
step = 640.0 / M
X = np.concatenate([[0], np.cumsum(-np.cos(turn[:-1]) * step)])
Y = np.concatenate([[0], np.cumsum(np.sin(turn[:-1]) * step)])
X += W / 2 - (X.min() + X.max()) / 2; Y += 270 - (Y.min() + Y.max()) / 2


def girth(s):
    head = 23 * math.exp(-((s - 0.045) / 0.05) ** 2)
    return max(1.6, head, 21 * math.exp(-((s - 0.30) / 0.2) ** 2), 17 * (1 - s) ** 1.25 + 1.6)


R = np.array([girth(s) for s in S])
LIMBS = []                                   # each: s where it joins, and three points (joint, knee, foot) with toes
for at, back in ((0.17, False), (0.47, True)):
    i = int(at * (M - 1)); tx, ty = X[i - 3] - X[i + 3], Y[i - 3] - Y[i + 3]; n = math.hypot(tx, ty); tx, ty = tx / n, ty / n
    for side in (1, -1):
        nx, ny = -ty * side, tx * side
        f = -1 if back else 1                # front legs reach forward, hind legs trail
        a = (X[i] + nx * R[i] * 0.6, Y[i] + ny * R[i] * 0.6)
        b = (a[0] + nx * 36 + tx * 16 * f, a[1] + ny * 36 + ty * 16 * f)
        c = (b[0] + nx * 22 + tx * 34 * f, b[1] + ny * 22 + ty * 34 * f)
        toes = [(c[0] + (nx * math.cos(q) + tx * f * math.sin(q)) * 17, c[1] + (ny * math.cos(q) + ty * f * math.sin(q)) * 17) for q in (0.2, 0.9, 1.6)]
        LIMBS.append((at, a, b, c, toes))

# ---- the flourish under the name: one looping stroke of a broad pen (a figure of eight), as on old title pages
TAU = np.linspace(math.pi / 2, 2 * math.pi + math.pi / 2, 420)        # starts and ends at the right-hand tip, where the joins do not show
FX = W / 2 + 250 * np.cos(TAU) / (1 + np.sin(TAU) ** 2)
FY = 872 + 135 * np.sin(TAU) * np.cos(TAU) / (1 + np.sin(TAU) ** 2)
name_w = font.getlength("Salamanca") / SS


def ink(s, t):
    """Mostly the colour of the page's ink; a glow like embers wanders along the body (the salamander of the fire)."""
    k = 0.5 + 0.5 * math.sin(2 * math.pi * (0.9 * s - 0.16 * t))
    c = INK + (EMBER - INK) * 0.75 * k ** 2
    return int(c[0]), int(c[1]), int(c[2]), 255


def frame(i):
    t = i / FPS
    P = lambda v: v * SS
    art = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(art)
    dot = lambda x, y, r, fill: d.ellipse([P(x - r), P(y - r), P(x + r), P(y + r)], fill=fill)
    drawn = smooth((t - 0.3) / 1.7)                                      # how far along the body the pen has got
    for at, a, b, c, toes in LIMBS:
        k = smooth((drawn - at) / 0.12)
        if k > 0:
            col = ink(at, t)
            m = lambda p, q, u: (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u)
            b2 = m(a, b, min(1, 2 * k)); d.line([P(a[0]), P(a[1]), P(b2[0]), P(b2[1])], fill=col, width=int(P(10)))
            dot(*b2, 5.5, col)
            if k > 0.5:
                c2 = m(b, c, 2 * k - 1); d.line([P(b[0]), P(b[1]), P(c2[0]), P(c2[1])], fill=col, width=int(P(7))); dot(*c2, 4.5, col)
            if k >= 1:
                for toe in toes:
                    d.line([P(c[0]), P(c[1]), P(toe[0]), P(toe[1])], fill=col, width=int(P(3.2))); dot(*toe, 2.6, col)
    for j in range(int(drawn * (M - 1)) + 1):                            # the body: a row of touching discs, so its outline is smooth
        dot(X[j], Y[j], R[j], ink(S[j], t))
    if drawn > 0.08:                                                     # eyes, and a row of spots down the back, left as bare page
        tx, ty = X[0] - X[8], Y[0] - Y[8]; n = math.hypot(tx, ty); tx, ty = tx / n, ty / n
        for side in (1, -1):
            dot(X[12] - ty * side * 13, Y[12] + tx * side * 13, 4.2, (0, 0, 0, 255))
        for s in np.arange(0.16, 0.62, 0.052):
            if s < drawn:
                j = int(s * (M - 1)); dot(X[j], Y[j], R[j] * 0.3, (0, 0, 0, 255))
    wrote = smooth((t - 1.9) / 1.5)                                      # the name, written from the left
    if wrote > 0:
        word = Image.new("L", (W * SS, H * SS), 0)
        ImageDraw.Draw(word).text((P(W / 2), P(690)), "Salamanca", font=font, fill=255, anchor="ms")
        edge = P(W / 2 - name_w / 2 - 40 + (name_w + 120) * wrote)
        mask = np.clip((edge - np.arange(W * SS)) / P(70), 0, 1)[None, :]
        alpha = Image.fromarray((np.asarray(word, dtype=np.float32) * mask).astype(np.uint8))
        art.paste(Image.new("RGBA", art.size, tuple(int(v) for v in INK) + (255,)), (0, 0), alpha)
        d = ImageDraw.Draw(art)
    swept = smooth((t - 3.2) / 1.3)                                      # the flourish, in one sweep
    for j in range(1, int(swept * (len(TAU) - 1)) + 1):
        ang = math.atan2(FY[j] - FY[j - 1], FX[j] - FX[j - 1])
        w = 1.2 + 5.2 * abs(math.sin(ang - math.radians(35))) ** 1.5      # a broad nib held at an angle: thick and thin by direction
        d.line([P(FX[j - 1]), P(FY[j - 1]), P(FX[j]), P(FY[j])], fill=tuple(int(v) for v in INK) + (255,), width=max(1, int(P(w))))
    art = art.resize((W, H), Image.LANCZOS)
    glow = Image.eval(art.filter(ImageFilter.GaussianBlur(18)), lambda v: int(v * 0.32))
    base = Image.alpha_composite(Image.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, 255)), glow), art).convert("RGB")
    return (np.asarray(base, dtype=np.float32) * (1 - smooth((t - (LENGTH - 1.1)) / 1.0))).astype(np.uint8)


def pluck(freq, length, bright=0.5):
    n = int(SR / freq); buf = rng.uniform(-1, 1, n)
    buf = np.convolve(buf, np.ones(3) / 3, "same") * (1 - bright) + buf * bright
    y = np.empty(int(SR * length))
    for i in range(len(y)):
        y[i] = buf[i % n]
        buf[i % n] = 0.996 * 0.5 * (buf[i % n] + buf[(i + 1) % n])
    return y * np.minimum(1, np.linspace(1, 0, len(y)) * 6)


def sound():
    """A lute: four rising notes while the salamander is drawn, then a full open chord as the name is written."""
    hz = lambda semis: 293.66 * 2 ** (semis / 12)
    notes = [(0.35, -12, 0.3), (0.75, -5, 0.4), (1.15, 0, 0.5), (1.55, 3, 0.5), (1.95, 7, 0.55),
             (2.45, -24, 0.25), (2.47, -17, 0.3), (2.49, -12, 0.35), (2.51, -5, 0.45), (2.53, 0, 0.5), (2.55, 4, 0.5), (2.57, 12, 0.55)]
    x = np.zeros(int(SR * (LENGTH + 5)))
    for at, semis, bright in notes:
        y = pluck(hz(semis), 4.5, bright); a = int(SR * at); x[a:a + len(y)] += 0.5 * y
    for dly, g in ((0.031, 0.35), (0.047, 0.3), (0.089, 0.22), (0.151, 0.16), (0.233, 0.1)):
        s = int(SR * dly); x[s:] += g * x[:-s]
    x = x[:int(SR * LENGTH)]
    x[-int(SR * 1.2):] *= np.linspace(1, 0, int(SR * 1.2))
    return 0.8 * x / np.abs(x).max()


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
