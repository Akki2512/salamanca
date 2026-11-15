"""The title, version 2: a salamander drawn as a jointed rig crawls in, settles above the name, and keeps changing colour.

    python tools/cut_title.py <output .mp4>
The mark is the idea of the project in one drawing: a body of linked joints (a motion-capture rig, a robot's
kinematic chain) in the shape of a salamander, seen from above, its colour running along it. Drawn here frame by
frame (1920x1080, 24 fps); the sound is made here too. Nothing is downloaded.
"""
import colorsys, math, subprocess, sys, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SR, SS = 1920, 1080, 24, 48000, 2            # SS: drawn at twice the size, then reduced, for clean edges
LENGTH, ARRIVE, NAME_AT = 6.5, 2.0, 1.75                # seconds: the whole clip, the salamander in place, the name starts
BG, WHITE = (7, 9, 13), (236, 240, 245)
out = Path(sys.argv[1])
smooth = lambda k: (lambda k: k * k * (3 - 2 * k))(min(1.0, max(0.0, k)))
font = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", 104 * SS, index=5)       # Medium
N, BODY = 15, 560.0                                     # spine joints; length in pixels, nose to tail tip
BIG = 1.6
LEGS = (0.20, 0.52)                                     # where along the body the shoulders and the hips are


def radius(s):
    """How thick the body is at s (0 the nose, 1 the tail's tip)."""
    head = 21 * math.exp(-((s - 0.05) / 0.07) ** 2)
    trunk = 20 * math.exp(-((s - 0.36) / 0.22) ** 2)
    return max(3.2, head, trunk, 15 * (1 - s) ** 1.3 + 3)


def colour(s, t, light=1.0):
    """Colour runs along the body and never stops moving: teal, through green and gold, to coral."""
    h = 0.30 + 0.21 * math.sin(2 * math.pi * (0.55 * s - 0.22 * t))          # 0.09 (orange) .. 0.51 (cyan)
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.78, light)
    return int(r * 255), int(g * 255), int(b * 255), 255


def salamander(t):
    """Joints and limbs at time t, in picture coordinates: (spine [(x, y, r, s)], legs [(shoulder, elbow, foot, s)])."""
    k = smooth(t / ARRIVE)
    travelled = 1150 * k                                               # it comes in from the left
    nose_x, mid_y = W / 2 + BODY / 2 - 1150 + travelled, 390
    gait = travelled / 150.0 * 2 * math.pi + 0.5 * t                   # the body's wave keeps pace with the ground covered
    amp = 30 * (1 - k) + 9                                             # a wide swing while it walks, a slow breath once it is still
    pts = []
    for i in range(N):
        s = i / (N - 1)
        y = mid_y + amp * (0.25 + s) * math.sin(gait - 2 * math.pi * 0.95 * s) + 34 * math.sin(math.pi * (s - 0.15)) * k * (s - 0.5)
        pts.append((nose_x - BODY * s, y, radius(s), s))
    legs = []
    for j, at in enumerate(LEGS):
        i = round(at * (N - 1)); (x, y, r, s) = pts[i]
        tx, ty = pts[i - 1][0] - pts[i + 1][0], pts[i - 1][1] - pts[i + 1][1]
        n = math.hypot(tx, ty); tx, ty = tx / n, ty / n
        for side in (1, -1):
            swing = math.sin(gait + (0 if (j == 0) == (side == 1) else math.pi)) * (1 - 0.75 * k)      # diagonal pairs step together
            nx, ny = -ty * side, tx * side
            sh = (x + nx * r * 0.7, y + ny * r * 0.7)
            el = (sh[0] + nx * 34 + tx * (10 + 20 * swing), sh[1] + ny * 34 + ty * (10 + 20 * swing))
            ft = (el[0] + nx * 22 + tx * (22 + 30 * swing), el[1] + ny * 22 + ty * (22 + 30 * swing))
            legs.append((sh, el, ft, s))
    z = lambda x, y: (W / 2 + (x - W / 2) * BIG, 380 + (y - 390) * BIG)          # the whole animal, larger, about its place above the name
    return [(*z(x, y), r * BIG, s) for x, y, r, s in pts], [(z(*sh), z(*el), z(*ft), s) for sh, el, ft, s in legs]


def frame(i):
    t = i / FPS
    pts, legs = salamander(t)
    P = lambda v: v * SS
    art = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(art)
    dot = lambda x, y, r, fill: d.ellipse([P(x - r), P(y - r), P(x + r), P(y + r)], fill=fill)
    for sh, el, ft, s in legs:                                         # limbs: two links and a foot, like the rig of an arm
        c = colour(s, t)
        d.line([P(sh[0]), P(sh[1]), P(el[0]), P(el[1]), P(ft[0]), P(ft[1])], fill=c, width=int(P(5 * BIG)), joint="curve")
        dot(*el, 6.5 * BIG, c); dot(*ft, 8.5 * BIG, c); dot(*ft, 3.5 * BIG, BG + (255,))
    for a, b in zip(pts, pts[1:]):                                     # the spine's links
        d.line([P(a[0]), P(a[1]), P(b[0]), P(b[1])], fill=colour((a[3] + b[3]) / 2, t), width=int(P(min(a[2], b[2]) * 1.15)))
    for x, y, r, s in pts:                                             # ... and its joints: a ring with a dark centre
        dot(x, y, r, colour(s, t)); dot(x, y, r * 0.42, BG + (255,))
    hx, hy, hr, _ = pts[0]                                             # two eyes on the head
    tx, ty = pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]; n = math.hypot(tx, ty); tx, ty = tx / n, ty / n
    for side in (1, -1):
        dot(hx + tx * 3 - ty * side * hr * 0.62, hy + ty * 3 + tx * side * hr * 0.62, 4.6 * BIG, WHITE + (255,))
    # the name: letters come together from wide apart as the salamander arrives
    k = smooth((t - NAME_AT) / 1.1)
    if k > 0:
        text, track = "SALAMANCA", (0.30 + 0.5 * (1 - k)) * 104
        widths = [font.getlength(ch) / SS for ch in text]
        total = sum(widths) + track * (len(text) - 1)
        x = W / 2 - total / 2
        for ch, w in zip(text, widths):
            d.text((P(x), P(690)), ch, font=font, fill=WHITE + (int(255 * k),), anchor="ls")
            x += w + track
        rule = 300 * smooth((t - NAME_AT - 0.5) / 0.9)                 # a thin line under the name, in the body's colours
        for j in range(int(rule)):
            for sx in (-1, 1):
                d.point((P(W / 2 + sx * j), P(750)), fill=colour(0.5 + sx * j / 600, t))
        if rule:
            d.line([P(W / 2 - rule), P(750), P(W / 2 + rule), P(750)], fill=colour(0.5, t)[:3] + (0,), width=1)
            for j in range(-int(rule), int(rule)):
                d.rectangle([P(W / 2 + j), P(749), P(W / 2 + j + 1), P(751)], fill=colour(0.5 + j / 600, t))
    art = art.resize((W, H), Image.LANCZOS)
    glow = art.filter(ImageFilter.GaussianBlur(22))
    base = Image.new("RGBA", (W, H), BG + (255,))
    base = Image.alpha_composite(base, Image.eval(glow, lambda v: int(v * 0.55)))
    base = Image.alpha_composite(base, art).convert("RGB")
    fade = smooth(t / 0.35) * (1 - smooth((t - (LENGTH - 0.9)) / 0.8))
    return (np.asarray(base, dtype=np.float32) * fade).astype(np.uint8)


def sound():
    t = np.arange(int(SR * LENGTH)) / SR
    hit = NAME_AT + 0.25
    rise = np.sin(2 * np.pi * (70 * t + 55 * t * t / (2 * hit))) * np.clip(t / hit, 0, 1) ** 2 * (t < hit) * 0.35      # a low swell up to the name
    u = np.clip(t - hit, 0, None); on = (t >= hit)
    sub = np.sin(2 * np.pi * (42 * u + 28 * (1 - np.exp(-u * 10)) / 10)) * np.exp(-u / 0.5) * on                       # the strike
    pad = sum(a * np.sin(2 * np.pi * f * (1 + 0.0015 * np.sin(2 * np.pi * 0.7 * u + p)) * u) for f, a, p in
              ((146.83, 0.5, 0), (220.0, 0.45, 1), (293.66, 0.4, 2), (369.99, 0.3, 3), (440.0, 0.22, 4), (880.0, 0.08, 5)))
    pad = pad * (1 - np.exp(-u / 0.05)) * np.exp(-u / 2.4) * on * 0.3                                                 # an open major chord that rings on
    steps = np.zeros_like(t)                                                                                           # soft ticks as it walks in
    for k in range(7):
        a = int(SR * (0.25 + 0.24 * k)); n = int(SR * 0.05); tt = np.arange(n) / SR
        steps[a:a + n] += np.sin(2 * np.pi * (900 + 60 * (k % 2)) * tt) * np.exp(-tt / 0.008) * 0.12
    x = rise + 0.9 * sub + pad + steps
    x[-int(SR * 0.9):] *= np.linspace(1, 0, int(SR * 0.9))
    return 0.85 * x / np.abs(x).max()


n = int(LENGTH * FPS)
if "--still" in sys.argv:
    Image.fromarray(frame(int(float(sys.argv[sys.argv.index("--still") + 1]) * FPS))).save(out)
    sys.exit()
wav = out.with_suffix(".wav")
with wave.open(str(wav), "wb") as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes((sound() * 32767).astype(np.int16).tobytes())
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav),
                      "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                      "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
for i in range(n):
    p.stdin.write(frame(i).tobytes())
p.stdin.close(); p.wait(); wav.unlink()
print("wrote", out, LENGTH, "s")
