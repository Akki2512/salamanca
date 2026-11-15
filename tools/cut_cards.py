"""The demo video's three clips after the opening shot (same size and frame rate: 1920x1080, 24 fps).

    PYTHONPATH=. python tools/cut_cards.py <last frame of the previous clip .png> <output folder>

    02_static.mp4     silence: the picture breaks up into television static, then goes black
    03_question.mp4   the question in an old book face (Luminari), with a plucked-string tune made here
    04_title.mp4      the title, on a hard cut, with a low strike and a bell chord
Everything is made in this file: no footage, fonts or sounds are downloaded.
"""
import os, subprocess, sys, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1920, 1080, 24, 48000
# RH_FONT: another face for the question (Aquiline Two, a renaissance hand: renders/opening/fonts/aquiline_two);
# RH_ONLY=question makes just that clip, as 03_question_v2.mp4; RH_SIZE its letter size.
FONT = os.environ.get("RH_FONT", "/System/Library/Fonts/Supplemental/Luminari.ttf")
ONLY, SIZE = os.environ.get("RH_ONLY"), int(os.environ.get("RH_SIZE", "96"))
INK = (233, 221, 196)                       # old paper, on black
last, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)
smooth = lambda k: (lambda k: k * k * (3 - 2 * k))(float(np.clip(k, 0, 1)))


def write(name, frames, n, sound):
    if ONLY and ONLY not in name:
        return
    name = name + "_v2" if ONLY else name
    """frames: a function of the frame number giving an HxWx3 uint8 picture. sound: mono float samples or None (silence)."""
    wav = out / (name + ".wav")
    x = np.zeros(int(SR * n / FPS)) if sound is None else np.pad(sound, (0, max(0, int(SR * n / FPS) - len(sound))))[:int(SR * n / FPS)]
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav),
                          "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                          "-movflags", "+faststart", str(out / (name + ".mp4"))], stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(np.ascontiguousarray(frames(i), dtype=np.uint8).tobytes())
    p.stdin.close(); p.wait(); wav.unlink()
    print("wrote", out / (name + ".mp4"), round(n / FPS, 2), "s")


# ---------------------------------------------------------------- 02: static, then black, in silence
still = np.asarray(Image.open(last).convert("RGB").resize((W, H)), dtype=np.float32)
grey = still.mean(axis=2, keepdims=True)
rows = np.arange(H)


def static(i):
    t = i / FPS
    k = smooth(t / 3.4)                                  # 0 the picture, 1 all static
    img = still + (grey - still) * min(1.0, 1.4 * k)     # the colour drains first
    if k > 0.03:                                         # bands of the picture slip sideways, more and more of them
        shift = np.zeros(H, dtype=int)
        for _ in range(int(2 + 26 * k)):
            a = rng.integers(0, H); h = rng.integers(4, int(20 + 160 * k))
            shift[a:a + h] = rng.integers(-int(8 + 150 * k * k), int(8 + 150 * k * k) + 1)
        img = img[rows[:, None], (np.arange(W)[None, :] + shift[:, None]) % W]
    snow = np.repeat(np.repeat(rng.integers(0, 256, (H // 2, W // 2), dtype=np.uint8), 2, 0), 2, 1).astype(np.float32)[:, :, None]
    bar = 0.75 + 0.25 * np.sin((rows / H * 2.2 - t * 1.3) * 2 * np.pi)[:, None, None]           # the slow dark band of a bad signal
    img = img * (1 - k ** 1.4) * (1 - 0.3 * k + 0.3 * k * rng.random()) + snow * bar * k ** 1.4
    img[::3] *= 1 - 0.22 * k                                                                     # scan lines
    return np.clip(img * (1 - smooth((t - 4.4) / 0.9)), 0, 255)                                  # ... and out to black


write("02_static", static, int(6.4 * FPS), None)


# ---------------------------------------------------------------- 03: the question
def text_layer(lines, size, spacing=1.5):
    """White-on-black picture of centred lines, one array per line, so each can come up on its own."""
    font = ImageFont.truetype(FONT, size)
    total = size * spacing * (len(lines) - 1)
    layers = []
    for j, line in enumerate(lines):
        im = Image.new("L", (W, H), 0)
        ImageDraw.Draw(im).text((W / 2, H / 2 - total / 2 + j * size * spacing), line, font=font, fill=255, anchor="mm")
        layers.append(np.asarray(im, dtype=np.float32)[:, :, None] / 255)
    return layers


def pluck(freq, length, bright=0.5):
    """A plucked string (Karplus-Strong): a burst of noise going round a delay line and softening."""
    n = int(SR / freq); buf = rng.uniform(-1, 1, n)
    buf = np.convolve(buf, np.ones(3) / 3, "same") * (1 - bright) + buf * bright
    y = np.empty(int(SR * length))
    for i in range(len(y)):
        y[i] = buf[i % n]
        buf[i % n] = 0.996 * 0.5 * (buf[i % n] + buf[(i + 1) % n])
    return y * np.minimum(1, np.linspace(1, 0, len(y)) * 6)


def tune(length):
    """A slow air in D dorian for a lute: one line over an open-fifth bass."""
    hz = lambda semis: 293.66 * 2 ** (semis / 12)                       # D4 = 0
    line = [(0.6, 0), (1.5, 3), (2.4, 7), (3.3, 5), (3.75, 3), (4.2, 2), (5.1, 0), (6.0, -2), (6.45, 0), (6.9, 3), (7.8, 2), (8.7, 0)]
    bass = [(0.6, -24), (0.6, -17), (3.3, -19), (3.3, -12), (5.1, -24), (5.1, -17), (6.9, -22), (8.7, -24), (8.7, -17), (8.7, -12)]
    x = np.zeros(int(SR * (length + 4)))
    for at, semis in line:
        y = pluck(hz(semis), 3.2, 0.55); a = int(SR * at); x[a:a + len(y)] += 0.55 * y
    for at, semis in bass:
        y = pluck(hz(semis), 4.0, 0.25); a = int(SR * (at + 0.012 * (semis % 5))); x[a:a + len(y)] += 0.5 * y
    for d, g in ((0.031, 0.35), (0.047, 0.3), (0.089, 0.22), (0.151, 0.16), (0.233, 0.1)):    # a small stone room
        s = int(SR * d); x[s:] += g * x[:-s]
    x = x[:int(SR * length)]
    x[-int(SR * 1.5):] *= np.linspace(1, 0, int(SR * 1.5))
    return 0.7 * x / np.abs(x).max()


q1, q2 = text_layer(["Can robots help our loved ones", "when we can’t?"], SIZE)
ink = np.array(INK, dtype=np.float32)[None, None, :]
QLEN = 10.0


def question(i):
    t = i / FPS
    gone = 1 - smooth((t - 8.2) / 1.4)
    return (q1 * smooth((t - 0.6) / 1.6) + q2 * smooth((t - 3.0) / 1.6)) * gone * ink


write("03_question", question, int(QLEN * FPS), tune(QLEN))

# ---------------------------------------------------------------- 04: the title
title = Image.new("L", (W, H), 0)
ImageDraw.Draw(title).text((W / 2, H / 2), "Salamanca", font=ImageFont.truetype(FONT, 250), fill=255, anchor="mm")
TLEN = 5.0


def title_frame(i):
    t = i / FPS
    z = 1.0 + 0.035 * t / TLEN                                           # it drifts very slightly towards us
    w, h = int(W / z) // 2 * 2, int(H / z) // 2 * 2
    im = title.crop(((W - w) // 2, (H - h) // 2, (W + w) // 2, (H + h) // 2)).resize((W, H), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32)[:, :, None] / 255 * ink * (1 - smooth((t - 4.0) / 0.9))


def strike():
    t = np.arange(int(SR * TLEN)) / SR
    drum = np.sin(2 * np.pi * (38 * t + 30 * (1 - np.exp(-t * 9)) / 9)) * np.exp(-t / 0.55)      # a low strike falling in pitch
    bell = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau) for f, a, tau in
               ((146.83, 0.6, 2.2), (220.0, 0.5, 2.0), (293.66, 0.5, 1.8), (440.0, 0.25, 1.4), (587.33 * 1.004, 0.22, 1.2), (880.0 * 1.007, 0.1, 0.8)))
    x = (0.9 * drum + 0.5 * bell) * np.minimum(1, t / 0.003)
    x[-int(SR * 0.8):] *= np.linspace(1, 0, int(SR * 0.8))
    return 0.85 * x / np.abs(x).max()


write("04_title", title_frame, int(TLEN * FPS), strike())
