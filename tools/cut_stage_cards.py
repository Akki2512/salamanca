"""Chapter cards for the talking part of the demo video, in the element-tile style of the title.

    python tools/cut_stage_cards.py <output folder> [--sheet]
One short clip per stage (1920x1080, 24 fps, 3.5 s, a thud as the tiles land). In each title the letters that
are a real element's symbol are set as that element's tile. --sheet also writes stage_cards.jpg, one frame of each.
"""
import math, subprocess, sys, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SR = 1920, 1080, 24, 48000
LENGTH = 3.5
BG, WHITE = (6, 9, 11), (238, 242, 244)
AV = "/System/Library/Fonts/Avenir Next.ttc"
SIZE, TILE, GAP, BASE = 170, 206, 12, 590
big, sym = ImageFont.truetype(AV, SIZE, index=2), ImageFont.truetype(AV, 116, index=0)
small, sub, count = ImageFont.truetype(AV, 25, index=5), ImageFont.truetype(AV, 44, index=7), ImageFont.truetype(AV, 30, index=2)
TEAL, AMBER = ((22, 150, 136), (10, 84, 80)), ((226, 148, 52), (150, 78, 30))
EL = {"S": (16, "32.06", "Sulfur"), "Re": (75, "186.21", "Rhenium"), "Al": (13, "26.982", "Aluminium"), "Es": (99, "(252)", "Einsteinium"),
      "Ta": (73, "180.95", "Tantalum"), "Cr": (24, "51.996", "Chromium")}
# file name, the title in pieces (a piece that is a key of EL becomes a tile), the line under it
STAGES = [("05_stage_1_suit", ["S", "uit setup"], "the worker's body, streamed live"),
          ("05_stage_2_unreal", ["Un", "Re", "Al", " Engine"], "a realistic room, linked to the robot"),
          ("05_stage_3_quest", ["Qu", "Es", "t VR"], "seeing through the robot's eyes"),
          ("05_stage_4_recorder", ["Da", "Ta", " recorder"], "every run becomes training data"),
          ("05_stage_5_crawling", ["Cr", "awling"], "what the data taught a robot")]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
smooth = lambda k: (lambda k: k * k * (3 - 2 * k))(min(1.0, max(0.0, k)))
rng = np.random.default_rng(3)
haze = [Image.fromarray((rng.random((27, 48)) * 255).astype(np.uint8)).resize((2600, 1460), Image.BICUBIC).filter(ImageFilter.GaussianBlur(95)) for _ in range(2)]


def tile(symbol, colours, k):
    num, mass, name = EL[symbol]; top, bottom = colours
    t_ = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0)); d = ImageDraw.Draw(t_)
    for y in range(TILE):
        d.line([0, y, TILE, y], fill=tuple(int(a + (b - a) * y / TILE) for a, b in zip(top, bottom)) + (255,))
    d.rectangle([0, 0, TILE - 1, TILE - 1], outline=WHITE + (255,), width=4)
    d.text((TILE / 2, TILE * 0.72), symbol, font=sym, fill=WHITE + (255,), anchor="ms")
    d.text((TILE - 14, 12), str(num), font=small, fill=WHITE + (255,), anchor="ra")
    d.text((14, 12), mass, font=small, fill=WHITE + (230,), anchor="la")
    d.text((TILE / 2, TILE - 15), name.upper(), font=ImageFont.truetype(AV, 18, index=5), fill=WHITE + (230,), anchor="ms")
    z = 1 + 0.5 * (1 - smooth(k)) ** 2
    t_ = t_.resize((int(TILE * z), int(TILE * z)), Image.LANCZOS)
    t_.putalpha(Image.eval(t_.getchannel("A"), lambda v: int(v * smooth(k * 2))))
    return t_


def frame(stage, number, i):
    _, pieces, under = stage
    t = i / FPS
    base = Image.new("RGB", (W, H), BG)
    for j, hz in enumerate(haze):
        ox, oy = int(340 + 300 * math.sin(0.11 * t + 2 * j + number)), int(190 + 160 * math.cos(0.09 * t + j + number))
        m = Image.eval(hz.crop((ox, oy, ox + W, oy + H)), lambda v: int(max(0, v - 105) * 1.6))
        base.paste(Image.new("RGB", (W, H), ((18, 120, 110), (150, 90, 30))[j]), (0, 0), m)
    art = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(art)
    widths = [TILE + 2 * GAP if p in EL else big.getlength(p) for p in pieces]
    x = W / 2 - sum(widths) / 2
    d.text((W / 2, BASE - 260), "STAGE %d OF %d" % (number, len(STAGES)), font=count, fill=WHITE + (int(170 * smooth((t - 0.15) / 0.5)),), anchor="ms")
    flash, tiles = 0.0, 0
    for p, w in zip(pieces, widths):
        if p in EL:
            k = (t - 0.5 - 0.3 * tiles) / 0.28
            if k > 0:
                tl = tile(p, (TEAL, AMBER)[tiles % 2], min(1.0, k))
                art.alpha_composite(tl, (int(x + GAP + TILE / 2 - tl.width / 2), int(BASE - TILE * 0.72 + TILE / 2 - tl.height / 2)))
                d = ImageDraw.Draw(art)
                flash = max(flash, 1 - abs(k - 1) / 0.9 if 0.4 < k < 1.9 else 0)
            tiles += 1
        else:
            d.text((x, BASE), p, font=big, fill=WHITE + (int(255 * smooth((t - 0.3) / 0.5)),), anchor="ls")
        x += w
    d.text((W / 2, BASE + 130), under, font=sub, fill=WHITE + (int(200 * smooth((t - 1.2) / 0.6)),), anchor="ms")
    glow = Image.eval(art.filter(ImageFilter.GaussianBlur(24)), lambda v: int(v * 0.35))
    pic = Image.alpha_composite(Image.alpha_composite(base.convert("RGBA"), glow), art).convert("RGB")
    a = np.asarray(pic, dtype=np.float32) * (1 + 0.18 * flash)
    return np.clip(a * smooth(t / 0.25) * (1 - smooth((t - (LENGTH - 0.6)) / 0.5)), 0, 255).astype(np.uint8)


def sound(tiles):
    t = np.arange(int(SR * LENGTH)) / SR
    x = 0.10 * np.sin(2 * np.pi * 55 * t) * np.clip(t / 0.6, 0, 1)
    for j in range(tiles):
        at = 0.5 + 0.3 * j + 0.28; u = np.clip(t - at, 0, None)
        x += (t >= at) * np.sin(2 * np.pi * ((44 - 4 * j) * u + 34 * (1 - np.exp(-u * 12)) / 12)) * np.exp(-u / 0.4)
    x[-int(SR * 0.6):] *= np.linspace(1, 0, int(SR * 0.6))
    return 0.85 * x / np.abs(x).max()


stills = []
for number, stage in enumerate(STAGES, 1):
    path = out / (stage[0] + ".mp4"); wav = path.with_suffix(".wav")
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((sound(sum(p in EL for p in stage[1])) * 32767).astype(np.int16).tobytes())
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav),
                          "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                          "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for i in range(int(LENGTH * FPS)):
        fr = frame(stage, number, i)
        if i == int(2.4 * FPS): stills.append(Image.fromarray(fr).resize((640, 360)))
        p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait(); wav.unlink()
    print("wrote", path)
if "--sheet" in sys.argv:
    sheet = Image.new("RGB", (1280, 360 * 3), (0, 0, 0))
    for j, im in enumerate(stills):
        sheet.paste(im, (640 * (j % 2), 360 * (j // 2)))
    sheet.save(out / "stage_cards.jpg", quality=90)
