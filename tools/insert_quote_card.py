"""Puts the question card into the team's intro, at the cut into the dialogue clip.

    python tools/insert_quote_card.py "video/Timeline 1.mov" 257 "video/Timeline 1 with quote.mp4" [--card-only card.mp4]
257 is the first frame of the dialogue clip (24 fps). The card is the question typed out in Departure Mono, the
pixel face of the Junction 2026 poster, white capitals on black; it is silent, and the intro's own sound stops
for it and carries on after. The original file is not changed.
"""
import subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, CUT, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
W, H, FPS = 1920, 1080, 24
SECONDS = 4.5
HERE = Path(__file__).resolve().parent.parent
font = ImageFont.truetype(str(HERE / "unreal/renders/opening/fonts/departure_mono/DepartureMono-1.500/DepartureMono-Regular.otf"), 77)   # 7 x its 11-pixel grid: sharp
LINES = ["CAN ROBOTS HELP OUR LOVED ONES", "WHEN WE CAN'T?"]
TOTAL = sum(len(l) for l in LINES)
TYPE_FROM, TYPE_FOR = 0.35, 2.0             # seconds: typing starts, and how long it takes


def card(i):
    t = i / FPS
    shown = int(TOTAL * min(1.0, max(0.0, (t - TYPE_FROM) / TYPE_FOR)))
    im = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(im)
    cell = font.getlength("M")
    for j, line in enumerate(LINES):
        x, y = W / 2 - cell * len(line) / 2, H / 2 - 77 + j * 132
        part = line[:max(0, shown - sum(len(l) for l in LINES[:j]))]
        d.text((round(x), round(y)), part, font=font, fill=(255, 255, 255))
        here = sum(len(l) for l in LINES[:j]) <= shown < sum(len(l) for l in LINES[:j + 1]) or (j == len(LINES) - 1 and shown >= TOTAL)
        if here and (shown < TOTAL or int(t * 2.5) % 2 == 0):               # a block cursor: steady while typing, blinking after
            cx = round(x + cell * len(part))
            d.rectangle([cx + 4, round(y) + 8, cx + cell - 6, round(y) + 74], fill=(255, 255, 255))
    return np.asarray(im, dtype=np.uint8)


n = int(SECONDS * FPS)
if "--card-only" in sys.argv:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-f", "lavfi", "-i",
                          "anullsrc=r=48000:cl=stereo", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                          sys.argv[sys.argv.index("--card-only") + 1]], stdin=subprocess.PIPE)
    for i in range(n): p.stdin.write(card(i).tobytes())
    p.stdin.close(); p.wait()
at = CUT / FPS
# video: the intro up to the cut, the card (fed in as raw frames), the rest; audio: the same split with silence between, tiny fades at the joins
graph = (f"[0:v]fps={FPS},scale={W}:{H},setsar=1,split[v0][v1];[v0]trim=end_frame={CUT},setpts=PTS-STARTPTS[a];[v1]trim=start_frame={CUT},setpts=PTS-STARTPTS[b];"
         f"[1:v]setsar=1,format=yuv420p[c];[a]format=yuv420p[a2];[b]format=yuv420p[b2];[a2][c][b2]concat=n=3:v=1:a=0[v];"
         f"[0:a]aresample=48000,asplit[x0][x1];[x0]atrim=end={at},asetpts=PTS-STARTPTS,afade=t=out:st={at - 0.04}:d=0.04[xa];"
         f"[x1]atrim=start={at},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.04[xb];anullsrc=r=48000:cl=stereo,atrim=end={SECONDS}[xs];[xa][xs][xb]concat=n=3:v=0:a=1[au]")
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                      "-filter_complex", graph, "-map", "[v]", "-map", "[au]", "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
for i in range(n): p.stdin.write(card(i).tobytes())
p.stdin.close(); print("ffmpeg exit", p.wait(), "->", out)
