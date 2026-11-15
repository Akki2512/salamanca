"""Footage for the talking part of the demo video, cut from a recorded suit run (recordings/dataset/episode_NNNNNN).

    python tools/cut_footage.py recordings/dataset/episode_000004 <output folder>

    06_end_to_end_20s.mp4    the whole story in 20 seconds: the robot's eyes, the man's view inset, a caption per step
    05b_quest_view.mp4       stage 3: what the worker sees in the headset, at real speed (glass picked up, then handed over)
    05b_recorder_view.mp4    stage 4: both recorded cameras side by side with the numbers saved for that step
Silent (voice goes over them). 1920x1080, 24 fps.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 24
ep, out = Path(sys.argv[1]), Path(sys.argv[2])
AV = "/System/Library/Fonts/Avenir Next.ttc"
cap, tiny, mono = ImageFont.truetype(AV, 46, index=2), ImageFont.truetype(AV, 28, index=5), ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 27)
steps = [json.loads(l) for l in open(ep / "steps.jsonl")]
T = np.array([s["t"] for s in steps])
at = lambda t: steps[min(len(steps) - 1, int(np.searchsorted(T, t)))]
cache = {}


def picture(cam, k, size):
    key = (cam, k, size)
    if key not in cache:
        if len(cache) > 40: cache.clear()
        cache[key] = Image.open(ep / "frames" / cam / ("%06d.jpg" % k)).convert("RGB").resize(size, Image.LANCZOS)
    return cache[key]


def write(name, n, frame):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p",
                          "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out / name)], stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(np.asarray(frame(i), dtype=np.uint8).tobytes())
    p.stdin.close(); p.wait(); print("wrote", out / name, round(n / FPS, 1), "s")


def label(d, xy, text, font=cap, anchor="ls"):
    box = d.textbbox(xy, text, font=font, anchor=anchor)
    d.rounded_rectangle([box[0] - 18, box[1] - 10, box[2] + 18, box[3] + 12], 10, fill=(6, 9, 11))
    d.text(xy, text, font=font, fill=(238, 242, 244), anchor=anchor)


def story(s):
    L = s["label"]
    if L.get("pushing"): return "Wheels him to the bed"
    if L["phase"] == "bed": return "He is in bed"
    if L["phase"] == "drink": return "He drinks"
    if L["phase"] == "cough": return "Brings a glass of water" if L.get("assist") in ("approach", "descend", "closing", "holding") and s["t"] > 60 else "He coughs and rings his bell"
    return "Picks the sweater up off the floor" if L.get("reach") or L.get("assist") == "holding" else "A worker in a suit drives the robot"


# ---- the whole story in 20 seconds
A, B, N = 26.0, 196.0, 20 * FPS


def end_to_end(i):
    s = at(A + (B - A) * i / N)
    im = picture("robot_eyes", s["frame"]["robot_eyes"] or 0, (W, H)).copy()
    inset = picture("person", s["frame"]["person"] or 0, (512, 288))
    im.paste(Image.new("RGB", (520, 296), (238, 242, 244)), (W - 520 - 36, 36)); im.paste(inset, (W - 516 - 36, 40))
    d = ImageDraw.Draw(im)
    label(d, (W - 36 - 520, 372), "the man's view", tiny, "la")
    label(d, (60, H - 70), story(s))
    label(d, (W - 60, H - 70), "%.1fx speed" % ((B - A) / 20), tiny, "rs")
    return im


if "--only-recorder" not in sys.argv: write("06_end_to_end_20s.mp4", N, end_to_end)

# ---- stage 3: the headset's picture at real speed
PARTS = [(66.0, 74.5, "In the headset: the robot's own eyes"), (104.5, 112.0, "Trigger: the glass is handed over")]
frames = [(a + j / FPS, text) for a, b, text in PARTS for j in range(int((b - a) * FPS))]


def quest(i):
    t, text = frames[i]
    im = picture("robot_eyes", at(t)["frame"]["robot_eyes"] or 0, (W, H)).copy()
    label(ImageDraw.Draw(im), (60, H - 70), text)
    return im


if "--only-recorder" not in sys.argv: write("05b_quest_view.mp4", len(frames), quest)

# ---- stage 4: what one step of a recording holds
R0, RN = 126.0, 12 * FPS


def recorder(i):
    s = at(R0 + i / FPS)
    im = Image.new("RGB", (W, H), (6, 9, 11))
    im.paste(picture("robot_eyes", s["frame"]["robot_eyes"] or 0, (944, 531)), (12, 60))
    im.paste(picture("person", s["frame"]["person"] or 0, (944, 531)), (964, 60))
    d = ImageDraw.Draw(im)
    d.text((12, 14), "camera 1: robot's eyes", font=tiny, fill=(238, 242, 244)); d.text((964, 14), "camera 2: the man's view", font=tiny, fill=(238, 242, 244))
    st, L = s["state"], s["label"]
    f = lambda v: " ".join("%+.2f" % x for x in v)
    lines = ["step %06d   t %7.2f s   30 steps per second" % (s["i"], s["t"]),
             "phase %-6s  grasp %-9s  pushing %-5s  man %s" % (L.get("phase"), L.get("assist"), L.get("pushing"), L.get("man")),
             "base   x y yaw   " + f(st["base"]),
             "arms   14 joints " + f(st["arm_q"]),
             "legs   12 joints " + f(st["leg_q"]),
             "waist   3 joints " + f(st["waist_q"]) + "      action (arm targets) " + f(list(s["action"].values())[0][:5]) + " ..."]
    for j, line in enumerate(lines):
        d.text((40, 640 + 66 * j), line, font=mono, fill=(238, 242, 244) if j else (226, 148, 52))
    return im


write("05b_recorder_view.mp4", RN, recorder)
