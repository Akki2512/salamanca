"""Cuts the demo video's opening shot from the frames render_shot.py made.

    PYTHONPATH=. python tools/cut_opening.py unreal/renders/opening/take3 recordings/shot_opening_packets.jsonl out.mp4

Adds the bell's sound where the button goes down, and from FLICKER_AT the picture flickers between the robot and
a motion-capture figure standing exactly where the robot stands, ending on the figure (the cut to the worker in
the suit comes next). The figure is the robot's own joints for that frame (the packets), worked out with the G1
model and drawn through the shot's camera (shot.json) over a frame taken before the robot came in.
"""
import json, math, subprocess, sys, wave
from pathlib import Path
import numpy as np
import pinocchio as pin
from PIL import Image, ImageDraw, ImageFilter
from remote_hands.walk import URDF

frames_dir, packets_path, out_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
shot = json.load(open(frames_dir / "shot.json"))
FPS, (W, H), OFFSET = shot["fps"], shot["size"], shot["offset"]
RING_AT, PLATE_AT, FLICKER_AT = 0.9, 4.45, 7.6
# (seconds of the figure, seconds of the robot), then the figure stays
BURSTS = [(0.08, 0.42), (0.12, 0.30), (0.17, 0.21), (0.25, 0.17), (0.42, 0.12), (0.6, 0.08)]
FADE = 0.06                                   # each change is a quick dissolve, not a hard cut
packets = open(packets_path).read().splitlines()
model = pin.buildModelFromUrdf(str(URDF)); data = model.createData()
BONES = [("pelvis", "chest"), ("chest", "head"), ("chest", "left_shoulder_roll_link"), ("chest", "right_shoulder_roll_link")]
for s in ("left", "right"):
    BONES += [(f"{s}_shoulder_roll_link", f"{s}_elbow_link"), (f"{s}_elbow_link", f"{s}_wrist_yaw_link"), (f"{s}_wrist_yaw_link", f"{s}_hand_palm_link"),
              ("pelvis", f"{s}_hip_pitch_link"), (f"{s}_hip_pitch_link", f"{s}_knee_link"), (f"{s}_knee_link", f"{s}_ankle_roll_link")]
NAMES = sorted({n for b in BONES for n in b} - {"chest", "head"}) + ["head_link"]


def joints_world(packet):
    """The figure's joints in the level's frame (cm)."""
    q = pin.neutral(model)
    for name, v in {**packet["joints"], **packet["ext"]["joints"]}.items():
        if model.existJointName(name):
            q[model.joints[model.getJointId(name)].idx_q] = v
    pin.forwardKinematics(model, data, q)
    pos, (w, _, _, z) = np.array(packet["ext"]["pelvis"]["pos"]), packet["ext"]["pelvis"]["quat"]
    yaw = 2 * math.atan2(z, w); c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    P = {n: pos + Rz @ pin.updateFramePlacement(model, data, model.getFrameId(n)).translation for n in NAMES}
    P["chest"] = (P["left_shoulder_roll_link"] + P["right_shoulder_roll_link"]) / 2
    P["head"] = P["chest"] + (P["chest"] - P["pelvis"]) / np.linalg.norm(P["chest"] - P["pelvis"]) * 0.21
    return {n: np.array([150 + 100 * p[0], 230 - 100 * p[1], 100 * p[2]]) for n, p in P.items()}


def project(p, cam):
    C, L = np.array(cam["cam"]), np.array(cam["look"])
    F = (L - C) / np.linalg.norm(L - C)
    R = np.cross([0, 0, 1], F); R /= np.linalg.norm(R)
    U = np.cross(F, R)
    d = p - C; depth = d @ F
    k = (W / 2) / math.tan(math.radians(cam["fov"]) / 2)
    return np.array([W / 2 + k * (d @ R) / depth, H / 2 - k * (d @ U) / depth]), depth


def figure(i, plate):
    cam = shot["frames"][i]
    J = joints_world(json.loads(packets[min(len(packets) - 1, int(round((cam["t"] + OFFSET) * 30)))]))
    pts = {n: project(p, cam) for n, p in J.items()}
    scale = (W / 2) / math.tan(math.radians(cam["fov"]) / 2) / pts["pelvis"][1]            # pixels per cm at the figure
    scale = min(scale, 11.0)                    # this close, the lines stay lines
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    line = (64, 224, 255, 255)
    for a, b in BONES:
        d.line([tuple(pts[a][0]), tuple(pts[b][0])], fill=line, width=max(3, int(2.2 * scale)))
    hx, hy = pts["head"][0]; r = 9.5 * scale
    d.ellipse([hx - r, hy - r, hx + r, hy + r], outline=line, width=max(3, int(2.2 * scale)))
    for n, (xy, _) in pts.items():
        if n not in ("head", "head_link"):
            r = (3.4 if n in ("pelvis", "chest") else 2.8) * scale
            d.ellipse([xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r], fill=(255, 255, 255, 255), outline=line, width=max(2, int(scale)))
    glow = layer.filter(ImageFilter.GaussianBlur(float(min(40.0, 2.8 * scale))))
    base = Image.eval(plate.convert("RGB"), lambda v: int(v * 0.72)).convert("RGBA")       # the room dims a little behind the figure
    base = Image.alpha_composite(base, glow); base = Image.alpha_composite(base, glow)
    return Image.alpha_composite(base, layer).convert("RGB")


def weight(t):
    """How much of the figure is in the picture at t: 0 the robot, 1 the figure."""
    at, k = FLICKER_AT, 0.0
    for on, off in BURSTS:
        k = max(k, min(1.0, (t - at) / FADE, (at + on - t) / FADE + 1.0)) if at <= t < at + on + FADE else k
        at += on + off
    return max(0.0, min(1.0, max(k, (t - at) / FADE)))


def bell_wav(path, length):
    """One strike of a desk bell for every press, each a little different, ringing over the next."""
    sr = 48000; t = np.arange(int(sr * 2.2)) / sr; rng = np.random.default_rng(7)
    x = np.zeros(int(sr * length) + len(t))
    for ring in shot.get("rings", [RING_AT]):
        f0 = 2093 * (1 + rng.uniform(-0.004, 0.004))
        tone = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau) for f, a, tau in
                   ((f0, 1.0, 0.6), (f0 * 2.01, 0.25, 0.45), (f0 * 2.76, 0.5, 0.3), (f0 * 5.4, 0.3, 0.1), (f0 * 8.9, 0.15, 0.04)))
        tone *= np.minimum(1.0, t / 0.002) * rng.uniform(0.65, 1.0)
        a = int(sr * (ring + 0.02)); x[a:a + len(tone)] += tone
    x = x[:int(sr * length)]
    x[-int(sr * 0.4):] *= np.linspace(1, 0, int(sr * 0.4))
    x = (0.6 * x / np.abs(x).max() * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(x.tobytes())


n = len(shot["frames"])
plates = Path(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else None      # the same frames without the robot
plate = Image.open(frames_dir / ("frame_%04d.png" % round(PLATE_AT * FPS)))
work = frames_dir / "cut"; work.mkdir(exist_ok=True)
for i in range(n):
    src = frames_dir / ("frame_%04d.png" % i)
    k = weight(i / FPS)
    if k <= 0:
        img = Image.open(src).convert("RGB")
    else:
        own = plates / ("frame_%04d.png" % i) if plates else None
        img = Image.blend(Image.open(src).convert("RGB"), figure(i, Image.open(own) if own and own.exists() else plate), k)
    img.save(work / ("cut_%04d.png" % i), compress_level=1)
if "--check" in sys.argv:                      # the figure half over the robot, to see that they line up
    i = n - 1
    Image.blend(Image.open(frames_dir / ("frame_%04d.png" % i)).convert("RGB"), figure(i, Image.open(frames_dir / ("frame_%04d.png" % i))), 0.6).save(frames_dir / "check_alignment.jpg")
bell_wav(work / "bell.wav", n / FPS)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(work / "cut_%04d.png"), "-i", str(work / "bell.wav"),
                "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_path)], check=True)
print("wrote", out_path, n, "frames", round(n / FPS, 2), "s")
