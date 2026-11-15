"""Boxes around the items in the robot's head-camera frames, worked out from the recorded step data (no vision).

A recorded episode has, per control tick, where the robot's head camera is ("eye"), where the robot stands
("state"."base") and where every item is ("items"), and per saved frame the head's direction and the lens
("head": [yaw_deg, pitch_deg, fov_deg, aspect]). That is enough to say where each item falls in the picture.

Conventions (all in the room frame: x forward and y left of the robot's starting pose, z up from the floor, m):
    camera position   step["eye"] of the step whose tick is nearest the frame's tick
    heading           base yaw (rad, anticlockwise) + head yaw (deg, to the LEFT of the way the robot faces)
    pitch             head pitch (deg), positive looks UP; no roll
    lens              pinhole, square pixels, fov_deg is the HORIZONTAL field of view of the full image width:
                      focal length in pixels f = (width / 2) / tan(fov / 2), principal point at the image middle
    pixels            u to the right, v DOWN, (0, 0) the top left corner
    depth             distance along the camera's forward axis (not the straight-line distance)

An item is taken as a ball of home.ITEMS "radius" around pos + "aim" (pos is where the hand takes hold of it).
Its box is the square of half-size f * radius / depth around the projected middle: exact for a ball on the
camera's axis, a little too small towards the edges of a wide lens.

The sweater is not a ball:
    lying      a flat disc on whatever it lies on (floor or bed), radius = its "radius", 0.10 m thick, around pos (x, y)
    held       hanging under the hand: an upright roll from 0.05 m above pos down to 0.70 m under it (not below
               the floor), 0.18 m radius
Both are boxed by projecting points on the rims of those shapes and taking what encloses them.

The man in the wheelchair: the fixed step format has no chair position. If a step does carry
step["scene"]["chair"] = [x, y, heading] (what DemoScene.state() gives) he is boxed as an upright roll at the
chair (name "man in wheelchair", or "wheelchair" once the scene says he is in bed); otherwise he is left out.

LIMITS: this knows nothing about what is in the way. A thing behind furniture, behind the robot's own arm or
inside its hand still gets a full box. A thing whose middle is behind the camera gets none. "aim" is added in
room axes, so for a held thing that has been turned the middle is off by up to |aim| (12 cm for the jug).
Frames with "head": null come from a camera whose pose was not recorded: no boxes, "labelled": false.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import home

NEAR = 0.05                     # m: anything closer than this along the view axis counts as behind the camera
SWEATER_THICK = 0.10            # m: how high the lying sweater stands off the floor
SWEATER_HANG = 0.70             # m: how far it hangs under the hand
SWEATER_HANG_R = 0.18           # m: half its width while hanging
MAN_R, MAN_TOP, CHAIR_TOP = 0.40, 1.30, 0.95   # m: the seated man with his chair, as an upright roll from the floor
DEFAULT_HEIGHT = 1024           # pixels, used with the frame's aspect when the picture itself cannot be read


def camera(step, frame_meta):
    """The camera of one frame: {"eye", "forward", "left", "up", "fov" (rad, horizontal), "aspect"}, or None when
    the frame has no recorded head direction (or the step has no eye)."""
    head = (frame_meta or {}).get("head")
    eye = (step or {}).get("eye")
    if not head or eye is None:
        return None
    yaw_deg, pitch_deg, fov_deg, aspect = (float(v) for v in head[:4])
    h = float(step["state"]["base"][2]) + math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    ch, sh, cp, sp = math.cos(h), math.sin(h), math.cos(p), math.sin(p)
    return {"eye": np.array(eye, float),
            "forward": np.array([ch * cp, sh * cp, sp]),
            "left": np.array([-sh, ch, 0.0]),
            "up": np.array([-ch * sp, -sh * sp, cp]),
            "fov": math.radians(fov_deg), "aspect": aspect}


def focal(cam, width):
    """Focal length in pixels (the same across and down: square pixels)."""
    return 0.5 * width / math.tan(0.5 * cam["fov"])


def project(point_room, cam, width, height):
    """A room point in the picture: (u, v, depth), u right and v down in pixels, depth along the view axis (m).
    None if it is behind the camera. The point may fall outside the picture."""
    d = np.asarray(point_room, float) - cam["eye"]
    depth = float(d @ cam["forward"])
    if depth < NEAR:
        return None
    f = focal(cam, width)
    return (0.5 * width - f * float(d @ cam["left"]) / depth, 0.5 * height - f * float(d @ cam["up"]) / depth, depth)


def _ring(centre, radius, z, n=16):
    a = np.arange(n) * (2 * math.pi / n)
    return np.stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a), np.full(n, z)], axis=1)


def _roll(centre_xy, radius, z_low, z_high):
    """Points on the top and bottom rims of an upright roll."""
    return np.concatenate([_ring(centre_xy, radius, z_low), _ring(centre_xy, radius, z_high)])


def _clip(name, raw, centre, depth, held_by, width, height):
    """Cut an unclipped box [x0, y0, x1, y1] to the picture; None if nothing of it is inside."""
    x0, y0, x1, y1 = (float(v) for v in raw)
    cx0, cy0, cx1, cy1 = max(x0, 0.0), max(y0, 0.0), min(x1, float(width)), min(y1, float(height))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    area = (x1 - x0) * (y1 - y0)
    return {"name": name, "box": [round(cx0, 1), round(cy0, 1), round(cx1, 1), round(cy1, 1)],
            "center": [round(centre[0], 1), round(centre[1], 1)], "depth": round(depth, 3),
            "visible": round((cx1 - cx0) * (cy1 - cy0) / area, 3) if area > 0 else 0.0, "held_by": held_by}


def _ball_box(name, centre, radius, held_by, cam, width, height):
    c = project(centre, cam, width, height)
    if c is None:
        return None
    half = focal(cam, width) * radius / c[2]
    return _clip(name, (c[0] - half, c[1] - half, c[0] + half, c[1] + half), c, c[2], held_by, width, height)


def _points_box(name, centre, points, held_by, cam, width, height):
    """Box around a handful of room points; the points behind the camera are dropped."""
    c = project(centre, cam, width, height)
    if c is None:
        return None
    uv = [q[:2] for q in (project(p, cam, width, height) for p in points) if q is not None]
    if not uv:
        return None
    uv = np.array(uv)
    return _clip(name, (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()), c, c[2], held_by, width, height)


def _item_box(item, cam, width, height):
    name, pos, held = item["name"], np.array(item["pos"], float), item.get("held_by")
    info = home.BY_NAME.get(name)
    if info is None:                                # not in the table of items: no size known
        return None
    if name == "sweater":
        if held:
            low, high = max(pos[2] - SWEATER_HANG, 0.0), pos[2] + 0.05
            mid = (pos[0], pos[1], 0.5 * (low + high))
            return _points_box(name, mid, _roll(pos[:2], SWEATER_HANG_R, low, high), held, cam, width, height)
        base = max(float(pos[2]) - 0.07, 0.0)        # what it lies on: the floor, or the bed (pos is 7 cm above its underside)
        mid = (pos[0], pos[1], base + 0.5 * SWEATER_THICK)
        return _points_box(name, mid, _roll(pos[:2], info["radius"], base, base + SWEATER_THICK), held, cam, width, height)
    return _ball_box(name, pos + np.array(info["aim"], float), info["radius"], held, cam, width, height)


def _man_box(step, cam, width, height):
    sc = step.get("scene") or {}
    chair = sc.get("chair")
    if not chair or len(chair) < 2:
        return None
    in_bed = sc.get("man") == "bed" or sc.get("phase") == "bed"
    top = CHAIR_TOP if in_bed else MAN_TOP
    return _points_box("wheelchair" if in_bed else "man in wheelchair", (chair[0], chair[1], 0.5 * top),
                       _roll(chair[:2], MAN_R, 0.0, top), None, cam, width, height)


def boxes(step, frame_meta, width, height):
    """Boxes of everything in one frame: a list of {"name", "box": [x0, y0, x1, y1] (pixels, cut to the picture),
    "center": [u, v] (may be outside the picture), "depth" (m along the view axis), "visible" (the share of the
    uncut box that is inside the picture), "held_by"}. Things wholly outside the picture or behind the camera are
    left out. Empty for a frame without a head direction (see `camera`). Nothing here knows what hides what:
    a thing behind furniture still gets its box."""
    cam = camera(step, frame_meta)
    if cam is None:
        return []
    out = [_item_box(it, cam, width, height) for it in step.get("items") or []]
    out.append(_man_box(step, cam, width, height))
    return [b for b in out if b is not None]


def _read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def frame_size(path, meta):
    """(width, height) of a saved frame: from the file when Pillow can open it, else from the frame's aspect."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        head = (meta or {}).get("head")
        aspect = float(head[3]) if head else 1.25
        return int(round(DEFAULT_HEIGHT * aspect)), DEFAULT_HEIGHT


def frames(episode_dir):
    """Every robot_eyes frame of an episode with its step: yields (meta, step, jpg path). The step is the one
    whose tick is nearest the frame's tick."""
    ep = Path(episode_dir)
    steps = _read_jsonl(ep / "steps.jsonl")
    ticks = np.array([s["tick"] for s in steps], float)
    for meta in _read_jsonl(ep / "frames" / "robot_eyes" / "index.jsonl"):
        step = steps[int(np.argmin(np.abs(ticks - meta["tick"])))] if len(steps) else None
        yield meta, step, ep / "frames" / "robot_eyes" / ("%06d.jpg" % meta["k"])


def label_episode(episode_dir):
    """Write <episode>/labels/robot_eyes.jsonl, one line per frame: {"k", "labelled", "width", "height", "boxes"}.
    Returns {"frames", "labelled", "boxes"} counts. Occlusion is not known (see the top of this file)."""
    ep = Path(episode_dir)
    (ep / "labels").mkdir(exist_ok=True)
    n = {"frames": 0, "labelled": 0, "boxes": 0}
    with open(ep / "labels" / "robot_eyes.jsonl", "w") as out:
        for meta, step, jpg in frames(ep):
            w, h = frame_size(jpg, meta)
            ok = step is not None and camera(step, meta) is not None
            bx = boxes(step, meta, w, h) if ok else []
            out.write(json.dumps({"k": meta["k"], "labelled": ok, "width": w, "height": h, "boxes": bx}) + "\n")
            n["frames"] += 1
            n["labelled"] += int(ok)
            n["boxes"] += len(bx)
    return n


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:]:
        print(d, label_episode(d))
