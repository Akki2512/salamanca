"""The home scene: what is on the table and what each thing is called.

The simulator knows objects only by colour + shape. Here each colour + shape pair is one everyday item
(its real outline and parts are in home/home_items.py, used by our simulator copy), so the person can ask for "my pills" and everything downstream (grasp assist, the worker's page)
still finds the object by colour + shape. `home/scene.yaml` holds the table's geometry.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

SCENE_PATH = Path(os.environ.get("REMOTE_HANDS_SCENE", Path(__file__).resolve().parents[1] / "home" / "scene.yaml"))
DESTINATION = "person"          # every fetch ends with the item handed to the person in bed
# Where the item is handed to the person, in the home frame (the robot's own frame at its starting place by the
# bedside table: x forward, y left, z up from the pelvis, m). The person sits in a wheelchair by the radio, across the room:
# this is just in front of their lap. The robot has to walk there (unreal build_bedroom.py WHEELCHAIR).
WHEELCHAIR = (-4.90, 1.70, 19.0)        # x, y in the home frame, and the way it faces (degrees, towards the robot's start)
HANDOVER = (-4.50, 1.56, 0.10)        # 42 cm in front of the chair's middle: over his knees
LAP_Z = -0.06                           # the top of his knees, from the robot's pelvis height: what is let go there rests on it

# The things on the bedside table that can be asked for. They match the Unreal bedroom
# (unreal/RemoteHandsBedroom/Scripts/build_bedroom.py); colour + shape is only the internal key the simulator
# code needs. All three stand on the right half of the table: only the right hand can reach the bed.
# "aim" is the middle of the thing as seen (offset from "at", m) and "radius" its size, for picking by pointing.
# "at" is where the hand takes hold (robot frame x, y). "grip_z": that point's height above the table when it is
# not a small thing gripped near the table top: the jug is held by the top of its handle, which faces the robot.
ITEMS = (
    {"name": "water jug", "color": "blue", "shape": "cylinder", "looks": "a white jug with flowers painted on it",
     "words": ("jug", "pitcher"), "at": (0.30, -0.22), "grip_z": 0.155,
     "aim": (0.12, 0.0, -0.06), "radius": 0.11},
    {"name": "alarm clock", "color": "green", "shape": "cylinder", "looks": "a small green alarm clock with two bells",
     "words": ("clock", "alarm", "time"), "at": (0.43, -0.04), "aim": (0.0, 0.0, 0.065), "radius": 0.08},
    {"name": "glass of water", "color": "white", "shape": "cylinder", "looks": "a glass of water",
     "words": ("glass", "water", "drink", "thirsty"), "at": (0.29, 0.02), "grip_z": 0.055, "aim": (0.0, 0.0, 0.0), "radius": 0.07},
    # Not on the table: a sweater lying on the floor out in the room ("room": x, y in the home frame). It is
    # picked from where the robot stands (remote_hands/scene.py), not by reaching down.
    {"name": "sweater", "color": "red", "shape": "cube", "looks": "a dark red sweater lying on the floor",
     "words": ("sweater", "jumper", "pullover"), "at": (0.45, 0.25), "room": (-0.55, 1.35), "grip_z": 0.07,
     "aim": (0.0, 0.0, 0.0), "radius": 0.30},
)
NAMES = tuple(i["name"] for i in ITEMS)
BY_NAME = {i["name"]: i for i in ITEMS}
BY_LOOK = {(i["color"], i["shape"]): i for i in ITEMS}

# The same positions in the scene file's table coordinates (x = robot y + 0.45, y = robot x - 0.14).
LAYOUT = tuple({"color": i["color"], "shape": i["shape"], "x": round(i["at"][1] + 0.45, 3),
                "y": round(i["at"][0] - 0.14, 3), "yaw": 0.0} for i in ITEMS)


def load_scene():
    from aaltonomous.scene.scene import Scene
    return Scene.load(SCENE_PATH)


def layout():
    return [dict(o) for o in LAYOUT]


def name_of(color, shape):
    """Everyday name of a colour + shape pair; falls back to "<colour> <shape>" for anything not in ITEMS."""
    i = BY_LOOK.get((color, shape))
    return i["name"] if i else f"{color} {shape}"


def match_words(text):
    """Items a sentence names, by plain word matching (the item's name, its other words, or its colour).
    Used when Nemotron does not answer in time. Returns a list of ITEMS entries, best first, no duplicates."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    hits = []
    for i in ITEMS:
        named = set(i["name"].split()) <= words or any(w in words for w in i["words"])
        if named or i["color"] in words:
            hits.append((0 if named else 1, i))
    if any(rank == 0 for rank, _ in hits):          # a name beats a bare colour ("the red can" -> soda can)
        hits = [h for h in hits if h[0] == 0]
    return [i for _, i in hits]


def home_renderer(scene, cal):
    """The mock robot's head-camera picture for this scene: wooden table, no printed grid."""
    from aaltonomous.baselines.classical import mock_world as mw

    class HomeRenderer(mw.Renderer):
        def _background(self):
            keep = mw.MAT_RGB
            mw.MAT_RGB = (0.0, 0.0, 0.0)            # the base class adds its 5 cm checker on top of this
            try:
                rgb, zb = super()._background()
            finally:
                mw.MAT_RGB = keep
            sc = self.scene
            t = self._plane(sc.counter_z)
            on = np.isfinite(t) & (np.abs(zb - t) < 1e-9)       # pixels where the table top is what is seen
            p = self.o + np.where(np.isfinite(t), t, 0.0)[..., None] * self.rays
            plank = np.floor(p[..., 1] / 0.11)                   # boards running away from the robot
            shade = 1.0 + 0.06 * np.sin(plank * 12.9898) + 0.025 * np.sin(p[..., 0] * 90 + plank * 5.0)
            seam = np.abs(p[..., 1] / 0.11 - np.round(p[..., 1] / 0.11)) < 0.012
            wood = np.array([176.0, 128.0, 84.0]) * shade[..., None]
            wood[seam] *= 0.72
            rgb[on] = wood[on]
            floor = np.isfinite(zb) & ~on & (np.abs(zb - self._plane(mw.FLOOR_Z)) < 1e-9)
            rgb[floor] = (112.0, 104.0, 96.0)                    # a warm floor instead of lab grey
            return rgb, zb

    return HomeRenderer(scene, cal)
