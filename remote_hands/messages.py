"""The wrist-pose message: the only thing the mocap bridge sends to the robot node (spec 4.3 + architecture 7).

Frame contract (the bridge's job to meet, whatever the mocap source is):
  positions are metres, relative to the helper's chest, x forward, y to the helper's left, z up.
Only differences of positions are used downstream (clutched mapping), so a constant offset does not matter;
the axes do.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

SIDES = ("left", "right")


@dataclass
class WristPose:
    seq: int
    t: float                      # source time, seconds
    left: tuple                   # (x, y, z) left wrist
    right: tuple                  # (x, y, z) right wrist
    grip: tuple = (0.0, 0.0)      # (left, right), 0 = open .. 1 = closed
    clutch: bool = True           # False = helper is not driving (resting, calibrating): robot holds
    source: str = "synthetic"     # rokoko | replay | synthetic
    body: dict | None = None      # hips and legs for walking (remote_hands/walk.py Walker.update), when the source has them

    def to_json(self) -> str:
        return json.dumps({"type": "pose", "seq": self.seq, "t": self.t, "clutch": self.clutch,
                           "source": self.source,
                           "left_wrist": {"p": list(self.left)}, "right_wrist": {"p": list(self.right)},
                           "grip": {"left": self.grip[0], "right": self.grip[1]},
                           **({"body": self.body} if self.body else {})})

    @classmethod
    def from_dict(cls, d: dict) -> "WristPose":
        g = d.get("grip") or {}
        return cls(seq=int(d.get("seq", 0)), t=float(d.get("t", 0.0)),
                   left=tuple(float(v) for v in d["left_wrist"]["p"]),
                   right=tuple(float(v) for v in d["right_wrist"]["p"]),
                   grip=(float(g.get("left", 0.0)), float(g.get("right", 0.0))),
                   clutch=bool(d.get("clutch", True)), source=str(d.get("source", "unknown")), body=d.get("body"))

    def p(self, side: str) -> tuple:
        return self.left if side == "left" else self.right

    def g(self, side: str) -> float:
        return self.grip[0] if side == "left" else self.grip[1]
