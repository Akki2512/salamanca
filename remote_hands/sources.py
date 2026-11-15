"""Mocap sources for the bridge. Each yields WristPose at its own rate; the bridge forwards them.

  SyntheticSource -- a scripted human: reaches, holds, grips, rests. Moves at human speed (well above what
                     the robot is allowed), so it exercises the rate limiting. No hardware needed.
  rokoko          -- the Smartsuit Pro II through the Rokoko Device SDK driver (rokoko.py).
"""
from __future__ import annotations

import numpy as np

from .messages import WristPose

# Chest-relative rest positions of a person standing with forearms raised, hands in front (m).
REST = {"left": np.array([0.30, 0.20, -0.15]), "right": np.array([0.30, -0.20, -0.15])}

# (duration s, right-hand offset from rest, right grip, clutch). Offsets are human-scale; at the default
# scale 0.75 from the robot's ready pose they land on an object in the far-right corner of the object band,
# then on the near part of the right bin (positions from the Aalto scene.yaml).
OBJ = (0.063, 0.057, -0.268)        # -> robot (0.33, -0.17, 0.035): grasp height of a 45 mm object
LIFT = (0.063, 0.057, -0.088)       # -> robot z 0.17: carry height, clears the bin walls
BIN = (0.223, -0.063, -0.088)       # -> robot (0.45, -0.26, 0.17): release point in the right bin
SCRIPT = [
    (1.0, (0, 0, 0), 0.0, False),   # not driving yet (calibration pose)
    (1.0, (0, 0, 0), 0.0, True),    # engage
    (0.6, OBJ, 0.0, True),          # reach down to the object at human speed (~0.5 m/s)
    (2.0, OBJ, 0.0, True),          # wait for the robot to arrive
    (1.0, OBJ, 1.0, True),          # close the hand
    (0.4, LIFT, 1.0, True),         # lift
    (0.5, BIN, 1.0, True),          # carry to the bin
    (2.0, BIN, 1.0, True),          # wait for the robot
    (1.0, BIN, 0.0, True),          # release
    (0.6, (0, 0, 0), 0.0, True),    # back to rest
    (2.0, (0, 0, 0), 0.0, True),
]


class SyntheticSource:
    name = "synthetic"

    def __init__(self, hz=30.0, loop=True, script=SCRIPT):
        self.hz, self.loop, self.script = hz, loop, script
        self.period = sum(s[0] for s in script)

    def at(self, t: float, seq: int = 0) -> WristPose:
        """Pose at source time t. Positions move linearly between script waypoints."""
        tt = t % self.period if self.loop else min(t, self.period - 1e-9)
        prev = np.zeros(3)
        for dur, off, grip, clutch in self.script:
            if tt < dur:
                a = tt / dur
                off_now = (1 - a) * prev + a * np.asarray(off, float)
                left = REST["left"] + np.array([0.0, 0.0, 0.01 * np.sin(2 * np.pi * 0.3 * t)])  # idle sway
                return WristPose(seq=seq, t=t, left=tuple(left), right=tuple(REST["right"] + off_now),
                                 grip=(0.0, grip), clutch=clutch, source=self.name)
            tt -= dur
            prev = np.asarray(off, float)
        raise AssertionError("unreachable")

    def __iter__(self):
        n = int(self.period * self.hz) if not self.loop else None
        k = 0
        while n is None or k < n:
            yield self.at(k / self.hz, k)
            k += 1


class ReplaySource:
    """A suit recording played back as if it were live (tools/record_studio.py makes the .rkkudp file).
    The wrist positions are cached next to it as NAME_wrists.npz, so later starts are instant.
    start / end: seconds into the recording. The clip loops; at the loop point the clutch opens for half a
    second so the robot re-ties its hands to the clip's start instead of jumping back."""
    name = "replay"

    def __init__(self, path, start=0.0, end=None, loop=True, speed=1.0, hz=30.0, **_):
        from pathlib import Path
        p = Path(path)
        cache = p if p.suffix == ".npz" else p.with_name(p.stem + "_wrists.npz")
        if not cache.exists():
            self._convert(p, cache)
        d = np.load(cache)
        keep = (d["t"] >= start) & (d["t"] <= (end if end is not None else d["t"][-1]))
        if keep.sum() < 2:
            raise ValueError(f"nothing between {start} s and {end} s in {cache} (it is {d['t'][-1]:.0f} s long)")
        self.t = d["t"][keep] - d["t"][keep][0]
        self.w = {"left": d["left"][keep], "right": d["right"][keep]}
        self.hz, self.loop, self.speed, self.period = hz, loop, speed, float(self.t[-1])

    @staticmethod
    def _convert(src, cache):
        import json, struct
        from .rokoko import RokokoStudioSource
        T, L, Rr = [], [], []
        with open(src, "rb") as f:
            while True:
                h = f.read(12)
                if len(h) < 12:
                    break
                t, n = struct.unpack("<dI", h)
                data = f.read(n)
                try:
                    if data[:4] == b"\x04\x22\x4d\x18":
                        import lz4.frame
                        data = lz4.frame.decompress(data)
                    w = RokokoStudioSource.wrists_from_frame(json.loads(data))
                except (ValueError, KeyError, IndexError, RuntimeError):
                    continue
                T.append(t); L.append(w["left"]); Rr.append(w["right"])
        np.savez_compressed(cache, t=np.array(T), left=np.array(L), right=np.array(Rr))

    def at(self, t: float, seq: int = 0) -> WristPose:
        tt = t * self.speed
        gap = 0.5
        if self.loop:
            tt = tt % (self.period + gap)
        live = tt <= self.period
        tt = min(tt, self.period)
        p = {s: tuple(float(np.interp(tt, self.t, self.w[s][:, k])) for k in range(3)) for s in ("left", "right")}
        return WristPose(seq=seq, t=t, left=p["left"], right=p["right"], grip=(0.0, 0.0), clutch=live,
                         source=self.name)


def _rokoko(**kw):
    from .rokoko import RokokoSource     # needs the Rokoko driver; imported only when asked for
    return RokokoSource(auto_calibrate=False, **kw)    # calibration starts when the helper is ready


def _studio(**kw):
    from .rokoko import RokokoStudioSource
    return RokokoStudioSource(**kw)


SOURCES = {"synthetic": SyntheticSource, "rokoko": _rokoko, "studio": _studio, "replay": ReplaySource}
