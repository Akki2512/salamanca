"""Grasp assist: finishes the last centimetres of a pick and the release, for a helper without gloves.

The suit gives wrist positions that are roughly right, not exact, and no fingers. Given the task's object
(colour + shape from Nemotron) and where the objects are (scene service on the sim, the mock world on a laptop),
one hand goes through:

    free       the helper drives. A hand that comes within `capture_r` (horizontally) of the object, low enough,
               is captured.
    approach   the hand is pulled to a point straight above the object, fingers in the wide "preshape" claw.
    descend    straight down to the grasp height.
    closing    arm frozen, hand closes.                                     -> event "grasped"
    holding    the helper drives again, hand stays closed. Near the destination bin's drop point the hand is
               pulled onto it, at a height that clears the rim.
    releasing  arm frozen, hand opens.                                      -> event "released"
    done       nothing more for this request.

The helper stays in charge: pulling the hand away during approach / descend lets go of it (back to free), and
with the clutch open nothing moves. If the object is not in the hand a moment after lifting, the state goes back
to free (event "dropped") so the helper can try again.

Pure logic, no I/O: update() is called once per control tick with times in robot (sim) seconds.
"""
from __future__ import annotations

import numpy as np

from .messages import SIDES


def drop_point(scene, bin_name, inset=0.05, above_rim=0.04):
    """Where the grasp point should be to let go into a bin (pelvis frame): the point of the bin's inside
    nearest the robot, `inset` from the walls (as the Aalto baseline does; side-bin centres are out of reach)."""
    lo, hi = scene.bin_inner_box(bin_name)
    a, b = scene.mat_to_pelvis(lo), scene.mat_to_pelvis(hi)
    lo_p, hi_p = np.minimum(a, b), np.maximum(a, b)
    xy = np.clip([0.0, 0.0], lo_p[:2] + inset, hi_p[:2] - inset)
    z = hi_p[2] + scene.object_height() / 2 + above_rim
    return np.array([xy[0], xy[1], z])


class GraspAssist:
    def __init__(self, scene, reach_min, reach_max, grasp_z, approach_height=0.08, capture_r=0.11,
                 capture_h=0.17, escape_r=0.19, bin_r=0.10, close_s=0.6, release_s=0.7):
        self.scene, self.reach_min, self.reach_max = scene, reach_min, reach_max
        self.grasp_z = grasp_z                  # grasp-point height for an object standing on the counter
        self.approach_height = approach_height
        self.capture_r, self.capture_h, self.escape_r, self.bin_r = capture_r, capture_h, escape_r, bin_r
        self.close_s, self.release_s = close_s, release_s
        self.task = None
        self._reset()

    def _reset(self):
        self.state, self.side, self.obj = "free", None, None
        self._frozen, self._t0, self._dwell, self._last = None, 0.0, None, None
        self.drop = None
        self._grab, self._forced = None, False     # a pick asked for outright (controller trigger), and one under way

    def set_task(self, task):
        """task: {"request_id", "color", "shape", "destination"} or None."""
        self.task = task
        self._reset()
        if task:
            name = str(task.get("destination") or "person").split()[0].lower()
            if name == "hold":                      # pick it up and keep holding it: never let go
                self.drop = None
            elif name == "person":                  # hand it to the person in bed
                from . import home
                self.drop = np.array(home.HANDOVER, float)
            elif name in self.scene.bin_names:
                self.drop = drop_point(self.scene, name)

    def grab(self, side="right"):
        """Pick the task's item now, wherever the worker's hand is: the hand is taken to it, lowered and closed.
        Prefers `side`; uses the other hand if only that one can reach. Nothing the worker does cancels it."""
        self._grab = side

    def let_go(self, t, hand):
        """Open the hand that is holding the item, where it is. hand: that hand's current position."""
        if self.state == "holding":
            self.state, self._t0, self._frozen = "releasing", t, np.asarray(hand, float).copy()
            return True
        return False

    # ------------------------------------------------------------------ helpers
    def _matches(self, objects):
        t = self.task
        return [o for o in objects
                if (not t.get("color") or o["color"] == t["color"]) and (not t.get("shape") or o["shape"] == t["shape"])]

    def _reachable(self, side, p):
        return bool(np.all(p >= self.reach_min[side] - 1e-6) and np.all(p <= self.reach_max[side] + 1e-6))

    def _sides_for(self, obj_p):
        """Hands that may take the object: those that can also reach the drop point, if any can."""
        g = np.array([obj_p[0], obj_p[1], max(self.grasp_z, float(obj_p[2]))])
        can = [s for s in SIDES if self._reachable(s, g)]
        both = [s for s in can if self.drop is not None and self._reachable(s, self.drop)]
        return both or can

    def info(self):
        return {"state": self.state, "side": self.side,
                "object": None if self.obj is None else np.round(self.obj["pos"], 3).tolist(),
                "drop": None if self.drop is None else np.round(self.drop, 3).tolist()}

    # ------------------------------------------------------------------ one tick
    def update(self, t, targets, hands, objects):
        """targets: {side: helper's (clamped) hand target} or None while not driving. hands: {side: the robot's
        commanded grasp point}. objects: [{"color", "shape", "pos"}] or None when unknown.
        Returns (targets, {side: posture name}, [event names])."""
        if not self.task or self.state == "done":
            return targets, {}, []
        events, s = [], self.side
        cand = self._matches(objects or [])
        if self.state in ("free", "approach", "descend"):
            if cand:                                    # follow the object: it may have been nudged
                ref = (targets or hands)
                key = (lambda o: np.linalg.norm(np.asarray(o["pos"])[:2] - ref[s][:2])) if s else \
                      (lambda o: min(np.linalg.norm(np.asarray(o["pos"])[:2] - ref[x][:2]) for x in SIDES))
                self.obj = dict(min(cand, key=key))
                self.obj["pos"] = np.asarray(self.obj["pos"], float)
            elif self.state == "free":
                self.obj = None
        elif self.obj is not None and "name" in self.obj:   # holding: keep watching the same object
            for o in objects or []:
                if o.get("name") == self.obj["name"]:
                    self.obj["pos"] = np.asarray(o["pos"], float)
        if self.obj is None:
            return targets, {}, []
        op = self.obj["pos"]
        # Where to take hold: the item's own grip point when it is above table-grasp height (a jug's handle),
        # else the usual height for a small thing standing on the table.
        gz = max(self.grasp_z, float(op[2]))
        # A raised grip point is near where the hands rest: there the worker has to come down to within 5 cm
        # above it, or the robot would take over the moment the clutch goes on.
        ch = self.capture_h if gz <= self.grasp_z + 1e-6 else 0.05
        above = np.array([op[0], op[1], gz + self.approach_height])
        grasp = np.array([op[0], op[1], gz])

        if self.state == "free" and self._grab:
            can = [x for x in (self._grab, *[x for x in SIDES if x != self._grab]) if self._reachable(x, grasp)]
            self._grab = None
            if can:
                self.state, self.side, s, self._forced = "approach", can[0], can[0], True
            else:
                events.append("unreachable")            # too far from the robot: walk closer first
        if self.state == "free":
            if targets is not None:
                for x in self._sides_for(op):
                    d = np.linalg.norm(targets[x][:2] - op[:2])
                    if d < self.capture_r and targets[x][2] < gz + ch:
                        self.state, self.side, s = "approach", x, x
                        break
            if self.state == "free":
                return targets, {}, []

        out = None if targets is None else dict(targets)
        post = {}
        if self.state in ("approach", "descend"):
            post[s] = "preshape"
            if targets is not None:
                d = np.linalg.norm(targets[s][:2] - op[:2])
                if not self._forced and (d > self.escape_r or targets[s][2] > gz + 2 * self.capture_h):
                    self._reset_to_free()               # the helper pulled away
                    return targets, {}, events
            if self.state == "approach":
                goal = above
                if np.linalg.norm(hands[s] - np.clip(above, self.reach_min[s], self.reach_max[s])) < 0.012:
                    self.state, self._t0, self._last = "descend", t, None
            if self.state == "descend":
                goal = grasp
                e = hands[s] - grasp
                near = np.linalg.norm(e[:2]) < 0.008 and abs(e[2]) < 0.008
                moved = self._last is None or np.linalg.norm(hands[s] - self._last[0]) > 5e-4
                if moved:
                    self._last = (hands[s].copy(), t)
                # resting: the table guard keeps the fingertips off the counter, so the hand may stop a
                # little above the grasp height. It has stopped there for 0.4 s: that is as low as it goes.
                stuck = t - self._last[1] > 0.4 and np.linalg.norm(e[:2]) < 0.008 and -0.012 < e[2] < 0.03
                if near or stuck:
                    self.state, self._t0, self._frozen = "closing", t, hands[s].copy()
            if self.state != "closing" and out is not None:
                out[s] = np.clip(goal, self.reach_min[s], self.reach_max[s])
        if self.state == "closing":
            post[s] = "closed"
            if out is not None:
                out[s] = self._frozen
            if t - self._t0 >= self.close_s:
                self.state, self._t0, self._dwell = "holding", t, None
                events.append("grasped")
        elif self.state == "holding":
            post[s] = "closed"
            lifted = hands[s][2] > gz + 0.05
            if objects is not None and lifted and t - self._t0 > 1.0 and np.linalg.norm(op - hands[s]) > 0.07:
                self._reset_to_free()                   # it is not in the hand
                return targets, {}, ["dropped"]
            if self.drop is not None and self._reachable(s, self.drop):
                if out is not None and np.linalg.norm(targets[s][:2] - self.drop[:2]) < self.bin_r:
                    out[s] = np.array([self.drop[0], self.drop[1], max(targets[s][2], self.drop[2])])
                over = np.linalg.norm(hands[s][:2] - self.drop[:2]) < 0.015 and hands[s][2] > self.drop[2] - 0.01
                self._dwell = (self._dwell if self._dwell is not None else t) if over else None
                if over and t - self._dwell >= 0.3:
                    self.state, self._t0, self._frozen = "releasing", t, hands[s].copy()
        if self.state == "releasing":
            post[s] = "release"
            if out is not None:
                out[s] = self._frozen
            if t - self._t0 >= self.release_s:
                self.state = "done"
                events.append("released")
                post[s] = "open"
        return out, post, events

    def _reset_to_free(self):
        self.state, self.side, self._frozen, self._dwell, self._forced = "free", None, None, None, False
