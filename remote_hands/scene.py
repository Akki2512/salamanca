"""The demo scene's story, kept by the robot node (Unreal and the headset page only show it).

    tidy   the robot (the worker in the suit) picks a sweater up off the floor: a household chore
    cough  a few seconds after the sweater is picked up the old man starts coughing and ringing his bell
    drink  the robot brings the glass of water and lets go of it at his lap: he drinks, the coughing stops
    bed    the robot wheels him to the bed (trigger from behind the chair = take hold / let go of it); let go
           beside the bed, he lies down in it

What the controller's trigger does (robot_node._trigger asks here first):
    pushing the chair -> let go of it
    pointing at a thing on the floor up to 1.8 m away (the sweater) -> the robot steps up to it, squats and bows
        until its right hand reaches it, closes the hand on it (grasp assist's own pick) and stands up again
    pointing at nothing, within 1.3 m of the glass on the table -> the robot steps to the table and picks the glass up
    holding the glass within 1.5 m of the man -> he has it (wherever the hand is)
    pointing at nothing, standing behind the wheelchair -> both hands take the chair's handles and it is pushed
Everything else (things on the table, letting go of what is held) stays with grasp assist.

Frames: the room ("home") frame of walk.py: x forward and y left of the robot's starting pose, metres; angles
anticlockwise in radians. The chair's pose and the hand-over point are published into home.HANDOVER and
walk.CHAIR every tick, so the rest of the node follows the chair when it moves.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from . import home, walk

log = logging.getLogger("remote_hands.scene")

COUGH_AFTER_S = 3.0             # from picking the sweater up to the first cough
FIT_S = 3.4                     # one fit: a cough at 0 s, three rings of the bell from 1.5 s
RINGS = (1.5, 2.0, 2.5)
HAND_UP_S = 1.2                 # his hand is at his mouth for this long after each cough
PUSH_GAP = 0.72                 # chair's middle ahead of the robot's pelvis while it is pushed (m)
BED = (-0.19, -1.15)            # middle of the bed
BED_NEAR = 1.7                  # let go of the chair this close to it and he goes to bed
DRINK_AT = (0.183, 0.018, 0.514)  # the glass at his mouth, in the chair's frame (forward, left, up from the pelvis height)
SEAT_AT = (0.10, 0.0, -0.10)    # where the glass is left once he is in bed: on the seat
HANDLES = {"left": (-0.30, 0.27, 0.20), "right": (-0.30, -0.27, 0.20)}    # the chair's push handles, in the chair's frame
TABLE_STANCE = (0.29, 0.02)     # where a thing on the bedside table has to be for the right hand, the robot facing the table
GLASS_NEAR = 1.3                # trigger this close to the glass (m, pelvis to glass) picks it up without pointing
GIVE_NEAR = 2.2                 # trigger this close to the man while holding the glass gives it to him
CLOTH_LEN = 0.35                # the hanging sweater swings like a pendulum this long (m)
STANCE = (0.22, -0.16)          # where a thing on the floor has to be, ahead and to the right of the pelvis, to be reached


class DemoScene:
    def __init__(self, node):
        self.node = node
        self.reset()

    def reset(self):
        x, y, face = home.WHEELCHAIR
        self.chair = np.array([x, y, -math.radians(face)])     # x, y, heading
        self.phase = "tidy"
        self.pushing = False
        self._push_t = 0.0
        self.reach = None           # {"stage", "i", "name", "t"}: picking a thing up off the floor (or, "table": off the table)
        self._given = False         # the glass was let go near him: he has it
        # The sweater in the hand hangs and swings: a damped pendulum under the hand, pushed by the hand's own
        # acceleration (it trails when the robot sets off, swings on when it stops). Angles in radians: how far
        # the hanging end is out along the room's x and y.
        self.swing, self._swing_v = np.zeros(2), np.zeros(2)
        self._pivot = None          # (the hand's place, its velocity) last tick
        self.node.walker.bend_goal = 0.0
        self.cough = 0              # counts coughs; the page plays a cough for every step
        self.bell = 0               # counts rings
        self._cough_from = None     # when the coughing starts / started
        self._fit = -1
        self._rings = 0
        self._last_cough = -100.0
        self._bed_at = None
        self._publish()

    # ------------------------------------------------------------------ helpers
    def _objects(self):
        return getattr(self.node.env.backend, "objects", None) or []

    def _index(self, name):
        for i, o in enumerate(self._objects()):
            if home.name_of(o["color"], o["shape"]) == name:
                return i
        return None

    def _held(self, i):
        held = getattr(self.node.env.backend, "held", None) or {}
        return any(h is not None and h[0] == i for h in held.values())

    def _in_chair(self, p):
        c, s = math.cos(self.chair[2]), math.sin(self.chair[2])
        return np.array([self.chair[0] + c * p[0] - s * p[1], self.chair[1] + s * p[0] + c * p[1], p[2]])

    def _publish(self):
        f = self._in_chair((0.42, 0.0, 0.10))
        home.HANDOVER = (float(f[0]), float(f[1]), 0.10)
        walk.CHAIR = (float(self.chair[0]), float(self.chair[1]), 0.0 if self.pushing else 0.80)

    def _behind_chair(self):
        b = self.node.walker.base
        d = b[:2] - self.chair[:2]
        dist = float(np.linalg.norm(d))
        fwd = np.array([math.cos(self.chair[2]), math.sin(self.chair[2])])
        return 0.3 < dist < 1.5 and float(d @ fwd) < -0.3 * dist

    # ------------------------------------------------------------------ the trigger
    def trigger(self):
        """True if the trigger was used up here."""
        n = self.node
        if self.reach is not None:
            return True                                      # busy picking something up
        if self.pushing:
            self.pushing = False
            log.info("let go of the wheelchair at %s", np.round(self.chair, 2).tolist())
            if self.phase != "bed" and math.hypot(self.chair[0] - BED[0], self.chair[1] - BED[1]) < BED_NEAR:
                self._bed_at = n.t + 1.0
            return True
        a = n.assist
        if a is not None and a.task and a.state == "holding":
            if a.task.get("name") == "glass of water" and self.phase in ("tidy", "cough") \
                    and math.hypot(n.walker.base[0] - self.chair[0], n.walker.base[1] - self.chair[1]) < GIVE_NEAR:
                self._given = True                           # near enough: he takes it, wherever the hand is
                log.info("the glass is handed over")
            return False                                     # grasp assist opens the hand
        gi = self._index("glass of water")
        if not n._pointed and gi is not None and self.phase in ("tidy", "cough") and not self._held(gi):
            g = self._objects()[gi]["pos"]
            on_table = 0.14 <= g[0] <= 0.80 and abs(g[1]) <= 0.33 and g[2] > -0.2
            if on_table and math.hypot(g[0] - n.walker.base[0], g[1] - n.walker.base[1]) < GLASS_NEAR:
                self.reach = {"stage": "walk", "i": gi, "name": "glass of water", "t": n.t, "table": True}
                log.info("near the table: going to pick the glass up")
                return True
        if n._pointed:
            i = self._index(n._pointed)
            if i is not None:
                g = self._objects()[i]["pos"]
                on_table = 0.14 <= g[0] <= 0.80 and abs(g[1]) <= 0.33 and g[2] > -0.2
                plan = None if on_table else self._plan(float(g[2]))
                if plan and math.hypot(g[0] - n.walker.base[0], g[1] - n.walker.base[1]) < 2.2:
                    # on the floor, on the bed, anywhere but the table: step up to it and bend as far as it needs
                    self.reach = {"stage": "walk", "i": i, "name": n._pointed, "t": n.t, "d": plan[0], "bend": plan[1]}
                    log.info("going to pick the %s up (%.2f m ahead, bend %.2f)", n._pointed, plan[0], plan[1])
                    return True
            return False
        if self._behind_chair() and self.phase != "bed":
            self.pushing, self._push_t = True, n.t
            log.info("took hold of the wheelchair")
            return True
        return False

    @property
    def busy(self):
        """The robot is moving itself (stepping up to a thing on the floor): the suit must not walk it meanwhile."""
        return self.reach is not None

    def handle_targets(self):
        """Where the two hands go while the chair is pushed: its handles, in the robot's own frame."""
        return {s: self.node.walker.to_pelvis(self._in_chair((p[0], p[1], p[2]))) for s, p in HANDLES.items()}

    def _plan(self, z):
        """How to get the right hand onto a thing at height z (room frame, from the standing pelvis height):
        (how far ahead of the pelvis it has to be, how much to squat and bow: 0 = standing .. 1 = down to the
        floor). Every depth and distance is tried on the walker's own kinematics and the one that leaves the thing
        deepest inside the hand's reach is taken (so a few centimetres of error do not matter); None if none does."""
        w = self.node.walker
        keep, best = (w.dz, w.pitch), None
        try:
            for e in np.linspace(0.0, 1.0, 21):
                w.dz, w.pitch = w.bend_depth(float(e))
                for d in np.arange(0.14, 0.44, 0.02):
                    p = w.room_to_pelvis([0.0, 0.0, 0.0])                # (unused: keeps the base out of it)
                    c, s_ = math.cos(w.base[2]), math.sin(w.base[2])
                    at = [w.base[0] + c * d - s_ * STANCE[1], w.base[1] + s_ * d + c * STANCE[1], z]
                    p = w.to_pelvis(at)                                  # where that puts it in the hand's (torso's) frame
                    margin = min(p[0] - 0.24, 0.46 - p[0], p[2] - 0.05, 0.30 - p[2])
                    if margin > 0.005 and (best is None or margin > best[0] + 1e-6):
                        best = (margin, float(d), float(math.acos(1 - 2 * e) / math.pi))    # apply_bend eases: this bend gives that depth
        finally:
            w.dz, w.pitch = keep
        return best and best[1:]

    def _reach(self):
        """One tick of picking a thing up that is not on the bedside table: off the floor, the bed, anywhere low."""
        from .robot_node import task_target
        n, r = self.node, self.reach
        w, t = n.walker, n.t
        ob = self._objects()[r["i"]]
        table = bool(r.get("table"))                         # on the bedside table: face it, no squat
        if r["stage"] == "walk":                             # step until the thing is where the right hand comes down
            b = w.base
            d = np.asarray(ob["pos"][:2], float) - b[:2]
            c, s = math.cos(b[2]), math.sin(b[2])
            fx, fy = c * d[0] + s * d[1], -s * d[0] + c * d[1]
            stance = TABLE_STANCE if table else (r.get("d", STANCE[0]), STANCE[1])
            ex, ey = fx - stance[0], fy - stance[1]
            if table:                                        # the table is ahead of the home pose: heading 0
                turn = walk._wrap(0.0 - b[2])
            else:
                turn = math.atan2(fy - stance[1], max(fx, 0.05)) if math.hypot(fx, fy) > 0.6 or fx < 0.1 else 0.0
            if t - r["t"] > 8.0 and math.hypot(ex, ey) > 0.12:   # something is in the way: give up, do not bend at nothing
                log.warning("could not get close enough to the %s", r["name"])
                self.reach = None
            elif (math.hypot(ex, ey) < 0.02 and abs(turn) < 0.1) or t - r["t"] > 8.0:
                r.update(stage="bend", t=t)
                w.bend_goal = 0.0 if table else r.get("bend", 1.0)
            elif abs(turn) > 0.25:
                w.drive(0.0, 0.0, float(np.clip(2.0 * turn, -1.4, 1.4)), n.dt)
            else:
                w.drive(float(np.clip(2.0 * ex, -0.5, 0.5)), float(np.clip(2.0 * ey, -0.4, 0.4)), float(np.clip(2.0 * turn, -1.0, 1.0)), n.dt)
        elif r["stage"] == "bend":
            if table or w.bend >= r.get("bend", 1.0) - 0.005:
                n.task_target = task_target({"request_id": "floor-%d" % n.tick_i, "name": r["name"], "destination": "hold"})
                n._grab_next = True
                r.update(stage="grab", t=t)
        elif r["stage"] == "grab":
            a = n.assist
            if a.task and a.state == "holding" and t - r["t"] > 1.0:
                w.bend_goal = 0.0
                r.update(stage="rise", t=t)
                if r["name"] == "sweater" and self.phase == "tidy" and self._cough_from is None:
                    self._cough_from = t + COUGH_AFTER_S
            elif t - r["t"] > 10.0:
                log.warning("could not get hold of the %s: standing up again", r["name"])
                n.task_target = None
                w.bend_goal = 0.0
                r.update(stage="rise", t=t)
        elif r["stage"] == "rise" and w.bend <= 0.005:
            self.reach = None

    def cue(self, name):
        """Stage manager's override (the page's number keys): jump to a point in the story."""
        t = self.node.t
        if name == "reset":                              # everything: the robot and the items too
            self.node.reset_scene()
        elif name == "cough":
            self.phase, self._cough_from, self._fit, self._rings = "cough", t, -1, 0
        elif name == "drink":
            self.phase = "drink"
        elif name == "bed":
            self.pushing, self.phase = False, "bed"
        log.info("scene cue: %s", name)

    # ------------------------------------------------------------------ every tick
    def update(self):
        n = self.node
        t = n.t
        if self.reach is not None:
            self._reach()
        if self.pushing:                                     # the chair goes ahead of the robot, facing its way
            b = n.walker.base
            want = np.array([b[0] + PUSH_GAP * math.cos(b[2]), b[1] + PUSH_GAP * math.sin(b[2]), b[2]])
            k = min(1.0, (t - self._push_t) / 0.7) * 0.35 + (0.65 if t - self._push_t > 0.7 else 0.0)
            self.chair[:2] += (want[:2] - self.chair[:2]) * k
            self.chair[2] += walk._wrap(want[2] - self.chair[2]) * k
        if self.phase == "tidy" and self._cough_from is not None and t >= self._cough_from:
            self.phase = "cough"
            log.info("the man starts coughing")
        if self.phase == "cough":
            since = t - (self._cough_from if self._cough_from is not None else t)
            fit, into = int(since // FIT_S), since % FIT_S
            if fit != self._fit:
                self._fit, self._rings = fit, 0
                self.cough += 1
                self._last_cough = t
            while self._rings < len(RINGS) and into >= RINGS[self._rings]:
                self._rings += 1
                self.bell += 1
        gi = self._index("glass of water")
        if gi is not None:
            g = self._objects()[gi]
            near = math.hypot(g["pos"][0] - home.HANDOVER[0], g["pos"][1] - home.HANDOVER[1]) < 0.45 and g["pos"][2] > -0.35
            if self.phase in ("tidy", "cough") and not self._held(gi) and (near or self._given):
                self.phase = "drink"
                log.info("he has the water: drinking")
            if self.phase == "drink":
                g["pos"] = self._in_chair(DRINK_AT)
            elif self.phase == "bed" and self._bed_done:
                g["pos"] = self._in_chair(SEAT_AT)
        if self._bed_at is not None and t >= self._bed_at:
            self._bed_at, self.phase = None, "bed"
            log.info("he is in bed")
        self._cloth()
        self._publish()

    @property
    def _bed_done(self):
        return self.phase == "bed"

    def _cloth(self):
        """One tick of the hanging sweater's swing."""
        i = self._index("sweater")
        if i is None or not self._held(i):
            self.swing[:], self._swing_v[:], self._pivot = 0.0, 0.0, None
            return
        dt, p = self.node.dt, np.asarray(self._objects()[i]["pos"], float)
        if self._pivot is None:
            self._pivot = (p.copy(), np.zeros(3))
            self._swing_v[:] = (0.9, -0.6)                   # picked up: it starts with a swing
        p0, v0 = self._pivot
        v = v0 + 0.5 * ((p - p0) / dt - v0)                  # smoothed: the hand's place jitters
        a = np.clip((v - v0) / dt, -8.0, 8.0)
        self._pivot = (p.copy(), v)
        for k in range(2):
            acc = -(9.81 / CLOTH_LEN) * math.sin(self.swing[k]) - (a[k] / CLOTH_LEN) * math.cos(self.swing[k]) - 1.6 * self._swing_v[k]
            self._swing_v[k] += acc * dt
            self.swing[k] = float(np.clip(self.swing[k] + self._swing_v[k] * dt, -1.1, 1.1))

    def hanging(self):
        """The sweater in the hand: {"pos": [x, y, z] in the room (z from the floor), "quat": [w, x, y, z]} for the
        hanging mesh (held by its top), or None when it lies somewhere."""
        i = self._index("sweater")
        if i is None or not self._held(i):
            return None
        p = self._objects()[i]["pos"]
        rx, ry = float(self.swing[1]), -float(self.swing[0])  # the end out along +y: turn about x; along +x: about -y
        ang = math.hypot(rx, ry)
        s = math.sin(ang / 2) / ang if ang > 1e-6 else 0.5
        qs = (math.cos(ang / 2), rx * s, ry * s, 0.0)        # the swing, in the room's axes
        h = self.node.walker.base[2] / 2
        qz = (math.cos(h), 0.0, 0.0, math.sin(h))            # turned with the robot
        w1, x1, y1, z1 = qs
        w2, x2, y2, z2 = qz
        q = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
        return {"pos": [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]) + 0.793, 4)], "quat": [round(v, 5) for v in q]}

    def man(self):
        if self.phase == "bed":
            return "bed"
        if self.phase == "drink" or (self.phase == "cough" and self.node.t - self._last_cough < HAND_UP_S):
            return "hand_up"
        return "idle"

    def _glass_hint(self):
        """What the trigger would do with the glass right now, with nothing pointed at: "pick", "give" or None."""
        n, gi = self.node, self._index("glass of water")
        if gi is None or self.phase not in ("tidy", "cough") or self.reach is not None or self.pushing:
            return None
        a, b = n.assist, n.walker.base
        if a is not None and a.task and a.state == "holding" and a.task.get("name") == "glass of water":
            return "give" if math.hypot(b[0] - self.chair[0], b[1] - self.chair[1]) < GIVE_NEAR else None
        g = self._objects()[gi]["pos"]
        if not self._held(gi) and 0.14 <= g[0] <= 0.80 and abs(g[1]) <= 0.33 and g[2] > -0.2 \
                and math.hypot(g[0] - b[0], g[1] - b[1]) < GLASS_NEAR:
            return "pick"
        return None

    def state(self):
        return {"phase": self.phase, "man": self.man(), "chair": [round(float(v), 4) for v in self.chair],
                "pushing": self.pushing, "can_push": bool(not self.pushing and self.phase != "bed" and self._behind_chair()),
                "cough": self.cough, "bell": self.bell, "glass": self._glass_hint(),
                "reach": self.reach and {"stage": self.reach["stage"], "name": self.reach["name"]}}
