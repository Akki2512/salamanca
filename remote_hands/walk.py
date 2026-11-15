"""Walking: the suit's hips and legs move the robot through the room.

The robot's arms are still driven in the robot's own (pelvis) frame, as before. This module adds where the
pelvis IS: a position and heading in the room, a height, and twelve leg joint angles.

    base      follows the worker's hips. From the moment the clutch engages, the robot's displacement is the
              worker's hip displacement times (robot leg length / worker leg length), so one human step is one
              robot step, and its heading turns by as much as the worker turns.
    legs      each robot leg copies the direction of the worker's thigh and shin (hip pitch and roll, knee);
              the ankle keeps the sole level.
    height    whatever puts the lower foot on the floor: bend your knees and the robot crouches.

Frames. "Home" is the robot's own frame at its starting place by the bedside table, at standing height:
x forward, y left, z up, origin at the pelvis. Items and the hand-over point are fixed in the home frame.
to_pelvis() / to_home() convert between it and where the pelvis is now.

The suit has no absolute position sensing, so the worker's position drifts slowly; re-engaging the clutch
re-ties the robot to wherever the worker is. Nothing here balances: the base is placed, not simulated.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pinocchio as pin

from . import AALTO_REPO
from .messages import SIDES

URDF = AALTO_REPO / "third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
LEG_JOINTS = tuple(f"{s}_{j}_joint" for s in SIDES for j in LEG)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
PELVIS_HEIGHT = 0.793           # standing, legs straight (m above the floor)

# Where the pelvis may not go (home frame, m): the bedside table is in front, the bed on the right,
# the walls around. Other furniture is not known here.
X_MAX = 0.02                    # the table edge is 14 cm in front of the start position
BED_Y, BED_X_FROM = -0.15, -1.40    # right of this line (y below it) is the bed, for x beyond BED_X_FROM
WALLS = ((-5.70, -1.70), (0.02, 4.00))      # (x min, y min), (x max, y max)
# Furniture the pelvis may not enter: (x min, x max, y min, y max) in the home frame, from the footprints in
# unreal/RemoteHandsBedroom/Scripts/build_bedroom.py (home x = (room x - 150) / 100, home y = (230 - room y) / 100).
FURNITURE = (
    (-1.11, 0.90, -1.90, -0.55),    # the bed (out by the nearest side: no cutting across it from its foot)
    (-4.77, -2.03, 3.34, 4.26),    # sofa
    (-3.72, -2.68, 2.38, 3.42),    # coffee table
    (-4.86, -3.14, -1.84, -0.83),    # chest of drawers
    (-6.00, -5.49, -1.89, -1.27),    # grandfather clock
    (-1.93, -1.31, -1.64, -0.76),    # ottoman
    (-3.10, -2.50, -1.90, -1.30),    # basket
    (-5.82, -4.95, 3.25, 4.12),    # armchair
    (-6.00, -5.35, 0.45, 1.15),    # radio table
    (-6.00, -5.40, -1.15, -0.25),    # shelf unit
    (-1.65, -0.55, 3.70, 4.30),    # bookshelf
    (0.28, 0.78, 2.55, 3.05),    # plant
)
BODY = 0.28                                 # how far the pelvis stays from a piece of furniture (m)
# Walking gain: metres the robot goes for each metre the worker walks. 2 = the 6.85 m room fits in about 3.4 m of
# real floor. (Before 15 Nov it was the ratio of leg lengths, about 0.78.) REMOTE_HANDS_WALK_GAIN changes it.
# Turning is not scaled.
GAIN = float(os.environ.get("REMOTE_HANDS_WALK_GAIN", "2.0"))
SQUAT, BOW = 1.10, math.radians(80.0)       # reaching the floor: thigh angle of the squat, and the bow at the waist (rad)
CHAIR = (-4.90, 1.70, 0.80)                 # the wheelchair: x, y and the distance the pelvis keeps from its middle


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Walker:
    def __init__(self, max_speed=1.4, max_turn=3.5, smooth=0.45):
        self.m = pin.buildModelFromUrdf(str(URDF))
        self.d = self.m.createData()
        self.qi = {n: self.m.joints[self.m.getJointId(n)].idx_q for n in LEG_JOINTS}
        self.lim = {n: (float(self.m.lowerPositionLimit[i]), float(self.m.upperPositionLimit[i])) for n, i in self.qi.items()}
        self.ankles = [self.m.getFrameId(f"{s}_ankle_roll_link") for s in SIDES]
        self._z0 = self._lowest_ankle({})                # ankle height (pelvis frame) with the legs straight
        hip, knee = self.m.getFrameId("left_hip_pitch_link"), self.m.getFrameId("left_knee_link")
        self.leg_len = float(np.linalg.norm(self.d.oMf[hip].translation - self.d.oMf[knee].translation)
                             + np.linalg.norm(self.d.oMf[knee].translation - self.d.oMf[self.ankles[0]].translation))
        self.max_speed, self.max_turn, self.smooth = max_speed, max_turn, smooth
        self.base = np.zeros(3)                          # x, y, heading in the home frame
        self.dz = 0.0                                    # pelvis height relative to standing (<= 0)
        self.legs = {n: 0.0 for n in LEG_JOINTS}
        self._ref = None                                 # (worker hip xy, worker heading, robot base) at engagement
        self._phase, self._gait = 0.0, 0.0               # joystick walking: where in the stride, and how much of one
        # Reaching down to the floor: a deep squat plus a bow at the waist. `bend` runs 0..1 towards `bend_goal`.
        # The arms and head sit on the torso, so "the robot's own frame" used for the hands (to_home / to_pelvis)
        # is the torso's: it tips with the bow, and everything aimed in it (grasp assist, reach limits) follows.
        self.bend, self.bend_goal, self.pitch, self._free = 0.0, 0.0, 0.0, None
        # How the worker's chest is turned from their hips (waist yaw, roll, pitch; radians), copied to the waist.
        self.torso = np.zeros(3)
        self._waist_lim = [(float(self.m.lowerPositionLimit[i]), float(self.m.upperPositionLimit[i])) for i in
                           (self.m.joints[self.m.getJointId(n)].idx_q for n in WAIST_JOINTS)]
        self.waist = self.d.oMf[self.m.getFrameId("torso_link")].translation.copy()   # where the torso tips, pelvis frame

    # ------------------------------------------------------------------ frames
    def bend_depth(self, k):
        """What a squat-and-bow of depth k (0..1, after easing) does: (pelvis height change, bow angle), from the
        legs' real kinematics and joint limits (straight legs otherwise). For planning a reach."""
        a, legs = SQUAT * k, {}
        for side in SIDES:
            for j, v in (("hip_pitch", -a), ("knee", 2 * a), ("ankle_pitch", -a)):
                n = f"{side}_{j}_joint"
                legs[n] = float(np.clip(v, *self.lim[n]))
        return min(0.0, self._z0 - self._lowest_ankle(legs)), BOW * k

    def apply_bend(self, dt):
        """Move `bend` towards `bend_goal` and pose the legs and waist for it. Call once a tick, after walking."""
        if self.bend_goal <= 0.0 and self.bend <= 0.0:
            return
        if self._free is None:
            self._free = dict(self.legs)                 # the legs as they were: they come back to this
        self.bend = float(np.clip(self.bend + np.clip(self.bend_goal - self.bend, -0.7 * dt, 0.7 * dt), 0.0, 1.0))
        k = 0.5 - 0.5 * math.cos(math.pi * self.bend)    # eased
        a = SQUAT * k
        for side in SIDES:
            squat = {"hip_pitch": -a, "hip_roll": 0.0, "hip_yaw": 0.0, "knee": 2 * a, "ankle_pitch": -a, "ankle_roll": 0.0}
            for j, v in squat.items():
                n = f"{side}_{j}_joint"
                lo, hi = self.lim[n]
                self.legs[n] = float(np.clip((1 - k) * self._free[n] + k * v, lo, hi))
        self.dz = min(0.0, self._z0 - self._lowest_ankle(self.legs))
        self.pitch = BOW * k
        if self.bend <= 0.0:
            self._free = None

    def _torso_R(self):
        """The torso's rotation from the pelvis: waist yaw, then roll, then pitch (the bow added to the pitch)."""
        y, r, p = float(self.torso[0]), float(self.torso[1]), float(self.torso[2]) + self.pitch
        cy, sy, cr, sr, cp, sp = math.cos(y), math.sin(y), math.cos(r), math.sin(r), math.cos(p), math.sin(p)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        return Rz @ Rx @ Ry

    def _tip(self, p, sign):
        if not self.pitch and not self.torso.any():
            return np.asarray(p, float)
        R = self._torso_R()
        d = np.asarray(p, float) - self.waist
        return (R @ d if sign > 0 else R.T @ d) + self.waist

    def room_to_pelvis(self, p):
        """A point in the room, in the pelvis's own frame (which does not tip with the torso): for Unreal."""
        c, s = math.cos(self.base[2]), math.sin(self.base[2])
        d = np.asarray(p, float) - [self.base[0], self.base[1], self.dz]
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])

    def to_home(self, p):
        c, s = math.cos(self.base[2]), math.sin(self.base[2])
        p = self._tip(p, 1.0)
        return np.array([c * p[0] - s * p[1] + self.base[0], s * p[0] + c * p[1] + self.base[1], p[2] + self.dz])

    def to_pelvis(self, p):
        c, s = math.cos(self.base[2]), math.sin(self.base[2])
        d = np.asarray(p, float) - [self.base[0], self.base[1], self.dz]
        return self._tip(np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]]), -1.0)

    @property
    def moved(self):
        return bool(np.abs(self.base).max() > 1e-4 or abs(self.dz) > 1e-4)

    # ------------------------------------------------------------------ legs
    def _lowest_ankle(self, legs):
        q = pin.neutral(self.m)
        for n, v in legs.items():
            q[self.qi[n]] = v
        pin.forwardKinematics(self.m, self.d, q)
        pin.updateFramePlacements(self.m, self.d)
        return float(min(self.d.oMf[a].translation[2] for a in self.ankles))

    def _leg_angles(self, side, thigh, shin):
        """Joint angles that point the robot's thigh and shin the way the worker's are (unit vectors in the
        worker's heading frame: x forward, y left, z up). Straight down is zero."""
        t = np.asarray(thigh, float)
        t = t / max(1e-9, np.linalg.norm(t))
        sh = np.asarray(shin, float)
        sh = sh / max(1e-9, np.linalg.norm(sh))
        roll = math.asin(float(np.clip(t[1], -0.6, 0.6)))
        pitch = math.atan2(-t[0], -t[2])
        knee = math.acos(float(np.clip(t @ sh, -1.0, 1.0)))
        want = {"hip_pitch": pitch, "hip_roll": roll, "hip_yaw": 0.0, "knee": knee,
                "ankle_pitch": -(pitch + knee), "ankle_roll": -roll}          # keeps the sole level
        return {f"{side}_{j}_joint": v for j, v in want.items()}

    # ------------------------------------------------------------------ one tick
    def release(self):
        """Clutch open or no body data: stay where we are; the next engagement ties on afresh."""
        self._ref = None

    def update(self, body, dt):
        """body: {"hip": [x, y, z], "yaw": heading, "leg_len": m, "legs": {side: {"thigh": v, "shin": v}}}
        in a world frame that does not turn with the worker (x, y horizontal, z up)."""
        hip, psi = np.asarray(body["hip"], float)[:2], float(body["yaw"])
        if self._ref is None:
            self._ref = (hip.copy(), psi, self.base.copy())
        hip0, psi0, base0 = self._ref
        k = GAIN                                         # metres the robot goes for each metre the worker goes
        d = k * (hip - hip0)
        a = base0[2] - psi0                              # turn from the worker's world into the home frame
        c, s = math.cos(a), math.sin(a)
        want = np.array([base0[0] + c * d[0] - s * d[1], base0[1] + s * d[0] + c * d[1], base0[2] + _wrap(psi - psi0)])
        want[:2] = self._keep_out(want[:2])
        step = want[:2] - self.base[:2]
        n = float(np.linalg.norm(step))
        if n > self.max_speed * dt:                      # a tracking jump must not teleport the robot
            step *= self.max_speed * dt / n
        # checked again where the robot really ends up: a target pushed out to a piece of furniture's far side
        # must not drag the robot through it on the way
        self.base[:2] = self._keep_out(self.base[:2] + step)
        self.base[2] += float(np.clip(_wrap(want[2] - self.base[2]), -self.max_turn * dt, self.max_turn * dt))
        for side in SIDES:
            leg = (body.get("legs") or {}).get(side)
            if not leg:
                continue
            for n_, v in self._leg_angles(side, leg["thigh"], leg["shin"]).items():
                lo, hi = self.lim[n_]
                self.legs[n_] += self.smooth * (float(np.clip(v, lo, hi)) - self.legs[n_])
        self.dz = min(0.0, self._z0 - self._lowest_ankle(self.legs))    # lower foot on the floor
        t = body.get("torso")
        if t:                                            # the chest's turn, lean and bend, inside the waist's limits
            for i, key in enumerate(("yaw", "roll", "pitch")):
                lo, hi = self._waist_lim[i]
                self.torso[i] += self.smooth * (float(np.clip(t.get(key, 0.0), lo, hi)) - self.torso[i])

    def drive(self, vx, vy, wz, dt, stride=0.55):
        """Joystick walking, for when there is no suit: move at vx forward / vy left (m/s) and turn at wz (rad/s),
        all relative to the way the robot faces. The legs are not anyone's: a plain alternating stride whose size
        follows the speed, so the robot steps as far as it travels."""
        self._ref = None                                 # a suit that takes over later ties on afresh
        self.torso *= 0.85                               # no suit: the waist straightens
        c, s = math.cos(self.base[2]), math.sin(self.base[2])
        want = self._keep_out(self.base[:2] + np.array([c * vx - s * vy, s * vx + c * vy]) * dt)
        moved = float(np.linalg.norm(want - self.base[:2]))
        self.base[:2] = want
        self.base[2] = _wrap(self.base[2] + wz * dt)
        back = -1.0 if vx < -abs(vy) else 1.0            # walking backwards: run the stride the other way
        self._phase += back * 2 * math.pi * (moved + 0.10 * abs(wz) * dt) / stride
        amount = float(np.clip(moved / dt / 0.6, 0.0, 1.0)) if dt > 0 else 0.0
        if amount < 0.25 and abs(wz) > 0.2:
            amount = 0.3                                 # turning on the spot: small steps
        self._stride(amount, dt)

    def settle(self, dt):
        """No joystick input: bring the feet together. Returns True while the legs are still moving."""
        if self._gait < 0.01:
            return False
        self._stride(0.0, dt)
        return True

    def _stride(self, amount, dt):
        self._gait += float(np.clip(amount - self._gait, -4.0 * dt, 4.0 * dt))
        g = self._gait
        for i, side in enumerate(SIDES):
            ph = self._phase + i * math.pi               # the two legs half a stride apart
            pitch = -0.38 * g * math.sin(ph)             # negative = thigh forward
            knee = 0.80 * g * max(0.0, math.cos(ph)) + 0.10 * g      # bent while the leg swings through
            want = {"hip_pitch": pitch, "hip_roll": 0.0, "hip_yaw": 0.0, "knee": knee,
                    "ankle_pitch": -(pitch + knee), "ankle_roll": 0.0}
            for j, v in want.items():
                n = f"{side}_{j}_joint"
                self.legs[n] += 0.5 * (v - self.legs[n])
        self.dz = min(0.0, self._z0 - self._lowest_ankle(self.legs))

    @staticmethod
    def _keep_out(p):
        x = float(np.clip(p[0], WALLS[0][0], WALLS[1][0]))
        y = float(np.clip(p[1], WALLS[0][1], WALLS[1][1]))
        x = min(x, X_MAX)
        for x0, x1, y0, y1 in FURNITURE:                 # out by the nearest side
            x0, x1, y0, y1 = x0 - BODY, x1 + BODY, y0 - BODY, y1 + BODY
            if x0 < x < x1 and y0 < y < y1:
                gaps = (x - x0, x1 - x, y - y0, y1 - y)
                k = gaps.index(min(gaps))
                x, y = (x0, y) if k == 0 else (x1, y) if k == 1 else (x, y0) if k == 2 else (x, y1)
        dx, dy = x - CHAIR[0], y - CHAIR[1]
        d = math.hypot(dx, dy)
        if d < CHAIR[2]:                                 # pushed back out along the line from the chair
            k = CHAIR[2] / max(d, 1e-6)
            x, y = CHAIR[0] + dx * k, CHAIR[1] + dy * k
        return np.array([x, y])

    # ------------------------------------------------------------------ for the Unreal stream
    def pelvis_world(self):
        """The pelvis in the room: origin on the floor under the robot's starting place, axes as the home frame.
        pos [x, y, z] in metres and quat [w, x, y, z]."""
        return {"pos": [float(self.base[0]), float(self.base[1]), PELVIS_HEIGHT + self.dz],
                "quat": [math.cos(self.base[2] / 2), 0.0, 0.0, math.sin(self.base[2] / 2)]}

    def joints(self):
        return {**{n: round(v, 4) for n, v in self.legs.items()}, "waist_yaw_joint": round(float(self.torso[0]), 4), "waist_roll_joint": round(float(self.torso[1]), 4),
                "waist_pitch_joint": round(float(self.torso[2]) + self.pitch, 4)}
