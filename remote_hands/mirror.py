"""Arms that look like the worker's: the robot's upper arms and forearms point the way the suit's do.

The other way of driving the arms (retarget.py + the IK) only places the hand, inside a small box over the
table: right for picking, but the elbows and the rest of the arm go wherever the IK likes, so the robot does
not look like the person in the suit. Here the suit's two arm segments are copied as directions:

    suit, in the worker's chest frame (x forward, y left, z up):  upper arm (shoulder -> elbow), forearm (elbow -> wrist)
    robot: shoulder pitch / roll / yaw and elbow chosen so its own two segments point the same way

Solved numerically on the G1 model (a few damped Gauss-Newton steps from the last answer, inside the joint
limits). The three wrist joints stay at zero: the hand continues the forearm. The robot node uses this whenever
grasp assist is not moving a hand itself (a pick, a release, the wheelchair's handles).
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

SIDES = ("left", "right")
JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")


class ArmMirror:
    def __init__(self, model, data):
        self.m, self.d = model, data
        self.q = pin.neutral(model)
        self.idx = {s: [model.joints[model.getJointId(f"{s}_{j}_joint")].idx_q for j in JOINTS] for s in SIDES}
        self.lim = {s: (np.array([model.lowerPositionLimit[i] for i in self.idx[s]]) + 0.05,
                        np.array([model.upperPositionLimit[i] for i in self.idx[s]]) - 0.05) for s in SIDES}
        self.frames = {s: [model.getFrameId(f"{s}_{n}") for n in ("shoulder_roll_link", "elbow_link", "wrist_yaw_link")] for s in SIDES}
        self.a = {s: np.zeros(4) for s in SIDES}         # the last answer: [pitch, roll, yaw, elbow]
        for s in SIDES:
            self.a[s][3] = 0.6

    def _dirs(self, side, a):
        """The robot's upper-arm and forearm directions (pelvis frame = torso frame at a straight waist)."""
        for i, v in zip(self.idx[side], a):
            self.q[i] = v
        pin.forwardKinematics(self.m, self.d, self.q)
        sh, el, wr = (pin.updateFramePlacement(self.m, self.d, f).translation for f in self.frames[side])
        u, f = el - sh, wr - el
        return np.concatenate([u / np.linalg.norm(u), f / np.linalg.norm(f)])

    def solve(self, arms, iterations=3):
        """arms: {side: {"upper": unit vector, "fore": unit vector}} in the chest frame.
        Returns the 14 arm joint angles in the node's order (left 7, right 7), or None if a side is missing."""
        out = []
        for s in SIDES:
            arm = (arms or {}).get(s)
            if not arm:
                return None
            want = np.concatenate([np.asarray(arm["upper"], float), np.asarray(arm["fore"], float)])
            a, (lo, hi) = self.a[s].copy(), self.lim[s]
            for _ in range(iterations):
                r = self._dirs(s, a) - want
                J = np.empty((6, 4))
                for k in range(4):
                    b = a.copy()
                    b[k] += 1e-4
                    J[:, k] = (self._dirs(s, b) - self._dirs(s, a)) / 1e-4
                # damped: near a straight arm the twist about the upper arm is free, so it stays where it was
                step = np.linalg.solve(J.T @ J + 0.02 * np.eye(4), -J.T @ r)
                a = np.clip(a + np.clip(step, -0.5, 0.5), lo, hi)
            self.a[s] = a
            out += [*a, 0.0, 0.0, 0.0]
        return np.array(out)
