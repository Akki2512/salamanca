"""Wrist poses -> G1 arm joint targets.

Three parts, in the order the data flows:

  ClutchMap    human wrist position -> robot grasp-point target (pelvis frame). Relative ("clutched") mapping:
               when the helper engages, their hand position and the robot's current hand position are tied
               together; from then on the robot hand moves by scale * (human hand movement). No absolute
               calibration between the two bodies is needed, and re-engaging removes accumulated mocap drift.
  IKWorker     grasp-point targets -> desired arm joints, on its own thread. Measured on a laptop: median
               2 ms, ~15% of solves 10-100 ms, an unreachable target ~0.5 s. The 30 Hz control loop must
               never wait for that (1 s without a command latches the safety filter's safe stop), so the
               loop only ever reads the newest finished solution.
  slew         (in the robot node, before IK) the IK target follows the helper's hand at a capped Cartesian
               speed. Consecutive IK problems are then centimetres apart, so each solve starts next to its
               answer, the arm does not flip configuration, and the hand travels in a straight line the
               helper can predict. Without it a fast human move produced a far-away joint solution and the
               joint-space path to it took the hand through the counter.
  rate_limit   desired joints -> the command actually sent, moving at most `max_vel` rad/s from the last
               command. The Aalto safety filter does the same clamp at 1.0 rad/s but counts every clamped
               tick as a "clipped step" and aborts after 30; staying under it means it never fires.
"""
from __future__ import annotations

import threading

import numpy as np

from .messages import SIDES, WristPose

# Where the robot's grasp point may be commanded (pelvis frame, m). Inside the safety filter's workspace
# box and above its counter keep-out; per side, a hand may not cross far over the body's midline.
# Checked by walking the IK in 1 cm steps from the ready pose: the object band, carry height and the near
# part of all three bins are reachable palm-down without the arm changing configuration; x = 0.50 at the side
# bins is not.
REACH_MIN = {"left": np.array([0.22, -0.05, 0.035]), "right": np.array([0.22, -0.36, 0.035])}
REACH_MAX = {"left": np.array([0.48, 0.36, 0.32]), "right": np.array([0.48, 0.05, 0.32])}


class ClutchMap:
    def __init__(self, scale=0.75):
        self.scale = scale          # robot reach / human reach (G1 ~0.55 m, adult ~0.70 m)
        self.engaged = False
        self._h0 = {}               # human wrist at engage
        self._r0 = {}               # robot grasp point at engage

    def update(self, pose: WristPose, robot_grasp: dict) -> dict | None:
        """robot_grasp: {side: (3,)} current commanded grasp points. Returns {side: target (3,)} or None
        while the clutch is open."""
        if not pose.clutch:
            self.engaged = False
            return None
        if not self.engaged:
            self._h0 = {s: np.asarray(pose.p(s), float) for s in SIDES}
            self._r0 = {s: np.asarray(robot_grasp[s], float).copy() for s in SIDES}
            self.engaged = True
        out = {}
        for s in SIDES:
            p = self._r0[s] + self.scale * (np.asarray(pose.p(s), float) - self._h0[s])
            out[s] = np.clip(p, REACH_MIN[s], REACH_MAX[s])
        return out

    def release(self):
        self.engaged = False


class IKWorker:
    """Solves the newest target; older unsolved targets are dropped. `sync=True` solves inline (tests)."""

    def __init__(self, kin, q_init, sync=False, max_jump=0.35, q_ready=None):
        self.kin = kin
        self.q_ready = None if q_ready is None else np.asarray(q_ready, float).copy()
        self.sync = sync
        self.max_jump = max_jump    # rad; a "solution" further than this from the seed is another arm
                                    # configuration: moving there in joint space sweeps the hand anywhere
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._target = None
        self._q = np.asarray(q_init, float).copy()   # newest solution (14,)
        self._ok = {s: True for s in SIDES}
        self.solves = 0
        self._epoch = 0
        self._done = {}             # {side: last target solved}
        self._relax = set()
        self._stop = False
        if not sync:
            self._thread = threading.Thread(target=self._run, name="ik", daemon=True)
            self._thread.start()

    def submit(self, targets: dict, relax=()):
        """relax: sides whose next solve may exceed max_jump (recovery from the edge of reach, where a
        centimetre of hand travel can need a large joint change)."""
        with self._lock:
            self._target = {s: np.asarray(targets[s], float).copy() for s in SIDES}
            self._relax = set(relax)
        if self.sync:
            self._solve_once()
        else:
            self._wake.set()

    def reseed(self, q):
        """Forget the last solution (e.g. after a reset or a long hold): continue from where the robot is."""
        with self._lock:
            self._q = np.asarray(q, float).copy()
            self._target = None
            self._done = {}
            self._epoch += 1

    def latest(self):
        with self._lock:
            return self._q.copy(), dict(self._ok)

    def close(self):
        self._stop = True
        self._wake.set()

    def _solve_once(self):
        with self._lock:
            tgt, self._target = self._target, None
            q = self._q.copy()
            epoch, relax = self._epoch, set(self._relax)
            if tgt is None:
                return
            done = dict(self._done)
        ok = dict(self._ok)
        for s in SIDES:
            if s in done and np.abs(tgt[s] - done[s]).max() < 1e-3:
                continue            # same target as the last solve for this arm: the answer has not changed
            done[s] = tgt[s]
            # Recovery solves may also start from the ready pose: after working at the edge of reach the
            # arm can sit in a configuration from which nearby points have no palm-down solution at all.
            extra = [self.q_ready[self.kin.sl(s)]] if (s in relax and self.q_ready is not None) else ()
            q_new, ok[s], _, _ = self.kin.solve(s, tgt[s], q, top_down=True, extra_seeds=extra, restarts=False)
            ok[s] = bool(ok[s]) and (s in relax or float(np.abs(q_new - q).max()) <= self.max_jump)
            if ok[s]:               # an unreachable target (or a configuration flip) leaves that arm where it was
                q = q_new
        with self._lock:
            if self._epoch == epoch:    # a reseed() during the solve wins: drop this result
                self._q, self._ok, self._done = q, ok, done
            self.solves += 1

    def _run(self):
        while not self._stop:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                break
            try:
                self._solve_once()
            except Exception:       # a solver failure must not kill the thread: the loop keeps holding
                import logging
                logging.getLogger("remote_hands.ik").exception("IK solve failed")


def slew(p_last, p_target, max_speed, dt):
    """Move a point towards p_target by at most max_speed * dt (straight line)."""
    d = np.asarray(p_target, float) - p_last
    n = float(np.linalg.norm(d))
    step = max_speed * dt
    return p_last + d if n <= step else p_last + d * (step / n)


def rate_limit(q_last, q_des, max_vel, dt):
    """Move from q_last towards q_des by at most max_vel * dt per joint."""
    d = max_vel * dt
    return q_last + np.clip(np.asarray(q_des, float) - q_last, -d, d)
