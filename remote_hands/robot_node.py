"""Robot node: the 30 Hz control loop between the mocap bridge and the Aalto robot interface.

Runs next to the simulator (the sim backend talks DDS, which does not leave the machine). On a laptop it runs
against AaltoEnv's mock backend: same API, no physics.

    python -m remote_hands.robot_node --backend mock --source synthetic --seconds 20   # everything in one process
    python -m remote_hands.robot_node --backend mock --listen 8765                     # wait for a bridge
    python -m remote_hands.robot_node --backend sim --listen 8765                      # on the GPU pod

Every tick steps the env, whatever happens upstream:
    homing   move to the ready pose (hands out of the head camera's view); poses are ignored
    idle     no fresh pose, or the helper's clutch is open: hold the last command
    driving  follow the helper's hands through ClutchMap -> IKWorker -> rate_limit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import threading
import time

import numpy as np

from . import AALTO_REPO  # noqa: F401  (puts the Aalto repo on sys.path)
from . import home
from .messages import SIDES, WristPose
from .grasp_assist import GraspAssist
from .retarget import REACH_MAX, REACH_MIN, ClutchMap, IKWorker, rate_limit, slew
from .walk import Walker

from aaltonomous.baselines.classical.config import hand_postures, home_arm, load_config  # noqa: E402
from aaltonomous.baselines.classical.kinematics import ArmKinematics  # noqa: E402
from aaltonomous.interface.env import AaltoEnv  # noqa: E402
from aaltonomous.interface.safety import SafetyConfig  # noqa: E402

log = logging.getLogger("remote_hands.node")
EYE = np.array([0.0537, 0.0175, 0.4739])    # the head camera (d435_link) in the robot's own frame, from the URDF: pointing is judged from here
HZ = 30.0


class RobotNode:
    def __init__(self, backend="mock", realtime=None, sync_ik=False, scale=0.75, arm_vel=0.9, hand_vel=2.5,
                 stale_s=0.2, cart_vel=0.30, log_path=None, assist=True, **backend_kwargs):
        self.cfg = load_config()
        self.kin = ArmKinematics(self.cfg)          # FK for this loop
        self.postures = hand_postures(self.cfg)
        self.home = self._ready_pose(home_arm(self.cfg))
        self.arm_vel, self.hand_vel, self.stale_s = arm_vel, hand_vel, stale_s
        self.cart_vel = cart_vel    # m/s the robot hand's target may travel (0.9 rad/s on a ~0.4 m arm ~ 0.35 m/s)
        self._refused = {s: 0 for s in SIDES}   # consecutive ticks the IK refused that arm's request
        self._goal = None           # {side: point the IK is asked for}; trails the helper's hand
        self._lift = {s: 0.0 for s in SIDES}    # m added to that hand's goal height by the table guard
        # Limits stay as shipped (1 rad/s arms, 3 rad/s hands, workspace box). Two episode rules are relaxed
        # for teleoperation: no time limit, and no abort after 30 clipped ticks (a hand resting on the
        # workspace boundary clips every tick; that is expected, not a fault).
        if backend == "sim":
            backend_kwargs = self._sim_instance_kwargs(backend_kwargs)
        safety = SafetyConfig(dt=1.0 / HZ, max_clipped_steps=10 ** 9)
        if backend == "mock":
            # No real robot behind the mock (it feeds Unreal): the arms may move as fast as the worker's and
            # anywhere they can reach (hanging by the sides, raised, behind), so they can look like the worker's.
            # The simulator and the robot keep the shipped limits.
            safety = SafetyConfig(dt=1.0 / HZ, max_clipped_steps=10 ** 9, arm_max_vel=6.0, workspace=False)
            self.arm_vel = max(self.arm_vel, 5.0)
        self.env = AaltoEnv(backend=backend, time_limit=1e9, safety=safety, realtime=realtime,
                            instruction="teleoperation", **backend_kwargs)
        self.env.scene = home.load_scene()      # the bedside table and what is on it
        self.walker = Walker()                  # where the pelvis is in the room; stays put unless the source sends hips and legs
        from .mirror import ArmMirror
        self.mirror = ArmMirror(self.walker.m, self.walker.m.createData())   # arms that point the way the worker's do
        self.dt = self.env.dt
        self._objects = None        # callable -> [{"name", "color", "shape", "pos"}] or None (unknown)
        if assist and backend == "mock":
            self._objects = self._mock_world()
        elif assist and backend == "sim":
            self._objects = SceneFeed().latest
        counter_z = float(self.env.scene.cfg["counter"]["mat_origin_in_pelvis"][2])
        self.floor_z = counter_z + float(self.cfg["grasp"]["fingertip_clearance"])   # hand points stay above this
        self._floor_home = self.floor_z                 # that height at standing height; it shifts when the robot crouches
        sc, b = self.env.scene, self.env.scene.cfg["bins"]
        self._view = {"counter_z": counter_z, "bins": {}, "handover": list(home.HANDOVER)}   # for the worker page's drawing
        self.t = 0.0
        self._clock = time.monotonic if self.env.realtime else (lambda: self.t)
        self.clutch = ClutchMap(scale=scale)
        grasp_z = max(counter_z + float(self.cfg["grasp"]["height_frac"]) * self.env.scene.size,
                      max(float(REACH_MIN[s][2]) for s in SIDES))
        self.assist = GraspAssist(self.env.scene, REACH_MIN, REACH_MAX, grasp_z,
                                  approach_height=float(self.cfg["grasp"]["approach_height"])) if assist else None
        self._assist_req = None
        self.on_progress = None     # callable(request_id, state): "in_progress" | "grasped" | "released" | "dropped"
        ik_kin = ArmKinematics(self.cfg)                # own model: used from the IK thread
        # Reset last: the filter's watchdog starts counting at reset, so nothing slow may come after it.
        obs = self.env.reset()
        if assist and backend == "mock":         # reset() sampled a random layout: put the home items out
            self.env.backend.reset_scene(home.layout(), "home")
            for o in self.env.backend.objects:      # an item's position is where the hand takes hold of it
                grip_z = (home.BY_LOOK.get((o["color"], o["shape"])) or {}).get("grip_z")
                if grip_z:
                    o["pos"][2] = counter_z + grip_z
                room = (home.BY_LOOK.get((o["color"], o["shape"])) or {}).get("room")
                if room:                            # lying on the floor out in the room
                    o["pos"] = np.array([room[0], room[1], -0.793 + (grip_z or 0.05)])
        self._items0 = [np.array(o["pos"], float) for o in getattr(self.env.backend, "objects", None) or []]   # for reset_scene
        self._reset_req = False
        from .scene import DemoScene
        self.scene = DemoScene(self) if assist and backend == "mock" else None
        self.arm_cmd = np.asarray(obs["arm_q"], float).copy()
        self.hand_cmd = np.asarray(obs["hand_q"], float).copy()
        self.ik = IKWorker(ik_kin, self.arm_cmd, sync=sync_ik, q_ready=self.home)   # own model: other thread
        self.state = "homing"
        self.tick_i = 0
        self._pose_lock = threading.Lock()
        self._pov_lock = threading.Lock()
        self._pose = None           # (WristPose, receive time)
        self.clutch_on = True       # the helper's own switch (helper page / bridge); off = the robot holds
        self._targets = None
        self.last = {}              # newest telemetry
        self.listeners = []         # callables(telemetry dict)
        self.task_target = None     # {"request_id", "color", "shape", "destination"} from the session server
        self._drive = None          # (forward m/s, left m/s, turn rad/s, when) from a controller's joysticks
        self._point = None          # (yaw left, pitch up, when), degrees from the way the robot faces: where a controller points
        self._press = False         # the controller's trigger was pulled since the last tick
        self._pointed = None        # name of the item being pointed at
        self._grab_next = False     # the next task is to be picked outright (trigger), not waited for
        self._announced = None
        self._log = open(log_path, "a") if log_path else None
        self.recorder = None        # remote_hands/recorder.py, when attached
        self._last_body = None      # the suit's raw body data of the newest tick, for the recorder

    def _mock_world(self):
        """Give the mock backend objects and a head camera: the Aalto baseline's kinematic world (an object
        near a closing hand sticks to it and drops straight down when the hand opens; a ray-cast render of the
        table from the robot's head, about 7 ms, redone only when an object moves). It does not draw the arms."""
        from aaltonomous.baselines.classical.mock_world import RenderMockBackend
        from aaltonomous.interface.calibration import Calibration

        self._cal = Calibration.load()
        walker = self.walker

        class RoomBackend(RenderMockBackend):
            """Items live in the room (home frame), the hands in the robot's own frame: the same sticking rule
            as the parent class, with the hand's grasp point taken into the room first, so it still holds
            when the robot has walked or crouched."""

            def advance(self, dt):
                super(RenderMockBackend, self).advance(dt)
                for side in SIDES:
                    g = walker.to_home(self.kin.grasp_pose(side, self.arm_q)[0])
                    f, h = self._closure(side), self.held[side]
                    if h is None and f > 0.6:
                        free = [i for i in range(len(self.objects)) if all(x is None or x[0] != i for x in self.held.values())]
                        if free:
                            i = min(free, key=lambda i: np.linalg.norm(self.objects[i]["pos"] - g))
                            if np.linalg.norm(self.objects[i]["pos"] - g) < self.attach_radius:
                                self.held[side] = (i, walker.to_pelvis(self.objects[i]["pos"]) - walker.to_pelvis(g))
                    elif h is not None:
                        i, off = h
                        self.objects[i]["pos"] = walker.to_home(walker.to_pelvis(g) + off)
                        if f < 0.3:
                            self._drop(self.objects[i])
                            self.held[side] = None

            def _drop(self, ob):
                """Let go: over the bedside table it comes to rest on the table; near the hand-over point it
                rests on the person's lap; anywhere else it falls to the floor."""
                grip_z = (home.BY_LOOK.get((ob["color"], ob["shape"])) or {}).get("grip_z") or ob["height"] / 2
                counter = float(self.scene.cfg["counter"]["mat_origin_in_pelvis"][2])
                x, y = float(ob["pos"][0]), float(ob["pos"][1])
                if 0.14 <= x <= 0.80 and abs(y) <= 0.33:
                    ob["pos"] = np.array([x, y, counter + grip_z])
                elif -1.15 <= x <= 0.80 and -1.85 <= y <= -0.25:
                    ob["pos"] = np.array([x, y, -0.353 + grip_z])             # on the bed
                elif np.hypot(x - home.HANDOVER[0], y - home.HANDOVER[1]) <= 0.35:
                    ob["pos"] = np.array([x, y, home.LAP_Z + grip_z])         # on the person's lap
                else:
                    ob["pos"] = np.array([x, y, -0.793 + grip_z])

        self.env.backend = RoomBackend(self.env.scene, self._cal, ArmKinematics(self.cfg), self.postures,
                                       depth_noise_mm=0.0, hole_frac=0.0)
        b = self.env.backend
        b.renderer = home.home_renderer(self.env.scene, self._cal)
        # for grasp assist and the worker page: where the items are as the robot sees them now (its own frame)
        return lambda: [{"name": f"{o['color']}_{o['shape']}_{i}", "color": o["color"], "shape": o["shape"],
                         "pos": walker.to_pelvis(o["pos"])} for i, o in enumerate(b.objects)]

    @staticmethod
    def _sim_instance_kwargs(kw):
        """Decide WHICH simulator this node commands, explicitly.

        The Aalto DDS backend always talks to the main sim (DDS domain 1) unless apply_instance_env() has
        remapped it; setting AALTO_DDS_DOMAIN alone does nothing. On 2026-11-14 a trial meant for a second
        instance therefore commanded the main one while an evaluation was running on it. So:
          REMOTE_HANDS_SIM=main      command the main sim (domain 1)
          REMOTE_HANDS_SIM=instance  command the instance named by AALTO_DDS_DOMAIN (must not be 1) and the
                                     AALTO_*_PORT variables (see aaltonomous/sim/run_sim.sh)
        Anything else refuses to start."""
        which = os.environ.get("REMOTE_HANDS_SIM")
        dom = os.environ.get("AALTO_DDS_DOMAIN")
        if which == "instance":
            if not dom or int(dom) == 1 or not os.environ.get("AALTO_SCENE_PORT"):
                raise RuntimeError("REMOTE_HANDS_SIM=instance needs AALTO_DDS_DOMAIN (not 1), AALTO_SCENE_PORT, "
                                   "AALTO_CAM_PORT_BASE and AALTO_RGBD_PORT set to the instance's values")
            from aaltonomous.sim.evaluate import apply_instance_env
            apply_instance_env()
        elif which == "main":
            if dom and int(dom) != 1:
                raise RuntimeError("REMOTE_HANDS_SIM=main but AALTO_DDS_DOMAIN points at another instance")
        else:
            raise RuntimeError("set REMOTE_HANDS_SIM=main or REMOTE_HANDS_SIM=instance: the robot node will not "
                               "guess which simulator to command")
        kw = dict(kw)
        kw.setdefault("img_server_ip", "127.0.0.1")     # the backend's default is the real robot's address
        log.warning("commanding the %s simulator (DDS domain %s)", which, dom or "1")
        return kw

    def _ready_pose(self, home):
        """The pose the node homes to: hands where the Aalto home pose puts them, but palms down.
        The live IK always asks for palms down; starting from a pose that already is one keeps every later
        solve a small step from its seed (from the Aalto home pose the first solve was a 1.3 rad jump)."""
        q = np.asarray(home, float).copy()
        for s in SIDES:
            q_new, ok, _, _ = self.kin.solve(s, self.kin.grasp_pose(s, home)[0], q, top_down=True, restarts=True)
            if not ok:
                raise RuntimeError(f"no palm-down ready pose for the {s} arm")
            q = q_new
        return q

    # ------------------------------------------------------------------ input
    def push(self, pose: WristPose):
        """Called from any thread. Newest pose wins; older ones are dropped, never queued."""
        with self._pose_lock:
            self._pose = (pose, self._clock())

    def drive(self, vx, vy, wz):
        """Called from any thread: walk at vx forward / vy left (m/s), turn at wz (rad/s). It has to be repeated
        several times a second; the robot stops 0.35 s after the last one."""
        clip = lambda v, m: float(max(-m, min(m, float(v or 0.0))))
        self._drive = (clip(vx, 0.8), clip(vy, 0.5), clip(wz, 1.6), self._clock())

    def point(self, yaw, pitch, press=False):
        """Called from any thread: where a controller points, as seen from the robot's eyes (degrees, yaw to the
        left of the way the robot faces, pitch up). press: its trigger was just pulled: pick the item pointed at,
        or let go of the one being held."""
        self._point = (float(yaw), float(pitch), self._clock())
        if press:
            self._press = True

    def _pointed_item(self):
        """The item nearest the pointing direction, if one is within its own apparent size of it."""
        pt = self._point
        if pt is None or self._clock() - pt[2] > 0.6 or not self._objects:
            return None
        y, p = np.radians(pt[0]), np.radians(pt[1])
        d = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
        best, best_err = None, None
        c, s = np.cos(-self.walker.base[2]), np.sin(-self.walker.base[2])
        for o in self._objects():
            item = home.BY_LOOK.get((o["color"], o["shape"]))
            if not item:
                continue
            ax, ay, az = item.get("aim", (0.0, 0.0, 0.0))            # offset in the room's axes -> the robot's
            v = np.asarray(o["pos"], float) + [c * ax - s * ay, s * ax + c * ay, az] - EYE
            dist = float(np.linalg.norm(v))
            ang = float(np.arccos(np.clip(v @ d / dist, -1.0, 1.0)))
            err = ang / max(np.radians(8.0), np.arctan(item.get("radius", 0.06) / dist) * 1.4)
            if err < 1.0 and (best_err is None or err < best_err):
                best, best_err = item["name"], err
        return best

    def _trigger(self):
        """Controller trigger: let go if something is held, else pick what is pointed at."""
        a = self.assist
        if a is None:
            return
        if self.scene is not None and self.scene.trigger():     # far pick, put down, the wheelchair
            return
        if a.task and a.state == "holding":
            a.let_go(self.t, self.grasp_points(self.arm_cmd)[a.side])
            log.info("trigger: letting go of %s", a.task.get("name"))
        elif self._pointed and (not a.task or a.state in ("free", "done")):
            self.task_target = task_target({"request_id": "point-%d" % self.tick_i, "name": self._pointed, "destination": "hold"})
            self._grab_next = True
            log.info("trigger: picking the %s", self._pointed)

    def _fresh_pose(self):
        with self._pose_lock:
            item = self._pose
        if item is None or not self.clutch_on or self._clock() - item[1] > self.stale_s:
            return None
        return item[0]

    def _lowest(self, arm_q, hand_q):
        """Height (pelvis z) of the lowest palm / finger / fingertip point of either hand."""
        return min(float(self.kin.hand_points(s, arm_q, hand_q)[:, 2].min()) for s in SIDES)

    def grasp_points(self, arm_q):
        return {s: self.kin.grasp_pose(s, arm_q)[0] for s in SIDES}

    # ------------------------------------------------------------------ one tick
    def _viz(self):
        """What the mirror page needs to draw boxes round the items and show the joints: all in the room frame
        (x forward and y left of the robot's starting pose, z up from the floor, m), as the recorder writes them."""
        try:
            from .recorder import LEG_JOINTS, WAIST_JOINTS
            from .walk import PELVIS_HEIGHT
            up = np.array([0.0, 0.0, PELVIS_HEIGHT])
            b = self.env.backend
            held = {h[0]: side for side, h in (getattr(b, "held", None) or {}).items() if h is not None}
            items = []
            for k, o in enumerate(getattr(b, "objects", None) or []):
                name = home.name_of(o["color"], o["shape"])
                info = home.BY_NAME.get(name) or {}
                items.append({"name": name, "pos": np.round(np.asarray(o["pos"], float) + up, 3).tolist(), "held_by": held.get(k),
                              "radius": info.get("radius"), "aim": [float(v) for v in info.get("aim", (0.0, 0.0, 0.0))]})
            j = self.walker.joints()
            return {"eye": np.round(np.asarray(self.walker.to_home(EYE), float) + up, 3).tolist(), "items": items,
                    "leg_q": [round(float(j[n]), 3) for n in LEG_JOINTS], "waist_q": [round(float(j[n]), 3) for n in WAIST_JOINTS]}
        except Exception:                                   # graphics only: never in the way of the control loop
            return None

    def reset_scene(self):
        """Called from any thread: put everything back to the start of the story at the next tick."""
        self._reset_req = True

    def _reset_scene(self):
        b, w = self.env.backend, self.walker
        for s in SIDES:
            if getattr(b, "held", None):
                b.held[s] = None
        for o, p in zip(getattr(b, "objects", None) or [], self._items0):
            o["pos"] = p.copy()
        self.task_target, self._grab_next, self._press, self._drive = None, False, False, None
        self.__dict__["_given"] = {}                    # what unreal_stream remembers as handed over
        w.base[:] = 0.0
        w.dz, w.bend, w.bend_goal, w.pitch, w._free, w._ref, w._gait = 0.0, 0.0, 0.0, 0.0, None, None, 0.0
        w.torso[:] = 0.0
        for n in w.legs:
            w.legs[n] = 0.0
        self.clutch.release()
        self.state, self._targets, self._goal = "homing", None, None     # the arms go back to their ready pose
        if self.scene is not None:
            self.scene.reset()
        if self.recorder is not None:
            self.recorder.restart()                     # a running episode ends here and a new one begins
        log.info("scene reset: robot, items, wheelchair and the man are back at the start")

    def tick(self):
        if self._reset_req:
            self._reset_req = False
            self._reset_scene()
        pose = self._fresh_pose()
        ik_ok = {s: True for s in SIDES}
        assist_post = {}
        body = getattr(pose, "body", None) if pose is not None else None
        self._last_body, trig = body, False
        d = self._drive
        if d is not None and self._clock() - d[3] < 0.35 and any(abs(v) > 1e-3 for v in d[:3]) and self.state != "homing":
            self.walker.drive(d[0], d[1], d[2], self.dt)        # a controller's joysticks: walk without the suit
        elif self.walker.settle(self.dt):
            pass                                                # joystick let go: the feet are coming together
        elif self.scene is not None and self.scene.busy:
            self.walker.release()                               # the robot is walking itself to something on the floor
        elif body and self.state != "homing":           # the worker's hips and legs: walk
            try:
                self.walker.update(body, self.dt)
            except (KeyError, TypeError, ValueError) as e:
                log.warning("body data not usable: %r", e)
                self.walker.release()
        else:
            self.walker.release()
        self.walker.apply_bend(self.dt)
        self.floor_z = self._floor_home - self.walker.dz
        over_table = any(0.10 <= h[0] <= 0.85 and abs(h[1]) <= 0.60 for h in
                         (self.walker.to_home(p) for p in self.grasp_points(self.arm_cmd).values()))
        if self.walker.bend > 0.01 or not over_table:
            self.floor_z = -10.0                                # no hand is over the bedside table: nothing under them to guard
        if self.state == "homing":
            q_des = self.home
            if np.abs(self.arm_cmd - self.home).max() < 1e-3:
                self.state = "idle"
                self.ik.reseed(self.arm_cmd)
        else:
            targets = self.clutch.update(pose, self.grasp_points(self.arm_cmd)) if pose is not None else None
            self._pointed = self._pointed_item()
            if self.scene is not None:
                self.scene.update()
            if self._press:
                self._press = False
                trig = True
                self._trigger()
            a = self.assist
            if targets is None and a is not None and (self._grab_next or (a.task and a._forced and a.state in ("approach", "descend", "closing"))
                                                      or (a.task and a.state == "releasing")):
                # a pick or release asked for by the trigger has to go through even with no suit driving the arms
                targets = {s: p.copy() for s, p in self.grasp_points(self.arm_cmd).items()}
            pushing = self.scene is not None and self.scene.pushing
            # Arms like the worker's: when the suit sends its arm segments and grasp assist is not moving a hand
            # itself, the arm joints come straight from the suit (mirror.py) instead of from the hand IK.
            picking = lambda: a is not None and bool(a.task) and a.state in ("approach", "descend", "closing", "releasing")
            mirror_q = None
            if pose is not None and (body or {}).get("arms") and not pushing and not picking():
                mirror_q = self.mirror.solve(body["arms"])
                if mirror_q is not None:
                    targets = {s: p.copy() for s, p in self.grasp_points(self.arm_cmd).items()}
            if pushing:                                 # both hands go to the wheelchair's handles and close on them
                targets = self.scene.handle_targets()
            raw = targets
            targets, assist_post = self._assist(targets)
            if pushing:
                assist_post = {**assist_post, "left": "closed", "right": "closed"}
            if targets is None:                     # dropout or clutch open: hold where we are
                self.clutch.release()
                self.ik.reseed(self.arm_cmd)
                self.state, self._targets, self._goal = "idle", None, None
            else:
                if self._goal is None:              # just engaged: start from where the robot hand is
                    self._goal = self.grasp_points(self.arm_cmd)
                q_seed, last_ok = self.ik.latest()
                relax = []
                for s in SIDES:
                    self._refused[s] = 0 if last_ok[s] else self._refused[s] + 1
                    if last_ok[s]:
                        self._goal[s] = slew(self._goal[s], targets[s], self.cart_vel, self.dt)
                    elif self._refused[s] > 15:
                        # Refused for half a second: take one 1.5 cm hop even if it needs a large joint
                        # change. The table guard and the safety filter still bound where the hand can go.
                        self._goal[s] = slew(self.kin.grasp_pose(s, q_seed)[0], targets[s], 0.015, 1.0)
                        if self._refused[s] > 30:           # a full second: go for the hand's position itself
                            self._goal[s] = targets[s]
                        relax.append(s)
                    else:
                        # The last request was refused (out of reach, or only reachable by flipping the arm).
                        # Ask again a short hop from where the IK last succeeded, towards the hand; vary the
                        # hop so one refused point cannot block the way back.
                        hop = (0.03, 0.015, 0.06)[self.tick_i % 3]
                        self._goal[s] = slew(self.kin.grasp_pose(s, q_seed)[0], targets[s], hop, 1.0)
                for s in SIDES:                     # far above the counter again: forget the guard's lift
                    if self._goal[s][2] > self.floor_z + 0.08:
                        self._lift[s] = 0.0
                self.ik.submit({s: self._goal[s] + [0.0, 0.0, self._lift[s]] for s in SIDES}, relax=relax)
                self.state, self._targets = "driving", raw
            q_des, ik_ok = self.ik.latest()
            if mirror_q is not None and not picking() and not self._grab_next:
                q_des, self._goal = mirror_q, None              # the IK starts again from the hand when a pick begins
                self.ik.reseed(self.arm_cmd)

        arm = rate_limit(self.arm_cmd, q_des, self.arm_vel, self.dt)
        hand_des = self.hand_cmd.copy()
        if pose is not None and self.state == "driving":
            for s, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
                hand_des[sl] = self.postures["closed" if pose.g(s) > 0.5 else "open"][sl]
        elif self.state == "homing":
            hand_des = self.postures["open"]
        for s, name in assist_post.items():         # grasp assist decides that hand, driving or not
            sl = slice(0, 7) if s == "left" else slice(7, 14)
            hand_des[sl] = self.postures[name][sl]
        hand = rate_limit(self.hand_cmd, hand_des, self.hand_vel, self.dt)

        # Table guard. The Aalto filter keeps wrists and palms out of the counter but not the fingers
        # (safety.py: "Not covered"). Refuse an arm step that takes any hand point lower while below the floor.
        # Judged per hand and in the room: only a hand that is over the bedside table is held back, and only
        # that arm stops (the other keeps following); a hand beside or behind the table may hang as low as it likes.
        guard = False
        if self.walker.bend <= 0.01 and self.state != "homing":     # going back to the ready pose must never be held up
            def low(side, a, h):                         # room height of that hand's lowest point
                return min(float(self.walker.to_home(p)[2]) for p in self.kin.hand_points(side, a, h))
            for s_, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
                g = self.walker.to_home(self.kin.grasp_pose(s_, arm)[0])
                if not (0.10 <= g[0] <= 0.85 and abs(g[1]) <= 0.60):
                    self._lift[s_] = 0.0
                    continue
                new, now = low(s_, arm, hand), low(s_, self.arm_cmd, self.hand_cmd)
                if new < self._floor_home and new < now - 1e-6:
                    # Holding alone can deadlock: the path to a goal just above the table may dip below it on
                    # the way. So also ask for that hand a little higher, until the path clears.
                    self._lift[s_] = min(self._lift[s_] + 0.002, 0.04)
                    arm = arm.copy()
                    arm[sl] = self.arm_cmd[sl]           # hold this arm; the fingers may still move
                    guard = True
                    if low(s_, arm, hand) < min(self._floor_home, now - 1e-6):
                        hand = hand.copy()
                        hand[sl] = self.hand_cmd[sl]     # the fingers alone would touch: hold them too

        obs, info = self.env.step({"arm_q_target": arm, "hand_q_target": hand})
        # Continue from what the safety filter actually sent, so a workspace clip does not build up error.
        self.arm_cmd = self.env.filter.arm_cmd.copy()
        self.hand_cmd = self.env.filter.hand_cmd.copy()
        self.t = info["t"]
        self.tick_i += 1
        self.obs = obs
        self.last = {"type": "telemetry", "tick": self.tick_i, "t": round(self.t, 4), "state": self.state,
                     "q_cmd": np.round(self.arm_cmd, 4).tolist(), "q_des": np.round(q_des, 4).tolist(),
                     "hand_cmd": np.round(self.hand_cmd, 3).tolist(),
                     "safety_mode": info["safety_mode"], "clip_reasons": info["clip_reasons"],
                     "clipped_steps": info["clipped_steps"], "ik_ok": ik_ok, "table_guard": guard,
                     "pose_seq": pose.seq if pose is not None else None, "clutch_on": self.clutch_on,
                     "task_target": self.task_target, "lag_cm": self._lag_cm(),
                     "assist": self.assist.info() if self.assist and self.task_target else None,
                     "hands": {s: np.round(v, 3).tolist() for s, v in self.grasp_points(self.arm_cmd).items()},
                     "closed": {s: bool(np.abs(self.hand_cmd[sl] - self.postures["closed"][sl]).max() < 0.15)
                                for s, sl in (("left", slice(0, 7)), ("right", slice(7, 14)))},
                     "objects": [{"item": home.name_of(o["color"], o["shape"]), "color": o["color"], "shape": o["shape"],
                                  "pos": np.round(o["pos"], 3).tolist()}
                                 for o in (self._objects() or [])] if self._objects else [],
                     "view": self._view,
                     "pointed": self._pointed, "trigger": trig,      # trigger: a trigger press was handled this tick
                     "suit_yaw": (body or {}).get("yaw"), "suit_head_rel": (body or {}).get("head_rel"),            # which way the worker's hips face (rad, the suit's own frame)
                     "recording": self.recorder.status() if self.recorder is not None else None,
                     "scene": self.scene.state() if self.scene is not None else None,
                     "base": [round(float(v), 4) for v in (*self.walker.base, self.walker.dz)],     # x, y, heading, height change
                     "viz": self._viz(),                             # for the data graphics on the mirror page
                     # target = where the helper's hand is (the ghost); goal = where the robot is being sent now
                     "target": {s: np.round(v, 4).tolist() for s, v in (self._targets or {}).items()},
                     "goal": {s: np.round(v, 4).tolist() for s, v in (self._goal or {}).items()}}
        tt = self.task_target
        if tt and self.state == "driving" and self._announced != tt["request_id"] and self.on_progress:
            self._announced = tt["request_id"]
            self.on_progress(tt["request_id"], "in_progress")
        if self._log:
            self._log.write(json.dumps(self.last) + "\n")
        for fn in self.listeners:
            fn(self.last)
        return self.last

    def _assist(self, targets):
        """Run grasp assist for this tick. Returns (targets, {side: posture name})."""
        tt = self.task_target
        if self.assist is None:
            return targets, {}
        if (tt or {}).get("request_id") != self._assist_req:
            self._assist_req = (tt or {}).get("request_id")
            self.assist.set_task(tt)
            if self._grab_next and tt:
                self.assist.grab("right")
            self._grab_next = False
        if not tt:
            return targets, {}
        objects = self._objects() if self._objects else None
        if self.assist.drop is not None and str(tt.get("destination") or "person").startswith("person"):
            self.assist.drop = self.walker.to_pelvis(home.HANDOVER)     # the hand-over point is fixed in the room
        targets, post, events = self.assist.update(self.t, targets, self.grasp_points(self.arm_cmd), objects)
        for ev in events:
            log.info("grasp assist: %s (%s)", ev, tt)
            if self.on_progress:
                self.on_progress(tt["request_id"], ev)
        return targets, post

    def pov_jpeg(self, quality=70, overlay=True):
        with self._pov_lock:
            return self._pov_jpeg(quality, overlay)

    def _pov_jpeg(self, quality, overlay):
        """The robot's head camera as JPEG bytes (None before the first tick). On the mock backend the render
        has no arms, so the hands are drawn in: every palm / finger point projected into the image."""
        import io
        from PIL import Image, ImageDraw
        obs = getattr(self, "obs", None)
        if obs is None or "head_rgb" not in obs:
            return None
        # Called from the camera thread, never from the control loop (a slow encode there trips the safety
        # watchdog). So: copies of the commands, and a kinematics model of its own.
        if not hasattr(self, "_pov_kin"):
            self._pov_kin = ArmKinematics(self.cfg)
        kin, arm_q, hand_q = self._pov_kin, self.arm_cmd.copy(), self.hand_cmd.copy()
        img = Image.fromarray(np.asarray(obs["head_rgb"], np.uint8))
        cal = getattr(self, "_cal", None)
        if cal is not None and overlay:
            d = ImageDraw.Draw(img)
            for s, col in (("left", (93, 202, 165)), ("right", (197, 155, 245))):
                uvz = cal.project("head", kin.hand_points(s, arm_q, hand_q))
                pts = [(float(u), float(v)) for u, v, z in uvz if z > 0.05]
                for u, v in pts:
                    d.ellipse([u - 7, v - 7, u + 7, v + 7], fill=col, outline=(20, 20, 20))
                g = cal.project("head", kin.grasp_pose(s, arm_q)[0])
                if g[2] > 0.05:
                    d.ellipse([g[0] - 12, g[1] - 12, g[0] + 12, g[1] + 12], outline=(255, 210, 63), width=3)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    def _lag_cm(self):
        """How far the robot's right/left hands trail the helper's (largest of the two), for the helper."""
        if not self._targets:
            return None
        g = self.grasp_points(self.arm_cmd)
        return round(100 * max(float(np.linalg.norm(g[s] - self._targets[s])) for s in SIDES), 1)

    def run(self, seconds=None, source=None):
        """source: optional in-process mocap source with .at(t, seq)."""
        n = None if seconds is None else int(seconds * HZ)
        while n is None or self.tick_i < n:
            if source is not None:
                self.push(source.at(self.t, self.tick_i))
            self.tick()

    def close(self):
        self.ik.close()
        self.env.close()
        if self._log:
            self._log.close()


def task_target(msg):
    """A `target` message -> the node's task: {"request_id", "name", "color", "shape", "destination"}.
    The item may be given by name ("pill bottle") or by colour + shape."""
    item = home.BY_NAME.get(str(msg.get("name") or "").lower())
    color, shape = (item["color"], item["shape"]) if item else (msg.get("color"), msg.get("shape"))
    return {"request_id": msg.get("request_id"), "name": home.name_of(color, shape), "color": color,
            "shape": shape, "destination": msg.get("destination") or home.DESTINATION}


class SceneFeed:
    """Object poses from the simulator's scene service, polled on a thread (the call can block for seconds;
    the control loop only reads the newest answer). Uses $AALTO_SCENE_PORT, so it follows the sim instance."""

    def __init__(self, period=0.2):
        self._objs, self.period = None, period
        threading.Thread(target=self._run, name="scene", daemon=True).start()

    def latest(self):
        return self._objs

    def _run(self):
        from aaltonomous.sim.scene_client import SceneClient
        client = None
        while True:
            try:
                if client is None:
                    client = SceneClient(timeout_ms=3000)
                    have = {(o["color"], o["shape"]) for o in client.state()["objects"] if o.get("active")}
                    if have != {(o["color"], o["shape"]) for o in home.LAYOUT}:
                        r = client.reset(home.layout())
                        log.info("home items placed on the simulator's table: %s", r)
                self._objs = [o for o in client.state()["objects"] if o.get("active")]
            except Exception as e:      # sim restarting, service busy: keep the last answer, try again
                log.warning("scene service: %s", e)
            time.sleep(self.period)


# ---------------------------------------------------------------------- network
def serve(node: RobotNode, host: str, port: int, telemetry_every=2):     # 15 a second: the headset page lines its view up with the hips' heading from this
    """WebSocket endpoint in a background thread. Any client may send pose messages; every client receives
    telemetry (every 3rd tick = 10 Hz). One socket serves the mocap bridge and, later, the helper station."""
    import websockets

    loop = asyncio.new_event_loop()
    clients = set()

    async def handler(ws):
        clients.add(ws)
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    if msg.get("type") == "pose":
                        node.push(WristPose.from_dict(msg))
                    elif msg.get("type") == "point":        # where a controller points; its trigger picks / lets go
                        node.point(msg.get("yaw", 0.0), msg.get("pitch", 0.0), bool(msg.get("press")))
                    elif msg.get("type") == "drive":        # controller joysticks: walk the robot about
                        node.drive(msg.get("vx", 0.0), msg.get("vy", 0.0), msg.get("wz", 0.0))
                    elif msg.get("type") == "pov_ack":      # the page has shown the last camera frame
                        acks.add(ws)
                        waiting.pop(ws, None)
                    elif msg.get("type") == "target":       # set by hand (worker station, tests) when
                        node.task_target = task_target(msg)   # Nemotron is unavailable
                        log.info("task target (manual): %s", node.task_target)
                    elif msg.get("type") == "scene" and node.scene is not None:    # stage manager: jump in the story
                        node.scene.cue(str(msg.get("cue")))
                    elif msg.get("type") == "cancel":
                        node.task_target = None
                    elif msg.get("type") == "record" and node.recorder is not None:    # start / stop a demonstration
                        node.recorder.start(msg.get("task")) if msg.get("on") else node.recorder.stop(msg.get("success"))
                    elif msg.get("type") == "clutch":
                        node.clutch_on = bool(msg.get("on"))
                        log.info("clutch %s", "on" if node.clutch_on else "off")
                except (ValueError, KeyError, TypeError) as e:
                    log.warning("bad message dropped: %s", e)
        finally:
            clients.discard(ws)

    def on_telemetry(tel):
        if tel["tick"] % telemetry_every or not clients:
            return
        data = json.dumps(tel)

        def send():
            now = time.monotonic()
            for ws in list(clients):
                if ws in acks and now - waiting.get(ws, now) > 0.7:
                    continue                    # this page's link is behind: do not add to its queue
                asyncio.ensure_future(_safe_send(ws, data, "telemetry"))
        loop.call_soon_threadsafe(send)

    # Flow control for the camera. The link from the pod can be slower than the picture stream (160 KB/s
    # measured on 2026-11-14), and an SSH tunnel buffers megabytes before a send ever blocks, so "newest wins"
    # below is not enough: every page answers each frame with {"type": "pov_ack"} and gets the next one only
    # then (or after 3 s, for pages and scripts that never answer).
    acks, waiting = set(), {}       # clients that have ever acked; client -> when its unanswered frame was sent

    def send_pov(msg):
        now = time.monotonic()
        for ws in list(clients):
            if now - waiting.get(ws, -1e9) < 3.0:
                continue
            waiting[ws] = now
            asyncio.ensure_future(_safe_send(ws, msg, "pov"))

    busy = set()    # (client, kind) with a send still in flight

    async def _safe_send(ws, data, kind):
        """Newest wins: a client that has not finished receiving the previous message of this kind (slow page,
        slow link) skips this one. Queueing instead builds a backlog that grows without limit, delays the
        picture by minutes and clogs the tunnel for everyone."""
        key = (ws, kind)
        if key in busy:
            return
        busy.add(key)
        try:
            await asyncio.wait_for(ws.send(data), 5.0)
        except Exception:
            clients.discard(ws)
            acks.discard(ws)
            waiting.pop(ws, None)
            asyncio.ensure_future(ws.close())
        finally:
            busy.discard(key)

    async def main():
        async with websockets.serve(handler, host, port, max_size=2 ** 20):
            await asyncio.Future()

    def camera(period=0.15):        # the robot's camera, about 7 frames a second, for the worker's view
        import base64
        last, last_t = None, 0.0
        while True:
            t0 = time.monotonic()
            try:
                jpeg = node.pov_jpeg(quality=50) if clients else None
                if jpeg is not None and jpeg == last and t0 - last_t < 1.0:
                    jpeg = None                         # nothing moved: one frame a second is enough
                if jpeg is not None:
                    last, last_t = jpeg, t0
                    node.last_pov = jpeg
                    msg = json.dumps({"type": "pov", "jpeg_b64": base64.b64encode(jpeg).decode()})
                    loop.call_soon_threadsafe(send_pov, msg)
            except Exception:
                log.exception("camera frame failed")
            time.sleep(max(0.01, period - (time.monotonic() - t0)))

    node.listeners.append(on_telemetry)
    threading.Thread(target=camera, name="camera", daemon=True).start()
    threading.Thread(target=lambda: loop.run_until_complete(main()), name="ws", daemon=True).start()


def connect_session(node: RobotNode, url: str):
    """Background link to the session server as the robot: receives the task target, reports progress."""
    from websockets.sync.client import connect

    def loop():
        while True:
            try:
                with connect(url) as ws:
                    log.info("session server connected: %s", url)
                    node.on_progress = lambda rid, state: ws.send(json.dumps(
                        {"type": "progress", "request_id": rid, "state": state}))
                    for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "target":
                            node.task_target = task_target(msg)
                            log.info("task target: %s", node.task_target)
                        elif msg.get("type") == "cancel":
                            node.task_target = None
                        elif msg.get("type") == "frame_request":    # the caller asked for something: what the robot sees
                            import base64
                            jpeg = node.pov_jpeg(quality=85, overlay=False)   # no drawn-in hands for Nemotron
                            ws.send(json.dumps({"type": "frame", "id": msg.get("id"),
                                                "jpeg_b64": base64.b64encode(jpeg).decode() if jpeg else None}))
            except Exception as e:
                node.on_progress = None
                log.warning("session server link lost (%s); retrying", e)
                time.sleep(1.0)
    threading.Thread(target=loop, name="session", daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="mock", choices=["mock", "sim"])
    ap.add_argument("--source", default=None, help="in-process mocap source (synthetic); omit to use --listen")
    ap.add_argument("--listen", type=int, default=None, help="WebSocket port for the bridge / helper station")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--scale", type=float, default=0.75, help="robot hand travel per metre of human hand travel")
    ap.add_argument("--session-url", default=None, help="e.g. ws://127.0.0.1:8000/ws/robot?session=default")
    ap.add_argument("--log", default=None, help="append telemetry as JSON lines to this file")
    ap.add_argument("--unreal-udp", default=None, metavar="HOST:PORT[,HOST:PORT]",
                    help="send the robot's joints and the items to Unreal every tick (see unreal_stream.py)")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="record demonstrations under DIR when told to (see recorder.py)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("aaltonomous.safety").setLevel(logging.ERROR)   # clips are reported in the summary

    node = RobotNode(backend=args.backend, realtime=True, scale=args.scale, log_path=args.log)
    source = None
    if args.source:
        from .sources import SOURCES
        source = SOURCES[args.source]()
    if args.listen:
        serve(node, args.host, args.listen)
        log.info("listening on ws://%s:%d", args.host, args.listen)

    if args.session_url:
        connect_session(node, args.session_url)
    if args.unreal_udp:
        from . import unreal_stream
        unreal_stream.attach(node, args.unreal_udp)
        log.info("sending robot state for Unreal to udp://%s", args.unreal_udp)
    if args.record:
        from . import recorder
        recorder.attach(node, args.record)
        log.info("recorder ready, episodes go to %s", args.record)

    def report(tel):
        if tel["tick"] % 30 == 0:
            tgt = tel["target"].get("right")
            got = np.round(node.grasp_points(node.arm_cmd)["right"], 3).tolist()
            log.info("t=%5.1f %-7s safety=%s clipped=%d right hand at %s target %s", tel["t"], tel["state"],
                     tel["safety_mode"], tel["clipped_steps"], got, tgt)
    node.listeners.append(report)
    try:
        node.run(seconds=args.seconds, source=source)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("done: %s", node.env.filter.summary())
        if node.recorder is not None:
            node.recorder.close()
        node.close()


if __name__ == "__main__":
    main()
