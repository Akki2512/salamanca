"""Demonstration recorder: writes what the robot did, tick by tick, and what its cameras saw, to disk.

    .venv/bin/python -m remote_hands.robot_node --backend mock --listen 8765 --record recordings/demos
    then send {"type": "record", "on": true, "task": "bring the glass"} / {"type": "record", "on": false}

One folder per episode (numbers go up, the next free one is taken):

    <root>/episode_000000/
      meta.json     {"episode": 0, "task": "<free text or null>", "started": "<ISO time>", "ended": "<ISO time>", "fps": 30,
                     "ticks": <n>, "success": null, "robot": "unitree_g1_29dof_hand14 (kinematic mock)",
                     "walk_gain": <float>, "cameras": {"robot_eyes": {"frames": <count>}, "person": {"frames": <count>}},
                     "notes": "<short limitations text>"}
                    (written once at the start with "ended": null and again, complete, at the end)
      steps.jsonl   one JSON object per control tick:
         {"i": <index in the episode from 0>, "t": <seconds since the episode's first tick>, "tick": <node tick>,
          "state": {"arm_q": [14], "hand_q": [14], "leg_q": [12], "waist_q": [3], "base": [x, y, yaw, dz]},
          "action": {"arm_q": [14], "hand_q": [14]},      arm: where the arms were being sent this tick (telemetry q_des);
                                                          hand: the hand command of this tick
          "hands": {"left": [x, y, z], "right": [x, y, z]},   grasp point of each hand in the robot's own (torso) frame
          "hands_room": {"left": [x, y, z], "right": [x, y, z]},   the same points in the room, z above the floor
          "closed": {"left": bool, "right": bool},
          "eye": [x, y, z],                               head camera position in the room, z = height above the floor
          "items": [{"name": str, "pos": [x, y, z], "held_by": "left" | "right" | null}],   room frame, z above the floor
          "label": {"task": str | null, "assist": str | null, "phase": str | null, "man": str | null,
                    "pushing": bool, "reach": str | null, "pointed": str | null, "trigger": bool},
          "scene": {...} | null,                          scene.DemoScene.state(): "chair" [x, y, heading], "man", "phase", ...
          "suit": {...} | null,                           the raw body dict from the suit for this tick
          "frame": {"robot_eyes": <int | null>, "person": <int | null>}}   index of the newest saved frame at this tick
      frames/robot_eyes/000000.jpg, 000001.jpg, ...
      frames/robot_eyes/index.jsonl   one line per saved frame: {"k": <index>, "t": <s since the episode's first tick>,
                                      "tick": <node tick when saved>, "head": [yaw_deg, pitch_deg, fov_deg, aspect] | null,
                                      "x_frame": <int | null>}
      frames/person/000000.jpg, ... and frames/person/index.jsonl (same; "head" and "x_frame" are always null there)

Joint order. arm_q: left 7 then right 7 (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw). hand_q: left 7
then right 7, as unreal_stream.HAND. leg_q: for side in (left, right): hip_pitch, hip_roll, hip_yaw, knee,
ankle_pitch, ankle_roll. waist_q: yaw, roll, pitch. Angles in radians, lengths in metres.
Room frame: origin on the floor under the robot's starting place, x forward, y to the robot's left, z up.

The joint state is the command after the safety filter (the mock backend is kinematic: it has nothing to measure).
label.task is the name of the item the robot is fetching; the free text given to start() is in meta.json.

The pictures come from the Unreal editor over HTTP (<frames_url>/eyes.jpg and /person.jpg), fetched by two
background threads, about 15 and 5 a second, only while an episode is running. If Unreal does not answer, the
episode is recorded without pictures (frame indices null). eyes.jpg is asked for with NO query string: a query
steers the live headset view.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
import time
import urllib.request

import numpy as np

from . import home
from .messages import SIDES
from .walk import GAIN, LEG_JOINTS, PELVIS_HEIGHT, WAIST_JOINTS

log = logging.getLogger("remote_hands.recorder")
FRAMES_URL = "http://127.0.0.1:8011"
CAMERAS = {"robot_eyes": ("eyes.jpg", 15.0), "person": ("person.jpg", 5.0)}     # name -> (file on the server, frames a second)
NOTES = ("Kinematic mock, no physics: state is the commanded joint position after the safety filter, not a measurement. "
         "action.hand_q equals state.hand_q. Items stick to a closing hand and drop straight down. "
         "Camera frames are fetched on their own clock; steps point at the newest frame saved by then.")


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _plain(o):
    """For json.dumps: numpy numbers and arrays as plain ones."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _dump(o):
    return json.dumps(o, separators=(",", ":"), default=_plain)


class _Episode:
    """One episode being written. The tick thread writes the steps; the camera threads write the frames (under
    `lock`, so a frame is never written into an episode that has been closed)."""

    def __init__(self, root, number, task, fps):
        self.number, self.task, self.fps = number, task, fps
        self.dir = os.path.join(root, "episode_%06d" % number)
        os.makedirs(self.dir)
        self.started = _now()
        self.steps = open(os.path.join(self.dir, "steps.jsonl"), "w", buffering=1 << 16)
        self.n = 0                                  # steps written
        self.t0 = None                              # node time of the first step
        self.t, self.tick, self.wall = 0.0, None, time.monotonic()     # the newest step: episode time, node tick, when
        self.lock = threading.Lock()
        self.closed = False
        self.count = {c: 0 for c in CAMERAS}        # frames saved
        self.index = {}
        for c in CAMERAS:
            os.makedirs(os.path.join(self.dir, "frames", c))
            self.index[c] = open(os.path.join(self.dir, "frames", c, "index.jsonl"), "w")

    def meta(self, ended=None, success=None):
        return {"episode": self.number, "task": self.task, "started": self.started, "ended": ended, "fps": self.fps,
                "ticks": self.n, "success": success, "robot": "unitree_g1_29dof_hand14 (kinematic mock)",
                "walk_gain": GAIN, "cameras": {c: {"frames": self.count[c]} for c in CAMERAS}, "notes": NOTES}

    def write_meta(self, **kw):
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self.meta(**kw), f, indent=1)

    def save_frame(self, cam, jpeg, head, x_frame):
        with self.lock:
            if self.closed or self.tick is None:    # no step yet: there is no episode time to give the frame
                return
            k = self.count[cam]
            with open(os.path.join(self.dir, "frames", cam, "%06d.jpg" % k), "wb") as f:
                f.write(jpeg)
            # between two ticks the frame's time runs on with the wall clock, at most one tick far
            t = self.t + min(time.monotonic() - self.wall, 1.0 / self.fps)
            self.index[cam].write(_dump({"k": k, "t": round(t, 4), "tick": self.tick, "head": head, "x_frame": x_frame}) + "\n")
            self.count[cam] = k + 1

    def close(self, success):
        with self.lock:
            self.closed = True
            for f in self.index.values():
                f.close()
        self.steps.close()
        self.write_meta(ended=_now(), success=success)


class Recorder:
    def __init__(self, node, root_dir, frames_url=FRAMES_URL):
        from .robot_node import EYE, HZ             # here, not at the top: robot_node imports this module too
        self.node, self.root, self._eye, self.fps = node, str(root_dir), EYE, int(round(HZ))
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.Lock()               # guards _todo and the episode's start / end
        self._todo = []                             # ("start", task) / ("stop", success), done at the next tick
        self._ep = None
        self._quit = threading.Event()
        self.frames_url = frames_url.rstrip("/") if frames_url else None
        if self.frames_url:
            for cam in CAMERAS:
                threading.Thread(target=self._camera, args=(cam,), name="rec-" + cam, daemon=True).start()

    # ------------------------------------------------------------------ control (any thread)
    def start(self, task=None):
        """Begin a new episode at the next tick (one that is running is finished first)."""
        with self._lock:
            self._todo.append(("start", None if task is None else str(task)))

    def stop(self, success=None):
        """Finish the running episode at the next tick and write its meta.json."""
        with self._lock:
            self._todo.append(("stop", success))

    def restart(self):
        """The scene was reset: finish the running episode and begin a new one with the same task."""
        with self._lock:
            on, task = self._ep is not None, self._ep.task if self._ep else None
            for what, arg in self._todo:            # asked for but not yet done
                on, task = (True, arg) if what == "start" else (False, None)
            if on:
                self._todo += [("stop", None), ("start", task)]

    def status(self):
        ep = self._ep
        return {"on": ep is not None, "episode": ep.number if ep else None, "ticks": ep.n if ep else 0}

    def close(self):
        """Finish now (the node is shutting down) and stop the camera threads."""
        self._quit.set()
        with self._lock:
            self._todo.clear()
            self._end(None)

    # ------------------------------------------------------------------ one tick
    def _next_number(self):
        taken = [int(m.group(1)) for m in (re.fullmatch(r"episode_(\d{6})", n) for n in os.listdir(self.root)) if m]
        return max(taken) + 1 if taken else 0

    def _end(self, success):
        ep, self._ep = self._ep, None
        if ep is not None:
            ep.close(success)
            log.info("episode %d saved: %d ticks, %s frames, in %s", ep.number, ep.n, ep.count, ep.dir)

    def on_tick(self, tel):
        """Listener of the robot node: called once after every control tick."""
        with self._lock:
            todo, self._todo = self._todo, []
            for what, arg in todo:
                if what == "stop":
                    self._end(arg if arg is None else bool(arg))
                else:
                    self._end(None)
                    self._ep = _Episode(self.root, self._next_number(), arg, self.fps)
                    self._ep.write_meta()
                    log.info("recording episode %d (%s)", self._ep.number, arg or "no task given")
            ep = self._ep
            if ep is None:
                return
            if ep.t0 is None:
                ep.t0 = tel["t"]
            t = round(tel["t"] - ep.t0, 4)
            ep.steps.write(_dump(self.step(tel, ep.n, t, {c: ep.count[c] - 1 if ep.count[c] else None for c in CAMERAS})) + "\n")
            ep.n += 1
            ep.t, ep.tick, ep.wall = t, tel["tick"], time.monotonic()

    def step(self, tel, i, t, frame):
        """The steps.jsonl line for one telemetry message (see the module text)."""
        node = self.node
        w = node.walker
        up = np.array([0.0, 0.0, PELVIS_HEIGHT])        # room positions are kept relative to the standing pelvis
        room = lambda p: np.round(np.asarray(p, float) + up, 4).tolist()
        j = w.joints()
        arm_q, hand_q = np.round(node.arm_cmd, 5).tolist(), np.round(node.hand_cmd, 5).tolist()
        b = node.env.backend
        held = {h[0]: side for side, h in (getattr(b, "held", None) or {}).items() if h is not None}
        items = [{"name": home.name_of(o["color"], o["shape"]), "pos": room(o["pos"]), "held_by": held.get(k)}
                 for k, o in enumerate(getattr(b, "objects", None) or [])]
        sc = tel.get("scene") or {}
        return {"i": i, "t": t, "tick": tel["tick"],
                "state": {"arm_q": arm_q, "hand_q": hand_q, "leg_q": [float(j[n]) for n in LEG_JOINTS],
                          "waist_q": [float(j[n]) for n in WAIST_JOINTS], "base": tel["base"]},
                "action": {"arm_q": tel["q_des"], "hand_q": hand_q},
                "hands": tel["hands"],
                "hands_room": {s: room(w.to_home(node.grasp_points(node.arm_cmd)[s])) for s in SIDES},
                "closed": tel["closed"],
                "eye": room(w.to_home(self._eye)),
                "items": items,
                "label": {"task": (tel.get("task_target") or {}).get("name"), "assist": (tel.get("assist") or {}).get("state"),
                          "phase": sc.get("phase"), "man": sc.get("man"), "pushing": bool(sc.get("pushing")),
                          "reach": (sc.get("reach") or {}).get("stage"), "pointed": tel.get("pointed"),
                          "trigger": bool(tel.get("trigger"))},
                "scene": tel.get("scene"),               # the story's own state: the wheelchair's place, the man's pose, ...
                "suit": getattr(node, "_last_body", None),
                "frame": frame}

    # ------------------------------------------------------------------ camera frames (own threads)
    def _camera(self, cam):
        path, hz = CAMERAS[cam]
        url = self.frames_url + "/" + path          # no query string: a query on eyes.jpg steers the live headset view
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        last, warned = None, False
        while not self._quit.is_set():
            ep, t0, wait = self._ep, time.monotonic(), 1.0 / hz
            if ep is None:
                last, wait = None, 0.05             # not recording: ask for nothing
            else:
                try:
                    with opener.open(url, timeout=0.5) as r:
                        jpeg, xf, xh = r.read(), r.headers.get("X-Frame"), r.headers.get("X-Head")
                    key = xf if xf is not None else jpeg        # person.jpg has no frame counter: compare the picture
                    if jpeg and (key != last or ep.count[cam] == 0):
                        last = key
                        try:
                            head = [float(v) for v in xh.split(",")] if xh and cam == "robot_eyes" else None
                        except ValueError:                      # "flat": the view is not steered by a head
                            head = None
                        ep.save_frame(cam, jpeg, head, int(xf) if xf and xf.strip().lstrip("-").isdigit() else None)
                    warned = False
                except Exception as e:              # Unreal closed or busy: record without pictures, try again
                    if not warned:
                        log.warning("no %s frames from %s (%s); recording goes on without them", cam, url, e)
                        warned = True
                    wait = 1.0
            self._quit.wait(max(0.005, wait - (time.monotonic() - t0)))


def attach(node, root_dir, frames_url=FRAMES_URL):
    """Give `node` a recorder writing under `root_dir`. Nothing is recorded until start() (or a "record" message)."""
    rec = Recorder(node, root_dir, frames_url)
    node.recorder = rec
    node.listeners.append(rec.on_tick)
    return rec
