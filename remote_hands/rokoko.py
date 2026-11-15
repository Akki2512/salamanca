"""Rokoko Smartsuit Pro II as a mocap source, through the Rokoko Device SDK driver.

    suit --Wi-Fi/UDP--> `rokoko-sdk` driver (this laptop) --TCP 12276, RGMP v2--> RokokoSource --> bridge

The SDK does not give a skeleton. It gives, per sensor, an orientation quaternion (x, y, z, w) that rotates the
sensor's own axes into a north-east-down world frame (checked on our suit 2026-11-14: rotating each sensor's
measured acceleration by it gives (0, 0, -1 g) for all 17 sensors). So the wrist position is computed here:

    wrist = shoulder + upper_arm_length * (upper arm direction) + forearm_length * (forearm direction)

The direction of each arm segment in its sensor's axes is not known (it depends on how the suit sits), nor is
which way the helper faces. Two calibration poses, each held still for a second, give both:

    A  arms hanging straight down     -> each segment points straight down: its direction in the sensor's axes
    B  arms straight out in front     -> the forearms point forward: which way the helper faces, and how the
                                         lower-back sensors sit on the torso (so turning the body is followed)

Only arm directions are used, so sensor drift in heading (magnetic disturbance indoors) shows up as the hand
slowly wandering sideways; the node's clutch absorbs it each time the helper re-engages. Recalibrate when the
robot hand no longer moves the way the helper's does.
"""
from __future__ import annotations

import json
import logging
import shutil
import socket
import struct
import subprocess
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from .messages import SIDES, WristPose

log = logging.getLogger("remote_hands.rokoko")
DOWN = np.array([0.0, 0.0, 1.0])            # north-east-down world
SIZES = {"INT32": 4, "UINT32": 4, "INT64": 8, "UINT64": 8, "FLOAT": 4, "DOUBLE": 8}


def _size(data_type: str) -> int:
    base, _, dims = data_type.partition("[")
    n = int(np.prod([int(x) for x in dims.rstrip("]").split(",")])) if dims else 1
    return SIZES[base.strip()] * n


class RgmpReader:
    """Minimal RGMP v2 client (framing from the SDK's rgmp_core): keeps the newest orientation per sensor."""

    def __init__(self, host="127.0.0.1", port=12276):
        self.addr = (host, port)
        self.lock = threading.Lock()
        self.quats = {}             # sensor name (without "_sensor") -> (x, y, z, w)
        self.stamp = 0.0            # time.monotonic() of the newest data frame
        self.device = None
        self._slots = {}            # device_id -> {group_id: [(name, offset)]}
        self._stop = False
        threading.Thread(target=self._run, name="rgmp", daemon=True).start()

    def close(self):
        self._stop = True

    def snapshot(self):
        with self.lock:
            return dict(self.quats), self.stamp

    def _run(self):
        while not self._stop:
            try:
                with socket.create_connection(self.addr, timeout=3.0) as sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    log.info("connected to the Rokoko driver at %s:%d", *self.addr)
                    f = sock.makefile("rb")
                    while not self._stop:
                        head = f.read(8)
                        if len(head) < 8:
                            raise ConnectionError("driver closed the connection")
                        kind, n = struct.unpack("<II", head)
                        payload = f.read(n)
                        if len(payload) < n:
                            raise ConnectionError("driver closed the connection")
                        if kind == 1:
                            self._definition(json.loads(payload.decode("utf-8")))
                        elif kind == 2:
                            self._data(payload)
            except (OSError, ValueError) as e:
                log.warning("Rokoko driver link: %s (is `rokoko-sdk` running?); retrying", e)
                time.sleep(1.0)

    def _definition(self, d):
        groups = {}
        for gid, g in enumerate(d.get("groups") or []):
            off, slots = 0, []
            for s in g.get("streams") or []:
                if s.get("measure_type") == "ORIENTATION" and s.get("data_type") == "FLOAT[4]":
                    slots.append((str(s.get("target_frame", "")).removesuffix("_sensor"), off))
                off += _size(s["data_type"])
            groups[gid] = slots
        self._slots[d["device_id"]] = groups
        self.device = d.get("device_type")
        log.info("device %s (%s): %d orientation streams", d["device_id"], self.device,
                 sum(len(v) for v in groups.values()))

    def _data(self, payload):
        dev, gid, _ts = struct.unpack_from("<IIQ", payload, 0)
        slots = self._slots.get(dev, {}).get(gid)
        if not slots:
            return
        new = {name: struct.unpack_from("<4f", payload, 16 + off) for name, off in slots}
        with self.lock:
            self.quats.update(new)
            self.stamp = time.monotonic()


class ArmSolver:
    """Sensor orientations -> wrist positions in the helper's frame (x forward, y left, z up; metres).
    The frame turns with the helper's torso about the vertical only: leaning forward moves the hands forward."""

    NEED = [f"{s}_{part}" for s in SIDES for part in ("upper_arm", "lower_arm", "lower_back")]

    def __init__(self, upper_arm=0.30, forearm=0.35, shoulder_half_width=0.19):
        # forearm = elbow to the middle of the palm (the robot's grasp point is in its palm)
        self.upper, self.fore = upper_arm, forearm
        self.shoulder = {"left": np.array([0.0, shoulder_half_width, 0.0]),
                         "right": np.array([0.0, -shoulder_half_width, 0.0])}
        self.axis = None            # {sensor: segment direction in the sensor's axes}
        self.torso = None           # {side: helper frame expressed in that lower-back sensor's axes}

    @property
    def ready(self):
        return self.axis is not None and self.torso is not None

    def calibrate_down(self, quats):
        """Pose A: arms hanging straight down."""
        self.axis = {f"{s}_{part}": R.from_quat(quats[f"{s}_{part}"]).inv().apply(DOWN)
                     for s in SIDES for part in ("upper_arm", "lower_arm")}
        self.torso = None

    def calibrate_forward(self, quats):
        """Pose B: arms straight out in front. Returns how far below horizontal the forearms were (degrees),
        a check that the pose was held."""
        f = sum(R.from_quat(quats[f"{s}_lower_arm"]).apply(self.axis[f"{s}_lower_arm"]) for s in SIDES) / 2
        droop = float(np.degrees(np.arcsin(np.clip(f[2] / np.linalg.norm(f), -1, 1))))
        x = np.array([f[0], f[1], 0.0])
        if np.linalg.norm(x) < 0.3:
            raise ValueError("the arms were not out in front (still hanging down?)")
        x /= np.linalg.norm(x)
        up = -DOWN
        W = np.column_stack([x, np.cross(up, x), up])           # helper frame -> world
        self.torso = {s: R.from_quat(quats[f"{s}_lower_back"]).inv().as_matrix() @ W for s in SIDES}
        return droop

    def wrists(self, quats):
        # the helper frame now, from both lower-back sensors, then levelled (yaw only)
        Ws = [R.from_quat(quats[f"{s}_lower_back"]).as_matrix() @ self.torso[s] for s in SIDES]
        x = Ws[0][:, 0] + Ws[1][:, 0]
        x[2] = 0.0
        x /= np.linalg.norm(x)
        up = -DOWN
        W = np.column_stack([x, np.cross(up, x), up])
        out = {}
        for s in SIDES:
            u = R.from_quat(quats[f"{s}_upper_arm"]).apply(self.axis[f"{s}_upper_arm"])
            l = R.from_quat(quats[f"{s}_lower_arm"]).apply(self.axis[f"{s}_lower_arm"])
            out[s] = self.shoulder[s] + W.T @ (self.upper * u + self.fore * l)
        return out


def announce(text):
    """Tell the helper, who is standing away from the screen: print, and speak on macOS."""
    log.info(">>> %s", text)
    if shutil.which("say"):
        subprocess.Popen(["say", text])


class RokokoSource:
    name = "rokoko"

    def __init__(self, hz=30.0, host="127.0.0.1", port=12276, height=None, stale_s=0.15, reader=None, auto_calibrate=True):
        self.hz, self.stale_s = hz, stale_s
        kw = {} if height is None else {"upper_arm": 0.172 * height, "forearm": 0.146 * height + 0.09,
                                        "shoulder_half_width": 0.11 * height}
        self.solver = ArmSolver(**kw)
        self.reader = reader or RgmpReader(host, port)
        self._last = {s: (0.3, 0.2 if s == "left" else -0.2, -0.15) for s in SIDES}
        self._busy = False
        if auto_calibrate:
            self.calibrate()
        else:
            log.info(">>> put the suit on, then press Enter here to calibrate (two poses, 5 seconds each)")

    def _mean(self, seconds):
        """Average orientation of each needed sensor over `seconds`. Raises ValueError saying what went wrong."""
        got, t_end = {k: [] for k in ArmSolver.NEED}, time.monotonic() + seconds
        while time.monotonic() < t_end:
            quats, stamp = self.reader.snapshot()
            if time.monotonic() - stamp > self.stale_s or any(k not in quats for k in got):
                raise ValueError("No data from the suit.")
            for k in got:
                got[k].append(quats[k])
            time.sleep(0.02)
        out, worst = {}, (0.0, "")
        for k, v in got.items():
            r = R.from_quat(np.array(v))
            m = r.mean()
            worst = max(worst, (float(np.degrees((m.inv() * r).magnitude().max())), k))
            out[k] = m.as_quat()
        if worst[0] > 10.0:
            log.info("moved: %s turned %.0f degrees during the capture", worst[1], worst[0])
            raise ValueError("You moved.")
        return out

    def calibrate(self, settle=4.0):
        """Run the two-pose calibration once, on a thread (Enter in the bridge terminal starts it). Poses are
        sent with the clutch open until it has succeeded. A failed attempt says why and stops."""
        if self._busy:
            return
        self._busy = True
        self.solver.axis = self.solver.torso = None

        def run():
            try:
                announce("Calibration. Stand straight, arms hanging down by your sides. Hold still.")
                time.sleep(settle)
                a = self._mean(1.0)
                self.solver.calibrate_down(a)
                announce("Now both arms straight out in front, palms down. Hold still.")
                time.sleep(settle)
                droop = self.solver.calibrate_forward(self._mean(1.0))
                log.info("forearms were %.0f degrees below horizontal in the second pose", droop)
                announce("Calibrated. You can lower your arms.")
            except ValueError as e:
                self.solver.axis = self.solver.torso = None
                msg = "Your arms were not out in front." if "front" in str(e) else str(e)
                announce(f"Calibration failed. {msg} Press Enter to try again.")
            finally:
                self._busy = False
        threading.Thread(target=run, name="calibrate", daemon=True).start()

    def at(self, t: float, seq: int = 0) -> WristPose:
        quats, stamp = self.reader.snapshot()
        live = self.solver.ready and time.monotonic() - stamp < self.stale_s
        if live:
            try:
                w = self.solver.wrists(quats)
                self._last = {s: tuple(float(v) for v in w[s]) for s in SIDES}
            except KeyError:
                live = False
        return WristPose(seq=seq, t=t, left=self._last["left"], right=self._last["right"], grip=(0.0, 0.0),
                         clutch=live, source=self.name)


class RokokoStudioSource:
    """Wrist positions from Rokoko Studio's own body solve, instead of ArmSolver.

    In Studio: Livestream -> Custom -> forward to this laptop (127.0.0.1), port 14043, data format "JSON v3"
    (plain or LZ4). One UDP datagram per frame. scene.actors[0].body.<joint>.position / .rotation are world
    values in Unity's convention: metres, left-handed, y up, z forward (format checked on a live frame
    2026-11-14). Calibration and the actor profile are done in Studio. The wrists are expressed in a frame at
    the chest that turns with the shoulders about the vertical only, converted to ours (x forward, y left, z up)."""
    name = "rokoko-studio"

    def __init__(self, hz=30.0, port=14043, stale_s=0.25, **_):
        self.hz, self.stale_s = hz, stale_s
        self.lock = threading.Lock()
        self._wrists, self._stamp, self.frames, self.error = None, 0.0, 0, None
        self._last = {s: (0.3, 0.2 if s == "left" else -0.2, -0.15) for s in SIDES}
        self._curl, self._grip, self.meta, self._said = {}, {s: 0.0 for s in SIDES}, {}, 0.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        log.info(">>> waiting for Rokoko Studio: Livestream -> Custom, 127.0.0.1 port %d, format JSON v3", port)
        threading.Thread(target=self._run, name="studio-udp", daemon=True).start()

    @staticmethod
    def wrists_from_frame(frame):
        """One decoded JSON v3 frame -> {side: (x, y, z)} in our frame, or raises KeyError / IndexError."""
        body = frame["scene"]["actors"][0]["body"]
        ours = lambda p: np.array([p["z"], -p["x"], p["y"]], float)     # Unity (x right, y up, z fwd) -> ours
        # Facing direction from where the two shoulder joints are, not from a bone rotation: the bones' own
        # axes differ per joint in this format (seen live: chest is turned 180 degrees about x from the hip).
        left = ours(body["leftUpperArm"]["position"]) - ours(body["rightUpperArm"]["position"])
        left[2] = 0.0
        left /= np.linalg.norm(left)
        f = np.cross(left, [0.0, 0.0, 1.0])
        chest = ours(body["chest"]["position"])
        out = {}
        for s in SIDES:
            d = ours(body[f"{s}Hand"]["position"]) - chest
            out[s] = (float(d @ f), float(d @ left), float(d[2]))
        return out

    @staticmethod
    def body_from_frame(frame):
        """Hips and legs for walking: {"hip": position, "yaw": heading, "leg_len": metres, "legs": {side:
        {"thigh": unit vector, "shin": unit vector}}}. hip and yaw are in Studio's world turned to our axes
        (x = Unity z, y = -Unity x, z up), which does not turn with the worker; the leg directions are in the
        worker's own heading frame (x forward, y left, z up), heading taken from the line between the hip joints."""
        body = frame["scene"]["actors"][0]["body"]
        ours = lambda k: np.array([body[k]["position"]["z"], -body[k]["position"]["x"], body[k]["position"]["y"]], float)
        left = ours("leftUpLeg") - ours("rightUpLeg")
        left[2] = 0.0
        left /= np.linalg.norm(left)
        f = np.cross(left, [0.0, 0.0, 1.0])
        out = {"hip": np.round(ours("hip"), 4).tolist(), "yaw": round(float(np.arctan2(f[1], f[0])), 4), "legs": {}}
        lengths = []
        for s in SIDES:
            hipj, knee, foot = ours(f"{s}UpLeg"), ours(f"{s}Leg"), ours(f"{s}Foot")
            th, sh = knee - hipj, foot - knee
            lengths.append(float(np.linalg.norm(th) + np.linalg.norm(sh)))
            heading = lambda v: [round(float(v @ f), 4), round(float(v @ left), 4), round(float(v[2]), 4)]
            out["legs"][s] = {"thigh": heading(th / np.linalg.norm(th)), "shin": heading(sh / np.linalg.norm(sh))}
        out["leg_len"] = round(float(np.mean(lengths)), 3)
        # The chest's own axes (x forward, y left, z up), from the shoulders and the spine; in them, the way
        # each upper arm and forearm points ("arms": for arms that look like the worker's, mirror.py); and how
        # the chest is turned from the hips' heading ("torso": yaw, roll, pitch in radians, for the robot's waist:
        # yaw to the left, roll to the right side down, pitch forward).
        try:
            cl = ours("leftUpperArm") - ours("rightUpperArm")
            cl /= np.linalg.norm(cl)
            cu = ours("neck") - ours("spine")
            cu -= (cu @ cl) * cl
            cu /= np.linalg.norm(cu)
            cf = np.cross(cl, cu)
            chest = lambda v: [round(float(v @ cf), 4), round(float(v @ cl), 4), round(float(v @ cu), 4)]
            unit = lambda v: v / np.linalg.norm(v)
            out["arms"] = {s: {"upper": chest(unit(ours(f"{s}LowerArm") - ours(f"{s}UpperArm"))),
                               "fore": chest(unit(ours(f"{s}Hand") - ours(f"{s}LowerArm")))} for s in SIDES}
            up = np.array([0.0, 0.0, 1.0])
            R = np.array([[a @ b for b in (cf, cl, cu)] for a in (f, left, up)])     # chest axes in the heading frame
            out["torso"] = {"yaw": round(float(np.arctan2(-R[0, 1], R[1, 1])), 4), "roll": round(float(np.arcsin(np.clip(R[2, 1], -1, 1))), 4),
                            "pitch": round(float(np.arctan2(-R[2, 0], R[2, 2])), 4)}
        except (KeyError, ZeroDivisionError, FloatingPointError):
            pass
        # Which way the head faces, from the hips' heading (radians, left positive), from the head sensor's own
        # orientation: its forward (Unity +z) laid flat, helped by its up axis so that it stays steady when the
        # head nods far down or up. Both headings are in the suit's frame, so a drift of that frame cancels. The
        # headset page uses it to line the headset's heading up with the suit's (helper/quest.html).
        try:
            q = body["head"]["rotation"]
            x, y, z, w = q["x"], q["y"], q["z"], q["w"]
            ours_v = lambda v: np.array([v[2], -v[0], v[1]])
            fwd = ours_v([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
            up = ours_v([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
            flat = fwd[:2] - fwd[2] * up[:2]
            out["head_rel"] = round(float((np.arctan2(flat[1], flat[0]) - np.arctan2(f[1], f[0]) + np.pi) % (2 * np.pi) - np.pi), 4)
        except (KeyError, TypeError):
            pass
        return out

    @staticmethod
    def curl_from_frame(frame):
        """Gloves: how far each hand is closed, {side: 0 (flat) .. 1 (fist)}; a side with no finger joints in
        the frame is left out. Per finger (thumb excluded): straight-line distance from the knuckle to the
        fingertip over the finger's length along its joints. A straight finger gives 1, a curled one about 0.5."""
        actor = frame["scene"]["actors"][0]
        body, meta = actor["body"], actor.get("meta") or {}
        P = lambda k: np.array([body[k]["position"][a] for a in "xyz"], float)
        out = {}
        for s in SIDES:
            if meta and not (meta.get("hasGloves") or meta.get(f"has{s.capitalize()}Glove")):
                continue
            ratios = []
            for finger in ("Index", "Middle", "Ring", "Little"):
                chain = [f"{s}{finger}{seg}" for seg in ("Proximal", "Medial", "Distal", "Tip")]
                if not all(k in body for k in chain):
                    continue
                pts = [P(k) for k in chain]
                length = sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:]))
                if length > 1e-4:
                    ratios.append(np.linalg.norm(pts[-1] - pts[0]) / length)
            if ratios:
                out[s] = float(np.clip((0.95 - np.mean(ratios)) / 0.40, 0.0, 1.0))
        return out

    def _run(self):
        while True:
            data, _ = self.sock.recvfrom(1 << 20)
            try:
                if data[:4] == b"\x04\x22\x4d\x18":       # Studio's "JSON v3 (LZ4)" option: an LZ4 frame
                    import lz4.frame
                    data = lz4.frame.decompress(data)
                frame = json.loads(data)
                curl = self.curl_from_frame(frame)
                w = self.wrists_from_frame(frame)
                try:
                    self._body = self.body_from_frame(frame)
                except (KeyError, IndexError, TypeError, ZeroDivisionError):
                    self._body = None                   # no legs in the stream: arms only
                with self.lock:
                    self._wrists, self._stamp, self.error = w, time.monotonic(), None
                    self._curl, self.meta = curl, frame["scene"]["actors"][0].get("meta") or {}
                    self.frames += 1
                if time.monotonic() - self._said > 3.0:
                    self._said = time.monotonic()
                    log.info("studio: %d frames, body %s, gloves L %s R %s, hand closed %s, wrists L %s R %s",
                             self.frames, self.meta.get("hasBody"), self.meta.get("hasLeftGlove"),
                             self.meta.get("hasRightGlove"), {k: round(v, 2) for k, v in curl.items()},
                             np.round(w["left"], 2).tolist(), np.round(w["right"], 2).tolist())
            except (ValueError, KeyError, IndexError, TypeError) as e:
                if self.error is None:
                    log.warning("Studio frame not usable: %r (first 200 bytes: %r)", e, data[:200])
                self.error = e

    def at(self, t: float, seq: int = 0) -> WristPose:
        with self.lock:
            w, stamp, curl = self._wrists, self._stamp, dict(self._curl)
        live = w is not None and time.monotonic() - stamp < self.stale_s
        if live:
            self._last = w
            for s, c in curl.items():       # closed above 0.65, open again below 0.35: no chatter in between
                self._grip[s] = 1.0 if c > 0.65 else 0.0 if c < 0.35 else self._grip[s]
        return WristPose(seq=seq, t=t, left=self._last["left"], right=self._last["right"],
                         grip=(self._grip["left"], self._grip["right"]), body=getattr(self, "_body", None) if live else None,
                         clutch=live, source=self.name)
