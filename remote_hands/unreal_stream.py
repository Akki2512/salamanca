"""Robot state for Unreal Engine: one small JSON datagram per control tick (30 a second) over UDP.

Unreal only draws. The robot node stays the source of truth: it applies the G1's joint limits and reach, runs
grasp assist and decides where every item is (an item near a closing hand sticks to it; once it is let go at
the hand-over point the person has it and it stays there). Unreal sets the robot's joints and the items' positions from each packet; it simulates nothing.

    .venv/bin/python -m remote_hands.robot_node --backend mock --listen 8765 --unreal-udp 127.0.0.1:7777

Packet (UTF-8 JSON, under 2 kB):
    {"type": "robot_state", "v": 1, "seq": 123, "t": 4.1,
     "joints": {"left_shoulder_pitch_joint": 0.12, ...},   28 angles in radians: 14 arm + 14 hand joints,
                                                            names as in the G1 URDF (g1_body29_hand14)
     "hands": {"left": [x, y, z], "right": [x, y, z]},     grasp point of each hand, robot frame, metres
     "closed": {"left": false, "right": true},
     "items": [{"name": "water jug", "pos": [x, y, z], "held_by": "right" | "person" | null}, ...],
     "task": {"item": "water jug", "state": "holding", "side": "right"} | null,
     "handover": [x, y, z],                                where the robot holds the item out for the person
     "table_z": 0.01}                                      height of the table top
Robot frame: origin at the robot's pelvis (waist joint), x forward, y to the robot's LEFT, z up, metres,
right-handed. Unreal is left-handed in centimetres: UE (X, Y, Z) = (100 x, -100 y, 100 z) relative to the
pelvis, and a rotation about an axis changes sign with the handedness (try the sign per joint on the rig).
"""
from __future__ import annotations

import json
import socket

import numpy as np

from . import home

ARM = [f"{s}_{j}_joint" for s in ("left", "right") for j in
       ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
HAND = ([f"left_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")] +
        [f"right_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")])


def packet(node, tel) -> dict:
    """The packet for one telemetry message of `node` (see the module text)."""
    held = {}
    b = node.env.backend
    for side, h in (getattr(b, "held", None) or {}).items():
        if h is not None:
            held[h[0]] = side
    tt, a = tel.get("task_target"), tel.get("assist")
    walker = node.walker
    handover = walker.room_to_pelvis(home.HANDOVER)     # fixed in the room; here from the robot's pelvis
    given = node.__dict__.setdefault("_given", {})      # item index -> where in the room it was handed over
    items = []
    scene = getattr(node, "scene", None)
    for i, o in enumerate(getattr(b, "objects", None) or []):
        name = home.name_of(o["color"], o["shape"])
        # `doing` is the task grasp assist is really on: for one tick after a new request the telemetry still
        # shows the previous fetch's "done", which must not mark the new item as already handed over.
        doing = (node.assist.task or {}) if node.assist is not None else {}
        if tt and name == tt.get("name") == doing.get("name") and doing.get("request_id") == tt.get("request_id") \
                and (a or {}).get("state") in ("releasing", "done") and i not in given \
                and str(tt.get("destination") or "person").startswith("person"):     # put down, not handed over: it lies where it fell
            given[i] = np.asarray(o["pos"] if i in held else home.HANDOVER, float)
        room, by = (given[i], "person") if i in given else (np.asarray(o["pos"], float), held.get(i))
        items.append({"name": name, "pos": np.round(walker.room_to_pelvis(room), 4).tolist(), "held_by": by})
    # The sweater has two shapes in Unreal: lying ("sweater") and hanging from the hand ("sweater hanging", placed
    # with its swing by ext.items). The one not in use is parked far under the floor.
    hang = scene.hanging() if scene is not None else None
    away = [0.0, 0.0, -60.0]
    if scene is not None:
        for it in items:
            if it["name"] == "sweater" and hang:
                items.append({"name": "sweater hanging", "pos": it["pos"], "held_by": it["held_by"]})
                it["pos"] = away
                break
        else:
            items.append({"name": "sweater hanging", "pos": away, "held_by": None})
    task = {"item": tt.get("name"), "state": (a or {}).get("state"), "side": (a or {}).get("side")} if tt else None
    out = {"type": "robot_state", "v": 1, "seq": tel["tick"], "t": tel["t"],
           "joints": {**dict(zip(ARM, tel["q_cmd"])), **dict(zip(HAND, tel["hand_cmd"]))},
           "hands": tel["hands"], "closed": tel["closed"], "items": items, "task": task,
           "handover": np.round(handover, 4).tolist(),
           "tray": np.round(handover, 4).tolist(),       # old name, still read by the Unreal receiver
           "table_z": tel["view"]["counter_z"] - walker.dz}
    # Walking: where the pelvis is in the room (origin on the floor under the robot's starting place) and the
    # leg and waist joints. Every v1 position above is in the robot's own frame, which moves with it.
    out["ext"] = {"v": 1, "pelvis": walker.pelvis_world(), "joints": walker.joints()}
    if scene is not None:       # the wheelchair's place and the person's pose (remote_hands/scene.py)
        out["ext"]["scene"] = scene.state()
        if hang:
            out["ext"]["items"] = {"sweater hanging": hang}
    return out


def attach(node, endpoints):
    """Send a packet to every "HOST:PORT" in `endpoints` (a list, or one comma-separated string) after every tick."""
    if isinstance(endpoints, str):
        endpoints = endpoints.split(",")
    dests = []
    for e in endpoints:
        host, _, port = e.strip().rpartition(":")
        dests.append((host or "127.0.0.1", int(port)))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(tel):
        data = json.dumps(packet(node, tel), separators=(",", ":")).encode()
        for d in dests:
            sock.sendto(data, d)
    node.listeners.append(send)
    return sock
