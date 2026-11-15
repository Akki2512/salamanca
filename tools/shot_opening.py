"""The demo video's opening shot, robot side: the robot waits out of sight, then walks in and stands beside the man.

    python tools/shot_opening.py recordings/shot_opening_packets.jsonl [wait_seconds]
Writes one robot_state packet per 30 Hz tick (driven with the joystick path, no suit). The camera move and the
bell are in unreal/RemoteHandsBedroom/Scripts/render_shot.py; the cut is made by tools/cut_opening.py.
"""
import sys, math, json, logging
import numpy as np
logging.basicConfig(level=logging.WARNING)
logging.getLogger("aaltonomous.safety").setLevel(logging.ERROR)
from remote_hands.robot_node import RobotNode
from remote_hands import unreal_stream

node = RobotNode(backend="mock", sync_ik=True)
out = open(sys.argv[1], "w")
wait = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
n = [0]
sideways = [False]


def tick(k=1, drive=None):
    for _ in range(k):
        if drive: node.drive(*drive)
        p = unreal_stream.packet(node, node.tick())
        # Nobody is in the suit for this shot: the arms hang at the robot's sides and swing a little against the legs.
        legs = p["ext"]["joints"]
        if sideways[0]:          # the node's stride swings the legs forward and back; going sideways they open and close instead
            for side in ("left", "right"):
                swing = legs[f"{side}_hip_pitch_joint"]
                legs[f"{side}_hip_roll_joint"] = round(0.9 * swing, 4)
                legs[f"{side}_hip_pitch_joint"] = round(-0.5 * legs[f"{side}_knee_joint"], 4)      # the knee lifts the foot straight up
                legs[f"{side}_ankle_pitch_joint"] = round(-0.5 * legs[f"{side}_knee_joint"], 4)
        for side, other, sign in (("left", "right", 1.0), ("right", "left", -1.0)):
            p["joints"].update({f"{side}_shoulder_pitch_joint": round(0.25 + 0.7 * legs[f"{other}_hip_pitch_joint"], 4), f"{side}_shoulder_roll_joint": 0.18 * sign,
                                f"{side}_shoulder_yaw_joint": 0.0, f"{side}_elbow_joint": 1.05, f"{side}_wrist_roll_joint": 0.0,
                                f"{side}_wrist_pitch_joint": 0.0, f"{side}_wrist_yaw_joint": 0.0})
        out.write(json.dumps(p) + "\n"); n[0] += 1


def walk_to(x, y, yaw=None, tol=0.08, speed=0.7):
    for _ in range(3000):
        b = node.walker.base
        d = np.array([x, y]) - b[:2]; dist = float(np.linalg.norm(d))
        if dist < tol: break
        e = (math.atan2(d[1], d[0]) - b[2] + math.pi) % (2 * math.pi) - math.pi
        tick(drive=(speed * max(0.0, math.cos(e)) * min(1, dist / 0.3 + 0.2), 0.0, float(np.clip(2.5 * e, -1.5, 1.5))))
    if yaw is not None:
        for _ in range(600):
            e = (yaw - node.walker.base[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(e) < 0.04: break
            tick(drive=(0.0, 0.0, float(np.clip(2.0 * e, -1.2, 1.2))))


while node.state == "homing": node.tick()
# The wide end of the camera move (render_shot.py), in the room's frame: where it stands and the way it looks.
CAM, LOOK = np.array([-3.78, 1.017]), np.array([-0.875, 0.485])
RIGHT = np.array([0.485, 0.875])                                           # screen right
stop = CAM + 0.50 * LOOK                                                   # straight in front of the lens, between it and the man
wings = stop + 0.80 * RIGHT                                                # out of the picture, short of the coffee table
yaw = math.atan2(LOOK[1], LOOK[0])                                         # its back to the camera
walk_to(-1.3, 1.6, tol=0.25); walk_to(-3.3, 1.9, tol=0.2); walk_to(wings[0], wings[1], yaw, tol=0.06)
tick(int(wait * 30))
start = n[0]
sideways[0] = True
for _ in range(75):                                                       # sidestep in, facing the same way throughout
    b = node.walker.base
    d = stop - b[:2]
    c, s_ = math.cos(b[2]), math.sin(b[2])
    v = np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]])
    v = v / max(np.linalg.norm(v), 0.25) * 0.38
    e = (yaw - b[2] + math.pi) % (2 * math.pi) - math.pi
    tick(drive=(float(v[0]), float(v[1]), float(np.clip(2.0 * e, -1, 1))))
arrived = n[0]
sideways[0] = False
tick(30 * 8)
print(json.dumps({"ticks": n[0], "wings": np.round(wings, 3).tolist(), "wanted": np.round(stop, 3).tolist(), "step_from_s": start / 30, "arrived_s": arrived / 30,
                  "stop": np.round(node.walker.base, 3).tolist()}))
json.dump({"step_from_s": start / 30, "arrived_s": arrived / 30}, open(sys.argv[1] + ".meta.json", "w"))
node.close()
