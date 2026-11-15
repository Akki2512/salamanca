"""Step 3 checks: grasp assist on the mock backend with the Aalto kinematic world (objects stick to a closed hand).
The "helper" aims a few centimetres off, has no gloves (grip 0), and never signals a release."""
import logging
import math
import unittest

import numpy as np

from remote_hands import home
from remote_hands.messages import WristPose
from remote_hands.robot_node import HZ, RobotNode
from remote_hands.sources import REST

f = math.radians(home.WHEELCHAIR[2])
STAND = (home.WHEELCHAIR[0] + 0.80 * math.cos(-f), home.WHEELCHAIR[1] + 0.80 * math.sin(-f), math.pi - f)

logging.getLogger("aaltonomous.safety").setLevel(logging.ERROR)


def run_pick(want, miss=(0.03, -0.03), seconds=30.0, dest="person"):
    node = RobotNode(backend="mock", sync_ik=True)
    while node.state == "homing":
        node.tick()
    objs = node._objects()
    side_of = lambda o: "right" if o["pos"][1] < 0 else "left"
    # `want`: which side of the table the item is on (that hand takes it), or None for the first item
    ob = next(o for o in objs if want is None or side_of(o) == want)
    side = want or side_of(ob)
    start = np.array(ob["pos"], float)
    events = []
    node.on_progress = lambda rid, state: events.append(state)
    node.task_target = {"request_id": "r1", "color": ob["color"], "shape": ob["shape"], "destination": dest}
    node.tick()                              # the node picks the task up
    hand0 = node.grasp_points(node.arm_cmd)[side]
    # helper's plan in robot coordinates: over the object (off by `miss`), down, up; then (the robot having walked
    # over to the wheelchair) out to the hand-over point, a few centimetres off too
    way = [(1.0, hand0), (2.5, np.r_[start[:2] + miss, 0.06]), (6.0, np.r_[start[:2] + miss, 0.06]),
           (2.0, np.r_[start[:2] + miss, 0.18]), (3.0, None), (seconds, None)]
    k, t0, states = 0, node.t, []
    while node.t - t0 < seconds and "released" not in events:
        tt, prev = node.t - t0, hand0
        if tt >= 11.5 and not node.walker.moved:      # stands 80 cm in front of the chair, facing it
            node.walker.base[:] = STAND
        give = np.r_[node.walker.to_pelvis(home.HANDOVER)[:2] + (-0.04, 0.03), 0.16]
        way[4], way[5] = (3.0, give), (seconds, give)
        for dur, p in way:
            if tt < dur:
                robot_p = prev + (p - prev) * (tt / dur)
                break
            tt -= dur
            prev = p
        human = {s: np.array(REST[s], float) for s in ("left", "right")}
        human[side] = REST[side] + (robot_p - hand0) / node.clutch.scale
        node.push(WristPose(seq=k, t=k / HZ, left=tuple(human["left"]), right=tuple(human["right"]),
                            grip=(0.0, 0.0), clutch=True))
        tel = node.tick()
        states.append(tel["assist"]["state"])
        k += 1
    release_at = node.grasp_points(node.arm_cmd)[side]
    for _ in range(30):                      # let the hand finish opening
        node.push(WristPose(seq=k, t=k / HZ, left=tuple(human["left"]), right=tuple(human["right"]),
                            grip=(0.0, 0.0), clutch=True))
        node.tick(); k += 1
    end = np.array(next(o for o in node._objects() if o["name"] == ob["name"])["pos"], float)
    summary = node.env.filter.summary()
    scene = node.env.scene
    node.close()
    # handed over: let go at the hand-over point over the person's knees
    on_tray = bool(np.linalg.norm(np.array(release_at)[:2] - node.walker.to_pelvis(home.HANDOVER)[:2]) < 0.02)
    return events, states, on_tray, summary


class GraspAssistTest(unittest.TestCase):
    def check(self, side):
        events, states, in_bin, s = run_pick(side)
        self.assertEqual(events, ["in_progress", "grasped", "released"], states[-1])
        self.assertTrue(in_bin)
        self.assertEqual((s["mode"], s["clipped_steps"]), ("ok", 0), s)
        seen = [x for i, x in enumerate(states) if i == 0 or states[i - 1] != x]
        self.assertEqual(seen, ["free", "approach", "descend", "closing", "holding", "releasing", "done"])

    def test_first_item_is_handed_over(self):
        self.check(None)

    def test_right_hand_hands_over(self):
        self.check("right")

    def test_pulling_away_lets_go(self):
        node = RobotNode(backend="mock", sync_ik=True)
        while node.state == "homing":
            node.tick()
        ob = node._objects()[0]
        node.task_target = {"request_id": "r", "color": ob["color"], "shape": ob["shape"], "destination": "person"}
        a = node.assist
        hands = node.grasp_points(node.arm_cmd)
        near = {s: np.r_[np.array(ob["pos"])[:2] + 0.02, ob["pos"][2] + 0.03] for s in hands}
        a.set_task(node.task_target)
        a.update(0.0, near, hands, node._objects())
        self.assertEqual(a.state, "approach")
        far = {s: v + np.array([0.0, -0.3, 0.0]) for s, v in near.items()}       # pulled 30 cm to the side
        out, post, _ = a.update(0.1, far, hands, node._objects())
        self.assertEqual((a.state, post), ("free", {}))
        self.assertIs(out, far)
        node.close()


if __name__ == "__main__":
    unittest.main()
