"""Step 1 checks, on AaltoEnv's mock backend (no GPU, no suit).
Run from remote-hands/:  .venv/bin/python -m unittest discover -s tests -v"""
import logging
import time
import unittest

import numpy as np

from remote_hands.messages import WristPose
from remote_hands.robot_node import HZ, RobotNode
from remote_hands.sources import REST, SyntheticSource

logging.getLogger("aaltonomous.safety").setLevel(logging.ERROR)


def homed(**kw):
    node = RobotNode(backend="mock", sync_ik=True, **kw)
    while node.state == "homing":
        node.tick()
    return node


def pose(seq, right_offset=(0, 0, 0), clutch=True, grip=0.0):
    return WristPose(seq=seq, t=seq / HZ, left=tuple(REST["left"]),
                     right=tuple(REST["right"] + np.asarray(right_offset, float)), grip=(0.0, grip), clutch=clutch)


class RobotNodeTest(unittest.TestCase):
    def test_scripted_pick_and_place_tracks_without_clips(self):
        node, src = homed(), SyntheticSource()
        t0, at = node.t, {}
        for k in range(int(src.period * HZ)):
            node.push(src.at(node.t - t0, k))
            tel = node.tick()
            if tel["state"] == "driving":
                at[round(node.t - t0, 2)] = (node.grasp_points(node.arm_cmd)["right"], np.array(tel["target"]["right"]),
                                             node.hand_cmd.copy())
        s = node.env.filter.summary()
        self.assertEqual((s["mode"], s["clipped_steps"]), ("ok", 0), s)
        # end of "wait for the robot" over the object (t = 4.6) and over the bin (t = 8.5)
        for t, want in ((4.5, (0.33, -0.17, 0.035)), (8.4, (0.45, -0.26, 0.17))):
            got, target, _ = at[min(at, key=lambda x: abs(x - t))]
            self.assertLess(np.linalg.norm(got - target), 0.01, f"t={t}: hand {got} target {target}")
            self.assertLess(np.linalg.norm(target - np.array(want)), 0.01, f"t={t}: target {target}")
        closed = node.postures["closed"][7:]
        self.assertLess(np.abs(at[min(at, key=lambda x: abs(x - 5.5))][2][7:] - closed).max(), 0.05, "hand not closed")
        node.close()

    def test_human_speed_never_reaches_the_safety_filter(self):
        """A 0.3 m hand jump in one frame: the command still changes by at most arm_vel * dt per tick."""
        node = homed()
        node.push(pose(0)); node.tick()
        prev, worst = node.arm_cmd.copy(), 0.0
        for k in range(1, 90):
            node.push(pose(k, right_offset=(0.25, -0.05, -0.25)))
            node.tick()
            worst = max(worst, np.abs(node.arm_cmd - prev).max() * HZ)
            prev = node.arm_cmd.copy()
        self.assertLessEqual(worst, node.arm_vel + 1e-6)
        self.assertGreater(worst, 0.1, "the arm did not move")
        self.assertEqual(node.env.filter.summary()["counts"].get("arm_velocity", 0), 0)
        node.close()

    def test_mocap_dropout_holds_and_resumes_without_a_jump(self):
        node = homed()
        for k in range(45):
            node.push(pose(k, right_offset=(0.05, 0.0, -0.10)))
            node.tick()
        held = node.arm_cmd.copy()
        for _ in range(int(3 * HZ)):                # 3 s of silence: three times the filter's stop timeout
            tel = node.tick()
        self.assertEqual(tel["state"], "idle")
        self.assertEqual(tel["safety_mode"], "ok")  # never "stale" or "safe_stop": the node kept commanding
        self.assertLess(np.abs(node.arm_cmd - held).max(), 0.05)    # stopped within a tick or two
        before = node.arm_cmd.copy()
        node.push(pose(999, right_offset=(0.30, 0.2, 0.2)))        # helper is somewhere else entirely now
        node.tick()
        self.assertLess(np.abs(node.arm_cmd - before).max(), 1e-3, "re-engaging must not move the robot")
        node.close()

    def test_clutch_open_holds(self):
        node = homed()
        for k in range(30):
            node.push(pose(k)); node.tick()
        before = node.arm_cmd.copy()
        for k in range(30, 60):
            node.push(pose(k, right_offset=(0.2, 0.0, -0.2), clutch=False))
            tel = node.tick()
        self.assertEqual(tel["state"], "idle")
        self.assertLess(np.abs(node.arm_cmd - before).max(), 1e-6)
        node.close()

    def test_out_of_reach_target_stops_at_the_edge_and_comes_back(self):
        node = homed()
        node.push(pose(0)); node.tick()                            # engage at rest
        for k in range(1, 150):
            node.push(pose(k, right_offset=(1.0, -1.0, 0.5)))     # far outside anything the arm can do
            node.tick()
            self.assertGreaterEqual(node._lowest(node.arm_cmd, node.hand_cmd), node.floor_z - 1e-3)
        self.assertEqual(node.env.filter.summary()["mode"], "ok")
        for k in range(150, 330):
            node.push(pose(k))                                     # back to where the helper engaged
            tel = node.tick()
        got = node.grasp_points(node.arm_cmd)["right"]
        self.assertLess(np.linalg.norm(got - np.array(tel["target"]["right"])), 0.01)
        node.close()

    def test_hand_never_goes_below_the_table(self):
        """Push the hand down hard everywhere over the counter: no finger point passes the floor height."""
        node = homed()
        node.push(pose(0)); node.tick()
        k, low = 1, 9.0
        for off in ((0.0, 0.0, -1.0), (0.3, 0.1, -1.0), (0.3, -0.3, -1.0), (-0.3, 0.3, -1.0), (0.0, 0.0, 0.0)):
            for _ in range(90):
                node.push(pose(k, right_offset=off, grip=float(k % 60 < 30))); node.tick(); k += 1
                low = min(low, node._lowest(node.arm_cmd, node.hand_cmd))
        self.assertGreaterEqual(low, node.floor_z - 1e-3)
        self.assertEqual(node.env.filter.summary()["mode"], "ok")
        node.close()

    def test_slow_ik_does_not_starve_the_loop(self):
        """Real time, IK on its thread and made to take 0.4 s per solve: the loop still steps at 30 Hz."""
        node = RobotNode(backend="mock", realtime=True)
        solve = node.ik.kin.solve

        def slow(*a, **kw):
            time.sleep(0.4)
            return solve(*a, **kw)
        node.ik.kin.solve = slow
        t0, n = time.monotonic(), 0
        while time.monotonic() - t0 < 4.0:
            node.push(pose(n, right_offset=(0.05, 0.0, -0.05)))
            tel = node.tick()
            n += 1
        s = node.env.filter.summary()
        self.assertGreater(n, 110, "loop ran below 27.5 Hz")
        self.assertEqual(s["mode"], "ok")
        self.assertEqual(s["counts"].get("watchdog_hold", 0), 0)
        self.assertGreater(node.ik.solves, 0)
        node.close()


class MessageTest(unittest.TestCase):
    def test_round_trip(self):
        import json
        p = pose(7, right_offset=(0.1, 0.2, 0.3), grip=1.0)
        q = WristPose.from_dict(json.loads(p.to_json()))
        self.assertEqual((q.seq, q.clutch, q.grip), (7, True, (0.0, 1.0)))
        np.testing.assert_allclose(q.right, p.right)


if __name__ == "__main__":
    unittest.main()
