"""Rokoko source checks without a suit: the arm calculation against made-up sensor mountings, and the RGMP
reader against a fake driver that speaks the SDK's wire format."""
import json
import socket
import struct
import threading
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from remote_hands.rokoko import ArmSolver, RgmpReader, RokokoSource

NED_FROM_BODY = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])    # body x fwd, y left, z up at heading 0


def suit(rng):
    """Random fixed mounting of every needed sensor on its segment."""
    return {k: R.random(random_state=rng) for k in ArmSolver.NEED}


def quats(mount, heading_deg, upper, lower, lean_deg=0.0):
    """upper / lower: {side: segment direction in the body frame}. Segment frame: x along the segment."""
    yaw = R.from_euler("z", heading_deg, degrees=True)             # NED: z down, positive = turning right
    out = {}
    for k, m in mount.items():
        side, part = k.split("_", 1)
        if part == "lower_back":
            seg = R.from_euler("y", -lean_deg, degrees=True)
        else:
            d = NED_FROM_BODY @ np.asarray((upper if part == "upper_arm" else lower)[side], float)
            seg, _ = R.align_vectors([d / np.linalg.norm(d)], [[1.0, 0, 0]])
        out[k] = (yaw * seg * m).as_quat()
    return out


both = lambda v: {"left": v, "right": v}


class SolverTest(unittest.TestCase):
    def test_wrists_follow_arm_directions_whatever_the_mounting_and_heading(self):
        rng = np.random.default_rng(3)
        for heading in (0.0, 75.0, -160.0):
            m, s = suit(rng), ArmSolver(upper_arm=0.30, forearm=0.35, shoulder_half_width=0.19)
            s.calibrate_down(quats(m, heading, both((0, 0, -1)), both((0, 0, -1))))
            droop = s.calibrate_forward(quats(m, heading, both((1, 0, -0.1)), both((1, 0, -0.1))))
            self.assertAlmostEqual(droop, np.degrees(np.arctan(0.1)), places=3)
            # elbows bent 90 degrees: upper arms down, forearms forward; then the helper turns 40 degrees
            for turn in (0.0, 40.0):
                w = s.wrists(quats(m, heading + turn, both((0, 0, -1)), both((1, 0, 0))))
                np.testing.assert_allclose(w["left"], [0.35, 0.19, -0.30], atol=1e-6)
                np.testing.assert_allclose(w["right"], [0.35, -0.19, -0.30], atol=1e-6)
            # right hand out to the right, left hand up; leaning the torso forward does not tilt the frame
            w = s.wrists(quats(m, heading, {"left": (0, 0, 1), "right": (0, -1, 0)},
                               {"left": (0, 0, 1), "right": (0, -1, 0)}, lean_deg=20))
            np.testing.assert_allclose(w["right"], [0, -0.19 - 0.65, 0], atol=1e-6)
            np.testing.assert_allclose(w["left"], [0, 0.19, 0.65], atol=1e-6)

    def test_second_pose_with_arms_down_is_rejected(self):
        m, s = suit(np.random.default_rng(1)), ArmSolver()
        down = quats(m, 0, both((0, 0, -1)), both((0, 0, -1)))
        s.calibrate_down(down)
        with self.assertRaises(ValueError):
            s.calibrate_forward(down)


def fake_driver(frames):
    """A TCP server that sends one stream definition and then data frames, as rokoko-sdk does."""
    srv = socket.create_server(("127.0.0.1", 0))
    names = sorted(frames[0])
    streams = []
    for n in names:                     # as on the real suit: other streams sit between the orientations
        streams += [{"data_type": "FLOAT[3]", "measure_type": "ANGULAR_VELOCITY", "target_frame": n + "_sensor"},
                    {"data_type": "FLOAT[4]", "measure_type": "ORIENTATION", "target_frame": n + "_sensor",
                     "reference_frame": "LTP_NED"},
                    {"data_type": "UINT32", "measure_type": "STATUS_FLAGS", "target_frame": n + "_sensor"}]
    definition = json.dumps({"protocol_name": "RGMP", "device_id": 7, "device_type": "SmartSuitPro",
                             "groups": [{"name": "default", "expected_rate_hz": 100, "streams": streams}]}).encode()

    def serve():
        conn, _ = srv.accept()
        conn.sendall(struct.pack("<II", 1, len(definition)) + definition)
        k = 0
        try:
            while True:
                fr = frames[min(k // 100, len(frames) - 1)]         # each entry lasts one second
                body = b"".join(struct.pack("<3f", 0, 0, 0) + struct.pack("<4f", *fr[n]) + struct.pack("<I", 35)
                                for n in names)
                payload = struct.pack("<IIQ", 7, 0, k * 10000) + body
                conn.sendall(struct.pack("<II", 2, len(payload)) + payload)
                k += 1
                time.sleep(0.01)
        except OSError:
            pass
    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1]


class SourceTest(unittest.TestCase):
    def test_reader_and_calibration_end_to_end(self):
        m = suit(np.random.default_rng(5))
        down = quats(m, 30, both((0, 0, -1)), both((0, 0, -1)))
        fwd = quats(m, 30, both((1, 0, 0)), both((1, 0, 0)))
        bent = quats(m, 30, both((0, 0, -1)), both((1, 0, 0)))
        # 0-2 s arms down, 2-4 s arms forward, then elbows bent
        port = fake_driver([down, down, fwd, fwd, bent])
        import remote_hands.rokoko as rk
        rk.announce = lambda text: None                              # no speech in tests
        src = RokokoSource(reader=RgmpReader(port=port), auto_calibrate=False)
        self.assertFalse(src.at(0.0).clutch)                         # not calibrated: the robot must hold
        time.sleep(0.3)
        src.solver.calibrate_down(src._mean(0.5))
        time.sleep(2.2)
        src.solver.calibrate_forward(src._mean(0.5))
        time.sleep(2.2)
        p = src.at(1.0, 5)
        self.assertTrue(p.clutch)
        np.testing.assert_allclose(p.right, [0.35, -0.19, -0.30], atol=1e-4)
        np.testing.assert_allclose(p.left, [0.35, 0.19, -0.30], atol=1e-4)
        src.reader.close()


class StudioTest(unittest.TestCase):
    def test_json_v3_frame_to_chest_relative_wrists(self):
        from remote_hands.rokoko import RokokoStudioSource
        P = lambda x, y, z: {"position": {"x": x, "y": y, "z": z}, "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}}
        # Unity: x right, y up, z forward. Facing +z: right hand 0.4 m ahead, 0.2 m to the right, 0.3 m below.
        body = {"chest": P(1.0, 1.3, 2.0), "rightHand": P(1.2, 1.0, 2.4), "leftHand": P(0.8, 1.0, 2.0),
                "leftUpperArm": P(0.8, 1.4, 2.0), "rightUpperArm": P(1.2, 1.4, 2.0)}
        w = RokokoStudioSource.wrists_from_frame({"scene": {"actors": [{"body": body}]}})
        np.testing.assert_allclose(w["right"], [0.4, -0.2, -0.3], atol=1e-9)
        np.testing.assert_allclose(w["left"], [0.0, 0.2, -0.3], atol=1e-9)
        # the actor turns to face +x: left shoulder is now towards +z. The same pose gives the same answer.
        body = {"chest": P(1.0, 1.3, 2.0), "rightHand": P(1.4, 1.0, 1.8), "leftHand": P(1.0, 1.0, 2.2),
                "leftUpperArm": P(1.0, 1.4, 2.2), "rightUpperArm": P(1.0, 1.4, 1.8)}
        w = RokokoStudioSource.wrists_from_frame({"scene": {"actors": [{"body": body}]}})
        np.testing.assert_allclose(w["right"], [0.4, -0.2, -0.3], atol=1e-9)
        np.testing.assert_allclose(w["left"], [0.0, 0.2, -0.3], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
