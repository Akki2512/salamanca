"""Recorder checks, on the mock backend with no Unreal: an episode's files, one step per tick, the trigger flag,
and camera frames from a small local web server standing in for the Unreal editor."""
import http.server
import io
import json
import logging
import os
import tempfile
import threading
import time
import unittest

import numpy as np

from remote_hands import home, recorder
from remote_hands.robot_node import EYE, RobotNode

logging.getLogger("aaltonomous.safety").setLevel(logging.ERROR)
logging.getLogger("remote_hands.node").setLevel(logging.WARNING)

STEP_KEYS = {"scene", "i", "t", "tick", "state", "action", "hands", "hands_room", "closed", "eye", "items", "label", "suit", "frame"}
LABEL_KEYS = {"task", "assist", "phase", "man", "pushing", "reach", "pointed", "trigger"}


def homed(root, frames_url=None):
    node = RobotNode(backend="mock", sync_ik=True)
    rec = recorder.attach(node, root, frames_url=frames_url)
    while node.state == "homing":
        node.tick()
    return node, rec


def aim_at(node, name):
    """Yaw and pitch (degrees) from the robot's eyes to that item, as a controller would point."""
    o = next(o for o in node._objects() if home.name_of(o["color"], o["shape"]) == name)
    v = np.asarray(o["pos"], float) + np.asarray(home.BY_NAME[name].get("aim", (0, 0, 0))) - EYE
    return float(np.degrees(np.arctan2(v[1], v[0]))), float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))


def read(ep, *path):
    with open(os.path.join(ep, *path)) as f:
        return [json.loads(line) for line in f]


class RecorderTest(unittest.TestCase):
    def test_episode_files_and_steps(self):
        with tempfile.TemporaryDirectory() as root:
            node, rec = homed(root)
            self.assertEqual(node.last["recording"], {"on": False, "episode": None, "ticks": 0})
            rec.start("bring the glass")
            aim = aim_at(node, "glass of water")
            ticks, press_tick = [], None
            for k in range(60):
                if 20 <= k < 34:
                    node.point(aim[0], aim[1], press=(k == 28))
                tel = node.tick()
                ticks.append(tel["tick"])
                if k == 28:
                    press_tick = tel["tick"]
            self.assertEqual(node.last["recording"], {"on": True, "episode": 0, "ticks": 59})    # read before this tick's step
            self.assertEqual(rec.status(), {"on": True, "episode": 0, "ticks": 60})
            rec.stop()
            node.tick()                             # the stop takes effect here; this tick is not recorded
            self.assertEqual(rec.status(), {"on": False, "episode": None, "ticks": 0})
            node.close()

            ep = os.path.join(root, "episode_000000")
            for p in ("meta.json", "steps.jsonl", "frames/robot_eyes/index.jsonl", "frames/person/index.jsonl"):
                self.assertTrue(os.path.isfile(os.path.join(ep, p)), p)
            steps = read(ep, "steps.jsonl")
            self.assertEqual(len(steps), 60)
            self.assertEqual([s["i"] for s in steps], list(range(60)))
            self.assertEqual([s["tick"] for s in steps], ticks)
            self.assertEqual(steps[0]["t"], 0.0)
            self.assertAlmostEqual(steps[-1]["t"], 59 / 30, places=3)
            for s in steps:
                self.assertEqual(set(s), STEP_KEYS)
                self.assertEqual(set(s["label"]), LABEL_KEYS)
                self.assertEqual([len(s["state"][k]) for k in ("arm_q", "hand_q", "leg_q", "waist_q", "base")], [14, 14, 12, 3, 4])
                self.assertEqual([len(s["action"][k]) for k in ("arm_q", "hand_q")], [14, 14])
                self.assertEqual([len(s[k][side]) for k in ("hands", "hands_room") for side in ("left", "right")], [3] * 4)
                self.assertEqual(len(s["eye"]), 3)
                self.assertEqual(s["frame"], {"robot_eyes": None, "person": None})
                self.assertIsNone(s["suit"])
            first = steps[0]
            self.assertAlmostEqual(first["eye"][2], 0.793 + EYE[2], places=3)      # standing: the eyes are 1.27 m up
            sweater = next(it for it in first["items"] if it["name"] == "sweater")
            self.assertAlmostEqual(sweater["pos"][2], 0.07, places=3)              # lying on the floor
            self.assertEqual({it["name"] for it in first["items"]}, set(home.NAMES))
            # the trigger: flagged on the tick it was handled, the glass pointed at, and the pick under way after it
            self.assertEqual([s["tick"] for s in steps if s["label"]["trigger"]], [press_tick])
            self.assertEqual(steps[28]["label"]["pointed"], "glass of water")
            self.assertEqual(steps[40]["label"]["task"], "glass of water")
            self.assertIn(steps[40]["label"]["assist"], ("approach", "descend", "closing", "holding"))
            self.assertEqual(steps[10]["label"]["task"], None)

            with open(os.path.join(ep, "meta.json")) as f:
                meta = json.load(f)
            self.assertEqual((meta["episode"], meta["task"], meta["ticks"], meta["fps"], meta["success"]),
                             (0, "bring the glass", 60, 30, None))
            self.assertEqual(meta["cameras"], {"robot_eyes": {"frames": 0}, "person": {"frames": 0}})
            self.assertTrue(meta["started"] and meta["ended"] and meta["notes"])
            self.assertIsInstance(meta["walk_gain"], float)

    def test_scene_reset_starts_a_new_episode_with_the_same_task(self):
        with tempfile.TemporaryDirectory() as root:
            node, rec = homed(root)
            node.reset_scene()                      # not recording: a reset starts nothing
            node.tick()
            self.assertFalse(rec.status()["on"])
            rec.start("tidy up")
            for _ in range(5):
                node.tick()
            node.reset_scene()
            for _ in range(3):
                node.tick()
            self.assertEqual(rec.status(), {"on": True, "episode": 1, "ticks": 3})
            rec.stop(success=True)
            node.tick()
            node.close()
            metas = []
            for n in (0, 1):
                with open(os.path.join(root, "episode_%06d" % n, "meta.json")) as f:
                    metas.append(json.load(f))
            self.assertEqual([(m["task"], m["ticks"], m["success"]) for m in metas], [("tidy up", 5, None), ("tidy up", 3, True)])
            self.assertEqual(len(read(root, "episode_000001", "steps.jsonl")), 3)

    def test_camera_frames_from_a_local_server(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (16, 12), (40, 90, 160)).save(buf, "JPEG")
        jpeg, asked = buf.getvalue(), []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                asked.append(self.path)
                eyes = self.path.startswith("/eyes.jpg")
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                if eyes:                            # a new frame every second request; the head only now and then
                    n = sum(p.startswith("/eyes.jpg") for p in asked)
                    self.send_header("X-Frame", str(100 + n // 2))
                    self.send_header("X-Head", "flat" if n < 3 else "12.5,-30,90,1.778")
                self.end_headers()
                self.wfile.write(jpeg)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as root:
                node, rec = homed(root, frames_url="http://127.0.0.1:%d" % srv.server_address[1])
                rec.start()
                for _ in range(45):
                    node.tick()
                    time.sleep(0.02)                # the mock ticks faster than 30 Hz: give the cameras time
                rec.stop()
                node.tick()
                rec.close()
                node.close()
                ep = os.path.join(root, "episode_000000")
                with open(os.path.join(ep, "meta.json")) as f:
                    meta = json.load(f)
                steps = read(ep, "steps.jsonl")
                eyes, person = read(ep, "frames", "robot_eyes", "index.jsonl"), read(ep, "frames", "person", "index.jsonl")
                self.assertGreaterEqual(len(eyes), 3)
                self.assertEqual(len(person), 1)                    # the same picture every time: saved once
                self.assertEqual(meta["cameras"], {"robot_eyes": {"frames": len(eyes)}, "person": {"frames": len(person)}})
                self.assertEqual([e["k"] for e in eyes], list(range(len(eyes))))
                xs = [e["x_frame"] for e in eyes]
                self.assertEqual(len(set(xs)), len(xs))             # only saved when X-Frame changed
                self.assertTrue(all(isinstance(x, int) for x in xs))
                self.assertIsNone(eyes[0]["head"])                  # "flat"
                self.assertEqual(eyes[-1]["head"], [12.5, -30.0, 90.0, 1.778])
                self.assertEqual((person[0]["head"], person[0]["x_frame"]), (None, None))
                tick_of = {s["tick"] for s in steps}
                for e in eyes:
                    self.assertIn(e["tick"], tick_of)
                    self.assertTrue(0.0 <= e["t"] <= steps[-1]["t"] + 0.04)
                    with open(os.path.join(ep, "frames", "robot_eyes", "%06d.jpg" % e["k"]), "rb") as f:
                        self.assertEqual(f.read(), jpeg)
                seen = [s["frame"]["robot_eyes"] for s in steps]
                self.assertIn(seen[-1], (len(eyes) - 1, len(eyes) - 2))     # one more may arrive after the last tick
                self.assertEqual([k for k in seen if k is not None], sorted(k for k in seen if k is not None))
                self.assertEqual(steps[-1]["frame"]["person"], 0)
                self.assertTrue(all("?" not in p for p in asked), asked)     # a query would steer the live headset view
        finally:
            srv.shutdown()
            srv.server_close()

    def test_unreal_not_reachable_records_without_frames(self):
        with tempfile.TemporaryDirectory() as root:
            node, rec = homed(root, frames_url="http://127.0.0.1:9")       # nothing listens there
            logging.getLogger("remote_hands.recorder").setLevel(logging.ERROR)
            rec.start()
            for _ in range(10):
                node.tick()
            rec.close()                             # shutting down finishes the episode without another tick
            node.close()
            logging.getLogger("remote_hands.recorder").setLevel(logging.NOTSET)
            steps = read(root, "episode_000000", "steps.jsonl")
            self.assertEqual(len(steps), 10)
            self.assertEqual(steps[-1]["frame"], {"robot_eyes": None, "person": None})


if __name__ == "__main__":
    unittest.main()
