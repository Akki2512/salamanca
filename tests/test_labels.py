"""Checks of remote_hands/labels.py on made-up cases with known answers (no recorded frames needed)."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from remote_hands import home, labels

W, H = 1280, 1024
EYE = [0.0, 0.0, 1.5]


def step(items=(), base=(0.0, 0.0, 0.0, 0.0), eye=EYE, tick=0, **more):
    return {"i": tick, "t": tick / 30, "tick": tick, "state": {"base": list(base)}, "eye": list(eye),
            "items": [dict(i) for i in items], "frame": {"robot_eyes": None, "person": None}, **more}


def meta(yaw=0.0, pitch=0.0, fov=110.0, aspect=1.25, k=0, tick=0):
    return {"k": k, "t": tick / 30, "tick": tick, "head": [yaw, pitch, fov, aspect], "x_frame": k}


def clock(x, y, z, held=None):
    """An alarm clock whose visual middle (pos + aim) is at x, y, z."""
    a = home.BY_NAME["alarm clock"]["aim"]
    return {"name": "alarm clock", "pos": [x - a[0], y - a[1], z - a[2]], "held_by": held}


def one(items, m=None, **kw):
    b = labels.boxes(step(items, **kw), m or meta(), W, H)
    return b[0] if b else None


class Projection(unittest.TestCase):
    def test_straight_ahead_is_the_image_centre(self):
        cam = labels.camera(step(), meta())
        u, v, d = labels.project([2.0, 0.0, 1.5], cam, W, H)
        self.assertAlmostEqual(u, W / 2)
        self.assertAlmostEqual(v, H / 2)
        self.assertAlmostEqual(d, 2.0)
        b = one([clock(2.0, 0.0, 1.5)])
        self.assertEqual(b["center"], [W / 2, H / 2])
        self.assertEqual(b["visible"], 1.0)
        self.assertIsNone(b["held_by"])

    def test_left_is_left_and_above_is_above(self):
        self.assertLess(one([clock(2.0, 0.5, 1.5)])["center"][0], W / 2)
        self.assertGreater(one([clock(2.0, -0.5, 1.5)])["center"][0], W / 2)
        self.assertLess(one([clock(2.0, 0.0, 2.0)])["center"][1], H / 2)
        self.assertGreater(one([clock(2.0, 0.0, 1.0)])["center"][1], H / 2)

    def test_known_pixel(self):
        # 45 degrees to the left with a 90 degree lens is the left edge; 45 degrees up is f above the centre
        cam = labels.camera(step(), meta(fov=90.0))
        u, v, _ = labels.project([1.0, 1.0, 1.5], cam, W, H)
        self.assertAlmostEqual(u, 0.0)
        u, v, _ = labels.project([1.0, 0.0, 2.5], cam, W, H)
        self.assertAlmostEqual(v, H / 2 - W / 2)

    def test_behind_gives_nothing(self):
        cam = labels.camera(step(), meta())
        self.assertIsNone(labels.project([-2.0, 0.0, 1.5], cam, W, H))
        self.assertIsNone(one([clock(-2.0, 0.0, 1.5)]))

    def test_outside_the_picture_is_left_out(self):
        self.assertIsNone(one([clock(0.5, 5.0, 1.5)]))          # in front, but far off to the left

    def test_head_yaw_and_pitch(self):
        it = [clock(2.0, 0.0, 1.5)]
        self.assertGreater(one(it, meta(yaw=20.0))["center"][0], W / 2)       # head left -> thing moves right
        self.assertLess(one(it, meta(yaw=-20.0))["center"][0], W / 2)
        self.assertGreater(one(it, meta(pitch=15.0))["center"][1], H / 2)     # head up -> thing moves down
        self.assertLess(one(it, meta(pitch=-15.0))["center"][1], H / 2)
        f = labels.focal(labels.camera(step(), meta()), W)
        self.assertAlmostEqual(one(it, meta(yaw=20.0))["center"][0], W / 2 + f * math.tan(math.radians(20)), delta=0.1)
        self.assertAlmostEqual(one(it, meta(pitch=15.0))["center"][1], H / 2 + f * math.tan(math.radians(15)), delta=0.1)

    def test_base_yaw_and_place_count(self):
        # the robot stands at (1, 2) facing +y; a thing 2 m further along +y is straight ahead
        b = one([clock(1.0, 4.0, 1.5)], base=(1.0, 2.0, math.pi / 2, 0.0), eye=(1.0, 2.0, 1.5))
        self.assertEqual(b["center"], [W / 2, H / 2])
        self.assertAlmostEqual(b["depth"], 2.0, places=3)
        # base yaw and head yaw add up: base 30 degrees left, head 30 degrees right -> looking along +x again
        b = one([clock(2.0, 0.0, 1.5)], meta(yaw=-30.0), base=(0.0, 0.0, math.radians(30), 0.0))
        self.assertEqual(b["center"], [W / 2, H / 2])


class Sizes(unittest.TestCase):
    def half(self, dist, fov=110.0):
        b = one([clock(dist, 0.0, 1.5)], meta(fov=fov))
        self.assertAlmostEqual(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1], delta=0.2)   # square
        return (b["box"][2] - b["box"][0]) / 2

    def test_size_follows_distance_and_fov(self):
        r = home.BY_NAME["alarm clock"]["radius"]
        self.assertGreater(self.half(1.0), self.half(2.0))
        self.assertAlmostEqual(self.half(1.0) / self.half(2.0), 2.0, delta=0.01)
        for fov in (110.0, 60.0):
            for dist in (1.0, 3.0):
                f = (W / 2) / math.tan(math.radians(fov) / 2)
                self.assertAlmostEqual(self.half(dist, fov), f * math.tan(math.atan(r / dist)), delta=0.1)

    def test_clipping_and_visible(self):
        # the middle exactly on the left edge (90 degree lens): half the box is inside
        b = one([clock(1.0, 1.0, 1.5)], meta(fov=90.0))
        self.assertEqual(b["box"][0], 0.0)
        self.assertAlmostEqual(b["visible"], 0.5, places=2)
        self.assertAlmostEqual(b["center"][0], 0.0, places=1)

    def test_aim_is_added(self):
        # the jug's middle is 12 cm ahead of and 6 cm under where it is gripped
        b = one([{"name": "water jug", "pos": [2.0, 0.0, 1.5], "held_by": "right"}])
        self.assertEqual(b["center"][0], W / 2)
        self.assertGreater(b["center"][1], H / 2)
        self.assertAlmostEqual(b["depth"], 2.12, places=3)
        self.assertEqual(b["held_by"], "right")


class Sweater(unittest.TestCase):
    def test_lying_is_wide_and_flat(self):
        b = one([{"name": "sweater", "pos": [2.0, 0.0, 0.07], "held_by": None}], meta(pitch=-30.0))
        w, h = b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]
        self.assertGreater(w, 1.3 * h)                          # seen from 1.5 m up and 2 m away: 111 x 80 pixels
        self.assertAlmostEqual((b["box"][0] + b["box"][2]) / 2, W / 2, delta=1.0)
        self.assertGreater(b["center"][1], H / 2 - 1)           # on the floor: at or under where the camera looks

    def test_held_is_tall_and_hangs_under_the_hand(self):
        b = one([{"name": "sweater", "pos": [2.0, 0.0, 1.5], "held_by": "right"}])
        w, h = b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]
        self.assertGreater(h, 1.5 * w)
        self.assertLess(b["box"][1], H / 2)                     # starts just above the hand
        self.assertGreater(b["box"][3], H / 2 + 0.25 * labels.focal(labels.camera(step(), meta()), W))
        self.assertEqual(b["held_by"], "right")


class Man(unittest.TestCase):
    def test_only_when_the_step_has_the_chair(self):
        self.assertEqual(labels.boxes(step(), meta(), W, H), [])
        b = labels.boxes(step(scene={"phase": "tidy", "man": "idle", "chair": [3.0, 0.0, 0.3]}), meta(), W, H)
        self.assertEqual([x["name"] for x in b], ["man in wheelchair"])
        b = labels.boxes(step(scene={"phase": "bed", "man": "bed", "chair": [3.0, 0.0, 0.3]}), meta(), W, H)
        self.assertEqual([x["name"] for x in b], ["wheelchair"])


class Episode(unittest.TestCase):
    def test_head_null_is_unlabelled(self):
        m = meta()
        m["head"] = None
        self.assertIsNone(labels.camera(step([clock(2.0, 0.0, 1.5)]), m))
        self.assertEqual(labels.boxes(step([clock(2.0, 0.0, 1.5)]), m, W, H), [])

    def test_label_episode_writes_the_file(self):
        with tempfile.TemporaryDirectory() as root:
            ep = Path(root) / "episode_000000"
            (ep / "frames" / "robot_eyes").mkdir(parents=True)
            # the clock moves from the left to the right, one step per tick; frames saved at ticks 1, 9 and 16
            steps = [step([clock(2.0, 1.0 - 0.1 * i, 1.5)], tick=i) for i in range(20)]
            (ep / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
            metas = [meta(k=0, tick=1), meta(k=1, tick=9), meta(k=2, tick=16)]
            metas[1]["head"] = None
            metas[2]["tick"] = 16.4                              # nearest step is 16
            (ep / "frames" / "robot_eyes" / "index.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas))
            try:                                                 # a real (small) picture for frame 0; the others are missing
                from PIL import Image
                Image.new("RGB", (640, 512)).save(ep / "frames" / "robot_eyes" / "000000.jpg")
                size0 = (640, 512)
            except ImportError:
                size0 = (W, H)
            n = labels.label_episode(ep)
            self.assertEqual(n, {"frames": 3, "labelled": 2, "boxes": 2})
            rows = [json.loads(line) for line in (ep / "labels" / "robot_eyes.jsonl").read_text().splitlines()]
            self.assertEqual([r["k"] for r in rows], [0, 1, 2])
            self.assertEqual([r["labelled"] for r in rows], [True, False, True])
            self.assertEqual(rows[1]["boxes"], [])
            self.assertEqual((rows[0]["width"], rows[0]["height"]), size0)
            self.assertEqual((rows[2]["width"], rows[2]["height"]), (W, H))      # no picture: 1024 high, aspect 1.25
            self.assertLess(rows[0]["boxes"][0]["center"][0], size0[0] / 2)      # tick 1: y = +0.9, left
            self.assertGreater(rows[2]["boxes"][0]["center"][0], W / 2)          # tick 16: y = -0.6, right
            f = (W / 2) / math.tan(math.radians(55))
            self.assertAlmostEqual(rows[2]["boxes"][0]["center"][0], W / 2 + f * 0.6 / 2.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
