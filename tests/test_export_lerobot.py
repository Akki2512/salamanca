"""LeRobot exporter checks on two small made-up episodes: reading, the state / action arrays, which camera
picture each step shows, task text, stats, the meta files, the videos (needs ffmpeg) and the parquet files
(needs pyarrow, skipped without it)."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools import export_lerobot as ex

HAVE_FFMPEG = Path(ex.FFMPEG).exists()
HAVE_PYARROW = importlib.util.find_spec("pyarrow") is not None


def step(i, frame, task=None):
    """Step i with numbers that say where they come from: state arm = i + 0.01 * joint, action arm = 100 + ..."""
    ramp = lambda base, n: [base + 0.01 * j for j in range(n)]
    return {"i": i, "t": i / 30, "tick": 500 + i,
            "state": {"arm_q": ramp(i, 14), "hand_q": ramp(i + 0.2, 14), "leg_q": ramp(i + 0.6, 12),
                      "waist_q": ramp(i + 0.4, 3), "base": ramp(i + 0.8, 4)},
            "action": {"arm_q": ramp(100 + i, 14), "hand_q": ramp(200 + i, 14)},
            "hands": {"left": [0.3, 0.2, 0.1], "right": [0.3, -0.2, 0.1]}, "closed": {"left": False, "right": i > 2},
            "eye": [0.0, 0.0, 0.45], "items": [{"name": "water jug", "pos": [0.5, 0.1 * i, 0.02], "held_by": None}],
            "label": {"task": task, "assist": None, "phase": "reach", "man": None, "pushing": False, "reach": None,
                      "pointed": None, "trigger": False},
            "suit": None, "frame": frame}


def make_episode(root, n, frames, sizes, meta_task, label_tasks, ended="2026-11-15T10:00:05"):
    """frames: {camera: per step picture number or None}; sizes: {camera: {picture number: (w, h)}}."""
    d = Path(root) / f"episode_{n:06d}"
    d.mkdir(parents=True)
    count = len(next(iter(frames.values())))
    with open(d / "steps.jsonl", "w") as f:
        for i in range(count):
            f.write(json.dumps(step(i, {c: frames[c][i] for c in frames}, label_tasks[i])) + "\n")
    for cam, by_k in sizes.items():
        (d / "frames" / cam).mkdir(parents=True)
        with open(d / "frames" / cam / "index.jsonl", "w") as f:
            for k, size in by_k.items():
                Image.new("RGB", size, (40 * k + 10, 100, 200)).save(d / "frames" / cam / f"{k:06d}.jpg")
                f.write(json.dumps({"k": k, "t": k / 15, "tick": 500 + 2 * k, "head": None, "x_frame": k}) + "\n")
    (d / "meta.json").write_text(json.dumps({
        "episode": n, "task": meta_task, "started": "2026-11-15T10:00:00", "ended": ended, "fps": 30, "ticks": count,
        "success": None, "robot": "unitree_g1_29dof_hand14 (kinematic mock)", "walk_gain": 1.0,
        "cameras": {c: {"frames": len(by_k)} for c, by_k in sizes.items()}, "notes": ""}))
    return d


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "episodes"
        # episode 0: 6 steps, robot_eyes pictures 0..2 (the last one bigger, head-tracked), person pictures 0..1;
        #            no task in meta, labels say "fetch the jug" most of the time
        make_episode(cls.root, 0,
                     {"robot_eyes": [None, 0, 0, 1, 2, 2], "person": [None, None, 0, 0, 0, 1]},
                     {"robot_eyes": {0: (64, 36), 1: (64, 36), 2: (64, 52)}, "person": {0: (32, 24), 1: (32, 24)}},
                     None, [None, "fetch the jug", "fetch the jug", "fetch the jug", "wave", None])
        # episode 1: 4 steps, robot_eyes only; its index points at picture 5, which was never saved
        make_episode(cls.root, 1, {"robot_eyes": [0, 0, 5, 1], "person": [None] * 4},
                     {"robot_eyes": {0: (64, 36), 1: (64, 36)}}, "bring the water jug", [None] * 4)
        # not complete: still being recorded (no end time), and a folder with no meta at all
        make_episode(cls.root, 2, {"robot_eyes": [None], "person": [None]}, {}, "x", [None], ended=None)
        (cls.root / "episode_000003").mkdir()
        cls.eps, cls.skipped = ex.load_episodes(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_loading(self):
        self.assertEqual([ep["dir"].name for ep in self.eps], ["episode_000000", "episode_000001"])
        self.assertEqual(self.skipped, ["episode_000002", "episode_000003"])
        self.assertEqual([len(ep["steps"]) for ep in self.eps], [6, 4])
        self.assertEqual(self.eps[0]["frames"], {"robot_eyes": {0, 1, 2}, "person": {0, 1}})
        self.assertEqual(self.eps[1]["frames"]["person"], set())

    def test_half_written_last_line_is_dropped(self):
        with tempfile.TemporaryDirectory() as t:
            d = make_episode(t, 0, {"robot_eyes": [None] * 3, "person": [None] * 3}, {}, "x", [None] * 3)
            with open(d / "steps.jsonl", "a") as f:
                f.write('{"i": 3, "t": 0.1, "sta')
            self.assertEqual(len(ex.load_episode(d)["steps"]), 3)

    def test_names(self):
        self.assertEqual(len(ex.NAMES), 47)
        self.assertEqual(len(set(ex.NAMES)), 47)
        self.assertEqual(ex.NAMES[0], "left_shoulder_pitch_joint")
        self.assertEqual(ex.NAMES[13], "right_wrist_yaw_joint")
        self.assertEqual(ex.NAMES[14], "left_hand_thumb_0_joint")
        self.assertEqual(ex.NAMES[28:31], ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])
        self.assertEqual(ex.NAMES[31], "left_hip_pitch_joint")
        self.assertEqual(ex.NAMES[42], "right_ankle_roll_joint")
        self.assertEqual(ex.NAMES[43:], ["base_x", "base_y", "base_yaw", "base_dz"])

    def test_state_and_action(self):
        state, action = ex.episode_arrays(self.eps[0])
        self.assertEqual(state.shape, (6, 47))
        self.assertEqual(action.shape, (6, 47))
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(action.dtype, np.float32)
        # state of step 2: arm, hand, waist, leg, base in that order
        np.testing.assert_allclose(state[2, [0, 13, 14, 28, 30, 31, 42, 43, 46]],
                                   [2.0, 2.13, 2.2, 2.4, 2.42, 2.6, 2.71, 2.8, 2.83], rtol=1e-6)
        # action: this step's commanded arm and hand ...
        np.testing.assert_allclose(action[2, [0, 13, 14, 27]], [102.0, 102.13, 202.0, 202.13], rtol=1e-6)
        # ... then waist, legs and base of the NEXT step's state; the last step repeats its own
        np.testing.assert_array_equal(action[:-1, 28:], state[1:, 28:])
        np.testing.assert_array_equal(action[-1, 28:], state[-1, 28:])

    def test_frame_refs(self):
        # before the first picture: the first one; after that the newest; picture 2 is repeated
        self.assertEqual(ex.frame_refs(self.eps[0], "robot_eyes"), [0, 0, 0, 1, 2, 2])
        self.assertEqual(ex.frame_refs(self.eps[0], "person"), [0, 0, 0, 0, 0, 1])
        # picture 5 is not on disk: the one before is repeated
        self.assertEqual(ex.frame_refs(self.eps[1], "robot_eyes"), [0, 0, 0, 1])
        # no picture at all in this episode
        self.assertEqual(ex.frame_refs(self.eps[1], "person"), [None] * 4)

    def test_task_text(self):
        self.assertEqual(ex.task_text(self.eps[0]), "fetch the jug")          # most common label
        self.assertEqual(ex.task_text(self.eps[1]), "bring the water jug")    # meta wins
        bare = {"meta": {"task": None}, "steps": [{"label": {"task": None}}, {"label": None}, {}]}
        self.assertEqual(ex.task_text(bare), "teleoperation")
        self.assertEqual(ex.task_table(self.eps + self.eps[:1]),
                         (["fetch the jug", "bring the water jug"], [0, 1, 0]))

    def test_feature_stats(self):
        a = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 40.0]])
        st = ex.feature_stats(a)
        np.testing.assert_allclose(st["min"], [0, 10])
        np.testing.assert_allclose(st["max"], [4, 40])
        np.testing.assert_allclose(st["mean"], [2, 20])
        np.testing.assert_allclose(st["std"], a.std(0))
        np.testing.assert_allclose(st["q50"], [2, 10])
        self.assertEqual(st["count"].tolist(), [3])
        self.assertEqual(ex.feature_stats(np.arange(5))["mean"].shape, (1,))  # a column of single numbers

    def test_image_stats(self):
        px = np.zeros((2, 4, 4, 3), np.uint8)
        px[..., 0], px[1, ..., 1] = 255, 255
        st = ex.image_stats(px, 30)
        self.assertEqual(st["mean"].shape, (3, 1, 1))
        np.testing.assert_allclose(st["mean"].ravel(), [1.0, 0.5, 0.0])
        np.testing.assert_allclose(st["max"].ravel(), [1.0, 1.0, 0.0])
        self.assertEqual(st["count"].tolist(), [30])

    def test_camera_sizes(self):
        self.assertEqual(ex.camera_sizes(self.eps), {"robot_eyes": (64, 36), "person": (32, 24)})
        self.assertEqual(ex.camera_sizes(self.eps, (40, 30)), {"robot_eyes": (40, 30), "person": (40, 30)})
        self.assertEqual(ex.camera_sizes(self.eps[1:]), {"robot_eyes": (64, 36)})   # person: nothing saved
        self.assertEqual(ex.parse_size("1281x721"), (1280, 720))                    # made even for yuv420p

    def test_picture_bytes(self):
        d = self.eps[0]["dir"] / "frames/robot_eyes"
        self.assertEqual(len(ex.picture_bytes(d / "000000.jpg", (64, 36))), 64 * 36 * 3)
        tall = np.frombuffer(ex.picture_bytes(d / "000002.jpg", (64, 36)), np.uint8).reshape(36, 64, 3)
        self.assertEqual(tall[18, 0].tolist(), [0, 0, 0])              # 64x52 fitted into 64x36: black bars at the sides
        self.assertGreater(int(tall[18, 32, 2]), 150)                  # the picture itself in the middle
        self.assertEqual(ex.picture_bytes(None, (4, 2)), bytes(24))    # no picture: black

    def test_build(self):
        sizes = ex.camera_sizes(self.eps)
        b = ex.build(self.eps, 30, sizes)
        c = b["columns"]
        self.assertEqual(c["observation.state"].shape, (10, 47))
        self.assertEqual(c["action"].shape, (10, 47))
        self.assertEqual(c["index"].tolist(), list(range(10)))
        self.assertEqual(c["frame_index"].tolist(), [0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
        self.assertEqual(c["episode_index"].tolist(), [0] * 6 + [1] * 4)
        self.assertEqual(c["task_index"].tolist(), [0] * 6 + [1] * 4)
        np.testing.assert_allclose(c["timestamp"], np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3]) / 30, rtol=1e-6)
        self.assertEqual(c["timestamp"].dtype, np.float32)
        # the next-state rule does not reach over into the next episode
        np.testing.assert_array_equal(c["action"][5, 28:], c["observation.state"][5, 28:])
        e0, e1 = b["episodes"]
        self.assertEqual((e0["length"], e0["dataset_from_index"], e0["dataset_to_index"]), (6, 0, 6))
        self.assertEqual((e1["length"], e1["dataset_from_index"], e1["dataset_to_index"]), (4, 6, 10))
        self.assertEqual(e1["tasks"], ["bring the water jug"])
        key = "videos/observation.images.robot_eyes/"
        self.assertAlmostEqual(e0[key + "from_timestamp"], 0.0)
        self.assertAlmostEqual(e0[key + "to_timestamp"], 0.2)
        self.assertAlmostEqual(e1[key + "from_timestamp"], 0.2)
        self.assertAlmostEqual(e1[key + "to_timestamp"], 10 / 30)
        self.assertEqual(len(e1["stats/observation.state/mean"]), 47)
        self.assertEqual(e1["stats/observation.state/count"], [4])
        self.assertEqual(np.shape(e1["stats/observation.images.person/max"]), (3, 1, 1))
        # one picture path per data row; episode 1 has no person camera, so it is black (None) there
        pics = b["pictures"]
        self.assertEqual([p.name for p in pics["robot_eyes"]],
                         [f"{k:06d}.jpg" for k in (0, 0, 0, 1, 2, 2, 0, 0, 0, 1)])
        self.assertEqual(pics["robot_eyes"][6].parent.parent.parent.name, "episode_000001")
        self.assertEqual(pics["person"][6:], [None] * 4)
        # dataset stats are over all rows of both episodes
        st = b["stats"]
        np.testing.assert_allclose(st["observation.state"]["mean"], c["observation.state"].mean(0), rtol=1e-5)
        np.testing.assert_allclose(st["action"]["std"], c["action"].astype(np.float64).std(0), rtol=1e-5)
        np.testing.assert_allclose(st["observation.state"]["min"], c["observation.state"].min(0))
        self.assertEqual(st["index"]["max"], [9.0])
        self.assertEqual(st["observation.state"]["count"], [10])
        self.assertEqual(np.shape(st["observation.images.robot_eyes"]["mean"]), (3, 1, 1))
        self.assertEqual(st["observation.images.robot_eyes"]["count"], [10])
        self.assertLessEqual(np.max(st["observation.images.robot_eyes"]["max"]), 1.0)

    def test_build_without_cameras(self):
        b = ex.build(self.eps, 30, {})
        self.assertEqual(b["pictures"], {})
        self.assertFalse(any(k.startswith("videos/") for k in b["episodes"][0]))
        self.assertNotIn("observation.images.robot_eyes", b["stats"])
        i = ex.info(b, 30, {})
        self.assertIsNone(i["video_path"])
        self.assertFalse(any(f["dtype"] == "video" for f in i["features"].values()))

    def test_meta_files(self):
        sizes = ex.camera_sizes(self.eps)
        with tempfile.TemporaryDirectory() as out:
            ex.write_meta(out, ex.build(self.eps, 30, sizes), 30, sizes, "unitree_g1_29dof_hand14 (kinematic mock)")
            info = json.loads(Path(out, "meta/info.json").read_text())
            stats = json.loads(Path(out, "meta/stats.json").read_text())
        self.assertEqual(info["codebase_version"], "v3.0")
        self.assertEqual((info["total_episodes"], info["total_frames"], info["total_tasks"]), (2, 10, 2))
        self.assertEqual(info["fps"], 30)
        self.assertEqual(info["splits"], {"train": "0:2"})
        self.assertEqual(info["robot_type"], "unitree_g1_29dof_hand14 (kinematic mock)")
        self.assertEqual(info["data_path"], "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        self.assertEqual(info["video_path"],
                         "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        self.assertEqual(info["data_path"].format(chunk_index=0, file_index=0), "data/chunk-000/file-000.parquet")
        f = info["features"]
        self.assertEqual(set(f), {"observation.state", "action", "observation.images.robot_eyes",
                                  "observation.images.person", "timestamp", "frame_index", "episode_index",
                                  "index", "task_index"})
        self.assertEqual(f["observation.state"], {"dtype": "float32", "shape": [47], "names": ex.NAMES})
        self.assertEqual(f["action"]["shape"], [47])
        eyes = f["observation.images.robot_eyes"]
        self.assertEqual((eyes["dtype"], eyes["shape"]), ("video", [36, 64, 3]))
        self.assertEqual(eyes["names"], ["height", "width", "channels"])
        self.assertEqual((eyes["info"]["video.fps"], eyes["info"]["video.codec"]), (30, "h264"))
        self.assertEqual(f["timestamp"], {"dtype": "float32", "shape": [1], "names": None})
        self.assertEqual(f["index"], {"dtype": "int64", "shape": [1], "names": None})
        for key in f:
            for s in ("min", "max", "mean", "std", "count"):
                self.assertIn(s, stats[key], key)
        self.assertEqual(len(stats["action"]["max"]), 47)

    def test_extras(self):
        with tempfile.TemporaryDirectory() as out:
            ex.write_extras(out, self.eps)
            lines = [json.loads(x) for x in Path(out, "extras/episode_000001.jsonl").read_text().splitlines()]
            summary = [json.loads(x) for x in Path(out, "extras/episodes.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 4)
        src = self.eps[1]["steps"][2]
        self.assertEqual((lines[2]["index"], lines[2]["episode_index"], lines[2]["frame_index"]), (8, 1, 2))
        for k in ("i", "t", "tick", "hands", "closed", "eye", "items", "label", "suit", "frame"):
            self.assertEqual(lines[2][k], src[k], k)
        self.assertEqual(summary[0]["source"], "episode_000000")
        self.assertEqual(summary[0]["task"], "fetch the jug")
        self.assertEqual(summary[1]["meta"], self.eps[1]["meta"])

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not found")
    def test_video(self):
        sizes = ex.camera_sizes(self.eps)
        b = ex.build(self.eps, 30, sizes)
        with tempfile.TemporaryDirectory() as out:
            for cam in sizes:                                          # robot_eyes mixes two sizes, person has black
                mp4 = ex.write_video(b["pictures"][cam], sizes[cam], 30, Path(out) / cam / "file-000.mp4")
                probe = Path(ex.FFMPEG).with_name("ffprobe")
                if not probe.exists():
                    self.assertGreater(mp4.stat().st_size, 500)
                    continue
                r = subprocess.run([str(probe), "-v", "error", "-select_streams", "v:0", "-count_frames",
                                    "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames",
                                    "-of", "json", str(mp4)], capture_output=True, check=True)
                s = json.loads(r.stdout)["streams"][0]
                self.assertEqual((s["width"], s["height"]), sizes[cam])
                self.assertEqual((s["codec_name"], s["pix_fmt"], s["r_frame_rate"]), ("h264", "yuv420p", "30/1"))
                self.assertEqual(int(s["nb_read_frames"]), 10)         # one video frame per data row

    @unittest.skipIf(HAVE_PYARROW, "pyarrow is installed")
    def test_message_without_pyarrow(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(SystemExit) as stop:
                ex.export(self.root, Path(out) / "ds", log=lambda *a: None)
            self.assertIn(".venv/bin/pip install pyarrow", str(stop.exception))
            self.assertFalse((Path(out) / "ds").exists())              # stopped before writing anything

    @unittest.skipUnless(HAVE_PYARROW and HAVE_FFMPEG, "needs pyarrow (.venv/bin/pip install pyarrow) and ffmpeg")
    def test_parquet(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ds"
            ex.export(self.root, out, log=lambda *a: None)
            data = pq.ParquetFile(out / "data/chunk-000/file-000.parquet")
            t = data.read()
            self.assertEqual(t.num_rows, 10)
            self.assertEqual(data.num_row_groups, 2)                   # one per episode
            self.assertEqual(t.column_names, ["observation.state", "action", "timestamp", "frame_index",
                                              "episode_index", "index", "task_index"])
            # the inner field's name is "item" or "element" depending on the pyarrow version and the parquet round trip
            self.assertIn(str(t.schema.field("observation.state").type), ("fixed_size_list<item: float>[47]", "fixed_size_list<element: float>[47]"))
            self.assertEqual(str(t.schema.field("timestamp").type), "float")
            self.assertEqual(str(t.schema.field("index").type), "int64")
            state, _ = ex.episode_arrays(self.eps[1])
            np.testing.assert_array_equal(np.array(t.column("observation.state").to_pylist()[6:], np.float32), state)
            tasks = pq.read_table(out / "meta/tasks.parquet")
            self.assertEqual(tasks.to_pydict(), {"task_index": [0, 1],
                                                 "task": ["fetch the jug", "bring the water jug"]})
            self.assertEqual(json.loads(tasks.schema.metadata[b"pandas"])["index_columns"], ["task"])
            eps = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
            self.assertEqual([e["length"] for e in eps], [6, 4])
            self.assertEqual(eps[1]["tasks"], ["bring the water jug"])
            self.assertEqual((eps[1]["dataset_from_index"], eps[1]["dataset_to_index"]), (6, 10))
            self.assertIn("videos/observation.images.robot_eyes/from_timestamp", eps[0])
            self.assertEqual(len(eps[0]["stats/action/mean"]), 47)
            for rel in ("meta/info.json", "meta/stats.json", "extras/episode_000000.jsonl",
                        "videos/observation.images.robot_eyes/chunk-000/file-000.mp4",
                        "videos/observation.images.person/chunk-000/file-000.mp4"):
                self.assertTrue((out / rel).is_file(), rel)


if __name__ == "__main__":
    unittest.main()
