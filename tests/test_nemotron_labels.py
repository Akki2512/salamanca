"""tools/label_with_nemotron.py on a small made-up episode. The model call is a fake: no network."""
import contextlib, io, json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import label_with_nemotron as ln
from PIL import Image


def step(i, phase="tidy", assist="free", reach=None, pushing=False, held=None, frame=None):
    return {"i": i, "t": round(i / 30, 4),
            "items": [{"name": "sweater", "pos": [0, 0, 0], "held_by": "right" if held == "sweater" else None},
                      {"name": "glass", "pos": [1, 0, 1], "held_by": "left" if held == "glass" else None}],
            "label": {"task": None, "assist": assist, "phase": phase, "man": None, "pushing": pushing,
                      "reach": reach, "pointed": None, "trigger": False},
            "frame": {"robot_eyes": frame, "person": None}}


def story():
    """120 ticks: walk (30), pick from the floor (30), hold the sweater (30), cough phase with empty hands (30).
    One camera frame for every second tick."""
    rows = [dict()] * 30 + [dict(assist="descend", reach="bend")] * 30 + [dict(assist="holding", held="sweater")] * 30 \
        + [dict(phase="cough", assist="done")] * 30
    return [step(i, frame=i // 2, **kw) for i, kw in enumerate(rows)]


def write_episode(root, steps, frames=60, name="episode_000000"):
    ep = Path(root) / name
    (ep / "frames" / "robot_eyes").mkdir(parents=True)
    (ep / "meta.json").write_text(json.dumps({"episode": 0, "task": "tidy up", "fps": 30, "ticks": len(steps),
                                              "success": None}))
    (ep / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    for k in range(frames):
        Image.new("RGB", (1280, 720), (k * 4 % 255, 80, 120)).save(ep / "frames" / "robot_eyes" / f"{k:06d}.jpg")
    return ep


class FakeModel:
    """Stands in for nemotron_call: remembers what it was sent, answers from a list of scripted replies."""
    def __init__(self, fail=()):
        self.sent, self.fail = [], set(fail)          # fail: numbers of the calls (from 1) that raise

    def __call__(self, messages, timeout):
        self.sent.append(messages)
        if len(self.sent) in self.fail:
            raise TimeoutError("too slow")
        if "whole recording" in messages[0]["content"]:
            return '{"task": "Tidy the sweater away.", "success": true, "reason": "The sweater is off the floor."}', "fake-vl"
        return ('<think>hm</think> {"instruction": "Pick up the sweater.", "objects": ["sweater"], '
                '"confidence": 0.8}'), "fake-vl"


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*a, **kw)
    return out, buf.getvalue()


class Segments(unittest.TestCase):
    def test_boundaries_follow_the_recorded_labels(self):
        segs = ln.segment(story())
        self.assertEqual([(s["from_i"], s["to_i"]) for s in segs], [(0, 29), (30, 59), (60, 89), (90, 119)])
        self.assertEqual(segs[1]["truth"], {"phase": "tidy", "assist": "picking", "reach": True, "pushing": False,
                                            "held": []})
        self.assertEqual(segs[2]["truth"]["held"], ["sweater"])
        self.assertEqual(segs[3]["truth"]["phase"], "cough")
        self.assertAlmostEqual(segs[1]["t0"], 1.0); self.assertAlmostEqual(segs[1]["t1"], 59 / 30, places=3)

    def test_states_of_one_group_do_not_split(self):
        steps = [step(i, assist=a) for i, a in enumerate(["approach"] * 20 + ["descend"] * 20 + ["closing"] * 20)]
        self.assertEqual(len(ln.segment(steps)), 1)
        steps = [step(i, assist=a) for i, a in enumerate(["free"] * 20 + ["done"] * 20 + [None] * 20)]
        self.assertEqual(len(ln.segment(steps)), 1)

    def test_each_part_of_the_tuple_splits(self):
        for change in (dict(phase="bed"), dict(assist="releasing"), dict(reach="walk"), dict(pushing=True),
                       dict(held="glass")):
            steps = [step(i) for i in range(30)] + [step(30 + i, **change) for i in range(30)]
            self.assertEqual(len(ln.segment(steps)), 2, change)

    def test_short_segment_joins_a_neighbour(self):
        # 10 ticks (0.33 s) of pushing between two long parts: too short to stand alone
        steps = [step(i) for i in range(40)] + [step(40 + i, pushing=True) for i in range(10)] \
            + [step(50 + i, assist="holding", held="glass") for i in range(40)]
        segs = ln.segment(steps)
        self.assertEqual([(s["from_i"], s["to_i"]) for s in segs], [(0, 49), (50, 89)])
        self.assertFalse(segs[0]["truth"]["pushing"])          # the long part's facts are kept
        # 15 ticks is exactly 0.5 s: kept
        steps = [step(i) for i in range(40)] + [step(40 + i, pushing=True) for i in range(15)]
        self.assertEqual(len(ln.segment(steps)), 2)

    def test_cap_merges_the_shortest(self):
        steps, i = [], 0
        for n, kw in ((60, {}), (20, dict(pushing=True)), (60, dict(phase="bed")), (30, dict(assist="holding")),
                      (60, dict(phase="drink"))):
            steps += [step(i + j, **kw) for j in range(n)]; i += n
        self.assertEqual(len(ln.segment(steps, max_segments=8)), 5)
        segs = ln.segment(steps, max_segments=3)
        self.assertEqual(len(segs), 3)
        self.assertEqual((segs[0]["from_i"], segs[-1]["to_i"]), (0, 229))       # nothing lost
        self.assertTrue(all(a["to_i"] + 1 == b["from_i"] for a, b in zip(segs, segs[1:])))
        self.assertNotIn(True, [s["truth"]["pushing"] for s in segs])           # the 20-tick part went first
        self.assertEqual(len(ln.segment(steps, max_segments=1)), 1)

    def test_middle_frame(self):
        steps = story()
        segs = ln.segment(steps)
        self.assertEqual([ln.middle_frame(steps, s, "robot_eyes") for s in segs], [7, 22, 37, 52])
        self.assertIsNone(ln.middle_frame(steps, segs[0], "person"))            # that camera saved nothing
        # no frame at the middle tick: the nearest tick in the segment that has one is used
        for s in steps[40:50]:
            s["frame"]["robot_eyes"] = None
        self.assertEqual(ln.middle_frame(steps, segs[1], "robot_eyes"), 19)     # tick 39


class Labelling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.ep = write_episode(self.tmp.name, story())
        self.file = self.ep / "labels" / "nemotron.json"

    def test_output_file(self):
        fake = FakeModel()
        _, text = quiet(ln.label_episode, self.ep, call=fake)
        d = json.loads(self.file.read_text())
        self.assertEqual(len(fake.sent), 5)                                     # 4 segments + the whole episode
        self.assertEqual((d["model"], d["camera"], d["calls"]), ("fake-vl", "robot_eyes", 5))
        self.assertEqual((d["task"], d["success"]), ("Tidy the sweater away.", True))
        self.assertTrue(d["reason"])
        self.assertEqual([s["frame_k"] for s in d["segments"]], [7, 22, 37, 52])
        for s in d["segments"]:
            self.assertEqual(set(s), {"from_i", "to_i", "t0", "t1", "frame_k", "truth", "instruction", "objects",
                                      "confidence", "error", "took_s"})
            self.assertEqual((s["instruction"], s["objects"], s["confidence"], s["error"]),
                             ("Pick up the sweater.", ["sweater"], 0.8, None))
        self.assertIn("Pick up the sweater.", text); self.assertIn("holds sweater", text)

    def test_the_model_is_not_told_the_recorded_label(self):
        fake = FakeModel()
        quiet(ln.label_episode, self.ep, call=fake)
        seg_call = fake.sent[1]
        said = json.dumps(seg_call)
        self.assertIn("sweater", seg_call[0]["content"]); self.assertIn("glass", seg_call[0]["content"])  # item names
        for word in ("descend", "picking", "holding", "tidy up", '"reach"', "cough"):
            self.assertNotIn(word, said)
        images = [c for c in seg_call[1]["content"] if c["type"] == "image_url"]
        self.assertEqual(len(images), 1)
        self.assertEqual(len([c for c in fake.sent[-1][1]["content"] if c["type"] == "image_url"]), 3)

    def test_images_are_made_smaller(self):
        import base64
        url = ln.data_url(self.ep / "frames" / "robot_eyes" / "000000.jpg")
        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        self.assertEqual(img.size, (768, 432))

    def test_failed_call_is_retried_once_then_recorded(self):
        fake = FakeModel(fail={1})                                              # first try fails, the retry works
        quiet(ln.label_episode, self.ep, call=fake)
        d = json.loads(self.file.read_text())
        self.assertEqual((len(fake.sent), d["calls"]), (6, 6))
        self.assertTrue(all(s["instruction"] for s in d["segments"]))
        fake = FakeModel(fail={1, 2})                                           # both tries of segment 1 fail
        quiet(ln.label_episode, self.ep, call=fake, redo=True)
        d = json.loads(self.file.read_text())
        self.assertIsNone(d["segments"][0]["instruction"])
        self.assertIn("TimeoutError", d["segments"][0]["error"])
        self.assertTrue(all(s["instruction"] for s in d["segments"][1:]))       # the run went on
        self.assertEqual(d["task"], "Tidy the sweater away.")

    def test_second_run_does_only_what_is_missing(self):
        quiet(ln.label_episode, self.ep, call=FakeModel(fail={2, 3}))           # segment 2 fails twice
        d = json.loads(self.file.read_text())
        self.assertIsNone(d["segments"][1]["instruction"])
        again = FakeModel()
        quiet(ln.label_episode, self.ep, call=again)
        self.assertEqual(len(again.sent), 1)                                    # only the failed segment
        d = json.loads(self.file.read_text())
        self.assertTrue(all(s["instruction"] and not s["error"] for s in d["segments"]))
        self.assertEqual(d["calls"], 7)                                         # 6 the first time + 1
        nothing = FakeModel()
        quiet(ln.label_episode, self.ep, call=nothing)
        self.assertEqual(len(nothing.sent), 0)
        redo = FakeModel()
        quiet(ln.label_episode, self.ep, call=redo, redo=True)
        self.assertEqual(len(redo.sent), 5)

    def test_progress_is_saved_after_every_call(self):
        seen = []

        def call(messages, timeout):
            seen.append(json.loads(self.file.read_text())["calls"] if self.file.exists() else 0)
            return FakeModel()(messages, timeout)
        quiet(ln.label_episode, self.ep, call=call)
        self.assertEqual(seen, [0, 1, 2, 3, 4])

    def test_dry_run_calls_nothing_and_writes_nothing(self):
        fake = FakeModel()
        _, text = quiet(ln.label_episode, self.ep, call=fake, dry_run=True)
        self.assertEqual(len(fake.sent), 0)
        self.assertFalse(self.file.exists())
        self.assertIn("4 segment(s) + 1 for the whole episode", text)

    def test_episode_without_frames(self):
        ep = write_episode(self.tmp.name, story(), frames=0, name="episode_000001")
        fake = FakeModel()
        out, _ = quiet(ln.label_episode, ep, call=fake)
        self.assertEqual(len(fake.sent), 0)
        self.assertEqual([s["frame_k"] for s in out["segments"]], [None] * 4)
        self.assertIsNone(out["task"])
        self.assertTrue((ep / "labels" / "nemotron.json").exists())

    def test_root_or_single_episode(self):
        write_episode(self.tmp.name, story()[:40], frames=20, name="episode_000001")
        self.assertEqual([p.name for p in ln.episodes_in(self.tmp.name)], ["episode_000000", "episode_000001"])
        self.assertEqual(ln.episodes_in(self.ep), [self.ep])
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            ln.main([self.tmp.name, "--dry-run", "--max-segments", "2"])
        self.assertIn("episode_000001", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
