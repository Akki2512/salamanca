"""Request lifecycle on the session server with a stand-in extractor (no network)."""
import base64, io, json, socket, threading, time, unittest

import uvicorn
from PIL import Image
from websockets.sync.client import connect

from remote_hands import session_server as ss
from remote_hands.extractor import validate

buf = io.BytesIO(); Image.new("RGB", (640, 480), "white").save(buf, "JPEG")
FRAME = base64.b64encode(buf.getvalue()).decode()
REPLY = {"status": "ok", "target": {"item": "water jug", "box": [350, 800, 450, 980]},
         "confidence": 0.9, "say": "Getting your water jug.",
         "visible": [{"item": "water jug"}, {"item": "teapot"}]}


def until(ws, pred):
    for _ in range(20):
        m = json.loads(ws.recv(timeout=5))
        if pred(m):
            return m
    raise AssertionError("message not seen")


class Server:
    """The real app on a real socket (uvicorn in a thread): every client shares one event loop, as deployed."""

    def __init__(self):
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); self.port = sock.getsockname()[1]; sock.close()
        self.server = uvicorn.Server(uvicorn.Config(ss.app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def ws(self, role, session="t"):
        return connect(f"ws://127.0.0.1:{self.port}/ws/{role}?session={session}", max_size=2 ** 24)

    def stop(self):
        self.server.should_exit = True
        self.thread.join(5)


class Lifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = Server()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def setUp(self):
        ss.sessions.clear()
        self.reply = dict(REPLY)

        async def fake(jpeg, text, w, h):
            return validate(self.reply, w, h)
        ss.app.state.extract = fake
        self.c = self.srv

    def test_request_accept_done(self):
        with self.c.ws("helper") as helper, \
             self.c.ws("robot") as robot, \
             self.c.ws("wearer") as wearer:
            wearer.send(json.dumps({"type": "request", "text": "could I have some water", "frame_jpeg_b64": FRAME}))
            task = until(helper, lambda m: m["type"] == "task")
            self.assertEqual(task["target"], {"name": "water jug", "color": "blue", "shape": "cylinder",
                                              "box_xyxy": [224, 384, 288, 470]})
            self.assertEqual(task["instruction"], "Pick up the water jug and hand it to the person.")
            self.assertEqual(task["visible"], ["water jug"])        # out-of-vocabulary dropped
            self.assertIn("frame_jpeg_b64", task)
            self.assertEqual(until(wearer, lambda m: m.get("state") == "offered")["say"],
                             "Getting your water jug. Asking a helper.")
            helper.send(json.dumps({"type": "status", "state": "accepted"}))
            tgt = until(robot, lambda m: m["type"] == "target")
            self.assertEqual((tgt["name"], tgt["color"], tgt["shape"], tgt["destination"]),
                             ("water jug", "blue", "cylinder", "person"))
            until(wearer, lambda m: m.get("state") == "accepted")
            robot.send(json.dumps({"type": "progress", "state": "in_progress"}))
            until(wearer, lambda m: m.get("state") == "in_progress")
            helper.send(json.dumps({"type": "status", "state": "done"}))
            self.assertEqual(until(wearer, lambda m: m.get("state") == "done")["say"], "Here is the water jug.")
            until(robot, lambda m: m["type"] == "cancel")
            self.assertIsNone(ss.sessions["t"].request)       # frame dropped with the request

    def test_robot_release_finishes_the_request(self):
        with self.c.ws("helper") as helper, \
             self.c.ws("robot") as robot, \
             self.c.ws("wearer") as wearer:
            wearer.send(json.dumps({"type": "request", "text": "could I have some water", "frame_jpeg_b64": FRAME}))
            until(helper, lambda m: m["type"] == "task")
            helper.send(json.dumps({"type": "status", "state": "accepted"}))
            rid = until(robot, lambda m: m["type"] == "target")["request_id"]
            robot.send(json.dumps({"type": "progress", "request_id": "stale", "state": "released"}))   # ignored
            robot.send(json.dumps({"type": "progress", "request_id": rid, "state": "grasped"}))
            self.assertEqual(until(wearer, lambda m: m.get("state") == "in_progress")["say"],
                             "The robot has the water jug.")
            robot.send(json.dumps({"type": "progress", "request_id": rid, "state": "released"}))
            self.assertEqual(until(wearer, lambda m: m.get("state") == "done")["say"], "Here is the water jug.")
            until(robot, lambda m: m["type"] == "cancel")

    def test_a_call_without_a_photo_uses_the_robots_camera(self):
        with self.c.ws("helper") as helper, \
             self.c.ws("robot") as robot, \
             self.c.ws("caller") as caller:
            caller.send(json.dumps({"type": "request", "text": "could I have some water"}))
            req = until(robot, lambda m: m["type"] == "frame_request")
            robot.send(json.dumps({"type": "frame", "id": req["id"], "jpeg_b64": FRAME}))
            task = until(helper, lambda m: m["type"] == "task")
            self.assertEqual(task["frame_jpeg_b64"], FRAME)
            self.assertEqual(until(caller, lambda m: m.get("state") == "offered")["say"],
                             "Getting your water jug. Asking a helper.")

    def test_ambiguous_goes_back_to_the_wearer_as_a_question(self):
        self.reply = {"status": "ambiguous", "target": None, "say": "The water jug or the tea cup?", "visible": []}
        with self.c.ws("helper") as helper, \
             self.c.ws("wearer") as wearer:
            wearer.send(json.dumps({"type": "request", "text": "something to drink", "frame_jpeg_b64": FRAME}))
            m = until(wearer, lambda m: m.get("state") == "failed")
            self.assertEqual(m["say"], "The water jug or the tea cup?")

    def test_no_helper_and_bad_frame(self):
        with self.c.ws("wearer") as wearer:
            wearer.send(json.dumps({"type": "request", "text": "the blue one", "frame_jpeg_b64": "not base64!"}))
            self.assertEqual(until(wearer, lambda m: m["type"] == "status")["state"], "failed")
            wearer.send(json.dumps({"type": "request", "text": "the blue one", "frame_jpeg_b64": FRAME}))
            self.assertIn("No helper", until(wearer, lambda m: m.get("state") == "failed")["say"])

    def test_extractor_failure_fails_the_request_not_the_session(self):
        async def boom(*a):
            raise RuntimeError("down")
        ss.app.state.extract = boom
        with self.c.ws("wearer") as wearer:
            wearer.send(json.dumps({"type": "request", "text": "bring me the thing", "frame_jpeg_b64": FRAME}))
            until(wearer, lambda m: m.get("state") == "failed")
            self.assertEqual(ss.sessions["t"].state, "idle")

    def test_extractor_failure_falls_back_to_word_matching(self):
        async def boom(*a):
            raise RuntimeError("down")
        ss.app.state.extract = boom
        with self.c.ws("helper") as helper, \
             self.c.ws("wearer") as wearer:
            wearer.send(json.dumps({"type": "request", "text": "I am so thirsty", "frame_jpeg_b64": FRAME}))
            task = until(helper, lambda m: m["type"] == "task")
            self.assertEqual((task["source"], task["target"]["name"], task["target"]["box_xyxy"]),
                             ("keywords", "glass of water", None))       # "thirsty" now means the glass, not the jug
            wearer.send(json.dumps({"type": "request", "text": "x", "frame_jpeg_b64": FRAME}))   # busy
        with self.c.ws("wearer", "u") as wearer:
            wearer.send(json.dumps({"type": "request", "text": "the clock or the glass", "frame_jpeg_b64": FRAME}))
            self.assertEqual(until(wearer, lambda m: m.get("state") == "failed")["say"],
                             "Do you mean the alarm clock or the glass of water?")


class Validate(unittest.TestCase):
    def test_rejects_what_a_robot_cannot_use(self):
        for bad in ({**REPLY, "target": {"item": "teapot", "box": [1, 1, 50, 50]}},
                    {**REPLY, "target": {"item": "cup", "box": [0, 0, 1000, 1000]}},
                    {**REPLY, "target": {"item": "cup", "box": [10, 10, 10, 10]}},
                    {**REPLY, "status": "sure"}, {**REPLY, "say": ""}, {**REPLY, "target": None}):
            with self.assertRaises(ValueError):
                validate(bad, 640, 480)
        self.assertEqual(validate({**REPLY, "destination": "the moon"}, 640, 480)["destination"], "person")


if __name__ == "__main__":
    unittest.main()
