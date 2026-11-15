"""Session server: pairs a wearer with a helper and a robot, runs the request lifecycle, calls Nemotron, logs.

    .venv/bin/python -m uvicorn remote_hands.session_server:app --port 8000

Clients connect to  ws://HOST/ws/{role}?session=NAME  with role = wearer | helper | robot.

  wearer -> {"type": "request", "text": "...", "frame_jpeg_b64": "...", "log_frame": false}
  helper -> {"type": "status", "state": "accepted" | "done" | "failed", "say": "..."}
  robot  -> {"type": "progress", "state": "in_progress" | "grasped" | "dropped" | "released"}   (released ends the request as done)

  -> wearer: {"type": "status", "request_id", "state", "say"}            every state change; `say` is spoken
  -> helper: {"type": "task", ...extractor output..., "text", "frame_jpeg_b64"}   then status + progress
  -> robot:  {"type": "target", "request_id", "color", "shape", "destination"}    on accept; "cancel" on end

States: idle -> extracting -> offered -> accepted -> in_progress -> done | failed
The wearer's frame is held in memory until the request ends and is never written to disk unless the
request says "log_frame": true (spec section 8).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import extractor

ROLES = ("wearer", "helper", "robot")      # "wearer" = the caller (the person asking for help); "caller" is accepted too
LOG_DIR = Path(os.environ.get("REMOTE_HANDS_LOG_DIR", Path(__file__).resolve().parents[1] / "logs"))
# On a public address the session name is the shared secret: only names listed here are accepted (comma
# separated). Empty = accept any name (local use).
ALLOWED_SESSIONS = {n for n in os.environ.get("REMOTE_HANDS_SESSIONS", "").split(",") if n}
HELPER_PAGE = Path(__file__).resolve().parents[1] / "helper" / "index.html"

app = FastAPI(title="Remote Hands session server")
app.state.extract = extractor.extract       # replaceable (tests, stand-in)
sessions: dict[str, "Session"] = {}


def jpeg_size(jpeg: bytes) -> tuple[int, int]:
    from PIL import Image
    with Image.open(io.BytesIO(jpeg)) as im:
        return im.width, im.height


class Session:
    def __init__(self, name: str):
        self.name = name
        self.sockets: dict[str, set[WebSocket]] = {r: set() for r in ROLES}
        self.state = "idle"
        self.request = None         # {"id", "text", "frame" (bytes), "task", "log_frame", "t0"}
        self.lock = asyncio.Lock()

    # -------------------------------------------------------------- plumbing
    async def send(self, role: str, msg: dict):
        data = json.dumps(msg)
        for ws in list(self.sockets[role]):
            try:
                await ws.send_text(data)
            except Exception:
                self.sockets[role].discard(ws)

    def log(self, event: str, **fields):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        rec = {"t": round(time.time(), 3), "session": self.name, "event": event,
               "request_id": self.request["id"] if self.request else None, **fields}
        with open(LOG_DIR / f"session_{self.name}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if event in ("extract_error", "task", "state"):     # also to stdout: visible in `modal app logs`
            shown = {k: v for k, v in fields.items() if k != "task"}
            if "task" in fields:
                shown.update({k: fields["task"].get(k) for k in ("status", "latency_s", "attempts", "attempt_errors")})
            print(f"[{rec['request_id']}] {event} {shown}", flush=True)

    async def set_state(self, state: str, say: str | None = None):
        self.state = state
        msg = {"type": "status", "request_id": self.request["id"] if self.request else None,
               "state": state, "say": say}
        self.log("state", state=state, say=say)
        await self.send("wearer", msg)
        await self.send("helper", msg)

    def presence(self) -> dict:
        return {"type": "presence", "state": self.state, **{r: len(self.sockets[r]) for r in ROLES}}

    async def end(self, state: str, say: str):
        """done / failed: tell everyone, stop the robot's assist, drop the frame."""
        await self.set_state(state, say)
        await self.send("robot", {"type": "cancel", "request_id": self.request["id"] if self.request else None})
        self.request = None         # the frame goes with it
        self.state = "idle"

    # -------------------------------------------------------------- wearer
    async def on_request(self, msg: dict):
        text = str(msg.get("text", "")).strip()
        try:
            if msg.get("frame_jpeg_b64"):
                frame = base64.b64decode(msg["frame_jpeg_b64"], validate=True)
            else:                   # a call, not a photo: use what the robot's camera sees right now
                frame = await self.robot_frame()
            w, h = jpeg_size(frame)
        except Exception:
            await self.send("wearer", {"type": "status", "request_id": None, "state": "failed",
                                       "say": "I can't see through the robot's camera right now. Please try again."})
            return
        if not text:
            await self.send("wearer", {"type": "status", "request_id": None, "state": "failed",
                                       "say": "I didn't catch that. Please say it again."})
            return
        async with self.lock:
            if self.state not in ("idle",):
                await self.send("wearer", {"type": "status", "request_id": self.request["id"], "state": self.state,
                                           "say": "Still working on your last request."})
                return
            self.request = {"id": uuid.uuid4().hex[:8], "text": text, "frame": frame, "task": None,
                            "log_frame": bool(msg.get("log_frame", False)), "t0": time.time()}
            self.log("request", text=text, frame_bytes=len(frame), size=[w, h])
            if self.request["log_frame"]:
                (LOG_DIR / f"frame_{self.request['id']}.jpg").write_bytes(frame)
            await self.set_state("extracting", "One moment.")
        try:
            task = await app.state.extract(frame, text, w, h)
        except Exception as e:      # extractor down or out of retries: fail the request, keep the session
            self.log("extract_error", error=str(e)[:300])
            task = extractor.keyword_task(text)     # Nemotron did not answer in time: plain word matching
            if task is None:
                await self.end("failed", "I couldn't work that out just now. Please try again.")
                return
        self.log("task", task=task)
        if task["status"] != "ok":  # ambiguous / none: the model's question goes back to the wearer
            await self.send("helper", {**task, "request_id": self.request["id"], "text": text})
            await self.end("failed", task["say"])
            return
        self.request["task"] = task
        await self.send("helper", {**task, "request_id": self.request["id"], "text": text,
                                   "frame_jpeg_b64": base64.b64encode(frame).decode()})
        if not self.sockets["helper"]:
            await self.end("failed", "No helper is connected right now.")
            return
        await self.set_state("offered", task["say"] + " Asking a helper.")

    async def robot_frame(self, timeout=4.0) -> bytes:
        self._frame = asyncio.get_running_loop().create_future()
        await self.send("robot", {"type": "frame_request", "id": "f"})
        b64 = await asyncio.wait_for(self._frame, timeout)
        return base64.b64decode(b64, validate=True)

    # -------------------------------------------------------------- helper
    async def on_helper_status(self, msg: dict):
        state = msg.get("state")
        if self.request is None or self.request["task"] is None:
            return
        task = self.request["task"]
        thing = "the " + (task["target"].get("name") or f'{task["target"]["color"]} {task["target"]["shape"]}')
        if state == "accepted" and self.state == "offered":
            await self.send("robot", {"type": "target", "request_id": self.request["id"],
                                      "name": task["target"].get("name"),
                                      "color": task["target"]["color"], "shape": task["target"]["shape"],
                                      "destination": task["destination"]})
            await self.set_state("accepted", msg.get("say") or f"A helper is getting {thing} for you.")
        elif state == "done" and self.state in ("accepted", "in_progress"):
            self.log("duration", seconds=round(time.time() - self.request["t0"], 1))
            await self.end("done", msg.get("say") or f"Here is {thing}.")
        elif state == "failed" and self.state in ("offered", "accepted", "in_progress"):
            await self.end("failed", msg.get("say") or "Sorry, the helper couldn't do that one.")

    # -------------------------------------------------------------- robot
    async def on_progress(self, msg: dict):
        if self.request is None:
            return
        self.log("progress", state=msg.get("state"))
        await self.send("helper", {"type": "progress", "request_id": self.request["id"], "state": msg.get("state")})
        state, task = msg.get("state"), self.request["task"]
        if msg.get("request_id") not in (None, self.request["id"]) or task is None:
            return
        thing = "the " + (task["target"].get("name") or f'{task["target"]["color"]} {task["target"]["shape"]}')
        if state == "in_progress" and self.state == "accepted":
            await self.set_state("in_progress", None)
        elif state == "grasped" and self.state in ("accepted", "in_progress"):
            await self.set_state("in_progress", f"The robot has {thing}.")
        elif state == "dropped" and self.state == "in_progress":
            await self.set_state("in_progress", "It slipped. Trying again.")
        elif state == "released" and self.state in ("accepted", "in_progress"):
            # the robot let go at the hand-over point: the request is finished without the helper pressing Done
            self.log("duration", seconds=round(time.time() - self.request["t0"], 1))
            await self.end("done", f"Here is {thing}.")


@app.websocket("/ws/{role}")
async def ws_endpoint(ws: WebSocket, role: str, session: str = "default"):
    if role == "caller":
        role = "wearer"
    if role not in ROLES or (ALLOWED_SESSIONS and session not in ALLOWED_SESSIONS):
        await ws.close(code=4000)   # unknown role, or not a session name this deployment accepts
        return
    await ws.accept()
    s = sessions.setdefault(session, Session(session))
    s.sockets[role].add(ws)
    for r in ROLES:
        await s.send(r, s.presence())
    try:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except ValueError:
                continue
            kind = msg.get("type") if isinstance(msg, dict) else None
            if role == "wearer" and kind == "request":
                asyncio.ensure_future(s.on_request(msg))    # do not block this socket on Nemotron
            elif role == "helper" and kind == "status":
                await s.on_helper_status(msg)
            elif role == "robot" and kind == "progress":
                await s.on_progress(msg)
            elif role == "robot" and kind == "frame":
                fut = getattr(s, "_frame", None)
                if fut is not None and not fut.done():
                    fut.set_result(msg.get("jpeg_b64") or "")
    except WebSocketDisconnect:
        pass
    finally:
        s.sockets[role].discard(ws)
        for r in ROLES:
            await s.send(r, s.presence())


@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "sessions": len(sessions)})    # no names: they are the secret


@app.get("/")
async def helper_page():
    return FileResponse(HELPER_PAGE)


@app.get("/quest")          # the worker's headset: the robot's eyes from Unreal, full screen
async def quest_page():
    return FileResponse(HELPER_PAGE.with_name("quest.html"))


@app.get("/mirror")         # the laptop's screen: what the worker sees in the headset, for the audience
async def mirror_page():
    return FileResponse(HELPER_PAGE.with_name("mirror.html"))


@app.get("/pov")            # full-screen robot camera: the phone in the video call to the worker's glasses shows this
async def pov_page():
    return FileResponse(HELPER_PAGE.with_name("pov.html"))


@app.get("/caller")         # the person asking for help
async def caller_page():
    return FileResponse(HELPER_PAGE.with_name("caller.html"))
