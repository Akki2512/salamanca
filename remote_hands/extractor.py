"""Task extractor: one first-person frame + one spoken sentence -> a structured robot task, using
NVIDIA Nemotron 3.5 Super VL through NVIDIA's OpenAI-compatible API.

Called once per request (never per frame). What is done to make the output usable by a robot:
  - the model lists what it sees and counts the matches BEFORE it commits to a target (key order in the
    reply); with its built-in reasoning switched off this is what keeps it from guessing
  - closed vocabulary: the model must pick colour, shape and destination from the lists the scene actually has
  - JSON only, validated here; one repair round-trip if the reply does not validate
  - the box is asked for in a fixed 0-1000 grid and converted to pixels here, then sanity-checked
  - the model must say when the sentence fits more than one object or none ("ambiguous" / "none") and
    produce the question to ask back, instead of guessing
  - retries with backoff on rate limits (free tier ~40 requests/minute per key)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time

import httpx

# The Junction key is a Verda inference key: Verda hosts the Nemotron endpoint with NVIDIA (challenge text).
# NVIDIA's own API (https://integrate.api.nvidia.com/v1, model "nvidia/nemotron-3.5-super-vl-120b-a12b")
# works the same way with an nvapi- key: set NEMOTRON_BASE_URL and NEMOTRON_MODEL.
BASE_URL = os.environ.get("NEMOTRON_BASE_URL", "https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1")
MODEL = os.environ.get("NEMOTRON_MODEL")      # None: looked up once from GET /models
from . import home

ITEMS = home.NAMES
DESTINATION = home.DESTINATION          # the robot hands the item to the person
_CATALOGUE = "\n".join(f'  - "{i["name"]}": {i["looks"]}' for i in home.ITEMS)

SYSTEM = f"""You turn what a person at home says into one fetch task for a robot helper.
The person is in a wheelchair with limited mobility and is asking for something from the bedside table. You get one
photo from the robot's camera, looking down at the bedside table, and the sentence the person said. The robot's
own arms and hands may be in the photo. Other things on the table (a lamp, a plant, a photo frame, a saucer) and
the bed beside it are not things the robot can fetch.
The things on the table that the robot can fetch:
{_CATALOGUE}

Reply with ONE JSON object and nothing else, with the keys in this order:
{{
  "visible": [{{"item": <item name>, "where": <"left" | "centre" | "right">}}, ...],
  "matches": <how many of the visible items fit the sentence>,
  "status": "ok" | "ambiguous" | "none",
  "target": {{"item": <item name>, "box": [x1, y1, x2, y2]}},
  "confidence": <0.0-1.0>,
  "instruction": <one short plain instruction for the human worker who will move the robot's arms>,
  "say": <one short friendly sentence spoken back to the person>
}}
Rules:
- Item names must be exactly from this list: {list(ITEMS)}.
- "visible": look carefully and list every item on the table first, each once, left to right.
- People ask in their own words: "I'm thirsty", "some water" = water jug; "what time is it" = alarm clock;
  "my tea" = tea cup. Work out which listed item they mean.
- "matches": count the entries of "visible" that fit the sentence. 1 -> status "ok"; 2 or more -> "ambiguous"
  unless the sentence says which (left / right / nearest); 0 -> "none".
- "box": the target's bounding box on a 0-1000 grid (x to the right, y down, 1000 = full width / height).
- status "ambiguous": more than one item fits and the sentence does not settle it. Set "target" to null and
  make "say" a question naming the choices.
- status "none": nothing visible fits, or the sentence is not asking for a thing on the table. Set "target" to
  null and make "say" explain briefly.
- "instruction" for status ok: e.g. "Pick up the water jug (on the right) and hand it to the person."
- "say" for status ok: confirm what will be fetched, e.g. "Getting your water jug."
"""


class ExtractionError(RuntimeError):
    pass


_model_cache = {}


async def resolve_model(client: httpx.AsyncClient, api_key: str) -> str:
    """The Nemotron vision model this key can call. Fails with a message that says what to do."""
    if MODEL:
        return MODEL
    if "id" not in _model_cache:
        r = await client.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {api_key}"})
        if r.status_code != 200:
            raise ExtractionError(f"listing models failed: http {r.status_code}: {r.text[:200]}")
        ids = [m["id"] for m in r.json().get("data", [])]
        nem = [i for i in ids if "nemotron" in i.lower()]
        if not nem:
            raise ExtractionError(f"this key can call no Nemotron model at {BASE_URL} (models enabled: {ids}). "
                                  "Enable Nemotron for the project in the provider's console.")
        vl = [i for i in nem if "vl" in i.lower() or "omni" in i.lower()]
        _model_cache["id"] = (vl or nem)[0]
    return _model_cache["id"]


def _parse(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def validate(d: dict, width: int, height: int) -> dict:
    """Check the model's reply and convert it to the `task` message. Raises ValueError with a reason the
    model can act on."""
    status = d.get("status")
    if status not in ("ok", "ambiguous", "none"):
        raise ValueError('"status" must be "ok", "ambiguous" or "none"')
    say = d.get("say")
    if not isinstance(say, str) or not say.strip():
        raise ValueError('"say" must be a non-empty sentence')
    visible = [v.get("item") for v in (d.get("visible") or []) if isinstance(v, dict) and v.get("item") in ITEMS]
    out = {"type": "task", "status": status, "action": "pick_and_place", "target": None,
           "destination": DESTINATION, "confidence": 0.0, "say": say.strip(), "visible": visible,
           "instruction": str(d.get("instruction") or "").strip(), "source": "nemotron", "model": MODEL}
    if status != "ok":
        return out
    t = d.get("target")
    if not isinstance(t, dict):
        raise ValueError('status "ok" needs a "target" object')
    if t.get("item") not in ITEMS:
        raise ValueError(f'"target.item" must be one of {list(ITEMS)}')
    item = home.BY_NAME[t["item"]]
    box = t.get("box")
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
        raise ValueError('"target.box" must be four numbers [x1, y1, x2, y2] on the 0-1000 grid')
    x1, y1, x2, y2 = (min(1000.0, max(0.0, float(v))) for v in box)
    if x2 - x1 < 5 or y2 - y1 < 5:
        raise ValueError('"target.box" is empty: x2 > x1 and y2 > y1 on the 0-1000 grid')
    if (x2 - x1) * (y2 - y1) > 0.25 * 1e6:
        raise ValueError('"target.box" covers more than a quarter of the photo; box only the one small object')
    conf = d.get("confidence")
    out["confidence"] = min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) else 0.5
    out["instruction"] = out["instruction"] or f'Pick up the {item["name"]} and hand it to the person.'
    out["target"] = {"name": item["name"], "color": item["color"], "shape": item["shape"],
                     "box_xyxy": [round(x1 * width / 1000), round(y1 * height / 1000),
                                  round(x2 * width / 1000), round(y2 * height / 1000)]}
    return out


def keyword_task(text: str) -> dict | None:
    """The same `task` message from plain word matching, for when Nemotron does not answer in time.
    One item named -> ok (no box: nothing looked at the picture). Several -> a question. None -> None."""
    hits = home.match_words(text)
    if not hits:
        return None
    out = {"type": "task", "action": "pick_and_place", "target": None, "destination": DESTINATION,
           "confidence": 0.5, "visible": [], "source": "keywords", "model": None, "instruction": ""}
    if len(hits) > 1:
        names = " or the ".join(h["name"] for h in hits)
        return {**out, "status": "ambiguous", "say": f"Do you mean the {names}?"}
    i = hits[0]
    return {**out, "status": "ok", "say": f'Getting your {i["name"]}.',
            "instruction": f'Pick up the {i["name"]} and hand it to the person.',
            "target": {"name": i["name"], "color": i["color"], "shape": i["shape"], "box_xyxy": None}}


async def _attempt(client, api_key, model, messages, width, height) -> dict:
    """One call, plus one repair call if the reply does not validate. Raises on failure."""
    msgs, err = list(messages), None
    for _ in range(2):
        r = await client.post(f"{BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {api_key}"},
                              json={"model": model, "messages": msgs, "temperature": 0.0, "max_tokens": 500,
                                    # The model reasons before answering by default (1-3k characters) and
                                    # sometimes spent the whole token budget on it, leaving an empty answer.
                                    # This task does not need it: off = ~120 output tokens instead of ~600.
                                    "chat_template_kwargs": {"enable_thinking": False}})
        if r.status_code != 200:
            raise ExtractionError(f"http {r.status_code}")
        reply = r.json()["choices"][0]["message"].get("content") or ""
        try:
            return validate(_parse(reply), width, height)
        except (ValueError, json.JSONDecodeError) as e:
            err = f"invalid reply: {str(e)[:80]}"
            msgs = list(messages) + [{"role": "assistant", "content": reply},
                                     {"role": "user", "content": f"That reply was rejected: {e}. "
                                      "Send the corrected JSON object only."}]
    raise ExtractionError(err)


async def extract(jpeg: bytes, text: str, width: int, height: int, api_key: str | None = None,
                  client: httpx.AsyncClient | None = None, max_tries: int = 3, attempt_timeout: float | None = None,
                  hedge_after: float = 4.0, budget_s: float | None = None) -> dict:
    """Returns the `task` message (see validate) with "latency_s", "attempts" and "attempt_errors" added.

    The shared endpoint is usually 1-4 s but some calls take 20 s or more. So attempts are hedged: if one has
    not answered after `hedge_after` seconds another is started alongside it (up to `max_tries`), and the
    first valid answer wins. The whole call gives up after `budget_s`: a wearer should hear "try again"
    rather than wait a minute."""
    api_key = api_key or os.environ.get("NEMOTRON_API_KEY") or os.environ["NVIDIA_API_KEY"]
    budget_s = float(os.environ.get("NEMOTRON_BUDGET_S", 22.0)) if budget_s is None else budget_s
    attempt_timeout = max(12.0, budget_s) if attempt_timeout is None else attempt_timeout
    hedge_after = max(hedge_after, budget_s / 3)        # a slow endpoint is not helped by piling calls on it
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(attempt_timeout, connect=5.0))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
            {"type": "text", "text": f'The person said: "{text}"'}]},
    ]
    t0, errors, running, started = time.monotonic(), [], set(), 0
    try:
        model = await resolve_model(client, api_key)
        while True:
            left = budget_s - (time.monotonic() - t0)
            if left <= 0 or (not running and started >= max_tries):
                break
            if started < max_tries and (not running or time.monotonic() - t0 >= started * hedge_after):
                running.add(asyncio.ensure_future(_attempt(client, api_key, model, messages, width, height)))
                started += 1
            next_hedge = started * hedge_after - (time.monotonic() - t0) if started < max_tries else left
            done, running = await asyncio.wait(running, timeout=max(0.05, min(left, next_hedge)),
                                               return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                if d.exception() is None:
                    task = d.result()
                    task.update(model=model, latency_s=round(time.monotonic() - t0, 2), attempts=started,
                                attempt_errors=errors)
                    return task
                e = d.exception()
                errors.append(str(e) if isinstance(e, ExtractionError) else f"network: {type(e).__name__}")
        raise ExtractionError(f"no valid task from {started} attempts in {time.monotonic() - t0:.0f} s ({errors})")
    finally:
        for t in running:
            t.cancel()
        if own:
            await asyncio.gather(*running, return_exceptions=True)
            await client.aclose()
