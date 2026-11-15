"""Language labels for recorded episodes from Nemotron (a vision-language model), next to the recorded ones.

    .venv/bin/python tools/label_with_nemotron.py <episodes_root_or_episode_dir>
        [--max-segments 8] [--timeout 120] [--camera robot_eyes] [--redo] [--dry-run]

What it does for each episode:
  1. cuts the episode into segments where the recorded labels change (story phase, grasp state, floor reach,
     wheelchair pushing, what is held), merges the very short ones, and keeps at most --max-segments
  2. shows Nemotron the camera frame from the middle of each segment and asks what the robot is doing there.
     The model is NOT told the recorded label: its answer is an independent second opinion
  3. shows it three frames (start, middle, end) and asks for the task of the whole episode and whether it worked
  4. writes <episode>/labels/nemotron.json and prints a table: recorded label next to Nemotron's sentence

The model has been slow (50-80 s a call), so: few calls, a long timeout, one retry, the file is saved after
every call, and a second run only does what is still missing (--redo starts again).
"""
import argparse, asyncio, base64, io, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCENE = ("A humanoid robot (Unitree G1), driven remotely by a person, helps an elderly man in a wheelchair in "
         "his bedroom: it tidies things off the floor, brings him things from the bedside table and moves his "
         "wheelchair.")
VIEWS = {"robot_eyes": "The photo is from the robot's own head camera, so its arms and hands may be in view.",
         "person": "The photo is from a camera in the room that looks at the man in the wheelchair."}
GROUP = {None: "idle", "free": "idle", "done": "idle", "approach": "picking", "descend": "picking",
         "closing": "picking", "holding": "holding", "releasing": "releasing"}
MIN_SEGMENT_S = 0.5
MAX_WIDTH = 768


# ---------------------------------------------------------------- segments from the recorded labels
def truth_of(step: dict) -> dict:
    """The recorded facts that define a segment, as a small dict."""
    lab = step.get("label") or {}
    held = sorted(i["name"] for i in step.get("items") or [] if i.get("held_by"))
    return {"phase": lab.get("phase"), "assist": GROUP.get(lab.get("assist"), "idle"),
            "reach": lab.get("reach") is not None, "pushing": bool(lab.get("pushing")), "held": held}


def _merge_shortest(segs: list) -> None:
    """Merge the shortest segment into a neighbour: the one whose recorded facts differ least, the longer one
    on a tie. The merged segment keeps the facts of its longer part."""
    n = lambda s: s["b"] - s["a"] + 1
    j = min(range(len(segs)), key=lambda k: (n(segs[k]), k))
    same = lambda k: sum(segs[j]["truth"][f] == segs[k]["truth"][f] for f in segs[j]["truth"])
    k = max((k for k in (j - 1, j + 1) if 0 <= k < len(segs)), key=lambda k: (same(k), n(segs[k])))
    lo, hi = min(j, k), max(j, k)
    truth = segs[j]["truth"] if n(segs[j]) > n(segs[k]) else segs[k]["truth"]
    segs[lo:hi + 1] = [{"a": segs[lo]["a"], "b": segs[hi]["b"], "truth": truth}]


def segment(steps: list, fps: float = 30.0, max_segments: int = 8, min_s: float = MIN_SEGMENT_S) -> list:
    """Cut the ticks into segments. "a" and "b" are the first and last position in `steps` (both included)."""
    segs = []
    for n, s in enumerate(steps):
        t = truth_of(s)
        if segs and segs[-1]["truth"] == t:
            segs[-1]["b"] = n
        else:
            segs.append({"a": n, "b": n, "truth": t})
    short = lambda: min(s["b"] - s["a"] + 1 for s in segs) / fps < min_s
    while len(segs) > 1 and (short() or len(segs) > max(1, max_segments)):
        _merge_shortest(segs)
    for s in segs:
        first, last = steps[s["a"]], steps[s["b"]]
        s.update(from_i=first.get("i", s["a"]), to_i=last.get("i", s["b"]), t0=first.get("t"), t1=last.get("t"))
    return segs


def frame_near(steps: list, n: int, camera: str, a: int = 0, b: int | None = None) -> int | None:
    """The saved frame of `camera` at tick position n, or at the nearest tick in [a, b] that has one."""
    b = len(steps) - 1 if b is None else b
    for d in range(max(n - a, b - n) + 1):
        for m in (n - d, n + d):
            if a <= m <= b:
                k = (steps[m].get("frame") or {}).get(camera)
                if k is not None:
                    return int(k)
    return None


def middle_frame(steps: list, seg: dict, camera: str) -> int | None:
    return frame_near(steps, (seg["a"] + seg["b"]) // 2, camera, seg["a"], seg["b"])


# ---------------------------------------------------------------- the model
def data_url(path: Path, max_width: int = MAX_WIDTH) -> str:
    """The picture as a base64 JPEG, made smaller first so the request stays small."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def nemotron_call(messages: list, timeout: float) -> tuple[str, str]:
    """ONE request to Nemotron. Returns (reply text, model id). The only function here that touches the
    network: tests pass their own function instead."""
    import httpx
    from remote_hands import extractor as ex          # importing remote_hands loads .env (key, endpoint, model)
    key = os.environ.get("NEMOTRON_API_KEY") or os.environ["NVIDIA_API_KEY"]

    async def go():
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
            model = await ex.resolve_model(client, key)
            r = await client.post(f"{ex.BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                                  json={"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 300,
                                        "chat_template_kwargs": {"enable_thinking": False}})   # see extractor.py
            if r.status_code != 200:
                raise ex.ExtractionError(f"http {r.status_code}")
            return r.json()["choices"][0]["message"].get("content") or "", model
    return asyncio.run(go())


def _clean(e: Exception) -> str:
    """Error text for the file and the screen, with the key and the endpoint address taken out."""
    text = f"{type(e).__name__}: {e}"[:300]
    for name in ("NEMOTRON_API_KEY", "NVIDIA_API_KEY", "NEMOTRON_BASE_URL"):
        if os.environ.get(name):
            text = text.replace(os.environ[name], "<hidden>")
    ex = sys.modules.get("remote_hands.extractor")
    return text.replace(ex.BASE_URL, "<hidden>") if ex else text


def ask(call, messages: list, timeout: float, stats: dict) -> tuple[dict | None, str | None, float]:
    """Ask for a JSON object, once more if the first try fails. Returns (reply, error text, seconds)."""
    from remote_hands.extractor import _parse
    t0, err = time.monotonic(), None
    for _ in range(2):
        stats["calls"] += 1
        try:
            text, model = call(messages, timeout)
            stats["model"] = model or stats.get("model")
            d = _parse(text)
            if not isinstance(d, dict):
                raise ValueError("the reply is not a JSON object")
            return d, None, round(time.monotonic() - t0, 2)
        except Exception as e:                        # a failed call must never stop the run
            err = _clean(e)
    return None, err, round(time.monotonic() - t0, 2)


def segment_messages(image: str, items: list, camera: str) -> list:
    system = (f"{SCENE} {VIEWS.get(camera, '')}\nYou label short pieces of a recording of the robot for training data. "
              f"Things in the room that the robot can handle: {items}.\n"
              "Say what the robot is doing at the moment of the photo as one short instruction in the imperative, "
              'as if telling the robot to do it (e.g. "Walk to the bedside table.").\n'
              "Reply with ONE JSON object and nothing else:\n"
              '{"instruction": <the instruction>, "objects": [<the things the robot is acting on, names from the '
              'list where they fit>], "confidence": <0.0-1.0>}')
    return [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                         {"type": "text", "text": "What is the robot doing here?"}]}]


def episode_messages(images: list, items: list, camera: str) -> list:
    system = (f"{SCENE} {VIEWS.get(camera, '')}\nYou get {len(images)} photo(s) from one recording, in time order "
              f"(start to end). Things in the room that the robot can handle: {items}.\n"
              "Say in one sentence what task the robot did over the whole recording, and whether it succeeded.\n"
              "Reply with ONE JSON object and nothing else:\n"
              '{"task": <one sentence>, "success": true | false | null (null = cannot tell from the photos), '
              '"reason": <one short sentence>}')
    return [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}} for u in images]
             + [{"type": "text", "text": "What was the task, and did it succeed?"}]}]


# ---------------------------------------------------------------- one episode
def _read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _save(path: Path, out: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(path)                                 # never leaves a half-written file behind


def _truth_text(t: dict) -> str:
    return " ".join([t["phase"] or "-", t["assist"]] + ["reach"] * t["reach"] + ["pushing"] * t["pushing"]
                    + (["holds " + "+".join(t["held"])] if t["held"] else []))


def label_episode(ep, call=nemotron_call, max_segments: int = 8, timeout: float = 120.0, camera: str = "robot_eyes",
                  redo: bool = False, dry_run: bool = False) -> dict:
    """Label one episode folder; returns what was (or, with dry_run, would be) written."""
    ep = Path(ep)
    meta = json.loads((ep / "meta.json").read_text()) if (ep / "meta.json").exists() else {}
    steps = _read_jsonl(ep / "steps.jsonl")
    items = sorted({i["name"] for s in steps for i in s.get("items") or []})
    fdir = ep / "frames" / camera
    jpg = lambda k: fdir / f"{k:06d}.jpg"
    has = lambda k: k is not None and jpg(k).exists()
    path = ep / "labels" / "nemotron.json"
    old = json.loads(path.read_text()) if path.exists() and not redo else {}
    if old.get("camera") != camera:
        old = {}                                      # labels from another camera are not the same labels
    old_segs = {(s["from_i"], s["to_i"]): s for s in old.get("segments", [])}

    out = {"model": old.get("model"), "camera": camera, "task": old.get("task"), "success": old.get("success"),
           "reason": old.get("reason"), "error": None, "segments": [], "calls": old.get("calls", 0),
           "took_s": old.get("took_s", 0.0)}
    for s in segment(steps, float(meta.get("fps") or 30.0), max_segments):
        k = middle_frame(steps, s, camera)
        row = {"from_i": s["from_i"], "to_i": s["to_i"], "t0": s["t0"], "t1": s["t1"], "frame_k": k if has(k) else None,
               "truth": s["truth"], "instruction": None, "objects": [], "confidence": None, "error": None, "took_s": 0.0}
        done = old_segs.get((s["from_i"], s["to_i"]))
        if done and done.get("instruction"):          # labelled on an earlier run: keep it
            row.update({f: done.get(f) for f in ("frame_k", "instruction", "objects", "confidence", "took_s")})
        out["segments"].append(row)
    ends = [frame_near(steps, n, camera) for n in (0, len(steps) // 2, len(steps) - 1)] if steps else []
    ends = [k for n, k in enumerate(ends) if has(k) and k not in ends[:n]]          # start, middle, end, each once
    todo = [r for r in out["segments"] if not r["instruction"] and r["frame_k"] is not None]
    need_task = out["task"] is None and bool(ends)

    print(f"\n{ep.name}: {len(steps)} ticks, {len(out['segments'])} segments, items {items}, "
          f"recorded task {meta.get('task')!r}, recorded success {meta.get('success')}")
    if not fdir.exists() or not ends:
        print(f"   no saved frames for camera {camera!r}: nothing to show the model")
    print(f"   calls to make: {len(todo)} segment(s) + {int(need_task)} for the whole episode"
          + ("   (dry run: none made)" if dry_run else ""))
    if not dry_run:
        stats = {"calls": 0, "model": out["model"]}

        def account(took):
            out.update(model=stats["model"], calls=old.get("calls", 0) + stats["calls"],
                       took_s=round(out["took_s"] + took, 2))
            _save(path, out)                          # after every call: a stopped run loses nothing

        for r in todo:
            try:
                d, r["error"], r["took_s"] = ask(call, segment_messages(data_url(jpg(r["frame_k"])), items, camera),
                                                 timeout, stats)
            except Exception as e:                    # e.g. a frame file that cannot be read
                d, r["error"], r["took_s"] = None, _clean(e), 0.0
            if d is not None:
                c = d.get("confidence")
                r["instruction"] = str(d.get("instruction") or "").strip() or None
                r["objects"] = [str(o) for o in d.get("objects") or []] if isinstance(d.get("objects"), list) else []
                r["confidence"] = min(1.0, max(0.0, float(c))) if isinstance(c, (int, float)) else None
                r["error"] = None if r["instruction"] else "ValueError: the reply has no instruction"
            account(r["took_s"])
            print(f"   segment {r['from_i']}-{r['to_i']}: {r['took_s']:.1f} s  {r['error'] or 'ok'}", flush=True)
        if need_task:
            try:
                d, out["error"], took = ask(call, episode_messages([data_url(jpg(k)) for k in ends], items, camera),
                                            timeout, stats)
            except Exception as e:
                d, out["error"], took = None, _clean(e), 0.0
            if d is not None:
                out["task"] = str(d.get("task") or "").strip() or None
                out["success"] = d.get("success") if isinstance(d.get("success"), bool) else None
                out["reason"] = str(d.get("reason") or "").strip() or None
            account(took)
            print(f"   whole episode: {took:.1f} s  {out['error'] or 'ok'}", flush=True)
        _save(path, out)

    print(f"   {'time':>13}  {'frame':>6}  {'recorded':<34}  Nemotron")
    for r in out["segments"]:
        t = f"{r['t0']:6.1f}-{r['t1']:6.1f}" if r["t0"] is not None and r["t1"] is not None else f"{r['from_i']}-{r['to_i']}"
        said = r["instruction"] or (f"[{r['error']}]" if r["error"] else "[no frame]" if r["frame_k"] is None else "-")
        conf = f"  ({r['confidence']:.2f})" if r["confidence"] is not None else ""
        print(f"   {t:>13}  {'-' if r['frame_k'] is None else r['frame_k']:>6}  {_truth_text(r['truth']):<34}  {said}{conf}")
    if not dry_run or out["task"]:
        print(f"   task: {out['task']!r}   success: {out['success']}   reason: {out['reason']!r}"
              + (f"   [{out['error']}]" if out["error"] else ""))
    if not dry_run:
        print(f"   {out['calls']} call(s), {out['took_s']:.1f} s in all -> {path}")
    return out


def episodes_in(root) -> list:
    root = Path(root)
    if (root / "steps.jsonl").exists():
        return [root]
    return sorted(p for p in root.glob("episode_*") if (p / "steps.jsonl").exists())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="a folder of episode_NNNNNN folders, or one episode folder")
    ap.add_argument("--max-segments", type=int, default=8, help="most segments (= calls) per episode")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for one reply")
    ap.add_argument("--camera", default="robot_eyes", help="which frames/<camera> to show the model")
    ap.add_argument("--redo", action="store_true", help="label again what is already labelled")
    ap.add_argument("--dry-run", action="store_true", help="print the segments and the number of calls, call nothing")
    a = ap.parse_args(argv)
    eps = episodes_in(a.root)
    if not eps:
        sys.exit(f"no episode (a folder with steps.jsonl) under {a.root}")
    for ep in eps:
        label_episode(ep, max_segments=a.max_segments, timeout=a.timeout, camera=a.camera, redo=a.redo,
                      dry_run=a.dry_run)


if __name__ == "__main__":
    main()
