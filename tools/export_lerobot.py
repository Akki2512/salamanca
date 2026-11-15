"""Turn recorded teleoperation episodes into a LeRobot dataset (Hugging Face `LeRobotDataset`).

    .venv/bin/python tools/export_lerobot.py <episodes_root> <out_dir> [--repo-id NAME] [--fps 30] [--video-size WxH]

Target: LeRobot dataset format v3.0 ("codebase_version": "v3.0"), the format of lerobot >= 0.4.0 and of the
current main branch. v2.1 (one parquet and one mp4 per episode, meta/*.jsonl) is the old layout: current lerobot
does not load it, it asks for a conversion to v3.0, so writing it would only add a step. Checked on 15 Nov 2026
against:
    https://huggingface.co/docs/lerobot/lerobot-dataset-v3                      (layout, what each file is)
    https://github.com/huggingface/lerobot/tree/main/src/lerobot/datasets       (utils.py: paths and DatasetInfo;
        dataset_metadata.py + dataset_writer.py: episode columns; compute_stats.py: stats; io_utils.py: tasks)
    https://huggingface.co/datasets/lerobot/pusht/tree/main                     (a real v3.0 dataset, info.json)

What is written:
    meta/info.json                               features, fps, counts, path templates
    meta/stats.json                              min / max / mean / std / count / q01 q10 q50 q90 q99 per feature
    meta/tasks.parquet                           task text (as the table index "task") -> task_index
    meta/episodes/chunk-000/file-000.parquet     one row per episode: length, tasks, where its rows and its video
                                                 seconds are, and its own stats ("stats/<feature>/<stat>")
    data/chunk-000/file-000.parquet              one row per 30 Hz step, all episodes, one row group per episode
    videos/<camera key>/chunk-000/file-000.mp4   all episodes of one camera one after the other, one frame per step
    extras/episode_000000.jsonl, extras/episodes.jsonl    NOT part of LeRobot: labels, items, hands, eye, suit and
                                                 the recorded time of every step, and each episode's meta.json
Everything goes into ONE data file and ONE video file per camera. LeRobot itself starts a new file at 100 MB /
200 MB; a reader finds episodes through meta/episodes, not through file sizes, so one file is valid, only big.

observation.state = arm 14 + hand 14 + waist 3 + leg 12 + base 4 = 47 numbers. action has the same layout: the
commanded arm and hand angles of this step, then waist, legs and base of the NEXT step's state (nobody commands
those directly: the walker makes them, so "where they go next" is the action). The last step repeats itself.
timestamp is frame_index / fps as LeRobot wants; the recorded time of each step is kept in extras.

Cameras are saved slower than the steps (about 15 and 5 pictures a second), so each step shows the newest saved
picture (its "frame" index) and a picture is repeated until the next one comes. Steps before the first picture
show the episode's first picture; an episode with no picture for a camera that other episodes have is black. A
camera with no picture in any episode is left out of the dataset, and the tool says so. Pictures of other sizes
are fitted into the video size with black bars (not stretched).

The parquet files need pyarrow, which is imported only when they are written. Everything else (reading episodes,
arrays, stats, JSON, videos, extras) works without it. Videos are made by ffmpeg (H.264, yuv420p, a keyframe every
2 frames like LeRobot's own, so any frame is quick to reach).
"""
import argparse, collections, json, shutil, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_hands.unreal_stream import ARM, HAND      # the 14 arm and 14 hand joint names, in recording order

VERSION = "v3.0"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
CAMERAS = ("robot_eyes", "person")
WAIST = [f"waist_{a}_joint" for a in ("yaw", "roll", "pitch")]
LEG = [f"{s}_{j}_joint" for s in ("left", "right") for j in
       ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
BASE = ["base_x", "base_y", "base_yaw", "base_dz"]
NAMES = ARM + HAND + WAIST + LEG + BASE               # the 47 dimensions of observation.state and of action
NEXT = len(ARM) + len(HAND)                           # action[NEXT:] comes from the next step's state
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
CRF, GOP = 23, 2
NEEDS_PYARROW = ("pyarrow is not installed, and the LeRobot data files are parquet. "
                 "Install it with:  .venv/bin/pip install pyarrow")


def video_key(cam):
    return "observation.images." + cam


# ---------------------------------------------------------------- reading the recording

def load_episode(d):
    """One episode folder -> {"dir", "meta", "steps", "frames": {camera: set of saved picture numbers}}, or None
    when it is not complete (no meta.json, not ended yet, or no steps)."""
    d = Path(d)
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        return None
    if not meta.get("ended") or not (d / "steps.jsonl").is_file():
        return None
    steps = []
    for line in (d / "steps.jsonl").read_text().splitlines():
        try:
            steps.append(json.loads(line))
        except ValueError:                             # a half-written last line
            break
    if not steps:
        return None
    frames = {c: {int(p.stem) for p in (d / "frames" / c).glob("*.jpg") if p.stem.isdigit()} for c in CAMERAS}
    return {"dir": d, "meta": meta, "steps": steps, "frames": frames}


def load_episodes(root):
    """All complete episodes under root, in folder order, and the names of the folders that were skipped."""
    eps, skipped = [], []
    for d in sorted(Path(root).glob("episode_*")):
        if d.is_dir():
            ep = load_episode(d)
            eps.append(ep) if ep else skipped.append(d.name)
    return eps, skipped


def episode_arrays(ep):
    """(state, action): two float32 arrays of shape (steps, 47). See the module text for the action rule."""
    def row(s, arm, hand):
        st = s["state"]
        return list(arm) + list(hand) + list(st["waist_q"]) + list(st["leg_q"]) + list(st["base"])
    steps = ep["steps"]
    state = np.array([row(s, s["state"]["arm_q"], s["state"]["hand_q"]) for s in steps], np.float32)
    action = np.array([row(n, s["action"]["arm_q"], s["action"]["hand_q"])
                       for s, n in zip(steps, steps[1:] + steps[-1:])], np.float32)
    if state.shape[1] != len(NAMES):
        raise ValueError(f"{ep['dir'].name}: a step has {state.shape[1]} state numbers, expected {len(NAMES)}")
    return state, action


def frame_refs(ep, cam):
    """For every step, the number of the saved picture of `cam` to show, or None for all steps when the episode
    has no picture of it. A step whose index is empty or points to a file that is not there repeats the one
    before; steps before the first picture get the first one."""
    have, out, last = ep["frames"].get(cam) or set(), [], None
    for s in ep["steps"]:
        k = (s.get("frame") or {}).get(cam)
        if k is not None and k in have:
            last = k
        out.append(last)
    first = next((k for k in out if k is not None), None)
    return [first if k is None else k for k in out]


def task_text(ep):
    """The task of an episode: what meta.json says, else the label seen on most steps, else "teleoperation"."""
    t = ep["meta"].get("task")
    if isinstance(t, str) and t.strip():
        return t.strip()
    seen = collections.Counter(t for s in ep["steps"] if (t := (s.get("label") or {}).get("task")))
    return seen.most_common(1)[0][0] if seen else "teleoperation"


def task_table(eps):
    """(tasks, per episode task_index): each different task text once, in order of first use."""
    tasks = []
    for ep in eps:
        if (t := task_text(ep)) not in tasks:
            tasks.append(t)
    return tasks, [tasks.index(task_text(ep)) for ep in eps]


# ---------------------------------------------------------------- stats

def feature_stats(a, count=None):
    """Stats of a (rows, dims) array per dimension, in the shape LeRobot keeps them: lists of length dims."""
    a = np.asarray(a, np.float64)
    a = a.reshape(len(a), -1)
    st = {"min": a.min(0), "max": a.max(0), "mean": a.mean(0), "std": a.std(0),
          "count": np.array([len(a) if count is None else count])}
    for q in QUANTILES:
        st[f"q{int(q * 100):02d}"] = np.quantile(a, q, axis=0)
    return st


def image_stats(samples, count):
    """Stats of pictures per colour channel, 0..1, each of shape (3, 1, 1) as LeRobot keeps them. `samples` is
    (pictures, height, width, 3) uint8; `count` is the number of video frames they stand for."""
    px = np.asarray(samples).reshape(-1, 3) / 255.0
    return {k: v if k == "count" else v.reshape(3, 1, 1) for k, v in feature_stats(px, count).items()}


def sample_pictures(paths, size, most=100, small=160):
    """Up to `most` pictures spread evenly over `paths` (None = black), made small, as one uint8 array."""
    w, h = size
    step = max(1, max(w, h) // small)
    idx = sorted(set(np.round(np.linspace(0, len(paths) - 1, min(most, len(paths)))).astype(int).tolist()))
    return np.stack([np.frombuffer(picture_bytes(paths[i], size), np.uint8).reshape(h, w, 3)[::step, ::step]
                     for i in idx])


def to_lists(stats):
    return {k: {s: np.asarray(v).tolist() for s, v in st.items()} for k, st in stats.items()}


# ---------------------------------------------------------------- pictures and videos

def parse_size(text):
    w, h = (int(v) for v in text.lower().split("x"))
    return even((w, h))


def even(size):
    """yuv420p video needs an even width and height."""
    return tuple(max(2, int(v) // 2 * 2) for v in size)


def camera_sizes(eps, forced=None):
    """{camera: (width, height)} for each camera that has a picture in some episode: the size of its first
    picture, or `forced` for all."""
    from PIL import Image
    sizes = {}
    for cam in CAMERAS:
        for ep in eps:
            k = next((k for k in frame_refs(ep, cam) if k is not None), None)
            if k is not None:
                with Image.open(picture_path(ep, cam, k)) as im:
                    sizes[cam] = forced or even(im.size)
                break
    return sizes


def picture_path(ep, cam, k):
    return None if k is None else ep["dir"] / "frames" / cam / f"{k:06d}.jpg"


def picture_bytes(path, size):
    """The picture as raw RGB bytes of exactly `size`, fitted in with black bars. None = a black picture."""
    from PIL import Image, ImageOps
    if path is None:
        return bytes(size[0] * size[1] * 3)
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != tuple(size):
            im = ImageOps.pad(im, tuple(size), Image.BILINEAR, color=(0, 0, 0))
        return im.tobytes()


def write_video(paths, size, fps, dest):
    """Encode one picture per entry of `paths` (None = black) into an mp4 with exactly len(paths) frames."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(GOP),
           "-crf", str(CRF), "-movflags", "+faststart", str(dest)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    last, raw = False, None
    try:
        for path in paths:
            if path != last:                           # repeated pictures are decoded once
                last, raw = path, picture_bytes(path, size)
            p.stdin.write(raw)
        p.stdin.close()
    except BrokenPipeError:
        pass
    with p.stderr:
        err = p.stderr.read().decode(errors="replace")
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {dest}: {err.strip()}")
    return dest


# ---------------------------------------------------------------- the dataset tables (plain Python, no pyarrow)

def build(eps, fps=30, sizes=None):
    """Everything that goes into the dataset except the files themselves:
    {"tasks", "columns" (the data table, column name -> array), "episodes" (one dict per episode, the rows of
    meta/episodes), "stats", "pictures" ({camera: one path or None per data row})}."""
    sizes = sizes or {}
    tasks, task_of = task_table(eps)
    cols = collections.defaultdict(list)
    pictures = {c: [] for c in sizes}
    rows, samples, start = [], {c: [] for c in sizes}, 0
    for e, ep in enumerate(eps):
        state, action = episode_arrays(ep)
        n = len(state)
        frame = np.arange(n, dtype=np.int64)
        part = {"observation.state": state, "action": action, "timestamp": (frame / fps).astype(np.float32),
                "frame_index": frame, "episode_index": np.full(n, e, np.int64), "index": frame + start,
                "task_index": np.full(n, task_of[e], np.int64)}
        ep_stats = {k: feature_stats(v) for k, v in part.items()}
        row = {"episode_index": e, "tasks": [tasks[task_of[e]]], "length": n,
               "data/chunk_index": 0, "data/file_index": 0, "dataset_from_index": start, "dataset_to_index": start + n}
        for cam, size in sizes.items():
            paths = [picture_path(ep, cam, k) for k in frame_refs(ep, cam)]
            pictures[cam] += paths
            samples[cam].append(sample_pictures(paths, size))
            ep_stats[video_key(cam)] = image_stats(samples[cam][-1], n)
            row.update({f"videos/{video_key(cam)}/chunk_index": 0, f"videos/{video_key(cam)}/file_index": 0,
                        f"videos/{video_key(cam)}/from_timestamp": start / fps,
                        f"videos/{video_key(cam)}/to_timestamp": (start + n) / fps})
        row.update({"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        row.update({f"stats/{k}/{s}": v for k, st in to_lists(ep_stats).items() for s, v in st.items()})
        rows.append(row)
        for k, v in part.items():
            cols[k].append(v)
        start += n
    columns = {k: np.concatenate(v) for k, v in cols.items()}
    stats = {k: feature_stats(v) for k, v in columns.items()}
    for cam in sizes:                                  # pictures: the samples of all episodes together
        stats[video_key(cam)] = image_stats(np.concatenate(samples[cam]), start)
    return {"tasks": tasks, "columns": columns, "episodes": rows, "stats": to_lists(stats), "pictures": pictures}


def features(fps, sizes):
    """The "features" part of meta/info.json."""
    f = {"observation.state": {"dtype": "float32", "shape": [len(NAMES)], "names": list(NAMES)},
         "action": {"dtype": "float32", "shape": [len(NAMES)], "names": list(NAMES)}}
    for cam, (w, h) in sizes.items():
        f[video_key(cam)] = {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
                             "info": {"video.height": h, "video.width": w, "video.codec": "h264",
                                      "video.pix_fmt": "yuv420p", "video.fps": fps, "video.channels": 3,
                                      "has_audio": False, "video.g": GOP, "video.crf": CRF, "is_depth_map": False}}
    f["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for k in ("frame_index", "episode_index", "index", "task_index"):
        f[k] = {"dtype": "int64", "shape": [1], "names": None}
    return f


def info(built, fps, sizes, robot_type=None):
    """The content of meta/info.json."""
    n = len(built["episodes"])
    return {"codebase_version": VERSION, "robot_type": robot_type, "total_episodes": n,
            "total_frames": int(len(built["columns"]["index"])), "total_tasks": len(built["tasks"]),
            "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200, "fps": fps,
            "splits": {"train": f"0:{n}"}, "data_path": DATA_PATH, "video_path": VIDEO_PATH if sizes else None,
            "features": features(fps, sizes)}


def write_meta(out, built, fps, sizes, robot_type=None):
    """meta/info.json and meta/stats.json (the two meta files that are JSON)."""
    out = Path(out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta/info.json").write_text(json.dumps(info(built, fps, sizes, robot_type), indent=4))
    (out / "meta/stats.json").write_text(json.dumps(built["stats"], indent=4))


def write_extras(out, eps):
    """extras/: every step with its vectors taken out (they are in the dataset) and all the rest untouched, plus
    one line per episode with its source folder and meta.json. `index` is the row in the LeRobot data."""
    ex = Path(out) / "extras"
    ex.mkdir(parents=True, exist_ok=True)
    index = 0
    with open(ex / "episodes.jsonl", "w") as summary:
        for e, ep in enumerate(eps):
            with open(ex / f"episode_{e:06d}.jsonl", "w") as f:
                for n, s in enumerate(ep["steps"]):
                    rest = {k: v for k, v in s.items() if k not in ("state", "action")}
                    f.write(json.dumps({"index": index, "episode_index": e, "frame_index": n, **rest}) + "\n")
                    index += 1
            summary.write(json.dumps({"episode_index": e, "source": ep["dir"].name, "task": task_text(ep),
                                      "meta": ep["meta"]}) + "\n")


# ---------------------------------------------------------------- parquet (the only part that needs pyarrow)

def need_pyarrow():
    try:
        import pyarrow as pa, pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(NEEDS_PYARROW) from None
    return pa, pq


def write_parquet(out, built):
    """data/, meta/tasks.parquet and meta/episodes/ as LeRobot v3.0 lays them out."""
    pa, pq = need_pyarrow()
    out, c = Path(out), built["columns"]
    # the step table: vectors as fixed-size lists of float32 (what `datasets.Sequence(length=47)` is in arrow)
    # the list's inner field is named "item", as Hugging Face datasets writes it (pyarrow's own default name differs by version)
    vec = lambda a: pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), pa.float32()),
                                                      type=pa.list_(pa.field("item", pa.float32()), a.shape[1]))
    table = pa.table({"observation.state": vec(c["observation.state"]), "action": vec(c["action"]),
                      "timestamp": pa.array(c["timestamp"], pa.float32()),
                      **{k: pa.array(c[k], pa.int64()) for k in ("frame_index", "episode_index", "index", "task_index")}})
    path = out / DATA_PATH.format(chunk_index=0, file_index=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(str(path), table.schema, compression="snappy") as w:
        for ep in built["episodes"]:                   # one row group per episode, as LeRobot's writer does
            w.write_table(table.slice(ep["dataset_from_index"], ep["length"]), row_group_size=ep["length"])
    # tasks: LeRobot writes a pandas table whose INDEX is the task text. The "pandas" note in the file is what
    # makes pandas.read_parquet give that index back, so it is written by hand here (pandas is not needed).
    tasks = pa.table({"task_index": pa.array(range(len(built["tasks"])), pa.int64()),
                      "task": pa.array(built["tasks"], pa.string())})
    note = {"index_columns": ["task"], "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode",
                                                           "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}],
            "columns": [{"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
                         "numpy_type": "int64", "metadata": None},
                        {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object",
                         "metadata": None}],
            "creator": {"library": "pyarrow", "version": pa.__version__}, "pandas_version": "2.2.0"}
    pq.write_table(tasks.replace_schema_metadata({"pandas": json.dumps(note)}), str(out / "meta/tasks.parquet"))
    # episodes: one row each, the keys of built["episodes"] as columns
    rows = built["episodes"]
    path = out / EPISODES_PATH.format(chunk_index=0, file_index=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), str(path), compression="snappy")


# ---------------------------------------------------------------- command line

def export(root, out, fps=30, video_size=None, repo_id=None, log=print):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty. Give a new folder, so old and new files do not mix.")
    need_pyarrow()                                     # stop now, not after minutes of video encoding
    eps, skipped = load_episodes(root)
    if skipped:
        log(f"skipped, not complete: {', '.join(skipped)}")
    if not eps:
        raise SystemExit(f"no complete episode in {root}")
    sizes = camera_sizes(eps, video_size)
    for cam in CAMERAS:
        if cam not in sizes:
            log(f"camera {cam}: no saved picture in any episode, left out of the dataset")
    if sizes and not Path(FFMPEG).exists():
        raise SystemExit(f"ffmpeg not found ({FFMPEG}); it is needed for the videos")
    recorded = {ep["meta"].get("fps") for ep in eps} - {None, fps}
    if recorded:
        log(f"warning: recorded at {sorted(recorded)} steps a second, written as {fps}")
    built = build(eps, fps, sizes)
    robots = {ep["meta"].get("robot") for ep in eps} - {None}
    write_meta(out, built, fps, sizes, robots.pop() if len(robots) == 1 else None)
    write_extras(out, eps)
    write_parquet(out, built)
    for cam, size in sizes.items():
        write_video(built["pictures"][cam], size, fps,
                    out / VIDEO_PATH.format(video_key=video_key(cam), chunk_index=0, file_index=0))
        none = [e for e, ep in enumerate(eps) if not ep["frames"][cam]]
        log(f"camera {cam}: {size[0]}x{size[1]}" + (f", black in episodes {none} (nothing saved)" if none else ""))
    log(f"{len(eps)} episodes, {len(built['columns']['index'])} steps, {len(built['tasks'])} tasks -> {out}"
        + (f"   (load: LeRobotDataset('{repo_id}', root='{out}'))" if repo_id else ""))
    return built


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Recorded episodes -> LeRobot v3.0 dataset")
    ap.add_argument("episodes_root")
    ap.add_argument("out_dir")
    ap.add_argument("--repo-id", default="local/remote_hands", help="name to load it under; only printed, the "
                    "format has no place for it")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--video-size", type=parse_size, default=None, metavar="WxH")
    a = ap.parse_args()
    export(a.episodes_root, a.out_dir, a.fps, a.video_size, a.repo_id)
