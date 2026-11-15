"""Have Nemotron look at each robot motion clip and label its action and visible defects.

Reads NEMOTRON_BASE_URL, NEMOTRON_API_KEY and NEMOTRON_MODEL (OpenAI-compatible endpoint).
With --dry_run only the contact sheets are written and no request is made.
"""
import argparse
import json
import pathlib
import re
import time

import mujoco
import numpy as np
from PIL import Image

from nemotron import ask, strict_schema

ACTIONS = ["roll-over", "sit-lie", "lie-sit", "stand-sit", "burpee", "crawl", "other"]
PROMPT = (
    "The image is a contact sheet of 8 frames from one humanoid robot motion clip, in time order, "
    "left to right then top to bottom. Label the action. sit-lie: starts sitting on the floor and "
    "ends lying. lie-sit: starts lying and ends sitting. stand-sit: starts standing and ends sitting "
    "on the floor. roll-over: stays lying and turns between back and belly. burpee: stands, drops to "
    "a plank and returns. crawl: moves on hands and knees. Also report whether the robot floats "
    "above the floor or sinks into it in any frame."
)
SCHEMA = strict_schema("clip_label", {
    "action": {"type": "string", "enum": ACTIONS},
    "floats": {"type": "boolean"},
    "sinks": {"type": "boolean"},
    "reason": {"type": "string"},
})


def contact_sheet(model, data, renderer, cam, qpos, n=8):
    tiles = []
    for q in qpos[np.linspace(0, len(qpos) - 1, n).astype(int)]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [q[0], q[1], 0.45]
        renderer.update_scene(data, cam)
        tiles.append(renderer.render().copy())
    half = n // 2
    return Image.fromarray(np.vstack([np.hstack(tiles[:half]), np.hstack(tiles[half:])]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--motions", required=True, help="folder of .npz clips with a qpos array")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--pace_s", type=float, default=1.6, help="delay between requests")
    p.add_argument("--expected", help="true action for every clip; default reads it from the file name")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    model = mujoco.MjModel.from_xml_path(a.scene)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 240, 320)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 120

    out = pathlib.Path(a.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    clips = sorted(pathlib.Path(a.motions).rglob("*.npz"))
    if a.limit:
        clips = clips[:: max(1, len(clips) // a.limit)][: a.limit]

    rows = []
    for clip in clips:
        sheet = contact_sheet(model, data, renderer, cam, np.load(clip)["qpos"])
        sheet.save(out / "sheets" / f"{clip.stem}.jpg", quality=85)
        if a.dry_run:
            continue
        # File names carry the benchmark's own action label, e.g. roll-over_12.
        row = {"clip": clip.stem, "expected": a.expected or re.sub(r"[_-]?\d+$", "", clip.stem)}
        try:
            row.update(ask(PROMPT, SCHEMA, sheet))
        except Exception as e:
            row["error"] = repr(e)
        rows.append(row)
        print(json.dumps(row), flush=True)
        time.sleep(a.pace_s)

    if rows:
        (out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        answered = [r for r in rows if "action" in r]
        right = sum(r["action"] == r["expected"] for r in answered)
        print(f"answered {len(answered)}/{len(rows)}, action correct {right}/{len(answered)}")


if __name__ == "__main__":
    main()
