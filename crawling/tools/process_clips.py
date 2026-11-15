"""Turn retargeted suit takes into crawl training clips.

Fixes root height against the floor, keeps the hands-and-knees parts, cuts them into
benchmark-length windows and adds left-right mirrored copies.
"""
import argparse
import json
import pathlib

import numpy as np
from scipy.signal import butter, filtfilt

from floor import Floor

# Joint order after the 7 root values: left leg 0-5, right leg 6-11, waist 12-14,
# left arm 15-21, right arm 22-28. Roll and yaw joints flip sign under a left-right mirror.
SWAP = list(range(6, 12)) + list(range(0, 6)) + [12, 13, 14] + list(range(22, 29)) + list(range(15, 22))
FLIP = np.ones(29)
FLIP[[1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28]] = -1
# Lowest bounding-box point of the benchmark's own crawl clips sits this far below the floor.
FLOOR_TARGET = -0.05


def mirror(qpos):
    out = qpos.copy()
    out[:, 1] *= -1
    out[:, [4, 6]] *= -1
    out[:, 7:] = qpos[:, 7:][:, SWAP] * FLIP
    return out


def forward_pitch_deg(qpos):
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    return np.degrees(np.arcsin(np.clip(-2 * (x * z - w * y), -1, 1)))


def speed_up(qpos, speed):
    """Play a clip `speed` times faster by resampling it in time."""
    t = np.arange(0, len(qpos) - 1, speed)
    i, f = t.astype(int), (t % 1)[:, None]
    out = qpos[i] * (1 - f) + qpos[i + 1] * f
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def runs(mask, min_len):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("takes", nargs="+")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--window_s", type=float, default=12)
    p.add_argument("--stride_s", type=float, default=4)
    p.add_argument("--keep_height", action="store_true", help="skip the floor fix, for data that is already grounded")
    p.add_argument("--speeds", type=float, nargs="+", default=[1.0], help="playback speed-ups to add")
    a = p.parse_args()

    floor = Floor(a.scene)
    out = pathlib.Path(a.out)
    (out / "motions").mkdir(parents=True, exist_ok=True)
    b, c = butter(2, 1.0 / (a.fps / 2))
    entries = []
    for take in a.takes:
        name = pathlib.Path(take).stem
        qpos = np.load(take)["qpos"].astype(np.float64)
        low = np.full(len(qpos), FLOOR_TARGET) if a.keep_height else np.array([floor.lowest(q) for q in qpos])
        qpos[:, 2] += filtfilt(b, c, FLOOR_TARGET - low)
        # Keep the quaternion sign continuous so clips can be interpolated in time.
        flips = np.sign(np.sum(qpos[1:, 3:7] * qpos[:-1, 3:7], axis=1))
        qpos[1:, 3:7] *= np.cumprod(np.where(flips == 0, 1, flips))[:, None]
        crawling = (qpos[:, 2] > 0.25) & (qpos[:, 2] < 0.6) & (forward_pitch_deg(qpos) > 35)
        n = 0
        for speed in a.speeds:
            win, stride = int(a.window_s * a.fps * speed), int(a.stride_s * a.fps * speed)
            for start, end in runs(crawling, 6 * a.fps):
                for s0 in range(start, max(start + 1, end - win + 1), stride):
                    clip = speed_up(qpos[s0:min(s0 + win, end)], speed)
                    clip[:, :2] -= clip[0, :2]
                    for tag, data in (("", clip), ("_mirror", mirror(clip))):
                        rel = f"{name}_crawl_x{speed:.1f}_{n:02d}{tag}.npz"
                        np.savez(out / "motions" / rel, qpos=data.astype(np.float32))
                        entries.append({"path": rel, "category": "Ground", "frames": len(data)})
                    n += 1
        print(f"{name}: {crawling.mean():.0%} of frames crawling, {n} windows, height shift median {np.median(FLOOR_TARGET - low):+.2f} m")
    (out / "clips.json").write_text(json.dumps(entries, indent=1))
    print(f"{len(entries)} clips, {sum(e['frames'] for e in entries) / a.fps / 60:.1f} minutes including mirrors and overlap")


if __name__ == "__main__":
    main()
