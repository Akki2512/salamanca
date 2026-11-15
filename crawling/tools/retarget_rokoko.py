"""Retarget Rokoko BVH takes to Unitree G1 qpos at 50 Hz with GMR."""
import argparse
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

import rokoko_bvh

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "GMR"))
from general_motion_retargeting import IK_CONFIG_DICT  # noqa: E402
from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402

IK_CONFIG_DICT["bvh_rokoko"] = {"unitree_g1": HERE / "bvh_rokoko_to_g1.json"}
BODIES = ["Hips", "Spine3"] + [s + b for s in ("Left", "Right") for b in ("Thigh", "Shin", "Foot", "Arm", "ForeArm", "Hand")]

p = argparse.ArgumentParser()
p.add_argument("bvh", nargs="+")
p.add_argument("--out", required=True)
p.add_argument("--height", type=float, default=1.72, help="performer height in metres")
p.add_argument("--fps", type=float, default=50)
a = p.parse_args()

out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for path in a.bvh:
    names, parents, fps, pos, rot = rokoko_bvh.read(path)
    ix = [names.index(b) for b in BODIES]
    quat = R.from_matrix(rot[:, ix].reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(len(pos), len(ix), 4)
    retargeter = GMR(src_human="bvh_rokoko", tgt_robot="unitree_g1", actual_human_height=a.height, verbose=False)
    # Frame 0 is the suit's T-pose; resample the rest to the benchmark rate.
    src = np.round(np.arange(1, len(pos) - 1, fps / a.fps)).astype(int)
    qpos = []
    for f in src:
        frame = {b: [pos[f, j], quat[f, k]] for k, (b, j) in enumerate(zip(BODIES, ix))}
        qpos.append(retargeter.retarget(frame).copy())
    qpos = np.asarray(qpos, dtype=np.float32)
    t_frame = {b: [pos[0, j], quat[0, k]] for k, (b, j) in enumerate(zip(BODIES, ix))}
    for _ in range(20):
        tpose = retargeter.retarget(t_frame).copy()
    np.savez(out / (pathlib.Path(path).stem + ".npz"), qpos=qpos, tpose=tpose.astype(np.float32))
    print(path, qpos.shape, "pelvis z min/median/max", qpos[:, 2].min().round(2), np.median(qpos[:, 2]).round(2), qpos[:, 2].max().round(2), flush=True)
