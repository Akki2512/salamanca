"""Convert G1 qpos clips (npz, 50 Hz) into SONIC's motion_lib PKL files, one per clip.

Run with SONIC's training environment, from the GR00T-WholeBodyControl repo root.
"""
import argparse
import pathlib

import joblib
import numpy as np

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import convert_sequence

p = argparse.ArgumentParser()
p.add_argument("--src", nargs="+", required=True, help="folders of .npz clips with a qpos array")
p.add_argument("--out", required=True)
p.add_argument("--fps", type=int, default=50)
a = p.parse_args()

n = 0
for src in a.src:
    src = pathlib.Path(src)
    dst = pathlib.Path(a.out) / src.parent.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.npz")):
        qpos = np.load(f)["qpos"].astype(np.float32)
        # qpos is root xyz, root quaternion wxyz, then 29 joints in MuJoCo order.
        entry = convert_sequence({
            "joint_pos": qpos[:, 7:],
            "body_pos_w": qpos[:, None, :3],
            "body_quat_w": qpos[:, None, 3:7],
            "joint_order": "mj",
        }, a.fps)
        name = f.stem.replace(".", "p")
        joblib.dump({name: entry}, dst / f"{name}.pkl")
        n += 1
print(n, "clips written to", a.out)
