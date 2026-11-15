"""Read a Rokoko Studio BVH into world-frame joint positions and orientations.

Output frame is robot convention: x forward, y left, z up, metres. Rokoko writes its T-pose as
frame 0, and joint orientations are returned relative to that frame so they are identity in the
T-pose.
"""
import re

import numpy as np
from scipy.spatial.transform import Rotation as R

# BVH is y-up with the character facing +z and its left along +x.
BVH_TO_ROBOT = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def read(path):
    """Return names, parents, fps, pos (T, J, 3) in metres and rot (T, J, 3, 3)."""
    names, parents, offsets, orders, stack = [], [], [], [], []
    with open(path) as f:
        lines = iter(f)
        for line in lines:
            tok = line.split()
            if not tok:
                continue
            if tok[0] in ("ROOT", "JOINT"):
                names.append(tok[1])
                parents.append(stack[-1] if stack else -1)
                stack.append(len(names) - 1)
            elif tok[0] == "End":
                stack.append(None)
            elif tok[0] == "OFFSET" and stack[-1] is not None:
                offsets.append([float(v) for v in tok[1:4]])
            elif tok[0] == "CHANNELS":
                orders.append("".join(c[0] for c in tok[2:] if c.endswith("rotation")))
            elif tok[0] == "}":
                stack.pop()
            elif tok[0] == "Frame" and tok[1] == "Time:":
                fps = 1.0 / float(tok[2])
                break
        data = np.loadtxt(lines)
    offsets = np.array(offsets) / 100.0
    T, J = len(data), len(names)
    assert data.shape[1] == 3 + 3 * J, (data.shape, J)
    root_pos = data[:, :3] / 100.0
    euler = data[:, 3:].reshape(T, J, 3)
    C = BVH_TO_ROBOT
    pos = np.zeros((T, J, 3))
    rot = np.zeros((T, J, 3, 3))
    for j in range(J):
        local = R.from_euler(orders[j].upper(), euler[:, j], degrees=True).as_matrix()
        if parents[j] < 0:
            rot[:, j], pos[:, j] = local, root_pos
        else:
            p = parents[j]
            rot[:, j] = rot[:, p] @ local
            pos[:, j] = pos[:, p] + rot[:, p] @ offsets[j]
    rot = C @ rot @ C.T
    return names, parents, fps, pos @ C.T, rot @ np.swapaxes(rot[:1], -1, -2)
