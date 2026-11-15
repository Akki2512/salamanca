"""Lowest point of the robot above the floor, from the bounding boxes of its geoms."""
import mujoco
import numpy as np

CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


class Floor:
    def __init__(self, scene):
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] != 0]

    def lowest(self, qpos):
        """Height of the lowest robot point for one pose."""
        m, d = self.model, self.data
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        low = np.inf
        for g in self.geoms:
            centre, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            world = d.geom_xpos[g] + (centre + CORNERS * half) @ d.geom_xmat[g].reshape(3, 3).T
            low = min(low, world[:, 2].min())
        return low
