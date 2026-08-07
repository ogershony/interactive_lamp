"""
Lamp MuJoCo model: kinematics, base re-anchoring, load-time calibration,
and the offscreen renderer.
"""

import math

import numpy as np

import config  # sets MUJOCO_GL before mujoco import  # noqa: F401
import mujoco

from config import BASE_BODY, HEAD_BODY, HOME4, JOINTS, SCENE_XML


class Lamp:
    """
    scene.xml is rooted mid-chain at the lower arm with a freejoint
    (onshape-to-robot artifact); the base hangs *down* the tree. For
    kinematic replay we re-anchor: after setting joint qpos, the freejoint
    is set to the rigid transform that puts the base body back at its
    fixed pose on the floor.
    """

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.base_pos = self.data.body(BASE_BODY).xpos.copy()
        self.base_quat = self.data.body(BASE_BODY).xquat.copy()
        self.jq = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.jid = [self.model.joint(n).id for n in JOINTS]
        self.lo = np.array([self.model.joint(n).range[0] for n in JOINTS])
        self.hi = np.array([self.model.joint(n).range[1] for n in JOINTS])
        self._calibrate()

    def set_pose(self, q):
        d, m = self.data, self.model
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[self.jq] = q
        mujoco.mj_forward(m, d)
        bp, bq = d.body(BASE_BODY).xpos.copy(), d.body(BASE_BODY).xquat.copy()
        neg, cq, cp = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(neg, bq)
        mujoco.mju_mulQuat(cq, self.base_quat, neg)
        mujoco.mju_rotVecQuat(cp, bp, cq)
        d.qpos[0:3] = self.base_pos - cp
        d.qpos[3:7] = cq
        mujoco.mj_forward(m, d)

    def root_pose_for(self, q):
        """Freejoint (pos, quat) consistent with base-on-floor at joints q."""
        self.set_pose(q)
        return self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy()

    def gaze_elev(self):
        """Elevation (rad, + up) of the lamphead gaze axis (local +y)."""
        gz = self.data.body(HEAD_BODY).xmat.reshape(3, 3)[2, 1]
        return math.asin(max(-1.0, min(1.0, gz)))

    def head_z(self):
        return float(self.data.body(HEAD_BODY).xpos[2])

    def fk(self, q):
        self.set_pose(q)
        return self.gaze_elev(), self.head_z()

    def _calibrate(self):
        """Verify assumed joint semantics; fit the nod compensation."""
        self.set_pose(np.zeros(5))
        assert abs(self.data.xaxis[self.jid[0]][2]) > 0.99, \
            "J1 is not a vertical yaw axis"

        # GAZE_LEVEL: q5 where gaze is level, at home posture, bisected
        def elev_at_q5(q5):
            self.set_pose(np.array([0, 0, 0, HOME4, q5]))
            return self.gaze_elev()
        lo5, hi5 = 1.0, 2.2
        for _ in range(48):
            mid = 0.5 * (lo5 + hi5)
            if elev_at_q5(mid) < 0:
                lo5 = mid
            else:
                hi5 = mid
        self.gaze_level_q5 = 0.5 * (lo5 + hi5)
        self.set_pose(np.array([0, 0, 0, HOME4, self.gaze_level_q5]))
        assert abs(self.data.xaxis[self.jid[4]][2]) < 0.03, \
            "J5 nod axis not level at HOME4"

        # nod compensation: gaze pitches by a2*dq2 + a3*dq3 (exact: J2/J3
        # axes are antiparallel, so the coupling is planar/linear)
        q0 = np.array([0, 0, 0, HOME4, self.gaze_level_q5])
        e0, _ = self.fk(q0)
        eps = 1e-4
        a = []
        for j in (1, 2):
            qp = q0.copy()
            qp[j] += eps
            a.append((self.fk(qp)[0] - e0) / eps)
        self.pitch_coef = np.array(a)          # d elev / d (q2, q3)
        assert all(0.9 < abs(c) < 1.1 for c in a), f"pitch coef {a}"
        qbig = q0.copy()
        qbig[1] += 0.4
        qbig[2] += 0.2
        lin = e0 + a[0] * 0.4 + a[1] * 0.2
        assert abs(self.fk(qbig)[0] - lin) < 0.02, "nod coupling not linear"


class LampRenderer:
    """Offscreen lamp render; lamphead tint follows the light01 channel."""

    W, H = 420, 300

    def __init__(self, lamp):
        self.lamp = lamp
        self.renderer = mujoco.Renderer(lamp.model, self.H, self.W)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 0.85
        self.cam.azimuth = 215
        self.cam.elevation = -14
        self.cam.lookat[:] = (-0.02, 0.0, 0.20)
        b = lamp.model.body(HEAD_BODY)
        adr, num = int(b.geomadr[0]), int(b.geomnum[0])
        self.head_geoms = list(range(adr, adr + num))
        self.base_rgba = lamp.model.geom_rgba[self.head_geoms].copy()

    def frame(self, q, light01):
        m = self.lamp.model
        warm = np.array([1.0, 0.85, 0.45, 1.0])
        for g in self.head_geoms:
            m.geom_rgba[g] = (1 - light01) * self.base_rgba[0] + light01 * warm
        self.lamp.set_pose(q)
        self.renderer.update_scene(self.lamp.data, camera=self.cam)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
