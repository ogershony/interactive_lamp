"""
Minimal Cozmo replica for MuJoCo, replaying NPZ channel arrays.

Kinematic replay only: qpos is set directly from the arrays each frame and
mj_forward computes poses — no dynamics, no actuators. The same pattern
(qpos <- channel arrays -> render) is what lamp retargeting will use.

Dimensions follow pycozmo geometry where it matters (head pivot, lift pivot
45 mm up / 66 mm arm) and eyeballed Cozmo proportions elsewhere
(chassis ~74x56x36 mm). Everything is in meters (mm * 1e-3).
"""

import math
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

MM = 1e-3

# qpos layout (all kinematic):
#   0 base_x (m), 1 base_y (m), 2 base_yaw (rad),
#   3 head_pitch (rad, +up),   4 lift_angle (rad, +up)

COZMO_XML = f"""
<mujoco model="cozmo_replica">
  <compiler angle="radian"/>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/>
    <global offwidth="840" offheight="600"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.83 0.79 0.72" rgb2="0.74 0.69 0.61"/>
    <material name="floor" texture="floor" texrepeat="12 12" reflectance="0.05"/>
    <material name="shell"  rgba="0.92 0.92 0.94 1"/>
    <material name="dark"   rgba="0.22 0.22 0.24 1"/>
    <material name="track"  rgba="0.13 0.13 0.14 1"/>
    <material name="red"    rgba="0.78 0.12 0.10 1"/>
    <material name="screen" rgba="0.02 0.02 0.03 1"/>
    <material name="arm"    rgba="0.85 0.86 0.88 1"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.4 0.4 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="-0.4 0.2 0.6" dir="0.5 -0.3 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>

    <body name="base" pos="0 0 0">
      <joint name="base_x"   type="slide" axis="1 0 0"/>
      <joint name="base_y"   type="slide" axis="0 1 0"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1"/>

      <!-- tracks -->
      <geom type="box" size="{37*MM} {8*MM} {12*MM}" pos="0 {-24*MM} {12*MM}" material="track"/>
      <geom type="box" size="{37*MM} {8*MM} {12*MM}" pos="0 {24*MM} {12*MM}" material="track"/>
      <!-- chassis -->
      <geom type="box" size="{34*MM} {17*MM} {14*MM}" pos="{-2*MM} 0 {24*MM}" material="shell"/>
      <!-- backpack strip -->
      <geom type="box" size="{16*MM} {12*MM} {5*MM}" pos="{-14*MM} 0 {41*MM}" material="red"/>

      <!-- head on pitch hinge at the head pivot -->
      <body name="head" pos="{18*MM} 0 {38*MM}">
        <joint name="head_pitch" type="hinge" axis="0 -1 0"
               range="{math.radians(-30)} {math.radians(50)}"/>
        <geom type="box" size="{15*MM} {21*MM} {15*MM}" pos="{6*MM} 0 {4*MM}" material="dark"/>
        <!-- screen plate on the front face -->
        <geom type="box" size="{1.2*MM} {16*MM} {10*MM}" pos="{21.5*MM} 0 {4*MM}" material="screen"/>
        <!-- eyes: two cyan quads, roughly where the screen eyes sit -->
        <geom name="eye_l" type="box" size="{0.8*MM} {5*MM} {5*MM}"
              pos="{22.6*MM} {7*MM} {5*MM}" rgba="0.1 0.95 0.95 1"/>
        <geom name="eye_r" type="box" size="{0.8*MM} {5*MM} {5*MM}"
              pos="{22.6*MM} {-7*MM} {5*MM}" rgba="0.1 0.95 0.95 1"/>
      </body>

      <!-- lift: arm on hinge at the front pivot, 66 mm to the fork -->
      <body name="lift" pos="{22*MM} 0 {45*MM}">
        <joint name="lift_angle" type="hinge" axis="0 -1 0"
               range="{math.radians(-45)} {math.radians(50)}"/>
        <geom type="box" size="{33*MM} {2.5*MM} {2.5*MM}" pos="{33*MM} {19*MM} 0" material="arm"/>
        <geom type="box" size="{33*MM} {2.5*MM} {2.5*MM}" pos="{33*MM} {-19*MM} 0" material="arm"/>
        <!-- fork plate -->
        <geom type="box" size="{2*MM} {22*MM} {13*MM}" pos="{66*MM} 0 {-4*MM}" material="arm"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# neutral pose when a channel is NaN (clip never animates it)
NEUTRAL_HEAD_DEG = 0.0
NEUTRAL_LIFT_DEG = math.degrees(math.asin((32.0 - 45.0) / 66.0))  # lift at rest


def make():
    model = mujoco.MjModel.from_xml_string(COZMO_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def replay_frame(model, data, npz, i):
    """Set qpos from NPZ arrays at frame i and recompute kinematics."""
    head = npz['head_deg'][i]
    lift = npz['lift_deg'][i]
    data.qpos[0] = npz['x_mm'][i] * MM
    data.qpos[1] = npz['y_mm'][i] * MM
    data.qpos[2] = npz['yaw_rad'][i]
    data.qpos[3] = math.radians(NEUTRAL_HEAD_DEG if np.isnan(head) else float(head))
    data.qpos[4] = math.radians(NEUTRAL_LIFT_DEG if np.isnan(lift) else float(lift))
    mujoco.mj_forward(model, data)


class ClipRenderer:
    """Offscreen renderer with a front-3/4 camera tracking the base."""

    def __init__(self, width=420, height=300):
        self.model, self.data = make()
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = self.model.body("base").id
        self.cam.distance = 0.34
        self.cam.azimuth = 160        # from the front, slightly off-axis
        self.cam.elevation = -18
        self.cam.lookat[:] = (0, 0, 0.04)

    def frame(self, npz, i):
        replay_frame(self.model, self.data, npz, i)
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()   # H x W x 3 uint8

    def close(self):
        self.renderer.close()
