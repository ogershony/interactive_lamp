"""
Live MuJoCo viewer: watch the lamp move while the session runs.

A scheduler sink like the servo bus and the LED ring -- it receives the
same governed frames those do, so what you see on screen is the commanded
stream, not the model's raw output. `runtime/eval/replay.py` renders the
same stream offline into an mp4 afterwards; this is the realtime window
onto it.

Held to the same rule as the drivers: never let the display stall the
30 Hz control loop. `sync()` is called at most VIEW_FPS times a second
(the physical lamp does not care about frames the human eye cannot
resolve), and a viewer that has been closed, or that throws for any
reason, disables itself rather than propagating into the tick.

Needs a windowed GL context, so it runs on a desktop, not the Pi. Under
WSL2 that is WSLg's Wayland/X11 display; the offscreen EGL context the
offline pipeline forces via MUJOCO_GL cannot open a window, so
`MUJOCO_GL=egl` is cleared for the viewer's own import.
"""

import os
import time

import numpy as np

VIEW_FPS = 30.0


class LiveViewer:
    """Duck-types a driver sink: .write(cmd9) then .close()."""

    def __init__(self, view_fps=VIEW_FPS):
        self._min_dt = 1.0 / view_fps if view_fps else 0.0
        self._last = 0.0
        self._alive = True
        self.lamp = None
        self.viewer = None

    def _open(self):
        """GL init is deferred to the first frame so the window is created
        on whichever thread drives the ticks -- in --converse that is the
        scheduler's realtime thread, not main -- since GLFW wants window
        creation and its event polling on one and the same thread."""
        # the viewer needs a window; the pipeline's default EGL context
        # is offscreen-only and would fail here
        if os.environ.get("MUJOCO_GL", "").lower() in ("egl", "osmesa"):
            os.environ["MUJOCO_GL"] = "glfw"

        import mujoco.viewer

        from pipeline import Lamp        # via runtime._paths shim

        self.lamp = Lamp()
        self._head_geoms, self._base_rgba = self._head_material()
        self.viewer = mujoco.viewer.launch_passive(
            self.lamp.model, self.lamp.data,
            show_left_ui=False, show_right_ui=False)
        from config import (CAM_AZIMUTH, CAM_DISTANCE,  # via _paths shim
                            CAM_ELEVATION, CAM_LOOKAT)
        cam = self.viewer.cam                # one framing for every lamp view
        cam.distance, cam.azimuth = CAM_DISTANCE, CAM_AZIMUTH
        cam.elevation = CAM_ELEVATION
        cam.lookat[:] = CAM_LOOKAT

    def _head_material(self):
        """Lamphead geoms + their base colors, for the LED tint."""
        from config import HEAD_BODY       # via runtime._paths shim

        b = self.lamp.model.body(HEAD_BODY)
        adr, num = int(b.geomadr[0]), int(b.geomnum[0])
        geoms = list(range(adr, adr + num))
        return geoms, self.lamp.model.geom_rgba[geoms].copy()

    def write(self, cmd):
        """`cmd` is a (9,) governed frame: 5 joints, light01, rgb."""
        if not self._alive:
            return
        now = time.monotonic()
        if now - self._last < self._min_dt:
            return                       # drop: the eye cannot resolve it
        self._last = now
        cmd = np.asarray(cmd, np.float64)
        try:
            if self.viewer is None:
                self._open()
            if not self.viewer.is_running():
                self.close()             # window closed by hand
                return
            light = float(cmd[5])
            warm = np.array([1.0, 0.85, 0.45, 1.0])
            for i, g in enumerate(self._head_geoms):
                self.lamp.model.geom_rgba[g] = \
                    (1 - light) * self._base_rgba[i] + light * warm
            self.lamp.set_pose(cmd[:5])
            self.viewer.sync()
        except Exception:                # noqa: BLE001
            self.close()                 # a dead window must not stop the lamp

    def close(self):
        self._alive = False
        if self.viewer is None:
            return
        try:
            self.viewer.close()
        except Exception:                # noqa: BLE001
            pass
