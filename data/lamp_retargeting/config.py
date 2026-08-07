"""
Paths and mapping constants for the Cozmo -> lamp retargeting pipeline.

Single source of truth: every other module imports paths and constants
from here (motion_prior/dataset.py deliberately duplicates a handful to
stay dependency-light on the GPU box -- see its header).

The mapping constants below were calibrated via rendered pose probes;
`mapping.mapping_constants()` bundles them into every retargeted clip
and `mapping.constants_sha1()` hashes them so metric diffs can catch
edits made without a MAPPING_VERSION bump.
"""

import math
import os
import pathlib
import sys

import numpy as np

# mujoco is imported by lamp_model/verify; the GL backend must be chosen
# before that import happens, and every module imports config first
os.environ.setdefault("MUJOCO_GL", "egl")

RETARGET = pathlib.Path(__file__).resolve().parent   # data/lamp_retargeting
ROOT = RETARGET.parents[1]                           # repo root
COZMO = ROOT / "data" / "cozmo_data"
NPZ_IN = COZMO / "animations" / "npz"
LABELS_CSV = COZMO / "labels.csv"
SCENE_XML = ROOT / "assets" / "scene.xml"
NPZ_ROOT = RETARGET / "npz"      # one subdirectory per retarget run
SCRATCH_RUN = "scratch"          # single-clip debug output; never a run
PREVIEW = RETARGET / "preview"
METRICS_DIR = RETARGET / "metrics"
CURATION_CSV = RETARGET / "curation.csv"
DATASET_DIR = ROOT / "data" / "dataset"   # diffusion-ready exports

sys.path.insert(0, str(COZMO / "scripts"))   # cozmo_model (GIF renderer)

MAPPING_VERSION = "1.4"
DT = 0.033
JOINTS = ["1", "2", "3", "4", "5"]
BASE_BODY = "scs215_v5"          # contains lamp_base; welded/re-anchored
HEAD_BODY = "diffuser"

# ---- mapping constants (calibrated via rendered pose probes) -------------
HOME4 = -0.643        # wrist roll that levels the J5 nod axis
HOME_PITCH = -0.35    # home gaze elevation (rad): classic desk-lamp droop
CROUCH23 = np.array([1.10, 0.27])    # (dq2,dq3) at crouch = -1 (lift 0mm)
TALL23 = np.array([-0.50, -0.95])    # (dq2,dq3) at crouch = +1 (lift 92mm)
LEAN23 = np.array([0.55, -0.25])     # (dq2,dq3) per unit forward lean
COMP_GAIN = 0.80      # fraction of posture pitch the nod compensates:
                      # full (1.0) would exhaust J5 range when crouched;
                      # partial keeps range and lets a crouch read "down"
LIFT_PARK = 32.0      # mm; physical lift minimum (commands below are
                      # firmware-clamped: 0 mm in the data == parked)
LIFT_MAX = 92.0
K_SLUMP = 0.8         # how strongly head-down pulls the posture into a
                      # crouch (Cozmo performs a slump as head-down; the
                      # lift can't go below park, so the lamp's fold-down
                      # range is driven by the head channel)
HEAD_DOWN_REF = math.radians(25.0)   # Cozmo head at -25 deg = full slump
YAW_SOFT = 1.00       # rad; soft clamp scale for base yaw
K_GX, GX_CLAMP = 0.011, 0.20         # gaze_x px -> rad, clamp
K_GY, GY_CLAMP = 0.008, 0.25         # gaze_y px -> rad, clamp
ROLL_CLAMP = 0.55     # rad; face-angle -> wrist roll clamp
K_TILT = 0.15         # rad of head tilt (J4) per rad/s of base yaw:
                      # the head banks into turns (secondary motion) --
                      # J4 was pinned at HOME4 in 100% of v1.3 frames
                      # because Cozmo's face-roll channel barely fires
TILT_CLAMP = 0.30     # rad; max banking tilt (~17 deg)
TILT_LP_HZ = 1.5      # low-pass on yaw velocity before tilt: the head
                      # leans in and out of a turn smoothly
LEAN_VREF = 120.0     # mm/s; lean = tanh(v / LEAN_VREF)
LEAN_LP_HZ = 1.5      # low-pass on body velocity before lean
FILT_HZ = 2.5         # Butterworth cutoff on joint targets: gestures
                      # shorter than ~0.4 s melt into the surrounding
                      # motion -- deliberate, graceful, Pixar-lamp calm
                      # (1.3; was 6 in 1.0, 4 in 1.2)
RATE_CAP = 1.8        # rad/s max joint speed (~103 deg/s): a 90 deg
                      # head turn takes ~1 s. Calm > punchy (1.3; was
                      # 4.0 -> 2.5 in earlier passes)
ACCEL_CAP = 15.0      # rad/s^2 onset accel: eases to RATE_CAP over
                      # ~120 ms (4 frames); scaled down with the speed
                      # cap so onsets stay proportionally gentle
BRAKE_MULT = 2.0      # stops/reversals brake at BRAKE_MULT*ACCEL_CAP:
                      # keeps the no-overshoot braking envelope's chase
                      # lag small (v^2/2A = 3.0 deg at RATE_CAP) so fast
                      # head pops ("huh!") stay crisp -- symmetric soft
                      # braking cost r_head 0.84 -> 0.12 on the worst
                      # cap-saturated clip when tried in 1.1
LIMIT_MARGIN = math.radians(2.0)
LIGHT_CLOSE_W = 11    # frames (363 ms); closing removes blink- and
                      # flutter-length light dips, keeps sustained
                      # closures as dimming
LIGHT_LP_HZ = 1.0     # low-pass after closing: fades become smooth
                      # S-curves instead of linear slew ramps
LIGHT_FLOOR = 0.15    # lamp never fully off: sustained eye closure reads
                      # as a sleeping glow, not a dead lamp
LIGHT_SLEW = 0.8      # max |d light01|/s: full-range fade takes ~1.1 s,
                      # a calm breath rather than a blink

DYN_SAMPLE = 50       # clips dynamics-checked by default in --all
GIF_EMOTIONS = ["joy", "sorrow", "anger", "surprise", "fear", "boredom"]
