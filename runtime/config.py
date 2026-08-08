"""
All runtime tunables in one place, plus re-exports of the offline
pipeline's constants (imported, never redefined -- see the plan's
Appendix A). Anything you would want to change while tuning the robot
on real users lives here.
"""

import numpy as np

from runtime._paths import MOTION_GEN_DIR, RETARGET_DIR, ROOT  # noqa: F401 (sys.path shim)

# ---- offline-pipeline constants (single source of truth) ------------------
from config import (DT, HOME4, HOME_PITCH, LIGHT_FLOOR,  # noqa: E402
                    LIGHT_SLEW, LIMIT_MARGIN, RATE_CAP)
from labels import EMOTIONS  # noqa: E402  11 labels as of dataset v1.5

# JOINT_LO/HI, T_MAX live in motion_generator/dataset.py (the
# deliberately dependency-light duplicate), which imports torch; pulling
# torch in just for five floats is wrong on the Pi's control path, so
# these are the only constants duplicated here. The test suite asserts
# they match dataset.py.
JOINT_LO = np.array([-5.021, -1.0843, -2.82, -3.7865, -0.8533])
JOINT_HI = np.array([1.2622, 2.0573, 0.3216, 2.4966, 2.2883])
T_MAX = 240
N_CHANNELS = 9
T_MIN = 8                      # sample.py's floor on generated length

FPS = 30
LIGHT_CH, RGB_CH = 5, slice(6, 9)

# ---- checkpoint / generation service --------------------------------------
DEFAULT_CKPT = MOTION_GEN_DIR / "runs" / "fm-v1" / "ckpt_best.pt"
# The motion generation service (runtime/motion/service.py) runs on the
# GPU box; the runtime is its client. MOTION_SERVICE_URL env overrides.
MOTION_SERVICE_URL = "http://127.0.0.1:8031"
MOTION_SERVICE_TIMEOUT_S = 1.0   # socket timeout, strictly < TIMEOUT_MOTION
MOTION_BREAKER_FAILS = 3         # consecutive failures before opening
MOTION_BREAKER_RESET_S = 5.0     # open duration (backs off to 30 s)
ENGINE_STEPS = 10                # Euler ODE steps per generated clip

# ---- endpointing ----------------------------------------------------------
T_END_MS = 700          # silence that ends a user turn
MIN_SPEECH_MS = 300     # minimum accumulated speech before an endpoint
AUDIO_BLOCK_MS = 10     # VAD granularity (160 samples @ 16 kHz)
AUDIO_SR = 16000
RING_SECONDS = 30       # preallocated mic ring buffer
HALF_DUPLEX_TAIL_MS = 150   # keep mic muted this long after playback ends
VAD_ENERGY_THRESHOLD = 0.01  # RMS (of full-scale 1.0) for the energy VAD

# ---- dialogue LLM (Gemini) ------------------------------------------------
# The plan's latency budget (section 7) calls for a small model with a
# short system prompt and capped max_tokens.
GEMINI_MODEL = "gemini-3.5-flash-lite"  # fastest tier; thinking_level=low
LLM_MAX_TOKENS = 500
MAX_HISTORY_TURNS = 12  # user+assistant message pairs kept per session

# ---- TTS ------------------------------------------------------------------
TTS_WPM = 170           # SilentTts duration estimate (words per minute)
TTS_SR = 22050          # sample rate local backends synthesize at

# ---- continuity blending (section 6.1) ------------------------------------
BLEND_ALPHA = 0.40      # fraction of the rate budget reserved for blending
YAW_RELATIVE = True     # L1: play J1 as deltas on the current heading
TAIL_BLEND_MS = 300     # L2: drift toward the next clip's start pose
PREFIX_FRAMES = 5       # L4: commanded-frame ring buffer length
CORRECTION_MOTION_MAX = 0.15   # P1 gate threshold

# ---- affect -> cfg (section 4.6) ------------------------------------------
CFG_MIN, CFG_MAX = 1.0, 3.5
MAX_AFFECT_KEYS = 3     # per segment; more averages into mush
MAX_SEGMENTS = 8
FALLBACK_AFFECT = "interest"

# ---- speech-motion alignment (section 6.2) --------------------------------
STRETCH_MAX = 0.10      # uniform-resample clips within +/-10% mismatch

# ---- envelope modulation (section 6.3) ------------------------------------
K_NOD = 0.04            # rad, envelope -> J5 gain (~2.3 deg)
K_LED = 0.05            # envelope -> brightness gain
ENV_LP_HZ = 4.0         # envelope low-pass (syllable rate)
ENV_CLIP = 2.0          # normalized envelope clipped to [-ENV_CLIP, ENV_CLIP]

# ---- reaction clips (section 6.4) -----------------------------------------
REACT_MIN_S, REACT_MAX_S = 0.6, 1.2

# ---- timeouts (section 5), seconds ----------------------------------------
TIMEOUT_ASR = 3.0
TIMEOUT_LLM = 4.0
TIMEOUT_TTS = 3.0
TIMEOUT_MOTION = 1.5

# ---- ambient motion (between speech, instead of static idle) --------------
AMBIENT_AFFECTS = ("interest", "boredom", "understanding")
AMBIENT_SECONDS = 2.5        # filler clip duration
AMBIENT_POOL_SIZE = 2        # prefetched ambient clips kept ready
AMBIENT_STALE_S = 60.0       # drop pooled clips older than this on pop

# ---- idle motion ----------------------------------------------------------
BREATH_PERIOD_S = 4.0
BREATH_LIGHT_AMP = 0.06      # light01 swing around the resting glow
BREATH_JOINT_AMP = 0.015     # rad, J2/J3/J5 sway amplitude
IDLE_POSE = np.array([0.0, 0.55, -1.05, HOME4, HOME_PITCH + 0.55])
IDLE_LIGHT = 0.45
IDLE_RGB = np.array([1.0, 0.85, 0.6])   # warm white

# ---- servo bus ------------------------------------------------------------
SERVO_IDS = [1, 2, 3, 4, 5]        # Feetech bus ids, J1..J5
SERVO_TICKS_PER_REV = 4096         # STS3215
SERVO_CENTER_TICKS = 2048
SERVO_SIGNS = np.array([1, 1, 1, 1, 1])   # calibrate at P0 on hardware
SERVO_OFFSETS_TICKS = np.zeros(5)         # calibrate at P0 on hardware

N_LEDS = 24
