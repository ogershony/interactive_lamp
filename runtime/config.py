"""
All runtime tunables in one place, plus re-exports of the offline
pipeline's constants (imported, never redefined -- see the plan's
Appendix A). Anything you would want to change while tuning the robot
on real users lives here.
"""

import numpy as np

from runtime import _dotenv  # noqa: F401 (loads .env into os.environ)
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
T_END_SHORT_MS = 1100   # ... but this much after a *short* utterance. Two
                        # words then a pause is usually someone assembling a
                        # sentence, and cutting in there is the most
                        # conversation-breaking thing the lamp does. A pause
                        # only reliably means "your turn" once the user has
                        # been talking for a while.
SHORT_SPEECH_MS = 1200  # accumulated speech below which "short" applies
ASR_SPECULATE_MS = 250  # silence after which ASR starts *early*, in parallel
                        # with the rest of the endpoint window. Measured on a
                        # live session, endpoint->speech was 2.09 s of which
                        # ASR was 0.87 s -- and the 700 ms window before it
                        # was spent doing nothing at all. Transcribing from
                        # here overlaps the two; if speech resumes the guess
                        # is discarded. Lower means more wasted transcriptions
                        # of mid-sentence pauses, not worse turns.
MIN_SPEECH_MS = 300     # minimum accumulated speech before an endpoint
AUDIO_BLOCK_MS = 10     # VAD granularity (160 samples @ 16 kHz)
AUDIO_SR = 16000
RING_SECONDS = 30       # preallocated mic ring buffer
HALF_DUPLEX_TAIL_MS = 150   # reverb tail: mic stays muted this long after
                            # the last sample is *audible*
OUTPUT_LAG_MS = 320         # speaker latency: how long after the playback
                            # queue empties the sound is still coming out.
                            # The gate used to open 150 ms after the queue
                            # emptied, which is when the audio reaches
                            # PortAudio -- not the speaker. On this box
                            # (PulseAudio over WSLg) a live session had the
                            # lamp transcribe the tail of its own sentences
                            # on 3 of 8 turns and answer its own question
                            # verbatim. 320 = the demo box's measured
                            # chirp loop (p50 254, max 265 ms) plus 20%,
                            # against PortAudio's self-reported 50 ms --
                            # that is the ALSA figure and excludes the
                            # PulseAudio buffer entirely.
                            # Note this is smaller than the session's
                            # behaviour implied, and the loop was measured
                            # on an idle box while the session had the
                            # viewer, the motion engine and Whisper
                            # competing for CPU (3 scheduler overruns).
                            # The echo guard in behavior.py is the belt to
                            # this braces, and does not depend on getting
                            # this number right.
                            # AudioIO.start() raises this to the stream's
                            # reported latency when that is larger. MEASURE
                            # IT: scripts/calibrate_audio_loop.py.
VAD_ENERGY_THRESHOLD = 0.005  # RMS (of full-scale 1.0) for the energy VAD.
                      # Calibrated on the demo box (Blue Snowball over WSLg)
                      # with scripts/calibrate_vad.py: room tone p95 0.0024,
                      # voiced speech p90 0.0124 -- 14 dB apart. Picked as
                      # the most sensitive threshold that still leaks zero
                      # room tone, because the failure mode here is
                      # under-accumulation: at 0.01 only the loudest blocks
                      # counted (550 ms of speech per sample vs 750 at
                      # 0.005), voiced runs came out short, the gaps between
                      # them ran past T_END_MS, and utterances closed below
                      # MIN_SPEECH_MS -- onsets with no endpoint, so no turn
                      # ever started. Re-measure per box; --vad-threshold
                      # overrides per run.
                      # This is the *fallback* VAD now; see VAD_DEFAULT.
VAD_DEFAULT = "silero"        # {"silero", "energy"}; audio/vad.make_vad
VAD_SILERO_THRESHOLD = 0.5    # speech probability. Box-independent, unlike
                              # the energy threshold above -- that is the
                              # whole point of using it.

# ---- dialogue LLM (Gemini) ------------------------------------------------
# The plan's latency budget (section 7) calls for a small model with a
# short system prompt and capped max_tokens.
GEMINI_MODEL = "gemini-3.5-flash-lite"  # fastest tier; thinking_level=low
LLM_MAX_TOKENS = 500
MAX_HISTORY_TURNS = 12  # user+assistant message pairs kept per session

# ---- TTS ------------------------------------------------------------------
TTS_WPM = 170           # base speaking rate (words per minute)
TTS_SR = 22050          # sample rate local backends synthesize at

# ---- voice prosody (dialogue/affect.py -> audio/tts.py) -------------------
# The lamp's body performs the affect the LLM chose; the voice used to read
# every line with identical parameters, which made the two look unrelated.
# Valence is the primary axis (pitch, word gap, amplitude); arousal drives
# rate only, so an alarmed lamp doesn't speak as slowly as a sad one.
VOICE_PITCH_SPAN = 18        # espeak -p points either side of 50
VOICE_RATE_SPAN = 0.18       # fraction of TTS_WPM either side of the base
VOICE_GAP_MAX = 6            # espeak -g units (10 ms) at full negative valence
VOICE_AMP_SPAN = 15          # espeak -a points either side of 100
VOICE_INTENSITY_FLOOR = 0.4  # a zero-intensity line still carries this much
#                              of its affect, so the voice never goes flat

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
ASR_MODEL = "base.en"   # faster-whisper size. Measured on the demo box over
                        # one utterance, median of 3: base.en 559 ms,
                        # small.en 1567 ms. In a live --converse session
                        # small.en ran 2.0-2.35 s -- the extra ~0.5 s is CPU
                        # contention with the 30 Hz loop, the motion engine
                        # and --view, so the model is only part of it.
                        # That was two thirds of a 3.3 s endpoint-to-speech
                        # budget, and wide enough that the user routinely
                        # started their next sentence inside it.
                        # The cost is real: on the same clip base.en heard
                        # "what do you think come out of the weather today"
                        # where small.en got "about" right. Being answered
                        # promptly is worth more here -- the dialogue layer
                        # tolerates a scrappy transcript, and barge-in now
                        # lets the user cut off a wrong answer. Set
                        # --asr-model small.en to trade back.

TIMEOUT_ASR = 8.0       # local faster-whisper, not the cloud streaming ASR
                        # the plan's latency budget assumed. Measured on the
                        # demo box: 1.5 s idle, 2.0 s with the motion engine
                        # generating, and the live viewer + 30 Hz scheduler
                        # push it further (a --view --motion local session
                        # logged 3 overruns, i.e. a saturated CPU). At 3.0 s
                        # every turn timed out into "Sorry, I missed that.";
                        # 8 s covers the loaded case with margin. Lower it
                        # when a streaming cloud ASR replaces WhisperAsr.
TIMEOUT_LLM = 4.0
TIMEOUT_TTS = 3.0
TIMEOUT_MOTION = 1.5

# ---- conversational mood (dialogue/mood.py) -------------------------------
# The lamp's affect memory. `vec` (which emotion) persists for the whole
# session; `level` (how exaggerated) relaxes toward the floor, so a mood
# is kept but stops shouting. Tuned so a strong reaction is visibly
# subdued after roughly a minute of silence but never neutral again.
MOOD_DECAY_S = 75.0          # time constant of the intensity relaxation
MOOD_FLOOR_LEVEL = 0.25      # subdued floor: remembered, never forgotten
MOOD_BASE_LEVEL = 0.35       # level at session start
MOOD_W_PROSODY = 0.15        # weight of the instant, words-free estimate
#                              made at the endpoint, before ASR returns
MOOD_W_USER = 0.35           # weight of the user's affect once the
#                              transcript exists to inform it
MOOD_W_LAMP = 0.55           # weight of the lamp's own reply affect; higher
#                              because the reply already reflects the LLM's
#                              reading of the whole conversation. Strictly
#                              above 0.5 on purpose: at exactly a half the
#                              newest reply and the accumulated past tie,
#                              and a cheerful answer after a sad stretch
#                              would leave the mood undecided.
MOOD_PUBLISH_DELTA = 0.03    # republish the decaying mood to the ambient
#                              prefetcher once it has drifted this far
MOOD_REACT_PRIOR = 0.3       # mood's share of the REACT affect estimate, so
#                              the reaction continues the mood instead of
#                              snapping back to FALLBACK_AFFECT

# ---- ambient motion (between speech, instead of static idle) --------------
AMBIENT_AFFECTS = ("interest", "boredom", "understanding")
AMBIENT_SECONDS = 2.5        # filler clip duration
AMBIENT_POOL_SIZE = 3        # prefetched ambient clips kept ready (7.5 s of
#                              buffer -- enough to cover a turn's ASR+LLM
#                              on CPU whisper without the pool draining)
AMBIENT_STALE_S = 60.0       # refresh pooled clips older than this
AMBIENT_MOOD_MIX = 0.75      # ambient affect = MIX*mood + (1-MIX)*random
#                              draw from AMBIENT_AFFECTS (kept as a variety
#                              term so a held mood doesn't get monotonous)
AMBIENT_INTENSITY_SCALE = 0.6   # ambient is a damped echo of the mood, not
#                                 a restatement of it
AMBIENT_AMP_MIN, AMBIENT_AMP_MAX = 0.55, 1.0   # post-hoc gesture damping
#                              (motion/align.damp_amplitude) as a function
#                              of mood level. CFG alone cannot do this:
#                              CFG_MIN=1.0 is still a full-size gesture.
AMBIENT_MOOD_REFRESH_COS = 0.8  # flush pooled ambient when the mood turns
#                                 this far, so a new mood lands within one
#                                 clip instead of after the queue drains
AMBIENT_MOOD_REFRESH_LEVEL = 0.25   # ... or when the level moves this much

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
