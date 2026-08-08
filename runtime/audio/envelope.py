"""
TTS PCM -> 30 Hz speech envelope (plan 6.3): short-time RMS on the
frame grid, per-segment normalization, clip, 4 Hz low-pass. The result
rides on J5 and LED brightness via motion/modulate.py.
"""

import numpy as np

import runtime.config as C


def envelope_30hz(pcm, sample_rate, lp_hz=None, clip=None):
    """int16 (or float) mono PCM -> (T,) normalized envelope at 30 Hz,
    T = ceil(n / hop). Zero-mean unit-std per segment, clipped to
    [-ENV_CLIP, ENV_CLIP], then one-pole low-passed at ENV_LP_HZ so it
    tracks syllables (4-7 Hz) without jitter."""
    lp_hz = C.ENV_LP_HZ if lp_hz is None else lp_hz
    clip = C.ENV_CLIP if clip is None else clip
    x = np.asarray(pcm)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    else:
        x = x.astype(np.float64)
    if len(x) == 0:
        return np.zeros(0)
    hop = int(round(sample_rate / C.FPS))
    T = int(np.ceil(len(x) / hop))
    pad = np.pad(x, (0, T * hop - len(x)))
    rms = np.sqrt((pad.reshape(T, hop) ** 2).mean(axis=1))

    e = (rms - rms.mean()) / (rms.std() + 1e-8)
    e = np.clip(e, -clip, clip)

    # one-pole low-pass on the 30 Hz grid
    alpha = 1.0 - np.exp(-2.0 * np.pi * lp_hz * C.DT)
    out = np.empty_like(e)
    acc = e[0]
    for i, v in enumerate(e):
        acc += alpha * (v - acc)
        out[i] = acc
    return out


def duration_s(pcm, sample_rate):
    """Exact segment duration from returned PCM length (plan 4.5): the
    v1 way to get motion-speech alignment without TTS timing metadata."""
    return len(pcm) / float(sample_rate)


def resample(pcm, sr_from, sr_to):
    """Linear-interpolation resample of int16 PCM (TTS voices synth at
    22.05 kHz, the duplex stream runs at 16 kHz). Good enough for a
    small speaker; duration is preserved exactly."""
    if sr_from == sr_to or len(pcm) == 0:
        return np.asarray(pcm, np.int16)
    n_out = int(round(len(pcm) * sr_to / sr_from))
    src = np.linspace(0.0, len(pcm) - 1, n_out)
    return np.interp(src, np.arange(len(pcm)),
                     np.asarray(pcm, np.float64)).astype(np.int16)
