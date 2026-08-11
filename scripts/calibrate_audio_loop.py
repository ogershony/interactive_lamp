#!/usr/bin/env python3
"""
Measure this box's speaker->mic loop delay and recommend OUTPUT_LAG_MS.

The lamp is half duplex: it mutes its own microphone while it speaks, so
it does not hear itself. That gate has to stay shut until the sound has
actually left the speaker -- and the runtime cannot know when that is.
The playback queue emptying only means PortAudio has the samples; behind
it sit PortAudio's own buffer, PulseAudio's, and under WSLg a socket to
Windows. PortAudio reports the first of those and knows nothing about the
rest.

Getting this wrong is quiet and expensive. With the gate opening 150 ms
after the queue emptied, a live session had the lamp transcribe the tail
of its own sentences as the user's next turn on 3 of 8 turns -- it asked
"Did something happen?", heard itself, and answered its own question.

This plays a short chirp, records the whole time, and cross-correlates
to find how late the chirp came back:

    uv run scripts/calibrate_audio_loop.py
    uv run scripts/calibrate_audio_loop.py --repeats 7 --volume 0.5

Then set OUTPUT_LAG_MS in runtime/config.py, or pass it per run:

    uv run runtime/main.py --converse --output-lag-ms 320

Run it with the speaker and mic you will actually converse with, at the
volume you will actually use. Headphones give a much shorter loop than
speakers, and a Bluetooth device a much longer one.
"""

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import runtime.config as C


def chirp(seconds, sr, f0=800.0, f1=4000.0):
    """A short rising sweep: broadband enough to survive a cheap speaker,
    and its autocorrelation is a single sharp peak, unlike a tone."""
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    x = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / seconds * t ** 2))
    ramp = int(0.005 * sr)                     # avoid a click at the edges
    env = np.ones_like(x)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return x * env


def measure(sr, chirp_s, pad_s, volume, quiet=False):
    """Play a chirp into a simultaneously recording duplex stream and
    return the loop delay in ms, plus the correlation peak strength."""
    import sounddevice as sd

    probe = chirp(chirp_s, sr)
    n_out = int((chirp_s + pad_s) * sr)
    out = np.zeros(n_out, np.float32)
    out[:len(probe)] = probe * volume
    rec = sd.playrec(out, samplerate=sr, channels=1, blocking=True)
    heard = np.asarray(rec, np.float64).ravel()

    # normalized cross-correlation; the lag of the peak is the round trip
    a = heard - heard.mean()
    b = probe - probe.mean()
    corr = np.correlate(a, b, mode="valid")
    denom = np.sqrt((b ** 2).sum() * np.convolve(
        a ** 2, np.ones(len(b)), mode="valid")) + 1e-12
    ncorr = corr / denom
    lag = int(np.argmax(np.abs(ncorr)))
    return lag * 1000.0 / sr, float(np.abs(ncorr[lag]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--sr", type=int, default=C.AUDIO_SR)
    p.add_argument("--chirp-ms", type=float, default=120.0)
    p.add_argument("--pad-s", type=float, default=2.0,
                   help="how long to keep recording after the chirp; must "
                        "exceed the delay you are trying to measure")
    p.add_argument("--volume", type=float, default=0.4,
                   help="0..1, the level you would actually converse at")
    args = p.parse_args()

    print(f"playing {args.repeats} chirps at volume {args.volume:.2f}; "
          f"keep the room quiet and don't move the mic")
    lags, weak = [], 0
    for i in range(args.repeats):
        ms, strength = measure(args.sr, args.chirp_ms / 1000.0,
                               args.pad_s, args.volume)
        ok = strength >= 0.20
        weak += not ok
        print(f"  {i+1}/{args.repeats}  {ms:7.1f} ms   "
              f"peak {strength:.2f}{'' if ok else '   (weak -- ignored)'}")
        if ok:
            lags.append(ms)
        time.sleep(0.25)

    if not lags:
        print("\nNo chirp came back. The mic never heard the speaker: check "
              "that both are the default devices, raise --volume, and make "
              "sure you are not on headphones (nothing leaks back). If you "
              "*will* converse on headphones, the loop really is silent and "
              "OUTPUT_LAG_MS only needs to cover the speaker latency -- "
              "PortAudio's reported figure is enough there.")
        return 1

    lags = np.array(lags)
    p50, p90 = float(np.percentile(lags, 50)), float(np.percentile(lags, 90))
    print(f"\nloop delay: p50 {p50:.0f} ms   p90 {p90:.0f} ms   "
          f"max {lags.max():.0f} ms   ({len(lags)} good of {args.repeats})")
    if weak:
        print(f"{weak} chirp(s) came back too faintly to trust; if most of "
              f"them did, raise --volume and rerun.")

    # Recommend the worst observed loop, rounded up, plus a little margin.
    # Erring high costs responsiveness after the lamp speaks; erring low
    # costs the lamp hearing itself, which corrupts the conversation.
    pick = int(np.ceil((lags.max() * 1.2) / 10.0) * 10)
    print(f"\nrecommended OUTPUT_LAG_MS = {pick}"
          f"   (worst loop + 20% margin; current {C.OUTPUT_LAG_MS})")
    print(f"try it:  uv run runtime/main.py --converse --motion local "
          f"--output-lag-ms {pick}")
    print(f"keep it: set OUTPUT_LAG_MS = {pick} in runtime/config.py")
    if pick > 600:
        print("\nThat is a long loop. It means the lamp cannot hear you "
              "until ~%.1f s after it stops talking. Headphones, or a "
              "speaker closer to the lamp and further from the mic, will "
              "buy that back." % (pick / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
