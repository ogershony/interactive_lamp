#!/usr/bin/env python3
"""
Measure this box's mic levels and recommend VAD_ENERGY_THRESHOLD.

The energy VAD (runtime/audio/vad.py) calls a 10 ms block speech when its
RMS clears VAD_ENERGY_THRESHOLD, and the Endpointer ends a turn only after
MIN_SPEECH_MS of accumulated speech followed by T_END_MS of continuous
silence. A threshold above your speaking level breaks that quietly rather
than loudly: onsets still fire on the loud syllables, but the accumulated
speech never reaches MIN_SPEECH_MS, so no endpoint fires, no turn starts,
and the lamp just breathes at you. The session log shows it as
speech_onset events with no matching endpoint.

This records a silence sample and a speech sample, prints the block-RMS
distribution of each, and replays the real Endpointer over the speech
sample at candidate thresholds so you can see which ones would actually
have ended a turn.

    uv run scripts/calibrate_vad.py
    uv run scripts/calibrate_vad.py --seconds 6
    uv run scripts/calibrate_vad.py --compare      # ... vs the silero VAD

The energy VAD is the fallback now (VAD_DEFAULT = "silero"), so the
threshold below only matters on a box without onnxruntime, or when
--compare says silero did worse here. --compare is the honest way to
decide: it replays both detectors over the same two recordings.

Then either put the recommendation in runtime/config.py
(VAD_ENERGY_THRESHOLD) or pass it per-run:

    uv run runtime/main.py --converse --vad-threshold 0.004
"""

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import runtime.config as C
from runtime.audio.vad import ENDPOINT, ONSET, EnergyVad, Endpointer


def record(seconds, sr, label):
    import sounddevice as sd

    print(f"\n{label}  ({seconds:.0f}s)", flush=True)
    for i in (3, 2, 1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1.0)
    print("  recording NOW      ", flush=True)
    pcm = sd.rec(int(sr * seconds), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    print("  done", flush=True)
    return pcm.ravel()


def block_rms(pcm, block):
    n = len(pcm) // block
    x = pcm[:n * block].reshape(n, block).astype(np.float64) / 32768.0
    return np.sqrt((x ** 2).mean(axis=1))


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def _runs(pcm, block, vad):
    """Number of separate voiced runs a VAD finds in one sample."""
    dec = [bool(vad(pcm[i:i + block]))
           for i in range(0, len(pcm) - block + 1, block)]
    return sum(1 for i, d in enumerate(dec) if d and not (i and dec[i - 1]))


def replay(pcm, block, vad):
    """Run a VAD + the real Endpointer over a sample; return the counts."""
    ep = Endpointer()
    onsets = endpoints = 0
    speech_blocks = 0
    for i in range(0, len(pcm) - block + 1, block):
        is_speech = vad(pcm[i:i + block])
        speech_blocks += bool(is_speech)
        ev = ep.update(is_speech)
        onsets += ev == ONSET
        endpoints += ev == ENDPOINT
    # a turn also ends when the stream stops: flush trailing silence
    for _ in range(int(C.T_END_SHORT_MS / C.AUDIO_BLOCK_MS) + 1):
        endpoints += ep.update(False) == ENDPOINT
    return onsets, endpoints, speech_blocks * C.AUDIO_BLOCK_MS


def simulate(pcm, block, threshold):
    """Energy VAD at `threshold`, replayed through the real endpointer."""
    return replay(pcm, block, EnergyVad(threshold=threshold))


def compare_silero(quiet, speech, block):
    """Score the Silero VAD on the same two samples as the energy sweep.

    Worth its own section because the interesting number is not the
    threshold -- Silero's is box-independent, which is the point -- but
    *fragmentation*: how many separate voiced runs one utterance breaks
    into. Every extra run is a gap the endpointer might mistake for the
    end of a turn."""
    from runtime.audio.vad import SileroVad

    out = {}
    for label, pcm in (("speech", speech), ("silence", quiet)):
        vad = SileroVad()
        dec = [bool(vad(pcm[i:i + block]))
               for i in range(0, len(pcm) - block + 1, block)]
        runs = sum(1 for i, d in enumerate(dec) if d and not (i and dec[i - 1]))
        vad.reset()
        o, e, ms = replay(pcm, block, vad)
        out[label] = dict(onsets=o, endpoints=e, speech_ms=ms, runs=runs)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seconds", type=float, default=5.0,
                   help="length of each sample")
    p.add_argument("--sr", type=int, default=C.AUDIO_SR)
    p.add_argument("--save", default="runtime/vad_calib.npz",
                   help="where to keep the raw samples ('' to skip)")
    p.add_argument("--analyze", default=None,
                   help="re-analyze a saved .npz instead of recording")
    p.add_argument("--compare", action="store_true",
                   help="also score the silero VAD on the same samples")
    args = p.parse_args()

    if args.analyze:
        d = np.load(args.analyze)
        quiet, speech = d["quiet"], d["speech"]
        args.sr = int(d["sr"])
        args.save = ""                      # already on disk
        print(f"analyzing {args.analyze} "
              f"({len(quiet)/args.sr:.1f}s silence, "
              f"{len(speech)/args.sr:.1f}s speech)")

    block = int(args.sr * C.AUDIO_BLOCK_MS / 1000)
    print(f"block {C.AUDIO_BLOCK_MS} ms ({block} samples @ {args.sr} Hz)")
    print(f"current VAD_ENERGY_THRESHOLD = {C.VAD_ENERGY_THRESHOLD}")
    print(f"endpoint rule: >= {C.MIN_SPEECH_MS} ms speech, then "
          f">= {C.T_END_MS} ms continuous silence")

    if not args.analyze:
        quiet = record(args.seconds, args.sr,
                       "1/2  SAY NOTHING -- room tone")
        speech = record(args.seconds, args.sr,
                        "2/2  TALK NORMALLY -- as you would to the lamp")

    q, s = block_rms(quiet, block), block_rms(speech, block)
    print("\n---- block RMS (full scale 1.0) ----")
    print(f"{'':10s} {'p50':>9s} {'p90':>9s} {'p95':>9s} {'max':>9s}")
    print(f"{'silence':10s} {pct(q,50):9.5f} {pct(q,90):9.5f} "
          f"{pct(q,95):9.5f} {q.max():9.5f}")
    print(f"{'speech':10s} {pct(s,50):9.5f} {pct(s,90):9.5f} "
          f"{pct(s,95):9.5f} {s.max():9.5f}")

    # A threshold has to sit above room tone and below the *voiced* part of
    # speech. Compare the noise floor against speech p90, NOT speech p50:
    # half the blocks in a normal utterance are the gaps between words, so
    # the speech median is a silent block and would make even an excellent
    # mic look indistinguishable from the room.
    floor, voiced = pct(q, 95), pct(s, 90)
    print(f"\nnoise ceiling (silence p95): {floor:.5f}")
    print(f"voiced level  (speech  p90): {voiced:.5f}")
    if voiced > floor:
        print(f"headroom                   : "
              f"{20*np.log10(voiced/max(floor,1e-9)):.1f} dB")

    # The verdict comes from replaying the real endpointer at each candidate
    # over both samples -- a threshold is only good if it ends a turn on
    # speech AND stays quiet on room tone. Percentiles pick the candidates;
    # the simulation decides.
    print("\n---- real VAD + Endpointer replayed on both samples ----")
    print(f"{'threshold':>10s} {'onsets':>7s} {'endpoints':>10s} "
          f"{'speech ms':>10s} {'false ms':>9s}")
    cands = sorted({C.VAD_ENERGY_THRESHOLD,
                    round(float(np.sqrt(floor * voiced)), 5),
                    round(floor * 1.2, 5), round(floor * 1.6, 5),
                    round(voiced * 0.25, 5), round(voiced * 0.4, 5),
                    round(voiced * 0.6, 5)})
    rows = []
    for t in cands:
        o, e, ms = simulate(speech, block, t)
        _, _, false_ms = simulate(quiet, block, t)     # room tone leakage
        rows.append((t, o, e, ms, false_ms))
        mark = "   (current)" if t == C.VAD_ENERGY_THRESHOLD else ""
        print(f"{t:10.5f} {o:7d} {e:10d} {ms:10.0f} {false_ms:9.0f}{mark}")

    if args.compare:
        print("\n---- silero VAD on the same samples ----")
        try:
            sil = compare_silero(quiet, speech, block)
        except Exception as e:      # noqa: BLE001 -- missing model or ORT
            print(f"unavailable: {type(e).__name__}: {e}")
        else:
            eng_runs = _runs(speech, block, EnergyVad(
                threshold=C.VAD_ENERGY_THRESHOLD))
            print(f"{'':10s} {'onsets':>7s} {'endpoints':>10s} "
                  f"{'speech ms':>10s} {'runs':>6s}")
            for label in ("speech", "silence"):
                r = sil[label]
                print(f"{label:10s} {r['onsets']:7d} {r['endpoints']:10d} "
                      f"{r['speech_ms']:10.0f} {r['runs']:6d}")
            print(f"\nvoiced runs in one utterance: silero "
                  f"{sil['speech']['runs']}, energy {eng_runs} "
                  f"(fewer is better -- each gap is a chance to cut you off)")
            if sil["silence"]["endpoints"] == 0 \
                    and sil["speech"]["endpoints"] >= 1 \
                    and sil["speech"]["runs"] <= max(1, eng_runs):
                print("silero wins on this box; keep VAD_DEFAULT = 'silero'")
            else:
                print("silero did not clearly beat the energy VAD here; "
                      "run with --vad energy and the threshold below")

    # good = ends a turn, clears MIN_SPEECH_MS, and barely triggers on room
    # tone; among those prefer the largest (most noise margin)
    good = [r for r in rows
            if r[2] >= 1 and r[3] >= C.MIN_SPEECH_MS
            and r[4] <= C.MIN_SPEECH_MS / 2]
    print()
    if not good:
        print("No candidate both ends a turn and stays quiet on room tone.")
        if voiced <= floor:
            print("Speech never rises above the room: check the mic's "
                  "physical gain/pad switch, aim it at you, and rerun.")
        else:
            print("There is headroom, so this is an endpointing problem "
                  "rather than a level one -- send the numbers above on.")
        return 1

    # Among thresholds that end a turn without triggering on room tone,
    # take the one that captures the MOST speech, not the one with the most
    # noise margin. The observed live failure is under-accumulation --
    # onsets fire but never reach MIN_SPEECH_MS, so the turn never ends --
    # and a higher threshold makes that strictly worse. Ties go to the
    # larger threshold, which keeps whatever margin is free.
    clean = [r for r in good if r[4] == 0] or good
    pick = max(clean, key=lambda r: (r[3], r[0]))[0]
    print(f"recommended VAD_ENERGY_THRESHOLD = {pick:.5f}")
    print(f"try it:  uv run runtime/main.py --converse --motion local "
          f"--vad-threshold {pick:.5f}")
    print(f"keep it: set VAD_ENERGY_THRESHOLD = {pick:.5f} in "
          f"runtime/config.py")
    if args.save:
        np.savez_compressed(args.save, quiet=quiet, speech=speech,
                            sr=args.sr, block_ms=C.AUDIO_BLOCK_MS)
        print(f"\nsamples saved to {args.save} -- rerun the analysis without "
              f"re-recording:\n  uv run scripts/calibrate_vad.py "
              f"--analyze {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
