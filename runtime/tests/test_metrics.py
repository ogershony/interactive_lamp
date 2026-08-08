import numpy as np

import runtime.config as C
from runtime.eval.metrics import (boundary_offsets, correction_motion,
                                  invariant_scan, stage_latencies,
                                  sync_errors)


def _clean_stream(T=200):
    x = np.zeros((T, 9))
    t = np.arange(T) * C.DT
    x[:, :5] = C.IDLE_POSE + 0.1 * np.sin(2 * np.pi * 0.5 * t)[:, None]
    x[:, 5] = 0.5 + 0.05 * np.sin(2 * np.pi * 0.3 * t)
    x[:, 6:] = 0.8
    return x


def test_invariant_scan_clean():
    scan = invariant_scan(_clean_stream())
    assert scan["total"] == 0 and scan["per_1000"] == 0


def test_invariant_scan_catches_injected_faults():
    x = _clean_stream()
    x[50, 0] += 0.5                # rate spike (0.5 rad in one frame)
    x[100, 5] = 0.05               # under the light floor
    x[150, 2] = C.JOINT_LO[2]      # inside the 2-deg margin
    scan = invariant_scan(x)
    assert scan["rate_cap"] >= 1
    assert scan["light_floor"] == 1
    assert scan["joint_limits"] >= 1
    assert scan["total"] > 0


def test_correction_motion_opposed_vs_aligned():
    t = np.linspace(0, 2 * np.pi, 100)
    gen = np.zeros((100, 9))
    gen[:, :5] = np.sin(t)[:, None] * 0.2
    aligned = correction_motion(gen, gen)
    opposed = correction_motion(-gen, gen)
    assert aligned == 0.0
    assert opposed > 0.9


def test_boundary_offsets():
    events = [{"kind": "clip_start", "offset": o}
              for o in [0.1, 0.2, 0.3, 0.4, 0.5]]
    r = boundary_offsets(events)
    assert r["n"] == 5 and r["p50"] == 0.3
    assert boundary_offsets([])["n"] == 0


def test_subtitle_track():
    from runtime.eval.replay import subtitle_track
    events = [
        {"kind": "user_turn", "text": "hello lamp", "frame": 10},
        {"kind": "segment_planned", "text": "Hi!", "start_frame": 60,
         "seconds": 1.0},
        {"kind": "segment_planned", "text": "Nice to hear you.",
         "start_frame": 90, "seconds": 2.0},
        {"kind": "user_turn", "text": "", "frame": 200},   # empty: dropped
    ]
    subs = subtitle_track(events)
    assert subs[0] == (10, 60, "you", "hello lamp")        # ends at reply
    assert subs[1] == (60, 90, "lamp", "Hi!")
    assert subs[2][1] == 90 + round(2.0 / C.DT)
    assert len(subs) == 3


def test_stage_latencies_and_sync():
    events = [
        {"t": 0.0, "kind": "endpoint"},
        {"t": 0.15, "kind": "react_motion"},
        {"t": 1.1, "kind": "first_audio"},
        {"t": 2.0, "kind": "endpoint"},
        {"t": 2.2, "kind": "react_motion"},
        {"t": 3.4, "kind": "first_audio"},
        {"t": 5.0, "kind": "audio_seg_start", "seg": 0},
        {"t": 5.03, "kind": "motion_seg_start", "seg": 0},
        {"t": 7.0, "kind": "audio_seg_start", "seg": 1},
        {"t": 7.12, "kind": "motion_seg_start", "seg": 1},
    ]
    lat = stage_latencies(events, [("endpoint", "react_motion", "react"),
                                   ("endpoint", "first_audio", "audio")])
    assert lat["react"]["n"] == 2 and abs(lat["react"]["p50"] - 0.175) < 1e-9
    assert lat["audio"]["n"] == 2

    sync = sync_errors(events)
    assert sync["n"] == 2
    assert abs(sync["mean"] - 0.075) < 1e-9
