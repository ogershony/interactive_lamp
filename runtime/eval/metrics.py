"""
Rater-free runtime metrics (plan section 10), computed from the session
recorder's artifacts: commanded.npz (the 30 Hz stream that reached the
servos) and session_log.jsonl (timestamped events).

The safety numbers are computed here on the *commanded* stream, not on
generated clips -- the invariants are a property of what reached the
servos.
"""

import json
import pathlib

import numpy as np

import runtime.config as C


def load_session(session_dir):
    d = pathlib.Path(session_dir)
    events = [json.loads(line) for line in
              (d / "session_log.jsonl").read_text().splitlines() if line]
    frames = None
    if (d / "commanded.npz").exists():
        z = np.load(d / "commanded.npz", allow_pickle=True)
        frames = dict(frame=z["frame"], cmd=z["cmd"], tag=z["tag"])
    return events, frames


# ---- safety ---------------------------------------------------------------

def invariant_scan(cmd, tol=1e-4):
    """Counts of invariant violations over a commanded (N, 9) stream.
    Target: all zeros (per 1000 frames and absolutely).

    tol is in rad/s (and limit/light units): the recorder stores frames
    as float32, whose quantization puts an at-the-cap diff up to ~1e-5
    rad/s over the exact cap. 1e-4 is far below anything physical
    (~0.006 deg/s) while ignoring pure storage noise."""
    cmd = np.asarray(cmd, np.float64)
    dq = np.abs(np.diff(cmd[:, :5], axis=0)) / C.DT
    dlight = np.abs(np.diff(cmd[:, C.LIGHT_CH])) / C.DT
    out = dict(
        n_frames=len(cmd),
        rate_cap=int((dq > C.RATE_CAP + tol).any(axis=1).sum()),
        joint_limits=int(((cmd[:, :5] < C.JOINT_LO + C.LIMIT_MARGIN - tol) |
                          (cmd[:, :5] > C.JOINT_HI - C.LIMIT_MARGIN + tol))
                         .any(axis=1).sum()),
        light_slew=int((dlight > C.LIGHT_SLEW + tol).sum()),
        light_floor=int((cmd[:, C.LIGHT_CH] < C.LIGHT_FLOOR - tol).sum()),
        non_finite=int((~np.isfinite(cmd)).any(axis=1).sum()),
    )
    out["total"] = sum(v for k, v in out.items() if k != "n_frames")
    out["per_1000"] = 1000.0 * out["total"] / max(1, out["n_frames"])
    return out


# ---- continuity (drives the L4 go/no-go, plan 6.1 / 10) -------------------

def correction_motion(cmd_seg, gen_seg, eps=1e-4):
    """Fraction of frames in a blend window where the commanded velocity
    opposes the generated clip's velocity, per joint, averaged. This is
    what a viewer perceives as the lamp fighting itself; a large offset
    traversed *with* the motion looks fine. Gate: < 0.15."""
    dq_c = np.diff(np.asarray(cmd_seg, np.float64)[:, :5], axis=0)
    dq_g = np.diff(np.asarray(gen_seg, np.float64)[:, :5], axis=0)
    n = min(len(dq_c), len(dq_g))
    if n == 0:
        return 0.0
    dq_c, dq_g = dq_c[:n], dq_g[:n]
    active = (np.abs(dq_g) > eps) | (np.abs(dq_c) > eps)
    opposed = (np.sign(dq_c) != np.sign(dq_g)) & active
    return float(opposed.sum() / max(1, active.sum()))


def boundary_offsets(events):
    """Max-joint offsets logged by the scheduler at each clip activation
    (post-L1, pre-L3: what the blend had to absorb). Compare p50/p90
    against the dataset baseline 0.29 / 0.99 rad to confirm L1+L2 work."""
    d = [e["offset"] for e in events
         if e["kind"] == "clip_start" and "offset" in e]
    if not d:
        return dict(n=0)
    return dict(n=len(d), p50=float(np.percentile(d, 50)),
                p90=float(np.percentile(d, 90)), max=float(np.max(d)))


# ---- motion provenance ----------------------------------------------------

def motion_sources(events, frames=None):
    """Where motion came from. `requests` counts per-clip requests by
    source (engine / pool_forced / none -- engine covers both the
    remote service and a local in-process model). With the commanded
    stream, `frame_share` gives the fraction of played frames per clip
    family -- react and ambient clips come from the prefetched
    MotionPool by construction, speak clips follow the per-request
    sources."""
    req = {}
    for e in events:
        if e["kind"] == "motion_source":
            req[e["source"]] = req.get(e["source"], 0) + 1
    out = {"requests": req}
    if frames is not None:
        tags = [str(t).split(":")[0] for t in frames["tag"]]
        n = max(1, len(tags))
        out["frame_share"] = {k: round(tags.count(k) / n, 3)
                              for k in sorted(set(tags))}
    return out


# tags the scheduler falls back to when it has no clip to play. `idle` is
# procedural breathing, `hold` is the last commanded frame repeated, `cold`
# is the constant start pose -- all three read to a viewer as the lamp
# having stopped performing.
DEAD_TAGS = ("idle", "hold", "cold")


def dead_motion(frames, dead_tags=DEAD_TAGS):
    """How much of the session the lamp spent not performing -- the
    numeric form of "movement gaps". `fraction` is the share of commanded
    frames on a dead tag; `longest_s` is the longest continuous stretch,
    which is what a viewer actually notices (thirty scattered frames are
    invisible, one three-second hole is not)."""
    tags = [str(t).split(":")[0] for t in frames["tag"]]
    n = len(tags)
    if n == 0:
        return dict(n_frames=0, fraction=0.0, longest_s=0.0, runs=0)
    dead = [t in dead_tags for t in tags]
    longest = run = runs = 0
    for d in dead:
        if d:
            run += 1
            if run == 1:
                runs += 1
            longest = max(longest, run)
        else:
            run = 0
    return dict(n_frames=n, fraction=round(sum(dead) / n, 4),
                longest_s=round(longest * C.DT, 3), runs=runs)


# ---- mood -----------------------------------------------------------------

def mood_track(events):
    """The conversation's affect memory over time, from the `mood` events
    the behavior layer emits on every update: [(t, emotion, level)]. This
    is how "did it stay sad?" gets answered without watching the video."""
    return [(e["t"], e["emotion"], e["level"]) for e in events
            if e["kind"] == "mood"]


# ---- latency / sync -------------------------------------------------------

# The turn pipeline's stage boundaries, as (from_kind, to_kind, name) for
# stage_latencies(). `reaction` is the plan's sub-200 ms promise; `answer`
# is what the user experiences as "how long until it says something".
TURN_STAGES = [
    ("endpoint", "react_motion", "reaction"),
    ("endpoint", "asr_final", "asr"),
    # `llm_first` is when the first phrase became speakable, `llm_reply`
    # when the model stopped writing. The gap between them is what
    # streaming the reply buys back.
    ("asr_final", "llm_first", "llm_first"),
    ("asr_final", "llm_reply", "llm_full"),
    ("endpoint", "segment_planned", "answer"),
]


def stage_latencies(events, stages=None):
    """Per-turn deltas between event kinds. `stages` is a list of
    (from_kind, to_kind, name); within each turn the first `to` after
    each `from` is matched. Defaults to TURN_STAGES. Returns
    name -> dict(n, p50, p95) seconds."""
    out = {}
    for src, dst, name in (TURN_STAGES if stages is None else stages):
        deltas = []
        t_src = None
        for e in events:
            if e["kind"] == src:
                t_src = e["t"]
            elif e["kind"] == dst and t_src is not None:
                deltas.append(e["t"] - t_src)
                t_src = None
        out[name] = dict(n=len(deltas),
                         p50=float(np.percentile(deltas, 50)) if deltas
                         else None,
                         p95=float(np.percentile(deltas, 95)) if deltas
                         else None)
    return out


def sync_errors(events):
    """Per speech segment, how far the motion actually started from where
    it was planned to, in seconds. Target p95 < 0.100.

    Measured against the plan rather than against the audio events,
    because the audio player is a queue: `play()` returns as soon as a
    segment is enqueued, so an audio_seg_start timestamp is when the
    segment joined the queue, not when it was heard. Motion and audio are
    laid out from one cursor of measured TTS durations (behavior._speak),
    so a clip that starts on its planned frame is aligned with its
    speech by construction -- and a clip that starts late is exactly the
    desync a viewer sees."""
    planned = {e["seg"]: e["start_frame"] for e in events
               if e["kind"] == "segment_planned" and "start_frame" in e}
    errs = [abs(e["frame"] - planned[e["seg"]]) * C.DT for e in events
            if e["kind"] == "motion_seg_start" and e.get("seg") in planned
            and "frame" in e]
    if not errs:
        return dict(n=0)
    return dict(n=len(errs), mean=float(np.mean(errs)),
                p95=float(np.percentile(errs, 95)),
                max=float(np.max(errs)))
