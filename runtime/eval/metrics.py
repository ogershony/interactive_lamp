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
    source (cache / engine / cache_forced / none). With the commanded
    stream, `frame_share` gives the fraction of played frames per clip
    family -- react and ambient clips are cache-only by construction,
    speak clips follow the per-request sources."""
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


# ---- latency / sync -------------------------------------------------------

def stage_latencies(events, stages):
    """Per-turn deltas between event kinds. `stages` is a list of
    (from_kind, to_kind, name); within each turn the first `to` after
    each `from` is matched. Returns name -> dict(n, p50, p95) seconds."""
    out = {}
    for src, dst, name in stages:
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
    """Per speech segment, |motion_start - audio_start| in seconds.
    Events carry a shared `seg` id. Target p95 < 0.100."""
    audio = {e["seg"]: e["t"] for e in events
             if e["kind"] == "audio_seg_start"}
    errs = [abs(e["t"] - audio[e["seg"]]) for e in events
            if e["kind"] == "motion_seg_start" and e.get("seg") in audio]
    if not errs:
        return dict(n=0)
    return dict(n=len(errs), mean=float(np.mean(errs)),
                p95=float(np.percentile(errs, 95)))
