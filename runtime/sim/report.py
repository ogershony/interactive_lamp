"""
What a simulated session actually tells you.

Everything here is computed from the session directory the run wrote --
the same artifacts a live `--converse` session leaves -- so the same
report can be produced for a real conversation. The gates at the bottom
are the ones that correspond to the complaints this harness exists to
catch:

    turns       every scripted line produced exactly one turn. Fails when
                the endpointer cuts a line in two, or never fires.
    heard       the ASR got roughly the right words.
    gaps        the lamp kept performing. `longest_s` is the number that
                matters -- scattered idle frames are invisible, one
                three-second hole is the thing that reads as "it forgot
                about me".
    safety      zero post-governor invariant violations. Non-negotiable.

`mood` is not gated, because there is no single right trajectory -- it
is printed as a track so a person can see whether the lamp stayed sad.
"""

import difflib
import json

import numpy as np

import runtime.config as C
from runtime.eval import metrics

GATE_DEAD_FRACTION = 0.05      # share of frames not performing
GATE_DEAD_LONGEST_S = 1.5      # longest single stretch
GATE_HEARD = 0.6               # transcript similarity, 0..1


def _similar(a, b):
    norm = str.maketrans("", "", ".,!?;:'\"")
    a = " ".join(a.lower().translate(norm).split())
    b = " ".join(b.lower().translate(norm).split())
    return difflib.SequenceMatcher(None, a, b).ratio()


def _turn_table(events, expected):
    """Scripted line vs what the lamp heard and said, in order. Replies
    are attributed to the most recent user_turn, which is exact because
    the runtime is strictly one turn at a time."""
    rows = []
    for e in events:
        if e["kind"] == "user_turn":
            i = len(rows)
            want = expected[i] if i < len(expected) else None
            rows.append(dict(
                i=i, t=round(e["t"], 2), scripted=want,
                heard=e.get("text", ""), replied=[],
                match=round(_similar(want, e.get("text", "")), 3)
                if want is not None else None))
        elif e["kind"] == "segment_planned" and rows:
            rows[-1]["replied"].append(dict(
                text=e.get("text", ""), affect=e.get("affect") or {},
                intensity=e.get("intensity"), cfg=e.get("cfg"),
                prosody=e.get("prosody") or {}))
    return rows


def affect_summary(events):
    """Every conditioning vector the flow-matching prior was given, by
    where it came from. `speak` is what the LLM chose for a reply;
    `react` and `ambient` are what the mood chose while nobody was
    talking. The split is the thing to look at -- most of a session's
    motion is the second kind, and it used to leave no trace at all."""
    by_source = {}
    for e in events:
        if e["kind"] == "motion_source":
            kind = str(e.get("tag", "")).split(":")[0] or "speak"
        elif e["kind"] == "pool_request":
            kind = e.get("role", "pool")
        else:
            continue
        d = by_source.setdefault(kind, dict(n=0, cfg=[], weight={}))
        d["n"] += 1
        if e.get("cfg") is not None:
            d["cfg"].append(float(e["cfg"]))
        for name, w in (e.get("affect") or {}).items():
            d["weight"][name] = d["weight"].get(name, 0.0) + float(w)
    out = {}
    for kind, d in sorted(by_source.items()):
        total = sum(d["weight"].values()) or 1.0
        top = sorted(d["weight"].items(), key=lambda kv: -kv[1])[:4]
        out[kind] = dict(
            n=d["n"],
            cfg=round(float(np.mean(d["cfg"])), 2) if d["cfg"] else None,
            affect={k: round(v / total, 3) for k, v in top})
    return out


def talked_over(marks, busy, gate_s=None):
    """Scripted lines the lamp was physically incapable of hearing --
    spoken while it was still talking, or inside the half-duplex gate
    afterwards.

    Worth its own number because it looks identical to an endpointing bug
    from the outside ("I talked and nothing happened") and has a
    completely different cause. The lamp is half duplex: while it speaks,
    the microphone is muted and whatever you say is discarded. Widening
    the gate to stop it hearing itself widens this window too, so the
    trade has to stay visible."""
    gate_s = (C.OUTPUT_LAG_MS + C.HALF_DUPLEX_TAIL_MS) / 1000.0 \
        if gate_s is None else gate_s
    deaf = [(s, e + gate_s) for s, e in busy]
    out = []
    for t0, t1, text in marks:
        covered = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in deaf)
        if covered > 0.5 * (t1 - t0):
            out.append(dict(t=round(t0, 2), text=text,
                            covered=round(covered / (t1 - t0), 2)))
    return out


def build(session_dir, script=None, extra=None, marks=None, busy=None):
    events, frames = metrics.load_session(session_dir)
    expected = list(script.lines) if script is not None else []
    rows = _turn_table(events, expected)
    over = talked_over(marks, busy) if marks and busy is not None else []

    dead = metrics.dead_motion(frames) if frames is not None else {}
    scan = metrics.invariant_scan(frames["cmd"]) if frames is not None else {}
    matches = [r["match"] for r in rows if r["match"] is not None]

    engine_ms = [e["ms"] for e in events
                 if e["kind"] == "motion_source" and "ms" in e]
    rep = dict(
        session=str(session_dir),
        script=None if script is None else script.name,
        seconds=round(len(frames["cmd"]) * C.DT, 2) if frames is not None
        else 0.0,
        turns=dict(scripted=len(expected), detected=len(rows),
                   discarded=len([e for e in events
                                  if e["kind"] == "speech_discarded"]),
                   talked_over=over,
                   barge_ins=len([e for e in events
                                  if e["kind"] == "barge_in"])),
        heard=dict(n=len(matches),
                   mean=round(float(np.mean(matches)), 3) if matches else None,
                   worst=round(float(np.min(matches)), 3) if matches else None),
        latency=metrics.stage_latencies(events),
        sync=metrics.sync_errors(events),
        motion=metrics.motion_sources(events, frames),
        gaps=dead,
        continuity=metrics.boundary_offsets(events),
        affect=affect_summary(events),
        safety=scan,
        engine_ms=dict(
            n=len(engine_ms),
            p50=float(np.percentile(engine_ms, 50)) if engine_ms else None,
            p95=float(np.percentile(engine_ms, 95)) if engine_ms else None),
        mood=[dict(t=round(t, 2), emotion=e, level=round(lv, 3))
              for t, e, lv in metrics.mood_track(events)],
        turn_table=rows,
        **(extra or {}),
    )
    rep["gates"] = _gates(rep)
    rep["ok"] = all(g["ok"] for g in rep["gates"])
    return rep


def _gates(rep):
    g = []
    if rep["turns"]["scripted"]:
        over = len(rep["turns"]["talked_over"])
        detail = (f"{rep['turns']['detected']} detected of "
                  f"{rep['turns']['scripted']} scripted")
        if over:
            detail += (f"; {over} spoken while the lamp was talking "
                       f"(half-duplex, not an endpointing fault)")
        g.append(dict(name="turns",
                      ok=rep["turns"]["detected"] + over
                      >= rep["turns"]["scripted"],
                      detail=detail))
    if rep["heard"]["mean"] is not None:
        g.append(dict(name="heard", ok=rep["heard"]["mean"] >= GATE_HEARD,
                      detail=f"mean transcript match {rep['heard']['mean']} "
                             f"(worst {rep['heard']['worst']})"))
    if rep["gaps"]:
        ok = (rep["gaps"]["fraction"] <= GATE_DEAD_FRACTION
              and rep["gaps"]["longest_s"] <= GATE_DEAD_LONGEST_S)
        g.append(dict(name="gaps", ok=ok,
                      detail=f"{rep['gaps']['fraction']:.1%} of frames not "
                             f"performing, longest "
                             f"{rep['gaps']['longest_s']:.2f}s"))
    if rep["safety"]:
        g.append(dict(name="safety", ok=rep["safety"]["total"] == 0,
                      detail=f"{rep['safety']['total']} invariant violations"))
    return g


def render(rep):
    """The human-readable form. Kept in one place so the CLI and the
    tests agree on what a run looked like."""
    out = [f"session {rep['session']}",
           f"script  {rep['script']}   {rep['seconds']:.1f}s simulated"]

    out.append("\n---- turns ----")
    for r in rep["turn_table"]:
        out.append(f"[{r['t']:6.2f}s] said : {r['scripted']!r}")
        out.append(f"           heard: {r['heard']!r}"
                   + (f"   (match {r['match']:.2f})"
                      if r["match"] is not None else ""))
        for seg in r["replied"]:
            out.append(f"           lamp : {seg['text']!r}")
            chip = " ".join(f"{k} {v:.2f}" for k, v in
                            list(seg["affect"].items())[:3])
            if chip:
                pr = seg.get("prosody") or {}
                voice = (f"  voice v{pr['valence']:+.2f} a{pr['arousal']:+.2f}"
                         if "valence" in pr else "")
                out.append(f"                  [{chip}  cfg "
                           f"{seg['cfg']:.2f}]{voice}")
    for o in rep["turns"]["talked_over"]:
        out.append(f"  !! [{o['t']:6.2f}s] {o['text']!r} was said while the "
                   f"lamp was still talking ({o['covered']:.0%} covered) -- "
                   f"half duplex, the mic was muted")
    missed = (rep["turns"]["scripted"] - rep["turns"]["detected"]
              - len(rep["turns"]["talked_over"]))
    if missed > 0:
        out.append(f"  !! {missed} scripted line(s) never became a turn")
    if rep["turns"]["barge_ins"]:
        out.append(f"  -- {rep['turns']['barge_ins']} barge-in(s): a reply "
                   f"was abandoned because the user started talking")
    if rep["turns"]["discarded"]:
        out.append(f"  !! {rep['turns']['discarded']} utterance(s) discarded "
                   f"below MIN_SPEECH_MS (onset with no endpoint)")

    out.append("\n---- latency (s) ----")
    for name, d in rep["latency"].items():
        if d["n"]:
            out.append(f"  {name:9s} n={d['n']:<3d} p50 {d['p50']:.2f}  "
                       f"p95 {d['p95']:.2f}")
    if rep["sync"].get("n"):
        out.append(f"  motion vs plan p95 {rep['sync']['p95']:.3f}s "
                   f"max {rep['sync']['max']:.3f}s "
                   f"(n={rep['sync']['n']}, target < 0.100)")
    if rep["engine_ms"]["n"]:
        out.append(f"  motion engine p50 {rep['engine_ms']['p50']:.0f} ms  "
                   f"p95 {rep['engine_ms']['p95']:.0f} ms")

    out.append("\n---- motion ----")
    out.append(f"  sources     {rep['motion'].get('requests')}")
    out.append(f"  frame share {rep['motion'].get('frame_share')}")
    if rep["gaps"]:
        out.append(f"  not performing {rep['gaps']['fraction']:.1%} of frames"
                   f"  (longest {rep['gaps']['longest_s']:.2f}s over "
                   f"{rep['gaps']['runs']} stretches)")
    if rep["continuity"].get("n"):
        c = rep["continuity"]
        out.append(f"  boundary offsets p50 {c['p50']:.2f} p90 {c['p90']:.2f} "
                   f"rad  (dataset baseline 0.29 / 0.99)")

    if rep["affect"]:
        out.append("\n---- affect conditioning the motion prior ----")
        for kind, d in rep["affect"].items():
            mix = " ".join(f"{k} {v:.2f}" for k, v in d["affect"].items())
            cfg = "" if d["cfg"] is None else f"  cfg~{d['cfg']:.2f}"
            out.append(f"  {kind:9s} n={d['n']:<4d}{cfg}   {mix}")

    if rep["mood"]:
        out.append("\n---- mood ----")
        prev = None
        for m in rep["mood"]:
            bar = "#" * int(round(m["level"] * 20))
            mark = " " if m["emotion"] == prev else ">"
            out.append(f"  {mark}{m['t']:7.2f}s  {m['emotion']:14s} "
                       f"{m['level']:.2f} {bar}")
            prev = m["emotion"]

    out.append("\n---- gates ----")
    for g in rep["gates"]:
        out.append(f"  [{'ok' if g['ok'] else 'FAIL'}] {g['name']:8s} "
                   f"{g['detail']}")
    out.append(f"\n{'PASS' if rep['ok'] else 'FAIL'}")
    return "\n".join(out)


def write(session_dir, rep):
    import pathlib
    p = pathlib.Path(session_dir) / "sim_report.json"
    p.write_text(json.dumps(rep, indent=2))
    return p
