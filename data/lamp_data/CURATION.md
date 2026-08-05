# Retargeting iteration + dataset curation

> **STATUS (2026-08-04): dataset v1 FROZEN at mapping v1.4.** All 926
> clips reviewed (47 manual by Oren; the rest audited against rules
> calibrated on those verdicts + visual contact-sheet review of the
> 168 ambiguous clips). v1.4 re-audited: J4 banking validated on the
> 12 highest-yaw keeps, 0 real regressions, 0 rescues (3 candidates
> examined, still inexpressive). Final: **819 keep / 107 drop** ->
> `dataset/lamp_dataset_v1.4.npz`: 740 train / 79 val clips (grouped
> by base animation, no head-angle leakage), 86,855 frames, 47.3 min,
> every emotion covered by >= 77 clips at fraction >= 0.5, and all
> five joints alive (J4: 0.49 rad of banking; was 0 through v1.3).

Goal: iteratively improve the Cozmo->lamp mapping (now in
`data/pipeline.py`; was `data/retarget.py` when the runs below were made)
and curate the trajectory set until it is ready for generative-model
training (flow matching as of plan v2; was CVAE). Cuteness is the lamp's
*identity*: it lives in the mapping and in curation, so that everything
the model learns is cute. The model conditions on the 16-d soft emotion
vector (content), never on a "cute" knob.

## Layout

```
data/lamp_data/
  npz/<run>/            one directory per retarget run (gitignored).
                        run.json marks a run complete; tools default to
                        the newest complete run. scratch/ = single-clip
                        debug output, never a run. Metric diffs need the
                        runs' CSVs (git-tracked), not the npz. Kept on
                        disk: v1.0-baseline + v1.4 (dataset freeze);
                        v1.1-v1.3 deleted 2026-08-05, regenerable from
                        git history of the mapping.
  metrics/<run>.csv     per-clip quality metrics (gitignored; keep the
                        local copies -- v1.1-v1.3 CSVs are the only
                        remaining record of those runs)
  metrics/<run>_emotions.md   emotion-preservation report
  curation.csv          human verdicts (git-tracked; the audit log is
                        git history). Survives all re-runs by design.
  review_gifs/          sha1-cached side-by-side GIFs (gitignored)
  dataset/              exported training sets (gitignored)
```

Never store precious files inside an npz run directory.

## The loop (one mapping change per cycle; regeneration is ~4 s)

1. Implement **one** mapping change in `data/pipeline.py`, bump
   `MAPPING_VERSION`, add a changelog line below.
2. `uv run data/pipeline.py retarget --all --run v1.X-<slug>`
3. `uv run data/pipeline.py metrics --diff <prev-run> --emotions`
   -- check the flag deltas, regressed-clip list, and that no emotion's
   probe R^2 collapsed.
4. `uv run data/scripts/curate.py panel` then open `/panel` -- judge the
   variant on the 10-clip panel against previous columns **before**
   paying for a full GIF re-render. If it reads worse, revert.
5. `uv run data/scripts/curate.py render` (sha1 cache: only clips whose
   qpos changed re-render), then `serve` and review the queue:
   re-queued fix_mapping clips first, then flagged-unreviewed by
   severity, then an unbiased sample.
6. While reviewing, classify every bad clip: mapping bug ->
   `fix_mapping` (+ note); untranslatable source -> `drop` + a sticky
   tag (`source-static`, `too-short`, `source-degenerate`,
   `untranslatable`). Sticky drops never return to the queue; everything
   else re-queues after the next version bump.
7. Cluster the fix_mapping notes into the next cycle's hypothesis.

When the queue is clean: `uv run data/pipeline.py export` (drop
`--include-unreviewed` once review is complete).

## Cuteness hypothesis backlog (A/B on the panel, one per version)

1. ~~**Ease-in/out**~~ -- DONE in 1.1 (`ease_track`: accel-limited
   onsets, harder braking, exact landing).
2. **Overshoot / follow-through** -- lightly underdamped 2nd-order
   filter (zeta ~0.6-0.8) on J4/J5 so the head settles with a bounce.
3. **Anticipation** -- 2-3 frame counter-dip on J2/J3/J5 before large
   fast excursions.
4. **Head-tilt secondary motion** -- couple J4 to base-yaw velocity
   (J4 sits dead at HOME4 in most clips).
5. **Squash-and-stretch** -- nonlinear gamma on the crouch signal to
   exaggerate posture extremes.
6. **Gaze leading** -- J1/J5 lead J2/J3 by a frame or two.
7. **Breathing / idle micro-motion** -- open question: probably better
   applied at runtime on the robot than baked into training data (the
   CVAE would learn the synthetic sinusoid as content). Decide via
   panel A/B.

## Changelog

- **1.4** (v1.4): head-tilt secondary motion (backlog item 4). J4 banks
  into turns: `tilt = clip(K_TILT * lowpass(d(yaw)/dt, 1.5 Hz), +-0.30)`
  with K_TILT 0.15. J4 was pinned at HOME4 in 100% of v1.3 frames
  (Cozmo's face-roll channel barely fires); now 0.49 rad p1-p99 in the
  dataset. Correlations/emotions unchanged (rho +0.90, jerk 5059).
  Curation re-audit: 2 LOWCORR threshold-noise regressions (r_lift
  0.30->0.29), no real ones; J4 visually validated on the 12 max-yaw
  keeps; laser_drive_01/pounce_drive_01/pounce_lookloop_01 re-examined
  for rescue (tilt gave them 0.23 rad range) -- still inexpressive,
  stayed dropped.

- **1.0** (v1.0-baseline): initial feature-space mapping, re-baselined
  under the uv venv (scipy 1.18.0). 926 clips; 590 carry at least one
  metric flag (RATE 419, LOWCORR 167, SHORT 154, TINY 50, SAT 40,
  STATIC 39, DEGENERATE 2). Emotion preservation: Spearman rho +0.91
  (speed) / +0.86 (yaw) across emotions; probe R^2 lamp == source
  (mean +0.016 both sides).
- **1.3** (v1.3): Pixar-calm pass ("cuteness and calmness"). RATE_CAP
  2.5 -> 1.8 rad/s (103 deg/s), ACCEL_CAP 30 -> 15, FILT_HZ 4 -> 2.5;
  light: CLOSE_W 7 -> 11, new 1 Hz lowpass (S-fades), SLEW 2.0 -> 0.8
  (full fade ~1.1 s). Jerk 11554 -> 4972 (-84% cumulative vs 1.0);
  SAT 23, DYN 17. Fidelity: r_head/yaw +0.92, STATIC +4; emotion
  Spearman held (+0.91/+0.87) but lamp probe R2 slipping (+0.009 vs
  source +0.016) -- 1.8 rad/s is close to the calm/affectless floor;
  don't lower the cap further without checking the emotions report.
  Sub-second twitch clips (e.g. explorer_huh_01, 0.46 s) are content
  problems, not filter problems: already excluded from export (T<30),
  sticky-drop them in review.
- **1.2** (v1.2): "make everything smoother" pass. RATE_CAP 4.0 -> 2.5
  rad/s (143 deg/s), ACCEL_CAP 60 -> 30 (83 ms ease-in), FILT_HZ
  6 -> 4 Hz. Corpus jerk mean 23949 -> 11554 (-52% vs 1.1, -62% vs
  1.0); DYN 31 -> 26, SAT 39 -> 32. Fidelity cost, accepted by design:
  r_head 0.96 -> 0.94, r_yaw 0.97 -> 0.94, LOWCORR 169 -> 192, RATE
  521 (more time cruising at the lower cap -- expected, eased);
  emotions Spearman 0.90, probe R2 lamp +0.013 vs source +0.016.
  Fast-pop clips (explorer_huh r_head 0.49) inherently can't pop at
  this speed -- review candidates. eval fix: rate_cap_frac now reads
  RATE_CAP from each run's stored constants.
- **1.1** (v1.1): style pass -- motion easing + light calming.
  `ease_track` (ACCEL_CAP 60, BRAKE_MULT 2) replaces the hard rate
  limiter: S-curve onsets, no overshoot, bitwise identity for sub-cap
  motion; `lowpass` padlen fix ends the T<15 Butterworth bypass (TINY
  clips 6x less jerk). `calm_light` (closing 7f, floor 0.15, slew
  2.0/s): FLICKER 242 -> 0, light never hard-steps, lamp never fully
  off. Corpus: jerk mean -21%, flagged 74% -> 64%, correlations held
  (r_head +0.96, r_yaw +0.97), emotions Spearman/probe unchanged.
  5 LOWCORR regressions, all cap-saturation pathologies or 0.30->0.29
  threshold noise (worst real: codelab_cubetap r_lift 0.74->0.20,
  17-frame 87%-at-cap tap -- review candidates). DYN 31 observed both
  runs (pre-existing kp=17.8 servo tracking, not a mapping issue --
  future item).
