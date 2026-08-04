#!/usr/bin/env python3
"""
Standalone visualizer for the preprocessed Cozmo dataset (NOT a pipeline
stage). Reads ONLY the pipeline outputs animations/npz/ and labels.csv — what you see
is exactly the data retargeting will consume, derived channels included.
animations/json/*.json is never touched here; it is the keyframe ground truth for the
pipeline, not the consumable dataset.

    # one clip -> stacked timeline PNG, one lane per animated channel
    python scripts/visualize.py anim_bored_01

    # one clip -> animated GIF (real-time, 30 fps): rendered procedural
    # face + side-view head/lift schematic + top-down drive path.
    # Prints the clip's YouTube URL for a side-by-side smoke check.
    python scripts/visualize.py anim_bored_01 --gif

    # random labelled clips, PNG+GIF each, with their YouTube URLs
    python scripts/visualize.py --sample 6

    # corpus summary / listing
    python scripts/visualize.py --overview
    python scripts/visualize.py --list 'anim_bored*'

Channels drawn only when the clip animates them (NaN-filled channels and
all-zero body/LED/audio channels are skipped): head angle (deg), lift
height (mm), drive speed (mm/s), turn rate (rad/s), gaze (px), eye
openness (derived), backpack LEDs, audio events.
"""

import argparse
import collections
import fnmatch
import math
import pathlib
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data_preprocessing_pipeline'))

import cozmo_common as CC  # noqa: E402

# --- Pillow 12 compat for pycozmo's face renderer -------------------------
# pycozmo's ProceduralLid.render emits inverted chord/ellipse bounding boxes
# for some face params (e.g. anim_bored_02); Pillow >= 10 raises
# "y1 must be greater than or equal to y0" where old Pillow tolerated it.
# Normalize the bbox before drawing.
from PIL import ImageDraw as _ImageDraw  # noqa: E402


def _norm_bbox(xy):
    p = list(xy)
    (x0, y0), (x1, y1) = p if len(p) == 2 else ((p[0], p[1]), (p[2], p[3]))
    return ((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)))


_orig_chord = _ImageDraw.ImageDraw.chord
_orig_ellipse = _ImageDraw.ImageDraw.ellipse
_ImageDraw.ImageDraw.chord = (
    lambda self, xy, start, end, **kw: _orig_chord(self, _norm_bbox(xy), start, end, **kw))
_ImageDraw.ImageDraw.ellipse = (
    lambda self, xy, **kw: _orig_ellipse(self, _norm_bbox(xy), **kw))
# --------------------------------------------------------------------------

LED_ROWS = ['Left', 'Front', 'Middle', 'Back', 'Right']

THEMES = {
    'light': dict(surface='#fcfcfb', page='#f9f9f7', ink='#0b0b0b', ink2='#52514e',
                  muted='#898781', grid='#e1e0d9', axis='#c3c2b7', off='#e1e0d9',
                  series=('#2a78d6', '#eb6834', '#1baf7a')),
    'dark': dict(surface='#1a1a19', page='#0d0d0d', ink='#ffffff', ink2='#c3c2b7',
                 muted='#898781', grid='#2c2c2a', axis='#383835', off='#2c2c2a',
                 series=('#3987e5', '#d95926', '#199e70')),
}


# ---------------------------------------------------------------- data loading

def npz_names():
    return sorted(p.stem for p in CC.NPZ_DIR.glob('*.npz'))


def load_npz(name):
    return np.load(CC.NPZ_DIR / f'{name}.npz')


def load_labels():
    """clip name -> {'emotions': {e: frac}, 'strong': [e], 'description',
    'youtube_url'} from labels.csv."""
    import csv
    out = {}
    if not CC.LABELS_CSV.exists():
        return out
    for r in csv.DictReader(open(CC.LABELS_CSV, newline='')):
        fr = {e: float(r[e]) for e in CC.EMOTIONS}
        out[r['clip_name']] = {
            'emotions': fr,
            'strong': sorted((e for e, v in fr.items() if v >= 0.5),
                             key=lambda e: -fr[e]),
            'description': r['descriptions'].split(' | ')[0],
            'youtube_url': r['youtube_url'],
        }
    return out


def animated(arr):
    """True if a float channel carries data (not all-NaN, not all-zero)."""
    if arr.dtype.kind != 'f':
        return arr.any()
    return not np.isnan(arr).all() and np.nan_to_num(arr).any() or \
        (not np.isnan(arr).all() and np.unique(np.nan_to_num(arr)).size > 1)


# -------------------------------------------------------------------- chrome

def style(theme):
    t = THEMES[theme]
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'figure.facecolor': t['page'],
        'axes.facecolor': t['surface'],
        'savefig.facecolor': t['page'],
        'text.color': t['ink'],
        'axes.labelcolor': t['ink2'],
        'xtick.color': t['muted'],
        'ytick.color': t['muted'],
        'axes.edgecolor': t['axis'],
    })
    return t


def dress(ax, t, last=False, grid='y'):
    if grid:
        ax.grid(axis=grid, color=t['grid'], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['axis'])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8, labelsize=8)
    if not last:
        ax.tick_params(labelbottom=False)


def legend(ax, t, ncol=3):
    return ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), borderaxespad=0.1,
                     fontsize=8, frameon=False, ncol=ncol, handlelength=1.4,
                     handletextpad=0.5, columnspacing=1.2, labelcolor=t['ink2'])


# --------------------------------------------------------------- lane drawing

def lane_series(ax, t, ts, vs, ylabel, color, unit_fmt='{:g}', zero_line=False):
    """One resampled channel as a continuous 30 Hz line, extremes labelled."""
    ax.plot(ts, vs, color=color, linewidth=1.8, solid_joinstyle='round', zorder=3)
    if zero_line:
        ax.axhline(0, color=t['axis'], linewidth=0.8, zorder=2)
    ax.set_ylabel(ylabel, fontsize=8.5, color=t['ink2'])
    finite = np.asarray(vs)[np.isfinite(vs)]
    if not finite.size:
        return
    lo, hi = float(finite.min()), float(finite.max())
    pad = max(1e-6, (hi - lo) * 0.30) or 1.0
    ax.set_ylim(lo - pad, hi + pad)
    # Direct-label the extremes only — never a number on every point.
    for value in {hi, lo}:
        at = ts[int(np.nanargmax(vs) if value == hi else np.nanargmin(vs))]
        ax.annotate(unit_fmt.format(value), (at, value), textcoords='offset points',
                    xytext=(5, 4 if value == hi else -10),
                    fontsize=7.5, color=t['ink2'])


def lane_pair(ax, t, ts, series, ylabel, zero_line=False):
    """Two related channels sharing a lane, direct-labelled + legend."""
    for (vs, label), color in zip(series, t['series']):
        ax.plot(ts, vs, color=color, linewidth=1.8, label=label, zorder=3)
        j = int(len(ts) * 0.98)
        ax.annotate(label, (ts[-1], vs[j]), textcoords='offset points',
                    xytext=(5, 5 if label in ('x', 'left') else -9),
                    fontsize=8, color=color, annotation_clip=False)
    if zero_line:
        ax.axhline(0, color=t['axis'], linewidth=0.8, zorder=2)
    ax.set_ylabel(ylabel, fontsize=8.5, color=t['ink2'])
    legend(ax, t, ncol=2)


def lane_lights(ax, t, ts, leds, end_s):
    """leds: T x 5 x 3 uint8. Contiguous same-colour runs drawn as one bar."""
    ax.set_ylim(0, len(LED_ROWS))
    ax.set_yticks([i + 0.5 for i in range(len(LED_ROWS))])
    ax.set_yticklabels(list(reversed(LED_ROWS)), fontsize=7.5, color=t['ink2'])
    ax.grid(False)
    dt = ts[1] - ts[0] if len(ts) > 1 else 0.033
    for li in range(len(LED_ROWS)):
        row = leds[:, li]                      # T x 3
        i = 0
        while i < len(row):
            if not row[i].any():
                i += 1
                continue
            j = i
            while j + 1 < len(row) and (row[j + 1] == row[i]).all():
                j += 1
            y = len(LED_ROWS) - 1 - li          # physical order top-to-bottom
            dur = max((j - i + 1) * dt, end_s * 0.004)
            ax.add_patch(Rectangle((ts[i], y + 0.12), dur, 0.76,
                                   facecolor=tuple(row[i] / 255.0),
                                   edgecolor='none', zorder=3))
            i = j + 1
    ax.set_ylabel('backpack LEDs\n(blank = off)', fontsize=8.5, color=t['ink2'])


def lane_audio(ax, t, at, ids, end_s):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    for x in at:
        ax.vlines(x, 0.12, 0.5, color=t['ink2'], linewidth=1.4, zorder=3)
        ax.plot([x], [0.5], 'o', color=t['ink2'], markersize=3.4, zorder=4)
    if len(at) <= 8:
        for n, x in enumerate(at):
            ev = [str(v)[-5:] for v in ids[n] if v >= 0]
            ha = 'left' if x < end_s * 0.03 else 'right' if x > end_s * 0.97 else 'center'
            ax.annotate(','.join(ev), (x, 0.60 if n % 2 else 0.82),
                        ha=ha, va='center', fontsize=7, color=t['muted'])
    ax.set_ylabel('audio\n(%d events)' % len(at), fontsize=8.5, color=t['ink2'])


# ------------------------------------------------------------------- timeline

def plot_timeline(name, labels, theme, out_path, show):
    t = style(theme)
    d = load_npz(name)
    ts = d['t']
    end_s = float(d['duration_s'])

    has_head = animated(d['head_deg'])
    has_lift = animated(d['lift_mm'])
    has_v = d['body_v_mmps'].any()
    has_w = d['body_omega_radps'].any()
    has_face = animated(d['gaze_x']) or animated(d['eye_open_l'])
    has_leds = d['leds_rgb'].any()
    has_audio = len(d['audio_t']) > 0

    lanes = []
    if has_head:
        lanes.append((lambda ax: lane_series(ax, t, ts, d['head_deg'],
                                             'head angle\n(deg)', t['series'][0],
                                             '{:g}°'), 1.0, 'y'))
    if has_lift:
        lanes.append((lambda ax: lane_series(ax, t, ts, d['lift_mm'],
                                             'lift height\n(mm)', t['series'][1],
                                             '{:g} mm'), 1.0, 'y'))
    if has_v:
        lanes.append((lambda ax: lane_series(ax, t, ts, d['body_v_mmps'],
                                             'drive speed\n(mm/s)', t['series'][0],
                                             '{:g}', zero_line=True), 1.0, 'y'))
    if has_w:
        lanes.append((lambda ax: lane_series(ax, t, ts, d['body_omega_radps'],
                                             'turn rate\n(rad/s)', t['series'][1],
                                             '{:.2g}', zero_line=True), 1.0, 'y'))
    if has_face:
        lanes.append((lambda ax: lane_pair(ax, t, ts,
                                           [(d['gaze_x'], 'x'), (d['gaze_y'], 'y')],
                                           'gaze\n(px)', zero_line=True), 1.0, 'y'))
        lanes.append((lambda ax: lane_pair(ax, t, ts,
                                           [(d['eye_open_l'], 'left'),
                                            (d['eye_open_r'], 'right')],
                                           'eye open\n(derived)'), 1.0, 'y'))
    if has_leds:
        lanes.append((lambda ax: lane_lights(ax, t, ts, d['leds_rgb'], end_s),
                      0.95, ''))
    if has_audio:
        lanes.append((lambda ax: lane_audio(ax, t, d['audio_t'], d['audio_ids'],
                                            end_s), 0.55, ''))

    if not lanes:
        print(f'  !! {name}: no animated channels', file=sys.stderr)
        return None

    heights = [h for _, h, _ in lanes]
    fig_h = 1.9 + sum(heights) * 0.95
    fig, axes = plt.subplots(len(lanes), 1, figsize=(11.5, fig_h), sharex=True,
                             gridspec_kw={'height_ratios': heights, 'hspace': 0.42})
    axes = [axes] if len(lanes) == 1 else list(axes)

    for i, ((draw, _, grid), ax) in enumerate(zip(lanes, axes)):
        draw(ax)
        dress(ax, t, last=(i == len(lanes) - 1), grid=grid)
    axes[-1].set_xlabel('time (s)', fontsize=8.5, color=t['ink2'])
    axes[-1].set_xlim(-end_s * 0.01, end_s * 1.01 if end_s else 1)

    lab = labels.get(name, {})
    sub = (f"{d['source_bin']}  ·  {end_s:.2f} s  ·  {len(ts)} frames @ 30.3 Hz")
    if lab.get('strong'):
        sub += '  ·  ' + ', '.join(f"{e} {lab['emotions'][e]:.0%}"
                                   for e in lab['strong'][:4])
    fig.text(0.012, 1 - 0.30 / fig_h, name, ha='left', va='center',
             fontsize=13, color=t['ink'])
    fig.text(0.012, 1 - 0.52 / fig_h, sub, ha='left', va='center',
             fontsize=8.5, color=t['ink2'])
    desc = lab.get('description')
    if desc:
        fig.text(0.012, 1 - 0.72 / fig_h,
                 '“' + desc[:150] + ('…' if len(desc) > 150 else '') + '”',
                 ha='left', va='center', fontsize=8, color=t['muted'], style='italic')

    fig.subplots_adjust(left=0.085, right=0.975, top=1 - 0.95 / fig_h,
                        bottom=0.52 / fig_h)
    return finish(fig, out_path, show)


# ------------------------------------------------------------------ GIF

def _clip_frames(name, labels):
    """Build the animation frames for a clip: MuJoCo 3D replica replay
    (main pane) + rendered procedural face inset + drive-path inset.
    Returns (list of PIL Images, t array)."""
    from PIL import Image, ImageDraw
    CC.import_pycozmo()
    from pycozmo import procedural_face as pf
    import cozmo_model

    d = load_npz(name)
    T = len(d['t'])
    W, H = 720, 340
    lab = labels.get(name, {})
    x_mm, y_mm, yaw = d['x_mm'], d['y_mm'], d['yaw_rad']
    head_deg = np.nan_to_num(d['head_deg'], nan=0.0)
    lift_mm = np.nan_to_num(d['lift_mm'], nan=32.0)
    span = max(np.ptp(x_mm), np.ptp(y_mm), 1.0)

    r3d = cozmo_model.ClipRenderer(width=420, height=300)
    imgs = []
    try:
        for i in range(T):
            im = Image.new('RGB', (W, H), '#101010')
            dr = ImageDraw.Draw(im)

            # -- main pane (right): MuJoCo 3D replica replaying the NPZ
            im.paste(Image.fromarray(r3d.frame(d, i)), (290, 34))
            dr.rectangle([290, 34, 710, 334], outline='#333333')

            # -- face inset (left top): actual procedural face render
            fp = d['face_params'][i]
            if not np.isnan(fp).any():
                params = ([fp[1], fp[2], fp[3], fp[4], fp[0]]
                          + list(fp[5:24]) + list(fp[24:43]))
                face_im = pf.ProceduralFace(params).render().convert('L')
            else:
                face_im = Image.new('L', (128, 64), 0)
            face_rgb = Image.merge('RGB', (face_im.point(lambda p: 0),
                                           face_im, face_im))  # cozmo cyan
            im.paste(face_rgb.resize((256, 128), Image.NEAREST), (14, 40))
            dr.rectangle([14, 40, 270, 168], outline='#333333')
            dr.text((14, 172), f'head {head_deg[i]:+5.1f}°   '
                               f'lift {lift_mm[i]:5.1f} mm', fill='#9a9a9a')

            # -- drive path inset (left bottom, top-down)
            bx, by, bs = 14, 208, 96
            dr.rectangle([bx, by, bx + bs, by + bs], outline='#333333')
            cx, cy = bx + bs / 2, by + bs / 2
            sc = (bs * 0.4) / (span / 2 + 1e-6) if span > 2 else 1.0
            pts = [(cx + (x_mm[j] - x_mm.mean()) * sc,
                    cy - (y_mm[j] - y_mm.mean()) * sc)
                   for j in range(0, i + 1, 2)]
            if len(pts) > 1:
                dr.line(pts, fill='#3987e5', width=2)
            px, py = (cx + (x_mm[i] - x_mm.mean()) * sc,
                      cy - (y_mm[i] - y_mm.mean()) * sc)
            dr.line([px, py, px + 9 * math.cos(yaw[i]), py - 9 * math.sin(yaw[i])],
                    fill='#e0e0e0', width=2)
            dr.ellipse([px - 3, py - 3, px + 3, py + 3], fill='#3987e5')
            dr.text((bx, by + bs + 4), 'drive path (top-down)', fill='#666666')

            # -- header
            dr.text((14, 8), name, fill='#ffffff')
            if lab.get('strong'):
                dr.text((14, 22), ', '.join(lab['strong'][:4]), fill='#9a9a9a')
            dr.text((W - 96, 8), f't = {d["t"][i]:5.2f} s', fill='#9a9a9a')
            imgs.append(im)
    finally:
        r3d.close()
    return imgs, d['t']


def render_gif(name, labels, out_path):
    """Real-time 30 fps GIF of the clip (see _clip_frames)."""
    imgs, _ = _clip_frames(name, labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=33, loop=0, optimize=True)
    return out_path


# ------------------------------------------------------------------- overview

def plot_overview(labels, theme, out_path, show):
    t = style(theme)
    blue = t['series'][0]

    durations = []
    coverage = collections.Counter()
    heads = []
    total_frames = 0
    names = npz_names()
    for name in names:
        d = load_npz(name)
        durations.append(float(d['duration_s']))
        total_frames += len(d['t'])
        if animated(d['head_deg']):
            coverage['head'] += 1
            heads.extend(d['head_deg'][np.isfinite(d['head_deg'])][::3])
        if animated(d['lift_mm']):
            coverage['lift'] += 1
        if d['body_v_mmps'].any() or d['body_omega_radps'].any():
            coverage['body'] += 1
        if animated(d['gaze_x']) or animated(d['eye_open_l']):
            coverage['face'] += 1
        if d['leds_rgb'].any():
            coverage['LEDs'] += 1
        if len(d['audio_t']):
            coverage['audio'] += 1
    durations.sort()
    emotions = collections.Counter(
        em for v in labels.values() for em in v.get('strong', []))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    (ax1, ax2), (ax3, ax4) = axes

    capped = [min(d_, 15) for d_ in durations]
    ax1.hist(capped, bins=30, color=blue, linewidth=0)
    ax1.set_title('clip duration', loc='left', fontsize=10.5, color=t['ink'])
    ax1.set_xlabel('seconds (15 s bin holds the long tail)', fontsize=8.5, color=t['ink2'])
    ax1.set_ylabel('clips', fontsize=8.5, color=t['ink2'])
    dress(ax1, t, last=True)
    med = durations[len(durations) // 2]
    ax1.axvline(med, color=t['ink2'], linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax1.annotate(f'median {med:.1f} s', (med, ax1.get_ylim()[1] * 0.92),
                 xytext=(5, 0), textcoords='offset points', fontsize=8, color=t['ink2'])

    order = [k for k, _ in coverage.most_common()][::-1]
    vals = [coverage[k] / len(names) * 100 for k in order]
    ax2.barh(range(len(order)), vals, color=blue, linewidth=0, height=0.62)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order, fontsize=8)
    ax2.set_title('channel coverage (npz)', loc='left', fontsize=10.5, color=t['ink'])
    ax2.set_xlabel('% of clips animating the channel', fontsize=8.5, color=t['ink2'])
    dress(ax2, t, last=True, grid='x')
    for i, v in enumerate(vals):
        ax2.annotate(f'{v:.0f}%', (v, i), xytext=(4, 0),
                     textcoords='offset points', va='center', fontsize=7.5,
                     color=t['ink2'])
    ax2.set_xlim(0, 112)

    ax3.hist(heads, bins=40, color=blue, linewidth=0)
    ax3.set_title('head angle across all frames', loc='left', fontsize=10.5, color=t['ink'])
    ax3.set_xlabel('degrees', fontsize=8.5, color=t['ink2'])
    ax3.set_ylabel('frames (subsampled)', fontsize=8.5, color=t['ink2'])
    dress(ax3, t, last=True)

    if emotions:
        items = emotions.most_common(16)[::-1]
        ax4.barh(range(len(items)), [c for _, c in items], color=blue,
                 linewidth=0, height=0.62)
        ax4.set_yticks(range(len(items)))
        ax4.set_yticklabels([e for e, _ in items], fontsize=8)
        for i, (_, c) in enumerate(items):
            ax4.annotate(str(c), (c, i), xytext=(4, 0), textcoords='offset points',
                         va='center', fontsize=7.5, color=t['ink2'])
        ax4.set_xlim(0, max(c for _, c in items) * 1.12)
    else:
        ax4.text(0.5, 0.5, 'labels.csv not found', ha='center',
                 fontsize=9, color=t['muted'], transform=ax4.transAxes)
    ax4.set_title('emotions with annotator fraction ≥ 0.5', loc='left',
                  fontsize=10.5, color=t['ink'])
    ax4.set_xlabel('clips', fontsize=8.5, color=t['ink2'])
    dress(ax4, t, last=True, grid='x')

    fig.text(0.012, 0.968, 'Cozmo animation corpus (npz)', ha='left', va='center',
             fontsize=13, color=t['ink'])
    fig.text(0.012, 0.935,
             f'{len(names)} clips  ·  {total_frames:,} frames @ 30.3 Hz  ·  '
             f'{len(labels)} labelled',
             ha='left', va='center', fontsize=8.5, color=t['ink2'])
    fig.tight_layout(rect=(0, 0, 1, 0.915), h_pad=2.6, w_pad=3.0)
    return finish(fig, out_path, show)


def finish(fig, out_path, show):
    if show:
        plt.show()
        plt.close(fig)
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('clip', nargs='?', help='clip name or glob pattern')
    p.add_argument('--gif', action='store_true', help='also render animated GIF')
    p.add_argument('--sample', type=int, metavar='N',
                   help='render N random clips (PNG+GIF)')
    p.add_argument('--list', dest='list_only', nargs='?', const='*', metavar='PATTERN',
                   help='list matching clip names and exit')
    p.add_argument('--overview', action='store_true',
                   help='corpus summary instead of a single clip')
    p.add_argument('--all', action='store_true',
                   help='render every clip matching the pattern')
    p.add_argument('--dark', action='store_true', help='dark-mode palette')
    p.add_argument('--show', action='store_true',
                   help='open an interactive window instead of writing a PNG')
    p.add_argument('--out-dir', default=str(ROOT / 'output' / 'figures'),
                   help='where PNGs go (default: output/figures/); '
                        'GIFs go to output/gifs/')
    p.add_argument('--seed', type=int, default=0, help='seed for --sample')
    args = p.parse_args()

    if not args.show:
        matplotlib.use('Agg')

    names = npz_names()
    if not names:
        sys.exit('ERROR: no animations/npz/; run data_preprocessing_pipeline/run_all.py first')
    labels = load_labels()
    theme = 'dark' if args.dark else 'light'
    out_dir = pathlib.Path(args.out_dir)
    gif_dir = ROOT / 'output' / 'gifs'

    if args.list_only:
        for name in fnmatch.filter(names, args.list_only):
            d = load_npz(name)
            lab = labels.get(name, {})
            mark = ','.join(lab.get('strong', [])) or '-'
            print(f'{name:<62} {float(d["duration_s"]):6.2f}s  {mark}')
        return

    if args.overview:
        out = plot_overview(labels, theme, out_dir / f'overview_{theme}.png', args.show)
        if out:
            print(f'wrote {out}')
        return

    if args.sample:
        rng = random.Random(args.seed)
        matches = rng.sample(names, args.sample)
    else:
        if not args.clip:
            p.error('give a clip name, or --sample / --overview / --list')
        matches = fnmatch.filter(names, args.clip)
        if not matches:
            near = fnmatch.filter(names, f'*{args.clip}*')[:10]
            sys.exit(f'no clip matches {args.clip!r}' +
                     (f'\ndid you mean:\n  ' + '\n  '.join(near) if near else ''))
        if len(matches) > 1 and not args.all:
            print(f'{len(matches)} clips match; pass --all to render them, or pick one:',
                  file=sys.stderr)
            for m in matches[:20]:
                print('  ' + m, file=sys.stderr)
            sys.exit(1)

    for name in matches:
        out = plot_timeline(name, labels, theme,
                            out_dir / f'{name}_{theme}.png', args.show)
        if out:
            print(f'wrote {out}')
        if args.gif or args.sample:
            g = render_gif(name, labels, gif_dir / f'{name}.gif')
            print(f'wrote {g}')
        url = labels.get(name, {}).get('youtube_url')
        if url:
            print(f'  compare against: {url}')


if __name__ == '__main__':
    main()
