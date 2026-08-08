#!/usr/bin/env python3
"""
Render a recorded session (plan section 8: eval/replay.py): the
commanded 30 Hz stream through the MuJoCo lamp renderer, with the
conversation burned in as subtitles -- user turns and the lamp's spoken
segments, timed from the session log's frame annotations.

    uv run runtime/eval/replay.py runtime/sessions/<ts> [--out lamp.mp4]

Output is .mp4 (imageio-ffmpeg) or .gif (--gif / no ffmpeg). Needs
mujoco with a working GL context (the demo box, not the Pi).
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

import runtime.config as C
from runtime.eval import metrics

USER_HOLD_S = 2.5      # user subtitle lingers this long if nothing follows


def subtitle_track(events):
    """[(start_frame, end_frame, speaker, text)] from the session log."""
    subs = []
    for e in events:
        if e["kind"] == "user_turn" and e.get("text"):
            subs.append([e["frame"], None, "you", e["text"]])
        elif e["kind"] == "segment_planned" and e.get("text"):
            start = e["start_frame"]
            subs.append([start, start + int(round(e["seconds"] / C.DT)),
                         "lamp", e["text"]])
    subs.sort(key=lambda s: s[0])
    for i, s in enumerate(subs):
        if s[1] is None:            # user turn: until the next subtitle
            nxt = next((n[0] for n in subs[i + 1:]), None)
            s[1] = nxt if nxt is not None \
                else s[0] + int(USER_HOLD_S / C.DT)
    return [tuple(s) for s in subs]


def _font(size):
    from PIL import ImageFont
    for path in ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_session(session_dir, out=None, scale=2, gif=False):
    from PIL import Image, ImageDraw

    from pipeline import Lamp, LampRenderer    # mujoco; via _paths shim

    events, fr = metrics.load_session(session_dir)
    assert fr is not None, f"no commanded.npz in {session_dir}"
    cmd, fidx = fr["cmd"], fr["frame"]
    subs = subtitle_track(events)

    lamp = Lamp()
    r = LampRenderer(lamp)
    W, H = LampRenderer.W * scale, LampRenderer.H * scale
    font = _font(14 * scale)
    imgs = []
    for i in range(len(cmd)):
        im = Image.fromarray(r.frame(cmd[i, :5], float(cmd[i, 5])))
        if scale != 1:
            im = im.resize((W, H), Image.LANCZOS)
        active = [s for s in subs if s[0] <= fidx[i] < s[1]]
        if active:
            draw = ImageDraw.Draw(im)
            speaker, text = active[-1][2], active[-1][3]
            label = text if speaker == "lamp" else f"[you]  {text}"
            fill = (255, 255, 255) if speaker == "lamp" \
                else (255, 225, 120)
            lines = _wrap(draw, label, font, W - 30 * scale)
            lh = int(font.size * 1.35)
            y = H - 8 * scale - lh * len(lines)
            for line in lines:
                x = (W - draw.textlength(line, font=font)) / 2
                draw.text((x, y), line, font=font, fill=fill,
                          stroke_width=scale, stroke_fill=(0, 0, 0))
                y += lh
        imgs.append(np.asarray(im))
    r.close()

    session_dir = pathlib.Path(session_dir)
    if out is None:
        out = session_dir / ("replay.gif" if gif else "replay.mp4")
    out = pathlib.Path(out)
    if gif or out.suffix == ".gif":
        frames = [Image.fromarray(a) for a in imgs]
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 * C.DT), loop=0, optimize=True)
    else:
        import imageio.v2 as imageio
        imageio.mimwrite(out, imgs, fps=C.FPS,
                         codec="libx264", quality=8,
                         pixelformat="yuv420p")
    return out, len(imgs)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("session_dir")
    p.add_argument("--out", default=None)
    p.add_argument("--scale", type=int, default=2,
                   help="render scale over the native 420x300")
    p.add_argument("--gif", action="store_true")
    args = p.parse_args()
    out, n = render_session(args.session_dir, args.out, args.scale,
                            args.gif)
    print(f"wrote {out}  ({n} frames, {n * C.DT:.1f} s)")


if __name__ == "__main__":
    main()
