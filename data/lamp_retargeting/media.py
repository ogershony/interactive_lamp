"""
Media generation: side-by-side Cozmo|lamp GIFs, the calibrated keypose
sheet, and the per-emotion preview picks. Everything here needs a
working GL context (mujoco offscreen render); nothing else does.
"""

import math

import numpy as np

from config import (COMP_GAIN, CROUCH23, GIF_EMOTIONS, HOME4, HOME_PITCH,
                    LEAN23, LIMIT_MARGIN, NPZ_IN, TALL23)
from lamp_model import LampRenderer
from labels import top_emotions


def pick_gif_samples(labels):
    picks = []
    for emo in GIF_EMOTIONS:
        cand = sorted(labels, key=lambda s: (-float(labels[s][emo]), s))[0]
        if cand not in picks:
            picks.append(cand)
    return picks


def _font(size=15):
    from PIL import ImageFont
    try:
        import matplotlib.font_manager as fm
        path = fm.findfont("DejaVu Sans")
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def render_gif(stem, arrays, labels, lamp, out_path):
    from PIL import Image, ImageDraw
    from cozmo_model import ClipRenderer
    z = np.load(NPZ_IN / f"{stem}.npz")
    cozmo = ClipRenderer(420, 300)
    lampr = LampRenderer(lamp)
    q = arrays["qpos"]
    light = arrays["light01"]
    row = labels.get(stem, {})
    emo = top_emotions(row) if row else ""
    font = _font(15)
    small = _font(12)
    frames = []
    T = len(q)
    for i in range(T):
        canvas = Image.new("RGB", (840, 340), (16, 16, 20))
        canvas.paste(Image.fromarray(cozmo.frame(z, i)), (0, 40))
        canvas.paste(Image.fromarray(lampr.frame(q[i], float(light[i]))),
                     (420, 40))
        dr = ImageDraw.Draw(canvas)
        dr.text((10, 4), stem, font=font, fill=(240, 240, 240))
        dr.text((10, 22), emo, font=small, fill=(170, 170, 180))
        dr.text((770, 4), f"t={z['t'][i]:5.2f}s", font=small,
                fill=(170, 170, 180))
        dr.text((430, 22), "cozmo (source)", font=small, fill=(120, 200, 220))
        dr.text((826, 22), "lamp", font=small, fill=(250, 210, 120),
                anchor="ra")
        frames.append(canvas)
    cozmo.close()
    lampr.close()
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=33, loop=0, optimize=False)
    print(f"wrote {out_path}")


def render_keypose_sheet(lamp, out_path):
    from PIL import Image, ImageDraw
    lampr = LampRenderer(lamp)
    lvl = lamp.gaze_level_q5
    a2, a3 = lamp.pitch_coef

    def pose(dq2, dq3, roll=0.0, pitch=0.0):
        q5 = lvl + HOME_PITCH + pitch - COMP_GAIN * (dq2 * a2 + dq3 * a3)
        return np.clip(np.array([0, dq2, dq3, HOME4 + roll, q5]),
                       lamp.lo + LIMIT_MARGIN, lamp.hi - LIMIT_MARGIN)

    poses = [
        ("HOME", pose(0, 0)),
        ("CROUCH (head down)", pose(*CROUCH23 * 1.0)),
        ("TALL (lift raised)", pose(*TALL23 * 1.0)),
        ("LEAN FWD (drive+)", pose(*LEAN23 * 1.0)),
        ("RECOIL (drive-)", pose(*LEAN23 * -1.0)),
        ("LOOK UP (+45deg)", pose(0, 0, pitch=math.radians(45))),
        ("LOOK DOWN (-25deg)", pose(0, 0, pitch=math.radians(-25))),
        ("HEAD TILT (face roll)", pose(0, 0, roll=0.45)),
    ]
    font = _font(14)
    tiles = []
    for name, q in poses:
        img = Image.fromarray(lampr.frame(q, 0.8))
        ImageDraw.Draw(img).text((8, 6), name, font=font, fill=(255, 255, 0))
        tiles.append(img)
    lampr.close()
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 420, rows * 300))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * 420, (i // cols) * 300))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"wrote {out_path}")
