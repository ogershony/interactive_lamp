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
from runtime.audio.envelope import resample
from runtime.eval import metrics

USER_HOLD_S = 2.5      # user subtitle lingers this long if nothing follows
MUX_SR = 44100         # rate the replay's audio is encoded at (see below)


def affect_chip(e):
    """The one-line summary of what conditioned this segment's motion,
    e.g. `sorrow .70 · understanding .30 — cfg 2.8`. Drawn under the
    lamp's subtitle so the affect can be read against the gesture it
    produced instead of grepped for afterwards."""
    affect = e.get("affect") or {}
    if not affect:
        return None
    parts = " · ".join(f"{k} {v:.2f}".replace("0.", ".")
                       for k, v in list(affect.items())[:3])
    cfg = e.get("cfg")
    return parts if cfg is None else f"{parts}  —  cfg {cfg:.1f}"


def _frame_at(events):
    """A t -> frame mapping fitted from events carrying both, used to
    place things the log only timestamped. Falls back to the nominal
    frame rate when a session has no such event."""
    pts = [(e["t"], e["frame"]) for e in events
           if e.get("frame") is not None and "t" in e]
    if len(pts) < 2:
        return lambda t: int(round(t / C.DT))
    t = np.array([p[0] for p in pts])
    f = np.array([p[1] for p in pts], float)
    a, b = np.polyfit(t, f, 1)
    return lambda x: int(round(a * x + b))


def subtitle_track(events):
    """[(start_frame, end_frame, speaker, text, chip)] from the session
    log. `chip` is the affect annotation, None for user turns."""
    subs = []
    at = _frame_at(events)
    onset = None
    for e in events:
        if e["kind"] == "speech_onset":
            onset = e.get("t")
        elif e["kind"] == "user_turn" and e.get("text"):
            # Sessions recorded before `user_turn.frame` meant "when the
            # user spoke" stamped it with the frame ASR *finished* on,
            # putting the subtitle seconds behind its own audio. The
            # onset was always logged, so recover the real start and take
            # whichever is earlier -- a no-op on sessions recorded since.
            start = e["frame"]
            if onset is not None:
                start = min(start, at(onset))
            subs.append([start, None, "you", e["text"], None])
            onset = None
        elif e["kind"] == "segment_planned" and e.get("text"):
            start = e["start_frame"]
            subs.append([start, start + int(round(e["seconds"] / C.DT)),
                         "lamp", e["text"], affect_chip(e)])
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
    chip_font = _font(10 * scale)
    imgs = []
    for i in range(len(cmd)):
        im = Image.fromarray(r.frame(cmd[i, :5], float(cmd[i, 5])))
        if scale != 1:
            im = im.resize((W, H), Image.LANCZOS)
        active = [s for s in subs if s[0] <= fidx[i] < s[1]]
        if active:
            draw = ImageDraw.Draw(im)
            speaker, text, chip = active[-1][2], active[-1][3], active[-1][4]
            label = text if speaker == "lamp" else f"[you]  {text}"
            fill = (255, 255, 255) if speaker == "lamp" \
                else (255, 225, 120)
            lines = _wrap(draw, label, font, W - 30 * scale)
            lh = int(font.size * 1.35)
            ch = int(chip_font.size * 1.35) if chip else 0
            y = H - 8 * scale - lh * len(lines) - ch
            for line in lines:
                x = (W - draw.textlength(line, font=font)) / 2
                draw.text((x, y), line, font=font, fill=fill,
                          stroke_width=scale, stroke_fill=(0, 0, 0))
                y += lh
            if chip:
                x = (W - draw.textlength(chip, font=chip_font)) / 2
                draw.text((x, y), chip, font=chip_font, fill=(150, 200, 235),
                          stroke_width=scale, stroke_fill=(0, 0, 0))
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
        kw = {}
        wav = _audio_slice(session_dir, fidx, len(cmd))
        if wav is not None:
            # ffmpeg muxes the track straight in; imageio's writer forwards
            # both kwargs to imageio_ffmpeg.write_frames
            kw = dict(audio_path=str(wav), audio_codec="aac")
        try:
            imageio.mimwrite(out, imgs, fps=C.FPS, codec="libx264",
                             quality=8, pixelformat="yuv420p",
                             macro_block_size=1, **kw)
        finally:
            if wav is not None:
                wav.unlink(missing_ok=True)
    return out, len(imgs)


def _audio_slice(session_dir, fidx, n_frames):
    """The session's audio.wav trimmed to the rendered frame span, as a
    temp wav for ffmpeg. Returns None when the session has no audio (any
    session recorded before audio.wav existed, or a motion-only run), in
    which case the render is silent exactly as it used to be."""
    src = pathlib.Path(session_dir) / "audio.wav"
    if not src.exists():
        return None
    import tempfile
    import wave
    with wave.open(str(src), "rb") as w:
        sr, pcm = w.getframerate(), np.frombuffer(
            w.readframes(w.getnframes()), np.int16)
    hop = sr / C.FPS
    a = int(round(int(fidx[0]) * hop))
    b = a + int(round(n_frames * hop))
    clip = pcm[a:b]
    if len(clip) < b - a:                       # audio ended early: pad
        clip = np.concatenate([clip, np.zeros(b - a - len(clip), np.int16)])
    if not clip.any():
        return None                             # silent: don't bother
    # Upsample before handing it to the encoder. The session track is
    # 16 kHz because that is what the duplex stream runs at, but low-rate
    # mono AAC is where players get picky -- some desktop and browser
    # players show an audio stream and play nothing. 44.1 kHz costs a
    # couple of hundred KB and removes the question.
    if sr < MUX_SR:
        clip = resample(clip, sr, MUX_SR)
        sr = MUX_SR
    fh = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(fh, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(clip.tobytes())
    fh.close()
    return pathlib.Path(fh.name)


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
