#!/usr/bin/env python3
"""
Convert the fm-v1 checkpoint to ONNX so the browser can sample from it.

One ORT run = one guided Euler step. The wrapper below folds
classifier-free guidance into the graph (batch the conditional and the
learned-null branch, combine with the cfg weight), so `docs/js/generate.js`
is a plain loop over `steps` runs with no model surgery on the JS side.

The padding mask is built inside the graph from x's shape: sampling always
asks for exactly T real frames, so the mask is all-ones by construction and
does not need to cross the language boundary.

Everything else the browser needs to turn samples back into motion --
per-channel normalization stats, the train-data envelope, the per-affect
duration table, the affect order -- goes to model_meta.json, read straight
out of the checkpoint so the two can never drift.

    uv run --group web scripts/export_onnx.py

Writes docs/assets/fm-v1.onnx, docs/assets/model_meta.json and the
projection fixtures used by docs/tests/project.test.mjs.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn

import _webpaths  # noqa: F401  (sys.path shim)

from dataset import DT, EMOTIONS, N_CHANNELS, T_MAX, unit
from sample import denormalize, generate, load_model, project

CKPT = _webpaths.ROOT / "motion_generator" / "runs" / "fm-v1" / "ckpt_best.pt"


class CfgDenoiser(nn.Module):
    """(x, t, label, log_dur, cfg) -> guided velocity field.

    Runs the conditional and unconditional branches as one batch-2 forward
    and returns v_u + cfg * (v_c - v_u), matching sample.generate()'s
    `v_at`. `drop` is a constant buffer so the null branch is selected
    inside the graph rather than by a second call.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("drop", torch.tensor([False, True]))

    def forward(self, x, t, label, log_dur, cfg):
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([t, t], dim=0)
        l2 = torch.cat([label, label], dim=0)
        d2 = torch.cat([log_dur, log_dur], dim=0)
        mask = torch.ones(x2.shape[0], x2.shape[1],
                          dtype=torch.bool, device=x.device)
        v = self.net(x2, mask, t2, l2, d2, drop=self.drop)
        v_c, v_u = v[0:1], v[1:2]
        return v_u + cfg * (v_c - v_u)


def example_inputs(T=90):
    return (
        torch.randn(1, T, N_CHANNELS),
        torch.tensor([0.3]),
        torch.from_numpy(unit(np.eye(len(EMOTIONS))[6])).float()[None],
        torch.tensor([float(np.log(T * DT))]),
        torch.tensor([2.5]),
    )


def export(wrapper, path, dynamic=True):
    """Try a T-dynamic graph first; fall back to a fixed T_MAX graph.

    Dynamic T is worth the attempt: typical clips are 40-95 frames, so a
    graph padded to T_MAX=240 would do 3-6x the work on the WASM backend.
    """
    args = example_inputs()
    names = ["x", "t", "label", "log_dur", "cfg"]
    if dynamic:
        try:
            batch = torch.export.Dim("T", min=8, max=T_MAX)
            torch.onnx.export(
                wrapper, args, str(path), input_names=names,
                output_names=["v"], dynamo=True,
                dynamic_shapes={"x": {1: batch}, "t": None, "label": None,
                                "log_dur": None, "cfg": None})
            return "dynamic"
        except Exception as e:                       # noqa: BLE001
            print(f"  dynamic export failed ({type(e).__name__}: {e});"
                  " falling back to fixed T")
    args = example_inputs(T_MAX)
    torch.onnx.export(wrapper, args, str(path), input_names=names,
                      output_names=["v"], dynamo=True)
    return "fixed"


def inline_weights(path):
    """Fold the .data sidecar back into the .onnx file.

    The dynamo exporter externalizes initializers by default, which would
    make the browser fetch two files and configure ORT's externalData
    mapping. One self-contained ~10 MB file is simpler to serve and to
    cache.
    """
    import onnx
    sidecar = path.with_suffix(path.suffix + ".data")
    if not sidecar.exists():
        return
    model = onnx.load(str(path))          # pulls the sidecar in
    onnx.save(model, str(path), save_as_external_data=False)
    sidecar.unlink()


def ort_generate(sess, label, T, cfg_w, steps, seed):
    """Mirror of sample.generate(), but stepping the ONNX graph.

    Uses torch's RNG for the initial noise so a given seed produces the
    same x0 as the reference implementation and the two can be compared
    sample-for-sample.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn((1, T, N_CHANNELS), generator=g).numpy().astype(np.float32)
    lab = label.astype(np.float32)[None]
    ld = np.array([np.log(T * DT)], np.float32)
    cw = np.array([cfg_w], np.float32)
    dt = 1.0 / steps
    for k in range(steps):
        v = sess.run(["v"], {
            "x": x, "t": np.array([k * dt], np.float32),
            "label": lab, "log_dur": ld, "cfg": cw})[0]
        x = x + dt * v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, default=CKPT)
    ap.add_argument("--out", type=pathlib.Path, default=_webpaths.WEB_ASSETS)
    ap.add_argument("--cases", type=int, default=12)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, ck = load_model(args.ckpt, "cpu", use_ema=True)
    n_affect = ck["config"].get("n_affect", len(EMOTIONS))
    assert n_affect == len(EMOTIONS), \
        f"checkpoint has {n_affect} affects, taxonomy has {len(EMOTIONS)}"
    wrapper = CfgDenoiser(model).eval()

    onnx_path = args.out / "fm-v1.onnx"
    kind = export(wrapper, onnx_path)
    inline_weights(onnx_path)
    print(f"fm-v1.onnx: {kind} T, {onnx_path.stat().st_size / 1e6:.2f} MB")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])

    # ---- parity: single velocity evaluation -----------------------------
    rng = np.random.default_rng(0)
    worst_v, scale_v = 0.0, 0.0
    for _ in range(args.cases):
        T = int(rng.integers(20, 160))
        lab = unit(rng.random(len(EMOTIONS)) ** 3)
        x = rng.standard_normal((1, T, N_CHANNELS)).astype(np.float32)
        t = np.array([rng.random()], np.float32)
        ld = np.array([np.log(T * DT)], np.float32)
        cw = np.array([rng.uniform(0.0, 3.5)], np.float32)
        got = sess.run(["v"], {"x": x, "t": t, "label": lab[None],
                               "log_dur": ld, "cfg": cw})[0]
        with torch.no_grad():
            want = wrapper(torch.from_numpy(x), torch.from_numpy(t),
                           torch.from_numpy(lab[None]), torch.from_numpy(ld),
                           torch.from_numpy(cw)).numpy()
        worst_v = max(worst_v, float(np.abs(got - want).max()))
        scale_v = max(scale_v, float(np.abs(want).max()))
    # fp32 op ordering differs between backends, so compare relative to the
    # size of the field itself rather than against a bare absolute epsilon
    rel_v = worst_v / max(scale_v, 1e-9)
    print(f"velocity parity: max |onnx - torch| = {worst_v:.2e} "
          f"({rel_v:.1e} of peak |v| = {scale_v:.2f})")
    assert rel_v < 1e-4, f"velocity parity {rel_v} too loose"

    # ---- parity: a whole 10-step sample ---------------------------------
    worst_s = 0.0
    for i in range(4):
        T, cfg_w, seed = 90, 2.5, 1000 + i
        lab = unit(np.eye(len(EMOTIONS))[i * 2])
        xo = ort_generate(sess, lab, T, cfg_w, 10, seed)
        xt = generate(model, lab, T, n=1, cfg_w=cfg_w, steps=10,
                      device="cpu", seed=seed)
        rms = float(np.sqrt(((xo - xt) ** 2).mean()))
        worst_s = max(worst_s, rms)
    print(f"sample parity:   max RMS over 4 clips = {worst_s:.2e}")
    assert worst_s < 1e-3, f"sample parity {worst_s} too loose"

    # ---- metadata the browser needs -------------------------------------
    stats = ck["norm_stats"]
    meta = {
        "emotions": EMOTIONS,
        "n_channels": N_CHANNELS,
        "t_max": T_MAX,
        "t_min": 8,
        "dt": DT,
        "steps": 10,
        "cfg_default": 2.5,
        "cfg_min": 0.0,
        "cfg_max": 3.5,
        "graph": kind,
        "norm_stats": {k: np.asarray(v).tolist()
                       for k, v in stats.items()},
        "duration_table": {
            k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                for kk, vv in v.items()}
            if isinstance(v, dict) else v
            for k, v in ck["duration_table"].items()},
        "train_step": int(ck["step"]),
        "best_val": float(ck["best_val"]),
        "params": sum(p.numel() for p in model.parameters()),
    }
    (args.out / "model_meta.json").write_text(json.dumps(meta))
    print(f"model_meta.json: {meta['params']} params, "
          f"step {meta['train_step']}, val {meta['best_val']:.4f}")

    # ---- fallback clips: one per affect, so the playground works before
    # (or without) the 10 MB model download -------------------------------
    table = ck["duration_table"]["seconds"]
    fallback = []
    for i, emo in enumerate(EMOTIONS):
        lab = unit(np.eye(len(EMOTIONS))[i])
        qs = table[emo]
        T = int(np.clip(round(qs[len(qs) // 2] / DT), 8, T_MAX))
        xn = generate(model, lab, T, n=1, cfg_w=2.5, steps=10,
                      device="cpu", seed=100 + i)
        x = project(denormalize(xn, stats)[0])
        fallback.append({
            "affect": emo,
            "cfg": 2.5,
            "seed": 100 + i,
            "T": int(T),
            "qpos": np.round(x[:, :5], 4).tolist(),
            "light": np.round(x[:, 5], 3).tolist(),
            "rgb": np.round(x[:, 6:9] * 255).astype(int).tolist(),
        })
    cdir = args.out / "clips"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "fallback.json").write_text(
        json.dumps({"note": "sampled offline from fm-v1 at cfg 2.5",
                    "clips": fallback}, separators=(",", ":")))
    print(f"fallback clips: {len(fallback)} "
          f"({(cdir / 'fallback.json').stat().st_size / 1e3:.0f} kB)")

    # ---- fixtures for the JS projection port ----------------------------
    fx = []
    for i in range(6):
        lab = unit(np.eye(len(EMOTIONS))[i])
        T = 70 + 13 * i
        xn = generate(model, lab, T, n=1, cfg_w=3.0, steps=10,
                      device="cpu", seed=7 + i)
        raw = denormalize(xn, stats)[0]
        fx.append({
            "affect": EMOTIONS[i],
            "raw": np.round(raw, 6).tolist(),
            "projected": np.round(project(raw), 6).tolist(),
        })
    tdir = _webpaths.DOCS / "tests" / "fixtures"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "project_cases.json").write_text(
        json.dumps({"dt": DT, "cases": fx}, separators=(",", ":")))
    print(f"fixtures: {len(fx)} projection cases")


if __name__ == "__main__":
    main()
