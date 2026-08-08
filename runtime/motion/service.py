#!/usr/bin/env python3
"""
The motion generation service: MotionEngine behind plain HTTP, meant
to run on the GPU box while the Pi's runtime calls it over the LAN
(~20-50 ms/clip on an RTX 2080 Ti at 10 Euler steps).

    uv run runtime/motion/service.py --device cuda --host 0.0.0.0

    POST /generate   {"affect": [11 floats], "seconds": 2.4|null,
                      "cfg": 2.5, "seed": 123|null, "steps": 10|null,
                      "tag": "speak:0"}
                  -> npz bytes, key "x" = (T, 9) float32, physical
                     units, duration-clamped and projected server-side
                     (headers X-Gen-Ms, X-Seconds)
    GET  /health  -> {"emotions": [...], "ckpt": ..., "step": ...,
                      "device": ..., "durations": {...}}

Affect is validated and renormalized to unit L2 at this boundary (400
on garbage); a lock serializes engine calls (one GPU, tiny model).
stdlib only -- no web framework. Unauthenticated LAN HTTP by design:
put it on the robot's network, not the internet.
"""

import argparse
import io
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

import runtime.config as C
from runtime.types import MotionRequest


def make_server(engine, host="127.0.0.1", port=8031):
    """ThreadingHTTPServer wrapping any `.clip(MotionRequest, steps=)`
    engine (the real MotionEngine in production, fakes in tests)."""
    lock = threading.Lock()
    ck = getattr(engine, "ck", {})
    health = {
        "emotions": list(C.EMOTIONS),
        "n_affect": len(C.EMOTIONS),
        "ckpt": str(ck.get("config", {}).get("ckpt", "")) or None,
        "step": ck.get("step"),
        "device": getattr(engine, "device", None),
        "durations": ck.get("duration_table", {}).get("seconds", {}),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):     # quiet; errors still raise
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, health)
            else:
                self._json(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/generate":
                self._json(404, {"error": "unknown path"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                v = np.asarray(req["affect"], np.float64)
                assert v.shape == (len(C.EMOTIONS),) and \
                    np.isfinite(v).all() and (v >= 0).all() and v.sum() > 0
                v = (v / np.linalg.norm(v)).astype(np.float32)
                cfg = float(np.clip(req.get("cfg", 2.5),
                                    C.CFG_MIN, C.CFG_MAX))
                seconds = req.get("seconds")
                seconds = None if seconds is None else float(seconds)
                seed = req.get("seed")
                seed = None if seed is None else int(seed)
                steps = req.get("steps")
                steps = None if steps is None else int(steps)
                tag = str(req.get("tag", ""))
            except Exception as e:  # noqa: BLE001 -- trust boundary
                self._json(400, {"error": f"bad request: {e}"})
                return
            try:
                import time
                t0 = time.perf_counter()
                with lock:
                    x = engine.clip(MotionRequest(
                        affect=v, cfg=cfg, seconds=seconds, tag=tag,
                        seed=seed), steps=steps)
                gen_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:  # noqa: BLE001 -- surface to client
                self._json(500, {"error": f"generation failed: {e}"})
                return
            buf = io.BytesIO()
            np.savez_compressed(buf, x=np.asarray(x, np.float32))
            body = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Gen-Ms", f"{gen_ms:.0f}")
            self.send_header("X-Seconds", f"{len(x) * C.DT:.3f}")
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ckpt", default=str(C.DEFAULT_CKPT))
    p.add_argument("--device", default="cpu", help="cuda on the GPU box")
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 to serve the LAN")
    p.add_argument("--port", type=int, default=8031)
    args = p.parse_args()

    from runtime.motion.engine import MotionEngine
    engine = MotionEngine(args.ckpt, device=args.device)
    srv = make_server(engine, args.host, args.port)
    print(f"motion service on {args.host}:{args.port}  "
          f"(ckpt step {engine.ck['step']}, device {args.device}, "
          f"{C.ENGINE_STEPS} steps)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
