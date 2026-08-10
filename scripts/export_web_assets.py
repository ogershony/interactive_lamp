#!/usr/bin/env python3
"""
Export the lamp geometry + kinematics for the browser demo (docs/).

The MJCF is rooted mid-chain at the lower arm with a freejoint (an
onshape-to-robot artifact) and the base hangs *down* the tree, so it is not
directly usable as a web scene graph. `lamp_model.Lamp` already solves this
for rendering by re-anchoring the base to the floor every frame; this script
reuses that and converts the result into the one representation a browser
wants: a **product of exponentials**.

For a serial revolute chain with the base pinned, each joint is exactly a
rotation about a fixed line in the rest-pose world frame, so

    T_k(q) = exp([S_1] q_1) ... exp([S_k] q_k) . T_k(0)

Baking rest-pose world vertices and nesting one pivot per joint (each with a
constant axis in its parent's frame) reproduces that product exactly -- which
is what `docs/js/lamp.js` does. The screw axes are extracted numerically from
MuJoCo rather than read off the XML, so no assumption about tree direction or
joint sign can silently be wrong, and `--check` proves it over random poses.

    uv run scripts/export_web_assets.py

Writes docs/assets/lamp.glb and docs/assets/rig.json.
"""

import argparse
import json
import pathlib

import numpy as np

import _webpaths  # noqa: F401  (sys.path shim; must precede the flat imports)
from _glb import GlbBuilder

import config  # sets MUJOCO_GL before mujoco  # noqa: F401
import mujoco
from config import (BASE_BODY, CAM_AZIMUTH, CAM_DISTANCE, CAM_ELEVATION,
                    CAM_LOOKAT, HEAD_BODY, HOME4, HOME_PITCH, JOINTS,
                    LIGHT_FLOOR, LIGHT_SLEW, RATE_CAP, ACCEL_CAP, BRAKE_MULT,
                    LIMIT_MARGIN, DT)
from lamp_model import Lamp

# Visual shells plus the servo casings and horns. The remaining group-2
# meshes (ge_27, sg_ziji_15, xg_ziji_16, zk_122, pcb_chazuo_92) are internal
# gearbox/PCB parts hidden inside those casings: dropping them takes the scene
# from ~300k to ~129k triangles with no visible difference.
KEEP_MESHES = {
    "lamp_base", "lamp_base_cover",
    "lamparm__base_elbow", "lamparm__elbow_wrist", "lamparm__wrist_head",
    "lamphead", "diffuser",
    "motor_1723_3", "金属舵盘_从动__v2", "金属舵盘_驱动__v2",
}
# the geoms the browser tints with light01, matching LampRenderer's head tint
GLOW_BODY = HEAD_BODY
WARM = [1.0, 0.85, 0.45]


# ---------------------------------------------------------------- rigid math

def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def quat2mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def axis_angle_mat(u, th):
    u = np.asarray(u, float)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def screw_transform(axis, point, th):
    """Rigid transform of rotating by `th` about the world line (point, axis)."""
    R = axis_angle_mat(axis, th)
    return R, point - R @ point


def mat2axis_angle(R):
    """Rotation matrix -> (unit axis, angle in [0, pi])."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(np.pi - th) < 1e-6:            # 180 deg: axis from R + I
        w, V = np.linalg.eigh((R + np.eye(3)) / 2.0)
        return V[:, int(np.argmax(w))], th
    u = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return u / (2.0 * np.sin(th)), th


def body_pose(lamp, body):
    b = lamp.data.body(body)
    return b.xpos.copy(), b.xmat.reshape(3, 3).copy()


# --------------------------------------------------------------- chain + rig

def physical_chain(model):
    """Body names from the floor-anchored base up to the head.

    Walks BASE_BODY and HEAD_BODY to the world, splices them at their common
    ancestor. The base branch is traversed child->parent, so the chain that
    comes out is the physical one even though the MJCF tree is not.
    """
    def to_root(name):
        out, b = [], mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        while b > 0:
            out.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b))
            b = model.body_parentid[b]
        return out

    up_base, up_head = to_root(BASE_BODY), to_root(HEAD_BODY)
    common = next(b for b in up_base if b in up_head)
    # up_base already runs base -> common; up_head runs head -> common
    return up_base[:up_base.index(common) + 1] + up_head[:up_head.index(common)][::-1]


def build_rig(lamp, probe=0.4):
    """Extract each joint's screw axis in the rest-pose world frame."""
    model = lamp.model
    chain = physical_chain(model)
    assert len(chain) == len(JOINTS) + 1, f"unexpected chain {chain}"

    lamp.set_pose(np.zeros(5))
    rest = {b: body_pose(lamp, b) for b in chain}
    distal = chain[-1]
    p0, R0 = rest[distal]

    joints = []
    for i, jname in enumerate(JOINTS):
        q = np.zeros(5)
        q[i] = probe
        lamp.set_pose(q)
        p1, R1 = body_pose(lamp, distal)

        R = R1 @ R0.T
        t = p1 - R @ p0
        u, th = mat2axis_angle(R)
        if th < 1e-6:
            raise SystemExit(f"joint {jname} does not move {distal}")
        # mat2axis_angle returns th in [0, pi] about +u, so u already points
        # the way that makes a positive q a positive rotation
        if not np.isclose(th, abs(probe), atol=1e-6):
            raise SystemExit(f"joint {jname}: angle {th} != probe {abs(probe)}")
        if probe < 0:
            u = -u
        # point on the axis: t = (I - R) a, minimum-norm solution
        a, *_ = np.linalg.lstsq(np.eye(3) - R, t, rcond=None)

        Rc, tc = screw_transform(u, a, probe)
        err = max(np.abs(Rc - R).max(), np.abs((Rc @ p0 + tc) - p1).max())
        if err > 1e-9:
            raise SystemExit(f"joint {jname}: screw reconstruction err {err}")

        joints.append({
            "name": jname,
            "index": i,
            "axis": [float(x) for x in u],
            "point": [float(x) for x in a],
            "lo": float(lamp.lo[i]),
            "hi": float(lamp.hi[i]),
            "segment": chain[i + 1],
        })

    lamp.set_pose(np.zeros(5))
    return chain, joints


def poe(joints, q):
    """T(q) = exp([S_1]q_1) . exp([S_2]q_2) ... , in the rest-pose world frame.

    Space-form product of exponentials: the joint-1 factor is outermost, so
    each new joint composes on the *right* -- which is the same thing as
    nesting its pivot deeper in the scene graph.
    """
    R, t = np.eye(3), np.zeros(3)
    for j, qi in zip(joints, q):
        Ri, ti = screw_transform(j["axis"], np.array(j["point"]), float(qi))
        R, t = R @ Ri, R @ ti + t
    return R, t


def check_fk(lamp, chain, joints, n=200, seed=0, tol=1e-3):
    """Prove the browser's FK equals MuJoCo's over random legal poses."""
    lamp.set_pose(np.zeros(5))
    rest = {b: body_pose(lamp, b) for b in chain}
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(lamp.lo, lamp.hi)
        lamp.set_pose(q)
        for k, body in enumerate(chain):
            # segment k is moved by joints 0..k-1
            R, t = poe(joints[:k], q[:k])
            p0, R0 = rest[body]
            pp, pR = R @ p0 + t, R @ R0
            pa, Ra = body_pose(lamp, body)
            worst = max(worst, float(np.abs(pp - pa).max()),
                        float(np.abs(pR - Ra).max()))
    lamp.set_pose(np.zeros(5))
    return worst


# ------------------------------------------------------------------- meshes

def export_glb(lamp, chain, path):
    model = lamp.model
    lamp.set_pose(np.zeros(5))
    glb = GlbBuilder()
    mats, per_segment, kept, tris = {}, {}, set(), 0

    for seg, body in enumerate(chain):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        bpos, bmat = body_pose(lamp, body)
        names = []
        adr, num = model.body_geomadr[bid], model.body_geomnum[bid]
        for g in range(adr, adr + num):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if model.geom_group[g] != 2:            # 2 = visual, 3 = collision
                continue
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                     model.geom_dataid[g])
            if mesh not in KEEP_MESHES:
                continue
            kept.add(mesh)

            di = model.geom_dataid[g]
            va, vn = model.mesh_vertadr[di], model.mesh_vertnum[di]
            fa, fn = model.mesh_faceadr[di], model.mesh_facenum[di]
            v = model.mesh_vert[va:va + vn].reshape(-1, 3).astype(float)
            nrm = model.mesh_normal[va:va + vn].reshape(-1, 3).astype(float)
            f = model.mesh_face[fa:fa + fn].reshape(-1, 3)

            gR = quat2mat(model.geom_quat[g])
            gp = model.geom_pos[g]
            world = (bmat @ (gR @ v.T + gp[:, None])).T + bpos
            wn = (bmat @ (gR @ nrm.T)).T

            # onshape-to-robot puts the CAD colours on MJCF materials and
            # leaves geom_rgba at the 0.5 grey default, so prefer the material
            mid = int(model.geom_matid[g])
            src = np.array(model.mat_rgba[mid] if mid >= 0
                           else model.geom_rgba[g], dtype=float)
            # MuJoCo treats rgba as display values; glTF baseColorFactor is
            # linear. Writing 0.302 straight through renders the dark grey
            # shells as mid grey, so convert sRGB -> linear here.
            rgba = tuple(np.round(np.concatenate([srgb_to_linear(src[:3]),
                                                  src[3:4]]), 5))
            if rgba not in mats:
                mats[rgba] = glb.material(rgba, f"m{len(mats)}")
            node = f"{body}__{mesh}__{g}"
            glb.add_mesh(node, world, f, mats[rgba], normals=wn)
            names.append(node)
            tris += fn
        per_segment[seg] = names

    missing = KEEP_MESHES - kept
    if missing:
        raise SystemExit(f"KEEP_MESHES never matched: {sorted(missing)}")
    size = glb.write(path)
    return per_segment, tris, size


def glow_nodes(per_segment, chain):
    body = GLOW_BODY
    seg = chain.index(body)
    return per_segment[seg]


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=_webpaths.WEB_ASSETS)
    ap.add_argument("--poses", type=int, default=200,
                    help="random poses for the FK self-check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lamp = Lamp()
    chain, joints = build_rig(lamp)
    print("chain:", " -> ".join(chain))

    worst = check_fk(lamp, chain, joints, n=args.poses)
    print(f"FK self-check: worst error {worst * 1000:.4f} mm "
          f"over {args.poses} random poses")
    if worst > 1e-3:
        raise SystemExit(f"FK mismatch {worst:.6f} exceeds 1 mm")

    per_segment, tris, size = export_glb(lamp, chain, args.out / "lamp.glb")
    print(f"lamp.glb: {tris} triangles, {size / 1e6:.2f} MB")

    lamp.set_pose(np.zeros(5))
    head_rest = body_pose(lamp, HEAD_BODY)[0]
    rig = {
        "chain": chain,
        "joints": joints,
        "segment_nodes": {str(k): v for k, v in per_segment.items()},
        "glow_nodes": glow_nodes(per_segment, chain),
        "glow_warm": WARM,
        "camera": {
            "azimuth": CAM_AZIMUTH, "elevation": CAM_ELEVATION,
            "distance": CAM_DISTANCE, "lookat": list(CAM_LOOKAT),
        },
        "limits": {
            "lo": [float(x) for x in lamp.lo],
            "hi": [float(x) for x in lamp.hi],
        },
        "invariants": {
            "dt": DT, "rate_cap": RATE_CAP, "accel_cap": ACCEL_CAP,
            "brake_mult": BRAKE_MULT, "limit_margin": LIMIT_MARGIN,
            "light_floor": LIGHT_FLOOR, "light_slew": LIGHT_SLEW,
        },
        "home": {"home4": HOME4, "home_pitch": HOME_PITCH,
                 "gaze_level_q5": float(lamp.gaze_level_q5)},
        "head_rest_z": float(head_rest[2]),
        "fk_check_mm": worst * 1000.0,
    }
    (args.out / "rig.json").write_text(json.dumps(rig, indent=1))
    print(f"rig.json: {len(rig['joints'])} joints, "
          f"{sum(len(v) for v in per_segment.values())} mesh nodes")


if __name__ == "__main__":
    main()
