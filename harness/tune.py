"""Per-machine kernel tuning sweep.

Measures device memory bandwidth, sweeps threadgroup shapes for every
kernel on a representative workload, prints the table, and writes the
winners to mlx_tomo/tuning_local.json (loaded automatically; delete the
file to fall back to shipped defaults). Run this once per machine —
e.g. after moving from M1 to M5 Max.

Usage: python harness/tune.py [--nvox 512] [--ndet 512] [--views 10]
       [--dry]   (dry: print winners, don't write the file)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

import mlx_tomo
from mlx_tomo import projector as pr
from mlx_tomo import tuning
from gen import default_phantom_mm, ellipsoid_phantom, machine

CAND_3D = [(8, 8, 1), (8, 4, 1), (4, 4, 1), (16, 4, 1), (16, 8, 1),
           (32, 4, 1), (32, 8, 1), (16, 16, 1), (64, 4, 1),
           (32, 4, 2), (16, 4, 4), (8, 8, 4), (8, 8, 2), (16, 8, 2)]
CAND_TEX = [(8, 8), (4, 4), (8, 4), (16, 4), (16, 8), (32, 4), (16, 16)]


def measure_bandwidth():
    """Sustained read+write GB/s via a streaming op (a += 1)."""
    n = 2**28  # 1 GiB float32
    a = mx.zeros((n,), dtype=mx.float32)
    mx.eval(a)
    best = np.inf
    for _ in range(5):
        mx.synchronize()
        t0 = time.perf_counter()
        a = a + 1.0
        mx.eval(a)
        mx.synchronize()
        best = min(best, time.perf_counter() - t0)
    return 2 * n * 4 / best / 1e9


def _time(fn, reps=3):
    fn()  # warm
    best = np.inf
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def sweep(kind, cands, run, views):
    rows = []
    for tg in cands:
        try:
            dt = _time(lambda: run(tg))
            rows.append((dt, tg))
            print(f"  {kind:10s} tg={str(tg):12s} {dt / views * 1e3:8.2f} ms/view")
        except Exception as e:
            print(f"  {kind:10s} tg={str(tg):12s} FAIL {str(e)[:50]}")
    rows.sort()
    return rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvox", type=int, default=512)
    ap.add_argument("--ndet", type=int, default=512)
    ap.add_argument("--views", type=int, default=10)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    print(f"machine: {machine()}  mlx {mx.__version__}")
    bw = measure_bandwidth()
    print(f"sustained read+write bandwidth: {bw:.0f} GB/s\n")

    geo = mlx_tomo.geometry_default(high_resolution=True)
    geo.nVoxel = np.array([args.nvox] * 3)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array([args.ndet] * 2)
    geo.dDetector = np.array([0.8, 0.8]) * (512 / args.ndet)
    geo.sDetector = geo.nDetector * geo.dDetector
    angles = np.linspace(0, 2 * np.pi, args.views, endpoint=False)
    vol = ellipsoid_phantom(geo, default_phantom_mm(geo)).astype(np.float32)
    vol_mx = mx.array(vol)
    mx.eval(vol_mx)
    proj = mx.array(np.random.default_rng(0).random(
        (args.views, args.ndet, args.ndet)).astype(np.float32))
    mx.eval(proj)

    g = geo.copy()
    g.check_geo(angles)
    winners = {}

    def run_ax(ptype):
        def run(tg):
            # dispatch through the public path with a patched tuning table
            tuning._cache = dict(tuning._load())
            tuning._cache["ax_interp" if ptype == "interpolated" else "ax_siddon"] = tg
            p = pr.ax_gpu(vol_mx, g, ptype)
            mx.eval(p)
        return run

    def run_atb(tg):
        tuning._cache = dict(tuning._load())
        tuning._cache["atb"] = tg
        v = pr.atb_gpu(proj, g, "FDK")
        mx.eval(v)

    dt, tg = sweep("ax_interp", CAND_3D, run_ax("interpolated"), args.views)
    winners["ax_interp"] = tg
    print(f"-> ax_interp best {tg}: {dt / args.views * 1e3:.2f} ms/view\n")

    dt, tg = sweep("ax_siddon", CAND_3D, run_ax("Siddon"), args.views)
    winners["ax_siddon"] = tg
    print(f"-> ax_siddon best {tg}: {dt / args.views * 1e3:.2f} ms/view\n")

    dt, tg = sweep("atb", CAND_3D, run_atb, args.views)
    winners["atb"] = tg
    print(f"-> atb best {tg}: {dt / args.views * 1e3:.2f} ms/view\n")

    tuning.reload()
    if mlx_tomo.texture_available():
        plan = mlx_tomo.TextureProjector(vol, geo)
        def run_tex(tg):
            plan.project(angles, threadgroup=tg)
        dt, tg = sweep("texture", CAND_TEX, run_tex, args.views)
        winners["texture"] = tg
        print(f"-> texture best {tg}: {dt / args.views * 1e3:.2f} ms/view\n")
        plan.close()
    else:
        print("texture bridge not built; skipping texture sweep\n")

    out = {"machine": machine(), "mlx": mx.__version__,
           "bandwidth_gbs": round(bw, 1),
           "workload": {"nvox": args.nvox, "ndet": args.ndet, "views": args.views},
           "threadgroups": {k: list(v) for k, v in winners.items()}}
    print(json.dumps(out, indent=1))
    if not args.dry:
        path = os.path.join(os.path.dirname(__file__), "..", "mlx_tomo",
                            "tuning_local.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        tuning.reload()
        print(f"\nwrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
