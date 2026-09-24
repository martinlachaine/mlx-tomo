"""Backprojection throughput. Usage: python harness/bench_atb.py [--quick]"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

import mlx_tomo
from gen import machine


def bench(nvox, ndet, nviews, bptype, reps=9):
    geo = mlx_tomo.geometry_default(high_resolution=True)
    geo.nVoxel = np.array([nvox] * 3)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array([ndet, ndet])
    geo.dDetector = np.array([0.8, 0.8]) * (512 / ndet)
    geo.sDetector = geo.nDetector * geo.dDetector
    angles = np.linspace(0, 2 * np.pi, nviews, endpoint=False)
    proj = mx.array(np.random.default_rng(0).random(
        (nviews, ndet, ndet)).astype(np.float32))

    v = mlx_tomo.Atb(proj, geo, angles, backprojection_type=bptype,
                     return_np=False)
    mx.eval(v)
    times = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        v = mlx_tomo.Atb(proj, geo, angles, backprojection_type=bptype,
                         return_np=False)
        mx.eval(v)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    times = np.sort(np.asarray(times))
    med, lo, hi = float(np.median(times)), float(times[0]), float(times[-1])
    print(f"{nvox}^3 <- {ndet}^2 x{nviews:4d} {bptype:8s}: "
          f"{med/nviews*1e3:7.2f} ms/view median "
          f"[{lo/nviews*1e3:.2f}-{hi/nviews*1e3:.2f}] over {reps} reps")
    return {
        "nvox": nvox, "ndet": ndet, "nviews": nviews,
        "bptype": bptype, "median_s": med, "min_s": lo, "max_s": hi,
        "ms_per_view": med / nviews * 1e3, "reps": reps,
    }


if __name__ == "__main__":
    print(f"machine: {machine()}  mlx {mx.__version__}")
    rows = [bench(256, 256, 100, "FDK")]
    if "--quick" not in sys.argv:
        rows.extend([
            bench(256, 256, 100, "matched"),
            bench(512, 512, 100, "FDK"),
            bench(512, 512, 360, "FDK"),
        ])
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"),
                exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "bench_atb.json")
    with open(out, "w") as f:
        json.dump({"machine": machine(), "mlx": mx.__version__, "rows": rows},
                  f, indent=1)
