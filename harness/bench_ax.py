"""Forward-projection throughput benchmark.

Measures warm ms/view for clinical-ish sizes on the local GPU.
Usage: python harness/bench_ax.py [--quick]
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

import mlx_tomo
from gen import default_phantom_mm, ellipsoid_phantom, machine


def bench_case(nvox, ndet, nviews, ptype, mode="cone", accuracy=0.5, reps=9):
    if mode == "cone":
        geo = mlx_tomo.geometry_default(high_resolution=True)
        geo.nVoxel = np.array([nvox] * 3)
        geo.dVoxel = geo.sVoxel / geo.nVoxel
        geo.nDetector = np.array([ndet, ndet])
        geo.dDetector = np.array([0.8, 0.8]) * (512 / ndet)
        geo.sDetector = geo.nDetector * geo.dDetector
    else:
        geo = mlx_tomo.geometry(mode="parallel", nVoxel=np.array([nvox] * 3))
        geo.nDetector = np.array([ndet, ndet])
        geo.dDetector = np.array([nvox / ndet, nvox / ndet])
        geo.sDetector = geo.nDetector * geo.dDetector
    geo.accuracy = accuracy

    vol = ellipsoid_phantom(geo, default_phantom_mm(geo)).astype(np.float32)
    vol_mx = mx.array(vol)
    angles = np.linspace(0, 2 * np.pi, nviews, endpoint=False)

    # Warm-up: compiles the kernel, allocates pools.
    p = mlx_tomo.Ax(vol_mx, geo, angles, projection_type=ptype, return_np=False)
    mx.eval(p)
    mx.synchronize()

    times = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        p = mlx_tomo.Ax(vol_mx, geo, angles, projection_type=ptype, return_np=False)
        mx.eval(p)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    times = np.sort(np.asarray(times))
    return float(np.median(times)), float(times[0]), float(times[-1])


def main():
    quick = "--quick" in sys.argv
    print(f"machine: {machine()}")
    print(f"mlx {mx.__version__}")
    cases = [
        # (nvox, ndet, nviews, ptype)
        (256, 256, 1, "interpolated"),
        (256, 256, 1, "Siddon"),
        (256, 256, 100, "interpolated"),
        (256, 256, 100, "Siddon"),
        (512, 512, 1, "interpolated"),
        (512, 512, 1, "Siddon"),
    ]
    if not quick:
        cases += [
            (512, 512, 100, "interpolated"),
            (512, 512, 100, "Siddon"),
            (512, 768, 10, "interpolated"),
        ]
    rows = []
    for nvox, ndet, nviews, ptype in cases:
        med, lo, hi = bench_case(nvox, ndet, nviews, ptype)
        ms_view = med / nviews * 1e3
        rows.append(dict(nvox=nvox, ndet=ndet, nviews=nviews, ptype=ptype,
                         median_s=med, min_s=lo, max_s=hi,
                         ms_per_view=ms_view, reps=9))
        print(f"{nvox}^3 -> {ndet}^2 x{nviews:4d} {ptype:13s}: "
              f"{ms_view:7.2f} ms/view median "
              f"[{lo/nviews*1e3:.2f}-{hi/nviews*1e3:.2f}] over 9 reps")
    peak = mx.get_peak_memory() / 2**30
    print(f"peak GPU memory: {peak:.2f} GiB")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "bench_ax.json")
    with open(out, "w") as f:
        json.dump(dict(machine=machine(), mlx=mx.__version__, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
