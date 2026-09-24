"""Texture backend vs buffer backend throughput.

Reports nine-run medians and separates GPU-resident buffer output from the
host-returning paths. Texture upload is measured with the pipeline already
compiled, so it does not include one-time shader compilation.
Usage: python harness/bench_texture.py [--quick]
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


def _measure(fn, reps):
    fn()  # warm
    times = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    times = np.sort(np.asarray(times))
    return float(np.median(times)), float(times[0]), float(times[-1])


def bench(nvox, ndet, nviews, dtype=np.float32, reps=9):
    geo = mlx_tomo.geometry_default(high_resolution=True)
    geo.nVoxel = np.array([nvox] * 3)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array([ndet, ndet])
    geo.dDetector = np.array([0.8, 0.8]) * (512 / ndet)
    geo.sDetector = geo.nDetector * geo.dDetector

    vol = ellipsoid_phantom(geo, default_phantom_mm(geo)).astype(dtype)
    angles = np.linspace(0, 2 * np.pi, nviews, endpoint=False)

    # Buffer backend with output left GPU-resident.
    vol_mx = mx.array(vol)
    def buffer_gpu():
        return mlx_tomo.Ax(
            vol_mx, geo, angles, projection_type="interpolated",
            return_np=False)
    bg_med, bg_lo, bg_hi = _measure(buffer_gpu, reps)

    # Same buffer kernel with a host numpy result, matching the texture
    # backend's current output residency.
    def buffer_host():
        return mlx_tomo.Ax(
            vol_mx, geo, angles, projection_type="interpolated",
            return_np=True)
    bh_med, bh_lo, bh_hi = _measure(buffer_host, reps)

    # Compile once outside the upload measurement, then construct the plan
    # used for timing. The second construction reuses the cached pipeline.
    warm_plan = mlx_tomo.TextureProjector(vol, geo)
    warm_plan.close()
    t0 = time.perf_counter()
    plan = mlx_tomo.TextureProjector(vol, geo)
    plan_s = time.perf_counter() - t0
    def texture_host():
        return plan.project(angles)
    tx_med, tx_lo, tx_hi = _measure(texture_host, reps)
    plan.close()

    tag = f"{nvox}^3 -> {ndet}^2 x{nviews:4d} {np.dtype(dtype).name}"
    scale = 1e3 / nviews
    print(f"{tag}:\n"
          f"  buffer GPU  {bg_med*scale:7.2f} ms/view "
          f"[{bg_lo*scale:.2f}-{bg_hi*scale:.2f}]\n"
          f"  buffer host {bh_med*scale:7.2f} ms/view "
          f"[{bh_lo*scale:.2f}-{bh_hi*scale:.2f}]\n"
          f"  texture host{tx_med*scale:7.2f} ms/view "
          f"[{tx_lo*scale:.2f}-{tx_hi*scale:.2f}] "
          f"({bh_med/tx_med:4.2f}x vs buffer host) | "
          f"upload {plan_s*1e3:6.1f} ms")
    return {
        "nvox": nvox, "ndet": ndet, "nviews": nviews,
        "dtype": np.dtype(dtype).name, "reps": reps,
        "buffer_gpu": {"median_s": bg_med, "min_s": bg_lo, "max_s": bg_hi},
        "buffer_host": {"median_s": bh_med, "min_s": bh_lo, "max_s": bh_hi},
        "texture_host": {"median_s": tx_med, "min_s": tx_lo, "max_s": tx_hi},
        "texture_upload_s": plan_s,
    }


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    print(f"machine: {machine()}  mlx {mx.__version__}")
    if not mlx_tomo.texture_available():
        print("SKIP texture bridge not built")
        raise SystemExit(0)
    rows = [bench(256, 256, 100)]
    if not quick:
        rows.extend([
            bench(512, 512, 1),
            bench(512, 512, 100),
            bench(512, 512, 100, dtype=np.float16),
        ])
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"),
                exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "bench_texture.json")
    with open(out, "w") as f:
        json.dump({"machine": machine(), "mlx": mx.__version__, "rows": rows},
                  f, indent=1)
