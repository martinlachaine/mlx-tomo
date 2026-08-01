"""Texture backend vs buffer backend throughput.

Measures plan (texture upload) time separately from warm per-view time.
Usage: python harness/bench_texture.py [--quick]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

import mlx_tomo
from gen import default_phantom_mm, ellipsoid_phantom, machine


def bench(nvox, ndet, nviews, dtype=np.float32, reps=3):
    geo = mlx_tomo.geometry_default(high_resolution=True)
    geo.nVoxel = np.array([nvox] * 3)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array([ndet, ndet])
    geo.dDetector = np.array([0.8, 0.8]) * (512 / ndet)
    geo.sDetector = geo.nDetector * geo.dDetector

    vol = ellipsoid_phantom(geo, default_phantom_mm(geo)).astype(dtype)
    angles = np.linspace(0, 2 * np.pi, nviews, endpoint=False)

    # Buffer backend (warm)
    vol_mx = mx.array(vol)
    p = mlx_tomo.Ax(vol_mx, geo, angles, projection_type="interpolated",
                    return_np=False)
    mx.eval(p)
    buf_best = np.inf
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        p = mlx_tomo.Ax(vol_mx, geo, angles, projection_type="interpolated",
                        return_np=False)
        mx.eval(p)
        mx.synchronize()
        buf_best = min(buf_best, time.perf_counter() - t0)

    # Texture backend: plan once, project many
    t0 = time.perf_counter()
    plan = mlx_tomo.TextureProjector(vol, geo)
    plan_s = time.perf_counter() - t0
    plan.project(angles[:1])  # warm (pipeline already compiled at plan)
    tex_best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        plan.project(angles)
        tex_best = min(tex_best, time.perf_counter() - t0)
    plan.close()

    tag = f"{nvox}^3 -> {ndet}^2 x{nviews:4d} {np.dtype(dtype).name}"
    print(f"{tag}: buffer {buf_best/nviews*1e3:7.2f} ms/view | "
          f"texture {tex_best/nviews*1e3:7.2f} ms/view "
          f"({buf_best/tex_best:4.1f}x) | upload {plan_s*1e3:6.1f} ms")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    print(f"machine: {machine()}  mlx {mx.__version__}")
    if not mlx_tomo.texture_available():
        print("SKIP texture bridge not built")
        raise SystemExit(0)
    bench(256, 256, 100)
    if not quick:
        bench(512, 512, 1)
        bench(512, 512, 100)
        bench(512, 512, 100, dtype=np.float16)
