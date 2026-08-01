"""At-scale FDK validation and the documented off-plane error budget.

Not part of the default suite (harness/run_tests.py globs test_*.py):
this allocates GBs and runs for a minute or two. Companion to
harness/test_fdk.py, in the spirit of check_bigvol.py. First run on an
M5 Max, 128 GB; numbers quoted in the CHANGELOG.

Sections:
1. 512^3 <- 360 x 512^2 FDK: recovered-mu accuracy at scale and the
   wall-clock breakdown (weight+filter vs backprojection).
2. Off-plane error budget: uniform tall ellipsoid spanning ~7 deg of
   half-cone angle; per-z interior recovered mu. FDK is exact in the
   central plane and degrades smoothly with cone angle — this table IS
   the baseline's error budget, not a bug to fix.
3. Multi-ellipsoid phantom (contrast steps): central profiles and
   per-region means vs the exact rasterized ground truth.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

import mlx_tomo
from mlx_tomo.algorithms import _cosine_weight
from mlx_tomo.filtering import _filter_core
from gen import (analytic_projection, default_phantom_mm,
                 ellipsoid_phantom, machine, rel_l2)

fails = 0
MU = 0.02


def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"{status} {name} {detail}", flush=True)


def big_geo():
    geo = mlx_tomo.geometry_default(high_resolution=True)  # 512^3, 512^2
    return geo


def timing_and_accuracy():
    print("\n== 512^3 <- 360 x 512^2 FDK: accuracy and timing ==", flush=True)
    geo = big_geo()
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ells = [(MU, (0.0, 0.0, 0.0), (100.0, 100.0, 110.0))]
    t0 = time.time()
    proj = analytic_projection(geo, angles, ells).astype(np.float32)
    print(f"  analytic projections: {time.time() - t0:.0f}s", flush=True)

    vol = mlx_tomo.FDK(proj, geo, angles)  # warm (compiles kernels)

    # recovered mu, central slice and near-plane
    nz = vol.shape[0]
    y = (np.arange(512) - 255.5) * geo.dVoxel[1]
    x = (np.arange(512) - 255.5) * geo.dVoxel[2]
    r = np.hypot(x[None, :], y[:, None])
    inner = r < 60.0
    ratio = vol[nz // 2][inner].mean() / MU
    check("512^3 central-slice recovered mu", abs(ratio - 1) < 0.005,
          f"ratio={ratio:.4f}")
    # FOV radius is 133 mm; keep the exterior check inside it
    ext = abs(vol[nz // 2][(r > 125.0) & (r < 127.0)].mean()) / MU
    check("512^3 exterior (in-FOV) clean", ext < 0.01, f"|mean|/mu={ext:.4f}")

    # timing: end-to-end and breakdown, warm, best of 3
    proj_mx = mx.array(proj)
    mx.eval(proj_mx)
    g = geo.copy()
    g.check_geo(angles)

    def run_full():
        v = mlx_tomo.FDK(proj_mx, g, angles, return_np=False)
        mx.eval(v)

    def run_filter():
        f = _filter_core(proj_mx * _cosine_weight(g), g, 0.0, 1.0, None)
        mx.eval(f)

    best_full = best_filt = np.inf
    run_full()
    for _ in range(3):
        mx.synchronize()
        t0 = time.perf_counter()
        run_full()
        mx.synchronize()
        best_full = min(best_full, time.perf_counter() - t0)
    run_filter()
    for _ in range(3):
        mx.synchronize()
        t0 = time.perf_counter()
        run_filter()
        mx.synchronize()
        best_filt = min(best_filt, time.perf_counter() - t0)
    print(f"  FDK end-to-end: {best_full:.3f} s "
          f"({best_full / 360 * 1e3:.2f} ms/view)")
    print(f"  weight+filter : {best_filt:.3f} s "
          f"({best_filt / best_full * 100:.0f}% of total)")
    check("512^3 recon lands near target", best_full < 1.2,
          f"{best_full:.3f} s (target ~0.6 s, stop-optimizing bound 2x)")
    del proj, proj_mx, vol
    return best_full


def offplane_budget():
    print("\n== off-plane error budget (uniform ellipsoid, ~7 deg cone) ==",
          flush=True)
    geo = big_geo()
    # halve z-resolution to keep runtime modest; accuracy is set by the
    # detector, not the voxel grid
    geo.nVoxel = np.array([256, 512, 512])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ells = [(MU, (0.0, 0.0, 0.0), (100.0, 100.0, 120.0))]
    proj = analytic_projection(geo, angles, ells).astype(np.float32)
    vol = mlx_tomo.FDK(proj, geo, angles)

    nz = vol.shape[0]
    dz = geo.dVoxel[0]
    y = (np.arange(512) - 255.5) * geo.dVoxel[1]
    x = (np.arange(512) - 255.5) * geo.dVoxel[2]
    rr = np.hypot(x[None, :], y[:, None])
    print("  z(mm)  cone(deg)  interior mean/mu")
    worst_low = 0.0   # |z| < 60 mm (cone < 3.4 deg)
    worst_high = 0.0  # up to ~110 mm (cone ~6.3 deg)
    for zmm in (0, 20, 40, 60, 80, 100, 110):
        iz = int(round(zmm / dz + nz / 2 - 0.5))
        rz2 = (zmm / 120.0) ** 2
        mask = rr < 100.0 * np.sqrt(max(0.36 - rz2, 0.01))
        ratio = vol[iz][mask].mean() / MU
        cone = np.degrees(np.arctan(zmm / geo.DSO))
        print(f"  {zmm:5.0f}  {cone:9.2f}  {ratio:.4f}")
        err = abs(ratio - 1)
        if zmm <= 60:
            worst_low = max(worst_low, err)
        worst_high = max(worst_high, err)
    check("off-plane budget: <=3.4deg cone within 1%", worst_low < 0.01,
          f"worst={worst_low * 100:.2f}%")
    check("off-plane budget: <=6.3deg cone within 3%", worst_high < 0.03,
          f"worst={worst_high * 100:.2f}%")
    del proj, vol


def contrast_phantom():
    print("\n== multi-ellipsoid contrast phantom vs ground truth ==",
          flush=True)
    geo = big_geo()
    geo.nVoxel = np.array([256, 512, 512])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    # default phantom scaled by mu; radii 0.36*256=92mm < 133mm FOV
    ells = [(rho * MU, c, ax) for rho, c, ax in default_phantom_mm(geo)]
    proj = analytic_projection(geo, angles, ells).astype(np.float32)
    vol = mlx_tomo.FDK(proj, geo, angles)
    gt = ellipsoid_phantom(geo, ells)

    # central slice, eroded per-region means (avoid edge ringing)
    sl = vol[128]
    gsl = gt[128]
    for name, ell in zip(("body", "hole", "insert"), ells):
        rho, c, ax = ell
        y = (np.arange(512) + 0.5) * geo.dVoxel[1] - 128.0 - c[1]
        x = (np.arange(512) + 0.5) * geo.dVoxel[2] - 128.0 - c[0]
        rz2 = (0.0 - c[2]) ** 2 / ax[2] ** 2
        rr = (x[None, :] / ax[0]) ** 2 + (y[:, None] / ax[1]) ** 2
        m = rr < max(0.55 - rz2, 0.02)
        got = sl[m].mean()
        want = gsl[m].mean()
        check(f"contrast phantom {name} mean",
              abs(got - want) < 0.02 * MU + 0.02 * abs(want),
              f"got={got:.5f} want={want:.5f}")
    # profile through the center: report L2 in the interior band
    prof = sl[256, :]
    gprof = gsl[256, :]
    band = slice(96, 416)  # +-80 mm
    d = rel_l2(prof[band], gprof[band])
    check("central profile rel_l2 (edges included)", d < 0.08,
          f"rel_l2={d:.4f}")
    print("  profile (x mm, recon, truth) every 32 voxels:")
    for i in range(96, 417, 32):
        xmm = (i + 0.5) * geo.dVoxel[2] - 128.0
        print(f"    {xmm:+7.1f}  {prof[i]:+.5f}  {gprof[i]:+.5f}")
    del proj, vol, gt


if __name__ == "__main__":
    print(f"machine: {machine()}  mlx {mx.__version__}", flush=True)
    timing_and_accuracy()
    offplane_budget()
    contrast_phantom()
    peak = mx.get_peak_memory() / 2**30
    print(f"\npeak GPU memory: {peak:.1f} GiB")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
