"""Full-scale validation of volume slab splitting (needs ~96 GB free RAM).

Not part of the default suite (run_tests.py globs test_*.py): this script
allocates tens of GB and runs for minutes. It is the at-scale companion to
harness/test_slabs.py and was first run on an M5 Max, 128 GB.

Layers of evidence, mirroring the repo's validation doctrine:

1. 1024^3 (fits a single dispatch): slab-forced vs natural single-dispatch
   at the float32 floor — the only size where slab-vs-unsplit can be compared
   directly at scale.
2. 2048^3 (8.59e9 elements > 2^31-1, real slab path, 5 slabs of 511):
   - cut-position invariance: natural 511-layer slabs vs forced 400-layer
     slabs (different cut planes, both in-contract dispatches) must agree
     at the float32 floor; any mis-count at a cut plane depends on where the
     cuts are and breaks this.
   - analytic ellipsoid oracle: line integrals of an exact phantom.
   - impulses on the natural cut layers, each compared against an
     independent unsplit reference: the same impulse in an
     offOrigin-shifted 64-layer sub-volume occupying the same world
     positions (single legal dispatch, no cut planes). A dropped or
     doubled tap at the cut reads ~0.5x/2x.
   - adjoint identity <Ax,y>/<x,Atb y> == dVol/dPix at full scale, both
     operators slabbed.
   - Atb cut-position invariance: 511- vs 400-layer output tiles.
3. A projection stack genuinely over 2^31-1 elements (600 x 2048^2)
   exercising real view-axis chunking: chunked views must be bitwise
   identical to per-view dispatches, Atb chunk accumulation at fp noise.

Usage: python harness/check_bigvol.py [--quick]   (--quick: 1024^3 only)
"""

import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

import mlx_tomo
from mlx_tomo import projector as pr
from gen import analytic_projection, default_phantom_mm, machine, rel_l2

fails = 0


def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"{status} {name} {detail}", flush=True)


def clear():
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()


def rel_l2_mx_slabbed(got, ref, chunk=128):
    """rel_l2 for device arrays too big to reduce in one op: MLX
    reductions flatten, and flatten caps shape dims at int32."""
    nz = int(got.shape[0])
    num = den = 0.0
    for z0 in range(0, nz, chunk):
        dz = got[z0:z0 + chunk] - ref[z0:z0 + chunk]
        num += float(mx.sum(dz.astype(mx.float32) ** 2))
        den += float(mx.sum(ref[z0:z0 + chunk].astype(mx.float32) ** 2))
    return (num / den) ** 0.5 if den else float(num > 0)


def big_geo(nvox, ndet=None, accuracy=0.5):
    ndet = ndet or nvox
    geo = mlx_tomo.geometry_default(high_resolution=True)
    geo.nVoxel = np.array([nvox] * 3)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array([ndet, ndet])
    geo.dDetector = np.array([0.8, 0.8]) * (512 / ndet)
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.accuracy = accuracy
    return geo


def rasterize_phantom(geo, out_dtype=np.float32, chunk=64):
    """Slab-wise ellipsoid rasterization (gen.ellipsoid_phantom would
    build full-volume float64 meshgrids — 8x the memory)."""
    ells = default_phantom_mm(geo)
    nz, ny, nx = (int(k) for k in geo.nVoxel)
    dz, dy, dx = geo.dVoxel
    sz, sy, sx = geo.sVoxel
    y = ((np.arange(ny, dtype=np.float32) + 0.5) * dy - sy / 2)[None, :, None]
    x = ((np.arange(nx, dtype=np.float32) + 0.5) * dx - sx / 2)[None, None, :]
    vol = np.zeros((nz, ny, nx), dtype=out_dtype)
    for z0 in range(0, nz, chunk):
        z = ((np.arange(z0, min(z0 + chunk, nz), dtype=np.float32) + 0.5)
             * dz - sz / 2)[:, None, None]
        acc = np.zeros((z.shape[0], ny, nx), dtype=np.float32)
        for rho, c, ax in ells:
            q = (((x - c[0]) / ax[0]) ** 2 + ((y - c[1]) / ax[1]) ** 2
                 + ((z - c[2]) / ax[2]) ** 2)
            acc[q <= 1.0] += rho
        vol[z0:z0 + z.shape[0]] = acc
    return vol, ells


def check_1024_slab_vs_unsplit():
    print("\n== 1024^3: slab-forced vs single dispatch ==", flush=True)
    geo = big_geo(1024, ndet=1024)
    vol, _ = rasterize_phantom(geo)
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    n_v, n_u = (int(k) for k in geo.nDetector)
    proj = np.random.default_rng(0).random((4, n_v, n_u)).astype(np.float32)

    # Siddon tolerance: each slab re-enters the DDA with a fresh
    # ray-parameter division where the unsplit kernel arrives via ~nz
    # accumulated `azn += daz` roundings, so the two cut cells' segment
    # lengths differ by ~nz*ulp — seam noise that grows with volume size
    # (measured 3e-5 at 1024^3, vs ~7e-6 at 64^3). A genuine counting bug
    # (dropped/doubled cell on cut-crossing rays) would sit near 1e-3.
    for ptype, tol in (("interpolated", 1e-6), ("Siddon", 1e-4)):
        full = mlx_tomo.Ax(vol, geo, angles, projection_type=ptype)
        pr._SLAB_ELEMS = 300 * 1024 * 1024  # 300-layer slabs
        try:
            got = mlx_tomo.Ax(vol, geo, angles, projection_type=ptype)
        finally:
            pr._SLAB_ELEMS = pr._MK_MAX_ELEMS
        d = rel_l2(got, full)
        check(f"1024^3 Ax {ptype} slab == unsplit", d < tol, f"d={d:.2e}")
        del full, got
        clear()

    full = mlx_tomo.Atb(proj, geo, angles, backprojection_type="FDK",
                        return_np=False)
    mx.eval(full)
    pr._SLAB_ELEMS = 300 * 1024 * 1024
    try:
        got = mlx_tomo.Atb(proj, geo, angles, backprojection_type="FDK",
                           return_np=False)
        mx.eval(got)
    finally:
        pr._SLAB_ELEMS = pr._MK_MAX_ELEMS
    d = float(mx.linalg.norm(got - full) / mx.linalg.norm(full))
    check("1024^3 Atb FDK slab == unsplit", d == 0.0, f"d={d:.2e}")
    del vol, proj, full, got
    clear()


def check_2048_forward():
    print("\n== 2048^3 forward (real slab path, 5 slabs of 511) ==", flush=True)
    geo = big_geo(2048, ndet=2048)
    t0 = time.time()
    vol, ells = rasterize_phantom(geo)
    print(f"  rasterized 2048^3 phantom in {time.time() - t0:.0f}s", flush=True)
    angles = np.array([0.0, 0.7, 2.1])
    ana = analytic_projection(geo, angles, ells)

    got = {}
    for ptype in ("interpolated", "Siddon"):
        t0 = time.time()
        got[ptype] = mlx_tomo.Ax(vol, geo, angles, projection_type=ptype)
        dt = time.time() - t0
        d = rel_l2(got[ptype], ana)
        # voxelization error of the rasterized phantom dominates; geometry
        # bugs would be orders of magnitude above this
        check(f"2048^3 Ax {ptype} vs analytic oracle", d < 0.02,
              f"rel_l2={d:.4f} ({dt / len(angles) * 1e3:.0f} ms/view warm+slabs)")

    # cut-position invariance: natural 511-layer cuts vs forced 400-layer
    # (the forced size must itself stay under 2^31-1 elements per slab:
    # 400 * 2048^2 = 1.68e9)
    pr._SLAB_ELEMS = 400 * 2048 * 2048
    try:
        alt = mlx_tomo.Ax(vol, geo, angles, projection_type="interpolated")
    finally:
        pr._SLAB_ELEMS = pr._MK_MAX_ELEMS
    d = rel_l2(alt, got["interpolated"])
    check("2048^3 Ax interp cut-position invariance", d < 1e-6, f"d={d:.2e}")
    del alt, got, ana
    clear()

    # Impulses on the natural cut layers (z = 511k), each validated
    # against an INDEPENDENT unsplit reference: a 64-layer sub-volume
    # holding the same impulse, offOrigin-shifted so its voxels occupy
    # the same world positions — one legal single dispatch, no cut
    # planes. A dropped or double-counted tap at the cut would show up
    # as ~0.5x/2x here; agreement is at the float32 floor.
    nz, ny, nx = (int(k) for k in geo.nVoxel)
    dz = float(geo.dVoxel[0])
    marks = [510, 511, 512, 1021, 1022, 2043, 2044]
    ang1 = np.array([0.4])
    for iz in marks:
        v1 = np.zeros((nz, ny, nx), dtype=np.float32)
        v1[iz, ny // 2 - 7, nx // 2 + 5] = 1.0
        p_big = mlx_tomo.Ax(v1, geo, ang1, projection_type="interpolated")[0]
        del v1
        clear()

        a = min(max(iz - 32, 0), nz - 64)
        sub = geo.copy()
        sub.nVoxel = np.array([64, ny, nx])
        sub.sVoxel = sub.nVoxel * geo.dVoxel
        sub.dVoxel = geo.dVoxel.copy()
        sub.offOrigin = np.array([(a + 32 - nz / 2.0) * dz, 0.0, 0.0])
        vs = np.zeros((64, ny, nx), dtype=np.float32)
        vs[iz - a, ny // 2 - 7, nx // 2 + 5] = 1.0
        p_ref = mlx_tomo.Ax(vs, sub, ang1, projection_type="interpolated")[0]
        d = rel_l2(p_big, p_ref)
        # The reference is a DIFFERENT geometry (sVoxel_z differs), so its
        # float32 view params round differently and len can differ by one:
        # two legitimate quadratures of the same integral agree to ~1e-4
        # on a sparse impulse (measured 5e-6..2e-4 across marks). A
        # dropped or doubled cut-layer tap reads ~0.3-1.0 here.
        check(f"2048^3 cut-layer impulse z={iz} vs unsplit sub-volume ref",
              d < 1e-3, f"rel_l2={d:.2e}")
        del vs, p_big, p_ref
        clear()


def check_2048_adjoint_and_atb():
    print("\n== 2048^3 adjoint + Atb tiles ==", flush=True)
    geo = big_geo(2048, ndet=2048, accuracy=0.25)
    vol, _ = rasterize_phantom(geo)
    angles = np.array([0.3, 1.7])
    n_v, n_u = (int(k) for k in geo.nDetector)
    y = np.random.default_rng(5).random((2, n_v, n_u)).astype(np.float32)

    ax = mlx_tomo.Ax(vol, geo, angles, projection_type="interpolated")
    t0 = time.time()
    aty = mlx_tomo.Atb(y, geo, angles, backprojection_type="matched",
                       return_np=False)
    mx.eval(aty)
    dt = time.time() - t0
    num = float((ax.astype(np.float64) * y).sum())
    den = 0.0  # <x, Atb y> in float64, slab-wise off the device
    for z0 in range(0, 2048, 256):
        den += float(np.einsum(
            "ijk,ijk->", vol[z0:z0 + 256].astype(np.float64),
            np.array(aty[z0:z0 + 256], copy=False).astype(np.float64)))
    measure = float(np.prod(geo.dVoxel) / np.prod(geo.dDetector))
    r = (num / den) / measure
    check("2048^3 adjoint <Ax,y>/<x,Atb y>", 0.95 < r < 1.05,
          f"ratio={r:.4f} (Atb {dt / 2 * 1e3:.0f} ms/view warm-ish)")
    del ax, vol
    clear()

    # Atb cut-position invariance: 511-layer tiles vs 400-layer tiles
    pr._SLAB_ELEMS = 400 * 2048 * 2048
    try:
        aty2 = mlx_tomo.Atb(y, geo, angles, backprojection_type="matched",
                            return_np=False)
        mx.eval(aty2)
    finally:
        pr._SLAB_ELEMS = pr._MK_MAX_ELEMS
    d = rel_l2_mx_slabbed(aty2, aty)
    check("2048^3 Atb tile-position invariance", d == 0.0, f"d={d:.2e}")
    del aty, aty2, y
    clear()


def check_big_projection_stack():
    print("\n== >2^31-element projection stack (real view chunking) ==",
          flush=True)
    geo = big_geo(512, ndet=2048)
    vol, _ = rasterize_phantom(geo)
    n = 600  # 600 * 2048^2 = 2.5e9 elements > 2^31-1
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    proj = mlx_tomo.Ax(vol, geo, angles, projection_type="Siddon",
                       return_np=False)
    mx.eval(proj)
    check("big-stack Ax shape", tuple(proj.shape) == (n, 2048, 2048),
          f"{tuple(proj.shape)}, {n * 2048 * 2048:,} elements")
    ok = True
    for i in (0, 511, 512, 599):  # chunk boundary is at 511
        single = mlx_tomo.Ax(vol, geo, angles[i:i + 1],
                             projection_type="Siddon", return_np=False)
        d = float(mx.linalg.norm(proj[i] - single[0]))
        if d != 0.0:
            ok = False
            check(f"big-stack view {i} bitwise", False, f"|diff|={d:.2e}")
    check("big-stack views == per-view dispatch (bitwise)", ok)

    atb = mlx_tomo.Atb(proj, geo, angles, backprojection_type="FDK",
                       return_np=False)
    mx.eval(atb)
    h1 = mlx_tomo.Atb(proj[:300], geo, angles[:300],
                      backprojection_type="FDK", return_np=False)
    h2 = mlx_tomo.Atb(proj[300:], geo, angles[300:],
                      backprojection_type="FDK", return_np=False)
    d = float(mx.linalg.norm(atb - (h1 + h2)) / mx.linalg.norm(atb))
    check("big-stack Atb == split-and-summed halves", d < 1e-6, f"d={d:.2e}")
    del vol, proj, atb, h1, h2
    clear()


if __name__ == "__main__":
    print(f"machine: {machine()}  mlx {mx.__version__}", flush=True)
    check_1024_slab_vs_unsplit()
    if "--quick" not in sys.argv:
        check_2048_forward()
        check_2048_adjoint_and_atb()
        check_big_projection_stack()
    peak = mx.get_peak_memory() / 2**30
    print(f"\npeak GPU memory: {peak:.1f} GiB")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
