"""Volume z-slab splitting must reproduce the single-dispatch paths.

Forces an artificially low slab threshold so the slab code runs at small
sizes (the real 2^31-1 threshold needs volumes > 8 GiB). Expectations:

- Atb slabs are independent output tiles computed with bitwise-identical
  per-voxel expressions, so slab == unsplit exactly. (A future MLX/Metal
  compiler bump may relax this to ~1e-7; today it is a free exactness
  check.)
- Ax slabs are partial projections of zero-padded sub-volumes; the sum
  reassociates float32 adds, so agreement is at the float32 floor, not bitwise.
  Siddon additionally recomputes the slab-entry ray parameter per slab
  (fresh division vs the unsplit kernel's accumulated plane stepping),
  which perturbs the two cut-plane cells' segment lengths by ~1 ulp of
  the ray parameter — seam noise well under 1e-5, still far below any
  real counting bug (one dropped/doubled cell would show ~1e-4 here).
- An impulse sitting exactly on a cut layer is the sharpest double-count
  probe: its full projection lives in the taps of two adjacent slabs.
"""

import numpy as np

from gen import default_phantom_mm, ellipsoid_phantom, rel_l2

import mlx.core as mx

import mlx_tomo
from mlx_tomo import projector as pr
from mlx_tomo.ref import atb_ref, ax_ref

fails = 0


def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"{status} {name} {detail}")


def slabbed(layers, fn, plane=None):
    """Run fn() with the slab threshold forced to `layers` z-layers.

    plane: (ny, nx) of the volume in play; defaults to the module geo.
    """
    ny, nx = plane if plane is not None else (NY, NX)
    pr._SLAB_ELEMS = layers * ny * nx
    try:
        return fn()
    finally:
        pr._SLAB_ELEMS = pr._MK_MAX_ELEMS


def cone_geo(with_offsets=True):
    geo = mlx_tomo.geometry_default(high_resolution=False)  # 64^3, 128^2
    if with_offsets:
        geo.offOrigin = np.array([5.0, -8.0, 3.0])
        geo.offDetector = np.array([4.0, -6.0])
        geo.COR = 2.0
        geo.rotDetector = np.array([0.02, -0.03, 0.05])
    return geo


GEO = cone_geo()
NZ, NY, NX = (int(k) for k in GEO.nVoxel)
N_V, N_U = (int(k) for k in GEO.nDetector)
ANGLES = np.linspace(0, 2 * np.pi, 7, endpoint=False)
VOL = ellipsoid_phantom(GEO, default_phantom_mm(GEO)).astype(np.float32)
PROJ = np.random.default_rng(2).random((7, N_V, N_U)).astype(np.float32)


def test_slab_vs_unsplit():
    ax_i = mlx_tomo.Ax(VOL, GEO, ANGLES, projection_type="interpolated")
    ax_s = mlx_tomo.Ax(VOL, GEO, ANGLES, projection_type="Siddon")
    atb_f = mlx_tomo.Atb(PROJ, GEO, ANGLES, backprojection_type="FDK")
    atb_m = mlx_tomo.Atb(PROJ, GEO, ANGLES, backprojection_type="matched")
    for layers in (1, 5, 17):  # incl. single-layer slabs and a remainder
        d = rel_l2(slabbed(layers, lambda: mlx_tomo.Ax(
            VOL, GEO, ANGLES, projection_type="interpolated")), ax_i)
        check(f"Ax interp slab({layers}) == unsplit", d < 1e-6, f"d={d:.2e}")
        d = rel_l2(slabbed(layers, lambda: mlx_tomo.Ax(
            VOL, GEO, ANGLES, projection_type="Siddon")), ax_s)
        check(f"Ax Siddon slab({layers}) == unsplit", d < 2e-5, f"d={d:.2e}")
        d = rel_l2(slabbed(layers, lambda: mlx_tomo.Atb(
            PROJ, GEO, ANGLES, backprojection_type="FDK")), atb_f)
        check(f"Atb FDK slab({layers}) == unsplit", d == 0.0, f"d={d:.2e}")
        d = rel_l2(slabbed(layers, lambda: mlx_tomo.Atb(
            PROJ, GEO, ANGLES, backprojection_type="matched")), atb_m)
        check(f"Atb matched slab({layers}) == unsplit", d == 0.0, f"d={d:.2e}")


def test_zyz_and_parallel():
    # ZYZ trajectories, incl. theta=pi/2 (rays crossing cut planes head-on)
    zyz = np.array([[0.0, 0.0, 0.0], [0.0, np.pi / 2, 0.0],
                    [0.3, 1.2, -0.4], [2.1, 0.6, 1.0]])
    geo = cone_geo(with_offsets=False)
    for ptype, tol in (("interpolated", 1e-6), ("Siddon", 2e-5)):
        full = mlx_tomo.Ax(VOL, geo, zyz, projection_type=ptype)
        d = rel_l2(slabbed(5, lambda: mlx_tomo.Ax(
            VOL, geo, zyz, projection_type=ptype)), full)
        check(f"Ax {ptype} slab ZYZ == unsplit", d < tol, f"d={d:.2e}")

    pgeo = mlx_tomo.geometry(mode="parallel", nVoxel=np.array([32, 48, 64]))
    pgeo.nDetector = np.array([48, 96])
    pgeo.sDetector = pgeo.nDetector * pgeo.dDetector
    pvol = ellipsoid_phantom(pgeo, default_phantom_mm(pgeo)).astype(np.float32)
    pproj = np.random.default_rng(4).random((7, 48, 96)).astype(np.float32)
    pplane = (48, 64)
    for ptype, tol in (("interpolated", 1e-6), ("Siddon", 2e-5)):
        full = mlx_tomo.Ax(pvol, pgeo, ANGLES, projection_type=ptype)
        got = slabbed(3, lambda: mlx_tomo.Ax(
            pvol, pgeo, ANGLES, projection_type=ptype), plane=pplane)
        d = rel_l2(got, full)
        check(f"Ax {ptype} slab parallel == unsplit", d < tol, f"d={d:.2e}")
    full = mlx_tomo.Atb(pproj, pgeo, ANGLES, backprojection_type="FDK")
    d = rel_l2(slabbed(3, lambda: mlx_tomo.Atb(
        pproj, pgeo, ANGLES, backprojection_type="FDK"), plane=pplane), full)
    check("Atb FDK slab parallel == unsplit", d == 0.0, f"d={d:.2e}")


def test_slab_plus_view_chunking():
    """Both split paths engaged at once (huge volume AND huge stack)."""
    ax_full = mlx_tomo.Ax(VOL, GEO, ANGLES, projection_type="interpolated")
    atb_full = mlx_tomo.Atb(PROJ, GEO, ANGLES, backprojection_type="FDK")
    pr._CHUNK_ELEMS = 3 * N_V * N_U  # 7 views -> 3+3+1
    try:
        ax_c = slabbed(5, lambda: mlx_tomo.Ax(
            VOL, GEO, ANGLES, projection_type="interpolated"))
        atb_c = slabbed(5, lambda: mlx_tomo.Atb(
            PROJ, GEO, ANGLES, backprojection_type="FDK"))
    finally:
        pr._CHUNK_ELEMS = pr._MK_MAX_ELEMS
    d = rel_l2(ax_c, ax_full)
    check("Ax slab+chunk == unsplit", d < 1e-6, f"d={d:.2e}")
    # chunked Atb accumulates float32 across view chunks: fp noise allowed
    d = rel_l2(atb_c, atb_full)
    check("Atb slab+chunk == unsplit", d < 1e-6, f"d={d:.2e}")


def test_vs_float64_reference():
    """Slab-forced GPU vs the float64 TIGRE-port oracle (never slabbed)."""
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    for ptype in ("interpolated", "Siddon"):
        ref = ax_ref(VOL.astype(np.float64), GEO, angles, ptype)
        got = slabbed(5, lambda: mlx_tomo.Ax(
            VOL, GEO, angles, projection_type=ptype))
        d = rel_l2(got, ref)
        check(f"slab gpu-vs-ref Ax {ptype}", d < 2e-4, f"rel_l2={d:.2e}")
    proj4 = PROJ[:4]
    for bptype in ("FDK", "matched"):
        ref = atb_ref(proj4.astype(np.float64), GEO, angles, bptype)
        got = slabbed(5, lambda: mlx_tomo.Atb(
            proj4, GEO, angles, backprojection_type=bptype))
        d = rel_l2(got, ref)
        check(f"slab gpu-vs-ref Atb {bptype}", d < 2e-4, f"rel_l2={d:.2e}")


def test_adjoint_with_slabs():
    """<Ax,y>/<x,Atb y> with BOTH operators slab-split."""
    geo = mlx_tomo.geometry_default(high_resolution=False)
    geo.accuracy = 0.25
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x = ellipsoid_phantom(geo, default_phantom_mm(geo)).astype(np.float32)
    n_v, n_u = (int(k) for k in geo.nDetector)
    y = np.random.default_rng(5).random((len(angles), n_v, n_u)).astype(np.float32)
    ax = slabbed(5, lambda: mlx_tomo.Ax(
        x, geo, angles, projection_type="interpolated"))
    aty = slabbed(5, lambda: mlx_tomo.Atb(
        y, geo, angles, backprojection_type="matched"))
    num = float((ax.astype(np.float64) * y).sum())
    den = float((x.astype(np.float64) * aty).sum())
    measure = float(np.prod(geo.dVoxel) / np.prod(geo.dDetector))
    r = (num / den) / measure
    check("adjoint ratio with slabs", 0.95 < r < 1.05, f"ratio={r:.4f}")


def test_impulse_on_cut_plane():
    """Impulses on/next to cut layers: taps split across two slabs must
    sum to exactly one impulse (double-count would read ~2x)."""
    geo = cone_geo(with_offsets=False)
    angles = np.array([0.0, 0.7, np.pi / 2, 2.5])
    layers = 5  # cuts at z = 5, 10, ...
    for iz in (4, 5, 6, 9, 10):
        vol = np.zeros((NZ, NY, NX), dtype=np.float32)
        vol[iz, NY // 2 - 3, NX // 2 + 2] = 1.0
        full = mlx_tomo.Ax(vol, geo, angles, projection_type="interpolated")
        got = slabbed(layers, lambda: mlx_tomo.Ax(
            vol, geo, angles, projection_type="interpolated"))
        d = rel_l2(got, full)
        ratio = float(got.max() / full.max())
        ok = d < 1e-6 and 0.999 < ratio < 1.001
        check(f"impulse at z={iz} (cuts every {layers})", ok,
              f"d={d:.2e} peak_ratio={ratio:.6f}")


def test_per_angle_fields_through_chunk_and_slab():
    """Per-angle DSD/DSO/COR/offsets/rotDetector VARYING across views:
    the chunk/slab paths slice these arrays per view chunk, and a
    slice-offset bug is invisible with broadcast-constant fields."""
    geo = cone_geo(with_offsets=False)
    n = 7
    geo.DSD = np.linspace(1500.0, 1560.0, n)
    geo.DSO = np.linspace(980.0, 1015.0, n)
    geo.COR = np.linspace(-2.0, 2.5, n)
    geo.offOrigin = np.stack([np.linspace(-5, 5, n), np.linspace(3, -4, n),
                              np.linspace(0, 6, n)], axis=1)
    geo.offDetector = np.stack([np.linspace(-4, 4, n),
                                np.linspace(2, -6, n)], axis=1)
    geo.rotDetector = np.stack([np.linspace(-0.03, 0.02, n),
                                np.linspace(0.01, -0.02, n),
                                np.linspace(0.0, 0.04, n)], axis=1)
    ax_full = mlx_tomo.Ax(VOL, geo, ANGLES, projection_type="interpolated")
    atb_full = mlx_tomo.Atb(PROJ, geo, ANGLES, backprojection_type="matched")
    pr._CHUNK_ELEMS = 3 * N_V * N_U
    try:
        ax_c = slabbed(5, lambda: mlx_tomo.Ax(
            VOL, geo, ANGLES, projection_type="interpolated"))
        atb_c = slabbed(5, lambda: mlx_tomo.Atb(
            PROJ, geo, ANGLES, backprojection_type="matched"))
    finally:
        pr._CHUNK_ELEMS = pr._MK_MAX_ELEMS
    d = rel_l2(ax_c, ax_full)
    check("Ax varying per-angle fields chunk+slab", d < 1e-6, f"d={d:.2e}")
    d = rel_l2(atb_c, atb_full)
    check("Atb varying per-angle fields chunk+slab", d < 1e-6, f"d={d:.2e}")


def test_degenerate_ray_hugging_cut_plane():
    """Parallel-beam rays with vect.z == 0 exactly, whose z sits within
    ~len*1e-9 voxels below a cut plane: the slab's z-clip must not drop
    their low-weight taps (regression for the sign-blind degenerate-
    slope clip; the pre-fix kernel loses most of the lower slab's
    contribution for such rays). The lower-slab partial is compared to
    the same data zero-padded to the full volume and projected unsplit —
    identical owed taps, no cut plane, so any clip defect is a
    large RELATIVE error on the probe row's (epsilon-sized) partial."""
    cut = 8
    geo = mlx_tomo.geometry(mode="parallel", nVoxel=np.array([16, 64, 64]))
    geo.nDetector = np.array([4, 64])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.accuracy = 0.02  # len = 1600 -> ulp band below the cut is wide
    angles = np.array([0.0])
    rng = np.random.default_rng(11)
    vol = rng.random((16, 64, 64)).astype(np.float32)
    padded = vol.copy()
    padded[cut:] = 0.0

    for k in (1, 2):  # k float32 ulps below the cut plane
        target = np.float32(cut)
        for _ in range(k):
            target = np.nextafter(target, np.float32(0.0))
        # solve offDetector[0] so output row 0's float32 S.z == target
        g = geo.copy()
        g.offDetector = np.zeros(2)
        for _ in range(4):
            gc = g.copy()
            gc.check_geo(angles)
            sz = float(pr.build_view_params(gc, "interpolated")[0, 11])
            err = float(target) - sz
            if err == 0.0:
                break
            g.offDetector = g.offDetector + np.array([err * float(g.dVoxel[0]), 0.0])
        gc = g.copy()
        gc.check_geo(angles)
        sz = float(pr.build_view_params(gc, "interpolated")[0, 11])
        band = 1600 * 1e-9
        check(f"probe row constructed in-band (k={k})",
              0.0 < cut - sz < band, f"S.z={sz!r} cut-S.z={cut - sz:.2e}")

        part = np.array(pr._ax_dispatch(
            mx.array(vol[:cut]), gc, "interpolated", 0))
        ref = mlx_tomo.Ax(padded, g, angles, projection_type="interpolated")
        # the probe row's partial is epsilon-sized, so compare it in its
        # own scale: the pre-fix clip loses most of it (d ~ 0.5); the
        # residual is guarded-vs-fast-path rounding over ~1600 samples
        row_d = rel_l2(part[0, 0], ref[0, 0])
        check(f"degenerate ray at cut plane (k={k} ulp)",
              row_d < 1e-4, f"probe-row d={row_d:.2e}")


def test_dtypes_and_containers():
    """float16 volumes and device-resident mx inputs through the slab path."""
    full16 = mlx_tomo.Ax(VOL.astype(np.float16), GEO, ANGLES,
                         projection_type="interpolated")
    got16 = slabbed(5, lambda: mlx_tomo.Ax(
        VOL.astype(np.float16), GEO, ANGLES, projection_type="interpolated"))
    d = rel_l2(got16, full16)
    check("Ax float16 slab == unsplit", d < 1e-6, f"d={d:.2e}")

    # mx input: the slab loop slices the device array (offset views)
    vol_mx = mx.array(VOL)
    full = mlx_tomo.Ax(VOL, GEO, ANGLES, projection_type="Siddon")
    got = slabbed(5, lambda: mlx_tomo.Ax(
        vol_mx, GEO, ANGLES, projection_type="Siddon", return_np=False))
    check("Ax mx-input slab returns mx", isinstance(got, mx.array))
    d = rel_l2(np.array(got), full)
    check("Ax mx-input slab == unsplit", d < 2e-5, f"d={d:.2e}")


if __name__ == "__main__":
    test_slab_vs_unsplit()
    test_zyz_and_parallel()
    test_slab_plus_view_chunking()
    test_vs_float64_reference()
    test_adjoint_with_slabs()
    test_impulse_on_cut_plane()
    test_per_angle_fields_through_chunk_and_slab()
    test_degenerate_ray_hugging_cut_plane()
    test_dtypes_and_containers()
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
