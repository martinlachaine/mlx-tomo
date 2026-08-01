"""FDK / FBP quantitative validation against the analytic oracle.

The oracle (gen.analytic_projection) produces EXACT line integrals of
ellipsoid phantoms, so recovered attenuation is compared to known
ground truth — the property that matters for a research comparator.

Strategy (decoupled):
1. Parallel FBP first: Radon inversion is exact, so interior recovery
   must be near-perfect (measured 3e-4 relative at these sizes) — any
   miss is a filter/padding/scale bug, gated tightly.
2. Cone FDK: exact in the central plane of a circular orbit; gated
   tightly there. Off-plane FDK error grows with cone angle — the
   at-scale error budget lives in harness/check_fdk.py; here the small
   test cone (<2 deg) must stay sub-percent.

Phantoms must fit the scan FOV (radius ~ nU/2*dU*DSO/DSD): truncated
projections legitimately inflate FBP/FDK values (verified ~19% for a
90 mm phantom in a 67 mm FOV) and would mask real bugs.
"""

import warnings

import numpy as np

from gen import analytic_projection, rel_l2

import mlx.core as mx

import mlx_tomo
from mlx_tomo.filtering import parker_weights, ramp_filter

fails = 0
MU = 0.02  # 1/mm


def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"{status} {name} {detail}")


def parallel_geo():
    geo = mlx_tomo.geometry(mode="parallel", nVoxel=np.array([16, 128, 128]))
    geo.nDetector = np.array([16, 128])
    geo.sDetector = geo.nDetector * geo.dDetector
    return geo


def cone_geo():
    geo = mlx_tomo.geometry_default(high_resolution=False)
    geo.nVoxel = np.array([64, 128, 128])
    geo.sVoxel = np.array([256.0, 256.0, 256.0])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array([128, 128])
    geo.dDetector = np.array([1.6, 1.6])
    geo.sDetector = geo.nDetector * geo.dDetector
    return geo


def disc_masks(geo, R, c=(0.0, 0.0), fov=None):
    """Interior disc and an exterior ring INSIDE the scan FOV (voxels
    beyond the FOV radius are not reconstructable and read nonzero)."""
    ny, nx = int(geo.nVoxel[1]), int(geo.nVoxel[2])
    y = (np.arange(ny) + 0.5) * geo.dVoxel[1] - geo.sVoxel[1] / 2 - c[1]
    x = (np.arange(nx) + 0.5) * geo.dVoxel[2] - geo.sVoxel[2] / 2 - c[0]
    r = np.hypot(x[None, :], y[:, None])
    if fov is None:
        fov = 0.5 * geo.sVoxel[2]  # parallel: detector spans the volume
    return r < 0.6 * R, (r > 1.25 * R) & (r < 0.95 * fov)


def test_fbp_parallel_exact():
    geo = parallel_geo()
    R = 40.0
    ells = [(MU, (0.0, 0.0, 0.0), (R, R, 1e4))]
    inner, outer = disc_masks(geo, R)
    for n_ang, arc in [(360, 2 * np.pi), (180, np.pi)]:
        angles = np.linspace(0, arc, n_ang, endpoint=False)
        proj = analytic_projection(geo, angles, ells).astype(np.float32)
        vol = mlx_tomo.FBP(proj, geo, angles)
        sl = vol[8]
        mean_r = sl[inner].mean() / MU
        max_r = np.abs(sl[inner] - MU).max() / MU
        ext = abs(sl[outer].mean()) / MU
        check(f"FBP {np.degrees(arc):.0f}deg interior mean",
              abs(mean_r - 1) < 5e-3, f"ratio={mean_r:.4f}")
        check(f"FBP {np.degrees(arc):.0f}deg interior pointwise",
              max_r < 0.02, f"max|err|/mu={max_r:.4f}")
        check(f"FBP {np.degrees(arc):.0f}deg exterior clean",
              ext < 5e-3, f"|mean|/mu={ext:.4f}")


def test_fbp_windows_and_cutoff():
    geo = parallel_geo()
    R = 40.0
    ells = [(MU, (0.0, 0.0, 0.0), (R, R, 1e4))]
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    proj = analytic_projection(geo, angles, ells).astype(np.float32)
    inner, _ = disc_masks(geo, R)
    for name in ("shepp_logan", "cosine", "hamming", "hann"):
        vol = mlx_tomo.FBP(proj, geo, angles, filter=name)
        r = vol[8][inner].mean() / MU
        check(f"FBP filter={name} interior mean", abs(r - 1) < 0.02,
              f"ratio={r:.4f}")
    vol = mlx_tomo.FBP(proj, geo, angles, d=0.5)
    r = vol[8][inner].mean() / MU
    check("FBP cutoff d=0.5 interior mean", abs(r - 1) < 0.03,
          f"ratio={r:.4f}")
    # geo.filter is honored when set (same result as explicit kwarg)
    geo2 = parallel_geo()
    geo2.filter = "hann"
    v1 = mlx_tomo.FBP(proj, geo2, angles)
    v2 = mlx_tomo.FBP(proj, geo, angles, filter="hann")
    check("geo.filter honored", rel_l2(v1, v2) == 0.0,
          f"d={rel_l2(v1, v2):.2e}")


def cone_fov(geo):
    return (geo.nDetector[1] / 2 * geo.dDetector[1]) * geo.DSO / geo.DSD


def test_fdk_cone_central_and_offsets():
    geo = cone_geo()
    ells = [(MU, (0.0, 0.0, 0.0), (42.0, 42.0, 58.0))]
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    proj = analytic_projection(geo, angles, ells).astype(np.float32)
    vol = mlx_tomo.FDK(proj, geo, angles)
    inner, outer = disc_masks(geo, 42.0, fov=cone_fov(geo))
    ratios = [vol[iz][inner].mean() / MU for iz in (24, 28, 32, 36, 40)]
    check("FDK central slice", abs(ratios[2] - 1) < 0.01,
          f"ratio={ratios[2]:.4f}")
    check("FDK near-plane slices (<2deg cone)",
          max(abs(r - 1) for r in ratios) < 0.015,
          f"ratios={[f'{r:.4f}' for r in ratios]}")
    check("FDK exterior clean", abs(vol[32][outer].mean()) / MU < 0.01,
          f"|mean|/mu={abs(vol[32][outer].mean())/MU:.4f}")

    # offOrigin + offDetector, phantom off-center
    geo2 = cone_geo()
    geo2.offOrigin = np.array([8.0, -12.0, 10.0])
    geo2.offDetector = np.array([6.0, -9.0])
    ells2 = [(MU, (8.0, -5.0, 4.0), (45.0, 48.0, 52.0))]
    proj2 = analytic_projection(geo2, angles, ells2).astype(np.float32)
    vol2 = mlx_tomo.FDK(proj2, geo2, angles)
    nz, ny, nx = vol2.shape
    z = (np.arange(nz) + 0.5) * geo2.dVoxel[0] - geo2.sVoxel[0] / 2 + geo2.offOrigin[0]
    y = (np.arange(ny) + 0.5) * geo2.dVoxel[1] - geo2.sVoxel[1] / 2 + geo2.offOrigin[1]
    x = (np.arange(nx) + 0.5) * geo2.dVoxel[2] - geo2.sVoxel[2] / 2 + geo2.offOrigin[2]
    c, ax = ells2[0][1], ells2[0][2]
    rr = (((x[None, None, :] - c[0]) / ax[0]) ** 2
          + ((y[None, :, None] - c[1]) / ax[1]) ** 2
          + ((z[:, None, None] - c[2]) / ax[2]) ** 2)
    r = vol2[rr < 0.25].mean() / MU
    check("FDK offOrigin+offDetector interior", abs(r - 1) < 0.015,
          f"ratio={r:.4f}")


def test_fdk_short_scan_parker():
    geo = cone_geo()
    ells = [(MU, (0.0, 0.0, 0.0), (42.0, 42.0, 58.0))]
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    proj = analytic_projection(geo, angles, ells).astype(np.float32)
    full = mlx_tomo.FDK(proj, geo, angles)

    fan = 2 * np.arctan((geo.nDetector[1] / 2 * geo.dDetector[1]) / geo.DSD)
    arc = np.pi + fan + 0.1
    n_s = int(round(360 * arc / (2 * np.pi)))
    angles_s = np.linspace(0, arc, n_s, endpoint=False)
    proj_s = analytic_projection(geo, angles_s, ells).astype(np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        short = mlx_tomo.FDK(proj_s, geo, angles_s)  # parker auto
        auto_warned = any("Parker" in str(x.message) for x in w)
    check("short scan auto-applies parker (warns)", auto_warned)
    inner, _ = disc_masks(geo, 42.0, fov=cone_fov(geo))
    r_full = full[32][inner].mean()
    r_short = short[32][inner].mean()
    check("short scan == full scan (central interior)",
          abs(r_short / r_full - 1) < 0.01,
          f"short/full={r_short/r_full:.4f}")
    check("short scan absolute mu", abs(r_short / MU - 1) < 0.015,
          f"ratio={r_short/MU:.4f}")
    # POINTWISE gate: the mirrored Atb distance weight shaded short
    # scans antisymmetrically by +-6% while every mean-based gate read
    # 1.00 — means must never be the only gate on short scans.
    pmax = np.abs(short[32][inner] - MU).max() / MU
    check("short scan pointwise interior", pmax < 0.02,
          f"max|err|/mu={pmax:.4f}")

    # parker=False on a short scan under-recovers (redundancy unweighted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noparker = mlx_tomo.FDK(proj_s, geo, angles_s, parker=False)
    r_np = noparker[32][inner].mean() / MU
    check("short scan without parker under-recovers", 0.4 < r_np < 0.75,
          f"ratio={r_np:.4f}")

    # weight properties: range, boundary taper, smoothness. The Wesarg
    # transitions are steep (width ~ q*(2*delta - 2*alpha + epsilon)),
    # so gate CONVERGENCE under view refinement — a sign/indexing bug
    # gives hard 0<->1 jumps that do not shrink with denser sampling.
    g = geo.copy()
    g.check_geo(angles_s)
    w = parker_weights(g, g.angles, 1.0)
    check("parker weights in [0,1]",
          w.min() > -1e-9 and w.max() < 1 + 1e-9,
          f"range=[{w.min():.3f},{w.max():.3f}]")
    step1 = np.abs(np.diff(w, axis=0)).max()
    g4 = geo.copy()
    g4.check_geo(np.linspace(0, arc, 4 * n_s, endpoint=False))
    step4 = np.abs(np.diff(parker_weights(g4, g4.angles, 1.0), axis=0)).max()
    check("parker weights smooth (refine 4x -> steps ~/4)",
          step1 < 0.5 and step4 < 0.35 * step1,
          f"max step {step1:.3f} -> {step4:.3f}")
    mid_u = w.shape[1] // 2
    check("parker taper at scan ends",
          w[0, mid_u] < 0.05 and w[-1, mid_u] < 0.35,
          f"w[0]={w[0, mid_u]:.3f} w[-1]={w[-1, mid_u]:.3f}")


def test_filtering_function():
    geo = cone_geo()
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    rng = np.random.default_rng(0)
    proj = rng.random((8, 128, 128)).astype(np.float32)

    # numpy in -> numpy out; mx in -> mx out; input not mutated
    keep = proj.copy()
    out = mlx_tomo.filtering(proj, geo, angles)
    check("filtering returns numpy for numpy", isinstance(out, np.ndarray))
    check("filtering does not mutate input", np.array_equal(proj, keep))
    out_mx = mlx_tomo.filtering(mx.array(proj), geo, angles)
    check("filtering returns mx for mx", isinstance(out_mx, mx.array))
    check("filtering mx == np path", rel_l2(np.array(out_mx), out) == 0.0)

    # a constant sinogram is a rect after zero-padding, so the filtered
    # result is edge response, not DC leakage; a wrong-DC filter would
    # read ~1x the reference, correct behavior reads well under 10%
    const = np.ones((4, 128, 128), dtype=np.float32)
    fc = mlx_tomo.filtering(const, geo, angles[:4])
    ref = np.abs(mlx_tomo.filtering(
        rng.random((1, 128, 128)).astype(np.float32), geo, angles[:1])).max()
    check("constant sinogram ~suppressed", np.abs(fc).max() < 0.1 * ref,
          f"max|filt(const)|={np.abs(fc).max():.2e} vs {ref:.2e}")

    # frequency response (rfft half-spectrum, length flen//2+1; Nyquist
    # gain 1.0). The spatial-kernel route's signature: DC is small but
    # strictly POSITIVE, ~4/(pi^2*flen) from kernel truncation — the
    # term a naive |omega| filter loses (causing cupping).
    flen = 1024
    f = ramp_filter("ram_lak", flen)
    k = np.arange(len(f))
    check("Nyquist gain ~1 (minus truncation deficit 4/pi^2/flen)",
          abs(f[-1] - 1.0) < 2.0 / flen, f"f[-1]={f[-1]:.6f}")
    check("DC small and positive", 0 < f[0] < 1.0 / flen,
          f"f[0]={f[0]:.2e} (theory {4/np.pi**2/flen:.2e})")
    lin = f[16:513] / (2 * k[16:513] / flen)
    check("ramp midband linear", np.abs(lin - 1).max() < 0.01,
          f"max dev={np.abs(lin-1).max():.4f}")
    fh = ramp_filter("hann", flen)
    check("hann tapers to 0 at Nyquist", fh[-1] < 1e-9 and fh[1] > 0)
    fd = ramp_filter("ram_lak", flen, d=0.5)
    check("cutoff d zeroes above pi*d", fd[257:].max() == 0.0
          and fd[250] > 0, f"fd[257:].max()={fd[257:].max():.1e}")

    for bad in ("gauss", "ramlak"):
        try:
            mlx_tomo.filtering(proj, geo, angles, filter=bad)
            check(f"unknown filter {bad!r} raises", False)
        except ValueError:
            check(f"unknown filter {bad!r} raises", True)
    try:
        mlx_tomo.filtering(proj, geo, angles, d=1.5)
        check("bad cutoff raises", False)
    except ValueError:
        check("bad cutoff raises", True)


def test_scan_direction_and_wrapped_angles():
    """Descending (clockwise) short scans need the mirrored Parker fan
    sign, and only an OFF-CENTER phantom can tell (a centered one is
    mirror-symmetric in u). Wrapped mod-2*pi angle arrays must not
    inflate the measured step or trigger auto-Parker."""
    geo = cone_geo()
    ells = [(MU, (20.0, -15.0, 0.0), (32.0, 32.0, 50.0))]
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    proj = analytic_projection(geo, angles, ells).astype(np.float32)
    full = mlx_tomo.FDK(proj, geo, angles)
    # interior mask at the phantom position
    y = (np.arange(128) + 0.5) * geo.dVoxel[1] - 128.0 + 15.0
    x = (np.arange(128) + 0.5) * geo.dVoxel[2] - 128.0 - 20.0
    inner = np.hypot(x[None, :], y[:, None]) < 0.6 * 32.0
    ref = full[32][inner].mean()

    fan = 2 * np.arctan((geo.nDetector[1] / 2 * geo.dDetector[1]) / geo.DSD)
    arc = np.pi + fan + 0.1
    n_s = int(round(360 * arc / (2 * np.pi)))
    got = {}
    for tag, ang_s in (("ascending", np.linspace(0, arc, n_s, endpoint=False)),
                       ("descending", np.linspace(arc, 0, n_s, endpoint=False))):
        proj_s = analytic_projection(geo, ang_s, ells).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vol_s = mlx_tomo.FDK(proj_s, geo, ang_s)
        got[tag] = vol_s[32][inner].mean()
        # (with the mirrored Atb distance weight this read 1.038; the
        # residual "Parker approximation" was that bug, not Parker)
        check(f"{tag} short scan off-center vs full",
              abs(got[tag] / ref - 1) < 0.01, f"short/full={got[tag]/ref:.4f}")
    # mirror symmetry: the wrong fan-angle sign for descending scans
    # breaks this by ~2% (and 3x the peak error); correct code ~0.01%
    d = abs(got["descending"] / got["ascending"] - 1)
    check("descending == mirrored ascending (Parker fan sign)",
          d < 0.005, f"desc/asc={got['descending']/got['ascending']:.4f}")

    # wrapped full scan: same recon as unwrapped, no parker warning
    wrapped = (angles + np.pi) % (2 * np.pi)
    proj_w = analytic_projection(geo, wrapped, ells).astype(np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vol_w = mlx_tomo.FDK(proj_w, geo, wrapped)
        parkered = any("Parker" in str(x.message) for x in w)
    check("wrapped angles: no spurious auto-parker", not parkered)
    got = vol_w[32][inner].mean()
    check("wrapped angles: correct scale", abs(got / ref - 1) < 0.005,
          f"wrapped/full={got/ref:.4f}")


def test_trajectory_cor_and_anisotropy():
    """(1) A (0, theta, 0) ZYZ orbit must not be misread as a 0-degree
    short scan (auto-Parker would zero every weight and return an
    all-zero volume). (2) COR end-to-end: Atb's COR handling never had
    an oracle-based check, and COR must NOT enter the FDK distance
    weight (it shifts source and detector together). (3) Anisotropic
    detector pixels: a dV/dU mix-up in the 1/(4*dU) filter scale would
    rescale mu by exactly dV/dU."""
    # (1) rotation living in the theta column: full-circle coverage
    geo = cone_geo()
    ells = [(MU, (0.0, 0.0, 0.0), (42.0, 42.0, 42.0))]
    zyz = np.stack([np.zeros(72),
                    np.linspace(0, 2 * np.pi, 72, endpoint=False),
                    np.zeros(72)], axis=1)
    proj = analytic_projection(geo, zyz, ells).astype(np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vol = mlx_tomo.FDK(proj, geo, zyz)
        parkered = any("Parker" in str(x.message) for x in w)
    check("ZYZ theta-orbit: no spurious auto-parker", not parkered)
    check("ZYZ theta-orbit: volume not zeroed", float(np.abs(vol).max()) > 0.5 * MU,
          f"max={np.abs(vol).max():.4f}")

    # (2) COR != 0, full scan, off-center phantom, pointwise gate
    geo = cone_geo()
    geo.COR = 4.0
    ells = [(MU, (18.0, -12.0, 0.0), (34.0, 34.0, 48.0))]
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    proj = analytic_projection(geo, angles, ells).astype(np.float32)
    vol = mlx_tomo.FDK(proj, geo, angles)
    y = (np.arange(128) + 0.5) * geo.dVoxel[1] - 128.0 + 12.0
    x = (np.arange(128) + 0.5) * geo.dVoxel[2] - 128.0 - 18.0
    inner = np.hypot(x[None, :], y[:, None]) < 0.6 * 34.0
    r = vol[32][inner].mean() / MU
    pmax = np.abs(vol[32][inner] - MU).max() / MU
    check("COR=4mm full scan interior mean", abs(r - 1) < 0.01,
          f"ratio={r:.4f}")
    check("COR=4mm full scan pointwise", pmax < 0.03,
          f"max|err|/mu={pmax:.4f}")

    # (3) anisotropic detector (dV != dU), parallel FBP
    geo = mlx_tomo.geometry(mode="parallel", nVoxel=np.array([16, 128, 128]))
    geo.nDetector = np.array([12, 128])
    geo.dDetector = np.array([2.4, 1.0])
    geo.sDetector = geo.nDetector * geo.dDetector
    ells = [(MU, (0.0, 0.0, 0.0), (40.0, 40.0, 1e4))]
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    proj = analytic_projection(geo, angles, ells).astype(np.float32)
    vol = mlx_tomo.FBP(proj, geo, angles)
    inner = disc_masks(geo, 40.0)[0]
    r = vol[8][inner].mean() / MU
    check("anisotropic detector (dV=2.4,dU=1.0) mu", abs(r - 1) < 0.01,
          f"ratio={r:.4f} (a dV/dU scale mix-up would read ~2.4)")


def test_cosine_weight_matches_oracle_rays():
    """The FDK pre-weight DSD/sqrt(DSD^2+u^2+v^2) must equal cos(gamma)
    of the actual source->pixel ray bundle from the analytic oracle —
    pins the (v,u) field ordering, the offDetector sign, the
    piercing-point convention, and that COR (which shifts source and
    detector together) does not enter."""
    from gen import _euler_zyz_mat, rays_world

    from mlx_tomo.algorithms import _cosine_weight

    geo = cone_geo()
    geo.offDetector = np.array([11.0, -7.0])  # (v, u) mm, asymmetric
    geo.COR = 3.0
    angles = np.array([0.0, 0.9, 2.2])
    g = geo.copy()
    g.check_geo(angles)
    w = np.array(_cosine_weight(g))  # (1, nV, nU): offsets constant
    S, P = rays_world(geo, angles)
    worst = 0.0
    for i in range(len(angles)):
        d = P[i] - S[i]
        dn = d / np.linalg.norm(d, axis=-1, keepdims=True)
        n_hat = _euler_zyz_mat(*g.angles[i]) @ np.array([-1.0, 0.0, 0.0])
        worst = max(worst, np.abs(w[0] - dn @ n_hat).max())
    check("cosine weight == oracle cos(gamma)", worst < 1e-6,
          f"max|err|={worst:.2e}")


def test_api_conventions():
    geo = parallel_geo()
    R = 40.0
    ells = [(MU, (0.0, 0.0, 0.0), (R, R, 1e4))]
    angles = np.linspace(0, np.pi, 60, endpoint=False)
    proj = analytic_projection(geo, angles, ells).astype(np.float32)

    # FDK on parallel dispatches to FBP
    v1 = mlx_tomo.FDK(proj, geo, angles)
    v2 = mlx_tomo.FBP(proj, geo, angles)
    check("FDK(parallel) == FBP", rel_l2(v1, v2) == 0.0)

    # mx passthrough
    vm = mlx_tomo.FDK(mx.array(proj), geo, angles)
    check("FDK mx in -> mx out", isinstance(vm, mx.array))
    check("FDK mx == np", rel_l2(np.array(vm), v1) == 0.0)
    vn = mlx_tomo.FDK(mx.array(proj), geo, angles, return_np=True)
    check("FDK return_np override", isinstance(vn, np.ndarray))

    # dtype policing
    try:
        mlx_tomo.FDK(proj.astype(np.float64), geo, angles)
        check("float64 proj raises", False)
    except TypeError:
        check("float64 proj raises", True)

    # FBP refuses cone geometry
    cg = cone_geo()
    try:
        mlx_tomo.FBP(proj, cg, angles)
        check("FBP cone raises", False)
    except ValueError:
        check("FBP cone raises", True)

    # lowercase TIGRE aliases
    check("fdk/fbp aliases", mlx_tomo.fdk is mlx_tomo.FDK
          and mlx_tomo.fbp is mlx_tomo.FBP)


if __name__ == "__main__":
    test_fbp_parallel_exact()
    test_fbp_windows_and_cutoff()
    test_fdk_cone_central_and_offsets()
    test_fdk_short_scan_parker()
    test_scan_direction_and_wrapped_angles()
    test_trajectory_cor_and_anisotropy()
    test_filtering_function()
    test_cosine_weight_matches_oracle_rays()
    test_api_conventions()
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
