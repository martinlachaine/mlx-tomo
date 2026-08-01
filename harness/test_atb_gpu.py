"""GPU backprojector vs float64 reference, plus adjoint pairing checks.

The matched backprojector is TIGRE's voxel-driven approximate adjoint of
the interpolated forward projector — the discretizations differ, so the
inner-product identity <Ax, y> = <x, Atb y> holds to ~1%, not machine
precision. GPU-vs-reference parity is the tight gate (float32 floor).
"""

import numpy as np

from gen import default_phantom_mm, ellipsoid_phantom, rel_l2

import mlx_tomo
from mlx_tomo.ref import atb_ref

fails = 0


def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"{status} {name} {detail}")


def run_case(mode, bptype, with_offsets=False):
    if mode == "cone":
        geo = mlx_tomo.geometry_default(high_resolution=False)
    else:
        geo = mlx_tomo.geometry(mode="parallel", nVoxel=np.array([32, 48, 64]))
        geo.nDetector = np.array([48, 96])
        geo.sDetector = geo.nDetector * geo.dDetector
    if with_offsets:
        geo.offOrigin = np.array([5.0, -8.0, 3.0])
        geo.offDetector = np.array([4.0, -6.0])
        geo.COR = 2.0 if mode == "cone" else 0.0
        geo.rotDetector = np.array([0.02, -0.03, 0.05]) if mode == "cone" else np.zeros(3)
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)

    rng = np.random.default_rng(3)
    n_v, n_u = (int(k) for k in geo.nDetector)
    proj = rng.random((len(angles), n_v, n_u)).astype(np.float32)

    ref = atb_ref(proj.astype(np.float64), geo, angles, bptype)
    got = mlx_tomo.Atb(proj, geo, angles, backprojection_type=bptype)
    err = rel_l2(got, ref)
    tag = f"{mode}/{bptype}" + ("+off" if with_offsets else "")
    check(f"atb gpu-vs-ref {tag}", err < 2e-4, f"rel_l2={err:.2e}")


def adjoint_ratio(mode):
    """<Ax, y> / <x, Atb y> for the interpolated/matched pair."""
    if mode == "cone":
        geo = mlx_tomo.geometry_default(high_resolution=False)
    else:
        geo = mlx_tomo.geometry(mode="parallel", nVoxel=np.array([64, 64, 64]))
    geo.accuracy = 0.25
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x = ellipsoid_phantom(geo, default_phantom_mm(geo)).astype(np.float32)
    rng = np.random.default_rng(5)
    n_v, n_u = (int(k) for k in geo.nDetector)
    y = rng.random((len(angles), n_v, n_u)).astype(np.float32)

    ax = mlx_tomo.Ax(x, geo, angles, projection_type="interpolated")
    aty = mlx_tomo.Atb(y, geo, angles, backprojection_type="matched")
    num = float((ax.astype(np.float64) * y).sum())
    den = float((x.astype(np.float64) * aty).sum())
    # TIGRE's matched pair is the adjoint of the CONTINUOUS operators:
    # <Ax,y> du dv = <x, Atb y> dVol. In plain l2 the ratio is therefore
    # voxel-volume / pixel-area; ~1% residual is discretization mismatch.
    measure = float(np.prod(geo.dVoxel) / np.prod(geo.dDetector))
    r = (num / den) / measure
    check(f"adjoint ratio {mode}", 0.95 < r < 1.05,
          f"normalized <Ax,y>/<x,Atby>={r:.4f} (measure={measure:.3f})")


def regressions():
    """Pin the fixes from the adversarial review."""
    import warnings

    geo = mlx_tomo.geometry_default(high_resolution=False)
    angles = np.array([0.0, 1.0])
    n_v, n_u = (int(k) for k in geo.nDetector)
    proj = np.random.default_rng(1).random((2, n_v, n_u)).astype(np.float32)

    # legacy TIGRE<=2.0 krylov= must be honored, not dropped
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = mlx_tomo.Atb(proj, geo, angles, krylov="matched")
        b = mlx_tomo.Atb(proj, geo, angles, backprojection_type="matched")
    check("krylov= honored as alias", rel_l2(a, b) == 0.0)

    # NaN geometry must raise ValueError, not compile 'nanf' or emit NaNs
    bad = mlx_tomo.geometry_default(high_resolution=False)
    bad.sVoxel = np.array([np.nan, 256.0, 256.0])
    bad.dVoxel = bad.sVoxel / bad.nVoxel
    try:
        mlx_tomo.Atb(proj, bad, angles)
        check("NaN geometry rejected", False, "no exception")
    except ValueError:
        check("NaN geometry rejected", True)
    bad2 = mlx_tomo.geometry_default(high_resolution=False)
    bad2.DSD = np.nan
    try:
        mlx_tomo.Ax(np.zeros((64, 64, 64), dtype=np.float32), bad2, angles)
        check("NaN DSD rejected", False, "no exception")
    except ValueError:
        check("NaN DSD rejected", True)


if __name__ == "__main__":
    for mode in ("cone", "parallel"):
        for bptype in ("FDK", "matched"):
            run_case(mode, bptype)
    run_case("cone", "FDK", with_offsets=True)
    run_case("cone", "matched", with_offsets=True)
    adjoint_ratio("cone")
    adjoint_ratio("parallel")
    regressions()
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
