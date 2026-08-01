"""Validate the TIGRE-port reference projector against the analytic oracle.

Two independent constructions of the same geometry must agree:
- mlx_tomo.ref ports TIGRE's computeDeltas voxel-space pipeline;
- gen.rays_world builds world-space rays from the documented convention.

Checks: (1) ellipsoid projections agree to discretization error, (2) a
single bright voxel at a known world position lands on the predicted
detector pixel, for cone and parallel modes, several angles, offsets, COR.
"""

import numpy as np

from gen import analytic_projection, default_phantom_mm, ellipsoid_phantom, rays_world, rel_l2

import mlx_tomo
from mlx_tomo.ref import ax_ref

fails = 0


def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"{status} {name} {detail}")


def make_geo(mode="cone"):
    if mode == "cone":
        geo = mlx_tomo.geometry_default(high_resolution=False)  # 64^3, 128^2 det
    else:
        geo = mlx_tomo.geometry(mode="parallel", nVoxel=np.array([64, 64, 64]))
    return geo


def test_ellipsoids(mode, projection_type, tol):
    geo = make_geo(mode)
    geo.accuracy = 0.1  # fine sampling to isolate geometry errors
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ell = default_phantom_mm(geo)
    vol = ellipsoid_phantom(geo, ell).astype(np.float64)
    ref = ax_ref(vol, geo, angles, projection_type)
    ana = analytic_projection(geo, angles, ell)
    err = rel_l2(ref, ana)
    check(f"ellipsoid {mode}/{projection_type}", err < tol, f"rel_l2={err:.4f} tol={tol}")


def test_impulse(mode, projection_type):
    """A single voxel must project to the analytically predicted pixel."""
    geo = make_geo(mode)
    geo.accuracy = 0.2
    nz, ny, nx = (int(k) for k in geo.nVoxel)
    angles = np.array([0.0, 0.7, np.pi / 2, 2.5])

    # off-center voxel, known world position (center convention)
    iz, iy, ix = nz // 2 + 6, ny // 2 - 9, nx // 2 + 4
    wz = (iz + 0.5) * geo.dVoxel[0] - geo.sVoxel[0] / 2
    wy = (iy + 0.5) * geo.dVoxel[1] - geo.sVoxel[1] / 2
    wx = (ix + 0.5) * geo.dVoxel[2] - geo.sVoxel[2] / 2
    vol = np.zeros((nz, ny, nx))
    vol[iz, iy, ix] = 1.0

    proj = ax_ref(vol, geo, angles, projection_type)

    S, P = rays_world(geo, angles)
    n_v, n_u = proj.shape[1:]
    ok_all, worst = True, 0.0
    for i in range(len(angles)):
        pk = np.unravel_index(np.argmax(proj[i]), proj[i].shape)
        # Predict: intersect the ray from S through the voxel with the
        # detector plane. Solve in the rotated frame instead: express pixel
        # coords whose ray passes through the voxel.
        # Use brute force on the analytic ray grid: pixel whose ray passes
        # closest to the voxel position.
        w = np.array([wx, wy, wz])
        d = P[i] - S[i]
        t = ((w - S[i]) * d).sum(-1) / (d * d).sum(-1)
        closest = S[i] + t[..., None] * d
        dist = np.linalg.norm(closest - w, axis=-1)
        pred = np.unravel_index(np.argmin(dist), dist.shape)
        off = np.hypot(pk[0] - pred[0], pk[1] - pred[1])
        worst = max(worst, off)
        if off > 1.5:
            ok_all = False
    check(f"impulse {mode}/{projection_type}", ok_all, f"worst offset={worst:.2f} px")


def test_offsets_cor(projection_type):
    """Offsets and COR shift the image identically in both constructions."""
    geo = make_geo("cone")
    geo.accuracy = 0.1
    geo.offOrigin = np.array([6.0, -10.0, 8.0])   # (z, y, x) mm
    geo.offDetector = np.array([5.0, -7.0])       # (v, u) mm
    geo.COR = 3.0
    angles = np.array([0.3, 1.9, 4.0])
    ell = default_phantom_mm(geo)
    vol = ellipsoid_phantom(geo, ell).astype(np.float64)
    ref = ax_ref(vol, geo, angles, projection_type)
    ana = analytic_projection(geo, angles, ell)
    err = rel_l2(ref, ana)
    check(f"offsets+COR cone/{projection_type}", err < 0.03, f"rel_l2={err:.4f}")


if __name__ == "__main__":
    for mode in ("cone", "parallel"):
        for ptype in ("interpolated", "Siddon"):
            # Tolerance is voxelization error of the rasterized phantom vs
            # exact chords at 64^3 (~3%); geometry bugs show up as >>10%.
            test_ellipsoids(mode, ptype, tol=0.035)
            test_impulse(mode, ptype)
    test_offsets_cor("interpolated")
    test_offsets_cor("Siddon")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
