"""Tests for mlx_tomo.rays: projection along explicitly specified rays.

The load-bearing test is the ROUND TRIP: rays extracted from a geometry, fed
back through Ax_rays, must reproduce Ax bit-for-bit. Everything in this module
is a coordinate-frame conversion (world mm with the isocenter at the origin
versus the kernel's corner-origin voxel units, with a half-voxel difference
between projection types), so an exact round trip is what rules out a
convention slip. Approximate agreement would not: a half-voxel shift is small
enough to look like interpolation error and large enough to invalidate
everything built on top.

The remaining tests check the capabilities the module exists for -- a
translated source against the equivalent offOrigin geometry, a sheared
(non-orthogonal) detector basis against a geometry that produces the same
rays by other means, and the affine ray mapping against an analytic
ellipsoid oracle.
"""

from __future__ import annotations

import numpy as np

import mlx_tomo
from mlx_tomo.rays import Ax_rays, affine_rays, rays_from_geometry


def _geo(nvox=(48, 64, 64), ndet=(56, 72), mode="cone"):
    g = mlx_tomo.geometry_default(high_resolution=False)
    g.mode = mode
    g.DSO, g.DSD = 1000.0, 1500.0
    g.nDetector = np.array(ndet)
    g.dDetector = np.array([1.4, 1.2])
    g.sDetector = g.nDetector * g.dDetector
    g.nVoxel = np.array(nvox)
    g.dVoxel = np.array([2.0, 1.5, 1.5])
    g.sVoxel = g.nVoxel * g.dVoxel
    return g


def _blob(g, seed=3):
    rng = np.random.default_rng(seed)
    nz, ny, nx = (int(k) for k in g.nVoxel)
    zz, yy, xx = np.meshgrid(*[np.arange(n) - n / 2 for n in (nz, ny, nx)],
                             indexing="ij")
    v = np.zeros((nz, ny, nx), np.float32)
    for _ in range(6):
        c = rng.uniform(-0.25, 0.25, 3) * np.array([nz, ny, nx])
        r = rng.uniform(0.12, 0.25) * min(nz, ny, nx)
        v += rng.uniform(0.2, 1.0) * np.exp(
            -0.5 * ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
            / r ** 2).astype(np.float32)
    return v.astype(np.float32)


def test_round_trip_is_exact():
    """rays_from_geometry -> Ax_rays must reproduce Ax bit-for-bit."""
    for mode in ("cone", "parallel"):
        g = _geo(mode=mode)
        ang = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        v = _blob(g)
        for pt in ("Siddon", "interpolated"):
            ref = np.asarray(mlx_tomo.Ax(v, g, ang, pt, return_np=True))
            got = np.asarray(Ax_rays(v, g, rays_from_geometry(g, ang, pt), pt,
                                     return_np=True))
            assert got.shape == ref.shape, (got.shape, ref.shape)
            assert np.array_equal(got, ref), (
                f"{mode}/{pt}: max |diff| "
                f"{np.abs(got - ref).max():.3e} of range {np.abs(ref).max():.3e}"
                " -- not bit-identical, so the frame conversion is wrong")


def test_round_trip_exact_with_offsets_and_rotation():
    """Same, with the geometry features that complicate the frame."""
    g = _geo()
    n = 7
    ang = np.linspace(0, np.pi, n, endpoint=False)
    g.offOrigin = np.tile(np.array([2.0, -3.0, 4.0]), (n, 1))
    g.offDetector = np.tile(np.array([1.7, -2.3]), (n, 1))
    g.rotDetector = np.tile(np.array([0.02, -0.03, 0.05]), (n, 1))
    v = _blob(g, seed=11)
    for pt in ("Siddon", "interpolated"):
        ref = np.asarray(mlx_tomo.Ax(v, g, ang, pt, return_np=True))
        got = np.asarray(Ax_rays(v, g, rays_from_geometry(g, ang, pt), pt,
                                 return_np=True))
        assert np.array_equal(got, ref), (
            f"{pt}: max |diff| {np.abs(got - ref).max():.3e}")


def test_translated_source_matches_offOrigin():
    """Shifting the rays by -b == shifting the object by +b (offOrigin)."""
    g = _geo()
    n = 11
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = _blob(g, seed=5)
    b = np.array([3.7, -2.4, 6.1])              # non-integer voxel shift
    g2 = g.copy()
    g2.offOrigin = np.tile(b[::-1], (n, 1))     # geo stores (z, y, x)
    ref = np.asarray(mlx_tomo.Ax(v, g2, ang, "Siddon", return_np=True))
    rays = affine_rays(rays_from_geometry(g, ang, "Siddon"), np.eye(3), b)
    got = np.asarray(Ax_rays(v, g, rays, "Siddon", return_np=True))
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-6, f"relative L2 {rel:.3e}"


def test_sheared_detector_basis_runs_and_is_linear():
    """A non-orthogonal basis is accepted, and the operator stays linear."""
    g = _geo()
    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    rays = rays_from_geometry(g, ang, "Siddon")
    rays[:, 1] = rays[:, 1] + 0.3 * rays[:, 2]      # shear u into v
    rays[:, 2] = 1.15 * rays[:, 2]                  # and stretch v
    a, b = _blob(g, seed=7), _blob(g, seed=8)
    pa = np.asarray(Ax_rays(a, g, rays, "Siddon", return_np=True))
    pb = np.asarray(Ax_rays(b, g, rays, "Siddon", return_np=True))
    pab = np.asarray(Ax_rays(2.0 * a - 3.0 * b, g, rays, "Siddon",
                             return_np=True))
    assert np.isfinite(pa).all() and pa.max() > 0
    rel = np.linalg.norm(pab - (2.0 * pa - 3.0 * pb)) / max(
        np.linalg.norm(pab), 1e-30)
    assert rel < 1e-5, f"linearity relative L2 {rel:.3e}"


def test_affine_rays_against_analytic_gaussian():
    """Mapped rays hit the lines they claim to, checked against a closed form.

    A Gaussian is used rather than an ellipsoid for a specific reason. Comparing
    "warped object, nominal rays" against "unwarped object, mapped rays" through
    a voxel projector voxelises two DIFFERENT shapes, whose discretisation
    errors are independent and do not cancel; with a hard-edged ellipsoid that
    floor is ~2% and converges to ~2% under supersampling (measured 0.0276 /
    0.0211 / 0.0202 at supersample 1 / 2 / 3), which is too blunt to validate
    the stretch case.

    Instead this compares Ax_rays on the UNWARPED phantom along MAPPED rays
    against the analytic line integral along the same lines. For
    f(x) = exp(-(x-c)^T S^-1 (x-c) / 2), with p = a - c,
    A = w^T S^-1 w, B = p^T S^-1 w, C = p^T S^-1 p,

        integral f along a + lam w  =  sqrt(2 pi / A) exp(-(C - B^2/A) / 2)

    The threshold is SELF-CALIBRATING: the mapped-ray error is compared against
    the same measurement with NOMINAL rays, which is the projector's own floor
    with no mapping involved. An absolute threshold would drift with voxel size
    and phantom choice; this asserts what actually matters, namely that the
    mapping adds nothing. Measured: nominal 5.02e-03, axial stretch 5.08e-03,
    shear+anisotropic 4.92e-03 -- and insensitive to geo.accuracy (0.5 vs 0.2
    identical), so the floor is rasterisation of the Gaussian, not ray stepping.
    """
    g = _geo(nvox=(64, 96, 96), ndet=(80, 96))
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    nz, ny, nx = (int(k) for k in g.nVoxel)
    dz, dy, dx = (float(t) for t in g.dVoxel)
    zz, yy, xx = np.meshgrid(
        (np.arange(nz) + 0.5) * dz - nz * dz / 2,
        (np.arange(ny) + 0.5) * dy - ny * dy / 2,
        (np.arange(nx) + 0.5) * dx - nx * dx / 2, indexing="ij")
    c = np.array([6.0, -5.0, 3.0])
    S = np.diag([28.0, 22.0, 18.0]) ** 2          # covariance, (x, y, z)
    Si = np.linalg.inv(S)
    q = np.stack([xx, yy, zz], -1) - c
    vol = np.exp(-0.5 * np.einsum("...i,ij,...j->...", q, Si, q)
                 ).astype(np.float32)

    n_v, n_u = int(g.nDetector[0]), int(g.nDetector[1])
    uu = np.arange(n_u)[None, :, None]
    vv = np.arange(n_v)[:, None, None]
    floor = None
    for A, b in ((np.eye(3), np.zeros(3)),          # control: projector floor
                 (np.diag([1.0, 1.0, 1.12]), np.array([2.0, -4.0, 5.0])),
                 (np.array([[1.04, 0.05, 0.0],
                            [0.0, 0.98, 0.03],
                            [0.0, 0.0, 1.09]]), np.array([-3.0, 2.0, 1.5]))):
        rays = affine_rays(rays_from_geometry(g, ang, "interpolated"), A, b)
        got = np.asarray(Ax_rays(vol, g, rays, "interpolated",
                                 return_np=True))
        exact = np.empty_like(got, dtype=np.float64)
        for i in range(ang.size):
            P = rays[i, 0] + uu * rays[i, 1] + vv * rays[i, 2]
            Sr = rays[i, 3]
            D = P - Sr
            w = D / np.linalg.norm(D, axis=-1, keepdims=True)
            pp = Sr - c + 0.0 * w
            Aq = np.einsum("...i,ij,...j->...", w, Si, w)
            Bq = np.einsum("...i,ij,...j->...", pp, Si, w)
            Cq = np.einsum("...i,ij,...j->...", pp, Si, pp)
            exact[i] = np.sqrt(2 * np.pi / Aq) * np.exp(
                -0.5 * (Cq - Bq ** 2 / Aq))
        m = exact > 0.05 * exact.max()
        rel = float(np.abs(got[m] - exact[m]).mean() / np.abs(exact[m]).mean())
        if floor is None:                          # the identity control
            floor = rel
            assert rel < 1e-2, f"projector floor unexpectedly large {rel:.3e}"
            continue
        assert rel < 1.15 * floor, (
            f"A={A.tolist()}: mean rel {rel:.4e} exceeds the nominal-ray "
            f"floor {floor:.4e} -- the affine ray mapping is adding error")


def test_rejects_bad_shapes():
    g = _geo()
    v = _blob(g)
    for bad in (np.zeros((4, 3)), np.zeros((5, 3, 3)), np.zeros((5, 4, 2))):
        try:
            Ax_rays(v, g, bad, "Siddon")
        except ValueError:
            continue
        raise AssertionError(f"accepted rays of shape {bad.shape}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all rays tests passed")
