"""Pure-NumPy float64 reference forward projector.

This module is the accuracy oracle for the Metal kernels. It implements the
projector geometry that TIGRE defines (computeDeltas, computeDeltas_parallel)
and evaluates the same sampling rules in float64:

- ``interpolated``: fixed-step ray marching with trilinear interpolation,
  voxel-center convention (TIGRE translates by sVoxel/2 - dVoxel/2 and
  samples ``tex3D(t + 0.5)``; net effect: voxel ``j`` is centered at
  normalized coordinate ``j``).
- ``Siddon``: exact cell-constant radiological path (Jacobs/Siddon),
  voxel-corner convention (cell ``j`` spans ``[j, j+1)``).

offDetector is applied once, in all modes. See https://github.com/martinlachaine/mlx-tomo/blob/main/docs/compatibility.md for the
behavioral differences from TIGRE.

This implementation prioritizes clarity and numerical traceability over
performance and is intended for small validation problems.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ax_ref", "view_geometry"]


def _roll_pitch_yaw(rot, pts):
    """TIGRE rollPitchYaw: rot = (yaw, pitch, roll) as stored in rotDetector."""
    yaw, pitch, roll = rot[0], rot[1], rot[2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cr * cp, cr * sp * sy - sr * cy, cr * sp * cy + sr * sy],
        [sr * cp, sr * sp * sy + cr * cy, sr * sp * cy - cr * sy],
        [-sp, cp * sy, cp * cy],
    ])
    return pts @ R.T


def _euler_zyz(abc, pts):
    """TIGRE eulerZYZ with angles (alpha, theta, psi)."""
    a, t, p = abc
    ca, sa = np.cos(a), np.sin(a)
    ct, st = np.cos(t), np.sin(t)
    cp, sp = np.cos(p), np.sin(p)
    R = np.array([
        [ca * ct * cp - sa * sp, -ca * ct * sp - sa * cp, ca * st],
        [sa * ct * cp + ca * sp, -sa * ct * sp + ca * cp, sa * st],
        [-st * cp, st * sp, ct],
    ])
    return pts @ R.T


def view_geometry(geo, i, projection_type="interpolated"):
    """Per-view ray geometry in normalized (voxel-scaled) coordinates.

    Returns (uvOrigin, deltaU, deltaV, source) as float64 length-3 arrays,
    ordered (x, y, z). ``geo`` must already be validated via check_geo.

    Follows TIGRE's computeDeltas (cone) / computeDeltas_parallel exactly,
    with the half-voxel shift chosen by projection_type:
    interpolated -> center convention, Siddon -> corner convention.
    """
    dsd, dso = geo.DSD[i], geo.DSO[i]
    # dVoxel/sVoxel stored (z,y,x) -> to (x,y,z)
    dvox = geo.dVoxel[::-1].astype(np.float64)
    svox = geo.sVoxel[::-1].astype(np.float64)
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    d_v, d_u = float(geo.dDetector[0]), float(geo.dDetector[1])
    off_v, off_u = geo.offDetector[i][0], geo.offDetector[i][1]
    off_orig = geo.offOrigin[i][::-1].astype(np.float64)  # -> (x,y,z)
    alpha = geo.angles[i]
    cor = geo.COR[i]

    parallel = geo.mode == "parallel"

    # Pixel (u=0, v_row=nV-1  i.e. pixelV index 0 in TIGRE's flipped count)
    # P: detector pixel (0,0); Pu0/Pv0: neighbours along u and v.
    px = -(dsd - dso)
    P = np.array([px, d_u * (-n_u / 2.0 + 0.5), d_v * (n_v / 2.0 - 0.5)])
    Pu = np.array([px, d_u * (-n_u / 2.0 + 1.5), d_v * (n_v / 2.0 - 0.5)])
    Pv = np.array([px, d_u * (-n_u / 2.0 + 0.5), d_v * (n_v / 2.0 - 1.5)])

    if parallel:
        S = np.array([dso, P[1], P[2]])
    else:
        S = np.array([dso, 0.0, 0.0])

    # Detector roll/pitch/yaw around the detector center (x set to 0).
    pts = np.stack([P, Pu, Pv])
    pts[:, 0] = 0.0
    pts = _roll_pitch_yaw(geo.rotDetector[i], pts)
    pts[:, 0] += px
    if parallel:
        # The parallel source sits on the detector normal through the pixel;
        # it rotates with the detector.
        S = _roll_pitch_yaw(geo.rotDetector[i], np.array([[0.0, S[1], S[2]]]))[0]
        S[0] += dso

    # Detector offset (applied once; see module docstring).
    pts[:, 1] += off_u
    pts[:, 2] += off_v
    if parallel:
        S[1] += off_u
        S[2] += off_v

    # Gantry rotation (ZYZ Euler) applied to detector points and source.
    allpts = np.vstack([pts, S[None]])
    allpts = _euler_zyz(alpha, allpts)

    # Move image offset onto everything else, then shift origin to the
    # volume corner and scale to voxel units.
    allpts -= off_orig
    if projection_type == "interpolated":
        allpts += (svox / 2.0 - dvox / 2.0)
    else:  # Siddon: corner convention
        allpts += svox / 2.0
    allpts /= dvox

    # Center-of-rotation correction.
    cshift = np.array([-cor * np.sin(alpha[0]) / dvox[0],
                       cor * np.cos(alpha[0]) / dvox[1], 0.0])
    allpts += cshift

    P, Pu, Pv, S = allpts
    return P, Pu - P, Pv - P, S


def _trilinear_border0(vol_xyz, pts):
    """Trilinear interpolation, voxel j centered at coordinate j.

    vol_xyz is indexed [x, y, z]; pts is (..., 3) in normalized coords.
    Out-of-range taps contribute 0 (CUDA border addressing).
    """
    nx, ny, nz = vol_xyz.shape
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    ix, iy, iz = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
    fx, fy, fz = x - ix, y - iy, z - iz

    out = np.zeros(x.shape, dtype=np.float64)
    for dx in (0, 1):
        wx = fx if dx else 1.0 - fx
        jx = ix + dx
        okx = (jx >= 0) & (jx < nx)
        for dy in (0, 1):
            wy = fy if dy else 1.0 - fy
            jy = iy + dy
            oky = okx & (jy >= 0) & (jy < ny)
            for dz in (0, 1):
                wz = fz if dz else 1.0 - fz
                jz = iz + dz
                ok = oky & (jz >= 0) & (jz < nz)
                v = np.zeros_like(out)
                v[ok] = vol_xyz[jx[ok].clip(0, nx - 1),
                                jy[ok].clip(0, ny - 1),
                                jz[ok].clip(0, nz - 1)]
                out += wx * wy * wz * v
    return out


def _pixel_grid(geo, i, projection_type):
    """P (ray end) and S (ray start) for every detector pixel of view i.

    Returns arrays of shape (nV, nU, 3) in normalized coordinates, with the
    output row order of TIGRE projections (row v: pixelV = nV-1-v).
    """
    uvo, dU, dV, src = view_geometry(geo, i, projection_type)
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    u = np.arange(n_u, dtype=np.float64)
    v_row = np.arange(n_v, dtype=np.float64)
    pixel_v = (n_v - 1.0) - v_row  # TIGRE flips V for output row order
    duv = (u[None, :, None] * dU[None, None, :]
           + pixel_v[:, None, None] * dV[None, None, :])
    P = uvo[None, None, :] + duv
    if geo.mode == "parallel":
        S = src[None, None, :] + duv
    else:
        S = np.broadcast_to(src, P.shape).copy()
    return P, S


def _ax_view_interpolated(vol_xyz, geo, i):
    P, S = _pixel_grid(geo, i, "interpolated")
    dvox = geo.dVoxel[::-1].astype(np.float64)
    acc = float(geo.accuracy)

    d = P - S
    length = np.ceil(np.sqrt((d * d).sum(-1)) / acc)  # samples per ray
    length = np.maximum(length, 1.0)
    vect = d / length[..., None]

    nmax = int(length.max())
    steps = np.arange(nmax + 1, dtype=np.float64)  # i = 0..length inclusive
    total = np.zeros(P.shape[:2], dtype=np.float64)
    # Chunk over steps to bound memory.
    chunk = max(1, int(2e7) // (P.shape[0] * P.shape[1] + 1))
    for s0 in range(0, nmax + 1, chunk):
        ss = steps[s0:s0 + chunk]
        pos = S[:, :, None, :] + vect[:, :, None, :] * ss[None, None, :, None]
        vals = _trilinear_border0(vol_xyz, pos)
        vals[ss[None, None, :] > length[..., None]] = 0.0
        total += vals.sum(axis=2)

    deltalength = np.sqrt(((vect * dvox[None, None, :]) ** 2).sum(-1))
    return total * deltalength


def _ax_view_siddon(vol_xyz, geo, i):
    """Exact cell-constant line integral (corner convention)."""
    P, S = _pixel_grid(geo, i, "Siddon")
    nx, ny, nz = vol_xyz.shape
    dvox = geo.dVoxel[::-1].astype(np.float64)

    d = P - S  # (nV, nU, 3)
    phys_len = np.sqrt(((d * dvox[None, None, :]) ** 2).sum(-1))

    # Ray/box intersection in alpha in [0, 1] where x(alpha) = S + alpha d.
    eps = 1e-12
    dd = np.where(np.abs(d) < eps, eps, d)
    a_lo = (0.0 - S) / dd
    a_hi = (np.array([nx, ny, nz], dtype=np.float64) - S) / dd
    a_min = np.minimum(a_lo, a_hi).max(axis=-1).clip(0.0, 1.0)
    a_max = np.maximum(a_lo, a_hi).min(axis=-1).clip(0.0, 1.0)

    # All plane crossings as candidate alphas, per axis.
    alphas = [((np.arange(n + 1, dtype=np.float64)[None, None, :] - S[..., k:k + 1])
               / dd[..., k:k + 1]) for k, n in enumerate((nx, ny, nz))]
    alph = np.concatenate(alphas + [a_min[..., None], a_max[..., None]], axis=-1)
    inside = (alph >= a_min[..., None]) & (alph <= a_max[..., None])
    # Out-of-range crossings collapse onto a_max: zero-length segments.
    alph = np.where(inside, alph, a_max[..., None])
    alph.sort(axis=-1)

    seg = np.diff(alph, axis=-1)
    mids = alph[..., :-1] + 0.5 * seg
    valid = seg > eps
    mids = np.where(valid, mids, 0.0)

    pos = S[..., None, :] + mids[..., :, None] * d[..., None, :]
    ijk = np.floor(pos).astype(np.int64)
    ok = (valid
          & (ijk[..., 0] >= 0) & (ijk[..., 0] < nx)
          & (ijk[..., 1] >= 0) & (ijk[..., 1] < ny)
          & (ijk[..., 2] >= 0) & (ijk[..., 2] < nz))
    ijk = np.clip(ijk, 0, np.array([nx - 1, ny - 1, nz - 1]))
    vals = vol_xyz[ijk[..., 0], ijk[..., 1], ijk[..., 2]]
    vals = np.where(ok, vals, 0.0)
    seg = np.where(ok, seg, 0.0)
    return (vals * seg).sum(axis=-1) * phys_len


def ax_ref(img, geo, angles, projection_type="Siddon"):
    """Reference forward projection.

    img: (nz, ny, nx) array; returns (n_angles, nV, nU) float64.
    """
    geo = geo.copy()
    geo.check_geo(angles)
    if tuple(img.shape) != tuple(int(n) for n in geo.nVoxel):
        raise ValueError(f"img shape {img.shape} != geo.nVoxel {geo.nVoxel}")

    vol_xyz = np.ascontiguousarray(np.asarray(img, dtype=np.float64).transpose(2, 1, 0))
    n = geo.angles.shape[0]
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    out = np.empty((n, n_v, n_u), dtype=np.float64)
    for i in range(n):
        if projection_type in ("interpolated",):
            out[i] = _ax_view_interpolated(vol_xyz, geo, i)
        elif projection_type in ("Siddon", "ray-voxel"):
            out[i] = _ax_view_siddon(vol_xyz, geo, i)
        else:
            raise ValueError(f"unknown projection_type {projection_type!r}")
    return out


# ----------------------------------------------------------------------
# Backprojection (port of TIGRE's voxel_backprojection*.cu)

def _euler_zyz_T(abc, pts):
    """Transposed eulerZYZ (rotate the volume instead of the rays)."""
    a, t, p = abc
    ca, sa = np.cos(a), np.sin(a)
    ct, st = np.cos(t), np.sin(t)
    cp, sp = np.cos(p), np.sin(p)
    R = np.array([
        [ca * ct * cp - sa * sp, -ca * ct * sp - sa * cp, ca * st],
        [sa * ct * cp + ca * sp, -sa * ct * sp + ca * cp, sa * st],
        [-st * cp, st * sp, ct],
    ])
    return pts @ R  # pts @ R == (R.T @ pts.T).T


def _roll_pitch_yaw_T(rot, pts):
    yaw, pitch, roll = rot[0], rot[1], rot[2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cr * cp, cr * sp * sy - sr * cy, cr * sp * cy + sr * sy],
        [sr * cp, sr * sp * sy + cr * cy, sr * sp * cy - cr * sy],
        [-sp, cp * sy, cp * cy],
    ])
    return pts @ R


def bp_view_geometry(geo, i):
    """Per-view voxel-grid basis in the detector-pixel frame (float64).

    Port of TIGRE computeDeltasCube / computeDeltasCubeParallel: returns
    (xyzOrigin, deltaX, deltaY, deltaZ, S) where y is in detector-U pixel
    units, z in detector-V pixel units, x in mm along the beam axis.
    """
    dvox = geo.dVoxel[::-1].astype(np.float64)   # (x, y, z)
    svox = geo.sVoxel[::-1].astype(np.float64)
    d_v, d_u = float(geo.dDetector[0]), float(geo.dDetector[1])
    off_v, off_u = geo.offDetector[i][0], geo.offDetector[i][1]
    dsd, dso = geo.DSD[i], geo.DSO[i]
    oo = geo.offOrigin[i][::-1].astype(np.float64)
    parallel = geo.mode == "parallel"

    P = -(svox / 2.0 - dvox / 2.0) + oo
    pts = np.stack([P, P + [dvox[0], 0, 0], P + [0, dvox[1], 0], P + [0, 0, dvox[2]]])
    pts = _euler_zyz_T(geo.angles[i], pts)
    pts[:, 1] -= off_u
    pts[:, 2] -= off_v
    pts[:, 0] += (dsd - dso)
    pts = _roll_pitch_yaw_T(geo.rotDetector[i], pts)
    pts[:, 0] -= (dsd - dso)

    S = np.array([0.0 if parallel else dsd, -off_u, -off_v])
    S = _roll_pitch_yaw_T(geo.rotDetector[i], S[None])[0]
    S[0] -= (dsd - dso)

    pts[:, 1] /= d_u
    pts[:, 2] /= d_v
    S[1] /= d_u
    S[2] /= d_v
    P, Px, Py, Pz = pts
    return P, Px - P, Py - P, Pz - P, S


def _bilinear_border0(img2d, u, v):
    """Bilinear sample of img2d[v_row, u_col], pixel j centered at j + 0.5."""
    nv, nu = img2d.shape
    ub, vb = u - 0.5, v - 0.5
    iu, iv = np.floor(ub).astype(np.int64), np.floor(vb).astype(np.int64)
    fu, fv = ub - iu, vb - iv
    out = np.zeros(u.shape, dtype=np.float64)
    for du in (0, 1):
        wu = fu if du else 1.0 - fu
        ju = iu + du
        oku = (ju >= 0) & (ju < nu)
        for dv in (0, 1):
            wv = fv if dv else 1.0 - fv
            jv = iv + dv
            ok = oku & (jv >= 0) & (jv < nv)
            val = np.zeros_like(out)
            val[ok] = img2d[jv[ok].clip(0, nv - 1), ju[ok].clip(0, nu - 1)]
            out += wu * wv * val
    return out


def atb_ref(proj, geo, angles, backprojection_type="FDK"):
    """Reference backprojection: (n, nV, nU) -> (nz, ny, nx) float64."""
    geo = geo.copy()
    geo.check_geo(angles)
    n = geo.angles.shape[0]
    nz, ny, nx = (int(k) for k in geo.nVoxel)
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    proj = np.asarray(proj, dtype=np.float64)
    if proj.shape != (n, n_v, n_u):
        raise ValueError(f"proj shape {proj.shape} != {(n, n_v, n_u)}")
    if backprojection_type not in ("FDK", "matched"):
        raise ValueError(f"unknown backprojection_type {backprojection_type!r}")

    dvox = geo.dVoxel[::-1].astype(np.float64)
    svox = geo.sVoxel[::-1].astype(np.float64)
    d_v, d_u = float(geo.dDetector[0]), float(geo.dDetector[1])
    s_v, s_u = float(geo.sDetector[0]), float(geo.sDetector[1])
    parallel = geo.mode == "parallel"

    ix = np.arange(nx, dtype=np.float64)[None, None, :]
    iy = np.arange(ny, dtype=np.float64)[None, :, None]
    iz = np.arange(nz, dtype=np.float64)[:, None, None]

    out = np.zeros((nz, ny, nx), dtype=np.float64)
    for i in range(n):
        o, dX, dY, dZ, S = bp_view_geometry(geo, i)
        dsd, dso = geo.DSD[i], geo.DSO[i]
        cor = geo.COR[i]
        aux_cor = cor / d_u
        Pgx = o[0] + ix * dX[0] + iy * dY[0] + iz * dZ[0]
        Pgy = o[1] + ix * dX[1] + iy * dY[1] + iz * dZ[1] - aux_cor
        Pgz = o[2] + ix * dX[2] + iy * dY[2] + iz * dZ[2]
        if parallel:
            y, z = Pgy, Pgz
        else:
            t = (dso - dsd - S[0]) / (Pgx - S[0])
            y = (Pgy - S[1]) * t + S[1]
            z = (Pgz - S[2]) * t + S[2]
        u = y + n_u * 0.5
        v = z + n_v * 0.5
        sample = _bilinear_border0(proj[i], u, v)

        if parallel:
            w = 1.0
        elif backprojection_type == "FDK":
            # U for the source at (DSO cosA, DSO sinA); see the frame
            # note in projector._atb_source — TIGRE's literal kernel
            # expression evaluates U at -A and folds COR into realy,
            # which cancels on full scans but shades short scans.
            sa, ca = np.sin(geo.angles[i][0]), np.cos(geo.angles[i][0])
            oo = geo.offOrigin[i][::-1]
            realx = -(svox[0] - dvox[0]) * 0.5 + ix * dvox[0] + oo[0]
            realy = -(svox[1] - dvox[1]) * 0.5 + iy * dvox[1] + oo[1]
            w = (dso / (dso - realx * ca - realy * sa)) ** 2
        else:  # matched
            sa, ca = np.sin(geo.angles[i][0]), np.cos(geo.angles[i][0])
            oo = geo.offOrigin[i][::-1]
            rvx = -svox[0] / 2 + dvox[0] / 2 + oo[0] + ix * dvox[0]
            rvy = -svox[1] / 2 + dvox[1] / 2 + oo[1] + iy * dvox[1]
            rvz = -svox[2] / 2 + dvox[2] / 2 + oo[2] + iz * dvox[2]
            rSx, rSy = dso * ca, dso * sa
            rDaux_x = -(dsd - dso)
            rDaux_y = (-s_u + d_u) * 0.5 + u * d_u + geo.offDetector[i][1]
            rDz = (-s_v + d_v) * 0.5 + v * d_v + geo.offDetector[i][0]
            rDx = rDaux_x * ca - rDaux_y * sa
            rDy = rDaux_x * sa + rDaux_y * ca
            L = np.sqrt((rSx - rDx) ** 2 + (rSy - rDy) ** 2 + rDz ** 2)
            lsq = (rSx - rvx) ** 2 + (rSy - rvy) ** 2 + rvz ** 2
            w = L ** 3 / (dsd * lsq)
        out += sample * w
    return out
