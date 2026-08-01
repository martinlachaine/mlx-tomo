"""Problem generators and independent analytic oracle.

The analytic projector here derives detector/source positions directly from
the documented TIGRE convention — it shares no code with mlx_tomo.ref (which
ports TIGRE's computeDeltas). Agreement between the two validates the port.

World frame (mm): at angles=(0,0,0) the source is at (DSO, 0, 0), detector
center at (-(DSD-DSO), 0, 0); u along +y, output row v along +z
(row v center: z = (v - nV/2 + 0.5) * dV). Volume centered at origin,
offOrigin shifts the volume (z,y,x ordering in the geometry fields).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def rel_l2(u, ref):
    u = np.asarray(u, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


def _euler_zyz_mat(a, t, p):
    ca, sa, ct, st, cp, sp = np.cos(a), np.sin(a), np.cos(t), np.sin(t), np.cos(p), np.sin(p)
    return np.array([
        [ca * ct * cp - sa * sp, -ca * ct * sp - sa * cp, ca * st],
        [sa * ct * cp + ca * sp, -sa * ct * sp + ca * cp, sa * st],
        [-st * cp, st * sp, ct],
    ])


def rays_world(geo, angles):
    """Source S and pixel P world positions per (angle, v, u).

    Returns (S, P) with shape (n, nV, nU, 3), ordered (x, y, z), in mm.
    Supports offDetector, offOrigin, COR, per-angle DSD/DSO; assumes
    rotDetector = 0 (the analytic oracle keeps detector unrotated).
    """
    g = geo.copy()
    g.check_geo(angles)
    n = g.angles.shape[0]
    n_v, n_u = int(g.nDetector[0]), int(g.nDetector[1])
    d_v, d_u = float(g.dDetector[0]), float(g.dDetector[1])

    u = np.arange(n_u) + 0.5 - n_u / 2.0
    v = np.arange(n_v) + 0.5 - n_v / 2.0

    S = np.empty((n, n_v, n_u, 3))
    P = np.empty((n, n_v, n_u, 3))
    for i in range(n):
        dsd, dso = g.DSD[i], g.DSO[i]
        off_v, off_u = g.offDetector[i][0], g.offDetector[i][1]
        R = _euler_zyz_mat(*g.angles[i])
        py = u * d_u + off_u
        pz = v * d_v + off_v
        pix = np.stack(np.broadcast_arrays(
            np.full((n_v, n_u), -(dsd - dso)),
            py[None, :].repeat(n_v, 0),
            pz[:, None].repeat(n_u, 1)), axis=-1)
        if g.mode == "parallel":
            src = pix.copy()
            src[..., 0] = dso
        else:
            src = np.broadcast_to(np.array([dso, 0.0, 0.0]), pix.shape).copy()
        P[i] = pix @ R.T
        S[i] = src @ R.T
        # COR correction shifts source and detector (world equivalent of
        # TIGRE's normalized-space shift).
        a = g.angles[i][0]
        cshift = np.array([-g.COR[i] * np.sin(a), g.COR[i] * np.cos(a), 0.0])
        P[i] += cshift
        S[i] += cshift
        # offOrigin moves the volume; equivalently move rays the other way.
        oo = g.offOrigin[i][::-1]  # (z,y,x) -> (x,y,z)
        P[i] -= oo
        S[i] -= oo
    return S, P


def ellipsoid_phantom(geo, ellipsoids):
    """Rasterize axis-aligned ellipsoids into a (nz, ny, nx) volume.

    ellipsoids: list of (rho, (cx, cy, cz), (ax, ay, az)) in mm.
    """
    nz, ny, nx = (int(k) for k in geo.nVoxel)
    dz, dy, dx = geo.dVoxel
    sz, sy, sx = geo.sVoxel
    z = (np.arange(nz) + 0.5) * dz - sz / 2.0
    y = (np.arange(ny) + 0.5) * dy - sy / 2.0
    x = (np.arange(nx) + 0.5) * dx - sx / 2.0
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    vol = np.zeros((nz, ny, nx), dtype=np.float64)
    for rho, c, ax in ellipsoids:
        q = (((X - c[0]) / ax[0]) ** 2 + ((Y - c[1]) / ax[1]) ** 2
             + ((Z - c[2]) / ax[2]) ** 2)
        vol[q <= 1.0] += rho
    return vol


def analytic_projection(geo, angles, ellipsoids):
    """Exact line integrals of ellipsoids: (n, nV, nU) float64, in mm units."""
    S, P = rays_world(geo, angles)
    d = P - S
    norm = np.linalg.norm(d, axis=-1)
    out = np.zeros(S.shape[:3], dtype=np.float64)
    for rho, c, ax in ellipsoids:
        c = np.asarray(c, dtype=np.float64)
        ax = np.asarray(ax, dtype=np.float64)
        o = (S - c) / ax
        dn = d / ax
        A = (dn * dn).sum(-1)
        B = 2.0 * (o * dn).sum(-1)
        C = (o * o).sum(-1) - 1.0
        disc = B * B - 4.0 * A * C
        hit = disc > 0
        chord = np.zeros_like(out)
        chord[hit] = np.sqrt(disc[hit]) / A[hit] * norm[hit]
        out += rho * chord
    return out


def default_phantom_mm(geo):
    """A smooth-ish three-ellipsoid phantom scaled to the volume size."""
    sx = float(geo.sVoxel[2])
    return [
        (1.0, (0.0, 0.0, 0.0), (0.36 * sx, 0.30 * sx, 0.30 * sx)),
        (-0.5, (0.08 * sx, 0.05 * sx, 0.04 * sx), (0.14 * sx, 0.10 * sx, 0.12 * sx)),
        (0.8, (-0.12 * sx, -0.08 * sx, -0.06 * sx), (0.06 * sx, 0.08 * sx, 0.07 * sx)),
    ]


def machine():
    import subprocess
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                         capture_output=True, text=True).stdout.strip()
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"],
                         capture_output=True, text=True).stdout.strip()
    gb = int(mem) / 2**30 if mem else 0
    return f"{cpu}, {gb:.0f} GB"
