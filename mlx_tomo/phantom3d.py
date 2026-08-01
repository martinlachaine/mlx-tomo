"""3D Shepp-Logan phantom (modified contrast), mm units, mlx-tomo axes.

The ellipsoid table is the standard modified 3D Shepp-Logan (the one
ODL ships as odl.phantom.shepp_logan): ten
ellipsoids, additive values, rotations about z only. Coordinates follow
mlx_tomo/geometry.py: volumes are img[z, y, x]; world (x, y, z) mm with
the volume centered on the origin, voxel centers at
(i + 0.5) * dVoxel - sVoxel / 2.

Each ellipsoid is (value, center(x, y, z), semiaxes(ax, ay, az), phi)
with phi a rotation about +z (radians), all in the phantom's unit box
[-1, 1]^3; `shepp_logan_mm(scale_mm, mu)` converts to mm and to
attenuation units so recovered values read directly as mu (1/mm).
"""

from __future__ import annotations

import numpy as np

# value, a(x), b(y), c(z), x0, y0, z0, phi(deg)  -- modified contrast
_TABLE = [
    (1.00, 0.6900, 0.9200, 0.810, 0.0, 0.0, 0.0, 0.0),
    (-0.80, 0.6624, 0.8740, 0.780, 0.0, -0.0184, 0.0, 0.0),
    (-0.20, 0.1100, 0.3100, 0.220, 0.22, 0.0, 0.0, -18.0),
    (-0.20, 0.1600, 0.4100, 0.280, -0.22, 0.0, 0.0, 18.0),
    (0.10, 0.2100, 0.2500, 0.410, 0.0, 0.35, -0.15, 0.0),
    (0.10, 0.0460, 0.0460, 0.050, 0.0, 0.10, 0.25, 0.0),
    (0.10, 0.0460, 0.0460, 0.050, 0.0, -0.10, 0.25, 0.0),
    (0.10, 0.0460, 0.0230, 0.050, -0.08, -0.605, 0.0, 0.0),
    (0.10, 0.0230, 0.0230, 0.020, 0.0, -0.606, 0.0, 0.0),
    (0.10, 0.0230, 0.0460, 0.020, 0.06, -0.605, 0.0, 0.0),
]


def shepp_logan_mm(scale_mm=115.0, mu=0.02):
    """Ellipsoid list [(rho, (x0,y0,z0), (ax,ay,az), phi_rad)] in mm.

    scale_mm maps the unit box to +-scale_mm; mu multiplies the
    dimensionless table values (peak interior sum ~0.2 -> ~0.004/mm
    inside 'brain' tissue, 0.02/mm in the shell, with mu=0.02).
    """
    out = []
    for v, a, b, c, x0, y0, z0, phi in _TABLE:
        out.append((v * mu,
                    (x0 * scale_mm, y0 * scale_mm, z0 * scale_mm),
                    (a * scale_mm, b * scale_mm, c * scale_mm),
                    np.deg2rad(phi)))
    return out


def rasterize(geo, ells, supersample=2, chunk=32):
    """Rasterize rotated ellipsoids into (nz, ny, nx) float32.

    supersample: linear subdivision per axis (2 -> 8 subsamples/voxel),
    giving partial-volume edges — a fairer ground truth for comparing
    band-limited reconstructions than binary voxel-center assignment.
    """
    nz, ny, nx = (int(k) for k in geo.nVoxel)
    dz, dy, dx = (float(v) for v in geo.dVoxel)
    sz, sy, sx = (float(v) for v in geo.sVoxel)
    s = int(supersample)
    off = (np.arange(s) + 0.5) / s  # subsample offsets in voxel units

    xs = (np.add.outer(np.arange(nx), off).ravel()) * dx - sx / 2
    ys = (np.add.outer(np.arange(ny), off).ravel()) * dy - sy / 2
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    for z0 in range(0, nz, chunk):
        zc = min(chunk, nz - z0)
        zs = (np.add.outer(np.arange(z0, z0 + zc), off).ravel()) * dz - sz / 2
        acc = np.zeros((zc * s, ny * s, nx * s), dtype=np.float32)
        for rho, c, ax, phi in ells:
            cp, sp = np.cos(phi), np.sin(phi)
            # rotate (r - c) by -phi about z into the ellipsoid frame
            xr = xs[None, None, :] - c[0]
            yr = ys[None, :, None] - c[1]
            zr = zs[:, None, None] - c[2]
            u = cp * xr + sp * yr
            v = -sp * xr + cp * yr
            q = (u / ax[0]) ** 2 + (v / ax[1]) ** 2 + (zr / ax[2]) ** 2
            acc += np.where(q <= 1.0, np.float32(rho), np.float32(0.0))
        blk = acc.reshape(zc, s, ny, s, nx, s).mean(axis=(1, 3, 5))
        vol[z0:z0 + zc] = blk
    return vol


def region_masks(geo, ells, erode=0.6):
    """Homogeneous interior mask per ellipsoid.

    The Shepp-Logan ellipsoids are NESTED (every insert lies inside the
    skull and brain), so a region mask must stay away from every OTHER
    ellipsoid's boundary while matching the containment status of the
    region's center: strictly inside (q < 0.8^2) the ellipsoids that
    contain the center, strictly outside (q > 1.2^2) the rest.
    """
    nz, ny, nx = (int(k) for k in geo.nVoxel)
    dz, dy, dx = (float(v) for v in geo.dVoxel)
    sz, sy, sx = (float(v) for v in geo.sVoxel)
    x = (np.arange(nx) + 0.5) * dx - sx / 2
    y = (np.arange(ny) + 0.5) * dy - sy / 2
    z = (np.arange(nz) + 0.5) * dz - sz / 2

    def q_of(c, ax, phi):
        cp, sp = np.cos(phi), np.sin(phi)
        xr = x[None, None, :] - c[0]
        yr = y[None, :, None] - c[1]
        zr = z[:, None, None] - c[2]
        u = cp * xr + sp * yr
        v = -sp * xr + cp * yr
        return (u / ax[0]) ** 2 + (v / ax[1]) ** 2 + (zr / ax[2]) ** 2

    def q_point(p, c, ax, phi):
        cp, sp = np.cos(phi), np.sin(phi)
        xr, yr, zr = p[0] - c[0], p[1] - c[1], p[2] - c[2]
        u = cp * xr + sp * yr
        v = -sp * xr + cp * yr
        return (u / ax[0]) ** 2 + (v / ax[1]) ** 2 + (zr / ax[2]) ** 2

    qs = [q_of(c, ax, phi) for _, c, ax, phi in ells]
    masks = []
    for i, (_, c_i, _, _) in enumerate(ells):
        m = qs[i] < erode ** 2
        for j, (_, c_j, ax_j, phi_j) in enumerate(ells):
            if j == i:
                continue
            if q_point(c_i, c_j, ax_j, phi_j) < 1.0:  # center inside j
                m &= qs[j] < 0.8 ** 2
            else:
                m &= qs[j] > 1.2 ** 2
        masks.append(m)
    return masks


def ground_truth_values(ells):
    """Additive ground-truth value inside each region mask (the sum of
    all ellipsoids containing that region's center)."""
    vals = []
    for i, (rho_i, c_i, ax_i, phi_i) in enumerate(ells):
        total = 0.0
        for rho, c, ax, phi in ells:
            cp, sp = np.cos(phi), np.sin(phi)
            xr, yr, zr = (c_i[0] - c[0]), (c_i[1] - c[1]), (c_i[2] - c[2])
            u = cp * xr + sp * yr
            v = -sp * xr + cp * yr
            q = (u / ax[0]) ** 2 + (v / ax[1]) ** 2 + (zr / ax[2]) ** 2
            if q < 1.0:
                total += rho
        vals.append(total)
    return vals
