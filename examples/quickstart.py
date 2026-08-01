"""Minimal mlx-tomo example: cone-beam DRRs of a simple phantom.

Run: python examples/quickstart.py
"""

import numpy as np

import mlx_tomo as tigre  # TIGRE-compatible for the operations it implements

# Geometry (identical fields and defaults to TIGRE's)
geo = tigre.geometry_default(high_resolution=False)  # 64^3 volume, 128^2 det
print(geo)

# A ball phantom
nz, ny, nx = (int(n) for n in geo.nVoxel)
z, y, x = np.mgrid[:nz, :ny, :nx].astype(np.float32)
r2 = ((x - nx / 2) ** 2 + (y - ny / 2) ** 2 + (z - nz / 2) ** 2)
img = (r2 < (nx / 4) ** 2).astype(np.float32)

# 100 projections over 2*pi
angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
proj = tigre.Ax(img, geo, angles, projection_type="interpolated")
print("projections:", proj.shape, proj.dtype, "max:", proj.max())

# Parallel beam
geo_p = tigre.geometry(mode="parallel", nVoxel=np.array([64, 64, 64]))
proj_p = tigre.Ax(img, geo_p, angles)
print("parallel projections:", proj_p.shape)
