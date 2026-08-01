"""Projection along explicitly specified rays.

``Ax`` derives its rays from a TIGRE-style geometry: source on a circle at
DSO, flat detector at DSD, orthogonal pixel grid, plus offsets and detector
rotations. That covers ideal circular scans. This module exposes the more general
contract the kernel already accepts.

The kernel's native per-view contract is four vectors::

    pixel(v, u) = uvOrigin + u * deltaU + v * deltaV        (source separate)

with ``deltaU`` and ``deltaV`` arbitrary -- nothing requires them to be
orthogonal or equal in length. ``build_view_params`` is a convenience adapter
from the geometry model onto that contract, so the restriction to orthogonal
grids lives in the adapter, not in the projector. This module exposes the
contract directly.

Applications:

* **Geometrically calibrated scans.** Calibration of a real C-arm or CBCT
  gantry (sag, wobble, non-ideal detector mounting) yields a 3x4 projection
  matrix per view. That decomposes exactly into a source position and a
  detector basis, but maps only lossily onto DSO/DSD-plus-offsets.
* **Non-circular and robotic trajectories**, tomosynthesis arcs, and any scan
  whose source positions are measured rather than idealized.
* **Motion compensation.** If the object moves by an affine map,
  ``f_t(x) = f_ref(A^-1 (x - b))``, then every measured ray is a ray of the
  static reference from a virtual source, and because ``A^-1`` is linear the
  affine image of a parallelogram grid is another parallelogram grid::

      source' = A^-1 (source - b)      deltaU' = A^-1 deltaU
      uvOrigin' = A^-1 (uvOrigin - b)  deltaV' = A^-1 deltaV

  so the moving-object projection is a rays call on the *unwarped* volume.
  Transforming the rays avoids an additional volume-resampling step, which
  would otherwise contribute its own interpolation error.

Conventions. Rays are given in **world millimeters with the isocenter at the
origin**, the frame ``geo`` fields are specified in, and indices match the
returned array directly: ``pixel(v, u)`` is the sample at ``proj[i, v, u]``.
The half-voxel and corner-origin bookkeeping that ``view_geometry`` uses
internally is handled here, and ``rays_from_geometry`` is the inverse map, so
``Ax_rays(img, geo, rays_from_geometry(geo, angles, t), t)`` reproduces
``Ax(img, geo, angles, t)`` exactly. That round trip is the gate on this
module (https://github.com/martinlachaine/mlx-tomo/blob/main/harness/test_rays.py).
"""

from __future__ import annotations

import numpy as np

from . import projector as _pr
from .ref import view_geometry

__all__ = ["rays_from_geometry", "Ax_rays", "affine_rays"]

_UVO, _DU, _DV, _SRC = 0, 1, 2, 3


def _frame(geo, projection_type):
    """(scale, shift) mapping world mm -> the kernel's normalized coordinates.

    ``view_geometry`` moves the origin to the volume corner and divides by the
    voxel size, with a half-voxel difference between the two projection types
    (center convention for `interpolated`, corner for `Siddon`).
    """
    dvox = np.asarray(geo.dVoxel, np.float64)[::-1]          # -> (x, y, z)
    svox = np.asarray(geo.sVoxel, np.float64)[::-1]
    if projection_type in ("Siddon", "ray-voxel"):
        shift = svox / 2.0
    elif projection_type == "interpolated":
        shift = svox / 2.0 - dvox / 2.0
    else:
        raise ValueError(f"unknown projection_type {projection_type!r}")
    return dvox, shift


def rays_from_geometry(geo, angles, projection_type="Siddon"):
    """Extract the rays a geometry describes, as (n, 4, 3) world mm.

    Returns, per view, ``[uvOrigin, deltaU, deltaV, source]`` with
    ``pixel(v, u) = uvOrigin + u*deltaU + v*deltaV`` indexed as the returned
    projection is. Useful on its own for inspecting or perturbing a geometry,
    and it is the exact inverse of what Ax_rays consumes.
    """
    g = geo.copy()
    g.check_geo(angles)
    v = _pr.build_view_params(g, "Siddon" if projection_type in
                              ("Siddon", "ray-voxel") else projection_type)
    v = np.asarray(v, np.float64)
    dvox, shift = _frame(g, projection_type)
    out = np.empty((v.shape[0], 4, 3), dtype=np.float64)
    out[:, _UVO] = v[:, 0:3] * dvox - shift
    out[:, _DU] = v[:, 3:6] * dvox              # differences: scale only
    out[:, _DV] = v[:, 6:9] * dvox
    out[:, _SRC] = v[:, 9:12] * dvox - shift
    return out


def _to_view_params(geo, rays, projection_type):
    rays = np.asarray(rays, np.float64)
    if rays.ndim != 3 or rays.shape[1:] != (4, 3):
        raise ValueError(f"rays must be (n, 4, 3), got {rays.shape}")
    dvox, shift = _frame(geo, projection_type)
    out = np.zeros((rays.shape[0], 16), dtype=np.float64)
    out[:, 0:3] = (rays[:, _UVO] + shift) / dvox
    out[:, 3:6] = rays[:, _DU] / dvox
    out[:, 6:9] = rays[:, _DV] / dvox
    out[:, 9:12] = (rays[:, _SRC] + shift) / dvox
    return out.astype(np.float32)


def Ax_rays(img, geo, rays, projection_type="Siddon", return_np=None):
    """Forward project along explicitly given rays.

    img:  (nz, ny, nx) float32/float16, shape == geo.nVoxel.
    geo:  supplies ONLY nVoxel, dVoxel, sVoxel, nDetector, accuracy and mode.
          DSO, DSD, angles, offOrigin, offDetector, rotDetector and COR are
          IGNORED -- the rays replace them. Pass the same geo you would use
          with Ax and it will size everything consistently.
    rays: (n, 4, 3) float, world mm, isocenter at the origin, per view
          [uvOrigin, deltaU, deltaV, source]; see rays_from_geometry.
          In parallel mode the source is per-pixel, offset by the same
          u*deltaU + v*deltaV, matching Ax's parallel convention.

    Returns (n, nV, nU) float32 line integrals in mm, exactly as Ax does.
    """
    import mlx.core as mx

    is_mx = isinstance(img, mx.array)
    if is_mx:
        if img.dtype not in (mx.float32, mx.float16):
            raise TypeError(f"img must be float32 (or float16), got {img.dtype}")
        img_mx = img
    else:
        img = np.asarray(img)
        if img.dtype not in (np.float32, np.float16):
            raise TypeError(f"img must be float32 (or float16), got {img.dtype}")
        img_mx = mx.array(img) if img.size <= _pr._SLAB_ELEMS else img

    geo = geo.copy()
    if tuple(img_mx.shape) != tuple(int(n) for n in geo.nVoxel):
        raise ValueError(
            f"img shape {tuple(img_mx.shape)} != geo.nVoxel "
            f"{tuple(int(n) for n in geo.nVoxel)}")
    views = _to_view_params(geo, rays, projection_type)
    proj = _pr.ax_gpu(img_mx, geo, projection_type, views=views)
    if return_np is None:
        return np.asarray(proj) if not is_mx else proj
    return np.asarray(proj) if return_np else proj


def affine_rays(rays, A, b):
    """Map rays to those of the static reference under f_t(x)=f_ref(A^-1(x-b)).

    Applies the linear part to the detector basis and the full inverse affine
    to the two points, which is exact because an affine image of a
    parallelogram grid is a parallelogram grid.

    NOTE this returns the GEOMETRY only. The line integral of the moving object
    also carries the change-of-variables factor: with
    ``jac = |A^-1 omega|`` for the unit ray direction ``omega``,

        g_measured = (1/jac) * integral f_ref along the mapped ray

    so recovering a reference-object line integral means MULTIPLYING the
    measured value by ``jac``. ``jac`` varies per pixel unless ``A`` is a
    rotation, and it is deliberately not folded in here: the caller knows
    which direction they are going.
    """
    rays = np.asarray(rays, np.float64)
    Ai = np.linalg.inv(np.asarray(A, np.float64).reshape(3, 3))
    b = np.asarray(b, np.float64).reshape(3)
    out = np.empty_like(rays)
    out[:, _UVO] = np.einsum("ij,...j->...i", Ai, rays[:, _UVO] - b)
    out[:, _DU] = np.einsum("ij,...j->...i", Ai, rays[:, _DU])
    out[:, _DV] = np.einsum("ij,...j->...i", Ai, rays[:, _DV])
    out[:, _SRC] = np.einsum("ij,...j->...i", Ai, rays[:, _SRC] - b)
    return out
