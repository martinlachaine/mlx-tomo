"""TIGRE-compatible user-facing API."""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from . import projector as _pr
from .projector import ax_gpu

__all__ = ["Ax", "Atb"]


def Ax(img, geo, angles, projection_type="Siddon", return_np=None,
       backend="buffer", **kwargs):
    """Forward projection; TIGRE-compatible signature.

    img: (nz, ny, nx) float32 numpy array or mx.array, shape == geo.nVoxel.
    geo: mlx_tomo.Geometry (or any object with TIGRE geometry fields).
    angles: (n,) or (n, 3) radians.
    projection_type: 'Siddon' (default, exact cell-constant path, like
        TIGRE) or 'interpolated' (fixed-step trilinear sampling,
        geo.accuracy voxels per step). 'ray-voxel' is accepted as an alias
        of 'Siddon'.
    return_np: force numpy (True) or mx.array (False) output; default
        mirrors the input type.
    backend: 'buffer' (default; exact float32 trilinear/Siddon via MLX
        kernels) or 'texture' (hardware sampler, interpolated only,
        ~1e-4 measured accuracy, volume upload per call — prefer
        TextureProjector for repeated projections of one volume).

    Returns (n, nV, nU) float32 line integrals in mm units.
    """
    if kwargs.pop("gpuids", None) is not None:
        import warnings
        warnings.warn("mlx-tomo ignores gpuids (single Apple GPU)")
    if kwargs:
        import warnings
        warnings.warn(f"ignoring unknown Ax options: {sorted(kwargs)}")

    is_mx = isinstance(img, mx.array)
    if is_mx:
        if img.dtype not in (mx.float32, mx.float16):
            raise TypeError(f"img must be float32 (or float16), got {img.dtype}")
        img_mx = img
    else:
        img = np.asarray(img)
        if img.dtype not in (np.float32, np.float16):
            raise TypeError(f"img must be float32 (or float16), got {img.dtype}")
        # Volumes beyond the slab threshold stay numpy here; ax_gpu
        # converts one z-slab at a time to bound peak memory.
        img_mx = mx.array(img) if img.size <= _pr._SLAB_ELEMS else img

    geo = geo.copy()
    geo.check_geo(angles)
    if tuple(img_mx.shape) != tuple(int(n) for n in geo.nVoxel):
        raise ValueError(
            f"img shape {tuple(img_mx.shape)} != geo.nVoxel "
            f"{tuple(int(n) for n in geo.nVoxel)}")

    if return_np is None:
        return_np = not is_mx
    if backend == "texture":
        from .texture import TextureProjector
        src = img if not is_mx else img_mx  # keep numpy as-is (no round trip)
        with TextureProjector(src, geo, projection_type) as plan:
            return plan.project(angles, return_np=return_np)
    elif backend != "buffer":
        raise ValueError(f"unknown backend {backend!r}")
    proj = ax_gpu(img_mx, geo, projection_type)
    mx.eval(proj)
    return np.array(proj) if return_np else proj


def Atb(proj, geo, angles, backprojection_type="FDK", return_np=None, **kwargs):
    """Backprojection; TIGRE-compatible signature.

    proj: (n, nV, nU) float32 numpy array or mx.array.
    backprojection_type: 'FDK' (distance-weighted, for FDK reconstruction)
        or 'matched' (approximate adjoint of the interpolated Ax).
        The legacy TIGRE<=2.0 kwarg krylov= is honored as a deprecated
        alias so migrated code keeps its requested weighting.
    Returns (nz, ny, nx) float32.
    """
    if "krylov" in kwargs:
        import warnings
        backprojection_type = kwargs.pop("krylov")
        warnings.warn("krylov= is deprecated; use backprojection_type=",
                      DeprecationWarning)
    if kwargs.pop("gpuids", None) is not None:
        import warnings
        warnings.warn("mlx-tomo ignores gpuids (single Apple GPU)")
    if kwargs:
        import warnings
        warnings.warn(f"ignoring unknown Atb options: {sorted(kwargs)}")

    from .projector import atb_gpu

    is_mx = isinstance(proj, mx.array)
    if is_mx:
        if proj.dtype != mx.float32:
            raise TypeError(f"proj must be float32, got {proj.dtype}")
        proj_mx = proj
    else:
        proj = np.asarray(proj)
        if proj.dtype != np.float32:
            raise TypeError(f"proj must be float32, got {proj.dtype}")
        proj_mx = mx.array(proj)

    geo = geo.copy()
    geo.check_geo(angles)
    n = geo.angles.shape[0]
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    if tuple(proj_mx.shape) != (n, n_v, n_u):
        raise ValueError(f"proj shape {tuple(proj_mx.shape)} != {(n, n_v, n_u)}")

    img = atb_gpu(proj_mx, geo, backprojection_type)
    mx.eval(img)
    if return_np is None:
        return_np = not is_mx
    return np.array(img) if return_np else img
