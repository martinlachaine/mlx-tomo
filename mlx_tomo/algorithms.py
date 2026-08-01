"""Single-pass analytic reconstruction: FDK (cone) and FBP (parallel).

    FDK  = cosine pre-weight -> ramp filtering -> Atb(..., "FDK")
    FBP  = ramp filtering -> Atb -> scale by DSO/DSD   (parallel only)

The total FDK weighting is split between this module and ``Atb``: the detector
cosine weight is applied here before filtering, and the per-voxel distance
weight is applied inside ``Atb``'s FDK mode. The absolute scale is pinned by
analytic-phantom recovery tests in https://github.com/martinlachaine/mlx-tomo/blob/main/harness/test_fdk.py.

For an ideal circular trajectory under the standard FDK assumptions,
reconstruction is exact in the central plane and approximate away from it.
Recovered attenuation degrades smoothly with cone angle; the measured error
budget is in https://github.com/martinlachaine/mlx-tomo/blob/main/harness/check_fdk.py.

See https://github.com/martinlachaine/mlx-tomo/blob/main/docs/implementation.md for the weight split and scaling, and
https://github.com/martinlachaine/mlx-tomo/blob/main/docs/compatibility.md for behavioral differences.
"""

from __future__ import annotations

import warnings

import numpy as np

import mlx.core as mx

from .api import Atb
from .filtering import (_covered_range, _filter_core, _mean_step,
                        _resolve_parker)

__all__ = ["FDK", "FBP", "fdk", "fbp"]


def _to_mx(proj, who):
    is_mx = isinstance(proj, mx.array)
    if is_mx:
        if proj.dtype != mx.float32:
            raise TypeError(f"{who}: proj must be float32, got {proj.dtype}")
        return proj, True
    proj = np.asarray(proj)
    if proj.dtype != np.float32:
        raise TypeError(f"{who}: proj must be float32, got {proj.dtype}")
    return mx.array(proj), False


def _pop_common_kwargs(kwargs, who):
    if kwargs.pop("gpuids", None) is not None:
        warnings.warn(f"mlx-tomo ignores gpuids (single Apple GPU)")
    kwargs.pop("verbose", None)
    if kwargs.pop("dowang", None):
        warnings.warn(
            f"{who}: TIGRE's Wang half-fan weighting (dowang) is not "
            "implemented; offset detectors are handled through "
            "piercing-point-aware cosine weights instead, which is "
            "correct for untruncated projections")
    if kwargs:
        warnings.warn(f"ignoring unknown {who} options: {sorted(kwargs)}")


def _cosine_weight(geo):
    """FDK cosine pre-weight, (1, nV, nU) or (n, nV, nU) float32 mx.

    w = DSD / sqrt(DSD^2 + u^2 + v^2) with (u, v) the pixel offsets
    from the piercing point: centered pixel coordinates plus
    offDetector (per-angle when it varies). COR shifts source and
    detector together, leaving pixel-to-piercing distances unchanged,
    so it does not enter.
    """
    n = geo.angles.shape[0]
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    d_v, d_u = float(geo.dDetector[0]), float(geo.dDetector[1])
    cu = (np.arange(n_u) - n_u / 2 + 0.5) * d_u
    cv = (np.arange(n_v) - n_v / 2 + 0.5) * d_v
    varying = (np.ptp(geo.DSD) > 0) or bool(np.ptp(geo.offDetector, axis=0).any())
    if not varying:
        u = cu + geo.offDetector[0][1]
        v = cv + geo.offDetector[0][0]
        w = geo.DSD[0] / np.sqrt(geo.DSD[0] ** 2
                                 + u[None, :] ** 2 + v[:, None] ** 2)
        return mx.array(w[None].astype(np.float32))
    w = np.empty((n, n_v, n_u), dtype=np.float32)
    for i in range(n):
        u = cu + geo.offDetector[i][1]
        v = cv + geo.offDetector[i][0]
        w[i] = geo.DSD[i] / np.sqrt(geo.DSD[i] ** 2
                                    + u[None, :] ** 2 + v[:, None] ** 2)
    return mx.array(w)


def _auto_parker(geo):
    """Parker q for parker=None: 1.0 for short scans, else 0 (off).

    Auto-application requires a monotonic circular arc (rotation in the
    first Euler angle, theta/psi constant): Parker redundancy weighting
    is undefined for other trajectories, and misreading e.g. a
    (0, theta, 0) orbit as a 0-degree scan would silently zero every
    weight. Such trajectories get no weighting unless parker is forced.
    """
    a = geo.angles
    if a.shape[0] < 2:
        return 0.0
    rot = np.unwrap(a[:, 0])
    d = np.diff(rot)
    monotonic_arc = (np.all(d > 0) or np.all(d < 0)) \
        and np.ptp(a[:, 1]) == 0 and np.ptp(a[:, 2]) == 0
    if not monotonic_arc:
        return 0.0
    covered = _covered_range(a) + _mean_step(a)
    if covered < 2 * np.pi - 1e-6:
        warnings.warn(
            f"FDK: short scan detected ({np.degrees(covered):.1f} deg "
            "covered): applying Parker weights (pass parker=False to "
            "disable)")
        return 1.0
    return 0.0


def FDK(proj, geo, angles, filter=None, parker=None, d=1.0,
        return_np=None, **kwargs):
    """FDK reconstruction; TIGRE-compatible signature.

    proj: (n, nV, nU) float32 numpy array or mx.array (line integrals,
        as produced by Ax).
    filter: 'ram_lak' (default), 'shepp_logan', 'cosine', 'hamming',
        'hann'; None uses geo.filter when set.
    parker: None (auto: applied for scans covering < 2*pi), False
        (never), True (q=1) or a positive float q (Wesarg smoothness).
    d: filter cutoff frequency in (0, 1].
    return_np: force numpy (True) or mx.array (False); default mirrors
        the input type.

    Returns (nz, ny, nx) float32. For geo.mode == 'parallel' this
    dispatches to FBP (parker is meaningless there and ignored).
    """
    _pop_common_kwargs(kwargs, "FDK")
    geo = geo.copy()
    geo.check_geo(angles)
    if geo.mode == "parallel":
        if _resolve_parker(parker) > 0:
            warnings.warn("FDK: Parker weights are for divergent short "
                          "scans; ignored in parallel mode")
        return FBP(proj, geo, angles, filter=filter, d=d,
                   return_np=return_np)
    proj_mx, is_mx = _to_mx(proj, "FDK")
    n = geo.angles.shape[0]
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    if tuple(proj_mx.shape) != (n, n_v, n_u):
        raise ValueError(f"proj shape {tuple(proj_mx.shape)} != {(n, n_v, n_u)}")
    if np.any(geo.rotDetector):
        warnings.warn("FDK: cosine weights assume an untilted detector; "
                      "rotDetector != 0 degrades quantitative accuracy")

    q = _auto_parker(geo) if parker is None else _resolve_parker(parker)
    weighted = proj_mx * _cosine_weight(geo)
    filtered = _filter_core(weighted, geo, q, d, filter)
    vol = Atb(filtered, geo, geo.angles, backprojection_type="FDK",
              return_np=False)
    if return_np is None:
        return_np = not is_mx
    return np.array(vol) if return_np else vol


def FBP(proj, geo, angles, filter=None, d=1.0, return_np=None, **kwargs):
    """Filtered backprojection for parallel-beam data (tigre.fbp).

    Same conventions as FDK; geo.mode must be 'parallel'.
    """
    _pop_common_kwargs(kwargs, "FBP")
    geo = geo.copy()
    geo.check_geo(angles)
    if geo.mode != "parallel":
        raise ValueError("Only use FBP for parallel beam. Check geo.mode.")
    proj_mx, is_mx = _to_mx(proj, "FBP")
    n = geo.angles.shape[0]
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    if tuple(proj_mx.shape) != (n, n_v, n_u):
        raise ValueError(f"proj shape {tuple(proj_mx.shape)} != {(n, n_v, n_u)}")

    filtered = _filter_core(proj_mx, geo, 0.0, d, filter)
    # filtering carries TIGRE's DSD/DSO cone factor; cancel it here
    # exactly as tigre's fbp does.
    vol = Atb(filtered, geo, geo.angles, backprojection_type="FDK",
              return_np=False)
    vol = vol * (float(geo.DSO[0]) / float(geo.DSD[0]))
    mx.eval(vol)
    if return_np is None:
        return_np = not is_mx
    return np.array(vol) if return_np else vol


fdk = FDK
fbp = FBP
