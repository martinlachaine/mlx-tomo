"""Frequency-domain filtering utilities for FDK and FBP.

Implements band-limited ramp filters, configurable spectral windows,
geometry-dependent scaling, and optional Parker short-scan weighting.

``filtering(proj, geo, angles)`` returns a new array and does not modify its
input. The cone-beam cosine pre-weight is not applied here; it lives in ``FDK``.

Available windows: ``shepp_logan``, ``cosine``, ``hamming``, ``hann``, plus a
cutoff parameter in (0, 1] that zeroes frequencies above pi * d.

See https://github.com/martinlachaine/mlx-tomo/blob/main/docs/implementation.md for the ramp construction, padding and scaling
conventions, and https://github.com/martinlachaine/mlx-tomo/blob/main/docs/compatibility.md for behavioral differences.
"""

from __future__ import annotations

import warnings

import numpy as np

import mlx.core as mx

__all__ = ["filtering", "ramp_flat", "ramp_filter", "parker_weights",
           "FILTERS"]

FILTERS = ("ram_lak", "shepp_logan", "cosine", "hamming", "hann")

# Keep every padded FFT batch comfortably below the int32 dispatch
# limit (complex64 intermediate of n*nV*flen elements).
_FFT_CHUNK_ELEMS = 2**29


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def ramp_flat(n):
    """Band-limited spatial ramp kernel at unit spacing, length n.

    h[center] = 1/4, h[odd offsets] = -1/(pi k)^2, h[even] = 0
    (Kak & Slaney eq. 3.29 with tau = 1; the physical 1/tau^2 enters
    via the 1/dU in the filtering scale).
    """
    nn = np.arange(-n / 2, n / 2)
    h = np.zeros(n, dtype=np.float64)
    h[n // 2] = 1.0 / 4.0
    odd = (nn % 2) == 1
    h[odd] = -1.0 / (np.pi * nn[odd]) ** 2
    return h, nn


def ramp_filter(name, flen, d=1.0):
    """Half-spectrum (rfft layout, length flen//2+1) windowed ramp.

    name: one of FILTERS (None -> ram_lak). d: cutoff in (0, 1].
    Built in float64; callers cast as needed.
    """
    if name is None:
        name = "ram_lak"
    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}; options: {FILTERS}")
    if not (0.0 < d <= 1.0):
        raise ValueError(f"filter cutoff d must be in (0, 1], got {d}")
    h, _ = ramp_flat(flen)
    filt = np.abs(np.fft.fft(h))[: flen // 2 + 1] * 2.0
    w = 2.0 * np.pi * np.arange(flen // 2 + 1) / flen  # [0, pi]
    if name == "shepp_logan":
        filt[1:] *= np.sin(w[1:] / (2 * d)) / (w[1:] / (2 * d))
    elif name == "cosine":
        filt[1:] *= np.cos(w[1:] / (2 * d))
    elif name == "hamming":
        filt[1:] *= 0.54 + 0.46 * np.cos(w[1:] / d)
    elif name == "hann":
        filt[1:] *= (1 + np.cos(w[1:] / d)) / 2
    filt[w > np.pi * d] = 0.0
    return filt


def parker_weights(geo, angles, q=1.0):
    """Wesarg/Parker short-scan weights, shape (n, nU) float64.

    geo must be check_geo'd. angles is the (n, 3) validated array; only
    the rotation component angles[:, 0] is used, assumed monotonic
    (either direction; values may be wrapped mod 2*pi). The fan angle
    runs along the detector U axis. Detector offsets and COR are
    ignored (as in TIGRE); large offsets need Wang weighting instead,
    which is out of scope.
    """
    n_u = int(geo.nDetector[1])
    d_u = float(geo.dDetector[1])
    u = (np.arange(n_u) - n_u / 2 + 0.5) * d_u
    rot = np.unwrap(angles[:, 0])
    # Conjugate-ray pairing under the repo convention is
    # (beta, u) <-> (beta + pi - 2*arctan(u/DSD), -u) for an ASCENDING
    # scan; a descending scan mirrors it, so the fan-angle sign follows
    # the rotation direction (a centered phantom is blind to this; an
    # off-center one shows ~3x peak error with the wrong sign).
    direction = np.sign(rot[-1] - rot[0]) or 1.0
    alpha = -direction * np.arctan(u / float(np.mean(geo.DSD)))  # (nU,)
    delta = abs(alpha[0] - alpha[-1]) / 2.0
    beta = np.abs(rot - rot[0])[:, None]  # (n, 1)
    totangles = float(np.abs(rot[-1] - rot[0]))
    if totangles >= 2 * np.pi:
        warnings.warn(
            "Parker weights requested for a scan range >= 2*pi; "
            "a full scan needs no redundancy weighting.")
    if totangles < np.pi + 2 * delta:
        warnings.warn(
            "Scan range below pi + fan angle: limited-angle data; "
            "Parker weighting cannot restore the missing views.")
    epsilon = max(totangles - (np.pi + 2 * delta), 0.0)

    def s(x):
        out = np.zeros_like(x)
        mid = np.abs(x) < 0.5
        out[mid] = 0.5 * (1 + np.sin(np.pi * x[mid]))
        out[x >= 0.5] = 1.0
        return out

    def b(a):
        # b -> 0 at the detector edge when epsilon == 0 (exact pi+2*delta
        # or limited-angle scans); floor it so the transition saturates
        # deterministically instead of dividing by zero (TIGRE inherits
        # the same degeneracy and warns from numpy instead).
        return np.maximum(q * (2 * delta - 2 * a + epsilon), 1e-12)

    w = 0.5 * (
        s(beta / b(alpha) - 0.5)
        + s((beta - 2 * delta + 2 * alpha - epsilon) / b(alpha) + 0.5)
        - s((beta - np.pi + 2 * alpha) / b(-alpha) - 0.5)
        - s((beta - np.pi - 2 * delta - epsilon) / b(-alpha) + 0.5)
    )
    return w


def _mean_step(angles):
    """Mean |rotation step| of a validated (n, 3) angle array.

    The rotation component is unwrapped first: gantry angles often
    arrive wrapped mod 2*pi, and the seam jump would otherwise inflate
    the mean step (~2x) and mis-scale the reconstruction.
    """
    a = np.unwrap(angles[:, 0])
    if a.shape[0] < 2:
        return 2 * np.pi
    steps = np.abs(np.diff(a))
    m = float(np.mean(steps))
    return m if m > 0 else 2 * np.pi / a.shape[0]


def _covered_range(angles):
    """|last - first| of the unwrapped rotation component."""
    a = np.unwrap(angles[:, 0])
    return float(np.abs(a[-1] - a[0]))


def _resolve_parker(parker):
    """None/False -> 0 (off); True -> q=1; positive float -> q."""
    if parker is None or parker is False:
        return 0.0
    if parker is True:
        return 1.0
    q = float(parker)
    if q <= 0:
        return 0.0
    return q


def _filter_core(proj_mx, geo, parker_q, d, name):
    """Filter a validated (n, nV, nU) mx.array; returns mx.array.

    geo must be check_geo'd; name overrides geo.filter when not None.
    """
    n, n_v, n_u = (int(k) for k in proj_mx.shape)
    d_u = float(geo.dDetector[1])
    if name is None:
        name = geo.filter
    flen = max(64, _next_pow2(2 * n_u))
    filt = mx.array(ramp_filter(name, flen, d).astype(np.float32))

    if parker_q > 0:
        w = parker_weights(geo, geo.angles, parker_q)
        proj_mx = proj_mx * mx.array(w.astype(np.float32))[:, None, :]

    # Per-view scale (the x2 baked into ramp_filter composes with the
    # /4 here to the Feldkamp 1/2):
    # - parallel: TIGRE's literal (2*pi/n)/(4*dU). Per view this is
    #   pi/n x ramp, which self-normalizes for BOTH pi- and 2*pi-range
    #   scans (line space has measure pi; uniform coverage assumed) —
    #   verified exact against the analytic oracle for both.
    # - cone: measured mean step / (4*dU); equals TIGRE for standard
    #   full scans (step = 2*pi/n) and stays correct for short scans,
    #   where Parker weights (conjugate pairs summing to 1) replace the
    #   Feldkamp 1/2, hence the doubling.
    if geo.mode == "parallel":
        dbeta = 2 * np.pi / n
    else:
        dbeta = _mean_step(geo.angles)
    scale = (float(geo.DSD[0]) / float(geo.DSO[0])) * dbeta / (4 * d_u)
    if parker_q > 0:
        scale *= 2.0

    rows_per_view = n_v * flen
    max_views = max(1, _FFT_CHUNK_ELEMS // rows_per_view)
    parts = []
    for lo in range(0, n, max_views):
        chunk = proj_mx[lo:lo + max_views]
        spec = mx.fft.rfft(chunk, n=flen, axis=-1)
        filtered = mx.fft.irfft(spec * filt, n=flen, axis=-1)[..., :n_u]
        parts.append(filtered * scale)
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
    return out.astype(mx.float32)


def filtering(proj, geo, angles, parker=False, d=1.0, filter=None,
              return_np=None):
    """Ramp-filter a projection stack (TIGRE's ``filtering`` semantics).

    proj: (n, nV, nU) float32 numpy array or mx.array.
    parker: False/None (off), True (Parker weights, q=1) or a positive
        float q (Wesarg smoothness parameter, as TIGRE's ``parker``).
    d: filter cutoff in (0, 1].
    filter: window name; defaults to geo.filter, else ram_lak.
    Cone-beam cosine pre-weighting is NOT applied here (it belongs to
    FDK, as in TIGRE). Does not mutate proj. Returns float32,
    numpy/mx.array mirroring the input unless return_np is forced.
    """
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

    out = _filter_core(proj_mx, geo, _resolve_parker(parker), d, filter)
    mx.eval(out)
    if return_np is None:
        return_np = not is_mx
    return np.array(out) if return_np else out
