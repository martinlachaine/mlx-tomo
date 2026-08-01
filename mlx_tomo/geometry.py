"""TIGRE-compatible scan geometry (cone and parallel beam).

Axis conventions (identical to TIGRE's Python API):

- Volume arrays are indexed ``img[z, y, x]``; ``nVoxel``, ``sVoxel``,
  ``dVoxel`` and ``offOrigin`` are ordered ``(z, y, x)``.
- Projection arrays are indexed ``proj[angle, v, u]``; ``nDetector``,
  ``sDetector``, ``dDetector`` and ``offDetector`` are ordered ``(v, u)``.
  ``v`` runs along the rotation axis (z), ``u`` along the detector row.
- At gantry angle 0 the source sits at ``(x, y, z) = (DSO, 0, 0)`` and the
  detector center at ``(-(DSD - DSO), 0, 0)``. ``u`` increases along +y,
  ``v`` (as stored, row index) increases along +z.
- ``angles`` may be ``(n,)`` (rotation around z) or ``(n, 3)`` ZYZ Euler
  angles ``(alpha, theta, psi)`` in radians. Rotation follows TIGRE's
  eulerZYZ: positive alpha rotates the source counter-clockwise around +z
  when seen from +z.
- ``rotDetector`` is ``(yaw, pitch, roll)`` in radians; ``COR`` is the
  center-of-rotation correction in mm.

Distances are in mm. Projection values are line integrals in units of
(volume value) x mm.
"""

from __future__ import annotations

import copy

import numpy as np

__all__ = ["Geometry", "geometry", "geometry_default", "ParallelGeo"]

_MANDATORY = ("nVoxel", "sVoxel", "dVoxel", "nDetector", "sDetector",
              "dDetector", "DSO", "DSD")


class Geometry:
    """Container for scan geometry, field-compatible with TIGRE."""

    def __init__(self):
        self.mode = None
        self.n_proj = None
        self.angles = None
        self.filter = None
        self.accuracy = 0.5
        self.nVoxel = np.zeros(3)
        self.dVoxel = np.zeros(3)
        self.sVoxel = np.zeros(3)
        self.nDetector = np.zeros(2)
        self.dDetector = np.zeros(2)
        self.sDetector = np.zeros(2)

    # ------------------------------------------------------------------
    def check_geo(self, angles):
        """Validate fields and broadcast per-angle parameters.

        After this call ``self.angles`` is ``(n, 3)`` and DSD, DSO, COR are
        ``(n,)`` while offOrigin is ``(n, 3)``, offDetector ``(n, 2)`` and
        rotDetector ``(n, 3)``.
        """
        angles = np.atleast_1d(np.asarray(angles, dtype=np.float64))
        if angles.ndim == 1:
            self.n_proj = angles.shape[0]
            z = np.zeros((self.n_proj, 1))
            self.angles = np.hstack((angles.reshape(-1, 1), z, z))
        elif angles.ndim == 2 and angles.shape[1] == 3:
            self.n_proj = angles.shape[0]
            self.angles = np.array(angles, dtype=np.float64)
        else:
            raise ValueError(f"angles must be (n,) or (n, 3), got {angles.shape}")

        if self.mode is None:
            self.mode = "cone"
        if self.mode not in ("cone", "parallel"):
            raise ValueError(f"geo.mode must be 'cone' or 'parallel', got {self.mode!r}")

        missing = [a for a in _MANDATORY if not hasattr(self, a)]
        if missing:
            raise AttributeError(f"mandatory geometry fields missing: {missing}")

        for name in ("nVoxel", "sVoxel", "dVoxel"):
            arr = np.asarray(getattr(self, name)).reshape(-1)
            if arr.shape != (3,):
                raise ValueError(f"geo.{name} must have shape (3,)")
            setattr(self, name, arr)
        for name in ("nDetector", "sDetector", "dDetector"):
            arr = np.asarray(getattr(self, name)).reshape(-1)
            if arr.shape != (2,):
                raise ValueError(f"geo.{name} must have shape (2,)")
            setattr(self, name, arr)

        # NaN-safe comparisons: `not (x <= tol)` is True for NaN, `x > tol`
        # is not, so a NaN field must not silently pass validation.
        if not (np.abs(self.dVoxel * self.nVoxel - self.sVoxel).sum() <= 1e-6):
            raise ValueError("nVoxel*dVoxel != sVoxel")
        if not (np.abs(self.dDetector * self.nDetector - self.sDetector).sum() <= 1e-6):
            raise ValueError("nDetector*dDetector != sDetector")

        n = self.n_proj
        self.DSD = self._per_angle(self.DSD, n, "DSD")
        self.DSO = self._per_angle(self.DSO, n, "DSO")
        if not np.all(self.DSD >= self.DSO):
            raise ValueError("DSD must be >= DSO (and finite)")

        self.offOrigin = self._per_angle_vec(getattr(self, "offOrigin", np.zeros(3)), n, 3, "offOrigin")
        self.offDetector = self._per_angle_vec(getattr(self, "offDetector", np.zeros(2)), n, 2, "offDetector")
        self.rotDetector = self._per_angle_vec(getattr(self, "rotDetector", np.zeros(3)), n, 3, "rotDetector")
        self.COR = self._per_angle(getattr(self, "COR", 0.0), n, "COR")

        if not hasattr(self, "accuracy") or self.accuracy is None:
            self.accuracy = 0.5
        self.accuracy = float(self.accuracy)
        if not (0 < self.accuracy <= 8):
            raise ValueError("geo.accuracy must be in (0, 8] voxels/sample")

        # These values reach f-string-generated Metal source; a NaN/Inf
        # would compile to garbage or poison outputs silently.
        for name in ("nVoxel", "sVoxel", "dVoxel", "nDetector", "sDetector",
                     "dDetector", "DSD", "DSO", "offOrigin", "offDetector",
                     "rotDetector", "COR", "angles"):
            if not np.all(np.isfinite(np.asarray(getattr(self, name), dtype=np.float64))):
                raise ValueError(f"geo.{name} contains NaN/Inf")

    # ------------------------------------------------------------------
    @staticmethod
    def _per_angle(val, n, name):
        arr = np.asarray(val, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.full(n, arr[0])
        if arr.shape == (n,):
            return arr
        raise ValueError(f"geo.{name} must be scalar or shape ({n},), got {arr.shape}")

    @staticmethod
    def _per_angle_vec(val, n, k, name):
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim == 1 and arr.shape == (k,):
            return np.tile(arr, (n, 1))
        if arr.shape == (n, k):
            return np.array(arr)
        raise ValueError(f"geo.{name} must have shape ({k},) or ({n}, {k}), got {arr.shape}")

    # ------------------------------------------------------------------
    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        lines = ["mlx-tomo geometry (TIGRE-compatible)",
                 f"  mode        : {self.mode}",
                 f"  DSD / DSO   : {np.atleast_1d(self.DSD)[0]} / {np.atleast_1d(self.DSO)[0]} mm",
                 f"  nVoxel (zyx): {self.nVoxel}",
                 f"  dVoxel (zyx): {self.dVoxel} mm",
                 f"  nDetector(vu): {self.nDetector}",
                 f"  dDetector(vu): {self.dDetector} mm",
                 f"  accuracy    : {self.accuracy}"]
        return "\n".join(lines)


class ParallelGeo(Geometry):
    """Parallel-beam preset, mirrors tigre.geometry(mode='parallel')."""

    def __init__(self, nVoxel):
        if nVoxel is None:
            raise ValueError("nVoxel is required for parallel geometry")
        super().__init__()
        self.mode = "parallel"
        self.nVoxel = np.asarray(nVoxel)
        self.dVoxel = np.array([1.0, 1.0, 1.0])
        self.sVoxel = self.nVoxel.astype(np.float64)

        self.DSO = np.float64(self.nVoxel[0])
        self.DSD = np.float64(self.nVoxel[0] * 2)

        self.dDetector = np.array([1.0, 1.0])
        self.nDetector = np.array([self.nVoxel[0], max(self.nVoxel[1], self.nVoxel[2])], dtype=np.int64)
        self.sDetector = self.nDetector.astype(np.float64)

        self.offOrigin = np.zeros(3)
        self.offDetector = np.zeros(2)
        self.rotDetector = np.zeros(3)
        self.accuracy = 0.5


def geometry(mode="cone", nVoxel=None, default=False, high_resolution=True):
    """TIGRE-compatible geometry constructor."""
    if mode == "cone":
        if default:
            return geometry_default(high_resolution, nVoxel)
        return Geometry()
    if mode == "parallel":
        return ParallelGeo(nVoxel)
    raise ValueError(f"mode {mode!r} not recognized (use 'cone' or 'parallel')")


def geometry_default(high_resolution=True, nVoxel=None):
    """Default cone-beam geometry, identical values to TIGRE's."""
    geo = Geometry()
    geo.mode = "cone"
    geo.DSD = 1536.0
    geo.DSO = 1000.0
    if nVoxel is not None:
        geo.nVoxel = np.asarray(nVoxel)
        geo.nDetector = np.array([nVoxel[1], nVoxel[2]])
        geo.dDetector = np.array([0.8, 0.8])
    elif high_resolution:
        geo.nVoxel = np.array([512, 512, 512])
        geo.nDetector = np.array([512, 512])
        geo.dDetector = np.array([0.8, 0.8])
    else:
        geo.nVoxel = np.array([64, 64, 64])
        geo.nDetector = np.array([128, 128])
        geo.dDetector = np.array([0.8, 0.8]) * 4
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.sVoxel = np.array([256.0, 256.0, 256.0])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.zeros(3)
    geo.offDetector = np.zeros(2)
    geo.rotDetector = np.zeros(3)
    geo.accuracy = 0.5
    geo.filter = None
    return geo
