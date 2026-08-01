"""Optional hardware-texture backend (interpolated mode only).

Uses Metal 3D textures + the hardware trilinear sampler through a small
C-ABI dylib (texture_bridge/), which mx.fast.metal_kernel cannot access.
Sampling semantics match the buffer backend and TIGRE: voxel j centered at
normalized coordinate (j + 0.5)/N, zero outside (clamp_to_zero == CUDA
border). Measured accuracy on an Apple M1: relative L2 error of approximately 6e-5
against the float64 reference, close to the buffer backend's float32 floor.

Measured on an Apple M1 relative to the buffer backend: approximately 1.3x
lower single-view latency, approximately 1.3x faster with float16 volumes,
and roughly at parity for float32 many-view batches. These ratios depend on
texture-unit count and filtering rate and will differ on other GPUs.

Primary API is the plan: upload the volume once, project many times.

    plan = TextureProjector(img, geo)          # texture upload happens here
    proj = plan.project(angles)                # fast, repeatable

Unlike the buffer backend, results land in host numpy memory (the GPU
writes into a page-aligned numpy buffer zero-copy when possible);
return_np=False re-wraps as mx.array with one copy.

`Ax(..., backend="texture")` constructs a plan on each call, so the volume
upload cost is included every time. Use TextureProjector directly when
projecting the same volume repeatedly.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

import mlx.core as mx

from . import tuning
from .projector import _MK_MAX_ELEMS, build_view_params

_lib = None
_load_error = None

_CAND = [
    os.path.join(os.path.dirname(__file__), "..", "texture_bridge",
                 "libmlxtomo_bridge.dylib"),
    os.path.join(os.path.dirname(__file__), "libmlxtomo_bridge.dylib"),
]


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return
    paths = ([os.environ["MLX_TOMO_BRIDGE_LIB"]]
             if "MLX_TOMO_BRIDGE_LIB" in os.environ else _CAND)
    for p in paths:
        if os.path.exists(p):
            try:
                lib = ctypes.CDLL(os.path.abspath(p))
            except OSError as e:
                _load_error = f"failed to load {p}: {e}"
                return
            lib.mcb_error.restype = ctypes.c_char_p
            lib.mcb_init.restype = ctypes.c_int
            lib.mcb_compile.restype = ctypes.c_void_p
            lib.mcb_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            lib.mcb_texture3d.restype = ctypes.c_void_p
            lib.mcb_texture3d.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
            lib.mcb_project.restype = ctypes.c_int
            lib.mcb_project.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.mcb_release.argtypes = [ctypes.c_void_p]
            _lib = lib
            return
    _load_error = ("texture bridge dylib not found; build it with "
                   "texture_bridge/build.sh or set MLX_TOMO_BRIDGE_LIB")


def available():
    _load()
    return _lib is not None


def require():
    _load()
    if _lib is None:
        raise RuntimeError(f"texture backend unavailable: {_load_error}")
    if _lib.mcb_init() != 0:
        raise RuntimeError(f"texture backend init failed: "
                           f"{_lib.mcb_error().decode()}")


_PAGE = 16384  # Apple silicon page size


def _tex_kernel_source(nx, ny, nz, nu, nv, acc, dlx, dly, dlz, parallel):
    par_s = "S += fu * dU + fv * dV;" if parallel else ""
    return f"""
#include <metal_stdlib>
using namespace metal;

kernel void ax_tex(texture3d<float, access::sample> vol [[texture(0)]],
                   const device float* views [[buffer(0)]],
                   device float* proj [[buffer(1)]],
                   uint3 tid [[thread_position_in_grid]]) {{
    uint u = tid.x, v = tid.y, w = tid.z;
    if (u >= {nu}u || v >= {nv}u) return;

    const device float* vp = views + w * 16u;
    float3 uvo = float3(vp[0], vp[1], vp[2]);
    float3 dU  = float3(vp[3], vp[4], vp[5]);
    float3 dV  = float3(vp[6], vp[7], vp[8]);
    float3 S   = float3(vp[9], vp[10], vp[11]);
    float fu = (float)u, fv = (float)v;
    float3 P = uvo + fu * dU + fv * dV;
    {par_s}

    float3 d = P - S;
    float len = ceil(length(d) / {acc}f);
    float3 vect = d / len;

    float3 va = float3(abs(vect.x) < 1e-9f ? 1e-9f : vect.x,
                       abs(vect.y) < 1e-9f ? 1e-9f : vect.y,
                       abs(vect.z) < 1e-9f ? 1e-9f : vect.z);
    float3 inv = 1.0f / va;
    float3 t0 = (float3(-1.0f, -1.0f, -1.0f) - S) * inv;
    float3 t1 = (float3({nx}.0f, {ny}.0f, {nz}.0f) - S) * inv;
    float tin  = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    float tout = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    int i0 = max(0, (int)ceil(tin));
    int i1 = min((int)len, (int)floor(tout));

    constexpr sampler smp(coord::normalized, address::clamp_to_zero,
                          filter::linear);
    const float3 invN = float3({1.0 / nx}f, {1.0 / ny}f, {1.0 / nz}f);
    float sum = 0.0f;
    for (int i = i0; i <= i1; ++i) {{
        float3 t = S + vect * (float)i;
        sum += vol.sample(smp, (t + 0.5f) * invN).r;
    }}
    float dl = length(vect * float3({dlx}f, {dly}f, {dlz}f));
    proj[(w * {nv}u + v) * {nu}u + u] = sum * dl;
}}
"""


_PSO_CACHE: dict = {}


class TextureProjector:
    """Plan-style forward projector on a hardware 3D texture.

    img: (nz, ny, nx) float32 or float16 (numpy or mx.array). The volume is
    copied into an r32Float/r16Float texture once at construction.
    geo: geometry without angles (angles go to project()).
    Only projection_type='interpolated' exists on this backend; use the
    buffer backend for Siddon.
    """

    def __init__(self, img, geo, projection_type="interpolated"):
        require()
        if projection_type != "interpolated":
            raise ValueError("texture backend supports only 'interpolated'")
        if isinstance(img, mx.array):
            img = np.array(img)
        img = np.ascontiguousarray(img)
        if img.dtype not in (np.float32, np.float16):
            raise TypeError(f"img must be float32/float16, got {img.dtype}")
        # Keep a pristine copy; only static fields are needed at plan time
        # (angles and per-view fields are handled in project()).
        self.geo = geo.copy()
        if self.geo.mode is None:
            self.geo.mode = "cone"
        if not hasattr(self.geo, "accuracy") or self.geo.accuracy is None:
            self.geo.accuracy = 0.5
        nz, ny, nx = (int(k) for k in self.geo.nVoxel)
        if img.shape != (nz, ny, nx):
            raise ValueError(f"img shape {img.shape} != geo.nVoxel {(nz, ny, nx)}")
        # Metal 3D textures max out at 2048 per axis (the buffer backend
        # only has the 2^31-element cap); guard here because Metal aborts
        # the process on oversized descriptors rather than erroring.
        if not all(1 <= d <= 2048 for d in (nx, ny, nz)):
            raise ValueError(
                f"texture backend requires 1 <= dim <= 2048 per axis, got "
                f"{(nz, ny, nx)}; use the buffer backend for larger volumes")
        self._dims = (nx, ny, nz)
        n_v, n_u = int(self.geo.nDetector[0]), int(self.geo.nDetector[1])
        self._det = (n_u, n_v)

        half = img.dtype == np.float16
        self._tex = _lib.mcb_texture3d(
            img.ctypes.data_as(ctypes.c_void_p), nx, ny, nz, int(half))
        if not self._tex:
            raise RuntimeError(f"texture upload failed: {_lib.mcb_error().decode()}")

        dl = tuple(float(x) for x in self.geo.dVoxel[::-1])
        src = _tex_kernel_source(nx, ny, nz, n_u, n_v, float(self.geo.accuracy),
                                 dl[0], dl[1], dl[2],
                                 self.geo.mode == "parallel")
        key = src
        pso = _PSO_CACHE.get(key)
        if pso is None:
            pso = _lib.mcb_compile(src.encode(), b"ax_tex")
            if not pso:
                raise RuntimeError(f"kernel compile failed: "
                                   f"{_lib.mcb_error().decode()}")
            _PSO_CACHE[key] = pso
        self._pso = pso

    def project(self, angles, return_np=True, threadgroup=None):
        """Forward projection at `angles`: (n, nV, nU) float32."""
        if threadgroup is None:
            threadgroup = tuning.get("texture")
        if self._tex is None:
            raise RuntimeError("TextureProjector is closed")
        tgx, tgy = (int(t) for t in threadgroup)
        if tgx < 1 or tgy < 1 or tgx * tgy > 1024:
            raise ValueError(f"invalid threadgroup {threadgroup}; "
                             "need 1 <= tgx*tgy <= 1024")
        geo = self.geo.copy()
        geo.check_geo(angles)
        n = geo.angles.shape[0]
        n_u, n_v = self._det
        if n == 0:
            out = np.zeros((0, n_v, n_u), dtype=np.float32)
            return out if return_np else mx.array(out)
        if n * n_u * n_v > _MK_MAX_ELEMS:
            raise ValueError("projection stack exceeds 2^31-1 elements")

        views = np.ascontiguousarray(build_view_params(geo, "interpolated"))
        # Page-padded numpy target: large macOS allocations are page-aligned,
        # so the bridge's zero-copy (bytesNoCopy) path engages and the GPU
        # writes straight into this buffer; otherwise it stages + memcpys.
        nfloats = n * n_v * n_u
        padded = nfloats + (-nfloats * 4) % _PAGE // 4
        raw = np.empty(padded, dtype=np.float32)
        rc = _lib.mcb_project(
            self._pso, self._tex,
            views.ctypes.data_as(ctypes.c_void_p), n,
            raw.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint64(padded * 4),
            n_u, n_v, tgx, tgy)
        if rc != 0:
            raise RuntimeError(f"texture projection failed (rc={rc}): "
                               f"{_lib.mcb_error().decode()}")
        out = raw[:nfloats].reshape(n, n_v, n_u)
        return out if return_np else mx.array(out)

    def close(self):
        # Swap-then-release narrows the double-release window if close()
        # races __del__ on another thread.
        tex, self._tex = getattr(self, "_tex", None), None
        if tex:
            _lib.mcb_release(tex)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
