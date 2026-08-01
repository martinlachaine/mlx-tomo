"""Metal projection and backprojection kernels (cone/parallel x interpolated/Siddon).

Design notes
------------
- Per-view ray geometry (uvOrigin, deltaU, deltaV, source) is computed on
  the host in float64 (mlx_tomo.ref.view_geometry — the validated TIGRE
  port) and uploaded as a small float32 buffer: 16 floats per view. The
  GPU pipeline itself is float32 throughout, like TIGRE's.
- Thread mapping is (u, v, view) with u fastest: adjacent SIMD lanes write
  adjacent projection pixels and march through nearby volume regions.
- The interpolated kernel takes the same sample positions as TIGRE's
  ray_interpolated_projection.cu (integer steps i of size `accuracy` voxels
  along S->P), but clips the loop to the ray/volume-support intersection
  analytically instead of TIGRE's conservative crop heuristic; excluded
  samples are exactly the zero-valued ones.
- The Siddon kernel is an incremental digital differential analyzer over
  cells (corner convention), matching TIGRE's Siddon output semantics.
- Kernels are compiled once per (volume dims, detector dims, accuracy,
  voxel size, mode, type) and cached module-wide.

Buffer limits: kernel buffers are int32-indexed by MLX (<= 2**31-1
elements). Projection stacks beyond that are split transparently along
the view axis; volumes beyond it are split into z-slabs (see the slab
notes on ax_gpu/atb_gpu). Each single dispatch stays under the limit.

Slab splitting (volumes > 2^31-1 elements)
------------------------------------------
Forward: Ax is linear in the volume, and both samplers have border-zero
semantics, so Ax(vol) == sum_k Ax(slab_k zero-padded). Each slab kernel
runs in global voxel coordinates (the per-view ray setup is bitwise
identical to the unsplit kernel — same S, P, step count, sample
positions) with the z clip window narrowed from (-1, nz) to
(z0-1, z0+snz) and buffer indices offset by the integer z0. A trilinear
tap (or Siddon cell) in layer j is fetched only by the slab that owns j,
so every tap/cell is counted exactly once by integer comparisons — no
halo layers and no floating-point ownership decisions at the cut plane.
A sample whose support straddles the cut is visited by both neighboring
slabs, each contributing only its own layers' taps; the two partial
sums add to exactly the full trilinear value. Backprojection slabs are
independent output tiles (voxel-driven): the kernel gets fz = zv + z0
and writes a (snz, ny, nx) block; results concatenate along z.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from .ref import view_geometry
from . import tuning

_MK_MAX_ELEMS = 2**31 - 1


def _clamp_tg(tg, gx, gy, gz):
    return (min(tg[0], gx), min(tg[1], gy), min(tg[2] if len(tg) > 2 else 1, gz))
# Chunk threshold for the projection-stack (view) axis; tests lower it to
# exercise the chunked paths at small sizes.
_CHUNK_ELEMS = _MK_MAX_ELEMS
# Slab threshold for the volume (z) axis; tests lower it likewise.
_SLAB_ELEMS = _MK_MAX_ELEMS

_HDR = """#include <metal_math>
#include <metal_geometric>
using namespace metal;
"""


def _interp_source(nx, ny, nz, nu, nv, acc, dlx, dly, dlz, parallel, half, z0):
    # nz is the slab's layer count; z0 its global offset (0, nz_total when
    # unsplit). All geometry runs in global voxel coordinates — only the
    # z clip window and the buffer index change with (z0, nz).
    nxy = nx * ny
    par_s = "S += fu * dU + fv * dV;" if parallel else ""
    ld = "(float)img[" if half else "img["
    z_lo, z_hi = z0, z0 + nz  # slab owns global layers [z_lo, z_hi)
    border = (
        f"#define BORDER_SAMPLE(I) {{ "
        f"float3 t = S + vect * (float)(I); float3 f = floor(t); "
        f"int ix = (int)f.x, iy = (int)f.y, iz = (int)f.z; float3 fr = t - f; "
        f"for (int dz = 0; dz < 2; ++dz) {{ int jz = iz + dz; "
        f"if (jz < {z_lo} || jz >= {z_hi}) continue; float wz = dz ? fr.z : 1.0f - fr.z; "
        f"for (int dy = 0; dy < 2; ++dy) {{ int jy = iy + dy; "
        f"if (jy < 0 || jy >= {ny}) continue; float wy = dy ? fr.y : 1.0f - fr.y; "
        f"for (int dx = 0; dx < 2; ++dx) {{ int jx = ix + dx; "
        f"if (jx < 0 || jx >= {nx}) continue; float wx = dx ? fr.x : 1.0f - fr.x; "
        f"sum += wz * wy * wx * {ld}((uint)(jz - {z_lo}) * {ny}u + (uint)jy) * {nx}u + (uint)jx]; "
        f"}} }} }} }}"
    )
    return f"""
{border}
    uint u = thread_position_in_grid.x;
    uint v = thread_position_in_grid.y;
    uint w = thread_position_in_grid.z;
    if (u >= {nu}u || v >= {nv}u) return;

    const device float* vp = views + (uint)w * 16u;
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

    // Clip sampling to the trilinear support box of the slab's own
    // layers: (-1, N) in x/y, (z0-1, z0+nz) in z (== (-1, N) unsplit).
    // Degenerate axes (|vect| < 1e-9) cannot use the slope form — a
    // signed pseudo-slope reads as a fictitious drift and prematurely
    // clips rays hugging a face (or an interior cut plane) from inside,
    // dropping their low-weight taps — so membership of S in the open
    // support interval decides instead.
    float3 va = float3(abs(vect.x) < 1e-9f ? 1e-9f : vect.x,
                       abs(vect.y) < 1e-9f ? 1e-9f : vect.y,
                       abs(vect.z) < 1e-9f ? 1e-9f : vect.z);
    float3 inv = 1.0f / va;
    float3 blo = float3(-1.0f, -1.0f, {z_lo - 1}.0f);
    float3 bhi = float3({nx}.0f, {ny}.0f, {z_hi}.0f);
    float3 tA = (blo - S) * inv;
    float3 tB = (bhi - S) * inv;
    float3 tmin3 = min(tA, tB), tmax3 = max(tA, tB);
    bool3 deg = abs(vect) < float3(1e-9f);
    bool3 insb = (S > blo) & (S < bhi);
    tmin3 = select(tmin3, select(float3( 3e38f), float3(-3e38f), insb), deg);
    tmax3 = select(tmax3, select(float3(-3e38f), float3( 3e38f), insb), deg);
    float tin  = max(max(tmin3.x, tmin3.y), tmin3.z);
    float tout = min(min(tmax3.x, tmax3.y), tmax3.z);

    // clamp before the int casts: out-of-range float->int is undefined
    int i0 = (int)ceil(clamp(tin, 0.0f, len + 1.0f));
    int i1 = (int)floor(clamp(tout, -1.0f, len));

    // Interior sub-range: every trilinear tap is in-slab, no per-tap
    // guards needed. Boundary-straddling samples run the guarded path.
    // The upper bound is strict (ceil - 1): a sample exactly on it has a
    // weight-0 far tap one plane outside the slab buffer, which must not
    // be fetched. This window is a performance hint only — the fast path
    // verifies its footprint per sample — so degenerate axes need no
    // special handling here.
    float3 q0 = (float3(0.0f, 0.0f, {z_lo}.0f) - S) * inv;
    float3 q1 = (float3({nx - 1}.0f, {ny - 1}.0f, {z_hi - 1}.0f) - S) * inv;
    float qin  = max(max(min(q0.x, q1.x), min(q0.y, q1.y)), min(q0.z, q1.z));
    float qout = min(min(max(q0.x, q1.x), max(q0.y, q1.y)), max(q0.z, q1.z));
    int j0 = max(i0, (int)ceil(clamp(qin, 0.0f, len + 1.0f)));
    int j1 = min(i1, (int)ceil(clamp(qout, -1.0f, len)) - 1);
    if (j0 > j1) {{ j0 = i1 + 1; j1 = i1; }}  // no interior: all guarded

    float sum = 0.0f;
    // guarded boundary samples before the interior
    for (int i = i0; i < j0; ++i) {{
        BORDER_SAMPLE(i)
    }}
    // interior fast path: all 8 taps in-slab. The [j0, j1] window comes
    // from float32 plane crossings while t is evaluated as S + vect*i, so a
    // sample can sit an ulp outside the window's promise; verify the
    // footprint and send the (measure-zero) misses to the guarded path —
    // an unchecked floor one short would wrap the uint index far out of
    // the buffer.
    for (int i = j0; i <= j1; ++i) {{
        float3 t = S + vect * (float)i;
        float3 f = floor(t);
        int jx = (int)f.x, jy = (int)f.y, jz = (int)f.z - {z_lo};
        if (jx < 0 || jx >= {nx - 1} || jy < 0 || jy >= {ny - 1}
            || jz < 0 || jz >= {nz - 1}) {{
            BORDER_SAMPLE(i)
            continue;
        }}
        float3 fr = t - f;
        uint base = ((uint)jz * {ny}u + (uint)jy) * {nx}u + (uint)jx;
        float c00 = mix({ld}base],                {ld}base + 1u],                fr.x);
        float c10 = mix({ld}base + {nx}u],        {ld}base + {nx + 1}u],         fr.x);
        float c01 = mix({ld}base + {nxy}u],       {ld}base + {nxy + 1}u],        fr.x);
        float c11 = mix({ld}base + {nxy + nx}u],  {ld}base + {nxy + nx + 1}u],   fr.x);
        sum += mix(mix(c00, c10, fr.y), mix(c01, c11, fr.y), fr.z);
    }}
    // guarded boundary samples after the interior
    for (int i = max(j1 + 1, j0); i <= i1; ++i) {{
        BORDER_SAMPLE(i)
    }}
    float dl = length(vect * float3({dlx}f, {dly}f, {dlz}f));
    proj[((uint)w * {nv}u + (uint)v) * {nu}u + (uint)u] = sum * dl;
"""


def _siddon_source(nx, ny, nz, nu, nv, dlx, dly, dlz, parallel, half, z0):
    # nz is the slab's layer count; z0 its global offset. The DDA runs in
    # global voxel coordinates, clipped in z to the slab's cell range
    # [z0, z0+nz] so each cell is integrated by exactly one slab (iz is
    # monotonic along the ray).
    par_s = "S += fu * dU + fv * dV;" if parallel else ""
    ld = "(float)img[" if half else "img["
    z_lo, z_hi = z0, z0 + nz
    return f"""
    uint u = thread_position_in_grid.x;
    uint v = thread_position_in_grid.y;
    uint w = thread_position_in_grid.z;
    if (u >= {nu}u || v >= {nv}u) return;

    const device float* vp = views + (uint)w * 16u;
    float3 uvo = float3(vp[0], vp[1], vp[2]);
    float3 dU  = float3(vp[3], vp[4], vp[5]);
    float3 dV  = float3(vp[6], vp[7], vp[8]);
    float3 S   = float3(vp[9], vp[10], vp[11]);
    float fu = (float)u, fv = (float)v;
    float3 P = uvo + fu * dU + fv * dV;
    {par_s}

    float3 d = P - S;
    uint oidx = ((uint)w * {nv}u + (uint)v) * {nu}u + (uint)u;

    float3 va = float3(abs(d.x) < 1e-9f ? 1e-9f : d.x,
                       abs(d.y) < 1e-9f ? 1e-9f : d.y,
                       abs(d.z) < 1e-9f ? 1e-9f : d.z);
    float3 inv = 1.0f / va;
    float3 a0 = (float3(0.0f, 0.0f, {z_lo}.0f) - S) * inv;
    float3 a1 = (float3({nx}.0f, {ny}.0f, {z_hi}.0f) - S) * inv;
    float amin = max(max(min(a0.x, a1.x), min(a0.y, a1.y)), min(a0.z, a1.z));
    float amax = min(min(max(a0.x, a1.x), max(a0.y, a1.y)), max(a0.z, a1.z));
    amin = max(amin, 0.0f);
    amax = min(amax, 1.0f);
    if (amax <= amin) {{ proj[oidx] = 0.0f; return; }}

    // Entry cell (nudged inside).
    float aeps = 1e-6f;
    float3 pin = S + (amin + aeps) * d;
    int ix = clamp((int)floor(pin.x), 0, {nx - 1});
    int iy = clamp((int)floor(pin.y), 0, {ny - 1});
    int iz = clamp((int)floor(pin.z), {z_lo}, {z_hi - 1});

    int sx = d.x > 0.0f ? 1 : -1;
    int sy = d.y > 0.0f ? 1 : -1;
    int sz = d.z > 0.0f ? 1 : -1;
    float dax = abs(inv.x), day = abs(inv.y), daz = abs(inv.z);
    // Degenerate axes never cross a plane.
    bool degx = abs(d.x) < 1e-9f, degy = abs(d.y) < 1e-9f, degz = abs(d.z) < 1e-9f;
    float axn = degx ? INFINITY : ((float)(ix + (sx > 0 ? 1 : 0)) - S.x) * inv.x;
    float ayn = degy ? INFINITY : ((float)(iy + (sy > 0 ? 1 : 0)) - S.y) * inv.y;
    float azn = degz ? INFINITY : ((float)(iz + (sz > 0 ? 1 : 0)) - S.z) * inv.z;

    float sum = 0.0f;
    float a = amin;
    int guard = {2 * (nx + ny + nz) + 4};
    while (a < amax && guard-- > 0) {{
        float anext = min(min(axn, ayn), min(azn, amax));
        float seg = anext - a;
        if (seg > 0.0f && ix >= 0 && ix < {nx} && iy >= 0 && iy < {ny}
            && iz >= {z_lo} && iz < {z_hi}) {{
            sum += seg * {ld}((uint)(iz - {z_lo}) * {ny}u + (uint)iy) * {nx}u + (uint)ix];
        }}
        a = anext;
        if (axn <= ayn && axn <= azn)      {{ ix += sx; axn += dax; }}
        else if (ayn <= azn)               {{ iy += sy; ayn += day; }}
        else                               {{ iz += sz; azn += daz; }}
        if (ix < 0 || ix >= {nx} || iy < 0 || iy >= {ny} || iz < {z_lo} || iz >= {z_hi}) break;
    }}
    float3 dphys = d * float3({dlx}f, {dly}f, {dlz}f);
    proj[oidx] = sum * length(dphys);
"""


_KERNEL_CACHE: dict = {}


def _get_kernel(key, ptype, nx, ny, nz, nu, nv, acc, dl, parallel, half, z0):
    k = _KERNEL_CACHE.get(key)
    if k is not None:
        return k
    suffix = ("h" if half else "f") + (f"_s{z0}" if z0 else "")
    if ptype == "interpolated":
        src = _interp_source(nx, ny, nz, nu, nv, acc, dl[0], dl[1], dl[2], parallel, half, z0)
        name = f"ax_interp_{nx}x{ny}x{nz}_{nu}x{nv}_{suffix}"
    else:
        src = _siddon_source(nx, ny, nz, nu, nv, dl[0], dl[1], dl[2], parallel, half, z0)
        name = f"ax_siddon_{nx}x{ny}x{nz}_{nu}x{nv}_{suffix}"
    k = mx.fast.metal_kernel(
        name=name,
        input_names=["img", "views"],
        output_names=["proj"],
        header=_HDR,
        source=src,
    )
    _KERNEL_CACHE[key] = k
    return k


def build_view_params(geo, ptype):
    """(n, 16) float32 per-view buffer with the output V-flip absorbed.

    Layout per view: uvOrigin(3), deltaU(3), deltaV(3), source(3), pad(4).
    """
    n = geo.angles.shape[0]
    n_v = int(geo.nDetector[0])
    out = np.zeros((n, 16), dtype=np.float64)
    for i in range(n):
        uvo, dU, dV, src = view_geometry(geo, i, ptype)
        # TIGRE writes output row v from pixelV = nV-1-v; absorb the flip.
        uvo_f = uvo + (n_v - 1) * dV
        dV_f = -dV
        if geo.mode == "parallel":
            src = src + (n_v - 1) * dV
        out[i, 0:3] = uvo_f
        out[i, 3:6] = dU
        out[i, 6:9] = dV_f
        out[i, 9:12] = src
    return out.astype(np.float32)


def ax_gpu(img_mx, geo, ptype, views=None):
    """Forward-project a volume. geo must be validated (check_geo).

    img_mx: (nz, ny, nx) float32/float16 mx.array (or numpy array when the
    volume exceeds the slab threshold — slabs are converted one at a time
    to bound peak memory). Returns (n, nV, nU) mx.array.

    views: optional (n, 16) float32 per-view buffer replacing the one
    build_view_params would derive from geo (see mlx_tomo.rays). When given,
    geo's trajectory fields are unused and the view count comes from views.
    """
    nz, ny, nx = (int(k) for k in geo.nVoxel)
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    n = geo.angles.shape[0] if views is None else int(views.shape[0])

    # Oversized projection stacks are split transparently along views.
    if n_v * n_u > _CHUNK_ELEMS:
        raise ValueError(
            f"detector plane ({n_v}x{n_u}) alone exceeds the 2^31-1 "
            "element dispatch limit; view chunking cannot help")
    max_views = max(1, _CHUNK_ELEMS // (n_v * n_u))
    if n > max_views:
        parts = []
        for lo in range(0, n, max_views):
            if views is not None:
                # explicit rays: slice the buffer, geo carries only sizing
                parts.append(ax_gpu(img_mx, geo, ptype,
                                    views=views[lo:lo + max_views]))
                continue
            sub = geo.copy()
            for f in ("DSD", "DSO", "COR"):
                setattr(sub, f, getattr(geo, f)[lo:lo + max_views])
            for f in ("angles", "offOrigin", "offDetector", "rotDetector"):
                setattr(sub, f, getattr(geo, f)[lo:lo + max_views])
            sub.n_proj = sub.angles.shape[0]
            parts.append(ax_gpu(img_mx, sub, ptype))
        return mx.concatenate(parts, axis=0)

    if ptype in ("Siddon", "ray-voxel"):
        ptype = "Siddon"
    elif ptype != "interpolated":
        raise ValueError(f"unknown projection_type {ptype!r}")

    # Oversized volumes: z-slabs, each a zero-padded partial projection;
    # the sum over slabs is the full projection (see module docstring).
    if nx * ny * nz > _SLAB_ELEMS:
        if nx * ny > _SLAB_ELEMS:
            raise ValueError(
                f"volume xy plane ({nx}x{ny}) alone exceeds the 2^31-1 "
                "element dispatch limit; z-slab splitting cannot help")
        layers = max(1, _SLAB_ELEMS // (nx * ny))
        proj = None
        for z0 in range(0, nz, layers):
            part = _ax_dispatch(img_mx[z0:z0 + layers], geo, ptype, z0,
                                views=views)
            proj = part if proj is None else proj + part
            mx.eval(proj)  # bound live memory to one slab at a time
        return proj
    return _ax_dispatch(img_mx, geo, ptype, 0, views=views)


def _ax_dispatch(img_mx, geo, ptype, z0, views=None):
    """Single-dispatch forward projection of one z-slab at offset z0."""
    if not isinstance(img_mx, mx.array):
        img_mx = mx.array(np.ascontiguousarray(img_mx))
    snz, ny, nx = (int(k) for k in img_mx.shape)
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    n = geo.angles.shape[0] if views is None else int(views.shape[0])

    parallel = geo.mode == "parallel"
    half = img_mx.dtype == mx.float16
    dl = tuple(float(x) for x in geo.dVoxel[::-1])  # (x, y, z)
    acc = float(geo.accuracy)
    key = (ptype, nx, ny, snz, n_u, n_v, acc if ptype == "interpolated" else None,
           dl, parallel, half, z0)
    kernel = _get_kernel(key, ptype, nx, ny, snz, n_u, n_v, acc, dl, parallel,
                         half, z0)

    vbuf = mx.array(build_view_params(geo, ptype) if views is None
                    else np.ascontiguousarray(views, dtype=np.float32))
    (proj,) = kernel(
        inputs=[img_mx, vbuf],
        output_shapes=[(n, n_v, n_u)],
        output_dtypes=[mx.float32],
        grid=(n_u, n_v, n),
        # Shapes are measured per machine; see mlx_tomo/tuning.py and
        # https://github.com/martinlachaine/mlx-tomo/blob/main/harness/tune.py (M1: square (8,8) tiles maximize volume-cache
        # sharing for trilinear; the Siddon DDA prefers wide (32,4)).
        threadgroup=_clamp_tg(
            tuning.get("ax_interp" if ptype == "interpolated" else "ax_siddon"),
            n_u, n_v, n),
    )
    return proj


# ----------------------------------------------------------------------
# Backprojection (Atb)

def _atb_source(nx, ny, nz, nu, nv, dvox, svox, ddet, sdet, wmode, z0):
    """Voxel-driven backprojection kernel body.

    wmode: 0 = FDK cone weight, 1 = matched cone weight, 2 = parallel
    (weight 1). dvox/svox are (x, y, z) mm; ddet/sdet are (u, v) mm.
    nz is the output slab's layer count and z0 its global offset: output
    slabs are independent tiles, so the only slab dependence is the
    global voxel z index fz = zv + z0 (svox stays the full-volume size).
    """
    dx, dy, dz = dvox
    sx, sy, sz = svox
    du_mm, dv_mm = ddet
    su_mm, sv_mm = sdet
    if wmode == 2:
        uv = """
        float y = P.y, z = P.z;"""
    else:
        uv = """
        float t = (DSO - DSD - S.x) / (P.x - S.x);
        float y = (P.y - S.y) * t + S.y;
        float z = (P.z - S.z) * t + S.z;"""
    if wmode == 0:
        # U = source-to-voxel-plane distance for the source at
        # (DSO cosA, DSO sinA) — the repo's validated CCW convention.
        # TIGRE's literal CUDA expression (DSO + realy sinA - realx cosA,
        # with COR folded into realy) evaluates U at -A: its wrappers
        # compensate with an axis flip ours does not have. The mirror
        # cancels exactly on full 2pi scans (which is why it survived
        # v0.3-0.6 validation) but shades Parker short scans by ~+-6%
        # antisymmetrically in y. COR does not enter U at all: it shifts
        # source and detector together, perpendicular to the ray axis.
        weight = f"""
        float realx = {-0.5 * (sx - dx)}f + fx * {dx}f + vp[20];
        float realy = {-0.5 * (sy - dy)}f + fy * {dy}f + vp[21];
        float wgt = DSO / (DSO - realx * cosA - realy * sinA);
        wgt = wgt * wgt;"""
    elif wmode == 1:
        # Same frame fix as wmode 0: source at (DSO cosA, DSO sinA) and
        # the detector element rotated by R(+A); the ported expressions
        # used R(-A) (L2 is rotation-invariant, lsq is not).
        weight = f"""
        float rvx = {-0.5 * (sx - dx)}f + fx * {dx}f + vp[20];
        float rvy = {-0.5 * (sy - dy)}f + fy * {dy}f + vp[21];
        float rvz = {-0.5 * (sz - dz)}f + fz * {dz}f + vp[22];
        float rSx = DSO * cosA, rSy = DSO * sinA;
        float rDauxx = -(DSD - DSO);
        float rDauxy = {0.5 * (du_mm - su_mm)}f + u * {du_mm}f + vp[24];
        float rDz    = {0.5 * (dv_mm - sv_mm)}f + v * {dv_mm}f + vp[25];
        float rDx = rDauxx * cosA - rDauxy * sinA;
        float rDy = rDauxx * sinA + rDauxy * cosA;
        float L2 = (rSx - rDx) * (rSx - rDx) + (rSy - rDy) * (rSy - rDy)
                 + rDz * rDz;
        float lsq = (rSx - rvx) * (rSx - rvx) + (rSy - rvy) * (rSy - rvy)
                  + rvz * rvz;
        float wgt = L2 * sqrt(L2) / (DSD * lsq);"""
    else:
        weight = """
        float wgt = 1.0f;"""
    return f"""
    uint x = thread_position_in_grid.x;
    uint yv = thread_position_in_grid.y;
    uint zv = thread_position_in_grid.z;
    if (x >= {nx}u || yv >= {ny}u) return;
    float fx = (float)x, fy = (float)yv, fz = (float)(zv + {z0}u);

    float acc = 0.0f;
    for (uint w = 0; w < NVIEWS; ++w) {{
        const device float* vp = views + w * 32u;
        float3 dX = float3(vp[0], vp[1], vp[2]);
        float3 dY = float3(vp[3], vp[4], vp[5]);
        float3 dZ = float3(vp[6], vp[7], vp[8]);
        float3 org = float3(vp[9], vp[10], vp[11]);
        float3 S = float3(vp[12], vp[13], vp[14]);
        float sinA = vp[15], cosA = vp[16];
        float DSD = vp[18], DSO = vp[19];

        float3 P = org + fx * dX + fy * dY + fz * dZ;
        P.y -= vp[17];  // COR / dDetU
        {uv}
        float u = y + {nu / 2.0}f;
        float v = z + {nv / 2.0}f;

        // bilinear, pixel j centered at j + 0.5, border 0
        float ub = u - 0.5f, vb = v - 0.5f;
        float fub = floor(ub), fvb = floor(vb);
        int iu = (int)fub, iv = (int)fvb;
        float au = ub - fub, av = vb - fvb;
        float s = 0.0f;
        if (iu >= 0 && iu < {nu - 1} && iv >= 0 && iv < {nv - 1}) {{
            uint base = (w * {nv}u + (uint)iv) * {nu}u + (uint)iu;
            float c0 = mix(proj[base],          proj[base + 1u],          au);
            float c1 = mix(proj[base + {nu}u],  proj[base + {nu + 1}u],   au);
            s = mix(c0, c1, av);
        }} else if (iu >= -1 && iu <= {nu - 1} && iv >= -1 && iv <= {nv - 1}) {{
            for (int dvi = 0; dvi < 2; ++dvi) {{
                int jv = iv + dvi;
                if (jv < 0 || jv >= {nv}) continue;
                float wv = dvi ? av : 1.0f - av;
                for (int dui = 0; dui < 2; ++dui) {{
                    int ju = iu + dui;
                    if (ju < 0 || ju >= {nu}) continue;
                    float wu = dui ? au : 1.0f - au;
                    s += wv * wu * proj[(w * {nv}u + (uint)jv) * {nu}u + (uint)ju];
                }}
            }}
        }}
        {weight}
        acc += s * wgt;
    }}
    img[((uint)zv * {ny}u + (uint)yv) * {nx}u + (uint)x] = acc;
"""


def build_bp_view_params(geo):
    """(n, 32) float32 per-view backprojection parameters."""
    from .ref import bp_view_geometry
    n = geo.angles.shape[0]
    d_u = float(geo.dDetector[1])
    out = np.zeros((n, 32), dtype=np.float64)
    for i in range(n):
        o, dX, dY, dZ, S = bp_view_geometry(geo, i)
        out[i, 0:3] = dX
        out[i, 3:6] = dY
        out[i, 6:9] = dZ
        out[i, 9:12] = o
        out[i, 12:15] = S
        out[i, 15] = np.sin(geo.angles[i][0])
        out[i, 16] = np.cos(geo.angles[i][0])
        out[i, 17] = geo.COR[i] / d_u
        out[i, 18] = geo.DSD[i]
        out[i, 19] = geo.DSO[i]
        out[i, 20:23] = geo.offOrigin[i][::-1]
        out[i, 23] = geo.COR[i]
        out[i, 24] = geo.offDetector[i][1]
        out[i, 25] = geo.offDetector[i][0]
    return out.astype(np.float32)


def atb_gpu(proj_mx, geo, bptype):
    """Backproject an MLX projection stack. geo must be validated."""
    nz, ny, nx = (int(k) for k in geo.nVoxel)
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    if bptype not in ("FDK", "matched"):
        raise ValueError(f"unknown backprojection_type {bptype!r}")
    if n_v * n_u > _CHUNK_ELEMS:
        raise ValueError(
            f"detector plane ({n_v}x{n_u}) alone exceeds the 2^31-1 "
            "element dispatch limit; view chunking cannot help")
    # Oversized volumes: independent output z-slabs, concatenated
    # (transient peak ~2x the output volume: all tiles live plus the
    # concatenate copy). Each slab runs the view-chunk accumulation
    # below with a slab-sized accumulator.
    if nx * ny * nz > _SLAB_ELEMS:
        if nx * ny > _SLAB_ELEMS:
            raise ValueError(
                f"volume xy plane ({nx}x{ny}) alone exceeds the 2^31-1 "
                "element dispatch limit; z-slab splitting cannot help")
        layers = max(1, _SLAB_ELEMS // (nx * ny))
        parts = []
        for z0 in range(0, nz, layers):
            part = _atb_dispatch(proj_mx, geo, bptype, z0,
                                 min(layers, nz - z0))
            mx.eval(part)
            parts.append(part)
        return mx.concatenate(parts, axis=0)
    return _atb_dispatch(proj_mx, geo, bptype, 0, nz)


def _atb_dispatch(proj_mx, geo, bptype, z0, snz):
    """Backproject into one output z-slab: layers [z0, z0+snz)."""
    nz, ny, nx = (int(k) for k in geo.nVoxel)
    n_v, n_u = int(geo.nDetector[0]), int(geo.nDetector[1])
    n = geo.angles.shape[0]
    # Oversized projection stacks: backproject view chunks and accumulate.
    max_views = max(1, _CHUNK_ELEMS // (n_v * n_u))
    if n > max_views:
        acc = None
        for lo in range(0, n, max_views):
            sub = geo.copy()
            for f in ("DSD", "DSO", "COR"):
                setattr(sub, f, getattr(geo, f)[lo:lo + max_views])
            for f in ("angles", "offOrigin", "offDetector", "rotDetector"):
                setattr(sub, f, getattr(geo, f)[lo:lo + max_views])
            sub.n_proj = sub.angles.shape[0]
            part = _atb_dispatch(proj_mx[lo:lo + max_views], sub, bptype,
                                 z0, snz)
            acc = part if acc is None else acc + part
        return acc

    parallel = geo.mode == "parallel"
    wmode = 2 if parallel else (0 if bptype == "FDK" else 1)
    dvox = tuple(float(v) for v in geo.dVoxel[::-1])
    svox = tuple(float(v) for v in geo.sVoxel[::-1])
    ddet = (float(geo.dDetector[1]), float(geo.dDetector[0]))
    sdet = (float(geo.sDetector[1]), float(geo.sDetector[0]))

    # svox and sdet are in the key explicitly: everything baked into the
    # generated source must be distinguished (with slabs, snz no longer
    # determines the full-volume physical size the weight terms bake in,
    # and check_geo's 1e-6 tolerance means sdet is not exactly
    # nDetector * dDetector).
    key = ("atb", nx, ny, snz, n_u, n_v, dvox, svox, ddet, sdet, wmode, n, z0)
    kern = _KERNEL_CACHE.get(key)
    if kern is None:
        src = ("#define NVIEWS " + str(n) + "u\n"
               + _atb_source(nx, ny, snz, n_u, n_v, dvox, svox, ddet, sdet,
                             wmode, z0))
        suffix = f"_s{z0}" if z0 else ""
        kern = mx.fast.metal_kernel(
            name=f"atb_{wmode}_{nx}x{ny}x{snz}_{n_u}x{n_v}_n{n}{suffix}",
            input_names=["proj", "views"],
            output_names=["img"],
            header=_HDR,
            source=src,
        )
        _KERNEL_CACHE[key] = kern
    views = mx.array(build_bp_view_params(geo))
    (img,) = kern(
        inputs=[proj_mx, views],
        output_shapes=[(snz, ny, nx)],
        output_dtypes=[mx.float32],
        grid=(nx, ny, snz),
        threadgroup=_clamp_tg(tuning.get("atb"), nx, ny, snz),
    )
    return img
