"""View-axis chunking must be bit-consistent with single-dispatch paths.

Forces an artificially low chunk threshold so the chunked code runs at
small sizes (the real 2^31-1 threshold needs >8 GiB stacks).
"""

import numpy as np

from gen import default_phantom_mm, ellipsoid_phantom, rel_l2

import mlx_tomo
from mlx_tomo import projector as pr

fails = 0


def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"{status} {name} {detail}")


if __name__ == "__main__":
    geo = mlx_tomo.geometry_default(high_resolution=False)
    geo.offDetector = np.array([4.0, -6.0])
    geo.COR = 2.0
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    vol = ellipsoid_phantom(geo, default_phantom_mm(geo)).astype(np.float32)
    n_v, n_u = (int(k) for k in geo.nDetector)
    proj = np.random.default_rng(2).random((7, n_v, n_u)).astype(np.float32)

    ax_full = mlx_tomo.Ax(vol, geo, angles, projection_type="interpolated")
    atb_full = mlx_tomo.Atb(proj, geo, angles, backprojection_type="FDK")

    # Force 3-views-per-chunk (7 views -> chunks of 3+3+1).
    pr._CHUNK_ELEMS = 3 * n_v * n_u
    try:
        ax_chunk = mlx_tomo.Ax(vol, geo, angles, projection_type="interpolated")
        atb_chunk = mlx_tomo.Atb(proj, geo, angles, backprojection_type="FDK")
    finally:
        pr._CHUNK_ELEMS = pr._MK_MAX_ELEMS

    check("Ax chunked == unchunked", rel_l2(ax_chunk, ax_full) == 0.0,
          f"d={rel_l2(ax_chunk, ax_full):.2e}")
    # Atb accumulation reorders float adds across chunks; allow float32 noise.
    d = rel_l2(atb_chunk, atb_full)
    check("Atb chunked == unchunked", d < 1e-6, f"d={d:.2e}")

    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
