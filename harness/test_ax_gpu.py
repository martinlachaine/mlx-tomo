"""GPU forward projector vs float64 reference (the TIGRE-port oracle).

The GPU pipeline is float32 with the same sample positions as the
reference, so agreement should be at the float32 floor (~1e-6..1e-5 relative L2),
far below discretization error. Any geometry/convention bug shows up
orders of magnitude above that.
"""

import numpy as np

from gen import default_phantom_mm, ellipsoid_phantom, rel_l2

import mlx_tomo
from mlx_tomo.ref import ax_ref

fails = 0


def check(name, cond, detail=""):
    global fails
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails += 1
    print(f"{status} {name} {detail}")


def run_case(mode, ptype, with_offsets=False, angles_3d=False):
    if mode == "cone":
        geo = mlx_tomo.geometry_default(high_resolution=False)
    else:
        geo = mlx_tomo.geometry(mode="parallel", nVoxel=np.array([32, 48, 64]))
        geo.nDetector = np.array([48, 96])
        geo.sDetector = geo.nDetector * geo.dDetector
    if with_offsets:
        geo.offOrigin = np.array([5.0, -8.0, 3.0])
        geo.offDetector = np.array([4.0, -6.0])
        geo.COR = 2.0 if mode == "cone" else 0.0
        geo.rotDetector = np.array([0.02, -0.03, 0.05]) if mode == "cone" else np.zeros(3)

    if angles_3d:
        rng = np.random.default_rng(7)
        angles = rng.uniform(0, 2 * np.pi, (5, 3))
    else:
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)

    ell = default_phantom_mm(geo)
    vol = ellipsoid_phantom(geo, ell).astype(np.float32)

    ref = ax_ref(vol.astype(np.float64), geo, angles, ptype)
    got = mlx_tomo.Ax(vol, geo, angles, projection_type=ptype)
    err = rel_l2(got, ref)
    tag = f"{mode}/{ptype}" + ("+off" if with_offsets else "") + ("+zyz" if angles_3d else "")
    check(f"gpu-vs-ref {tag}", err < 2e-4, f"rel_l2={err:.2e}")
    return err


if __name__ == "__main__":
    for mode in ("cone", "parallel"):
        for ptype in ("interpolated", "Siddon"):
            run_case(mode, ptype)
    run_case("cone", "interpolated", with_offsets=True)
    run_case("cone", "Siddon", with_offsets=True)
    run_case("cone", "interpolated", angles_3d=True)
    run_case("cone", "Siddon", angles_3d=True)
    run_case("parallel", "interpolated", with_offsets=True)

    # float16 volume storage (opt-in): quantization-limited accuracy
    import mlx.core as mx
    geo = mlx_tomo.geometry_default(high_resolution=False)
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    vol = ellipsoid_phantom(geo, default_phantom_mm(geo)).astype(np.float32)
    ref = ax_ref(vol.astype(np.float64), geo, angles, "interpolated")
    for ptype in ("interpolated", "Siddon"):
        r = ax_ref(vol.astype(np.float64), geo, angles, ptype)
        got = mlx_tomo.Ax(mx.array(vol).astype(mx.float16), geo, angles,
                          projection_type=ptype, return_np=True)
        err = rel_l2(got, r)
        check(f"gpu float16 {ptype}", err < 5e-4, f"rel_l2={err:.2e}")

    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
