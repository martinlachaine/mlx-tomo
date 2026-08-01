"""Texture backend vs float64 reference and vs the buffer backend.

The acceptance gate is looser than the buffer backend's float32 floor to
allow for hardware filtering differences across GPU generations; measured
error on M1 is ~6e-5 (Apple samplers filter r32Float at high precision).
The test prints the measured error so regressions and platform
differences are visible.
Skips cleanly when the bridge dylib is not built.
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


def main():
    if not mlx_tomo.texture_available():
        print("SKIP texture bridge not built (texture_bridge/build.sh)")
        return 0

    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)

    for mode in ("cone", "parallel"):
        if mode == "cone":
            geo = mlx_tomo.geometry_default(high_resolution=False)
            geo.offOrigin = np.array([5.0, -8.0, 3.0])
            geo.offDetector = np.array([4.0, -6.0])
        else:
            geo = mlx_tomo.geometry(mode="parallel", nVoxel=np.array([64, 64, 64]))
        vol = ellipsoid_phantom(geo, default_phantom_mm(geo)).astype(np.float32)
        ref = ax_ref(vol.astype(np.float64), geo, angles, "interpolated")

        got = mlx_tomo.Ax(vol, geo, angles, projection_type="interpolated",
                          backend="texture")
        err = rel_l2(got, ref)
        check(f"texture-vs-ref {mode}", err < 5e-3, f"rel_l2={err:.2e}")

        buf = mlx_tomo.Ax(vol, geo, angles, projection_type="interpolated")
        err_b = rel_l2(got, buf)
        check(f"texture-vs-buffer {mode}", err_b < 5e-3, f"rel_l2={err_b:.2e}")

    # Plan reuse: two project() calls must agree with one-shot Ax.
    geo = mlx_tomo.geometry_default(high_resolution=False)
    vol = ellipsoid_phantom(geo, default_phantom_mm(geo)).astype(np.float32)
    plan = mlx_tomo.TextureProjector(vol, geo)
    p1 = plan.project(angles)
    p2 = plan.project(angles[:3])
    one = mlx_tomo.Ax(vol, geo, angles, projection_type="interpolated",
                      backend="texture")
    check("plan repeat consistency", rel_l2(p1, one) == 0.0 and
          rel_l2(p2, one[:3]) == 0.0,
          f"d1={rel_l2(p1, one):.1e} d2={rel_l2(p2, one[:3]):.1e}")
    plan.close()

    # float16 texture
    plan16 = mlx_tomo.TextureProjector(vol.astype(np.float16), geo)
    ref = ax_ref(vol.astype(np.float64), geo, angles, "interpolated")
    err16 = rel_l2(plan16.project(angles), ref)
    check("texture float16", err16 < 5e-3, f"rel_l2={err16:.2e}")
    plan16.close()

    # Guard-rail regressions (each previously killed the process or
    # silently returned wrong data):
    plan = mlx_tomo.TextureProjector(vol, geo)
    try:
        plan.project(angles, threadgroup=(64, 32))
        check("oversized threadgroup rejected", False, "no exception")
    except (ValueError, RuntimeError) as e:
        check("oversized threadgroup rejected", True, type(e).__name__)
    empty = plan.project(np.zeros((0,)))
    check("empty angles -> empty result", empty.shape[0] == 0, f"{empty.shape}")
    plan.close()
    try:
        plan.project(angles)
        check("project-after-close raises", False, "no exception")
    except RuntimeError:
        check("project-after-close raises", True)

    big = mlx_tomo.geometry_default(high_resolution=False)
    big.nVoxel = np.array([16, 16, 4096])
    big.dVoxel = big.sVoxel / big.nVoxel
    try:
        mlx_tomo.TextureProjector(
            np.zeros((16, 16, 4096), dtype=np.float32), big)
        check(">2048 axis rejected", False, "no exception")
    except ValueError:
        check(">2048 axis rejected", True)

    # isinstance must work on the re-exported class
    from mlx_tomo.texture import TextureProjector as TP
    check("re-export is the class", mlx_tomo.TextureProjector is TP)
    return 0


if __name__ == "__main__":
    main()
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
