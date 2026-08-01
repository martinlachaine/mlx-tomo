# mlx-tomo

GPU-accelerated X-ray projection and backprojection for Apple silicon.

[![tests](https://github.com/martinlachaine/mlx-tomo/actions/workflows/ci.yml/badge.svg)](https://github.com/martinlachaine/mlx-tomo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlx-tomo.svg)](https://pypi.org/project/mlx-tomo/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/martinlachaine/mlx-tomo/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/martinlachaine/mlx-tomo/blob/main/pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-lightgrey.svg)](#requirements-and-installation)

mlx-tomo provides parallel-beam, fan-beam, and cone-beam forward and
backprojection using custom Metal kernels through MLX. It offers a
TIGRE-compatible Python interface for supported operations, making it easier to
adapt existing tomography workflows to Apple GPUs.

Host-side geometry calculations use float64, while GPU computation uses float32.
Numerical accuracy is validated against an independent float64 reference
implementation, analytic ellipsoid projections, and impulse-placement tests.

The package includes projection operators, geometry utilities, and analytic FDK
and FBP reconstruction. Iterative and learned reconstruction algorithms are
intentionally outside its current scope.

mlx-tomo is research software and has not been validated for clinical or
diagnostic use.

## Requirements and installation

Requires an Apple silicon Mac (Metal/MLX) and Python 3.11 or newer.

```bash
pip install mlx-tomo
```

Or install a tagged release from GitHub:

```bash
pip install "git+https://github.com/martinlachaine/mlx-tomo.git@v0.1.0"
```

`mlx` is pinned to a validated version; `numpy` has a lower bound. Check the
install with:

```python
import numpy as np
import mlx_tomo

geo = mlx_tomo.geometry_default(high_resolution=False)
proj = mlx_tomo.Ax(np.ones(tuple(geo.nVoxel), np.float32), geo, np.array([0.0]))
print(mlx_tomo.__version__, proj.shape)
```

### Development installation

```bash
git clone https://github.com/martinlachaine/mlx-tomo && cd mlx-tomo
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -e ".[dev]"
cd texture_bridge && ./build.sh && cd ..   # optional texture backend
.venv/bin/python harness/run_tests.py      # full correctness suite
```

## Quickstart

```python
import numpy as np
import mlx_tomo as tigre

geo = tigre.geometry_default(high_resolution=False)   # 64^3 volume, 128^2 detector
vol = tigre.rasterize(geo, tigre.shepp_logan_mm(scale_mm=100.0))

angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
proj = tigre.Ax(vol, geo, angles, projection_type="interpolated")
rec = tigre.FDK(proj, geo, angles)
```

`proj` is `(n_angles, nDetector[0], nDetector[1])` float32 line integrals in
millimeter units; `rec` matches `geo.nVoxel` as `[z, y, x]`.

`examples/quickstart.py` runs cone and parallel projections and both
reconstructions end to end.

## API

| | |
|---|---|
| `geometry(mode=...)`, `geometry_default()` | geometry container |
| `Ax(img, geo, angles, projection_type)` | forward projection; `"interpolated"` or `"Siddon"` |
| `Atb(proj, geo, angles, backprojection_type)` | backprojection; `"FDK"` or `"matched"` |
| `Ax_rays`, `rays_from_geometry`, `affine_rays` | projection along user-supplied rays |
| `FDK`, `FBP`, `filtering` | analytic reconstruction; ramp filters, windows, Parker weights |
| `TextureProjector`, `texture_available` | optional hardware-texture backend |
| `shepp_logan_mm`, `rasterize` | 3D Shepp-Logan phantom in millimeter units |

Field names, array layouts and call signatures follow TIGRE. See
[docs/compatibility.md](https://github.com/martinlachaine/mlx-tomo/blob/main/docs/compatibility.md) for what is and is not supported,
and where behavior differs.

## Limitations

- **Iterative and learned reconstruction are not included** — the package
  provides the operators these methods are built from.
- **`Atb(backprojection_type="matched")` is an approximate adjoint.** Ray-driven
  forward projection and voxel-driven backprojection are not exact transposes.
  The measured discrepancy is 1.8e-4 relative on random test vectors; see
  [docs/validation.md](https://github.com/martinlachaine/mlx-tomo/blob/main/docs/validation.md) for the method and for guidance if you
  intend to use it in an iterative solver.
- **Half-fan offset-detector reconstruction (`dowang`) is not implemented**, so
  transaxially truncated half-fan data is not handled.
- **MLX reductions reject arrays over 2^31-1 elements.** Very large volumes must
  be reduced in chunks.
- **The hardware-texture backend is capped at 2048 elements per axis** by Metal.
- **`mlx` is pinned**, because the custom Metal kernels are validated against a
  specific version.

## Validation

Correctness rests on three complementary layers: a float64 NumPy reference
implementation that the GPU kernels must agree with to approximately 1e-5
relative L2 in the tested configurations, an analytic ellipsoid line-integral
oracle sharing no code with that reference, and impulse-placement tests. Adjoint
identities and analytic reconstruction comparisons are also checked.

Volumes and projection stacks beyond 2^31-1 elements are handled by view and
z-slab splitting, validated at 2048³ against the analytic oracle.

Full detail and how to run everything: [docs/validation.md](https://github.com/martinlachaine/mlx-tomo/blob/main/docs/validation.md).

## Performance

Milliseconds per view, median of nine timed repetitions after a warm-up, on an
Apple M5 Max (40-core GPU, 128 GB, macOS 26.5.1, Python 3.14.6, MLX 0.32.0,
mains power, no other GPU load):

| problem | views | interpolated | Siddon | Atb (FDK) |
|---|---|---|---|---|
| 256³ ↔ 256² | 100 | 0.28 | 0.15 | 0.23 |
| 512³ ↔ 512² | 100 | 2.08 | 0.98 | 1.67 |

Observed scaling across machines is consistent with bandwidth-bound execution.
Kernel launch parameters are measured per device; run `harness/tune.py` on your
own hardware.

Methodology, run-to-run spread, Apple M1 figures, the large-volume measurement,
and tuning guidance: [docs/performance.md](https://github.com/martinlachaine/mlx-tomo/blob/main/docs/performance.md).

## Documentation

- [docs/compatibility.md](https://github.com/martinlachaine/mlx-tomo/blob/main/docs/compatibility.md) — supported API, and behavioral
  differences from TIGRE
- [docs/validation.md](https://github.com/martinlachaine/mlx-tomo/blob/main/docs/validation.md) — how correctness is established, how
  to run the suite, adjoint consistency
- [docs/performance.md](https://github.com/martinlachaine/mlx-tomo/blob/main/docs/performance.md) — benchmark methodology and results
- [docs/implementation.md](https://github.com/martinlachaine/mlx-tomo/blob/main/docs/implementation.md) — precision model, kernels,
  array splitting, tuning
- [CONTRIBUTING.md](https://github.com/martinlachaine/mlx-tomo/blob/main/CONTRIBUTING.md) — bug reports and development setup
- [CHANGELOG.md](https://github.com/martinlachaine/mlx-tomo/blob/main/CHANGELOG.md) — release history

## Development and validation

Generative AI tools assisted with implementation, refactoring, test scaffolding,
and documentation. The author defined the project's scope and numerical
requirements, reviewed the implementation, and is responsible for validation and
releases.

## License and citation

BSD-3-Clause; see [LICENSE](https://github.com/martinlachaine/mlx-tomo/blob/main/LICENSE). mlx-tomo adopts TIGRE's geometry
conventions and was validated against TIGRE; see [NOTICE](https://github.com/martinlachaine/mlx-tomo/blob/main/NOTICE) for
attribution and the TIGRE paper to cite, and [CITATION.cff](https://github.com/martinlachaine/mlx-tomo/blob/main/CITATION.cff) to
cite mlx-tomo itself.
