# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-18

### Changed

- The interpolated projector keeps the current cell's eight corners in
  registers and reloads them only when a sample enters a new cell, with
  identical sums. 512³ → 512² at 100 views on an M5 Max: 1.90 to 1.49 ms/view
  (with the launch shape re-tuned by `harness/tune.py`).
- Backprojection on circular trajectories with an unrotated detector derives
  the FDK distance weight from the perspective divide's reciprocal, one division
  per voxel and view instead of two, and both backprojection kernels compile
  with Metal's relaxed math mode. 512³ ← 512² at 100 views on an M5 Max: FDK
  1.61 to 0.92 ms/view, matched 1.85 to 1.30; results within 1e-7 of 0.1.0.
- The hardware-texture projector uploads volumes into GPU-private tiled storage,
  with a zero-copy blit for page- and row-aligned volumes and a z-chunked staging
  buffer capped at 256 MiB otherwise. 512³ at 100 views on an M5 Max: 7.49 to
  3.77 ms/view float32, 4.58 to 2.37 float16.
- The shipped threadgroup defaults are re-measured on the M1 with the new
  kernels: interpolated (8, 8, 2), backprojection (8, 8, 1).
- `mlx` is pinned to 0.32.4.
- The benchmarks report nine-run medians, distinguish GPU- and host-resident
  output, exclude shader compilation from upload timing and write JSON results;
  the validation runner imports the current checkout.

## [0.1.0] - 2026-08-02

Initial public release of mlx-tomo. The API is usable but may evolve during the
0.x release series; pin a released version for reproducible environments.

### Added

- **Geometry**: TIGRE-compatible container (`geometry`, `geometry_default`),
  cone and parallel modes, per-angle `DSD`/`DSO`/`COR`/`offOrigin`/`offDetector`/
  `rotDetector`, validated and broadcast by `check_geo`.
- **Forward projection** (`Ax`): `interpolated` (trilinear) and `Siddon`
  (exact ray-voxel intersection) Metal kernels, cone and parallel.
- **Backprojection** (`Atb`): `FDK` (distance-weighted, for analytic
  reconstruction) and `matched` (approximate adjoint of the interpolated `Ax`;
  see README for the measured mismatch).
- **Arbitrary rays** (`Ax_rays`, `rays_from_geometry`, `affine_rays`): 
  projection along explicitly specified source/direction pairs, for DRR
  generation, calibrated non-ideal orbits, and rigid or affine transforms
  applied in ray space.
- **Analytic reconstruction** (`FDK`, `FBP`, `filtering`): ramp filters with
  Hann/Hamming/Shepp-Logan/cosine windows, Wesarg Parker weights auto-applied
  for short scans, piercing-point-aware cosine weights for offset detectors.
- **Hardware-texture backend** (optional): Metal 3D-texture sampling for the
  interpolated projector via a small C-ABI bridge (`texture_bridge/`).
- **Large-problem handling**: projection stacks split transparently along the
  view axis; volumes split into z-slabs, so sizes beyond the int32 indexing
  limit of a single Metal dispatch work unchanged (2048³ validated).
- **Per-device tuning** (`harness/tune.py`): measures device bandwidth, sweeps
  kernel threadgroup shapes, writes `mlx_tomo/tuning_local.json`.
- **Phantom** (`shepp_logan_mm`, `rasterize`): 3D modified Shepp-Logan in mm
  units on the library's geometry convention.
- **Validation harness**: float64 reference projector (`mlx_tomo/ref.py`),
  analytic ellipsoid line-integral oracle, impulse-placement tests, adjoint
  identities, and slab-splitting regression tests.

### Known limitations

- `Atb(backprojection_type="matched")` is an approximate adjoint of the
  interpolated forward projector. For the validation geometry documented in
  `docs/validation.md`, the normalized discrepancy was 1.8e-4 for random vectors
  and 2.1e-3 for smoothed random vectors. Adjoint inconsistency may affect
  iterative methods, so users should evaluate convergence for their own
  geometry. An exact scatter-based transpose is not implemented.
- Half-fan `dowang` offset-detector reconstruction is not implemented.
- The texture backend is capped at 2048 elements per axis by Metal.
- MLX reductions flatten and therefore reject arrays over 2^31-1 elements;
  reduce very large volumes in chunks.
- `mlx` is pinned to 0.32.0 because the custom Metal kernels are validated
  against it.
