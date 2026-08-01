# TIGRE compatibility

mlx-tomo provides a TIGRE-compatible Python interface for the operations it
implements. Geometry field names, array layouts, angle conventions and the
`Ax` / `Atb` / `FDK` / `FBP` call signatures follow TIGRE, so many TIGRE
workflows can be adapted by changing the import.

This document lists what is supported, what is not, and where mlx-tomo's
behavior differs from the TIGRE-Python implementation consulted during
development.

> **Reference version.** The behavioral comparisons below were made against
> `CERN/TIGRE` at commit
> [`e70247328fd897ca5dee1351893b66efe741dc73`](https://github.com/CERN/TIGRE/tree/e70247328fd897ca5dee1351893b66efe741dc73)
> (`master`, 2026-06-25), specifically
> `Python/tigre/utilities/filtering.py`,
> `Python/tigre/utilities/parkerweight.py` and
> `Python/tigre/algorithms/single_pass_algorithms.py`.
> TIGRE continues to evolve, so if parity matters to your work, compare against
> your own installed version.

## Supported

| Feature | Status |
|---|---|
| `geometry`, `geometry_default`, `Geometry`, `ParallelGeo` | supported |
| Cone and parallel modes | supported |
| Per-angle `DSD`, `DSO`, `COR`, `offOrigin`, `offDetector`, `rotDetector` | supported |
| `Ax` — `interpolated`, `Siddon` | supported |
| `Atb` — `FDK`, `matched` (legacy `krylov=` alias honored) | supported |
| `FDK`, `FBP`, `filtering` | supported |
| Euler `ZYZ` angle triples | supported |
| `gpuids` | accepted, ignored, and warned about (single Apple GPU) |
| Iterative solvers (SART, OS-SART, CGLS, ASD-POCS, …) | not implemented — out of scope |
| Half-fan offset-detector reconstruction (`dowang`) | not implemented |
| Multi-GPU execution | not supported |

## Behavioral differences

Each row states what mlx-tomo does, why, and the measured effect if you were
relying on the other behavior.

| Area | mlx-tomo behavior | Rationale and measured effect |
|---|---|---|
| Trilinear interpolation | Full float32 weights | CUDA texture units interpolate with 9-bit fractional weights. mlx-tomo's `interpolated` mode therefore does not reproduce that quantization, and will not match a texture-based implementation to better than that quantization level. |
| `offDetector` in parallel interpolated mode | Applied once, in all modes | Applying it once is consistent across modes. If a workflow was tuned against an implementation that applies it twice in this mode, parallel projections will shift by one `offDetector` amount. |
| Ramp filter padding | Pads to the smallest power of two ≥ 2·nU | Avoids over-padding: a 512-wide detector pads to 1024 rather than 2048. Affects only speed and memory, not values. |
| Per-view FDK/FBP scale | Uses the measured mean angular step | A fixed 2π/n scale is correct for full scans but mis-scales a short scan by approximately π/(angular range). Full-scan values are unchanged. |
| Parker short-scan weights | Wesarg (2002) formulation applied along the U axis; auto-applied when coverage < 2π, disable with `parker=False` | Short-scan reconstructions are redundancy-weighted by default. Full scans are unaffected. A warning is emitted when weights are applied automatically. |
| Offset-detector weighting | Piercing-point-aware cosine weights | Correct for untruncated projections. The Wang-style zero-pad path used for truncated half-fan data is not implemented, so genuinely truncated half-fan data is not handled. |
| `FDK` on a parallel geometry | Dispatches to `FBP` | Convenience; `parker` is meaningless there and ignored with a warning. |
| `FDK(filter=None)` | Honors a pre-set `geo.filter` | Lets the geometry object carry the filter choice. Pass `filter=` explicitly to override. |
| `filtering` | Does not mutate its input | Pure function. Code relying on in-place modification must take the return value. |
| `Atb` distance weights | Frame convention verified against the float64 reference and the analytic oracle; `COR` is excluded from the weight | The weight is a distance ratio, and center-of-rotation offset does not enter it. Full scans are unaffected; on Parker-weighted short scans the difference reaches approximately ±6% in soft-tissue shading. |
| Nonuniform angular sampling | The per-view scale uses the *mean* rotation step | Uniformly sampled scans, full or short, are handled correctly. Materially nonuniform angles are averaged rather than weighted per view, so clustered views are under-weighted relative to sparse regions; scale each view by its own share of the arc before filtering — see [implementation.md](implementation.md) for the exact operation. |
| Automatic Parker weighting | Requires a strictly monotonic arc in the first Euler angle with the other two constant; warns when applied | Other trajectories receive no weighting unless `parker` is passed explicitly, because Parker weighting is undefined for them. A second warning is emitted when coverage falls below pi plus the full detector fan angle, the minimum for a valid short scan — note that weights are still applied in that case, so the warning reports insufficient data rather than declining to weight. |
| Large arrays | Projection stacks split along the view axis; volumes split into z-slabs | A single Metal dispatch is int32-indexed, so volumes beyond 2^31-1 elements are dispatched as multiple slabs. `Ax` slab contributions sum; `Atb` slabs are independent output tiles. Transparent to the caller. |

## Known limitations

- **`Atb(backprojection_type="matched")` is an approximate adjoint.** See
  [validation.md](validation.md) for the measured discrepancy and what it means
  for iterative use.
- **MLX reductions flatten**, so `mx.sum`, `mx.linalg.norm` and similar reject
  arrays over 2^31-1 elements. Reduce very large volumes in chunks.
- **The hardware-texture backend is capped at 2048 elements per axis** by Metal.
  Larger volumes use the buffer path.
- **`mlx` is pinned** to the version in `pyproject.toml`, because the custom
  Metal kernels are validated against it.
