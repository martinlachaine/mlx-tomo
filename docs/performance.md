# Performance

## Methodology

Timings come from `harness/bench_ax.py` and `harness/bench_atb.py`, which measure
as follows:

- One untimed warm-up call, which compiles the Metal kernel and allocates memory
  pools.
- Then nine timed repetitions. Each calls `mx.synchronize()` before starting the
  clock and both `mx.eval()` and `mx.synchronize()` before stopping it, so the
  measurement covers full GPU execution rather than dispatch only.
- **The reported figure is the median of the nine repetitions**, with the
  observed minimum and maximum also printed so run-to-run spread is visible.
- Wall-clock time for the whole projection stack, divided by the number of views.
- Volumes are float32; the phantom is the ellipsoid phantom in `harness/gen.py`.
- `harness/tune.py` was run first, so kernel launch parameters are the local
  measurements rather than the shipped defaults.

Both scripts print the machine identifier and MLX version they ran under, and
write results to `results/`.

## Measured throughput

Milliseconds per view, median of nine repetitions.

**Apple M5 Max** — 40-core GPU, 128 GB unified memory, macOS 26.5.1 (build
25F80), Python 3.14.6, MLX 0.32.0, mains power, no other GPU load. Measured
sustained read+write bandwidth approximately 500 GB/s.

| problem | views | interpolated | Siddon |
|---|---|---|---|
| 256³ → 256² | 1 | 1.17 `[1.02–2.45]` | 0.59 `[0.51–1.05]` |
| 256³ → 256² | 100 | 0.28 `[0.27–0.29]` | 0.15 `[0.15–0.16]` |
| 512³ → 512² | 1 | 3.14 `[2.99–4.43]` | 1.78 `[1.53–4.10]` |
| 512³ → 512² | 100 | 2.08 `[2.06–2.10]` | 0.98 `[0.97–0.99]` |
| 512³ → 768² | 10 | 3.84 `[3.77–4.00]` | — |

| backprojection | views | FDK | matched |
|---|---|---|---|
| 256³ ← 256² | 100 | 0.23 `[0.23–0.25]` | 0.27 `[0.26–0.27]` |
| 512³ ← 512² | 100 | 1.67 `[1.67–1.76]` | — |
| 512³ ← 512² | 360 | 1.72 `[1.71–1.73]` | — |

Peak GPU memory across the whole forward sweep was 0.70 GiB, set by its largest
case (a 512³ volume with a 768² detector).

Single-view rows carry noticeably wider spread than 100-view rows, because
fixed per-call overhead is amortized over one view instead of a hundred.

### Earlier measurements on an Apple M1

These were taken with an earlier version of the benchmark that reported the
**fastest of three** repetitions rather than a median, so they are not directly
comparable with the table above and are somewhat optimistic. They are retained
because the M1-to-M5 comparison is informative.

Apple M1, 8-core GPU, 16 GB unified memory; measured sustained read+write
bandwidth approximately 50 GB/s.

| problem | views | interpolated | Siddon | Atb (FDK) |
|---|---|---|---|---|
| 256³ ↔ 256² | 100 | 6.1 | 2.6 | 2.3 |
| 512³ ↔ 512² | 100 | 48.2 | 16.6 | 17.6 |

Throughput across the two machines tracks measured memory bandwidth more closely
than GPU core count. Observed scaling is consistent with bandwidth-bound
execution, though no roofline analysis has been performed.

### Large-volume measurement

Separate from the standard sweep above: a 2048³ volume with a 2048² detector runs
as five z-slab dispatches on the M5 Max, at roughly 800 ms/view interpolated,
310 ms/view Siddon, and 800 ms/view for matched backprojection. This was a
single measurement taken to confirm the slab path performs sensibly at scale, not
a median over repetitions, and it should be read as an order of magnitude rather
than a benchmark. Correctness at this size is covered in
[validation.md](validation.md).

## Tuning for your machine

Kernel threadgroup shapes are measured rather than derived, and the shipped
defaults were selected on an Apple M1. On other hardware run:

```bash
python harness/tune.py
```

which measures device bandwidth, sweeps each kernel's launch parameters, and
writes `mlx_tomo/tuning_local.json`. Delete that file to fall back to the shipped
defaults.

Confirm the sweep result at your real view count before accepting it. On both
machines tested here a short 10-view sweep selected an `interpolated` threadgroup
shape that was slower at 100 views, so `harness/bench_ax.py` should be run at a
representative view count as a check.

The backprojection kernel was insensitive to threadgroup shape on the M5 Max,
within about 1% across every shape swept.

## Reproducing

```bash
python harness/bench_ax.py         # forward projection; --quick for a subset
python harness/bench_atb.py        # backprojection
python harness/bench_texture.py    # hardware-texture backend
```
