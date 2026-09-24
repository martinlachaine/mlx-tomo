# Performance

## Methodology

Timings come from `harness/bench_ax.py`, `harness/bench_atb.py`, and
`harness/bench_texture.py`, which measure as follows:

- One untimed warm-up call, which compiles the Metal kernel and allocates memory
  pools.
- Then nine timed repetitions. Each calls `mx.synchronize()` before starting the
  clock and both `mx.eval()` and `mx.synchronize()` before stopping it, so the
  measurement covers full GPU execution rather than dispatch only.
- **The reported figure is the median of the nine repetitions**, with the
  observed minimum and maximum also printed so run-to-run spread is visible.
- Wall-clock time for the whole projection stack, divided by the number of views.
- Volumes are float32 unless the table says otherwise; the phantom is the
  ellipsoid phantom in `harness/gen.py`.
- `harness/tune.py` was run first, so kernel launch parameters are the local
  measurements rather than the shipped defaults.

All three scripts print the machine identifier and MLX version they ran under,
and write structured JSON results to `results/`. The texture benchmark reports
the buffer backend both with its output left GPU-resident and with a host numpy
result; comparisons with the texture backend use the latter because texture
results currently land in host memory.

## Measured throughput

Milliseconds per view, median of nine repetitions.

**Apple M5 Max** — 40-core GPU, 128 GB unified memory, macOS 27.0 (build
26A428), Python 3.14.7, MLX 0.32.0, mains power, no thermal or performance
warnings. Measured sustained read+write bandwidth 517 GB/s. The benchmark used
commit `e7416da` plus the unreleased private-texture change described below.

| problem | views | interpolated | Siddon |
|---|---|---|---|
| 256³ → 256² | 1 | 2.20 `[2.09–2.56]` | 0.92 `[0.87–1.34]` |
| 256³ → 256² | 100 | 0.26 `[0.26–0.26]` | 0.14 `[0.14–0.15]` |
| 512³ → 512² | 1 | 3.03 `[2.93–6.34]` | 2.19 `[1.56–6.08]` |
| 512³ → 512² | 100 | 1.90 `[1.88–1.94]` | 0.90 `[0.90–0.91]` |
| 512³ → 768² | 10 | 3.55 `[3.54–3.59]` | — |

| backprojection | views | FDK | matched |
|---|---|---|---|
| 256³ ← 256² | 100 | 0.23 `[0.22–0.24]` | 0.26 `[0.25–0.26]` |
| 512³ ← 512² | 100 | 1.61 `[1.61–1.71]` | — |
| 512³ ← 512² | 360 | 1.62 `[1.62–1.64]` | — |

Peak GPU memory across the whole forward sweep was 0.70 GiB, set by its largest
case (a 512³ volume with a 768² detector).

Single-view rows carry noticeably wider spread than 100-view rows, because
fixed per-call overhead is amortized over one view instead of a hundred.

### Hardware-texture backend

The optional texture backend stores the volume in a GPU-private 3D texture and
returns projections in host numpy memory. The directly comparable buffer column
therefore also includes conversion to host numpy. Upload time is measured after
the Metal pipeline has been compiled and is paid once per `TextureProjector`.

| problem | views | dtype | buffer GPU | buffer host | texture host | upload |
|---|---:|---|---:|---:|---:|---:|
| 256³ → 256² | 100 | float32 | 0.26 | 0.27 | 0.42 | 2.8 ms |
| 512³ → 512² | 1 | float32 | 2.99 | 2.97 | 3.54 | 18.6 ms |
| 512³ → 512² | 100 | float32 | 1.92 | 1.94 | 3.92 | 24.3 ms |
| 512³ → 512² | 100 | float16 | 2.01 | 2.02 | 2.58 | 12.4 ms |

Projection columns are milliseconds per view. On this M5 Max the private
texture is not faster than the buffer kernel, but private tiled storage is a
substantial improvement over the former shared texture: the 512³/100-view time
fell from 7.49 to 3.92 ms/view for float32 and from 4.58 to 2.58 ms/view for
float16. Sampler throughput differs across GPU generations, so applications
should benchmark both backends on their target machine.

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
