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
- Both runs below used launch parameters measured with `harness/tune.py`; on
  the M1 these are the shipped defaults.

All three scripts print the machine identifier and MLX version they ran under,
and write structured JSON results to `results/`. The texture benchmark reports
the buffer backend both with its output left GPU-resident and with a host numpy
result; comparisons with the texture backend use the latter because texture
results currently land in host memory.

## Measured throughput

Milliseconds per view, median of nine repetitions.

**Apple M5 Max**: 40-core GPU, 128 GB unified memory, macOS 27.0.1 (build
26A434), Python 3.14.7, MLX 0.32.4, mains power, no thermal or performance
warnings, display asleep for the whole run. Measured sustained read+write
bandwidth 520 GB/s. Release 0.2.0 with
the launch shapes written by `harness/tune.py` on this machine. The unrounded
results are in [forward projection](benchmark-data/m5-2026-10-17/bench_ax.json),
[backprojection](benchmark-data/m5-2026-10-17/bench_atb.json) and
[texture comparison](benchmark-data/m5-2026-10-17/bench_texture.json).

| problem | views | interpolated | Siddon |
|---|---|---|---|
| 256³ → 256² | 1 | 1.80 `[1.76–2.00]` | 1.29 `[0.93–1.46]` |
| 256³ → 256² | 100 | 0.20 `[0.20–0.21]` | 0.14 `[0.14–0.14]` |
| 512³ → 512² | 1 | 1.96 `[1.89–3.83]` | 1.52 `[1.49–6.34]` |
| 512³ → 512² | 100 | 1.49 `[1.48–1.51]` | 0.90 `[0.90–0.90]` |
| 512³ → 768² | 10 | 3.10 `[3.09–3.14]` |  |

| backprojection | views | FDK | matched |
|---|---|---|---|
| 256³ ← 256² | 100 | 0.14 `[0.13–0.14]` | 0.18 `[0.18–0.18]` |
| 512³ ← 512² | 100 | 0.92 `[0.91–0.99]` |  |
| 512³ ← 512² | 360 | 0.93 `[0.93–0.94]` |  |

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
| 256³ → 256² | 100 | float32 | 0.20 | 0.21 | 0.40 | 2.1 ms |
| 512³ → 512² | 1 | float32 | 2.74 | 1.98 | 3.40 | 15.3 ms |
| 512³ → 512² | 100 | float32 | 1.49 | 1.51 | 3.77 | 15.4 ms |
| 512³ → 512² | 100 | float16 | 1.63 | 1.65 | 2.38 | 7.5 ms |

Projection columns are milliseconds per view. On this M5 Max the private
texture is not faster than the buffer kernel; private tiled storage is still a
substantial improvement over the former shared texture (the 512³/100-view
float32 time was 7.49 ms/view with it, and 4.58 for float16). Sampler
throughput differs across GPU generations, so applications should benchmark
both backends on their target machine.

### Apple M1 (2026-10-17, release 0.2.0)

Mac mini (`Macmini9,1`), Apple M1 with an 8-core GPU and 16 GB unified memory;
macOS 27.0 (build 26A428), Python 3.13.9, MLX 0.32.4, mains power, no thermal
or performance warnings. Measured sustained read+write bandwidth 61 GB/s.
Release 0.2.0 with the launch shapes written by `harness/tune.py` on this
machine, which are the shipped defaults. Values are milliseconds per view,
nine-run median `[minimum–maximum]`. The unrounded results are in
[forward projection](benchmark-data/m1-2026-10-17/bench_ax.json),
[backprojection](benchmark-data/m1-2026-10-17/bench_atb.json) and
[texture comparison](benchmark-data/m1-2026-10-17/bench_texture.json).

| problem | views | interpolated | Siddon |
|---|---:|---:|---:|
| 256³ → 256² | 1 | 4.75 `[4.55–7.54]` | 3.67 `[3.50–4.47]` |
| 256³ → 256² | 100 | 3.65 `[3.62–3.68]` | 1.65 `[1.60–1.67]` |
| 512³ → 512² | 1 | 36.98 `[35.91–38.00]` | 32.93 `[32.55–34.50]` |
| 512³ → 512² | 100 | 30.21 `[30.13–30.33]` | 15.03 `[14.90–15.22]` |
| 512³ → 768² | 10 | 41.35 `[41.25–41.64]` | |

| backprojection | views | FDK | matched |
|---|---:|---:|---:|
| 256³ ← 256² | 100 | 1.11 `[1.07–1.15]` | 1.68 `[1.65–1.69]` |
| 512³ ← 512² | 100 | 8.33 `[8.28–8.71]` | |
| 512³ ← 512² | 360 | 8.50 `[8.32–8.57]` | |

Peak GPU memory across the forward sweep was 0.70 GiB. The 0.1.0 gap between
the 100- and 360-view FDK rows (19.27 against 23.71 ms/view) is gone.

Against the 0.1.0 run on the same machine
([data](benchmark-data/m1-2026-09-24/)), the interpolated projector is 1.5 to
1.7× faster and FDK 2.3 to 2.8×. Part of that is the session rather than the
kernels: the Siddon kernel and the texture path, whose code did not change
between the two releases, measured 1.1 to 1.2× and 1.6× faster respectively,
with MLX at 0.32.4 instead of 0.32.0.

The texture table uses the same output-residency definitions as the M5 table:
the buffer GPU output remains on the GPU, while buffer host and texture host
return NumPy arrays. Projection columns are milliseconds per view. Upload is
the one-time `TextureProjector` construction after shader compilation, measured
once per case in milliseconds; it is **not** a nine-run median.

| problem | views | dtype | buffer GPU | buffer host | texture host | upload |
|---|---:|---|---:|---:|---:|---:|
| 256³ → 256² | 100 | float32 | 3.66 `[3.63–3.67]` | 3.67 `[3.65–3.69]` | 2.61 `[2.58–2.63]` | 6.0 ms |
| 512³ → 512² | 1 | float32 | 35.75 `[34.67–37.20]` | 36.06 `[35.79–36.75]` | 20.90 `[20.76–21.36]` | 55.3 ms |
| 512³ → 512² | 100 | float32 | 29.83 `[29.81–29.91]` | 29.90 `[29.77–29.93]` | 21.08 `[21.02–21.25]` | 46.1 ms |
| 512³ → 512² | 100 | float16 | 26.89 `[26.85–26.95]` | 26.98 `[26.93–27.66]` | 17.01 `[16.94–17.56]` | 28.3 ms |

On the M1 the texture path is 1.4× as fast as buffer host on the 512³/100-view
float32 case (1.90× in 0.1.0; the buffer kernel gained more than the texture
path) and 1.6× for float16. For a one-off single-view projection the upload
cost makes the texture path slower; the advantage appears once a volume is
reused or projected over multiple views. The full correctness suite passed
8/8 test files, including odd-sized texture volumes.

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

Confirm the sweep result at your real view count before accepting it: the
sweep uses 10 views, and on the M1 the `interpolated` winner was 7% faster
than the runner-up at 100 views but 4% slower on single-view cases, so run
`harness/bench_ax.py` at a representative view count as a check.

The backprojection kernel is insensitive to threadgroup shape, within about 1%
across every shape swept on the M5 Max and 3% on the M1.

## Reproducing

```bash
python harness/bench_ax.py         # forward projection; --quick for a subset
python harness/bench_atb.py        # backprojection
python harness/bench_texture.py    # hardware-texture backend
```
