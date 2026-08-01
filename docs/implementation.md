# Implementation notes

Background on how the package is put together, for readers extending it or
debugging against it. None of this is needed to use the library.

## Precision model

Geometry is computed on the host in float64: source and detector positions,
rotation matrices, per-angle offsets, and the ray parameterization derived from
them. Those results are passed to the GPU as float32, and the projection and
backprojection kernels run entirely in float32.

Apple GPUs have no native double precision, so this split keeps the
precision-sensitive part — coordinate setup, where a small angular error
displaces a ray across the whole volume — in double precision, while the
accumulation runs at the width the hardware supports. Accuracy is bounded by the
float32 accumulation, which is what the reference comparison in
[validation.md](validation.md) measures.

## Array layouts

- Volumes are `img[z, y, x]`, float32.
- Projection stacks are `proj[view, v, u]`, float32.
- World coordinates are (x, y, z) in millimeters, with the volume centered on the
  origin and voxel centers at `(i + 0.5) * dVoxel - sVoxel / 2`.

These follow TIGRE, so geometry objects and arrays transfer between the two
without reordering.

## Forward projectors

Two kernels, selected by `projection_type`:

- **`interpolated`** samples the volume with trilinear interpolation at fixed
  steps along each ray. Step length is set by `geo.accuracy` in voxel units.
  Cost is proportional to ray length over step size, independent of volume
  content.
- **`Siddon`** walks the exact ray-voxel intersections, accumulating each cell's
  path length. It is implemented as an incremental digital differential
  analyzer: the next cell boundary in each axis is tracked and the smallest is
  advanced, which avoids recomputing intersections and keeps the inner loop free
  of most branches.

`interpolated` is the usual choice; `Siddon` is exact for piecewise-constant
volumes and is used in the analytic comparisons.

## Backprojectors

Both are voxel-driven: each output voxel gathers from the projections. Two cone
weights are available:

- **`FDK`** applies the distance weighting that Feldkamp-Davis-Kress
  reconstruction requires.
- **`matched`** applies a weight chosen to approximate the adjoint of the
  interpolated forward projector. See [validation.md](validation.md) for how
  close the approximation is.

In the cone-beam cosine weight, detector coordinates are measured from the
piercing point — the intersection of the central ray with the detector plane —
rather than from the detector center. The two coincide only when the detector is
not offset.

## Large arrays

A single Metal dispatch indexes with int32, which caps one dispatch at 2^31-1
elements. Two splits handle anything larger, both transparent to the caller:

- **Projection stacks** split along the view axis. Views are independent, so the
  results concatenate.
- **Volumes** split into contiguous z-slabs. For `Ax`, each slab produces a
  partial projection over the rays that cross it, and the partials sum. For
  `Atb`, each slab is an independent output tile, so no combination is needed.

Split thresholds live in `mlx_tomo/projector.py` and are lowered artificially by
`harness/test_slabs.py` so the logic is covered on small problems.

Note that MLX reductions flatten their input and therefore reject arrays over
2^31-1 elements. Reducing a very large volume requires chunking, which the
library does not do on the caller's behalf.

## Optional hardware-texture backend

The interpolated projector can sample through Metal's 3D texture units instead
of buffer loads. Because `mx.fast.metal_kernel` cannot bind textures, this
requires a small C-ABI dylib built from `texture_bridge/`, loaded at runtime by
`mlx_tomo/texture.py`. `texture_available()` reports whether it loaded, and the
buffer path is used if it did not.

Two constraints: Metal caps textures at 2048 elements per axis, and
`Ax(..., backend="texture")` builds its plan on each call, so the volume upload
cost is included in every call. For repeated projection of the same volume,
construct a `TextureProjector` once and reuse it.

## Per-device tuning

Kernel threadgroup shapes are measured rather than derived. `mlx_tomo/tuning.py`
holds the shipped defaults, selected on an Apple M1, and reads
`mlx_tomo/tuning_local.json` if present — written by `harness/tune.py` — or a
path given in `MLX_TOMO_TUNING`. See [performance.md](performance.md).

## Reference implementation

`mlx_tomo/ref.py` is a NumPy float64 implementation of the same projections. It
prioritizes clarity and numerical traceability over speed and is intended for
small validation problems, not production use. It is the primary accuracy oracle
for the GPU kernels.

## Analytic reconstruction: weights, ramp and scaling

### Weight split

FDK's total weighting is applied in two places:

- the detector cosine weight `DSD / sqrt(DSD² + u² + v²)`, applied before
  filtering in `algorithms.py`;
- the per-voxel distance weight `(DSO/U)²`, applied inside `Atb`'s `FDK` mode.

`u` and `v` in the cosine weight are measured from the piercing point, so
per-angle `offDetector` is honored directly. FBP is the parallel counterpart:
ramp filtering, backprojection, then a `DSO/DSD` scale.

For an ideal circular trajectory under the standard FDK assumptions,
reconstruction is exact in the central plane and approximate away from it;
recovered attenuation degrades smoothly with cone angle.

### Ramp filter construction

The band-limited ramp is built in the **spatial** domain following Kak and
Slaney — `h[0] = 1/4` and `h[n] = -1/(pi n)²` for odd `n` at unit spacing, with
detector spacing entering through a `1/dU` factor in the scale — and taken to
the frequency domain as `|FFT(h)| * 2`. The magnitude removes the linear phase
of the uncentered kernel, and the factor of two composes with the `1/4` in the
scale to give the Feldkamp `1/2`. Constructing the ramp this way gives the
correct DC and Nyquist response, which a directly sampled `|omega|` ramp does
not.

Windows multiply the half-spectrum on `w = 2*pi*k/n` over `[0, pi]`:

| window | form |
|---|---|
| `shepp_logan` | `sinc(w / 2d)` |
| `cosine` | `cos(w / 2d)` |
| `hamming` | `0.54 + 0.46 cos(w/d)` |
| `hann` | `(1 + cos(w/d)) / 2` |

The cutoff `d` in `(0, 1]` zeroes frequencies above `pi * d`.

### Padding

Projections are zero-padded to the smallest power of two at least `2 * nU`, and
at least 64. Power-of-two lengths keep MLX's FFT on its native fast path. Data
sits at the head of the padded window, which is circularly equivalent to
centered placement for a zero-phase filter.

### Per-view scale and angular sampling

The per-view scale is `(DSD/DSO) * (2 * dbeta) / (4 * dU)`, where `dbeta` is the
**mean** step of the unwrapped rotation component. For a full scan sampled
uniformly this reduces to the familiar `(2*pi/n) / (4*dU)`. For a short scan it
scales by the actual coverage rather than assuming a full rotation.

Because `dbeta` is a mean, **genuinely nonuniform angular sampling is averaged
rather than weighted per view**. Views clustered in one part of the arc are
under-weighted relative to sparse regions. To correct for it, scale each view by
its own share of the arc before filtering:

```python
d = np.diff(np.unwrap(angles))                       # per-view steps
d = np.append(d, d[-1])                              # last view keeps its step
proj = proj * (d / d.mean())[:, None, None]          # then filter and backproject
```

With Parker weights active the scale doubles, because Parker weighting
normalizes conjugate-ray pairs to sum to one and therefore replaces the
Feldkamp `1/2`.

### Parker short-scan weighting

Parker weights use Wesarg's smooth formulation (Med. Phys. 29(3), 2002), with
the fan angle taken along the detector U axis.

Automatic application (`parker=None`) is deliberately conservative. It requires
a strictly monotonic arc in the first Euler angle with the other two constant,
because Parker weighting is undefined for other trajectories and misreading one
could silently zero every weight. Trajectories that fail that test receive no
weighting unless `parker` is passed explicitly.

When coverage is below `2*pi` a warning is emitted and weights are applied.

A second guard covers inadequate coverage. Parker weighting requires at least
`pi` plus the **full** detector fan angle — in the code, `pi + 2*delta`, where
`delta` is half the detector's angular width. Below that a further warning is
emitted, but **the weights are still computed and applied**: the internal
`epsilon` term clamps to zero and the weighting proceeds. The warning is
therefore not a refusal. Such data are genuinely limited-angle, the redundancy
Parker weighting assumes does not exist, and the reconstruction remains
incomplete no matter how it is weighted. Treat that warning as a statement that
the acquisition is insufficient, not as a recoverable condition.

## References

- A. C. Kak and M. Slaney, *Principles of Computerized Tomographic Imaging*.
  IEEE Press, 1988 (reprinted SIAM, 2001). Ramp filter construction and the
  spatial-domain kernel used here.
- L. A. Feldkamp, L. C. Davis, and J. W. Kress, "Practical cone-beam algorithm,"
  *Journal of the Optical Society of America A* 1(6), 612–619 (1984).
  doi:10.1364/JOSAA.1.000612
- S. Wesarg, M. Ebert, and T. Bortfeld, "Parker weights revisited,"
  *Medical Physics* 29(3), 372–378 (2002). doi:10.1118/1.1450132
  The smooth short-scan redundancy weighting implemented here.
- D. L. Parker, "Optimal short scan convolution reconstruction for fan beam CT,"
  *Medical Physics* 9(2), 254–257 (1982). doi:10.1118/1.595078
- R. L. Siddon, "Fast calculation of the exact radiological path for a
  three-dimensional CT array," *Medical Physics* 12(2), 252–255 (1985).
  doi:10.1118/1.595715
- A. Biguri, M. Dosanjh, S. Hancock, and M. Soleimani, "TIGRE: a MATLAB-GPU
  toolbox for CBCT image reconstruction," *Biomedical Physics & Engineering
  Express* 2(5), 055010 (2016). doi:10.1088/2057-1976/2/5/055010
