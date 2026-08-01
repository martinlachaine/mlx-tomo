# Validation

## How correctness is established

Three complementary layers, all in `harness/`:

1. **Float64 reference implementation** — `mlx_tomo/ref.py` computes the same
   projections in double precision with NumPy. The GPU kernels are required to
   agree with it to approximately 1e-5 relative L2 in the tested
   configurations. This is an independent implementation, but it deliberately
   shares the same mathematical model, so it bounds arithmetic and kernel error
   rather than modeling error — which is what layers 2 and 3 address.
2. **Analytic oracle** — exact ellipsoid line integrals derived directly from
   the documented geometry convention. This shares no code with the reference
   implementation, so a shared misunderstanding of the geometry would not pass
   both.
3. **Impulse placement** — single-voxel volumes must project onto the
   analytically predicted detector pixel, across projection modes, angles, and
   detector and origin offsets.

Adjoint identities are checked between `Ax` and `Atb`, and reconstruction is
checked against the analytic oracle for FDK and FBP.

## Running the suite

From a development checkout:

```bash
python harness/run_tests.py
```

This runs every `harness/test_*.py` and exits non-zero on any failure. It is the
same command CI runs.

Individual files can be run directly, for example:

```bash
python harness/test_ax_gpu.py        # forward projection vs the float64 reference
python harness/test_atb_gpu.py       # backprojection, including adjoint checks
python harness/test_fdk.py           # FDK/FBP vs the analytic oracle
python harness/test_ref_geometry.py  # geometry transforms and impulse placement
python harness/test_slabs.py         # slab splitting at reduced thresholds
python harness/test_rays.py          # projection along explicit rays
python harness/test_texture.py       # hardware-texture backend, if built
```

## Large-array validation

Slab splitting is exercised two ways. `harness/test_slabs.py` runs in the
default suite with artificially lowered split thresholds, so the splitting logic
is covered on small problems. `harness/check_bigvol.py` validates it at real
sizes beyond 2^31-1 elements and is not part of the default suite because it
needs roughly 96 GB of free memory:

```bash
python harness/check_bigvol.py
```

At 1024³ it compares slab-split against unsplit output within the same float32 tolerance; at
2048³ it compares against the analytic oracle, checks that results are
independent of where the cuts fall, counts impulses placed on a cut plane, and
verifies the adjoint identity at full scale.

## Adjoint consistency

Ray-driven forward projection and voxel-driven backprojection are not exact
transposes of one another. `backprojection_type="matched"` uses a cone weight
chosen to be closer to the adjoint than the FDK weight, but it remains an
approximation.

Measured on a 128³ volume with a 666² detector over 180 views, cone geometry
with DSO 1000 mm and DSD 1500 mm, using `mlx==0.32.0` on an Apple M5 Max:

```
|<Ax x, y> - <x, Atb y>| / (||Ax x|| ||y||)  =  1.8e-4   (random x, y)
                                               2.1e-3   (smoothed random x, y)
```

In the geometries tested here this discrepancy did not materially affect FDK,
FBP or DRR results. Iterative methods can be more sensitive to adjoint
inconsistency, so if you build a Krylov solver or an unrolled network on the
normal equations, evaluate adjoint consistency and convergence for your own
geometry before relying on it. An exactly matched pair — a scatter-based
transpose of the trilinear weights — is not implemented.

### Measuring it yourself

Normalize by `||Ax x|| ||y||`, not by `<Ax x, y>`. With random test vectors both
inner products are near-zero sums of many canceling terms, so normalizing by
the inner product measures cancellation noise rather than adjoint error and can
report a very large relative discrepancy for an adjoint pair that is in fact
accurate. `harness/test_atb_gpu.py` shows the pattern used here.
