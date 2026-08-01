# Contributing to mlx-tomo

Thanks for your interest. This is a small, focused library; contributions, bug
reports, and questions are all welcome.

## Scope

mlx-tomo provides the projection operators and the classical analytic
reconstructions built directly on them (FDK/FBP). Iterative and learned
reconstruction are currently outside this project's scope and can be developed
in downstream packages.

## Reporting a bug

Open a [GitHub issue](https://github.com/martinlachaine/mlx-tomo/issues) and
include:

- your hardware and OS (e.g. *Apple M2 Pro, macOS 15.4*),
- Python, `mlx-tomo`, and `mlx` versions (`pip show mlx-tomo mlx`),
- a minimal snippet that reproduces the problem,
- the geometry object involved (`print(geo)`), and
- what you expected versus what happened.

Accuracy reports are most useful as a relative-L2 error against a stated
reference — the float64 implementation in `mlx_tomo/ref.py`, an analytic
phantom, or TIGRE itself. See `harness/` for the patterns used in the suite.

## Development setup

Requires an Apple silicon Mac (Metal/MLX).

```bash
git clone https://github.com/martinlachaine/mlx-tomo && cd mlx-tomo
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -e ".[dev]"
cd texture_bridge && ./build.sh && cd ..   # optional texture backend
```

## Running the tests

```bash
.venv/bin/python harness/run_tests.py
```

This runs every `harness/test_*.py` and exits non-zero on any failure. Please
make sure it passes before opening a pull request, and add or extend a test
under `harness/` for any behavior change.

New kernel code is expected to come with: a float64 reference path in
`mlx_tomo/ref.py`, an analytic or adjoint oracle, and a GPU-vs-reference test
at the documented float32 tolerance.

The `mlx` dependency is pinned (`mlx==0.32.0`) because the custom Metal kernels
are validated against it. If you need to bump it, re-validate the full suite
first and say so in the PR.

## Pull requests

- Keep changes focused and match the surrounding code style.
- Note any change to the public API or to TIGRE parity in the PR description,
  and update `CHANGELOG.md` plus the relevant file under `docs/`.
- Kernel tuning is per-device; if you change launch parameters, state which
  machine you measured on and confirm at a representative view count (see
  [docs/performance.md](docs/performance.md)).
- By contributing, you agree your contributions are licensed under the
  project's [BSD-3-Clause license](LICENSE).
