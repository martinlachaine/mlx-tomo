"""mlx-tomo: parallel-, fan- and cone-beam projectors on Apple GPUs (Metal/MLX).

TIGRE-compatible Python interface for supported operations: ``import mlx_tomo as
tigre`` then use ``tigre.Ax(img, geo, angles, projection_type)`` with a
``tigre.geometry()`` object. See https://github.com/martinlachaine/mlx-tomo/blob/main/docs/compatibility.md for what is supported and
where behavior differs.
"""

from .geometry import Geometry, ParallelGeo, geometry, geometry_default

__version__ = "0.1.0"

__all__ = [
    "Geometry", "ParallelGeo", "geometry", "geometry_default",
    "Ax", "Atb", "Ax_rays", "rays_from_geometry", "affine_rays",
    "FDK", "FBP", "fdk", "fbp", "filtering",
    "TextureProjector", "texture_available",
    "shepp_logan_mm", "rasterize", "__version__",
]


def __getattr__(name):  # PEP 562: lazy re-exports (real class, not a wrapper)
    if name == "TextureProjector":
        from .texture import TextureProjector
        return TextureProjector
    if name == "texture_available":
        from .texture import available
        return available
    if name in ("Ax_rays", "rays_from_geometry", "affine_rays"):
        from . import rays
        return getattr(rays, name)
    if name in ("shepp_logan_mm", "rasterize", "region_masks",
                "ground_truth_values"):
        from . import phantom3d
        return getattr(phantom3d, name)
    if name in ("FDK", "FBP", "fdk", "fbp"):
        from . import algorithms
        return getattr(algorithms, name)
    raise AttributeError(f"module 'mlx_tomo' has no attribute {name!r}")


# Bound eagerly: the import system would otherwise set the package
# attribute to the SUBMODULE mlx_tomo.filtering the first time anything
# imports it, shadowing the function the public API promises.
from .filtering import filtering as filtering  # noqa: E402


def Ax(img, geo, angles, projection_type="Siddon", **kwargs):
    from .api import Ax as _Ax
    return _Ax(img, geo, angles, projection_type=projection_type, **kwargs)


def Atb(proj, geo, angles, backprojection_type="FDK", **kwargs):
    from .api import Atb as _Atb
    return _Atb(proj, geo, angles, backprojection_type=backprojection_type,
                **kwargs)
