"""gpufluid — GPU FLIP/PIC fluid simulator on NVIDIA Warp.

Layered architecture: G1 primitives → S2 schemes → F3 solver →
D4 domain → M5 meshing → I6 io → C7 cli → A8 addon.
See ``docs/DESIGN.md`` for the full block hierarchy.
"""
__version__ = "0.0.2"

# Public API — convenience re-exports. Prefer explicit submodule imports
# in non-trivial code so the layer is obvious.
from .solvers.solver3d import FlipSolver3D
from .solvers.solver2d import FlipSolver2D
from .meshing.surface import MeshExtractor, particles_to_mesh
from .meshing.smoothing import smooth_taubin, smooth_laplacian  # registers M5.5
from .meshing.decimate import decimate_mesh                       # registers M5.6
from .sim.reseed import reseed_particles, ReseedConfig            # registers S2.11
from .sim.whitewater import WhitewaterSystem, WhitewaterConfig    # registers W7.x
from .primitives.sdf import (
    cell_centers,
    sdf_sphere, sdf_box, sdf_cylinder_y, sdf_plane, sdf_union,
)
from .domain.sdf import mark_solid_from_sdf
from .io.ply import write_ply, read_ply, write_particles_npy
from .blocks import BlockError, block, get_registry, by_layer

__all__ = [
    "__version__",
    "FlipSolver3D", "FlipSolver2D",
    "MeshExtractor", "particles_to_mesh",
    "cell_centers", "sdf_sphere", "sdf_box", "sdf_cylinder_y", "sdf_plane",
    "sdf_union", "mark_solid_from_sdf",
    "write_ply", "read_ply", "write_particles_npy",
    "BlockError", "block", "get_registry", "by_layer",
]
