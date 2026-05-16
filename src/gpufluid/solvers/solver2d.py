"""[Layer F3] 2D FLIP/PIC solver (legacy module re-export).

The 2D solver pre-dates the layered refactor. It still works and is kept
for visualization and tests; it will be rewritten on the G1 primitives
in v0.2. For now we re-export so callers can import from the
``gpufluid.solvers`` namespace.
"""
# [BLK F3.1]
from ..solver2d_legacy import FlipSolver2D  # noqa: F401
