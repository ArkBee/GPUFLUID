"""[Layer S2.17 + F3.7] MPM contact/inflow helpers and the solver adapter.

Public API:
    MpmSolver — F3.7 shell-out adapter around third_party/warp-mpm
    MpmDivergenceError — raised on NaN-divergence mid-bake (round-20)
    apply_patches() — apply S2.17.PATCH.SLIP + S2.17.PATCH.EOS at import time

Sub-modules:
    colliders   — S2.17.1 (SDF box collider grid kernel)
    pushback    — S2.17.2/3 (cube + wall particle pushback)
    velcaps     — S2.17.4/5 (tap terminal velocity + above-cube anti-splash)
    _patches    — S2.17.PATCH.* (overlay fixes to warp-mpm upstream)
    solver      — F3.7 MpmSolver orchestrator

See DESIGN.md §5.3 (S2.17.*) and §6.7 (F3.7) for block specs.
"""
from __future__ import annotations

from .solver import MpmSolver, MpmDivergenceError
from ._patches import apply_patches

__all__ = ["MpmSolver", "MpmDivergenceError", "apply_patches"]
