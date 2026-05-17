"""Back-compat shim — Motion + evaluate_center moved to
``gpufluid.primitives.animation`` on 2026-05-17 (F3.6.B; see
``docs/DESIGN.md §3.2.4.2``).

Mesh-based animated obstacles are still supported via per-frame
``mesh_to_sdf`` (D4.3) calls; their integration with this module is
via ``evaluate_center`` which is re-exported below.

New code should import directly from ``gpufluid.primitives.animation``.
This shim exists so existing scripts continue working during the
migration window.
"""
from ..primitives.animation import (  # noqa: F401
    LinearMotion,
    KeyframeMotion,
    Motion,
    evaluate_center,
)
