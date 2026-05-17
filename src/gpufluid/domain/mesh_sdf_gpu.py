"""Back-compat shim — the real implementation moved to
``gpufluid.schemes.mesh_marker`` on 2026-05-17 (F3.6.A2; see
``docs/DESIGN.md §3.2.4.2``).

New code should import from ``gpufluid.schemes.mesh_marker`` directly.
This shim exists so existing scripts continue working during the
migration window; it may be removed once the F3.6 macro closes.
"""
from ..schemes.mesh_marker import (  # noqa: F401
    k3_mesh_to_solid,
    k3_mesh_to_solid_bvh,
    DEFAULT_BVH_THRESHOLD,
    mesh_indicator_sdf_gpu,
    mark_solid_from_mesh_gpu,
)
