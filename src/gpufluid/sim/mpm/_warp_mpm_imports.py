"""Lazy import of warp-mpm structs.

Importing this module:
    1. Adds `third_party/warp-mpm/` to `sys.path`.
    2. Applies S2.17.PATCH.* (the upstream overlay patches) BEFORE the
       structs are imported, so Warp picks up the patched kernel source.
    3. Imports the three structs we need.

If warp-mpm is not available the import raises a clear error pointing
to the third_party directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ._patches import apply_patches

# Locate the vendored warp-mpm and apply patches
_patch_result = apply_patches()  # idempotent, no-op if already patched
_WARP_MPM = _patch_result["root"]
if str(_WARP_MPM) not in sys.path:
    sys.path.insert(0, str(_WARP_MPM))

try:
    from warp_utils import (  # type: ignore[import-not-found]  # noqa: F401
        Dirichlet_collider,
        MPMStateStruct,
        MPMModelStruct,
    )
    from mpm_solver_warp import MPM_Simulator_WARP  # type: ignore[import-not-found]  # noqa: F401
except ImportError as exc:
    raise ImportError(
        f"warp-mpm could not be imported from {_WARP_MPM}. "
        "Ensure the submodule is checked out (git submodule update --init) "
        "and that h5py is installed in the active env."
    ) from exc
