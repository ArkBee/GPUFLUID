"""pytest fixtures and CUDA-availability gate."""
import sys
from pathlib import Path

# Make `addon/` importable for tests covering the Blender bridge helpers
# (gpufluid_blender.render_bridge). The addon is intentionally OUTSIDE the
# `src/` tree so it can be zipped standalone — tests need explicit pathing.
_ADDON_DIR = Path(__file__).resolve().parents[1] / "addon"
if str(_ADDON_DIR) not in sys.path:
    sys.path.insert(0, str(_ADDON_DIR))

import pytest
import warp as wp


def _has_cuda() -> bool:
    try:
        wp.init()
    except Exception:
        return False
    return any("cuda" in str(d).lower() for d in wp.get_devices())


HAS_CUDA = _has_cuda()


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.gpu tests when CUDA absent."""
    if HAS_CUDA:
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
