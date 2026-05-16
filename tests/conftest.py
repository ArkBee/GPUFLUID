"""pytest fixtures and CUDA-availability gate."""
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
