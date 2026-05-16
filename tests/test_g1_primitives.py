"""Layer G1 tests — primitives, runtime, gridmath."""
import numpy as np
import pytest
import warp as wp

from gpufluid.primitives.runtime import init, device, zeros, zeros_int
from gpufluid.blocks import get_registry, by_layer, BlockError


def test_g1_1_init_idempotent():
    d1 = init()
    d2 = init()
    assert d1 == d2
    assert "cuda" in d1 or "cpu" in d1


def test_g1_1_device_known():
    dev = device()
    assert dev == init()


@pytest.mark.gpu
def test_g1_2_zeros_alloc_and_shape():
    a = zeros((4, 5, 6))
    assert a.shape == (4, 5, 6)
    assert (a.numpy() == 0).all()


@pytest.mark.gpu
def test_g1_2_zeros_int_alloc():
    a = zeros_int((3, 3, 3))
    assert a.shape == (3, 3, 3)
    assert (a.numpy() == 0).all()


def test_block_registry_populated():
    reg = get_registry()
    # All six v0.1 layers must have at least one registered block.
    for layer in ["G1", "S2", "F3", "D4", "M5", "I6"]:
        assert len(by_layer(layer)) > 0, f"layer {layer} has no registered blocks"


def test_block_error_format():
    e = BlockError("S2.6.1", "test")
    assert "S2.6.1" in str(e)
    assert e.block_id == "S2.6.1"
