"""[Layer G1] Warp runtime: init, device selection, allocation helpers."""
from __future__ import annotations
import warp as wp
from typing import Tuple, Union
from ..blocks import block

_INITIALIZED = False
_DEVICE: str = "cuda:0"


# [BLK G1.1]
@block("G1.1", "Warp init & default-device selection")
def init(device: str = "cuda:0") -> str:
    """Initialise Warp and pin a default device. Idempotent."""
    global _INITIALIZED, _DEVICE
    if not _INITIALIZED:
        wp.init()
        _INITIALIZED = True
    devices = [str(d) for d in wp.get_devices()]
    if device not in devices:
        # fall back to cpu if requested CUDA missing
        device = "cpu" if "cpu" in devices else devices[0]
    _DEVICE = device
    return _DEVICE


def device() -> str:
    if not _INITIALIZED:
        init()
    return _DEVICE


Shape = Union[int, Tuple[int, ...]]


# [BLK G1.2]
@block("G1.2", "zeros allocation helper on default device")
def zeros(shape: Shape, dtype=float, *, dev: str | None = None):
    """Allocate a Warp array of zeros on the chosen (or default) device."""
    return wp.zeros(shape, dtype=dtype, device=dev or device())


# [BLK G1.2]
@block("G1.2", "zeros allocation helper for int arrays")
def zeros_int(shape: Shape, *, dev: str | None = None):
    return wp.zeros(shape, dtype=int, device=dev or device())
