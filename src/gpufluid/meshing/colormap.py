"""[B11.3] Temperature → RGB colormaps for vertex colouring.

``get_colormap(name)`` returns a callable that maps a ``(N,)`` value array plus
a ``(t_min, t_max)`` range to an ``(N, 3)`` float32 RGB array in ``[0, 1]``,
backed by matplotlib's named colormaps (``inferno``/``viridis``/``magma``/
``plasma``/``coolwarm``/``turbo``/``jet``/…).

audit-2026-06-14 round-2 #3: this module was imported by the CLI mesh pass
(``cli/commands.py``: ``from ..meshing.colormap import get_colormap``) but never
existed — any scene setting ``[output] temperature_colormap`` raised
``ModuleNotFoundError`` and took the whole bake down. matplotlib is an optional
``viz`` extra, so a missing install degrades to a clear BlockError, not a crash.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..blocks import BlockError


def get_colormap(name: str) -> Callable[[np.ndarray, float, float], np.ndarray]:
    """Resolve a matplotlib colormap by ``name`` into a value→RGB mapper.

    The returned callable takes ``(values, t_min, t_max)`` and returns
    ``(N, 3)`` float32 RGB in ``[0, 1]`` — values are normalised to ``[0, 1]``
    over ``[t_min, t_max]`` (clamped) before the lookup.
    """
    try:
        import matplotlib
    except Exception as exc:  # matplotlib is an optional `viz` extra
        raise BlockError(
            "B11.3", f"temperature_colormap='{name}' needs matplotlib "
            f"(install the 'viz' extra): {exc}")
    try:
        registry = matplotlib.colormaps        # mpl ≥ 3.5 dict-like registry
    except AttributeError:                      # very old mpl: deprecated API
        from matplotlib import cm as _cm
        try:
            cmap = _cm.get_cmap(name)
        except Exception:
            raise BlockError("B11.3", f"unknown temperature colormap '{name}'")
    else:
        try:
            cmap = registry[name]
        except KeyError:                        # genuinely unknown name
            raise BlockError("B11.3", f"unknown temperature colormap '{name}'")

    def _apply(values: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
        v = np.asarray(values, dtype=np.float32)
        # dodge: a flat field (t_max == t_min) would div-by-zero — map it all
        # to the colormap's low end rather than NaN.
        denom = float(t_max) - float(t_min)
        if denom == 0.0:
            denom = 1.0
        t = np.clip((v - float(t_min)) / denom, 0.0, 1.0)
        rgba = cmap(t)                              # (N, 4) in [0, 1]
        return np.ascontiguousarray(rgba[:, :3], dtype=np.float32)

    return _apply
