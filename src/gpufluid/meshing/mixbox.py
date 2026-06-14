"""[BLK M5.11.5 / B18.5] Mixbox pigment-space colour mixing.

Wrapper around `pymixbox <https://github.com/scrtwpns/mixbox>`_ (MIT
licence, Šochorová & Jamriška 2021 — "Practical Pigment Mixing for
Digital Painting"). Implements a per-mesh-vertex KNN colour blend in
**latent pigment space** instead of linear RGB.

The visible payoff vs the linear path (B18.4): blue + yellow blend to
green (R<127, G>140, B<127) instead of muddy grey (~127,127,127). All
other mixes also look more like physical pigment behaviour — desaturated
mid-mixes, hue rotation toward the perceptual blend axis.

When ``pymixbox`` is not importable, this module falls back to the
linear-RGB inverse-distance² blend (same result as the GPU pass in
:mod:`gpufluid.meshing.surface`) and logs a single warning. Users who
want the green-not-grey behaviour install the optional dep::

    pip install pymixbox

The fallback path is deterministic and never crashes, so the addon /
CLI can default ``mix_mode = "mixbox"`` without runtime guards — if
the LUT is absent, the user simply gets the linear result.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..blocks import block

try:  # pragma: no cover - import side effect
    import mixbox  # pymixbox (MIT, ~120 KB wheel)
    HAVE_MIXBOX = True
except ImportError:  # pragma: no cover
    mixbox = None  # type: ignore[assignment]
    HAVE_MIXBOX = False

_log = logging.getLogger(__name__)
_WARNED_NO_MIXBOX = False


def _warn_once_no_mixbox() -> None:
    global _WARNED_NO_MIXBOX
    if not _WARNED_NO_MIXBOX:
        _log.warning(
            "mix_mode='mixbox' requested but pymixbox is not installed — "
            "falling back to linear RGB blend. Install via "
            "`pip install pymixbox` for true pigment-space mixing."
        )
        _WARNED_NO_MIXBOX = True


# ─── single-pair API ─────────────────────────────────────────────────

def mix_rgb(
    c1: tuple[float, float, float],
    c2: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    """Mix two RGB colours in pigment space at parameter ``t ∈ [0, 1]``.

    Inputs/outputs are in the unit interval ``[0, 1]``; conversion to
    ``pymixbox``'s ``[0, 255]`` uint8 happens internally.

    Example
    -------
    >>> blue, yellow = (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)
    >>> r, g, b = mix_rgb(blue, yellow, 0.5)
    >>> g > r and g > b  # greener than red/blue
    True
    """
    # Round-42: clamp inputs + outputs to [0, 1]. Pre-round-42 an
    # HDR-linear caller (e.g. value 1.2) hit `int(round(1.2*255))=306`
    # which is out of pymixbox's expected uint8 range — behaviour was
    # undefined. The fallback branch could also return values >1.
    def _clip3(c):
        return (max(0.0, min(1.0, float(c[0]))),
                max(0.0, min(1.0, float(c[1]))),
                max(0.0, min(1.0, float(c[2]))))
    c1 = _clip3(c1); c2 = _clip3(c2)
    t = max(0.0, min(1.0, float(t)))
    if HAVE_MIXBOX:
        a = tuple(int(round(x * 255.0)) for x in c1)
        b = tuple(int(round(x * 255.0)) for x in c2)
        r, g, bl = mixbox.lerp(a, b, t)
        return (r / 255.0, g / 255.0, bl / 255.0)
    _warn_once_no_mixbox()
    return (
        c1[0] * (1 - t) + c2[0] * t,
        c1[1] * (1 - t) + c2[1] * t,
        c1[2] * (1 - t) + c2[2] * t,
    )


# ─── per-vertex KNN remix (B18.5) ────────────────────────────────────

@block("M5.11.5",
       "Mixbox pigment-space per-vertex colour blend (B18.5)")
def remix_vertices_mixbox(
    verts_world: np.ndarray,
    particle_pos: np.ndarray,
    particle_colors: np.ndarray,
    search_radius_world: float,
    max_neighbours: int = 16,
) -> np.ndarray:
    """Per-vertex pigment-space blend from K nearest particle colours.

    Mirrors the GPU linear-RGB pass in
    :meth:`MeshExtractor.compute_vertex_colors` but does the blend in
    Mixbox latent space (vec7), giving a perceptual-pigment result.
    CPU-only for now: pymixbox is a pure-Python LUT that doesn't
    JIT-compile to GPU; the cost is acceptable (~10-50k vertices × 16
    neighbours = ~0.5 s on a typical demo frame).

    Parameters
    ----------
    verts_world : (N, 3) float32 — mesh vertex positions.
    particle_pos : (P, 3) float32 — particle positions, **same row
        order** as ``particle_colors`` (the bake-side cache filtering
        ensures this).
    particle_colors : (P, 3) float32 in [0, 1] — particle RGB.
    search_radius_world : neighbourhood radius, world units.
    max_neighbours : K cap to bound CPU time (default 16, matches the
        Mixbox-paper recommended density).

    Returns
    -------
    (N, 3) uint8 RGB — ready for the PLY writer.

    Notes
    -----
    Latent-space averaging is a well-defined N-way Mixbox blend:
    ``latent_avg = Σ wᵢ · latent(cᵢ) / Σ wᵢ``, then back to RGB. This
    is the same operation pymixbox's :func:`mixbox.lerp` does internally
    for the 2-colour case, generalised to weighted N.
    """
    from scipy.spatial import cKDTree  # imported here to keep top-level cheap

    n_v = verts_world.shape[0]
    if n_v == 0 or particle_pos.shape[0] == 0:
        return np.zeros((n_v, 3), dtype=np.uint8)
    if particle_colors.shape != particle_pos.shape:
        raise ValueError(
            f"particle_colors shape {particle_colors.shape} != "
            f"particle_pos shape {particle_pos.shape}"
        )

    if not HAVE_MIXBOX:
        _warn_once_no_mixbox()
        # Linear-RGB fallback: cheap reproduction of the GPU pass on CPU.
        # Same KDTree query, inverse-distance² weights, no latent space.
        tree = cKDTree(particle_pos.astype(np.float32))
        out = np.full((n_v, 3), 255, dtype=np.uint8)
        for i, v in enumerate(verts_world):
            idx = tree.query_ball_point(v, r=search_radius_world)
            if not idx:
                continue
            idx = np.asarray(idx)
            d = particle_pos[idx] - v
            d_sq = (d * d).sum(axis=1)
            if idx.size > max_neighbours:
                # audit-2026-06-14r2 #7: keep the K NEAREST. query_ball_point
                # returns neighbours in unordered tree-traversal order, so the
                # old `[:max_neighbours]` blended an ARBITRARY 16, not the
                # nearest 16 — the mixbox branch below already sorts; this
                # fallback (the live path when pymixbox is absent) had drifted.
                order = np.argsort(d_sq)[:max_neighbours]
                idx = idx[order]
                d_sq = d_sq[order]
            w = 1.0 / (d_sq + 1e-8)
            w = w / w.sum()
            rgb = (w[:, None] * particle_colors[idx]).sum(axis=0)
            out[i] = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        return out

    # Mixbox latent-space remix. Pre-compute latents for ALL particles
    # once (LATENT_SIZE × n_particles floats), then per-vertex weighted
    # average is a cheap dot product.
    LS = mixbox.LATENT_SIZE  # 7 floats
    latents = np.empty((particle_pos.shape[0], LS), dtype=np.float32)
    for pi, c in enumerate(particle_colors):
        latents[pi] = mixbox.float_rgb_to_latent(
            (float(c[0]), float(c[1]), float(c[2]))
        )
    tree = cKDTree(particle_pos.astype(np.float32))
    out = np.full((n_v, 3), 255, dtype=np.uint8)
    for i, v in enumerate(verts_world):
        idx = tree.query_ball_point(v, r=search_radius_world)
        if not idx:
            continue
        idx = np.asarray(idx)
        if idx.size > max_neighbours:
            # Keep the K nearest (query_ball_point is unordered).
            d_all = particle_pos[idx] - v
            d_sq_all = (d_all * d_all).sum(axis=1)
            order = np.argsort(d_sq_all)[:max_neighbours]
            idx = idx[order]
            d_sq = d_sq_all[order]
        else:
            d = particle_pos[idx] - v
            d_sq = (d * d).sum(axis=1)
        w = 1.0 / (d_sq + 1e-8)
        w = w / w.sum()
        latent_avg = (w[:, None] * latents[idx]).sum(axis=0)
        r, g, b = mixbox.latent_to_float_rgb(tuple(float(x) for x in latent_avg))
        out[i] = (
            int(np.clip(round(r * 255.0), 0, 255)),
            int(np.clip(round(g * 255.0), 0, 255)),
            int(np.clip(round(b * 255.0), 0, 255)),
        )
    return out
