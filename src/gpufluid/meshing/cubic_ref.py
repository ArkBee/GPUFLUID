"""[Layer M5] CPU reference for M5.9 wider cubic kernel scatter.

Discovered as the cheap-but-effective slice of B13 Yu-Turk evaluation:
the kernel SIZE change (M5.1 trilinear -> cubic R=2cells) reduces mesh
fragment count by 29x on real demo30 data; the additional anisotropic
shape change moves the needle by <=10%. So M5.9 ships the cheap part.

See DESIGN.md §8.3 for the full algorithm and rationale.

Pure numpy. Slow (no neighbour search, but per-particle AABB iteration
in Python). Used as the GPU parity baseline only.
"""
from __future__ import annotations
import numpy as np


def cubic_isotropic_density_cpu(
    positions: np.ndarray,
    nx: int, ny: int, nz: int,
    dx: float,
    radius_cells: float = 2.0,
) -> np.ndarray:
    """Per-particle cubic-kernel density scatter with fixed sphere support.

    For each particle at world position x_i, deposit a contribution
    ``(1 - r²)³`` into every cell within ``radius_cells`` cells of x_i,
    where ``r²`` is the squared normalised distance from the cell centre
    to x_i.

    Parameters
    ----------
    positions : (N, 3) float64
        Particle positions in world coordinates.
    nx, ny, nz : int
        Density grid dimensions in cells.
    dx : float
        World size per cell.
    radius_cells : float, default 2.0
        Support radius in cells. Beyond this distance the kernel is 0.

    Returns
    -------
    density : (nx, ny, nz) float64
        Same shape and indexing convention as M5.1 output, suitable to
        feed M5.2 box-blur and M5.4 marching cubes unchanged.
    """
    positions = np.asarray(positions, dtype=np.float64)
    density = np.zeros((nx, ny, nz), dtype=np.float64)
    # Round-30: align with sdf_ref + adaptive_cubic_ref defensive
    # behaviour. Pre-round-30 an empty positions array was tolerated
    # (loop didn't iterate) but a NaN row crashed at `int(np.floor(
    # nan/dx))` — both paths now early-return the zero grid.
    if positions.size == 0:
        return density
    if not np.all(np.isfinite(positions)):
        # Drop NaN/inf rows defensively; alternative would be raise,
        # but mesher caller is hot-path and a single corrupt particle
        # shouldn't kill the whole frame.
        positions = positions[np.all(np.isfinite(positions), axis=1)]
        if positions.size == 0:
            return density
    radius_world = radius_cells * dx
    inv_r_sq = 1.0 / (radius_world * radius_world)
    half_cells = int(np.ceil(radius_cells)) + 1

    for p in positions:
        cx = int(np.floor(p[0] / dx - 0.5))
        cy = int(np.floor(p[1] / dx - 0.5))
        cz = int(np.floor(p[2] / dx - 0.5))
        i_lo = max(0, cx - half_cells); i_hi = min(nx, cx + half_cells + 1)
        j_lo = max(0, cy - half_cells); j_hi = min(ny, cy + half_cells + 1)
        k_lo = max(0, cz - half_cells); k_hi = min(nz, cz + half_cells + 1)
        if i_lo >= i_hi or j_lo >= j_hi or k_lo >= k_hi:
            continue
        ii, jj, kk = np.meshgrid(
            np.arange(i_lo, i_hi), np.arange(j_lo, j_hi), np.arange(k_lo, k_hi),
            indexing="ij",
        )
        cx_w = (ii + 0.5) * dx
        cy_w = (jj + 0.5) * dx
        cz_w = (kk + 0.5) * dx
        r_sq = ((cx_w - p[0]) ** 2 + (cy_w - p[1]) ** 2 + (cz_w - p[2]) ** 2) * inv_r_sq
        mask = r_sq < 1.0
        contribution = np.zeros_like(r_sq)
        contribution[mask] = (1.0 - r_sq[mask]) ** 3
        density[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi] += contribution

    return density
