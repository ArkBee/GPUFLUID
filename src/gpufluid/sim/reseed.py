"""[Layer S2 / BLK S2.11 + S2.11.GPU] Particle reseeding — keep per-cell density bounded.

Over time a FLIP/PIC simulation develops two problems:

* **Voids**: cells inside the fluid drop to 0-1 particles, marching cubes
  flips them out of the surface, holes appear.
* **Over-sampling**: a few cells accumulate >50 particles after collisions,
  wasting work and biasing the surface.

This module ships two implementations:

* **S2.11** — host-side, simple to read, fine up to ~200k particles.
* **S2.11.GPU** — device-resident count + rank + compact (D4.7.GPU pattern)
  + minimal host work for the emit list. Auto-engaged above
  ``RESEED_GPU_THRESHOLD`` particles via ``reseed_particles_gpu``.

Both leave new particles with zero velocity; one solver step blends them
back into the flow via G2P. For higher fidelity, sample the grid
velocity at the emitted position — left as a future micro-optimisation.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from dataclasses import dataclass
from typing import Optional, Tuple

from ..blocks import block
from ..primitives.runtime import init as warp_init, device as default_device

warp_init()


@dataclass
class ReseedConfig:
    min_per_cell: int = 4      # below this → emit
    max_per_cell: int = 16     # above this → cull
    jitter: float = 0.3        # fraction of dx for emission jitter
    every_n_frames: int = 5


# [BLK S2.11]
@block("S2.11", "Reseed particles to maintain bounded per-cell density")
def reseed_particles(
    pos: np.ndarray, vel: np.ndarray,
    marker_host: np.ndarray,
    dx: float,
    cfg: ReseedConfig,
    rng: np.random.Generator,
    attr_color: np.ndarray | None = None,
    attr_temperature: np.ndarray | None = None,
):
    """Return ``(new_pos, new_vel, n_emitted, n_culled, new_color,
    new_temperature)``.

    Round-36: ``attr_color`` and ``attr_temperature`` are optional row-
    aligned with ``pos``; when provided, they ride the same keep_mask +
    emit-append as positions, so the CPU reseed stays in lockstep with
    pos/vel. GPU sibling (``reseed_particles_gpu``) had this since the
    start; the CPU path was forgotten — exact lesson §9.6 mirror-drift
    that bit on round-35 reviewer's sweep.

    Emitted particles get default attrs (zero colour, 20°C temperature),
    matching the prepare_frame emit-append convention (round-34).

    ``marker_host``: (nx, ny, nz) int — only cells marked fluid (==1) are reseeded.
    Cells marked solid (==2) are skipped entirely so no particles end up in solids.
    """
    nx, ny, nz = marker_host.shape

    # ---- 1. bin particles into cells (handle empty array)
    if len(pos) > 0:
        ix = np.clip((pos[:, 0] / dx).astype(np.int32), 0, nx - 1)
        iy = np.clip((pos[:, 1] / dx).astype(np.int32), 0, ny - 1)
        iz = np.clip((pos[:, 2] / dx).astype(np.int32), 0, nz - 1)
        cell_idx = ix * (ny * nz) + iy * nz + iz   # flat
        counts = np.bincount(cell_idx, minlength=nx * ny * nz).reshape(nx, ny, nz)
    else:
        cell_idx = np.zeros(0, dtype=np.int32)
        counts = np.zeros((nx, ny, nz), dtype=np.int32)

    # ---- 2. emission targets
    # A cell counts as fluid if it's explicitly fluid in the marker OR if it
    # currently contains particles AND isn't solid. This handles the common
    # case where the caller only populates wall/solid markers; fluid presence
    # is then inferred from the particle distribution.
    solid_mask = (marker_host == 2)
    fluid_mask = ((marker_host == 1) | (counts > 0)) & ~solid_mask
    deficit = np.clip(cfg.min_per_cell - counts, 0, None)
    deficit[~fluid_mask] = 0
    emit_per_cell = deficit.astype(np.int32)
    n_emitted = int(emit_per_cell.sum())

    emit_pos = np.empty((n_emitted, 3), dtype=np.float32) if n_emitted else None
    emit_vel = np.zeros((n_emitted, 3), dtype=np.float32) if n_emitted else None
    if n_emitted > 0:
        # build flat list of (i, j, k) per emission slot
        nz_idx = np.argwhere(emit_per_cell > 0)   # (M, 3)
        rep = emit_per_cell[emit_per_cell > 0]
        # repeat each (i,j,k) by its emit count
        cells = np.repeat(nz_idx, rep, axis=0)    # (n_emitted, 3)
        jitter = (rng.random((n_emitted, 3)) - 0.5).astype(np.float32) * cfg.jitter
        emit_pos[:] = (cells.astype(np.float32) + 0.5 + jitter) * dx

    # ---- 3. culling — for non-solid cells over capacity, drop random extras
    over_cells = counts - cfg.max_per_cell
    over_cells[solid_mask] = 0
    cull_per_cell = np.clip(over_cells, 0, None)
    n_to_cull = int(cull_per_cell.sum())

    keep_mask = np.ones(len(pos), dtype=bool)
    if n_to_cull > 0:
        # For each cell with excess, randomly pick that many particles to drop.
        # Vectorised via grouping argsort over cell_idx.
        order = np.argsort(cell_idx, kind="stable")
        cell_sorted = cell_idx[order]
        # boundaries between cells
        change = np.diff(cell_sorted, prepend=cell_sorted[0] - 1)
        starts = np.where(change != 0)[0]
        ends = np.concatenate([starts[1:], [len(cell_sorted)]])
        for s, e in zip(starts, ends):
            ci = cell_sorted[s]
            ii, jj, kk = ci // (ny * nz), (ci // nz) % ny, ci % nz
            excess = cull_per_cell[ii, jj, kk]
            if excess <= 0:
                continue
            picks = rng.choice(e - s, size=int(excess), replace=False)
            for p in picks:
                keep_mask[order[s + p]] = False

    # audit-2026-06-15r11 (§9.6 mirror of the GPU k_mark_keep_by_rank
    # `if marker==2: alive=0` drop): also remove any particle whose host cell is
    # SOLID, regardless of capacity. A fast particle can penetrate one cell into
    # an interior obstacle before boundary handling stops it; the GPU reseed
    # purges those, the CPU path did not — so below RESEED_GPU_THRESHOLD they
    # accumulated inside solids (the iso-surface then bleeds into the obstacle).
    # This also makes the code honour the docstring's "no particles end up in
    # solids" promise.
    if len(pos) > 0:
        keep_mask &= ~solid_mask[ix, iy, iz]

    new_pos = pos[keep_mask]
    new_vel = vel[keep_mask]
    if n_emitted > 0:
        new_pos = np.concatenate([new_pos, emit_pos], axis=0)
        new_vel = np.concatenate([new_vel, emit_vel], axis=0)
    # Round-36: variant return — preserve 4-tuple back-compat for the
    # ~10 existing call sites that didn't pass attrs (tests etc.).
    # 6-tuple form only fires when caller passed at least one attr.
    if attr_color is None and attr_temperature is None:
        return new_pos, new_vel, n_emitted, n_to_cull
    new_color = attr_color[keep_mask] if attr_color is not None else None
    new_temperature = (attr_temperature[keep_mask]
                       if attr_temperature is not None else None)
    if n_emitted > 0:
        if attr_color is not None:
            emit_col = np.zeros((n_emitted, 3), dtype=np.float32)
            new_color = np.concatenate([new_color, emit_col], axis=0)
        if attr_temperature is not None:
            emit_t = np.full((n_emitted,), 20.0, dtype=np.float32)
            new_temperature = np.concatenate([new_temperature, emit_t],
                                              axis=0)
    return new_pos, new_vel, n_emitted, n_to_cull, new_color, new_temperature


# =============================================================================
# S2.11.GPU — GPU port (count + rank + compact + emit)
# =============================================================================

RESEED_GPU_THRESHOLD = 100_000


@wp.kernel
def k_count_particles_per_cell(
    pos: wp.array(dtype=wp.vec3),
    counts: wp.array3d(dtype=int),
    dx: float, nx: int, ny: int, nz: int,
):
    """[BLK S2.11.GPU.COUNT] Per-particle atomic increment of cell counts."""
    p = wp.tid()
    pp = pos[p]
    i = int(wp.floor(pp[0] / dx))
    j = int(wp.floor(pp[1] / dx))
    k = int(wp.floor(pp[2] / dx))
    if i < 0: i = 0
    if j < 0: j = 0
    if k < 0: k = 0
    if i >= nx: i = nx - 1
    if j >= ny: j = ny - 1
    if k >= nz: k = nz - 1
    wp.atomic_add(counts, i, j, k, 1)


block("S2.11.GPU.COUNT", "Atomic per-cell particle count")(k_count_particles_per_cell)


@wp.kernel
def k_mark_keep_by_rank(
    pos: wp.array(dtype=wp.vec3),
    marker: wp.array3d(dtype=int),
    seen_count: wp.array3d(dtype=int),
    max_per_cell: int,
    alive: wp.array(dtype=int),
    dx: float, nx: int, ny: int, nz: int,
):
    """[BLK S2.11.GPU.RANK] Each particle atomically grabs a rank-in-cell.

    Rules:
      * Particles whose host cell is solid (marker==2) are dropped (alive=0).
        This shouldn't happen in a well-behaved sim but reseeding cures it.
      * The first `max_per_cell` particles per cell (by atomic order) are
        kept; later arrivals are dropped.
    """
    p = wp.tid()
    pp = pos[p]
    i = int(wp.floor(pp[0] / dx))
    j = int(wp.floor(pp[1] / dx))
    k = int(wp.floor(pp[2] / dx))
    if i < 0: i = 0
    if j < 0: j = 0
    if k < 0: k = 0
    if i >= nx: i = nx - 1
    if j >= ny: j = ny - 1
    if k >= nz: k = nz - 1
    if marker[i, j, k] == 2:
        alive[p] = 0
        return
    rank = wp.atomic_add(seen_count, i, j, k, 1)
    if rank >= max_per_cell:
        alive[p] = 0
    else:
        alive[p] = 1


block("S2.11.GPU.RANK", "Per-particle rank-in-cell + alive mask")(k_mark_keep_by_rank)


@wp.kernel
def k_scatter_alive_pv(
    pos_in: wp.array(dtype=wp.vec3),
    vel_in: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=int),
    prefix: wp.array(dtype=int),
    pos_out: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
):
    """If alive[i], copy pos_in[i]/vel_in[i] to slot prefix[i]-1.

    Distinct from `k3_scatter_alive` in solver3d.py only by name; kept local
    to avoid a cross-module import dance from sim/ → solvers/."""
    i = wp.tid()
    if alive[i] == 1:
        dst = prefix[i] - 1
        pos_out[dst] = pos_in[i]
        vel_out[dst] = vel_in[i]


@wp.kernel
def k_scatter_alive_color(
    color_in: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=int),
    prefix: wp.array(dtype=int),
    color_out: wp.array(dtype=wp.vec3),
):
    """Separate kernel for the S2.15 colour sidecar to keep the main scatter
    branchless when no colours are present."""
    i = wp.tid()
    if alive[i] == 1:
        color_out[prefix[i] - 1] = color_in[i]


@wp.kernel
def k_scatter_alive_scalar(
    scalar_in: wp.array(dtype=float),
    alive: wp.array(dtype=int),
    prefix: wp.array(dtype=int),
    scalar_out: wp.array(dtype=float),
):
    """B11.3 follow-up — sibling of `k_scatter_alive_color` for the S2.18
    scalar attribute (temperature). Same compaction pattern."""
    i = wp.tid()
    if alive[i] == 1:
        scalar_out[prefix[i] - 1] = scalar_in[i]


@block("S2.11.GPU", "GPU reseed orchestrator: count → rank-cull → compact → emit")
def reseed_particles_gpu(
    pos_wp: wp.array,
    vel_wp: wp.array,
    marker_wp: wp.array,
    dx: float,
    cfg: ReseedConfig,
    rng: np.random.Generator,
    attr_color_wp: Optional[wp.array] = None,
    attr_temperature_wp: Optional[wp.array] = None,
    device: Optional[str] = None,
) -> Tuple[wp.array, wp.array, Optional[wp.array], Optional[wp.array], int, int]:
    """Return ``(new_pos_wp, new_vel_wp, new_color_wp, new_temperature_wp,
    n_emitted, n_culled)``.

    ``attr_color_wp`` / ``attr_temperature_wp`` are optional; if provided
    they're compacted in lockstep with positions/velocities.
    Output Warp arrays are fresh allocations (the caller's old buffers
    can be dropped). Velocities of emitted particles are zero; their
    colour is zero (0,0,0) and temperature 20.0 — matching the CPU
    reseed path and FlipSolver3D.prepare_frame (audit-2026-06-14r2 #6;
    was white/0.0, a §9.6 mirror-drift). The next G2P sweep repaints them.
    """
    import warp.utils as wputils
    dev = device or default_device()
    n = pos_wp.shape[0]
    nx, ny, nz = marker_wp.shape

    # 1) Count particles per cell on GPU
    counts = wp.zeros((nx, ny, nz), dtype=int, device=dev)
    if n > 0:
        wp.launch(k_count_particles_per_cell, dim=n,
                  inputs=[pos_wp, counts, dx, nx, ny, nz], device=dev)

    # 2) Rank-cull: per-particle alive flag (1 if rank < max_per_cell)
    alive = wp.zeros(max(n, 1), dtype=int, device=dev)
    if n > 0:
        seen = wp.zeros((nx, ny, nz), dtype=int, device=dev)
        wp.launch(k_mark_keep_by_rank, dim=n,
                  inputs=[pos_wp, marker_wp, seen, int(cfg.max_per_cell),
                          alive, dx, nx, ny, nz], device=dev)

    # 3) Inclusive prefix-sum → scatter alive into compacted buffers
    if n > 0:
        prefix = wp.zeros(n, dtype=int, device=dev)
        wputils.array_scan(alive[:n], prefix, inclusive=True)
        n_alive = int(prefix[n - 1:n].numpy()[0])
        pos_kept = wp.zeros(max(n_alive, 1), dtype=wp.vec3, device=dev)
        vel_kept = wp.zeros(max(n_alive, 1), dtype=wp.vec3, device=dev)
        wp.launch(k_scatter_alive_pv, dim=n,
                  inputs=[pos_wp, vel_wp, alive, prefix, pos_kept, vel_kept],
                  device=dev)
        color_kept = None
        if attr_color_wp is not None:
            color_kept = wp.zeros(max(n_alive, 1), dtype=wp.vec3, device=dev)
            wp.launch(k_scatter_alive_color, dim=n,
                      inputs=[attr_color_wp, alive, prefix, color_kept],
                      device=dev)
        temp_kept = None
        if attr_temperature_wp is not None:
            temp_kept = wp.zeros(max(n_alive, 1), dtype=float, device=dev)
            wp.launch(k_scatter_alive_scalar, dim=n,
                      inputs=[attr_temperature_wp, alive, prefix, temp_kept],
                      device=dev)
    else:
        n_alive = 0
        pos_kept = wp.zeros(0, dtype=wp.vec3, device=dev)
        vel_kept = wp.zeros(0, dtype=wp.vec3, device=dev)
        color_kept = wp.zeros(0, dtype=wp.vec3, device=dev) if attr_color_wp is not None else None
        temp_kept = wp.zeros(0, dtype=float, device=dev) if attr_temperature_wp is not None else None
    n_culled = n - n_alive

    # 4) Build emit list on host (small, O(cells with deficit), not O(particles))
    counts_host = counts.numpy()
    marker_host = marker_wp.numpy()
    solid = (marker_host == 2)
    fluid = ((marker_host == 1) | (counts_host > 0)) & ~solid
    deficit = np.clip(cfg.min_per_cell - counts_host, 0, None)
    deficit[~fluid] = 0
    n_emit = int(deficit.sum())
    if n_emit == 0:
        return (
            pos_kept[:n_alive],
            vel_kept[:n_alive],
            (color_kept[:n_alive] if color_kept is not None else None),
            (temp_kept[:n_alive] if temp_kept is not None else None),
            0, n_culled,
        )

    # repeat (i,j,k) by deficit count, then jitter
    idx_nz = np.argwhere(deficit > 0)
    rep = deficit[deficit > 0]
    cells = np.repeat(idx_nz, rep, axis=0).astype(np.float32)
    jit = (rng.random((n_emit, 3)) - 0.5).astype(np.float32) * cfg.jitter
    emit_pos = (cells + 0.5 + jit) * dx
    emit_vel = np.zeros_like(emit_pos)

    pos_kept_np = pos_kept[:n_alive].numpy() if n_alive > 0 else np.zeros((0, 3), dtype=np.float32)
    vel_kept_np = vel_kept[:n_alive].numpy() if n_alive > 0 else np.zeros((0, 3), dtype=np.float32)
    new_pos_np = np.concatenate([pos_kept_np, emit_pos], axis=0)
    new_vel_np = np.concatenate([vel_kept_np, emit_vel], axis=0)
    new_pos = wp.array(new_pos_np, dtype=wp.vec3, device=dev)
    new_vel = wp.array(new_vel_np, dtype=wp.vec3, device=dev)

    new_color = None
    if color_kept is not None:
        color_kept_np = color_kept[:n_alive].numpy() if n_alive > 0 else np.zeros((0, 3), dtype=np.float32)
        # audit-2026-06-14r2 #6: emitted particles get ZERO (black), matching the
        # CPU reseed path (reseed_particles) and the authoritative inflow-emit
        # convention in FlipSolver3D.prepare_frame. The GPU sibling had drifted
        # to white(1,1,1) (§9.6 mirror-drift), so the same scene flips emit
        # colour the moment particle count crosses RESEED_GPU_THRESHOLD.
        emit_col = np.zeros((n_emit, 3), dtype=np.float32)
        new_color_np = np.concatenate([color_kept_np, emit_col], axis=0)
        new_color = wp.array(new_color_np, dtype=wp.vec3, device=dev)

    new_temperature = None
    if temp_kept is not None:
        temp_kept_np = temp_kept[:n_alive].numpy() if n_alive > 0 else np.zeros(0, dtype=np.float32)
        # audit-2026-06-14r2 #6: emitted particles get T=20.0, matching the CPU
        # reseed path and prepare_frame's unspecified-temperature default. The
        # GPU sibling had drifted to 0.0 (§9.6), causing a 20°C->0°C jump in the
        # temperature-colormap output as particle count crosses the GPU threshold.
        emit_t = np.full(n_emit, 20.0, dtype=np.float32)
        new_temperature_np = np.concatenate([temp_kept_np, emit_t], axis=0)
        new_temperature = wp.array(new_temperature_np, dtype=float, device=dev)

    return new_pos, new_vel, new_color, new_temperature, n_emit, n_culled
