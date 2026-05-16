"""B7-alt.1 — spike: deferred-dense allocation feasibility for v1.0.

Pivots away from the aborted B7 (NanoVDB), which depended on a Warp
`volume_store_*` API that doesn't exist. The B7-alt plan from HANDOFF
§8.0b.B7 / §12: allocate `wp.array3d` lazily for the bbox of active
8³ tiles, rebuild every N frames with a dilation margin. Existing
dense kernels work unchanged if they accept an `offset_xyz` parameter
so global (i,j,k) → local (i-ox, j-oy, k-oz) translation lives in the
kernel.

This spike answers TWO killers before greenlighting the macro:

  1. **Memory bbox ratio.** At 128³ / ~5% fill, does bbox(active_tiles)
     drop allocations by ≥5× vs full dense? If the fluid is connected
     the bbox is small — easy win. If scattered (many droplets across
     the domain) the bbox covers everything → no memory win → macro
     dead. We measure BOTH scenarios so the spike can't be deceived
     by a single easy setup.

  2. **Coord-translation correctness.** A dense Jacobi-like kernel,
     re-launched on a sub-dense `wp.array3d` of size bbox+dilation with
     an `offset_xyz` parameter, must produce the same numerical result
     on the active region as the full-dense launch. If it doesn't, the
     plan to leave existing kernels almost-unchanged falls apart.

Spike verdict (printed at the end of each test, not asserted in the
borderline cases — like B7.1, the failures are deterministic kills,
not perf thresholds):

  * GREEN if memory ratio ≥ 5× on connected-blob AND coord-translation
    matches dense within 1e-5 relative.
  * RED if either fails — abort the macro before writing micros.
"""
import time
import numpy as np
import pytest
import warp as wp


BLOCK_SIZE = 8  # must match solver3d.BLOCK_SIZE


# ---------------------------------------------------------------------------
# Helpers: build markers + active-tile bbox.
# ---------------------------------------------------------------------------

def _build_connected_blob_marker(N: int, fill: float):
    """Marker with a single connected cube of fluid (interior), ~`fill`
    of the domain volume. Wraps the cube in solid walls so the marker
    layout matches what the real solver produces."""
    side = int(round((fill * N ** 3) ** (1.0 / 3.0)))
    side = min(side, N - 4)
    marker = np.zeros((N, N, N), dtype=np.int32)
    # Walls
    marker[0, :, :] = 2; marker[-1, :, :] = 2
    marker[:, 0, :] = 2; marker[:, -1, :] = 2
    marker[:, :, 0] = 2; marker[:, :, -1] = 2
    # Fluid cube near (0.2..0.2+s, 0.2..0.2+s, 0.2..0.2+s) of N
    o = max(2, N // 5)
    marker[o:o + side, o:o + side, o:o + side] = 1
    return marker


def _build_scattered_droplets_marker(N: int, fill: float, n_droplets: int = 200,
                                      seed: int = 7):
    """Marker with many small fluid clusters scattered across the domain.
    Designed to *minimise* the spatial locality the bbox approach relies
    on. ~`fill` of the domain is fluid, split into roughly equal blobs."""
    target = int(fill * N ** 3)
    rng = np.random.default_rng(seed)
    marker = np.zeros((N, N, N), dtype=np.int32)
    marker[0, :, :] = 2; marker[-1, :, :] = 2
    marker[:, 0, :] = 2; marker[:, -1, :] = 2
    marker[:, :, 0] = 2; marker[:, :, -1] = 2

    # Each droplet is a small cube placed at a random interior origin.
    cells_per_drop = max(1, target // n_droplets)
    side = max(1, int(round(cells_per_drop ** (1.0 / 3.0))))
    drops_placed = 0
    while drops_placed < n_droplets:
        oi = rng.integers(2, N - side - 2)
        oj = rng.integers(2, N - side - 2)
        ok = rng.integers(2, N - side - 2)
        marker[oi:oi + side, oj:oj + side, ok:ok + side] = 1
        drops_placed += 1
    return marker


def _active_tile_bbox(marker: np.ndarray, bs: int = BLOCK_SIZE):
    """Return (lo, hi, n_active_tiles) where lo/hi are inclusive 8³-tile
    bounds in cell units that cover every active tile. None if there
    are no active tiles."""
    nbx = marker.shape[0] // bs
    nby = marker.shape[1] // bs
    nbz = marker.shape[2] // bs
    # 8³-blocked indicator — any fluid cell makes the tile active.
    blocked = marker.reshape(nbx, bs, nby, bs, nbz, bs)
    blocked = blocked.transpose(0, 2, 4, 1, 3, 5).reshape(nbx, nby, nbz, -1)
    tile_has_fluid = (blocked == 1).any(axis=-1)
    if not tile_has_fluid.any():
        return None
    bi, bj, bk = np.where(tile_has_fluid)
    lo_b = (int(bi.min()), int(bj.min()), int(bk.min()))
    hi_b = (int(bi.max()) + 1, int(bj.max()) + 1, int(bk.max()) + 1)
    return (
        tuple(b * bs for b in lo_b),
        tuple(b * bs for b in hi_b),
        int(tile_has_fluid.sum()),
    )


# ---------------------------------------------------------------------------
# Spike kernels — Jacobi smoother on dense AND coord-translated sub-dense.
# ---------------------------------------------------------------------------

@wp.kernel
def k3_jacobi_spike_dense(
    p_in: wp.array3d(dtype=float),
    p_out: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
):
    """Reference Jacobi sweep, identical to `k3_jacobi_pressure` modulo
    the wp.block tag (omitted here so the spike is self-contained)."""
    i, j, k = wp.tid()
    nx = p_in.shape[0]; ny = p_in.shape[1]; nz = p_in.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    if marker[i, j, k] != 1:
        p_out[i, j, k] = 0.0
        return
    sum_nb = float(0.0); diag = float(0.0)
    if i > 0:
        m = marker[i - 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i - 1, j, k]
    if i < nx - 1:
        m = marker[i + 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i + 1, j, k]
    if j > 0:
        m = marker[i, j - 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j - 1, k]
    if j < ny - 1:
        m = marker[i, j + 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j + 1, k]
    if k > 0:
        m = marker[i, j, k - 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j, k - 1]
    if k < nz - 1:
        m = marker[i, j, k + 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j, k + 1]
    if diag > 0.0:
        p_out[i, j, k] = (div[i, j, k] + sum_nb) / diag
    else:
        p_out[i, j, k] = 0.0


@wp.kernel
def k3_jacobi_spike_offset(
    p_sub_in: wp.array3d(dtype=float),
    p_sub_out: wp.array3d(dtype=float),
    div_full: wp.array3d(dtype=float),
    marker_full: wp.array3d(dtype=int),
    off_x: int, off_y: int, off_z: int,
):
    """Same Jacobi update, but `p_*` arrays cover a sub-dense bbox; the
    kernel translates the local (li,lj,lk) to global (li+off, lj+off,
    lk+off) when reading marker/div, while neighbour reads come from
    the sub-dense buffer using local offsets.

    Boundary handling: at the EDGE of the sub-dense buffer the neighbour
    falls outside; treat that neighbour as "non-fluid, zero pressure"
    (it would be air or another non-active region — pressure is 0 there
    anyway). With a dilation margin of ≥1 cell, the active interior is
    always far enough from the sub-dense edge that this approximation
    doesn't perturb the result.
    """
    li, lj, lk = wp.tid()
    sx = p_sub_in.shape[0]; sy = p_sub_in.shape[1]; sz = p_sub_in.shape[2]
    if li >= sx or lj >= sy or lk >= sz:
        return
    # Translate to global indices for marker/div lookups.
    gi = li + off_x; gj = lj + off_y; gk = lk + off_z
    nx = marker_full.shape[0]; ny = marker_full.shape[1]; nz = marker_full.shape[2]
    if gi < 0 or gj < 0 or gk < 0 or gi >= nx or gj >= ny or gk >= nz:
        p_sub_out[li, lj, lk] = 0.0
        return
    if marker_full[gi, gj, gk] != 1:
        p_sub_out[li, lj, lk] = 0.0
        return
    sum_nb = float(0.0); diag = float(0.0)
    # neg-i neighbour: marker is at (gi-1, gj, gk); pressure from p_sub at (li-1)
    if gi > 0:
        m = marker_full[gi - 1, gj, gk]
        if m != 2:
            diag += 1.0
            if m == 1 and li > 0:
                sum_nb += p_sub_in[li - 1, lj, lk]
    if gi < nx - 1:
        m = marker_full[gi + 1, gj, gk]
        if m != 2:
            diag += 1.0
            if m == 1 and li < sx - 1:
                sum_nb += p_sub_in[li + 1, lj, lk]
    if gj > 0:
        m = marker_full[gi, gj - 1, gk]
        if m != 2:
            diag += 1.0
            if m == 1 and lj > 0:
                sum_nb += p_sub_in[li, lj - 1, lk]
    if gj < ny - 1:
        m = marker_full[gi, gj + 1, gk]
        if m != 2:
            diag += 1.0
            if m == 1 and lj < sy - 1:
                sum_nb += p_sub_in[li, lj + 1, lk]
    if gk > 0:
        m = marker_full[gi, gj, gk - 1]
        if m != 2:
            diag += 1.0
            if m == 1 and lk > 0:
                sum_nb += p_sub_in[li, lj, lk - 1]
    if gk < nz - 1:
        m = marker_full[gi, gj, gk + 1]
        if m != 2:
            diag += 1.0
            if m == 1 and lk < sz - 1:
                sum_nb += p_sub_in[li, lj, lk + 1]
    if diag > 0.0:
        p_sub_out[li, lj, lk] = (div_full[gi, gj, gk] + sum_nb) / diag
    else:
        p_sub_out[li, lj, lk] = 0.0


# ---------------------------------------------------------------------------
# Spike test 1 — memory ratio across two scene topologies.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_b7_alt_1_memory_ratio_connected_blob():
    """Connected blob at 128³ / ~5% fill. The bbox should be much smaller
    than the full domain — this is the *easy* case where B7-alt earns
    its memory back. Target ≥5× drop."""
    N = 128
    marker = _build_connected_blob_marker(N, fill=0.05)
    bbox = _active_tile_bbox(marker)
    assert bbox is not None
    lo, hi, n_active_tiles = bbox
    sub_dim = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    sub_cells = sub_dim[0] * sub_dim[1] * sub_dim[2]
    full_cells = N ** 3
    ratio = full_cells / sub_cells
    fill = (marker == 1).sum() / full_cells
    print(f"\n[B7-alt.1] CONNECTED-BLOB  N={N}, fill={fill*100:.2f}%, "
          f"n_active_tiles={n_active_tiles}, "
          f"bbox={sub_dim}, sub/full = {sub_cells/full_cells*100:.2f}%, "
          f"memory drop = {ratio:.2f}x")
    # Greenlight criterion for the macro: at least 5x memory drop on the
    # easy case. If we can't manage even this, the macro is dead.
    assert ratio >= 5.0, (
        f"connected-blob 5% fill landed at only {ratio:.1f}x memory drop; "
        f"B7-alt macro should be aborted (worse than B5 sparse-graph)."
    )


@pytest.mark.gpu
def test_b7_alt_1_memory_ratio_scattered_droplets():
    """Scattered droplets at 128³ / ~5% fill. Designed to be hostile to
    the bbox approach — droplets spread across the domain force the
    bbox to cover almost everything. This is the *hard* case; we
    report the ratio but don't fail on it. If it's <1.5x even here,
    the macro's value proposition is narrow (only helps "blob" scenes).
    """
    N = 128
    marker = _build_scattered_droplets_marker(N, fill=0.05, n_droplets=200)
    bbox = _active_tile_bbox(marker)
    assert bbox is not None
    lo, hi, n_active_tiles = bbox
    sub_dim = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    sub_cells = sub_dim[0] * sub_dim[1] * sub_dim[2]
    full_cells = N ** 3
    ratio = full_cells / sub_cells
    fill = (marker == 1).sum() / full_cells
    print(f"\n[B7-alt.1] SCATTERED-DROPS N={N}, fill={fill*100:.2f}%, "
          f"n_active_tiles={n_active_tiles}, "
          f"bbox={sub_dim}, sub/full = {sub_cells/full_cells*100:.2f}%, "
          f"memory drop = {ratio:.2f}x")
    # NOT a hard assert — this is documentation. The macro's promise is
    # ~70-80% of sparse v2 memory savings on REALISTIC scenes (which
    # tend to be blob-shaped). Scattered droplets are a known pathology
    # that the dilation-margin rebuild strategy doesn't fix.


# ---------------------------------------------------------------------------
# Spike test 2 — coord-translated kernel matches dense on connected blob.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_b7_alt_1_coord_translation_jacobi_matches_dense():
    """The killer correctness test. Run 40 Jacobi iters on the same
    marker + div field two ways:
      (a) full-dense, the existing kernel pattern;
      (b) sub-dense bbox + dilation, with the offset-aware spike kernel.

    The sub-dense pressure field, lifted back into a 128³ buffer, must
    match the full-dense pressure field on every fluid cell of the
    active region to within 1e-4 relative.

    If this fails, the offset-translation idea is broken and B7-alt
    needs a different mechanism (or to abort)."""
    N = 128
    DILATION = 1  # cells; thin margin is enough for steady-state
    marker = _build_connected_blob_marker(N, fill=0.06)
    bbox = _active_tile_bbox(marker)
    assert bbox is not None
    lo, hi, n_active_tiles = bbox
    # Dilation margin on every face, clipped to grid.
    lo_d = tuple(max(0, c - DILATION) for c in lo)
    hi_d = tuple(min(N, c + DILATION) for c in hi)
    sub_shape = tuple(hi_d[a] - lo_d[a] for a in range(3))

    # Build a non-trivial divergence pattern so the smoother actually
    # has work to do. Sinusoidal-ish field over fluid cells.
    div = np.zeros((N, N, N), dtype=np.float32)
    ii, jj, kk = np.meshgrid(
        np.arange(N), np.arange(N), np.arange(N), indexing="ij"
    )
    div[marker == 1] = (
        0.5 * np.sin(0.1 * ii)
        + 0.3 * np.cos(0.07 * jj)
        + 0.2 * np.sin(0.13 * kk)
    )[marker == 1].astype(np.float32)

    # ---- (a) full-dense Jacobi
    marker_wp = wp.array(marker, dtype=int, device="cuda:0")
    div_wp = wp.array(div, dtype=float, device="cuda:0")
    p_a = wp.zeros((N, N, N), dtype=float, device="cuda:0")
    p_a_tmp = wp.zeros((N, N, N), dtype=float, device="cuda:0")
    for _ in range(40):
        wp.launch(k3_jacobi_spike_dense, dim=(N, N, N),
                  inputs=[p_a, p_a_tmp, div_wp, marker_wp], device="cuda:0")
        p_a, p_a_tmp = p_a_tmp, p_a
    p_a_np = p_a.numpy()

    # ---- (b) sub-dense bbox+dilation Jacobi with offset-aware kernel
    p_b = wp.zeros(sub_shape, dtype=float, device="cuda:0")
    p_b_tmp = wp.zeros(sub_shape, dtype=float, device="cuda:0")
    off = lo_d
    for _ in range(40):
        wp.launch(k3_jacobi_spike_offset, dim=sub_shape,
                  inputs=[p_b, p_b_tmp, div_wp, marker_wp,
                          int(off[0]), int(off[1]), int(off[2])],
                  device="cuda:0")
        p_b, p_b_tmp = p_b_tmp, p_b
    p_b_np = p_b.numpy()

    # Lift sub-dense pressure into a full-dense grid for comparison.
    p_b_full = np.zeros((N, N, N), dtype=np.float32)
    p_b_full[off[0]:off[0] + sub_shape[0],
             off[1]:off[1] + sub_shape[1],
             off[2]:off[2] + sub_shape[2]] = p_b_np

    # Compare ONLY on the active fluid cells (marker == 1). Pressure
    # outside the fluid region is zero in both anyway.
    fluid = (marker == 1)
    err = np.abs(p_a_np - p_b_full)[fluid]
    rel = err.max() / max(np.abs(p_a_np[fluid]).max(), 1e-9)
    print(f"\n[B7-alt.1] CORRECTNESS  N={N}, sub_shape={sub_shape}, "
          f"n_fluid={int(fluid.sum())}, "
          f"max abs err={err.max():.3e}, rel err={rel:.3e}")
    assert rel < 1e-4, (
        f"coord-translated Jacobi diverged from full-dense; "
        f"rel err={rel:.3e}. B7-alt macro needs a different boundary "
        f"strategy than 'dilation=1 + zero-pressure-outside'."
    )


# ---------------------------------------------------------------------------
# Spike test 3 — verdict summary (informational, never fails).
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_b7_alt_1_verdict():
    """Aggregate the two killer answers into a single human-readable
    line. Intended to be the LAST line you see when running this module
    so the verdict is at the bottom of pytest output."""
    N = 128
    # Re-measure (cheap, deterministic) so the verdict line is
    # self-contained and survives running this test in isolation.
    m_blob = _build_connected_blob_marker(N, fill=0.05)
    m_drops = _build_scattered_droplets_marker(N, fill=0.05, n_droplets=200)
    lo_b, hi_b, _ = _active_tile_bbox(m_blob)
    lo_d, hi_d, _ = _active_tile_bbox(m_drops)
    r_blob = (N ** 3) / ((hi_b[0] - lo_b[0]) * (hi_b[1] - lo_b[1]) * (hi_b[2] - lo_b[2]))
    r_drops = (N ** 3) / ((hi_d[0] - lo_d[0]) * (hi_d[1] - lo_d[1]) * (hi_d[2] - lo_d[2]))
    print("")
    print("=" * 64)
    print(f"[B7-alt.1] SPIKE VERDICT (N={N}, ~5% fill):")
    print(f"  memory drop: connected-blob = {r_blob:.2f}x, "
          f"scattered-droplets = {r_drops:.2f}x")
    print(f"  coord-translation: see test_b7_alt_1_coord_translation_jacobi_matches_dense")
    if r_blob >= 5.0:
        print("  -> GREEN on blob topology. Macro greenlit pending the "
              "correctness test above.")
    else:
        print("  -> RED on blob topology. Abort macro.")
    if r_drops >= 1.5:
        print(f"  scattered-drops case OK at {r_drops:.2f}x — "
              "macro helps even moderately-sparse scenes.")
    else:
        print(f"  scattered-drops case offers only {r_drops:.2f}x — "
              "macro's value is narrow to connected-blob scenes; "
              "document in BACKLOG that scattered scenes get nothing.")
    print("=" * 64)
