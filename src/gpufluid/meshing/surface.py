"""[Layer M5] Particles → surface mesh via density grid + marching cubes."""
from __future__ import annotations
import numpy as np
import warp as wp
import skimage.measure as skm
from typing import Optional, Tuple

from ..blocks import block
from ..primitives.runtime import device as default_device
from ..primitives.gridmath import k_box_blur_3d  # reuse the G1 box blur


# [BLK M5.1]
@wp.kernel
def k_density_scatter(
    pos: wp.array(dtype=wp.vec3),
    density: wp.array3d(dtype=float),
    dx: float,
    nx: int, ny: int, nz: int,
):
    """Trilinear scatter particle 'mass' = 1 into a cell-centred density grid."""
    pid = wp.tid()
    p = pos[pid]
    fx = p[0] / dx - 0.5
    fy = p[1] / dx - 0.5
    fz = p[2] / dx - 0.5
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    sx = fx - float(i0); sy = fy - float(j0); sz = fz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if 0 <= ii and ii < nx and 0 <= jj and jj < ny and 0 <= kk and kk < nz:
                    wxi = float(0.0); wyi = float(0.0); wzi = float(0.0)
                    if di == 0: wxi = 1.0 - sx
                    else:       wxi = sx
                    if dj == 0: wyi = 1.0 - sy
                    else:       wyi = sy
                    if dk == 0: wzi = 1.0 - sz
                    else:       wzi = sz
                    wp.atomic_add(density, ii, jj, kk, wxi * wyi * wzi)


block("M5.1", "Particle density scatter (trilinear, atomic)")(k_density_scatter)


# [BLK M5.9.1] Wider cubic kernel scatter.
# Discovered in B13.8 evaluation: replacing M5.1's trilinear delta-footprint
# with a cubic kernel of fixed radius reduces mesh fragment count 29x on
# real demo30 data. See DESIGN.md §8.3 for the full algorithm + rationale.
# Per-particle thread; each iterates an AABB of (2R+1)^3 cells and
# atomic_add's the cubic-falloff weight where r^2 < 1.
@wp.kernel
def k_cubic_scatter(
    pos: wp.array(dtype=wp.vec3),
    density: wp.array3d(dtype=float),
    dx: float,
    inv_r_world_sq: float,   # 1/(radius_cells*dx)^2 — kernel falloff scaler
    half_cells: int,         # ceil(radius_cells) + 1 — AABB half-extent
    nx: int, ny: int, nz: int,
):
    pid = wp.tid()
    p = pos[pid]
    cx = int(wp.floor(p[0] / dx - 0.5))
    cy = int(wp.floor(p[1] / dx - 0.5))
    cz = int(wp.floor(p[2] / dx - 0.5))
    i_lo = wp.max(0, cx - half_cells); i_hi = wp.min(nx, cx + half_cells + 1)
    j_lo = wp.max(0, cy - half_cells); j_hi = wp.min(ny, cy + half_cells + 1)
    k_lo = wp.max(0, cz - half_cells); k_hi = wp.min(nz, cz + half_cells + 1)
    for i in range(i_lo, i_hi):
        cw_x = (float(i) + 0.5) * dx
        ddx = cw_x - p[0]
        for j in range(j_lo, j_hi):
            cw_y = (float(j) + 0.5) * dx
            ddy = cw_y - p[1]
            for k in range(k_lo, k_hi):
                cw_z = (float(k) + 0.5) * dx
                ddz = cw_z - p[2]
                d_sq = ddx * ddx + ddy * ddy + ddz * ddz
                r_sq = d_sq * inv_r_world_sq
                if r_sq < 1.0:
                    one_minus = 1.0 - r_sq
                    val = one_minus * one_minus * one_minus
                    wp.atomic_add(density, i, j, k, val)


block("M5.9.1", "Wider cubic kernel particle scatter (per-particle AABB iter)")(k_cubic_scatter)


# [BLK M5.11.2] Bridson SDF compute.
# Per-cell parallel: each thread queries the HashGrid for the nearest
# particle within `search_radius`, then writes phi = nearest_dist -
# particle_radius. Cells with no particle in range get phi = +search_radius
# (positive sentinel = "definitely outside"). MC at level 0 then extracts
# the union of particle-spheres — smooth by construction. See
# DESIGN.md §8.5 for algorithm and rationale.
_HG = wp.uint64


@wp.kernel
def k_sdf_field(
    grid: _HG,
    pos: wp.array(dtype=wp.vec3),
    phi: wp.array3d(dtype=float),
    dx: float,
    nx: int, ny: int, nz: int,
    search_radius: float,
    particle_radius: float,
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    cell = wp.vec3((float(i) + 0.5) * dx,
                   (float(j) + 0.5) * dx,
                   (float(k) + 0.5) * dx)
    min_d_sq = search_radius * search_radius
    for pid in wp.hash_grid_query(grid, cell, search_radius):
        d = cell - pos[pid]
        d_sq = wp.dot(d, d)
        if d_sq < min_d_sq:
            min_d_sq = d_sq
    phi[i, j, k] = wp.sqrt(min_d_sq) - particle_radius


block("M5.11.2", "Per-cell SDF compute kernel (HashGrid nearest-particle)")(k_sdf_field)


# [BLK M5.11.4] Per-vertex colour KNN blend (B18.4).
@wp.kernel
def k_vertex_color_knn(
    grid: _HG,
    pos: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    verts: wp.array(dtype=wp.vec3),
    out_color: wp.array(dtype=wp.vec3),
    search_radius: float,
):
    """Inverse-distance² weighted colour blend per vertex.

    Each vertex queries the SDF HashGrid for particles within
    ``search_radius`` and accumulates ``colors[pid] / (d² + eps)``.
    The eps avoids division blow-up when a vertex lands exactly on a
    particle (which the Bridson SDF would, at the particle-sphere
    surface for tiny ``particle_radius``).

    Linear-RGB blend. Mixbox pigment-space mixing (B18.5) is done on the
    CPU after this kernel: the GPU pass produces a "neutral" linear
    average that the Mixbox host-side path can then re-mix pairwise.
    """
    v = wp.tid()
    p = verts[v]
    w_sum = float(0.0)
    c_sum = wp.vec3(0.0, 0.0, 0.0)
    eps = float(1.0e-8)
    for pid in wp.hash_grid_query(grid, p, search_radius):
        d = p - pos[pid]
        d_sq = wp.dot(d, d)
        w = 1.0 / (d_sq + eps)
        c_sum = c_sum + w * colors[pid]
        w_sum = w_sum + w
    if w_sum > 0.0:
        out_color[v] = c_sum / w_sum
    else:
        out_color[v] = wp.vec3(1.0, 1.0, 1.0)


block("M5.11.4",
      "Per-vertex KNN inverse-distance² colour blend (B18.4)")(k_vertex_color_knn)


@wp.kernel
def _k_zero3(a: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    if i < a.shape[0] and j < a.shape[1] and k < a.shape[2]:
        a[i, j, k] = 0.0


@wp.kernel
def k_mc_zero_walls(
    field: wp.array3d(dtype=float),
    margin: int,
):
    """[BLK M5.7] GPU wall-mask: zero density within `margin` cells of any
    domain face. Replaces the host-side numpy slice writes so the M5.4
    GPU MC path can run without an intermediate D→H copy."""
    i, j, k = wp.tid()
    nx = field.shape[0]; ny = field.shape[1]; nz = field.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    if (i < margin or i >= nx - margin
        or j < margin or j >= ny - margin
        or k < margin or k >= nz - margin):
        field[i, j, k] = 0.0


block("M5.7", "GPU wall-mask kernel (zero density within margin cells of walls)")(k_mc_zero_walls)


class MeshExtractor:
    """[BLK M5] Reusable particle→mesh extractor.

    Holds two density buffers across calls so per-frame extraction does
    not pay re-allocation cost. Thread-unsafe (single-stream usage).
    """

    def __init__(self, nx: int, ny: int, nz: int, dx: float, device: Optional[str] = None,
                 use_gpu_mc: Optional[bool] = None):
        self.nx, self.ny, self.nz, self.dx = nx, ny, nz, dx
        self.device = device or default_device()
        self.dens = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        self.dens_tmp = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        # M5.4 GPU MC: auto-engage at ≥64³ grids (CPU skimage faster below).
        if use_gpu_mc is None:
            use_gpu_mc = (max(nx, ny, nz) >= 64)
        self.use_gpu_mc = bool(use_gpu_mc)
        self._mc = None      # lazy wp.MarchingCubes
        self._mc_max_verts = 0
        self._mc_max_tris = 0
        # M5.11.1 — lazy HashGrid for SDF mesh_method. 64³ matches W7.7's
        # tested setup; covers the unit-cube sim space with cells ~1.6 cm
        # at dx=5.2 mm (192³ grid), which gives 1-2 cell traversal per
        # SDF query at typical search_radius = 4 cells = 2.1 cm.
        self._sdf_grid = None
        self._sdf_grid_cells = 64

    @block("M5.11.4.H",
           "Host wrapper: per-vertex linear-RGB colour from particle attrs (B18.4)")
    def compute_vertex_colors(
        self,
        verts_world: np.ndarray,
        pos: "wp.array",
        colors: "wp.array",
        search_radius_world: float,
    ) -> np.ndarray:
        """Per-vertex colour via inverse-distance² KNN blend against particles.

        Requires the SDF HashGrid to already be built (call after
        :meth:`extract` with ``mesh_method='sdf'``). For ``trilinear`` /
        ``cubic`` meshers, this builds a HashGrid on the fly.

        Parameters
        ----------
        verts_world : (N, 3) float32 — mesh vertices in world coordinates.
        pos : ``wp.array(dtype=vec3)`` — particle positions (same that
            went into :meth:`extract`).
        colors : ``wp.array(dtype=vec3)`` — particle RGB in [0, 1].
        search_radius_world : neighbourhood for KNN — typically the same
            as the SDF ``search_radius`` to keep cache-coherent queries.

        Returns
        -------
        ``(N, 3) uint8`` — RGB ready for the PLY writer.
        """
        if self._sdf_grid is None:
            self._build_sdf_hashgrid(pos, search_radius_world)
        n_v = verts_world.shape[0]
        verts_wp = wp.array(verts_world, dtype=wp.vec3, device=self.device)
        out = wp.zeros(n_v, dtype=wp.vec3, device=self.device)
        wp.launch(
            k_vertex_color_knn, dim=n_v,
            inputs=[self._sdf_grid.id, pos, colors, verts_wp, out,
                    float(search_radius_world)],
            device=self.device,
        )
        cols = out.numpy().astype(np.float32)
        # Clamp & convert to uint8. Slight headroom on the high end avoids
        # banding from float→int truncation: round, then clip.
        cols_u8 = np.clip(np.round(cols * 255.0), 0, 255).astype(np.uint8)
        return cols_u8

    @block("M5.11.1", "Lazy wp.HashGrid build for SDF nearest-particle queries")
    def _build_sdf_hashgrid(self, pos, search_r_world: float):
        if self._sdf_grid is None:
            self._sdf_grid = wp.HashGrid(
                self._sdf_grid_cells, self._sdf_grid_cells,
                self._sdf_grid_cells, device=self.device)
        self._sdf_grid.build(pos, search_r_world)
        return self._sdf_grid

    # [BLK M5.2]
    @block("M5.2", "Density grid box-blur (N passes)")
    def _smooth(self, passes: int):
        a, b = self.dens, self.dens_tmp
        for _ in range(passes):
            wp.launch(k_box_blur_3d, dim=(self.nx, self.ny, self.nz),
                      inputs=[a, b], device=self.device)
            a, b = b, a
        # leave the final result in self.dens
        self.dens, self.dens_tmp = a, b

    def _mc_extract_cpu(self, iso_level: float, sdf: bool = False):
        """[BLK M5.3] skimage marching_cubes — D→H copy of density first.

        ``sdf=True`` flips the early-out: SDF cells with no particle in
        range hold +search_radius (positive sentinel); the surface exists
        iff at least one cell has phi < 0.
        """
        wp.synchronize()
        d = self.dens.numpy()
        if sdf:
            if d.min() >= 0.0:
                return None, None
        else:
            if d.max() < iso_level * 0.5:
                return None, None
        try:
            verts, faces, _, _ = skm.marching_cubes(
                d, level=iso_level, spacing=(self.dx, self.dx, self.dx))
        except (ValueError, RuntimeError):
            return None, None
        # audit-2026-06-14 #12: cell-centred grid -> world = (g+0.5)*dx. skimage
        # `spacing` gives g*dx, so add the half-cell to match the GPU path and
        # the cell-centred obstacle SDFs / raw particles.
        return (verts.astype(np.float32) + 0.5 * self.dx,
                faces.astype(np.int32))

    @block("M5.4", "GPU marching cubes via wp.MarchingCubes (device-resident)")
    def _mc_extract_gpu(self, iso_level: float):
        """Run wp.MarchingCubes on the device density. Returns
        ``(verts_world, faces)`` rescaled to world coordinates.

        Lazy-allocates the MC context on first call with a generous
        buffer budget; ``resize()`` is called if a frame would exceed
        the budget (rare for stable scenes, fast when it happens)."""
        nx, ny, nz = self.nx, self.ny, self.nz
        if self._mc is None:
            # Heuristic budget: O(nx·ny·nz / 100) verts is plenty for fluid
            # surfaces; minimum 4096 so small grids don't trip the cap.
            cap_v = max(4096, (nx * ny * nz) // 100)
            cap_t = cap_v * 2
            self._mc = wp.MarchingCubes(nx=nx, ny=ny, nz=nz,
                                        max_verts=cap_v, max_tris=cap_t,
                                        device=self.device)
            self._mc_max_verts = cap_v
            self._mc_max_tris = cap_t
        # surface() may throw if the buffer is too small — bump and retry until
        # it fits. audit-2026-06-14r6 #3: the resize MUST be followed by another
        # surface() attempt, and final exhaustion MUST raise. The old range(2)
        # capped this at two surface() calls, so a frame needing >2x the current
        # budget left the buffer resized but surface() never re-run — and the
        # code below then read self._mc.verts, which a throwing surface() leaves
        # holding the PREVIOUS frame's mesh, silently emitting a stale surface.
        for _attempt in range(6):
            try:
                self._mc.surface(self.dens, threshold=float(iso_level))
                break
            except Exception:
                self._mc_max_verts *= 2
                self._mc_max_tris *= 2
                self._mc.resize(max_verts=self._mc_max_verts,
                                max_tris=self._mc_max_tris)
        else:
            # Exhausted every retry — fail loud rather than emit a stale mesh.
            raise RuntimeError(
                f"marching cubes buffer exhausted after retries "
                f"(max_verts={self._mc_max_verts}, max_tris={self._mc_max_tris}); "
                f"the iso-surface did not fit — scene density may have diverged")
        n_v = int(self._mc.verts.shape[0])
        if n_v == 0:
            return None, None
        verts = self._mc.verts.numpy().astype(np.float32)
        # Warp MC returns vertices in *grid-index* coordinates; scale to world.
        # audit-2026-06-14 #12: the density/SDF grid is CELL-CENTRED — the
        # scatter kernels place a particle at world p into sample index
        # i = p/dx - 0.5 (so index i ↔ world (i+0.5)*dx, and the SDF evaluates
        # phi at (i+0.5)*dx). MC verts are in grid-index space, so the world
        # map is (g+0.5)*dx, NOT g*dx — the old g*dx shifted the fluid mesh by
        # -0.5*dx relative to the cell-centred obstacle SDFs and the raw
        # particles. Verified by particle-cloud vs mesh-bbox centre comparison.
        verts = (verts + 0.5) * self.dx
        faces = self._mc.indices.numpy().reshape(-1, 3).astype(np.int32)
        return verts, faces

    # [BLK M5.3]
    @block("M5.3", "MC + smoothing + wall mask + optional decimation")
    def extract(
        self,
        pos: wp.array,
        iso_level: float = 0.5,
        smooth_passes: int = 2,
        mesh_smooth_passes: int = 0,
        mesh_smooth_method: str = "taubin",
        wall_margin_cells: int = 0,
        decimate_ratio: float = 1.0,
        mesh_method: str = "trilinear",
        cubic_radius_cells: float = 2.0,
        sdf_particle_radius_cells: float = 1.0,
        sdf_search_radius_cells: float = 4.0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run density scatter + box-blur + marching cubes, then (optionally)
        post-smooth the mesh.

        Parameters
        ----------
        pos : wp.array of vec3, particle positions in world units
        iso_level : MC iso-value on the smoothed density grid (typical 0.3..1.0)
        smooth_passes : box-blur passes on the *density grid* before MC
        mesh_smooth_passes : Taubin/Laplacian passes on the *output mesh*
        mesh_smooth_method : "taubin" (no shrinkage) or "laplacian" (cheaper, shrinks)

        Returns
        -------
        (verts, faces) or (None, None) when no surface emerges.
        """
        # Round-36: empty input guard. Pre-round-36 a zero-particle
        # extract (every particle exited an outflow in one frame, or
        # MPM diverged pre-empting all particles, or solver state was
        # reset) hit two failure modes: (a) wp.HashGrid.build on an
        # empty pos array raises on some Warp versions; (b) CPU MC
        # path's `self.dens.numpy()` returned the PREVIOUS frame's
        # data (dens never cleared) so the prior frame's surface
        # ghost-froze into all subsequent empty frames silently. Now:
        # short-circuit cleanly with (None, None).
        n_particles = int(pos.shape[0]) if pos is not None else 0
        if n_particles == 0:
            return None, None
        wp.launch(_k_zero3, dim=(self.nx, self.ny, self.nz), inputs=[self.dens], device=self.device)
        if mesh_method == "trilinear":
            wp.launch(k_density_scatter, dim=pos.shape[0],
                      inputs=[pos, self.dens, self.dx, self.nx, self.ny, self.nz],
                      device=self.device)
        elif mesh_method == "cubic":
            # M5.9: wider cubic kernel. iso_level needs to be ~5-10x higher
            # than M5.1 values because cubic kernel concentrates more mass
            # at peak — see DESIGN.md §8.3 migration note.
            r_world = float(cubic_radius_cells) * self.dx
            inv_r_world_sq = 1.0 / (r_world * r_world)
            half_cells = int(np.ceil(cubic_radius_cells)) + 1
            wp.launch(k_cubic_scatter, dim=pos.shape[0],
                      inputs=[pos, self.dens, self.dx, inv_r_world_sq,
                              half_cells, self.nx, self.ny, self.nz],
                      device=self.device)
        elif mesh_method == "sdf":
            # M5.11: Bridson SDF. phi(cell) = dist(cell, nearest_particle) -
            # particle_radius_world. MC threshold is forced to 0 below.
            search_r_world = float(sdf_search_radius_cells) * self.dx
            particle_r_world = float(sdf_particle_radius_cells) * self.dx
            # Seed self.dens (= phi storage) to +search_r_world so empty
            # cells outside any particle's reach naturally read "outside".
            # The kernel only writes cells whose hash-grid query finds at
            # least one particle, so the sentinel covers the rest.
            self.dens.fill_(search_r_world)
            self._build_sdf_hashgrid(pos, search_r_world)
            wp.launch(k_sdf_field,
                      dim=(self.nx, self.ny, self.nz),
                      inputs=[self._sdf_grid.id, pos, self.dens, self.dx,
                              self.nx, self.ny, self.nz,
                              search_r_world, particle_r_world],
                      device=self.device)
        else:
            raise ValueError(
                f"unknown mesh_method: {mesh_method!r}; "
                f"expected 'trilinear', 'cubic', or 'sdf'"
            )
        self._smooth(smooth_passes)
        # Wall mask on GPU (M5.7) — keeps the field device-resident so the
        # M5.4 GPU MC path doesn't pay an extra D→H roundtrip.
        # Skip for SDF: zeroing phi at the wall would create a phi=0 contour
        # there, i.e. a fake surface. Empty cells already read +search_radius
        # (positive = outside), which is what we want.
        if wall_margin_cells > 0 and mesh_method != "sdf":
            wp.launch(k_mc_zero_walls, dim=(self.nx, self.ny, self.nz),
                      inputs=[self.dens, int(wall_margin_cells)], device=self.device)
        # SDF semantics force MC threshold = 0 (the zero level set). User's
        # iso_level is ignored — DESIGN.md §8.5 migration note.
        mc_level = 0.0 if mesh_method == "sdf" else float(iso_level)
        if self.use_gpu_mc:
            verts, faces = self._mc_extract_gpu(mc_level)
        else:
            verts, faces = self._mc_extract_cpu(mc_level, sdf=(mesh_method == "sdf"))
        if verts is None:
            return None, None
        if mesh_smooth_passes > 0 and len(verts) > 0:
            from .smoothing import smooth_taubin, smooth_laplacian
            if mesh_smooth_method == "taubin":
                verts = smooth_taubin(verts, faces, passes=mesh_smooth_passes)
            elif mesh_smooth_method == "laplacian":
                verts = smooth_laplacian(verts, faces, passes=mesh_smooth_passes)
            else:
                raise ValueError(f"unknown mesh_smooth_method: {mesh_smooth_method}")
        if decimate_ratio < 1.0 and len(faces) > 4:
            from .decimate import decimate_mesh
            verts, faces = decimate_mesh(verts, faces, target_ratio=decimate_ratio)
        return verts, faces


# Convenience function (back-compat with previous flat API).
def particles_to_mesh(pos, nx, ny, nz, dx, iso_level=0.5, smooth_passes=2, **_ignored):
    ex = MeshExtractor(nx, ny, nz, dx)
    return ex.extract(pos, iso_level=iso_level, smooth_passes=smooth_passes)


# M5.11.3 — SDF smoothing reuses M5.2 box-blur unchanged on the phi field
# (smoothing a signed-distance field yields a smoothed SDF, Bridson §5.4).
# Registered as a separate block ID so BLOCKS.md ↔ registry stays in sync;
# the underlying callable is `_smooth` which is also registered as M5.2.
block("M5.11.3", "SDF smoothing — reuses M5.2 box-blur on the phi field")(
    MeshExtractor._smooth
)
# M5.9.H, M5.11.H — host wrappers are branches inside `MeshExtractor.extract()`
# (itself decorated as M5.3). One callable, multiple block IDs so the registry
# can locate each mesh_method opt-in.
block("M5.9.H", "MeshExtractor mesh_method='cubic' host wrapper")(
    MeshExtractor.extract
)
block("M5.11.H", "MeshExtractor mesh_method='sdf' host wrapper")(
    MeshExtractor.extract
)
