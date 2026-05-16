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

    def _mc_extract_cpu(self, iso_level: float):
        """[BLK M5.3] skimage marching_cubes — D→H copy of density first."""
        wp.synchronize()
        d = self.dens.numpy()
        if d.max() < iso_level * 0.5:
            return None, None
        try:
            verts, faces, _, _ = skm.marching_cubes(
                d, level=iso_level, spacing=(self.dx, self.dx, self.dx))
        except (ValueError, RuntimeError):
            return None, None
        return verts.astype(np.float32), faces.astype(np.int32)

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
        # surface() may throw if the buffer is too small — bump and retry.
        for _attempt in range(2):
            try:
                self._mc.surface(self.dens, threshold=float(iso_level))
                break
            except Exception:
                self._mc_max_verts *= 2
                self._mc_max_tris *= 2
                self._mc.resize(max_verts=self._mc_max_verts,
                                max_tris=self._mc_max_tris)
        n_v = int(self._mc.verts.shape[0])
        if n_v == 0:
            return None, None
        verts = self._mc.verts.numpy().astype(np.float32)
        # Warp MC returns vertices in *grid-index* coordinates; scale to world.
        verts = verts * self.dx
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
        wp.launch(_k_zero3, dim=(self.nx, self.ny, self.nz), inputs=[self.dens], device=self.device)
        wp.launch(k_density_scatter, dim=pos.shape[0],
                  inputs=[pos, self.dens, self.dx, self.nx, self.ny, self.nz],
                  device=self.device)
        self._smooth(smooth_passes)
        # Wall mask on GPU (M5.7) — keeps the field device-resident so the
        # M5.4 GPU MC path doesn't pay an extra D→H roundtrip.
        if wall_margin_cells > 0:
            wp.launch(k_mc_zero_walls, dim=(self.nx, self.ny, self.nz),
                      inputs=[self.dens, int(wall_margin_cells)], device=self.device)
        if self.use_gpu_mc:
            verts, faces = self._mc_extract_gpu(iso_level)
        else:
            verts, faces = self._mc_extract_cpu(iso_level)
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
