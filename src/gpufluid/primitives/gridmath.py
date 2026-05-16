"""[Layer G1] Grid index math: clamp, trilinear weights/sample/scatter, box-blur.

These are pure GPU helpers — Warp `@wp.func` for in-kernel reuse and
`@wp.kernel` for stand-alone launches. Higher layers compose them.
"""
from __future__ import annotations
import warp as wp
from ..blocks import block


# [BLK G1.3]
@wp.func
def clamp_int(v: int, lo: int, hi: int) -> int:
    """Integer clamp [lo, hi]. Used at grid-boundary indexing."""
    if v < lo: return lo
    if v > hi: return hi
    return v


# Register the @wp.func too (decorator is no-op at runtime; for docs/check).
block("G1.3", "clamp_int integer clamp helper")(clamp_int)


# [BLK G1.4]  (Trilinear corner weights computed inline in scatter/sample
#              for hot-path inlining; documented here for completeness.)


# [BLK G1.5]
@wp.func
def sample3(
    f: wp.array3d(dtype=float),
    fx: float, fy: float, fz: float,
    nx_max: int, ny_max: int, nz_max: int,
) -> float:
    """Trilinear sample of a 3D field at (fx,fy,fz) in grid coords."""
    fx = wp.clamp(fx, 0.0, float(nx_max))
    fy = wp.clamp(fy, 0.0, float(ny_max))
    fz = wp.clamp(fz, 0.0, float(nz_max))
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    if i0 >= nx_max: i0 = nx_max - 1
    if j0 >= ny_max: j0 = ny_max - 1
    if k0 >= nz_max: k0 = nz_max - 1
    if i0 < 0: i0 = 0
    if j0 < 0: j0 = 0
    if k0 < 0: k0 = 0
    i1 = i0 + 1; j1 = j0 + 1; k1 = k0 + 1
    if i1 >= nx_max: i1 = nx_max - 1
    if j1 >= ny_max: j1 = ny_max - 1
    if k1 >= nz_max: k1 = nz_max - 1
    sx = wp.clamp(fx - float(i0), 0.0, 1.0)
    sy = wp.clamp(fy - float(j0), 0.0, 1.0)
    sz = wp.clamp(fz - float(k0), 0.0, 1.0)
    c000 = f[i0, j0, k0]; c100 = f[i1, j0, k0]
    c010 = f[i0, j1, k0]; c110 = f[i1, j1, k0]
    c001 = f[i0, j0, k1]; c101 = f[i1, j0, k1]
    c011 = f[i0, j1, k1]; c111 = f[i1, j1, k1]
    c00 = c000 * (1.0 - sx) + c100 * sx
    c10 = c010 * (1.0 - sx) + c110 * sx
    c01 = c001 * (1.0 - sx) + c101 * sx
    c11 = c011 * (1.0 - sx) + c111 * sx
    c0 = c00 * (1.0 - sy) + c10 * sy
    c1 = c01 * (1.0 - sy) + c11 * sy
    return c0 * (1.0 - sz) + c1 * sz


block("G1.5", "Trilinear sample of 3D field")(sample3)


# [BLK G1.6]
@wp.func
def scatter_face(
    field: wp.array3d(dtype=float),
    weight: wp.array3d(dtype=float),
    val: float,
    fx: float, fy: float, fz: float,
    nx: int, ny: int, nz: int,
):
    """Trilinear scatter of `val` into 8 corners of (fx,fy,fz) using atomic_add."""
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    sx = fx - float(i0); sy = fy - float(j0); sz = fz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if ii >= 0 and ii < nx and jj >= 0 and jj < ny and kk >= 0 and kk < nz:
                    wxi = float(0.0); wyi = float(0.0); wzi = float(0.0)
                    if di == 0: wxi = 1.0 - sx
                    else:       wxi = sx
                    if dj == 0: wyi = 1.0 - sy
                    else:       wyi = sy
                    if dk == 0: wzi = 1.0 - sz
                    else:       wzi = sz
                    w = wxi * wyi * wzi
                    wp.atomic_add(field, ii, jj, kk, val * w)
                    wp.atomic_add(weight, ii, jj, kk, w)


block("G1.6", "Trilinear scatter with atomic_add into 8 corners")(scatter_face)


# [BLK G1.7]
@wp.kernel
def k_box_blur_3d(
    src: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
):
    """3x3x3 box-blur of src → dst (boundary-aware: divides by count of valid voxels)."""
    i, j, k = wp.tid()
    nx = src.shape[0]; ny = src.shape[1]; nz = src.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    s = float(0.0)
    n = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                ii = i + di; jj = j + dj; kk = k + dk
                if 0 <= ii and ii < nx and 0 <= jj and jj < ny and 0 <= kk and kk < nz:
                    s += src[ii, jj, kk]
                    n += 1.0
    dst[i, j, k] = s / n


block("G1.7", "3D box-blur kernel (3^3)")(k_box_blur_3d)
