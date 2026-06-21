"""[Layer I6] Binary PLY mesh writer (Blender-importable)."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Union
from ..blocks import block, BlockError


# [BLK I6.1]
@block("I6.1", "Binary little-endian PLY writer (optional per-vertex RGB)")
def write_ply(
    path: Union[str, Path],
    verts: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None = None,
) -> None:
    """Write a triangle mesh to a binary PLY file.

    Parameters
    ----------
    path : path-like
    verts : (N, 3) float32-castable
    faces : (M, 3) int32-castable
    vertex_colors : (N, 3) uint8-castable, optional (B18.4)
        When given, the file gains ``property uchar red/green/blue`` after
        the xyz vertex properties; Blender's PLY importer reads these as
        a ``Col`` vertex colour layer that any shader graph can hook into.

    Raises
    ------
    BlockError [I6.1] if vertex or face shapes are not (N,3)/(M,3), or if
    ``vertex_colors`` is given but its row count does not match ``verts``.
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise BlockError("I6.1", f"verts must be (N,3); got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise BlockError("I6.1", f"faces must be (M,3); got {faces.shape}")
    n_v, n_f = len(verts), len(faces)

    has_color = vertex_colors is not None
    if has_color:
        vc = np.asarray(vertex_colors)
        if vc.ndim != 2 or vc.shape != (n_v, 3):
            raise BlockError(
                "I6.1",
                f"vertex_colors must be (N={n_v}, 3); got {vc.shape}",
            )
        vc = np.clip(vc, 0, 255).astype(np.uint8, copy=False)
    color_header = (
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        if has_color else ""
    )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"{color_header}"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if has_color:
            # Interleaved xyz (12 bytes) + rgb (3 bytes) per vertex.
            # Build via a structured-dtype view so we avoid a Python loop.
            buf = np.empty(n_v, dtype=[("xyz", np.float32, 3),
                                       ("rgb", np.uint8, 3)])
            buf["xyz"] = verts
            buf["rgb"] = vc
            f.write(buf.tobytes())
        else:
            f.write(verts.tobytes())
        face_buf = bytearray(n_f * (1 + 3 * 4))
        offset = 0
        for tri in faces:
            face_buf[offset] = 3
            offset += 1
            face_buf[offset:offset + 12] = tri.astype(np.int32).tobytes()
            offset += 12
        f.write(bytes(face_buf))


# [BLK I6.1]
@block("I6.1", "Binary little-endian PLY reader (mirror of write_ply)")
def read_ply(path: Union[str, Path], return_colors: bool = False):
    """Read a triangle mesh written by :func:`write_ply`.

    Returns
    -------
    (verts, faces) by default. When ``return_colors=True``, returns
    ``(verts, faces, colors_or_None)`` where ``colors`` is ``(N, 3)``
    uint8 if the file has ``red/green/blue uchar`` vertex properties
    (B18.4), or ``None`` otherwise.

    Notes
    -----
    Parser handles the two layouts emitted by :func:`write_ply`:
    ``xyz`` (12 B/vertex) and ``xyz+rgb`` (15 B/vertex). Other vertex
    property combinations are rejected — generic PLYs need trimesh.
    """
    path = Path(path)
    with open(path, "rb") as f:
        # parse ASCII header line-by-line until "end_header"
        header_bytes = b""
        while True:
            line = f.readline()
            if not line:
                raise BlockError("I6.1", f"truncated PLY header: {path}")
            header_bytes += line
            if line.strip() == b"end_header":
                break
        header = header_bytes.decode("ascii", errors="replace")
        if "format binary_little_endian" not in header:
            raise BlockError("I6.1", f"unsupported PLY format in {path}")
        n_v = 0; n_f = 0
        # audit-2026-06-14r2 #4: collect the vertex element's property names in
        # order and validate the layout, instead of only sniffing for `red`.
        # An xyz+normals (or any other) layout was silently read as 12 B/vertex,
        # reinterpreting normal bytes as coordinates — contradicting the
        # docstring's "other combinations are rejected" claim.
        vertex_props = []
        cur_element = None
        for line in header.splitlines():
            ls = line.strip()
            if ls.startswith("element vertex"):
                n_v = int(ls.split()[2]); cur_element = "vertex"
            elif ls.startswith("element face"):
                n_f = int(ls.split()[2]); cur_element = "face"
            elif ls.startswith("element"):
                cur_element = "other"
            elif ls.startswith("property") and cur_element == "vertex":
                vertex_props.append(ls.split()[-1])  # the property name
        if vertex_props == ["x", "y", "z"]:
            has_color = False
        elif vertex_props == ["x", "y", "z", "red", "green", "blue"]:
            has_color = True
        else:
            raise BlockError(
                "I6.1", f"unsupported PLY vertex layout {vertex_props} in {path}; "
                "read_ply parses only xyz or xyz+rgb(uchar) — use trimesh for "
                "generic PLYs")
        if has_color:
            # 12 B xyz + 3 B rgb per vertex, interleaved
            vbuf = f.read(n_v * 15)
            arr = np.frombuffer(vbuf, dtype=np.uint8).reshape(n_v, 15)
            verts = np.ascontiguousarray(arr[:, :12]).view(np.float32).reshape(n_v, 3).copy()
            colors = arr[:, 12:].copy()
        else:
            verts = np.frombuffer(f.read(n_v * 12), dtype=np.float32).reshape(n_v, 3).copy()
            colors = None
        # [BLK I6.1.MESH] Vectorized face parse — each record is
        # uchar(=3) + 3 int32 (13 bytes total). Reinterpret as (N_f, 13)
        # uint8 → slice off count column → view as int32. Replaces a
        # per-face Python loop that was ~50ms on 10k faces; this is <1ms.
        face_bytes = f.read(n_f * 13)
        if n_f > 0:
            if len(face_bytes) < n_f * 13:
                raise BlockError("I6.1", f"truncated PLY face data in {path}")
            rows = np.frombuffer(face_bytes, dtype=np.uint8).reshape(n_f, 13)
            # audit-2026-06-14r2 #1: every face record MUST be a triangle
            # (count byte == 3). A quad/n-gon is 17+ bytes; reading it as 13
            # misaligns the byte stream from that face on and yields scrambled
            # indices with NO error. The addon reader (cache_loader/_ply.py) got
            # this guard in round-32; the library reader had drifted (§9.6).
            if (rows[:, 0] != 3).any():
                raise BlockError(
                    "I6.1", f"non-triangle face in PLY {path}; cache PLYs must "
                    "be triangulated (read_ply parses fixed 3-index records)")
            faces = np.ascontiguousarray(rows[:, 1:]).view(np.int32).reshape(n_f, 3).copy()
        else:
            faces = np.empty((0, 3), dtype=np.int32)
    if return_colors:
        return verts, faces, colors
    return verts, faces


# [BLK I6.1.MESH] companion to I6.1 for points-only PLYs (no faces)
@block("I6.1.MESH", "Binary PLY reader/writer — points-only and active-filtered variants for solver caches")
def read_points_ply(path: Union[str, Path]) -> np.ndarray:
    """Read positions from a points-only PLY (no faces). Returns (N, 3) float32.

    Format expected: binary_little_endian, `property float x/y/z` only.
    Use :func:`read_ply` for mesh PLYs (with `element face`).
    """
    path = Path(path)
    with open(path, "rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            if not line:
                raise BlockError("I6.1.MESH", f"truncated PLY header: {path}")
            header_bytes += line
            if line.strip() == b"end_header":
                break
        header = header_bytes.decode("ascii", errors="replace")
        if "format binary_little_endian" not in header:
            raise BlockError("I6.1.MESH", f"unsupported PLY format in {path}")
        # 2026-06-21 (reviewer-io, §9.6 mirror of read_ply): validate the vertex
        # layout. Pre-fix this read n_v*12 bytes assuming pure xyz; an xyz+rgb
        # PLY (15 B/vertex) was silently misread as float coords (garbage), no
        # error. Reject anything that isn't exactly 3 float props (x,y,z) and
        # point the caller at read_ply (the writer only ever emits pure xyz, so
        # the normal points pipeline is unaffected).
        n_v = 0
        in_vertex = False
        vert_props = []
        for line in header.splitlines():
            toks = line.split()
            if not toks:
                continue
            if toks[0] == "element":
                in_vertex = len(toks) >= 2 and toks[1] == "vertex"
                if in_vertex:
                    n_v = int(toks[2])
            elif toks[0] == "property" and in_vertex:
                vert_props.append(tuple(toks[1:]))
        names = [p[-1] for p in vert_props]
        types = [p[0] for p in vert_props]
        if names != ["x", "y", "z"] or any(t not in ("float", "float32")
                                            for t in types):
            raise BlockError(
                "I6.1.MESH",
                f"read_points_ply expects exactly 3 float vertex properties "
                f"(x,y,z); got {vert_props} in {path}. Use read_ply for mesh / "
                f"coloured PLYs (this guards against silently reading rgb bytes "
                f"as coordinates).")
        if n_v == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.frombuffer(f.read(n_v * 12), dtype=np.float32).reshape(n_v, 3).copy()


def write_points_ply(
    path: Union[str, Path],
    positions: np.ndarray,
    selection_mask: np.ndarray | None = None,
) -> int:
    """Write a points-only PLY. Optionally filter by `selection_mask`.

    [BLK S2.17.6] When `selection_mask` is given, only rows where
    `selection_mask == 0` are written — this matches warp-mpm semantics
    where ``particle_selection==1`` means "skip this particle" (used for
    killed/inactive particles). Filtering at write-time prevents
    killed particles from polluting downstream meshing/rendering.

    Parameters
    ----------
    path : path-like
    positions : (N, 3) float32-castable
    selection_mask : (N,) int-castable, optional. If None, write all.

    Returns
    -------
    int : the number of points actually written.
    """
    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise BlockError("S2.17.6", f"positions must be (N,3); got {pos.shape}")
    if selection_mask is not None:
        mask = np.asarray(selection_mask).ravel() == 0
        if mask.shape[0] != pos.shape[0]:
            raise BlockError("S2.17.6",
                f"selection_mask shape {mask.shape} != positions {pos.shape[0]}")
        pos = pos[mask]
    n = len(pos)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pos.tobytes())
    return n


# [BLK I6.3]
@block("I6.3", "Particle dump (numpy .npy of positions)")
def write_particles_npy(path: Union[str, Path], positions: np.ndarray) -> None:
    """Save particle positions (Nx3 float32) to .npy."""
    arr = np.asarray(positions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise BlockError("I6.3", f"positions must be (N,3); got {arr.shape}")
    np.save(path, arr)
