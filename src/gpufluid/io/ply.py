"""[Layer I6] Binary PLY mesh writer (Blender-importable)."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Union
from ..blocks import block, BlockError


# [BLK I6.1]
@block("I6.1", "Binary little-endian PLY writer")
def write_ply(path: Union[str, Path], verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle mesh to a binary PLY file.

    Parameters
    ----------
    path : path-like
    verts : (N, 3) float32-castable
    faces : (M, 3) int32-castable

    Raises
    ------
    BlockError [I6.1] if vertex or face shapes are not (N,3)/(M,3).
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise BlockError("I6.1", f"verts must be (N,3); got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise BlockError("I6.1", f"faces must be (M,3); got {faces.shape}")
    n_v, n_f = len(verts), len(faces)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
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
def read_ply(path: Union[str, Path]):
    """Read a triangle mesh written by :func:`write_ply`.

    Returns
    -------
    (verts, faces) : (N,3) float32, (M,3) int32

    Notes
    -----
    Parser is intentionally narrow: it understands the exact format produced
    by ``write_ply`` (binary_little_endian, `xyz` float vertex, `vertex_indices`
    uchar+int face list). Generic PLY files may not parse — use trimesh for those.
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
        for line in header.splitlines():
            if line.startswith("element vertex"):
                n_v = int(line.split()[2])
            elif line.startswith("element face"):
                n_f = int(line.split()[2])
        verts = np.frombuffer(f.read(n_v * 12), dtype=np.float32).reshape(n_v, 3).copy()
        # faces: each record is uchar(=3) + 3 int32
        face_bytes = f.read(n_f * (1 + 12))
        faces = np.empty((n_f, 3), dtype=np.int32)
        for i in range(n_f):
            base = i * 13
            # skip the 1-byte count
            faces[i] = np.frombuffer(face_bytes[base + 1:base + 13], dtype=np.int32)
    return verts, faces


# [BLK I6.3]
@block("I6.3", "Particle dump (numpy .npy of positions)")
def write_particles_npy(path: Union[str, Path], positions: np.ndarray) -> None:
    """Save particle positions (Nx3 float32) to .npy."""
    arr = np.asarray(positions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise BlockError("I6.3", f"positions must be (N,3); got {arr.shape}")
    np.save(path, arr)
