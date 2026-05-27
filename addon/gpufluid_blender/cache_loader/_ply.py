"""[BLK A8.6] Minimal binary-PLY reader for the addon cache loader.

Phase 4 split: extracted from ``cache_loader.py`` so the package stays
under the 500-line cap. Matches the format written by
``gpufluid.io.ply._write_ply_binary`` (verts as 3×float32, optional
``red/green/blue uchar`` per-vertex colour, faces as 1×uint8 count +
3×int32 indices, 13 bytes per triangle).
"""
from __future__ import annotations

import numpy as np


def _read_ply_minimal(path):
    """Vectorised binary-PLY reader matching gpufluid.io.ply format.

    Returns ``(verts, faces, colors_or_None)``. ``colors`` is a ``(N, 3)``
    uint8 array when the PLY has ``red/green/blue uchar`` properties
    (B18.4) — otherwise ``None``. Pre-B18 cache files (xyz-only verts)
    keep working unchanged.
    """
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), None)
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            if not line:
                return empty
            header += line
            if line.strip() == b"end_header":
                break
        text = header.decode("ascii", errors="replace")
        # Round-25: enforce binary_little_endian — pre-round-25 an
        # ASCII PLY (hand-edit or MeshLab export) was parsed as if
        # binary, yielding garbage floats reinterpreted from ASCII
        # bytes (scrambled mesh, no error). gpufluid.io.ply.read_ply
        # already had this guard; addon's parallel reader didn't.
        if "format binary_little_endian" not in text:
            raise ValueError(
                f"PLY {path}: only binary_little_endian supported "
                f"(got header without that format line)")
        n_v = 0
        n_f = 0
        has_color = False
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith("element vertex"):
                n_v = int(s.split()[2])
            elif s.startswith("element face"):
                n_f = int(s.split()[2])
            elif s == "property uchar red":
                has_color = True
        if has_color:
            vbuf = f.read(n_v * 15)
            arr = np.frombuffer(vbuf, dtype=np.uint8).reshape(n_v, 15)
            verts = (np.ascontiguousarray(arr[:, :12])
                       .view(np.float32).reshape(n_v, 3).copy())
            colors = arr[:, 12:].copy()
        else:
            verts = np.frombuffer(
                f.read(n_v * 12), dtype=np.float32).reshape(n_v, 3).copy()
            colors = None
        if n_f == 0:
            return verts, np.zeros((0, 3), np.int32), colors
        face_bytes = f.read(n_f * 13)
        rows = np.frombuffer(face_bytes, dtype=np.uint8).reshape(n_f, 13)
        faces = (np.ascontiguousarray(rows[:, 1:])
                   .view(np.int32).reshape(n_f, 3).copy())
        return verts, faces, colors
