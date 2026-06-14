"""[Layer I6 / BLK I6.5] USD time-sampled mesh writer.

Produces a single ``.usdc`` (binary USD) file containing one Mesh prim
with per-frame time-sampled vertices and faces. Blender 5.x's USD importer
turns this into a Mesh Sequence Cache modifier automatically, giving
proper motion blur and full animation playback in viewport without our
per-frame PLY swap handler.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

from ..blocks import block, BlockError


# [BLK I6.5]
@block("I6.5", "USD time-sampled mesh sequence writer")
def write_usd_mesh_sequence(
    path: Union[str, Path],
    frames: Iterable[Tuple[int, np.ndarray, np.ndarray]],
    *,
    fps: int = 24,
    prim_path: str = "/gpufluid/cache",
) -> Path:
    """Write a single USD prim with time-sampled mesh geometry.

    Parameters
    ----------
    path : output ``.usdc`` (binary) or ``.usda`` (ASCII) path.
    frames : iterable of ``(frame_index, verts, faces)`` triples.
             ``verts`` (N,3) float32, ``faces`` (M,3) int32. Empty frames
             allowed (pass ``np.zeros((0,3), float32)`` / ``int32``).
    fps : timecode rate stored as metadata. Blender uses scene fps when
          importing; this is mostly informational.
    prim_path : USD path of the mesh prim. Default ``/gpufluid/cache``.
    """
    from pxr import Usd, UsdGeom, Sdf, Vt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    parent_path = prim_path.rsplit("/", 1)[0] or "/"
    if parent_path and parent_path != "/":
        UsdGeom.Xform.Define(stage, parent_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    # audit-2026-06-14r4 #1: author subdivisionScheme=none. USD's documented
    # fallback when the attr is unauthored is 'catmullClark', so importers
    # (Blender, usdview, Hydra, Omniverse) would treat our marching-cubes
    # polygon mesh as a Catmull-Clark *control cage* and render a smoothed,
    # shrunken surface — not the baked mesh this cache is meant to reproduce
    # faithfully. 'none' pins it as a plain polygon mesh; rightHanded locks
    # the winding so faces don't flip across consumers.
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)

    points_attr = mesh.GetPointsAttr()
    fvc_attr = mesh.GetFaceVertexCountsAttr()
    fvi_attr = mesh.GetFaceVertexIndicesAttr()

    start_t = None
    end_t = None
    for frame_idx, verts, faces in frames:
        time = float(frame_idx)
        if start_t is None or time < start_t:
            start_t = time
        if end_t is None or time > end_t:
            end_t = time
        if verts is None or len(verts) == 0:
            points_attr.Set(Vt.Vec3fArray(), time)
            fvc_attr.Set(Vt.IntArray(), time)
            fvi_attr.Set(Vt.IntArray(), time)
            continue
        v = np.asarray(verts, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int32)
        points_attr.Set(Vt.Vec3fArray.FromNumpy(v), time)
        fvc_attr.Set(Vt.IntArray.FromNumpy(np.full(len(f), 3, dtype=np.int32)), time)
        fvi_attr.Set(Vt.IntArray.FromNumpy(f.ravel()), time)

    if start_t is not None:
        stage.SetStartTimeCode(start_t)
        stage.SetEndTimeCode(end_t)
    stage.GetRootLayer().Save()
    return path
