"""[BLK A8.5] Scene-collection helpers extracted from operators/bake.py.

Phase 4 split. Holds two small utilities that ``collect_scene`` calls but
which have no dependency on the bake operator's own state:

* ``_output_dict`` — flattens the Domain object's `output` properties
  (mesher params, whitewater toggles, particle/USD flags) into the dict
  shape ``config_builder.build_toml`` expects.
* ``_export_obj`` — quick wavefront-OBJ writer for mesh obstacles; called
  inline so the simulator subprocess sees a plain file next to scene.toml.
"""
from __future__ import annotations

import bpy


def _output_dict(dprops):
    out = {
        "cache_dir": bpy.path.abspath(dprops.cache_dir),
        "iso_level": dprops.iso_level, "smooth_passes": dprops.smooth_passes,
        "mesh_smooth_passes": dprops.mesh_smooth_passes,
        "mesh_smooth_method": dprops.mesh_smooth_method,
        "decimate_ratio": dprops.decimate_ratio,
        "wall_margin_cells": dprops.wall_margin_cells,
        "usd": dprops.write_usd,
        "particles": dprops.write_particles, "preview": False,
        # B18.5 — defaults to "linear" on the property; user opts into mixbox
        # via the Domain N-panel dropdown.
        "color_mix_mode": str(getattr(dprops, "color_mix_mode", "linear")),
    }
    ww = getattr(dprops, "whitewater_group", None)
    if ww is not None and ww.enable:
        out["whitewater"] = True
        out["whitewater_speed_threshold"] = float(ww.speed_threshold)
        out["whitewater_lifetime_sec"] = float(ww.lifetime_sec)
        out["whitewater_emit_per_frame_max"] = int(ww.emit_per_frame_max)
        out["whitewater_total_cap"] = int(ww.total_cap)
        if getattr(ww, "use_potential", False):
            out["whitewater_use_potential"] = True
            out["whitewater_potential_radius"] = float(ww.potential_radius)
            out["whitewater_potential_v_max"] = float(ww.potential_v_max)
            wcw = float(getattr(ww, "wave_crest_weight", 0.0))
            if wcw > 0.0:
                out["whitewater_wave_crest_weight"] = wcw
    return out


def _export_obj(obj, path):
    """Quick OBJ export of a single object's evaluated mesh."""
    dg = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(dg)
    mesh = eval_obj.to_mesh()
    try:
        with open(path, "w") as f:
            f.write(f"# gpufluid obstacle export: {obj.name}\n")
            mw = obj.matrix_world
            for v in mesh.vertices:
                wv = mw @ v.co
                f.write(f"v {wv.x} {wv.y} {wv.z}\n")
            for p in mesh.polygons:
                idx = " ".join(str(i + 1) for i in p.vertices)
                f.write(f"f {idx}\n")
    finally:
        eval_obj.to_mesh_clear()
