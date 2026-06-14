"""Audit 2026-06-15 round 8 — renderer fps, refractive scope, addon PLY layout,
whitewater trapped-air weight, inflow frame_start clamp.

5 confirmed findings (>=2/3 adversarial skeptics):
  #1 renderer --fps hardcoded 60, ignored cache.json's bake fps -> 24fps
     bakes played 2.5x too fast (sim-time clock + playback).
  #2 whitewater_trapped_air_weight was half-wired (config_builder emitted it,
     but _output_dict never produced it + no addon property) — §9.3 claim!=done.
  #3 addon _read_ply_minimal had no full vertex-layout validation, so an
     xyz+normals PLY was silently mis-strided (§9.6 vs library read_ply).
  #4 --refractive forced a glass-white base colour for ANY material,
     contradicting its documented water-only scope.
  #5 inflow frame_start wasn't clamped to sim.frames (frame_end was), so a
     too-large start made a clamp-induced reversed range -> raw ValueError.

#3 is a functional test (the reader loads without bpy); the rest are §9.12
source contracts (comment/docstring-stripped).
"""
from __future__ import annotations

import ast
import importlib.util
import os
import struct
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ADDON = REPO / "addon" / "gpufluid_blender"


def _code(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    doc = set()
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc.update(range(body[0].lineno, body[0].end_lineno + 1))
    return "\n".join(l for i, l in enumerate(src.splitlines(), 1)
                     if i not in doc and not l.lstrip().startswith("#"))


# ── #1 renderer + cmd_render fps fall back to cache.json ───────────────────

def test_r8_1_renderer_fps_falls_back_to_cache():
    code = _code(REPO / "examples" / "render_fluid_on_cube_eevee.py")
    assert 'ap.add_argument("--fps", type=int, default=None)' in code, \
        "r8 #1: renderer --fps must default None (so the cache fallback fires)"
    assert 'cache_meta.get("fps"' in code, \
        "r8 #1: the renderer must read the bake fps from cache.json"
    # the resolved local `fps`, not args.fps, must drive build_scene + loader
    assert "FrameLoader(cache, surf, sim_time_text, fps)" in code


def test_r8_1_cmd_render_resolves_fps_from_cache():
    code = _code(REPO / "src" / "gpufluid" / "cli" / "commands.py")
    assert 'p_render.add_argument("--fps", type=int, default=None)' in code
    assert 'fps = int(_json.loads(cache_json.read_text()).get("fps", 60) or 60)' in code, \
        "r8 #1: cmd_render must resolve fps from cache.json when --fps omitted"
    assert '"fps":    fps,' in code, "r8 #1: payload must carry the resolved fps"


# ── #2 trapped-air weight fully wired ──────────────────────────────────────

def test_r8_2_trapped_air_weight_wired_end_to_end():
    props = _code(ADDON / "properties.py")
    assert "trapped_air_weight: bpy.props.FloatProperty(" in props, \
        "r8 #2: the trapped-air weight property must exist on the WW group"
    collect = _code(ADDON / "operators" / "_collect.py")
    assert 'out["whitewater_trapped_air_weight"]' in collect, \
        "r8 #2: _output_dict must produce the trapped-air weight key"
    panel = _code(ADDON / "panels.py")
    assert 'pbox.prop(ww, "trapped_air_weight")' in panel, \
        "r8 #2: the panel must expose the trapped-air weight"


# ── #3 addon PLY reader validates the vertex layout ────────────────────────

def _load_ply_reader():
    spec = importlib.util.spec_from_file_location(
        "_gf_ply_under_test", ADDON / "cache_loader" / "_ply.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_ply(path, props, records):
    hdr = "ply\nformat binary_little_endian 1.0\n"
    hdr += f"element vertex {len(records)}\n"
    for name, typ in props:
        hdr += f"property {typ} {name}\n"
    hdr += "element face 0\nend_header\n"
    body = b"".join(records)
    Path(path).write_bytes(hdr.encode() + body)


def test_r8_3_ply_rejects_normals_layout():
    m = _load_ply_reader()
    d = tempfile.mkdtemp()
    p = os.path.join(d, "n.ply")
    props = [(c, "float") for c in ("x", "y", "z", "nx", "ny", "nz")]
    _write_ply(p, props, [struct.pack("<6f", 0, 0, 0, 0, 0, 1),
                          struct.pack("<6f", 1, 1, 1, 0, 0, 1)])
    with pytest.raises(ValueError):
        m._read_ply_minimal(p)


def test_r8_3_ply_accepts_xyz_and_rgb():
    m = _load_ply_reader()
    d = tempfile.mkdtemp()
    # xyz-only
    p = os.path.join(d, "x.ply")
    _write_ply(p, [(c, "float") for c in ("x", "y", "z")],
               [struct.pack("<3f", 0, 0, 0), struct.pack("<3f", 1, 1, 1)])
    v, f, c = m._read_ply_minimal(p)
    assert v.shape == (2, 3) and c is None
    # xyz+rgb
    p2 = os.path.join(d, "c.ply")
    props = [("x", "float"), ("y", "float"), ("z", "float"),
             ("red", "uchar"), ("green", "uchar"), ("blue", "uchar")]
    _write_ply(p2, props, [struct.pack("<3f", 0, 0, 0) + bytes([10, 20, 30]),
                           struct.pack("<3f", 1, 1, 1) + bytes([40, 50, 60])])
    v2, f2, c2 = m._read_ply_minimal(p2)
    assert v2.shape == (2, 3) and c2 is not None and c2.shape == (2, 3)


# ── #4 refractive is water-only ────────────────────────────────────────────

def test_r8_4_refractive_gated_on_water():
    code = _code(REPO / "examples" / "render_fluid_on_cube_eevee.py")
    assert 'if refractive and material == "water":' in code, \
        "r8 #4: the glass-white base-colour override must be gated on water"
    # the override block must drop the Col link only for water now — pin that
    # the glass tint assignment sits under the water-gated branch.
    water_gate = code.find('if refractive and material == "water":')
    assert water_gate > 0
    assert "(0.83, 0.91, 1.0, 1.0)" in code[water_gate:water_gate + 600], \
        "r8 #4: the glass-white tint must live inside the water-gated branch"


# ── #5 inflow frame_start clamped + warned ─────────────────────────────────

def test_r8_5_inflow_frame_start_clamped():
    code = _code(REPO / "src" / "gpufluid" / "cli" / "commands.py")
    assert "frame_start=int(min(inf.frame_start, sim.frames))" in code, \
        "r8 #5: frame_start must be clamped to sim.frames (mirror frame_end)"
    assert "it starts at/after the" in code, \
        "r8 #5: an inflow starting past the sim end must warn, not silently crash"
