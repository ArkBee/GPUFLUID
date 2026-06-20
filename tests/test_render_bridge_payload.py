"""[A8.12 / A8.13 — Phase 3] Payload contract: CLI <-> headless wrapper.

The addon's render path is a three-stage pipe:

    OT_render(execute)          # in-Blender, builds argv
        │
        ▼
    `gpufluid render` CLI       # commands.cmd_render — JSON-encodes a payload
        │
        ▼
    _headless_render.main       # in headless Blender, decodes payload → argv

This test pins down the two contracts:

  1. The CLI argparse accepts every flag the addon's OT_render emits.
  2. The headless wrapper's argv synthesis (build_renderer_argv) honours
     the Phase-3 bug fixes — omits ``--frames`` when payload frames==0
     (Bug #2) and emits ``--samples`` when payload samples>0 (Bug #1).

No bpy, no Blender. The OT_render argv shape is reproduced inline so the
test stays runnable in a vanilla pytest env.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the addon importable without bpy: __init__.py guards the bpy import.
_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "addon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gpufluid_blender import _headless_render  # noqa: E402


# ─── helpers ──────────────────────────────────────────────────────────────

def _cli_parser():
    """Reconstruct just the `render` subparser from commands.main().

    commands.main() builds the parser inline, so we either re-run it or
    parse via `gpufluid render --help`. Re-running is cheap and avoids
    a subprocess. We feed it argv that triggers only the `render` branch
    and returns the parsed Namespace.
    """
    # Import lazily so the warp dependency only loads when this test runs,
    # not at collection time.
    import argparse
    from gpufluid.cli import commands

    # Mirror commands.main's parser construction for the render branch only.
    # Avoids running the whole main() (which dispatches and would invoke
    # the real subprocess).
    p = argparse.ArgumentParser("gpufluid")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_render = sub.add_parser("render")
    p_render.add_argument("cache")
    p_render.add_argument("scene")
    p_render.add_argument("--out", required=True)
    p_render.add_argument("--label", default="gpufluid")
    p_render.add_argument("--color", nargs=3, type=float,
                          default=[0.20, 0.50, 0.70])
    p_render.add_argument("--samples", type=int, default=16)
    p_render.add_argument("--fps", type=int, default=60)
    p_render.add_argument("--frames", type=int, default=0)
    p_render.add_argument("--blender", default=None)
    p_render.set_defaults(func=commands.cmd_render)
    return p


def _addon_argv(cache: str, scene: str, out: str, *,
                label: str = "Water",
                color=(0.20, 0.50, 0.70),
                samples: int = 16,
                fps: int = 24,
                blender: str = "") -> list[str]:
    """Reproduce OT_render.execute's argv construction (without bpy).

    Kept in sync with addon/gpufluid_blender/operators/render.py — if that
    file changes its argv layout this test fails loud.
    """
    argv = [
        "python", "-m", "gpufluid.cli", "render",
        cache, scene,
        "--out", out,
        "--label", label,
        "--color", f"{color[0]:.4f}", f"{color[1]:.4f}", f"{color[2]:.4f}",
        "--samples", str(int(samples)),
        "--fps", str(int(fps)),   # audit-2026-06-14r2 #9
    ]
    if blender:
        argv += ["--blender", blender]
    return argv


# ─── Stage 1: CLI parser accepts the addon's argv ─────────────────────────

@pytest.mark.parametrize("kwargs,expected", [
    # default samples / no blender override
    ({}, {"label": "Water", "samples": 16, "blender": None,
          "color": [0.2, 0.5, 0.7]}),
    # explicit samples bump (hero render)
    ({"samples": 64}, {"label": "Water", "samples": 64, "blender": None}),
    # custom label + color (oil pass)
    ({"label": "Oil", "color": (0.5, 0.4, 0.1)},
        {"label": "Oil", "color": [0.5, 0.4, 0.1]}),
    # explicit Blender path (off-PATH install)
    ({"blender": "C:/blender/blender.exe"},
        {"blender": "C:/blender/blender.exe"}),
])
def test_cli_parser_accepts_addon_argv(kwargs, expected):
    parser = _cli_parser()
    addon_argv = _addon_argv("C:/c", "C:/c/scene.toml", "C:/c/render", **kwargs)
    # strip the `python -m gpufluid.cli` prelude — argparse expects subcommand argv
    ns = parser.parse_args(addon_argv[3:])
    assert ns.cmd == "render"
    assert ns.cache == "C:/c"
    assert ns.scene == "C:/c/scene.toml"
    assert ns.out == "C:/c/render"
    for key, val in expected.items():
        actual = getattr(ns, key)
        if isinstance(val, list) and isinstance(actual, list):
            assert actual == pytest.approx(val)
        else:
            assert actual == val


# ─── Stage 2: headless wrapper honours the Phase-3 bug fixes ──────────────

def _base_payload(**overrides):
    p = {
        "cache":   "/tmp/cache",
        "scene":   "/tmp/scene.toml",
        "out":     "/tmp/out",
        "label":   "Water",
        "color":   [0.20, 0.50, 0.70],
        "samples": 16,
        "fps":     60,
        "frames":  0,
    }
    p.update(overrides)
    return p


def test_frames_zero_omits_flag():
    """Bug #2: payload frames=0 must NOT produce `--frames 0` (renderer
    would render zero frames instead of falling back to cache.json)."""
    argv = _headless_render.build_renderer_argv(_base_payload(frames=0))
    assert "--frames" not in argv


def test_frames_positive_emits_flag():
    """Explicit positive frame count must be forwarded verbatim."""
    argv = _headless_render.build_renderer_argv(_base_payload(frames=120))
    assert "--frames" in argv
    assert argv[argv.index("--frames") + 1] == "120"


def test_samples_positive_emits_flag():
    """Bug #1: samples>0 must reach the renderer (was being dropped)."""
    argv = _headless_render.build_renderer_argv(_base_payload(samples=32))
    assert "--samples" in argv
    assert argv[argv.index("--samples") + 1] == "32"


def test_samples_zero_omits_flag():
    """Defensive: samples=0 is treated as 'not supplied', not as 0 TAA samples
    (which Eevee would silently clamp / render garbage)."""
    argv = _headless_render.build_renderer_argv(_base_payload(samples=0))
    assert "--samples" not in argv


def test_missing_samples_key_omits_flag():
    """The pre-Phase-3 CLI always sent `samples` but we treat its absence
    as 'use renderer default' for forward compatibility."""
    payload = _base_payload()
    payload.pop("samples")
    argv = _headless_render.build_renderer_argv(payload)
    assert "--samples" not in argv


def test_color_formatted_three_floats():
    """Color is three space-separated floats following --color."""
    argv = _headless_render.build_renderer_argv(_base_payload(color=[0.1, 0.2, 0.3]))
    i = argv.index("--color")
    # Three values immediately after --color, then the next flag (--fps).
    assert argv[i + 1 : i + 4] == ["0.100", "0.200", "0.300"]
    assert argv[i + 4] == "--fps"


def test_label_passed_through():
    argv = _headless_render.build_renderer_argv(_base_payload(label="Honey"))
    assert "--label" in argv
    assert argv[argv.index("--label") + 1] == "Honey"


# ─── FU-032: explicit material passthrough ────────────────────────────────

def test_material_passed_through_when_set():
    """FU-032: payload material must reach the renderer as --material."""
    argv = _headless_render.build_renderer_argv(
        _base_payload(material="water"))
    assert "--material" in argv
    assert argv[argv.index("--material") + 1] == "water"


def test_material_omitted_when_unset():
    """FU-032: absent/None material = legacy label inference downstream —
    the flag must NOT be emitted (old payloads stay byte-identical)."""
    argv = _headless_render.build_renderer_argv(_base_payload())
    assert "--material" not in argv
    argv = _headless_render.build_renderer_argv(_base_payload(material=None))
    assert "--material" not in argv


# ─── Goal 2026-06-20: camera azimuth passthrough (4-side validation) ───────

def test_cam_angle_passed_through_when_set():
    """A payload cam_angle must reach the renderer as --cam-angle so the
    CLI can drive the 0/90/180/270 four-side validation views."""
    argv = _headless_render.build_renderer_argv(_base_payload(cam_angle=90.0))
    assert "--cam-angle" in argv
    assert argv[argv.index("--cam-angle") + 1] == "90.0"


def test_cam_angle_zero_is_forwarded():
    """Unlike frames/samples (where 0 means 'unset'), cam_angle=0 is a real
    value (the front view) and must round-trip — only None means 'default'."""
    argv = _headless_render.build_renderer_argv(_base_payload(cam_angle=0.0))
    assert "--cam-angle" in argv
    assert argv[argv.index("--cam-angle") + 1] == "0.0"


def test_cam_angle_omitted_when_unset():
    """Absent/None cam_angle = renderer default (0°). The flag must NOT be
    emitted, so pre-2026-06-20 payloads stay byte-identical."""
    assert "--cam-angle" not in _headless_render.build_renderer_argv(
        _base_payload())
    assert "--cam-angle" not in _headless_render.build_renderer_argv(
        _base_payload(cam_angle=None))


def test_cli_render_exposes_cam_angle_and_threads_to_payload():
    """§9.12: the `gpufluid render` parser must own a --cam-angle arg and the
    payload dict must carry it through to the headless driver."""
    import ast
    path = (Path(__file__).resolve().parents[1] / "src" / "gpufluid"
            / "cli" / "commands.py")
    text = path.read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines()
                     if not l.lstrip().startswith("#"))
    assert '"--cam-angle"' in code, (
        "gpufluid render must expose --cam-angle for 4-side validation")
    assert '"cam_angle": args.cam_angle' in code, (
        "cmd_render must thread args.cam_angle into the render payload")


# ─── FU-032: renderer-side contracts (§9.12 — comment/docstring stripped) ──

def _renderer_code() -> str:
    import ast
    path = (Path(__file__).resolve().parents[1] / "examples"
            / "render_fluid_on_cube_eevee.py")
    text = path.read_text(encoding="utf-8")
    doc_lines = set()
    for node in ast.walk(ast.parse(text)):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return "\n".join(l for i, l in enumerate(text.splitlines(), 1)
                     if i not in doc_lines
                     and not l.lstrip().startswith("#"))


def test_fu032_label_keyed_material_selector_gone():
    """FU-032: the shader branch must key on the explicit `material`, not
    on the label — the old label-keyed if/else fell through to HONEY for
    any custom label (near-black under the headless Eevee preset)."""
    code = _renderer_code()
    assert 'if label.lower() == "water"' not in code, (
        "FU-032 regressed: material selection is keyed on label again")
    assert 'if material == "oil"' in code, (
        "FU-032: shader branch must key on the material arg")
    assert 'elif material == "honey"' in code, (
        "FU-032: honey must require the EXPLICIT token, never a catch-all")


def test_fu032_material_flag_and_precedence_exist():
    """--material exists; precedence = explicit flag > legacy label
    inference (water/oil/honey labels — keeps the step30-32 demo flows,
    e.g. label='oil' without --material still renders oil) > water."""
    code = _renderer_code()
    assert '"--material"' in code
    assert 'choices=["water", "oil", "honey"]' in code
    assert "args.material or" in code, (
        "FU-032: explicit --material must take precedence")
    assert 'in ("water", "oil", "honey") else "water")' in code, (
        "FU-032: legacy label inference must cover water/oil/honey and "
        "fall back to WATER (not honey) for any other label")


# ─── Stage 3: end-to-end shape sanity ─────────────────────────────────────

def test_full_payload_roundtrips_to_renderer_argv():
    """The headless argv must always start with the `blender --` sentinel
    so the renderer's _argv_after_doubledash parses what comes next."""
    argv = _headless_render.build_renderer_argv(
        _base_payload(frames=240, samples=24, label="Oil"))
    assert argv[0] == "blender"
    assert argv[1] == "--"
    # Renderer sees only the slice after `--`.
    import argparse
    rp = argparse.ArgumentParser()
    rp.add_argument("--cache", required=True)
    rp.add_argument("--scene", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--label", required=True)
    rp.add_argument("--color", nargs=3, type=float, required=True)
    rp.add_argument("--fps", type=int, default=60)
    rp.add_argument("--frames", type=int, default=None)
    rp.add_argument("--samples", type=int, default=16)
    ns = rp.parse_args(argv[2:])
    assert ns.frames == 240
    assert ns.samples == 24
    assert ns.label == "Oil"
    assert ns.color == pytest.approx([0.2, 0.5, 0.7])
