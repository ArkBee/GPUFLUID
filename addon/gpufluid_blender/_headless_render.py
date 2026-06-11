"""[Layer A8.12] Headless render driver — entry point for ``gpufluid render``.

Spawned via ``blender --background --python _headless_render.py -- <json_payload>``
by :func:`gpufluid.cli.commands.cmd_render`. Delegates to the existing
:mod:`examples.render_fluid_on_cube_eevee` for the actual scene
construction. The scene builder there has been refactored to import
the library helpers (A8.9 / A8.10 / A8.11) so no code is duplicated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make repo modules importable inside Blender
_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO / "src", _REPO / "addon", _REPO / "examples"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _argv_after_doubledash():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1:]


def build_renderer_argv(payload: dict) -> list[str]:
    """Convert the CLI `gpufluid render` JSON payload into argv for the
    example renderer's argparse.

    Split out from :func:`main` so unit tests can exercise the Bug #1
    (samples drop) and Bug #2 (`--frames 0` overrides cache fallback)
    fixes without spinning up Blender.

    The returned list starts with ``["blender", "--", ...]`` because the
    example renderer's :func:`_argv_after_doubledash` slices on the ``--``
    sentinel; preserving that shape keeps tests honest about the actual
    sys.argv we're feeding the renderer.
    """
    argv = [
        "blender", "--",
        "--cache",  payload["cache"],
        "--scene",  payload["scene"],
        "--out",    payload["out"],
        "--label",  payload["label"],
        "--color",  *(f"{c:.3f}" for c in payload["color"]),
        "--fps",    str(payload["fps"]),
    ]
    # Bug #2: only pass --frames when the user explicitly set a positive count;
    # otherwise the renderer treats `0` as "render 0 frames" instead of falling
    # back to cache.json's frame_count via its own default.
    frames = int(payload.get("frames", 0) or 0)
    if frames > 0:
        argv += ["--frames", str(frames)]
    # Bug #1: --samples was dropped on the floor; renderer fell back to its
    # hardcoded apply_eevee_preset(samples=16). Pass it through whenever the
    # payload supplies a positive value.
    samples = int(payload.get("samples", 0) or 0)
    if samples > 0:
        argv += ["--samples", str(samples)]
    # FU-032: optional material preset. NOT in the round-28 required-keys
    # list on purpose — a missing/None key means "legacy label inference"
    # in the renderer (old payloads keep working byte-identically).
    material = payload.get("material")
    if material:
        argv += ["--material", str(material)]
    return argv


def main():
    rest = _argv_after_doubledash()
    if not rest:
        raise SystemExit("[A8.12] no JSON payload supplied after `--`")
    # Round-25: pre-round-25 a malformed JSON payload raised
    # `json.JSONDecodeError` as a bare Python traceback, and any
    # missing required key (`payload["cache"]`) raised `KeyError`
    # the same way. Both bubbled up to Blender, which exited rc=1
    # indistinguishable from a real render failure — cmd_render in
    # the CLI (commands.py) just returns the rc with no context.
    # Catch + reformat both into a single-line SystemExit so the
    # parent process gets a useful one-line message.
    try:
        payload = json.loads(rest[0])
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"[A8.12] malformed JSON payload: {exc}; "
            f"first 80 chars: {rest[0][:80]!r}")
    # Round-28: include EVERY key build_renderer_argv unconditionally
    # indexes. Pre-round-28 only cache/scene/out were checked, and a
    # caller (hand-built JSON or future drift) missing `label` /
    # `color` / `fps` got a bare KeyError traceback through Blender
    # — exactly the failure mode round-25 set out to eliminate.
    for required in ("cache", "scene", "out", "label", "color", "fps"):
        if required not in payload:
            raise SystemExit(
                f"[A8.12] payload missing required key '{required}'; "
                f"got keys: {sorted(payload.keys())}")

    # Hand off to the example renderer's main entry — synthesise its argv.
    import render_fluid_on_cube_eevee as renderer  # type: ignore[import-not-found]
    sys.argv = build_renderer_argv(payload)
    renderer.main()


if __name__ == "__main__":
    main()
