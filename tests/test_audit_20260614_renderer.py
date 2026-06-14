"""Audit 2026-06-14 — renderer obstacle-path + framing bugs (#2, #11).

The renderer imports bpy at module top, so we exec the pure functions in
isolation (same trick as test_fu033_render_obstacles) and source-grep the rest.
"""
from __future__ import annotations

import ast
from pathlib import Path

RENDERER = (Path(__file__).resolve().parents[1] / "examples"
            / "render_fluid_on_cube_eevee.py")


def _func(name: str):
    src = RENDERER.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    ns: dict = {"Path": Path, "tomllib": __import__("tomllib")}
    exec(compile(ast.Module([node], []), str(RENDERER), "exec"), ns)
    return ns[name]


def _code() -> str:
    src = RENDERER.read_text(encoding="utf-8")
    return "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))


# ---- #2: relative mesh-obstacle path resolved against the scene dir -------

def test_relative_mesh_path_resolved_against_scene_dir(tmp_path):
    scene = tmp_path / "scenes" / "s.toml"
    scene.parent.mkdir(parents=True)
    scene.write_text('[[obstacle]]\ntype = "mesh"\npath = "ramp.obj"\n',
                     encoding="utf-8")
    obs = _func("_load_obstacles_from_scene")(scene)
    assert len(obs) == 1
    p = Path(obs[0]["path"])
    assert p.is_absolute(), "relative mesh path was not resolved to absolute"
    assert p == (scene.parent / "ramp.obj").resolve()


def test_absolute_mesh_path_is_left_untouched(tmp_path):
    abspath = (tmp_path / "x.obj").resolve().as_posix()  # has a drive on Win
    scene = tmp_path / "s.toml"
    scene.write_text(f'[[obstacle]]\ntype = "mesh"\npath = "{abspath}"\n',
                     encoding="utf-8")
    obs = _func("_load_obstacles_from_scene")(scene)
    assert obs[0]["path"] == abspath


def test_non_mesh_obstacle_path_key_not_invented(tmp_path):
    scene = tmp_path / "s.toml"
    scene.write_text('[[obstacle]]\ntype = "sphere"\n'
                     'center = [0.5, 0.5, 0.5]\nradius = 0.1\n', encoding="utf-8")
    obs = _func("_load_obstacles_from_scene")(scene)
    assert "path" not in obs[0]


# ---- #11: content bbox clamps to the real domain extent, not [0,1] --------

def test_content_bbox_clamp_uses_domain_size_not_hardcoded_one():
    code = _code()
    assert "min(1.0, x) for x in hi" not in code, \
        "audit #11 regressed: content bbox hi clamped to a hardcoded 1.0"
    assert '_cj.get("domain_size")' in code, \
        "audit #11: content bbox should read the real domain extent from cache.json"
