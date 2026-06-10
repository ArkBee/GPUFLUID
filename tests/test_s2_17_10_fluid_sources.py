"""S2.17.10 — fluid SOURCE shapes beyond cube (box / sphere / mesh).

"Mark as Source" works on any object; the bake fills that object's shape with
fluid particles. These tests cover the shape seeders (CPU, no GPU), the new
sphere config parse, and the addon emission contract.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpufluid.sim.mpm import seeding  # noqa: E402

# audit-20260610: §9.12 grep hygiene — strip docstrings (triple-quoted
# blocks) AND comment lines before any source assertion below.
_TRIPLE_RE = re.compile(r'("""|\'\'\')[\s\S]*?\1')


def _code(path: Path) -> str:
    src = _TRIPLE_RE.sub("", path.read_text(encoding="utf-8"))
    return "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))


def test_seed_box_fills_aabb():
    pts = seeding.seed_box((0.2, 0.2, 0.2), (0.6, 0.4, 0.4), 0.05)
    assert pts.shape[0] > 0 and pts.shape[1] == 3
    assert pts[:, 0].min() >= 0.2 - 1e-6 and pts[:, 0].max() <= 0.6 + 1e-6
    assert pts[:, 1].max() <= 0.4 + 1e-6


def test_seed_sphere_all_within_radius():
    c, r = (0.5, 0.5, 0.7), 0.12
    pts = seeding.seed_sphere(c, r, 0.02)
    assert pts.shape[0] > 0
    d = np.linalg.norm(pts - np.array(c), axis=1)
    assert d.max() <= r + 1e-6, "every seeded point must lie inside the sphere"
    # and it should NOT just be the bbox corners — a real volumetric fill
    assert pts.shape[0] > 50


def test_seed_sphere_denser_with_finer_spacing():
    a = seeding.seed_sphere((0.5, 0.5, 0.5), 0.2, 0.04).shape[0]
    b = seeding.seed_sphere((0.5, 0.5, 0.5), 0.2, 0.02).shape[0]
    assert b > a


def test_seed_mesh_fills_interior(tmp_path):
    """A unit-ish box mesh: seeded points must all fall inside it."""
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    m.apply_translation((0.5, 0.5, 0.5))
    p = tmp_path / "boxmesh.obj"
    m.export(str(p))
    pts = seeding.seed_mesh(str(p), 0.03)
    assert pts.shape[0] > 0
    # all inside the [0.35,0.65] box (with a small margin for surface samples)
    assert pts.min() >= 0.35 - 0.03 and pts.max() <= 0.65 + 0.03


def test_config_parses_sphere_fluid():
    from gpufluid.cli.config import load_scene, FluidSphereCfg
    import tempfile
    import os
    toml = ("[domain]\nresolution=[32,32,32]\n"
            "[[fluids]]\ntype=\"sphere\"\ncenter=[0.5,0.5,0.7]\nradius=0.1\nppc=8\n"
            "[simulation]\nsolver=\"mpm\"\ngravity=-9.81\nframes=5\n"
            "[output]\ncache_dir=\"out\"\n")
    fd, path = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    Path(path).write_text(toml, encoding="utf-8")
    try:
        s = load_scene(path)
        assert len(s.fluids) == 1
        assert isinstance(s.fluids[0], FluidSphereCfg)
        assert s.fluids[0].radius == 0.1
        assert s.fluids[0].center == (0.5, 0.5, 0.7)
    finally:
        os.unlink(path)


def test_addon_emits_source_shapes():
    """Contract: collect_scene emits sphere/mesh kinds (not only box) driven by
    GpufluidFluidProps.source_type, and the panel + property exist.
    audit-20260610: properties.py is now comment+docstring-stripped before
    asserting (the raw-text grep could be satisfied by a comment)."""
    base = Path(__file__).resolve().parents[1] / "addon" / "gpufluid_blender"
    bake = _code(base / "operators" / "bake.py")
    assert '"kind": "sphere"' in bake, "fluid source must be able to emit a sphere"
    assert '"kind": "mesh"' in bake, "fluid source must be able to emit a mesh"
    assert 'source_type' in bake
    props = _code(base / "properties.py")
    assert "source_type" in props and "SPHERE" in props and "MESH" in props


def test_cli_threads_shape_seeding():
    """The MPM CLI branch must dispatch on fluid type to the seeders."""
    code = _code(Path(__file__).resolve().parents[1] / "src" / "gpufluid"
                 / "cli" / "commands.py")
    assert "seed_sphere(" in code and "seed_mesh(" in code
    assert "FluidSphereCfg" in code and "FluidMeshCfg" in code
