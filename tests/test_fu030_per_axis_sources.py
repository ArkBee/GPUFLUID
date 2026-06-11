"""FU-030 — per-axis sphere/mesh fluid sources (non-cubic domains).

A world-space sphere on a non-cubic domain is an ELLIPSOID in the solver's
per-axis-normalised [0,1]³ space; a mesh source likewise needs per-axis
scale. Pre-FU-030 both used a single scalar (avg inverse domain size) →
wrong shape AND wrong position in sim space.

Schema under test (CLI side):
  [[fluids]] type="sphere": optional `radii = [rx, ry, rz]` (sim units,
      exclusive with scalar `radius`; radii wins).
  [[fluids]] type="mesh": `scale` accepts scalar (uniform, back-compat)
      or `[sx, sy, sz]`.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("warp")  # cli.commands imports warp at module load

from conftest import HAS_CUDA


# ── seed_sphere: ellipsoid (per-axis radii) ─────────────────────────────

def test_seed_sphere_radii_extents_per_axis():
    """radii (0.2, 0.2, 0.1) must give per-axis extents ~ (0.4, 0.4, 0.2) —
    the FU-030 symptom was the z-extent matching x/y (avg-scaled sphere)."""
    from gpufluid.sim.mpm.seeding import seed_sphere
    sp = 0.01
    pts = seed_sphere((0.5, 0.5, 0.5), 0.0, sp, radii=(0.2, 0.2, 0.1))
    assert pts.shape[0] > 0
    ext = pts.max(axis=0) - pts.min(axis=0)
    assert np.allclose(ext, [0.4, 0.4, 0.2], atol=2 * sp), (
        f"FU-030: ellipsoid extents must be per-axis, got {ext}")


def test_seed_sphere_radii_all_points_inside_ellipsoid():
    from gpufluid.sim.mpm.seeding import seed_sphere
    c = np.array([0.5, 0.5, 0.5])
    r3 = np.array([0.2, 0.2, 0.1])
    pts = seed_sphere(tuple(c), 0.0, 0.02, radii=tuple(r3))
    q = (((pts.astype(np.float64) - c) / r3) ** 2).sum(axis=1)
    assert float(q.max()) <= 1.0 + 1e-9, (
        "FU-030: every seeded point must satisfy the ellipsoid equation")


def test_seed_sphere_scalar_backcompat_unchanged():
    """Scalar radius path must stay byte-identical to pre-FU-030 (no radii
    kwarg vs radii=None must agree, and all points lie within r)."""
    from gpufluid.sim.mpm.seeding import seed_sphere
    c = (0.5, 0.5, 0.5)
    a = seed_sphere(c, 0.15, 0.01)
    b = seed_sphere(c, 0.15, 0.01, radii=None)
    assert np.array_equal(a, b)
    d = np.linalg.norm(a.astype(np.float64) - np.asarray(c), axis=1)
    assert float(d.max()) <= 0.15 + 1e-9
    ext = a.max(axis=0) - a.min(axis=0)
    assert np.allclose(ext, [0.3, 0.3, 0.3], atol=0.02)


def test_seed_sphere_degenerate_radii_empty():
    """Any non-positive per-axis radius → empty cloud (the CLI's 0-particle
    rc=2 guard handles the rest)."""
    from gpufluid.sim.mpm.seeding import seed_sphere
    for bad in [(0.0, 0.1, 0.1), (0.1, -0.2, 0.1), (0.1, 0.1, 0.0)]:
        pts = seed_sphere((0.5, 0.5, 0.5), 0.1, 0.02, radii=bad)
        assert pts.shape == (0, 3), f"radii={bad} must seed nothing"


# ── seed_mesh: per-axis scale ───────────────────────────────────────────

def _unit_cube_obj(tmp_path: Path) -> str:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))  # bounds ±0.5
    p = tmp_path / "cube.obj"
    mesh.export(p)
    return str(p)


def test_seed_mesh_vec3_scale_extents(tmp_path):
    """Unit cube scaled (0.25, 0.25, 0.5) then centred: seeded extents must
    be per-axis (0.25, 0.25, 0.5), not uniform."""
    from gpufluid.sim.mpm.seeding import seed_mesh
    sp = 0.02
    pts = seed_mesh(_unit_cube_obj(tmp_path), sp,
                    scale=(0.25, 0.25, 0.5),
                    translate=(0.5, 0.5, 0.5))
    assert pts.shape[0] > 0
    ext = pts.max(axis=0) - pts.min(axis=0)
    assert np.allclose(ext, [0.25, 0.25, 0.5], atol=2 * sp), (
        f"FU-030: mesh vec3 scale must be per-axis, got extents {ext}")
    centre = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    assert np.allclose(centre, [0.5, 0.5, 0.5], atol=2 * sp)


def test_seed_mesh_scalar_scale_backcompat(tmp_path):
    """Scalar scale stays uniform (back-compat)."""
    from gpufluid.sim.mpm.seeding import seed_mesh
    sp = 0.02
    pts = seed_mesh(_unit_cube_obj(tmp_path), sp,
                    scale=0.25, translate=(0.5, 0.5, 0.5))
    ext = pts.max(axis=0) - pts.min(axis=0)
    assert np.allclose(ext, [0.25, 0.25, 0.25], atol=2 * sp)


def test_seed_mesh_bad_scale_raises(tmp_path):
    from gpufluid.sim.mpm.seeding import seed_mesh
    with pytest.raises(ValueError, match="FU-030"):
        seed_mesh(_unit_cube_obj(tmp_path), 0.05, scale=(0.5, 0.5))


# ── TOML parsing ────────────────────────────────────────────────────────

def _load(tmp_path: Path, fluids_body: str):
    from gpufluid.cli.config import load_scene
    toml = tmp_path / "scene.toml"
    toml.write_text(
        "[domain]\nresolution = [32, 32, 32]\n\n"
        + fluids_body
        + "\n[simulation]\nsolver = \"mpm\"\nframes = 1\n",
        encoding="utf-8",
    )
    return load_scene(toml)


def test_toml_sphere_radii_roundtrip(tmp_path):
    scene = _load(tmp_path,
                  "[[fluids]]\ntype = \"sphere\"\n"
                  "center = [0.5, 0.5, 0.5]\nradii = [0.2, 0.2, 0.1]\n")
    f0 = scene.fluids[0]
    assert f0.radii == (0.2, 0.2, 0.1)
    assert all(isinstance(v, float) for v in f0.radii)


def test_toml_sphere_scalar_radius_backcompat(tmp_path):
    scene = _load(tmp_path,
                  "[[fluids]]\ntype = \"sphere\"\nradius = 0.15\n")
    f0 = scene.fluids[0]
    assert f0.radius == 0.15 and f0.radii is None


def test_toml_sphere_radius_and_radii_rejected(tmp_path):
    from gpufluid.blocks import BlockError
    with pytest.raises(BlockError, match="not both"):
        _load(tmp_path,
              "[[fluids]]\ntype = \"sphere\"\n"
              "radius = 0.1\nradii = [0.1, 0.1, 0.2]\n")


def test_toml_sphere_radii_wrong_length_rejected(tmp_path):
    from gpufluid.blocks import BlockError
    with pytest.raises(BlockError):
        _load(tmp_path,
              "[[fluids]]\ntype = \"sphere\"\nradii = [0.1, 0.1]\n")


def test_toml_mesh_scale_union(tmp_path):
    scene = _load(tmp_path,
                  "[[fluids]]\ntype = \"mesh\"\npath = \"m.obj\"\n"
                  "scale = [0.25, 0.25, 0.5]\n")
    assert scene.fluids[0].scale == (0.25, 0.25, 0.5)
    scene2 = _load(tmp_path,
                   "[[fluids]]\ntype = \"mesh\"\npath = \"m.obj\"\n"
                   "scale = 0.25\n")
    assert scene2.fluids[0].scale == 0.25
    assert isinstance(scene2.fluids[0].scale, float)


def test_toml_mesh_scale_malformed_rejected(tmp_path):
    from gpufluid.blocks import BlockError
    with pytest.raises(BlockError, match="FU-030"):
        _load(tmp_path,
              "[[fluids]]\ntype = \"mesh\"\npath = \"m.obj\"\n"
              "scale = \"big\"\n")


# ── GPU end-to-end: non-cubic domain, world-sphere round-trip ───────────

@pytest.mark.gpu
def test_world_sphere_roundtrip_on_noncubic_domain(tmp_path):
    """World 4×4×2 m domain, world-space 0.5 m-radius sphere at the centre →
    sim radii (0.125, 0.125, 0.25). Frame-0 particle cloud, mapped BACK to
    world metres per axis, must be a ~1 m sphere on every axis (pre-FU-030
    the avg-scaled source came back as an ellipsoid)."""
    from gpufluid.cli.config import load_scene
    from gpufluid.cli.commands import _cmd_simulate_mpm
    from gpufluid.io.ply import read_points_ply
    cache = tmp_path / "cache"
    toml = tmp_path / "scene.toml"
    toml.write_text(
        "[domain]\n"
        "resolution = [64, 64, 32]\n"
        "size_world = [4.0, 4.0, 2.0]\n"
        "\n"
        "[[fluids]]\n"
        'type = "sphere"\n'
        "center = [0.5, 0.5, 0.5]\n"
        "radii = [0.125, 0.125, 0.25]\n"
        "\n"
        "[simulation]\n"
        'solver = "mpm"\n'
        "frames = 1\n"
        "\n"
        "[output]\n"
        f'cache_dir = "{cache.as_posix()}"\n'
        "mesh = false\n",
        encoding="utf-8",
    )
    scene = load_scene(toml)
    try:
        rc = _cmd_simulate_mpm(SimpleNamespace(), scene)
    except Exception:
        # Skip ONLY when CUDA is genuinely absent; on a GPU box any bake
        # exception is a real regression and must FAIL (audit-20260610
        # honest-skip pattern).
        if HAS_CUDA:
            raise
        pytest.skip("no CUDA device for MPM bake")
    assert rc == 0
    ply0 = cache / "particles_raw" / "sim_0000000000.ply"
    assert ply0.exists(), "frame-0 PLY missing"
    pts = read_points_ply(ply0)
    assert pts.shape[0] > 100
    world = pts * np.array([4.0, 4.0, 2.0])  # per-axis sim → metres
    ext_w = world.max(axis=0) - world.min(axis=0)
    # Per-axis world extents must all be the sphere diameter (1 m); the
    # pre-FU-030 bug gave z ≈ x/y in SIM space → 0.5 m vs 1 m in world.
    assert np.allclose(ext_w, [1.0, 1.0, 1.0], atol=0.15), (
        f"FU-030: world-space round-trip must be a sphere, got extents "
        f"{ext_w} m")
    centre_w = 0.5 * (world.max(axis=0) + world.min(axis=0))
    assert np.allclose(centre_w, [2.0, 2.0, 1.0], atol=0.15)
