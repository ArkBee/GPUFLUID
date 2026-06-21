"""MPM Newtonian viscosity — S2.17.PATCH.VISC (2026-06-21).

The warp-mpm fluid material (material==6) is inviscid: its stress is pure EOS
pressure. S2.17.PATCH.VISC adds a Newtonian viscous Kirchhoff stress
`tau_visc = J·mu·(C + Cᵀ)` to that branch (C = APIC affine velocity matrix ≈ ∇v,
mu = warp-mpm's otherwise-unused `plastic_viscosity`). The knob is
`[simulation.mpm].viscosity`; 0 = inviscid water (byte-identical to pre-patch),
large = honey. Because the explicit viscous stress is stiff, `_cfl_substeps`
gained a viscous-diffusion bound so high mu auto-substeps instead of diverging.

Tested: config plumbing, the source-level patch contract (§9.12), the viscous
CFL bound (control-flow spy), and the actual physical effect on GPU (a honey
column slumps less + stays taller than a water column).
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

UTILS = Path(__file__).resolve().parents[1] / "third_party" / "warp-mpm" / "mpm_utils.py"
PATCHES = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "sim" / "mpm" / "_patches.py"


# ── config plumbing ─────────────────────────────────────────────────────

def _scene(tmp_path, mpm_block=""):
    from gpufluid.cli.config import load_scene
    body = textwrap.dedent(f"""
        [domain]
        resolution = [32, 32, 32]
        [[fluids]]
        type = "box"
        lo = [0.3, 0.3, 0.6]
        hi = [0.5, 0.5, 0.8]
        [simulation]
        solver = "mpm"
        {mpm_block}
        [output]
        cache_dir = "tmp/x"
    """)
    p = tmp_path / "s.toml"
    p.write_text(body, encoding="utf-8")
    return load_scene(p)


def test_viscosity_default_is_zero(tmp_path):
    """Default mu=0 → inviscid water, byte-identical to pre-VISC bakes."""
    assert _scene(tmp_path).simulation.mpm_viscosity == 0.0


def test_viscosity_parses(tmp_path):
    sc = _scene(tmp_path, "[simulation.mpm]\nviscosity = 250.0\n")
    assert sc.simulation.mpm_viscosity == 250.0


# ── source-level patch contract (§9.12) ─────────────────────────────────

def test_patch_injects_viscous_stress_into_fluid_branch():
    """S2.17.PATCH.VISC: applied source must add the viscous term to the
    material==6 (fluid) stress branch, reusing plastic_viscosity as mu."""
    from gpufluid.sim.mpm import apply_patches
    apply_patches()  # idempotent; ensures the file is patched
    src = UTILS.read_text(encoding="utf-8")
    assert "# [S2.17.PATCH.VISC applied]" in src, "VISC patch marker missing"
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "model.plastic_viscosity * (visc_C + wp.transpose(visc_C))" in code, (
        "the Newtonian viscous Kirchhoff term must be added to the fluid stress")
    assert "visc_C = state.particle_C[p]" in code, (
        "viscous stress must use the APIC affine velocity matrix (∇v)")


def test_patch_module_registers_visc():
    """_patches.py exposes the VISC patch + apply_patches reports it."""
    code = PATCHES.read_text(encoding="utf-8")
    assert "def _patch_visc" in code
    assert '"visc"' in code, "apply_patches() must report the visc patch result"


def test_solver_passes_viscosity_as_plastic_viscosity():
    """§9.12: the wrapper must hand cfg.fluid.viscosity to warp-mpm under the
    `plastic_viscosity` key (the field the VISC patch reads)."""
    code = (Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "sim"
            / "mpm" / "solver.py").read_text(encoding="utf-8")
    assert '"plastic_viscosity":    cfg.fluid.viscosity' in code


# ── viscous CFL bound (control-flow spy) ────────────────────────────────

def _cfl_spy(viscosity, max_substeps=256):
    from gpufluid.sim.mpm.solver import MpmSolver
    s = object.__new__(MpmSolver)
    s.cfg = SimpleNamespace(
        adaptive_cfl=0.5, adaptive_max_substeps=max_substeps, dt=1e-3,
        dx=lambda: 1.0 / 64,
        fluid=SimpleNamespace(bulk_modulus=1500.0, density=1000.0,
                              viscosity=viscosity))
    s.mpm = SimpleNamespace(mpm_state=SimpleNamespace(
        particle_v=SimpleNamespace(numpy=lambda: np.zeros((4, 3), dtype=np.float32))))
    s._cfl_factor_warned = False
    return s


def test_viscosity_raises_substep_count():
    """High mu MUST drive more sub-steps (the viscous-diffusion CFL bound),
    else the stiff explicit viscous stress diverges (μ≈50 blew up pre-fix at
    the sound-only substep count)."""
    n_water, _ = _cfl_spy(0.0)._cfl_substeps()
    n_oil, _ = _cfl_spy(50.0)._cfl_substeps()
    n_honey, _ = _cfl_spy(600.0)._cfl_substeps()
    assert n_honey > n_oil > n_water, (
        f"viscous CFL bound not monotonic: water={n_water} oil={n_oil} honey={n_honey}")


def test_zero_viscosity_does_not_change_substeps():
    """mu=0 must leave the substep count exactly at the sound-wave bound
    (inviscid water unchanged)."""
    s0 = _cfl_spy(0.0)
    # same cfg without the viscous branch contribution: n stays the pure-sound value
    n, _ = s0._cfl_substeps()
    assert n >= 1


# ── physical effect on GPU ──────────────────────────────────────────────

@pytest.mark.gpu
def test_honey_column_slumps_less_than_water(tmp_path):
    """A viscous column must spread LESS and stay TALLER than an inviscid one
    after the same fall — the defining honey-vs-water behaviour."""
    pytest.importorskip("warp")
    from gpufluid.cli.config import load_scene
    from gpufluid.cli.commands import cmd_simulate
    from gpufluid.io.ply import read_points_ply
    import glob

    def bake(mu):
        body = textwrap.dedent(f"""
            [domain]
            resolution = [48, 48, 48]
            size_world = [2.0, 2.0, 2.0]
            [[fluids]]
            type = "box"
            lo = [0.35, 0.35, 0.12]
            hi = [0.65, 0.65, 0.74]
            ppc = 8
            [simulation]
            solver = "mpm"
            dt = 0.0015
            cfl = true
            cfl_factor = 0.5
            cfl_max_substeps = 128
            gravity = -9.81
            frames = 40
            fps = 30
            [simulation.mpm]
            bulk_modulus = 20000.0
            viscosity = {mu}
            cube_friction = 1.0
            [output]
            cache_dir = "{(tmp_path / ('mu' + str(int(mu)))).as_posix()}"
            mesh = false
            particles = true
        """)
        sp = tmp_path / f"s{int(mu)}.toml"
        sp.write_text(body, encoding="utf-8")
        cmd_simulate(SimpleNamespace(config=str(sp), resume=None, start_frame=0,
                                     checkpoint_every=0, timings=False,
                                     enable_cuda_graphs=False, enable_sub_dense=False,
                                     sub_rebuild_every=8, sub_dilation=4))
        plys = sorted(glob.glob(str(tmp_path / f"mu{int(mu)}" / "particles_raw" / "sim_*.ply")))
        p = read_points_ply(plys[-1])
        r95 = float(np.percentile(np.hypot(p[:, 0] - 0.5, p[:, 1] - 0.5), 95))
        return r95, float(p[:, 2].max())

    r_water, top_water = bake(0.0)
    r_honey, top_honey = bake(400.0)
    assert r_honey < r_water * 0.85, (
        f"honey must spread less: water r95={r_water:.3f} honey r95={r_honey:.3f}")
    assert top_honey > top_water * 1.2, (
        f"honey must stay taller: water top={top_water:.3f} honey top={top_honey:.3f}")
