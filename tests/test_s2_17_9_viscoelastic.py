"""[BLK S2.17.9] viscoelastic (viscoplastic) MPM material — honey that can coil.

The default "fluid" material (warp id 6, weakly-compressible) cannot rope-coil
(swept exhaustively 2026-06-21, see project memory). warp-mpm also ships a
viscoplastic StVK + von-Mises material (id 3, selected via warp's "foam" name):
a coherent elastic rope that yields and flows above `yield_stress`, which CAN
buckle/coil. This wires it in behind `[simulation.mpm].material = "viscoelastic"`
with young_modulus / poisson / yield_stress (viscosity reused as the real
plastic_viscosity). Default stays "fluid" → byte-identical.

Config parsing is unit-tested; the GPU constitutive path (warp kernels, needs
CUDA) is covered by source-grep contracts (§9.12 / round-25 pattern) — the
critical one being that finalize_mu_lam_bulk IS called for non-fluid materials
(mu/lam are zero otherwise → silent garbage stress).
"""
from __future__ import annotations

from pathlib import Path

from gpufluid.cli.config import load_scene
from gpufluid.sim.mpm.solver import MpmFluidParams

_SOLVER = (Path(__file__).resolve().parents[1] / "src" / "gpufluid"
           / "sim" / "mpm" / "solver.py")

_BASE = """
[domain]
resolution = [32, 32, 32]
size_world = [1.0, 1.0, 1.0]
[[inflow]]
lo = [0.45, 0.45, 0.6]
hi = [0.55, 0.55, 0.65]
velocity = [0.0, 0.0, -0.3]
rate_per_sec = 1000
[simulation]
solver = "mpm"
frames = 10
fps = 24
"""


def _write(tmp_path, extra):
    p = tmp_path / "s.toml"
    p.write_text(_BASE + extra)
    return load_scene(p)


def test_default_material_is_fluid(tmp_path):
    sc = _write(tmp_path, "")
    assert sc.simulation.mpm_material == "fluid"


def test_viscoelastic_params_parse(tmp_path):
    sc = _write(tmp_path, """
[simulation.mpm]
material = "viscoelastic"
young_modulus = 33000.0
poisson = 0.35
yield_stress = 250.0
viscosity = 120.0
""")
    s = sc.simulation
    assert s.mpm_material == "viscoelastic"
    assert s.mpm_young_modulus == 33000.0
    assert s.mpm_poisson == 0.35
    assert s.mpm_yield_stress == 250.0
    assert s.mpm_viscosity == 120.0


def test_fluidparams_has_viscoelastic_fields():
    fp = MpmFluidParams()
    assert fp.material == "fluid"            # default unchanged
    assert hasattr(fp, "young_modulus")
    assert hasattr(fp, "poisson")
    assert hasattr(fp, "yield_stress")


def _code(path: Path) -> str:
    return "\n".join(
        ln for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#"))


def test_solver_maps_viscoelastic_to_foam():
    code = _code(_SOLVER)
    assert '"viscoelastic": "foam"' in code, (
        "S2.17.9: viscoelastic must map to warp-mpm id-3 material ('foam')")


def test_solver_finalizes_mu_lam_for_nonfluid():
    """The fluid path sets bulk directly; every elastoplastic material reads
    mu/lam/bulk which stay ZERO unless derived from E/nu. finalize_mu_lam_bulk
    MUST run for non-fluid — guard against silently dropping it."""
    code = _code(_SOLVER)
    assert "finalize_mu_lam_bulk(" in code
    assert 'if warp_mat != "fluid":' in code, (
        "finalize_mu_lam_bulk must be gated to non-fluid (running it for the "
        "fluid path would zero the directly-set bulk_modulus)")


def test_cfl_uses_elastic_wave_for_viscoelastic():
    """The substep CFL must use the stiff elastic P-wave (E-based) for the
    viscoelastic material, not the fluid bulk modulus, or a stiff E silently
    under-substeps and diverges."""
    code = _code(_SOLVER)
    assert 'material", "fluid") == "viscoelastic"' in code
    assert "young_modulus" in code and "1.0 - 2.0 * nu" in code
