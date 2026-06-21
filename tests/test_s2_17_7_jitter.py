"""[BLK S2.17.7.JITTER] inflow velocity-jitter — buckling/coiling seed.

2026-06-21: a perfectly axisymmetric, near-deterministic MPM viscous rope
lands dead-centre and slumps into a non-coiling heap (verified across
mu=80/140/200/250 and two fall heights — the descending thread stays within
~0.3 cells of axis, the pile spreads symmetrically, no annulus). Real
nozzles/threads carry sub-percent velocity noise that seeds the liquid-rope-
coiling buckling instability. `MpmInflow.velocity_jitter` adds that noise at
release: isotropic in XY, zero in Z (fall speed untouched), and ZERO by
default so every pre-existing scene is byte-identical.

The kernel needs CUDA + a live State struct, so (per §9.12 / the round-25
pattern) the release wiring is covered by a source-grep contract test; the
pure-numpy seed generator is unit-tested directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpufluid.sim.mpm.inflow import MpmInflow, make_velocity_jitter

_SRC = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "sim" / "mpm"


# ── pure-numpy seed generator ───────────────────────────────────────────

# A representative spawn-step vector: 5000 particles released over ~220 frames
# at dump_every=16 (one frame), sorted-ish but with overlap, like the real seed.
def _spawn(n=5000, frames=220, dump_every=16, seed=11):
    r = np.random.default_rng(seed)
    f = r.uniform(0, frames, size=n)
    return (f * dump_every).astype(np.int32)


def test_jitter_zero_magnitude_is_all_zero():
    """Default / off path: must be byte-identical to no jitter."""
    out = make_velocity_jitter(_spawn(1000), 0.0, rng=np.random.default_rng(0))
    assert out.shape == (1000, 3)
    assert out.dtype == np.float32
    assert np.count_nonzero(out) == 0


def test_jitter_only_perturbs_xy_never_z():
    """Z must stay exactly 0 so the fall speed (and coiling regime) is
    untouched — the seed breaks rotational symmetry only."""
    out = make_velocity_jitter(_spawn(), 0.05, rng=np.random.default_rng(1))
    assert np.all(out[:, 2] == 0.0)
    assert np.any(out[:, 0] != 0.0) and np.any(out[:, 1] != 0.0)


def test_jitter_magnitude_is_bounded_and_zero_mean():
    mag = 0.04
    out = make_velocity_jitter(_spawn(20000), mag, rng=np.random.default_rng(2))
    speed = np.hypot(out[:, 0], out[:, 1])
    # peak lateral speed is the magnitude (signal scaled so max == mag)
    assert speed.max() <= mag + 1e-6
    # zero-mean meander: no net lean / no prescribed drift direction
    assert abs(out[:, 0].mean()) < mag * 0.25
    assert abs(out[:, 1].mean()) < mag * 0.25


def test_jitter_is_temporally_coherent():
    """The defining property: particles released in the SAME frame share
    nearly the same kick (so the thread cross-section wanders together),
    while particles many frames apart are decorrelated. Per-particle noise
    (which averages out and never coils) would fail this."""
    spawn = _spawn(8000)
    out = make_velocity_jitter(spawn, 0.05, rng=np.random.default_rng(3),
                               coherence_steps=16)
    # same-frame spread must be much smaller than the overall signal spread
    order = np.argsort(spawn)
    s, v = spawn[order], out[order]
    same = (np.diff(s) == 0)
    if same.sum() > 50:
        within = np.hypot(np.diff(v[:, 0])[same], np.diff(v[:, 1])[same]).mean()
        overall = np.hypot(out[:, 0].std(), out[:, 1].std())
        assert within < 0.2 * overall, (
            f"jitter not temporally coherent: same-frame spread {within:.4f} "
            f"vs overall {overall:.4f} — would average out, no coil seed")


def test_jitter_is_reproducible_under_seed():
    sp = _spawn(100)
    a = make_velocity_jitter(sp, 0.03, rng=np.random.default_rng(7))
    b = make_velocity_jitter(sp, 0.03, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_jitter_edge_counts():
    assert make_velocity_jitter(np.zeros(0, dtype=int), 0.1).shape == (0, 3)
    assert make_velocity_jitter(np.array([5, 5, 5]), 0.1).shape == (3, 3)


def test_inflow_jitter_field_defaults_off():
    inf = MpmInflow(lo=(0, 0, 0), hi=(1, 1, 1))
    assert inf.velocity_jitter == 0.0


# ── source-grep contract: release adds the per-particle seed ────────────

def _code(path: Path) -> str:
    lines = [
        ln for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    return "\n".join(lines)


def test_kernel_release_adds_vjit():
    """The gate kernel must ADD the per-particle vjit at release, and vjit
    must be a kernel input array — otherwise the seed never reaches the GPU."""
    code = _code(_SRC / "inflow.py")
    assert "vjit: wp.array(dtype=wp.vec3)" in code, (
        "S2.17.7.JITTER: k_inflow_gate must take a per-particle vjit array")
    assert "wp.vec3(vx, vy, vz) + vjit[p]" in code, (
        "S2.17.7.JITTER: release must add the jitter (regressed to plain "
        "vec3(vx,vy,vz) → no buckling seed reaches particles)")


def test_solver_launches_with_vjit_and_seeds_it():
    """The solver must generate the per-inflow vjit array and pass it to the
    kernel launch (mirror gap guard, §9.6)."""
    code = _code(_SRC / "solver.py")
    assert "make_velocity_jitter(" in code, (
        "solver must build the per-inflow jitter array")
    assert 'g["vjit_wp"]' in code, (
        "solver _pre_step must pass vjit_wp into the k_inflow_gate launch")
