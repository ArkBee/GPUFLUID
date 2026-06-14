"""Audit 2026-06-14 round 3 — FLIP solver (solver3d.py) fixes.

These all live in GPU kernels / device-buffer bookkeeping that a vanilla
pytest run can't exercise without CUDA, so per CLAUDE.md §9.12 they are
source-grep contract tests on the comment/docstring-stripped source: each
pins the exact *shape* of a fix so the old buggy form can't silently
return. Every assert names the round + the bug it guards.

Bugs covered (round 3):
  #1 viscosity ping-ponged us/vs/ws — the FLIP reference that
     k3_g2p_and_advect reads — instead of dedicated visc_* scratch.
  #2 load_checkpoint ignored the saved transfer_mode (flip/pic/apic),
     so a flip checkpoint resumed as whatever the ctor defaulted to.
  #3 (REVERTED) a false-positive graph-eligibility guard; this file pins
     that the bogus guard stays gone and the real APIC guard stays.
  #4 FLIP inflow with a per-emit colour/temperature never padded the
     existing particles, so the attribute arrays desynced from pos/vel.
  #5 keyframe-obstacle coupling velocity hard-coded fps=24 (mirror-drift
     vs LinearMotion); now threaded through frame_dt. (Also closes the
     round-4 KeyframeMotion-fps finding — frame_dt is authoritative.)
  #6 seed colour pad was white (1.0) — should be black (0.0) to match the
     GPU-reseed / prepare_frame convention (round-2 #6).
  #7 the PCG apply-A operators (dense + per-tile) had no identity row for
     an all-solid-neighbour fluid cell, unlike Jacobi/GS/preconditioner.
  #8 seed temperature pad was 0.0 °C — should be the 20.0 ambient default.
"""
from __future__ import annotations

import ast
from pathlib import Path

SOLVER = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "solvers" / "solver3d.py"


def _code(path: Path) -> str:
    """Source with comments and docstrings removed (§9.12) — so a grep can't
    trip over the bug's name in a `# fix:` note or a docstring example."""
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


CODE = _code(SOLVER)


# ── #1 viscosity uses dedicated visc_* scratch, never the FLIP reference ──

def test_r3_1_viscosity_pingpongs_dedicated_scratch():
    # The dedicated scratch must be allocated…
    for c in ("visc_u", "visc_v", "visc_w"):
        assert f"self.{c} = zeros(" in CODE, \
            f"r3 #1: viscosity scratch self.{c} must be pre-allocated"
    # …passed into the Jacobi sweep, and ping-ponged with the live field.
    assert "self.visc_u, r], device=dev); self.u, self.visc_u = self.visc_u, self.u" in CODE, \
        "r3 #1: viscosity must iterate u <-> visc_u (not u <-> us)"
    assert "self.visc_v, r], device=dev); self.v, self.visc_v = self.visc_v, self.v" in CODE
    assert "self.visc_w, r], device=dev); self.w, self.visc_w = self.visc_w, self.w" in CODE


def test_r3_1_viscosity_does_not_touch_us_vs_ws():
    # The old bug swapped self.us/vs/ws inside the viscosity loop, trashing
    # the pre-pressure FLIP reference k3_g2p_and_advect reads. Those swaps
    # must not reappear anywhere near k3_jacobi_visc.
    assert "self.us, r]" not in CODE and "self.us = self.u" not in CODE, \
        "r3 #1: viscosity must not reuse the us/vs/ws FLIP reference buffers"


# ── #2 checkpoint round-trips transfer_mode ────────────────────────────────

def test_r3_2_checkpoint_saves_and_restores_transfer_mode():
    assert "transfer_mode=str(self.transfer_mode)" in CODE, \
        "r3 #2: save_checkpoint must persist transfer_mode"
    assert 'if "transfer_mode" in data.files:' in CODE, \
        "r3 #2: load_checkpoint must read transfer_mode if present"
    assert "self.transfer_mode = saved_mode" in CODE, \
        "r3 #2: load_checkpoint must restore the saved transfer_mode"
    assert "saved_mode != str(self.transfer_mode)" in CODE, \
        "r3 #2: a mismatch (ctor vs checkpoint) must be detected (and warned)"


# ── #3 the false-positive graph guard stays reverted; APIC guard stays ─────

def test_r3_3_no_false_positive_scratch_eligibility_guard():
    # Device-only wp.zeros scratch is fine mid-capture (B5.a/B5.b prove it);
    # the reverted guard refused eligibility and broke capture-on-first-step.
    for bogus in (
        "self.surface_tension > 0.0 and self._csf_chi is None",
        "self.attr_color is not None and self._cgrid_r is None",
        'pressure_solver == "pcg" and getattr(self, "_pcg_r", None) is None',
    ):
        assert bogus not in CODE, \
            f"r3 #3: false-positive graph-eligibility guard came back: {bogus}"


def test_r3_3_real_apic_capture_guard_remains():
    # The genuine alloc+H2D-copy hazard (APIC affine_C rebuild) must stay
    # guarded — that one really is forbidden mid-capture.
    assert 'self.transfer_mode == "apic" and self.affine_C is None' in CODE, \
        "r3 #3: the real APIC alloc-in-capture guard must remain"


# ── #4 FLIP inflow pads existing particles' colour / temperature ───────────

def test_r3_4_flip_inflow_pads_existing_colour_and_temp():
    assert "any_emit_color = any(e.color is not None for e in emit_events)" in CODE, \
        "r3 #4: must detect a per-emit colour even when no particle has one yet"
    assert "need_color_pad = (self.attr_color is not None) or any_emit_color" in CODE
    # Existing particles seed to the white/ambient defaults so the attr array
    # length matches pos/vel before the emit rows are concatenated.
    assert "np.ones((self.n_particles, 3), dtype=np.float32)" in CODE, \
        "r3 #4: existing particles must get a white colour pad on first colour emit"
    assert "np.full(self.n_particles, 20.0, dtype=np.float32)" in CODE, \
        "r3 #4: existing particles must get a 20.0 temp pad on first temp emit"


# ── #5 keyframe coupling velocity is fps-correct (frame_dt threaded) ───────

def test_r3_5_keyframe_velocity_uses_frame_dt_not_const24():
    assert "denom = 2.0 * float(frame_dt) if frame_dt else 1.0" in CODE, \
        "r3 #5: keyframe velocity must divide by the real frame_dt"
    assert "def _build_anim_sdf_per_obstacle(self, frame_idx: int, frame_dt: float):" in CODE, \
        "r3 #5: frame_dt must be threaded into the per-obstacle SDF build"
    assert "self._build_anim_sdf_per_obstacle(frame_idx, frame_dt)" in CODE, \
        "r3 #5: the call site must pass frame_dt through"
    # The hard-coded `m.fps if hasattr(m,'fps') else 24` consumer is gone.
    assert "m.fps if hasattr" not in CODE, \
        "r3 #5/round-4: the const-24 keyframe-fps fallback must be gone"


# ── #6 / #8 seed pads match the prepare_frame / reseed convention ──────────

def test_r3_6_seed_colour_pad_is_black():
    # Two seed sites (add_source, seed_box); both must pad colour with zeros.
    assert CODE.count("pad = np.zeros((self.n_particles - prev_n, 3), dtype=np.float32)") == 1, \
        "r3 #6: add_source colour pad must be black (np.zeros), not white"
    assert CODE.count("pad = np.zeros((len(positions) - len(prev_pos), 3), dtype=np.float32)") == 1, \
        "r3 #6: seed_box colour pad must be black (np.zeros), not white"
    assert "np.ones((self.n_particles - prev_n, 3)" not in CODE, \
        "r3 #6: the old white seed colour pad must not return"


def test_r3_8_seed_temperature_pad_is_ambient():
    assert CODE.count("pad = np.full(new_n, 20.0, dtype=np.float32)") == 2, \
        "r3 #8: both seed sites must pad temperature with the 20.0 ambient default"
    assert "pad = np.zeros(new_n" not in CODE, \
        "r3 #8: the old 0.0 °C seed temperature pad must not return"


# ── #7 PCG apply-A operators carry an identity row for solid-locked cells ──

def test_r3_7_both_pcg_applyA_have_singular_row_guard():
    # Dense k3_apply_A and sparse k3_apply_A_per_tile must each short-circuit a
    # cell whose neighbours are all solid (diag<0.5) to an identity row, like
    # Jacobi/GS/the preconditioner already do — else the PCG operator is
    # singular under a moving solid.
    assert "y[i, j, k] = x[i, j, k]" in CODE, \
        "r3 #7: dense k3_apply_A must give an all-solid-neighbour cell an identity row"
    assert "y[li, lj, lk] = x[li, lj, lk]" in CODE, \
        "r3 #7: sparse k3_apply_A_per_tile must give the same identity row (mirror)"
    # Both guards keyed on the same diag<0.5 singular test.
    assert CODE.count("if diag < 0.5:") >= 7, \
        "r3 #7: the diag<0.5 singular-row guard must be present in every apply-A path"
