"""Round-20: MpmDivergenceError contract tests.

Pre-round-20, MpmSolver.run silently `print + break` on NaN, leaving
the CLI to exit rc=0 with a misleading cache.json. Round-20 raises a
typed exception every step (was: every 200 steps gated on progress=True
— never fired on short bakes).

These tests do NOT spin up warp + h5py + the real warp-mpm — they
construct a stand-in solver via `object.__new__` and override the
GPU-touching hooks. The only thing under test is the run() loop
contract: when positions() reports NaN, raise MpmDivergenceError with
truthful `step` / `last_dumped_frame` / `requested_frames` fields.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpufluid.sim.mpm.solver import MpmDivergenceError, MpmSolver


def _make_fake_solver(n_frames: int, dump_every: int,
                      nan_at_step: int | None,
                      pos_size: int = 10):
    """Build a MpmSolver shell that:
      - skips the heavy GPU __init__
      - records save_frame_ply calls
      - returns clean positions until `nan_at_step`, then NaN forever
    """
    sol = object.__new__(MpmSolver)
    sol.cfg = SimpleNamespace(n_frames=n_frames, dump_every=dump_every)
    sol._step_count = 0
    sol._nan_at = nan_at_step
    sol.saved_frames: list[int] = []

    clean_pos = np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
                        (pos_size, 1))
    nan_pos = clean_pos.copy()
    nan_pos[0, 0] = np.nan

    def _step(_k):
        sol._step_count += 1

    def _save(_out_dir, frame_idx):
        sol.saved_frames.append(frame_idx)

    def _positions():
        if nan_at_step is None or sol._step_count < nan_at_step:
            return clean_pos
        return nan_pos

    sol.step = _step
    sol.save_frame_ply = _save
    sol.positions = _positions
    return sol


def test_no_nan_returns_out_dir(tmp_path: Path):
    """Clean run: returns out_dir, never raises, saved frames at every
    `dump_every` plus frame 0."""
    sol = _make_fake_solver(n_frames=20, dump_every=5, nan_at_step=None)
    result = sol.run(tmp_path)
    assert result == tmp_path
    # frame 0 + 5, 10, 15, 20
    assert sol.saved_frames == [0, 5, 10, 15, 20]


def test_nan_at_step_1_raises_immediately(tmp_path: Path):
    """NaN already present on first step → no dumped frame beyond 0."""
    sol = _make_fake_solver(n_frames=20, dump_every=5, nan_at_step=1)
    with pytest.raises(MpmDivergenceError) as ei:
        sol.run(tmp_path)
    assert ei.value.step == 1
    # 1 // 5 == 0  → no full dump_every frame landed
    assert ei.value.last_dumped_frame == 0
    assert ei.value.requested_frames == 20
    assert sol.saved_frames == [0]


def test_nan_mid_run_reports_last_completed_dump_frame(tmp_path: Path):
    """NaN at step 13 with dump_every=5 → last dumped frame is 10."""
    sol = _make_fake_solver(n_frames=50, dump_every=5, nan_at_step=13)
    with pytest.raises(MpmDivergenceError) as ei:
        sol.run(tmp_path)
    assert ei.value.step == 13
    assert ei.value.last_dumped_frame == 10
    assert ei.value.requested_frames == 50
    # save_frame_ply called for 0, 5, 10 (13 is NOT a dump step)
    assert sol.saved_frames == [0, 5, 10]


def test_nan_exactly_on_dump_step_reports_that_frame(tmp_path: Path):
    """NaN at step 15 (which IS a dump step) → save runs first, NaN
    check fires after, last_dumped_frame == 15."""
    sol = _make_fake_solver(n_frames=20, dump_every=5, nan_at_step=15)
    with pytest.raises(MpmDivergenceError) as ei:
        sol.run(tmp_path)
    assert ei.value.step == 15
    assert ei.value.last_dumped_frame == 15
    assert sol.saved_frames == [0, 5, 10, 15]


def test_error_message_contains_step_and_frame_counts(tmp_path: Path):
    sol = _make_fake_solver(n_frames=100, dump_every=10, nan_at_step=42)
    with pytest.raises(MpmDivergenceError) as ei:
        sol.run(tmp_path)
    msg = str(ei.value)
    assert "42" in msg
    assert "100" in msg
    # last_dumped = (42 // 10) * 10 == 40
    assert "40" in msg


def test_error_fields_are_ints_not_numpy_scalars(tmp_path: Path):
    """Round-20 contract: cache.json (CLI side) json.dumps the fields;
    numpy int64 would crash json — exception coerces to plain int."""
    sol = _make_fake_solver(n_frames=20, dump_every=5, nan_at_step=7)
    with pytest.raises(MpmDivergenceError) as ei:
        sol.run(tmp_path)
    assert type(ei.value.step) is int
    assert type(ei.value.last_dumped_frame) is int
    assert type(ei.value.requested_frames) is int


def test_subclass_of_runtimeerror(tmp_path: Path):
    """Callers may catch RuntimeError as broad-net — the typed exception
    must remain a subclass so existing handlers don't miss it."""
    sol = _make_fake_solver(n_frames=10, dump_every=5, nan_at_step=3)
    with pytest.raises(RuntimeError):
        sol.run(tmp_path)


def test_zero_size_positions_does_not_raise(tmp_path: Path):
    """`positions()` returning a 0-row array (no live particles yet)
    must not trip the NaN check — `np.any(np.isnan(empty))` is False,
    but the `pos.size` guard documents the intent."""
    sol = object.__new__(MpmSolver)
    sol.cfg = SimpleNamespace(n_frames=3, dump_every=1)
    sol.saved_frames = []
    sol.step = lambda k: None
    sol.save_frame_ply = lambda _o, f: sol.saved_frames.append(f)
    sol.positions = lambda: np.empty((0, 3), dtype=np.float32)
    out = sol.run(tmp_path)
    assert out == tmp_path
    assert sol.saved_frames == [0, 1, 2, 3]
