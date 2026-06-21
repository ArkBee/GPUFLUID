"""Prod-hardening (2026-06-21): guard the checkpoint/resume footguns.

Independent audit (reviewer-checkpoint) found:
- MPM has NO checkpoint support, yet `--resume`/`--start-frame` were accepted and
  SILENTLY restarted the bake from frame 0 (wasted run + stale tail).
- FLIP `--start-frame N>0` without `--resume` starts a fresh solver mid-range ->
  frames 0..N-1 never written (a gap) while cache.json claims the full count;
  `--resume` without `--start-frame` re-writes from 0 with mid-sim state.

These guards fail loud (or warn) instead. The MPM guard is the first statement
in _cmd_simulate_mpm, so it raises before any GPU work — unit-testable here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from gpufluid.blocks import BlockError
from gpufluid.cli.commands import _cmd_simulate_mpm

COMMANDS = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli" / "commands.py"


def _args(**kw):
    base = dict(resume=None, start_frame=0)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("kw", [{"resume": "ckpt.npz"}, {"start_frame": 7},
                                {"resume": "ckpt.npz", "start_frame": 7}])
def test_mpm_rejects_resume_and_start_frame(kw):
    """The guard is the first statement -> raises before touching scene/GPU."""
    with pytest.raises(BlockError) as ei:
        _cmd_simulate_mpm(_args(**kw), scene=object())
    assert "FLIP-only" in str(ei.value)


def test_mpm_guard_runs_before_gpu():
    """resume=None/start_frame=0 must NOT trip the guard (it would then proceed
    to real work — which we don't run here). We only assert the guard itself is
    not raised for the clean case by checking it gets PAST the guard and fails
    later on the dummy scene (AttributeError), not with the FLIP-only BlockError."""
    with pytest.raises(Exception) as ei:
        _cmd_simulate_mpm(_args(), scene=object())
    assert "FLIP-only" not in str(ei.value)


def test_source_contracts_for_both_guards():
    code = "\n".join(l for l in COMMANDS.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    # MPM: resume/start-frame rejected.
    assert 'checkpoint/resume (--resume / --start-frame) is FLIP-only' in code
    # FLIP: start-frame>0 requires resume.
    assert "requires --resume <checkpoint>" in code
    assert "if start_frame > 0 and not getattr(args, \"resume\", None):" in code
