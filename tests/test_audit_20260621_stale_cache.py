"""Root-cause fix (2026-06-21): `gpufluid simulate` must strip stale per-frame
outputs from a REUSED cache_dir before a fresh bake.

Symptom that exposed it: a verify script read a MIX of two bakes because MPM
particle dumps are named by cumulative sub-step count (sim_<substeps>.ply), which
differs every bake, so they accumulated. Mesh frames (fixed frame_NNNN names)
also leave a stale TAIL when a re-bake is shorter. Both bake paths now call
`_clear_stale_frame_outputs`, guarded so --resume / --start-frame keep the cache.

These are pure filesystem/args tests — no GPU bake needed.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from gpufluid.cli.commands import _clear_stale_frame_outputs

COMMANDS = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli" / "commands.py"


def _args(**kw):
    base = dict(resume=None, start_frame=0)
    base.update(kw)
    return argparse.Namespace(**base)


def _make_cache(tmp_path):
    """A cache_dir that looks like a prior bake: per-frame subdirs with files,
    plus root-level cache.json + a checkpoint that must SURVIVE the strip."""
    cd = tmp_path / "cache"
    (cd / "particles_raw").mkdir(parents=True)
    (cd / "mesh").mkdir()
    (cd / "colors").mkdir()
    for i in range(5):
        (cd / "particles_raw" / f"sim_{i:010d}.ply").write_text("x")
        (cd / "mesh" / f"frame_{i:04d}.ply").write_text("x")
    (cd / "colors" / "frame_0000.npy").write_text("x")
    (cd / "cache.json").write_text("{}")
    (cd / "checkpoint_frame_0009.npz").write_text("x")
    return cd


def test_fresh_bake_strips_per_frame_files(tmp_path):
    cd = _make_cache(tmp_path)
    n = _clear_stale_frame_outputs(
        cd, ("particles_raw", "mesh", "colors", "temperatures"), _args())
    assert n == 11, f"should remove 5+5+1 stale frame files, removed {n}"
    assert list((cd / "particles_raw").iterdir()) == []
    assert list((cd / "mesh").iterdir()) == []
    # root-level cache.json + checkpoint MUST survive (they're not per-frame).
    assert (cd / "cache.json").exists(), "cache.json must never be stripped"
    assert (cd / "checkpoint_frame_0009.npz").exists(), "checkpoints must survive"


def test_resume_keeps_everything(tmp_path):
    cd = _make_cache(tmp_path)
    n = _clear_stale_frame_outputs(
        cd, ("particles_raw", "mesh", "colors"), _args(resume="ckpt.npz"))
    assert n == 0, "a --resume bake must NOT strip the cache it continues"
    assert len(list((cd / "particles_raw").iterdir())) == 5


def test_start_frame_keeps_everything(tmp_path):
    cd = _make_cache(tmp_path)
    n = _clear_stale_frame_outputs(
        cd, ("particles_raw", "mesh", "colors"), _args(start_frame=20))
    assert n == 0, "a --start-frame>0 bake continues a cache; must not strip it"
    assert len(list((cd / "mesh").iterdir())) == 5


def test_missing_subdirs_are_noop(tmp_path):
    cd = tmp_path / "empty"
    cd.mkdir()
    assert _clear_stale_frame_outputs(cd, ("particles_raw", "mesh"), _args()) == 0


def test_strip_clears_whitewater_subdirs(tmp_path):
    """Review-wave bug (2026-06-21): the FLIP bake also writes per-frame
    whitewater/ + whitewater_kinds/ frame_NNNN.npy. A shorter re-bake left a
    stale foam tail the renderer mixed with the new bake — the §9.6 twin of
    the mesh tail. The helper must clear those subdirs when listed."""
    cd = tmp_path / "cache"
    (cd / "whitewater").mkdir(parents=True)
    (cd / "whitewater_kinds").mkdir()
    for i in range(4):
        (cd / "whitewater" / f"frame_{i:04d}.npy").write_text("x")
        (cd / "whitewater_kinds" / f"frame_{i:04d}.npy").write_text("x")
    n = _clear_stale_frame_outputs(
        cd, ("mesh", "particles", "colors", "temperatures",
             "whitewater", "whitewater_kinds"), _args())
    assert n == 8, f"should clear 4+4 stale whitewater frame files, removed {n}"
    assert list((cd / "whitewater").iterdir()) == []
    assert list((cd / "whitewater_kinds").iterdir()) == []


def test_flip_strip_includes_whitewater():
    """§9.12: the FLIP strip tuple must list BOTH whitewater subdirs (the
    review-wave gap). Without this the renderer reads a mixed-bake foam tail."""
    code = "\n".join(l for l in COMMANDS.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    assert '"whitewater", "whitewater_kinds"' in code, (
        "FLIP stale-strip must include whitewater + whitewater_kinds subdirs")


def test_both_bake_paths_call_the_strip():
    """§9.12: MPM and FLIP bakes must both invoke the helper (the FLIP twin was
    the §9.6 mirror — easy to fix one path and forget the other)."""
    code = "\n".join(l for l in COMMANDS.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    assert code.count("_clear_stale_frame_outputs(") >= 3, (
        "expected the def + an MPM call + a FLIP call to _clear_stale_frame_outputs")
    # the MPM path strips particles_raw; the FLIP path strips particles.
    assert '("particles_raw", "mesh"' in code, "MPM bake must strip particles_raw"
    assert '("mesh", "particles"' in code, "FLIP bake must strip its particles dir"
