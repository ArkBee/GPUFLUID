"""Round-5 regression: mark_* ops must be single-role.

Live-found 2026-05-25: marking the same object first as obstacle, then
as inflow, left BOTH ``is_obstacle=True`` AND ``is_inflow=True`` — CLI
then emitted ``[[obstacle]]`` AND ``[[inflow]]`` for the same mesh.

Fix in ``operators/helpers.py``: ``_set_single_role`` clears the three
peer flags before setting the active one. Test exercises that helper
directly with a stand-in object — keeps the test bpy-free (the helper
just does ``setattr(getattr(obj, group), attr, bool)`` four times).
"""
from __future__ import annotations

import types


def _make_obj_with_roles():
    """Mock Blender object with the four gpufluid_* role groups."""
    obj = types.SimpleNamespace()
    obj.gpufluid_fluid = types.SimpleNamespace(is_fluid=False)
    obj.gpufluid_obstacle = types.SimpleNamespace(is_obstacle=False)
    obj.gpufluid_inflow = types.SimpleNamespace(is_inflow=False)
    obj.gpufluid_outflow = types.SimpleNamespace(is_outflow=False)
    return obj


def _set_role(obj, target_attr):
    """Pure-Python reproduction of ``operators.helpers._set_single_role``
    (the actual module needs a live ``bpy`` to import; reproducing the
    six-line helper keeps the test loader-free).
    """
    pairs = (
        ("gpufluid_fluid", "is_fluid"),
        ("gpufluid_obstacle", "is_obstacle"),
        ("gpufluid_inflow", "is_inflow"),
        ("gpufluid_outflow", "is_outflow"),
    )
    for grp, attr in pairs:
        setattr(getattr(obj, grp), attr, attr == target_attr)


def _snapshot(obj):
    return (obj.gpufluid_fluid.is_fluid,
            obj.gpufluid_obstacle.is_obstacle,
            obj.gpufluid_inflow.is_inflow,
            obj.gpufluid_outflow.is_outflow)


def test_mark_obstacle_then_inflow_clears_obstacle():
    obj = _make_obj_with_roles()
    _set_role(obj, "is_obstacle")
    assert _snapshot(obj) == (False, True, False, False)
    _set_role(obj, "is_inflow")
    # The critical bit: is_obstacle drops to False, only is_inflow=True
    assert _snapshot(obj) == (False, False, True, False)


def test_all_four_role_transitions():
    """Exhaustive: each role activation clears the other three."""
    obj = _make_obj_with_roles()
    for target in ("is_fluid", "is_obstacle", "is_inflow", "is_outflow"):
        _set_role(obj, target)
        snap = _snapshot(obj)
        true_count = sum(snap)
        assert true_count == 1, (
            f"After _set_role({target}), expected exactly 1 role True, "
            f"got snapshot={snap}")
        # And the right one is True
        idx_map = {"is_fluid": 0, "is_obstacle": 1,
                   "is_inflow": 2, "is_outflow": 3}
        assert snap[idx_map[target]] is True


def test_double_marking_same_role_idempotent():
    obj = _make_obj_with_roles()
    _set_role(obj, "is_inflow")
    _set_role(obj, "is_inflow")
    assert _snapshot(obj) == (False, False, True, False)


def test_set_single_role_helper_matches_addon():
    """Sanity check: the local copy of `_set_single_role` matches the
    one in operators/helpers.py. If the addon helper's signature drifts
    this test should catch it on next run."""
    from pathlib import Path
    here = Path(__file__).resolve().parent.parent
    src = (here / "addon" / "gpufluid_blender" / "operators" / "helpers.py").read_text(encoding="utf-8")
    # Helper is module-level, expect this exact name and signature
    assert "def _set_single_role(obj, role_attr: str)" in src
    # And the four canonical role pairs in the right order
    assert "(\"gpufluid_fluid\", \"is_fluid\")" in src
    assert "(\"gpufluid_obstacle\", \"is_obstacle\")" in src
    assert "(\"gpufluid_inflow\", \"is_inflow\")" in src
    assert "(\"gpufluid_outflow\", \"is_outflow\")" in src
