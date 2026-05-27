"""[Layer S2.17.PATCH] Runtime overlay patches for upstream warp-mpm.

We do NOT fork warp-mpm — instead we monkey-patch two specific kernel
sources at import time. This way upgrading warp-mpm is a `git pull` plus
a re-run of :func:`apply_patches`, not a merge conflict.

Two patches:
    S2.17.PATCH.SLIP — fix slip-surface velocity-zero bug in
        `mpm_solver_warp.add_surface_collider`. The upstream kernel
        computes the projected velocity `v` correctly then overwrites
        with `wp.vec3(0,0,0)` on the final line, so "slip" behaves
        identical to "sticky". We patch the source to write `v`.

    S2.17.PATCH.EOS — clamp `J` in `mpm_utils.kirchoff_stress_water` so
        that `J^{-1.1}` does not blow up to +∞ when a particle is
        numerically overcompressed at a rigid collider. `J_safe = max(J, 0.5)`.

Both patches edit the kernel source files in-place before Warp's JIT
compiler reads them. Idempotent: running twice is a no-op (we detect
the marker `# [S2.17.PATCH.*]` and skip).
"""
from __future__ import annotations

import re
from pathlib import Path

from ...blocks import block

# Marker comments we insert after patching to make idempotent
_SLIP_MARKER = "# [S2.17.PATCH.SLIP applied]"
_EOS_MARKER = "# [S2.17.PATCH.EOS applied]"


def _find_warp_mpm_root() -> Path | None:
    """Locate `third_party/warp-mpm/` by walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "third_party" / "warp-mpm"
        if (candidate / "mpm_solver_warp.py").exists():
            return candidate
    return None


@block("S2.17.PATCH.SLIP",
       "Overlay fix: write projected v in slip-surface branch instead of zero")
def _patch_slip(solver_path: Path) -> bool:
    src = solver_path.read_text(encoding="utf-8")
    if _SLIP_MARKER in src:
        return False  # already patched
    # The bug: in the `else` (slip/separate) branch the final two lines
    # write `wp.vec3(0.0, 0.0, 0.0)` despite having computed a real `v`.
    pattern = re.compile(
        r"(if normal_component < 0\.0 and wp\.length\(v\) > 1e-20:.*?"
        r"\* wp\.normalize\(\s*v\s*\)\s*#\s*apply friction here\s*\n\s+)"
        r"state\.grid_v_out\[grid_x, grid_y, grid_z\] = wp\.vec3\(\s*"
        r"0\.0,\s*0\.0,\s*0\.0\s*\)",
        re.DOTALL,
    )
    new_src, n = pattern.subn(
        lambda m: m.group(1) + f"state.grid_v_out[grid_x, grid_y, grid_z] = v  {_SLIP_MARKER}",
        src,
    )
    if n == 0:
        # Already patched in a different form? Check for the fixed pattern.
        if "state.grid_v_out[grid_x, grid_y, grid_z] = v" in src:
            # Just stamp the marker so future calls are idempotent.
            new_src = src.replace(
                "state.grid_v_out[grid_x, grid_y, grid_z] = v",
                f"state.grid_v_out[grid_x, grid_y, grid_z] = v  {_SLIP_MARKER}",
                1,
            )
        else:
            raise RuntimeError(
                "S2.17.PATCH.SLIP: could not locate the slip-branch zero "
                f"write in {solver_path}. warp-mpm version may have changed."
            )
    # Round-38: tolerate read-only install (system site-packages, pipx,
    # Nix store, restricted CI worker). Pre-round-38 an OSError here
    # propagated out of `apply_patches` → `_warp_mpm_imports.py:20` →
    # entire MPM path crashed on import with an opaque OS error. Now:
    # log + return False so the package keeps loading (the kernel patch
    # is best-effort; the rest of MPM works without it).
    try:
        solver_path.write_text(new_src, encoding="utf-8")
    except OSError as exc:
        import logging as _log
        _log.getLogger("gpufluid.mpm.patches").warning(
            "S2.17.PATCH.SLIP: cannot write to %s (%s) — install is "
            "read-only? Patch skipped; bake may show pre-patch slip "
            "behaviour.", solver_path, exc)
        return False
    return True


@block("S2.17.PATCH.EOS",
       "Overlay: clamp J in kirchoff_stress_water to bound pressure spike at rigid contact")
def _patch_eos(utils_path: Path) -> bool:
    src = utils_path.read_text(encoding="utf-8")
    if _EOS_MARKER in src:
        return False
    # Look for: pressure = -bulk * (wp.pow(J, -gamma) - 1.)
    # We want:   pressure = -bulk * (wp.pow(wp.max(J, 0.5), -gamma) - 1.)
    pattern = re.compile(
        r"pressure = -bulk \* \(wp\.pow\(\s*J\s*,\s*-gamma\s*\) - 1\.?\)"
    )
    replacement = (
        f"J_safe = wp.max(J, 0.5)  {_EOS_MARKER}\n"
        "    pressure = -bulk * (wp.pow(J_safe, -gamma) - 1.)"
    )
    new_src, n = pattern.subn(replacement, src)
    if n == 0:
        # Maybe already patched (J_safe present); stamp marker idempotently
        if "wp.max(J, 0.5)" in src:
            new_src = src.replace("wp.max(J, 0.5)",
                                  f"wp.max(J, 0.5)  {_EOS_MARKER}", 1)
        else:
            raise RuntimeError(
                "S2.17.PATCH.EOS: could not locate kirchoff_stress_water "
                f"pressure line in {utils_path}."
            )
    # Round-38: same read-only tolerance as the SLIP patch above.
    try:
        utils_path.write_text(new_src, encoding="utf-8")
    except OSError as exc:
        import logging as _log
        _log.getLogger("gpufluid.mpm.patches").warning(
            "S2.17.PATCH.EOS: cannot write to %s (%s) — install is "
            "read-only? Patch skipped; bake may show pressure spikes "
            "at rigid contact.", utils_path, exc)
        return False
    return True


def apply_patches(warp_mpm_root: Path | str | None = None) -> dict:
    """Apply both upstream patches. Idempotent.

    Parameters
    ----------
    warp_mpm_root : Path-like, optional
        Override location of warp-mpm. Default: auto-discover under
        `<project_root>/third_party/warp-mpm/`.

    Returns
    -------
    dict
        ``{"slip": applied_bool, "eos": applied_bool, "root": Path}`` —
        boolean ``True`` means the patch was applied now (False = was
        already in place).
    """
    if warp_mpm_root is None:
        root = _find_warp_mpm_root()
        if root is None:
            raise RuntimeError(
                "apply_patches: could not locate third_party/warp-mpm/. "
                "Pass warp_mpm_root explicitly."
            )
    else:
        root = Path(warp_mpm_root)
    slip = _patch_slip(root / "mpm_solver_warp.py")
    eos = _patch_eos(root / "mpm_utils.py")
    return {"slip": slip, "eos": eos, "root": root}
