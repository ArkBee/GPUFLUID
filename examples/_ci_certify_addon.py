"""Single-command production-readiness gate for the gpufluid addon.

Run from the repo root:
    .venv/Scripts/python.exe examples/_ci_certify_addon.py

What it does:
  1. Runs the 48-case pytest suite for the addon (six test files).
  2. Runs the headless CI bake+render harness under Blender -b -P.
  3. Runs the 100-frame stress harness (also Blender -b -P).
  4. Greps the addon source tree for forbidden patterns
     (bare `print(`, leftover `bpy.ops.outliner.orphans_purge` calls,
     `_is_running = True` not on the class, etc.).
  5. Greps docs to confirm the round 1-11 contract is documented
     (sync mode, ESC abort, reentrance guard, A8.13 in BLOCKS, etc.).
  6. Checks the HEAD commit follows Conventional Commits format.

Exit 0 only if every check is green. Writes a `certification_report.md`
to the repo root with per-step result + timing. Designed as the
"is this addon shippable today?" oracle — single command, single
PASS/FAIL, no judgement calls left to the operator.

Usage from CI:
    .venv/Scripts/python.exe examples/_ci_certify_addon.py || exit 1

This is intentionally NOT a pytest — pytest can't drive Blender via
subprocess in the same run cleanly. Wrap it in a CI step.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
ADDON_DIR = REPO / "addon" / "gpufluid_blender"
DOCS = REPO / "docs"

# Colourisation for terminals that support it.
_RED, _GRN, _YLW, _RST = "\033[31m", "\033[32m", "\033[33m", "\033[0m"
if not sys.stdout.isatty() or os.name == "nt":
    _RED = _GRN = _YLW = _RST = ""


@dataclass
class StepResult:
    name: str
    passed: bool
    duration_s: float
    detail: str = ""
    notes: List[str] = field(default_factory=list)


# ─── Step implementations ───────────────────────────────────────────────

def step_unit_tests() -> StepResult:
    """All 6 addon pytest files, fail-fast."""
    t0 = time.time()
    files = [
        "tests/test_addon_schema_roundtrip.py",
        "tests/test_addon_preload_lru.py",
        "tests/test_render_bridge_payload.py",
        "tests/test_no_layer_exceptions.py",
        "tests/test_addon_role_single.py",
        "tests/test_addon_round8_regressions.py",
        "tests/test_preload_cache_invariants.py",
        "tests/test_scene_dict_validator.py",
        "tests/test_domain_transform.py",
        "tests/test_scene_validator.py",
        "tests/test_addon_root_pkg.py",
        "tests/test_cache_binding.py",
    ]
    cmd = [str(VENV_PY), "-m", "pytest", "-x", "--tb=short", *files]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    dt = time.time() - t0
    last_line = (p.stdout.strip().splitlines() or ["?"])[-1]
    return StepResult(
        name="unit_tests",
        passed=(p.returncode == 0),
        duration_s=dt,
        detail=last_line,
        notes=([] if p.returncode == 0 else [p.stdout[-2000:], p.stderr[-1000:]]),
    )


def step_headless_ci() -> StepResult:
    """`blender -b -P examples/_ci_headless_bake.py` — full pipeline."""
    t0 = time.time()
    import tempfile
    cache = Path(tempfile.gettempdir()) / "gpufluid_certify_ci"
    if cache.exists():
        import shutil
        shutil.rmtree(cache, ignore_errors=True)
    cmd = [str(BLENDER), "-b", "-P",
           str(REPO / "examples" / "_ci_headless_bake.py"),
           "--", str(cache)]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    dt = time.time() - t0
    ci_lines = [ln for ln in p.stderr.splitlines() if ln.startswith("[CI-")]
    passed = any("[CI-PASS]" in ln for ln in ci_lines) and p.returncode == 0
    return StepResult(
        name="headless_ci",
        passed=passed,
        duration_s=dt,
        detail=ci_lines[-1] if ci_lines else "no [CI-*] markers",
        notes=ci_lines,
    )


def step_stress() -> StepResult:
    """100-frame stress harness."""
    t0 = time.time()
    import tempfile
    cache = Path(tempfile.gettempdir()) / "gpufluid_certify_stress"
    if cache.exists():
        import shutil
        shutil.rmtree(cache, ignore_errors=True)
    cmd = [str(BLENDER), "-b", "-P",
           str(REPO / "examples" / "_ci_stress_bake.py"),
           "--", str(cache)]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    dt = time.time() - t0
    s_lines = [ln for ln in p.stderr.splitlines() if ln.startswith("[STRESS-")]
    passed = any("[STRESS-PASS]" in ln for ln in s_lines) and p.returncode == 0
    return StepResult(
        name="stress_100frame",
        passed=passed,
        duration_s=dt,
        detail=s_lines[-1] if s_lines else "no [STRESS-*] markers",
        notes=s_lines,
    )


def step_forbidden_patterns() -> StepResult:
    """Grep the addon for patterns that round 1-11 outlawed."""
    t0 = time.time()
    forbidden = [
        # CLAUDE.md §3 rule: no print() in addon (logger only).
        # Exception: explicit `[bake]`/`[render]` markers in sync-mode
        # subprocess drain — those are stdout passthrough by design so
        # CLI callers see progress (bake.py:496-ish, render.py:235-ish).
        (r"^\s*print\(",
         "bare print( in addon source",
         lambda line: ("[bake]" in line or "[render]" in line)),
        # Round-5 fix: orphans_purge was replaced with bpy.data.cache_files.remove
        (r"bpy\.ops\.outliner\.orphans_purge",
         "orphans_purge ops call (deprecated)",
         # Allow inside comments (we explain what was replaced).
         lambda line: line.lstrip().startswith("#")),
        # Round-6 fix: _is_running must be class-level, never instance
        (r"self\._is_running\s*=",
         "_is_running assigned on instance (must be class-level)",
         None),
    ]
    findings: list[str] = []
    for py in ADDON_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        for pattern, label, allow_if in forbidden:
            for n, line in enumerate(text.splitlines(), 1):
                if re.search(pattern, line):
                    if allow_if is not None and allow_if(line):
                        continue
                    findings.append(f"{py.relative_to(REPO)}:{n}: {label}: {line.strip()}")
    return StepResult(
        name="forbidden_patterns",
        passed=(not findings),
        duration_s=time.time() - t0,
        detail=f"{len(findings)} hits" if findings else "0 hits across addon source",
        notes=findings[:20],
    )


def step_docs_contract() -> StepResult:
    """Greps confirming round 1-11 contract is documented."""
    t0 = time.time()
    required = [
        (DOCS / "BLOCKS.md", "A8.13", "render operator block ID missing from BLOCKS"),
        (DOCS / "BLOCKS.md", "addon/gpufluid_blender/operators/render.py",
         "render.py source path missing in BLOCKS A8 table"),
        (DOCS / "DESIGN.md", "A8.13", "render operator missing from DESIGN A8 table"),
        (DOCS / "HANDOFF.md", "cache_loader/",
         "cache_loader as package not reflected in HANDOFF dir layout"),
        (DOCS / "HANDOFF.md", "sync mode",
         "sync mode not described in HANDOFF block-registry"),
        (DOCS / "QUICKSTART.md", "Esc",
         "ESC cancel not documented in QUICKSTART"),
        (DOCS / "QUICKSTART.md", "sync=True",
         "sync mode opt-in not documented in QUICKSTART"),
        (DOCS / "QUICKSTART.md", "TOML overrides",
         "overrides escape-hatch not documented in QUICKSTART"),
    ]
    missing = []
    for path, needle, msg in required:
        if not path.exists():
            missing.append(f"{path.relative_to(REPO)}: file missing — {msg}")
            continue
        if needle not in path.read_text(encoding="utf-8", errors="replace"):
            missing.append(f"{path.relative_to(REPO)}: missing '{needle}' — {msg}")
    return StepResult(
        name="docs_contract",
        passed=(not missing),
        duration_s=time.time() - t0,
        detail=f"{len(missing)} doc gaps" if missing else f"{len(required)}/{len(required)} contracts present",
        notes=missing,
    )


def step_head_commit_format() -> StepResult:
    """HEAD commit message must follow Conventional Commits."""
    t0 = time.time()
    p = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=REPO,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
    subj = p.stdout.strip()
    # Conventional Commits with one OR multiple scopes (comma-separated
    # OR `+`-joined which we use for cross-cutting commits like
    # `fix(addon)+docs: ...`).
    pattern = r"^(feat|fix|refactor|chore|docs|test|perf|build|ci|revert)" \
              r"(\([^)]+\)(\+(feat|fix|refactor|chore|docs|test|perf|build|ci|revert)(\([^)]+\))?)*)?" \
              r"(!)?:\s.+"
    passed = bool(re.match(pattern, subj))
    return StepResult(
        name="head_commit_format",
        passed=passed,
        duration_s=time.time() - t0,
        detail=subj[:120],
        notes=[] if passed else [
            "Conventional Commits required: feat(scope): ... / fix(scope): ... / etc."],
    )


# ─── Orchestrator ──────────────────────────────────────────────────────

STEPS = [
    step_unit_tests,
    step_forbidden_patterns,
    step_docs_contract,
    step_head_commit_format,
    step_headless_ci,
    step_stress,
]


def main() -> int:
    if not VENV_PY.exists():
        print(f"[CERT-FAIL] venv python missing: {VENV_PY}", file=sys.stderr)
        return 1
    if not BLENDER.exists():
        print(f"[CERT-FAIL] Blender binary missing: {BLENDER}", file=sys.stderr)
        return 1

    # Force UTF-8 on Windows where the default cp1251 codec chokes on
    # the dash + box-drawing chars used below.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(f"\n{'-'*72}\ngpufluid addon - production-readiness certification\n{'-'*72}\n")
    results: list[StepResult] = []
    for step_fn in STEPS:
        name = step_fn.__name__.replace("step_", "")
        print(f"→ {name} ... ", end="", flush=True)
        try:
            r = step_fn()
        except Exception as e:
            r = StepResult(name=name, passed=False, duration_s=0.0,
                           detail=f"exception: {e}", notes=[str(e)])
        colour = _GRN if r.passed else _RED
        tag = "PASS" if r.passed else "FAIL"
        print(f"{colour}{tag}{_RST} ({r.duration_s:.1f}s) — {r.detail}")
        results.append(r)
        if not r.passed:
            for ln in r.notes[:5]:
                print(f"    {_YLW}{ln}{_RST}")

    n_pass = sum(1 for r in results if r.passed)
    print(f"\n{n_pass}/{len(results)} steps green")

    # Write report
    report = ["# gpufluid addon — certification report",
              f"_Generated by `examples/_ci_certify_addon.py` at "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')}_\n",
              f"**Result: {n_pass}/{len(results)} steps green**\n",
              "| Step | Result | Duration | Detail |",
              "|------|--------|----------|--------|"]
    for r in results:
        tag = "✅ PASS" if r.passed else "❌ FAIL"
        # Pre-compute outside the f-string to dodge Py3.11 backslash-in-fstring.
        detail_safe = r.detail.replace("|", r"\|")
        report.append(f"| {r.name} | {tag} | {r.duration_s:.1f}s | "
                      f"{detail_safe} |")
    if any(not r.passed for r in results):
        report.append("\n## Failures detail\n")
        for r in results:
            if not r.passed:
                report.append(f"### {r.name}\n")
                for ln in r.notes:
                    report.append(f"- `{ln}`")
                report.append("")
    (REPO / "certification_report.md").write_text(
        "\n".join(report), encoding="utf-8")
    print(f"Report: {REPO / 'certification_report.md'}")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
