# Addon-audit branch — session handoff (last updated 2026-05-27, round-26)

> **Read this file first** if you're continuing work on the addon
> audit / refactor sprint. Self-contained — does NOT depend on the
> wider `HANDOFF.md` context. Last verified PASS: **cert 6/6 + 126
> unit tests + headless CI + stress @ 0.39ms/scrub-frame**.
>
> **Status as of round-19: every architectural-debt item the senior
> architect agent flagged in round-12 is closed.** Branch is ready
> to merge — see § "Pre-merge checklist" below.

---

## TL;DR

Branch `fix/addon-audit-fixes` is **42 commits** beyond `main`,
covering **19 rounds** of addon audit + senior debt closure.

All three senior architect's day-1 rewrites + all five code-smells
**shipped**. Production gate (cert harness) **green**. Last 3
reviewers (rounds 17/18 — 9th + 10th) each found 0 critical bugs.
Diminishing-returns curve has hit bottom.

Single-command verify the branch is shippable:

```bash
.venv/Scripts/python.exe examples/_ci_certify_addon.py
# expect: 6/6 steps green, ~35s wall (Blender startup dominates)
```

Single-command run unit tests only:

```bash
.venv/Scripts/python.exe -m pytest tests/test_scene_dict_validator.py \
  tests/test_addon_schema_roundtrip.py tests/test_a8_config_builder.py \
  tests/test_preload_cache_invariants.py tests/test_addon_round8_regressions.py \
  tests/test_addon_role_single.py tests/test_addon_preload_lru.py \
  tests/test_render_bridge_payload.py tests/test_no_layer_exceptions.py
# expect: 93 passed in ~2s
```

If either fails on a fresh clone: read § "Common failure modes" below.

---

## Where we are

| Phase | Status | Round | Commit |
|-------|--------|-------|--------|
| Phase 1: UI↔TOML↔core contract | ✅ | 1 | `2754ff5` |
| Phase 2: bake bugs + preload LRU | ✅ | 2 | `7bdd733` |
| Phase 3: render bridge + OT_render | ✅ | 3 | `83d3882` |
| Phase 4: hygiene + file splits | ✅ | 4 | `1b99c02 + f9e7e26` |
| Live tests (11 scenarios, MCP) | ✅ | 6-12 | various |
| Headless CI smoke harness | ✅ | 12 | `87bacc2` |
| Stress test harness | ✅ | 10 | `cf20a63` |
| Certification harness (single-cmd) | ✅ | 12 | `011c600` |
| Lessons doc (10 rules) | ✅ | 9 | `~/.claude/CLAUDE.md §9` |
| Senior day-1 #1: PreloadCache | ✅ | 13 | `ae639ba` |
| Senior day-1 #2: ModalSubprocessRunner | ✅ | 14 | `12d2cd2` |
| Senior day-1 #3-lite: SceneDict + validator | ✅ | 15-16 | `d0531d2 + 6ba47b2` |
| Docs sync (BLOCKS/DESIGN/HANDOFF/QUICKSTART) | ✅ | 11 | `0a2275d` |
| Senior code-smell #2: collect_scene SRP | ✅ | 17 | `7f7d3ab` |
| Senior code-smell #5: ADDON_PKG centralised | ✅ | 18 | `76744b7` |
| Senior code-smell #4: cache_binding helpers | ✅ | 19 | `a142484` |

**Last commit on branch:** `a142484` (round-19 — cache_binding helpers, last senior code-smell closed).

### Senior architect's full debt list — 8/8 closed

The round-12 staff-grade review flagged 3 day-1 rewrites + 5 code-smells.
All eight items shipped:

| # | Item | Round | Notes |
|---|------|-------|-------|
| Day-1 #1 | `PreloadCache` class (was module-global `_PRELOAD`) | 13 | back-compat dict shims kept |
| Day-1 #2 | `ModalSubprocessRunner` (was bake/render dup) | 14 | both ops now thin shells |
| Day-1 #3 | typed `SceneDict` + validator (was `Dict[str, Any]`) | 15-16 | tomli_w deferred |
| Smell #1 | `_PRELOAD` mutability surface (subsumed by day-1 #1) | 13 | — |
| Smell #2 | `bake.collect_scene` SRP (250-line mix) | 17 | → `DomainTransform` + `scene_validator` |
| Smell #3 | Subprocess lifecycle dup (subsumed by day-1 #2) | 14 | — |
| Smell #4 | Custom-props as data bus (~10 magic strings) | 19 | → `cache_binding.py` |
| Smell #5 | `addon_root_pkg()` rsplit fragility | 18 | → `ADDON_PKG` const at addon root |

---

## Open work (ranked by ROI, pick top of list)

> Updated post-round-19: most of the original open-work list is now
> SHIPPED. Remaining items are smaller / lower-impact than the day-1 +
> code-smell sprint that just landed.

### 1. **Merge to `main` (recommended next step)**

**Why:** cert 6/6, 126 unit tests, headless CI green, all 8 senior
debt items closed, last 3 reviewer rounds found 0 critical bugs.
Diminishing-returns curve has hit bottom. Holding the branch open
costs context for the next contributor without buying more quality.

**Pre-merge checklist:**
- [ ] Cert script PASS on a fresh checkout (someone else's machine):
  `.venv/Scripts/python.exe examples/_ci_certify_addon.py` → 6/6
- [ ] `git log main..HEAD --stat` reviewed for surprises (none
  expected — every commit ties to a rounds-1..19 narrative)
- [ ] Tag the merge commit (suggest `addon-audit-2026-05-26-round19`)
  for archaeology — round-N comments throughout reference each other
- [ ] Update `docs/HANDOFF.md §5` A8 row to point at the merge tag
- [ ] After merge: this file (`HANDOFF_addon_audit.md`) can be deleted
  or moved to `docs/_archive/` — its job is done

### 2. Vendor `tomli_w` for full day-1 #3 (only outstanding deferred)

**Why:** day-1 #3 shipped as "lite" — kept hand-rolled
`_emit_scalar`/`_emit_table` (post-rounds 8-11 hardening). Senior
wanted full delete in favour of `tomli_w`. The emitter is still the
highest-bug-density file by line. Vendoring ~300 lines of `tomli_w`
(MIT, 0-dep) would let us delete those round-1..16 hardening lines.

**Plan:**
- Drop `tomli_w` source into `addon/gpufluid_blender/_vendor/tomli_w/`.
- Verify `addon/gpufluid_blender/blender_manifest.toml` handles
  vendor declarations correctly (Blender 4.2+ extension format).
- In `config_builder.build_toml`, replace `_emit_toml(d)` →
  `tomli_w.dumps(d)`.
- Delete `_emit_scalar` / `_emit_table` / `_emit_with_key` /
  `_is_table` / `_is_array_of_tables`.
- Keep `_deep_merge` (overrides path) and `validate_scene_dict`
  (round-15/16 typed contract).
- Run cert. Expect: same TOML output. `test_addon_schema_roundtrip.py`
  is the gate — if CLI parses the new TOML identically, ship.

**Time:** ~30-60 min. Risk: medium — emit format may differ slightly
(spacing, list multi-line) and CLI might surprise. Test thoroughly.

**Skip-criterion:** if `tomli_w` formatting causes CLI grief, fall
back to current emitter and file as permanent BACKLOG.

### 3. 11th reviewer round on rounds 17-19 (low value)

**Why:** rounds 17-19 each had their own reviewer (10th was on
round-17). A combined sweep on the three together might find drift
between them — but each round was independently reviewed clean, so
the joint surface is unlikely to hide much. Reviewer-finding curve
is at 0/0/0 for last three.

**When to do:** if you want belt-and-braces before merge. Otherwise
skip; cert green is sufficient.

### 4. New scope (deferred)

Out of round-19 scope but listed for awareness:

- **Core MPM truncation fix** (filed in `docs/BACKLOG.md`): root
  cause identified at `src/gpufluid/sim/mpm/solver.py:469` (silent
  `break` on NaN-divergence). Addon side already mitigated; core
  needs a typed `MpmDivergenceError` + `cache.json` truncation
  marker. **Out of addon-branch scope.**
- **Render scene with real obstacles**: `render_fluid_on_cube_eevee.py`
  hardcodes the cube. BACKLOG.
- **Addon mesh-fill export pipeline**: `fill_mesh` prop hidden in UI
  pending real `.obj` runtime export. BACKLOG.

---

## Key files / landmarks

```
addon/gpufluid_blender/
├── __init__.py                   # register/unregister + @persistent handlers
│                                 #   ★ Round-18: ADDON_PKG constant at root
├── scene_dict.py                 # ★ Round-15/16: TypedDict + validate_scene_dict
├── cache_binding.py              # ★ Round-19: single source for ~10 magic-string keys
├── domain_transform.py           # ★ Round-17: pure world↔[0,1]³ math dataclass
├── scene_validator.py            # ★ Round-17: out_of_domain_warning pure fn
├── config_builder.py             # build_toml + _emit_table (calls validator)
├── cache_loader/
│   ├── __init__.py               # ★ Round-13: PreloadCache class
│   │                             #   uses cache_binding for prop reads (round-19)
│   ├── _ply.py                   # PLY reader
│   └── _ops.py                   # OT_attach_cache/OT_attach_ww/OT_detach
│                                 #   uses cache_binding.{set,clear}_* (round-19)
├── operators/
│   ├── _runner.py                # ★ Round-14: ModalSubprocessRunner
│   ├── bake.py                   # OT_bake — uses runner + cache_binding + DomainTransform
│   ├── render.py                 # OT_render — uses runner + cache_binding
│   ├── helpers.py                # mark_* + clear + Eevee preset + drain
│   ├── _animation.py             # F-curve helpers (Phase 4 split)
│   └── _collect.py               # _output_dict + _export_obj (small utils)
├── render_bridge.py              # FrameMeshLoader (A8.10) + Eevee preset (A8.9)
├── preferences.py                # interpreter_path + LRU caps + ADDON_PKG (round-18)
└── properties.py                 # Domain/Fluid/Obstacle/Inflow/Outflow/WW PGs

tests/
├── test_scene_dict_validator.py        # ★ Round-15/16: 25 cases (plural keys)
├── test_preload_cache_invariants.py    # ★ Round-13: 10 cases
├── test_addon_round8_regressions.py    # ★ Round-8..14: lifecycle source-grep
├── test_addon_role_single.py           # Round-5: mark_* single-role
├── test_addon_schema_roundtrip.py      # Round-1: contract test
├── test_addon_preload_lru.py           # Round-2: LRU cap behaviour
├── test_render_bridge_payload.py       # Round-3: render argv contract
├── test_a8_config_builder.py           # pre-existing
├── test_no_layer_exceptions.py         # F3.6 layer-boundary gate
├── test_domain_transform.py            # ★ Round-17: 22 cases (math invariants)
├── test_scene_validator.py             # ★ Round-17: 9 cases (warning thresholds)
├── test_addon_root_pkg.py              # ★ Round-18: 4 source-grep gates
└── test_cache_binding.py               # ★ Round-19: 7 cases incl. magic-string gate

examples/
├── _ci_certify_addon.py          # ★ Round-12/15/17/18/19: single-cmd production gate
├── _ci_headless_bake.py          # ★ Round-12: blender -b bake+render smoke
├── _ci_stress_bake.py            # ★ Round-10/12: 100-frame scrub @ <50ms gate
└── render_fluid_on_cube_eevee.py # Eevee preset renderer (round-10 Col attr fix)

docs/
├── BACKLOG.md                    # tier 4 + deferred items
├── HANDOFF.md                    # wider project context
├── HANDOFF_addon_audit.md        # ★ THIS file
├── QUICKSTART.md                 # user-facing (round-11 sync mode added)
├── DESIGN.md §11                 # A8 table (mostly synced)
└── BLOCKS.md                     # A8 block IDs (round-10 synced)
```

★ = created or majorly rewritten during this sprint.

---

## Architecture state after refactors

- **PreloadCache** (round-13) is the singleton for ALL preload mutations.
  Back-compat shims (`_PRELOAD`, `_free_table`, `_lru_touch`,
  `_lru_install`, `_prune_stale`) at module level delegate to the
  singleton — old call sites still work, new code uses class methods.

- **ModalSubprocessRunner** (round-14) owns ALL subprocess lifecycle.
  Both `OT_bake` and `OT_render` lazy-init `self._runner` in
  `execute()`, call `runner.start_sync(...)` OR `runner.start_modal(...)`,
  delegate `modal()` to `runner.tick_modal(...)` and `cancel()` to
  `runner.cancel(...)`. `_is_running` stays on each OT class as
  cross-instance reentrance lock.

- **SceneDict TypedDict** + **validate_scene_dict** (round-15/16)
  documents the scene_dict shape (build_toml input). Validator
  semantics: **"validate what's there, don't require completeness"** —
  sections present must have correct shape; absent sections allowed
  (test fixtures pre-round-15 rely on this). Uses **plural keys**
  (`obstacles`, `inflows`, `outflows`, `fluids`) per the
  collect_scene convention — round-16 fixed singular-key theater.

- **DomainTransform + scene_validator** (round-17) — pure (bpy-free)
  math for world↔[0,1]³ + out-of-domain warnings. Extracted from
  `bake.collect_scene` per senior code-smell #2. Unit-testable
  without Blender. `bake.collect_scene` now glues bpy iteration +
  domain transform + validator — single-responsibility per module.

- **ADDON_PKG** constant (round-18) declared at addon root in
  `__init__.py::ADDON_PKG: str = __package__`. Every preferences
  lookup (`cache_loader/__init__.py`, `preferences.py`) imports it.
  Eliminates the `__package__.rsplit('.', 1)[0]` "works by
  coincidence" pattern that round-1 silently miss-keyed on the
  Blender 4.2+ extension namespace.

- **cache_binding module** (round-19) is the single source for ~10
  custom-property magic strings (`gpufluid_cache_dir`,
  `gpufluid_cache_origin`, ..., `gpufluid_origin`,
  `gpufluid_dom_size`, ...). Three name groups (cache / whitewater /
  bake-trace) with subtly-different prefixes — round-5 reviewer
  found a missing-key bug born from exactly that asymmetry. Now
  every consumer goes through `cache_binding.get_*` / `set_*` /
  `clear_all_bindings`. Source-grep test enforces magic strings
  appear nowhere else in addon source.

---

## BACKLOG items still open (filed entries)

Listed in `docs/BACKLOG.md`:

1. **Senior architectural debt** (round-12 staff review) — 5 items
   from `cache_loader._PRELOAD` (done — round-13) through
   `subprocess lifecycle duplicated` (done — round-14) to
   `Dict[str, Any]` (done — round-15-lite) PLUS:
   - `bake.collect_scene` SRP violation — **open** (item #3 above)
   - Custom-properties as inter-op data bus — **open**
   - `addon_root_pkg()` discovery via rsplit — **open**

2. **Core MPM truncation at high res** — root cause traced
   (`solver.py:469` deliberate `break` on NaN-divergence,
   only `print()`, no exception). Two core fixes proposed
   (typed `MpmDivergenceError` + `truncated_at_frame` in
   `cache.json`). Out of addon-branch scope.

3. **Renderer respects mesh Col attribute** — ✅ SHIPPED round-10
   (kept in BACKLOG with status = shipped commit ref for traceability).

4. **TOML overrides drops table-valued fields in `[[array]]`**
   — ✅ SHIPPED round-10.

5. **Addon mesh-fill export pipeline** — `fill_mesh` prop hidden
   in UI (Phase 1), real `.obj` runtime export still TODO.

6. **Render scene with real obstacles** — `render_fluid_on_cube_eevee.py`
   hardcodes the cube. Filed as future work.

---

## Lessons from this sprint

**Codified at `~/.claude/CLAUDE.md §9`** (10 rules — auto-loads in every
session). Read them BEFORE starting on round-17. Key ones:

- §9.1 — Self-review is blind; spawn `*-reviewer` agent after ≥3
  related commits.
- §9.2 — "Tests: N/N green" ≠ "new code works". Round-15/16 was the
  archetype — added validator, claimed coverage, reviewer found the
  whole thing was theater because tests used different keys than
  production.
- §9.3 — claim_completion ≠ fix_done. Always grep before saying done.
- §9.4 — Symptom vs root cause. Test timeouts? Fix the API, not the
  test. Sync mode is the canonical example.
- §9.5 — Hot-path defensive code isn't free. Round-7 `_prune_stale`
  in frame_change at 120k ops/sec.
- §9.6 — Mirror operators drift. ModalSubprocessRunner is the
  prevention.
- §9.7 — Mock fidelity. Reviewer-flagged that `_FakeMeshesCollection`
  doesn't fully simulate StructRNA invalidation; low risk currently.
- §9.8 — Order of operations in state machines. `_lru_install`
  pre-swap pattern.
- §9.9 — When to stop digging: headless CI + reviewer + unit cov +
  no hot-path tail.
- §9.10 — Silent fallback = silent bug. Log on every fallback path.

**New for round-16:** when introducing a refactor that claims safety
(validator, type check, contract enforcement), **verify against the
real production call site** with grep, not synthetic test fixtures.
Reviewer found the singular/plural bug in 2 minutes; 20 unit tests
missed it for a commit.

---

## Common failure modes (for the next person)

### Cert script fails on `forbidden_patterns`

Round-12 gate disallows `print(`, `bpy.ops.outliner.orphans_purge`,
and instance-level `self._is_running = True` in `addon/gpufluid_blender/`.
Allow-list:
- `print("[bake] ...")`, `print("[render] ...")` in subprocess
  drain — explicit stdout passthrough by design.
- `orphans_purge` mentioned in comments documenting what was replaced.

If you genuinely need to add a new `print()`, route via the addon
logger (`from .. import logger`) instead.

### Cert script fails on `docs_contract`

The script greps for 8 specific anchors in BLOCKS/DESIGN/HANDOFF/
QUICKSTART (e.g. "A8.13" in BLOCKS, "sync mode" in HANDOFF,
"Esc" in QUICKSTART). If you remove a documented contract, also
update the gate's `required` list at `examples/_ci_certify_addon.py`.

### Headless CI fails to find Blender

The cert + stress + headless scripts hard-code
`C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`.
On a different machine, edit the `BLENDER` constant at the top
of `examples/_ci_certify_addon.py` and `examples/_ci_headless_bake.py`
+ `examples/_ci_stress_bake.py`. Should be a CLI flag — filed
mentally as polish.

### MCP screenshot tool returns stale image

Round-3 confirmed this is a Blender-MCP plugin bug (PNG capture
returns wrong frame, but `get_screenshot_of_window_as_json`
reports correct live state). Use JSON for state inspection;
distrust PNG for visual verification when something looks weird.

### `_PRELOAD` shows dead meshes after live testing

You disabled+re-enabled the addon and `bpy.data.meshes.remove`
freed the StructRNAs, but `_PRELOAD` dict still has references.
On next `_frame_change_handler` tick, the swap raises
ReferenceError which triggers `_PRELOAD.invalidate(name)` —
self-healing. If it crashes Blender instead, the
`_free_table_static` defensive eager-name probe in
`cache_loader/__init__.py` is the place to look.

---

## Don't-touch list

1. **`_runner.py::_clear_instance_state`** — round-7..10..14 invariant
   that abort/_finish both call it. Test
   `test_runner_finish_and_abort_clean_instance_state` asserts. Don't
   skip cleanup branches.

2. **PreloadCache.attach pre-swap** — without the `obj.data = next(iter(
   table.values()))` BEFORE freeing old, every re-attach leaks the
   current-frame mesh. Test `test_attach_replace_pre_swap_releases_old_current_mesh`.

3. **`@persistent` on `_frame_change_handler` and `_on_load_post`** —
   Blender wipes non-persistent handlers on file load. Without these,
   scrubbing a saved-then-reopened .blend leaves the cache frozen.

4. **Bake reentrance flag set BEFORE Popen** — round-6 race fix.
   Test `test_bake_modal_popen_wrapped_in_try_oserror` asserts.

5. **Validator uses plural keys** — round-16 fix. Don't "normalise"
   to singular without also changing the 4 call sites in
   `bake.collect_scene`.

---

## Useful one-liners

```bash
# Branch state
git log --oneline main..HEAD | wc -l         # commit count (was 37)
git diff main..HEAD --stat | tail -1         # total diff size

# Run only the round-N tests (e.g. round-15+16)
.venv/Scripts/python.exe -m pytest tests/test_scene_dict_validator.py -v

# Live test in Blender via MCP — full sync bake + render in one call
# (see HANDOFF_addon_audit.md § "Open work" item 1 for the prompt)

# Sync extension dir with repo (after edits to addon/)
EXT="/c/Users/timof/AppData/Roaming/Blender Foundation/Blender/5.1/extensions/user_default/gpufluid_blender"
cp -r addon/gpufluid_blender/. "$EXT/"
find "$EXT" -name __pycache__ -type d -exec rm -rf {} +

# Reload addon in Blender (via MCP)
# bpy.ops to disable+enable, plus sys.modules cleanup —
# see any of my MCP exec blocks for the pattern.
```

---

## Reviewer history (so you don't re-litigate)

| Round | Reviewer focus | Real bugs | Cosmetic |
|-------|---------------|-----------|----------|
| 5 | Phase 1-4 → round-2 hotfixes | 3 | 0 |
| 8 | round-6/7 fixes | 2 + honest test-coverage gap | 0 |
| 9 | round-8 fixes-of-fixes | 0 critical | 3 minor |
| 10 | round-9 deferred + watchdog | 0 critical | 3 minor + 1 test-quality |
| 11 | round-11 docs sync | 0 critical | 1 real (QUICKSTART defaults) |
| 12 | senior arch review (full branch) | 0 new bugs | **8 architectural debt items** (all closed by round-19) |
| 16 | day-1 trilogy sweep | 2 (validator theater + KeyError gap) | 2 cosmetic |
| 17 | round-16 fix-for-fix | 0 critical | 3 scope-claim gaps |
| 18 | round-17 collect_scene split | 0 critical | 3 suspicious (bool-numeric, type-hint, behaviour-change docs) |

Reviewer-finding curve: **3 → 5 → 2 → 2 → 0 → 1 → 0 → 2 → 0 → 0**.
Last three rounds: zero critical bugs. Curve has flattened.

Reviewer agents are at least as good as me at this point — every round
that spawned one found at least one issue I missed (until the curve
flattened). Plan on spawning one for any non-trivial change you make
post-merge.

---

## Final state of the certification report

Last cert run: **6/6 PASS** (machine: this Windows box, Blender 5.1.1,
`.venv` Python 3.11). 113 unit tests in cert harness, 126 total in
full suite. Written to `certification_report.md` at repo root,
regenerated on every cert run. Committed in `a142484`.

If anyone runs the cert and sees < 6/6: that's the signal that the
branch isn't shippable on their environment. Common cause is the
Blender path constant (see "Common failure modes" above).

---

## End-of-sprint summary

19 rounds. 42 commits. ~30 real bugs found and fixed across 10
reviewer rounds. 3 senior day-1 rewrites + 5 senior code-smells all
shipped. 126 unit tests + headless CI + stress harness + cert gate
all green. Documentation (DESIGN/BLOCKS/HANDOFF/QUICKSTART) synced.
Lessons codified at `~/.claude/CLAUDE.md §9` (10 rules).

The branch represents a complete production-hardening sprint of the
addon. If you're picking up after merge, the addon should behave
exactly as documented in `docs/QUICKSTART.md` and any deviation is a
regression worth investigating with the cert harness first.

---

## 2026-05-27 update: rounds 20-26 on `feat/mpm-stability`

After merging `fix/addon-audit-fixes` to main, a new sprint focused
on MPM solver stability + remaining whitewater bugs landed on a
fresh branch `feat/mpm-stability`. 7 commits, 3 reviewer rounds
(21/23/25/27) each independent, each found new real bugs.

**What shipped:**

- **Round-20** — `MpmDivergenceError`: solver raises typed exception
  on NaN (per-step check; was silent `print + break`); CLI writes
  truthful `cache.json` with `truncated_at_frame`, exits rc=2; addon
  `OT_bake` reads cache.json on rc=2 → friendly recovery message.
- **Round-21** — fixed CRITICAL `cmd_render` duplicate that left
  `cmd_bench` unregistered + HIGH mesh-pass sidecar index desync
  (gap in PLY sequence offset colours by N frames).
- **Round-22** — 4 MEDIUMs: render_bridge frame 0 → idx −1 (silent
  no-op + negative sim-time text), colour-count mismatch silent
  drop (now WARNING logged), `MpmSolver.__del__` `TypeError` at
  interpreter shutdown, hardcoded `default_rng(42)` → `cfg.seed`.
- **Round-23** — CRITICAL whitewater Z-up: entire feature applied
  gravity / spawn_offset / kick to Y while project is Z-up; tests
  were written to match the bug (best example of lesson §9.2).
  Plus HIGH: `scene.simulation.gravity` was dropped into legacy
  cfg fields the new step() doesn't read; HIGH:
  `whitewater_trapped_air_weight` missing from addon emit; LOW:
  start_sync's Thread() construction missing orphan-Popen-kill
  guard (mirror drift INSIDE one class — lesson §9.6).
- **Round-24** — closed deferred items: CI script wipes CACHE_DIR
  before run (was: stale files satisfied count gates, lesson §9.2),
  per-frame sidecar de-dup (write once at frame 0, mesher falls
  back), whitewater attach shape-mismatch guard.
- **Round-25** — HIGH: MPM inflow gate used `==` not `>=` — any
  particle with `spawn_step = 0` (rounded from `frame_floats < 1/
  dump_every`) stayed pinned forever, silently dropped from PLY.
  Plus MEDIUMs: `_rebuild_mesh` cleared geometry before validating
  (corrupt mid-sequence PLY blanked screen); `_headless_render`
  malformed-JSON traceback (now one-line SystemExit); LOW: ASCII
  PLY silently misparsed in addon's parallel reader; LOW:
  `seed_inflow_particles` reversed range now raises early.
- **Round-26 (self-review)** — zero-gravity scene bug I introduced
  in round-23: the ratio guard `if g_scene != 0.0 else 1.0` was
  meant for div-by-zero (no such risk) but inverted user intent —
  setting gravity=0 snapped per-class gravities back to defaults.
  Dropped guard, added source-grep regression test.

**State of test suite:** 82 unit tests green (rounds 20-26 added 26
new across `test_mpm_divergence`, `test_round22_fixes`,
`test_round23_fixes`, `test_round24_fixes`, `test_round25_fixes`).
Three source-grep contract tests (whitewater gravity ratio form,
MPM inflow gate form, cmd_render duplicate guard).

**Branch state:** `feat/mpm-stability` is 7 commits beyond `main`.
Ready for review/merge; cert harness needs re-run on this branch.

**Open follow-ups:**
- LOW: pushback `_post_step` doesn't re-gate held particles; only
  bookkeeping, no user-visible artefact.
- LOW: `MpmConfig.fps`/`device` defaults written to public dataclass
  but per-bake overrides in scene config — minor documentation gap.

Lessons added (to be added to `~/.claude/CLAUDE.md §9` on next
update):

- **§9.11 (candidate)**: when introducing a guard, explicitly state
  what it's for IN A COMMENT. Round-26 caught the round-23 guard
  inverting user intent because there was no comment to tell future-
  me what the guard was supposed to dodge.
- **§9.12 (candidate)**: source-grep contract tests are cheap regression
  insurance for "must / must not" patterns the actual test can't
  exercise (e.g. kernels needing CUDA, removed forms with no test
  representation). Round-21 cmd_render dup + round-25 inflow gate
  + round-26 gravity ratio all use this pattern.

---

## 2026-05-27 update: rounds 27-32 on `feat/mpm-stability`

Continuation of the MPM-stability sprint. 7 more commits, 3 more
independent reviewer rounds (27/29/31). The round-31 sweep ran as
a parallel orchestrator (3 background workers, non-overlapping
ownership) — first use of the orchestrator-rules ownership-map
discipline during this sprint.

**What shipped:**

- **Round-27** (independent reviewer, 7 issues) — 2 HIGH
  (`PreloadCache.pop` silent leak, `_headless_render` incomplete
  required-keys list); 3 MEDIUM (`blocks` dedup silent — the root
  cause that let round-21 cmd_render slip in unnoticed,
  `render_bridge` no face bounds-check, `_on_load_post` false
  warning on non-MESH `Empty`); 2 LOW (bake auto_attach race after
  `_is_running` cleared, `_on_load_post` no LRU-overflow warning).
- **Round-28** — shipped fixes for all 7 round-27 findings.
  Headline: `blocks/__init__.py` now LOGS WARNING on duplicate
  qualname registration (was silent — exactly the root cause of
  the round-21 cmd_render duplicate). `render_bridge` had
  mirror-drifted from `cache_loader`; bounds-check copied across.
  10 new tests in `test_round28_fixes.py` incl. a meta-test
  asserting the dedup warning fires.
- **Round-29** (independent reviewer) — 1 CRITICAL + 1 MEDIUM + 1
  LOW. CRITICAL: `mpm_initial_velocity` was DOUBLE-SCALED with the
  WRONG axis — addon scaled by `inv_size[1]` (Y), CLI then
  re-scaled by `inv_dom_z` (Z); on a 2:1 Y:Z aspect domain the
  user-specified velocity was 4× off. Every other MPM velocity
  knob ships unscaled — classic mirror-drift (§9.6). MEDIUM:
  `whitewater_potentials` no `v_max`/`radius`/`dx` validation.
  LOW: `cubic_ref` missing empty/NaN guard.
- **Round-30** — shipped all 3 round-29 fixes; added
  `@block("M5.11.5")` to mixbox; added `[BLK ...]` source-grep
  test refs for S2.17.7, M5.11.4, M5.11.4.H, M5.11.5. Also fixed
  a self-introduced round-28 test-isolation bug: popping
  `gpufluid.blocks` from `sys.modules` created a fresh
  `BlockError` class instance which broke `pytest.raises` matching
  in downstream tests.
- **Round-31 (parallel sprint)** — orchestrator ran 3 background
  workers concurrently on non-overlapping ownership:
  `bug-hunter-round31` (READ-ONLY on `src/` + `addon/`),
  `docs-syncer-lessons` (`~/.claude/CLAUDE.md` — out-of-repo),
  cert harness (`certification_report.md`). All three landed
  cleanly. Reviewer found 2 HIGH + 3 MEDIUM + 1 LOW. Cert returned
  6/6 PASS confirming branch still shippable.
- **Round-32** — shipped all 6 round-31 fixes. The big one: FLIP
  solver had NO NaN-divergence trap — exact symmetric oversight
  with the MPM round-20 fix. Now ships `FlipDivergenceError` +
  `has_diverged()` + CLI per-frame check + cache.json
  `truncated_at_frame` + rc=2 — mirrors MPM pattern bit-for-bit.
  Also added `solver` / `truncated_at_frame` / `truncation_reason`
  fields to the `CacheManifest` dataclass to close the
  schema-drift gap reviewer flagged (the typed reader had been
  silently dropping the fields MPM-side raw-JSON writers were
  injecting). Other fixes: PLY non-triangle face rejection,
  `DomainTransform` per-axis degeneracy guard, `OT_clear_cache`
  selective preload invalidate (was clearing ALL domains'
  preloads), pushback `snap_eps` now `dx*0.1` capped at 1e-4.
  8 new tests in `test_round32_fixes.py`.

**Lessons shipped to `~/.claude/CLAUDE.md` §9.11 + §9.12** (by
`docs-syncer-lessons` during the round-31 parallel sprint —
previously listed as candidates above, now live in global rules):

- **§9.11 — Guard comments mandatory**: every defensive guard
  requires an inline `# dodge: <scenario>` comment so future-you
  can spot when the guard inverts intent. Real example: round-23
  added `if g_scene != 0.0 else 1.0` "for div-by-zero" but there
  was no division → guard inverted zero-G scene intent → caught
  only on round-26 self-review.
- **§9.12 — Source-grep contract tests**: cheap regression
  insurance for code paths that can't be unit-tested (Warp
  kernels, removed forms, stacked decorators). Read source as
  text, strip comments, assert pattern (not) present. Three real
  examples now shipped: inflow gate `==` form, gravity ratio
  inversion, `mpm_initial_velocity` pre-scale.

**State of test suite:** 107 unit tests green (8 new in
`test_round32_fixes` on top of the round-20-26 26 new tests).
Source-grep contract tests still in place + extended.

- **Round-33** (independent reviewer, parallel sprint) — 2 CRITICAL
  + 2 MEDIUM on the FLIP attribute pipeline. CRITICAL:
  `_apply_outflows_gpu` didn't compact `attr_color`/`attr_temperature`
  alongside `pos`/`vel` → colour scrambling on every outflow event;
  `prepare_frame` emit-append path didn't grow attr arrays → OOB
  reads on inflow + base-coloured fluid where coloured was expected.
  MEDIUM: MPM SDF box collider top-friction damped the FULL velocity
  vector instead of tangential post-projection; MPM inflow keyframes
  silently mis-interpolated on non-monotone frame lists or `lo > hi`
  AABBs.
- **Round-34** — shipped all 4 round-33 fixes. Both CRITICALs guarded
  by source-grep contract tests (§9.12) since the actual codepath
  needs CUDA + a FLIP scene with attrs + outflow, no fixture
  exercises that today. 6 new tests in `test_round34_fixes.py`.
  Bonus cleanup: added `@block("M5.11.5")` + literal `[BLK ...]`
  test refs (block-registry hygiene); fixed `MpmDivergenceError`
  export from `sim.mpm` package `__init__`.
- **Round-35** (independent reviewer, parallel sprint with cert
  harness) — 1 HIGH + 2 MEDIUM + 2 LOW. HIGH: CPU reseed path
  silently dropped `attr_color`/`attr_temperature` — classic §9.6
  mirror-drift, round-34 fixed the GPU sibling but forgot the CPU
  <100k path. MEDIUM: MPM sidecar shim shape-stuck — verified
  FALSE ALARM after re-reading round-24 per-frame indexing logic;
  MeshExtractor.extract had no empty-pos guard, frame ghost-froze
  when all particles exited. LOW: `array_scan` sync hot-path,
  `__init__.py` warp import-error masking — both deferred.
- **Round-36** — shipped HIGH CPU-reseed attr-drift fix (variant
  return on `reseed_particles`: 4-tuple back-compat when no attrs,
  6-tuple with `new_color`/`new_temperature` when any passed —
  preserves the ~10 existing test callsites); MEDIUM
  MeshExtractor empty-pos guard returns `(None, None)`. 5 new tests
  in `test_round36_fixes.py` including a source-grep guard for the
  CLI reseed branch. Round-35 sidecar-shim flagged as false alarm
  in commit message. 2 LOW deferred. Cert harness verified 6/6 PASS
  after round-34.
- **Round-37** (parallel sprint with cert + docs) — 3 MEDIUM + 3 LOW.
  Animated-inflow keyframe loop blew up at default
  `frame_end=10000` (10–60s collect hang); `_patches.py`
  `write_text` crashed on read-only install; `start_sync`
  `TimeoutExpired` didn't flush drained stdout; `_export_obj`
  fan-triangulated n-gons (degenerate fans on concave quads);
  obstacle name `obstacle_{name}.obj` allowed Windows-illegal
  chars.
- **Round-38** — shipped 5 of 6 round-37 fixes (1 LOW deferred:
  dev-only `_consolidate_demos` hardcoded path). 6 new tests.
- **Round-39** (parallel sprint) — 1 HIGH + 2 MEDIUM + 1 LOW.
  HIGH: modal tick dropped queued stdout on subprocess exit
  (mirror-drift to the round-38 sync-path fix — classic §9.6).
  MEDIUM: `seed_box` + `load_checkpoint` didn't invalidate the
  captured CUDA graph; `enable_sub_dense` + `sub_dilation<1` hit
  GS-RB band-edge bug. LOW: `load_post` didn't surface
  `truncated_at_frame` warning.
- **Round-40** — shipped all 4. 7 new tests.
- **Round-41** (parallel sprint) — 1 CRITICAL + 2 HIGH + 2 MEDIUM
  + 1 LOW. CRITICAL: `load_checkpoint` silently dropped attrs +
  ignored `dx`/physics drift. HIGH: sub-dense proximity check
  used stale `marker_host` for non-animated scenes; inflow box
  overlapping obstacle spawned particles inside solid →
  explosive ejection. MEDIUM: `mesh_marker` no 0-tri guard;
  `--resume` seam double-emit. LOW: `mix_rgb` no clamp.
- **Round-42** — shipped CRITICAL + 2 HIGH + 1 MEDIUM + 1 LOW
  (10 new tests). Deferred 1 MEDIUM (`--resume` seam double-emit)
  as documented follow-up — needs inflow-event-state persisted
  across checkpoint, broader-scope than this sprint.
- **Round-43** (parallel sprint) — 2 MEDIUM + 2 LOW.
  Diminishing-returns visible. Pushback kernels ignored
  `particle_selection` (wasted F-resets on held particles);
  `OutflowBox` no `lo <= hi` validation (mirror gap of the
  round-33 MPM-inflow keyframe fix); `apply_eevee_preset`
  corrupted Eevee state under `engine=CYCLES`.
- **Round-44** — shipped all 4. 8 new tests.
- **Round-45** (parallel sprint, asked reviewer for explicit
  "0 bugs" stop-signal) — 1 CRITICAL + 1 MEDIUM. Sprint NOT
  done yet. CRITICAL: FLIP inflow path dropped per-source
  `color` + `temperature` silently (`InflowBox` had no field —
  classic §9.6 mirror-drift with MPM, which had it). MEDIUM:
  APIC + CUDA graph capture allocated mid-capture when
  `affine_C` was `None` after inflow event.
- **Round-46** — shipped both. Plumbed `color`/`temperature`
  through 4 files (`InflowBox` + `FluidEmitEvent` +
  `publish_for_frame` + `prepare_frame` `col_pad` with
  mask-lockstep + CLI). 10 new tests.
- **Round-47 (FINAL)** — reviewer reported **"0 substantive
  bugs after 8 files in ~30 min"**. Explicit STOP-the-sprint
  signal: *"The curve is flat. Three reviewer rounds at 0/0/0
  weren't a fluke; round-47 confirms it."* Cosmetic-only notes
  (asymmetric `keep_ref` pattern, `MpmConfig.seed` without TOML
  surface) flagged as future-considerations, not bugs.

---

## SPRINT COMPLETE

- **Sprint duration:** rounds 20–47 (**28 rounds**, ~73 real
  bugs across 14 reviewer rounds + 1 self-review).
- **Branch state:** `feat/mpm-stability` is **22 commits**
  beyond `main`.
- **Test suite:** **183 unit tests green** in the `test_round*`
  + ancillary suite.
- **Cert harness:** 6/6 PASS verified after rounds 32, 34, 36,
  38, 40, 42, 44, 46.
- **Lessons §9.11 + §9.12 shipped** to `~/.claude/CLAUDE.md`.
- **Open follow-ups:**
  - 1 MEDIUM (round-41 `--resume` seam double-emit; needs
    inflow event-state persistence — broader scope than this
    sprint).
  - 2 LOW (round-35 `array_scan` hot-path sync;
    `__init__.py` warp-error masking — both cosmetic).
  - 2 cosmetic future-considerations from round-47
    (`vel` buffer `keep_ref` symmetry, `MpmConfig.seed` TOML
    exposure).
- **No CRITICAL / HIGH currently open.**
- **Recommendation:** **ready for merge**. Pre-existing flakies
  (A8.* registry sync — addon-only, walker scans `src/`;
  `sparse_jacobi` timing test) remain out-of-scope.
