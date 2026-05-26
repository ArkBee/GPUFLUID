# Addon-audit branch — session handoff (2026-05-26)

> **Read this file first** if you're continuing work on the addon
> audit / refactor sprint. Self-contained — does NOT depend on the
> wider `HANDOFF.md` context. Last verified PASS: **cert 6/6 + 93
> unit tests + headless CI + stress @ 0.39ms/scrub-frame**.

---

## TL;DR

Branch `fix/addon-audit-fixes` is **37 commits** beyond `main`,
covering 16 rounds of addon audit + 3 senior day-1 refactors.
All three senior architect's day-1 rewrites **shipped**. Production
gate (cert harness) **green**. Last reviewer (8th, round-16) caught
2 real bugs in the validator — both fixed same round.

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

**Last commit on branch:** `6ba47b2` (round-16 — singular/plural validator fix).

---

## Open work (ranked by ROI, pick top of list)

### 1. Round-17: 9th independent reviewer on round-16 fix

**Why:** round-16 introduced the plural-key validator + sim sub-field
type checks. 9th reviewer would verify the fix doesn't have its own
drift bug. Reviewer-finding curve: 3→5→2→2→0→1→0→**2** (round-16
spiked because refactor opened new surface). One more pass to confirm
clean before declaring done.

**Pattern (copy-paste prompt):**
```
Repo: E:\projects\gpu_flip\gpufluid. Last commit `6ba47b2`. Review
only diff `d0531d2..6ba47b2`. Focus: did the plural-key fix miss
any singular-key call site? Are the new sim-sub-field checks
correctly placed (after the dict-type guard)? Test fixture
fidelity vs production scene_dict shape?
```

**Time:** ~5 min wall (background agent).
**Risk if skipped:** low — round-16 is small + cert is green.

### 2. Round-17: vendor `tomli_w` for full day-1 #3

**Why:** day-1 #3 shipped as "lite" — kept hand-rolled `_emit_scalar`/
`_emit_table`. Senior wanted full delete in favour of `tomli_w`.
`_emit_table` is still the highest-bug-density file. Vendoring ~300
lines of `tomli_w` (MIT, 0-dep) would let us replace our emitter +
delete those round-1..16 hardening lines.

**Plan:**
- Drop `tomli_w` source into `addon/gpufluid_blender/_vendor/tomli_w/`
  (or `addon/gpufluid_blender/vendor/tomli_w.py` if single-file works).
- Update `addon/gpufluid_blender/blender_manifest.toml` if extensions
  need vendor declarations.
- In `config_builder.build_toml`, replace `_emit_toml(d)` →
  `tomli_w.dumps(d)`.
- Delete `_emit_scalar` / `_emit_table` / `_emit_with_key` /
  `_is_table` / `_is_array_of_tables`.
- Keep `_deep_merge` (overrides path).
- Run cert. Expect: same TOML output, but emitter is now a stable
  third-party lib instead of a 200-line hand-rolled parser.

**Time:** ~30-60 min. Risk: medium — emit format may differ slightly
(spacing, list formatting), need to verify CLI parses identically.
`test_addon_schema_roundtrip.py` is the gate.

### 3. Round-17: bake.collect_scene SRP split

**Why:** senior code smell #2 — 250-line bpy→dict adapter that mixes
(a) bpy traversal, (b) world→unit-cube math, (c) validation,
(d) OBJ export side-effects, (e) per-solver param mapping. The shape
that let `_FLIP_ONLY_SIM_KEYS` filters leak from `config_builder`
into `collect_scene`.

**Plan:**
- Extract `addon/gpufluid_blender/scene_collector.py` (bpy I/O only).
- Extract `addon/gpufluid_blender/domain_transform.py` (pure numerics,
  unit-testable without bpy).
- Extract `addon/gpufluid_blender/scene_validator.py` (warnings —
  out-of-domain, animated-without-keyframes, etc.).
- `bake.execute` glues the three.

**Time:** ~60-90 min. Risk: medium — collect_scene is the workhorse,
breaking it risks regression in every live scenario. Cert harness
catches most; live MCP test catches the rest.

### 4. Round-17: documentation sync (DESIGN.md §11 + HANDOFF §5)

**Why:** the docs-auditor (round-10) flagged BLOCKS.md A8 section as
stale (round-10 fixed it). Round-11 synced HANDOFF §3 layout + §5
A8 row + QUICKSTART. Still drifted: A8.13 not in DESIGN §11 (only
BLOCKS), refactor introduced new modules not in HANDOFF directory
tree (`scene_dict.py`, `operators/_runner.py`).

**Time:** ~15 min. Risk: trivial (docs only).

### 5. Pause — declare branch ready to merge to `main`

**Why:** cert green, all senior day-1 done, 8 reviewer rounds, last
2 found only the singular/plural typo. Diminishing returns is steep.
Merging frees the next contributor from a 37-commit branch.

**Pre-merge checklist:**
- [ ] Cert script PASS on a fresh checkout (someone else's machine)
- [ ] `git log main..HEAD --stat` reviewed
- [ ] Tag the merge commit (e.g. `addon-audit-2026-05-26`)
- [ ] Update `docs/HANDOFF.md §5` A8 row reference to point at the
      merge tag for archaeology

---

## Key files / landmarks

```
addon/gpufluid_blender/
├── __init__.py                   # register/unregister + @persistent handlers
├── scene_dict.py                 # ★ Round-15: TypedDict + validate_scene_dict
├── config_builder.py             # build_toml + _emit_table (calls validator)
├── cache_loader/
│   ├── __init__.py               # ★ Round-13: PreloadCache class
│   ├── _ply.py                   # PLY reader
│   └── _ops.py                   # OT_attach_cache + OT_attach_ww + OT_detach
├── operators/
│   ├── _runner.py                # ★ Round-14: ModalSubprocessRunner
│   ├── bake.py                   # OT_bake — uses runner + PreloadCache
│   ├── render.py                 # OT_render — uses runner
│   ├── helpers.py                # mark_* + clear + Eevee preset + drain
│   ├── _animation.py             # F-curve helpers (Phase 4 split)
│   └── _collect.py               # scene_dict builder (split candidate, item #3)
├── render_bridge.py              # FrameMeshLoader (A8.10) + Eevee preset (A8.9)
├── preferences.py                # interpreter_path + LRU caps
└── properties.py                 # Domain/Fluid/Obstacle/Inflow/Outflow/WW PGs

tests/
├── test_scene_dict_validator.py        # ★ Round-15/16: 25 cases
├── test_preload_cache_invariants.py    # ★ Round-13: 10 cases
├── test_addon_round8_regressions.py    # ★ Round-8..14: lifecycle source-grep
├── test_addon_role_single.py           # Round-5: mark_* single-role
├── test_addon_schema_roundtrip.py      # Round-1: contract test
├── test_addon_preload_lru.py           # Round-2: LRU cap behaviour
├── test_render_bridge_payload.py       # Round-3: render argv contract
├── test_a8_config_builder.py           # pre-existing
└── test_no_layer_exceptions.py         # F3.6 layer-boundary gate

examples/
├── _ci_certify_addon.py          # ★ Round-12: single-command production gate
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

- **PreloadCache** is the singleton for ALL preload mutations.
  Back-compat shims (`_PRELOAD`, `_free_table`, `_lru_touch`,
  `_lru_install`, `_prune_stale`) at module level delegate to the
  singleton — old call sites still work, new code uses class methods.

- **ModalSubprocessRunner** owns ALL subprocess lifecycle. Both
  `OT_bake` and `OT_render` lazy-init `self._runner` in `execute()`,
  call `runner.start_sync(...)` OR `runner.start_modal(...)`, then
  delegate `modal()` to `runner.tick_modal(...)` and `cancel()` to
  `runner.cancel(...)`. `_is_running` stays on each OT class as
  cross-instance reentrance lock.

- **SceneDict TypedDict** + **validate_scene_dict** documents the
  scene_dict shape (build_toml input). Validator semantics:
  **"validate what's there, don't require completeness"** — sections
  present must have correct shape; absent sections allowed (test
  fixtures pre-round-15 rely on this). Uses **plural keys**
  (`obstacles`, `inflows`, `outflows`, `fluids`) per the
  collect_scene convention — round-16 fixed singular-key theater.

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
| 12 | senior arch review (full branch) | 0 new (5 reviewers cleared) | 5 architectural debt items |
| 13 (impl) | day-1 #1 ship-or-not | n/a (refactor commit) | n/a |
| 16 (this) | day-1 trilogy sweep | **2** (validator theater + KeyError gap) | 2 cosmetic |

Reviewer agents are AT LEAST as good as me at this point — every round
that spawned one found at least one issue I missed. Plan on spawning
one for any non-trivial change.

---

## Final state of the certification report

Last cert run: **6/6 PASS** (machine: this Windows box, Blender 5.1.1,
`.venv` Python 3.11). Written to `certification_report.md` at repo
root, regenerated on every cert run. Committed in `6ba47b2`.

If anyone runs the cert and sees < 6/6: that's the signal that the
branch isn't shippable on their environment. Common cause is the
Blender path constant (see "Common failure modes" above).
