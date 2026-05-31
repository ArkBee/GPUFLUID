# Follow-ups tracker

> Отложенные TODO. Numbering policy: каждый item получает стабильный `FU-NNN`,
> номер **не переиспользуется** после закрытия. Закрытые архивируются в
> `follow-ups-archive.md` когда файл > 300 строк (§6 global rules).
> Статусы: 📝 open · ⚠ in-progress · ✅ done · 🔴 blocked.

---

## Pre-public-test audit (2026-06-01)

Источник: multi-agent workflow `gpufluid-pretest-audit` (30 агентов, 6 поверхностей,
adversarial-verified). Вердикт: **🟡 GO-WITH-FIXES** — истинных блокеров нет, но кластер
«тихо-неверный-результат» (3 из 5 — рецидив round-61 темы честности в ветках, которые
тот фикс не тронул; §9.6 mirror-drift) надо закрыть до паблик-теста.

> **Update 2026-06-01:** all five should-fix (FU-001…005) + three trivial minors
> (FU-006/008/009) applied on branch `fix/round61-zero-frame-honesty`. New tests:
> `tests/test_audit_20260601_fixes.py` (12) + conftest made warp-optional so the
> warp-free suite runs locally. Verified: 34 passed (12 new + 22 round-61/62) +
> 53 passed (contract/round-fix regression set). FU-002a intentionally folded
> into FU-002b (per-frame cache.json read would be hot-path I/O, §9.5).

### Should-fix (гейт паблик-теста)

- ✅ **FU-001** — Whitewater attach не валидирует кол-во кадров. *(done: `cache_sanity.count_ww_frames`/`attach_ww_sanity` + wired in `_ops.py`; unit+contract tests)*
  - **Зачем:** пустой/обрезанный whitewater-бейк рапортует `INFO "attached"` и показывает
    пустоту — тот самый round-61 silent-success, только surface-путь починили, whitewater забыли.
  - **Контекст:** `addon/gpufluid_blender/cache_loader/_ops.py:240-241` рапортует безусловно;
    surface-путь эскалирует 0 кадров до WARNING (`_ops.py:175-181`).
  - **Шаги:** считать `whitewater/frame_*.npy` в `execute()`, WARNING/ERROR на 0;
    вынести общий `attach_cache_output_sanity()` helper для обоих путей.
  - **Готово когда:** unit-тест: attach пустого whitewater-кэша → reports WARNING; helper покрыт.

- ✅ **FU-002** — Whitewater preload игнорит `cache.json frame_count` + утечка stale-кадров. *(done via 002b: pre-bake cleanup теперь чистит `whitewater/`+`whitewater_kinds/` — закрывает leak. 002a per-frame cache.json read отклонён как hot-path I/O §9.5)*
  - **Зачем:** перебейкал короче → старые кадры прошлого бейка грузятся молча → неверное рендерится
    без предупреждения (surface-путь это логирует, whitewater — нет).
  - **Контекст:** `cache_loader/__init__.py:408-437` (surface кап) vs handler `576-610` (whitewater без капа);
    pre-bake cleanup `~645-649` чистит `mesh/particles_raw/colors`, но **не** `whitewater/`, `whitewater_kinds/`.
  - **Шаги:** читать `cache.json frame_count` в whitewater-секции handler'а, ломать цикл по `frame_count`;
    добавить `whitewater`/`whitewater_kinds` в cleanup-список.
  - **Готово когда:** тест re-bake(short) после bake(long) → whitewater не грузит stale кадры.

- ✅ **FU-003** — Animated inflow `frame_end` молча клампится. *(done: WARNING когда `fe_eff < fe` + `entry["frame_end"]=fe_eff` пишется в TOML; contract test)*
  - **Зачем:** keyframes режутся до `fe_eff`, но в TOML пишется исходный `frame_end` (дефолт 10000) →
    солвер сидит частицы на весь диапазон → они замерзают посреди симуляции, выглядит как баг солвера.
  - **Контекст:** `operators/bake.py:381-399` (кламп `fe_eff = min(fe, fs + dprops.frames)`),
    TOML пишет `fe` не `fe_eff`; solver warning (`commands.py:190`) ловит только `frame_end < sim.frames`.
  - **Шаги:** WARNING когда `fe_eff < fe`; рассмотреть запись `fe_eff` в TOML.
  - **Готово когда:** тест на animated inflow с `frame_end` > bake-range → WARNING эмитится.

- ✅ **FU-004** — Несогласованные единицы температуры (Fluid=Кельвины, Inflow=Цельсии). *(done: inflow default `20→293`, description `°C→Kelvin`; contract test)*
  - **Зачем:** смешал Fluid-источник и Inflow → 1500 K мешается с «20» (читается как 20 K, не 293) —
    физически бессмысленно, тихо неверный результат.
  - **Контекст:** `properties.py:243-246` (Fluid, «1500 K hot, 300 K cool») vs `315-318` (Inflow, «°C»);
    солвер трактует температуру как unitless скаляр в P2G/G2P.
  - **Шаги:** стандартизировать на Кельвины — Inflow default `20 → 293`, description `°C → K`,
    уточнить «absolute temperature» в обоих.
  - **Готово когда:** оба default'а в одной шкале; source-grep contract test (§9.12) на единицы.

- ✅ **FU-005** — Вырожденный домен (zero/inverted extent) → необработанный `ValueError`. *(done: `except (RuntimeError, ValueError)` вокруг collect_scene; contract test)*
  - **Зачем:** Domain Empty со scale=0 или инверсией Z → сырой traceback в попапе при Bake вместо внятной ошибки.
  - **Контекст:** `operators/bake.py:590-594` ловит только `RuntimeError`; `domain_transform.py:103` кидает `ValueError`.
  - **Шаги:** ранняя валидация в `execute()` (или `poll()`): `lo[i] < hi[i]`; расширить except до `(RuntimeError, ValueError)`.
  - **Готово когда:** Bake вырожденного домена → чистый ERROR-report, не traceback.

### Minor (в ту же руку)

- ✅ **FU-006** — `operators/helpers.py` — `subprocess.Popen(["xdg-open", cache])` обёрнут в
  `try/except OSError` → report WARNING с путём кэша (с `# dodge:` коммент §9.11).
- 📝 **FU-007** — `properties.py:291,325` — дефолт `frame_end=10000` против Domain `frames=120`:
  панель показывает 10000, хотя эффективно клампится до 120 (поведение корректно). Снизить дефолт
  или показывать эффективное значение. *(оставлено: поведение корректно, чисто косметика панели;
  FU-003 закрыл вредный animated-случай)*
- ✅ **FU-008** — `preferences.py` — захардкоженный `E:\...` пример заменён на generic
  Windows + macOS/Linux placeholders.
- ✅ **FU-009** — `addon/gpufluid_blender.zip` удалён с диска (был untracked + в `.gitignore`).

### Defer (не гейт)

- 📝 **FU-010** — Версии рассинхронены: `blender_manifest.toml:4` `0.8.0` / `__init__.py:40` `(0,8,0)` /
  `pyproject.toml:3` `0.0.2`. Скептик снял до nit: аддон не импортит библиотеку (шеллится в CLI),
  рантайм не ломается. Синхронизировать для чистоты.
- 📝 **FU-011** — `tests/test_round61_zero_frame_honesty.py:126-131` (0-frame WARNING) и `:171-184`
  (`frame_offset=0`) — **grep-only, не интеграционные** (§9.2). Имплементация верна, но регресс в
  возвращаемом значении они не поймают. Усилить интеграционными вызовами оператора.

### Live-smoke gaps (обязательно до выкатки, §9.13)

- 🔴 **FU-012** — **pytest локально НЕ гонялся** (нет `warp`). Полный сьют (особенно солвер/kernels)
  обязан быть зелёным на **GPU-CI** перед выкаткой — статика это не заменяет.
- 📝 **FU-013** — Default-scene MCP smoke: domain + sphere → `mark_inflow` без правок → `bake(sync=True)`
  → скриншот. Ни одна cache-honesty находка не наблюдалась вживую, только в исходниках.
- 📝 **FU-014** — Воспроизвести вживую: whitewater attach с пустой папкой (FU-001); re-bake короче + скраб
  за конец (FU-002); вырожденный домен (FU-005); animated inflow за пределами диапазона (FU-003).
- 📝 **FU-015** — «Open Cache Folder» на Linux без xdg-open (FU-006) — нельзя проверить на Windows-хосте.
