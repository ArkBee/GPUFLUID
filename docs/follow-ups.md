# Follow-ups tracker

> Отложенные TODO. Numbering policy: каждый item получает стабильный `FU-NNN`,
> номер **не переиспользуется** после закрытия. Закрытые архивируются в
> `follow-ups-archive.md` когда файл > 300 строк (§6 global rules).
> Статусы: 📝 open · ⚠ in-progress · ✅ done · 🔴 blocked.

---

## Capability: MPM outflow/drain (2026-06-02) ✅

- ✅ **FU-022** — MPM теперь поддерживает outflow (дренаж). Раньше MPM-ветка
  CLI игнорировала `scene.outflow` (только FLIP потреблял — §9.6 mirror drift):
  проточные сцены (водопад/фонтан/река) копили частицы до переполнения домена.
  - **Решение:** `src/gpufluid/sim/mpm/outflow.py` — `MpmOutflow` + kernel
    `k_outflow_despawn` (S2.17.8). MPM пре-аллоцирует фикс-массив (inflow-частицы
    time-gated, не спавнятся), компактить нельзя → despawn: живая частица
    (selection==0) внутри активного drain-AABB получает selection=1 (исключается
    из дампов и всех kernel'ов), v и F/C сброшены, паркуется в центр бокса.
    Зеркало inflow-гейта, one-way + идемпотентно.
  - **Wiring:** `MpmConfig.outflows`; solver строит `_outflow_params`
    (frame→step via dump_every), запускает в `_post_step` ПОСЛЕ физики. CLI
    строит `MpmOutflow` из `scene.outflow`. Аддон уже эмитил `[[outflow]]` +
    имеет `mark_outflow` → UI→TOML→solver путь полный end-to-end.
  - **Верифицировано (RTX 4080):** A/B на идентичном непрерывном 3-ступенчатом
    каскаде, drain vs no-drain. Кадр 300: no-drain=31250 live (=rate×time,
    безгранично), drain=18449 → удалено 12801 (40%), разрыв растёт монотонно.
    Без дивергенции. Тесты `test_s2_17_8_mpm_outflow.py` (4) + регрессия 60 зелёных.
  - **Замечено по ходу (live cascade тест, §9.13):** MPM дивергирует на
    накоплении при `dt=0.005`/`bulk=1500` (глубокая лужа) — стабильно на
    `dt=0.001`/`bulk≈900`. Стоит подобрать adaptive dt / лучший EOS как отдельный
    research-тикет (FU-023, не блокер).

- ✅ **FU-023** — MPM accumulation stability — **ИСПРАВЛЕНО 2026-06-02** (adaptive
  CFL substepping, S2.17.9). Юзер ловил "MPM solver diverged at frame 680".
  - **Решение:** `step()` опционально дробит frame dt на N суб-шагов по CFL
    `dt_sub ≤ cfl·dx/(c_sound+v_max)`, `c_sound=√(K/ρ)`. Стабильно при любой
    глубине лужи без ручного занижения bulk/dt. Opt-in, default OFF =
    byte-identical старому (1× p2g2p). Переиспользует существующие
    `[simulation] cfl/cfl_factor/cfl_max_substeps` → чекбокс "CFL Substepping"
    в аддоне теперь работает и для MPM (новый UI не нужен).
  - **2 неочевидных бага, пойманных live A/B (§9.4):** (1) pushback ОБЯЗАН
    запускаться каждый суб-шаг, иначе туннелирующая частица вылетает за грид →
    CUDA error 700 (не чистый NaN) — первая наивная версия упала на кадре 362;
    (2) при превышении cap CFL-demand — one-shot WARN, а не тихий under-substep.
  - **Верифицировано (RTX 4080):** идентичный каскад dt=0.003/bulk=1200. CFL off
    → дивергенция кадр 179. CFL on → все 300 кадров, 0 NaN, физично. Тесты
    `test_s2_17_9_adaptive_substep.py` (6) + регрессия 66 зелёных.

## Pre-public-test audit (2026-06-01)

Источник: multi-agent workflow `gpufluid-pretest-audit` (30 агентов, 6 поверхностей,
adversarial-verified). Вердикт: **🟡 GO-WITH-FIXES** — истинных блокеров нет, но кластер
«тихо-неверный-результат» (3 из 5 — рецидив round-61 темы честности в ветках, которые
тот фикс не тронул; §9.6 mirror-drift) надо закрыть до паблик-теста.

> **Update 2026-06-01:** all five should-fix (FU-001…005) + three trivial minors
> (FU-006/008/009) + the physics-fidelity **FU-016** applied on branch
> `fix/round61-zero-frame-honesty`. New tests: `tests/test_audit_20260601_fixes.py`
> + `tests/test_fu016_gravity_world_metres.py`; conftest made warp-optional so the
> warp-free suite runs locally. **Verified in project `.venv` (gpufluid+warp+CUDA):
> full suite green except 2 pre-existing unrelated fails** — `test_blocks_registry`
> (BLOCKS.md A8.* impl-rows without @block decorator, addon-can't-carry-warp
> limitation) and `test_s2_16_sparse_jacobi` (GPU perf-timing benchmark, hardware
> flake). Both confirmed failing without these changes. FU-002a folded into FU-002b
> (per-frame cache.json read = hot-path I/O, §9.5).

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

### Physics fidelity (не блокер паблик-теста, но честно зафиксировать)

- ✅ **FU-016** — MPM gravity масштабируется по aspect-ratio, а не по метрам. **CONFIRMED расчётом + ИСПРАВЛЕНО 2026-06-01.** *(addon форвардит `[domain] size_world` (метры) из `DomainTransform.dom_size`; CLI считает `SceneCfg.world_size` с fallback на `domain_size` для standalone TOML; gravity+velocity+v_terminal+anti-splash стали метро-корректны разом. 6 unit + 2 contract теста; backward-compatible.)*
  - **Зачем:** `commands.py:244` делит `g_norm = g_world / dom_z`, где
    `dom_z = scene.domain_size[2] = nz/max(res)` (`config.py:296-298`, `dx=1/max(res)`) —
    это нормализованный aspect-ratio, НЕ реальные метры (реальный world-extent в CLI не
    передаётся вообще). Итог `g_norm = g·max(world_extent)/world_z` вместо `g/world_z`.
    Комментарий `commands.py:237-240` («real metres») фактически неверен.
  - **Замер:** кубический домен 2.5 м → ускорение 2.52× завышено → вода падает
    **1.59× быстрее** реального; плоский 4×4×1 м → ускорение 4× → **2.0× быстрее**.
    Корректно только для истинного 1-м единичного куба.
  - **Шаги (большой рефактор, НЕ перед паблик-тестом):** пробросить world dom_size из
    аддона в TOML (`[domain] size_world`), считать `g_norm = g/world_z`; так же пересмотреть
    `initial_velocity_z`/`v_terminal`/anti-splash, которые делят на тот же `inv_dom_z`.
  - **Готово когда:** для домена H метров время падения нормализованной 1.0 == √(2H/g);
    unit-тест на 3 домена (1м/2.5м/плоский).
  - **Severity:** fidelity/look, не crash. Юзер компенсирует слайдерами gravity/dt.
    [[project_round61_honesty]] держал это как OPEN — теперь подтверждено замером.

- ✅ **FU-017** — MPM сделан дефолтным солвером (FLIP → legacy water-only). **2026-06-01.**
  - **Зачем:** продакт: «FLIP не хочу — он только для воды, MPM для всего». Аддон дефолтил на
    `flip` (`properties.py:155`), т.е. юзер на паблик-тесте первым же Bake попадал на legacy-путь,
    который вдобавок НЕ метро-скейлит гравитацию (FU-016 починил только MPM → был FLIP↔MPM
    рассинхрон, §9.6). Смена дефолта на `mpm` авто-разрешает дилемму: дефолтный путь теперь
    корректный, FLIP остаётся выбираемым опционом с честным описанием «legacy, water-only».
  - **Контекст:** `properties.py` EnumProperty `solver` — `mpm` теперь первый item + `default="mpm"`;
    описания обновлены. CLI-дефолт (`config.py:191 solver="flip"`) оставлен для backward-compat
    рукописных TOML (аддон всегда пишет solver явно).
  - **Готово когда:** contract test `test_mpm_is_default_solver`. ✅
  - **Note:** это переводит FLIP-gravity-баг (тот же класс, что FU-016) из «блокер дефолтного
    пути» в «известное ограничение legacy-опции». Полный фикс FLIP-гравитации — FU-018 (ниже),
    низкий приоритет раз FLIP не продакт-путь.

- 📝 **FU-018** — FLIP-солвер применяет гравитацию в сырых m/s², без метро-нормализации (тот же
  класс бага, что FU-016 для MPM). `solver3d.py:153 k3_add_gravity` `v += g*dt`, g=сырой `-9.81`,
  частицы в [0,1]³. Корректно только при домене 1 м по высоте. **НЕ блокер** (FLIP теперь legacy,
  не дефолт — FU-017). Чинить если/когда FLIP вернётся в продакт-сценарии; тогда — то же
  `world_size`-масштабирование, что в MPM-ветке.

### Deep-dive раунд 2 (2026-06-01) — obstacle per-axis scaling

Источник: workflow `ws3gzv5zn` (12 агентов, adversarial-verified) + личная перепроверка
кода (§9.1 — отчёт агента НЕ применялся на слово; его предложенный OBB-фикс был НЕВЕРЕН
для повёрнутого случая — см. ниже).

- ✅ **FU-019** — Obstacle half-sizes/scale усреднялись по осям вместо per-axis на non-cubic домене. **ИСПРАВЛЕНО частично + honest-warn 2026-06-01.**
  - **Box (axis-aligned):** `bake.py:213` использовал `inv_avg` даже для неповёрнутого
    бокса → коллайдер мис-сайзился на широком/плоском домене. Фикс: для identity-rotation
    юзаем per-axis `hx/hy/hz` (точно). **Повёрнутый бокс ОСТАВЛЕН на `inv_avg`** — там
    per-axis шиарит ortho-матрицу R (агент-отчёт это пропустил, я поймал перечитав код).
    На кубическом домене обе ветки совпадают → 0 изменений для дефолтной сцены.
  - **MESH:** `scale` в схеме `ObstacleMeshCfg` — СКАЛЯР (`config.py:134`), per-axis
    невозможен без смены схемы+солвера. MESH рекламируется как «exact fit» workaround для
    SPHERE/CYL, но молча сквошился. Фикс: добавлен WARNING на non-cubic (чтобы «exact fit»
    не был тихой ложью). Полноценный per-axis MESH scaling → отдельный follow-up (схема+солвер).
  - **size_world валидация:** `scene_dict.py` теперь проверяет `domain.size_world` =
    3-list положительных чисел → внятная ошибка вместо `TypeError` в глубине CLI.
  - **Готово когда:** `test_axis_aligned_box_uses_per_axis_halfsize`,
    `test_mesh_obstacle_warns_on_noncubic_domain`, `test_size_world_validation_*`. ✅
    95 тестов (obstacle/schema/OBB-emit) зелёные в venv — без регрессий.

- 📝 **FU-020** — Per-axis MESH-obstacle scaling (инвазивный). `ObstacleMeshCfg.scale` сделать
  vec3 (или передавать per-axis translate+scale), солвер mesh-collider должен принять
  анизотропный масштаб. Сейчас MESH на non-cubic домене сквошится (FU-019 повесил warning).
  **НЕ блокер** (warning есть, workaround = кубический домен). Низкий приоритет.

- 📝 **FU-021** (nit) — лог CLI и `cache.json` пишут `domain_size = aspect-ratio`
  (напр. `(1.0,1.0,0.25)`), а не реальные метры (`size_world`). Косметика/метадата, не
  физика (физика теперь на `world_size`, FU-016). Стоит печатать и `size_world` в логе и
  класть его в `cache.json` для прозрачности. Низкий приоритет.

### Live-smoke 2026-06-01 раунд 2 — NON-CUBIC GPU bake (FU-016/019 end-to-end) ✅

Прогон в `.venv` (RTX 4080, warp+CUDA) на плоском домене 4×4×1 м через **реальный
аддон-конвейер** (`config_builder.build_toml` → CLI `simulate` → MPM solver):
- `size_world = [4,4,1]` форвардится в TOML (FU-016 верифицирован end-to-end, не только unit).
- Гравитの: `g_norm = -9.81` (метро-корректно). Старый код дал бы `-39.24` (**4× too strong**) —
  подтверждено расчётом из загруженной сцены.
- Бейк: 9 mesh-кадров, без дивергенции, физичное падение (z_max 0.879→0.592 за 0.33с).
  Старая гравитация ×4 расплющила бы каплю за 2-3 кадра.
- Box-obstacle на non-cubic домене (FU-019 per-axis) — сцена забейкалась без артефактов.
ВЫВОД: дефолтный путь (MPM) метро-корректен по гравитации и obstacle-геометрии на
non-cubic домене. Остаётся live-MCP smoke В Blender (операторы держат старые модули до
рестарта) — но физика солвера проверена напрямую, что важнее UI-обвязки.

### Live-smoke gaps (обязательно до выкатки, §9.13)

- 🔴 **FU-012** — **pytest локально НЕ гонялся** (нет `warp`). Полный сьют (особенно солвер/kernels)
  обязан быть зелёным на **GPU-CI** перед выкаткой — статика это не заменяет.
  *(2026-06-01: warp-free часть теперь гоняется — 62 теста зелёные после conftest-фикса.
  GPU-часть всё ещё нужна на CI.)*

> **Live-smoke 2026-06-01 (§9.13):** Blender 5.1.1, addon=bl_ext.user_default.gpufluid_blender.
> Подтверждено: addon включён, namespace = extension (урок §9.10 в силе), дефолтная сцена
> `round61_smoke` (Domain res48 + Fluid + Obstacle + 40-кадровый cache) рендерит воду в
> вьюпорте. Мои 6 файлов синхронизированы в install-dir. **Caveat:** живой Python держит
> СТАРЫЕ модули — операторы (FU-001/005) живьём не пере-дёрнуты (reload рискован, §round-61);
> покрыты unit+contract тестами. Gravity (FU-016) подтверждён расчётом, не live-бейком.
- 📝 **FU-013** — Default-scene MCP smoke: domain + sphere → `mark_inflow` без правок → `bake(sync=True)`
  → скриншот. Ни одна cache-honesty находка не наблюдалась вживую, только в исходниках.
- 📝 **FU-014** — Воспроизвести вживую: whitewater attach с пустой папкой (FU-001); re-bake короче + скраб
  за конец (FU-002); вырожденный домен (FU-005); animated inflow за пределами диапазона (FU-003).
- 📝 **FU-015** — «Open Cache Folder» на Linux без xdg-open (FU-006) — нельзя проверить на Windows-хосте.
