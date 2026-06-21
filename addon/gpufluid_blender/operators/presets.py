"""[BLK A8.8] One-click MATERIAL presets — the "Кухарка / Cook" experience.

Pick what jizz you want → bake. Each preset bundles a known-good combo of the
material model + every stability knob + the pour setup, so a non-specialist
never touches an individual slider (and never hits a NaN blow-up).

Two material families (see solver S2.17.9):
  * FLUID — weakly-compressible water/honey (bulk_modulus + Newtonian viscosity).
    Cannot rope-coil; great for water, splashes, thick slumping honey.
  * VISCOELASTIC — viscoplastic StVK+von-Mises: a coherent elastic rope that
    yields and flows. CAN rope-COIL (honey/clay/caramel/chocolate). Coiling
    presets also turn the floor STICKY (no-slip) and set a slow, wobbling pour
    on every inflow — the geometry coiling needs — so it "just works".

`PRESETS` is pure data (a test loads this module under a bpy stub and asserts
the combos — tests/test_addon_presets.py). The `_inflow` subdict (if present)
is applied by the operator to every inflow object in the scene.
"""
import bpy

try:
    from .._blocks import block
except ImportError:
    # dodge: standalone test load has no parent package — registration is moot.
    def block(_bid, _desc=""):
        def _w(fn):
            return fn
        return _w


# ── preset tables (pure data) ──────────────────────────────────────────────
# Every preset turns CFL substepping ON. Domain keys map 1:1 to
# GpufluidDomainProps fields; the optional "_inflow" subdict is applied to each
# inflow object (velocity_z = a slow downward pour, plus the coil seed jitter).
_FLUID = {"mpm_material": "fluid", "mpm_floor_friction": 0.0,
          "use_cfl": True, "solver": "mpm"}
_VE = {"mpm_material": "viscoelastic", "mpm_floor_friction": 1.0,
       "use_cfl": True, "solver": "mpm", "mpm_rpic_damping": 0.05,
       "mpm_grid_v_damping": 0.999}

PRESETS = {
    # ── FLUID family ──────────────────────────────────────────────────────
    "water": {**_FLUID, "resolution": 96, "dt": 0.004, "cfl_factor": 0.4,
              "cfl_max_substeps": 24, "mpm_bulk_modulus": 900.0,
              "mpm_viscosity": 0.0, "mpm_rpic_damping": 0.15,
              "mpm_grid_v_damping": 0.998, "mpm_initial_velocity": -0.3},
    "splash": {**_FLUID, "resolution": 128, "dt": 0.003, "cfl_factor": 0.4,
               "cfl_max_substeps": 40, "mpm_bulk_modulus": 1200.0,
               "mpm_viscosity": 0.0, "mpm_rpic_damping": 0.05,
               "mpm_grid_v_damping": 0.999, "mpm_vz_max_splash": 0.6,
               "mpm_initial_velocity": -0.6},
    "honey_slump": {**_FLUID, "resolution": 96, "dt": 0.003, "cfl_factor": 0.4,
                    "cfl_max_substeps": 80, "mpm_bulk_modulus": 1500.0,
                    "mpm_viscosity": 180.0, "mpm_rpic_damping": 0.15,
                    "mpm_grid_v_damping": 0.999, "mpm_initial_velocity": -0.2},
    "draft": {**_FLUID, "resolution": 48, "dt": 0.005, "cfl_factor": 0.5,
              "cfl_max_substeps": 16, "mpm_bulk_modulus": 900.0,
              "mpm_viscosity": 0.0, "mpm_rpic_damping": 0.15,
              "mpm_grid_v_damping": 0.998, "mpm_initial_velocity": -0.3},
    # ── VISCOELASTIC family (rope-coiling) ────────────────────────────────
    "honey_coil": {**_VE, "resolution": 128, "dt": 0.001, "cfl_factor": 0.5,
                   "cfl_max_substeps": 360, "mpm_young_modulus": 250000.0,
                   "mpm_poisson": 0.35, "mpm_yield_stress": 1000.0,
                   "mpm_viscosity": 40.0,
                   "_inflow": {"velocity_z": -0.15, "velocity_jitter": 0.04}},
    "clay": {**_VE, "resolution": 128, "dt": 0.001, "cfl_factor": 0.5,
             "cfl_max_substeps": 460, "mpm_young_modulus": 400000.0,
             "mpm_poisson": 0.35, "mpm_yield_stress": 2000.0,
             "mpm_viscosity": 30.0,
             "_inflow": {"velocity_z": -0.12, "velocity_jitter": 0.04}},
    "slime": {**_VE, "resolution": 96, "dt": 0.0015, "cfl_factor": 0.5,
              "cfl_max_substeps": 200, "mpm_young_modulus": 80000.0,
              "mpm_poisson": 0.4, "mpm_yield_stress": 250.0,
              "mpm_viscosity": 80.0, "mpm_floor_friction": 0.5,
              "_inflow": {"velocity_z": -0.2, "velocity_jitter": 0.05}},
    "chocolate": {**_VE, "resolution": 128, "dt": 0.0012, "cfl_factor": 0.5,
                  "cfl_max_substeps": 300, "mpm_young_modulus": 120000.0,
                  "mpm_poisson": 0.35, "mpm_yield_stress": 500.0,
                  "mpm_viscosity": 70.0,
                  "_inflow": {"velocity_z": -0.15, "velocity_jitter": 0.04}},
    "caramel": {**_VE, "resolution": 128, "dt": 0.0012, "cfl_factor": 0.5,
                "cfl_max_substeps": 340, "mpm_young_modulus": 180000.0,
                "mpm_poisson": 0.38, "mpm_yield_stress": 700.0,
                "mpm_viscosity": 130.0,
                "_inflow": {"velocity_z": -0.13, "velocity_jitter": 0.04}},
    "gel": {**_VE, "resolution": 96, "dt": 0.0015, "cfl_factor": 0.5,
            "cfl_max_substeps": 220, "mpm_young_modulus": 60000.0,
            "mpm_poisson": 0.4, "mpm_yield_stress": 6000.0,
            "mpm_viscosity": 20.0, "mpm_floor_friction": 0.3,
            "mpm_rpic_damping": 0.1,
            "_inflow": {"velocity_z": -0.2, "velocity_jitter": 0.0}},
}

# Bilingual dropdown items (RU / EN in one label).
PRESET_ENUM_ITEMS = [
    ("water", "Вода / Water", "Спокойный полив воды (96³). / Calm water pour."),
    ("splash", "Всплеск / Splash",
     "Энергичные брызги (128³). / Energetic splashing."),
    ("honey_slump", "Мёд тягучий / Honey (slump)",
     "Густой мёд держит форму и медленно оседает. / "
     "Thick honey holds shape and slowly slumps."),
    ("honey_coil", "Мёд завивается / Honey (coil)",
     "Тонкая струйка мёда наматывается петлями. / "
     "Thin honey stream winds up in coils."),
    ("clay", "Глина/тесто / Clay",
     "Жёсткая масса складывается крупными витками. / "
     "Stiff paste folds in big coils."),
    ("slime", "Слайм / Slime",
     "Мягкий тягучий слайм. / Soft stretchy slime."),
    ("chocolate", "Шоколад / Chocolate",
     "Растопленный шоколад вьётся лентой. / Melted chocolate ribbon coils."),
    ("caramel", "Карамель / Caramel",
     "Тягучая карамель, длинные нити. / Stringy caramel coils."),
    ("gel", "Гель / Gel", "Упругий гель/желе, подрагивает. / Springy jiggly gel."),
    ("draft", "Черновик / Draft (fast)",
     "Грубая сетка 48³ для быстрого превью. / Coarse 48³ fast preview."),
]

HINT_TEXT = {
    "ru": [
        "Режим «Кухарка»: выбери жижу из списка и жми Bake. Всё остальное само.",
        "Завивается (coil): мёд/глина/шоколад/карамель — нужен тонкий поток сверху.",
        "Хочешь крутить вручную? Включи Pro в настройках аддона.",
        "Завивание любит МЕДЛЕННЫЙ полив (пресет ставит сам) и липкий пол.",
        "Не наматывается? Подними сопло пониже и ближе к полу, поток потоньше.",
    ],
    "en": [
        "Cook mode: pick a fluid and hit Bake. Everything else is automatic.",
        "Coil presets (honey/clay/chocolate/caramel) want a thin stream from above.",
        "Want to tweak by hand? Switch to Pro in the addon preferences.",
        "Coiling likes a SLOW pour (preset sets it) and a sticky floor.",
        "Not coiling? Lower the nozzle closer to the floor, make the stream thinner.",
    ],
}


def _apply_inflow_settings(context, spec) -> int:
    """Apply a preset's `_inflow` overrides to every inflow object. Returns the
    count touched. velocity_z overwrites the source's downward pour; the rest
    (velocity_jitter) maps 1:1 to GpufluidInflowProps fields."""
    n = 0
    for obj in context.scene.objects:
        ip = getattr(obj, "gpufluid_inflow", None)
        if ip is None or not getattr(ip, "is_inflow", False):
            continue
        if "velocity_z" in spec:
            vx, vy, _ = ip.velocity
            ip.velocity = (vx, vy, float(spec["velocity_z"]))
        if "velocity_jitter" in spec:
            ip.velocity_jitter = float(spec["velocity_jitter"])
        n += 1
    return n


@block("A8.8", "Helper operator family — apply material preset")
class GPUFLUID_OT_apply_preset(bpy.types.Operator):
    bl_idname = "gpufluid.apply_preset"
    bl_label = "Apply material preset"
    bl_description = ("Set the material + every stability knob + the pour to a "
                      "known-good combo for the chosen fluid — no need to "
                      "understand any individual setting")
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.EnumProperty(
        name="Preset", items=PRESET_ENUM_ITEMS, default="water")

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.gpufluid_domain.is_domain

    def execute(self, context):
        d = context.active_object.gpufluid_domain
        values = PRESETS.get(self.preset)
        if values is None:
            self.report({"WARNING"}, f"unknown preset {self.preset!r}")
            return {"CANCELLED"}
        inflow_spec = None
        for key, val in values.items():
            if key == "_inflow":
                inflow_spec = val
                continue
            setattr(d, key, val)
        n_inflow = _apply_inflow_settings(context, inflow_spec) if inflow_spec else 0
        label = dict((k, lbl) for k, lbl, _ in PRESET_ENUM_ITEMS)[self.preset]
        extra = (f", {n_inflow} inflow(s) set to a slow coiling pour"
                 if inflow_spec else "")
        self.report({"INFO"}, f"Preset: {label.split(' / ')[0]} "
                              f"(res={values['resolution']}, "
                              f"material={values['mpm_material']}, CFL on{extra})")
        return {"FINISHED"}
