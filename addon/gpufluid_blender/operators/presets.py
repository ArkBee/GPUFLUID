"""[BLK A8.8] One-click physics presets for the Domain.

The individual stability knobs (bulk modulus, dt, CFL substepping, RPIC
damping) are physics jargon — a non-specialist can't guess a stable combo,
and a wrong combo diverges to NaN mid-bake. These presets bundle known-good
combinations behind a single dropdown so the common scenarios "just work"
without touching any individual knob.

The preset tables (`PRESETS`) are pure data — a test loads this module under
a minimal bpy stub and asserts the combos directly
(tests/test_addon_presets.py). Only the operator class needs bpy proper.
"""
import bpy

try:
    from .._blocks import block
except ImportError:
    # dodge: standalone test load (spec_from_file_location) has no parent
    # package, so the relative import fails — registration is irrelevant there.
    def block(_bid, _desc=""):
        def _w(fn):
            return fn
        return _w


# ── preset value tables (pure data — no bpy needed) ────────────────────────
# Each preset writes these Domain-property fields. Every preset turns CFL
# substepping ON (use_cfl=True): without it a lively inflow blows the MPM
# pressure stack to NaN within ~20 steps (live-found 2026-06-15). The combos
# below were validated to run a full 120-frame inflow cascade without NaN.
PRESETS = {
    "water": {
        "solver": "mpm", "resolution": 96, "dt": 0.004,
        "use_cfl": True, "cfl_factor": 0.4, "cfl_max_substeps": 24,
        "mpm_bulk_modulus": 900.0, "mpm_rpic_damping": 0.15,
        "mpm_grid_v_damping": 0.998, "mpm_initial_velocity": -0.3,
    },
    "splash": {
        "solver": "mpm", "resolution": 128, "dt": 0.003,
        "use_cfl": True, "cfl_factor": 0.4, "cfl_max_substeps": 32,
        "mpm_bulk_modulus": 1200.0, "mpm_rpic_damping": 0.05,
        "mpm_grid_v_damping": 0.999, "mpm_vz_max_splash": 0.6,
        "mpm_initial_velocity": -0.6,
    },
    "viscous": {
        "solver": "mpm", "resolution": 96, "dt": 0.004,
        "use_cfl": True, "cfl_factor": 0.4, "cfl_max_substeps": 24,
        "mpm_bulk_modulus": 700.0, "mpm_rpic_damping": 0.45,
        "mpm_grid_v_damping": 0.99, "mpm_initial_velocity": -0.2,
    },
    "draft": {
        "solver": "mpm", "resolution": 48, "dt": 0.005,
        "use_cfl": True, "cfl_factor": 0.5, "cfl_max_substeps": 16,
        "mpm_bulk_modulus": 900.0, "mpm_rpic_damping": 0.15,
        "mpm_grid_v_damping": 0.998, "mpm_initial_velocity": -0.3,
    },
}

# Bilingual enum items. Labels carry BOTH languages so the dropdown reads
# the same regardless of the hint-language preference (the per-knob help
# text below is what actually switches).
PRESET_ENUM_ITEMS = [
    ("water", "Вода — стабильно / Water — stable",
     "Стабильный полив/каскад воды. Безопасные настройки, 96³. / "
     "Stable water pour/cascade. Safe stability settings, 96³."),
    ("splash", "Всплеск / Splash",
     "Энергичные брызги, мельче сетка (128³), живее. Медленнее бейк. / "
     "Energetic splashing, finer grid (128³), livelier. Slower bake."),
    ("viscous", "Вязкое — мёд / Viscous — honey",
     "Густая тягучая жидкость (мёд/лава). Высокое демпфирование. / "
     "Thick syrupy fluid (honey/lava). High damping."),
    ("draft", "Черновик — быстро / Draft — fast",
     "Грубая сетка (48³) для быстрого превью. Прогнать и посмотреть. / "
     "Coarse grid (48³) for a fast preview. Run-and-look."),
]

# Per-knob help, switched by the addon's hint-language preference. Kept here
# (not in panels.py) so the bilingual strings live with the preset data.
HINT_TEXT = {
    "ru": [
        "Не уверен что крутить? Выбери пресет выше — он выставит всё сам.",
        "Solver: MPM — для всего (вода, вязкое, на препятствиях).",
        "Resolution: детализация. Больше = чётче, но медленнее.",
        "Bulk Modulus (жёсткость): выше = «жёстче» вода, но менее стабильно.",
        "Viscosity (RPIC): 0 = живая вода, 0.3+ = как сироп.",
        "CFL Substepping: дробит шаг при быстром потоке — спасает от взрыва (NaN).",
    ],
    "en": [
        "Not sure what to tweak? Pick a preset above — it sets everything.",
        "Solver: MPM — for everything (water, viscous, on obstacles).",
        "Resolution: detail. Higher = sharper but slower.",
        "Bulk Modulus (stiffness): higher = 'harder' water but less stable.",
        "Viscosity (RPIC): 0 = lively water, 0.3+ = syrupy.",
        "CFL Substepping: splits dt on fast flow — prevents blow-up (NaN).",
    ],
}


@block("A8.8", "Helper operator family — apply physics preset")
class GPUFLUID_OT_apply_preset(bpy.types.Operator):
    bl_idname = "gpufluid.apply_preset"
    bl_label = "Apply physics preset"
    bl_description = ("Set all stability knobs (bulk modulus, dt, CFL, "
                      "damping, resolution) to a known-good combo for the "
                      "chosen scenario — no need to understand each knob")
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
        for key, val in values.items():
            setattr(d, key, val)
        label = dict((k, lbl) for k, lbl, _ in PRESET_ENUM_ITEMS)[self.preset]
        self.report({"INFO"}, f"Preset applied: {label.split(' / ')[0]} "
                              f"(res={values['resolution']}, "
                              f"bulk={values['mpm_bulk_modulus']:.0f}, CFL on)")
        return {"FINISHED"}
