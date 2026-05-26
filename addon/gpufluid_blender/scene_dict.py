"""[Round-15] Typed contract for the addon → CLI scene_dict.

Senior architect (round-12) flagged `scene_dict: Dict[str, Any]` plumbed
through bake.collect_scene → config_builder.build_toml as the single
biggest code smell: round-9's `_emit_table` data-loss bug would have
been caught at compile time by a typed shape. Round-15 introduces:

  1. `SceneDict` + section TypedDicts — document the contract.
  2. `validate_scene_dict(d)` — runtime check called from `build_toml`
     before emit. Raises `SceneDictError` with the offending key path
     instead of letting a malformed dict produce ill-typed TOML.

This is the "lite" version of the senior's day-1 #3:
  - Kept `_emit_scalar/_emit_table` (post-round-10 hardening, key-
    context errors, inf-reject — pragmatically solid).
  - Did NOT vendor `tomli_w` (not in Blender's bundled Python or our
    venv; vendoring an external library into an addon zip is a
    separate decision, filed in BACKLOG).
  - DID add the typed surface + validator (the actual safety win).

TypedDict (PEP 589) was chosen over `@dataclass` because:
  * Zero runtime overhead — it's just a dict at runtime, all type info
    is for static checkers / IDE.
  * Operates on the existing dict-shaped scene_dict without forcing
    a migration of every collect_scene return / test fixture.
  * Validator does runtime checks via a separate schema dict — separation
    of type-hint vs runtime-check is intentional (dataclass would conflate
    them and break the dict-of-dicts assumptions baked into `_emit_table`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Union


# ─── TypedDict shapes (static-only — runtime sees plain dicts) ──────────

class DomainDict(TypedDict):
    resolution: List[int]       # [rx, ry, rz]
    dx: float


class MpmDict(TypedDict, total=False):
    bulk_modulus: float
    rpic_damping: float
    grid_v_damping: float
    cube_friction: float
    v_terminal: float
    vz_max_splash: float
    initial_velocity: float


class SimulationDict(TypedDict, total=False):
    # required
    solver: str                  # "flip" | "mpm"
    dt: float
    frames: int
    fps: int
    gravity: float
    # FLIP-only (emitted conditionally, see config_builder gating):
    pressure_iters: int
    pressure_solver: str
    flip_blend: float
    surface_tension: float
    csf_smoothing_passes: int
    transfer_mode: str
    viscosity: float
    viscosity_iters: int
    # CFL (shared)
    cfl: bool
    cfl_factor: float
    cfl_max_substeps: int
    # Reseed (FLIP-leaning but emitted under either solver when reseed=True)
    reseed: bool
    reseed_every_n_frames: int
    reseed_min_per_cell: int
    reseed_max_per_cell: int
    # MPM nested sub-table
    mpm: MpmDict


class OutputDict(TypedDict, total=False):
    cache_dir: str
    mesh: bool
    iso_level: float
    smooth_passes: int
    mesh_smooth_passes: int
    mesh_smooth_method: str
    decimate_ratio: float
    wall_margin_cells: int
    usd: bool
    particles: bool
    preview: bool
    # Mesher / colormap escape-hatch fields (gate 0.3-b — set via Domain
    # toml_overrides). Validator allows them through; build_toml emits
    # whatever scalar/list shapes appear.
    mesh_method: str
    cubic_radius_cells: float
    sdf_particle_radius_cells: float
    sdf_search_radius_cells: float
    color_mix_mode: str
    temperature_colormap: str
    temperature_range: List[float]
    # Whitewater
    whitewater: bool
    whitewater_speed_threshold: float
    whitewater_lifetime_sec: float
    whitewater_emit_per_frame_max: int
    whitewater_total_cap: int
    whitewater_use_potential: bool
    whitewater_potential_radius: float
    whitewater_potential_v_max: float
    whitewater_wave_crest_weight: float
    whitewater_trapped_air_weight: float


class FluidBoxDict(TypedDict, total=False):
    type: str                   # "box"
    lo: List[float]
    hi: List[float]
    ppc: int
    color: List[float]
    temperature: float


class ObstacleDict(TypedDict, total=False):
    # discriminated union via `type` key; validator checks shape per type
    type: str                   # "box" | "sphere" | "cylinder_y" | "mesh" | "plane"
    center: List[float]
    half_size: List[float]
    radius: float
    half_height: float
    path: str
    scale: float
    translate: List[float]
    rotate_deg: List[float]
    point: List[float]
    normal: List[float]
    bbox_lo: List[float]
    bbox_hi: List[float]
    motion: Dict[str, Any]      # inline-table { kind, velocity, keyframes? }


class InflowDict(TypedDict, total=False):
    lo: List[float]
    hi: List[float]
    velocity: List[float]
    rate_per_sec: int
    frame_start: int
    frame_end: int
    keyframes: List[List[float]]
    color: List[float]
    temperature: float


class OutflowDict(TypedDict, total=False):
    lo: List[float]
    hi: List[float]
    frame_start: int
    frame_end: int


class SceneDict(TypedDict, total=False):
    """Top-level contract for the addon → CLI TOML scene config.

    Required at top-level (in production via collect_scene): `domain`,
    `simulation`, `output`. Optional arrays (PLURAL keys per the
    addon's collect_scene convention — singular would be a TOML
    section name, but scene_dict is the Python intermediate):
    `obstacles`, `inflows`, `outflows`, `fluids`.

    Round-16 reviewer caught: an earlier round-15 draft used singular
    keys (`obstacle`, `inflow`, `outflow`) — those NEVER match production
    data so the validator's array-shape + reversed-range checks were
    silently no-ops. Both validator + TypedDicts now use plural.
    """
    domain: DomainDict
    simulation: SimulationDict
    output: OutputDict
    fluids: List[FluidBoxDict]
    obstacles: List[ObstacleDict]
    inflows: List[InflowDict]
    outflows: List[OutflowDict]


# ─── Runtime validator ─────────────────────────────────────────────────

class SceneDictError(ValueError):
    """Raised when scene_dict fails the validate_scene_dict contract."""


# Schema rules: (key_path, required_type, allow_None_path) — checked
# top-down. `key_path` may be a literal key or a tuple-path like
# ("simulation", "frames"). `required_type` is a type or tuple of types
# accepted (None means "any").
_REQUIRED_TOP = {
    "domain": dict,
    "simulation": dict,
    "output": dict,
}
_REQUIRED_DOMAIN = {
    "resolution": (list, tuple),
    "dx": (int, float),
}
_REQUIRED_SIMULATION = {
    "solver": str,
    "dt": (int, float),
    "frames": int,
    "fps": int,
    "gravity": (int, float),
}
_REQUIRED_OUTPUT = {
    "cache_dir": str,
}
_ALLOWED_SOLVERS = ("flip", "mpm")


def _type_name(t) -> str:
    if isinstance(t, tuple):
        return "/".join(x.__name__ for x in t)
    return t.__name__


def _check_required(name: str, d: dict, schema: Dict[str, Any]) -> None:
    for key, expected_t in schema.items():
        if key not in d:
            raise SceneDictError(
                f"missing required key '{name}.{key}'")
        if not isinstance(d[key], expected_t):
            raise SceneDictError(
                f"'{name}.{key}' has type {type(d[key]).__name__}, "
                f"expected {_type_name(expected_t)}")


def validate_scene_dict(d: Any) -> None:
    """Runtime contract enforcement. Call from `build_toml` BEFORE
    emit so malformed dicts produce a clear ValueError instead of
    ill-typed TOML that the CLI then chokes on (or worse: silently
    drops fields as the round-9 `_emit_table` data-loss bug did).

    Semantics: **"validate what's there, don't require completeness"**.
    Sections that ARE present must have correct shape; sections that
    are absent are NOT required (the addon's `bake._collect_scene`
    always provides them via PropertyGroup defaults; partial dicts
    in test fixtures pre-date round-15 and remain valid input).
    If you need full-completeness check, call
    `validate_complete_scene_dict()` (used by bake.execute path).

    Raises `SceneDictError` (ValueError subclass) with a message that
    names the offending key path.
    """
    if not isinstance(d, dict):
        raise SceneDictError(
            f"scene_dict must be dict, got {type(d).__name__}")

    # Domain — validate IF present
    if "domain" in d:
        if not isinstance(d["domain"], dict):
            raise SceneDictError(
                f"'domain' must be dict, got {type(d['domain']).__name__}")
        if "resolution" in d["domain"]:
            res = d["domain"]["resolution"]
            if (not isinstance(res, (list, tuple)) or len(res) != 3
                    or not all(isinstance(x, int) and x > 0 for x in res)):
                raise SceneDictError(
                    f"'domain.resolution' must be a 3-list of positive ints, "
                    f"got {res!r}")
        if "dx" in d["domain"] and not isinstance(
                d["domain"]["dx"], (int, float)):
            raise SceneDictError(
                f"'domain.dx' must be numeric, "
                f"got {type(d['domain']['dx']).__name__}")

    # Simulation — validate IF present
    if "simulation" in d:
        if not isinstance(d["simulation"], dict):
            raise SceneDictError(
                f"'simulation' must be dict, got "
                f"{type(d['simulation']).__name__}")
        sim = d["simulation"]
        if "solver" in sim:
            if not isinstance(sim["solver"], str):
                raise SceneDictError(
                    f"'simulation.solver' must be str, got "
                    f"{type(sim['solver']).__name__}")
            if sim["solver"] not in _ALLOWED_SOLVERS:
                raise SceneDictError(
                    f"'simulation.solver' is {sim['solver']!r}, "
                    f"expected one of {_ALLOWED_SOLVERS}")
        if "frames" in sim:
            if not isinstance(sim["frames"], int):
                raise SceneDictError(
                    f"'simulation.frames' must be int, got "
                    f"{type(sim['frames']).__name__}")
            if sim["frames"] <= 0:
                raise SceneDictError(
                    f"'simulation.frames' must be > 0, got {sim['frames']}")
        # Round-16: type-check numeric sim sub-fields when present.
        # We don't REQUIRE them (test fixtures pre-round-15 use partial
        # dicts) but if a string/None sneaks in we want a clear key-path
        # error instead of a KeyError-style crash deep in build_toml.
        for fkey in ("dt", "fps", "gravity"):
            if fkey in sim and not isinstance(sim[fkey], (int, float, bool)):
                raise SceneDictError(
                    f"'simulation.{fkey}' must be numeric, got "
                    f"{type(sim[fkey]).__name__}")

    # Output — validate IF present
    if "output" in d:
        if not isinstance(d["output"], dict):
            raise SceneDictError(
                f"'output' must be dict, got {type(d['output']).__name__}")
        if "cache_dir" in d["output"] and not isinstance(
                d["output"]["cache_dir"], str):
            raise SceneDictError(
                f"'output.cache_dir' must be str, got "
                f"{type(d['output']['cache_dir']).__name__}")

    # Array-of-tables sections (optional, but if present must be list-of-dict).
    # PLURAL keys — round-16 reviewer caught the singular bug that made
    # this whole loop a no-op on real production scene_dicts.
    for arr_key in ("obstacles", "inflows", "outflows", "fluids"):
        if arr_key in d:
            if not isinstance(d[arr_key], list):
                raise SceneDictError(
                    f"'{arr_key}' must be a list (array of tables), "
                    f"got {type(d[arr_key]).__name__}")
            for i, entry in enumerate(d[arr_key]):
                if not isinstance(entry, dict):
                    raise SceneDictError(
                        f"'{arr_key}[{i}]' must be dict, "
                        f"got {type(entry).__name__}")
                # Inflow / outflow timing sanity (mirrors bake.execute
                # guard so emit-time can't slip past the check either).
                if arr_key in ("inflows", "outflows"):
                    fs = entry.get("frame_start")
                    fe = entry.get("frame_end")
                    if (isinstance(fs, int) and isinstance(fe, int)
                            and fs > fe):
                        raise SceneDictError(
                            f"'{arr_key}[{i}]' frame_start ({fs}) > "
                            f"frame_end ({fe}) — empty emission window")

    # NOTE on MPM sub-table contract: an earlier round-15 draft tried to
    # enforce "solver=mpm requires [simulation.mpm]" AND "solver=flip
    # forbids [simulation.mpm]" at validate-time. That was the wrong
    # contract — the real addon flow lets `_collect.py` decide whether
    # MPM tuning props differ enough from defaults to emit the sub-table,
    # and round-1's overrides escape-hatch can produce arbitrary cross-
    # solver scene_dicts. config_builder gating (emit only FLIP-only
    # fields when solver=flip; emit only [simulation.mpm] when solver=mpm)
    # is the actual gate. Validator stays out of solver-specific shape
    # claims — lesson 9.4 (don't bake into a stricter contract than
    # production currently honours; broken tests would be the warning).
