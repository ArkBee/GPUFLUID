"""[Layer C7 / BLK C7.1] TOML scene config schema and validator.

Example::

    [domain]
    resolution = [64, 64, 64]
    dx = 0.015625    # optional; defaults to 1/max(resolution)

    [fluid]
    type = "box"
    lo = [0.05, 0.05, 0.05]
    hi = [0.40, 0.70, 0.40]
    ppc = 8

    [[obstacle]]
    type = "sphere"
    center = [0.5, 0.3, 0.5]
    radius = 0.10

    [[obstacle]]
    type = "cylinder_y"
    center = [0.30, 0.30, 0.60]
    radius = 0.07
    half_height = 0.30

    [[obstacle]]
    type = "mesh"
    path = "obstacles/bunny.obj"
    scale = 0.25
    translate = [0.5, 0.0, 0.5]

    [simulation]
    dt = 0.005
    pressure_iters = 60
    flip_blend = 0.95
    gravity = -9.81
    frames = 250
    fps = 24

    [output]
    cache_dir = "out/scene01"
    mesh = true
    iso_level = 0.6
    smooth_passes = 2
    particles = false
    preview = false
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

from ..blocks import BlockError

if sys.version_info >= (3, 11):
    import tomllib  # stdlib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass
class DomainCfg:
    resolution: Tuple[int, int, int] = (64, 64, 64)
    dx: Optional[float] = None
    # FU-016 (2026-06-01): the REAL world extent of the domain in metres,
    # (sx, sy, sz). The Blender addon knows it (DomainTransform.dom_size) and
    # now forwards it. None for hand-written standalone TOMLs that don't set
    # it — Scene.world_size then falls back to the resolution aspect-ratio
    # `domain_size` (pre-FU-016 behaviour, correct only for a 1 m unit cube).
    size_world: Optional[Tuple[float, float, float]] = None


@dataclass
class FluidBoxCfg:
    type: Literal["box"] = "box"
    lo: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    hi: Tuple[float, float, float] = (0.40, 0.70, 0.40)
    ppc: int = 8
    color: Optional[Tuple[float, float, float]] = None   # S2.15: per-source RGB
    temperature: Optional[float] = None                  # B11.3 / S2.18: per-source scalar


@dataclass
class FluidMeshCfg:
    type: Literal["mesh"] = "mesh"
    path: str = ""
    scale: float = 1.0
    translate: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate_deg: Optional[Tuple[float, float, float]] = None
    ppc: int = 8
    color: Optional[Tuple[float, float, float]] = None
    temperature: Optional[float] = None                  # B11.3 / S2.18: per-source scalar


@dataclass
class ObstacleSphereCfg:
    type: Literal["sphere"]
    center: Tuple[float, float, float]
    radius: float


@dataclass
class ObstacleBoxCfg:
    type: Literal["box"]
    center: Tuple[float, float, float]
    half_size: Tuple[float, float, float]
    # Round-57: optional OBB rotation. 3×3 matrix as nested tuple, rows
    # in row-major. Columns are world-space box-local +X/+Y/+Z axes
    # (orthonormal — no scale folded in). None ⇒ axis-aligned.
    rotation: Optional[Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]] = None


@dataclass
class ObstacleCylinderYCfg:
    type: Literal["cylinder_y"]
    center: Tuple[float, float, float]
    radius: float
    half_height: float


@dataclass
class ObstacleMeshCfg:
    type: Literal["mesh"]
    path: str
    scale: float = 1.0
    translate: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate_deg: Optional[Tuple[float, float, float]] = None


@dataclass
class ObstaclePlaneCfg:
    type: Literal["plane"]
    point: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    bbox_lo: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # plane * bbox = finite ramp
    bbox_hi: Tuple[float, float, float] = (1.0, 1.0, 1.0)


# v0.5 — animated obstacle motion
@dataclass
class MotionCfg:
    kind: Literal["linear", "keyframes"]
    velocity: Optional[Tuple[float, float, float]] = None
    keyframes: Optional[List[Tuple[int, Tuple[float, float, float]]]] = None


@dataclass
class InflowCfg:
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rate_per_sec: float = 5000.0
    frame_start: int = 0
    frame_end: int = 1_000_000
    # S2.17.7.MOVING — optional per-frame AABBs for animated MPM emitters.
    # Each row: [frame, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z]. When set, this
    # supersedes the static lo/hi above (those become the keyframe at
    # frame_start for backward compat). Ignored by FLIP solver.
    keyframes: List[List[float]] = field(default_factory=list)
    # B18 — per-source per-particle attributes. Each particle emitted by
    # this inflow inherits the same colour / temperature. Consumed by the
    # MPM path (FlipSolver3D inflows currently emit white; FLIP-side wire-
    # up is tracked under B2-revival).
    color: Optional[Tuple[float, float, float]] = None
    temperature: Optional[float] = None


@dataclass
class OutflowCfg:
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    frame_start: int = 0
    frame_end: int = 1_000_000


ObstacleCfg = Union[ObstacleSphereCfg, ObstacleBoxCfg, ObstacleCylinderYCfg,
                    ObstacleMeshCfg, ObstaclePlaneCfg]


@dataclass
class SimulationCfg:
    solver: str = "flip"               # "flip" (FlipSolver3D) or "mpm" (F3.7 MpmSolver) — B17.12
    dt: float = 0.005
    pressure_iters: int = 60
    pressure_solver: str = "jacobi"   # "jacobi" or "pcg" (v0.4+)
    cfl: bool = False                  # if True, dispatch via step_cfl
    cfl_factor: float = 0.5
    cfl_max_substeps: int = 16
    flip_blend: float = 0.95
    gravity: float = -9.81
    rho: float = 1.0
    viscosity: float = 0.0
    viscosity_iters: int = 12
    surface_tension: float = 0.0    # S2.14 CSF coefficient (m³/s² in SI ≈ σ/ρ)
    csf_smoothing_passes: int = 2   # box-blur passes for χ → χ̃
    transfer_mode: str = "flip"   # "flip" | "pic" | "apic"
    reseed: bool = False           # S2.11 particle reseeding
    reseed_every_n_frames: int = 5
    reseed_min_per_cell: int = 4
    reseed_max_per_cell: int = 16
    frames: int = 100
    fps: int = 24
    # MPM-specific knobs (consumed only when solver == "mpm")
    mpm_bulk_modulus: float = 1500.0
    mpm_rpic_damping: float = 0.15
    mpm_grid_v_damping: float = 0.998
    mpm_cube_friction: float = 0.6
    mpm_v_terminal: float = -1.0
    mpm_vz_max_splash: float = 0.3
    mpm_initial_velocity: float = -0.3


@dataclass
class OutputCfg:
    cache_dir: str = "out/sim"
    mesh: bool = True
    iso_level: float = 0.6
    smooth_passes: int = 2
    mesh_smooth_passes: int = 0
    mesh_smooth_method: str = "taubin"
    decimate_ratio: float = 1.0   # M5.6 — keep this fraction of faces
    wall_margin_cells: int = 0   # M5.7 wall mask
    particles: bool = False
    # M5.9: opt-in to wider cubic kernel particle scatter for smoother
    # surfaces. Default "trilinear" preserves M5.1 behaviour bit-for-bit.
    # Requires iso_level to be raised ~5-10× vs M5.1 (cubic kernel
    # concentrates more mass at peak — see DESIGN.md §8.3).
    mesh_method: str = "trilinear"     # "trilinear" (M5.1) | "cubic" (M5.9) | "sdf" (M5.11)
    cubic_radius_cells: float = 2.0    # M5.9 support radius in cells
    # M5.11 Bridson SDF (DESIGN.md §8.5). When mesh_method="sdf":
    # iso_level is IGNORED — MC threshold is forced to 0 (zero level set).
    # particle_radius = radius of each particle-sphere; surface = union.
    # search_radius = max distance a cell looks for nearest particle
    # (cells with no particle in range get phi = +search_radius_world).
    sdf_particle_radius_cells: float = 1.0
    sdf_search_radius_cells: float = 4.0
    # B18.5 — per-vertex colour mixing mode. "linear" = inverse-distance²
    # RGB blend (GPU, fast, blue+yellow=grey). "mixbox" = pigment-space
    # blend in Mixbox latent space (CPU, slower, blue+yellow=green). "off"
    # disables per-vertex colour writing entirely. Default is "linear" so
    # bakes that DO carry per-particle colour render with visible
    # variation out of the box; users opt into mixbox.
    color_mix_mode: str = "linear"
    # B11.3 — when set (e.g. "blackbody"), the mesher derives per-vertex
    # RGB from the per-particle TEMPERATURE sidecar instead of the COLOUR
    # sidecar. Range is set by ``temperature_range = [t_min, t_max]``
    # below. Default `None` keeps the regular colour path intact.
    temperature_colormap: Optional[str] = None
    temperature_range: Tuple[float, float] = (20.0, 1500.0)
    preview: bool = False
    usd: bool = False            # I6.5 — write cache.usdc alongside the PLY sequence
    whitewater: bool = False     # W7.x — emit foam/spray particles
    whitewater_speed_threshold: float = 4.0
    whitewater_lifetime_sec: float = 1.5
    whitewater_emit_per_frame_max: int = 4000
    whitewater_total_cap: int = 80000
    # W7.7 trapped-air potential (B3.3): when True, the emit selector samples
    # particles weighted by the Ihmsen-2012 trapped-air potential instead of
    # uniform-random over the speed-gated set. Off by default for back-compat.
    whitewater_use_potential: bool = False
    whitewater_potential_radius: float = 0.0   # 0 ⇒ auto = 2.5·dx
    whitewater_potential_v_max: float = 10.0   # velocity normaliser
    # B3.2 — when use_potential AND wave_crest_weight > 0, emit weight is
    # `alpha * I_ta + beta * I_wc`. I_wc fires on surface curvature
    # (wave crests, breaking ridges) where I_ta misses static-but-curving
    # geometry. Default beta=1.0 (equal blend with trapped-air) once on.
    whitewater_wave_crest_weight: float = 0.0
    whitewater_trapped_air_weight: float = 1.0


@dataclass
class SceneCfg:
    domain: DomainCfg = field(default_factory=DomainCfg)
    fluid: object = field(default_factory=FluidBoxCfg)  # FluidBoxCfg | FluidMeshCfg
    fluids: List[object] = field(default_factory=list)  # S2.15: optional multi-source list (each may have `color`)
    obstacle: List[ObstacleCfg] = field(default_factory=list)
    obstacle_motion: List[Optional[MotionCfg]] = field(default_factory=list)  # per-obstacle, same len as obstacle
    inflow: List[InflowCfg] = field(default_factory=list)
    outflow: List[OutflowCfg] = field(default_factory=list)
    simulation: SimulationCfg = field(default_factory=SimulationCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    # Resolved (set during validation):
    config_dir: Optional[Path] = None  # for resolving relative obstacle paths

    @property
    def dx(self) -> float:
        if self.domain.dx is not None:
            return self.domain.dx
        return 1.0 / max(self.domain.resolution)

    @property
    def domain_size(self) -> Tuple[float, float, float]:
        """Normalised [0,1]³ extent (resolution aspect-ratio). This is NOT
        metres — it is nx/max(res) per axis. Kept for callers that genuinely
        want the normalised extent; use `world_size` for real-metre physics
        (gravity/velocity) scaling."""
        nx, ny, nz = self.domain.resolution
        return (nx * self.dx, ny * self.dx, nz * self.dx)

    @property
    def world_size(self) -> Tuple[float, float, float]:
        """Real world extent of the domain in metres. FU-016: prefer the
        forwarded `domain.size_world` (the addon knows it); fall back to the
        normalised `domain_size` when absent (hand-written TOML). The fallback
        is only physically correct for a 1 m unit cube — see FU-016."""
        if self.domain.size_world is not None:
            return self.domain.size_world
        return self.domain_size


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _tuple(seq, n, name):
    if not isinstance(seq, Sequence) or len(seq) != n:
        raise BlockError("C7.1", f"{name}: expected list of {n}; got {seq!r}")
    return tuple(seq)


def _parse_motion(d: dict) -> MotionCfg:
    k = d.get("kind")
    if k == "linear":
        return MotionCfg(kind="linear",
                         velocity=_tuple(d.get("velocity", [0, 0, 0]), 3, "motion.velocity"))
    if k == "keyframes":
        raw_kfs = d.get("keyframes", [])
        kfs = []
        for kf in raw_kfs:
            if not isinstance(kf, (list, tuple)) or len(kf) != 2:
                raise BlockError("C7.1", f"keyframe must be [frame, [x,y,z]]; got {kf!r}")
            kfs.append((int(kf[0]), _tuple(kf[1], 3, "motion.keyframe.pos")))
        return MotionCfg(kind="keyframes", keyframes=kfs)
    raise BlockError("C7.1", f"unknown motion kind: {k!r}")


def _parse_inflow(d: dict) -> InflowCfg:
    return InflowCfg(
        lo=_tuple(d["lo"], 3, "inflow.lo"),
        hi=_tuple(d["hi"], 3, "inflow.hi"),
        velocity=_tuple(d.get("velocity", [0, 0, 0]), 3, "inflow.velocity"),
        rate_per_sec=float(d.get("rate_per_sec", 5000.0)),
        frame_start=int(d.get("frame_start", 0)),
        frame_end=int(d.get("frame_end", 1_000_000)),
        keyframes=[
            [float(v) for v in row]
            for row in d.get("keyframes", [])
        ],
        color=_tuple(d["color"], 3, "inflow.color") if "color" in d else None,
        temperature=float(d["temperature"]) if "temperature" in d else None,
    )


def _parse_outflow(d: dict) -> OutflowCfg:
    return OutflowCfg(
        lo=_tuple(d["lo"], 3, "outflow.lo"),
        hi=_tuple(d["hi"], 3, "outflow.hi"),
        frame_start=int(d.get("frame_start", 0)),
        frame_end=int(d.get("frame_end", 1_000_000)),
    )


def _parse_obstacle(d: dict) -> ObstacleCfg:
    t = d.get("type")
    if t == "sphere":
        return ObstacleSphereCfg(
            type="sphere",
            center=_tuple(d["center"], 3, "obstacle.center"),
            radius=float(d["radius"]),
        )
    if t == "box":
        # Round-57: optional `rotation` as a 3×3 nested list. Each row
        # is parsed independently; absence ⇒ axis-aligned (rotation=None).
        rot_raw = d.get("rotation")
        rotation = None
        if rot_raw is not None:
            if len(rot_raw) != 3:
                raise BlockError("C7.1",
                    "obstacle.rotation must be a 3×3 matrix (got "
                    f"{len(rot_raw)} rows)")
            rotation = tuple(
                _tuple(row, 3, f"obstacle.rotation[{i}]")
                for i, row in enumerate(rot_raw)
            )
        return ObstacleBoxCfg(
            type="box",
            center=_tuple(d["center"], 3, "obstacle.center"),
            half_size=_tuple(d["half_size"], 3, "obstacle.half_size"),
            rotation=rotation,
        )
    if t == "cylinder_y":
        return ObstacleCylinderYCfg(
            type="cylinder_y",
            center=_tuple(d["center"], 3, "obstacle.center"),
            radius=float(d["radius"]),
            half_height=float(d["half_height"]),
        )
    if t == "plane":
        return ObstaclePlaneCfg(
            type="plane",
            point=_tuple(d["point"], 3, "obstacle.point"),
            normal=_tuple(d["normal"], 3, "obstacle.normal"),
            bbox_lo=_tuple(d.get("bbox_lo", [0, 0, 0]), 3, "obstacle.bbox_lo"),
            bbox_hi=_tuple(d.get("bbox_hi", [1, 1, 1]), 3, "obstacle.bbox_hi"),
        )
    if t == "mesh":
        return ObstacleMeshCfg(
            type="mesh",
            path=str(d["path"]),
            scale=float(d.get("scale", 1.0)),
            translate=_tuple(d.get("translate", [0, 0, 0]), 3, "obstacle.translate"),
            rotate_deg=_tuple(d["rotate_deg"], 3, "obstacle.rotate_deg") if "rotate_deg" in d else None,
        )
    raise BlockError("C7.1", f"unknown obstacle type: {t!r}")


# [BLK C7.1]
def load_scene(path: Union[str, Path]) -> SceneCfg:
    """Load and validate a TOML scene config. Returns a SceneCfg."""
    p = Path(path)
    if not p.exists():
        raise BlockError("C7.1", f"config not found: {p}")
    with open(p, "rb") as f:
        raw = tomllib.load(f)

    dom = raw.get("domain", {})
    domain = DomainCfg(
        resolution=_tuple(dom.get("resolution", [64, 64, 64]), 3, "domain.resolution"),
        dx=float(dom["dx"]) if "dx" in dom else None,
        # FU-016: real world extent in metres, forwarded by the addon.
        size_world=(tuple(float(v) for v in _tuple(
            dom["size_world"], 3, "domain.size_world"))
            if "size_world" in dom else None),
    )
    def _parse_fluid_entry(fl: dict) -> object:
        ftype = fl.get("type", "box")
        color = _tuple(fl["color"], 3, "fluid.color") if "color" in fl else None
        # B11.3 / S2.18 — per-source scalar (temperature). Plain float; the
        # solver attaches it as a per-particle attribute via seed_box/seed_mesh.
        temperature = float(fl["temperature"]) if "temperature" in fl else None
        if ftype == "box":
            return FluidBoxCfg(
                type="box",
                lo=_tuple(fl.get("lo", [0.05, 0.05, 0.05]), 3, "fluid.lo"),
                hi=_tuple(fl.get("hi", [0.40, 0.70, 0.40]), 3, "fluid.hi"),
                ppc=int(fl.get("ppc", 8)),
                color=color,
                temperature=temperature,
            )
        if ftype == "mesh":
            return FluidMeshCfg(
                type="mesh",
                path=str(fl.get("path", "")),
                scale=float(fl.get("scale", 1.0)),
                translate=_tuple(fl.get("translate", [0, 0, 0]), 3, "fluid.translate"),
                rotate_deg=_tuple(fl["rotate_deg"], 3, "fluid.rotate_deg") if "rotate_deg" in fl else None,
                ppc=int(fl.get("ppc", 8)),
                color=color,
                temperature=temperature,
            )
        raise BlockError("C7.1", f"unknown fluid type: {ftype!r}")

    fl = raw.get("fluid", {})
    fluid = _parse_fluid_entry(fl)
    # S2.15: optional [[fluids]] array (multi-source); each may carry `color`.
    fluids_raw = raw.get("fluids", [])
    fluids = [_parse_fluid_entry(f) for f in fluids_raw]
    obstacles = []
    obstacle_motions = []
    for d in raw.get("obstacle", []):
        obstacles.append(_parse_obstacle(d))
        obstacle_motions.append(_parse_motion(d["motion"]) if "motion" in d else None)
    inflows = [_parse_inflow(d) for d in raw.get("inflow", [])]
    outflows = [_parse_outflow(d) for d in raw.get("outflow", [])]

    sim = raw.get("simulation", {})
    mpm_sub = sim.get("mpm", {})  # nested [simulation.mpm] subsection
    simulation = SimulationCfg(
        solver=str(sim.get("solver", "flip")),
        dt=float(sim.get("dt", 0.005)),
        pressure_iters=int(sim.get("pressure_iters", 60)),
        pressure_solver=str(sim.get("pressure_solver", "jacobi")),
        cfl=bool(sim.get("cfl", False)),
        cfl_factor=float(sim.get("cfl_factor", 0.5)),
        cfl_max_substeps=int(sim.get("cfl_max_substeps", 16)),
        flip_blend=float(sim.get("flip_blend", 0.95)),
        gravity=float(sim.get("gravity", -9.81)),
        rho=float(sim.get("rho", 1.0)),
        viscosity=float(sim.get("viscosity", 0.0)),
        viscosity_iters=int(sim.get("viscosity_iters", 12)),
        surface_tension=float(sim.get("surface_tension", 0.0)),
        csf_smoothing_passes=int(sim.get("csf_smoothing_passes", 2)),
        transfer_mode=str(sim.get("transfer_mode", "flip")),
        reseed=bool(sim.get("reseed", False)),
        reseed_every_n_frames=int(sim.get("reseed_every_n_frames", 5)),
        reseed_min_per_cell=int(sim.get("reseed_min_per_cell", 4)),
        reseed_max_per_cell=int(sim.get("reseed_max_per_cell", 16)),
        frames=int(sim.get("frames", 100)),
        fps=int(sim.get("fps", 24)),
        mpm_bulk_modulus=float(mpm_sub.get("bulk_modulus", 1500.0)),
        mpm_rpic_damping=float(mpm_sub.get("rpic_damping", 0.15)),
        mpm_grid_v_damping=float(mpm_sub.get("grid_v_damping", 0.998)),
        mpm_cube_friction=float(mpm_sub.get("cube_friction", 0.6)),
        mpm_v_terminal=float(mpm_sub.get("v_terminal", -1.0)),
        mpm_vz_max_splash=float(mpm_sub.get("vz_max_splash", 0.3)),
        mpm_initial_velocity=float(mpm_sub.get("initial_velocity", -0.3)),
    )

    out = raw.get("output", {})
    output = OutputCfg(
        cache_dir=str(out.get("cache_dir", "out/sim")),
        mesh=bool(out.get("mesh", True)),
        iso_level=float(out.get("iso_level", 0.6)),
        smooth_passes=int(out.get("smooth_passes", 2)),
        mesh_smooth_passes=int(out.get("mesh_smooth_passes", 0)),
        mesh_smooth_method=str(out.get("mesh_smooth_method", "taubin")),
        decimate_ratio=float(out.get("decimate_ratio", 1.0)),
        wall_margin_cells=int(out.get("wall_margin_cells", 0)),
        usd=bool(out.get("usd", False)),
        whitewater=bool(out.get("whitewater", False)),
        whitewater_speed_threshold=float(out.get("whitewater_speed_threshold", 4.0)),
        whitewater_lifetime_sec=float(out.get("whitewater_lifetime_sec", 1.5)),
        whitewater_emit_per_frame_max=int(out.get("whitewater_emit_per_frame_max", 4000)),
        whitewater_total_cap=int(out.get("whitewater_total_cap", 80000)),
        whitewater_use_potential=bool(out.get("whitewater_use_potential", False)),
        whitewater_potential_radius=float(out.get("whitewater_potential_radius", 0.0)),
        whitewater_potential_v_max=float(out.get("whitewater_potential_v_max", 10.0)),
        whitewater_wave_crest_weight=float(out.get("whitewater_wave_crest_weight", 0.0)),
        whitewater_trapped_air_weight=float(out.get("whitewater_trapped_air_weight", 1.0)),
        particles=bool(out.get("particles", False)),
        preview=bool(out.get("preview", False)),
        mesh_method=str(out.get("mesh_method", "trilinear")),
        cubic_radius_cells=float(out.get("cubic_radius_cells", 2.0)),
        sdf_particle_radius_cells=float(out.get("sdf_particle_radius_cells", 1.0)),
        sdf_search_radius_cells=float(out.get("sdf_search_radius_cells", 4.0)),
        color_mix_mode=str(out.get("color_mix_mode", "linear")),
        temperature_colormap=(
            str(out["temperature_colormap"])
            if out.get("temperature_colormap") not in (None, "", "off")
            else None
        ),
        temperature_range=(
            _tuple(out["temperature_range"], 2, "output.temperature_range")
            if "temperature_range" in out
            else (20.0, 1500.0)
        ),
    )

    return SceneCfg(
        domain=domain, fluid=fluid, fluids=fluids, obstacle=obstacles,
        obstacle_motion=obstacle_motions,
        inflow=inflows, outflow=outflows,
        simulation=simulation, output=output, config_dir=p.parent.resolve(),
    )
