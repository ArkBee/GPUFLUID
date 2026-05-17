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
        nx, ny, nz = self.domain.resolution
        return (nx * self.dx, ny * self.dx, nz * self.dx)


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
        return ObstacleBoxCfg(
            type="box",
            center=_tuple(d["center"], 3, "obstacle.center"),
            half_size=_tuple(d["half_size"], 3, "obstacle.half_size"),
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
    simulation = SimulationCfg(
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
    )

    return SceneCfg(
        domain=domain, fluid=fluid, fluids=fluids, obstacle=obstacles,
        obstacle_motion=obstacle_motions,
        inflow=inflows, outflow=outflows,
        simulation=simulation, output=output, config_dir=p.parent.resolve(),
    )
