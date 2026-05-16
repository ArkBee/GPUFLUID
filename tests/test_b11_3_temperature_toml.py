"""B11.3 — `[[fluids]] temperature` TOML wiring + CLI dump.

Closes the last v0.9 micro: per-source temperature scalar reachable from
TOML, threaded through `cli.commands:cmd_simulate`, dumped per-frame to
`<cache>/particles/../temperatures/frame_NNNN.npy`.

Tests follow the HANDOFF §2.3 rule: at least one asserts the feature
*changes the answer* (cold + hot sources land in disjoint value sets),
not just "code runs". A pure-config sibling test runs CPU-only so the
parser regression is caught even on the CI box without CUDA.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from gpufluid.cli.config import FluidBoxCfg, load_scene
from gpufluid.cli.commands import main


# ---------------------------------------------------------------------------
# Pure-config: no GPU needed — runs even when CUDA is absent.
# ---------------------------------------------------------------------------

def test_b11_3_toml_temperature_parses(tmp_path: Path):
    """`[[fluids]] temperature = X` must surface as `FluidBoxCfg.temperature`
    on the parsed scene config. Single-`[fluid]` block must too.

    Pre-fix the field didn't exist on FluidBoxCfg, so this also pins
    that nothing reverts the dataclass."""
    cfg = tmp_path / "scene.toml"
    cfg.write_text("""
[domain]
resolution = [8, 8, 8]

[fluid]
type = "box"
lo = [0.10, 0.10, 0.10]
hi = [0.50, 0.50, 0.50]
ppc = 4
temperature = 25.0

[[fluids]]
type = "box"
lo = [0.10, 0.10, 0.10]
hi = [0.30, 0.40, 0.30]
ppc = 4
temperature = 80.0

[[fluids]]
type = "box"
lo = [0.55, 0.10, 0.55]
hi = [0.80, 0.40, 0.80]
ppc = 4
temperature = 20.0
""")
    scene = load_scene(cfg)
    # Single [fluid] block still parses with a temperature scalar attached.
    assert isinstance(scene.fluid, FluidBoxCfg)
    assert scene.fluid.temperature == pytest.approx(25.0)
    # Multi-source [[fluids]] surface independent temperature values.
    assert len(scene.fluids) == 2
    temps = [getattr(f, "temperature", None) for f in scene.fluids]
    assert temps == [pytest.approx(80.0), pytest.approx(20.0)]


def test_b11_3_toml_temperature_omitted_is_none(tmp_path: Path):
    """Absent `temperature` key ⇒ None (no implicit zero). Necessary so
    `cli.commands` can decide whether to allocate the scalar attribute at
    all — a scene with no temperature stays cheap."""
    cfg = tmp_path / "scene.toml"
    cfg.write_text("""
[domain]
resolution = [8, 8, 8]

[fluid]
type = "box"
lo = [0.10, 0.10, 0.10]
hi = [0.50, 0.50, 0.50]
ppc = 4
""")
    scene = load_scene(cfg)
    assert isinstance(scene.fluid, FluidBoxCfg)
    assert scene.fluid.temperature is None


# ---------------------------------------------------------------------------
# End-to-end bake: requires CUDA + Warp.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_b11_3_lava_bake_dumps_temperature_sidecar(tmp_path: Path):
    """Bake a 2-frame scene with HOT and COLD sources via TOML.

    Asserts:
      1. `<cache>/temperatures/frame_NNNN.npy` is written each frame.
      2. The dumped array carries values from BOTH sources — not all-zeros
         (would mean attribute was allocated but never seeded), not a
         single value (would mean the per-source temperature didn't reach
         the solver).

    This is the regression: pre-wire the array existed only on the
    Python side; nothing reached `solver.attr_temperature` from TOML.
    """
    cfg = tmp_path / "scene.toml"
    # Two well-separated boxes so the cells don't share a P2G stencil at
    # frame 0 — each source's particles keep their seed temperature until
    # advection mixes them.
    cfg.write_text("""
[domain]
resolution = [16, 16, 16]

[[fluids]]
type = "box"
lo = [0.10, 0.40, 0.10]
hi = [0.30, 0.60, 0.30]
ppc = 4
temperature = 90.0

[[fluids]]
type = "box"
lo = [0.65, 0.40, 0.65]
hi = [0.85, 0.60, 0.85]
ppc = 4
temperature = 10.0

[simulation]
dt = 0.005
pressure_iters = 10
frames = 2
fps = 24

[output]
cache_dir = "cache"
mesh = false
particles = true
""")
    rc = main(["simulate", str(cfg)])
    assert rc == 0, "cmd_simulate must complete cleanly"

    cache_dir = tmp_path / "cache"
    temps_dir = cache_dir / "temperatures"
    f0 = temps_dir / "frame_0000.npy"
    f1 = temps_dir / "frame_0001.npy"
    assert f0.exists() and f1.exists(), \
        "per-frame temperature sidecar must land alongside particles/"

    t0 = np.load(f0)
    # Bimodal: each source contributes its own seed value. After 0 sim
    # frames the per-particle scalar is *exactly* the seed (no P2G yet).
    assert t0.size > 0, "no particles written — seed paths broken"
    hot = t0[t0 > 50.0]
    cold = t0[t0 < 50.0]
    assert hot.size > 0 and cold.size > 0, (
        f"expected BOTH hot and cold particles; "
        f"got hot={hot.size}, cold={cold.size}, range={t0.min()}..{t0.max()}"
    )
    # Per-particle seed values land exactly on the source temperature in
    # frame 0 (G2P only runs at step time; this dump is post-seed,
    # pre-step). Allow a tiny tolerance for any internal copies.
    assert hot.mean() == pytest.approx(90.0, abs=1e-4)
    assert cold.mean() == pytest.approx(10.0, abs=1e-4)

    # Frame 1: after one full step, the interface region may mix slightly,
    # but the per-source bulk values must still be present (we don't
    # rebalance, we just transport).
    t1 = np.load(f1)
    assert t1.max() > 50.0, "hot bulk vanished after one step"
    assert t1.min() < 50.0, "cold bulk vanished after one step"
    # Sanity: the sidecar count matches the particle count.
    parts = np.load(cache_dir / "particles" / "frame_0000.npy")
    assert parts.shape[0] == t0.shape[0]
