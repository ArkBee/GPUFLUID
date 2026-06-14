"""[Layer I6 / BLK I6.2] Mesh-cache directory manifest.

A simulation produces a directory like::

    out/sim01/
        cache.json
        mesh/frame_0000.ply
        mesh/frame_0001.ply
        ...
        particles/frame_0000.npy   (optional)

`cache.json` is the source of truth a downstream consumer (Blender
addon, render farm) reads to know what's there.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional

from ..blocks import block, BlockError

CACHE_SCHEMA_VERSION = 1


@dataclass
class CacheManifest:
    schema_version: int = CACHE_SCHEMA_VERSION
    gpufluid_version: str = ""
    created_unix: float = 0.0
    fps: int = 24
    frame_count: int = 0
    mesh_pattern: str = "mesh/frame_{:04d}.ply"
    particles_pattern: Optional[str] = None
    domain_size: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    resolution: List[int] = field(default_factory=lambda: [64, 64, 64])
    dx: float = 1.0 / 64
    notes: str = ""
    # Round-32: round-31 reviewer caught schema drift between MPM and
    # FLIP cache.json writers. MPM writes raw dict with `solver`,
    # `truncated_at_frame`, `truncation_reason`; FLIP wrote via this
    # dataclass without those keys; round-32 added them to FLIP via
    # post-hoc raw-JSON injection but `read_cache_manifest` filtered
    # them out, dropping the info on the typed-API path. Now: the
    # dataclass carries the optional fields, both writers + the typed
    # reader see them consistently. `solver` defaults to "flip" for
    # backwards-compat with pre-round-32 caches.
    solver: str = "flip"
    truncated_at_frame: Optional[int] = None
    truncation_reason: Optional[str] = None

    def mesh_path(self, root: Path, frame: int) -> Optional[Path]:
        # audit-2026-06-14r2 #11: guard None symmetrically with particles_path
        # — an MPM particles-only bake writes mesh_pattern=null, and the old
        # `None.format(...)` raised AttributeError instead of returning None.
        if not self.mesh_pattern:
            return None
        return root / self.mesh_pattern.format(frame)

    def particles_path(self, root: Path, frame: int) -> Optional[Path]:
        if not self.particles_pattern:
            return None
        return root / self.particles_pattern.format(frame)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# [BLK I6.2]
@block("I6.2", "Write cache manifest (cache.json)")
def write_cache_manifest(cache_dir: Path, manifest: CacheManifest) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if manifest.created_unix == 0.0:
        manifest.created_unix = time.time()
    if not manifest.gpufluid_version:
        from .. import __version__
        manifest.gpufluid_version = __version__
    path = cache_dir / "cache.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


# [BLK I6.2]
@block("I6.2", "Read cache manifest (cache.json)")
def read_cache_manifest(cache_dir: Path) -> CacheManifest:
    path = Path(cache_dir) / "cache.json"
    if not path.exists():
        raise BlockError("I6.2", f"no cache.json in {cache_dir}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version", 0) != CACHE_SCHEMA_VERSION:
        raise BlockError("I6.2",
                         f"unsupported cache schema {raw.get('schema_version')}; "
                         f"expected {CACHE_SCHEMA_VERSION}")
    # Drop unknown keys defensively (forward-compat).
    fields = {f for f in CacheManifest.__dataclass_fields__}
    return CacheManifest(**{k: v for k, v in raw.items() if k in fields})
