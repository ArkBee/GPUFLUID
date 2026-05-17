"""Block registry and error-tagging.

Every public callable in `gpufluid` is registered with a stable Block ID
declared in `docs/DESIGN.md` §§4-11. The decorator is a no-op at runtime
apart from recording (id, description, callable, source location) into a
global table — used for documentation generation and the `--check`
command (see `gpufluid.blocks.check`).

Errors raised through `BlockError` carry their block ID so any failure
message points at an exact line in DESIGN/BLOCKS.

Block ID format: `<LAYER><MAJOR>.<MINOR>[.<SUB>]` e.g. `S2.6.1`.

See `docs/DESIGN.md §3` for the full contract and `§3.2` for the
`gpufluid blocks --check` enforcement spec.
"""
from __future__ import annotations
import inspect
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List

_BLOCK_ID_RE = re.compile(r"^[A-Z][0-9]+(\.[0-9A-Z]+){1,3}$")


@dataclass
class BlockInfo:
    block_id: str
    description: str
    qualname: str
    module: str
    source_file: str
    source_line: int
    layer: str = field(init=False)

    def __post_init__(self):
        # Layer prefix is the leading letter cluster (e.g. "S2" → layer S2).
        m = re.match(r"^([A-Z][0-9]+)", self.block_id)
        self.layer = m.group(1) if m else "?"


_REGISTRY: Dict[str, List[BlockInfo]] = {}


def block(block_id: str, description: str = "") -> Callable:
    """Decorator: register a callable under the given Block ID.

    Usage::

        @block("S2.3", "Apply gravity to v faces")
        def add_gravity(...):
            ...

    Warp kernels: apply the decorator BEFORE `@wp.kernel` so registration
    sees the Python function; runtime behaviour is unchanged.
    """
    if not _BLOCK_ID_RE.match(block_id):
        raise ValueError(f"bad block_id {block_id!r}; expected like 'S2.6.1'")

    def wrap(fn):
        try:
            src_file = inspect.getsourcefile(fn) or "?"
            _, src_line = inspect.getsourcelines(fn)
        except (TypeError, OSError):
            src_file, src_line = "?", 0
        info = BlockInfo(
            block_id=block_id,
            description=description or fn.__doc__ or "",
            qualname=getattr(fn, "__qualname__", fn.__name__),
            module=getattr(fn, "__module__", "?"),
            source_file=src_file,
            source_line=src_line,
        )
        bucket = _REGISTRY.setdefault(block_id, [])
        if not any(b.qualname == info.qualname for b in bucket):
            bucket.append(info)
        fn.__block_id__ = block_id
        return fn

    return wrap


def get_registry() -> Dict[str, List[BlockInfo]]:
    """Return a copy of the registry mapping block_id → list of impls."""
    return {k: list(v) for k, v in _REGISTRY.items()}


def find(block_id: str) -> List[BlockInfo]:
    return list(_REGISTRY.get(block_id, []))


def by_layer(layer: str) -> List[BlockInfo]:
    items: list[BlockInfo] = []
    for bucket in _REGISTRY.values():
        items.extend(b for b in bucket if b.layer == layer)

    def _key(b):
        out = []
        for p in b.block_id[len(layer):].lstrip(".").split("."):
            try:
                out.append((0, int(p)))
            except ValueError:
                out.append((1, p))
        return tuple(out)
    return sorted(items, key=_key)


class BlockError(RuntimeError):
    """Domain error annotated with its block ID.

    Use::

        raise BlockError("S2.6.1", "residual increased after iter")

    Renders as::

        BlockError [S2.6.1]: residual increased after iter
    """

    def __init__(self, block_id: str, message: str):
        self.block_id = block_id
        self.message = message
        super().__init__(f"[{block_id}] {message}")


def assert_block(condition: bool, block_id: str, message: str) -> None:
    """Raise BlockError with the given tag if condition is false."""
    if not condition:
        raise BlockError(block_id, message)


__all__ = [
    "BlockInfo", "BlockError", "block", "get_registry", "find",
    "by_layer", "assert_block",
]
