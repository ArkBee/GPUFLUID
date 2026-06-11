"""[BLK A8] `@block` shim for the addon layer (audit-20260610).

The block registry lives in the `gpufluid` package (`gpufluid.blocks`);
the addon ships WITHOUT that package. This shim lets every addon module
carry real `@block("A8.x", ...)` decorators (so `gpufluid blocks
--check` / tests/test_blocks_registry.py see the A8 layer registered)
while staying a guaranteed no-op inside Blender.

Root cause this closes: docs/BLOCKS.md marked 17 A8.* rows as `impl`
but ZERO decorators existed in addon/ — check 3 reported all of them
as impl-row-without-registry the moment a bpy stub was installed.

Consumers import it defensively::

    try:
        from ._blocks import block
    except ImportError:
        def block(_bid, _desc=""):
            def _w(fn):
                return fn
            return _w

because several tests load addon files STANDALONE via
``spec_from_file_location`` with no parent package (relative imports
unavailable there); registration in those harnesses is irrelevant.
"""
from __future__ import annotations

try:
    from gpufluid.blocks import block  # noqa: F401 — re-exported
except ImportError:
    # dodge: inside Blender the `gpufluid` package is absent BY DESIGN —
    # the addon shells out to a user-configured venv CLI and never imports
    # the library (heavy Warp/CUDA deps). Registration only matters in the
    # test/CLI environment (`gpufluid blocks --check`,
    # tests/test_blocks_registry.py) where gpufluid IS importable. The
    # no-op keeps runtime behaviour byte-identical in Blender.
    def block(block_id: str, description: str = ""):  # noqa: ARG001
        def _wrap(fn):
            return fn
        return _wrap
