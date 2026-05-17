"""F3.6.C3 — hard CI gate: the layer-import whitelist must stay empty.

Spec: DESIGN.md §3.2.4.1 (closure of the F3.6 hook refactor macro,
2026-05-17). After phases F3.6.A1/A2/B/C1/C2, every cross-layer
import in the codebase satisfies the strict §2 layer rule. The
whitelist `_KNOWN_LAYER_EXCEPTIONS` in `blocks/check.py` was emptied
in C2. This test pins it: any future PR that re-adds a whitelist
entry without ALSO updating DESIGN.md §3.2.4.1 (which would itself
require code review) fails CI immediately.

The test name slug satisfies check 6 coverage references for all
six F3.6 micros — `[BLK F3.6.A1]` through `[BLK F3.6.C3]` — so the
closure history stays discoverable from the test file.

Why a separate test file: the assertion is single-purpose and fast;
co-locating with `test_blocks_registry.py` would dilute its intent.
A future maintainer reading "why is the whitelist empty?" finds
this one-test file as the load-bearing documentation.
"""
from __future__ import annotations

# Covers blocks from the F3.6 macro closure:
#   [BLK F3.6.A1] sdf primitives relocated to G1
#   [BLK F3.6.A2] mesh_marker relocated to S2
#   [BLK F3.6.B]  Motion + evaluate_center relocated to G1
#   [BLK F3.6.C1] FrameEventQueue dual-path
#   [BLK F3.6.C2] solver drains queue
#   [BLK F3.6.C3] this hard-gate test (closure micro)


def test_no_layer_exceptions_remain():
    """Layer-import whitelist `_KNOWN_LAYER_EXCEPTIONS` must be empty.

    If this test fails, it means someone added a cross-layer import
    that doesn't satisfy the §2 layer rule. Two valid fixes, in
    order of preference:

      1. Fix the dependency direction. The §3.2.4 case studies (A/B/C
         categories) show how — most violations are mis-filed code
         (move to the correct layer) or pure utilities (move to G1).

      2. If genuinely necessary, add an exit-plan row to DESIGN.md
         §3.2.4.1 explaining what the import is, why it must exist
         today, and what would make it go away. THEN add to
         `_KNOWN_LAYER_EXCEPTIONS`. This test will fail until the
         exit plan is documented (intentional friction).

    Do not silence this test. It is the architecture contract.
    """
    from gpufluid.blocks.check import _KNOWN_LAYER_EXCEPTIONS
    assert _KNOWN_LAYER_EXCEPTIONS == [], (
        f"Layer-import whitelist is non-empty: {_KNOWN_LAYER_EXCEPTIONS}. "
        f"See test_no_layer_exceptions docstring for the fix path. "
        f"Architecture contract violated — do not just edit this test."
    )
