"""Audit 2026-06-14 round 5 — block-coverage substring collision.

Promoted from the round-4 *disputed* finding after confirming it live: the
block-coverage matcher used a plain ``slug in haystack`` substring test, so a
shorter-numbered block (``S2.1`` → slug ``s2_1``) counted as 'tested' on the
strength of any longer sibling's test (``test_s2_11_*``). 80 such collisions
exist in the registry. `_slug_in_test_haystack` now closes the trailing
numeric segment with a lookahead. These are direct unit tests of that matcher.
"""
from __future__ import annotations

from gpufluid.blocks.check import _slug_in_test_haystack, _normalise_id


def _hay(*test_names: str) -> str:
    # Mirror check.py: "|"-joined, lowercased.
    return "|".join(test_names).lower()


def test_r5_exact_slug_matches():
    hay = _hay("test_s2_1_p2g", "test_misc")
    assert _slug_in_test_haystack("s2_1", hay)


def test_r5_longer_numbered_sibling_does_not_cover_shorter():
    # The whole bug: a test for S2.11 / S2.14 must NOT cover S2.1.
    hay = _hay("test_s2_11_sort", "test_s2_14_csf", "test_s2_15_visc")
    assert not _slug_in_test_haystack("s2_1", hay), \
        "r5: s2_1 must not be 'covered' by s2_11/s2_14/s2_15 tests"
    # …but S2.11 itself is still covered by its own test.
    assert _slug_in_test_haystack("s2_11", hay)


def test_r5_child_block_test_still_covers_parent():
    # A refinement (GPU variant) test counts for the parent block — same
    # block, finer grain. Only the numeric-sibling case is a false positive.
    hay = _hay("test_s2_11_gpu_rank")
    assert _slug_in_test_haystack("s2_11", hay)
    assert _slug_in_test_haystack("s2_11_gpu", hay)


def test_r5_multi_segment_numeric_boundary():
    # m5_1 must not be covered by m5_11_4; m5_11_4 not by m5_11_4_h's digit.
    hay = _hay("test_m5_11_4_h_host")
    assert not _slug_in_test_haystack("m5_1", hay), \
        "r5: m5_1 must not match m5_11_4_h"
    assert _slug_in_test_haystack("m5_11_4", hay), \
        "r5: m5_11_4 IS covered by m5_11_4_h (trailing _ is a boundary)"
    assert _slug_in_test_haystack("m5_11_4_h", hay)


def test_r5_head_boundary_no_midtoken_match():
    # A slug must not match inside a longer leading token.
    hay = _hay("test_xs2_1_bogus")
    assert not _slug_in_test_haystack("s2_1", hay), \
        "r5: s2_1 must not match mid-token inside xs2_1"


def test_r5_normalise_id_feeds_the_matcher():
    # End-to-end: a dotted block id normalises to the slug the matcher sees.
    slug = _normalise_id("S2.14.6")
    assert slug == "s2_14_6"
    assert _slug_in_test_haystack(slug, _hay("test_s2_14_6_clamp"))
    assert not _slug_in_test_haystack(_normalise_id("S2.14"),
                                      _hay("test_s2_146_other"))
