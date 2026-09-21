"""P1 proportional module-offset decision: `p1_module_steps` acceptance table.

Normative cases from the P1 plan (§6.7). Pure function — no fakes needed.
"""

from calib_auto.calibrate import p1_module_steps


def test_reference_case_22_positive_gaps():
    # 20×+1, 2×+2 -> n=1, x=22/48=45.8% -> band 2 -> +26 steps.
    assert p1_module_steps([1] * 20 + [2] * 2 + [0] * 26) == (26, 2)


def test_band1_full_drum():
    assert p1_module_steps([1] * 36 + [0] * 12) == (42, 1)


def test_band1_multi_char():
    assert p1_module_steps([2] * 40 + [0] * 8) == (84, 1)


def test_band2_lower_boundary():
    assert p1_module_steps([1] * 6 + [0] * 42) == (7, 2)


def test_band3_below_minimum():
    assert p1_module_steps([1] * 5 + [0] * 43) == (0, 3)


def test_mixed_magnitudes_use_minimum():
    # 10×+2, 10×+3 -> n=2, x=41.7% -> band 2 -> +47 steps.
    assert p1_module_steps([2] * 10 + [3] * 10 + [0] * 28) == (47, 2)


def test_conflict_goes_to_p2():
    # 19×+1 vs 14×-1: minority 29% >= 12.5% -> band 3, no move.
    assert p1_module_steps([1] * 19 + [-1] * 14 + [0] * 15) == (0, 3)


def test_small_minority_is_noise():
    # 20×+1 vs 3×-1: minority 6.25% < 12.5% -> dominant wins -> +23.
    assert p1_module_steps([1] * 20 + [-1] * 3 + [0] * 25) == (23, 2)


def test_tie_goes_to_p2():
    assert p1_module_steps([1] * 10 + [-1] * 10 + [0] * 28) == (0, 3)


def test_all_zero_is_band3():
    assert p1_module_steps([0] * 48) == (0, 3)


def test_all_none_is_band3():
    assert p1_module_steps([None] * 48) == (0, 3)


def test_none_counts_toward_denominator():
    # 22 gaps + 26 excluded: x = 22/48 still, band 2 -> +26.
    assert p1_module_steps([1] * 20 + [2] * 2 + [None] * 26) == (26, 2)


def test_negative_direction():
    assert p1_module_steps([-1] * 36 + [0] * 12) == (-42, 1)
    assert p1_module_steps([-1] * 22 + [0] * 26) == (-26, 2)
