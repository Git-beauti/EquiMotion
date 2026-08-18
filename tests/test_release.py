import numpy as np

from equimotion.release import comfort_rows, safety_guard, safety_rows


def test_safety_guard_preserves_safety_and_jerk():
    lv = np.arange(200, dtype=float)[None, :]
    baseline = lv - 10.0
    unsafe = lv - 2.0
    length = np.asarray([4.5])
    guarded, shifts = safety_guard(unsafe, baseline, lv, length)
    baseline_safety = safety_rows(baseline, lv, length)[0]
    guarded_safety = safety_rows(guarded, lv, length)[0]
    assert shifts[0] > 0.0
    assert guarded_safety[0] >= baseline_safety[0]
    np.testing.assert_allclose(comfort_rows(guarded), comfort_rows(unsafe), atol=1e-12)
