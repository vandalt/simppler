import numpy as np

import simppler.utils as ut

def test_time_to_phase():
    p = {"per1": 1.0, "tc1": 0.0}
    t = np.linspace(0, 10, num=100)
    phase = ut.time_to_phase(p, t, 1)
    phase_double = ut.time_to_phase(p, t, 1, double=True)

    assert phase.shape == t.shape
    assert phase.min() >= 0.0
    assert phase.max() <= 1.0

    assert len(phase_double) == len(phase) * 2
    np.testing.assert_array_equal(phase_double, np.concatenate([phase, phase+1]))
