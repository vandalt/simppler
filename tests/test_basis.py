import inspect

import numpy as np
import pytest

import simppler.basis as sb

BASES = list(sb.BASIS_DICT.values())

def test_basis_dicts():
    all_basis_classes = [
        cls for name, cls in inspect.getmembers(sb, inspect.isclass)
        if cls.__module__ == sb.__name__ and  name != "Basis"
    ]
    assert list(sb.BASIS_DICT.values()) == list(sb.BASIS_PARAM_DICT.values())
    for cls in all_basis_classes:
        assert cls in sb.BASIS_DICT.values()

@pytest.mark.parametrize("basis_cls,basis_name", zip(BASES, list(sb.BASIS_DICT)))
def test_conversion(basis_cls, basis_name):
    basis = basis_cls()

    # Quick check that names from classes are consistent with names from the dict
    assert basis.name == basis_name

    p_synth = {"per1": 10.0, "tp1": 1000.0, "e1": 0.1, "w1": 1.0, "k1": 5.0}

    p_basis = basis.from_synth(p_synth, 1)

    # Check that the names are consistent with basis.pstr
    p_expect = [pname+"1" for pname in basis.pstr.split()]
    assert list(p_basis) == p_expect

    # Check that converting back to synth gives the same dict (order matters)
    p_synth_back = basis.to_synth(p_basis, 1)
    assert list(p_synth) == list(p_synth_back)
    np.testing.assert_allclose(list(p_synth.values()), list(p_synth_back.values()))
