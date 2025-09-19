import numpy as np
import pytest
import simpple.distributions as sdist

from simppler.model import RVModel


@pytest.fixture
def sine_data():
    t = np.linspace(0, 100, num=100)
    rv = 8 * np.sin(2 * np.pi * t / 4)
    erv = 0.8 * np.ones_like(rv)
    return t, rv, erv


@pytest.fixture
def default_params():
    return {
        "per1": sdist.LogUniform(1e-5, 100),
        "tc1": sdist.Uniform(-100, 100),
        "secosw1": sdist.Uniform(-1, 1),
        "sesinw1": sdist.Uniform(-1, 1),
        "k1": sdist.LogUniform(1e-5, 100),
    }


@pytest.fixture
def sine_model_default(sine_data, default_params):
    t, rv, erv = sine_data
    return RVModel(default_params, 1, t, rv, erv)
