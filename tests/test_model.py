import pytest
import numpy as np
import simpple.distributions as sdist
from scipy.stats import norm

from simppler.model import RVModel
from simppler.basis import DefaultBasis, SynthBasis


def test_default_attributes(sine_model_default):
    model = sine_model_default

    # Test that default attributes are what we expect
    assert model.time_base == 0
    assert isinstance(model.basis, DefaultBasis)
    np.testing.assert_array_equal(model.tmod, model.t)

def test_log_likelihood(sine_model_default):
    model = sine_model_default

    pvals = {"per1": 1.0, "tc1": 0.0, "secosw1": 0.0, "sesinw1": 0.0, "k1": 5.0}
    ll = model.log_likelihood(pvals)
    rvmod = model.forward(pvals, model.t)
    assert ll.ndim == 0
    assert np.isclose(ll,  norm(model.rv, model.erv).logpdf(rvmod).sum())

    ll_jit = model.log_likelihood(pvals | {"jit": 10.0})
    assert np.isclose(ll_jit,  norm(model.rv, np.sqrt(model.erv**2 + 10.0**2)).logpdf(rvmod).sum())


def test_basis(sine_data, default_params):
    with pytest.raises(KeyError, match="Required parameter tp1"):
        model = RVModel(default_params, 1, *sine_data, basis="synth")
    with pytest.raises(ValueError, match="Unknown basis"):
        model = RVModel(default_params, 1, *sine_data, basis="scoobydoo")

    synth_params = {
        "per1": sdist.LogUniform(1e-5, 100),
        "tp1": sdist.Uniform(-100, 100),
        "e1": sdist.Uniform(0, 1),
        "w1": sdist.Uniform(0, 2 * np.pi),
        "k1": sdist.LogUniform(1e-5, 100),
    }
    model = RVModel(synth_params, 1, *sine_data, basis="synth")
    assert isinstance(model.basis, SynthBasis)


def test_to_radvel(sine_model_default):
    model = sine_model_default
    pvals = {"per1": 1.0, "tc1": 0.0, "secosw1": 0.0, "sesinw1": 0.0, "k1": 5.0}
    rvmod = model.forward(pvals, model.t)

    radvel_post = model.to_radvel(init_values=pvals)
    rvmod_radvel = radvel_post.model(model.t)

    np.testing.assert_allclose(rvmod, rvmod_radvel)
    np.testing.assert_allclose(model.log_likelihood(pvals), radvel_post.likelihood.logprob())
    np.testing.assert_allclose(model.log_prob(pvals), radvel_post.logprob())

    model.parameters["per1"] = sdist.TruncatedNormal(0.0, 1.0, -5, 5)
    with pytest.raises(TypeError, match="Distribution of type"):
        model.to_radvel()

def test_num_planets(sine_data, default_params):
    with pytest.raises(ValueError, match="Unexpected parameter per1"):
        RVModel(default_params, 0, *sine_data)
    RVModel({}, 0, *sine_data)

    with pytest.raises(KeyError, match="Required parameter per2"):
        RVModel(default_params, 2, *sine_data)
    p2 = default_params | {k.replace("1", "2"): v for k, v in default_params.items()}
    RVModel(p2, 2, *sine_data)


def test_model_bad_params(sine_data, default_params):
    parameters = default_params | {
        "allo": sdist.LogUniform(1e-5, 100),
    }

    # Test that unexpected parameters get caught
    with pytest.raises(ValueError, match="Unexpected parameter"):
        RVModel(parameters, 1, *sine_data)

    # Test that missing parameters get caugth
    per1 = parameters.pop("per1")
    del parameters["allo"]
    with pytest.raises(KeyError, match="Required parameter per1"):
        RVModel(parameters, 1, *sine_data)

    # Test that optional parameters are allowed
    parameters["per1"] = per1
    optional_parameters = {
        "jit": sdist.Uniform(-100, 100),
        "gamma": sdist.Uniform(-100, 100),
        "dvdt": sdist.Uniform(-100, 100),
        "curv": sdist.Uniform(-100, 100),
    }
    parameters = parameters | optional_parameters
    model = RVModel(parameters, 1, *sine_data)
    assert list(model.keys()) == list(parameters.keys())
