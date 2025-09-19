import numpy as np
import pytest

from simppler.model import RVModel
from simppler.plot import plot_phase, plot_rv


def test_plot_rv(sine_model_default: RVModel):
    # TODO: Test handling of fixed parameters for all input types
    model = sine_model_default
    pvals = {"per1": 1.0, "tc1": 0.0, "secosw1": 0.0, "sesinw1": 0.0, "k1": 5.0}

    # Check that output shapes are as expected
    _, axs = plot_rv(model)
    assert len(axs) == 1

    _, axs = plot_rv(model, pvals)
    assert len(axs) == 2

    _, axs = plot_rv(model, pvals, residuals=False)
    assert len(axs) == 1

    # Test that array inputs work
    pvals_arr = np.array(list(pvals.values()))
    _, axs = plot_rv(model, pvals_arr)
    assert len(axs) == 2

    # Test that samples work, as dict or as array
    psamples = {k: np.full(200, v) for k, v in pvals.items()}
    _, axs = plot_rv(model, psamples)
    assert len(axs) == 2

    psamples_arr = np.array(list(psamples.values()))
    assert psamples_arr.shape == (len(pvals), 200)
    _, axs = plot_rv(model, psamples_arr)
    assert len(axs) == 2

    # Test that less desired samples than actual samples causes an error
    with pytest.raises(ValueError, match="Cannot take a larger sample"):
        psamples = {k: np.full(45, v) for k, v in pvals.items()}
        _, axs = plot_rv(model, psamples)

    with pytest.raises(ValueError, match="Cannot take a larger sample"):
        psamples = {k: np.full(45, v) for k, v in pvals.items()}
        psamples_arr = np.array(list(psamples.values()))
        _, axs = plot_rv(model, psamples_arr)


def test_plot_phase(sine_model_default: RVModel):
    model = sine_model_default

    pvals = {"per1": 1.0, "tc1": 0.0, "secosw1": 0.0, "sesinw1": 0.0, "k1": 5.0}
    _, axs = plot_phase(model, pvals)
    assert len(axs) == 1

    _, axs = plot_phase(model, pvals, [1])
    assert len(axs) == 1

    with pytest.raises(ValueError, match="'planets' cannot be an empty list"):
        _, axs = plot_phase(model, pvals, planets=[])

    with pytest.raises(
        ValueError, match="planet 2 is not a valid planet for this model"
    ):
        _, axs = plot_phase(model, pvals, [2])

    parameters_two = model.parameters | {
        k.replace("1", "2"): v for k, v in model.parameters.items()
    }
    pvals_two = pvals | {k.replace("1", "2"): v for k, v in pvals.items()}
    model_two = RVModel(parameters_two, 2, model.t, model.rv, model.erv)
    _, axs = plot_phase(model_two, pvals_two)
    assert len(axs) == 2
    _, axs = plot_phase(model_two, pvals_two, planets=[1])
    assert len(axs) == 1
    _, axs = plot_phase(model_two, pvals_two, planets=[2])
    assert len(axs) == 1
    with pytest.raises(
        ValueError, match="planet 3 is not a valid planet for this model"
    ):
        _, axs = plot_phase(model_two, pvals_two, planets=[3])
