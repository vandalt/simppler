from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from simppler.model import RVModel


def plot_orbit(
    model: RVModel,
    samples: np.ndarray,
    model_type: str = "samples",
    n_samples: int = 100,
) -> tuple[Figure, Axes]:
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axd, axr = axs
    axd.errorbar(
        model.t, model.rv, yerr=model.erv, fmt="k.", capsize=2, mfc="w", label="Data"
    )
    axd.set_ylabel("RV [m/s]")

    axr.axhline(0.0, linestyle="--")
    axr.set_ylabel("Residuals [m/s]")
    axr.set_xlabel("Time [d]")

    if model_type == "med":
        med_p = np.median(samples, axis=0)
        med_rv = model.forward(med_p, model.t)
        axd.plot(model.tmod, model.forward(med_p, model.tmod), label="Median model")
        axr.errorbar(
            model.t,
            model.rv - med_rv,
            yerr=model.erv,
            fmt="k.",
            capsize=2,
            mfc="w",
            label="Data",
        )
    elif model_type == "samples":
        posterior_preds = model.get_posterior_pred(samples.T, n_samples, model.tmod)
        posterior_preds_data = model.get_posterior_pred(samples.T, n_samples, model.t)
        for i in range(n_samples):
            axd.plot(
                model.tmod,
                posterior_preds[i],
                color="C0",
                alpha=0.1,
                label="Posterior samples" if i == 0 else None,
            )
            axr.errorbar(
                model.t,
                model.rv - posterior_preds_data[i],
                yerr=model.erv,
                alpha=0.1,
                fmt="k.",
                capsize=2,
                mfc="w",
                label="Data",
            )
    else:
        raise ValueError(f"Unknown model type {model_type}. Expected 'med' or 'samples'")

    axd.legend()

    return fig, axs
