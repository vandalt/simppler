from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from simppler.model import RVModel
from simppler.utils import time_to_phase, bin_data


def plot_rv(
    model: RVModel,
    parameters: np.ndarray | dict[str, float | np.ndarray] | None = None,
    n_samples: int = 100,
    residuals: bool = True,
    remove_sys: bool = True,
    inst: list[str] | None = None,
) -> tuple[Figure, Axes]:
    """Plot the RV timeseries and model

    :param model: Model whose data and basis will be used
    :param parameter: Parameter values. If an array, must be compatible with ``RVModel.forward`` or have shape ``(nparam, nsamples_total)``
                      If a dict, will be combined with fixed parameters, but if fixed parameters are specified here they have priority. Can map to parameter values or to arrays of parameter samples.
    :param n_samples: Number of samples to draw if samples are passed as parameters. Ignored for 1D parameters.
    :param residuals: Whether to show a residuals panel.
    """
    show_model = parameters is not None
    if show_model:
        if not isinstance(parameters, dict):
            parameters = dict(zip(model.keys(), parameters))
        ndim = np.array(list(parameters.values())[0]).ndim + 1
        if ndim == 1:
            parameters_single = parameters.copy()
        elif ndim == 2:
            parameters_single = {k: np.median(v) for k, v in parameters.items()}
        else:
            raise ValueError(
                f"Unexpected dimension {ndim} for parameters. Should be 1 or 2."
            )

    residuals = residuals and show_model

    fig, axs = plt.subplots(
        1 + residuals, 1, figsize=(12, 4 + 4 * residuals), sharex=True
    )

    if residuals:
        axd, axr = axs
    else:
        axs = [axs]
        axd = axs[0]
    if model.inst is None:
        ninst = 1
        axd.errorbar(
            model.t, model.rv, yerr=model.erv, fmt="k.", capsize=2, mfc="w", label="Data"
        )
    else:
        plot_inst = model.inst_unique if inst is None else inst
        ninst = len(plot_inst)
        for inst in plot_inst:
            inst_mask = model.inst == inst
            if remove_sys and parameters is not None:
                inst_sys = model.forward(parameters_single, model.t, planets=[], inst=inst)[inst_mask]
            else:
                inst_sys = 0.0
            axd.errorbar(
                model.t[inst_mask],
                model.rv[inst_mask] - inst_sys,
                yerr=model.erv[inst_mask],
                fmt=".",
                capsize=2,
                mfc="w",
                label=f"{inst}",
            )
    axd.set_ylabel("RV [m/s]")

    axs[-1].set_xlabel("Time [d]")
    axd.legend()

    if show_model:

        if residuals:
            axr.axhline(0.0, linestyle="--", color=f"C{ninst}")
            axr.set_ylabel("Residuals [m/s]")

        inst_times = model.t[np.isin(model.inst, inst)]
        mod_times = np.linspace(inst_times.min() - 5, inst_times.max() + 5, num=1000)
        # mod_times = model.tmod[(model.tmod > inst_times.min() - 5) & (model.tmod < inst_times.max() + 5)]
        # TODO: How do we account for mod_sys variability in the samples when ndim == 2?
        if ndim == 1:
            mod_rv = model.forward(parameters, model.t)
            axd.plot(mod_times, model.forward(parameters, mod_times, include_sys=False), label="Model", color=f"C{ninst}")
            if residuals:
                for inst in plot_inst:
                    inst_mask = model.inst == inst
                    axr.errorbar(
                        model.t[inst_mask],
                        model.rv[inst_mask] - mod_rv[inst_mask],
                        yerr=model.erv[inst_mask],
                        fmt=".",
                        capsize=2,
                        mfc="w",
                        label=f"{inst}",
                    )
        elif ndim == 2:
            posterior_preds = model.get_posterior_pred(
                parameters, n_samples, mod_times, include_sys=False
            )
            posterior_preds_data = model.get_posterior_pred(
                parameters, n_samples, model.t
            )
            for i in range(n_samples):
                axd.plot(
                    mod_times,
                    posterior_preds[i],
                    color=f"C{ninst}",
                    alpha=0.1,
                    label="Model samples" if i == 0 else None,
                )
                axr.errorbar(
                    model.t[inst_mask],
                    model.rv[inst_mask] - posterior_preds_data[i][inst_mask],
                    yerr=model.erv[inst_mask],
                    alpha=0.1,
                    fmt="k.",
                    capsize=2,
                    mfc="w",
                    label="Data",
                )
        else:
            raise ValueError(
                f"Unexpected dimension {ndim} for parameters. Should be 1 or 2."
            )

    axd.legend()

    return fig, axs


def plot_phase(
    model: RVModel,
    parameters: np.ndarray | dict[str, float | np.ndarray],
    planets: list[int] | None = None,
    n_samples: int = 100,
    inst: list[str] | None = None,
):
    """Plot the RV timeseries in a phase-folded plot for each planet

    :param model: Model whose data and basis will be used
    :param parameter: Parameter values. If an array, must be compatible with ``RVModel.forward`` or have shape ``(nparam, nsamples_total)``
                      If a dict, will be combined with fixed parameters, but if fixed parameters are specified here they have priority. Can map to parameter values or to arrays of parameter samples.
    :param planets: List of parameter indices (base 1) to show. Show all planets by default.
    :param n_samples: Number of samples to draw if samples are passed as parameters. Ignored for 1D parameters.
    """
    if not isinstance(parameters, dict):
        parameters = dict(zip(model.keys(), parameters))
    ndim = np.array(list(parameters.values())[0]).ndim + 1

    if ndim == 1:
        parameters_single = parameters.copy()
    elif ndim == 2:
        parameters_single = {k: np.median(v) for k, v in parameters.items()}
    else:
        raise ValueError(
            f"Unexpected dimension {ndim} for parameters. Should be 1 or 2."
        )
    parameters_all = model.fixed_p_vals | parameters_single

    all_planets = list(range(1, model.num_planets + 1))
    if planets is None:
        planets = all_planets.copy()
    elif len(planets) == 0:
        raise ValueError(
            "'planets' cannot be an empty list. It must contain at least one valid planet index"
        )
    else:
        for ipl in planets:
            if ipl not in all_planets:
                raise ValueError(
                    f"planet {ipl} is not a valid planet for this model. Available indices are {all_planets}"
                )

    rng = np.random.default_rng()

    if model.inst is not None:
        plot_inst = model.inst_unique if inst is None else inst

    nplanets = len(planets)
    fig, axs = plt.subplots(nplanets, 1, figsize=(12, 4 * nplanets))
    if nplanets == 1:
        axs = [axs]
    for ipl, planet in enumerate(planets):
        other_planets = [i for i in all_planets if i != planet]
        mod_others_data = model.forward(
            parameters_all, model.t, planets=other_planets
        )

        # Double RVs and errors to match phase
        phase_rv = np.concatenate([model.rv, model.rv])
        phase_erv = np.concatenate([model.erv, model.erv])
        mod_others_data = np.concatenate([mod_others_data, mod_others_data])
        if model.inst is not None:
            phase_inst = np.concatenate([model.inst, model.inst])

        # Calculate the phase
        phase = time_to_phase(parameters_all, model.t, planet, double=True) - 1

        # Sort data and phase
        phase_inds = np.argsort(phase)
        phase_rv = phase_rv[phase_inds]
        phase_erv = phase_erv[phase_inds]
        phase_inst = phase_inst[phase_inds]
        phase = phase[phase_inds]
        mod_others_data = mod_others_data[phase_inds]

        if model.inst is None:
            axs[ipl].errorbar(
                phase,
                phase_rv - mod_others_data,
                yerr=phase_erv,
                fmt="k.",
                capsize=2,
                mfc="w",
                label="Data",
            )
        else:
            ninst = len(plot_inst)
            for inst in plot_inst:
                inst_mask = phase_inst == inst
                axs[ipl].errorbar(
                    phase[inst_mask],
                    phase_rv[inst_mask] - mod_others_data[inst_mask],
                    yerr=phase_erv[inst_mask],
                    fmt=".",
                    capsize=2,
                    mfc="w",
                    label=f"{inst}",
                )
        bin_centers, bin_means, bin_stds = bin_data(
            phase + 1, phase_rv - mod_others_data, phase_erv, nbins=25
        )
        bin_centers -= 1
        axs[ipl].errorbar(
            bin_centers,
            bin_means,
            yerr=bin_stds,
            fmt="ro",
            capsize=3,
            label="Binned Data",
            zorder=100,
        )
        if ndim == 1:
            parameters_planet = model.fixed_p_vals | parameters

            mod_planet = model.forward(parameters_planet, model.tmod, planets=[planet], include_sys=False)
            mod_planet = np.concatenate([mod_planet, mod_planet])

            phase_mod = time_to_phase(parameters_all, model.tmod, planet, double=True) - 1
            phase_mod_inds = np.argsort(phase_mod)

            axs[ipl].plot(
                phase_mod[phase_mod_inds], mod_planet[phase_mod_inds], label="Model"
            )
        else:
            ntot = list(parameters.values())[0].size
            inds = rng.choice(ntot, size=n_samples, replace=False)
            for i in inds:
                parameters_planet = model.fixed_p_vals | {k: v[i] for k, v in parameters.items()}
                mod_planet = model.forward(parameters_planet, model.tmod, planets=[planet], include_sys=False)
                mod_planet = np.concatenate([mod_planet, mod_planet])

                phase_mod = time_to_phase(parameters_planet, model.tmod, planet, double=True) - 1
                phase_mod_inds = np.argsort(phase_mod)

                axs[ipl].plot(
                    phase_mod[phase_mod_inds],
                    mod_planet[phase_mod_inds],
                    color="C0",
                    alpha=0.1,
                    label="Model samples" if i == 0 else None,
                )
        axs[ipl].axhline(0.0, linestyle="--", color="k", alpha=0.5)
        axs[ipl].set_ylabel("RV [m/s]")
        axs[ipl].set_xlim(-0.5, 0.5)

        axs[ipl].text(
            0.02,
            0.98,
            f"Planet {chr(ord('a') + ipl + 1)}",
            transform=axs[ipl].transAxes,
            va="top",
            ha="left",
            fontsize=14,
        )

    axs[0].legend()
    axs[-1].set_xlabel("Phase")
    return fig, axs
