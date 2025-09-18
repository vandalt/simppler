from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def time_to_phase(
    p: dict, t: ArrayLike, planet: int, double: bool = False
) -> np.ndarray:
    """Convert time to phase
    :param p: Dictionary mapping parameter names to values. Must contain a period ``per{planet}`` and time of conjunction ``tc{planet}``.
    :param t: Array of times
    :param planet: Integer index of the planet (base 1)
    :param double: Wehther the phase array should be doubled. Useful for plotting.
    """
    per = p[f"per{planet}"]
    t0 = p[f"tc{planet}"]
    t = np.array(t)
    phase = (t - t0) % per / per
    if double:
        phase = np.concatenate([phase, phase + 1])
    return phase


def bin_data(
    t: np.ndarray, rv: np.ndarray, erv: np.ndarray, nbins: int = 30
) -> tuple[np.ndarray]:
    """Bin RV data

    The data is binned using a weighted average. Bins with less than 3 points are discarded.

    :param t: Times
    :param rv: RVs
    :param erv: RV uncertainties
    :param nbins: Number of bins to use
    """
    # Compute bins
    n, bins = np.histogram(t, bins=nbins)
    bin_inds = np.digitize(t, bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = np.array(
        [
            np.average(rv[bin_inds == i], weights=1 / erv[bin_inds == i] ** 2)
            if np.any(bin_inds == i)
            else np.nan
            for i in range(1, nbins + 1)
        ]
    )
    bin_stds = np.array(
        [
            np.sqrt(1 / np.sum(1 / erv[bin_inds == i] ** 2))
            if np.any(bin_inds == i)
            else np.nan
            for i in range(1, nbins + 1)
        ]
    )

    # Discard bins with less than 3 pts
    nmask = n >= 3
    bin_centers = bin_centers[nmask]
    bin_means = bin_means[nmask]
    bin_stds = bin_stds[nmask]
    return bin_centers, bin_means, bin_stds


def timeperi_to_timetrans(tp, per, ecc, omega, secondary=False):
    """
    Convert Time of Periastron to Time of Transit

    Copied from RadVel (https://github.com/California-Planet-Search/radvel/blob/master/radvel/orbit.py)
    License: https://github.com/California-Planet-Search/radvel/blob/master/LICENSE

    Args:
        tp (float): time of periastron
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): argument of peri (radians)
        secondary (bool): calculate time of secondary eclipse instead

    Returns:
        float: time of inferior conjunction (time of transit if system is transiting)

    """
    try:
        if ecc >= 1:
            return tp
    except ValueError:
        pass

    if secondary:
        f = 3 * np.pi / 2 - omega  # true anomaly during secondary eclipse
        ee = 2 * np.arctan(
            np.tan(f / 2) * np.sqrt((1 - ecc) / (1 + ecc))
        )  # eccentric anomaly

        # ensure that ee is between 0 and 2*pi (always the eclipse AFTER tp)
        if isinstance(ee, np.float64):
            ee = ee + 2 * np.pi
        else:
            ee[ee < 0.0] = ee + 2 * np.pi
    else:
        f = np.pi / 2 - omega  # true anomaly during transit
        ee = 2 * np.arctan(
            np.tan(f / 2) * np.sqrt((1 - ecc) / (1 + ecc))
        )  # eccentric anomaly

    tc = tp + per / (2 * np.pi) * (ee - ecc * np.sin(ee))  # time of conjunction

    return tc


def timetrans_to_timeperi(tc, per, ecc, omega):
    """
    Convert Time of Transit to Time of Periastron Passage

    Copied from RadVel (https://github.com/California-Planet-Search/radvel/blob/master/radvel/orbit.py)
    License: https://github.com/California-Planet-Search/radvel/blob/master/LICENSE

    Args:
        tc (float): time of transit
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)

    Returns:
        float: time of periastron passage

    """
    try:
        if ecc >= 1:
            return tc
    except ValueError:
        pass

    f = np.pi / 2 - omega
    ee = 2 * np.arctan(
        np.tan(f / 2) * np.sqrt((1 - ecc) / (1 + ecc))
    )  # eccentric anomaly
    tp = tc - per / (2 * np.pi) * (ee - ecc * np.sin(ee))  # time of periastron

    return tp
