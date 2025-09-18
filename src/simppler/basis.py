from abc import ABC, abstractmethod

import numpy as np

from simppler.utils import timeperi_to_timetrans, timetrans_to_timeperi


class Basis(ABC):
    """Abstract base class for an orbital basis"""

    @abstractmethod
    def from_synth(p: dict, num_planets: int) -> dict:
        """
        Function converting parameters from the :class:`SynthBasis` basis to the current basis.
        """
        pass

    @abstractmethod
    def to_synth():
        """
        Function converting parameters from the current basis to the :class:`SynthBasis` basis.
        """
        pass


class SynthBasis(Basis):
    """Synth basis

    This is the 'synth' basis from radvel, which is passed to the Keplerian solver.
    All Basis classes must be able to convert to and from this basis with
    `to_synth()` and `from_synth()` methods.

    The parameters are:
    
    - ``per``: Period
    - ``tp``: Time of periastron
    - ``e``: Eccentricity
    - ``w``: Argument of periastron
    - ``k``: Semi-amplitude
    """
    name = "synth"
    pstr = "per tp e w k"

    def to_synth(p: dict, num_planets: int):
        return p

    def from_synth(p: dict, num_planets: int):
        return p


class DefaultBasis(Basis):
    r"""Default basis

    Fitting basis used by default in RV models.
    The differences with the :class:`SynthBasis` are:

    - The time of conjunction (or transit for transiting planets) ``tc`` is used instead of the time of periastron ``tp``. This parameter is better defined for circular orbit.
    - ``e`` and ``w`` are replaced with ``secosw`` and ``sesinw`` (:math:`\sqrt{e}\cos{\omega}`, :math:`\sqrt{e}\sin{\omega}`)
    """
    name = "default"
    pstr = "per tc secosw sesinw k"

    def to_synth(p: dict, num_planets: int):
        p_synth = {}
        for i in range(1, num_planets + 1):
            p_synth[f"per{i}"] = p[f"per{i}"]
            p_synth[f"e{i}"] = p[f"secosw{i}"] ** 2 + p[f"sesinw{i}"] ** 2
            p_synth[f"w{i}"] = np.arctan2(p[f"sesinw{i}"], p[f"secosw{i}"])
            p_synth[f"tp{i}"] = timetrans_to_timeperi(
                p[f"tc{i}"],
                p_synth[f"per{i}"],
                p_synth[f"e{i}"],
                p_synth[f"w{i}"],
            )
            p_synth[f"k{i}"] = p[f"k{i}"]
        return p_synth

    def from_synth(p_synth: dict, num_planets: int):
        p = {}
        for i in range(1, num_planets + 1):
            p[f"per{i}"] = p_synth[f"per{i}"]
            p[f"secosw{i}"] = np.sqrt(p_synth[f"e{i}"]) * np.cos(p_synth[f"w{i}"])
            p[f"sesinw{i}"] = np.sqrt(p_synth[f"e{i}"]) * np.sin(p_synth[f"w{i}"])
            p[f"tc{i}"] = timeperi_to_timetrans(
                p_synth[f"tp{i}"],
                p_synth[f"per{i}"],
                p_synth[f"e{i}"],
                p_synth[f"w{i}"],
            )
            p[f"k{i}"] = p_synth[f"k{i}"]
        return p


class LogKBasis(Basis):
    """LogK Basis

    Same as the :class:`DefaultBasis`, but the logarithm of the semi-amplitude is fitted instead of the semi-amplitude.
    """
    name = "logk"
    pstr = "per tc secosw sesinw logk"

    def to_synth(p: dict, num_planets: int):
        p_synth = {}
        for i in range(1, num_planets + 1):
            p_synth[f"per{i}"] = p[f"per{i}"]
            p_synth[f"e{i}"] = p[f"secosw{i}"] ** 2 + p[f"sesinw{i}"] ** 2
            p_synth[f"w{i}"] = np.arctan2(p[f"sesinw{i}"], p[f"secosw{i}"])
            p_synth[f"tp{i}"] = timetrans_to_timeperi(
                p[f"tc{i}"],
                p_synth[f"per{i}"],
                p_synth[f"e{i}"],
                p_synth[f"w{i}"],
            )
            p_synth[f"k{i}"] = np.exp(p[f"logk{i}"])
        return p_synth

    def from_synth(p_synth: dict, num_planets: int):
        p = {}
        for i in range(1, num_planets + 1):
            p[f"per{i}"] = p_synth[f"per{i}"]
            p[f"secosw{i}"] = np.sqrt(p_synth[f"e{i}"]) * np.cos(p_synth[f"w{i}"])
            p[f"sesinw{i}"] = np.sqrt(p_synth[f"e{i}"]) * np.sin(p_synth[f"w{i}"])
            p[f"tc{i}"] = timeperi_to_timetrans(
                p_synth[f"tp{i}"],
                p_synth[f"per{i}"],
                p_synth[f"e{i}"],
                p_synth[f"w{i}"],
            )
            p[f"logk{i}"] = np.log(p_synth[f"k{i}"])
        return p


class LogPerBasis(Basis):
    """LogPer Basis

    Same as the :class:`DefaultBasis`, but the logarithm of the period is fitted instead of the semi-amplitude.
    """
    name = "logper"
    pstr = "logper tc secosw sesinw k"

    def to_synth(p: dict, num_planets: int):
        p_synth = {}
        for i in range(1, num_planets + 1):
            p_synth[f"per{i}"] = np.exp(p[f"logper{i}"])
            p_synth[f"e{i}"] = p[f"secosw{i}"] ** 2 + p[f"sesinw{i}"] ** 2
            p_synth[f"w{i}"] = np.arctan2(p[f"sesinw{i}"], p[f"secosw{i}"])
            p_synth[f"tp{i}"] = timetrans_to_timeperi(
                p[f"tc{i}"],
                p_synth[f"per{i}"],
                p_synth[f"e{i}"],
                p_synth[f"w{i}"],
            )
            p_synth[f"k{i}"] = p[f"k{i}"]
        return p_synth

    def from_synth(p_synth: dict, num_planets: int):
        p = {}
        for i in range(1, num_planets + 1):
            p[f"logper{i}"] = np.log(p_synth[f"per{i}"])
            p[f"secosw{i}"] = np.sqrt(p_synth[f"e{i}"]) * np.cos(p_synth[f"w{i}"])
            p[f"sesinw{i}"] = np.sqrt(p_synth[f"e{i}"]) * np.sin(p_synth[f"w{i}"])
            p[f"tc{i}"] = timeperi_to_timetrans(
                p_synth[f"tp{i}"],
                p_synth[f"per{i}"],
                p_synth[f"e{i}"],
                p_synth[f"w{i}"],
            )
            p[f"k{i}"] = p_synth[f"k{i}"]
        return p


class EccBasis(Basis):
    """Ecc Basis 

    Same as the :class:`SynthBasis`, but ``e`` and ``w`` are fitted directly instead of ``secosw`` and ``sesinw``.
    """
    name = "ecc"
    pstr = "per tc e w k"

    def to_synth(p: dict, num_planets: int) -> dict:
        p_synth = p.copy()
        for i in range(1, num_planets + 1):
            p_synth[f"tp{i}"] = timetrans_to_timeperi(
                p[f"tc{i}"],
                p_synth[f"per{i}"],
                p_synth[f"e{i}"],
                p_synth[f"w{i}"],
            )
        return p_synth

    def from_synth(p_synth: dict, num_planets: int) -> dict:
        p = p_synth.copy()
        for i in range(1, num_planets + 1):
            p[f"tc{i}"] = timeperi_to_timetrans(
                p_synth[f"tp{i}"],
                p_synth[f"per{i}"],
                p_synth[f"e{i}"],
                p_synth[f"w{i}"],
            )
        return p


BASIS_DICT = {
    "synth": SynthBasis,
    "default": DefaultBasis,
    "logk": LogKBasis,
    "logper": LogPerBasis,
    "ecc": EccBasis,
}
BASIS_PARAM_DICT = {b.pstr: b for b in BASIS_DICT.values()}
