from simpple.model import ForwardModel
from simpple.distributions import Distribution
import simpple.distributions as sdist
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from radvel.posterior import Posterior

import numpy as np

from simppler import kepler
from simppler.basis import BASIS_DICT, BASIS_PARAM_DICT, Basis


class RVModel(ForwardModel):
    """RV Model
    `simpple.model.Model <https://simpple.readthedocs.io/en/stable/api/model.html#simpple.model.ForwardModel>` subclass for RV models.

    :param parameters: Model parameters specified as a dictionary of `simpple.distribution.Distribution <https://simpple.readthedocs.io/en/stable/api/distributions.html>` objects.
    :param num_planets: Number of planets in the model
    :param t: Array of time at which RV observations were taken
    :param rv: Array of RV observations in m/s
    :param erv: Array of uncertainty on RV observations in m/s
    :param basis: Name, parameter str or Basis class of the fitting basis to use. See also :mod:`simppler.basis`. The :class:`simppler.basis.DefaultBasis` is used by default.
    :param tmod: Array of times at which to generate the model when plotting. Set to `t` by default.
    :param time_base: Time base to use as the "zero time" for trend components in the model.
    """

    def __init__(
        self,
        parameters: dict[str, Distribution],
        num_planets: int,
        t: np.ndarray,
        rv: np.ndarray,
        erv: np.ndarray,
        inst: np.ndarray | None = None,
        basis: str | Basis = "default",
        tmod: np.ndarray | None = None,
        time_base: float = 0.0,
    ):
        super().__init__(parameters)

        self.num_planets = num_planets
        self.t = t
        if tmod is not None:
            self.tmod = tmod
        else:
            self.tmod = self.t.copy()
        self.rv = rv
        self.erv = erv
        if isinstance(basis, Basis):
            self.basis = basis
        elif basis in BASIS_DICT:
            self.basis = BASIS_DICT[basis]()
        elif basis in BASIS_PARAM_DICT:
            self.basis = BASIS_PARAM_DICT[basis]()
        else:
            basis_names = list(BASIS_DICT)
            basis_pstrs = list(BASIS_PARAM_DICT)
            raise ValueError(
                f"Unknown basis {basis}. Must be a Basis object"
                f" one of {basis_names} or one of {basis_pstrs}"
            )
        self.time_base = time_base

        expected_prefixes = self.basis.pstr.split(" ")
        expected_params = [prefix + str(ipl) for prefix in expected_prefixes for ipl in range(1, self.num_planets+1)]
        for pname in expected_params:
            if pname not in parameters:
                raise KeyError(f"Required parameter {pname} not found in parameters dictionary with keys {parameters.keys()}")

        self.inst = inst
        if self.inst is not None:
            self.inst_unique = list(np.unique(self.inst))
        else:
            self.inst_unique = None

        optional_params_nosuffix = ["jit", "gamma", "dvdt", "curv"]
        if self.inst is not None:
            optional_params = [
                f"{pname}_{iname}"
                for pname in optional_params_nosuffix
                for iname in self.inst_unique
            ]
        else:
            optional_params = optional_params_nosuffix


        allowed_params = expected_params + optional_params
        for pname in parameters:
            if pname not in allowed_params:
                raise ValueError(f"Unexpected parameter {pname}. Allowed parameters are {allowed_params}")


    def _log_likelihood(self, p: dict) -> float:
        rvmod = self.forward(p, self.t)
        if self.inst is None:
            s2 = self.erv**2 + p.get("jit", 0.0) ** 2
        else:
            s2 = self.erv**2
            for inst in self.inst_unique:
                inst_mask = self.inst == inst
                s2[inst_mask] += p.get(f"jit_{inst}", 0.0) ** 2
        return -0.5 * np.sum(np.log(2 * np.pi * s2) + (self.rv - rvmod) ** 2 / s2)

    def _forward(
        self,
        params: dict,
        t: np.ndarray,
        planets: list[int] | None = None,
        include_sys: bool = True,
        inst: str | list[str] | None = None,
    ):
        vel = np.zeros(len(t))
        params_synth = self.basis.to_synth(params, self.num_planets)
        if planets is None:
            planets = range(1, self.num_planets + 1)

        for num_planet in planets:
            per = params_synth[f"per{num_planet}"]
            tp = params_synth[f"tp{num_planet}"]
            e = params_synth[f"e{num_planet}"]
            w = params_synth[f"w{num_planet}"]
            k = params_synth[f"k{num_planet}"]
            orbel_synth = np.array([per, tp, e, w, k])
            vel += kepler.rv_drive(t, orbel_synth)

        if include_sys:
            if self.inst is None:
                vel += params.get("gamma", 0.0)
                vel += params.get("dvdt", 0.0) * (t - self.time_base)
                vel += params.get("curv", 0.0) * (t - self.time_base) ** 2
            else:
                use_inst = self.inst_unique if inst is None else inst
                if isinstance(use_inst, str):
                    use_inst = [use_inst]
                for inst in use_inst:
                    inst_mask = self.inst == inst
                    vel[inst_mask] += params.get(f"gamma_{inst}", 0.0)
                    vel[inst_mask] += params.get(f"dvdt_{inst}", 0.0) * (t[inst_mask] - self.time_base)
                    vel[inst_mask] += params.get(f"curv_{inst}", 0.0) * (t[inst_mask] - self.time_base) ** 2
        return vel

    def to_radvel(self, init_values: str | np.ndarray = "sample") -> "Posterior":
        import radvel

        def _convert_dist(pname: str, pdist: Distribution) -> radvel.prior.Prior:
            if isinstance(pdist, sdist.Uniform):
                return radvel.prior.HardBounds(pname, pdist.low, pdist.high)
            elif isinstance(pdist, sdist.Normal):
                return radvel.prior.Gaussian(pname, pdist.mu, pdist.sigma)
            elif isinstance(pdist, sdist.LogUniform):
                return radvel.prior.Jeffreys(pname, pdist.low, pdist.high)
            else:
                raise TypeError(
                    f"Distribution of type {type(pdist)} cannot be converted to radvel"
                )

        radvel_params = radvel.Parameters(self.num_planets, basis=self.basis.pstr)
        priors = []
        for i, (pname, pdist) in enumerate(self.parameters.items()):
            if init_values == "sample":
                pval = pdist.sample()
            elif isinstance(init_values, (np.ndarray, list, tuple)):
                pval = init_values[i]
            elif isinstance(init_values, dict):
                pval = init_values[pname]
            else:
                raise ValueError(f"Invalid init_values {init_values}")

            radvel_params[pname] = radvel.Parameter(value=pval)
            if pname in self.fixed_p:
                radvel_params[pname].vary = False
            else:
                priors += [_convert_dist(pname, pdist)]
        mod = radvel.RVModel(radvel_params, time_base=self.time_base)
        like = radvel.likelihood.RVLikelihood(mod, self.t, self.rv, self.erv)
        post = radvel.posterior.Posterior(like)
        post.priors = priors
        return post
