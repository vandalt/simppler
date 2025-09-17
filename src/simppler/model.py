import warnings
from simpple.model import ForwardModel
from simpple.distributions import Distribution
import simpple.distributions as sdist

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ImportWarning, module="radvel")
    from radvel import kepler
import numpy as np

from simppler import basis


class RVModel(ForwardModel):
    def __init__(
        self,
        parameters: dict[str, Distribution],
        num_planets: int,
        t: np.ndarray,
        rv: np.ndarray,
        erv: np.ndarray,
        basis_name: str,
        tmod: np.ndarray | None = None,
        time_base: float = 0.0,
    ):
        super().__init__(parameters)

        # TODO: Check for parameters
        self.num_planets = num_planets
        self.t = t
        if tmod is not None:
            self.tmod = tmod
        else:
            self.tmod = self.t.copy()
        self.rv = rv
        self.erv = erv
        self.basis = basis_name
        self.time_base = time_base

    def _log_likelihood(self, p: dict) -> float:
        rvmod = self.forward(p, self.t)
        s2 = self.erv**2 + p.get("jit", 0.0) ** 2
        return -0.5 * np.sum(np.log(2 * np.pi * s2) + (self.rv - rvmod) ** 2 / s2)

    def _forward(self, params: dict, t: np.ndarray, planets: list[int] | None = None):
        vel = np.zeros(len(t))
        params_synth = basis.to_synth(params, self.num_planets, self.basis)
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
        vel += params["gamma"]
        vel += params["dvdt"] * (t - self.time_base)
        vel += params["curv"] * (t - self.time_base) ** 2
        return vel

    def to_radvel(self, init_values: str | np.ndarray = "sample"):
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

        radvel_params = radvel.Parameters(self.num_planets, basis=self.basis)
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
