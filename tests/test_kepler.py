import numpy as np
import pytest
import radvel.kepler
import radvel.orbit

from simppler.kepler import rv_drive, solve_danby, true_anomaly, solve_newton_raphson


@pytest.mark.parametrize(
    "solver,atol", [(solve_danby, 1e-12), (solve_newton_raphson, 1e-2)]
)
def test_solvers(solver, atol):
    t = np.linspace(0, 100, num=1000)
    per, tp, e, w, k = 1.0, 5.0, 0.3, np.pi, 1.0

    M = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))

    k = 0.85
    E0 = M + np.sign(np.sin(M)) * k * e

    E = solver(E0, e, M)
    eccarr = np.full_like(t, e)
    E_radvel = radvel.kepler.kepler(M, eccarr)
    np.testing.assert_allclose(E, E_radvel, atol=atol)

    f = true_anomaly(E, e)
    f_radvel = radvel.orbit.true_anomaly(t, tp, per, e)
    np.testing.assert_allclose(f, f_radvel, atol=1e-2)


def test_rv_drive():
    t = np.linspace(0, 100, num=1000)
    per, tp, e, w, k = 1.0, 5.0, 0.3, np.pi, 1.0
    orbel = np.array([per, tp, e, w, k])
    simpple_rvs = rv_drive(t, orbel)
    radvel_rvs = radvel.kepler.rv_drive(t, orbel)
    np.testing.assert_allclose(simpple_rvs, radvel_rvs, atol=1e-5)
