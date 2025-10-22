import numpy as np
from numba import jit


@jit
def true_anomaly(E, e):
    # Murray and Dermott p. 33
    tanfo2 = np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)
    f = 2 * np.arctan(tanfo2)
    return f


@jit
def func(E, e, M):
    return E - e * np.sin(E) - M


@jit
def dfunc(E, e):
    return 1 - e * np.cos(E)


@jit
def d2func(E, e):
    return e * np.sin(E)


@jit
def d3func(E, e):
    return e * np.cos(E)


@jit
def solve_newton_raphson(E0, e, M):
    # M is an array
    E = E0  # Start at E0
    E0 = np.inf  # Make sure first iter fails
    tol = 1e-12
    fail = np.abs(E - E0) >= tol
    while np.any(fail):
        E0 = E  # Update E0 to previous E
        E[fail] = E0[fail] - func(E0[fail], e, M[fail]) / dfunc(E0[fail], e)  # New E
        fail = np.abs(E - E0) >= tol
    return E


@jit
def solve_danby(E0, e, M):
    E = E0  # Start at E0
    E0 = np.inf  # Make sure first iter fails
    tol = 1e-12
    fail = np.abs(E - E0) >= tol
    while np.any(fail):
        E0 = E  # Update E0 to previous E
        Ef = E0[fail]
        # TODO: Maybe writing the expressions here would be faster? Maybe not with numba
        f = func(Ef, e, M[fail])
        fp = dfunc(Ef, e)
        f2p = d2func(Ef, e)
        f3p = d3func(Ef, e)

        d1 = -f / fp
        d2 = -f / (fp + 0.5 * d1 * f2p)
        d3 = -f / (fp + 0.5 * d2 * f2p + d2**2 * f3p / 6)
        E[fail] = Ef + d3
        fail = np.abs(E - E0) >= tol
    return E


@jit
def starter(M, ecc, ome):
    M2 = np.square(M)
    alpha = 3 * np.pi / (np.pi - 6 / np.pi)
    # TODO: += with type hints?
    alpha = alpha + 1.6 / (np.pi - 6 / np.pi) * (np.pi - M) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * M
    q = 2 * alphad * ome - M2
    q2 = np.square(q)
    w = np.square(np.cbrt(np.abs(r) + np.sqrt(q2 * q + r * r)))
    return (2 * r * w / (np.square(w) + w * q + q2) + M) / d


@jit
def refine(M, ecc, ome, E):
    sE = E - np.sin(E)
    cE = 1 - np.cos(E)

    f_0 = ecc * sE + E * ome - M
    f_1 = ecc * cE + ome
    f_2 = ecc * (E - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24)

    return E + dE


@jit
def solve_twostep(M, e):
    # Wrap into the right range
    M = M % (2 * np.pi)

    # We can restrict to the range [0, pi)
    high = M > np.pi
    M = np.where(high, 2 * np.pi - M, M)

    # Solve
    ome = 1 - e
    E = starter(M, e, ome)
    E = refine(M, e, ome, E)

    # Re-wrap back into the full range
    E = np.where(high, 2 * np.pi - E, E)

    return E


@jit
def rv_drive(t, orbel):
    per, tp, e, w, k = orbel

    # Performance boost for circular orbits, from RadVel
    if e == 0.0:
        m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
        return k * np.cos(m + w)

    # Handle invalid parameter values
    if per < 0:
        per = 1e-4
    if e < 0:
        e = 0
    if e > 0.99:
        e = 0.99

    # TODO: Modulo, floor, none of the two? What does it change? What is used elsewhere?
    # n = 2 * np.pi / per  # mean motion, Murray and Dermott eq. 2.25
    # M = n * (t - tp)  # Mean anom, Murray and Dermott eq. 2.39
    M = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))

    k_factor = 0.85
    E0 = M + np.sign(np.sin(M)) * k_factor * e

    E = solve_danby(E0, e, M)

    nu = true_anomaly(E, e)
    rv = k * (np.cos(nu + w) + e * np.cos(w))

    return rv
