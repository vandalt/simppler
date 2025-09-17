import numpy as np


def timetrans_to_timeperi(tc, per, ecc, omega):
    """
    Convert Time of Transit to Time of Periastron Passage

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


def to_synth(params: dict, num_planets: int, name: str) -> dict:
    params_synth = {}
    # TODO: Unspaghetti
    if name == "per tp e w k":
        return params
    elif name == "per tc secosw sesinw logk":
        for i in range(1, num_planets+1):
            params_synth[f"per{i}"] = params[f"per{i}"]
            params_synth[f"e{i}"] = (
                params[f"secosw{i}"] ** 2 + params[f"sesinw{i}"] ** 2
            )
            params_synth[f"w{i}"] = np.arctan2(
                params[f"sesinw{i}"], params[f"secosw{i}"]
            )
            params_synth[f"tp{i}"] = timetrans_to_timeperi(
                params[f"tc{i}"],
                params_synth[f"per{i}"],
                params_synth[f"e{i}"],
                params_synth[f"w{i}"],
            )
            params_synth[f"k{i}"] = np.exp(params[f"logk{i}"])
    else:
        raise ValueError(f"Basis {name} unsupported")

    return params_synth
