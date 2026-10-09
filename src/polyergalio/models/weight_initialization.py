import inspect
from functools import partial
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray



# -------------    weight initilization functions    ---------------
def lecun(rng, ni: int, no: int) -> NDArray:
    """variance 1 / ni; linear, selu and tanh-like layers"""
    return rng.normal(0.0, 1 / np.sqrt(ni), size=(ni, no))


def glorot(rng, ni: int, no: int) -> NDArray:
    """variance 2 / (ni + no); balances forward and backward signal"""
    return rng.normal(0.0, np.sqrt(2 / (ni + no)), size=(ni, no))


def glorot_uniform(rng, ni: int, no: int) -> NDArray:
    limit = np.sqrt(6 / (ni + no))
    return rng.uniform(-limit, limit, size=(ni, no))


def kaiming_uniform(rng, ni: int, no: int, slope: float = 0.0) -> NDArray:
    """relu family; slope is the leaky-relu negative slope"""
    limit = np.sqrt(6 / ((1 + slope**2) * ni))
    return rng.uniform(-limit, limit, size=(ni, no))


def truncated_normal(
    rng, ni: int, no: int, std: float = 0.02, cutoff: float = 2.0
) -> NDArray:
    """small fixed-scale init, redrawing values beyond cutoff * std"""
    weights = rng.normal(0.0, std, size=(ni, no))
    outside = np.abs(weights) > cutoff * std
    while outside.any():
        weights[outside] = rng.normal(0.0, std, size=outside.sum())
        outside = np.abs(weights) > cutoff * std
    return weights


def orthogonal(rng, ni: int, no: int, gain: float = 1.0) -> NDArray:
    """orthonormal rows or columns, preserves norms through deep linear stacks"""
    q, r = np.linalg.qr(rng.normal(size=(max(ni, no), min(ni, no))))
    q = q * np.sign(np.diag(r))
    return gain * (q if ni >= no else q.T)


def identity(rng, ni: int, no: int, noise: float = 0.0) -> NDArray:
    """starts a map as passthrough, optionally perturbed"""
    return np.eye(ni, no) + noise * rng.normal(size=(ni, no))


def siren(
    rng, ni: int, no: int, frequency: float = 30.0, first: bool = False
) -> NDArray:
    """sine layers with the frequency folded into the weights, Sitzmann et al. 2020"""
    limit = frequency / ni if first else np.sqrt(6 / ni)
    return rng.uniform(-limit, limit, size=(ni, no))


def complex_glorot(rng, ni: int, no: int) -> NDArray:
    """Rayleigh magnitude, uniform phase, E|w|^2 = 2 / (ni + no); Trabelsi et al. 2018"""
    magnitude = rng.rayleigh(scale=1 / np.sqrt(ni + no), size=(ni, no))
    phase = rng.uniform(-np.pi, np.pi, size=(ni, no))
    return magnitude * np.exp(1j * phase)


def zeros(rng, ni: int, no: int) -> NDArray:
    return np.zeros((ni, no))


def kaiming(rng, ni: int, no: int, slope: float = 0.0) -> NDArray:
    """relu family, variance 2 / ((1 + squareedslope) ni); slope is the leaky-relu negative slope"""
    scale = np.sqrt(2 / ((1 + slope**2) * ni))
    return rng.normal(0.0, scale, size=(ni, no))


WEIGHT_INIT_DISPATCHER = {
    "linear": lecun,
    "relu": kaiming,
    "relu_leaky": partial(kaiming, slope=0.01),
    "swish": kaiming,
    "sigmoid": glorot,
    "tanh": glorot,
    "softmax": glorot,
    "selu": lecun,
    "sine": siren,
    "kaiming": kaiming,
    "kaiming_uniform": kaiming_uniform,
    "lecun": lecun,
    "glorot": glorot,
    "glorot_uniform": glorot_uniform,
    "truncated_normal": truncated_normal,
    "orthogonal": orthogonal,
    "identity": identity,
    "siren": siren,
    "complex_glorot": complex_glorot,
    "zeros": zeros,
}


# our helper function--
def get_weight_init(name: str, **kwargs) -> Callable[..., NDArray]:
    """
    Initializer by activation or init name, with keyword arguments bound

    Parameters
    ----------
    name : key of WEIGHT_INIT_DISPATCHER
    kwargs : passed through to the initializer, e.g. std, gain, slope, frequency, first

    Returns
    -------
    function of (rng, ni, no)
    """
    if name not in WEIGHT_INIT_DISPATCHER:
        raise KeyError(
            f"unknown weight init {name!r}. Known: {sorted(WEIGHT_INIT_DISPATCHER)}"
        )
    init = WEIGHT_INIT_DISPATCHER[name]
    inspect.signature(init).bind_partial(None, 0, 0, **kwargs)
    return partial(init, **kwargs)
