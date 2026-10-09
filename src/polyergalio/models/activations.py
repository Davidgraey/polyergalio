import numpy as np
from numpy.typing import NDArray

from polyergalio.models.constants import MASKED_LOGIT

activation_dictionary = {}
activation = lambda f: activation_dictionary.setdefault(f.__name__, f)
derivative_dictionary = {}

derivative = lambda f: derivative_dictionary.setdefault(f.__name__[:-11], f)


@activation
def sigmoid(x_array: NDArray) -> NDArray:
    """Numerically stable version of sigmoid via clipping -- the bool masks previously were costly"""
    limit = np.log(np.finfo(x_array.dtype).max) - 1.0

    out = np.clip(x_array, -limit, limit)

    np.negative(out, out=out)
    np.exp(out, out=out)
    out += 1.0
    np.reciprocal(out, out=out)
    return out


@activation
def swish(x: NDArray):
    """
    f(x) = x * σ(x)
    Parameters
    ----------
    x :

    Returns
    -------

    """

    return x * sigmoid(x)


@activation
def elu(x: NDArray, alpha: float = 1.0) -> NDArray:
    """f(x) = x for x > 0, alpha * (exp(x) - 1) otherwise"""
    return np.where(x > 0, x, alpha * np.expm1(np.minimum(x, 0)))


@activation
def softmax(x_array: NDArray) -> NDArray:
    """
    N-dimensional vector with values that sum to one - probabilistic multiclass
    Parameters
    ----------
    x_array : incoming values in numpy array

    Returns
    -------
    evaluation (single val for input_sample (row) of x)
    """

    exps = np.exp(x_array - np.max(x_array, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def masked_softmax(
    logits: NDArray, token_mask: NDArray, temperature: float | NDArray = 1.0
) -> NDArray:
    """
    Softmax over each row's valid options only.

    Parameters
    ----------
    logits : (batch, options)
    token_mask : (batch, options), 1 for a real option
    temperature : scalar, or (batch,) per-row

    Returns
    -------
    (batch, options) probabilities, exactly 0 on padded slots
    """
    temperature = np.reshape(temperature, (-1, 1))
    scaled = np.where(token_mask, logits / temperature, MASKED_LOGIT)
    exps = np.exp(scaled - scaled.max(axis=-1, keepdims=True)) * token_mask
    return exps / exps.sum(axis=-1, keepdims=True)

@activation
def linear(x_array: NDArray) -> NDArray:
    """linear forward - for regression"""
    return x_array


@activation
def tanh(x: NDArray) -> NDArray:
    """
    **********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    """
    # return 2 / (1 + np.e ** (-2 * x)) - 1
    return np.tanh(x)


@activation
def relu(x: NDArray) -> NDArray:
    """**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    """
    return np.maximum(0, x)


@activation
def relu_leaky(x: NDArray, alpha=0.1) -> NDArray:
    """**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    """
    return np.where(x > 0, x, alpha * x)

@activation
def mod_relu(x: NDArray, bias: float = -0.2) -> NDArray:
    """
     modrelu = z / |z| * max(|z| +b, 0)
    """
    # z / |z|
    magnitude = np.abs(x)
    phase = np.divide(x, magnitude, out=np.zeros_like(x), where=magnitude != 0)

    activated = np.maximum(magnitude + bias, 0.0)

    return activated * phase


# ===================== and their derivatives ======================
#   gradient -- the post-activation output of the forward pass
#   x        -- the pre-activation z of the forward pass
#   upstream -- incoming dL/d(output)

@derivative
def sigmoid_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    return upstream * gradient * (1 - gradient)


@derivative
def relu_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    return upstream * (x > 0)


@derivative
def relu_leaky_derivative(
    gradient: NDArray, x: NDArray, upstream: NDArray, alpha: float = 0.1
) -> NDArray:
    return np.where(x > 0, upstream, upstream * alpha)


@derivative
def tanh_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    return upstream * (1 - gradient ** 2)


@derivative
def swish_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    s = sigmoid(x)
    return upstream * s * (1 + x * (1 - s))


@derivative
def elu_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    return upstream * np.where(x > 0, 1.0, np.exp(np.minimum(x, 0)))


@derivative
def linear_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    return upstream


@derivative
def softmax_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    """True VJP over the last axis, so it holds for any batch shape."""
    dot = np.sum(upstream * gradient, axis=-1, keepdims=True)
    return gradient * (upstream - dot)


@derivative
def mod_relu_derivative(z, beta, dout, eps=1e-8):
    r = np.abs(z)
    r_safe = r + eps
    mask = (r + beta) > 0

    scale = (r + beta) / r_safe
    proj = np.real(dout * np.conj(z)) / r_safe

    d_bias = (proj * mask).sum(axis=0)
    d_z = mask * (dout * scale + z * proj * (-beta / r_safe**2))
    return d_bias, d_z


if __name__ == "__main__":
    x = np.array([[1, 2, 3, 4, 5], [3, 2, 4, 5, 6], [9, 8, 7, 6, 5]]) / 9
    upstream = np.ones_like(x)
    for name, _func in activation_dictionary.items():
        if name == "mod_relu":
            continue
        x_hat = _func(x)
        delta = derivative_dictionary[name](x_hat, x, upstream)
        print(f"{name}: forward {x_hat.shape} -> delta {delta.shape}")
