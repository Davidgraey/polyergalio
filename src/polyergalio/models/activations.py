"""
Partially Optimized activation functions and their derivatives
- Reduced temporary assignments to keep memory pressure low
- Makes use of np.errorstate where possible
- uses np.{operation_name} when it's helpful - np.divide, np.exp, np.log - as these are very slightly quicker

"""
import numpy as np
from numpy.typing import NDArray

from polyergalio.models.constants import MASKED_LOGIT

activation_dictionary = {}
activation = lambda f: activation_dictionary.setdefault(f.__name__, f)
derivative_dictionary = {}
derivative = lambda f: derivative_dictionary.setdefault(f.__name__[:-11], f)


@activation
def sigmoid(x_array: NDArray) -> NDArray:
    """
    Numerically stable sigmoid with no clip pass.

    exp(-x) overflows to inf for very negative x; 1 + inf = inf and 1 / inf = 0 is exactly the correct
    limit, so only the overflow warning needs silencing. Saves a full sweep over memory and the
    finfo/log lookup the clipped version did on every call.
    """
    out = np.negative(x_array)
    with np.errstate(over="ignore"):
        np.exp(out, out=out)
    out += 1.0
    np.reciprocal(out, out=out)
    return out


@activation
def swish(x: NDArray):
    """
    f(x) = x * σ(x) = x / (1 + exp(-x)), fused into one buffer and four passes.
    """
    t = np.negative(x)
    with np.errstate(over="ignore"):
        np.exp(t, out=t)
    t += 1.0
    np.divide(x, t, out=t)
    return t


@activation
def elu(x: NDArray, alpha: float = 1.0) -> NDArray:
    """
    f(x) = x for x > 0, alpha * (exp(x) - 1) otherwise

    """
    out = np.minimum(x, 0)
    np.expm1(out, out=out)
    scalar_alpha = np.ndim(alpha) == 0
    if not scalar_alpha or alpha != 1.0:
        out *= alpha
    if scalar_alpha and alpha <= 1.0:
        np.maximum(out, x, out=out)
    else:
        out += np.maximum(x, 0)
    return out


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
    out = x_array - x_array.max(axis=-1, keepdims=True)
    np.exp(out, out=out)
    out /= out.sum(axis=-1, keepdims=True)
    return out


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
    (batch, options) probabilities, exactly 0 on padded slots, in logits.dtype
    """
    # Casting temperature to the logits dtype matters: np.reshape(1.0, (-1, 1)) is a float64 *array*, so
    # float32 logits / it silently promoted the whole (batch, options) pipeline to float64.
    temperature = np.asarray(temperature, dtype=logits.dtype).reshape(-1, 1)

    scaled = logits / temperature
    np.putmask(scaled, token_mask == 0, MASKED_LOGIT)
    scaled -= scaled.max(axis=-1, keepdims=True)
    np.exp(scaled, out=scaled)
    scaled *= token_mask
    scaled /= scaled.sum(axis=-1, keepdims=True)
    return scaled


@activation
def linear(x_array: NDArray) -> NDArray:
    """linear forward - for regression"""
    return x_array


@activation
def tanh(x: NDArray) -> NDArray:
    return np.tanh(x)


@activation
def relu(x: NDArray) -> NDArray:
    return np.maximum(0, x)


@activation
def relu_leaky(x: NDArray, alpha=0.1) -> NDArray:
    """
    f(x) = x for x > 0, alpha * x otherwise.
    """
    if np.ndim(alpha) != 0:  # per-channel alpha: keep the general path
        return np.where(x > 0, x, alpha * x)
    out = np.multiply(x, alpha)

    (np.maximum if alpha <= 1 else np.minimum)(out, x, out=out)
    return out


@activation
def mod_relu(x: NDArray, bias: float = -0.2) -> NDArray:
    """
    modrelu = z / |z| * max(|z| +b, 0)
    """
    magnitude = np.abs(x)
    activated = magnitude > np.maximum(-bias, 0.0)

    magnitude += ~activated
    inverse = np.divide(activated, magnitude, out=magnitude)

    inverse *= bias
    inverse += activated
    return x * inverse


# ===================== and their derivatives ======================
#   gradient -- the post-activation output of the forward pass
#   x        -- the pre-activation z of the forward pass
#   upstream -- incoming dL/d(output)

# which of the forward pass's output and pre-activation z each derivative reads; a layer keeps only these for backward
DERIVATIVE_INPUTS = {
    "linear": (),
    "sigmoid": ("output",),
    "tanh": ("output",),
    "softmax": ("output",),
    "relu": ("z",),
    "relu_leaky": ("z",),
    "swish": ("z",),
    "elu": ("z",),
}


@derivative
def sigmoid_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    d = np.subtract(1.0, gradient)
    d *= gradient
    d *= upstream
    return d


@derivative
def relu_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    return upstream * (x > 0)


@derivative
def relu_leaky_derivative(
        gradient: NDArray, x: NDArray, upstream: NDArray, alpha: float = 0.1
) -> NDArray:
    if np.ndim(alpha) == 0 and 0.0 <= alpha <= 1.0:
        slope = np.sign(x)
        np.maximum(slope, alpha, out=slope)
        slope *= upstream
        return slope
    out = np.multiply(upstream, alpha)
    np.putmask(out, x > 0, upstream)
    return out


@derivative
def tanh_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    d = np.square(gradient)
    np.subtract(1.0, d, out=d)
    d *= upstream
    return d


@derivative
def swish_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    """
    σ(x) * (1 + x * (1 - σ(x)))
    """
    s = sigmoid(x)
    d = np.subtract(1.0, s)
    d *= s
    d *= x
    d += s
    d *= upstream
    return d


@derivative
def elu_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    d = np.minimum(x, 0)
    np.exp(d, out=d)
    d *= upstream
    return d


@derivative
def linear_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    return upstream


@derivative
def softmax_derivative(gradient: NDArray, x: NDArray, upstream: NDArray) -> NDArray:
    """Over the last axis, so it holds for any batch shape."""
    # einsum does the row-wise dot without temporary
    dot = np.einsum("...i,...i->...", upstream, gradient)[..., None]
    out = upstream - dot
    out *= gradient
    return out


@derivative
def mod_relu_derivative(z, beta, dout, eps=1e-8):
    """
    Exact gradients of modReLU

    Parameters
    ----------
    z : pre-activation
    beta : bias, broadcast against z without its first axis
    dout : complex upstream gradient, dL/dRe + i dL/dIm

    Returns
    -------
    d_bias : summed over the first axis
    d_z : same shape as z
    """
    inverse = np.abs(z)
    activated = inverse > np.maximum(-beta, 0.0)
    inverse += ~activated
    np.divide(activated, inverse, out=inverse)

    radial = np.multiply(dout.real, z.real)
    shrink = np.multiply(dout.imag, z.imag)
    radial += shrink
    radial *= inverse

    d_bias = radial.sum(axis=0)

    np.multiply(inverse, beta, out=shrink)
    radial *= shrink
    radial *= inverse
    shrink += activated
    # free them before the complex
    del activated, inverse

    d_z = dout * shrink
    del shrink
    d_z -= z * radial
    return d_bias, d_z
