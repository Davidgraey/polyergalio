from __future__ import annotations

from abc import ABC
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.layers.basic_layers import FullyConnectedLayer, Layer, NormalizeLayer


def hartley(x_array: NDArray, axis: int = -1) -> NDArray:
    x_freq = np.fft.fft(x_array, axis=axis)
    return x_freq.real - x_freq.imag


def hartley_2d(x_array: NDArray, axes: tuple = (-2, -1)) -> NDArray:
    x_freq = np.fft.fft2(x_array, axes=axes)
    return x_freq.real - x_freq.imag


class FrequencyFFT(Layer):
    cache_names = ("in_shape", "input", "output")

    def __init__(self, max_sequence_length: int, window_size: int):
        """
        Seting up a process for FFT transformations of windows of audio data - expecting data of 1 size batch,
        Preprocessing assumed: sliding windows or patches.
        shape -> (number_of_windows, samples per window)

        We utilize a blackamn kernel to "ease in and out" -- frequency jumps from non-zero starts can give artifacts. Common for FFT / DFT application

        The transform is a discrete Hartley, not Re(fft). Re(fft) is blind to
        the circularly-odd part of the window, so phase information is lost
        and distinct windows collapse to the same output. Hartley is full rank
        and stays in real arithmetic, at the cost of a window_size wide output
        rather than window_size // 2 + 1.

        Parameters
        ----------
        max_sequence_length : the MAXIMUM supported sequence length - (windows)
        window_size : the number of samples in a window
        """
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.window_size = window_size
        self.window_kernel: NDArray = np.blackman(window_size)
        self.declare_shapes(inputs=((None, window_size),), outputs=((None, window_size),))

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (number_of_windows, samples per window)
        mask : unused; the transform mixes only within a window's own samples, never across windows, kept for
            pass-through compatibility with the graph

        Returns
        -------
        the Hartley transform of the windowed data, same width as the window
        """
        self.in_shape = incoming_x.shape

        assert self.in_shape[0] <= self.max_sequence_length, f"Shapes don't match in {self}"
        assert self.in_shape[-1] == self.window_size, (
            f"last axis must equal window_size {self.window_size}, got {self.in_shape[-1]}"
        )

        input_reshaped = incoming_x.reshape(-1, self.in_shape[-1])
        self.output = hartley(self.window_kernel * input_reshaped, axis=-1)
        return self.output.reshape(self.in_shape)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        """Hartley is its own adjoint, so the backward pass is the same transform followed by the same window weighting."""
        grad = self.window_kernel * hartley(incoming_grad.reshape(-1, self.in_shape[-1]), axis=-1)
        return grad.reshape(self.in_shape)


class HartleyLayer(Layer, ABC):
    """Shape-preserving Hartley transform over the last axis, or the last two when use_2d."""
    preserves_shape = True
    cache_names = ("input", "output", "gradient")

    def __init__(self, use_2d: bool = True):
        super().__init__()
        self.use_2d = use_2d

    @property
    def fft_axes(self) -> tuple[int, int] | int:
        return (-2, -1) if self.use_2d else -1

    def transform(self, array: NDArray) -> NDArray:
        """The Hartley transform over fft_axes."""
        if self.use_2d:
            return hartley_2d(array, axes=self.fft_axes)
        return hartley(array, axis=self.fft_axes)


class FourierLayer(HartleyLayer):
    #  https://ieeexplore.ieee.org/document/9616294

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Hartley transform

        mask : unused; the transform is linear and the sequence is already zero-padded upstream, so a padded
            position contributes nothing to any output frequency. Kept for pass-through compatibility with the graph.
        """
        self.output = self.transform(incoming_x)
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        """Hartley is symmetric, so the adjoint is the same transform."""
        self.gradient = self.transform(incoming_grad)
        return self.gradient


class InverseFourierLayer(HartleyLayer):
    # https://arxiv.org/pdf/2502.18394
    cache_names = ("scale",)

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Hartley is its own inverse up to 1/N, so the inverse direction is the same transform carrying that scale.
        Stacking this on top of FourierLayer reconstructs the input exactly.

        mask : unused, same reasoning as FourierLayer.forward.
        """
        shape = incoming_x.shape
        self.scale = shape[-2] * shape[-1] if self.use_2d else shape[-1]
        self.output = self.transform(incoming_x) / self.scale
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        """Symmetric transform, so the adjoint carries the same 1/N."""
        self.gradient = self.transform(incoming_grad) / self.scale
        return self.gradient


class FourierAttention(Layer):
    parameter_names = ("norm_a", "fc", "norm_b")
    cache_names = ("training_now", "output", "gradient")

    def __init__(self, ni: int, no: int, use_2d: bool = True):
        super().__init__()
        assert ni == no, f"the feed forward residual needs matching widths, got ni={ni} no={no}"
        self.ni, self.no, self.use_2d = ni, no, use_2d
        self.fftlayer = FourierLayer(use_2d)
        self.norm_a = NormalizeLayer(input_dimension=ni, shift_scale=False)
        self.fc = FullyConnectedLayer(input_dimension=ni, output_dimension=no, activation_type="relu")
        self.norm_b = NormalizeLayer(input_dimension=no, shift_scale=True)

        self.declare_shapes(inputs=((ni,),), outputs=((no,),))
        self.zero_gradients()

    def forward(self, x_data: NDArray, training_now: Optional[bool] = None, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        x_data : (batch, sequence, hidden), or (batch, hidden) when use_2d is False
        training_now : None follows the layer's train() / eval() mode
        mask : unused; neither the FFT mixing nor the per-position norm and feed-forward sublayers need it, kept
            for pass-through compatibility
        """
        if self.fftlayer.use_2d:
            assert x_data.ndim >= 3, (
                "use_2d mixes over the last two axes, which on a 2D "
                f"(batch, hidden) input means mixing across the batch and "
                f"leaking between samples. Got shape {x_data.shape}; pass "
                "(batch, sequence, hidden) or use use_2d=False."
            )
        self.training_now = self.training if training_now is None else training_now
        fft_x = self.norm_a(self.fftlayer(x_data) + x_data)
        self.output = self.norm_b(self.fc(fft_x) + fft_x)
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        grad = self.norm_b.backward(incoming_gradient)
        grad = self.fc.backward(grad) + grad
        grad = self.norm_a.backward(grad)
        grad = self.fftlayer.backward(grad) + grad
        self.gradient = grad
        return grad.real
