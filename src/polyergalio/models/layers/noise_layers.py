"""
Noise sampling for the latent diffusion sampler:
Spatially Adaptive Noise Injection (Lavda et al. 2026) on a rectified flow, with the per-element uncertainty read from
the denoiser's variance logits.
"""
import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import polyergalio.models.activations as activations
from polyergalio.models.constants import ANY_SHAPE
from polyergalio.models.layers.basic_layers import RNG, Layer

complementary_error = np.vectorize(math.erfc, otypes=[float])


class AdaptiveNoiseSampler(Layer):
    """
    One reverse step of a rectified flow, x_t to x_s with s < t, injecting noise where the latent is uncertain.

        clean = x_t - t * velocity
        noise = x_t + (1 - t) * velocity
        uncertainty v = (model variance - posterior variance) / clean_weight ** 2
        gate = erfc(sqrt(tolerance / (2 v)))
        injected = clean_weight ** 2 * v + gate * posterior variance  #  at most s ** 2
        x_s = (1 - s) * clean + sqrt(s ** 2 - injected) * noise + sqrt(injected) * z
        z ~ N(0, I)

    The model variance is the denoiser's variance logits projected between the two bounds of the step from t

    v becomes the posterior variance of the latent
    With nothing injected the step is the Euler step.

    A step that ends at s = 0 is deterministic

    Inputs: noisy latents (B, N, C), denoiser output (B, N, 2C)
    Output shape: (B, N, C)
    """
    cache_names = (
        "time", "earlier", "kept", "injected", "draw", "noise_estimate", "live", "floored", "clean_weight",
        "posterior_variance", "model_variance", "bound_spread", "fraction", "reach", "uncertainty",
    )

    def __init__(self, total_steps: int, tolerance: float = 1e-3, floor: float = 1e-6):
        """
        Parameters
        ----------
        total_steps : noise levels of the diffusion scheduler; step k is flow time k / (total_steps - 1)
        tolerance : uncertainty scale of the gate; larger injects less noise
        floor : lower bound on the uncertainty
        """
        super().__init__()
        self.total_steps = total_steps
        self.tolerance = tolerance
        self.floor = floor
        self.declare_shapes(inputs=(ANY_SHAPE, ANY_SHAPE), outputs=(ANY_SHAPE,))

    def step_times(self, steps: NDArray) -> NDArray:
        """Flow time in [0, 1] of integer steps, shaped (B, 1, 1)."""
        return (np.asarray(steps, dtype=float) / (self.total_steps - 1)).reshape(-1, 1, 1)

    def forward(self, noisy_latents: NDArray, prediction: NDArray, timesteps: NDArray,
                next_timesteps: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        noisy_latents : (B, N, C) at timesteps
        prediction : (B, N, 2C) denoiser output, velocity then variance logits
        timesteps : (B,) integer steps, at least 1
        next_timesteps : (B,) integer steps to move to, below timesteps; timesteps - 1 if None

        Returns
        -------
        (B, N, C) latents at next_timesteps
        """
        steps = np.asarray(timesteps)
        earlier_steps = steps - 1 if next_timesteps is None else np.asarray(next_timesteps)
        assert np.all(steps >= 1) and np.all(earlier_steps < steps) and np.all(earlier_steps >= 0), (
            "timesteps must be at least 1 and next_timesteps must fall in [0, timesteps)"
        )
        velocity, logits = np.split(prediction, 2, axis=-1)
        time, earlier = self.step_times(steps), self.step_times(earlier_steps)
        final = earlier == 0.0
        positive = np.where(final, 0.5 * time, earlier)

        retained = (1.0 - time) / (1.0 - positive)
        step_variance = time ** 2 - retained ** 2 * positive ** 2
        posterior_variance = step_variance * positive ** 2 / time ** 2
        clean_weight = (1.0 - positive) * step_variance / time ** 2

        fraction = activations.sigmoid(logits)
        bound_spread = np.log(step_variance) - np.log(posterior_variance)
        model_variance = posterior_variance * np.exp(fraction * bound_spread)
        raw = (model_variance - posterior_variance) / clean_weight ** 2
        floored = raw <= self.floor
        uncertainty = np.maximum(raw, self.floor)
        reach = np.sqrt(self.tolerance / (2.0 * uncertainty))

        limit = earlier ** 2
        injected = clean_weight ** 2 * uncertainty + complementary_error(reach) * posterior_variance
        live = ~final & (injected < limit)
        injected = np.where(final, 0.0, np.minimum(injected, limit))
        kept = np.sqrt(limit - injected)

        noise_estimate = noisy_latents + (1.0 - time) * velocity
        draw = RNG.standard_normal(noisy_latents.shape)
        if self.training:
            self.time, self.earlier, self.kept, self.injected = time, earlier, kept, injected
            self.draw, self.noise_estimate, self.live, self.floored = draw, noise_estimate, live, floored
            self.clean_weight, self.posterior_variance = clean_weight, posterior_variance
            self.model_variance, self.bound_spread, self.fraction = model_variance, bound_spread, fraction
            self.reach, self.uncertainty = reach, uncertainty
        return (1.0 - earlier) * (noisy_latents - time * velocity) + kept * noise_estimate + np.sqrt(injected) * draw

    def backward(self, incoming_gradient: NDArray) -> tuple[NDArray, NDArray]:
        """
        Returns
        -------
        gradient with respect to the noisy latents and to the denoiser output
        """
        clean_scale = 1.0 - self.earlier
        noisy_gradient = incoming_gradient * (clean_scale + self.kept)
        velocity_gradient = incoming_gradient * (self.kept * (1.0 - self.time) - clean_scale * self.time)

        deviation = np.where(self.live, np.sqrt(self.injected), 1.0)
        kept = np.where(self.live, self.kept, 1.0)
        injected_gradient = self.live * incoming_gradient * 0.5 * (self.draw / deviation - self.noise_estimate / kept)
        gate_slope = np.exp(-self.reach ** 2) * self.reach / (np.sqrt(np.pi) * self.uncertainty)
        uncertainty_gradient = ~self.floored * injected_gradient * (
            self.clean_weight ** 2 + self.posterior_variance * gate_slope)
        logits_gradient = (uncertainty_gradient / self.clean_weight ** 2 * self.model_variance * self.bound_spread
                           * self.fraction * (1.0 - self.fraction))
        return noisy_gradient, np.concatenate([velocity_gradient, logits_gradient], axis=-1)

    def __str__(self):
        return f"AdaptiveNoiseSampler over {self.total_steps} steps, tolerance {self.tolerance}"

    def __repr__(self):
        return self.__str__()
