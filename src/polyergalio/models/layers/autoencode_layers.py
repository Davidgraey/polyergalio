from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.activations import softmax
from polyergalio.models.layers.basic_layers import RNG, FullyConnectedLayer, Layer

LOG_TWO_PI = float(np.log(2.0 * np.pi))

# TODO move to utils
def log_sum_exp(values: NDArray, axis: int = -1) -> NDArray:
    """ Numerically stable log of the summed exponentials along axis"""
    peak = values.max(axis=axis, keepdims=True)
    return np.squeeze(peak + np.log(np.exp(values - peak).sum(axis=axis, keepdims=True)), axis=axis)


class GaussianMixtureLayer(Layer):
    """
    Learned Gaussian mixture noise, added to a latent projection.

    Each latent draws a component (hard in the forward pass, Gumbel-softmax relaxed in the backward pass) and a reparam Gaussian sample from

    logits, means and ln_variances all receive gradient

    The mixture is also the prior. The noise is shrunk by a learned scale, so the noised latent is a narrow copy of
    the mixture centred on the incoming latent. forward caches divergence, the KL from the prior to that
    distribution, and backward adds kl_weight times its gradient.
    """
    parameter_names = ("logits", "means", "log_variances", "log_scale")
    cache_names = ("in_shape", "component_weights", "relaxed_weights", "standard_noise", "component_noise", "token_mask",
                   "sampled_noise", "noisy_latents", "divergence")

    def __init__(self,
                 num_components: int,
                 latent_dimension: int,
                 temperature: float = 1.0,
                 mean_scale: float = 1.0,
                 kl_weight: float = 1.0,
                 noise_scale: float = 0.3,
                 ):
        """
        Parameters
        ----------
        num_components : number of mixture components K
        latent_dimension : width of the latent space, D
        temperature : of the relaxation that carries gradient to the component logits; lower is sharper
        mean_scale : standard deviation of the random component means at initialization
        kl_weight : weight of the mean divergence per latent in the loss; 0 turns the prior off
        noise_scale : initial width of the noise added to a latent, relative to the prior mixture
        """
        super().__init__()
        self.num_components = num_components
        self.latent_dimension = latent_dimension
        self.temperature = temperature
        self.mean_scale = mean_scale
        self.kl_weight = kl_weight
        self.noise_scale = noise_scale

        self.logits = np.zeros(num_components)
        self.means = RNG.normal(0.0, mean_scale, size=(num_components, latent_dimension))
        self.log_variances = np.zeros((num_components, latent_dimension))
        self.log_scale = np.full(latent_dimension, np.log(noise_scale))

        self.declare_shapes(inputs=((latent_dimension,),), outputs=((latent_dimension,),))
        self.preserves_shape = True
        self.zero_gradients()

    @property
    def noise(self) -> NDArray:
        """ Mean of the mixture, (latent_dimension,)."""
        return softmax(self.logits) @ self.means

    def component_log_density(self, latents: NDArray) -> NDArray:
        """
        Parameters
        ----------
        latents : (N, latent_dimension)

        Returns
        -------
        log of each component's weight times its density at each latent, (N, num_components)
        """
        difference = latents[:, None, :] - self.means
        log_normal = -0.5 * (
            LOG_TWO_PI + self.log_variances + difference ** 2 * np.exp(-self.log_variances)
        ).sum(axis=-1)
        return log_normal + self.logits - log_sum_exp(self.logits)

    def log_density(self, latents: NDArray) -> NDArray:
        """
        Parameters
        ----------
        latents : (N, latent_dimension)

        Returns
        -------
        log p(latent) under the mixture, (N,)
        """
        return log_sum_exp(self.component_log_density(latents))

    def density_gradients(self, latents: NDArray, upstream: NDArray) -> tuple[NDArray, dict]:
        """
        Parameters
        ----------
        latents : (N, latent_dimension)
        upstream : weight of each latent's log density in the objective, (N,)

        Returns
        -------
        gradient with respect to the latents, and a dict of gradients for logits, means and log_variances
        """
        joint_density = self.component_log_density(latents)
        responsibilities = np.exp(joint_density - log_sum_exp(joint_density)[:, None])
        weighted = upstream[:, None] * responsibilities

        difference = latents[:, None, :] - self.means
        scaled = weighted[:, :, None] * difference * np.exp(-self.log_variances)
        gradients = {
            "logits": (upstream[:, None] * (responsibilities - softmax(self.logits))).sum(axis=0),
            "means": scaled.sum(axis=0),
            "log_variances": -0.5 * (weighted.sum(axis=0)[:, None] - (scaled * difference).sum(axis=0)),
        }
        return -scaled.sum(axis=1), gradients

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : latent data
        mask : matching incoming_x's leading axes, 1 for a real token and 0 for padding

        Returns
        -------
        the latent state with mixture noise added, same shape as incoming_x
        """
        self.in_shape = incoming_x.shape
        assert self.in_shape[-1] == self.latent_dimension, (
            f"cannot read trailing axis {self.in_shape[-1]} as latent_dimension {self.latent_dimension}"
        )
        if not self.training:
            return incoming_x + np.exp(self.log_scale) * self.noise

        latents = incoming_x.reshape(-1, self.latent_dimension)
        if mask is not None:
            assert mask.shape == self.in_shape[:-1], (
                f"mask shape {mask.shape} must match incoming_x's leading axes {self.in_shape[:-1]}"
            )
        self.token_mask = None if mask is None else mask.reshape(-1, 1)

        gumbel = RNG.gumbel(size=(len(latents), self.num_components))
        self.relaxed_weights = softmax((self.logits + gumbel) / self.temperature)
        self.component_weights = np.eye(self.num_components)[self.relaxed_weights.argmax(axis=-1)]
        self.standard_noise = RNG.standard_normal((len(latents), self.num_components, self.latent_dimension))
        self.component_noise = self.means + np.exp(0.5 * self.log_variances) * self.standard_noise

        # nk, nkd -> nd
        noise = np.einsum("nk,nkd->nd", self.component_weights, self.component_noise)

        if self.token_mask is not None:
            # mask out zeros IF we've got a mask
            noise = noise * self.token_mask

        noisy = latents + np.exp(self.log_scale) * noise
        valid = np.ones((len(latents), 1)) if self.token_mask is None else self.token_mask
        divergence = self.log_density(noise) - self.log_scale.sum() - self.log_density(noisy)
        self.divergence = float((divergence * valid[:, 0]).sum() / max(valid.sum(), 1.0))
        self.sampled_noise = noise
        self.noisy_latents = noisy

        return noisy.reshape(self.in_shape)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        """
        Parameters
        ----------
        incoming_grad : gradient with respect to the noisy latent space

        Returns
        -------
        gradient wrt ins, including the divergence's pull on the latents
        """
        delta = incoming_grad.reshape(-1, self.latent_dimension)
        valid = np.ones(len(delta)) if self.token_mask is None else self.token_mask[:, 0]
        if self.token_mask is not None:
            delta = delta * self.token_mask

        row_weight = self.kl_weight * valid / max(valid.sum(), 1.0)
        noise_term, noise_gradients = self.density_gradients(self.sampled_noise, row_weight)
        latent_term, latent_gradients = self.density_gradients(self.noisy_latents, -row_weight)
        latent_delta = delta + latent_term
        scale = np.exp(self.log_scale)
        self.gradient_log_scale = scale * (latent_delta * self.sampled_noise).sum(axis=0) - row_weight.sum()
        delta = scale * latent_delta + noise_term

        weights = self.component_weights
        self.gradient_means = weights.T @ delta + noise_gradients["means"] + latent_gradients["means"]

        self.gradient_log_variances = 0.5 * np.exp(0.5 * self.log_variances) * np.einsum(
            "nk,nd,nkd->kd", weights, delta, self.standard_noise
        ) + noise_gradients["log_variances"] + latent_gradients["log_variances"]

        relaxed = self.relaxed_weights
        weight_grad = np.einsum("nd,nkd->nk", delta, self.component_noise)
        mixed = (relaxed * weight_grad).sum(axis=-1, keepdims=True)
        self.gradient_logits = (
            (relaxed * (weight_grad - mixed)).sum(axis=0) / self.temperature
            + noise_gradients["logits"] + latent_gradients["logits"]
        )

        return incoming_grad + latent_term.reshape(self.in_shape)

    def sample(self, num_samples: int) -> NDArray:
        """
        Parameters
        ----------
        num_samples : number of latents to draw

        Returns
        -------
        latents drawn from the hard mixture, (num_samples, latent_dimension)
        """
        components = RNG.choice(self.num_components, size=num_samples, p=softmax(self.logits))
        noise = RNG.standard_normal((num_samples, self.latent_dimension))
        return self.means[components] + np.exp(0.5 * self.log_variances[components]) * noise


class GMVAELayer(Layer):
    """
    A mixture variational autoencoder
    project down to a latent space, mix in learned / parameterized Gaussian noise, and project back up.

    The noised latent is cached for backprop
    """
    parameter_names = ("down_projection", "mixture", "up_projection")
    cache_names = ("latent",)

    def __init__(self,
                 input_dimension: int,
                 latent_dimension: int,
                 num_components: int,
                 activation_type: str = "tanh",
                 temperature: float = 1.0,
                 mean_scale: float = 1.0,
                 kl_weight: float = 1.0,
                 noise_scale: float = 0.3,
                 ):
        """
        Parameters
        ----------
        input_dimension : incoming last dim (hidden) of the incoming and reconstructed data
        latent_dimension : width of the latent space (downproject to this)
        num_components : number of mixture components in the noise
        activation_type : activation of the projection down into latent space
        temperature : of the relaxation that carries gradient to the mixture's component logits
        mean_scale : standard deviation of the mixture's random means at initialization
        kl_weight : weight of the mixture's mean divergence in the loss; 0 turns the prior off
        noise_scale : initial width of the noise added to a latent, relative to the prior mixture
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.latent_dimension = latent_dimension
        self.num_components = num_components
        self.activation_type = activation_type
        self.temperature = temperature
        self.mean_scale = mean_scale
        self.kl_weight = kl_weight
        self.noise_scale = noise_scale

        self.down_projection = FullyConnectedLayer(input_dimension, latent_dimension, activation_type)
        self.mixture = GaussianMixtureLayer(num_components, latent_dimension, temperature, mean_scale, kl_weight, noise_scale)
        self.up_projection = FullyConnectedLayer(input_dimension=latent_dimension,
                                                 output_dimension=input_dimension,
                                                 activation_type="swish",
                                                 is_output=True)
        self.declare_shapes(inputs=((input_dimension,),), outputs=((input_dimension,),))
        self.preserves_shape = True
        self.zero_gradients()

    @property
    def divergence(self) -> float:
        """Mean KL per latent from the last training forward; add kl_weight times it to the loss."""
        return self.mixture.divergence

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x :
        mask : matching incoming_x's leading axes, 1 for a real token and 0 for padding; keeps padding out of the
            mixture's gradients

        Returns
        -------
        the reconstruction, same shape as incoming_x
        """
        self.latent = self.mixture(self.down_projection(incoming_x), mask)
        return self.up_projection(self.latent)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        return self.down_projection.backward(self.mixture.backward(self.up_projection.backward(incoming_grad)))

    def generate(self, num_samples: int) -> NDArray:
        """
        Parameters
        ----------
        num_samples : number of outputs to draw

        Returns
        -------
        outputs decoded from latents drawn from the mixture, (num_samples, input_dimension)
        """
        return self.up_projection(self.mixture.sample(num_samples))
