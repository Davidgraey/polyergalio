from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.constants import LOG_TWO_PI
from polyergalio.models.layers.basic_layers import RNG, FullyConnectedLayer, Layer


VARIANCE_FLOOR = 1e-8
LOG_VARIANCE_BOUND = 10.0
LOG_SCALE_BOUND = LOG_VARIANCE_BOUND / 2.0


def log_sum_exp(values: NDArray, axis: int = -1) -> NDArray:
    """ stable log of the summed exponentials along axis """
    peak = values.max(axis=axis, keepdims=True)
    return np.squeeze(peak + np.log(np.exp(values - peak).sum(axis=axis, keepdims=True)), axis=axis)


def soft_bound(raw: NDArray, bound: float) -> NDArray:
    """ smoothly bound raw to (-bound, bound), for a log variance or log scale whose exp must not overflow """
    return bound * np.tanh(raw / bound)


def soft_bound_derivative(bounded: NDArray, bound: float) -> NDArray:
    """ derivative of soft_bound at the raw value that gave bounded """
    return 1.0 - (bounded / bound) ** 2


def gaussian_log_density(x: NDArray, means: NDArray, log_variances: NDArray, axis: int = -1) -> NDArray:
    """ log N(x | means, diag(exp(log_variances))), summed along axis """
    return -0.5 * (LOG_TWO_PI + log_variances + (x - means) ** 2 * np.exp(-log_variances)).sum(axis=axis)


def gaussian_kl(means: NDArray,
                log_variances: NDArray,
                alternate_means: NDArray,
                other_log_variances: NDArray,
                axis: int = -1) -> NDArray:
    """ KL(N(means, exp(log_variances)) || N(alternate_means, exp(other_log_variances))), summed over the last axis."""
    return 0.5 * (
            other_log_variances - log_variances
            + (np.exp(log_variances) + (means - alternate_means) ** 2) * np.exp(-other_log_variances) - 1.0
    ).sum(axis=axis)


HISTOGRAM_BINS = 60


def latent_grid(points: NDArray, num_points: int) -> tuple[NDArray, NDArray]:
    """
    Parameters
    ----------
    points : (N, 2) points the grid must cover
    num_points : grid resolution per axis

    Returns
    -------
    grid x and y coordinates, each (num_points, num_points), spanning the points with a 10% margin
    """
    lower, upper = points.min(axis=0), points.max(axis=0)
    margin = 0.1 * (upper - lower) + 1e-6
    return np.meshgrid(np.linspace(lower[0] - margin[0], upper[0] + margin[0], num_points),
                       np.linspace(lower[1] - margin[1], upper[1] + margin[1], num_points))


def gaussian_densities(grid_x: NDArray, grid_y: NDArray, means: NDArray, deviations: NDArray) -> NDArray:
    """
    Parameters
    ----------
    grid_x, grid_y : (P, P) grid coordinates
    means : (C, 2) centre of each diagonal Gaussian
    deviations : (C, 2) standard deviations of each

    Returns
    -------
    density of each Gaussian on the grid, (C, P, P)
    """
    standard_x = (grid_x[None] - means[:, 0, None, None]) / deviations[:, 0, None, None]
    standard_y = (grid_y[None] - means[:, 1, None, None]) / deviations[:, 1, None, None]
    scale = 2.0 * np.pi * deviations[:, 0, None, None] * deviations[:, 1, None, None]
    return np.exp(-0.5 * (standard_x ** 2 + standard_y ** 2)) / scale


def draw_densities(axis, grid_x: NDArray, grid_y: NDArray, densities: NDArray, centres: NDArray) -> None:
    """
    Fill the average of the densities in grey and draw each one's contours at exp(-2) and exp(-0.5) of its peak,
    which are 2 and 1 standard deviations for a Gaussian, coloured and numbered by index.

    Parameters
    ----------
    axis : matplotlib axis
    grid_x, grid_y : (P, P) grid coordinates
    densities : (C, P, P) density of each distribution on the grid
    centres : (C, 2) where to write each index
    """
    import matplotlib.pyplot as plt

    palette = plt.get_cmap("tab10")
    axis.contourf(grid_x, grid_y, densities.mean(axis=0), levels=20, cmap="Greys")
    for index, (density, centre) in enumerate(zip(densities, centres)):
        peak = density.max()
        if not peak > 0:
            continue
        color = palette(index % palette.N)
        axis.contour(grid_x, grid_y, density, levels=[peak * np.exp(-2.0), peak * np.exp(-0.5)],
                     colors=[color], linewidths=[1.0, 1.8])
        axis.annotate(str(index), centre, color=color, ha="center", va="center", fontweight="bold")


class VAELayer(Layer):
    """
    Variational autoencoder (Kingma & Welling 2014) with a standard normal prior, over rows of features.

    q(z|y) = N(mean, diag(exp(log_variance))) from a two-layer recognition network, p(z) = N(0, I), and mu(z) from a
    two-layer observation network. In training the latent is mean + noise_scale * std * eps; in eval it is the mean.
    kl_weight and noise_scale are plain attributes meant to be set from a Scheduler as training goes: a small
    noise_scale keeps the latent close to deterministic early, a small kl_weight lets it carry information first.

    forward caches `kl_divergence`, KL(q || p) averaged over valid rows. backward adds kl_weight times its gradient to
    the incoming one, which must be the gradient of the reconstruction loss, averaged over rows, with respect to mu(z).
    The KL is that of q itself; noise_scale only scales the draw.
    """
    parameter_names = (
        "recognition_first", "recognition_second", "mean_head", "log_variance_head",
        "observation_first", "observation_second", "observation_output",
    )
    cache_names = ("in_shape", "valid", "mean", "log_variance", "noise", "applied_noise_scale")

    def __init__(self,
                 input_dimension: int,
                 latent_dimension: int,
                 hidden_dimension: int = 120,
                 kl_weight: float = 1.0,
                 noise_scale: float = 1.0,
                 output_activation: str = "linear",
                 ):
        """
        Parameters
        ----------
        input_dimension : width of the observed y, and of its reconstruction
        latent_dimension : width of the latent z
        hidden_dimension : width of the two ReLU layers of the recognition and observation networks
        kl_weight : weight of the KL term in the loss; 0 turns the prior off
        noise_scale : multiplier on the sampled noise; 0 makes the latent deterministic
        output_activation : activation of mu(z); linear for a Gaussian observation, sigmoid for Bernoulli
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.latent_dimension = latent_dimension
        self.hidden_dimension = hidden_dimension
        self.kl_weight = kl_weight
        self.noise_scale = noise_scale
        self.output_activation = output_activation
        self.kl_divergence = 0.0

        self.recognition_first = FullyConnectedLayer(input_dimension, hidden_dimension, "relu")
        self.recognition_second = FullyConnectedLayer(hidden_dimension, hidden_dimension, "relu")
        self.mean_head = FullyConnectedLayer(hidden_dimension, latent_dimension, "linear")
        self.log_variance_head = FullyConnectedLayer(hidden_dimension, latent_dimension, "linear",
                                                     initialization_override="zeros")

        self.observation_first = FullyConnectedLayer(latent_dimension, hidden_dimension, "relu")
        self.observation_second = FullyConnectedLayer(hidden_dimension, hidden_dimension, "relu")
        self.observation_output = FullyConnectedLayer(hidden_dimension, input_dimension, output_activation,
                                                      is_output=True)

        self.declare_shapes(inputs=((input_dimension,),), outputs=((input_dimension,),))
        self.preserves_shape = True
        self.zero_gradients()

    def encode(self, y: NDArray) -> tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        y : (N, input_dimension)

        Returns
        -------
        mean and log variance of q(z|y), each (N, latent_dimension); log variances are bounded by soft_bound
        """
        hidden = self.recognition_second(self.recognition_first(y))
        return self.mean_head(hidden), soft_bound(self.log_variance_head(hidden), LOG_VARIANCE_BOUND)

    def decode(self, z: NDArray) -> NDArray:
        """
        Parameters
        ----------
        z : (..., latent_dimension)

        Returns
        -------
        mu(z) of the observation model, (..., input_dimension)
        """
        return self.observation_output(self.observation_second(self.observation_first(z)))

    def generate(self, num_samples: int) -> NDArray:
        """num_samples reconstructions decoded from z ~ N(0, I), (num_samples, input_dimension)"""
        return self.decode(RNG.standard_normal((num_samples, self.latent_dimension)))

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (..., input_dimension) observed y
        mask : matching incoming_x's leading axes, 1 for a real row and 0 for padding

        Returns
        -------
        mu(z), same shape as incoming_x; z is the mean in eval
        """
        in_shape = incoming_x.shape
        y = incoming_x.reshape(-1, self.input_dimension)

        mean, log_variance = self.encode(y)

        if not self.training:
            return self.decode(mean).reshape(in_shape)
        if mask is not None:
            assert mask.shape == in_shape[:-1], (
                f"mask shape {mask.shape} must match incoming_x's leading axes {in_shape[:-1]}"
            )
        valid = np.ones(len(y), dtype=y.dtype) if mask is None else mask.reshape(-1).astype(y.dtype)
        rows = valid / max(valid.sum(), 1.0)

        noise = RNG.standard_normal(mean.shape, dtype=y.dtype)

        latent = mean + self.noise_scale * np.exp(0.5 * log_variance) * noise
        self.kl_divergence = float((rows * 0.5 * (np.exp(log_variance) + mean ** 2 - 1.0 - log_variance)
                                    .sum(axis=-1)).sum())

        self.in_shape, self.valid = in_shape, valid
        self.mean, self.log_variance, self.noise = mean, log_variance, noise
        self.applied_noise_scale = self.noise_scale

        return self.decode(latent).reshape(in_shape)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        """
        Parameters
        ----------
        incoming_grad : gradient of the reconstruction loss, averaged over rows, with respect to mu(z)

        Returns
        -------
        gradient of the negative ELBO with respect to y
        """
        delta = incoming_grad.reshape(-1, self.input_dimension)
        latent_gradient = self.observation_first.backward(
            self.observation_second.backward(self.observation_output.backward(delta)))

        rows = (self.valid / max(self.valid.sum(), 1.0))[:, None] * self.kl_weight
        mean_gradient = latent_gradient + rows * self.mean
        log_variance_gradient = (
            latent_gradient * 0.5 * self.applied_noise_scale * np.exp(0.5 * self.log_variance) * self.noise
            + rows * 0.5 * (np.exp(self.log_variance) - 1.0)
        )
        hidden_gradient = (
            self.mean_head.backward(mean_gradient)
            + self.log_variance_head.backward(log_variance_gradient
                                              * soft_bound_derivative(self.log_variance, LOG_VARIANCE_BOUND))
        )
        gradient = self.recognition_first.backward(self.recognition_second.backward(hidden_gradient))
        return gradient.reshape(self.in_shape)


class GMVAELayer(Layer):
    """
    Gaussian mixture variational autoencoder - GMVAE by (Dilokthanakul et al. 2016).

    Generative model p(w) p(z) p(x|w, z) p(y|x):
        w ~ N(0, I)
        z ~ Mult(pi), pi_k = 1 / K
        x | z, w ~ prod_k N(mu_k(w), diag(sigma_k^2(w)))^z_k        prior network, beta
        y | x ~ N(mu(x), ...) or Bernoulli(mu(x))                     observation network, theta
    Recognition model q(x|y) q(w|y) p(z|x, w): q(x|y) and q(w|y) are Gaussians from one shared network, and the
    z-posterior p(z|x, w) follows from Bayes' rule (Eq. 3).

    forward returns mu(x) for a draw of x from q(x|y), or for its mean in eval, and caches the KL terms of the ELBO
    (Eq. 4), each averaged over valid rows:
        conditional_prior   E_q(w|y) p(z|x,w) [KL(q(x|y) || p(x|w, z))], over num_samples draws (Eq. 5)
        w_prior             KL(q(w|y) || p(w))
        z_prior             E_q(x|y) q(w|y) [KL(p(z|x, w) || p(z))], over the same draws

    backward adds their gradients to the incoming one, which must be the gradient of -log p(y|x) averaged over
    rows. The z-prior enters as max(information_constraint, z_prior) (Eq. 7), so its gradient is off below the
    constraint.
    """
    parameter_names = (
        "recognition_first", "recognition_second", "x_mean_head", "x_log_variance_head", "w_mean_head",
        "w_log_variance_head", "prior_hidden", "prior_output", "observation_first", "observation_second",
        "observation_output",
    )
    cache_names = (
        "in_shape", "valid", "x_mean", "x_log_variance", "w_mean", "w_log_variance", "x_noise", "w_noise",
        "x_samples", "component_means", "component_log_variances", "z_posterior_values", "log_z_posterior_values",
        "component_kl", "z_prior_active",
    )

    def __init__(self,
                 input_dimension: int,
                 latent_dimension: int,
                 w_dimension: int,
                 num_components: int,
                 hidden_dimension: int = 120,
                 prior_hidden_dimension: int = 120,
                 num_samples: int = 5,
                 information_constraint: float = 0.0,
                 output_activation: str = "linear",
                 ):
        """
        Parameters
        ----------
        input_dimension : width of the observed y
        latent_dimension : width of the continuous latent x, N_x
        w_dimension : width of the latent w that generates the component means and variances, also called `style`
        num_components : number of mixture components K
        hidden_dimension : width of the two ReLU layers of the recognition and observation networks
        prior_hidden_dimension : width of the tanh layer of the prior network
        num_samples : Monte Carlo draws M of x and w for the conditional prior and z-prior terms
        information_constraint : lambda of Eq. 7; 0 keeps the plain ELBO -- must stay below log(K)
        output_activation : activation of mu(x); linear for a Gaussian observation, sigmoid for Bernoulli
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.latent_dimension = latent_dimension
        self.w_dimension = w_dimension
        self.num_components = num_components
        self.hidden_dimension = hidden_dimension
        self.prior_hidden_dimension = prior_hidden_dimension
        self.num_samples = num_samples
        self.information_constraint = information_constraint
        self.output_activation = output_activation
        self.conditional_prior = 0.0
        self.w_prior = 0.0
        self.z_prior = 0.0

        self.recognition_first = FullyConnectedLayer(input_dimension, hidden_dimension, "relu")
        self.recognition_second = FullyConnectedLayer(hidden_dimension, hidden_dimension, "relu")

        self.x_mean_head = FullyConnectedLayer(hidden_dimension, latent_dimension, "linear")
        self.x_log_variance_head = FullyConnectedLayer(hidden_dimension, latent_dimension, "linear",
                                                       initialization_override="zeros")

        self.w_mean_head = FullyConnectedLayer(hidden_dimension, w_dimension, "linear")
        self.w_log_variance_head = FullyConnectedLayer(hidden_dimension, w_dimension, "linear",
                                                       initialization_override="zeros")

        self.prior_hidden = FullyConnectedLayer(w_dimension, prior_hidden_dimension, "tanh")
        self.prior_output = FullyConnectedLayer(prior_hidden_dimension,
                                                2 * num_components * latent_dimension,
                                                initialization_override="orthogonal",
                                                activation_type="linear")

        self.observation_first = FullyConnectedLayer(latent_dimension, hidden_dimension, "relu")
        self.observation_second = FullyConnectedLayer(hidden_dimension, hidden_dimension, "relu")
        self.observation_output = FullyConnectedLayer(hidden_dimension, input_dimension, output_activation,
                                                      is_output=True)

        self.declare_shapes(inputs=((input_dimension,),), outputs=((input_dimension,),))
        self.preserves_shape = True
        self.zero_gradients()

    def recognize(self, y: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Parameters
        ----------
        y : (N, input_dimension)

        Returns
        -------
        mean and log variance of q(x|y), each (N, N_x), and of q(w|y), each (N, N_w); log variances are bounded by
        soft_bound
        """
        hidden = self.recognition_second(self.recognition_first(y))
        return (self.x_mean_head(hidden), soft_bound(self.x_log_variance_head(hidden), LOG_VARIANCE_BOUND),
                self.w_mean_head(hidden), soft_bound(self.w_log_variance_head(hidden), LOG_VARIANCE_BOUND))

    def component_parameters(self, w: NDArray) -> tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        w : (N, N_w)

        Returns
        -------
        means mu_k(w) and log variances log sigma_k^2(w) of p(x|w, z), each (N, K, N_x); log variances are bounded by
        soft_bound
        """
        outputs = self.prior_output(self.prior_hidden(w)).reshape(len(w), 2, self.num_components, self.latent_dimension)
        return outputs[:, 0], soft_bound(outputs[:, 1], LOG_VARIANCE_BOUND)

    def z_posterior(self, x: NDArray, means: NDArray, log_variances: NDArray) -> tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        x : (..., N_x)
        means : (..., K, N_x) component means for each x
        log_variances : (..., K, N_x) component log variances for each x

        Returns
        -------
        p(z_k = 1|x, w) (Eq. 3) and its log, each (..., K)
        """
        log_joint = gaussian_log_density(x[..., None, :], means, log_variances) - np.log(self.num_components)
        log_posterior = log_joint - log_sum_exp(log_joint)[..., None]
        return np.exp(log_posterior), log_posterior

    def decode(self, x: NDArray) -> NDArray:
        """
        Parameters
        ----------
        x : (..., N_x)

        Returns
        -------
        mu(x) of the observation model, (..., input_dimension)
        """
        return self.observation_output(self.observation_second(self.observation_first(x)))

    def cluster(self, y: NDArray) -> NDArray:
        """
        Parameters
        ----------
        y : (N, input_dimension); call in eval mode

        Returns
        -------
        p(z|x, w) at the means of q(x|y) and q(w|y), (N, K)
        """
        x_mean, _, w_mean, _ = self.recognize(y)
        means, log_variances = self.component_parameters(w_mean)
        return self.z_posterior(x_mean, means, log_variances)[0]

    def generate(self,
                 num_samples: int,
                 components: Optional[NDArray] = None,
                 w: Optional[NDArray] = None,
                 ) -> NDArray:
        """
        Parameters
        ----------
        num_samples : number of outputs to draw
        components : component of each draw, (num_samples,) or one int; drawn from pi by default
        w : value of w, (num_samples, N_w) or (N_w,); drawn from N(0, I) by default

        Returns
        -------
        mu(x) for x drawn from the generative model, (num_samples, input_dimension)
        """
        if w is None:
            w = RNG.standard_normal((num_samples, self.w_dimension))
        w = np.broadcast_to(w, (num_samples, self.w_dimension))
        if components is None:
            components = RNG.integers(0, self.num_components, num_samples)
        components = np.broadcast_to(components, (num_samples,))
        means, log_variances = self.component_parameters(w)
        rows = np.arange(num_samples)
        x = means[rows, components] + np.exp(0.5 * log_variances[rows, components]) * RNG.standard_normal(
            (num_samples, self.latent_dimension))
        return self.decode(x)

    def plot_latent(self,
                    y: Optional[NDArray] = None,
                    labels: Optional[NDArray] = None,
                    w: Optional[NDArray] = None,
                    dimensions: Sequence[int] = (0, 1),
                    num_points: int = 200,
                    axis=None):
        """
        Plot each component of p(x|w, z) over two latent dimensions: contours at 1 and 2 standard deviations over
        the mixture's density, with the means of q(x|y) when y is given. Call in eval mode.

        Parameters
        ----------
        y : (N, input_dimension) data to place in the latent space
        labels : (N,) integer colour of each point of y; their components by default
        w : (N_w,) value of w that sets the components; 0 by default
        dimensions : the two latent dimensions to draw
        num_points : grid resolution per axis
        axis : matplotlib axis to draw on; a new figure by default

        Returns
        -------
        the matplotlib Figure
        """
        import matplotlib.pyplot as plt

        dimensions = list(dimensions)
        w = np.zeros((1, self.w_dimension)) if w is None else np.asarray(w, dtype=float).reshape(1, self.w_dimension)
        means, log_variances = self.component_parameters(w)
        means, deviations = means[0][:, dimensions], np.exp(0.5 * log_variances[0][:, dimensions])
        cover = [means - 3.0 * deviations, means + 3.0 * deviations]
        points = None
        if y is not None:
            points = self.recognize(y)[0][:, dimensions]
            labels = self.cluster(y).argmax(axis=1) if labels is None else labels
            cover.append(points)
        grid_x, grid_y = latent_grid(np.concatenate(cover), num_points)

        if axis is None:
            axis = plt.subplots(figsize=(6, 6))[1]
        draw_densities(axis, grid_x, grid_y, gaussian_densities(grid_x, grid_y, means, deviations), means)
        if points is not None:
            axis.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", vmin=0, vmax=9, s=3, alpha=0.4)
        axis.set(title=f"p(x|w, z) per component, w = {np.round(w[0], 2).tolist()}",
                 xlabel=f"x[{dimensions[0]}]", ylabel=f"x[{dimensions[1]}]")
        return axis.figure

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (..., input_dimension) observed y
        mask : matching incoming_x's leading axes, 1 for a real row and 0 for padding

        Returns
        -------
        mu(x), same shape as incoming_x
        """
        in_shape = incoming_x.shape
        y = incoming_x.reshape(-1, self.input_dimension)
        x_mean, x_log_variance, w_mean, w_log_variance = self.recognize(y)
        if not self.training:
            return self.decode(x_mean).reshape(in_shape)
        if mask is not None:
            assert mask.shape == in_shape[:-1], (
                f"mask shape {mask.shape} must match incoming_x's leading axes {in_shape[:-1]}"
            )
        valid = np.ones(len(y), dtype=y.dtype) if mask is None else mask.reshape(-1).astype(y.dtype)
        rows = valid / max(valid.sum(), 1.0)

        x_noise = RNG.standard_normal((self.num_samples, len(y), self.latent_dimension), dtype=y.dtype)
        w_noise = RNG.standard_normal((self.num_samples, len(y), self.w_dimension), dtype=y.dtype)

        x_samples = x_mean + np.exp(0.5 * x_log_variance) * x_noise
        w_samples = w_mean + np.exp(0.5 * w_log_variance) * w_noise

        means, log_variances = self.component_parameters(w_samples.reshape(-1, self.w_dimension))
        shape = (self.num_samples, len(y), self.num_components, self.latent_dimension)

        means, log_variances = means.reshape(shape), log_variances.reshape(shape)

        posterior, log_posterior = self.z_posterior(x_samples, means, log_variances)

        component_kl = gaussian_kl(x_mean[None, :, None, :], x_log_variance[None, :, None, :], means, log_variances)

        self.conditional_prior = float((rows * (posterior * component_kl).sum(axis=-1).mean(axis=0)).sum())
        self.z_prior = float((rows * (posterior * (log_posterior + np.log(self.num_components))).sum(axis=-1)
                              .mean(axis=0)).sum())
        self.w_prior = float((rows * 0.5 * (np.exp(w_log_variance) + w_mean ** 2 - 1.0 - w_log_variance)
                              .sum(axis=-1)).sum())
        self.z_prior_active = self.z_prior >= self.information_constraint

        self.in_shape = in_shape
        self.valid = valid
        self.x_mean, self.x_log_variance = x_mean, x_log_variance
        self.w_mean, self.w_log_variance = w_mean, w_log_variance
        self.x_noise, self.w_noise = x_noise, w_noise
        self.x_samples = x_samples
        self.component_means, self.component_log_variances = means, log_variances
        self.z_posterior_values, self.log_z_posterior_values = posterior, log_posterior
        self.component_kl = component_kl
        return self.decode(x_samples[0]).reshape(in_shape)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        """
        Parameters
        ----------
        incoming_grad : gradient of -log p(y|x), averaged over rows, with respect to mu(x)

        Returns
        -------
        gradient of the negative ELBO with respect to y
        """
        delta = incoming_grad.reshape(-1, self.input_dimension)
        reconstruction_gradient = self.observation_first.backward(
            self.observation_second.backward(self.observation_output.backward(delta)))

        rows = (self.valid / max(self.valid.sum(), 1.0))[None, :, None] / self.num_samples
        posterior, log_posterior = self.z_posterior_values, self.log_z_posterior_values
        inverse_variance = np.exp(-self.component_log_variances)
        sample_difference = self.x_samples[:, :, None, :] - self.component_means
        mean_difference = self.x_mean[None, :, None, :] - self.component_means
        z_weight = 1.0 if self.z_prior_active else 0.0

        pressure = rows * (self.component_kl + z_weight * (log_posterior + np.log(self.num_components)))
        spread = posterior * (pressure - (posterior * pressure).sum(axis=-1, keepdims=True))
        direct = rows * posterior
        density_pull = spread[..., None] * sample_difference * inverse_variance
        kl_pull = direct[..., None] * mean_difference * inverse_variance

        sample_gradient = -density_pull.sum(axis=2)
        sample_gradient[0] += reconstruction_gradient
        means_gradient = density_pull - kl_pull
        log_variances_gradient = (
            spread[..., None] * 0.5 * (sample_difference ** 2 * inverse_variance - 1.0)
            + direct[..., None] * 0.5 * (1.0 - (np.exp(self.x_log_variance)[None, :, None, :] + mean_difference ** 2)
                                         * inverse_variance)
        )
        x_mean_gradient = sample_gradient.sum(axis=0) + kl_pull.sum(axis=(0, 2))
        x_log_variance_gradient = (
            (sample_gradient * 0.5 * np.exp(0.5 * self.x_log_variance) * self.x_noise).sum(axis=0)
            + (direct[..., None] * 0.5 * (np.exp(self.x_log_variance)[None, :, None, :] * inverse_variance - 1.0))
            .sum(axis=(0, 2))
        )

        log_variances_gradient = log_variances_gradient * soft_bound_derivative(self.component_log_variances,
                                                                                LOG_VARIANCE_BOUND)
        prior_gradient = np.stack([means_gradient, log_variances_gradient], axis=2)
        w_gradient = self.prior_hidden.backward(self.prior_output.backward(
            prior_gradient.reshape(-1, 2 * self.num_components * self.latent_dimension)
        )).reshape(self.num_samples, -1, self.w_dimension)
        row_weights = rows[0] * self.num_samples
        w_mean_gradient = w_gradient.sum(axis=0) + row_weights * self.w_mean
        w_log_variance_gradient = (
            (w_gradient * 0.5 * np.exp(0.5 * self.w_log_variance) * self.w_noise).sum(axis=0)
            + row_weights * 0.5 * (np.exp(self.w_log_variance) - 1.0)
        )

        hidden_gradient = (
            self.x_mean_head.backward(x_mean_gradient)
            + self.x_log_variance_head.backward(x_log_variance_gradient
                                                * soft_bound_derivative(self.x_log_variance, LOG_VARIANCE_BOUND))
            + self.w_mean_head.backward(w_mean_gradient)
            + self.w_log_variance_head.backward(w_log_variance_gradient
                                                * soft_bound_derivative(self.w_log_variance, LOG_VARIANCE_BOUND))
        )
        gradient = self.recognition_first.backward(self.recognition_second.backward(hidden_gradient))
        return gradient.reshape(self.in_shape)


class AffineCouplingLayer(Layer):
    """
    Real NVP affine coupling (Dinh et al. 2017)

    Affine transform is applied via shift functions to half of the data; the other half is num_kept.
        g_kept = z_kept
        g_transformed = z_changed * exp(s(z_kept)) + t(z_kept)

    The output layers of s and t start at zero, so the coupling starts as the identity
    """
    parameter_names = ("scale_first", "scale_second", "scale_output",
                       "translation_first", "translation_second", "translation_output")
    cache_names = ("in_shape", "changed_input", "scale")

    def __init__(self,
                 input_dimension: int,
                 hidden_dimension: int = 64,
                 first_half: bool = False,
                 activation_type: str = "tanh"):
        """
        Parameters
        ----------
        input_dimension : width of the vector, at least 2
        hidden_dimension : width of the hidden layers of s and t
        parity : 0 keeps the first half, 1 keeps the second
        activation_type : activation of the hidden layers of s and t
        """
        super().__init__()
        assert input_dimension >= 2, "a coupling needs at least two dimensions"
        self.dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.first_half = first_half
        self.activation_type = activation_type
        self.log_determinant = None

        halves = np.arange(input_dimension) < input_dimension // 2
        self.num_kept = np.flatnonzero(halves if first_half else ~halves)
        self.changed = np.flatnonzero(~halves if first_half else halves)
        kept_width, changed_width = len(self.num_kept), len(self.changed)

        self.scale_first = FullyConnectedLayer(kept_width, hidden_dimension, activation_type)
        self.translation_first = FullyConnectedLayer(kept_width, hidden_dimension, activation_type)

        self.scale_second = FullyConnectedLayer(hidden_dimension, hidden_dimension, activation_type)
        self.translation_second = FullyConnectedLayer(hidden_dimension, hidden_dimension, activation_type)

        self.scale_output = FullyConnectedLayer(hidden_dimension, changed_width, "linear",
                                                initialization_override="zeros")
        self.translation_output = FullyConnectedLayer(hidden_dimension, changed_width, "linear",
                                                      initialization_override="zeros")

        self.declare_shapes(inputs=((input_dimension,),), outputs=((input_dimension,),))
        self.preserves_shape = True

        self.zero_gradients()

    def scale_and_translation(self, kept_input: NDArray) -> tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        kept_input : (N, num_kept width)

        Returns
        -------
        s, bounded by soft_bound, and t of the num_kept half, each (N, changed width)
        """
        scale = soft_bound(self.scale_output(self.scale_second(self.scale_first(kept_input))), LOG_SCALE_BOUND)
        translation = self.translation_output(self.translation_second(self.translation_first(kept_input)))
        return scale, translation

    def forward(self, incoming_x: NDArray) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (..., input_dimension) z

        Returns
        -------
        g = f(z), same shape
        """
        in_shape = incoming_x.shape
        z = incoming_x.reshape(-1, self.dimension)
        scale, translation = self.scale_and_translation(z[:, self.num_kept])
        output = z.copy()
        output[:, self.changed] = z[:, self.changed] * np.exp(scale) + translation
        self.log_determinant = scale.sum(axis=-1)

        if self.training:
            self.in_shape = in_shape
            self.changed_input = z[:, self.changed]
            self.scale = scale

        return output.reshape(in_shape)

    def inverse(self, outgoing_x: NDArray) -> NDArray:
        """
        Parameters
        ----------
        outgoing_x : (..., input_dimension) g

        Returns
        -------
        z = f^-1(g) -- same shape
        """
        g = outgoing_x.reshape(-1, self.dimension)
        scale, translation = self.scale_and_translation(g[:, self.num_kept])

        z = g.copy()
        z[:, self.changed] = (g[:, self.changed] - translation) * np.exp(-scale)

        return z.reshape(outgoing_x.shape)

    def backward(self, incoming_grad: NDArray, log_determinant_grad: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_grad : gradient wrt g
        log_determinant_grad : gradient wrt each row's log det (N,)

        Returns
        -------
        gradient; wrt z
        """
        delta = incoming_grad.reshape(-1, self.dimension)
        changed_delta = delta[:, self.changed]
        stretch = np.exp(self.scale)
        scale_grad = changed_delta * self.changed_input * stretch
        if log_determinant_grad is not None:
            scale_grad = scale_grad + log_determinant_grad[:, None]
        scale_grad = scale_grad * soft_bound_derivative(self.scale, LOG_SCALE_BOUND)
        kept_grad = (
            delta[:, self.num_kept]
            + self.scale_first.backward(self.scale_second.backward(self.scale_output.backward(scale_grad)))
            + self.translation_first.backward(self.translation_second.backward(
                self.translation_output.backward(changed_delta)))
        )
        gradient = np.empty_like(delta)
        gradient[:, self.num_kept] = kept_grad
        gradient[:, self.changed] = changed_delta * stretch
        return gradient.reshape(self.in_shape)


class NVPVAELayer(Layer):
    """
    Conditional sigma-VAE with a NVP (real) conditional prior (De 2025)

    With x the input, y the labels and z the latent:
        encoder q        q(z|x, y) = N(mu_q, sigma_q^2)
        encoder p        mu_p(y), sigma_p(y)
        flow f           a stack of affine couplings
        prior            p(z|y) = N(f(z); mu_p, sigma_p^2) |det df/dz|
        decoder          x_hat(z, y)
    forward returns x_hat for a draw of z from q (mu_q in eval) and caches the KL term averaged over valid
        rows, from that draw:
            kl_divergence = sum log sigma_p - log sigma_q + (f(z) - mu_p)^2 / (2 sigma_p^2) - 1/2 - log |det df/dz|

    backward adds its gradient to the incoming one, -log p(x|z, y) (averaged over batch/rows);
    reconstruction_loss gives that for the paper's decoder with the optimal variance. Without labels (label_dimension 0)
        encoder p sees a constant, so mu_p and sigma_p are learned constants.
    """
    cache_names = ("in_shape", "has_labels", "valid", "conditioning", "posterior_log_scale", "noise", "latent",
                   "flowed", "prior_mean", "prior_log_scale")

    def __init__(self,
                 input_dimension: int,
                 latent_dimension: int,
                 label_dimension: int = 0,
                 hidden_dimension: int = 128,
                 flow_hidden_dimension: int = 64,
                 num_couplings: int = 4,
                 activation_type: str = "relu",
                 flow_activation_type: str = "tanh",
                 ):
        """
        Parameters
        ----------
        input_dimension : width of x
        latent_dimension : width of z, at least 2
        label_dimension : width of the labels y; 0 for an unconditional model
        hidden_dimension : width of the two hidden layers of encoder q, encoder p and the decoder
        flow_hidden_dimension : width of the two hidden layers of each coupling's s and t
        num_couplings : number of affine couplings in f, alternating which half they keep
        activation_type : activation of the hidden layers of encoder q, encoder p and the decoder
        flow_activation_type : activation of the hidden layers of s and t; a bounded one keeps exp(s) in range
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.latent_dimension = latent_dimension
        self.label_dimension = label_dimension
        self.hidden_dimension = hidden_dimension
        self.flow_hidden_dimension = flow_hidden_dimension
        self.num_couplings = num_couplings
        self.activation_type = activation_type
        self.flow_activation_type = flow_activation_type

        self.kl_divergence = 0.0
        self.observation_variance = 1.0

        prior_width = max(label_dimension, 1)
        self.posterior_first = FullyConnectedLayer(input_dimension + label_dimension, hidden_dimension, activation_type)
        self.posterior_second = FullyConnectedLayer(hidden_dimension, hidden_dimension, activation_type)
        self.posterior_mean = FullyConnectedLayer(hidden_dimension, latent_dimension, "linear")
        self.posterior_log_scale_head = FullyConnectedLayer(hidden_dimension, latent_dimension, "linear",
                                                            initialization_override="zeros")
        self.prior_first = FullyConnectedLayer(prior_width, hidden_dimension, activation_type)
        self.prior_second = FullyConnectedLayer(hidden_dimension, hidden_dimension, activation_type)
        self.prior_mean_head = FullyConnectedLayer(hidden_dimension, latent_dimension, "linear")
        self.prior_log_scale_head = FullyConnectedLayer(hidden_dimension, latent_dimension, "linear",
                                                        initialization_override="zeros")
        coupling_names = tuple(f"coupling_{n}" for n in range(num_couplings))
        for n, name in enumerate(coupling_names):
            setattr(self, name, AffineCouplingLayer(latent_dimension, flow_hidden_dimension, n % 2,
                                                   flow_activation_type))
        self.decoder_first = FullyConnectedLayer(latent_dimension + label_dimension, hidden_dimension, activation_type)
        self.decoder_second = FullyConnectedLayer(hidden_dimension, hidden_dimension, activation_type)
        self.decoder_output = FullyConnectedLayer(hidden_dimension, input_dimension, "linear", is_output=True)
        self.parameter_names = (
            ("posterior_first", "posterior_second", "posterior_mean", "posterior_log_scale_head",
             "prior_first", "prior_second", "prior_mean_head", "prior_log_scale_head")
            + coupling_names
            + ("decoder_first", "decoder_second", "decoder_output")
        )

        self.declare_shapes(inputs=((input_dimension,),), outputs=((input_dimension,),))
        self.preserves_shape = True
        self.zero_gradients()

    @property
    def couplings(self) -> list[AffineCouplingLayer]:
        return [getattr(self, f"coupling_{n}") for n in range(self.num_couplings)]

    def read_labels(self, labels: Optional[NDArray], num_rows: int, dtype) -> NDArray:
        """
        Parameters
        ----------
        labels : (..., label_dimension) or None
        num_rows : rows of the flattened input
        dtype : type of the flattened input

        Returns
        -------
        labels as (num_rows, label_dimension); (num_rows, 0) without labels
        """
        if labels is None:
            assert self.label_dimension == 0, f"expected labels of width {self.label_dimension}"
            return np.zeros((num_rows, 0), dtype=dtype)
        assert labels.shape[-1] == self.label_dimension, (
            f"labels width {labels.shape[-1]} is not label_dimension {self.label_dimension}"
        )
        return labels.reshape(num_rows, self.label_dimension).astype(dtype, copy=False)

    def prior_parameters(self, conditioning: NDArray) -> tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        conditioning : (N, label_dimension) labels

        Returns
        -------
        mu_p and log sigma_p from encoder p, each (N, latent_dimension); log sigma_p is bounded by soft_bound
        """
        if self.label_dimension == 0:
            conditioning = np.ones((len(conditioning), 1), dtype=conditioning.dtype)
        hidden = self.prior_second(self.prior_first(conditioning))
        return self.prior_mean_head(hidden), soft_bound(self.prior_log_scale_head(hidden), LOG_SCALE_BOUND)

    def encode(self, x: NDArray, conditioning: NDArray) -> tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        x : (N, input_dimension)
        conditioning : (N, label_dimension) labels

        Returns
        -------
        mu_q and log sigma_q from encoder q, each (N, latent_dimension); log sigma_q is bounded by soft_bound
        """
        hidden = self.posterior_second(self.posterior_first(np.concatenate([x, conditioning], axis=-1)))
        return self.posterior_mean(hidden), soft_bound(self.posterior_log_scale_head(hidden), LOG_SCALE_BOUND)

    def decode(self, z: NDArray, conditioning: NDArray) -> NDArray:
        """
        Parameters
        ----------
        z : (N, latent_dimension)
        conditioning : (N, label_dimension) labels

        Returns
        -------
        x_hat, (N, input_dimension)
        """
        return self.decoder_output(self.decoder_second(self.decoder_first(np.concatenate([z, conditioning], axis=-1))))

    def flow(self, z: NDArray) -> tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        z : (N, latent_dimension)

        Returns
        -------
        f(z), and log |det df/dz| per row, (N,)
        """
        log_determinant = np.zeros(len(z), dtype=z.dtype)
        for coupling in self.couplings:
            z = coupling(z)
            log_determinant = log_determinant + coupling.log_determinant
        return z, log_determinant

    def inverse_flow(self, g: NDArray) -> NDArray:
        """
        Parameters
        ----------
        g : (N, latent_dimension)

        Returns
        -------
        f^-1(g), (N, latent_dimension); call in eval mode
        """
        for coupling in reversed(self.couplings):
            g = coupling.inverse(g)
        return g

    def forward(self, incoming_x: NDArray, labels: Optional[NDArray] = None, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (..., input_dimension) x
        labels : (..., label_dimension) y, matching incoming_x's leading axes; None when label_dimension is 0
        mask : matching incoming_x's leading axes, 1 for a real row and 0 for padding

        Returns
        -------
        x_hat, same shape as incoming_x
        """
        in_shape = incoming_x.shape
        x = incoming_x.reshape(-1, self.input_dimension)
        conditioning = self.read_labels(labels, len(x), x.dtype)
        posterior_mean, posterior_log_scale = self.encode(x, conditioning)
        if not self.training:
            return self.decode(posterior_mean, conditioning).reshape(in_shape)
        if mask is not None:
            assert mask.shape == in_shape[:-1], (
                f"mask shape {mask.shape} must match incoming_x's leading axes {in_shape[:-1]}"
            )
        valid = np.ones(len(x), dtype=x.dtype) if mask is None else mask.reshape(-1).astype(x.dtype)

        noise = RNG.standard_normal(posterior_mean.shape, dtype=x.dtype)
        latent = posterior_mean + np.exp(posterior_log_scale) * noise
        prior_mean, prior_log_scale = self.prior_parameters(conditioning)
        flowed, log_determinant = self.flow(latent)
        kl = (
            prior_log_scale - posterior_log_scale + 0.5 * (flowed - prior_mean) ** 2 * np.exp(-2.0 * prior_log_scale)
            - 0.5
        ).sum(axis=-1) - log_determinant
        self.kl_divergence = float((valid * kl).sum() / max(valid.sum(), 1.0))

        self.in_shape = in_shape
        self.has_labels = labels is not None
        self.valid = valid
        self.conditioning = conditioning
        self.posterior_log_scale = posterior_log_scale
        self.noise = noise
        self.latent = latent
        self.flowed = flowed
        self.prior_mean = prior_mean
        self.prior_log_scale = prior_log_scale
        return self.decode(latent, conditioning).reshape(in_shape)

    def backward(self, incoming_grad: NDArray) -> NDArray | tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        incoming_grad : gradient of -log p(x|z, y), averaged over rows, with respect to x_hat

        Returns
        -------
        gradient of the negative ELBO with respect to x, and with respect to the labels when they were given
        """
        delta = incoming_grad.reshape(-1, self.input_dimension)
        decoder_grad = self.decoder_first.backward(self.decoder_second.backward(self.decoder_output.backward(delta)))
        latent_grad = decoder_grad[:, :self.latent_dimension]
        label_grad = decoder_grad[:, self.latent_dimension:]

        rows = (self.valid / max(self.valid.sum(), 1.0))[:, None]
        inverse_variance = np.exp(-2.0 * self.prior_log_scale)
        offset = self.flowed - self.prior_mean
        flowed_grad = rows * offset * inverse_variance
        log_determinant_grad = -rows[:, 0]
        for coupling in reversed(self.couplings):
            flowed_grad = coupling.backward(flowed_grad, log_determinant_grad)
        latent_grad = latent_grad + flowed_grad

        prior_hidden_grad = (
            self.prior_mean_head.backward(-rows * offset * inverse_variance)
            + self.prior_log_scale_head.backward(rows * (1.0 - offset ** 2 * inverse_variance)
                                                 * soft_bound_derivative(self.prior_log_scale, LOG_SCALE_BOUND))
        )
        prior_input_grad = self.prior_first.backward(self.prior_second.backward(prior_hidden_grad))

        posterior_hidden_grad = (
            self.posterior_mean.backward(latent_grad)
            + self.posterior_log_scale_head.backward(
                (latent_grad * np.exp(self.posterior_log_scale) * self.noise - rows)
                * soft_bound_derivative(self.posterior_log_scale, LOG_SCALE_BOUND))
        )
        posterior_input_grad = self.posterior_first.backward(self.posterior_second.backward(posterior_hidden_grad))
        gradient = posterior_input_grad[:, :self.input_dimension].reshape(self.in_shape)
        if not self.has_labels:
            return gradient
        label_grad = label_grad + posterior_input_grad[:, self.input_dimension:] + prior_input_grad
        return gradient, label_grad.reshape(*self.in_shape[:-1], self.label_dimension)

    def reconstruction_loss(self,
                            reconstruction: NDArray,
                            target: NDArray,
                            mask: Optional[NDArray] = None,
                            ) -> tuple[float, NDArray]:
        """
        -log p(x|z, y) of a Gaussian decoder at its optimal variance, the batch's MSE

        Parameters
        ----------
        reconstruction : x_hat, (..., input_dimension)
        target : x, same shape
        mask : matching the leading axes, 1 for a real row and 0 for padding

        Returns
        -------
        the loss averaged over valid rows, and its gradient with respect to reconstruction, to pass to backward;
        the variance used is left in observation_variance
        """
        difference = (reconstruction - target).reshape(-1, self.input_dimension)
        valid = np.ones(len(difference)) if mask is None else mask.reshape(-1).astype(difference.dtype)
        count = max(valid.sum(), 1.0)
        variance = max(float((valid[:, None] * difference ** 2).sum() / (count * self.input_dimension)), VARIANCE_FLOOR)
        self.observation_variance = variance
        loss = 0.5 * self.input_dimension * (np.log(2.0 * np.pi * variance) + 1.0)
        gradient = valid[:, None] * difference / (count * variance)
        return float(loss), gradient.reshape(reconstruction.shape)

    def generate(self, num_samples: int, labels: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        num_samples : number of outputs to draw
        labels : (num_samples, label_dimension) or (label_dimension,); None when label_dimension is 0

        Returns
        -------
        x_hat for z = f^-1(u), u ~ N(mu_p(y), sigma_p(y)^2), (num_samples, input_dimension); call in eval mode
        """
        if labels is not None:
            labels = np.broadcast_to(labels, (num_samples, self.label_dimension))
        conditioning = self.read_labels(labels, num_samples, np.float64)
        prior_mean, prior_log_scale = self.prior_parameters(conditioning)
        flowed = prior_mean + np.exp(prior_log_scale) * RNG.standard_normal(prior_mean.shape)
        return self.decode(self.inverse_flow(flowed), conditioning)

    def plot_latent(self,
                    x: Optional[NDArray] = None,
                    labels: Optional[NDArray] = None,
                    conditions: Optional[NDArray] = None,
                    dimensions: Sequence[int] = (0, 1),
                    num_points: int = 200,
                    num_samples: int = 20000,
                    axes=None):
        """
        Plot the prior p(z|y) of each condition over two latent dimensions, in the flowed space, where it is the
        Gaussian N(mu_p(y), sigma_p(y)^2), and in z, where it is that Gaussian carried back through f^-1: exact when
        latent_dimension is 2, a histogram of num_samples draws otherwise. Contours at 1 and 2 standard deviations,
        or the same fractions of the peak in z, over the conditions' average density, with f(mu_q) and mu_q of x when
        given. Call in eval mode.

        Parameters
        ----------
        x : (N, input_dimension) data to place in the latent space
        labels : (N, label_dimension) labels of x; needed with x when label_dimension > 0
        conditions : (C, label_dimension) labels whose priors are drawn; each one-hot label by default, the one
            learned prior without labels
        dimensions : the two latent dimensions to draw
        num_points : grid resolution per axis
        num_samples : draws per condition, for the extent of z and its histogram
        axes : two matplotlib axes, flowed space then z; a new figure by default

        Returns
        -------
        the matplotlib Figure
        """
        import matplotlib.pyplot as plt

        dimensions = list(dimensions)
        if conditions is None:
            conditions = np.eye(self.label_dimension) if self.label_dimension else np.zeros((1, 0))
        conditions = np.atleast_2d(np.asarray(conditions, dtype=float))
        prior_means, prior_log_scales = self.prior_parameters(conditions)
        draws = np.stack([
            self.inverse_flow(mean + np.exp(log_scale) * RNG.standard_normal((num_samples, self.latent_dimension)))
            for mean, log_scale in zip(prior_means, prior_log_scales)
        ])[:, :, dimensions]
        latent_points = flowed_points = colours = None
        if x is not None:
            conditioning = self.read_labels(labels, len(x), x.dtype)
            latent_points = self.encode(x, conditioning)[0]
            flowed_points = self.flow(latent_points)[0][:, dimensions]
            latent_points = latent_points[:, dimensions]
            colours = conditioning.argmax(axis=1) if self.label_dimension else np.zeros(len(x), dtype=int)

        if axes is None:
            axes = plt.subplots(1, 2, figsize=(12, 6))[1]
        flowed_axis, latent_axis = axes

        means, deviations = prior_means[:, dimensions], np.exp(prior_log_scales[:, dimensions])
        cover = [means - 3.0 * deviations, means + 3.0 * deviations]
        cover += [] if flowed_points is None else [flowed_points]
        grid_x, grid_y = latent_grid(np.concatenate(cover), num_points)
        draw_densities(flowed_axis, grid_x, grid_y, gaussian_densities(grid_x, grid_y, means, deviations), means)

        cover = [np.quantile(draws.reshape(-1, 2), [0.005, 0.995], axis=0)]
        cover += [] if latent_points is None else [latent_points]
        if self.latent_dimension == 2:
            grid_x, grid_y = latent_grid(np.concatenate(cover), num_points)
            z = np.zeros((grid_x.size, 2))
            z[:, dimensions[0]], z[:, dimensions[1]] = grid_x.ravel(), grid_y.ravel()
            flowed, log_determinant = self.flow(z)
            densities = np.stack([
                np.exp(gaussian_log_density(flowed, mean, 2.0 * log_scale) + log_determinant).reshape(grid_x.shape)
                for mean, log_scale in zip(prior_means, prior_log_scales)
            ])
        else:
            grid_x, grid_y = latent_grid(np.concatenate(cover), HISTOGRAM_BINS + 1)
            edges_x, edges_y = grid_x[0], grid_y[:, 0]
            densities = np.stack([
                np.histogram2d(draw[:, 0], draw[:, 1], bins=[edges_x, edges_y], density=True)[0].T for draw in draws
            ])
            grid_x, grid_y = np.meshgrid(0.5 * (edges_x[1:] + edges_x[:-1]), 0.5 * (edges_y[1:] + edges_y[:-1]))
        draw_densities(latent_axis, grid_x, grid_y, densities, np.median(draws, axis=1))

        for axis, points in ((flowed_axis, flowed_points), (latent_axis, latent_points)):
            if points is not None:
                axis.scatter(points[:, 0], points[:, 1], c=colours, cmap="tab10", vmin=0, vmax=9, s=3, alpha=0.4)
        flowed_axis.set(title="flowed f(z): N(mu_p(y), sigma_p(y)^2) per condition",
                        xlabel=f"f(z)[{dimensions[0]}]", ylabel=f"f(z)[{dimensions[1]}]")
        latent_axis.set(title="latent z: p(z|y) per condition",
                        xlabel=f"z[{dimensions[0]}]", ylabel=f"z[{dimensions[1]}]")
        return flowed_axis.figure


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from polyergalio.generators.data_generators import make_clusters, make_pinwheel
    from polyergalio.models.optimizers import Adam

    SEED = 0
    NUM_POINTS = 10000
    STEPS = 6000
    BATCH = 128
    RATE = 1e-3
    READOUT_EVERY = 100
    LATENT_DIMENSION = 2
    GMVAE_W_DIMENSION = 2
    GMVAE_OBSERVATION_VARIANCE = 0.1
    NVPVAE_HIDDEN_DIMENSION = 64
    NVPVAE_FLOW_HIDDEN_DIMENSION = 32
    OUTPUT_PATH = "autoencode_toy.png"
    RUNS = [
        {"model": "GMVAELayer", "dataset": "clusters", "num_components": 3, "information_constraint": 0.0},
        {"model": "GMVAELayer", "dataset": "pinwheel", "num_components": 5, "information_constraint": 0.0},
        {"model": "GMVAELayer", "dataset": "pinwheel", "num_components": 5, "information_constraint": 1.4},
        {"model": "NVPVAELayer", "dataset": "pinwheel", "num_couplings": 0},
        {"model": "NVPVAELayer", "dataset": "pinwheel", "num_couplings": 4},
    ]

    DATASETS = {"clusters": make_clusters, "pinwheel": make_pinwheel}

    def train_gmvae(run: dict, data: NDArray, labels: NDArray) -> tuple[GMVAELayer, dict]:
        """
        Fit a GMVAELayer, unsupervised, with a Gaussian observation model of fixed variance.

        Returns
        -------
        the trained layer, and the ELBO terms over training
        """
        RNG.bit_generator.state = np.random.default_rng(SEED).bit_generator.state
        layer = GMVAELayer(2, LATENT_DIMENSION, GMVAE_W_DIMENSION, run["num_components"],
                           information_constraint=run["information_constraint"])
        optimizer = Adam(RATE)
        order = np.random.default_rng(SEED + 1)
        history = {name: [] for name in ("step", "reconstruction", "conditional_prior", "w_prior", "z_prior")}
        layer.train()
        for step in range(STEPS):
            batch = data[order.integers(0, len(data), BATCH)]
            layer.zero_gradients()
            error = layer(batch) - batch
            layer.backward(error / (len(batch) * GMVAE_OBSERVATION_VARIANCE))
            optimizer.step(layer)
            if step % READOUT_EVERY == 0 or step == STEPS - 1:
                history["step"].append(step)
                history["reconstruction"].append(0.5 * (error ** 2).sum(axis=1).mean() / GMVAE_OBSERVATION_VARIANCE)
                for name in ("conditional_prior", "w_prior", "z_prior"):
                    history[name].append(getattr(layer, name))
        layer.eval()
        return layer, history

    def train_nvpvae(run: dict, data: NDArray, labels: NDArray) -> tuple[NVPVAELayer, dict]:
        """
        Fit an NVPVAELayer conditioned on one-hot labels, with the optimal-variance reconstruction.

        Returns
        -------
        the trained layer, and the loss terms over training
        """
        one_hot = np.eye(labels.max() + 1)[labels]
        RNG.bit_generator.state = np.random.default_rng(SEED).bit_generator.state
        layer = NVPVAELayer(2, LATENT_DIMENSION, one_hot.shape[1], hidden_dimension=NVPVAE_HIDDEN_DIMENSION,
                            flow_hidden_dimension=NVPVAE_FLOW_HIDDEN_DIMENSION, num_couplings=run["num_couplings"])
        optimizer = Adam(RATE)
        order = np.random.default_rng(SEED + 1)
        history = {name: [] for name in ("step", "reconstruction", "kl_divergence", "observation_variance")}
        layer.train()
        for step in range(STEPS):
            rows = order.integers(0, len(data), BATCH)
            layer.zero_gradients()
            reconstruction, gradient = layer.reconstruction_loss(layer(data[rows], one_hot[rows]), data[rows])
            layer.backward(gradient)
            optimizer.step(layer)
            if step % READOUT_EVERY == 0 or step == STEPS - 1:
                history["step"].append(step)
                history["reconstruction"].append(reconstruction)
                history["kl_divergence"].append(layer.kl_divergence)
                history["observation_variance"].append(layer.observation_variance)
        layer.eval()
        return layer, history

    def score_gmvae(layer: GMVAELayer, data: NDArray, labels: NDArray) -> dict:
        """
        Returns
        -------
        each point's component, the purity of those against the true labels, and the number of components that
        hold at least 1% of the points
        """
        components = layer.cluster(data).argmax(axis=1)
        purity = sum(np.bincount(labels[components == k]).max() for k in np.unique(components)) / len(labels)
        shares = np.bincount(components, minlength=layer.num_components) / len(components)
        return {"components": components, "purity": float(purity), "components_used": int((shares >= 0.01).sum())}

    def score_nvpvae(layer: NVPVAELayer, data: NDArray, labels: NDArray) -> dict:
        """
        Returns
        -------
        reconstruction mean squared error, points generated for each label, and the share of those whose nearest
        data point carries the same label, with their mean distance to the data
        """
        one_hot = np.eye(layer.label_dimension)
        reference, reference_labels = data[::5], labels[::5]
        generated, generated_labels = [], []
        for label in range(layer.label_dimension):
            generated.append(layer.generate(len(data) // (5 * layer.label_dimension), one_hot[label]))
            generated_labels.append(np.full(len(generated[-1]), label))
        generated, generated_labels = np.concatenate(generated), np.concatenate(generated_labels)
        distances = np.linalg.norm(generated[:, None] - reference[None], axis=-1)
        return {
            "reconstruction_mse": float(((layer(data, one_hot[labels]) - data) ** 2).mean()),
            "generated": generated,
            "generated_labels": generated_labels,
            "on_label": float((reference_labels[distances.argmin(axis=1)] == generated_labels).mean()),
            "distance_to_data": float(distances.min(axis=1).mean()),
        }

    def plot_gmvae(axes, run: dict, data: NDArray, layer: GMVAELayer, history: dict, scores: dict) -> None:
        """The GMVAE paper's Figs. 2 and 3: data by component, latent with components at w = 0, samples, z-prior."""
        data_axis, latent_axis, generated_axis, curve_axis = axes
        title = f"GMVAELayer, {run['dataset']}, K={run['num_components']}, lambda={run['information_constraint']:g}"
        data_axis.scatter(data[:, 0], data[:, 1], c=scores["components"], cmap="tab10", s=2)
        data_axis.set(title=f"{title}\ndata by component (purity {scores['purity']:.2f})")

        layer.plot_latent(data, scores["components"], axis=latent_axis)

        generated = layer.generate(len(data))
        generated_axis.scatter(generated[:, 0], generated[:, 1], s=2, alpha=0.3)
        generated_axis.set(title="generated")

        curve_axis.plot(history["step"], history["z_prior"], label="z_prior")
        if run["information_constraint"]:
            curve_axis.axhline(run["information_constraint"], color="gray", linestyle="--",
                               label="information_constraint")
        curve_axis.set(title="z-prior over training", xlabel="step", ylabel="nats")
        curve_axis.legend(fontsize=8)

    def plot_nvpvae(axes, run: dict, data: NDArray, layer: NVPVAELayer, history: dict, scores: dict,
                    labels: NDArray) -> None:
        """Each label's prior in the flowed space and in z with the data, samples by label, KL over training."""
        flowed_axis, latent_axis, generated_axis, curve_axis = axes
        layer.plot_latent(data, np.eye(layer.label_dimension)[labels], axes=(flowed_axis, latent_axis))
        flowed_axis.set_title(f"NVPVAELayer, {run['dataset']}, num_couplings={run['num_couplings']}\n"
                              f"{flowed_axis.get_title()}")

        generated = scores["generated"]
        generated_axis.scatter(generated[:, 0], generated[:, 1], c=scores["generated_labels"], cmap="tab10", s=2,
                               alpha=0.5)
        generated_axis.set(title=f"generated by label ({scores['on_label']:.2f} on that label's arc)")

        curve_axis.plot(history["step"], history["kl_divergence"], label="kl_divergence")
        curve_axis.set(title="KL over training", xlabel="step", ylabel="nats")
        curve_axis.legend(fontsize=8)

    def main():
        """
        Train the GMVAELayer (unsupervised) on three Gaussian clusters and on the pinwheel, with and without the
        information constraint, and the NVPVAELayer (conditioned on the arc) on the pinwheel, with and without the
        flow. Prints each model's loss terms and scores, and saves one row of plots per run.
        """
        results = []
        for run in RUNS:
            data, labels = DATASETS[run["dataset"]](NUM_POINTS, np.random.default_rng(SEED))
            if run["model"] == "GMVAELayer":
                layer, history = train_gmvae(run, data, labels)
                scores = score_gmvae(layer, data, labels)
            else:
                layer, history = train_nvpvae(run, data, labels)
                scores = score_nvpvae(layer, data, labels)
            results.append((run, data, labels, layer, history, scores))

        gmvae_names = ("reconstruction", "conditional_prior", "w_prior", "z_prior")
        print(f"{'GMVAELayer':12s}{'num_components':>16s}{'information_constraint':>24s}"
              + "".join(f"{name:>19s}" for name in gmvae_names) + f"{'purity':>8s}{'components_used':>17s}")
        for run, _, _, _, history, scores in results:
            if run["model"] == "GMVAELayer":
                print(f"{run['dataset']:12s}{run['num_components']:16d}{run['information_constraint']:24g}"
                      + "".join(f"{history[name][-1]:19.3f}" for name in gmvae_names)
                      + f"{scores['purity']:8.2f}{scores['components_used']:17d}")
        print()
        nvpvae_names = ("reconstruction", "kl_divergence", "observation_variance")
        print(f"{'NVPVAELayer':12s}{'num_couplings':>15s}" + "".join(f"{name:>22s}" for name in nvpvae_names)
              + f"{'reconstruction_mse':>20s}{'on_label':>10s}{'distance_to_data':>18s}")
        for run, _, _, _, history, scores in results:
            if run["model"] == "NVPVAELayer":
                print(f"{run['dataset']:12s}{run['num_couplings']:15d}"
                      + "".join(f"{history[name][-1]:22.4f}" for name in nvpvae_names)
                      + f"{scores['reconstruction_mse']:20.4f}{scores['on_label']:10.2f}"
                      + f"{scores['distance_to_data']:18.3f}")

        figure, axes = plt.subplots(len(results), 4, figsize=(17, 4.2 * len(results)), squeeze=False)
        for row, (run, data, labels, layer, history, scores) in enumerate(results):
            if run["model"] == "GMVAELayer":
                plot_gmvae(axes[row], run, data, layer, history, scores)
            else:
                plot_nvpvae(axes[row], run, data, layer, history, scores, labels)
            for axis in axes[row][:3]:
                axis.set_aspect("equal", adjustable="datalim")
        figure.tight_layout()
        figure.savefig(OUTPUT_PATH, dpi=80)
        plt.show()

    main()
