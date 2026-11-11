"""
I started this work in 2022 - but I see there's a better version out now that is a better executed approach! SOM-OLP (2026)
Originally I called this Self Organizing Latent Space (SOLS)
Modifications: based on PLSOM:
    Epsilon - the magnitude of the update - poorly fitting samples (far from their BMU) should have a larger 'gravity'
    Theta - takes the max of the decayed Epsilon vs
    Gamma - the final squared theta falloff radius - temperature/θ²

But the Japanese team have dubbed it "SOM-OLP" - Ubukata, Notsu, Honda (IJCNN 2026)
チームの皆さん、がんばってください。
"""
import time
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from polyergalio.distances import pairwise_euclidian_distance
from polyergalio.fitted_model import FittedModel
from polyergalio.models.clustering.centroid_network import CentroidNeuralNetwork
from polyergalio.models.clustering.cluster_metrics import (
    homogeneity,
    quantization_error,
    silhouette_score,
    topology_preservation,
)
from polyergalio.models.clustering.plsom_utils import free_energy
from polyergalio.models.constants import EPSILON, GLOBAL_DTYPE
from polyergalio.transforms.projections import pca


class SelfOrganizingLatentSpace(FittedModel):
    """

    SOLS - a latent-space self organizing map (Self Organizing Latent Space)
        with patameterless neighborhoods added (PLSOLS)
    Adopting the SOM-OLP from Ubukata, Notsu, Honda (IJCNN 2026) https://arxiv.org/abs/2604.13622



    Every sample {i} owns a continuous latent position v_i on a fixed 2D lattice R = {r_j}, scaled to [-1, 1].
    The objective, with temperature as the entropy weight lambda:

        J = sum_ij p_ij (||x_i - w_j||^2 + gamma ||v_i - r_j||^2) + temperature * sum_ij p_ij ln p_ij

    Block coordinate descent minimizes J with closed-form updates (J should never increase through these steps:

        P is the soft-assignment / entropy regularized softmax (20)
        p_ij = softmax_j(-(||x_i - w_j||^2 + gamma ||v_i - r_j||^2) / temperature)

        V is the weighted average of node coordinates (the latent position of x_i (21)
        V_i = sum_j p_ij r_j

        W is the weighted centroid (neuron or node) of all assigned samples (23)
        W_j = sum_i p_ij x_i / sum_i p_ij


    There is no neuron-to-neuron gridded neighborhood, only a projection distance lattice
    gamma = 0  approximate an entropy-regularized fuzzy c-means
    gamma = 0 with temperature == 0 aproximates a k-means
    """

    parameter_names = ("weights",)
    state_names = ("hit_map", "node_error", "q_error_trace", "objective_trace")

    def __init__(
        self,
        width: int,
        height: int,
        input_dim: int,
        gamma: float = 0.5,
        temperature: float = 1.0,
        tolerance: float = 1e-4,
        lock_seed: int = 42,
        verbose: bool = False,
        raw_scale: bool = False,
        filter_dead_neurons: bool = True,
        theta_max: Optional[float] = None,
        theta_decay: float = 0.98,
        r_decay: float = 0.99,
    ):
        """
        Parameters
        ----------
        width, height : lattice of width x height neurons
        input_dim : dimensionality of the data
        gamma : weight of the latent-space proximity term; 0 ignores the lattice
        temperature : entropy weight lambda; lower gives harder assignments
        tolerance : relative change in J at which fit stops
        lock_seed : seed of the model generator
        verbose : plot after fitting
        raw_scale : fit on the data as given, as the paper does, instead of standardizing it
        filter_dead_neurons : cluster only the neurons that won samples in fit
        theta_max : starting reach of the lattice neighborhood in lattice units (the lattice spans [-1, 1]); each
            sample gets gamma_i = temperature / theta_i^2 with theta_i = max(theta_max * epsilon_i, theta_min), as in
            PLSOM. None keeps gamma fixed
        theta_decay : per-iteration multiplier on theta_max, floored at theta_min = sqrt(temperature / gamma)
        r_decay : per-iteration multiplier on the running error scale that epsilon is measured against
        """
        super().__init__(seed=lock_seed, input_dimension=input_dim, output_dimension=width * height)
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if theta_max is not None and theta_max <= 0:
            raise ValueError("theta_max must be positive")

        self.width = width
        self.height = height

        self.gamma = gamma
        self.temperature = temperature
        self.tolerance = tolerance
        self.theta_max = theta_max
        self.current_theta_max = theta_max
        self.theta_decay = theta_decay
        self.r_decay = r_decay
        self.error_scale = 0.0

        self.verbose = verbose
        self.raw_scale = raw_scale
        self.filter_dead_neurons = filter_dead_neurons

        self.N_DIMS = input_dim
        self.network_shape = [height, width]
        self.n_neurons = height * width
        self.lattice = self.build_lattice()

        self.weights = None
        self.latent_positions = None
        self.hit_map = np.zeros(shape=self.n_neurons)
        self.node_error = np.zeros(shape=self.n_neurons)
        self.q_error_trace = []
        self.objective_trace = []

    def build_lattice(self) -> NDArray:
        """
        Node coordinates (n_neurons, 2) as (row, column)
        Both axes share a normalized unit space, set so the longer axis spans [-1, 1]
        Each axis is zero-centered
        """
        rows, cols = self.network_shape
        row_idx, col_idx = np.divmod(np.arange(rows * cols), cols)
        spacing = 2.0 / max(max(rows, cols) - 1, 1)
        row_axis = (np.arange(rows) - (rows - 1) / 2) * spacing
        col_axis = (np.arange(cols) - (cols - 1) / 2) * spacing
        return np.stack([row_axis[row_idx], col_axis[col_idx]], axis=-1)

    @property
    def theta_min(self) -> float:
        """ Narrowest neighborhood reach in lattice units, the width derived from gamma and temperature """
        return float(np.sqrt(self.temperature / (self.gamma + EPSILON)))

    def initialize_params(self, x_data: NDArray) -> NDArray:
        """
        Start the standardization statistics and return the standardized data; with raw_scale the data is
        returned as given.

        Parameters
        ----------
        x_data : raw samples
        """
        if self.raw_scale:
            return x_data
        self.init_standardize(x_data)
        return self.standardize(x_data)

    def standardize(self, data_array: NDArray, mean: NDArray = None, stds: NDArray = None) -> NDArray:
        """ standardize with the running statistics; unchanged when self.raw_scale is True """
        if self.raw_scale:
            return data_array
        return super().standardize(data_array, mean, stds)

    def unstandardize(self, data_array: NDArray, mean: NDArray = None, stds: NDArray = None) -> NDArray:
        """ reverse transform the standardized data to its original units; unchanged when self.raw_scale is True """
        if self.raw_scale:
            return data_array
        return super().unstandardize(data_array, mean, stds)

    def initialize_weights(self, x_data: NDArray) -> NDArray:
        """
        spread the reference vectors over the leading principal component

        w_j = mean + (2 * sum_l r_jl * s_l * u_l) with s_l the spread along the unit direction u_l.

        Parameters
        ----------
        x_data : standardized samples

        Returns
        -------
        (n_neurons, input_dim) reference vectors
        """
        mean = x_data.mean(axis=0)
        centered = x_data - mean
        scores = pca(x_data, top_k_components=self.lattice.shape[1])
        # mean-centered
        scores = scores - scores.mean(axis=0)
        # least-squares solve of scores(weights)
        loadings = np.linalg.lstsq(scores, centered, rcond=None)[0]
        directions = loadings / (np.linalg.norm(loadings, axis=1, keepdims=True) + EPSILON)
        spread = np.std(centered @ directions.T, axis=0)
        offsets = self.lattice[:, : len(directions)] * spread

        return mean + 2.0 * offsets @ directions

    def calc_bmu(self, x_data: NDArray) -> tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        x_data : standardized samples

        Returns
        -------
        index of the nearest reference vector per sample, euclidean distances to every reference vector
        """
        dist = pairwise_euclidian_distance(x_data, self.weights)
        return np.argmin(dist, axis=-1), dist

    def calc_epsilon(self, data_dist: NDArray) -> NDArray:
        """
        Goodness of fit per sample: distance to the best matching node over a decayed running maximum of past distances

        Parameters
        ----------
        data_dist : (N, M) squared distances from samples to reference vectors

        Returns
        -------
        (N, 1) epsilon in [0, 1]
        """
        nearest = np.sqrt(data_dist.min(axis=1, keepdims=True))
        self.error_scale = max(self.error_scale * self.r_decay, float(nearest.max()))
        return nearest / (self.error_scale + EPSILON)

    def calc_theta(self, epsilon: NDArray) -> NDArray:
        """
        Neighborhood reach per sample, bounded by the decaying current theta_max and by theta_min
        Caps / bounds the neighborhood gravity

        Parameters
        ----------
        epsilon : (N, 1) goodness of fit
        """
        return np.maximum(self.current_theta_max * epsilon, self.theta_min)

    def calc_gamma(self, theta: NDArray) -> NDArray:
        """
        Latent-space weight - calculated for each sample.
        Gives the Gaussian neighborhood

        Parameters
        ----------
        theta : (N, 1) neighborhood reach
        """
        return self.temperature / theta ** 2

    def calc_scores(self, data_dist: NDArray, positions: NDArray, gamma: Optional[NDArray] = None) -> NDArray:
        """
        Cost of assigning each sample to each node: data-space distortion plus gamma times latent-space proximity.

        Parameters
        ----------
        data_dist : (N, M) squared distances from samples to reference vectors
        positions : (N, 2) latent positions
        gamma : scalar or (N, 1) latent-space weight; None uses the model's gamma
        """
        gamma = self.gamma if gamma is None else gamma
        return data_dist + gamma * pairwise_euclidian_distance(positions, self.lattice) ** 2

    def calc_assignment(self, scores: NDArray) -> tuple[NDArray, NDArray]:
        """
        Softmax assignment over neurons, and the free energy per sample that normalizes it.

        Parameters
        ----------
        scores : (N, M) assignment costs, lowest wins

        Returns
        -------
        Assignments: Softmax values for each sample
        Energy: the free-energy calculations
        """
        energy = free_energy(scores, self.temperature)
        return np.exp(-(scores - energy[:, None]) / self.temperature), energy

    def fit(self,
            x: NDArray,
            num_iterations: int = 1000,
            verbose: Optional[bool] = None,
            on_epoch: Optional[Callable] = None,
            ) -> None:
        """
        Parameters
        ----------
        x : (N, input_dim) samples
        num_iterations : ceiling on block coordinate descent cycles
        verbose : plot after fitting; None follows the constructor
        on_epoch : called as on_epoch(model, step) after each cycle
        """
        x = np.asarray(x, dtype=GLOBAL_DTYPE)
        _x = self.initialize_params(x)
        # TODO: if we've already fitted, continue training instead of re-init
        self.weights = self.initialize_weights(_x)

        data_distances = pairwise_euclidian_distance(_x, self.weights) ** 2

        # fuzzy assigmnet via softmax (Pij)
        assignment, _energy = self.calc_assignment(data_distances)

        self.q_error_trace = []
        self.objective_trace = []
        self.current_theta_max = self.theta_max
        self.error_scale = 0.0
        previous = None

        for step in range(num_iterations):

            # update V
            positions = assignment @ self.lattice
            # Update weights
            self.weights = (assignment.T @ _x) / (assignment.sum(axis=0)[:, None] + EPSILON)

            data_dist = pairwise_euclidian_distance(_x, self.weights) ** 2

            gamma = self.gamma
            active_decay = True

            if self.theta_max is not None:
                self.current_theta_max = max(self.current_theta_max * self.theta_decay, self.theta_min)
                active_decay = self.current_theta_max <= self.theta_min
                gamma = self.calc_gamma(self.calc_theta(self.calc_epsilon(data_dist)))

            # softmax entropy
            assignment, energy = self.calc_assignment(self.calc_scores(data_dist, positions, gamma))

            objective = energy.sum()
            self.objective_trace.append(objective)
            self.q_error_trace.append(np.sqrt(data_dist.min(axis=1)).mean())

            if on_epoch is not None:
                on_epoch(self, step)

            # early termination -----
            if previous is not None and np.abs(previous - objective) / max(1.0, np.abs(previous)) <= self.tolerance:
                break
            previous = objective if active_decay else None

        self.latent_positions = assignment @ self.lattice
        # single best matching unit
        bmu_i = np.argmin(data_dist, axis=1)
        self.hit_map = np.bincount(bmu_i, minlength=self.n_neurons)
        self.node_error = np.bincount(
            bmu_i, weights=np.sqrt(data_dist[np.arange(len(_x)), bmu_i]), minlength=self.n_neurons
        )
        self.fitted = True

        if self.verbose or verbose:
            plot_som_2d(self.weights, self.network_shape, samples=_x)
            plot_convergence(self.q_error_trace, title="SOLS quantization error")

    def transform(self, x_data: NDArray, num_iterations: int = 100) -> NDArray:
        """
        Latent positions for any samples, alternating assignment and position updates with the reference vectors fixed.

        Parameters
        ----------
        x_data : (N, input_dim) raw samples
        num_iterations : ceiling on cycles; stops earlier once positions move less than tolerance

        Returns
        -------
        (N, 2) positions inside the lattice
        """
        if not self.fitted:
            raise RuntimeError("fit the model before transform")
        _x = self.standardize(np.asarray(x_data, dtype=float))
        data_dist = pairwise_euclidian_distance(_x, self.weights) ** 2

        assignment, _ = self.calc_assignment(data_dist)
        positions = assignment @ self.lattice

        for _ in range(num_iterations):
            assignment, _ = self.calc_assignment(self.calc_scores(data_dist, positions))
            updated = assignment @ self.lattice
            shift = np.abs(updated - positions).max()
            positions = updated
            if shift <= self.tolerance:
                break
        return positions

    def topology_scores(self, x_data: NDArray, num_neighbors: int = 5) -> dict:
        """
        Trustworthiness and continuity of the latent positions, and mean squared quantization error, all in the
        scale the model was fitted on.

        Parameters
        ----------
        x_data : (N, input_dim) raw samples
        num_neighbors : neighborhood size for trustworthiness and continuity
        """
        _x = self.standardize(np.asarray(x_data, dtype=float))

        positions = self.transform(x_data)
        trust, continuity_score = topology_preservation(_x, positions, num_neighbors)

        return {
            "trustworthiness": trust,
            "continuity": continuity_score,
            "quantization_error": quantization_error(_x, self.weights, squared_distance=True),
        }

    def forward(self, x_data: NDArray) -> tuple[NDArray, NDArray]:
        return self.calc_bmu(x_data)

    def calculate_loss(self, **kwargs):
        return self.objective_trace[-1] if self.objective_trace else None

    def predict(self, x_data: NDArray, n_clusters: int, verbose: bool = False) -> NDArray:
        """
        Cluster the reference neurons with a centroid network, then label each sample with its assigned cluster
            membership

        Parameters
        ----------
        x_data : (N, input_dim) raw samples
        n_clusters : ceiling on clusters
        """
        if not self.fitted:
            raise RuntimeError("fit the model before predict")
        _x = self.standardize(np.asarray(x_data, dtype=float))
        alive = self.hit_map > 0 if self.filter_dead_neurons else np.ones(self.n_neurons, dtype=bool)
        prototypes = self.weights[alive]
        idxs = np.argmin(pairwise_euclidian_distance(_x, prototypes), axis=1)

        if verbose or self.verbose:
            plot_hit_map(self.hit_map, self.network_shape)

        self.clust_model = CentroidNeuralNetwork(
            max_clusters=min(n_clusters, len(prototypes)), seed=self.seed, initial_clusters=None, epsilon=1e-6
        )
        self.clust_model.fit_predict(
            org_x_data=prototypes,
            num_iterations=500,
            fast_forward=False,
            verbose=False,
            skip_standardize=False,
        )
        _, centroids, labels = self.clust_model.get_optimal()
        x_labels = np.take_along_axis(labels, idxs, axis=0)

        if verbose or self.verbose:
            plot_clusters(prototypes, labels, centroids)
            plot_clusters(x_data, x_labels)

        return x_labels

    def fit_predict(self,
                    x_data: NDArray,
                    num_iterations: int,
                    max_clusters: int,
                    verbose: Optional[bool] = None,
                    ) -> NDArray:
        self.fit(x_data, num_iterations)
        return self.predict(x_data, n_clusters=max_clusters, verbose=bool(verbose))

    @property
    def input_dim(self) -> int:
        return self.input_dimension

    @property
    def lock_seed(self) -> int:
        return self.seed


if __name__ == "__main__":
    from polyergalio.visuals.cluster_visuals import (
        plot_clusters,
        plot_convergence,
        plot_hit_map,
        plot_latent_space,
        plot_som_2d,
        plot_som_3d,
    )
    from polyergalio.generators import RandomDatasetGenerator

    feature_dim = 12
    gen = RandomDatasetGenerator(random_seed=125)
    x_clust, y_clust, meta = gen.generate(
        task="clustering",
        num_samples=7500,
        num_features=feature_dim,
        num_clusters=6,
        noise_scale=0.4,
    )

    def generate_saddle(num_samples: int = 500, noise_scale: float = 0.1, random_seed: int = 123) -> NDArray:
        """
        Saddle manifold x3 = x1^2 - x2^2 + noise, with x1 and x2 uniform on [-1, 1].
        This is a good reference datappoint for comparing against the paper

        Returns
        -------
        (num_samples, 3) samples
        """
        rng = np.random.default_rng(random_seed)
        x_plane = rng.uniform(-1.0, 1.0, size=(num_samples, 2))
        height = x_plane[:, 0] ** 2 - x_plane[:, 1] ** 2 + rng.normal(0.0, noise_scale, size=num_samples)
        return np.column_stack([x_plane, height])


    start = time.time()
    rows, columns = (12, 12)
    sols = SelfOrganizingLatentSpace(width=rows,
                                     height=columns,
                                     input_dim=feature_dim,
                                     tolerance=1e-6,
                                     gamma=100,
                                     temperature=1.0,
                                     verbose=True
                                     )
    predictions = sols.fit_predict(x_clust, num_iterations=100, max_clusters=10)
    plot_latent_space(sols.latent_positions, sols.lattice, y_clust)
    plot_latent_space(sols.latent_positions, sols.lattice, predictions)
    plot_som_3d(weights=sols.weights,
                shape=(rows, columns),
                samples=sols.standardize(x_clust),
                labels=y_clust,
                title=f"toy_data_fitting - labels")

    plot_som_3d(weights=sols.weights,
                shape=(rows, columns),
                samples=sols.standardize(x_clust),
                labels=predictions,
                title=f"toy_data_fitting - predictions")

    print(f"fit_predict took {time.time() - start:.2f}s over {len(sols.objective_trace)} iterations")
    _hom = homogeneity(y_clust, predictions)
    _sil = silhouette_score(x_clust, predictions)
    scores_dict = sols.topology_scores(x_clust)
    scores_dict["homogeneity"] = _hom
    scores_dict["silhouette"] = _sil
    print(scores_dict)

    for i, (score, value) in enumerate(scores_dict.items()):
        plt.bar(i, value, label=str(score))
    plt.xlabel(list(scores_dict.keys()))
    plt.legend()
    plt.show()

    # running the 3-D saddle shape from the paper ----
    # x_saddle = generate_saddle(num_samples=500, noise_scale=0.1, random_seed=123)
    # height_bins = np.digitize(x_saddle[:, 2], np.quantile(x_saddle[:, 2], np.linspace(0.0, 1.0, 9)[1:-1]))

    # for rows, cols in [(9, 18), (13, 13), (20, 20)]:
    #     iters = 2
    #     for _ in range(5):
    #         iters *=2
    #         saddle_map = SelfOrganizingLatentSpace(
    #             width=cols, height=rows, input_dim=3, gamma=75, temperature=1.6, tolerance=1e-6, raw_scale=True, verbose=False
    #         )
    #         saddle_map.fit(x_saddle, num_iterations=iters)
    #         print(f"saddle {rows}x{cols}: {len(saddle_map.objective_trace)} iterations, {saddle_map.topology_scores(x_saddle)}")
    #         plot_som_3d(
    #             saddle_map.weights,
    #             (rows, cols),
    #             samples=saddle_map.standardize(x_saddle),
    #             labels=height_bins,
    #             title=f"saddle {rows}x{cols}",
    #         )
