import copy
import time
from typing import Callable, Optional

import matplotlib.pyplot as plt
import polyergalio.distances as distances
import numpy as np
from polyergalio.models.clustering.centroid_network import CentroidNeuralNetwork
from polyergalio.fitted_model import FittedModel
from polyergalio.visuals.cluster_visuals import plot_clusters
from numpy.typing import NDArray

DISTANCE_DICT = {
    "euclidean": distances.euclidian_distance,
    "manhattan": distances.manhattan_distance,
    "hamming": distances.hamming_distance,
    "cosine": distances.cosine_distance,
}


class PLSOM(FittedModel):
    """
    Parameterless Self Organizing Map
    https://arxiv.org/pdf/0705.0199
    Unlike a traditional SOM, the PLSOM has minimal configurable hyperparameters.
    ε(t)   = ||x(t) - w_c(t)||_2 / r(t)
    r(0) = ||x(0)-w_c(0)||_2
    r(t)   = max( ||x(t)-w_c(t)||_2, r(t-1) )
    Θ(ε)   = neighborhood scale from ε via Eq. (9), (10), or (11)
    h_ci   = exp( - d(i,c)^2 / Θ(ε)^2 )
    Δw_i   = ε * h_ci * (x - w_i)


    The PLSOM algorithm works by adjusting the weights of the neurons to approximate the input data.  In this
    process, we reduce the sample dimensionality by calculating which of the neurons ("primitives") is a best fit for a datapoint.
    This accomplishes two goals: reduction in dimensionality (fewer samples) and pseudo-clustering (each neuron is a prototype for a set of samples).
    The PLSOM's neurons can be easily clustered using a more traditional algorithm; leaving us with a
    sample:pseudo-clustering:cluster membership chain.
    """

    parameter_names = ("weights",)
    state_names = ("hit_map", "node_error", "THETAMAX", "previous_step_r", "q_error_trace", "epsilon_trace")

    def __init__(
        self,
        width: int,
        height: int,
        input_dim: int,
        theta_min=None,
        theta_max=None,
        lock_seed: int = 42,
        distance="euclidean",
        verbose: bool = False,
        hit_decay: float = 0.9,
        r_decay: float = 0.99,
        validation_fraction: float = 0.0,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best: bool = True,
        freeze_fraction: float = 0.15,
    ):
        """
        Create the Self Organizing Map  - rectangular / square grids

        Parameters
        ----------
        width : for simplicity's sake, we make map of width x height neurons.
        height : should be the same as width, number of neurons in h
        input_dim : dimensionality of our input data (should be shape[-1]) -- this is the original number of features
            that we'll be reprojecting through the SOM's dimensions
        theta_min : smallest neighborhood value - set at 0
        theta_max : Maximum neighborhood value - set at or close to width
        lock_seed :int, give an int to lock numpy seed
        distance :distance measure to use, can be 'euclidean', 'manhattan', 'cosine', 'hamming'
        hit_decay : per-epoch multiplier on hit_map and node_error, so both
            describe recent activity rather than the whole training history. At
            0.9 a node's record is largely forgotten over about ten epochs.
        r_decay : per-epoch multiplier carried into the next epoch's starting
            previous_step_r (see calc_epsilon), so the error scale can relax
            over training instead of being pinned wherever an early epoch's
            first sample happened to set it.
        validation_fraction : fraction of x held out from fitting to score
            each epoch. 0 (default) disables validation entirely -- no split,
            no early stopping, no best-state rollback.
        patience : consecutive epochs without a validation improvement of at
            least min_delta before training stops early. Only used when
            validation_fraction > 0.
        min_delta : minimum drop in validation error that counts as an
            improvement.
        restore_best : if True and validation is enabled, the map is rolled
            back to its best-validation-epoch state at the end of fit(),
            rather than left at whatever the final epoch produced.
        freeze_fraction : fraction of num_iterations, at the end of training,
            during which structural changes (grow/prune/reinit) are disabled
            so a late-inserted node still gets real training time before the
            run ends. Has no effect on a fixed-lattice PLSOM, only on growing
            subclasses.
        """
        super().__init__(seed=lock_seed, input_dimension=input_dim, output_dimension=width * height)

        self.width = width
        self.height = height
        self.weights = None

        self.N_DIMS = input_dim
        self.VERBOSE = True

        # variables that will need to be updated as we grow / shrink
        self.network_shape = [height, width]
        self.n_neurons = height * width
        self.grid_distances = self.build_grid()

        self.hit_decay = hit_decay
        self.hit_map = np.zeros(shape=self.n_neurons)
        self.node_error = np.zeros(shape=self.n_neurons)
        self.verbose = verbose

        # "grid" judges neighborhood by lattice position (see calc_neighborhood);
        # a subclass with no lattice (e.g. a distance-only grower) sets this to
        # "dist" to judge it by distance between neurons' weight vectors instead
        self.neighborhood_method = "grid"

        # Constants
        # minimum theta (within neighborhood influence)- 1 for alternate equations - might be worth trying both!
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.THETAMIN = theta_min if theta_min else 1
        # Maximum value for neighborhood influence - also called Beta in some papers
        self.THETAMAX = theta_max if theta_max else width

        # TODO: add feedback messaging if invalid selection. Convert to Enum distances
        self.distance = distance
        self.distance_function = DISTANCE_DICT[distance]

        self.n_iter = 0

        self.q_error_trace = []
        self.epsilon_trace = []

        self.previous_step_r = 0
        self.r_decay = r_decay

        assert 0.0 <= validation_fraction < 1.0, "validation_fraction must be in [0, 1)"
        assert 0.0 <= freeze_fraction <= 1.0, "freeze_fraction must be in [0, 1]"
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.freeze_fraction = freeze_fraction

        self.val_error_trace: list[float] = []
        self.best_val_error: float = float("inf")
        self.best_epoch: Optional[int] = None
        self.structure_frozen: bool = False
        self._best_state: Optional[dict] = None

    def initialize_weights(self, x: NDArray, method="kaiming") -> NDArray:
        """
        We will use a nnet-style of initialization for weights
        Parameters
        ----------
        x : array of x dataponts
        method :  string - use 'kaiming', "he" or "sample_space"

        Returns
        -------
        returns the init weights -> to assing to self.weights
        """
        if method in ["kaiming", "he"]:
            # may need to scale down bound further!
            bound = np.sqrt(2 / self.N_DIMS)
            # all our values will be between 0 and 1 for categorical data
            return self.RNG.uniform(
                low=0, high=bound, size=(self.n_neurons, self.N_DIMS)
            )

        elif method == "sample_space":
            weights = np.zeros(shape=(self.n_neurons, self.N_DIMS))
            for dim in range(self.N_DIMS):
                low = np.min(x, axis=0)[dim]
                high = np.max(x, axis=0)[dim]
                weights[:, dim] = self.RNG.uniform(
                    low=low, high=high, size=(self.n_neurons)
                )
            return weights

        else:
            return self.RNG.uniform(
                low=-0.1, high=0.1, size=(self.n_neurons, self.N_DIMS)
            )

    def initalize_params(self, x_data: NDArray) -> NDArray:
        """
        Initalize the starting parameters for weights and standardization
        Parameters
        ----------
        x_data :

        Returns
        -------

        """
        _x = copy.deepcopy(x_data)
        self.init_standardize(_x)
        _x = self.standardize(_x)

        if self.weights is not None:
            pass
        else:
            self.weights = self.initialize_weights(_x, method="sample_space")

        return _x

    def fit(
        self,
        x: NDArray,
        num_iterations: int,
        verbose: Optional[bool] = None,
        on_epoch: Optional[Callable[["PLSOM", int], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        on_epoch : optional callback run at the end of every epoch, after
            after_epoch's own growth/prune decision, as (self, step). A seam
            for a caller that wants a snapshot per epoch (e.g. an animation)
            without altering training itself.
        """
        x = np.asarray(x)

        if self.validation_fraction > 0:
            n_val = max(1, int(len(x) * self.validation_fraction))
            perm = self.RNG.permutation(len(x))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            x_train, x_val_raw = x[train_idx], x[val_idx]
        else:
            x_train, x_val_raw = x, None

        # initalize our paramters, weights and standaridzaion trackers -- the
        # validation split is held out of this too, so its stats never leak in
        _x = self.initalize_params(x_train)
        _x_val = self.standardize(x_val_raw) if x_val_raw is not None else None

        epochs_without_improvement = 0

        for step in range(num_iterations):
            self.RNG.shuffle(_x)
            # Decay our maximum value of THETA slightly - To  keep the full grid from being pulled back and forth by outliers.
            # Floored at THETAMIN rather than reset above it: bouncing back up to
            # THETAMIN + 1 every time it dipped below created a permanent decay/jump
            # sawtooth once THETAMAX got close to THETAMIN, and each jump suddenly
            # widens the neighborhood again, which can collapse the whole map.
            self.THETAMAX = max(self.THETAMAX * 0.98, self.THETAMIN)

            # fade the per-node records so they track recent activity. Without
            # this a node that was dead early still reads as dead once busy.
            self.hit_map *= self.hit_decay
            self.node_error *= self.hit_decay

            # structural changes freeze for the last freeze_fraction of
            # training, so a late-inserted node still gets real training time
            # under a properly annealed neighborhood before the run ends
            self.structure_frozen = step >= num_iterations * (1 - self.freeze_fraction)

            bmu_i, bmu_dist = self.calc_bmu(_x[0])
            # carry the error scale forward across epochs (decayed by
            # r_decay) instead of resetting it to a single noisy point
            # estimate every epoch -- lets it relax as training converges
            # without being pinned wherever an early epoch's first sample set it
            self.previous_step_r = max(
                self.previous_step_r * self.r_decay, bmu_dist[bmu_i]
            )

            error_trace = []
            epsilon_trace = []

            for sample_i in range(1, _x.shape[0]):
                _xs = _x[sample_i : sample_i + 1, :]

                bmu_i, sample_distances = self.calc_bmu(_xs)
                bmu_distance = sample_distances[bmu_i]
                epsilon = self.calc_epsilon(bmu_distance)
                theta = self.calc_theta(epsilon)
                neighborhood = self.calc_neighborhood(theta, bmu_i)

                # neighborhood is shape (n_samples, n_neurons, dimensions)
                weight_update = epsilon * (
                    neighborhood.reshape(self.n_neurons, -1) * (_xs - self.weights)
                )

                self.weights += weight_update
                self.hit_map[bmu_i] += 1
                self.node_error[bmu_i] += bmu_distance

                # quantisation error is the distance to the winning unit. The
                # mean over every neuron grows as the map spreads out, so it
                # cannot be compared across maps of different sizes.
                error_trace.append(bmu_distance)
                epsilon_trace.append(epsilon)

                # if (self.verbose or verbose) and (sample_i % 10 == 0):
                #     self.plot_neighborhood(epsilon * (neighborhood.reshape(self.n_neurons, -1)))

            self.q_error_trace.append(np.mean(error_trace))
            self.epsilon_trace.append(np.mean(epsilon_trace))
            self.after_epoch(step)
            self.anneal_structure_thresholds()
            if on_epoch is not None:
                on_epoch(self, step)

            if _x_val is not None:
                _, val_dist = self.forward(_x_val)
                val_error = float(np.mean(np.min(val_dist, axis=-1)))
                self.val_error_trace.append(val_error)

                if val_error < self.best_val_error - self.min_delta:
                    self.best_val_error = val_error
                    self.best_epoch = step
                    self._best_state = copy.deepcopy(self.get_state())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if self.verbose or verbose:
                        print(
                            f"early stop at epoch {step}, best val error "
                            f"{self.best_val_error:.4f} at epoch {self.best_epoch}"
                        )
                    break

        if _x_val is not None and self.restore_best and self._best_state is not None:
            self.set_state(copy.deepcopy(self._best_state))

        self.fitted = True

        if self.verbose or verbose:
            self.plot_grid(samples=0, highlight_idx=np.argmin(self.hit_map))
            plt.plot(self.q_error_trace)
            plt.plot(self.epsilon_trace)
            plt.legend(["q_error", "epsilon"])
            plt.show()

    def after_epoch(self, step: int) -> None:
        """
        Seam for subclasses that restructure the map between epochs. A fixed
        lattice has nothing to do here, so this is deliberately empty.
        """
        pass

    def anneal_structure_thresholds(self) -> None:
        """
        Seam for growing subclasses to make structural change progressively
        harder to trigger as training proceeds. A fixed lattice has nothing
        to anneal, so this is deliberately empty.
        """
        pass

    def mean_node_error(self) -> NDArray:
        """
        Average quantisation error per sample assigned to each node.

        The raw node_error is a decayed sum over every sample the node won, so
        its scale rides on dataset size and would clear any fixed threshold.
        Dividing by the hit count puts it in the units of a single distance,
        which is what a growth threshold like GSOM's GT is expressed in.
        """
        return self.node_error / np.maximum(self.hit_map, 1.0)

    def calc_bmu(self, x: NDArray) -> tuple[NDArray | int, NDArray]:
        """
        find the best matching unit to the input x

        Parameters
        ----------
        x : input data

        Returns
        -------
        BMU for each sample (shape batch_size, 1), distances to each neuron (shape batch_size, n_neurons)
        """
        dist = self.distance_function(x, self.weights)
        bmu_i = np.argmin(dist, axis=-1)
        return bmu_i, dist

    def calc_epsilon(self, bmu_distance: NDArray) -> NDArray:
        """
        calculate epsilon -- value for the magnitude of the update - it's a value driven by the goodness-of-fit (how
        close / far is this sample from its best matching untit?)

        Parameters
        ----------
        bmu_distance : the distance array between the sample and its best matching unit

        Returns
        -------
        the single-point value epsilon
        """
        self.previous_step_r = np.max((bmu_distance, self.previous_step_r), axis=0)
        return bmu_distance / self.previous_step_r

    def calc_theta(self, epsilon: float | NDArray) -> float | NDArray:
        """
        sets bounds on the reach of the neighborhood function
        uses constants self.THETAMAX and self.THETAMIN - these are usually  set at 0, 1 or 2
        Parameters
        ----------
        epsilon : the calculated epsilon value

        Returns
        -------
        single-value theta for use in neighborhood
        """
        # Using the PLSOM epsilon scale
        theta = max(self.THETAMAX * epsilon, self.THETAMIN)

        return theta

    def get_lateral_distance(self, bmu_i: int, method: str = "dist") -> NDArray:
        """
        applies the distance function to find distance between the bmu and all other neurons
        Parameters
        ----------
        bmu_i : index of the best matching unit under consideration
        method : 'grid' or 'dist' - use the precomputed grid distances (manhattan),
            or calculate distance between the neurons' embedding dimensions

        Returns
        -------
        distnace array between bmu and all other neurons
        """
        # return distance between this unit and all others - return should be n_neurons, n_dims
        if method == "dist":
            bmw = self.weights[bmu_i, :]
            return self.distance_function(bmw, self.weights) ** 2

        elif method == "grid":
            return self.grid_distances[bmu_i, :] ** 2

    def calc_neighborhood(self, theta: float | NDArray, bmu_i: int) -> NDArray:
        """
        This is the magic -- PLSOM defines the neighborhood function as an area of influence around a neuron that's
        being updated. The neighborhood is defined as a Gaussian function (exp(-x**2) of the distance between the
        active and the other neurons.

        Parameters
        ----------
        theta : the derived bounds (see calc_theta)
        bmu_i : the index of the best matching unit

        Returns
        -------
        adjusted gaussian kernel applied to distances
        """

        return np.exp(
            -self.get_lateral_distance(bmu_i, method=self.neighborhood_method)
            / theta**2
        )

    def _idx_to_grid(self, idx: int) -> tuple[int, int]:
        """
        1d vector index to 2d [row, col]. Row major, so the divisor is the
        column count. Using the row count instead only agrees on square maps.
        """
        rows, cols = self.network_shape
        return (int(idx // cols), int(idx % cols))

    def _grid_to_idx(self, grid_i: tuple[int, int]) -> int:
        """2d [row, col] back to a 1d index, row major"""
        rows, cols = self.network_shape
        return int(grid_i[1] + (grid_i[0] * cols))

    @staticmethod
    def grid_manhattan_distance(
        row_a: int, col_a: int, row_b: int, col_b: int
    ) -> float:
        """returns the cityblock / manhattan dist between two grid points"""
        return abs(row_a - row_b) + abs(col_a - col_b)

    def build_grid(self) -> NDArray:
        """
        All pairs manhattan distance between lattice positions, shaped
        (n_neurons, n_neurons). Growing or shrinking the map invalidates this,
        so it is cheap enough to rebuild outright.
        """
        rows, cols = self.network_shape
        row_idx, col_idx = np.divmod(np.arange(rows * cols), cols)

        return (
            np.abs(row_idx[:, None] - row_idx[None, :])
            + np.abs(col_idx[:, None] - col_idx[None, :])
        ).astype(float)

    def plot_grid(self, samples=0, highlight_idx=None):
        """utility function to plot the first two dimensions of the grid of weights"""
        ws = self.weights.reshape(
            self.network_shape[0], self.network_shape[1], self.N_DIMS
        )
        # Draw lines between each 2d weight vector in grid
        plt.figure(figsize=(15, 10))
        plt.title(f"PLSOM Grid {self.network_shape[0]}x{self.network_shape[1]}")
        # np.size rather than a truth test, so an array of samples can be passed
        if np.size(samples) > 1:
            plt.scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.3)
        for row in range(self.network_shape[0]):
            plt.plot(ws[row, :, 0], ws[row, :, 1], "bo-")
        for col in range(self.network_shape[1]):
            plt.plot(ws[:, col, 0], ws[:, col, 1], "bo-")
        # index 0 is a valid unit to highlight, so test against None
        if highlight_idx is not None:
            focus = self._idx_to_grid(highlight_idx)
            plt.scatter(
                ws[focus[0], focus[1], 0], ws[focus[0], focus[1], 1], s=200.0, c="r"
            )
        plt.show()

    def plot_neighborhood(self, neighborhood):
        """utility function to plot the neighborhood, "gravity" or sphere of influence around the bmu"""
        # shape the weights into the 2D grid representation
        ns = neighborhood.reshape(self.network_shape[0], self.network_shape[1])
        # Draw lines between each 2d weight vector in grid
        plt.figure(figsize=(10, 10))
        plt.title("PLSOM neighborhood")
        for row in range(self.network_shape[0]):
            for col in range(self.network_shape[1]):
                plt.scatter(row, col, s=ns[row, col] * 1000)
        plt.show()

    def plot_heatmap(self):
        """utilty function to draw the hit map as aa heat map"""
        plt.figure(figsize=(10, 10))
        plt.title("PLSOM Heatmap")
        self.hit_map.reshape(self.network_shape)
        plt.imshow(
            self.hit_map.reshape(self.network_shape),
            cmap="hot",
            interpolation="nearest",
        )
        plt.show()

    def forward(self, x_data: NDArray) -> tuple[NDArray, NDArray]:
        idxs, dist = self.calc_bmu(x_data[:, np.newaxis, :])
        return idxs, dist

    def calculate_loss(self, **kwargs):
        pass

    def predict(self, x: NDArray, n_clusters: int, verbose: bool = False) -> NDArray:
        """
        Using a dedicated clustering method to predict classes of samples based on the SOM's re-projected
        representation
        We use the clustering method on the SOM's weights, rather than on the dataset directly
        """
        # determine which protoype (SOM neuron) is closest to each sample
        _x = self.standardize(x)

        idxs, dist = self.forward(_x)
        # grid_idxs = np.array(
        #     [self._idx_to_grid(_i) for _i in idxs]
        # )
        print(idxs)

        if verbose or self.verbose:
            print(f"Weights: {self.weights.shape}")
            self.plot_heatmap()

        self.clust_model = CentroidNeuralNetwork(
            max_clusters=n_clusters, seed=42, initial_clusters=None, epsilon=1e-4
        )

        _, _ = self.clust_model.fit_predict(
            org_x_data=self.weights,
            num_iterations=100,
            fast_forward=False,
            verbose=False,
            skip_standardize=True,
        )

        best_scoring, centroids, labels = self.clust_model.get_optimal()
        print(f"Optimal Clusters at {best_scoring}")

        # assign each sample to its closest SOM neuron / prototype (idxs), then assign that neuron to its cluster (
        # labels)
        x_labels = np.take_along_axis(labels, idxs, axis=0)

        if verbose or self.verbose:
            plot_clusters(self.weights, labels, centroids)
            plot_clusters(x, x_labels)

        return x_labels

    def fit_predict(
        self,
        x_data: NDArray,
        grid_dim: int,
        num_iterations: int,
        max_clusters: int,
        verbose: Optional[bool] = None,
    ):
        """

        Parameters
        ----------

        Returns
        -------

        """
        start = time.time()
        self.fit(x_data, num_iterations)

        prediction = self.predict(x=x_data, n_clusters=max_clusters, verbose=True)

        print("plsom_fit_predict took", time.time() - start)

        if self.verbose or verbose:
            self.plot_grid(samples=0, highlight_idx=None)
            plot_clusters(x_data, prediction)

        return prediction

    @property
    def input_dim(self) -> int:
        return self.input_dimension

    @property
    def lock_seed(self) -> int:
        return self.seed


if __name__ == "__main__":
    # Example usage
    from polyergalio.generators import RandomDatasetGenerator
    from polyergalio.models.clustering.cluster_metrics import *

    feature_dim = 2
    gen = RandomDatasetGenerator(random_seed=123)
    x_clust, y_clust, meta = gen.generate(
        task="clustering",
        num_samples=1500,
        num_features=feature_dim,
        num_clusters=6,
        noise_scale=0.4,
    )
    plot_clusters(x_clust, y_clust, meta["centroids"])

    max_clusters = 10

    for dim in [10]:
        num_steps = dim * 5
        st = time.time()

        som = PLSOM(
            width=dim,
            height=dim,
            input_dim=feature_dim,
            theta_min=0 + 0.01,
            theta_max=dim - 0.01,
            lock_seed=42,
            distance="euclidean",
            verbose=True,
        )

        predictions = som.fit_predict(
            x_data=x_clust,
            grid_dim=dim,
            num_iterations=num_steps,
            max_clusters=max_clusters,
            verbose=True,
        )

        print("one_cluster predict took: ", time.time() - st)

        print("Predictions:", predictions[:20])
        print("Truth:", y_clust[:20])
        print(homogeneity(predictions, y_clust))

        print(f"silhouette score (1 is best): {silhouette_score(x_clust, predictions)}")
        print(f"CH index (high): {calinski_harabasz_index(x_clust, predictions)}")
        print(f"DB index score (low): {davies_bouldin_index(x_clust, predictions)}")

        print(f"homogenity: {homogeneity(y_clust, predictions)}")
        print(f"Mutual Information: {mutual_information_score(y_clust, predictions)}")
