import time
from typing import Optional

import numpy as np
from polyergalio.models.clustering.plsom_clustering import DISTANCE_DICT, PLSOM
from polyergalio.fitted_model import FittedModel
from polyergalio.visuals.cluster_visuals import (
    plot_clusters,
    plot_convergence,
    plot_hit_map,
    plot_som_2d,
    plot_structure_growth,
)
from numpy.typing import NDArray


class FreePLSOM(PLSOM):
    """
    Non-gridded lattice restricted Parameterless Self Organizing Map, with Growing and Shrinking
    Uses Neural Gas (NG - Martinetz, Berkovich & Schulten 1993)
    Growing follows Growing Neural Gas (GNG - Fritzke)

    GPLSOM keeps a full rectangle so it can reuse manhattan distance for the neighborhood function.
    FreePLSOM removes grid constraints:
        there is no row/column, no boundary/interior split, and no shape to keep consistent.

    Growth and pruning add or drop a single neuron, and the final layout is whatever shape the data pulls it into
        rather than a rectangular mesh.

    Neighborhood is based on Neural Gas / PLSOM - but not plain distance: every neuron is ranked by distance to the
        current sample and updated by exp(-rank/theta) as the neighborhood (pseudo gaussian)

    A second, independent `merge_duplicates` prunes any neuron that still ends up a near-duplicate of another
        (distance below the `merge_radius` threshold)

    Growth follows GNG (Fritzke): the worst neuron by mean_neuron_error is split with its nearest neighbor.
        Inserting a new neuron interpolated between them, whenever the worst neuron's error passes.

        GT = -input_dim * ln(spread_factor)

    Pruning drops the neuron with the "coldest" share of recent hits, once it falls below prune_ratio.
    """

    state_names = (
        "n_neurons", "growth_threshold", "last_structural_epoch", "structure_trace",
        "theta_max_auto", "merge_radius_auto",
    )

    def __init__(
        self,
        n_neurons: int,
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
        filter_dead_neurons: bool = True,
        spread_factor: float = 0.5,
        growth_anneal: float = 1.0,
        prune_ratio: float = 0.25,
        min_neurons: int = 2,
        max_neurons: Optional[int] = None,
        settle_epochs: int = 3,
        merge_radius: Optional[float] = None,
    ):
        """

        Parameters
        ----------
        n_neurons : starting neuron count.
        input_dim, lock_seed, distance, verbose, hit_decay, r_decay,
        validation_fraction, patience, min_delta, restore_best,
        freeze_fraction, filter_dead_neurons : see PLSOM.
        theta_min, theta_max : as in PLSOM, but in units of neighborhood RANK rather than lattice/weight-space distance
            Left unset, theta_max is calibrated to the starting neuron count.
        spread_factor : GSOM spread factor in (0, 1). Lower raises the growth threshold and yields fewer neurons.
        growth_anneal : per-epoch multiplier on growth_threshold, so growth can be made progressively harder to trigger
            as training proceeds. 1.0 (default) leaves the threshold fixed.
        prune_ratio : a neuron becomes a pruning candidate when its share of recent hits drops below this fraction of
            the average.
        min_neurons : floor on neuron count, so pruning cannot empty the map.
        max_neurons : ceiling on neuron count. Defaults to four times the starting count.
        settle_epochs : epochs to wait between structural changes, so the map can converge before it is measured again.
        merge_radius : two neurons closer than this in x's unit space get merged (the colder of the pair is dropped)
            Left unset, it's calibrated to a small fraction of the weights' own spread, the same way THETAMAX is
        """
        # PLSOM.__init__ bakes in a width/height rectangle this class doesn't
        # have, so this skips it and calls FittedModel directly instead.
        FittedModel.__init__(self, seed=lock_seed, input_dimension=input_dim, output_dimension=n_neurons)

        self.width = None
        self.height = None
        self.network_shape = None
        self.grid_distances = None
        self.weights = None

        self.N_DIMS = input_dim
        self.VERBOSE = True

        self.n_neurons = n_neurons
        # informational only: calc_neighborhood is fully overridden below and
        # never consults this (unlike the base class's grid/dist dispatch)
        self.neighborhood_method = "rank"
        self._last_distances = None

        self.hit_decay = hit_decay
        self.filter_dead_neurons = filter_dead_neurons
        self.hit_map = np.zeros(shape=self.n_neurons)
        self.neuron_error = np.zeros(shape=self.n_neurons)
        self.verbose = verbose

        self.theta_min = theta_min
        self.theta_max = theta_max
        self.THETAMIN = theta_min if theta_min else 1

        # theta is a RANK scale here, not a weight-space distance
        self.theta_max_auto = theta_max is None
        self.THETAMAX = theta_max if theta_max else 1.0

        self.distance = distance
        self.distance_function = DISTANCE_DICT[distance]

        self.n_iter = 0
        self.q_error_trace = []
        self.epsilon_trace = []
        self.error_scale = 0
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

        assert 0.0 < spread_factor < 1.0, "spread_factor must be in (0, 1)"
        self.spread_factor = spread_factor
        self.growth_threshold = -input_dim * np.log(spread_factor)
        self.growth_anneal = growth_anneal
        self.prune_ratio = prune_ratio
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons if max_neurons else 4 * self.n_neurons
        self.settle_epochs = settle_epochs

        # scaled to the weights' own spread once it's known (initalize_params),
        # same as THETAMAX, when left unset -- see there for why
        self.merge_radius_auto = merge_radius is None
        self.merge_radius = merge_radius if merge_radius else 1e-3

        self.last_structural_epoch = -settle_epochs
        self.structure_trace: list[tuple[int, str, int]] = []

    def initalize_params(self, x_data: NDArray) -> NDArray:
        """
        As PLSOM, but also calibrates THETAMAX (a rank scale, when left unset) to the starting neuron count, and
        `merge_radius` (a weight-space distance, when left unset) to the initial weights' own spread
        """
        _x = super().initalize_params(x_data)
        if self.theta_max_auto:
            self.THETAMAX = max(1.0, self.n_neurons / 2)
        if self.merge_radius_auto:
            centroid = self.weights.mean(axis=0)
            spread = float(np.mean(self.distance_function(centroid, self.weights)))
            self.merge_radius = 1e-3 * spread
        return _x

    def calc_bmu(self, x: NDArray):
        """As PLSOM, but also stashes the per-neuron distances (to this sample) that calc_neighborhood ranks against."""
        bmu_i, distances = super().calc_bmu(x)
        self._last_distances = distances
        return bmu_i, distances

    def calc_neighborhood(self, theta: float, bmu_i: int) -> NDArray:
        """
        Neural Gas neighborhood: rank every neuron by distance to the CURRENT SAMPLE (not to the BMU's weight, and not
        to a fixed lattice position), then decay by rank: h_k = exp(-k/theta) k=0 for the nearest neuron (the BMU)
        """
        ranks = np.argsort(np.argsort(self._last_distances))
        return np.exp(-ranks / theta)

    # -------------    the grow / prune phase    --------------------
    # ---------------------------------------------------------------
    def after_epoch(self, step: int) -> None:
        """
        Merging duplicates, growth and pruning runs every epoch, ahead of annealing and decay of hyperparameters
        """
        if self.weights is None:
            return

        merged = self.merge_duplicates()
        if merged:
            self.structure_trace.append((step, "merge", self.n_neurons))
            self.last_structural_epoch = step
            return

        if self.structure_frozen:
            return

        if step - self.last_structural_epoch < self.settle_epochs:
            return

        errors = self.mean_neuron_error()
        if (
            errors.max() > self.growth_threshold
            and self.n_neurons + 1 <= self.max_neurons
        ):
            self.grow_neuron(int(np.argmax(errors)))
            self.structure_trace.append((step, "grow", self.n_neurons))
            self.last_structural_epoch = step
            return

        candidate = self.pick_prune_site()
        if candidate is not None:
            self.prune_neuron(candidate)
            self.structure_trace.append((step, "prune", self.n_neurons))
            self.last_structural_epoch = step
            return

        dead = self.pick_dead_site()
        if dead is not None:
            self.reinit_neuron(dead)
            self.structure_trace.append((step, "reinit", self.n_neurons))
            self.last_structural_epoch = step

    def decay_structure_thresholds(self) -> None:
        """make growth progressively harder to trigger as training proceeds"""
        self.growth_threshold *= self.growth_anneal

    def nearest_neighbor(self, index: int) -> int:
        """the closest other neuron to this one, in x's unit space"""
        distances = self.distance_function(self.weights[index], self.weights)
        distances = np.asarray(distances, dtype=float).copy()
        distances[index] = np.inf
        return int(np.argmin(distances))

    def pick_duplicate_site(self) -> Optional[int]:
        """
        One neuron of the closest pair in x's unit space.
        The colder (less recently active) of the two is returned; merging / absorbing the other
        """
        if self.n_neurons <= self.min_neurons:
            return None

        best_dist = np.inf
        best_pair = None
        for i in range(self.n_neurons):
            j = self.nearest_neighbor(i)
            d = self.distance_function(self.weights[i], self.weights[j : j + 1])[0]
            if d < best_dist:
                best_dist = d
                best_pair = (i, j)

        if best_pair is None or best_dist >= self.merge_radius:
            return None

        i, j = best_pair
        return i if self.hit_map[i] < self.hit_map[j] else j

    def merge_duplicates(self) -> int:
        """
        Drop every neuron currently collapsed onto a near-duplicate of another, one at a time (removing one changes
        who's nearest to everyone else).

        Returns a count of how many neurons were removed.
        """
        removed = 0
        while True:
            duplicate = self.pick_duplicate_site()
            if duplicate is None:
                break
            self.prune_neuron(duplicate)
            removed += 1
        return removed

    def pick_prune_site(self) -> Optional[int]:
        """
        Get the coldest neuron
        """
        if self.n_neurons <= self.min_neurons:
            return None

        average = self.hit_map.mean()
        if average <= 0:
            return None

        candidate = int(np.argmin(self.hit_map))
        if self.hit_map[candidate] < self.prune_ratio * average:
            return candidate
        return None

    def pick_dead_site(self) -> Optional[int]:
        """
        The coldest neuron, when it's cold enough that pick_prune_site would drop it but min_neurons blocks it.

        We could reinit the target instead of pruning, so a neuron stuck at the floor doesn't sit unused for the rest of
            training. This might not be necessary
        """
        if self.n_neurons > self.min_neurons:
            return None

        average = self.hit_map.mean()
        if average <= 0:
            return None

        candidate = int(np.argmin(self.hit_map))
        if self.hit_map[candidate] < self.prune_ratio * average:
            return candidate
        return None

    # --------------    Structural Modifications    --------------------
    # -----------------------------------------------------------------
    def grow_neuron(self, worst: int) -> None:
        """
        Split the worst neuron: insert a new neuron interpolated between it and its nearest neighbor in x's unit space,
            and relieve both parents so neither immediately triggers the next growth event.
        """
        nearest = self.nearest_neighbor(worst)

        new_weight = 0.5 * (self.weights[worst] + self.weights[nearest])
        new_hits = 0.5 * (self.hit_map[worst] + self.hit_map[nearest])

        self.weights = np.vstack([self.weights, new_weight[None, :]])
        self.hit_map = np.append(self.hit_map, new_hits)
        self.neuron_error = np.append(self.neuron_error, 0.0)

        self.neuron_error[worst] *= 0.5
        self.neuron_error[nearest] *= 0.5

        self.n_neurons += 1
        # changing the neuron count changes the scale of BMU distances, so the running maximum driving epsilon has to
        # re-establish itself
        self.error_scale = 0

    def prune_neuron(self, index: int) -> None:
        """drop a single neuron"""
        self.weights = np.delete(self.weights, index, axis=0)
        self.hit_map = np.delete(self.hit_map, index)
        self.neuron_error = np.delete(self.neuron_error, index)

        self.n_neurons -= 1
        self.error_scale = 0

    def reinit_neuron(self, index: int) -> None:
        """
        Reinitialize a neuron stuck at the size floor: relocate it next to the neuron with the worst mean error instead
            of leaving it unused, and clear its hit/error records so it isn't inhibited by its past performance
        """
        worst = int(np.argmax(self.mean_neuron_error()))
        self.weights[index] = self.weights[worst] + self.RNG.uniform(
            -self.THETAMIN, self.THETAMIN, size=self.weights[index].shape
        )
        self.hit_map[index] = 0.0
        self.neuron_error[index] = 0.0
        self.error_scale = 0

    def predict(self, x: NDArray, n_clusters: int, verbose: bool = False) -> NDArray:
        """
        As PLSOM.predict, but the neuron count can have shrunk since construction
        The downstream clusterer runs on the neurons rather than the samples - so it can give unexpected behaviors if
            you try too many clusters
        """
        affordable = max(2, self.n_neurons // 2)
        if n_clusters > affordable:
            print(
                f"n_clusters {n_clusters} exceeds what {self.n_neurons} neurons "
                f"support, clamping to {affordable}"
            )
            n_clusters = affordable

        return super().predict(x=x, n_clusters=n_clusters, verbose=verbose)

    def fit_predict(
        self,
        x_data: NDArray,
        grid_dim: int,
        num_iterations: int,
        max_clusters: int,
        verbose: Optional[bool] = None,
    ):
        """fit then predict, reporting where the neuron count ended up"""
        start = time.time()
        self.fit(x_data, num_iterations)

        prediction = self.predict(x=x_data, n_clusters=max_clusters, verbose=verbose)

        print(f"freeplsom_fit_predict took {time.time() - start:.2f}s")
        print(
            f"finished at {self.n_neurons} neurons after "
            f"{len(self.structure_trace)} structural changes"
        )

        if self.verbose or verbose:
            plot_som_2d(self.weights, self.network_shape)
            plot_clusters(x_data, prediction)

        return prediction

    def __str__(self):
        return (
            f"FreePLSOM, {self.n_neurons} neurons, "
            f"growth threshold {self.growth_threshold:.3f}, "
            f"{len(self.structure_trace)} structural changes"
        )


if __name__ == "__main__":
    from polyergalio.generators import RandomDatasetGenerator
    from polyergalio.models.clustering.cluster_metrics import (
        calinski_harabasz_index,
        davies_bouldin_index,
        homogeneity,
        silhouette_score,
    )

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

    # start deliberately undersized so growth has something to do
    som = FreePLSOM(
        n_neurons=9,
        input_dim=feature_dim,
        spread_factor=0.95,
        max_neurons=120,
        verbose=False,
    )

    start_neurons = som.n_neurons
    som.fit(x_clust, num_iterations=60)
    print(som)
    for step, action, n in som.structure_trace:
        print(f"  epoch {step:3d}  {action:5s} -> {n} neurons")
    print(
        f"quantisation error {som.q_error_trace[0]:.4f} -> {som.q_error_trace[-1]:.4f}"
    )

    plot_som_2d(
        som.weights,
        som.network_shape,
        samples=som.standardize(x_clust),
        highlight_idx=int(np.argmin(som.hit_map)),
    )
    plot_hit_map(som.hit_map, som.network_shape)

    predictions = som.predict(x=x_clust, n_clusters=6)
    print(f"silhouette (1 is best): {silhouette_score(x_clust, predictions):.4f}")
    print(
        f"CH index (high): {calinski_harabasz_index(x_clust, predictions):.4f}"
    )
    print(f"DB index (low): {davies_bouldin_index(x_clust, predictions):.4f}")
    print(f"homogeneity: {homogeneity(y_clust, predictions):.4f}")

    plot_clusters(x_clust, predictions)

    plot_convergence(som.q_error_trace, som.epsilon_trace, "FreePLSOM convergence")

    epochs = [0] + [step for step, *_ in som.structure_trace]
    neurons = [start_neurons] + [n for _, _, n in som.structure_trace]
    plot_structure_growth(epochs, neurons, "FreePLSOM neuron growth")
