import time

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from polyergalio.models.clustering.plsom_clustering import PLSOM
from polyergalio.visuals.cluster_visuals import plot_clusters


class GPLSOM(PLSOM):
    """
    Growing and Shrinking Parameterless SOM.

    The lattice stays a full rectangle. Growth and pruning add or drop a whole
    row or column, so everything inherited from PLSOM keeps working unchanged --
    manhattan distance on a grid is still well defined, it just covers a grid of
    a different size.

    Growth follows GSOM (Alahakoon et al.), comparing an accumulated
    quantisation error per node against a threshold set by a spread factor:

        GT = -input_dim * ln(spread_factor)

    When the worst node's accumulated error passes GT the map grows near that
    node. A node on the boundary grows outward, and the new line is
    extrapolated from the two lines behind it. An interior node splits between
    its neighbours, interpolated from them, which relieves crowding rather than
    only widening the footprint.

    Pruning is the mirror: a row or column whose share of recent activity falls
    well below the average line is dropped, subject to a floor on map size.
    Both signals read the decayed hit_map and node_error from PLSOM, so they
    describe recent behaviour rather than the whole training history.

    https://arxiv.org/pdf/0705.0199    (PLSOM)
    """

    state_names = (
        "network_shape", "height", "width", "n_neurons",
        "growth_threshold", "last_structural_epoch", "structure_trace",
    )

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
        spread_factor: float = 0.5,
        growth_anneal: float = 1.0,
        prune_ratio: float = 0.25,
        min_shape: tuple[int, int] = (2, 2),
        max_neurons: Optional[int] = None,
        settle_epochs: int = 3,
    ):
        """
        Parameters
        ----------
        width, height, input_dim, theta_min, theta_max, lock_seed, distance,
        verbose, hit_decay, r_decay, validation_fraction, patience, min_delta,
        restore_best, freeze_fraction : see PLSOM
        spread_factor : GSOM spread factor in (0, 1). Lower raises the growth
            threshold and yields a smaller final map.
        growth_anneal : per-epoch multiplier on growth_threshold, so growth
            can be made progressively harder to trigger as training proceeds.
            1.0 (default) leaves the threshold fixed.
        prune_ratio : a row or column becomes a pruning candidate when its
            share of recent hits drops below this fraction of the average line.
        min_shape : (rows, cols) floor, so pruning cannot collapse the map.
        max_neurons : ceiling on total neurons. Defaults to four times the
            starting count.
        settle_epochs : epochs to wait between structural changes, so the map
            can converge before it is measured again.
        """
        super().__init__(
            width=width,
            height=height,
            input_dim=input_dim,
            theta_min=theta_min,
            theta_max=theta_max,
            lock_seed=lock_seed,
            distance=distance,
            verbose=verbose,
            hit_decay=hit_decay,
            r_decay=r_decay,
            validation_fraction=validation_fraction,
            patience=patience,
            min_delta=min_delta,
            restore_best=restore_best,
            freeze_fraction=freeze_fraction,
        )

        assert 0.0 < spread_factor < 1.0, "spread_factor must be in (0, 1)"

        self.spread_factor = spread_factor
        self.growth_threshold = -input_dim * np.log(spread_factor)
        self.growth_anneal = growth_anneal
        self.prune_ratio = prune_ratio
        self.min_shape = min_shape
        self.max_neurons = max_neurons if max_neurons else 4 * self.n_neurons
        self.settle_epochs = settle_epochs

        self.last_structural_epoch = -settle_epochs
        self.structure_trace: list[tuple[int, str, int, int]] = []

    # -------------    the grow / prune decision    --------------------
    # ------------------------------------------------------------------
    def set_state(self, state: dict) -> None:
        """Restore the saved map, then rebuild the lattice distances for its shape."""
        super().set_state(state)
        self.grid_distances = self.build_grid()

    def after_epoch(self, step: int) -> None:
        """
        At most one structural change per epoch, growth taking precedence,
        then prune, then a dead-unit reinit. Doing more than one in an epoch
        oscillates, since the decisions read the same decayed counters.

        Structural changes are disabled entirely once structure_frozen is
        set (see PLSOM.fit), so the map gets a clean settling period before
        training ends rather than being restructured on its last epochs.
        """
        if self.weights is None:
            return
        if self.structure_frozen:
            return
        if step - self.last_structural_epoch < self.settle_epochs:
            return

        if self.mean_node_error().max() > self.growth_threshold:
            axis, position, edge = self.pick_growth_site()
            if self.growth_fits(axis):
                self.grow_line(axis, position, edge)
                self.structure_trace.append((step, "grow", *self.network_shape))
                self.last_structural_epoch = step
                return

        candidate = self.pick_prune_site()
        if candidate is not None:
            self.prune_line(*candidate)
            self.structure_trace.append((step, "prune", *self.network_shape))
            self.last_structural_epoch = step
            return

        dead = self.pick_dead_site()
        if dead is not None:
            self.reinit_line(*dead)
            self.structure_trace.append((step, "reinit", *self.network_shape))
            self.last_structural_epoch = step

    def anneal_structure_thresholds(self) -> None:
        """make growth progressively harder to trigger as training proceeds"""
        self.growth_threshold *= self.growth_anneal

    def growth_fits(self, axis: int) -> bool:
        """
        Whether a new line along this axis stays inside the neuron budget. The
        check has to be made against the post-growth count, since a line adds a
        whole row or column at once rather than a single node.
        """
        addition = self.network_shape[1 - axis]
        return self.n_neurons + addition <= self.max_neurons

    def pick_growth_site(self) -> tuple[int, int, bool]:
        """
        Find the worst node and decide where its new line goes.

        Returns
        -------
        (axis, position, edge): axis 0 inserts a row and 1 a column, position
        is the insertion index, and edge marks an outward extension rather than
        an interior split.
        """
        rows, cols = self.network_shape
        row, col = self._idx_to_grid(int(np.argmax(self.mean_node_error())))

        # grow along whichever axis the node sits further out on, so a node
        # stranded against an edge extends the map that way. The offsets are
        # normalised by each axis's extent, otherwise the longer axis keeps
        # winning and the lattice drifts into a stripe.
        row_offset = abs(row - (rows - 1) / 2) / max(rows - 1, 1)
        col_offset = abs(col - (cols - 1) / 2) / max(cols - 1, 1)
        axis = 0 if row_offset >= col_offset else 1

        index, extent = (row, rows) if axis == 0 else (col, cols)

        if index == 0:
            return axis, 0, True
        if index == extent - 1:
            return axis, extent, True

        return axis, index + 1, False

    def pick_prune_site(self) -> Optional[tuple[int, int]]:
        """
        The coldest row or column, when it is cold enough to be worth dropping
        and the map can afford to lose it.
        """
        candidates = []
        for axis in (0, 1):
            if self.network_shape[axis] <= self.min_shape[axis]:
                continue

            line_hits = self.hit_map.reshape(self.network_shape).sum(axis=1 - axis)
            average = line_hits.mean()
            if average <= 0:
                continue

            position = int(np.argmin(line_hits))
            if line_hits[position] < self.prune_ratio * average:
                candidates.append((float(line_hits[position]), axis, position))

        if not candidates:
            return None

        _, axis, position = min(candidates)
        return axis, position

    def pick_dead_site(self) -> Optional[tuple[int, int]]:
        """
        A row or column cold enough that pick_prune_site would drop it, but
        blocked by min_shape -- a reinit target instead of a removal, so a
        line stuck at the floor doesn't sit unused for the rest of training.
        """
        candidates = []
        for axis in (0, 1):
            if self.network_shape[axis] > self.min_shape[axis]:
                continue

            line_hits = self.hit_map.reshape(self.network_shape).sum(axis=1 - axis)
            average = line_hits.mean()
            if average <= 0:
                continue

            position = int(np.argmin(line_hits))
            if line_hits[position] < self.prune_ratio * average:
                candidates.append((float(line_hits[position]), axis, position))

        if not candidates:
            return None

        _, axis, position = min(candidates)
        return axis, position

    # -------------    the structural operations    --------------------
    # ------------------------------------------------------------------
    def weight_grid(self) -> NDArray:
        """weights as (rows, cols, dims), which is how the lattice reads"""
        return self.weights.reshape(*self.network_shape, self.N_DIMS)

    def new_line(self, axis: int, position: int, edge: bool) -> NDArray:
        """
        Weights for the line about to be inserted. Interior lines interpolate
        between the neighbours they separate, edge lines extrapolate outward
        from the two lines behind them.
        """
        lines = np.moveaxis(self.weight_grid(), axis, 0)

        if not edge:
            return 0.5 * (lines[position - 1] + lines[position])

        if position == 0:
            return 2 * lines[0] - lines[1]
        return 2 * lines[-1] - lines[-2]

    def grow_line(self, axis: int, position: int, edge: bool) -> None:
        """insert a row (axis 0) or column (axis 1) at position"""
        addition = self.new_line(axis, position, edge)
        grid = np.insert(self.weight_grid(), position, addition, axis=axis)

        self.network_shape[axis] += 1
        self.resync(grid, axis, position, inserted=True)

    def prune_line(self, axis: int, position: int) -> None:
        """drop a row (axis 0) or column (axis 1)"""
        grid = np.delete(self.weight_grid(), position, axis=axis)

        self.network_shape[axis] -= 1
        self.resync(grid, axis, position, inserted=False)

    def resync(self, grid: NDArray, axis: int, position: int, inserted: bool) -> None:
        """
        Return the map to a consistent state after a structural change: weights
        flattened, lattice distances rebuilt, per-node records resized, and the
        PLSOM radius reset.
        """
        rows, cols = self.network_shape
        self.height, self.width = rows, cols
        self.n_neurons = rows * cols
        self.weights = grid.reshape(self.n_neurons, self.N_DIMS)
        self.grid_distances = self.build_grid()

        # a new line inherits its neighbours' activity, otherwise it is the
        # coldest line in the map by construction and pruning removes it again
        self.hit_map = self.resize_record(
            self.hit_map, axis, position, inserted, fill="neighbour_mean"
        )
        # error, by contrast, starts at zero so the new line does not itself
        # look like it needs splitting
        self.node_error = self.resize_record(
            self.node_error, axis, position, inserted, fill="zero"
        )

        if inserted:
            # halve the records either side of a split so the same node does not
            # immediately trigger the next growth event
            lines = np.moveaxis(self.node_error.reshape(rows, cols), axis, 0)
            for neighbour in (position - 1, position + 1):
                if 0 <= neighbour < lines.shape[0]:
                    lines[neighbour] *= 0.5

        # changing the node count changes the scale of BMU distances, so the
        # running maximum driving epsilon has to re-establish itself
        self.previous_step_r = 0

    def resize_record(
        self,
        record: NDArray,
        axis: int,
        position: int,
        inserted: bool,
        fill: str = "zero",
    ) -> NDArray:
        """
        Insert a line into, or delete a line from, a per-node record so it stays
        aligned with the lattice. network_shape is already updated, so the
        record's own shape is still the pre-change one.

        fill picks what an inserted line starts at: "zero", or
        "neighbour_mean" to average whichever adjacent lines exist.
        """
        rows, cols = self.network_shape
        was = [rows, cols]
        was[axis] += -1 if inserted else 1

        grid = record.reshape(was[0], was[1])

        if not inserted:
            return np.delete(grid, position, axis=axis).reshape(-1)

        lines = np.moveaxis(grid, axis, 0)
        if fill == "neighbour_mean":
            neighbours = [
                lines[i] for i in (position - 1, position) if 0 <= i < lines.shape[0]
            ]
            addition = np.mean(neighbours, axis=0)
        else:
            addition = np.zeros(lines.shape[1])

        return np.insert(grid, position, addition, axis=axis).reshape(-1)

    def reinit_line(self, axis: int, position: int) -> None:
        """
        Reinitialize a row/column stuck at the size floor: relocate it next
        to the line with the worst mean error instead of leaving it unused,
        and clear its hit/error records so it gets a fresh trial period.
        """
        errors = self.mean_node_error().reshape(self.network_shape)
        lines_error = np.moveaxis(errors, axis, 0)
        worst = int(np.argmax(lines_error.sum(axis=1)))

        grid = self.weight_grid()
        lines = np.moveaxis(grid, axis, 0)
        lines[position] = lines[worst] + self.RNG.uniform(
            -self.THETAMIN, self.THETAMIN, size=lines[position].shape
        )
        self.weights = grid.reshape(self.n_neurons, self.N_DIMS)

        hit_grid = np.moveaxis(self.hit_map.reshape(self.network_shape), axis, 0)
        hit_grid[position] = 0.0

        error_grid = np.moveaxis(self.node_error.reshape(self.network_shape), axis, 0)
        error_grid[position] = 0.0

        self.previous_step_r = 0

    def predict(self, x: NDArray, n_clusters: int, verbose: bool = False) -> NDArray:
        """
        As PLSOM.predict, but the lattice can have shrunk since construction.
        The downstream clusterer runs on the neurons rather than the samples, so
        asking for more clusters than the map can support fails deep inside it
        with an out of bounds partition. Clamp instead.
        """
        affordable = max(2, self.n_neurons // 2)
        if n_clusters > affordable:
            print(
                f"n_clusters {n_clusters} exceeds what a {self.network_shape} "
                f"lattice supports, clamping to {affordable}"
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
        """fit then predict, reporting where the lattice ended up"""
        start = time.time()
        self.fit(x_data, num_iterations)

        prediction = self.predict(x=x_data, n_clusters=max_clusters, verbose=verbose)

        print(f"gplsom_fit_predict took {time.time() - start:.2f}s")
        print(
            f"lattice finished at {self.network_shape} after "
            f"{len(self.structure_trace)} structural changes"
        )

        if self.verbose or verbose:
            self.plot_grid(samples=0, highlight_idx=None)
            plot_clusters(x_data, prediction)

        return prediction

    def __str__(self):
        return (
            f"GPLSOM lattice {self.network_shape[0]}x{self.network_shape[1]}, "
            f"growth threshold {self.growth_threshold:.3f}, "
            f"{len(self.structure_trace)} structural changes"
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from polyergalio.generators import RandomDatasetGenerator
    from polyergalio.models.clustering.cluster_metrics import (
        silhouette_score,
        calinski_harabasz_index,
        davies_bouldin_index,
        homogeneity,
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

    # ground truth, before the map sees any of it
    plot_clusters(x_clust, y_clust, meta["centroids"])

    # start deliberately undersized so growth has something to do
    som = GPLSOM(
        width=3,
        height=3,
        input_dim=feature_dim,
        theta_min=0.01,
        theta_max=2.99,
        spread_factor=0.95,
        max_neurons=120,
        verbose=False,
    )

    start_neurons = som.n_neurons
    som.fit(x_clust, num_iterations=60)
    print(som)
    for step, action, rows, cols in som.structure_trace:
        print(f"  epoch {step:3d}  {action:5s} -> {rows}x{cols}")
    print(
        f"quantisation error {som.q_error_trace[0]:.4f} -> {som.q_error_trace[-1]:.4f}"
    )

    # the grown lattice, with the coldest unit marked. The weights live in
    # standardized space, so the samples have to be standardized to match
    som.plot_grid(
        samples=som.standardize(x_clust), highlight_idx=int(np.argmin(som.hit_map))
    )
    # where the decayed hits landed across the final lattice
    som.plot_heatmap()

    predictions = som.predict(x=x_clust, n_clusters=6)
    print(f"silhouette (1 is best): {silhouette_score(x_clust, predictions):.4f}")
    print(f"CH index (high):        {calinski_harabasz_index(x_clust, predictions):.4f}")
    print(f"DB index (low):         {davies_bouldin_index(x_clust, predictions):.4f}")
    print(f"homogeneity:            {homogeneity(y_clust, predictions):.4f}")

    # the clustering the SOM's prototypes imply for the original samples
    plot_clusters(x_clust, predictions)

    plt.plot(som.q_error_trace)
    plt.plot(som.epsilon_trace)
    plt.legend(["q_error", "epsilon"])
    plt.title("GPLSOM convergence")
    plt.show()

    # lattice size over training, which the fixed-size PLSOM has no analogue for
    epochs = [0] + [step for step, *_ in som.structure_trace]
    neurons = [start_neurons] + [
        rows * cols for _, _, rows, cols in som.structure_trace
    ]
    plt.step(epochs, neurons, where="post")
    plt.xlabel("epoch")
    plt.ylabel("neurons")
    plt.title("GPLSOM lattice growth")
    plt.show()
