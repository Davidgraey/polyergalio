"""
Differentiable prototype layers: CentroidLayer (soft k-means), PLSOMLayer
(soft parameterless SOM), GPLSOMLayer (growing lattice) and FreePLSOMLayer
(growing, lattice-free neural gas).

Each layer holds K prototypes, scores every sample against them by squared
distance, and replaces the hard argmin with a temperature softmax. Each also
carries its own clustering energy:

    E = (energy_weight / 2N) * sum_n w_n * F_n

where F_n is a free energy over the (optionally neighborhood-smoothed)
distances e for the centroid and lattice layers,

    F_n = -T log sum_k exp(-e_nk / T)        (T = 0: F_n = min_k e_nk)

and the neural gas energy F_n = sum_k exp(-rank_nk / theta) * d_nk for
FreePLSOMLayer.

At T = 0 the classic rules are gradient steps on E:
    k-means (Lloyd): one full-batch step with per-centroid step size N / n_k
    PLSOM family: one single-sample SGD step with learning rate 1,
    w_n = epsilon, theta_decay = 0
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.constants import EPSILON
from polyergalio.models.layers.basic_layers import RNG, Layer

OUTPUT_TYPES = ("assignment", "distance", "quantized", "coordinates")


def soft_assign(scores: NDArray, temperature: float) -> tuple[NDArray, NDArray]:
    """
    Assignment and free energy over rows of scores, lowest score wins.

    Returns
    -------
    (N, K) assignment probabilities, (N,) free energy per row
    """
    if temperature <= 0:
        winners = np.argmin(scores, axis=-1)
        assignment = np.zeros_like(scores)
        assignment[np.arange(len(scores)), winners] = 1.0
        return assignment, scores[np.arange(len(scores)), winners]

    logits = -scores / temperature
    peak = logits.max(axis=-1, keepdims=True)
    exponent = np.exp(logits - peak)
    total = exponent.sum(axis=-1, keepdims=True)
    free_energy = -temperature * (peak + np.log(total))[:, 0]
    return exponent / total, free_energy


class PrototypeLayer(Layer):
    """
    Shared forward / backward for layers built on K prototype vectors.

    Input shape: (..., input_dim)
    Output shape: (..., K) for assignment and distance, (..., input_dim) for
    quantized, (..., 2) for coordinates. K is declared as a wildcard on
    layers that grow.

    Not adaptive: SGD steps the prototypes at its own rate, and adaptive
    optimizers (Adam) apply a plain gradient step at learning_rate instead of
    their rescaling, so the epsilon and Lloyd step sizes are kept and no
    optimizer state has to follow structural changes.
    """

    positions: Optional[NDArray] = None
    grows: bool = False
    adaptive: bool = False
    parameter_names = ("weights",)
    state_names = ("initialized",)
    cache_names = ("input", "distance_cache", "kernel", "sample_weights", "assignment", "lead_shape", "energy_gradient")

    def __init__(
        self,
        num_prototypes: int,
        input_dimension: int,
        temperature: float = 1.0,
        energy_weight: float = 1.0,
        output_type: str = "assignment",
        learning_rate: float = 1.0,
    ):
        super().__init__()
        if output_type not in OUTPUT_TYPES:
            raise ValueError(f"output_type must be one of {OUTPUT_TYPES}, got {output_type!r}")
        if output_type == "coordinates" and self.positions is None:
            raise ValueError(f"{self.__class__.__name__} has no prototype positions to emit")

        self.input_dimension = input_dimension
        self.temperature = temperature
        self.energy_weight = energy_weight
        self.output_type = output_type
        self.learning_rate = learning_rate

        self.weights = RNG.normal(0.0, 0.1, size=(num_prototypes, input_dimension))
        self.initialized = False
        self.energy = 0.0

        count = None if self.grows else num_prototypes
        width = {"assignment": count, "distance": count, "quantized": input_dimension, "coordinates": 2}[output_type]
        self.declare_shapes(inputs=((input_dimension,),), outputs=((width,),))
        self.zero_gradients()

    @abstractmethod
    def neighborhood(self, distances: NDArray, training_now: bool) -> tuple[Optional[NDArray], NDArray]:
        """
        Returns
        -------
        (K, K) symmetric kernel smoothing distances across prototypes, or None
        for no smoothing; (N,) per-sample energy weights. Both are held
        constant in backward.
        """

    def energy_terms(
        self, distances: NDArray, kernel: Optional[NDArray], assignment: NDArray, free_energy: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Returns
        -------
        (N,) energy per sample, (N, K) its gradient with respect to distances
        """
        gradient = assignment if kernel is None else assignment @ kernel
        return free_energy, gradient

    def initialize(self, x_data: NDArray) -> None:
        """Place prototypes on randomly drawn samples, jittered so duplicates can separate."""
        rows = RNG.choice(len(x_data), size=self.num_prototypes, replace=len(x_data) < self.num_prototypes)
        jitter = 1e-3 * (np.std(x_data, axis=0) + EPSILON)
        self.weights = x_data[rows] + RNG.normal(size=(self.num_prototypes, self.input_dimension)) * jitter
        self.initialized = True

    def distances(self, x_data: NDArray) -> NDArray:
        """Squared euclidean distance of each row to each prototype, (N, K)."""
        squared = (
            np.sum(x_data**2, axis=-1, keepdims=True)
            - 2.0 * x_data @ self.weights.T
            + np.sum(self.weights**2, axis=-1)
        )
        return np.maximum(squared, 0.0)

    def labels(self, x_data: NDArray) -> NDArray:
        """Hard nearest-prototype index per sample."""
        flat = np.asarray(x_data).reshape(-1, self.input_dimension)
        return np.argmin(self.distances(flat), axis=-1).reshape(np.shape(x_data)[:-1])

    def anneal(self, factor: float) -> None:
        """Scale the temperature down, sharpening assignments towards hard clustering."""
        self.temperature *= factor

    def forward(self, x_data: NDArray, training_now: Optional[bool] = None) -> NDArray:
        """
        Parameters
        ----------
        x_data : (..., input_dim)
        training_now : whether running statistics update; None follows train() / eval()

        Returns
        -------
        the output selected at construction, leading axes kept
        """
        training_now = self.training if training_now is None else training_now
        x_data = np.asarray(x_data)
        self.lead_shape = x_data.shape[:-1]
        flat = x_data.reshape(-1, self.input_dimension)
        if not self.initialized:
            self.initialize(flat)

        distances = self.distances(flat)
        kernel, sample_weights = self.neighborhood(distances, training_now)
        scores = distances if kernel is None else distances @ kernel
        assignment, free_energy = soft_assign(scores, self.temperature)
        energy_per_sample, energy_gradient = self.energy_terms(distances, kernel, assignment, free_energy)

        self.input = flat
        self.distance_cache = distances
        self.kernel = kernel
        self.sample_weights = sample_weights
        self.assignment = assignment
        self.energy_gradient = energy_gradient
        self.energy = float(self.energy_weight * np.mean(sample_weights * energy_per_sample) / 2.0)

        if self.output_type == "assignment":
            result = assignment
        elif self.output_type == "distance":
            result = distances
        elif self.output_type == "quantized":
            result = assignment @ self.weights
        else:
            result = assignment @ self.positions
        return result.reshape(*self.lead_shape, -1)

    def backward(self, incoming_grad: Optional[NDArray] = None) -> NDArray:
        """
        Gradient of (downstream loss + self.energy). Pass None to learn from
        the energy alone.

        Returns
        -------
        (..., input_dim) gradient with respect to the input
        """
        count = len(self.input)
        grad_distances = self.energy_weight * self.sample_weights[:, None] * self.energy_gradient / (2.0 * count)
        grad_scores = np.zeros_like(self.distance_cache)
        grad_weights = np.zeros_like(self.weights)

        if incoming_grad is not None:
            grad_output = np.asarray(incoming_grad).reshape(count, -1)
            if self.output_type == "distance":
                grad_distances += grad_output
            else:
                if self.output_type == "quantized":
                    grad_weights += self.assignment.T @ grad_output
                if self.temperature > 0:
                    if self.output_type == "quantized":
                        grad_assignment = grad_output @ self.weights.T
                    elif self.output_type == "coordinates":
                        grad_assignment = grad_output @ self.positions.T
                    else:
                        grad_assignment = grad_output
                    centered = grad_assignment - np.sum(grad_assignment * self.assignment, axis=-1, keepdims=True)
                    grad_scores -= self.assignment * centered / self.temperature

        grad_distances += grad_scores if self.kernel is None else grad_scores @ self.kernel

        grad_weights += 2.0 * (grad_distances.sum(axis=0)[:, None] * self.weights - grad_distances.T @ self.input)
        self.gradient_weights = grad_weights

        grad_input = 2.0 * (grad_distances.sum(axis=1, keepdims=True) * self.input - grad_distances @ self.weights)
        return grad_input.reshape(*self.lead_shape, self.input_dimension)

    def set_weights(self, weights: dict) -> None:
        super().set_weights(weights)
        self.initialized = True
        self.zero_gradients()

    @property
    def num_prototypes(self) -> int:
        return len(self.weights)


class CentroidLayer(PrototypeLayer):
    """
    Soft k-means over num_centroids prototypes; temperature 0 recovers hard
    k-means inertia.
    """

    def __init__(
        self,
        num_centroids: int,
        input_dimension: int,
        temperature: float = 1.0,
        energy_weight: float = 1.0,
        output_type: str = "assignment",
        learning_rate: float = 1.0,
    ):
        super().__init__(num_centroids, input_dimension, temperature, energy_weight, output_type, learning_rate)

    @property
    def num_centroids(self) -> int:
        return self.num_prototypes

    def neighborhood(self, distances: NDArray, training_now: bool) -> tuple[None, NDArray]:
        return None, np.ones(len(distances))


class ParameterlessLayer(PrototypeLayer):
    """
    PLSOM machinery shared by the SOM layers.

    epsilon_n = ||x_n - w_bmu|| / r weights each sample's energy, r being a
    decaying running maximum of winner distance. One neighborhood serves the
    batch: theta = max(theta_max * running_epsilon, theta_min), where
    running_epsilon is a decayed average of batch epsilon. Per-prototype hit
    and error records accumulate in training and decay once per epoch
    (end_epoch), which is also where growing layers restructure.
    """

    state_names = ("scale", "running_epsilon", "hit_map", "node_error", "epoch", "frozen")

    def __init__(
        self,
        num_prototypes: int,
        input_dimension: int,
        theta_min: float,
        theta_max: Optional[float],
        r_decay: float,
        theta_decay: float,
        hit_decay: float,
        temperature: float,
        energy_weight: float,
        output_type: str,
        learning_rate: float,
    ):
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.r_decay = r_decay
        self.theta_decay = theta_decay
        self.hit_decay = hit_decay
        self.scale = 0.0
        self.running_epsilon = 1.0
        self.epoch = 0
        self.frozen = False
        self.structure_trace: list[tuple[int, str, int]] = []
        super().__init__(num_prototypes, input_dimension, temperature, energy_weight, output_type, learning_rate)
        self.hit_map = np.zeros(num_prototypes)
        self.node_error = np.zeros(num_prototypes)

    @abstractmethod
    def neighborhood_kernel(self) -> Optional[NDArray]:
        """(K, K) kernel at the current theta, or None."""

    @property
    def theta(self) -> float:
        return max(self.theta_max * self.running_epsilon, self.theta_min)

    def mean_node_error(self) -> NDArray:
        """Decayed winner distance per decayed hit, in units of a single distance."""
        return self.node_error / np.maximum(self.hit_map, 1.0)

    def neighborhood(self, distances: NDArray, training_now: bool) -> tuple[Optional[NDArray], NDArray]:
        winners = np.argmin(distances, axis=-1)
        winner_distance = np.sqrt(distances[np.arange(len(distances)), winners])
        if training_now:
            self.scale = max(self.scale * self.r_decay, float(winner_distance.max()))
            np.add.at(self.hit_map, winners, 1.0)
            np.add.at(self.node_error, winners, winner_distance)

        epsilon = np.clip(winner_distance / max(self.scale, EPSILON), 0.0, 1.0)
        if training_now:
            self.running_epsilon = (
                self.theta_decay * self.running_epsilon
                + (1.0 - self.theta_decay) * float(epsilon.mean())
            )
        return self.neighborhood_kernel(), epsilon

    def restructure(self) -> Optional[str]:
        """At most one structural change; returns its name, or None. Static maps never change."""
        return None

    def end_epoch(self) -> Optional[str]:
        """
        Call once per pass over the data: restructure, then decay the hit and
        error records.

        Returns
        -------
        the structural change made, or None
        """
        event = self.restructure()
        if event is not None:
            self.structure_trace.append((self.epoch, event, self.num_prototypes))
            self.last_structural_epoch = self.epoch
        self.hit_map *= self.hit_decay
        self.node_error *= self.hit_decay
        self.epoch += 1
        return event

    def after_restructure(self) -> None:
        """Resync counts and buffers after prototypes are added or removed; r re-establishes itself."""
        self.scale = 0.0
        self.zero_gradients()
        self.purge()

    def anneal(self, factor: float) -> None:
        """Scale temperature and theta_max down; theta_max is floored at theta_min."""
        super().anneal(factor)
        self.theta_max = max(self.theta_max * factor, self.theta_min)


class PLSOMLayer(ParameterlessLayer):
    """
    Soft parameterless SOM on a width x height lattice.

    Distances are smoothed by the lattice gaussian before assignment (soft
    topographic vector quantization). Memory is K x K for the neighborhood,
    independent of batch size.
    """

    def __init__(
        self,
        width: int,
        height: int,
        input_dimension: int,
        theta_min: float = 1.0,
        theta_max: Optional[float] = None,
        r_decay: float = 0.99,
        theta_decay: float = 0.9,
        hit_decay: float = 0.9,
        temperature: float = 1.0,
        energy_weight: float = 1.0,
        output_type: str = "assignment",
        learning_rate: float = 1.0,
    ):
        """
        Parameters
        ----------
        theta_decay : weight on the previous running_epsilon when it is updated
            with a batch's mean epsilon; 0 uses the current batch alone
        hit_decay : per-epoch decay on the hit and error records
        learning_rate : prototype step size under adaptive optimizers (Adam);
            SGD uses its own rate. 1.0 is the classic PLSOM step
        """
        self.width = width
        self.height = height
        self.build_lattice()
        super().__init__(
            width * height,
            input_dimension,
            theta_min,
            theta_max if theta_max else float(max(width, height)),
            r_decay,
            theta_decay,
            hit_decay,
            temperature,
            energy_weight,
            output_type,
            learning_rate,
        )

    def build_lattice(self) -> None:
        """Row-major lattice positions (K, 2) and manhattan distances between them (K, K)."""
        rows, cols = np.divmod(np.arange(self.width * self.height), self.width)
        self.positions = np.stack([rows, cols], axis=-1)
        self.lattice = np.abs(rows[:, None] - rows[None, :]) + np.abs(cols[:, None] - cols[None, :])

    def neighborhood_kernel(self) -> NDArray:
        """Lattice gaussian at the current theta, (K, K)."""
        return np.exp(-self.lattice**2 / self.theta**2)

    def weight_grid(self) -> NDArray:
        """Prototypes as (height, width, input_dim)."""
        return self.weights.reshape(self.height, self.width, self.input_dimension)


class GPLSOMLayer(PLSOMLayer):
    """
    Growing and shrinking PLSOM layer; the lattice stays a full rectangle.

    At end_epoch, when the worst node's mean error passes
    GT = -input_dim * ln(spread_factor), a row or column is added near it:
    extrapolated outward at a boundary, interpolated between neighbours in
    the interior. Otherwise the coldest row or column is dropped when its
    share of hits falls below prune_ratio of the average line, or
    reinitialized when min_shape blocks the drop. The assignment and distance
    widths change with the map, so connect downstream of quantized or
    coordinates output, or freeze the structure before connecting.
    """

    grows = True
    state_names = ("growth_threshold", "last_structural_epoch")

    def __init__(
        self,
        width: int,
        height: int,
        input_dimension: int,
        theta_min: float = 1.0,
        theta_max: Optional[float] = None,
        r_decay: float = 0.99,
        theta_decay: float = 0.9,
        hit_decay: float = 0.9,
        spread_factor: float = 0.5,
        growth_anneal: float = 1.0,
        prune_ratio: float = 0.25,
        min_shape: tuple[int, int] = (2, 2),
        max_neurons: Optional[int] = None,
        settle_epochs: int = 3,
        temperature: float = 1.0,
        energy_weight: float = 1.0,
        output_type: str = "assignment",
        learning_rate: float = 1.0,
    ):
        """
        Parameters
        ----------
        spread_factor : GSOM spread factor in (0, 1); lower grows a smaller map
        growth_anneal : per-epoch multiplier on the growth threshold
        prune_ratio : share of the average line's hits below which a line is cold
        min_shape : (rows, cols) floor
        max_neurons : ceiling on prototype count, default four times the start
        settle_epochs : epochs between structural changes
        """
        if not 0.0 < spread_factor < 1.0:
            raise ValueError("spread_factor must be in (0, 1)")
        self.spread_factor = spread_factor
        self.growth_threshold = -input_dimension * np.log(spread_factor)
        self.growth_anneal = growth_anneal
        self.prune_ratio = prune_ratio
        self.min_shape = tuple(min_shape)
        self.max_neurons = max_neurons if max_neurons else 4 * width * height
        self.settle_epochs = settle_epochs
        self.last_structural_epoch = -settle_epochs
        super().__init__(
            width, height, input_dimension, theta_min, theta_max, r_decay, theta_decay, hit_decay,
            temperature, energy_weight, output_type, learning_rate,
        )

    def restructure(self) -> Optional[str]:
        event = None
        if not self.frozen and self.epoch - self.last_structural_epoch >= self.settle_epochs:
            if self.mean_node_error().max() > self.growth_threshold:
                axis, position, edge = self.pick_growth_site()
                if self.growth_fits(axis):
                    self.grow_line(axis, position, edge)
                    event = "grow"
            if event is None:
                cold = self.pick_cold_line(at_floor=False)
                if cold is not None:
                    self.prune_line(*cold)
                    event = "prune"
            if event is None:
                dead = self.pick_cold_line(at_floor=True)
                if dead is not None:
                    self.reinit_line(*dead)
                    event = "reinit"
        self.growth_threshold *= self.growth_anneal
        return event

    def growth_fits(self, axis: int) -> bool:
        """Whether a new row (axis 0) or column (axis 1) stays within max_neurons."""
        addition = (self.height, self.width)[1 - axis]
        return self.num_prototypes + addition <= self.max_neurons

    def pick_growth_site(self) -> tuple[int, int, bool]:
        """
        Returns
        -------
        (axis, position, edge) for a line near the worst node; edge marks an
        outward extension rather than an interior split
        """
        rows, cols = self.height, self.width
        row, col = divmod(int(np.argmax(self.mean_node_error())), cols)
        row_offset = abs(row - (rows - 1) / 2) / max(rows - 1, 1)
        col_offset = abs(col - (cols - 1) / 2) / max(cols - 1, 1)
        axis = 0 if row_offset >= col_offset else 1
        index, extent = (row, rows) if axis == 0 else (col, cols)

        if index == 0:
            return axis, 0, True
        if index == extent - 1:
            return axis, extent, True
        return axis, index + 1, False

    def pick_cold_line(self, at_floor: bool) -> Optional[tuple[int, int]]:
        """
        The coldest row or column below prune_ratio of the average line, on
        axes above min_shape (at_floor False, a prune target) or at it
        (at_floor True, a reinit target).
        """
        shape = (self.height, self.width)
        candidates = []
        for axis in (0, 1):
            if (shape[axis] <= self.min_shape[axis]) != at_floor:
                continue
            line_hits = self.hit_map.reshape(shape).sum(axis=1 - axis)
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

    def grow_line(self, axis: int, position: int, edge: bool) -> None:
        """Insert a row (axis 0) or column (axis 1) at position."""
        lines = np.moveaxis(self.weight_grid(), axis, 0)
        if not edge:
            addition = 0.5 * (lines[position - 1] + lines[position])
        elif position == 0:
            addition = 2 * lines[0] - lines[1]
        else:
            addition = 2 * lines[-1] - lines[-2]

        previous = (self.height, self.width)
        grid = np.insert(self.weight_grid(), position, addition, axis=axis)
        self.hit_map = self.resize_record(self.hit_map, previous, axis, position, inserted=True, fill="neighbour_mean")
        self.node_error = self.resize_record(self.node_error, previous, axis, position, inserted=True, fill="zero")
        self.reshape_map(grid)

        error_lines = np.moveaxis(self.node_error.reshape(self.height, self.width), axis, 0)
        for neighbour in (position - 1, position + 1):
            if 0 <= neighbour < error_lines.shape[0]:
                error_lines[neighbour] *= 0.5

    def prune_line(self, axis: int, position: int) -> None:
        """Drop a row (axis 0) or column (axis 1)."""
        previous = (self.height, self.width)
        grid = np.delete(self.weight_grid(), position, axis=axis)
        self.hit_map = self.resize_record(self.hit_map, previous, axis, position, inserted=False)
        self.node_error = self.resize_record(self.node_error, previous, axis, position, inserted=False)
        self.reshape_map(grid)

    def reinit_line(self, axis: int, position: int) -> None:
        """Move a cold line at the size floor beside the line with the worst error, with fresh records."""
        shape = (self.height, self.width)
        worst = int(np.argmax(np.moveaxis(self.mean_node_error().reshape(shape), axis, 0).sum(axis=1)))
        grid = self.weight_grid().copy()
        lines = np.moveaxis(grid, axis, 0)
        lines[position] = lines[worst] + RNG.uniform(-self.theta_min, self.theta_min, size=lines[position].shape)
        np.moveaxis(self.hit_map.reshape(shape), axis, 0)[position] = 0.0
        np.moveaxis(self.node_error.reshape(shape), axis, 0)[position] = 0.0
        self.reshape_map(grid)

    def reshape_map(self, grid: NDArray) -> None:
        """Adopt a (rows, cols, input_dim) prototype grid and rebuild the lattice."""
        self.height, self.width = grid.shape[:2]
        self.weights = grid.reshape(-1, self.input_dimension)
        self.build_lattice()
        self.after_restructure()

    @staticmethod
    def resize_record(
        record: NDArray, previous: tuple[int, int], axis: int, position: int, inserted: bool, fill: str = "zero"
    ) -> NDArray:
        """
        Insert or delete a line in a per-node record laid out on the previous
        (rows, cols) lattice. fill is "zero" or "neighbour_mean".
        """
        grid = record.reshape(previous)
        if not inserted:
            return np.delete(grid, position, axis=axis).reshape(-1)

        lines = np.moveaxis(grid, axis, 0)
        if fill == "neighbour_mean":
            neighbours = [lines[i] for i in (position - 1, position) if 0 <= i < lines.shape[0]]
            addition = np.mean(neighbours, axis=0)
        else:
            addition = np.zeros(lines.shape[1])
        return np.insert(grid, position, addition, axis=axis).reshape(-1)


class FreePLSOMLayer(ParameterlessLayer):
    """
    Growing and shrinking PLSOM layer with no lattice (neural gas).

    The energy weights each prototype by exp(-rank / theta), rank being its
    distance order to the sample, held constant in backward. At end_epoch,
    near-duplicate prototypes (closer than merge_radius) are merged; then,
    paced by settle_epochs, the worst prototype is split with its nearest
    neighbour when its mean error passes GT = -input_dim * ln(spread_factor),
    or the coldest is pruned below prune_ratio of average hits, or
    reinitialized when min_neurons blocks the prune. The assignment and
    distance widths change with the map.
    """

    grows = True
    state_names = ("growth_threshold", "last_structural_epoch")

    def __init__(self,
                 n_neurons: int,
                 input_dimension: int,
                 theta_min: float = 1.0,
                 theta_max: Optional[float] = None,
                 merge_radius: Optional[float] = None,
                 r_decay: float = 0.99,
                 theta_decay: float = 0.9,
                 hit_decay: float = 0.9,
                 spread_factor: float = 0.5,
                 growth_anneal: float = 1.0,
                 prune_ratio: float = 0.25,
                 min_neurons: int = 2,
                 max_neurons: Optional[int] = None,
                 settle_epochs: int = 3,
                 temperature: float = 1.0,
                 energy_weight: float = 1.0,
                 output_type: str = "assignment",
                 learning_rate: float = 1.0,
                 ):
        """
        Parameters
        ----------
        theta_min, theta_max : neighborhood scale in units of rank; theta_max
            defaults to half the starting count
        merge_radius : weight-space distance below which two prototypes merge;
            defaults to 1e-3 of the initial prototypes' spread
        spread_factor, growth_anneal, prune_ratio, max_neurons, settle_epochs :
            see GPLSOMLayer
        min_neurons : floor on prototype count
        """
        if not 0.0 < spread_factor < 1.0:
            raise ValueError("spread_factor must be in (0, 1)")
        self.merge_radius = merge_radius
        self.spread_factor = spread_factor
        self.growth_threshold = -input_dimension * np.log(spread_factor)
        self.growth_anneal = growth_anneal
        self.prune_ratio = prune_ratio
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons if max_neurons else 4 * n_neurons
        self.settle_epochs = settle_epochs
        self.last_structural_epoch = -settle_epochs
        super().__init__(
            n_neurons, input_dimension, theta_min, theta_max if theta_max else max(1.0, n_neurons / 2),
            r_decay, theta_decay, hit_decay, temperature, energy_weight, output_type, learning_rate,
        )

    @property
    def n_neurons(self) -> int:
        return self.num_prototypes

    def initialize(self, x_data: NDArray) -> None:
        super().initialize(x_data)
        if self.merge_radius is None:
            spread = np.linalg.norm(self.weights - self.weights.mean(axis=0), axis=-1).mean()
            self.merge_radius = float(1e-3 * spread)

    def neighborhood_kernel(self) -> None:
        return None

    def energy_terms(
        self, distances: NDArray, kernel: None, assignment: NDArray, free_energy: NDArray
    ) -> tuple[NDArray, NDArray]:
        ranks = np.argsort(np.argsort(distances, axis=-1), axis=-1)
        rank_weights = np.exp(-ranks / self.theta)
        return np.sum(rank_weights * distances, axis=-1), rank_weights

    def restructure(self) -> Optional[str]:
        event = "merge" if self.merge_duplicates() else None
        if event is None and not self.frozen and self.epoch - self.last_structural_epoch >= self.settle_epochs:
            errors = self.mean_node_error()
            if errors.max() > self.growth_threshold and self.num_prototypes + 1 <= self.max_neurons:
                self.grow_neuron(int(np.argmax(errors)))
                event = "grow"
            elif (cold := self.pick_cold_neuron(at_floor=False)) is not None:
                self.prune_neuron(cold)
                event = "prune"
            elif (dead := self.pick_cold_neuron(at_floor=True)) is not None:
                self.reinit_neuron(dead)
                event = "reinit"
        self.growth_threshold *= self.growth_anneal
        return event

    def prototype_distances(self) -> NDArray:
        """Euclidean distance between every pair of prototypes, diagonal set to inf."""
        gaps = np.linalg.norm(self.weights[:, None] - self.weights[None], axis=-1)
        np.fill_diagonal(gaps, np.inf)
        return gaps

    def merge_duplicates(self) -> int:
        """Drop the colder of every pair closer than merge_radius, one at a time; returns how many."""
        removed = 0
        while self.num_prototypes > self.min_neurons:
            gaps = self.prototype_distances()
            i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
            if gaps[i, j] >= self.merge_radius:
                break
            self.prune_neuron(int(i if self.hit_map[i] < self.hit_map[j] else j))
            removed += 1
        return removed

    def pick_cold_neuron(self, at_floor: bool) -> Optional[int]:
        """
        The coldest prototype below prune_ratio of average hits, when above
        min_neurons (at_floor False, a prune target) or at it (at_floor True,
        a reinit target).
        """
        if (self.num_prototypes <= self.min_neurons) != at_floor:
            return None
        average = self.hit_map.mean()
        if average <= 0:
            return None
        candidate = int(np.argmin(self.hit_map))
        return candidate if self.hit_map[candidate] < self.prune_ratio * average else None

    def grow_neuron(self, worst: int) -> None:
        """Insert a prototype midway between the worst and its nearest neighbour; halve both parents' error."""
        nearest = int(np.argmin(self.prototype_distances()[worst]))
        self.weights = np.vstack([self.weights, 0.5 * (self.weights[worst] + self.weights[nearest])])
        self.hit_map = np.append(self.hit_map, 0.5 * (self.hit_map[worst] + self.hit_map[nearest]))
        self.node_error = np.append(self.node_error, 0.0)
        self.node_error[[worst, nearest]] *= 0.5
        self.after_restructure()

    def prune_neuron(self, index: int) -> None:
        self.weights = np.delete(self.weights, index, axis=0)
        self.hit_map = np.delete(self.hit_map, index)
        self.node_error = np.delete(self.node_error, index)
        self.after_restructure()

    def reinit_neuron(self, index: int) -> None:
        """Move a cold prototype at the size floor beside the worst one, with fresh records."""
        worst = int(np.argmax(self.mean_node_error()))
        self.weights[index] = self.weights[worst] + RNG.uniform(
            -self.theta_min, self.theta_min, size=self.input_dimension
        )
        self.hit_map[index] = 0.0
        self.node_error[index] = 0.0
        self.after_restructure()
