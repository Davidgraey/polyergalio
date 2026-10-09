"""
Incremental decomposition Layer mechanisms for use in our Network objects
"""
from typing import Optional
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.constants import EPSILON
from polyergalio.models.layers.basic_layers import Layer
from polyergalio.utilities import init_standardize, standardize, unstandardize, update_running_standardize


OUTPUT_FORMS = ("orthogonalized", "whitened", "projection")

class DecompositionForm(Enum):
    ORTHO = "orthogonalized"
    WHITENED = "whitened"
    PROJECTION = "projection"


class IncrementalSVDLayer(Layer):
    """
    Based on "Incremental Singular Value Decomposition for Streaming and Scalable Network Embedding" (2025, Zhang)

    Streams rows into a truncated SVD and emits their moving-standardized orthogonalization, (..., output_dimension).
    Rows must already be normalized / standardized per feature

    output_form picks what is emitted, always with the fixed width output_dimension; columns beyond the current rank
    are zeroed out (sparsification) so may impact downstream layers

    - "orthogonalized": the orthogonalized rows sqrt(n) x V S^-1 V', needs output_dimension equal to input_dimension
    - "whitened": the leading principal scores sqrt(n) x V S^-1, unit variance and uncorrelated
    - "projection": the leading principal scores x V, with variance equal to the eigenvalues

    Each merge sets eigenvector_change, the largest angle in radians between a previously held eigenvector and its
    update. As the eigenvectors converge towards a shared representation iteratively, the largest angle decreses

    freeze() makes training passes behave like eval ones, so repeated epochs do not merge the same rows again.

    backward passes the gradient through the current linear map to the input

    Early batches are projected with factors that have seen little data, so will be unstable early.
    Start with a larger batch, or set decay_rate below 1 so the early statistics fade.
    """
    parameter_names = ()
    state_names = (
        "singular_values", "eigenvectors", "num_seen_samples", "energy", "eigenvector_change", "frozen",
        "x_means", "x_stds",
    )
    cache_names = ("transform", "lead_shape")

    def __init__(
        self,
        input_dimension: int,
        num_components: Optional[int] = None,
        output_dimension: Optional[int] = None,
        output_form: DecompositionForm = DecompositionForm.ORTHO,
        standardize: bool = True,
        decay_rate: float = 0.99,
        rank_tolerance: float = 1e-10,
    ):
        """
        Parameters
        ----------
        input_dimension : size of the last axis
        num_components : rank kept, at most input_dimension; None keeps all, which makes the output exact
        output_dimension : width of the output, at most num_components below input_dimension; None is input_dimension
        output_form : "orthogonalized", "whitened" or "projection"
        standardize : subtract the moving mean and divide by the moving std of the projected rows; they are tracked either way
        decay_rate : in (0, 1]; weight on previously seen data at each batch, in the factors and in the statistics
        rank_tolerance : singular values below this fraction of the largest are ignored
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.num_components = input_dimension if num_components is None else num_components
        self.output_dimension = output_dimension = input_dimension if output_dimension is None else output_dimension

        if not 0 < self.num_components <= input_dimension:
            raise ValueError(f"num_components must be in [1, {input_dimension}], got {num_components}")
        if self.output_dimension < input_dimension and self.output_dimension > self.num_components:
            raise ValueError(f"output_dimension {output_dimension} needs num_components of at least as many")

        if output_form == "orthogonalized" and output_dimension != input_dimension:
            raise ValueError("output_form 'orthogonalized' needs output_dimension equal to input_dimension")

        self.output_form = output_form
        self.standardize = standardize
        self.decay_rate = decay_rate
        self.rank_tolerance = rank_tolerance

        self.singular_values = np.zeros(0)
        self.eigenvectors = np.zeros((input_dimension, 0))

        self.num_seen_samples: int = 0
        self.energy = 0.0
        self.eigenvector_change = float(np.pi / 2)

        self.frozen = False
        self.x_means = np.zeros(output_dimension)
        self.x_stds = np.ones(output_dimension)

        self.declare_shapes(inputs=((input_dimension,),), outputs=((output_dimension,),))
        self.zero_gradients()

    @property
    def eigenvalues(self) -> NDArray:
        """(k,) eigenvalues of X`@ X / n over the rows seen, largest first."""
        return self.singular_values ** 2 / max(self.num_seen_samples, 1.0)

    @property
    def explained_variance_ratio(self) -> NDArray:
        """(k,) share of the total squared norm of the rows seen carried by each component."""
        return self.singular_values**2 / max(self.energy, EPSILON)

    def transform_matrix(self, num_rows: Optional[float] = None) -> NDArray:
        """
        Parameters
        ----------
        num_rows : weight of the rows behind the factors; the number seen when None

        Returns
        -------
        (input_dimension, output_dimension) map from rows to the output with the output_form applied to the transform
        """
        if self.output_form == DecompositionForm.PROJECTION:
            basis = self.eigenvectors
        else:
            num_rows = self.num_seen_samples if num_rows is None else num_rows
            basis = self.eigenvectors / self.singular_values * np.sqrt(num_rows)

        if self.output_form == DecompositionForm.ORTHO:
            return basis @ self.eigenvectors.T

        else: # DecompositionForm.WHITENED
            transform = np.zeros((self.input_dimension, self.output_dimension))
            kept = min(basis.shape[1], self.output_dimension)
            transform[:, :kept] = basis[:, :kept]

        return transform

    def freeze(self) -> None:
        """Stop merging and updating statistics; training passes then act like eval ones."""
        self.frozen = True
        self.transform = None

    def unfreeze(self) -> None:
        """Resume merging in training passes."""
        self.frozen = False

    def merge_batch(self, batch: NDArray) -> None:
        """
        Fold rows into the factors (Zhang et al) - Orthogonalizing only new columns.
        Rows seen before are weighted by decay_rate, and their count and energy by decay_rate**2.

        Parameters
        ----------
        batch : (m, input_dimension) new rows
        """
        new_columns = batch.T
        projection = self.eigenvectors.T @ new_columns
        residual = new_columns - self.eigenvectors @ projection
        correction = self.eigenvectors.T @ residual
        residual -= self.eigenvectors @ correction
        projection += correction
        orthogonal, triangular = np.linalg.qr(residual)

        rank, width = self.singular_values.size, triangular.shape[0]
        core = np.zeros((rank + width, rank + len(batch)))
        core[:rank, :rank] = np.diag(self.decay_rate * self.singular_values)
        core[:rank, rank:] = projection
        core[rank:, rank:] = triangular
        rotation, values, _ = np.linalg.svd(core, full_matrices=False)

        keep = min(self.num_components, int(np.sum(values > self.rank_tolerance * values[0])))
        right = (np.hstack([self.eigenvectors, orthogonal]) @ rotation)[:, :keep]
        pivot = np.argmax(np.abs(right), axis=0)

        right = right * np.where(right[pivot, np.arange(keep)] < 0, -1.0, 1.0)

        held = min(self.eigenvectors.shape[1], keep)
        if held:
            alignment = np.abs(np.sum(self.eigenvectors[:, :held] * right[:, :held], axis=0))
            self.eigenvector_change = float(np.max(np.arccos(np.clip(alignment, 0.0, 1.0))))
        else:
            self.eigenvector_change = float(np.pi / 2)
        self.singular_values = values[:keep]
        self.eigenvectors = right
        self.num_seen_samples *= self.decay_rate**2
        self.energy = self.decay_rate**2 * self.energy + float(np.sum(batch**2))

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.transform = None

    def forward(self, x_data: NDArray, training_now: Optional[bool] = None) -> NDArray:
        """
        Parameters
        ----------
        x_data : (..., input_dimension)
        training_now : whether the batch updates the factors and statistics, unless frozen; None follows train() / eval()

        Returns
        -------
        (..., output_dimension) rows in the chosen output_form, standardized with the moving statistics
        """
        training_now = self.training & self.frozen

        self.num_samples = x_data.shape[:-1]
        flat = x_data.reshape(-1, self.input_dimension)

        if training_now:
            self.merge_batch(flat)
            self.transform = self.transform_matrix(self.num_seen_samples + len(flat))

        elif self.num_seen_samples == 0:
            raise ValueError("nothing has been merged yet; run a training pass first")

        elif self.transform is None:
            self.transform = self.transform_matrix()

        projected = flat @ self.transform
        if training_now and self.num_seen_samples == 0:
            init_standardize(self, projected)

        elif training_now:
            batch_std = projected.std(axis=0, ddof=min(1, len(projected) - 1))
            update_running_standardize(self, projected.mean(axis=0), batch_std, len(projected))

        if self.standardize:
            projected = standardize(self, projected)
        return projected.reshape(*self.num_samples, -1)

    def backward(self, incoming_grad: NDArray = None) -> NDArray:
        """
        The factors and statistics are held fixed, so the layer is the linear map `transform` and a fixed scale.

        Parameters
        ----------
        incoming_grad : (..., output_dimension) gradient with respect to the output, or None for nothing downstream

        Returns
        -------
        (..., input_dimension) gradient with respect to the input
        """
        grad = np.asarray(incoming_grad).reshape(-1, self.output_dimension)
        if self.standardize:
            grad = grad / (self.x_stds + EPSILON)
        return (grad @ self.transform.T).reshape(*self.num_samples, self.input_dimension)

    def inverse(self, output: NDArray) -> NDArray:
        """
        Parameters
        ----------
        output : (..., output_dimension), as forward returns

        Returns
        -------
        (..., input_dimension) rows in the original units, exact when no component was truncated
        """
        projected = np.asarray(output)
        if self.standardize:
            projected = unstandardize(self, projected)
        return projected @ np.linalg.pinv(self.transform_matrix())

    def __str__(self):
        return (
            f"Incremental SVD of {self.input_dimension} to {self.output_dimension}, "
            f"rank {self.singular_values.size}/{self.num_components}"
        )

    def __repr__(self):
        return self.__str__()
