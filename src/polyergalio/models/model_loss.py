"""
ERROR and LOSS FUNCTIONS all done in Numpy--
"""
import copy
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Callable, Tuple

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.activations import sigmoid, softmax
from polyergalio.models.constants import LOG_TWO_PI, ClassificationTask, DECISION_TYPES, EPSILON
from polyergalio.models.layers.model_heads.decision_utilities import decision_type_ids, masked_softmax

loss_dictionary, derivative_dictionary = {}, {}

loss_func = lambda f: loss_dictionary.setdefault(f.__name__, f)
derivative = lambda f: derivative_dictionary.setdefault(f.__name__, f)


class Loss(ABC):
    def __init__(self):
        self.targets = None
        self.prediction = None
        self.mask_column = None
        self.valid_elements = None

    @staticmethod
    def _mask_column(mask: Optional[NDArray], reference: NDArray) -> Optional[NDArray]:
        """
        Normalize a mask to broadcast against a per-position or elementwise array.

        Parameters
        ----------
        mask : (..., 1) or (...,) with 1 for real content, 0 for padding
        reference : array with a trailing feature axis, size 1 for a per-position loss

        Returns
        -------
        (..., 1) mask in reference's dtype or None if mask is None
        """
        if mask is None:
            return None
        positions = mask[..., 0] if mask.ndim == reference.ndim else mask
        return positions[..., None].astype(reference.dtype)

    @staticmethod
    def _position_mask(mask: Optional[NDArray], reference: NDArray) -> Optional[NDArray]:
        """
        Normalize a mask to a per-position array.

        Parameters
        ----------
        mask : (...,) or (..., 1) with 1 for real content, 0 for padding
        reference : (...,) per-position array

        Returns
        -------
        (...,) mask in reference's dtype or None if mask is None
        """
        if mask is None:
            return None
        positions = mask[..., 0] if mask.ndim == reference.ndim + 1 else mask
        return positions.astype(reference.dtype)

    @staticmethod
    def _valid_elements(mask_column: NDArray, reference: NDArray) -> float:
        return max(mask_column.sum() * reference.shape[-1], 1.0)

    @abstractmethod
    def forward(self, prediction: NDArray, targets: NDArray, mask: Optional[NDArray] = None) -> float | NDArray:
        pass

    @abstractmethod
    def backward(self) -> NDArray:
        """
        Returns dL / d(y_pred)
        """
        pass

    def __call__(self, predictions: NDArray, targets: NDArray, mask: Optional[NDArray] = None) -> float | NDArray:
        return self.forward(predictions, targets, mask)


class MultiHeadLossWrapper(Loss):
    """
    An "Aggregator" loss function to take in multiple loss functions

        losses:  a list of Loss functions (already instantiated)
        weights: a list or array of weights; 1:1 to
    """
    def __init__(self, losses: list[Loss], weights=None):
        super().__init__()
        self.losses = losses

        if weights:
            assert len(losses) == len(weights)
            self.weights = weights
        else:
            self.weights = [1.0] * len(losses)

    def forward(self, predictions: NDArray, targets, mask: Optional[list] = None):
        self.predictions = predictions
        self.targets = targets
        masks = mask if mask is not None else [None] * len(self.losses)

        total = 0.0
        self.last_losses = []

        for weight, loss_fn, prediction, target, mask in zip(self.weights, self.losses, predictions, targets, masks):
            L = loss_fn.forward(prediction, target, mask)
            self.last_losses.append(L)
            total += weight * L
        return total

    def backward(self):
        return [weight * loss_fn.backward() for weight, loss_fn in zip(self.weights, self.losses)]

    def __call__(self, predictions: NDArray, targets: NDArray, mask: Optional[list] = None):
        return self.forward(predictions, targets, mask)


class MultiStepLossWrapper(Loss):
    """
    Weighted sum over steps of a base loss between each step's prediction and the target that many steps ahead.

    Prediction at position t is scored against the target at [t : t + m]

    Uses MutliHeadLossWrapper as the aggregator

    Parameters
    ----------
    loss_class: callable returning a new scalar Loss, called once per step because a loss keeps its last inputs

    num_future_steps: M
    decay:  decay rate coefficient. step m is weighted decay ** (m - 1)
    loss_kwargs: keyword arguments to instantiate our Loss class
    weights : explicit (steps,) weights per future state

    Attributes
    ----------
    step_weights : (steps,)
    step_losses : (steps,) unweighted loss of each step from the latest forward
    """
    def __init__(self,
                 loss_class: Callable[[], Loss],
                 num_future_steps: int,
                 decay: float = 0.5,
                 loss_kwargs: dict = {},
                 weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.future_steps = num_future_steps
        self.step_weights = np.asarray(
            weights if weights is not None else decay ** np.arange(num_future_steps),
            dtype=np.float64)

        if self.step_weights.shape != (num_future_steps,):
            raise ValueError(f"Need {num_future_steps} weights; provided ws shape == {self.step_weights.shape}")

        # each step gets its own loss function initalized
        self.losses = [loss_class(**loss_kwargs) for _ in range(num_future_steps)]
        self.step_losses = np.zeros(num_future_steps)

    def forward(self, prediction: NDArray, targets: NDArray, mask: Optional[NDArray] = None) -> float:
        """
        Parameters
        ----------
        prediction: (batch, length, steps, ...) from a SpectreCausalGenerator with num_future_steps > 1
        targets: (batch, length, ...) the target at every position
        mask: (batch, length) or (batch, length, 1), 1 for a real position

        Returns
        -------
        weighted sum of the per-step losses
        """
        self.prediction = prediction
        length = prediction.shape[1]
        valid = np.ones(prediction.shape[:2]) if mask is None else (mask[..., 0] if mask.ndim == 3 else mask)
        reaches = range(1, min(self.future_steps, length - 1) + 1)

        self.heads = MultiHeadLossWrapper(self.losses[:len(reaches)],
                                          weights=self.step_weights[:len(reaches)].tolist())

        # tricky indexing here - :length-m is the start of unseen; m-1 is the first active
        # TODO: add an assertion for targets earlier to catch if things haven't been aligned properly
        total = self.heads.forward(
            [prediction[:, :length - m, m - 1] for m in reaches],
            [targets[:, m:] for m in reaches],
            [valid[:, :length - m] * valid[:, m:] for m in reaches],
        )
        self.step_losses[:] = 0.0
        self.step_losses[:len(reaches)] = self.heads.last_losses
        return float(total)

    def backward(self) -> NDArray:
        """Gradient wrt the prediction, same shape."""
        gradient = np.zeros_like(self.prediction)
        length = self.prediction.shape[1]

        for m, step_gradient in enumerate(self.heads.backward(), start=1):
            gradient[:, :length - m, m - 1] = step_gradient
        return gradient


class DifferenceLoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        """
        Just the absolute difference from prediction to target
        Parameters
        ----------
        prediction: logit or linear output from a model
        targets: target values. shape == prediction
        mask: mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid

        Returns
        -------
        mean error over the batch - (mask identifies valid content)
        """
        self.prediction = prediction
        self.targets = targets
        diff = np.abs(targets - prediction)
        self.mask_column = self._mask_column(mask, diff)
        return diff * self.mask_column if self.mask_column is not None else diff

    def backward(self):
        grad = np.sign(self.prediction - self.targets)
        return grad * self.mask_column if self.mask_column is not None else grad


class MSELoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        """
        Mean Squared Error loss fn
        Parameters
        ----------
        prediction: logit or linear output from a model
        targets: target values. shape == prediction
        mask: mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid

        Returns
        -------
        mean error over the batch - (mask identifies valid content)

        """
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        self.mask_column = self._mask_column(mask, diff)

        if self.mask_column is not None:
            self.valid_elements = self._valid_elements(self.mask_column, diff)
            return np.sum((diff ** 2) * self.mask_column) / self.valid_elements
        return np.mean(diff ** 2)

    def backward(self):
        """forward means over every element, so the reduction is size not shape[0]"""
        diff = self.prediction - self.targets
        if self.mask_column is not None:
            return (2 / self.valid_elements) * diff * self.mask_column
        return (2 / self.prediction.size) * diff


class HuberLoss(Loss):
    """
    Pseudo-Huber Loss
    L2 squared loss as loss values are < delta , and L1 (abs error) as loss is high
    (decreasing / reducing the impact of large values)

    smoothed approximation of Huber
    """
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta
        self.delta_sq = delta ** 2

    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        """
        Pseudo-Huber loss for regression
        Parameters
        ----------
        prediction: logit or linear output from a model
        targets: target values. shape == prediction
        mask: mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid

        Returns
        -------
        mean error over the batch - (mask identifies valid content)
        """
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        loss = self.delta_sq * (np.sqrt(1 + (diff / self.delta) ** 2) - 1)
        self.mask_column = self._mask_column(mask, diff)

        if self.mask_column is not None:
            self.valid_elements = self._valid_elements(self.mask_column, diff)
            return np.sum(loss * self.mask_column) / self.valid_elements
        return np.mean(loss)

    def backward(self):
        diff = self.prediction - self.targets
        grad = diff / np.sqrt(1 + (diff / self.delta) ** 2)
        if self.mask_column is not None:
            return grad * self.mask_column / self.valid_elements
        return grad / diff.size


class RMSELoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        """
        Root Mean Squared Error loss fn
        Parameters
        ----------
        prediction: logit or linear output from a model
        targets: target values. shape == prediction
        mask: mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid

        Returns
        -------
        mean error over the batch - (mask identifies valid content)
        """
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        self.mask_column = self._mask_column(mask, diff)

        if self.mask_column is not None:
            self.valid_elements = self._valid_elements(self.mask_column, diff)
            self.rmse = np.sqrt(np.sum((diff ** 2) * self.mask_column) / self.valid_elements)
        else:
            self.rmse = np.sqrt(np.mean(diff ** 2))
        return self.rmse

    def backward(self):
        diff = self.prediction - self.targets
        if self.mask_column is not None:
            return diff * self.mask_column / (self.valid_elements * (self.rmse + EPSILON))
        N = diff.size
        return diff / (N * (self.rmse + EPSILON))


class SSELoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        """
        Sum Square of Errors as Loss fn
        Parameters
        ----------
        prediction: logit or linear output from a model
        targets: target values. shape == prediction
        mask: mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid

        Returns
        -------
        Summed square of errors over the batch - (mask identifies valid content)
        """
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        self.mask_column = self._mask_column(mask, diff)
        if self.mask_column is not None:
            return np.sum((diff ** 2) * self.mask_column)
        return np.sum(diff ** 2)

    def backward(self):
        grad = 2 * (self.prediction - self.targets)
        return grad * self.mask_column if self.mask_column is not None else grad


class MAELoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        """
        Mean Absolute Error (loss)
        Parameters
        ----------
        prediction: logit or linear output from a model
        targets: target values. shape == prediction
        mask: mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid

        Returns
        -------
        mean loss over the batch - (mask identifies valid content)
        """
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        self.mask_column = self._mask_column(mask, diff)

        if self.mask_column is not None:
            self.valid_elements = self._valid_elements(self.mask_column, diff)
            return np.sum(np.abs(diff) * self.mask_column) / self.valid_elements
        return np.mean(np.abs(diff))

    def backward(self):
        diff = self.prediction - self.targets
        if self.mask_column is not None:
            return np.sign(diff) * self.mask_column / self.valid_elements
        return np.sign(diff) / diff.size


class CosineLoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None) -> NDArray:
        """

        Parameters
        ----------
        prediction: logit or linear output from a model
        targets: target values. shape == prediction
        mask: mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid

        Returns
        -------
        mean loss over the batch - (mask identifies valid content)
        """
        self.prediction = prediction
        self.targets = targets

        dot = np.sum(prediction * targets, axis=-1)
        norm_p = np.linalg.norm(prediction, axis=-1)
        norm_t = np.linalg.norm(targets, axis=-1)

        self.cos = dot / (norm_p * norm_t + 1e-8)
        per_position = (1 - self.cos)[..., None]
        self.mask_column = self._mask_column(mask, per_position)

        if self.mask_column is not None:
            self.valid_elements = self._valid_elements(self.mask_column, per_position)
            return np.sum(per_position * self.mask_column) / self.valid_elements

        self.valid_elements = per_position.size
        return np.mean(per_position)

    def backward(self):
        p = self.prediction
        t = self.targets

        norm_p = np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-8)
        norm_t = np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-8)

        grad = (
            p * np.sum(p * t, axis=-1, keepdims=True) / (norm_p**3 * norm_t)
            - t / (norm_p * norm_t)
        )
        if self.mask_column is not None:
            grad = grad * self.mask_column
        return grad / self.valid_elements


class CrossEntropyLoss(Loss):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def forward(self, prediction: NDArray, targets: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        prediction: logit or linear output from a model
        targets: target values. shape == prediction
        mask: mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid

        Returns
        -------
        mean loss over the batch - (mask identifies valid content)
        """
        self.prediction = prediction
        self.targets = targets

        if self.task == ClassificationTask.MULTINOMIAL:
            shifted = prediction - np.max(prediction, axis=-1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
            cls = np.argmax(targets, axis=-1)
            correct = np.take_along_axis(shifted, cls[..., None], axis=-1)
            loss = -correct + log_sum_exp

        elif self.task == ClassificationTask.BINARY:
            loss = (
                np.maximum(0, prediction)
                - targets * prediction
                + np.log(1 + np.exp(-np.abs(prediction)))
            )

        elif self.task == ClassificationTask.MULTILABEL:
            log_sum_exp = np.log(1 + np.exp(prediction))
            loss = log_sum_exp - targets * prediction

        self.mask_column = self._mask_column(mask, loss)
        self.valid_elements = (
            self._valid_elements(self.mask_column, loss) if self.mask_column is not None else loss.size
        )

        if self.mask_column is not None:
            return np.sum(loss * self.mask_column) / self.valid_elements
        return np.mean(loss)

    def backward(self):
        """
        dL / d(logits). forward() consumes logits and applies its own log-softmax / log-sigmoid
        we'll need to apply that here, too
        """
        activation = softmax if self.task == ClassificationTask.MULTINOMIAL else sigmoid
        grad = activation(self.prediction) - self.targets
        if self.mask_column is not None:
            grad = grad * self.mask_column
        return grad / self.valid_elements

    def __call__(self, predictions: NDArray, targets: NDArray, mask: Optional[NDArray] = None):
        return self.forward(predictions, targets, mask)


class SparseCrossEntropyLoss(Loss):
    """
    Multinomial cross-entropy on logits with integer class labels (no need for a one-hot)

    Predictions are (..., classes) logits and targets are (...) integers.

    A mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid
    """

    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        """
        Parameters
        ----------
        prediction: logit or linear output from a model
        targets: target values, where class is carried by integer labels. (num_samples, ..., 1)
        mask: mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid

        Returns
        -------
        mean loss over the batch - (mask identifies valid content)
        """
        labels = np.asarray(targets, dtype=int)
        shifted = prediction - np.max(prediction, axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        total = np.sum(exponentials, axis=-1, keepdims=True)
        self.probabilities = exponentials / total
        self.labels = labels
        correct = np.take_along_axis(shifted, labels[..., None], axis=-1)
        loss = np.log(total) - correct
        self.mask_column = self._mask_column(mask, loss)
        self.valid_elements = (
            self._valid_elements(self.mask_column, loss) if self.mask_column is not None else loss.size
        )
        if self.mask_column is not None:
            return np.sum(loss * self.mask_column) / self.valid_elements
        return np.mean(loss)

    def backward(self):
        """ dL / d(logits): softmax - 1  at each label, masked and averaged """
        grad = self.probabilities.copy()
        flat = grad.reshape(-1, grad.shape[-1])
        flat[np.arange(flat.shape[0]), self.labels.reshape(-1)] -= 1.0
        if self.mask_column is not None:
            grad = grad * self.mask_column
        return grad / self.valid_elements


class MatryoshkaLoss(Loss):
    """
    Apply a base loss to nested subsections of the embedding axis and sum them.

    Parameters
    ----------
    loss : base loss, copied per subsection so each keeps its own cached state
    dimensions : increasing subsection lengths, low to high (64, 128, 256, 512)
    weights : per-subsection weighting. Uniform by default - this will be norm'd to sum to 1 within the class
    """
    def __init__(self, loss: Loss,
                 dimensions: Sequence[int],
                 weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.dimensions = tuple(dimensions)
        weights = np.ones(len(self.dimensions)) if weights is None else np.asarray(weights, dtype=float)
        self.weights = weights / weights.sum()
        self.losses = [copy.deepcopy(loss) for _ in self.dimensions]

    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        """

        Parameters
        ----------
        prediction: logit or linear output from a model
        targets: target values. shape == prediction
        mask: mask is (...) or (..., 1) with 1 for real content, 0 for padding or invalid

        Returns
        -------
        Weighted sum of each of the fixed-length subsections or subselections of prediction
            (mask identifies valid content)
        """
        self.prediction = prediction
        self.targets = targets
        return sum(
            weight * loss(prediction[..., :dimension], targets[..., :dimension], mask)
            for weight, dimension, loss in zip(self.weights, self.dimensions, self.losses)
        )

    def backward(self):
        grad = np.zeros_like(self.prediction)
        for weight, dimension, loss in zip(self.weights, self.dimensions, self.losses):
            grad[..., :dimension] += weight * loss.backward()
        return grad


class DecisionLoss(Loss):
    """
    Loss for Decision system (System-One style model)
    Masked softmax cross-entropy over each row's options, shared by every
    question type. Score rows can add an ordinal penalty, the expected squared
    distance between the predicted and true level, so near misses cost less.
    """

    def __init__(self, ordinal_weight: float = 0.0):
        super().__init__()
        self.ordinal_weight = ordinal_weight
        self.probabilities = None

    def forward(self,
                prediction: NDArray,
                targets: NDArray,
                mask: Optional[NDArray] = None,
                decisiontypes: Optional[NDArray] = None,
                ) -> float:
        """
        Parameters
        ----------
        prediction : (batch, options) logits
        targets : (batch,) answer option index
        mask : (batch, options) token_mask; all options real if None
        decisiontypes : (batch,) DECISION_TYPES members or values, needed for the ordinal term
        """
        self.prediction = prediction
        self.targets = np.asarray(targets, dtype=int)
        self.token_mask = np.ones_like(prediction, dtype=bool) if mask is None else mask.astype(bool)
        self.probabilities = masked_softmax(prediction, self.token_mask)
        self.onehot = np.eye(prediction.shape[-1])[self.targets]

        picked = np.take_along_axis(self.probabilities, self.targets[:, None], axis=-1)[:, 0]
        loss = -np.log(np.maximum(picked, EPSILON))

        self.ordinal_rows = np.zeros(len(self.targets), dtype=bool)
        if self.ordinal_weight and decisiontypes is not None:
            self.ordinal_rows = decision_type_ids(decisiontypes) == DECISION_TYPES.SCORE.value
            levels = np.arange(prediction.shape[-1])
            self.distance = (levels[None, :] - self.targets[:, None]) ** 2 * self.token_mask
            expected = np.sum(self.probabilities * self.distance, axis=-1)
            loss = loss + self.ordinal_weight * expected * self.ordinal_rows

        return float(np.mean(loss))

    def backward(self) -> NDArray:
        grad = self.probabilities - self.onehot
        if self.ordinal_rows.any():
            expected = np.sum(self.probabilities * self.distance, axis=-1, keepdims=True)
            ordinal = self.probabilities * (self.distance - expected)
            grad = grad + self.ordinal_weight * ordinal * self.ordinal_rows[:, None]
        return grad * self.token_mask / len(self.targets)

    def __call__(self, predictions, targets, mask=None, decisiontypes=None) -> float:
        return self.forward(predictions, targets, mask, decisiontypes)


class MixtureNLLLoss(Loss):
    """
    Mean over valid positions of -log p(target), summed over the Z latent dimensions.

    Parameters
    ----------
    num_components : J, the number of gaussians we constructed
    latent_dimension : Z the dimensionality of the latent projection space

    Attributes
    ----------
    nll : (...,) per-position negative log-likelihood from the latest forward, 0 where masked
    """
    def __init__(self, num_components: int, latent_dimension: int):
        super().__init__()
        self.num_components = num_components
        self.latent_dimension = latent_dimension
        self.nll = None

    def forward(self,
                pred_pi: NDArray,
                pred_mu: NDArray,
                pred_sigma: NDArray,
                targets: NDArray,
                mask: Optional[NDArray] = None) -> float:
        """
        NOTE -- the loss expects the UNPACKED dense output from the MixtureDensityHead

        Parameters
        ----------
        prediction : (..., J + 2*J*Z) packed head output

        targets : (..., Z)
        mask : (...,) or (..., 1), 1 for a scored position and 0 for padding

        Returns
        -------
        mean negative log-likelihood per scored position
        """
        self.targets = targets

        self.log_pi = pred_pi
        self.mu = pred_mu
        self.sigma = pred_sigma

        log_density, self.responsibilities = self.mixture_log_density(self.log_pi, self.mu, self.sigma, targets)
        weights = self._position_mask(mask, log_density)

        self.weights = np.ones_like(log_density) if weights is None else weights
        self.scored = max(float(self.weights.sum()), 1.0)

        self.nll = -log_density * self.weights
        return float(self.nll.sum() / self.scored)

    def backward(self) -> NDArray:
        """Gradient with respect to the packed prediction, same shape."""
        scale = (self.weights / self.scored)[..., None, None]
        resp = self.responsibilities[..., None]
        difference = self.targets[..., None, :] - self.mu

        grad_log_pi = -self.responsibilities * (self.weights / self.scored)[..., None]
        grad_mu = -scale * resp * difference / self.sigma ** 2
        grad_sigma = scale * resp * (self.sigma ** 2 - difference ** 2) / self.sigma ** 3

        batch_shape = grad_log_pi.shape[:-1]
        return np.concatenate([grad_log_pi, grad_mu.reshape(*batch_shape, -1), grad_sigma.reshape(*batch_shape, -1)], axis=-1)

    @staticmethod
    def mixture_log_density(log_pi: NDArray, mu: NDArray, sigma: NDArray, targets: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Parameters
        ----------
        log_pi : (..., J)
        mu, sigma : (..., J, Z)
        targets : (..., Z)

        Returns
        -------
        tuple (
            log_density : (...,) log p(target) under the mixture,
            responsibility: weight of each component to the target : (..., J)
            )
        """
        standardized = (targets[..., None, :] - mu) / sigma
        component = log_pi - 0.5 * (LOG_TWO_PI + 2.0 * np.log(sigma) + standardized ** 2).sum(axis=-1)

        peak = component.max(axis=-1, keepdims=True)
        weights = np.exp(component - peak)
        total = weights.sum(axis=-1, keepdims=True)
        return (peak + np.log(total))[..., 0], weights / total


# -------  derivative functions -----------------------------------------------------------

@derivative
def mse_derivative(prediction, targets, **kwargs) -> float | NDArray:
    return prediction - targets


@derivative
def mae_derivative(prediction: NDArray, targets: NDArray) -> NDArray:
    return (prediction - targets) / np.abs(prediction - targets)


@derivative
def rmse_derivative(prediction: NDArray, targets: NDArray) -> NDArray:
    return np.abs(targets - prediction) / np.sqrt(prediction.shape[0])


@derivative
def cross_entropy_derivative(
    prediction: NDArray, targets: NDArray, **kwargs
) -> NDArray | float:
    """BACKPROP TRICKS for sigmoid / softmax: combine"""
    sample_count = targets.shape[0]
    return (prediction - targets) / sample_count


# ------------------------------------------------------------------
# free-function wrappers, delegating to the Loss classes above. Added so supervised.* (scg_regression, tree_models)
# can import cross_entropy / mse as plain functions rather than instantiate a class.
# TODO: modify these later on to make the supervised models work with the actual classes, not aliased
def mse(prediction: NDArray, targets: NDArray, **kwargs) -> float:
    return MSELoss()(prediction, targets)


def cross_entropy(
    prediction: NDArray, targets: NDArray, task: ClassificationTask, **kwargs
) -> float:
    return CrossEntropyLoss(task)(prediction, targets)
