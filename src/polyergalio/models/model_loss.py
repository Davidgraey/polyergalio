"""
ERROR and LOSS FUNCTIONS all done in Numpy--
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.activations import sigmoid, softmax
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.constants import DECISION_TYPES, EPSILON
from polyergalio.models.heads.decision_utilities import decision_type_ids, masked_softmax

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
        Normalize a mask to broadcast against an elementwise array.

        Parameters
        ----------
        mask : (..., 1) or (...,) with 1 for real content, 0 for padding
        reference : the elementwise array the mask will multiply

        Returns
        -------
        (..., 1) mask, or None if mask is None
        """
        if mask is None:
            return None
        positions = mask[..., 0] if mask.ndim == reference.ndim else mask
        return positions[..., None]

    @staticmethod
    def _position_mask(mask: Optional[NDArray], reference: NDArray) -> Optional[NDArray]:
        """
        Normalize a mask to match an already-reduced, per-position array.

        Parameters
        ----------
        mask : (..., 1) or (...,) with 1 for real content, 0 for padding
        reference : an array one axis narrower than the raw prediction,
            e.g. a per-position loss after reducing over the class axis

        Returns
        -------
        mask matching reference's shape, or None if mask is None
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


class DifferenceLoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
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


class RMSELoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
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
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        self.prediction = prediction
        self.targets = targets

        dot = np.sum(prediction * targets, axis=-1)
        norm_p = np.linalg.norm(prediction, axis=-1)
        norm_t = np.linalg.norm(targets, axis=-1)

        self.cos = dot / (norm_p * norm_t + 1e-8)
        per_position = 1 - self.cos
        self.position_mask = self._position_mask(mask, per_position)

        if self.position_mask is not None:
            self.valid_elements = max(self.position_mask.sum(), 1.0)
            return np.sum(per_position * self.position_mask) / self.valid_elements

        self.valid_elements = per_position.size
        return np.sum(per_position) / self.valid_elements

    def backward(self):
        p = self.prediction
        t = self.targets

        norm_p = np.linalg.norm(p, axis=-1, keepdims=True)
        norm_t = np.linalg.norm(t, axis=-1, keepdims=True)

        grad = (
            p * np.sum(p * t, axis=-1, keepdims=True) / (norm_p**3 * norm_t)
            - t / (norm_p * norm_t)
        )
        if self.position_mask is not None:
            grad = grad * self.position_mask[..., None]
        return grad / self.valid_elements


class CrossEntropyLoss(Loss):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        self.prediction = prediction
        self.targets = targets

        if self.task == ClassificationTask.MULTINOMIAL:
            shifted = prediction - np.max(prediction, axis=-1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
            cls = np.argmax(targets, axis=-1)
            correct = np.take_along_axis(shifted, cls[..., None], axis=-1)[..., 0]
            loss = -correct + log_sum_exp
            self.reduce_mask = self._position_mask(mask, loss)
            self.valid_elements = (
                max(self.reduce_mask.sum(), 1.0) if self.reduce_mask is not None else loss.size
            )

        elif self.task == ClassificationTask.BINARY:
            loss = (
                np.maximum(0, prediction)
                - targets * prediction
                + np.log(1 + np.exp(-np.abs(prediction)))
            )
            self.reduce_mask = self._mask_column(mask, loss)
            self.valid_elements = (
                self._valid_elements(self.reduce_mask, loss) if self.reduce_mask is not None else loss.size
            )

        elif self.task == ClassificationTask.MULTILABEL:
            log_sum_exp = np.log(1 + np.exp(prediction))
            loss = log_sum_exp - targets * prediction
            self.reduce_mask = self._mask_column(mask, loss)
            self.valid_elements = (
                self._valid_elements(self.reduce_mask, loss) if self.reduce_mask is not None else loss.size
            )

        if self.reduce_mask is not None:
            return np.sum(loss * self.reduce_mask) / self.valid_elements
        return np.mean(loss)

    def backward(self):
        """
        dL / d(logits). forward() consumes logits and applies its own
        log-softmax / log-sigmoid, so the squashing belongs here too.
        """
        if self.task == ClassificationTask.MULTINOMIAL:
            grad = softmax(self.prediction) - self.targets
            if self.reduce_mask is not None:
                grad = grad * self.reduce_mask[..., None]
            return grad / self.valid_elements

        grad = sigmoid(self.prediction) - self.targets
        if self.reduce_mask is not None:
            grad = grad * self.reduce_mask
        return grad / self.valid_elements

    def __call__(self, predictions: NDArray, targets: NDArray, mask: Optional[NDArray] = None):
        return self.forward(predictions, targets, mask)


class SparseCrossEntropyLoss(Loss):
    """
    Multinomial cross-entropy on logits with integer class labels (no need for a one-hot)

    Predictions are (..., classes) logits and targets are (...) integers.
    A mask is (...) or (..., 1) with 1 for real content, and masked positions still need valid labels.
    """

    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        """
        Parameters
        ----------
        prediction : (..., classes) logits
        targets : (...) integer class labels
        mask : optional per-position masking to identify targets

        Returns
        -------
        mean loss over the unmasked positions
        """
        labels = np.asarray(targets, dtype=int)
        shifted = prediction - np.max(prediction, axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        total = np.sum(exponentials, axis=-1, keepdims=True)
        self.probabilities = exponentials / total
        self.labels = labels
        correct = np.take_along_axis(shifted, labels[..., None], axis=-1)[..., 0]
        loss = np.log(total[..., 0]) - correct
        self.reduce_mask = self._position_mask(mask, loss)
        self.valid_elements = max(self.reduce_mask.sum(), 1.0) if self.reduce_mask is not None else loss.size
        if self.reduce_mask is not None:
            return np.sum(loss * self.reduce_mask) / self.valid_elements
        return np.mean(loss)

    def backward(self):
        """ dL / d(logits): softmax - 1  at each label, masked and averaged """
        grad = self.probabilities.copy()
        flat = grad.reshape(-1, grad.shape[-1])
        flat[np.arange(flat.shape[0]), self.labels.reshape(-1)] -= 1.0
        if self.reduce_mask is not None:
            grad = grad * self.reduce_mask[..., None]
        return grad / self.valid_elements


class MultiHeadLoss(Loss):
    def __init__(self, losses: list[Loss], weights=None):
        super().__init__()
        self.losses = losses
        self.weights = weights or [1.0] * len(losses)

    def forward(self, predictions: NDArray, targets, mask: Optional[list] = None):
        self.predictions = predictions
        self.targets = targets
        masks = mask if mask is not None else [None] * len(self.losses)

        total = 0.0
        self.last_losses = []
        for w, loss, yhat, y, m in zip(self.weights, self.losses, predictions, targets, masks):
            L = loss.forward(yhat, y, m)
            self.last_losses.append(L)
            total += w * L
        return total

    def backward(self):
        return [w * loss.backward() for w, loss in zip(self.weights, self.losses)]

    def __call__(self, predictions: NDArray, targets: NDArray, mask: Optional[list] = None):
        return self.forward(predictions, targets, mask)


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

# ------------------------------------------------------------------
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
