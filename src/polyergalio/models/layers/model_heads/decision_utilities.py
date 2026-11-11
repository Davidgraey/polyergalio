from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.activations import masked_softmax
from polyergalio.models.constants import (
    DECISION_TYPES,
    EPSILON
)

# -------------    question types    -------------------------------
def decision_type_ids(decisiontypes) -> NDArray:
    """
    Numeric type ids for a batch of question types. Convert our enums back to int for the sake of models

    Parameters
    ----------
    decisiontypes : DECISION_TYPES members, their integer values, or a mix

    Returns
    -------
    (batch,) int array of DECISION_TYPES values
    """
    return np.array([DECISION_TYPES(kind).value for kind in np.ravel(decisiontypes)], dtype=int)


def decision_confidence(probabilities: NDArray, token_mask: NDArray) -> NDArray:
    """
    one minus the normalized entropy for each row

    Returns
    -------
    (batch,) in [0, 1]; 1 for a one-hot answer or a single-option row
    """
    num_options = token_mask.sum(axis=-1)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, EPSILON)), axis=-1
    )
    return np.where(num_options > 1, 1 - entropy / np.log(np.maximum(num_options, 2)), 1.0)


def decode_decisions(probabilities: NDArray, decisiontypes: list[DECISION_TYPES]) -> NDArray:
    """
    Read each row's answer according to its own question type.

    Returns
    -------
    (batch,) BINARY -> P(true) with options ordered [false, true],
    CHOICE -> chosen option index, SCORE -> expected level
    """
    type_ids = decision_type_ids(decisiontypes)
    levels = np.arange(probabilities.shape[-1])
    return np.select(
        [type_ids == DECISION_TYPES.CHOICE.value, type_ids == DECISION_TYPES.SCORE.value],
        [np.argmax(probabilities, axis=-1), probabilities @ levels],
        default=probabilities[:, min(1, probabilities.shape[-1] - 1)],
    )


def decision_correct(
    decisions: NDArray, answer: NDArray, decisiontype: list[DECISION_TYPES]
) -> NDArray:
    """
    Per-row correctness of decoded decisions against the answer option index.

    Returns
    -------
    (batch,) True where right; a score is right when its expected level rounds
    to the answer, a binary when P(true) falls on the answer's side of 0.5
    """
    binary_pick = (decisions >= 0.5).astype(int)
    picked = np.where(
        decision_type_ids(decisiontype) == DECISION_TYPES.BINARY.value,
        binary_pick,
        np.rint(decisions).astype(int),
    )
    return picked == answer


# -------------    calibration    ----------------------------------
def option_bucket(decisiontype: DECISION_TYPES | int, num_options: int) -> str:
    """Temperature key by question type and option count, e.g. "choice:3-5"."""
    if num_options <= 2:
        size = "2"
    elif num_options <= 5:
        size = "3-5"
    elif num_options <= 10:
        size = "6-10"
    else:
        size = "11+"
    return f"{DECISION_TYPES(decisiontype).name.lower()}:{size}"


def row_buckets(token_mask: NDArray, decisiontypes: list[DECISION_TYPES]) -> list[str]:
    """Temperature bucket name for every row."""
    return [
        option_bucket(int(t), int(k))
        for t, k in zip(decision_type_ids(decisiontypes), token_mask.sum(axis=-1))
    ]


def fit_temperatures(
    logits: NDArray,
    answer: NDArray,
    token_mask: NDArray,
    decisiontypes: list[DECISION_TYPES],
    grid: NDArray,
) -> dict[str, float]:
    """
    Post-hoc temperature per (type, option count) bucket, chosen by grid
    search to minimize held-out NLL

    Returns
    -------
    bucket name -> temperature
    """
    buckets = np.array(row_buckets(token_mask, decisiontypes))
    temperatures = {}
    for one_bucket in np.unique(buckets):
        rows = buckets == one_bucket
        answer_rows = answer[rows][:, None]
        losses = []
        for t in grid:
            _probs = masked_softmax(logits=logits[rows],
                                    token_mask=token_mask[rows],
                                    temperature=t)
            _per_row = np.take_along_axis(_probs, answer_rows, axis=-1)
            _nll = -1 * (np.mean(np.log(np.maximum(_per_row, EPSILON))))

            losses.append(_nll)

        temperatures[str(one_bucket)] = float(grid[int(np.argmin(losses))])
    return temperatures


def calibrated_probabilities(
    logits: NDArray,
    token_mask: NDArray,
    decisiontypes: list[DECISION_TYPES],
    temperatures: dict[str, float],
) -> NDArray:
    """Masked softmax with each row scaled by its bucket's temperature (1.0 if unfitted)."""

    row_temperature = np.array(
        [temperatures.get(b, 1.0) for b in row_buckets(token_mask, decisiontypes)]
    )

    return masked_softmax(logits, token_mask, row_temperature)

