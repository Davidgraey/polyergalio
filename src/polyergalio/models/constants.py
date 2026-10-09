from enum import Enum

import numpy as np
from numpy.typing import NDArray

GLOBAL_DTYPE = np.float64
GLOBAL_COMPLEX_DTYPE = np.complex128
EPSILON = 1e-15
ANY_SHAPE = (None,)

MASKED_LOGIT = 1e-6

# TOLERANCES AND LIMITS FOR SCALED CONJUGATE GRADIENT DESCENT (scg_regression)
FLOAT_TOLERANCE = 1e-30
SIGMA_ZERO = 1e-6
LAMBDA_MAX = 1e24
LAMBDA_MIN = 1e-20

# TOKEN CONSTANTS
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
CLS_ID = 3
SEP_ID = 4
MARK_ID = 5
TOKEN_OFFSET = 6


class DECISION_TYPES(Enum):
    BINARY = 0
    CHOICE = 1
    SCORE = 2


class ClassificationTask(Enum):
    BINARY = "binary"
    MULTINOMIAL = "multinomial"
    MULTILABEL = "multilabel"


class Reductions(Enum):
    MEAN = "mean"
    SUM = "sum"
    NONE = None


def determine_classification_task(targets: NDArray) -> ClassificationTask:
    """
    Takes one-hot targets
    Parameters
    ----------
    targets :

    Returns
    -------

    """
    if len(targets.shape) == 1:
        targets = targets.reshape(-1, 1)

    num_numeric_classes = np.ptp(targets)
    last_dim = targets.shape[-1]
    num_classes = max(num_numeric_classes, last_dim)
    class_per_sample = max(np.sum(targets, axis=-1))

    if class_per_sample == 1:
        if num_classes <= 2:
            task = ClassificationTask("binary")

        elif num_classes > 2:
            task = ClassificationTask("multinomial")

    elif class_per_sample > 1:
        task = ClassificationTask("multilabel")

    return task
