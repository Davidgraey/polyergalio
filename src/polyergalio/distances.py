import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist
from polyergalio.utilities import preformat_expected_shapes
from polyergalio.models.constants import EPSILON, GLOBAL_DTYPE



########### DISTANCES ###########
def manhattan_distance(x: NDArray, y: NDArray, summed: bool = True) -> NDArray:
    """
    aka Chebyshev
    :param x:
    :param y:
    :return:
    """
    x, y = preformat_expected_shapes(x, y)
    # sum(np.abs(x-y), axis=?)
    if summed:
        return np.sum(np.abs(x - y), axis=-1)
    else:
        return np.abs(x - y)


def grid_manhattan_distance(row_a, col_a, row_b, col_b):
    return abs(row_a - row_b) + abs(col_a - col_b)


def euclidian_distance(x: NDArray, y: NDArray) -> NDArray:
    """
    for size (5000, 100), per sample, np.sqrtsum method took 1.07ms
    linalg.norm took 1.21ms
    numpy 1.20.3
    :param x:
    :param y:
    :return:
    """
    x, y = preformat_expected_shapes(x, y)
    if x.ndim == 3 and x.shape[1] == 1 and y.ndim == 2:
        return pairwise_euclidian_distance(x[:, 0, :], y)
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))  # + EPSILON
    # return cdist(x, y, metric='euclidean')


def pairwise_euclidian_distance(x: NDArray, y: NDArray) -> NDArray:
    """
    Euclidean distance between every row of x and every row of y, from the expansion
    |x - y|^2 = |x|^2 - 2 x.y + |y|^2 so memory is (N, M) rather than (N, M, D).

    Parameters
    ----------
    x : (N, D) rows
    y : (M, D) rows

    Returns
    -------
    (N, M) distances
    """
    x = np.asarray(x, dtype=GLOBAL_DTYPE)
    y = np.asarray(y, dtype=GLOBAL_DTYPE)
    center = y.mean(axis=0)
    x = x - center
    y = y - center
    squared = np.sum(x**2, axis=1, keepdims=True) - 2.0 * x @ y.T + np.sum(y**2, axis=1)
    return np.sqrt(np.maximum(squared, 0.0))


def norm_euclidian_distance(x: NDArray, y: NDArray) -> NDArray:
    # numerator = (np.linalg.norm((x - np.mean(x)) - (y - np.mean(y))) ** 2)
    # denom = (np.linalg.norm(x - np.mean(x)) ** 2 + np.linalg.norm(y - np.mean(y)) ** 2)
    # return 0.5 * (numerator / denom)

    vx = np.var(x, axis=-1)
    vy = np.var(y, axis=-1)
    return 0.5 * (vx / (vy + vx))


def mahalonobis_distance(x: NDArray, y: NDArray) -> NDArray:
    """

    multivariate equivilant of euclidian distance, comparing point to distribution
    :param x:
    :param y:
    :return:
    """
    stacked_vecs = np.vstack([x, y])
    covariance = np.cov(stacked_vecs.T)
    inv_covariance = np.linalg.inv(covariance)
    delta = x - y
    # may have to fix reshape for multi-dim...
    return np.sqrt(np.einsum('nj,jk,nk->n', delta, inv_covariance, delta)).reshape(x.shape[0], -1)


def hamming_distance(x: NDArray, y: NDArray) -> NDArray:
    """
    :param x:
    :param y:
    :return:
    """
    # axis?
    return np.count_nonzero(x != y, axis=-1)


def cosine_distance(x: NDArray, y: NDArray) -> NDArray:
    """
    inverse cosine similarity
    :param x:
    :param y:
    :return:
    """
    x, y = preformat_expected_shapes(x, y)
    return 1 - cosine_similarity(x, y)


# +---------Similarity---------
def cosine_similarity(x: NDArray, y: NDArray) -> NDArray:
    """
    requires numpy v1.7+ for 'where' in np.divide()
    numpy array is [samples, dimension_1, dimension_2, ...]
    we will compute cosine distance per samples input_dimension
        input array is [500, 5] - output will be [500, 1]
    :param x:
    :param y:
    :return:
    """
    dot = np.sum(x * y, axis=-1)
    x_norm = np.sqrt(np.sum(x * x, axis=-1))
    y_norm = np.sqrt(np.sum(y * y, axis=-1))

    denom = x_norm * y_norm
    return dot / np.maximum(denom, EPSILON)


def jaccard_similarity(x: NDArray, y: NDArray) -> NDArray:
    """aka Jaccard Index
    for use in boolean / on-hot / binary array
    """
    return np.bitwise_and(x, y).sum(axis=-1) / np.bitwise_or(x, y).sum(axis=-1)
