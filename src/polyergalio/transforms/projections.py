from numpy.typing import NDArray
import numpy as np

from polyergalio.models.constants import EPSILON


def mca(
        data: NDArray,
        top_k_components: int = 3
) -> NDArray:
    """
    Multiple correspondence analysis of a one-hot table.

    Correspondence analysis works on the correspondence matrix P = N / N.sum()
    and compares it against what independence would predict. The expected table
    is the outer product of the row and column MASSES -- the marginals of P, not
    the raw counts -- so that expectation and observation are on the same scale
    and the standardized residuals sum to the table's total inertia.

    Returns row principal coordinates, one row per sample, the same orientation
    pca() returns.

    Parameters
    ----------
    data : one-hot encoded array, (num_samples, num_indicator_columns)
    top_k_components : number of axes to keep

    Returns
    -------
    (num_samples, min(top_k_components, num_axes)) projection
    """
    total = np.sum(data)
    probability = data / total

    # masses, i.e. the marginals of P. Both sum to 1.
    row_mass = np.sum(probability, axis=1)
    column_mass = np.sum(probability, axis=0)

    expected = np.outer(row_mass, column_mass)
    residuals = (probability - expected) / np.sqrt(np.maximum(expected, EPSILON))

    left, singular, _ = np.linalg.svd(residuals, full_matrices=False)

    keep = min(top_k_components, singular.size)

    # row principal coordinates: F = D_r^(-1/2) U D
    scaling = 1.0 / np.sqrt(np.maximum(row_mass, EPSILON))
    return scaling[:, None] * left[:, :keep] * singular[:keep]


def pca(
        data: NDArray,
        top_k_components: int = 3

) -> NDArray:
    """
    Principal component analysis on the correlation matrix.

    Standardizes each feature, then projects onto the leading eigenvectors of
    the covariance of the standardized data.

    Parameters
    ----------
    data : (num_samples, ...) - any trailing shape is flattened per sample
    top_k_components : number of components to keep. Clamped to the number of
        available features.

    Returns
    -------
    (num_samples, min(top_k_components, num_features)) projection
    """
    org_shape = data.shape

    # a zero-variance feature has no scale to divide by -- so we have to pad/clip floor
    deviation = np.std(data, axis=0)
    standardized_data = (data - np.mean(data, axis=0)) / np.maximum(
        deviation, EPSILON
    )

    # TODO: this forces 2D -- this will get covaraince across all values - for future, implement a mechanim across
    #  the axes, ... cov_list = [np.cov(slice_2d) for slice_2d in data]
    standardized_data = standardized_data.reshape(org_shape[0], -1)
    num_features = standardized_data.shape[1]

    covariance_matrix = np.cov(standardized_data, ddof=1, rowvar=False)
    # np.cov collapses to a scalar for a single feature, so restore the matrix
    covariance_matrix = np.atleast_2d(covariance_matrix)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # eigh returns ascending, so the reverse slice takes the top k descending
    keep = min(top_k_components, num_features)
    top_components = np.argsort(eigenvalues)[: -(keep + 1) : -1]
    # take copy for memory optimization
    principal_components = eigenvectors[:, top_components].copy()

    return standardized_data @ principal_components
