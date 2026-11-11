import numpy as np
from numpy.typing import NDArray

from polyergalio.distances import pairwise_euclidian_distance
from polyergalio.models.constants import GLOBAL_DTYPE


def silhouette_score(x_data: NDArray, prediction: NDArray, block_rows: int = 1024) -> float:
    """
    Compute the mean silhouette score for a clustering prediction
    Vectorized over samples
    High scores (0 - 1) mean a distinctive structure
    Low scores (-1 - 0) mean a disperse, poor structure distinction

    Parameters
    ----------
    x_data : our original data samples
    prediction : predictied labels as integers
    block_rows : samples scored per distance block, which bounds memory at block_rows x N

    Returns
    -------
    mean silhouette score across the dataset -1 to 1
    """
    # Input validation
    if int(x_data.shape[0]) != int(prediction.shape[0]):
        raise ValueError("we need a label for every sample in x_data")

    unique_labels, inv = np.unique(prediction, return_inverse=True)
    num_samples = int(x_data.shape[0])
    num_clusters = int(unique_labels.size)
    if num_clusters < 2 or num_samples == 0:
        return 0.0

    cluster_sizes = np.bincount(inv, minlength=num_clusters)
    one_hot = np.zeros((num_samples, num_clusters), dtype=GLOBAL_DTYPE)
    one_hot[np.arange(num_samples), inv] = 1.0

    summed_distances = np.empty((num_samples, num_clusters), dtype=GLOBAL_DTYPE)
    for start in range(0, num_samples, block_rows):
        stop = min(start + block_rows, num_samples)
        block = pairwise_euclidian_distance(x_data[start:stop], x_data)
        block[np.arange(stop - start), np.arange(start, stop)] = 0.0
        summed_distances[start:stop] = block @ one_hot

    # intra-cluster mean distance for each sample (exclude self)
    own_sizes = cluster_sizes[inv]
    own_sums = summed_distances[np.arange(num_samples), inv]
    is_singleton = own_sizes <= 1
    with np.errstate(divide="ignore", invalid="ignore"):
        # I hate that npwhere doesn't like kwargs.
        intra_dist = np.where(
            own_sizes > 1,
            own_sums / (own_sizes - 1),
            0.0
        )

    # average distance from each sample to each cluster
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_distances = summed_distances / np.maximum(cluster_sizes, 1)

    # excluding self-references to the same cluster -this works better than setting infinity.
    eye = np.eye(num_clusters, dtype=bool)
    row_mask = eye[inv]
    masked_means = np.where(row_mask, np.inf, mean_distances)
    nearest_clust_dist = masked_means.min(axis=1)

    # score per sample
    largest_delta = np.maximum(intra_dist, nearest_clust_dist)
    with np.errstate(divide="ignore", invalid="ignore"):
        # I hate that `where` doesn't like kwargs.
        scores = np.where(
            largest_delta > 0,
            (nearest_clust_dist - intra_dist) / largest_delta,
            0.0
        )

    scores = np.where(is_singleton, 0.0, scores)

    return np.mean(scores)


def cluster_means(x_data: NDArray, inverse: NDArray, num_clusters: int) -> tuple[NDArray, NDArray]:
    """
    Parameters
    ----------
    x_data : (N, D) samples
    inverse : (N,) cluster index of each sample, every cluster in range(num_clusters) occupied
    num_clusters : cluster count

    Returns
    -------
    (K,) cluster sizes, (K, D) cluster means
    """
    counts = np.bincount(inverse, minlength=num_clusters)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    sums = np.add.reduceat(np.asarray(x_data, dtype=GLOBAL_DTYPE)[order], starts, axis=0)
    return counts, sums / counts[:, None]


def davies_bouldin_index(x_data: NDArray, prediction: NDArray) -> float:
    """
    DB index quantifies the within-cluster spread to inter-cluster distances
    Lower DB scores indicate a more compact, more seperated cluster

    Parameters
    ----------
    x_data : our original data samples
    prediction : predictied labels as integers

    Returns
    -------
    DB score - lower is "better"
    """
    unique_labels, inverse = np.unique(prediction, return_inverse=True)
    num_clusters = len(unique_labels)
    counts, centroids = cluster_means(x_data, inverse, num_clusters)

    own_distances = np.linalg.norm(x_data - centroids[inverse], axis=1)
    spread = np.bincount(inverse, weights=own_distances, minlength=num_clusters) / counts

    centroid_distances = pairwise_euclidian_distance(centroids, centroids)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = (spread[:, None] + spread[None, :]) / centroid_distances
    np.fill_diagonal(ratios, 0.0)

    return float(np.mean(ratios.max(axis=1)))


def calinski_harabasz_index(x_data: NDArray, prediction: NDArray) -> float:
    """
    Ratio of inter-cluster dispersion over the intra-cluster dispersion

    Higher CH index scores show us more dense, separated clusters

    Parameters
    ----------
    x_data : our original data samples
    prediction : predictied labels as integers

    Returns
    -------
    CH index - variance ratio criterion
    """
    unique_labels, inverse = np.unique(prediction, return_inverse=True)
    n_samples = x_data.shape[0]
    n_clusters = len(unique_labels)
    if n_clusters < 2 or n_samples == 0:
        return 0.0

    overall_mean = np.mean(x_data, axis=0)
    cluster_sizes, cluster_means_ = cluster_means(x_data, inverse, n_clusters)

    inter_dispersion = np.sum(cluster_sizes[:, None] * (cluster_means_ - overall_mean) ** 2)
    intra_dispersion = np.sum((x_data - cluster_means_[inverse]) ** 2)

    return (inter_dispersion / intra_dispersion) * ((n_samples - n_clusters) / (n_clusters - 1))


def cubic_clustering_criterion(x_data: NDArray, prediction: NDArray) -> float:
    """
    Cubic Clustering Criterion (CCC) for estimating the number of clusters - Sarle, W.S. (1983)

    Compares the observed R-squared of a clustering against the R-suare expected from points spread uniformly over the
    data's own principal-axis extent, with the same number of clusters

    Values above ~2-3 suggest real structure
    values near or below 0 suggest the clustering has no more structure than a random split of uniform data.

    Parameters
    ----------
    x_data : our original data samples
    prediction : predictied labels as integers

    Returns
    -------
    CCC statistic - higher is "better", roughly 0 for no structure
    """
    unique_labels, inverse = np.unique(prediction, return_inverse=True)
    n_samples, n_features = x_data.shape
    n_clusters = len(unique_labels)
    # same degenerate cases as calinski_harabasz_index, plus we need at
    # least 2 samples to form a covariance matrix
    if n_clusters < 2 or n_samples < 2:
        return 0.0

    x_centered = x_data - x_data.mean(axis=0)
    total_scatter = x_centered.T @ x_centered
    trace_total = np.trace(total_scatter)
    if trace_total <= 0:
        return 0.0

    _, means = cluster_means(x_data, inverse, n_clusters)
    within_scatter = np.sum((x_data - means[inverse]) ** 2)

    r_squared = 1.0 - within_scatter / trace_total

    # eigenvalues of the covariance matrix describe the data's own
    # principal-axis spread -- this is the "hyperbox" the null hypothesis
    # (uniformly distributed points) is compared against
    eigenvalues = np.linalg.eigvalsh(total_scatter / (n_samples - 1))
    eigenvalues = np.sort(eigenvalues)[::-1]
    sqrt_eigs = np.sqrt(np.maximum(eigenvalues, 0.0))
    s = np.where(sqrt_eigs > 0, sqrt_eigs, 1.0)
    vv = np.prod(s)

    # scale s down to a per-cluster cell size, then decide how many of the
    # largest (>= cell size) axes to treat exactly (hypercube) vs.
    # approximate (hypersphere tail) -- Sarle's mixed approximation
    cell_scale = (vv / n_clusters) ** (1.0 / n_features)
    u = s / cell_scale
    k1 = int(np.sum(u >= 1.0))
    p1 = min(k1, n_clusters - 1)

    if 0 < p1 < n_features:
        v1 = np.prod(s[:p1])
        cell_scale = (v1 / n_clusters) ** (1.0 / p1)
        u = s / cell_scale
        b1 = np.sum(1.0 / (n_samples + u[:p1]))
        tail = u[p1:]
        b2 = np.sum(tail ** 2 / (n_samples + tail))
        b_total = b1 + b2
        effective_dims = p1
    else:
        b_total = np.sum(1.0 / (n_samples + u))
        effective_dims = n_features

    sum_u2 = np.sum(u ** 2)
    if sum_u2 <= 0:
        return 0.0
    expected_r_squared = 1.0 - (b_total / sum_u2) * (
        (n_samples - n_clusters) ** 2 / n_samples
    ) * (1.0 + 4.0 / n_samples)

    ratio_num = 1.0 - expected_r_squared
    ratio_den = 1.0 - r_squared
    if ratio_num <= 0 or ratio_den <= 0:
        return 0.0

    ccc = np.log(ratio_num / ratio_den) * (
        np.sqrt(n_samples * effective_dims / 2.0)
        / (0.001 + expected_r_squared) ** 1.2
    )
    return float(ccc)


# ---- topology preservation and quantization ----
def partition_coefficient(memberships: NDArray, normalized: bool = False) -> float:
    """
    Fuzzy partition coefficient: the mean over samples of the summed squared memberships

    Parameters
    ----------
    memberships : (N, K) membership of each sample in each cluster, rows summing to 1
    normalized : rescale so scores compare across K

    Returns
    -------
    coefficient from 1/K (fully fuzzy) to 1 (crisp), or 0 to 1 if normalized - higher is "better"
    """
    num_clusters = memberships.shape[1]
    coefficient = float(np.mean(np.sum(memberships**2, axis=1)))
    if not normalized:
        return coefficient
    if num_clusters < 2:
        return float("nan")
    return 1.0 - (num_clusters / (num_clusters - 1)) * (1.0 - coefficient)


def xie_beni_index(
    x_data: NDArray, centroids: NDArray, memberships: NDArray, fuzzifier: float = 2.0
) -> float:
    """
    Xie-Beni (XB) index: membership-weighted 'compactness' over the smallest distance(sq) between centroids
    XB minimizes; lower is better

    Parameters
    ----------
    x_data : (N, D) data samples
    centroids : (K, D) cluster centers
    memberships : (N, K) membership of each sample in each cluster
    fuzzifier : exponent applied to the memberships

    Returns
    -------
    XB index
    """
    num_samples, num_clusters = memberships.shape
    if x_data.shape[0] != num_samples or centroids.shape[0] != num_clusters:
        raise ValueError("memberships must be (samples, centroids)")
    if num_clusters < 2:
        return float("nan")

    distances = pairwise_euclidian_distance(x_data, centroids) ** 2
    compactness = np.sum(memberships**fuzzifier * distances)

    separation = pairwise_euclidian_distance(centroids, centroids) ** 2
    np.fill_diagonal(separation, np.inf)
    smallest = separation.min()
    if smallest <= 0:
        return float("inf")
    return float(compactness / (num_samples * smallest))


def quantization_error(x_data: NDArray, prototypes: NDArray, squared_distance: bool = False) -> float:
    """
    Mean euclidean distance from each sample to its nearest prototype
    Lower error = better

    Parameters
    ----------
    x_data : (N, D) data samples
    prototypes : (K, D) reference vectors
    squared_distance : average of our squared distance instead

    Returns
    -------
    mean distance to the winning prototype
    """
    nearest = pairwise_euclidian_distance(x_data, prototypes).min(axis=1)

    return float(np.mean(nearest**2 if squared_distance else nearest))


def neighbor_ranks(points: NDArray) -> NDArray:
    """
    Parameters
    ----------
    points : (N, D) samples

    Returns
    -------
    (N, N) integer matrix whose [i, j] entry is the distance rank of j among
    the neighbors of i; a sample has rank 0 to itself and 1 to its nearest
    """
    distances = pairwise_euclidian_distance(points, points)
    np.fill_diagonal(distances, -np.inf)
    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(len(points))[:, None]
    ranks[rows, order] = np.arange(len(points))[None, :]
    return ranks


def check_neighbor_inputs(count: int, target_count: int, num_neighbors: int) -> None:
    if target_count != count:
        raise ValueError("source and target need the same number of samples")
    if not 0 < num_neighbors < count / 2:
        raise ValueError("num_neighbors must be in (0, N / 2)")


def distance_block(points: NDArray, start: int, stop: int) -> NDArray:
    """
    Parameters
    ----------
    points : (N, D) samples
    start, stop : rows to measure

    Returns
    -------
    (stop - start, N) distances to every sample
    """
    block = pairwise_euclidian_distance(points[start:stop], points)
    block[np.arange(stop - start), np.arange(start, stop)] = np.inf
    return block


def neighbor_rank_penalty(source_dist: NDArray, target_dist: NDArray, num_neighbors: int) -> float:
    """
    Summed rank excess, beyond num_neighbors, in the source space of each sample's nearest neighbors in the target
    space.

    Parameters
    ----------
    source_dist : (B, N) distance blocks over the same rows
    target_dist : (B, N) distance blocks over the same rows
    num_neighbors : neighborhood size k
    """
    neighbors = np.argpartition(target_dist, num_neighbors - 1, axis=1)[:, :num_neighbors]
    penalty = 0.0
    for column in range(num_neighbors):
        own = np.take_along_axis(source_dist, neighbors[:, column : column + 1], axis=1)
        ranks = np.sum(source_dist < own, axis=1) + 1
        penalty += float(np.sum(np.maximum(ranks - num_neighbors, 0)))
    return penalty


def rank_error(source: NDArray, target: NDArray, num_neighbors: int, block_rows: int = 256) -> float:
    """
    Normalized rank penalty for samples that are among the nearest num_neighbors in target but not in source

    Parameters
    ----------
    source : (N, D) space whose ranks are penalized
    target : (N, E) space whose neighborhoods are checked
    num_neighbors : neighborhood size k, below N / 2
    block_rows : samples processed per distance block, which bounds memory at block_rows x N

    Returns
    -------
    value in [0, 1], 0 when no sample enters a neighborhood from outside
    """
    count = len(source)
    check_neighbor_inputs(count, len(target), num_neighbors)

    penalty = 0.0
    for start in range(0, count, block_rows):
        stop = min(start + block_rows, count)
        penalty += neighbor_rank_penalty(
            distance_block(source, start, stop), distance_block(target, start, stop), num_neighbors
        )
    return penalty * 2.0 / (count * num_neighbors * (2 * count - 3 * num_neighbors - 1))


def trustworthiness(x_data: NDArray, embedding: NDArray, num_neighbors: int = 5, block_rows: int = 256) -> float:
    """
    How much of each sample's neighborhood in the embedding is also a neighborhood in the data (Venna & Kaski)
    maximize 0.0 to 1.0 (higher is better)

    Parameters
    ----------
    x_data : (N, D) original samples
    embedding : (N, E) low dimensional representation (latent embedding)
    num_neighbors : neighborhood size
    block_rows : samples processed per distance block

    Returns
    -------
    float - trustworthiness score in [0, 1]
    """
    return 1.0 - rank_error(x_data, embedding, num_neighbors, block_rows)


def continuity(x_data: NDArray, embedding: NDArray, num_neighbors: int = 5, block_rows: int = 256) -> float:
    """
    How much of each sample's neighborhood in the data survives in the embedding (Venna & Kaski)
    maximize 0.0 to 1.0 (higher is better)

    Parameters
    ----------
    x_data : (N, D) original samples
    embedding : (N, E) low dimensional representation, e.g. SOM coordinates
    num_neighbors : neighborhood size
    block_rows : samples processed per distance block

    Returns
    -------
    float - continuity score in [0, 1]
    """
    return 1.0 - rank_error(embedding, x_data, num_neighbors, block_rows)


def topology_preservation(
    x_data: NDArray, embedding: NDArray, num_neighbors: int = 5, block_rows: int = 256
) -> tuple[float, float]:
    """
    Trustworthiness and continuity in one pass, sharing each distance block

    Parameters
    ----------
    x_data : (N, D) original samples
    embedding : (N, E) low dimensional representation, e.g. SOM coordinates
    num_neighbors : neighborhood size
    block_rows : samples processed per distance block

    Returns
    -------
    trustworthiness and continuity, each in [0, 1] - higher is "better"
    """
    count = len(x_data)
    check_neighbor_inputs(count, len(embedding), num_neighbors)

    trust_penalty = continuity_penalty = 0.0
    for start in range(0, count, block_rows):
        stop = min(start + block_rows, count)
        data_dist = distance_block(x_data, start, stop)
        embedding_dist = distance_block(embedding, start, stop)
        trust_penalty += neighbor_rank_penalty(data_dist, embedding_dist, num_neighbors)
        continuity_penalty += neighbor_rank_penalty(embedding_dist, data_dist, num_neighbors)

    scale = 2.0 / (count * num_neighbors * (2 * count - 3 * num_neighbors - 1))
    return 1.0 - scale * trust_penalty, 1.0 - scale * continuity_penalty


# ---- performance metrics with labels available ----
def contingency_matrix(labels_true, labels_pred):
    """Create a contingency matrix for two labelings."""
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    cont_matrix = np.zeros((n_classes, n_clusters), dtype=np.int64)
    np.add.at(cont_matrix, (class_idx, cluster_idx), 1)
    return cont_matrix


def mutual_information_score(labels_true, labels_pred):
    """Compute the mutual information score between two clusterings."""
    contingency = contingency_matrix(labels_true, labels_pred)
    n_samples = np.sum(contingency)
    pi = contingency / n_samples
    pi_i = np.sum(pi, axis=1)
    pi_j = np.sum(pi, axis=0)

    non_zero = pi > 0
    mi = np.sum(pi[non_zero] * np.log(pi[non_zero] / np.outer(pi_i, pi_j)[non_zero]))
    return mi


def entropy_from_counts(counts: NDArray) -> float:
    """
    Shannon entropy of a frequency/count vector
    """
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log(probabilities))


def entropy_from_labels(labels):
    """Compute entropy of a label distribution"""
    _, counts = np.unique(labels, return_counts=True)
    return entropy_from_counts(counts)


def homogeneity(labels_true, labels_pred):
    """Compute homogeneity score of predicted labels given true labels."""
    # get unique class and cluster indices
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)

    # contingency matrix creation
    num_classes = classes.shape[0]
    num_clusters = clusters.shape[0]
    cont_matrix = np.zeros((num_classes, num_clusters), dtype=np.int64)
    np.add.at(cont_matrix, (class_idx, cluster_idx), 1)

    num_samples = np.sum(cont_matrix)
    class_freqs = np.sum(cont_matrix, axis=1)
    class_entropy = entropy_from_counts(class_freqs)

    # conditional entropy H(C|K)
    cond_ent = 0.0

    for i in range(num_clusters):
        cluster = cont_matrix[:, i]
        cluster_size = np.sum(cluster)
        if cluster_size > 0:
            cond_ent += (cluster_size / num_samples) * entropy_from_counts(cluster)

    # homogeneity score
    if class_entropy == 0:
        return 1.0
    return 1 - cond_ent / class_entropy
