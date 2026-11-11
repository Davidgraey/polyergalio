import copy

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from polyergalio.distances import pairwise_euclidian_distance
from polyergalio.fitted_model import FittedModel
from polyergalio.models.clustering import log
from polyergalio.models.clustering.cluster_metrics import silhouette_score
from polyergalio.models.constants import EPSILON
from polyergalio.visuals.cluster_visuals import plot_clusters


class CentroidNeuralNetwork(FittedModel):
    """The per-k growth history (centroid tracker, label tracker, metrics) is training-time bookkeeping and is not persisted."""

    parameter_names = ("centroids",)
    state_names = ("skip_standardize",)

    def __init__(
        self,
        max_clusters: int,
        seed: int = 42,
        initial_clusters: NDArray | None = None,
        epsilon: float = 2e-3,
    ):
        super().__init__(seed=seed)
        self.max_clusters = max_clusters
        self.initial_clusters = initial_clusters
        self.epsilon: float = epsilon
        self.num_dims: int = 0

        if initial_clusters is not None:
            self.centroids = initial_clusters
        else:
            self.centroids = None
        self.previous_labels = None
        self.volume = None

        self._label_tracker = {}
        self._centroid_tracker = {}

        self.skip_standardize = False

        self.metrics = {k: np.nan for k in range(2, max_clusters + 1)}
        self._scored = False

    @staticmethod
    def distance_metric(data_x: NDArray, data_y: NDArray) -> NDArray:
        """
        Calculate the distance metric between data points
        Parameters
        ----------
        data_x :
        data_y :

        Returns
        -------

        """
        return pairwise_euclidian_distance(data_x, data_y) ** 2

    def batch_update(self,
                     num_centroids: int,
                     x_data: NDArray,
                     new_labels: NDArray,
                     batch_mask: NDArray,
                     ) -> int:
        previous_labels = self._label_tracker.get(num_centroids)
        changed_mask = new_labels != previous_labels[batch_mask]
        changed_count = np.sum(changed_mask)

        one_hot = np.zeros((len(new_labels), num_centroids))
        one_hot[np.arange(len(new_labels)), new_labels] = 1.0
        member_counts = one_hot.sum(axis=0)
        occupied = member_counts > 0
        member_sums = one_hot.T @ x_data
        self.centroids[:num_centroids][occupied] = (
            member_sums[occupied] / member_counts[occupied, None]
        )

        self._update_labels(
            num_centroids, closest_centroids=new_labels, shuffle_mask=batch_mask
        )

        return changed_count

    def _update_labels(self,
                       num_clusters: int,
                       closest_centroids: NDArray,
                       shuffle_mask: NDArray
                       ) -> None:
        if num_clusters not in self._label_tracker.keys():
            self._label_tracker[num_clusters] = np.zeros_like(closest_centroids)

        self._label_tracker[num_clusters][shuffle_mask] = closest_centroids
        pass

    def split_clusters(self,
                       xs: NDArray,
                       closest_centroids: NDArray,
                       split_target: int
                       ) -> tuple[NDArray, NDArray]:
        """

        Parameters
        ----------
        xs : array -
        closest_centroids : array - the active closest centroid, per sample
        split_target : cluster index to be split

        Returns
        -------
        a tuple of both the new centroids, split along the vector of least density
        """

        cluster_xs = xs[closest_centroids == split_target]
        cluster_mean = np.mean(cluster_xs, axis=0)
        centered_xs = cluster_xs - cluster_mean
        cluster_covariance = np.cov(cluster_xs, rowvar=False)

        # get eigenvectors of covarainces
        eighvals, eighvecs = np.linalg.eigh(cluster_covariance)
        eigh_val_max, eigh_val_max_idx = np.max(eighvals), np.argmax(eighvals)
        principal_projection = eighvecs[:, eigh_val_max_idx]
        density_projection = centered_xs @ principal_projection

        # delta based on the eigenvalue, scaled spacing from density
        delta = np.sqrt(eigh_val_max) * np.maximum(
            self.density[split_target], EPSILON
        ) ** (-1 / self.num_dims)

        delta = min(delta, np.abs(density_projection).max())

        split_centroid_a = cluster_mean - principal_projection * delta
        split_centroid_b = cluster_mean + principal_projection * delta

        return (split_centroid_a, split_centroid_b)

    def fit(self,
            org_x_data: NDArray,
            verbose: bool = False,
            num_iterations: int = 25,
            fast_forward: bool = False,
            mini_batch: bool = False,
            skip_standardize: bool = False,
            ) -> tuple[NDArray, NDArray]:
        """
        Fit the clustering model to the data, return the centroids and labels for each data point.

        Parameters
        ----------
        org_x_data : numpy.ndarray - our input data
        verbose : bool Verbose will trigger plotting and additional outputs during the fit process
        fast_forward : bool - if True, skips metric calculations and plotting, and get_optimal()
            has nothing to select from (it will raise instead of silently returning the smallest k)

        Returns
        -------
        the unstandardized centroid locations and the class labels for each data point
        """
        # refitting an already-fitted instance starts clean rather than growing from wherever the previous fit left off
        if self.fitted:
            self.centroids = None
            self._label_tracker = {}
            self._centroid_tracker = {}
            self.metrics = {k: np.nan for k in range(2, self.max_clusters + 1)}
            self._scored = False
            self.fitted = False

        # check and see if we've done the standardization process:
        if skip_standardize:
            self.skip_standardize = skip_standardize
            x_data = org_x_data
        else:
            self.init_standardize(org_x_data)
            x_data = self.standardize(org_x_data)

        num_samples, num_features = x_data.shape
        self.num_dims = num_features

        # overwrite any prior labels -- set all to -1 (no cluster)
        previous_labels = np.full(num_samples, -1, dtype=int)
        starting_cent = np.mean(x_data, axis=0)

        # set up inertia for early termination
        if mini_batch and (num_samples > 500):
            batch_size = max((num_samples // 5), 500)
            log.info(f"processing in minibatches of size: {batch_size}")
        else:
            batch_size = num_samples
        breakpoint = max(1, int(batch_size * self.epsilon))

        # initalize the first two centroids ------------------
        if self.centroids is None:
            self.centroids = np.vstack(
                [
                    starting_cent
                    + self.RNG.uniform(-self.epsilon, self.epsilon, size=num_features),
                    starting_cent
                    - self.RNG.uniform(-self.epsilon, self.epsilon, size=num_features),
                ]
            )
        else:
            self.centroids = self.standardize(self.centroids)

        num_centroids = self.centroids.shape[0]
        self._label_tracker[num_centroids] = previous_labels
        self._batch = np.arange(0, num_samples)

        # =================== MAIN LOOP ====================
        while num_centroids <= self.max_clusters:
            # minibatch--------
            shuffle_mask = self.RNG.permutation(self._batch)[:batch_size].copy()
            xs = x_data[shuffle_mask, ...]
            for i in range(0, num_iterations):
                # get the "previous labels" --------------------------
                closest_centroids, full_distances = self.forward(
                    xs, num_centroids, axis=-1
                )

                changed_count = self.batch_update(
                    num_centroids=num_centroids,
                    x_data=xs,
                    new_labels=closest_centroids,
                    batch_mask=shuffle_mask,
                )

                if changed_count <= breakpoint:
                    log.info(
                        f"converged at iteration {i} for {num_centroids} centroids ({changed_count} changed)"
                    )
                    break

            self._centroid_tracker.update(
                {num_centroids: self.centroids[:num_centroids].copy()}
            )
            if fast_forward is False:
                self.metrics.update(
                    {num_centroids: silhouette_score(xs, closest_centroids)}
                )
                self._scored = True

            if num_centroids == self.max_clusters:
                log.info(
                    f"breaking from iterations at {num_centroids} centroids created "
                )
                break

            self.density = self.local_density(full_distances, k_radius=20)

            # calculate loss
            error = self.calculate_loss(
                distances=self.distance_metric(xs, self.centroids[:num_centroids]),
                closest_centroids=closest_centroids,
                num_centroids=num_centroids,
            )

            if verbose and (fast_forward is False):
                log.info(f"DENSITY: {self.density}")
                plot_clusters(xs, closest_centroids, self.centroids[:num_centroids])

            # Splitting the highest error cluster (the "worst fit" cluster)
            split_target = np.argmax(error)

            # ---- split_target
            cluster_a, cluster_b = self.split_clusters(xs, closest_centroids, split_target)
            self.centroids = np.vstack([self.centroids[:num_centroids], cluster_b])
            self.centroids[split_target] = cluster_a

            # cleanup, update & ending of the loop ---------------
            num_centroids += 1
            # carry this latest "closest" forward into next loop
            self._update_labels(num_centroids, closest_centroids, shuffle_mask)

        # ending conditions ========================================
        self.fitted = True

        if self.skip_standardize:
            return (
                self.centroids[:num_centroids, ...],
                self._label_tracker[num_centroids],
            )
        else:
            return (
                self.unstandardize(self.centroids[:num_centroids, ...]),
                self._label_tracker[num_centroids],
            )

    def fit_predict(
        self,
        org_x_data: NDArray,
        verbose: bool = False,
        num_iterations: int = 25,
        fast_forward: bool = False,
        mini_batch: bool = False,
        skip_standardize: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """
        Fit the clustering model to the data, return the centroids and labels for each data point.

        Parameters
        ----------
        org_x_data : numpy.ndarray
        verbose : bool Verbose will trigger plotting and additional outputs during the fit process
        fast_forward : bool - if True, will skip metric calculations and plotting

        Returns
        -------
        the unstandardized centroid locations and the class labels for each data point
        """

        return self.fit(
            org_x_data,
            verbose,
            num_iterations,
            fast_forward,
            mini_batch,
            skip_standardize,
        )

    def forward(
        self, x_data: NDArray, num_centroids: int, axis: int = -1
    ) -> tuple[NDArray, NDArray]:
        """
        Calculate the closest centroid to each sample in x_data

        Parameters
        ----------
        x_data : the data array
        num_centroids : integer, the number of centroids
        axis : which axis to collect across (usually -1)

        Returns
        -------
        array of integers (cluster membership for the closest centroid to each datapoint
        """
        full_distances = self.distance_metric(
            data_x=x_data,
            data_y=self.centroids[:num_centroids],
        )

        closest_centroids = np.argmin(full_distances, axis=axis)

        return closest_centroids, full_distances

    def local_density(self, distances: NDArray, k_radius: int = 10) -> NDArray:
        """
        Density around each centroid: points inside a ball of the mean k-nearest distance, per ball volume, as a
        multiple of the mean density over centroids. Computed in log space so it holds in any dimension.

        Parameters
        ----------
        distances : (num_points, num_centroids) squared distance from every point to every centroid.
        k_radius : target neighbor count, clamped to what num_points supports

        Returns
        -------
        (num_centroids,) relative density, 1 being the average centroid
        """
        num_points = distances.shape[0]
        if num_points < 2:
            return np.ones(distances.shape[1])

        k_radius = min(k_radius, num_points - 1)

        nearest = np.sqrt(np.partition(distances, k_radius, axis=0)[:k_radius, :])
        radius = 1.67 * np.mean(nearest, axis=0)

        half_dims = self.num_dims / 2
        self.volume = (
            half_dims * np.log(np.pi)
            - gammaln(half_dims + 1)
            + self.num_dims * np.log(radius + EPSILON)
        )
        point_count = np.sum(distances <= radius**2, axis=0)

        log_density = np.log(np.maximum(point_count, 1)) - self.volume
        density = np.exp(log_density - log_density.max())
        return density / density.mean()

    def predict(self, x_data: NDArray, num_centroids: int | None = None) -> NDArray:
        """
        Classify data against the fitted centroids

        Parameters
        ----------
        x_data : input samples, same feature input_dimension as the data used to fit

        num_centroids : how many of the fitted centroids to classify against.
            Defaults to every centroid currently held
            Pass get_optimal()'s best count explicitly to pick the optimal k

        Returns
        -------
        NDArray
            integer cluster label per sample, nearest centroid by distance
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        xs = x_data if self.skip_standardize else self.standardize(x_data)
        if num_centroids is None:
            num_centroids = len(self.centroids)

        labels, _ = self.forward(xs, num_centroids=num_centroids, axis=-1)
        return labels

    # predictions, targets
    def calculate_loss(
        self, distances: NDArray, closest_centroids: NDArray, num_centroids: int
    ) -> NDArray:
        """
        Summed squared distance of each cluster's members to its centroid.

        Parameters
        ----------
        distances : (N, num_centroids) squared distances to the current centroids
        closest_centroids : (N,) cluster of each sample
        num_centroids : cluster count

        Returns
        -------
        (num_centroids,) error per cluster
        """
        own_distances = distances[np.arange(len(closest_centroids)), closest_centroids]
        return np.bincount(
            closest_centroids, weights=own_distances, minlength=num_centroids
        )

    def growth_history(self) -> list[dict]:
        """
        Centroids and labels at each cluster count tried during fit
        In the order they were created (2, 3, ..., final)

        Returns
        -------
        List of dicts - Dictionary of
            number of centroids: int
            centroids: array - positions of centroids
            labels: array - the predicted cluster membersion (batch, idx)
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        history: list[dict] = []
        for num_centroids in sorted(self._centroid_tracker):
            centroids = self._centroid_tracker[num_centroids]
            if not self.skip_standardize:
                centroids = self.unstandardize(centroids)

            history.append(
                {"num_centroids": num_centroids,
                 "centroids": centroids,
                 "labels": self._label_tracker[num_centroids]}
            )
        return history

    def get_optimal(self) -> tuple[int, NDArray, NDArray]:
        """
        Get the optimal number of clusters based on the metrics collected during fitting.
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        if not self._scored:
            raise RuntimeError(
                "No cluster-count scoring was collected (fit ran with fast_forward=True), "
                "so there is no meaningful optimum to pick. Refit with fast_forward=False."
            )

        scored = {k: score for k, score in self.metrics.items() if not np.isnan(score)}
        best_scoring = max(scored, key=scored.get)
        if self.skip_standardize:
            _cents = self._centroid_tracker.get(best_scoring)
        else:
            _cents = self.unstandardize(self._centroid_tracker.get(best_scoring))
        return (
            best_scoring,
            _cents[:best_scoring, ...],
            self._label_tracker[best_scoring],
        )

    def get_state(self) -> dict:
        """Fitted state; the centroids are cut to the optimal cluster count and the per-k growth history is left out."""
        state = super().get_state()
        if self._centroid_tracker and self._scored:
            best_scoring, _, _ = self.get_optimal()
            state["centroids"] = copy.deepcopy(self._centroid_tracker[best_scoring])
        return state


# Example usage:
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    from polyergalio.generators import RandomDatasetGenerator
    from polyergalio.models.clustering.cluster_metrics import *

    gen = RandomDatasetGenerator(random_seed=42)
    X, y, meta = gen.generate(
        task="clustering",
        num_samples=2000,
        num_features=2,
        noise_scale=0.4,
        num_clusters=8,
        verbose=True,
    )

    plot_clusters(X, y, meta["centroids"])
    st = time.time()

    # for eps in np.arange(3, 5):
    cnn = CentroidNeuralNetwork(max_clusters=10, epsilon=0.2)
    centroids, labels = cnn.fit_predict(
        X, num_iterations=512, mini_batch=False, verbose=True, fast_forward=False
    )
    print(cnn.metrics)
    print("verbose took", (time.time() - st) * 1000)
    plot_clusters(X, labels, centroids)
    clust_count, opt_centroids, opt_labels = cnn.get_optimal()
    plot_clusters(X, opt_labels, opt_centroids)

    print(f"silhouette score (1 is best): {silhouette_score(X, opt_labels)}")
    print(f"CH index (high): {calinski_harabasz_index(X, opt_labels)}")
    print(f"DB index score (low): {davies_bouldin_index(X, opt_labels)}")

    print(f"homogenity: {homogeneity(y, opt_labels)}")
    print(f"Mutual Information: {mutual_information_score(y, opt_labels)}")

    for eps in np.arange(2, 5):
        st = time.time()
        cnn = CentroidNeuralNetwork(max_clusters=10, epsilon=1/eps)
        centroids, labels = cnn.fit_predict(X, verbose=False, fast_forward=True)
        print("EPS: ", 1/eps)
        print(cnn.metrics)
        print("verbose took", (time.time() - st) * 1000)
        plot_clusters(X, labels, centroids)
