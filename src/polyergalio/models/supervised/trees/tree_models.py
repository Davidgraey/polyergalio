import numpy as np
import copy
from numpy.typing import NDArray
from polyergalio.models.constants import EPSILON, SIGMA_ZERO, LAMBDA_MAX, LAMBDA_MIN
from typing import Optional, List, Tuple, Dict
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.model_loss import (
    cross_entropy,
    cross_entropy_derivative,
    mse,
    mse_derivative,
)
from polyergalio.models.activations import (
    softmax,
    sigmoid,
    linear,
)
from polyergalio.models.supervised import log
from polyergalio.fitted_model import FittedModel


class SupervisedTreeModel(FittedModel):
    pass


class ExplainableBoostedTreeModel(FittedModel):
    """
    Explainable Boosting Machine (EBM) — additive model with
    pairwise interactions, fit via cyclic gradient boosting of bagged / histogram learning

    supports regression and classification (binary / multinomial / multilabel)
    """
    parameter_names = ("main_effects", "interactions", "intercept")

    def __init__(
        self,
        input_dimension: int = 1,
        output_dimension: int = 1,
        task: ClassificationTask = ClassificationTask.BINARY,
        num_bins: int = 32,
        learning_rate: float = 0.01,
        num_rounds: int = 100,
        max_interaction_pairs: int = 10,
        interaction_num_bins: int = 16,
        seed: int = 42,
    ):
        super().__init__(seed=seed, input_dimension=input_dimension, output_dimension=output_dimension)
        self.task = task
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.num_rounds = num_rounds
        self.max_interaction_pairs = max_interaction_pairs
        self.interaction_num_bins = interaction_num_bins

        # --- Main-effect storage (populated during fit) ---
        # keyed by feature idx.  Each value is a dict:
        #   { "edges": NDArray, # quantile bounds
        #     "contributions": NDArray, # shape-function values per bin
        #     "num_bins": int }
        self.main_effects: Dict[int, dict] = {}

        # --- Pairwise interaction storage ---
        # Keyed by (feature_j, feature_k) tuple.  Each value is a dict:
        #   { "edges_j": NDArray, # quantile bounds for feature j
        #     "edges_k": NDArray, # quantile bounds for feature k
        #     "contributions": NDArray, # lookup
        #     "explained_variance": float } # selection score at time of ranking
        self.interactions: Dict[Tuple[int, int], dict] = {}

        self.intercept: float | NDArray = 0.0
        self.is_regression: bool = task is None

        self.is_multitarget: bool = False
        if (task == ClassificationTask.MULTINOMIAL) or (task == ClassificationTask.MULTILABEL):
            self.is_multitarget = True

    # Binning processes
    @staticmethod
    def _calculate_bin_edges(feature: NDArray, num_bins: int) -> NDArray:
        """ Return unique equal-frequency (quantile) cut-points for one feature's data """
        percentiles = np.linspace(0, 100, num_bins + 1)[1:-1]
        edges = np.percentile(feature, percentiles)
        return np.unique(edges)

    @staticmethod
    def _digitize(feature: NDArray, edges: NDArray) -> NDArray:
        """ Map each value to its bin index (0-based) """
        return np.digitize(feature, edges)

    @staticmethod
    def _fit_bins_1d(bin_indices: NDArray, num_bins: int, residuals: NDArray) -> NDArray:
        """ mean residual per bin — the gradient-boost update """
        contributions = np.zeros(num_bins)
        for b in range(num_bins):
            mask = bin_indices == b
            count = np.sum(mask)
            if count > 0:
                contributions[b] = np.mean(residuals[mask])
        return contributions

    @staticmethod
    def _fit_bins_1d_multi(bin_indices: NDArray, num_bins: int, residuals: NDArray) -> NDArray:
        """Multinomial variant: residuals have shape (N, C)."""
        n_classes = residuals.shape[1]
        contributions = np.zeros((num_bins, n_classes))
        for b in range(num_bins):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                contributions[b] = np.mean(residuals[mask], axis=0)
        return contributions

    @staticmethod
    def _fit_bins_2d(bins_j: NDArray, bins_k: NDArray,
                     n_bins_j: int, n_bins_k: int,
                     residuals: NDArray) -> NDArray:
        """Mean residual per 2-D bin cell."""
        contributions = np.zeros((n_bins_j, n_bins_k))
        for bj in range(n_bins_j):
            for bk in range(n_bins_k):
                mask = (bins_j == bj) & (bins_k == bk)
                if np.sum(mask) > 0:
                    contributions[bj, bk] = np.mean(residuals[mask])
        return contributions

    @staticmethod
    def _fit_bins_2d_multi(bins_j: NDArray, bins_k: NDArray,
                           n_bins_j: int, n_bins_k: int,
                           residuals: NDArray) -> NDArray:
        """Multinomial variant of 2-D bin fitting."""
        n_classes = residuals.shape[1]
        contributions = np.zeros((n_bins_j, n_bins_k, n_classes))
        for bj in range(n_bins_j):
            for bk in range(n_bins_k):
                mask = (bins_j == bj) & (bins_k == bk)
                if np.sum(mask) > 0:
                    contributions[bj, bk] = np.mean(residuals[mask], axis=0)
        return contributions

    # Interaction pair selection (greedy)
    def _select_interaction_pairs(
        self, x_data: NDArray, residuals: NDArray
    ) -> Dict[Tuple[int, int], dict]:
        """
        Rank all (j, k) pairs by the weighted variance of 2-D bin means and

        returns the top ranked interaction pairs as a dict[j, k]
        """
        n_features = x_data.shape[1]

        if n_features < 2:
            return {}

        # For multinomial residuals collapse to a scalar score per sample
        if residuals.ndim == 2:
            residuals_flat = np.sum(residuals ** 2, axis=1)
        else:
            residuals_flat = residuals

        candidates: List[Tuple[float, int, int]] = []
        for j in range(n_features):
            edges_j = self._calculate_bin_edges(x_data[:, j], self.interaction_num_bins)
            bins_j = self._digitize(x_data[:, j], edges_j)
            for k in range(j + 1, n_features):
                edges_k = self._calculate_bin_edges(x_data[:, k], self.interaction_num_bins)
                bins_k = self._digitize(x_data[:, k], edges_k)

                explained_var = 0.0
                for bj in range(len(edges_j) + 1):
                    for bk in range(len(edges_k) + 1):
                        mask = (bins_j == bj) & (bins_k == bk)
                        cnt = np.sum(mask)
                        if cnt > 1:
                            mean_r = np.mean(residuals_flat[mask])
                            explained_var += cnt * mean_r ** 2

                candidates.append((explained_var, j, k))

        candidates.sort(key=lambda c: -c[0])
        top = candidates[: self.max_interaction_pairs]

        result: Dict[Tuple[int, int], dict] = {}
        for explained_var, j, k in top:
            ej = self._calculate_bin_edges(x_data[:, j], self.interaction_num_bins)
            ek = self._calculate_bin_edges(x_data[:, k], self.interaction_num_bins)
            nbj, nbk = len(ej) + 1, len(ek) + 1

            if self.is_multitarget:
                contribs = np.zeros((nbj, nbk, self.output_dimension))
            else:
                contribs = np.zeros((nbj, nbk))

            result[(j, k)] = {
                "edges_j": ej,
                "edges_k": ek,
                "contributions": contribs,
                "explained_variance": float(explained_var),
            }

        return result

    # residuals (negative gradient)
    def _compute_residuals(self, scores: NDArray, targets: NDArray) -> NDArray:
        """
        Negative gradient of the loss -- not really residuals, pseudo resid?

        regression: target − score
        binary:        target − sigmoid(score)
        multinomial:   target − softmax(score)
        multilabel:    target − sigmoid(score)

        """
        if self.is_regression:
            return targets.ravel() - scores.ravel()

        if self.task == ClassificationTask.BINARY:
            return targets.ravel() - sigmoid(scores).ravel()
        elif self.task == ClassificationTask.MULTINOMIAL:
            return targets - softmax(scores)
        elif self.task == ClassificationTask.MULTILABEL:
            return targets - sigmoid(scores)

        return targets.ravel() - scores.ravel()

    def calculate_loss(self, scores: NDArray, targets: NDArray) -> float:
        """Internal loss dispatch used during training."""
        if self.is_regression:
            return float(mse(scores, targets))
        return float(cross_entropy(scores, targets, task=self.task))

    # --------- forward pass ----------
    def forward(self, x_data: NDArray, **kwargs) -> NDArray:
        """
        Compute raw additive scores for the regression and interaction effects
        """
        num_samples = x_data.shape[0]

        if self.is_multitarget:
            scores = np.full(
                shape=(num_samples, self.output_dimension),
                fill_value=self.intercept if np.isscalar(self.intercept) else 0.0
            )
            if not np.isscalar(self.intercept):
                scores += self.intercept
        else:
            scores = np.full(shape=num_samples, fill_value=(self.intercept))

        # Main effects
        for j, effect in self.main_effects.items():
            edges = effect["edges"]
            contribs = effect["contributions"]
            bins = np.clip(a=self._digitize(x_data[:, j], edges),
                           a_min=0,
                           a_max=len(contribs) - 1
                           )
            scores += contribs[bins]

        # Pairwise interactions
        for (j, k), interaction in self.interactions.items():
            contribs = interaction["contributions"]
            bj = np.clip(a=self._digitize(x_data[:, j], interaction["edges_j"]),
                         a_min=0,
                         a_max=contribs.shape[0] - 1
                         )
            bk = np.clip(a=self._digitize(x_data[:, k], interaction["edges_k"]),
                         a_min=0,
                         a_max=contribs.shape[1] - 1
                         )
            scores += contribs[bj, bk]
        # scores / logits
        return scores

    def predict(self, x_data: NDArray, **kwargs) -> NDArray:
        """ apply the appropriate activation and return hard predictions """
        scores = self.forward(x_data)

        if self.is_regression:
            return scores

        if self.task == ClassificationTask.BINARY:
            return (sigmoid(scores) >= 0.5).astype(int)
        elif self.task == ClassificationTask.MULTINOMIAL:
            return np.argmax(softmax(scores), axis=-1)
        elif self.task == ClassificationTask.MULTILABEL:
            return (sigmoid(scores) >= 0.5).astype(int)

        return scores

    def fit(self, x_data: NDArray, y_data: NDArray = None, **kwargs) -> List[float]:
        """
        Train the EBM in two phases:
          GB main effects f(x)
          GB pairwise interaction effects fj(j_x) +fk(k_x)

        Parameters
        ----------
        x_data: (num_samples, dimension) feature matrix
        y_data: (num_samples) or (num_samples, class_dim)

        Returns
        -------
        List of loss values recorded at the end of each gradient boosting round
        """
        num_samples, num_features = x_data.shape
        self.input_dimension = num_features
        # is_multi = self.task == ClassificationTask.MULTINOMIAL

        # build bins on quantiles
        self.main_effects = {}
        for j in range(num_features):
            edges = self._calculate_bin_edges(x_data[:, j], self.num_bins)
            num_actual = len(edges) + 1
            if self.is_multitarget:
                contribs = np.zeros((num_actual, self.output_dimension))
            else:
                contribs = np.zeros(num_actual)

            self.main_effects[j] = {
                "edges": edges,
                "contributions": contribs,
                "num_bins": num_actual,
            }

        # establish intercetps
        if self.is_regression:
            self.intercept = float(np.mean(y_data))
        elif self.task == ClassificationTask.BINARY:
            p = np.clip(np.mean(y_data), EPSILON, 1 - EPSILON)
            self.intercept = float(np.log(p / (1 - p)))
        elif self.is_multitarget:
            # per-class log-prior
            class_freq = np.mean(y_data, axis=0)
            class_freq = np.clip(class_freq, EPSILON, None)
            self.intercept = np.log(class_freq)
        else:
            self.intercept = 0.0

        # main effects f(x)
        losses: List[float] = []

        for round_i in range(self.num_rounds):
            scores = self.forward(x_data)
            residuals = self._compute_residuals(scores, y_data)

            for j, effect in self.main_effects.items():
                edges = effect["edges"]
                contribs = effect["contributions"]
                num_actual = effect["num_bins"]
                bins = np.clip(self._digitize(x_data[:, j], edges),
                               0, num_actual - 1)

                if self.is_multitarget:
                    update = self._fit_bins_1d_multi(bins, num_actual, residuals)
                else:
                    update = self._fit_bins_1d(bins, num_actual, residuals)

                effect["contributions"] = contribs + self.learning_rate * update
                # refresh residuals for the next feature in this round
                residuals -= self.learning_rate * update[bins]

            loss = self.calculate_loss(self.forward(x_data), y_data)
            losses.append(loss)

            if round_i > 5 and abs(losses[-1] - losses[-2]) < EPSILON:
                log.info(f"EBM main-effects converged at round {round_i}")
                break

        # pairwise effects
        scores = self.forward(x_data)
        residuals = self._compute_residuals(scores, y_data)
        self.interactions = self._select_interaction_pairs(x_data, residuals)

        interaction_rounds = max(1, self.num_rounds // 2)
        for round_i in range(interaction_rounds):
            scores = self.forward(x_data)
            residuals = self._compute_residuals(scores, y_data)

            for (j, k), interaction in self.interactions.items():
                contribs = interaction["contributions"]
                edges_j = interaction["edges_j"]
                edges_k = interaction["edges_k"]
                bj = np.clip(self._digitize(x_data[:, j], edges_j), 0, contribs.shape[0] - 1)
                bk = np.clip(self._digitize(x_data[:, k], edges_k), 0, contribs.shape[1] - 1)
                nbj, nbk = len(edges_j) + 1, len(edges_k) + 1

                if self.is_multitarget:
                    update = self._fit_bins_2d_multi(bj, bk, nbj, nbk, residuals)
                else:
                    update = self._fit_bins_2d(bj, bk, nbj, nbk, residuals)

                interaction["contributions"] = contribs + self.learning_rate * update
                residuals -= self.learning_rate * update[bj, bk]

            loss = self.calculate_loss(self.forward(x_data), y_data)
            losses.append(loss)

            if len(losses) > 2 and abs(losses[-1] - losses[-2]) < EPSILON:
                log.info(f"EBM interactions converged at {round_i}")
                break

        log.info(f"EBM training complete. Final loss: {losses[-1]:.6f}")
        self.fitted = True
        return losses

    def fit_predict(
        self,
        x_data: NDArray,
        verbose: bool = False,
        num_iterations: int = 100,
        labels: Optional[NDArray] = None,
        **kwargs,
    ) -> NDArray:
        self.num_rounds = num_iterations
        losses = self.fit(x_data, y_data=labels)
        if verbose:
            log.info(f"fit_predict losses: {losses[:5]}...{losses[-5:]}")
        return self.predict(x_data)

    # Interpretability helpers
    def get_feature_importance(self) -> Dict[int, float]:
        """ contribution per effect """
        return {
            j: float(np.mean(np.abs(effect["contributions"])))
            for j, effect in self.main_effects.items()
        }

    def get_shape_function(self, feature_index: int) -> Tuple[NDArray, NDArray]:
        """
        return (centers, contributions) for a single feature's shape - plotting helper
        """
        effect = self.main_effects[feature_index]
        edges = effect["edges"]
        sf = effect["contributions"]
        if len(edges) == 0:
            return np.array([0.0]), sf

        centers = np.empty(len(edges) + 1)
        gap = np.diff(edges)
        centers[0] = edges[0] - (gap[0] / 2 if len(gap) else 1.0)
        centers[-1] = edges[-1] + (gap[-1] / 2 if len(gap) else 1.0)
        for i in range(len(edges) - 1):
            centers[i + 1] = (edges[i] + edges[i + 1]) / 2
        return centers, sf

    def get_interaction_function(self, pair: Tuple[int, int]) -> dict:
        """
        Return the full interaction dict for the (j, k) pair
        "edges_j", "edges_k", "contributions", and
        "explained_variance".
        """
        return self.interactions[pair]

    @property
    def info(self) -> str:
        task_name = self.task.value if self.task else "regression"
        return (
            f"EBM: {self.input_dimension} features, {self.num_bins} bins, "
            f"{len(self.interactions)} interaction pairs, "
            f"task={task_name}, lr={self.learning_rate}, rounds={self.num_rounds}"
        )
