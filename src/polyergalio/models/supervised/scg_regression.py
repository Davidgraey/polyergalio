"""
Regression: Gradient Descent with Scaled Conjugate Gradient
SCG lets us avoid hyperparameter optimization step; makes it much more 'plug n play'
Scaled Conjugate Gradient adapted from implementation by Prof Charles
Anderson, CSU

Including regularization via elasticnet to extend usability a bit
"""

import numpy as np
import copy
from numpy.typing import NDArray
from polyergalio.models.constants import EPSILON, SIGMA_ZERO, LAMBDA_MAX, LAMBDA_MIN
from typing import Optional
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.weight_initialization import get_weight_init
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


class GradientDescent(FittedModel):
    """Linear and generalized linear regression fit by scaled conjugate gradient; y_means and y_stds stay None for classification."""
    parameter_names = ("weights",)
    state_names = ("y_means", "y_stds", "full_init", "input_dimension")

    def __init__(
        self,
        task: str | ClassificationTask = "regression",
        divisi: int = 3,
        reg_lambda: float = 0.1,
        reg_alpha: float = 0.5,
        use_elastic_reg: bool = False,
        early_termination: bool = True,
    ):
        """
        Parameters
        ----------
        task : Union[str, constants.ClassificationTask]
        divisi : int the divisor to slow down learning rate - higher = slower
        reg_lambda : float  overall regularization strength
        reg_alpha : float balance between lasso L1 (0.0) ridge L2 (1.0)
        use_elastic_reg : bool enable elastic regularization
        """
        super().__init__(seed=42)
        self.full_init = False
        self.early_termination = early_termination

        # hyperparameters for elasticnet regularization
        self.divisi = divisi
        self.use_elastic_reg = use_elastic_reg
        self.reg_lambda: float = reg_lambda  # overall regularization strength
        self.reg_alpha: float = (
            reg_alpha  # balance between lasso L1 (0.0) ridge L2 (1.0)
        )

        self.weights = None
        self.y_means = None
        self.y_stds = None

        self.task: ClassificationTask = task
        if task == "regression":
            self.func = {
                "activation": linear,
                "loss": mse,
                "loss_derivative": mse_derivative,
            }

        elif task == ClassificationTask.BINARY:
            self.func = {
                "activation": sigmoid,
                "loss": cross_entropy,
                "loss_derivative": cross_entropy_derivative,
            }

        elif task == ClassificationTask.MULTINOMIAL:
            self.func = {
                "activation": softmax,
                "loss": cross_entropy,
                "loss_derivative": cross_entropy_derivative,
            }

        elif task == ClassificationTask.MULTILABEL:
            self.func = {
                "activation": sigmoid,
                "loss": cross_entropy,
                "loss_derivative": cross_entropy_derivative,
            }
        else:
            raise ValueError(
                f"task must be one of regression, binary, multinomial, multilabel; got {task}"
            )
        self.zero_gradients()

    def init_weights(self, n_outputs: int = 1) -> None:
        """
        Initalize the beta values (coefficients / weights) using Kaiming mechanism
        Parameters
        ----------
        n_outputs : the number of output dimensions

        Returns
        -------
        None
        """
        coefficients = get_weight_init("kaiming")(self.RNG, ni=self.input_dimension, no=n_outputs)
        self.weights = np.vstack([np.zeros((1, n_outputs)), coefficients])

    def init_standardize(self, x_data: NDArray, y_data: Optional[NDArray]) -> None:
        """update the tracking means and standards"""
        self.x_means = np.mean(x_data, axis=0)
        self.x_stds = np.std(x_data, axis=0)

        if self.task == "regression":
            self.y_means = np.mean(y_data, axis=0)
            self.y_stds = np.std(y_data, axis=0)

    def _add_intercept(self, data_array: NDArray):
        """insert an intercept at 0th index"""
        return np.insert(data_array, 0, 1, -1)

    def log_likelihood(self, prediction: NDArray, targets: NDArray) -> NDArray | float:
        """
        Calc the LLH given the prediction in prob space, and our targets. (classification tasks only)
        Parameters
        ----------
        prediction : prediction with activation applied (probability values)
        targets : y data

        Returns
        -------
        log likelihood, float
        """
        _p = np.clip(prediction, EPSILON, 1 - EPSILON)
        if self.task == ClassificationTask.MULTINOMIAL:
            llh = np.mean(np.sum(targets * np.log(_p), axis=-1))

        elif self.task in (ClassificationTask.BINARY, ClassificationTask.MULTILABEL):
            llh = np.mean(
                np.sum(
                    targets * np.log(_p)
                    + (1 - targets) * np.log(1 - prediction + EPSILON),
                    axis=-1,
                )
            )
        return llh

    def _calculate_pseudo_r_squared(
        self, prediction: NDArray, targets: NDArray
    ) -> tuple[float, float]:
        """doing McFadden's R-squared for classificaiton"""
        # LLH of the model
        log_likelihood = self.log_likelihood(prediction, targets)

        # Log-likelihood of the null model (using the mean of y as intercept)
        p_null = np.mean(targets, axis=0)
        null_log_likelihood = self.log_likelihood(p_null, targets)

        # McFadden's R-squared
        self.r_square = 1 - (log_likelihood / null_log_likelihood)

        # no standardized adjusted version; so let's just... -1
        self.adjusted_r_square = -1

        return self.r_square, self.adjusted_r_square

    def _calculate_r_square(
        self, prediction: NDArray, targets: NDArray
    ) -> tuple[float, float]:
        if self.task == "regression":
            self.residuals = prediction - targets
            residual_sum_squares = np.sum(self.residuals**2)
            total_sum_squares = np.sum((targets - np.mean(targets)) ** 2)
            self.r_square = 1 - (residual_sum_squares / total_sum_squares)
            self.adjusted_r_square = 1 - (
                ((1 - self.r_square) * (self.num_samples - 1))
                / (self.num_samples - self.input_dimension - 1)
            )

        # ------- classification----- using mcfadden's R2 and no adjusted mechanism
        else:
            self.r_square, self.adjusted_r_square = self._calculate_pseudo_r_squared(
                prediction, targets
            )

        return self.r_square, self.adjusted_r_square

    def scaled_conjugate_gradient(
        self, x_data: NDArray, y_data: NDArray, iterations: int
    ) -> NDArray:
        """
        This is the SCG multi-step search; 2nd order derivatives lead us to a step that is less sensitive to local
        minima.  However, it's expensive to compute, and can overfit/converge too quickly; so we alternate between SCG
        and gradient descent.  It shares some stages with the Matlab implementation

        Parameters
        ----------
        x_data : numpy.ndarray
        y_data : numpy.ndarray
        iterations : int - number of substeps to take for this SCG update

        Returns
        -------
        numpy.ndarray
        """
        lamb = 1e-6
        lamb_ = 0

        self._weight_shape = self.weights.shape
        vector = self.weights.reshape(-1, 1).copy()
        grad_new, _ = self._calculate_gradients(x_data, y_data)
        grad_new = -1 * grad_new.reshape(-1, 1)
        r_new = grad_new.copy()
        success = True

        for _i in range(iterations):
            r = r_new.copy()
            grad = grad_new.copy()
            mu = grad.T @ grad

            if success:
                success = False
                sigma = SIGMA_ZERO / np.sqrt(mu)

                grad_old, _ = self._calculate_gradients(x_data, y_data)
                grad_old = grad_old.reshape(-1, 1)
                self.weights = (vector + (sigma * grad)).reshape(self._weight_shape)
                grad_step, _ = self._calculate_gradients(x_data, y_data)

                step = (grad_old - grad_step.reshape(-1, 1)) / sigma
                delta = grad.T @ step

            # increase the curvature / scale
            zeta = lamb - lamb_
            step += zeta * grad
            delta += zeta * mu

            if delta <= 0:
                step += (lamb - 2 * delta / mu) * grad
                lamb_ = 2 * (lamb - delta / mu)
                delta -= lamb * mu
                delta *= -1
                lamb = lamb_

            phi = grad.T @ r
            alpha = phi / delta

            vector_new = vector + alpha * grad
            loss_old = self.calculate_loss(x_data, y_data)
            self.weights = vector_new.copy().reshape(self._weight_shape)
            loss_new = self.calculate_loss(x_data, y_data)

            comparison = 2 * delta * (loss_old - loss_new) / (phi**2)

            if comparison >= 0:
                # break condition?
                vector = vector_new.copy()
                loss_old = loss_new
                self.weights = vector_new.copy().reshape(self._weight_shape)
                r_new, _ = self._calculate_gradients(x_data, y_data)
                r_new = -1 * r_new.reshape(-1, 1)
                success = True
                lamb_ = 0

                if _i % self._weight_shape[0] == 0:
                    grad_new = r_new
                else:
                    beta = ((r_new.T @ r_new) - (r_new.T @ r)) / phi
                    grad_new = r_new + beta * grad

                if comparison > 0.75:
                    lamb = max(0.5 * lamb, LAMBDA_MIN)
            else:
                lamb_ = lamb

            if comparison < 0.25:
                lamb = min(4 * lamb, LAMBDA_MAX)

        return vector_new.reshape(self._weight_shape)

    def fit(
        self,
        x_data: NDArray,
        y_data: NDArray,
        iterations: int = 100,
        add_constant: bool = True,
    ):
        self.num_samples = x_data.shape[0]
        self.num_outputs = y_data.shape[-1]

        # if we've never trained before, we need to take additional init steps
        self.input_dimension = x_data.shape[-1]

        if not self.full_init:
            self.init_weights(n_outputs=self.num_outputs)
            self.init_standardize(x_data=x_data, y_data=y_data)
            self.full_init = True

        # standardize our data ------------------------------------------
        xs = self.standardize(x_data, self.x_means, self.x_stds)
        if self.task == "regression":
            ys = self.standardize(y_data, self.y_means, self.y_stds)
        else:
            ys = y_data.copy()

        # if we need to add intercept ------------------------------------
        if add_constant:
            xs = self._add_intercept(data_array=xs)
        else:
            xs = xs.copy()

        # training loop ------------------------------------------------
        errors = []
        for i in range(iterations):
            shuffler = self.RNG.permutation(self.num_samples)
            xs = xs[shuffler]
            ys = ys[shuffler]

            loss = self.calculate_loss(xs, ys)
            # deepcopy - so we can exploit the SCG steps
            W = copy.deepcopy(self.weights)

            # weight updates -------------------------------------------------
            if i % 2 == 0:
                # Scaled Conjugate Gradient - take a partial step towards the SCG results
                # between current weights and SCG weights
                scg_weights = self.scaled_conjugate_gradient(xs, ys, iterations // 2)
                self.weights = (W + scg_weights) / (self.divisi)

            else:
                # standard gradient descent approach ---------
                delta_ws, prediction = self._calculate_gradients(xs, ys)
                self.weights = W - (delta_ws * (0.001 / self.divisi))

            epoch_loss = np.mean(loss)
            errors.append(epoch_loss)

            if i <= 5:
                continue
            if self.early_termination:
                loss_delta = np.array(errors[-min(i, 5) :]) - epoch_loss
                loss_new = np.mean(self.calculate_loss(xs, ys))
                if (np.mean(loss_delta) < 0.01) or ((epoch_loss - loss_new) >= 0):
                    self.divisi += 1

                if np.sum(loss_delta < 0) > 10:
                    print(f"early termination at {i} with error {epoch_loss}")
                    log.info(f"early termination at {i} with error {epoch_loss}")
                    break
                # not elif -- if cond
                if np.mean(loss_delta) < 1e-12:
                    print(f"early termination, loss stopped decreasing at {i}")
                    log.info(f"early termination, loss stopped decreasing at {i}")
                    break

        self.fitted = True
        r_squared = self._calculate_r_square(self.predict(x_data), y_data)
        log.info(f"fitted r_squared: {r_squared}")
        log.info(f"divisi {self.divisi} ")
        return errors

    def forward(self, x_data: NDArray, has_bias_present=True) -> NDArray:
        """
        If we already introduced bias into the X data (training), or if we need to add it as in prediction

        Parameters
        ----------
        x_data : numpy.ndarray
        has_bias_present : bool if we have the bias / intercept already present in our x_data array

        Returns
        -------
        numpy.ndarray
        """
        if has_bias_present:
            logit = x_data @ self.weights

        else:  # add intercept to our data
            x_data = self._add_intercept(x_data)
            logit = x_data @ self.weights

        return logit

    def _calculate_gradients(
        self, x_data: NDArray, y_data: NDArray
    ) -> tuple[NDArray, NDArray]:
        """calculate gradients for weight updates"""
        logits = self.forward(x_data)
        prediction = self.func["activation"](logits)
        delta_grad = self.func["loss_derivative"](
            targets=y_data, prediction=prediction, task=self.task
        )

        if self.use_elastic_reg:
            reg_gradient = self.elasticnet_grad()
            gradient = (x_data.T @ (delta_grad)) + reg_gradient
        else:
            gradient = x_data.T @ (delta_grad)

        return gradient, prediction

    def calculate_loss(self, xs: NDArray, ys: NDArray) -> float:
        """calculate loss for current weights and inputs"""
        logits = self.forward(xs)
        if self.use_elastic_reg:
            regularization_loss = self.elasticnet_loss()
        else:
            regularization_loss = 0

        return (
            self.func["loss"](
                prediction=logits, targets=ys, task=self.task, reduction="mean"
            )
            + regularization_loss
        )

    def predict(self, x_data: NDArray) -> NDArray:
        """forward pass, with final activation in place and the unstandardization (if we have a regression task)"""
        _x = self.standardize(x_data, self.x_means, self.x_stds)
        prediction = self.forward(_x, has_bias_present=False)

        if self.task == "regression":
            prediction = self.unstandardize(prediction, self.y_means, self.y_stds)
        else:
            prediction = prediction

        return self.func["activation"](prediction)

    def elasticnet_loss(self) -> NDArray:
        """implementing elasticnet factor on loss"""
        lasso_penalty = np.sum(np.abs(self.weights))
        ridge_penalty = np.sum(self.weights**2)
        reg_loss = self.reg_lambda * (
            (self.reg_alpha * lasso_penalty) + ((1 - self.reg_alpha) * ridge_penalty)
        )
        return reg_loss

    def elasticnet_grad(self) -> NDArray:
        """implementing elasticnet factor on gradients"""
        reg_grad = self.reg_lambda * (
            (self.reg_alpha * np.sign(self.weights))
            + ((1 - self.reg_alpha) * 2 * self.weights)
        )
        return reg_grad

    def get_residuals(self):
        """get the residuals from the last fit"""
        return self.residuals
