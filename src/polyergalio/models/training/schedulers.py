"""
Step schedules that move a value, or a whole array of features, from a start to an end along a curve.
"""
import numpy as np
from numpy.typing import NDArray


class Scheduler:
    def __init__(
        self,
        total_steps: int,
        parameters: dict[str, tuple[str, float, float]],
    ):
        """
        Moves each named parameter from its start to its end over total_steps along its own curve. An end below the
        start decays, an end above it accumulates.

        Parameters
        ----------
        total_steps : steps between start and end; step 0 gives the start and step total_steps - 1 the end
        parameters : name - (curve, start, end), curve "sigmoid" or "linear", e.g.
            {"learning_rate": ("sigmoid", 1e-3, 1e-6), "noise": ("linear", 0.1, 0.5)}
        """
        assert total_steps >= 2, f"total_steps must be at least 2, got {total_steps}"
        for name, (curve, _, _) in parameters.items():
            assert curve in ("sigmoid", "linear"), f"{name} curve must be 'sigmoid' or 'linear', got {curve}"
        self.total_steps: int = total_steps
        self.parameters = dict(parameters)
        self.current_step: int = 0
        self.current: dict[str, float] = {name: start for name, (_, start, _) in self.parameters.items()}

        sigmoid_steps = self.sigmoid(np.linspace(start=-10, stop=10, num=total_steps))
        self.sigmoid_steps = (sigmoid_steps - sigmoid_steps[0]) / (sigmoid_steps[-1] - sigmoid_steps[0])
        self.linear_steps = np.linspace(0, 1, num=total_steps)

    def step(self, current_step: int) -> None:
        """Move every parameter to the given step along its curve; values are read from `current`."""
        self.current_step = max(self.current_step, current_step)
        for name, (curve, start, end) in self.parameters.items():
            transition = self.sigmoid_transition if curve == "sigmoid" else self.linear_transition
            self.current[name] = float(transition(start, end, current_step))

    def get_curves(self) -> dict[str, float]:
        return dict(self.current)

    @staticmethod
    def sigmoid(x: NDArray):
        """Standard sigmoid function."""
        return 1 / (1 + np.exp(-x * 0.5))

    def sigmoid_transition(self, starting_value, ending_value, current_step):
        """
        Parameters
        ----------
        starting_value : float or array, the value at step 0
        ending_value : float or array broadcastable to starting_value, the value at the last step
        current_step : int

        Returns
        -------
        starting_value + (ending_value - starting_value) * sigmoid curve at current_step; the curve is normalized to
        run from exactly 0 to exactly 1
        """
        return self.interpolate(starting_value, ending_value, self.sigmoid_steps[current_step])

    def linear_transition(self, starting_value, ending_value, current_step):
        """As sigmoid_transition, along a straight line."""
        return self.interpolate(starting_value, ending_value, self.linear_steps[current_step])

    @staticmethod
    def interpolate(starting_value, ending_value, weight):
        """
        Parameters
        ----------
        starting_value, ending_value : float or array
        weight : float or array

        Returns
        -------
        (1 - weight) * starting_value + weight * ending_value
        """
        starting_value = np.asarray(starting_value, dtype=float)
        weight = np.asarray(weight, dtype=float)
        weight = weight.reshape(weight.shape + (1,) * (starting_value.ndim - weight.ndim))
        return starting_value + (np.asarray(ending_value, dtype=float) - starting_value) * weight
