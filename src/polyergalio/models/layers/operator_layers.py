"""
Operations -- track the incoming streams and backpropagate appropriately.
I started calling them latentXYZ because I was using them in latent spaces. No other reason.
"""
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.layers.basal_layers import ANY_SHAPE, Layer


class LatentStack(Layer):
    """
    stack via latent dimension (-1) position
    """

    def __init__(self):
        super().__init__()
        self.declare_shapes(inputs=(ANY_SHAPE, ANY_SHAPE), outputs=(ANY_SHAPE,))
        self.split_dims = ()

    def infer_output_shapes(self, input_shapes: tuple[tuple, ...]) -> tuple[tuple, ...]:
        """
        The output width is the sum of the two incoming widths, so it is only known once those widths enter the layer.
        """
        widths = [shape[-1] for shape in input_shapes]
        if any(width is None for width in widths):
            return (ANY_SHAPE,)
        return ((sum(widths),),)

    def forward(self, array_a: NDArray, array_b: NDArray = None) -> NDArray:
        """
        Stack the data together along the hidden dimension
        Parameters
        ----------
        array_a
        array_b

        Returns
        -------
        [array_a, array_b]
        """
        self.a_shape = array_a.shape
        self.b_shape = array_b.shape
        assert self.a_shape[0] == self.b_shape[0]

        self.split_dims = (self.a_shape[-1], self.b_shape[-1])

        if array_a.ndim == 1:
            return np.hstack((array_a.ravel(), array_b.ravel()))
        elif array_a.ndim == 2:
            return np.hstack((array_a, array_b))
        elif array_a.ndim == 3:
            return np.dstack((array_a, array_b))

    def backward(self, incoming_gradient: NDArray) -> tuple[NDArray, NDArray]:
        """
        # send the gradients to their correct inputs ("slices")
        Parameters
        ----------
        incoming_gradient : backpassed grad

        Returns
        -------
        the "split" gradients -> ordered in the same fashion as the inputs to the forward kwargs (a, b).
        """
        a_dim, b_dim = self.split_dims

        grad_a = incoming_gradient[..., :a_dim]
        grad_b = incoming_gradient[..., a_dim:a_dim + b_dim]

        return grad_a, grad_b

    def purge(self):
        self.split_dims, self.a_shape, self.b_shape = None, None, None

    def update_weights(self, **kwargs) -> None:
        pass

    def zero_gradients(self) -> None:
        pass

    def get_weights(self, for_serialize: bool = False):
        return {} if for_serialize else None

    def set_weights(self, weights: dict) -> None:
        pass


class LatentSum(Layer):
    """
    sum two arrays together, element-wise
    """
    def __init__(self, broadcast_axis: Optional[int] = None):
        super().__init__()
        self.array_a: Optional[NDArray] = None
        self.array_b: Optional[NDArray] = None
        self.output: Optional[NDArray] = None

        self.broadcast_axis = broadcast_axis
        self._expanded_dim = None


        # Gradients calculated during backward pass
        self.gradient_array_a: Optional[NDArray] = None
        self.gradient_array_b: Optional[NDArray] = None

        self.is_output: bool = False

        self.declare_shapes(inputs=(ANY_SHAPE, ANY_SHAPE), outputs=(ANY_SHAPE,))
        self.zero_gradients()


    @staticmethod
    def _reduce_to_shape(grad: NDArray, target_shape: tuple) -> NDArray:
        """ sum the gradients down to target_shape wherever we have to broadcast"""
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def forward(self,
                array_a: NDArray,
                array_b: NDArray,
                mask: Optional[NDArray] = None) -> NDArray:
        """
        Sums array_a and array_b element-wise.

        mask : unused -- the sum is per-position; accepted for pass-through
        compatibility with the graph.
        """
        self.in_shape_1 = array_a.shape
        self.in_shape_2 = array_b.shape
        ndim1, ndim2 = len(self.in_shape_1), len(self.in_shape_2)
        axis = self.broadcast_axis if self.broadcast_axis is not None else 1

        try:
            if ndim1 < ndim2:
                sum_array = np.expand_dims(array_a, axis) + array_b
                self._expanded_dim = (1, axis)
            elif ndim1 > ndim2:
                sum_array = array_a + np.expand_dims(array_b, axis)
                self._expanded_dim = (2, axis)
            else:
                sum_array = array_a + array_b
                self._expanded_dim = None
        except ValueError as e:
            raise RuntimeError(f"SummingLayer: Inputs not broadcastable. {e}")

        self.array_a = array_a
        self.array_b = array_b
        self.output = sum_array

        return self.output

    def backward(
            self, incoming_grad: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Backward pass. Calculates gradients for array_a and array_b.

        Parameters
        ----------
        incoming_grad : NDArray
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        Tuple[NDArray, NDArray]
        """
        if self._expanded_dim is not None:
            which, axis = self._expanded_dim
            if which == 1:
                grad_1 = incoming_grad.sum(axis=axis)
                grad_2 = incoming_grad
            else:
                grad_1 = incoming_grad
                grad_2 = incoming_grad.sum(axis=axis)
        else:
            grad_1 = incoming_grad
            grad_2 = incoming_grad

        self.gradient_array_a = self._reduce_to_shape(grad_1, self.in_shape_1)
        self.gradient_array_b = self._reduce_to_shape(grad_2, self.in_shape_2)

        return self.gradient_array_a, self.gradient_array_b

    def update_weights(self, **kwargs) -> None:
        """
        Pass through -- no weights to update
        """
        pass

    def purge(self) -> None:
        self.array_a = None
        self.array_b = None
        self.output = None
        self._expanded_dim = None

    def get_weights(self, for_serialize: bool = False):
        return {} if for_serialize else None

    def set_weights(self, weights: dict) -> None:
        pass

    def zero_gradients(self) -> None:
        """
        Zeroes the stored gradient values.
        """
        self.gradient_array_a = np.zeros_like(self.array_a)
        self.gradient_array_b = np.zeros_like(self.array_b)
        self.gradient = None

    @property
    def num_parameters(self) -> int:
        return 0

    def __str__(self):
        return "SummingLayer"

    def __repr__(self):
        return f"{self}"


class LatentProduct(Layer):
    """
    multiply two arrays together, element-wise
    """
    def __init__(self, broadcast_axis: Optional[int] = None):
        super().__init__()
        self.array_a: Optional[NDArray] = None
        self.array_b: Optional[NDArray] = None
        self.output: Optional[NDArray] = None

        self.broadcast_axis = broadcast_axis
        self._expanded_dim = None
        self._bcast_1: Optional[NDArray] = None
        self._bcast_2: Optional[NDArray] = None

        self.gradient_array_a: Optional[NDArray] = None
        self.gradient_array_b: Optional[NDArray] = None

        self.declare_shapes(inputs=(ANY_SHAPE, ANY_SHAPE), outputs=(ANY_SHAPE,))
        self.zero_gradients()

    @staticmethod
    def _reduce_to_shape(grad: NDArray, target_shape: tuple) -> NDArray:
        """sum the gradients down to target_shape wherever we have to broadcast"""
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def forward(self,
                array_a: NDArray,
                array_b: NDArray) -> NDArray:
        """
        Multiplies array_a and array_b element-wise.

        mask : unused -- the product is per-position; accepted for
        pass-through compatibility with the graph.
        """
        self.in_shape_1 = array_a.shape
        self.in_shape_2 = array_b.shape
        ndim1, ndim2 = len(self.in_shape_1), len(self.in_shape_2)
        axis = self.broadcast_axis if self.broadcast_axis is not None else 1

        try:
            if ndim1 < ndim2:
                self._bcast_1 = np.expand_dims(array_a, axis)
                self._bcast_2 = array_b
                self._expanded_dim = (1, axis)
            elif ndim1 > ndim2:
                self._bcast_1 = array_a
                self._bcast_2 = np.expand_dims(array_b, axis)
                self._expanded_dim = (2, axis)
            else:
                self._bcast_1 = array_a
                self._bcast_2 = array_b
                self._expanded_dim = None
            product_array = self._bcast_1 * self._bcast_2
        except ValueError as e:
            raise RuntimeError(f"LatentProduct: Inputs not broadcastable. {e}")

        self.array_a = array_a
        self.array_b = array_b
        self.output = product_array

        return self.output

    def backward(
            self, incoming_grad: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Backward pass. Product rule against the other (broadcast) input.

        Parameters
        ----------
        incoming_grad : NDArray
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        Tuple[NDArray, NDArray]
        """
        grad_1 = incoming_grad * self._bcast_2
        grad_2 = incoming_grad * self._bcast_1

        if self._expanded_dim is not None:
            which, axis = self._expanded_dim
            if which == 1:
                grad_1 = grad_1.sum(axis=axis)
            else:
                grad_2 = grad_2.sum(axis=axis)

        self.gradient_array_a = self._reduce_to_shape(grad_1, self.in_shape_1)
        self.gradient_array_b = self._reduce_to_shape(grad_2, self.in_shape_2)

        return self.gradient_array_a, self.gradient_array_b

    def update_weights(self, **kwargs) -> None:
        """
        Pass through -- no weights to update
        """
        pass

    def purge(self) -> None:
        self.array_a = None
        self.array_b = None
        self.output = None
        self._bcast_1 = None
        self._bcast_2 = None
        self._expanded_dim = None

    def get_weights(self, for_serialize: bool = False):
        return {} if for_serialize else None

    def set_weights(self, weights: dict) -> None:
        pass

    def zero_gradients(self) -> None:
        """
        Zeroes the stored gradient values.
        """
        self.gradient_array_a = np.zeros_like(self.array_a)
        self.gradient_array_b = np.zeros_like(self.array_b)
        self.gradient = None

    @property
    def num_parameters(self) -> int:
        return 0

    def __str__(self):
        return "LatentProduct"

    def __repr__(self):
        return f"{self}"


class LatentDifference(Layer):
    """
    subtract two arrays element-wise: array_a - array_b
    """
    def __init__(self, broadcast_axis: Optional[int] = None):
        super().__init__()
        self.array_a: Optional[NDArray] = None
        self.array_b: Optional[NDArray] = None
        self.output: Optional[NDArray] = None

        self.broadcast_axis = broadcast_axis
        self._expanded_dim = None

        self.gradient_array_a: Optional[NDArray] = None
        self.gradient_array_b: Optional[NDArray] = None

        self.declare_shapes(inputs=(ANY_SHAPE, ANY_SHAPE), outputs=(ANY_SHAPE,))
        self.zero_gradients()

    @staticmethod
    def _reduce_to_shape(grad: NDArray, target_shape: tuple) -> NDArray:
        """sum the gradients down to target_shape wherever we have to broadcast"""
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def forward(self,
                array_a: NDArray,
                array_b: NDArray,
                mask: Optional[NDArray] = None) -> NDArray:
        """
        Subtracts array_b from array_a, element-wise.

        mask : unused -- the difference is per-position; accepted for
        pass-through compatibility with the graph.
        """
        self.in_shape_1 = array_a.shape
        self.in_shape_2 = array_b.shape
        ndim1, ndim2 = len(self.in_shape_1), len(self.in_shape_2)
        axis = self.broadcast_axis if self.broadcast_axis is not None else 1

        try:
            if ndim1 < ndim2:
                diff_array = np.expand_dims(array_a, axis) - array_b
                self._expanded_dim = (1, axis)
            elif ndim1 > ndim2:
                diff_array = array_a - np.expand_dims(array_b, axis)
                self._expanded_dim = (2, axis)
            else:
                diff_array = array_a - array_b
                self._expanded_dim = None
        except ValueError as e:
            raise RuntimeError(f"LatentDifference: Inputs not broadcastable. {e}")

        self.array_a = array_a
        self.array_b = array_b
        self.output = diff_array

        return self.output

    def backward(
            self, incoming_grad: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Backward pass. array_b's gradient carries the sign flip from the
        subtraction; array_a's does not.

        Parameters
        ----------
        incoming_grad : NDArray
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        Tuple[NDArray, NDArray]
        """
        if self._expanded_dim is not None:
            which, axis = self._expanded_dim
            if which == 1:
                grad_1 = incoming_grad.sum(axis=axis)
                grad_2 = -incoming_grad
            else:
                grad_1 = incoming_grad
                grad_2 = -incoming_grad.sum(axis=axis)
        else:
            grad_1 = incoming_grad
            grad_2 = -incoming_grad

        self.gradient_array_a = self._reduce_to_shape(grad_1, self.in_shape_1)
        self.gradient_array_b = self._reduce_to_shape(grad_2, self.in_shape_2)

        return self.gradient_array_a, self.gradient_array_b

    def update_weights(self, **kwargs) -> None:
        """
        Pass through -- no weights to update
        """
        pass

    def purge(self) -> None:
        self.array_a = None
        self.array_b = None
        self.output = None
        self._expanded_dim = None

    def get_weights(self, for_serialize: bool = False):
        return {} if for_serialize else None

    def set_weights(self, weights: dict) -> None:
        pass

    def zero_gradients(self) -> None:
        """
        Zeroes the stored gradient values.
        """
        self.gradient_array_a = np.zeros_like(self.array_a)
        self.gradient_array_b = np.zeros_like(self.array_b)
        self.gradient = None

    @property
    def num_parameters(self) -> int:
        return 0

    def __str__(self):
        return "LatentDifference"

    def __repr__(self):
        return f"{self}"


class ShiftRight(Layer):
    """
    Shift a sequence right by one position along the sequence axis, so
    position t sees position t-1's value instead of its own
    """

    preserves_shape = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.start_token: NDArray = np.zeros((1, 1, hidden_dim))

        self.declare_shapes(inputs=((hidden_dim,),), outputs=((hidden_dim,),))
        self.zero_gradients()

    def forward(self, input_data: NDArray) -> NDArray:
        """
        input_data : (batch, sequence, hidden)
        """
        assert input_data.ndim == 3, (
            f"expected (batch, sequence, hidden), got shape {input_data.shape}"
        )
        assert input_data.shape[-1] == self.hidden_dim, (
            f"built for hidden_dim {self.hidden_dim}, got {input_data.shape[-1]}"
        )
        batch = input_data.shape[0]
        start = np.broadcast_to(self.start_token, (batch, 1, self.hidden_dim))
        self.output = np.concatenate([start, input_data[:, :-1, :]], axis=1)
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        self.gradient_start_token = incoming_gradient[:, :1, :].sum(
            axis=0, keepdims=True
        )
        # -1?
        grad_input = np.zeros_like(incoming_gradient)
        grad_input[:, :-1, :] = incoming_gradient[:, 1:, :]
        return grad_input

    def update_weights(self, gradient_start_token: NDArray) -> None:
        self.start_token -= gradient_start_token

    def zero_gradients(self) -> None:
        self.gradient_start_token = np.zeros_like(self.start_token)

    def get_weights(self, for_serialize: bool = False):
        if for_serialize:
            return {"start_token": self.start_token}
        return self.start_token

    def set_weights(self, weights: dict) -> None:
        if weights:
            self.start_token = np.asarray(weights["start_token"])

    def get_gradients(self) -> dict[str, NDArray]:
        return {"gradient_start_token": self.gradient_start_token}

    def purge(self) -> None:
        self.output = None

    @property
    def num_parameters(self) -> int:
        return self.start_token.size

    def __str__(self):
        return f"ShiftRight, hidden {self.hidden_dim}"

    def __repr__(self):
        return self.__str__()

class MaskGather(Layer):
    """
    Selects specific positions out of a (batch, sequence, hidden) tensor
    for a downstream head to run on; scatters the gradient back into the
    full shape on the way back. No learnable parameters.

    Input shape: (batch, sequence, hidden)
    Output shape: (num_selected, hidden) -- flattened across batch and
        sequence, since a downstream head only needs a bag of vectors,
        not which row or position each one came from
    """
    preserves_shape = False

    def __init__(self):
        super().__init__()
        self.declare_shapes(inputs=((None, None, None),), outputs=((None, None),))
        self.select_mask = None
        self.in_shape = None
        self.zero_gradients()

    def forward(self, incoming_x, mask=None, target_mask=None):
        """


        Parameters
        ----------
        incoming_x
        mask: (batch, sequence) or (batch, sequence, 1)
        target_mask: (batch, sequence) or (batch, sequence, 1) bool/0-1, which positions to pass forward (for training tasks like MLM)
        None keeps everything.

        Returns
        -------

        """
        self.in_shape = incoming_x.shape

        if mask is not None:
            self.select_mask = np.asarray(mask, dtype=bool)
        else:
            self.select_mask = np.ones(incoming_x.shape[:2], dtype=bool)

        if target_mask is not None:
            self.select_mask &= np.asarray(target_mask, dtype=bool)

        return incoming_x[self.select_mask]

    def backward(self, incoming_grad):
        full_grad = np.zeros(self.in_shape, dtype=incoming_grad.dtype)
        full_grad[self.select_mask] = incoming_grad
        return full_grad

    def update_weights(self): pass
    def zero_gradients(self): pass
    def get_weights(self, for_serialize=False): return {} if for_serialize else None
    def get_gradients(self): return {}
    def purge(self): self.select_mask = None; self.in_shape = None

    @property
    def num_parameters(self): return 0
