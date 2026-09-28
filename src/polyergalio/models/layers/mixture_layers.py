from typing import Optional

import numpy as np
from polyergalio.models.layers.basal_layers import (
    FullyConnectedLayer,
    Layer,
)
from polyergalio.models.constants import EPSILON
from numpy.typing import NDArray


class VotingBase(Layer):
    renormalize: bool = False
    activation: str = "linear"

    def __init__(self,
                 input_shape: int,
                 num_experts: int,
                 top_k: Optional[int] = None,
                 bias_update_speed: float = 0.0,
                 num_groups: Optional[int] = None,
                 top_groups: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        input_shape : width of the incoming hidden state, e.g. hidden_dim
        num_experts : one vote per expert out
        top_k : keep only the k largest votes, or None for a dense vote
        bias_update_speed : per-forward-pass adjustment to a routing bias added to each expert's vote before top-k
            SELECTION to order -- auxiliary-loss-free load-balancing mechanism
        num_groups : split the experts into this many equal groups for group-limited routing (DeepSeek-V3);
            None routes over every expert
        top_groups : groups each token may route into, ranked by the sum of each group's n highest scores
        """
        super().__init__()
        assert top_k is None or 1 <= top_k <= num_experts, (
            f"top_k must fall in [1, {num_experts}], or be None for a dense "
            f"vote, got {top_k}"
        )
        assert not (self.renormalize and top_k == 1), (
            "top_k=1 on a renormalised vote is untrainable, increase topk"
        )
        if num_groups is not None:
            assert top_k is not None, "group-limited routing needs top_k"
            assert num_experts % num_groups == 0, (
                f"num_groups {num_groups} must divide num_experts {num_experts}"
            )
            assert top_groups is not None and 1 <= top_groups <= num_groups, (
                f"top_groups must fall in [1, {num_groups}], got {top_groups}"
            )
            assert top_k <= top_groups * (num_experts // num_groups), (
                f"top_k {top_k} exceeds the {top_groups * (num_experts // num_groups)} experts "
                f"in the top {top_groups} groups"
            )
        self.input_shape = input_shape
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_groups = num_groups
        self.top_groups = top_groups

        # a slowly-adapting correction, not a per-batch scratch value. this is the auxiliary load-balancing approach
        self.bias_update_speed = bias_update_speed

        self.expert_bias = np.zeros(num_experts)
        # (input_shape,) > (num_experts,) layer
        self.declare_shapes(inputs=((input_shape,),), outputs=((num_experts,),))

        self.stack: tuple[FullyConnectedLayer, ...] = ()
        self.in_shape = None
        self.mask = None
        self.token_mask = None
        self.route_sum = None
        self.output = None

        self.zero_gradients()

    def forward(self,
                incoming_x: NDArray,
                training_now: Optional[bool] = None,
                mask: Optional[NDArray] = None,
                ) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : our incoming data, (..., input_shape) -- any number of leading batch/ sequence axes
        training_now : whether the expert-load bias is updated. False (inference) leaves expert_bias untouched;
            None follows the layer's train() / eval() mode
        mask : (...,) matching incoming_x's leading axes, 1 for a real token and 0 for padding

        Returns
        -------
        one vote per expert, (num_samples, num_experts)
        """
        training_now = self.training if training_now is None else training_now
        self.in_shape = incoming_x.shape
        assert self.in_shape[-1] == self.input_shape, (
            f"cannot read trailing axis {self.in_shape[-1]} as input_shape "
            f"{self.input_shape}"
        )

        votes = incoming_x.reshape(-1, self.input_shape)
        for layer in self.stack:
            votes = layer(votes)

        if mask is not None:
            assert mask.shape == self.in_shape[:-1], (
                f"mask shape {mask.shape} must match incoming_x's leading "
                f"axes {self.in_shape[:-1]}"
            )
        self.token_mask = None if mask is None else mask.reshape(-1)

        self.output = self._route(votes, training_now=training_now)
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        grad = self._route_backward(incoming_grad)
        for layer in self.stack[::-1]:
            grad = layer.backward(grad)

        return grad.reshape(self.in_shape)

    def _route(self, votes: NDArray, training_now: bool = True) -> NDArray:
        """
        Core Routing Process

        final selection considers votes + expert_bias.
        bias steers which experts fire without entering the combined output or its gradient

        Parameters
        ----------
        votes: the output / forward pass
        training_now: bool - if we're training vs inference (bias factored into the update or not)

        Returns
        -------

        """
        if self.top_k is None:
            return votes

        self.mask = self.select_experts(votes)

        if training_now and self.bias_update_speed:
            self.update_expert_bias()

        kept = votes * self.mask
        if not self.renormalize:
            return kept

        self.route_sum = np.sum(kept, axis=-1, keepdims=True) + EPSILON
        return kept / self.route_sum

    def select_experts(self, votes: NDArray) -> NDArray:
        """
        (num_samples, num_experts) bool mask of the top_k experts per row, ranked on votes + expert_bias.
        With num_groups set, only experts inside each row's top_groups groups are eligible.
        """
        scores = votes + self.expert_bias if self.bias_update_speed else votes
        if self.num_groups is not None:
            grouped = scores.reshape(
                -1, self.num_groups, self.num_experts // self.num_groups
            )
            best_two = min(2, grouped.shape[-1])
            group_scores = np.sort(grouped, axis=-1)[..., -best_two:].sum(axis=-1)
            kept_groups = np.argpartition(group_scores, -self.top_groups, axis=-1)[
                ..., -self.top_groups :
            ]
            group_mask = np.zeros(group_scores.shape, dtype=bool)
            np.put_along_axis(group_mask, kept_groups, True, axis=-1)
            eligible = np.repeat(
                group_mask, self.num_experts // self.num_groups, axis=-1
            )
            scores = np.where(eligible, scores, -np.inf)

        keep = np.argpartition(scores, -self.top_k, axis=-1)[..., -self.top_k :]
        mask = np.zeros(votes.shape, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=-1)
        return mask

    def update_expert_bias(self) -> None:
        """
        auxiliary-loss-free balancing update, similar to Deepseek v3 (entirely separate from the gradient)
        it's a load-balancing term to increase less-seen experts, while decreasing over-seen experts.

        padded rows (masks in Forward) are left out of the load average, otherwise they'd count as real usage
        """
        if self.token_mask is None:
            load = self.mask.mean(axis=0)
        else:
            valid = self.token_mask.astype(self.mask.dtype)[:, None]
            valid_count = np.maximum(valid.sum(), 1.0)
            load = (self.mask * valid).sum(axis=0) / valid_count

        fair_share = self.top_k / self.num_experts
        self.expert_bias += self.bias_update_speed * np.sign(fair_share - load)

    def _route_backward(self, incoming_grad: NDArray) -> NDArray:
        """
        backpass through the route
        """
        if self.top_k is None:
            return incoming_grad
        if not self.renormalize:
            return incoming_grad * self.mask

        dot = np.sum(incoming_grad * self.output, axis=-1, keepdims=True)
        return self.mask * (incoming_grad - dot) / self.route_sum

    def _named_stack(self) -> dict[str, FullyConnectedLayer]:
        """keys the optimizer round-trips through get_gradients/update_weights"""
        return {f"fc_{n}": layer for n, layer in enumerate(self.stack, start=1)}

    def get_weights(self, for_serialize: bool = False):
        named = self._named_stack()
        if for_serialize:
            weights = {
                name: layer.get_weights(for_serialize=True)
                for name, layer in named.items()
            }
            weights["expert_bias"] = self.expert_bias
            return weights
        return tuple(layer.get_weights(for_serialize=False) for layer in self.stack) + (
            self.expert_bias,
        )

    def set_weights(self, weights: dict) -> None:
        if not weights:
            return
        for name, layer in self._named_stack().items():
            if name in weights:
                layer.set_weights(weights[name])
        if "expert_bias" in weights:
            self.expert_bias = np.asarray(weights["expert_bias"])

    def get_gradients(self) -> dict[str, dict[str, NDArray]]:
        return {
            name: layer.get_gradients() for name, layer in self._named_stack().items()
        }

    def update_weights(self, **gradients: dict[str, NDArray]) -> None:
        for name, layer in self._named_stack().items():
            layer.update_weights(**gradients[name])

    def zero_gradients(self) -> None:
        for layer in self.stack:
            layer.zero_gradients()

    def purge(self) -> None:
        for layer in self.stack:
            layer.purge()
        self.in_shape = None
        self.mask = None
        self.token_mask = None
        self.route_sum = None
        self.output = None

    @property
    def num_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self.stack)

    def __str__(self):
        routing = f"top {self.top_k} of" if self.top_k else "dense over"
        return (
            f"{type(self).__name__}, {self.activation} {routing} "
            f"{self.num_experts} experts on input width {self.input_shape}"
        )

    def __repr__(self):
        return self.__str__()


class VotingWeight(VotingBase):
    def __init__(self,
                 input_shape: int,
                 num_experts: int,
                 top_k: Optional[int] = None,
                 bias_update_speed: float = 0.0,
                 ):
        """
        Independent per-expert weights between 0 and 1 as a single projection

        The votes do not compete. Each expert's activation is judged by the voting weights on its own, so the weights
        don't have to sum to 1 / be normalized. (independent consideration per expert)

        Outputs are weighted multiplicative elements. (expert 0 should be weighted (mult) by W[0] value

        Parameters
        ----------
        input_shape : hidden width of the incoming hidden state.
        num_experts : number of experts to weight. Forward returns one weight per expert
        top_k : keep only the k largest weights, zeroing the rest. None keeps
            every expert.
        bias_update_speed : see VotingBase class for more - this is the load-balancing mechanism
        """
        super().__init__(
            input_shape, num_experts, top_k, bias_update_speed=bias_update_speed
        )
        self.activation = "sigmoid"
        self.stack = (
            FullyConnectedLayer(
                ni=input_shape,
                no=num_experts,
                activation_type="sigmoid",
                is_output=True,
            ),
        )

        self.zero_gradients()


class VotingWeightBalanced(VotingBase):
    """determines the weighting of each expert in the final output"""

    renormalize = True

    def __init__(self,
                 input_shape: int,
                 hidden_size: Optional[int],
                 num_experts: int,
                 top_k: Optional[int] = None,
                 gate_activation: str = "softmax",
                 bias_update_speed: float = 0.0,
                 num_groups: Optional[int] = None,
                 top_groups: Optional[int] = None,
                 ):
        """
        Unlike VotingWeight the experts compete here: top_k always re-norms; so raising one vote's weights lowers anothers

        Parameters
        ----------
        input_shape : hidden width of the incoming hidden state.
        hidden_size : width of the relu hidden layer. None scores experts with a single projection, DeepSeek's
            per-expert centroid affinity
        num_experts : number of experts to vote over. The gated votes are used as weights, SUM(gate(vote) * expert_output)
        top_k : keep only the k largest votes and renormalise them back to a distribution None keeps every expert.
        gate_activation : final projection's activation
            "softmax" (default) makes every expert compete for one fixed budget of weight.
            "sigmoid" scores each expert independently before top-k selection and renorm
        bias_update_speed : see VotingBase. 0.0 (default) here too, so this stays a plain competitive gate
        num_groups, top_groups : group-limited routing, see VotingBase
        """
        super().__init__(input_shape,
                         num_experts,
                         top_k,
                         bias_update_speed=bias_update_speed,
                         num_groups=num_groups,
                         top_groups=top_groups,)
        self.activation = gate_activation
        self.gate_activation = gate_activation
        self.hidden_size = hidden_size
        score_width = input_shape if hidden_size is None else hidden_size
        hidden = (
            ()
            if hidden_size is None
            else (
                FullyConnectedLayer(
                    ni=input_shape, no=hidden_size, activation_type="relu"
                ),
            )
        )
        self.stack = hidden + (
            FullyConnectedLayer(
                ni=score_width,
                no=num_experts,
                activation_type=gate_activation,
                is_output=True,
            ),
        )


class VotingGate(VotingBase):
    """boolean pass/no-pass gate -- top_k experts fire at full strength, everyone else is off. No reweighting."""

    def __init__(self,
                 input_shape: int,
                 hidden_size: int,
                 num_experts: int,
                 top_k: int,
                 bias_update_speed: float = 0.0,
                 ):
        """
        Parameters
        ----------
        input_shape : hidden width of the incoming hidden state.
        hidden_size : width of the relu hidden layer.
        num_experts : number of experts to gate.
        top_k : how many experts pass per sample. Required -- a dense (top_k=None) boolean gate has nothing to gate.
        bias_update_speed : see VotingBase.
        """
        assert top_k is not None, (
            "VotingGate requires top_k -- a boolean gate with no top_k has "
            "nothing to gate"
        )
        super().__init__(input_shape,
                         num_experts,
                         top_k,
                         bias_update_speed=bias_update_speed,
                         )
        self.activation = "sigmoid"
        self.hidden_size = hidden_size
        self.stack = (
            FullyConnectedLayer(ni=input_shape, no=hidden_size, activation_type="relu"),
            FullyConnectedLayer(
                ni=hidden_size,
                no=num_experts,
                activation_type="sigmoid",
                is_output=True,
            ),
        )

    def _route(self, votes: NDArray, training_now: bool = True) -> NDArray:
        """
        Straight-through gate: forward is the pure top_k boolean mask, no vote
        magnitude passes through.

        Parameters
        ----------
        votes : the output / forward pass
        training_now : bool - if we're training vs inference (bias factored into the update or not)

        Returns
        -------
        """
        self.mask = self.select_experts(votes)

        if training_now and self.bias_update_speed:
            self.update_expert_bias()

        return self.mask.astype(votes.dtype)

    def _route_backward(self, incoming_grad: NDArray) -> NDArray:
        """
        Straight-through estimator: reuses the non-renormalised
        """
        return incoming_grad * self.mask


class Expert(Layer):
    """
    Gated feed-forward expert, down_proj(gate_proj(x) * up_proj(x))

    Input shape: (..., input_dim)
    Projected shape: (..., upscale_dim)
    Output shape: (..., hidden_dim)
    """

    def __init__(self,
                 input_dim: int,
                 upscale_dim: int,
                 hidden_dim: int,
                 activation_type: str = "swish",
                 ):
        """
        Parameters
        ----------
        input_dim : width of the incoming hidden state
        upscale_dim : width of the gated projection
        hidden_dim : width of the outgoing hidden state
        activation_type : activation on the gate projection; swish makes this SwiGLU
        """
        super().__init__()
        self.input_dim = input_dim
        self.upscale_dim = upscale_dim
        self.hidden_dim = hidden_dim
        self.activation_type = activation_type
        self.declare_shapes(inputs=((input_dim,),), outputs=((hidden_dim,),))

        self.gate_proj = FullyConnectedLayer(
            ni=input_dim, no=upscale_dim, activation_type=activation_type
        )
        self.up_proj = FullyConnectedLayer(
            ni=input_dim, no=upscale_dim, activation_type="linear"
        )
        self.down_proj = FullyConnectedLayer(
            ni=upscale_dim, no=hidden_dim, activation_type="linear"
        )

        self.gated = None
        self.up = None
        self.output = None

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """mask : unused -- each row is transformed on its own"""
        self.gated = self.gate_proj(incoming_x)
        self.up = self.up_proj(incoming_x)
        self.output = self.down_proj(self.gated * self.up)
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        grad_product = self.down_proj.backward(incoming_grad)
        return self.gate_proj.backward(grad_product * self.up) + self.up_proj.backward(
            grad_product * self.gated
        )

    def named_sublayers(self) -> dict[str, FullyConnectedLayer]:
        return {
            "gate_proj": self.gate_proj,
            "up_proj": self.up_proj,
            "down_proj": self.down_proj,
        }

    def get_weights(self, for_serialize: bool = False):
        return {
            name: layer.get_weights(for_serialize=for_serialize)
            for name, layer in self.named_sublayers().items()
        }

    def set_weights(self, weights: dict) -> None:
        if not weights:
            return
        for name, layer in self.named_sublayers().items():
            if name in weights:
                layer.set_weights(weights[name])

    def get_gradients(self) -> dict[str, dict]:
        return {
            name: layer.get_gradients()
            for name, layer in self.named_sublayers().items()
        }

    def update_weights(self, **gradients) -> None:
        for name, layer in self.named_sublayers().items():
            if name in gradients:
                layer.update_weights(**gradients[name])

    def zero_gradients(self) -> None:
        for layer in self.named_sublayers().values():
            layer.zero_gradients()

    def purge(self) -> None:
        for layer in self.named_sublayers().values():
            layer.purge()
        self.gated = None
        self.up = None
        self.output = None

    @property
    def num_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self.named_sublayers().values())

    def __str__(self):
        return (
            f"Expert, {self.activation_type}-gated, "
            f"{self.input_dim} -> {self.upscale_dim} -> {self.hidden_dim}"
        )

    def __repr__(self):
        return self.__str__()


class MixtureOfExperts(Layer):
    """
    DeepSeek-V3 mixture-of-experts FFN:

        output = sum(shared experts(x)) + routed_scaling * sum_{i in top_k} g_i * expert_i(x)

    Experts are FC / FFN with Swish. The gate scores each routed expert with sigmoid(x . e_i)
    top_k is chosen on score + expert_bias (auxiliary-loss-free balancing, the bias steers selection)
    and the chosen scores are renormalised to sum to 1
    optional group-limited routing restricts each token to its top_groups expert groups.

    Routed experts only run on the rows routed to them. Padded rows (mask == 0) are routed to no expert, so their
    output is the shared experts' alone.
    """
    def __init__(self,
                 input_dim: int,
                 upscale_dim: int,
                 hidden_dim: int,
                 num_shared_experts: int,
                 num_routed_experts: int,
                 top_k: int,
                 activation_type: str = "swish",
                 gate_activation: str = "sigmoid",
                 bias_update_speed: float = 1e-3,
                 routed_scaling: float = 1.0,
                 num_groups: Optional[int] = None,
                 top_groups: Optional[int] = None):
        """
        Parameters
        ----------
        input_dim : width of the incoming hidden state
        upscale_dim : width of each expert's gated projection
        hidden_dim : width of the outgoing hidden state
        num_shared_experts : experts every token passes through
        num_routed_experts : size of the routed expert pool the gate chooses top_k from
        top_k : routed experts per token, 2 <= top_k <= num_routed_experts -- the renormalised gate has no gradient at 1
        activation_type : the experts' gate-projection activation, swish for SwiGLU
        gate_activation : "sigmoid" (default, DeepSeek-V3) scores experts independently; "softmax" makes them compete
        bias_update_speed : gamma, the step of the load-balancing bias update
        routed_scaling : multiplier on the routed experts' combined output (DeepSeek-V3 uses 2.5)
        num_groups, top_groups : group-limited routing, see VotingBase. None routes over every expert
        """
        super().__init__()
        assert num_shared_experts >= 0, "num_shared_experts must be >= 0"
        assert 2 <= top_k <= num_routed_experts, (
            f"top_k must fall in [2, {num_routed_experts}], got {top_k}"
        )

        self.input_dim = input_dim
        self.upscale_dim = upscale_dim
        self.hidden_dim = hidden_dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.activation_type = activation_type
        self.gate_activation = gate_activation
        self.bias_update_speed = bias_update_speed
        self.routed_scaling = routed_scaling
        self.num_groups = num_groups
        self.top_groups = top_groups

        self.declare_shapes(inputs=((input_dim,),), outputs=((hidden_dim,),))

        self.shared_experts = tuple(
            Expert(input_dim, upscale_dim, hidden_dim, activation_type)
            for _ in range(num_shared_experts)
        )
        self.routed_experts = tuple(
            Expert(input_dim, upscale_dim, hidden_dim, activation_type)
            for _ in range(num_routed_experts)
        )

        self.gate = VotingWeightBalanced(
            input_shape=input_dim,
            hidden_size=None,
            num_experts=num_routed_experts,
            top_k=top_k,
            gate_activation=gate_activation,
            bias_update_speed=bias_update_speed,
            num_groups=num_groups,
            top_groups=top_groups,
        )

        self.in_shape = None
        self.gate_weights = None
        self.expert_rows: list[NDArray] = []
        self.expert_outputs: list[Optional[NDArray]] = []
        self.output = None

        self.zero_gradients()

    def forward(
        self,
        hidden_state: NDArray,
        training_now: Optional[bool] = None,
        mask: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Parameters
        ----------
        hidden_state : (..., input_dim), any number of leading batch/sequence axes
        training_now : whether the gate's load-balancing bias updates this pass; None follows train() / eval()
        mask : (...,) matching hidden_state's leading axes, 1 for a real token and 0 for padding

        Returns
        -------
        (..., hidden_dim)
        """
        training_now = self.training if training_now is None else training_now
        self.in_shape = hidden_state.shape
        rows = hidden_state.reshape(-1, self.input_dim)

        output = np.zeros((rows.shape[0], self.hidden_dim), dtype=rows.dtype)
        for expert in self.shared_experts:
            output = output + expert(rows)

        self.gate_weights = self.gate(
            hidden_state, training_now=training_now, mask=mask
        )
        routed = self.gate.mask
        if mask is not None:
            routed = routed & mask.reshape(-1, 1).astype(bool)

        self.expert_rows, self.expert_outputs = [], []
        for e, expert in enumerate(self.routed_experts):
            chosen = np.flatnonzero(routed[:, e])
            expert_out = expert(rows[chosen]) if chosen.size else None
            if chosen.size:
                output[chosen] += (self.routed_scaling * self.gate_weights[chosen, e : e + 1] * expert_out)

            self.expert_rows.append(chosen)
            self.expert_outputs.append(expert_out)

        self.output = output.reshape(self.in_shape[:-1] + (self.hidden_dim,))
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        grad_rows = incoming_grad.reshape(-1, self.hidden_dim)
        grad_input = np.zeros((grad_rows.shape[0], self.input_dim), dtype=grad_rows.dtype)

        for expert in self.shared_experts:
            grad_input = grad_input + expert.backward(grad_rows)

        gate_grad = np.zeros_like(self.gate_weights)
        for e, expert in enumerate(self.routed_experts):
            chosen, expert_out = self.expert_rows[e], self.expert_outputs[e]
            if not chosen.size:
                expert.zero_gradients()
                continue
            upstream = grad_rows[chosen]
            grad_input[chosen] += expert.backward(
                self.routed_scaling * self.gate_weights[chosen, e : e + 1] * upstream
            )
            gate_grad[chosen, e] = self.routed_scaling * np.sum(
                upstream * expert_out, axis=-1
            )

        grad_input = grad_input + self.gate.backward(gate_grad).reshape(grad_input.shape)
        return grad_input.reshape(self.in_shape)

    def named_sublayers(self) -> dict[str, Layer]:
        named = {f"shared_{n}": e for n, e in enumerate(self.shared_experts, start=1)}
        named.update(
            {f"routed_{n}": e for n, e in enumerate(self.routed_experts, start=1)}
        )
        named["gate"] = self.gate
        return named

    def get_weights(self, for_serialize: bool = False):
        return {
            name: layer.get_weights(for_serialize=for_serialize)
            for name, layer in self.named_sublayers().items()
        }

    def set_weights(self, weights: dict) -> None:
        if not weights:
            return
        for name, layer in self.named_sublayers().items():
            if name in weights:
                layer.set_weights(weights[name])

    def get_gradients(self) -> dict[str, dict]:
        return {
            name: layer.get_gradients()
            for name, layer in self.named_sublayers().items()
        }

    def update_weights(self, **gradients) -> None:
        for name, layer in self.named_sublayers().items():
            if name in gradients:
                layer.update_weights(**gradients[name])

    def zero_gradients(self) -> None:
        for layer in self.named_sublayers().values():
            layer.zero_gradients()

    def purge(self) -> None:
        for layer in self.named_sublayers().values():
            layer.purge()
        self.in_shape = None
        self.gate_weights = None
        self.expert_rows, self.expert_outputs = [], []
        self.output = None

    @property
    def num_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self.named_sublayers().values())

    def __str__(self):
        return (
            f"MixtureOfExperts, {self.num_shared_experts} shared + top "
            f"{self.top_k} of {self.num_routed_experts} routed experts, "
            f"{self.input_dim} -> {self.upscale_dim} -> {self.hidden_dim}"
        )

    def __repr__(self):
        return self.__str__()


class PoolingLayer(Layer):
    """
    Mean pooling layer to compute the average across the sequence dimension.

    Input shape: (batch, sequence, hidden)
    Output shape: (batch, 1, hidden)
    """

    preserves_shape = False

    def __init__(self):
        super().__init__()
        self.declare_shapes(inputs=((None, None, None),), outputs=((None, 1, None),))

        self.input = None
        self.weights = None
        self.counts = None

        self.zero_gradients()

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        mask : (batch, sequence), 1 for a real position and 0 for padding
        """
        self.input = incoming_x
        if mask is None:
            self.weights = None
            self.counts = None
            return incoming_x.mean(axis=1, keepdims=True)

        assert mask.shape == incoming_x.shape[:2], (
            f"mask shape {mask.shape} must match incoming_x's (batch, "
            f"sequence) axes {incoming_x.shape[:2]}"
        )
        self.weights = mask[..., None].astype(incoming_x.dtype)
        self.counts = np.maximum(self.weights.sum(axis=1, keepdims=True), 1.0)
        return (incoming_x * self.weights).sum(axis=1, keepdims=True) / self.counts

    def backward(self, incoming_grad: NDArray) -> NDArray:
        if self.input is None:
            return incoming_grad

        if self.counts is None:
            seq_len = self.input.shape[1]
            return np.broadcast_to(incoming_grad / seq_len, self.input.shape).copy()

        return (
            np.broadcast_to(incoming_grad / self.counts, self.input.shape)
            * self.weights
        )

    def update_weights(self) -> None:
        pass

    def zero_gradients(self) -> None:
        pass

    def get_weights(self, for_serialize: bool = False):
        return {} if for_serialize else None

    def get_gradients(self) -> dict[str, NDArray]:
        return {}

    def purge(self):
        self.input = None
        self.weights = None
        self.counts = None

    @property
    def num_parameters(self) -> int:
        return 0

    def __str__(self):
        return "Layer of Sequence Mean Pooling (avg over sequence)"

    def __repr__(self):
        return self.__str__()
