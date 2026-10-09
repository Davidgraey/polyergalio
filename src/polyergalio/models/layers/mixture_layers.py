from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.constants import EPSILON
from polyergalio.models.layers.basic_layers import FullyConnectedLayer, Layer


class VotingBase(Layer):
    """
    A stack of fully connected layers scoring experts, followed by a routing step.

    The stack is held as attributes fc_1, fc_2, ... so each layer is a named parameter.
    expert_bias steers expert selection and is moved by load balance, not by a loss gradient.
    """
    renormalize: bool = False
    activation: str = "linear"
    parameter_names = ("expert_bias",)
    cache_names = ("in_shape", "mask", "token_mask", "route_sum", "output")

    def __init__(self,
                 input_shape: int,
                 num_experts: int,
                 top_k: Optional[int] = None,
                 num_groups: Optional[int] = None,
                 top_groups: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        input_shape : width of the incoming hidden state, e.g. hidden_dim
        num_experts : one vote per expert out
        top_k : keep only the k largest votes, or leave None for a dense vote
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

        self.expert_bias = np.zeros(num_experts)
        self.declare_shapes(inputs=((input_shape,),), outputs=((num_experts,),))
        self.zero_gradients()

    @property
    def stack(self) -> tuple[FullyConnectedLayer, ...]:
        return tuple(getattr(self, name) for name in self.parameter_names[1:])

    def set_stack(self, *layers: FullyConnectedLayer) -> None:
        """Hold the layers as fc_1, fc_2, ... and register them as parameters."""
        names = tuple(f"fc_{n}" for n in range(1, len(layers) + 1))
        for name, layer in zip(names, layers):
            setattr(self, name, layer)
        self.parameter_names = ("expert_bias",) + names
        self.zero_gradients()

    def forward(self,
                incoming_x: NDArray,
                mask: Optional[NDArray] = None,
                ) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : our incoming data, (..., input_shape) -- any number of leading batch/ sequence axes
        mask : (...,) matching incoming_x's leading axes, 1 for a real token and 0 for padding

        Returns
        -------
        one vote per expert, (num_samples, num_experts)
        """
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

        self.output = self.route(votes)
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        self.gradient_expert_bias = -self.balance_signal()
        grad = self.route_backward(incoming_grad)
        for layer in self.stack[::-1]:
            grad = layer.backward(grad)

        return grad.reshape(self.in_shape)

    def route(self, votes: NDArray) -> NDArray:
        """
        Select experts on votes + expert_bias, which steers which experts fire without entering the combined
        output or its gradient.

        Parameters
        ----------
        votes : the stack's output

        Returns
        -------
        the votes of the selected experts, renormalised per row when renormalize is set
        """
        if self.top_k is None:
            return votes

        self.mask = self.select_experts(votes)

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
        scores = votes + self.expert_bias
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

    def balance_signal(self) -> NDArray:
        """
        Direction that moves expert_bias towards an even load, in the spirit of DeepSeek-V3's auxiliary-loss-free
        balancing: +1 for an under-used expert, -1 for an over-used one. Padded rows are left out of the load.
        """
        if self.mask is None:
            return np.zeros(self.num_experts)

        if self.token_mask is None:
            load = self.mask.mean(axis=0)
        else:
            valid = self.token_mask.astype(self.mask.dtype)[:, None]
            valid_count = np.maximum(valid.sum(), 1.0)
            load = (self.mask * valid).sum(axis=0) / valid_count

        fair_share = self.top_k / self.num_experts
        return np.sign(fair_share - load)

    def route_backward(self, incoming_grad: NDArray) -> NDArray:
        """Gradient through route()."""
        if self.top_k is None:
            return incoming_grad
        if not self.renormalize:
            return incoming_grad * self.mask

        dot = np.sum(incoming_grad * self.output, axis=-1, keepdims=True)
        return self.mask * (incoming_grad - dot) / self.route_sum

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
        """
        super().__init__(
            input_shape, num_experts, top_k
        )
        self.activation = "sigmoid"
        self.set_stack(
            FullyConnectedLayer(
                input_dimension=input_shape,
                output_dimension=num_experts,
                activation_type="sigmoid",
                is_output=True,
            ),
        )


class VotingWeightBalanced(VotingBase):
    """determines the weighting of each expert in the final output"""

    renormalize = True

    def __init__(self,
                 input_shape: int,
                 hidden_size: Optional[int],
                 num_experts: int,
                 top_k: Optional[int] = None,
                 gate_activation: str = "softmax",
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
        num_groups, top_groups : group-limited routing, see VotingBase
        """
        super().__init__(input_shape,
                         num_experts,
                         top_k,
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
                    input_dimension=input_shape, output_dimension=hidden_size, activation_type="relu"
                ),
            )
        )
        self.set_stack(
            *hidden,
            FullyConnectedLayer(
                input_dimension=score_width,
                output_dimension=num_experts,
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
                 ):
        """
        Parameters
        ----------
        input_shape : hidden width of the incoming hidden state.
        hidden_size : width of the relu hidden layer.
        num_experts : number of experts to gate.
        top_k : how many experts pass per sample. Required -- a dense (top_k=None) boolean gate has nothing to gate.
        """
        assert top_k is not None, (
            "VotingGate requires top_k -- a boolean gate with no top_k has "
            "nothing to gate"
        )
        super().__init__(input_shape,
                         num_experts,
                         top_k,
                         )
        self.activation = "sigmoid"
        self.hidden_size = hidden_size
        self.set_stack(
            FullyConnectedLayer(input_dimension=input_shape, output_dimension=hidden_size, activation_type="relu"),
            FullyConnectedLayer(
                input_dimension=hidden_size,
                output_dimension=num_experts,
                activation_type="sigmoid",
                is_output=True,
            ),
        )

    def route(self, votes: NDArray) -> NDArray:
        """
        Straight-through gate: forward is the pure top_k boolean mask, no vote
        magnitude passes through.
        """
        self.mask = self.select_experts(votes)

        return self.mask.astype(votes.dtype)

    def route_backward(self, incoming_grad: NDArray) -> NDArray:
        """Straight-through estimator: the gradient passes where the mask is set."""
        return incoming_grad * self.mask


class Expert(Layer):
    """
    Gated feed-forward expert, down_proj(gate_proj(x) * up_proj(x))

    Input shape: (..., input_dim)
    Projected shape: (..., upscale_dim)
    Output shape: (..., hidden_dim)
    """
    parameter_names = ("gate_proj", "up_proj", "down_proj")
    cache_names = ("gated", "up", "output")

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
            input_dimension=input_dim, output_dimension=upscale_dim, activation_type=activation_type
        )
        self.up_proj = FullyConnectedLayer(
            input_dimension=input_dim, output_dimension=upscale_dim, activation_type="linear"
        )
        self.down_proj = FullyConnectedLayer(
            input_dimension=upscale_dim, output_dimension=hidden_dim, activation_type="linear"
        )

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

    Experts are held as attributes shared_1.., routed_1.. and gate, which are the parameter names.
    """
    cache_names = ("in_shape", "gate_weights", "expert_rows", "expert_outputs", "output")

    def __init__(self,
                 input_dim: int,
                 upscale_dim: int,
                 hidden_dim: int,
                 num_shared_experts: int,
                 num_routed_experts: int,
                 top_k: int,
                 activation_type: str = "swish",
                 gate_activation: str = "sigmoid",
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
        top_k : routed experts per token, 2 <= top_k <= num_routed_experts
        activation_type : the experts' gate-projection activation, swish for Swish
        gate_activation : "sigmoid" (default, DeepSeek-V3) scores experts independently; "softmax" makes them compete
        routed_scaling : multiplier on the routed experts' combined output (DeepSeek-V3 uses 2.5)
        num_groups, top_groups : group-limited routing, see VotingBase. None routes over every expert
        """
        super().__init__()
        assert num_shared_experts >= 0, "num_shared_experts must be >= 0"
        assert 1 <= top_k <= num_routed_experts, (
            f"top_k must fall in [1, {num_routed_experts}], got {top_k}"
        )

        self.input_dim = input_dim
        self.upscale_dim = upscale_dim
        self.hidden_dim = hidden_dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.activation_type = activation_type
        self.gate_activation = gate_activation
        self.routed_scaling = routed_scaling
        self.num_groups = num_groups
        self.top_groups = top_groups

        self.declare_shapes(inputs=((input_dim,),), outputs=((hidden_dim,),))

        shared_names = tuple(f"shared_{n}" for n in range(1, num_shared_experts + 1))
        routed_names = tuple(f"routed_{n}" for n in range(1, num_routed_experts + 1))
        for name in shared_names + routed_names:
            setattr(self, name, Expert(input_dim, upscale_dim, hidden_dim, activation_type))
        self.gate = VotingWeightBalanced(
            input_shape=input_dim,
            hidden_size=None,
            num_experts=num_routed_experts,
            top_k=top_k,
            gate_activation=gate_activation,
            num_groups=num_groups,
            top_groups=top_groups,
        )
        self.parameter_names = shared_names + routed_names + ("gate",)
        self.zero_gradients()

    @property
    def shared_experts(self) -> tuple[Expert, ...]:
        return tuple(getattr(self, f"shared_{n}") for n in range(1, self.num_shared_experts + 1))

    @property
    def routed_experts(self) -> tuple[Expert, ...]:
        return tuple(getattr(self, f"routed_{n}") for n in range(1, self.num_routed_experts + 1))

    def forward(
        self,
        hidden_state: NDArray,
        mask: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Parameters
        ----------
        hidden_state : (..., input_dim), any number of leading batch/sequence axes
        mask : (...,) matching hidden_state's leading axes, 1 for a real token and 0 for padding

        Returns
        -------
        (..., hidden_dim)
        """
        self.in_shape = hidden_state.shape
        rows = hidden_state.reshape(-1, self.input_dim)

        output = np.zeros((rows.shape[0], self.hidden_dim), dtype=rows.dtype)
        for expert in self.shared_experts:
            output = output + expert(rows)

        self.gate_weights = self.gate(
            hidden_state, mask=mask
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

    def __str__(self):
        return (
            f"MixtureOfExperts, {self.num_shared_experts} shared + top "
            f"{self.top_k} of {self.num_routed_experts} routed experts, "
            f"{self.input_dim} -> {self.upscale_dim} -> {self.hidden_dim}"
        )

    def __repr__(self):
        return self.__str__()


class SimpleMixtureOfExperts(Layer):
    """
    Sample-level mixture of experts:

        output = sum(shared experts(x)) + mean_{i in top_k} routed_i(x)

    Routing is decided once per sample, not per token. Each sample's tokens are mean-pooled over the sequence
    (padding excluded) and a VotingGate picks the top_k routed experts from that pooled vector; every token of the
    sample then goes through the same experts. Shared experts are always on.

    The gate is boolean, so chosen experts fire at full strength and the gate trains by straight-through gradients.
    expert_bias evens out the load across samples.

    Input shape: (batch, ..., input_dim), axis 0 is the sample, the axes before input_dim are pooled for routing
    Output shape: (batch, ..., hidden_dim)
    """
    cache_names = ("in_shape", "selected", "expert_samples", "expert_outputs", "output")

    def __init__(self,
                 input_dim: int,
                 upscale_dim: int,
                 hidden_dim: int,
                 num_shared_experts: int,
                 num_routed_experts: int,
                 top_k: int,
                 gate_hidden: Optional[int] = None,
                 activation_type: str = "swish"):
        """
        Parameters
        ----------
        input_dim : width of the incoming hidden state
        upscale_dim : width of each expert's gated projection
        hidden_dim : width of the outgoing hidden state
        num_shared_experts : experts every sample passes through
        num_routed_experts : size of the pool the gate chooses top_k from
        top_k : routed experts per sample, 1 <= top_k <= num_routed_experts
        gate_hidden : width of the gate's relu hidden layer, input_dim if None
        activation_type : the experts' gate-projection activation, swish for SwiGLU
        """
        super().__init__()
        assert num_shared_experts >= 0, "num_shared_experts must be >= 0"
        assert 1 <= top_k <= num_routed_experts, (
            f"top_k must fall in [1, {num_routed_experts}], got {top_k}"
        )

        self.input_dim = input_dim
        self.upscale_dim = upscale_dim
        self.hidden_dim = hidden_dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.gate_hidden = gate_hidden
        self.activation_type = activation_type

        self.declare_shapes(inputs=((input_dim,),), outputs=((hidden_dim,),))

        shared_names = tuple(f"shared_{n}" for n in range(1, num_shared_experts + 1))
        routed_names = tuple(f"routed_{n}" for n in range(1, num_routed_experts + 1))
        for name in shared_names + routed_names:
            setattr(self, name, Expert(input_dim, upscale_dim, hidden_dim, activation_type))
        self.pooling = PoolingLayer()
        self.gate = VotingGate(
            input_shape=input_dim,
            hidden_size=gate_hidden or input_dim,
            num_experts=num_routed_experts,
            top_k=top_k,
        )
        self.parameter_names = shared_names + routed_names + ("gate",)
        self.zero_gradients()

    @property
    def shared_experts(self) -> tuple[Expert, ...]:
        return tuple(getattr(self, f"shared_{n}") for n in range(1, self.num_shared_experts + 1))

    @property
    def routed_experts(self) -> tuple[Expert, ...]:
        return tuple(getattr(self, f"routed_{n}") for n in range(1, self.num_routed_experts + 1))

    def forward(self, hidden_state: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        hidden_state : (batch, ..., input_dim)
        mask : (batch, ...) matching hidden_state's leading axes, 1 for a real token and 0 for padding;
            a sample with no real token routes to no expert

        Returns
        -------
        (batch, ..., hidden_dim)
        """
        self.in_shape = hidden_state.shape
        batch = self.in_shape[0]
        tokens = hidden_state.reshape(batch, -1, self.input_dim)

        token_mask, sample_mask = None, None
        if mask is not None:
            assert mask.shape == self.in_shape[:-1], (
                f"mask shape {mask.shape} must match hidden_state's leading axes {self.in_shape[:-1]}"
            )
            token_mask = mask.reshape(batch, -1)
            sample_mask = token_mask.any(axis=1)
        pooled = self.pooling(tokens, mask=token_mask)[:, 0]

        output = np.zeros((batch, tokens.shape[1], self.hidden_dim), dtype=tokens.dtype)
        for expert in self.shared_experts:
            output = output + expert(tokens)

        self.selected = self.gate(pooled, mask=sample_mask).astype(bool)
        if sample_mask is not None:
            self.selected = self.selected & sample_mask[:, None]

        scale = 1.0 / self.top_k
        self.expert_samples, self.expert_outputs = [], []
        for e, expert in enumerate(self.routed_experts):
            chosen = np.flatnonzero(self.selected[:, e])
            expert_out = expert(tokens[chosen]) if chosen.size else None
            if chosen.size:
                output[chosen] += scale * expert_out
            self.expert_samples.append(chosen)
            self.expert_outputs.append(expert_out)

        self.output = output.reshape(self.in_shape[:-1] + (self.hidden_dim,))
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        batch = self.in_shape[0]
        grad_tokens = incoming_grad.reshape(batch, -1, self.hidden_dim)
        grad_input = np.zeros((batch, grad_tokens.shape[1], self.input_dim), dtype=grad_tokens.dtype)

        for expert in self.shared_experts:
            grad_input = grad_input + expert.backward(grad_tokens)

        scale = 1.0 / self.top_k
        gate_grad = np.zeros((batch, self.num_routed_experts), dtype=grad_tokens.dtype)
        for e, expert in enumerate(self.routed_experts):
            chosen, expert_out = self.expert_samples[e], self.expert_outputs[e]
            if not chosen.size:
                expert.zero_gradients()
                continue
            upstream = grad_tokens[chosen]
            grad_input[chosen] += expert.backward(scale * upstream)
            gate_grad[chosen, e] = scale * np.sum(upstream * expert_out, axis=(1, 2))

        grad_pooled = self.gate.backward(gate_grad)
        grad_input = grad_input + self.pooling.backward(grad_pooled[:, None, :])
        return grad_input.reshape(self.in_shape)

    def __str__(self):
        return (
            f"SimpleMixtureOfExperts, {self.num_shared_experts} shared + top "
            f"{self.top_k} of {self.num_routed_experts} routed experts per sample, "
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
    cache_names = ("input", "weights", "counts")

    def __init__(self):
        super().__init__()
        self.declare_shapes(inputs=((None, None, None),), outputs=((None, 1, None),))

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

    def __str__(self):
        return "Layer of Sequence Mean Pooling (avg over sequence)"

    def __repr__(self):
        return self.__str__()
