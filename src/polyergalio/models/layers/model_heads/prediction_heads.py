"""
Slightly more abstract prediction layers as structural units - a prediction "head"

"""
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.constants import (
    DECISION_TYPES,
    EPSILON,
    MASKED_LOGIT,
)
from polyergalio.models.layers.basic_layers import (
    RNG,
    FullyConnectedLayer,
    Layer,
    NormalizeLayer,
)
from polyergalio.models.layers.model_heads.decision_utilities import decision_type_ids
from polyergalio.models.layers.mixture_layers import MixtureOfExperts
from polyergalio.models.weight_initialization import get_weight_init


class LowRankBottleNeck(Layer):
    """
    reduction in parameters as a linear bottleneck for high-dimensional output - like Masked Language Modeling task
        with high vocab
    """
    parameter_names = ("fc_down_projection", "fc_up_projection")

    def __init__(self,
                 input_dim: int,
                 low_rank_dim: int,
                 output_dim: int,
                 activation_type: str = "linear",
                 initialization_override: Optional[str] = None,
                 initialization_kwargs: Optional[dict] = None,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.low_rank_dim = low_rank_dim
        self.output_dim = output_dim

        self.activation_type = activation_type
        self.initialization_override = initialization_override
        self.initialization_kwargs = dict(initialization_kwargs or {})

        self.fc_down_projection = FullyConnectedLayer(input_dimension=input_dim,
                                                      output_dimension=low_rank_dim,
                                                      activation_type=activation_type,
                                                      initialization_override=initialization_override,
                                                      initialization_kwargs=initialization_kwargs
                                                      )
        self.fc_up_projection = FullyConnectedLayer(input_dimension=low_rank_dim,
                                                    output_dimension=output_dim,
                                                    activation_type="linear",
                                                    initialization_override=initialization_override,
                                                    initialization_kwargs=initialization_kwargs,
                                                    is_output=True
                                                    )


        self.declare_shapes(inputs=((input_dim,),), outputs=((output_dim,),))
        self.zero_gradients()

    def forward(self,
                incoming_x: NDArray,
                mask: Optional[NDArray] = None,
                ) -> NDArray:

            """
            Parameters
            ----------
            incoming_x : (..., ni) input, already standardized if called for
            mask : unused; every row is projected independently, num_kept for pass-through compatibility with the graph

            Returns
            -------
            (..., no) activated projection of the input
            """
            # downcast, upcast -->
            return self.fc_up_projection.forward(self.fc_down_projection.forward(incoming_x))

    def backward(self, incoming_gradient: NDArray) -> NDArray:

        return self.fc_down_projection.backward(self.fc_up_projection.backward(incoming_gradient))

    def __repr__(self):
        return f"Bottleneck Layer, shaped {self.fc_down_projection.shape} -> {self.fc_up_projection.shape}"



# ------- DECISION HEAD for System-One -- Encoder-> flexible prediction head - marker boudned tasks support -------
class DecisionHead(Layer):
    """
    system-one decision head for an encoder-only model, a-la Laya / Jev.

    every structured decision 'question' is one sequence laid out as
        [CLS] <type> <instructions> [SEP]
        [MARK] <option1> [MARK] <option2> ... [SEP] <state> [SEP]

    the head mean-pools the encoder's hidden state across each option's own span
        from its [MARK] to the next control token

    adds a learned embedding of the question / task type scores every option with one shared scorer

    Pooling the whole span, not just the [MARK] position, keeps a long option's tokens from being squeezed through a
        single position.

    A softmax over the valid options answers all three types:
        CHOICE: argmax between the valid options
        SCORE: expected level over the ordered levels,
        BINARY: P(true) with options fixed as [false, true].

    a second output head reads [CLS] into two act logits [defer, act], trained to predict whether the answer will be
        correct: this is the 'escalation' signal

    Input shape: (batch, sequence, hidden_dim)
    Output shape: (batch, options) logits, MASKED_LOGIT on padded slots
    """
    parameter_names = ("type_embedding", "embedding_norm", "trunk_a", "trunk_norm", "scorer", "act_head")
    cache_names = ("span_mask", "span_lengths", "token_mask", "decisiontype", "act_logits", "act_gradient")
    TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.05), np.log(10.0), 81))

    def __init__(self,
                 input_dimension: int,
                 hidden_dimension: int,
                 num_types: int = len(DECISION_TYPES),
                 activation_type: str = "swish",
                 act_weight: float = 1.0,
                 type_initialization: str = "truncated_normal",
                 type_initialization_kwargs: Optional[dict] = None,
                 moe_shared_experts: int = 2,
                 moe_routed_experts: int = 4,
                 moe_top_k: int = 2,
                 moe_routed_scaling: float = 1.0,
                 moe_num_groups: Optional[int] = None,
                 moe_top_groups: Optional[int] = None,
                 ):
        """
        Parameters
        ----------
        input_dimension : width of the encoder's hidden state
        hidden_dimension : width of the option scorer's hidden layer
        num_types : number of question types, see DECISION_TYPES
        activation_type : activation for the trunk experts
        act_weight : scale on the acttion loss gradient -- start small and grow over time
        type_initialization, type_initialization_kwargs : keyword arguments bound to initializer
        moe_shared_experts, moe_routed_experts, moe_top_k, moe_routed_scaling, moe_num_groups, moe_top_groups :
            see MixtureOfExperts
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.num_types = num_types

        self.activation_type = activation_type
        self.act_weight = act_weight
        self.moe_shared_experts = moe_shared_experts
        self.moe_routed_experts = moe_routed_experts
        self.moe_top_k = moe_top_k
        self.moe_routed_scaling = moe_routed_scaling
        self.moe_num_groups = moe_num_groups
        self.moe_top_groups = moe_top_groups

        self.declare_shapes(inputs=((None, None, input_dimension),), outputs=((None,),))

        self.type_initialization = type_initialization
        self.type_initialization_kwargs = dict(type_initialization_kwargs or {})
        type_initializer = get_weight_init(type_initialization, **self.type_initialization_kwargs)
        self.type_embedding = type_initializer(RNG, ni=num_types, no=input_dimension)
        self.embedding_norm = NormalizeLayer(input_dimension=input_dimension)

        self.trunk_a = MixtureOfExperts(input_dim=input_dimension,
                                        upscale_dim=int(1.5 * input_dimension),
                                        hidden_dim=input_dimension,
                                        num_shared_experts=moe_shared_experts,
                                        num_routed_experts=moe_routed_experts,
                                        top_k=moe_top_k,
                                        routed_scaling=moe_routed_scaling,
                                        num_groups=moe_num_groups,
                                        top_groups=moe_top_groups,
                                        activation_type=activation_type)

        self.trunk_norm = NormalizeLayer(input_dimension=input_dimension)
        self.scorer = FullyConnectedLayer(
            input_dimension=input_dimension, output_dimension=1, activation_type="linear", is_output=True
        )
        self.act_head = FullyConnectedLayer(
            input_dimension=input_dimension, output_dimension=2, activation_type="linear", is_output=True
        )

        self.zero_gradients()

    def infer_output_shapes(self, input_shapes: tuple[tuple, ...]) -> tuple[tuple, ...]:
        return ((None,),)

    def forward(
        self,
        hidden_state: NDArray,
        marker_pos: Optional[NDArray] = None,
        marker_end: Optional[NDArray] = None,
        token_mask: Optional[NDArray] = None,
        decisiontypes: Optional[list[DECISION_TYPES]] = None,
    ) -> NDArray:
        """
        Parameters
        ----------
        hidden_state : (batch, sequence, hidden_dim) encoder output, [CLS] at position 0
        marker_pos : (batch, options) position of each option's [MARK]
        marker_end : (batch, options) exclusive end of each option's span (its
            next control token's position, or the trailing [SEP]'s)
        token_mask : (batch, options), 1 for a real option; all real if None
        decisiontypes : (batch,) DECISION_TYPES members or their values; all CHOICE if None

        Returns
        -------
        (batch, options) option logits; act logits are left on self.act_logits
        """
        if marker_pos is None:
            raise ValueError(
                "DecisionHead needs marker_pos, the position of each option's [MARK]"
            )
        if marker_end is None:
            raise ValueError(
                "DecisionHead needs marker_end, the exclusive end of each option's span"
            )
        batch, sequence_length = hidden_state.shape[:2]
        marker_pos = np.asarray(marker_pos, dtype=int)
        marker_end = np.asarray(marker_end, dtype=int)
        token_mask = (np.ones(marker_pos.shape, dtype=bool)
                      if token_mask is None
                      else np.asarray(token_mask, dtype=bool)
                      )
        decisiontype = (np.full(batch, DECISION_TYPES.CHOICE.value)
                        if decisiontypes is None
                        else decision_type_ids(decisiontypes)
                        )

        positions = np.arange(sequence_length)
        span_mask = (
            (positions[None, None, :] >= marker_pos[..., None])
            & (positions[None, None, :] < marker_end[..., None])
            & token_mask[..., None]
        )
        span_lengths = np.maximum(span_mask.sum(axis=-1, keepdims=True), 1)
        # keep recast
        span_weights = (span_mask / span_lengths).astype(hidden_state.dtype)

        type_vector = self.type_embedding[decisiontype]
        markers = np.einsum("bot,bth->boh", span_weights, hidden_state)
        xs = self.trunk_a(self.embedding_norm(markers)+ type_vector[:, None, :], mask=token_mask)
        xs = self.trunk_norm(xs, mask=token_mask)

        scores = self.scorer(xs)[..., 0]

        self.act_logits = self.act_head(hidden_state[:, 0] + type_vector)
        self.act_gradient = None
        if self.training:
            self.span_mask, self.span_lengths = span_mask, span_lengths
            self.token_mask, self.decisiontype = token_mask, decisiontype
        return np.where(token_mask, scores, MASKED_LOGIT)

    @property
    def act_probabilities(self) -> NDArray:
        """(batch,) P(act): the head expects its own answer to be right."""
        exps = np.exp(self.act_logits - self.act_logits.max(axis=-1, keepdims=True))
        return exps[:, 1] / exps.sum(axis=-1)

    def score_act(self, correct: NDArray) -> float:
        """
        Cross-entropy of the act branch against per-row correctness. Caches its gradient so the next backward pass
            can train the act branch.

        Parameters
        ----------
        correct : (batch,) 1.0 where the answer was right; soft targets allowed

        Returns
        -------
        mean act loss
        """
        correct = np.asarray(correct).reshape(-1)
        act = np.clip(self.act_probabilities, EPSILON, 1 - EPSILON)
        loss = -np.mean(correct * np.log(act) + (1 - correct) * np.log(1 - act))
        delta = (act - correct) / correct.size
        self.act_gradient = self.act_weight * np.stack([-delta, delta], axis=-1)
        return float(loss)

    def escalate(self, threshold: float = 0.5) -> NDArray:
        """(batch,) true where prob(act) falls below threshold -- signaling the decision is NOT safe to act on """
        return self.act_probabilities < threshold

    def backward(self, incoming_grad: NDArray) -> NDArray:
        grad_scores = np.where(self.token_mask, incoming_grad, 0.0)[..., None]
        grad_trunk = self.trunk_a.backward(self.trunk_norm.backward(self.scorer.backward(grad_scores)))
        grad_typed = grad_trunk * self.token_mask[..., None]
        grad_markers = self.embedding_norm.backward(grad_typed) * self.token_mask[..., None]

        # keep recast
        span_weights = (self.span_mask / self.span_lengths).astype(grad_markers.dtype)
        grad_hidden = np.einsum("bot,boh->bth", span_weights, grad_markers)
        grad_type = np.sum(grad_typed, axis=1)

        if self.act_gradient is None:
            self.act_head.zero_gradients()
        else:
            grad_cls = self.act_head.backward(self.act_gradient)
            grad_hidden[:, 0] += grad_cls
            grad_type = grad_type + grad_cls
            self.act_gradient = None

        self.gradient_type_embedding = np.zeros_like(self.type_embedding)
        np.add.at(self.gradient_type_embedding, self.decisiontype, grad_type)
        return grad_hidden

    def __str__(self):
        return (
            f"DecisionHead, marker scorer {self.input_dimension} -> {self.hidden_dimension} -> 1 "
            f"over {self.num_types} question types, + act branch"
        )

    def __repr__(self):
        return self.__str__()


class MixtureDensityHead(Layer):
    """
    maps input to the parameters of J gaussian distributions over a latent-space of size Z.

    Controls added to try to keep bounds on the distributions with applied limits

    Output is packed along the last axis as [log_pi (J), mu (J*Z), sigma (J*Z)]:
        log_pi : log-softmax of logits soft-clipped to +-logit_limit, so each weight stays within
            exp(2 * logit_limit) of any other and no component dies or dominates outright
        mu : soft-clipped to +-mu_limit
        sigma : log-uniform between sigma_min and sigma_max (a sigmoid in log space)

    Parameters
    ----------
    input_dim : size of the incoming hidden state
    latent_dim : Z
    components : J
    logit_limit : bound on the mixture logits
    mu_limit : bound on every mean
    sigma_min, sigma_max : bounds on every standard deviation
    initial_sigma : sigma at initialization, within the bounds
    init_std : standard deviation of the projection's initial weights
    """
    parameter_names = ("projection",)
    cache_names = ("log_pi", "slope_logit", "slope_mu", "sigma", "sigma_gate")

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 components: int = 5,
                 logit_limit: float = 4.0,
                 mu_limit: float = 5.0,
                 sigma_min: float = 0.05,
                 sigma_max: float = 2.0,
                 initial_sigma: float = 0.5,
                 init_std: float = 0.01,
                 ):
        super().__init__()
        if not 0 < sigma_min <= initial_sigma <= sigma_max:
            raise ValueError("Need 0 < sigma_min <= initial_sigma <= sigma_max")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.components = components
        self.logit_limit = logit_limit
        self.mu_limit = mu_limit
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.initial_sigma = initial_sigma
        self.init_std = init_std
        self.log_ratio = float(np.log(sigma_max / sigma_min))

        self.projection = FullyConnectedLayer(
            input_dimension=input_dim,
            output_dimension=self.output_dim,
            activation_type="linear",
            initialization_override="truncated_normal",
            initialization_kwargs={"std": init_std},
            is_output=True,
        )

        start = np.log(initial_sigma / sigma_min) / self.log_ratio
        self.projection.bias[0, components + components * latent_dim:] = np.log(start / (1.0 - start)) if 0 < start < 1 else 0.0

        self.declare_shapes(inputs=((input_dim,),), outputs=((self.output_dim,),))
        self.zero_gradients()

    @staticmethod
    def soft_clip(values: NDArray, limit: float) -> Tuple[NDArray, NDArray]:
        """limit * tanh(values / limit): near identity around 0, bounded by +-limit. Returns it and its slope."""
        squashed = np.tanh(values / limit)
        return limit * squashed, 1.0 - squashed ** 2

    @property
    def output_dim(self) -> int:
        return self.components * (1 + 2 * self.latent_dim)

    def unpack(self, packed: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Parameters
        ----------
        packed : (..., J + 2*J*Z) output of forward

        Returns
        -------
        log_pi (..., J), mu (..., J, Z), sigma (..., J, Z)
        """
        j, z = self.components, self.latent_dim
        full_shape = packed.shape[:-1]
        return (packed[..., :j],
                packed[..., j:j + j * z].reshape(*full_shape, j, z),
                packed[..., j + j * z:].reshape(*full_shape, j, z))

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        # TODO: check our loss functions and all forward / backwards - currently we have different packing vs unpacking
        calling forward then unpack
        pred = self.forward(x)
        pred_pi, pred_mu, pred_sigma = self.unpack(pred)

        Parameters
        ----------
        incoming_x : (..., input_dim)
        mask : unused; every position is projected independently, num_kept for pass-through compatibility with the graph

        Returns
        -------
        Output is packed along the last axis as [log_pi (J), mu (J*Z), sigma (J*Z)]:
        self.unpack() gives back these correctly sliced

            log_pi : log-softmax of logits soft-clipped to +-logit_limit, so each weight stays within
                exp(2 * logit_limit) of any other and no component dies or dominates outright
            mu : soft-clipped to +-mu_limit
            sigma : log-uniform between sigma_min and sigma_max (a sigmoid in log space)
        """
        j, z = self.components, self.latent_dim
        raw = self.projection.forward(incoming_x)

        logits, self.slope_logit = self.soft_clip(raw[..., :j], self.logit_limit)
        mu, self.slope_mu = self.soft_clip(raw[..., j:j + j * z], self.mu_limit)
        # hard clipped between -50 || +50
        self.sigma_gate = 1.0 / (1.0 + np.exp(-np.clip(raw[..., j + j * z:], -50.0, 50.0)))
        self.sigma = self.sigma_min * np.exp(self.log_ratio * self.sigma_gate)

        peak = logits.max(axis=-1, keepdims=True)
        self.log_pi = logits - peak - np.log(np.exp(logits - peak).sum(axis=-1, keepdims=True))
        return np.concatenate([self.log_pi, mu, self.sigma], axis=-1)

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        """
        Parameters
        ----------
        incoming_gradient : (..., J + 2*J*Z) gradient with respect to the packed output

        Returns
        -------
        (..., input_dim) gradient with respect to the input
        """
        j, z = self.components, self.latent_dim
        grad_log_pi = incoming_gradient[..., :j]
        grad_mu = incoming_gradient[..., j:j + j * z]
        grad_sigma = incoming_gradient[..., j + j * z:]

        grad_logits = grad_log_pi - np.exp(self.log_pi) * grad_log_pi.sum(axis=-1, keepdims=True)
        raw_gradient = np.concatenate([
            grad_logits * self.slope_logit,
            grad_mu * self.slope_mu,
            grad_sigma * self.sigma * self.log_ratio * self.sigma_gate * (1.0 - self.sigma_gate),
        ], axis=-1)
        return self.projection.backward(raw_gradient)

    def sample(self, packed: NDArray, temperature: float = 0.0, rng: Optional[np.random.Generator] = None) -> NDArray:
        """
        Draw one latent per position from the predicted mixture.

        Temperature tau reshapes the mixture before drawing: weights go as pi^(1/tau) and every sigma is scaled
        by sqrt(tau). tau = 1 is the model's own density, below 1 concentrates on likely modes, above 1 spreads.
        tau = 0 is deterministic: the mean of the highest-weight component.

        Parameters
        ----------
        packed : (..., J + 2*J*Z) output of forward
        temperature : tau >= 0
        rng : source of randomness, required when temperature > 0

        Returns
        -------
        (..., Z) float64
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        log_pi, mu, sigma = self.unpack(packed)

        if temperature == 0:
            best = log_pi.argmax(axis=-1)[..., None, None]
            return np.take_along_axis(mu, best, axis=-2)[..., 0, :]

        if rng is None:
            raise ValueError("rng is required when temperature > 0")
        chosen = (log_pi / temperature + rng.gumbel(size=log_pi.shape)).argmax(axis=-1)[..., None, None]
        center = np.take_along_axis(mu, chosen, axis=-2)[..., 0, :]
        spread = np.take_along_axis(sigma, chosen, axis=-2)[..., 0, :]
        return center + spread * np.sqrt(temperature) * rng.standard_normal(center.shape)

    def __str__(self):
        return (f"MixtureDensityHead, {self.input_dim} -> {self.components} Gaussians over {self.latent_dim}, "
                f"mu +-{self.mu_limit:g}, sigma [{self.sigma_min:g}, {self.sigma_max:g}]")

    def __repr__(self):
        return self.__str__()
