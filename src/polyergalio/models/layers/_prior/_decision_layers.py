from __future__ import annotations

from typing import Optional

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
from polyergalio.models.layers.mixture_layers import MixtureOfExperts
from polyergalio.models.weight_initialization import get_weight_init


# -------------    question types    -------------------------------
def decision_type_ids(decisiontypes) -> NDArray:
    """
    Numeric type ids for a batch of question types.

    Parameters
    ----------
    decisiontypes : DECISION_TYPES members, their integer values, or a mix

    Returns
    -------
    (batch,) int array of DECISION_TYPES values
    """
    return np.array([DECISION_TYPES(kind).value for kind in np.ravel(decisiontypes)], dtype=int)


# -------------    distributions over options    ------------------
def masked_softmax(
    logits: NDArray, token_mask: NDArray, temperature: float | NDArray = 1.0
) -> NDArray:
    """
    Softmax over each row's real options only.

    Parameters
    ----------
    logits : (batch, options)
    token_mask : (batch, options), 1 for a real option
    temperature : scalar, or (batch,) per-row

    Returns
    -------
    (batch, options) probabilities, exactly 0 on padded slots
    """
    temperature = np.reshape(temperature, (-1, 1))
    scaled = np.where(token_mask, logits / temperature, MASKED_LOGIT)
    exps = np.exp(scaled - scaled.max(axis=-1, keepdims=True)) * token_mask
    return exps / exps.sum(axis=-1, keepdims=True)


def decision_confidence(probabilities: NDArray, token_mask: NDArray) -> NDArray:
    """
    one minus the normalized entropy for each row

    Returns
    -------
    (batch,) in [0, 1]; 1 for a one-hot answer or a single-option row
    """
    num_options = token_mask.sum(axis=-1)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, EPSILON)), axis=-1
    )
    return np.where(num_options > 1, 1 - entropy / np.log(np.maximum(num_options, 2)), 1.0)


def decode_decisions(probabilities: NDArray, decisiontypes: list[DECISION_TYPES]) -> NDArray:
    """
    Read each row's answer according to its own question type.

    Returns
    -------
    (batch,) BINARY -> P(true) with options ordered [false, true],
    CHOICE -> chosen option index, SCORE -> expected level
    """
    type_ids = decision_type_ids(decisiontypes)
    levels = np.arange(probabilities.shape[-1])
    return np.select(
        [type_ids == DECISION_TYPES.CHOICE.value, type_ids == DECISION_TYPES.SCORE.value],
        [np.argmax(probabilities, axis=-1), probabilities @ levels],
        default=probabilities[:, min(1, probabilities.shape[-1] - 1)],
    )


def decision_correct(
    decisions: NDArray, answer: NDArray, decisiontype: list[DECISION_TYPES]
) -> NDArray:
    """
    Per-row correctness of decoded decisions against the answer option index.

    Returns
    -------
    (batch,) True where right; a score is right when its expected level rounds
    to the answer, a binary when P(true) falls on the answer's side of 0.5
    """
    binary_pick = (decisions >= 0.5).astype(int)
    picked = np.where(
        decision_type_ids(decisiontype) == DECISION_TYPES.BINARY.value,
        binary_pick,
        np.rint(decisions).astype(int),
    )
    return picked == answer


# -------------    calibration    ----------------------------------
def option_bucket(decisiontype: DECISION_TYPES | int, num_options: int) -> str:
    """Temperature key by question type and option count, e.g. "choice:3-5"."""
    if num_options <= 2:
        size = "2"
    elif num_options <= 5:
        size = "3-5"
    elif num_options <= 10:
        size = "6-10"
    else:
        size = "11+"
    return f"{DECISION_TYPES(decisiontype).name.lower()}:{size}"


def row_buckets(token_mask: NDArray, decisiontypes: list[DECISION_TYPES]) -> list[str]:
    """Temperature bucket name for every row."""
    return [
        option_bucket(int(t), int(k))
        for t, k in zip(decision_type_ids(decisiontypes), token_mask.sum(axis=-1))
    ]


def fit_temperatures(
    logits: NDArray,
    answer: NDArray,
    token_mask: NDArray,
    decisiontypes: list[DECISION_TYPES],
    grid: NDArray,
) -> dict[str, float]:
    """
    Post-hoc temperature per (type, option count) bucket, chosen by grid
    search to minimize held-out NLL

    Returns
    -------
    bucket name -> temperature
    """
    buckets = np.array(row_buckets(token_mask, decisiontypes))
    temperatures = {}
    for one_bucket in np.unique(buckets):
        rows = buckets == one_bucket
        answer_rows = answer[rows][:, None]
        losses = []
        for t in grid:
            _probs = masked_softmax(logits=logits[rows],
                                    token_mask=token_mask[rows],
                                    temperature=t)
            _per_row = np.take_along_axis(_probs, answer_rows, axis=-1)
            _nll = -np.mean(np.log(np.maximum(_per_row, EPSILON)))

            losses.append(_nll)

        temperatures[str(one_bucket)] = float(grid[int(np.argmin(losses))])
    return temperatures


def calibrated_probabilities(
    logits: NDArray,
    token_mask: NDArray,
    decisiontypes: list[DECISION_TYPES],
    temperatures: dict[str, float],
) -> NDArray:
    """Masked softmax with each row scaled by its bucket's temperature (1.0 if unfitted)."""

    row_temperature = np.array(
        [temperatures.get(b, 1.0) for b in row_buckets(token_mask, decisiontypes)]
    )

    return masked_softmax(logits, token_mask, row_temperature)


# -------------    the head    -------------------------------------
class DecisionHead(Layer):
    """
    system-one decision head for an encoder-only model, a-la Laya / Jev.

    every structured decision 'question' is one sequence laid out as
        [CLS] <type> <instructions> [SEP]
        [MARK] <option1> [MARK] <option2> ... [SEP] <state> [SEP]

    the head mean-pools the encoder's hidden state across each option's own
    span -- from its [MARK] to the next control token -- adds a learned
    embedding of the question type, and scores every option with one shared
    scorer. Pooling the whole span, not just the [MARK] position, keeps a
    long option's tokens from being squeezed through a single position.
    A softmax over the valid options answers all three types:
        CHOICE: argmax between the valid options
        SCORE: expected level over the ordered levels,
        BINARY: P(true) with options fixed as [false, true].

    a second output head reads [CLS] into two act logits [defer, act], trained to
    predict whether the answer will be right: the escalation signal

    Input shape: (batch, sequence, hidden_dim)
    Output shape: (batch, options) logits, MASKED_LOGIT on padded slots
    """
    parameter_names = ("type_embedding", "embedding_norm", "trunk_a", "trunk_norm", "scorer", "act_head")
    cache_names = (
        "hidden_shape", "marker_pos", "marker_end", "span_mask", "span_lengths",
        "token_mask", "decisiontype", "act_logits", "act_gradient", "output",
    )
    TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.05), np.log(10.0), 81))

    def __init__(self,
                 hidden_dim: int,
                 head_hidden: int,
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
        hidden_dim : width of the encoder's hidden state
        head_hidden : width of the option scorer's hidden layer
        num_types : number of question types, see DECISION_TYPES
        activation_type : activation for the trunk experts
        act_weight : scale on the act loss gradient
        type_initialization : WEIGHT_INIT_DISPATCHER name for the question-type embedding
        type_initialization_kwargs : keyword arguments bound to that initializer
        moe_shared_experts, moe_routed_experts, moe_top_k, moe_routed_scaling, moe_num_groups, moe_top_groups :
            see MixtureOfExperts
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_hidden = head_hidden
        self.num_types = num_types

        self.activation_type = activation_type
        self.act_weight = act_weight
        self.moe_shared_experts = moe_shared_experts
        self.moe_routed_experts = moe_routed_experts
        self.moe_top_k = moe_top_k
        self.moe_routed_scaling = moe_routed_scaling
        self.moe_num_groups = moe_num_groups
        self.moe_top_groups = moe_top_groups

        self.declare_shapes(inputs=((None, None, hidden_dim),), outputs=((None,),))

        self.type_initialization = type_initialization
        self.type_initialization_kwargs = dict(type_initialization_kwargs or {})
        type_initializer = get_weight_init(type_initialization, **self.type_initialization_kwargs)
        self.type_embedding = type_initializer(RNG, ni=num_types, no=hidden_dim)
        self.embedding_norm = NormalizeLayer(input_dimension=hidden_dim)

        self.trunk_a = MixtureOfExperts(input_dim=hidden_dim,
                                        upscale_dim=int(1.5 * hidden_dim),
                                        hidden_dim=hidden_dim,
                                        num_shared_experts=moe_shared_experts,
                                        num_routed_experts=moe_routed_experts,
                                        top_k=moe_top_k,
                                        routed_scaling=moe_routed_scaling,
                                        num_groups=moe_num_groups,
                                        top_groups=moe_top_groups,
                                        activation_type=activation_type)

        self.trunk_norm = NormalizeLayer(input_dimension=hidden_dim)
        self.scorer = FullyConnectedLayer(
            input_dimension=hidden_dim, output_dimension=1, activation_type="linear", is_output=True
        )
        self.act_head = FullyConnectedLayer(
            input_dimension=hidden_dim, output_dimension=2, activation_type="linear", is_output=True
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
        self.hidden_shape = hidden_state.shape
        self.marker_pos = np.asarray(marker_pos, dtype=int)
        self.marker_end = np.asarray(marker_end, dtype=int)
        self.token_mask = (np.ones(self.marker_pos.shape, dtype=bool)
                           if token_mask is None
                           else np.asarray(token_mask, dtype=bool)
                           )
        self.decisiontype = (np.full(batch, DECISION_TYPES.CHOICE.value)
                             if decisiontypes is None
                             else decision_type_ids(decisiontypes)
                             )

        positions = np.arange(sequence_length)
        self.span_mask = (
            (positions[None, None, :] >= self.marker_pos[..., None])
            & (positions[None, None, :] < self.marker_end[..., None])
            & self.token_mask[..., None]
        )
        self.span_lengths = np.maximum(self.span_mask.sum(axis=-1, keepdims=True), 1)
        span_weights = self.span_mask / self.span_lengths

        type_vector = self.type_embedding[self.decisiontype]
        markers = np.einsum("bot,bth->boh", span_weights, hidden_state)
        xs = self.trunk_a(self.embedding_norm(markers)+ type_vector[:, None, :], mask=self.token_mask)
        xs = self.trunk_norm(xs, mask=self.token_mask)

        scores = self.scorer(xs)[..., 0]

        self.act_logits = self.act_head(hidden_state[:, 0] + type_vector)
        self.act_gradient = None
        self.output = np.where(self.token_mask, scores, MASKED_LOGIT)
        return self.output

    @property
    def act_probabilities(self) -> NDArray:
        """(batch,) P(act): the head expects its own answer to be right."""
        exps = np.exp(self.act_logits - self.act_logits.max(axis=-1, keepdims=True))
        return exps[:, 1] / exps.sum(axis=-1)

    def score_act(self, correct: NDArray) -> float:
        """
        Cross-entropy of the act branch against per-row correctness. Caches its
        gradient so the next backward pass trains the act branch.

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

        span_weights = self.span_mask / self.span_lengths
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
            f"DecisionHead, marker scorer {self.hidden_dim} -> {self.head_hidden} -> 1 "
            f"over {self.num_types} question types, + act branch"
        )

    def __repr__(self):
        return self.__str__()
