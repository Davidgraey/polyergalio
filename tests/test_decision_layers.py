"""
DecisionHead: exact gradients, the decision loss, decoding and calibration,
and end-to-end learning on the generated decision task.
"""

import numpy as np
import pytest
from conftest import GRADIENT_TOLERANCE, numeric_gradient, relative_error
from polyergalio.generators import RandomDatasetGenerator
from polyergalio.generators.data_generators import (
    CLS_ID,
    MARK_ID,
    SEP_ID,
    build_decision_sequence,
)
from polyergalio.models.constants import DECISION_TYPES
from polyergalio.models.model_loss import DecisionLoss
from polyergalio.models.heads import DecisionHead
from polyergalio.models.heads.decision_utilities import (calibrated_probabilities,
    decision_correct,
    decode_decisions,
    fit_temperatures,
    masked_softmax)
from polyergalio.models.optimizers import Adam

BATCH, SEQUENCE, HIDDEN = 4, 9, 6
BINARY, CHOICE, SCORE = (kind.value for kind in DECISION_TYPES)
GRID = np.exp(np.linspace(np.log(0.05), np.log(10.0), 81))


@pytest.fixture
def encoded():
    return np.random.default_rng(7).normal(size=(BATCH, SEQUENCE, HIDDEN))


@pytest.fixture
def markers():
    marker_pos = np.array([[2, 3, 4, 5], [2, 4, 0, 0], [3, 5, 7, 0], [2, 3, 0, 0]])
    token_mask = marker_pos > 0
    return dict(
        marker_pos=marker_pos,
        token_mask=token_mask,
        decisiontypes=[DECISION_TYPES.CHOICE, DECISION_TYPES.BINARY, DECISION_TYPES.SCORE, DECISION_TYPES.BINARY],
    )


def build_head(**kwargs):
    return DecisionHead(hidden_dim=HIDDEN, head_hidden=8, activation_type="tanh", **kwargs).eval()


def featurize(X, meta):
    """
    Stand-in for a contextual encoder: each position sees its own token and
    the next one (so a [MARK] sees its option), plus a summary of the state.
    """
    onehot = np.eye(meta["vocab_size"])[X]
    onehot = onehot + np.roll(onehot, -1, axis=1)
    separators = X == meta["sep_id"]
    segment = np.cumsum(separators, axis=1)
    token = np.eye(meta["vocab_size"])[X]
    counts = (token * ((segment == 2) & ~separators)[..., None]).sum(axis=1)
    query = (token * ((segment == 0) & (X != meta["cls_id"]))[..., None]).sum(axis=1)
    context = np.concatenate([counts * query, (counts > 0) * 1.0], axis=-1)
    return np.concatenate([onehot, np.broadcast_to(context[:, None], X.shape + context.shape[-1:])], axis=-1)


# -------------    sequence layout    ------------------------------
def test_build_decision_sequence_layout():
    ids, positions = build_decision_sequence([10, 11], [[20], [21, 22]], [30, 31, 32])
    assert ids == [CLS_ID, 10, 11, SEP_ID, MARK_ID, 20, MARK_ID, 21, 22, SEP_ID, 30, 31, 32, SEP_ID]
    assert [ids[p] for p in positions] == [MARK_ID, MARK_ID]


# -------------    generator    ------------------------------------
@pytest.fixture
def decision_data():
    return RandomDatasetGenerator(random_seed=3).generate(
        "decision", num_samples=300, vocab_size=10, max_seq_length=8, verbose=False
    )


def test_generator_answers_match_the_state(decision_data):
    X, y, meta = decision_data
    for row, kind, answer in zip(X, meta["decisiontypes"], y):
        separators = np.flatnonzero(row == SEP_ID)
        header, state = row[1:separators[0]], row[separators[1] + 1:separators[2]]
        options = row[separators[0] + 1:separators[1]]
        if kind == CHOICE:
            present = [token in state for token in options[options != MARK_ID]]
            assert present.index(True) == answer and sum(present) == 1
        elif kind == SCORE:
            assert np.sum(state == header[1]) == answer
        else:
            assert int(header[1] in state) == answer


# -------------    head    -----------------------------------------
def test_input_gradient(encoded, markers):
    head = build_head()
    upstream = np.random.default_rng(11).normal(size=markers["marker_pos"].shape) * markers["token_mask"]
    head.forward(encoded, **markers)
    analytic = head.backward(upstream)
    numeric = numeric_gradient(lambda: float((head.forward(encoded, **markers) * upstream).sum()), encoded)
    assert relative_error(analytic, numeric) < GRADIENT_TOLERANCE


PARAMETER_OWNERS = {
    "embedding_norm": (lambda head: head.embedding_norm, "scale_gamma", "gradient_gamma"),
    "trunk_a_shared": (lambda head: head.trunk_a.shared_experts[0].down_proj, "weights", "gradient_weights"),
    "trunk_a_gate": (lambda head: head.trunk_a.gate.stack[0], "weights", "gradient_weights"),
    "trunk_mid_norm": (lambda head: head.trunk_mid_norm, "scale_gamma", "gradient_gamma"),
    "trunk_b_shared": (lambda head: head.trunk_b.shared_experts[0].gate_proj, "weights", "gradient_weights"),
    "trunk_b_gate": (lambda head: head.trunk_b.gate.stack[0], "weights", "gradient_weights"),
    "scorer": (lambda head: head.scorer, "weights", "gradient_weights"),
    "type_embedding": (lambda head: head, "type_embedding", "gradient_type_embedding"),
}


@pytest.mark.parametrize("owner", PARAMETER_OWNERS)
def test_parameter_gradients(encoded, markers, owner):
    head = build_head()
    find, parameter, gradient_name = PARAMETER_OWNERS[owner]
    layer = find(head)
    upstream = np.random.default_rng(11).normal(size=markers["marker_pos"].shape) * markers["token_mask"]
    head.forward(encoded, **markers)
    head.backward(upstream)
    numeric = numeric_gradient(
        lambda: float((head.forward(encoded, **markers) * upstream).sum()), getattr(layer, parameter)
    )
    assert relative_error(getattr(layer, gradient_name), numeric) < GRADIENT_TOLERANCE


def test_padding_markers_receive_no_gradient(encoded, markers):
    head = build_head()
    head.forward(encoded, **markers)
    grad = head.backward(np.ones(markers["marker_pos"].shape))
    touched = np.zeros(encoded.shape[:2], dtype=bool)
    rows = np.broadcast_to(np.arange(BATCH)[:, None], markers["marker_pos"].shape)
    touched[rows[markers["token_mask"]], markers["marker_pos"][markers["token_mask"]]] = True
    assert not grad[~touched].any()


def test_act_gradient_matches_numeric(encoded, markers):
    head = build_head(act_weight=0.7)
    correct = np.array([1.0, 0.0, 1.0, 0.0])

    def act_loss():
        head.forward(encoded, **markers)
        return 0.7 * head.score_act(correct)

    head.forward(encoded, **markers)
    head.score_act(correct)
    analytic = head.backward(np.zeros(markers["marker_pos"].shape))
    assert relative_error(analytic, numeric_gradient(act_loss, encoded)) < GRADIENT_TOLERANCE
    numeric_weights = numeric_gradient(act_loss, head.act_head.weights)
    assert relative_error(head.act_head.gradient_weights, numeric_weights) < GRADIENT_TOLERANCE


# -------------    loss    -----------------------------------------
@pytest.mark.parametrize("ordinal_weight", (0.0, 0.5))
def test_decision_loss_gradient(markers, ordinal_weight):
    logits = np.random.default_rng(5).normal(size=markers["marker_pos"].shape)
    answer = np.array([2, 1, 1, 0])
    loss = DecisionLoss(ordinal_weight=ordinal_weight)
    loss(logits, answer, markers["token_mask"], markers["decisiontypes"])
    numeric = numeric_gradient(lambda: loss(logits, answer, markers["token_mask"], markers["decisiontypes"]), logits)
    loss(logits, answer, markers["token_mask"], markers["decisiontypes"])
    assert relative_error(loss.backward(), numeric * markers["token_mask"]) < GRADIENT_TOLERANCE


# -------------    decoding and calibration    ---------------------
def test_decode_by_question_type():
    probabilities = np.array([[0.1, 0.7, 0.2, 0.0], [0.0, 0.5, 0.5, 0.0], [0.3, 0.7, 0.0, 0.0]])
    kinds = [DECISION_TYPES.CHOICE, DECISION_TYPES.SCORE, DECISION_TYPES.BINARY]
    decisions = decode_decisions(probabilities, kinds)
    assert np.allclose(decisions, [1, 1.5, 0.7])
    assert np.array_equal(decisions, decode_decisions(probabilities, [kind.value for kind in kinds]))
    assert np.array_equal(decision_correct(decisions, np.array([1, 2, 1]), kinds), [1, 1, 1])
    assert np.array_equal(decision_correct(decisions, np.array([0, 1, 0]), kinds), [0, 0, 0])


def test_fit_temperatures_recovers_a_planted_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=3.0, size=(4000, 4))
    mask = np.ones_like(logits, dtype=bool)
    kinds = np.full(len(logits), CHOICE)
    true_probabilities = masked_softmax(logits, mask, 2.0)
    answer = np.array([rng.choice(4, p=p) for p in true_probabilities])
    temperatures = fit_temperatures(logits, answer, mask, kinds, GRID)
    assert temperatures["choice:3-5"] == pytest.approx(2.0, rel=0.15)
    assert np.allclose(calibrated_probabilities(logits, mask, kinds, temperatures).sum(axis=1), 1)


# -------------    end to end    -----------------------------------
def test_learns_all_three_question_types():
    X, y, meta = RandomDatasetGenerator(random_seed=0).generate(
        "decision", num_samples=2000, vocab_size=10, max_seq_length=8, verbose=False
    )
    features = featurize(X, meta)
    train, held_out = slice(0, 1600), slice(1600, None)
    options = {key: meta[key] for key in ("marker_pos", "token_mask", "decisiontypes")}

    head = DecisionHead(hidden_dim=features.shape[-1], head_hidden=64)
    loss = DecisionLoss(ordinal_weight=0.1)
    optimizer = Adam(0.01)
    for _ in range(300):
        batch = {key: value[train] for key, value in options.items()}
        logits = head.forward(features[train], **batch)
        loss(logits, y[train], batch["token_mask"], batch["decisiontypes"])
        decisions = decode_decisions(masked_softmax(logits, batch["token_mask"]), batch["decisiontypes"])
        head.score_act(decision_correct(decisions, y[train], batch["decisiontypes"]))
        head.backward(loss.backward())
        optimizer.step([head])

    batch = {key: value[held_out] for key, value in options.items()}
    probabilities = masked_softmax(head.forward(features[held_out], **batch), batch["token_mask"])
    correct = decision_correct(decode_decisions(probabilities, batch["decisiontypes"]), y[held_out], batch["decisiontypes"])
    for kind in range(len(DECISION_TYPES)):
        assert correct[batch["decisiontypes"] == kind].mean() > 0.9
    assert head.act_probabilities[correct == 1].mean() > 0.8
