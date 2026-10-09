"""
System-one decision example.

Wires the minimal encoder-head setup from decision-head-overview.md --
TextEmbedding -> SpectreAttention -> DecisionHead -- into one Network
graph, then trains it on RandomDatasetGenerator's "decision" task. A single
forward pass answers every question (no chain-of-thought, no decoding loop),
which is the "system one" part: one Spectre mixing pass over the sequence,
then one shared scorer read off the [MARK] positions.

Rows are split into train, calibration and test sets. The network trains on
the train rows (main loss plus the act branch), temperatures are fit on the
calibration rows, and the test rows are decoded with calibrated probabilities,
scored for confidence, and escalated by the act branch.

Run: python NNet_system_one_decision_example.py
"""

import numpy as np

from polyergalio.generators.data_generators import RandomDatasetGenerator
from polyergalio.models.constants import DECISION_TYPES, EPSILON
from polyergalio.models.embedding.embedding import TextEmbedding
from polyergalio.models.heads import (
    DecisionHead,
    calibrated_probabilities,
    decision_confidence,
    decision_correct,
    decode_decisions,
    fit_temperatures,
    masked_softmax,
)
from polyergalio.models.layers.spectre_layers import SpectreAttention
from polyergalio.models.model_loss import DecisionLoss
from polyergalio.models.network import Network
from polyergalio.models.optimizers import SGD

HIDDEN_DIM = 32
HEAD_HIDDEN = 16
NUM_HEADS = 4
LEARNING_RATE = 0.05
TRAIN_STEPS = 200
NUM_SAMPLES = 768
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
ESCALATION_THRESHOLD = 0.5


def build_network(vocab_size: int, sequence_length: int, padding_idx: int) -> Network:
    """
    Encoder-head system-one network: TextEmbedding -> SpectreAttention -> DecisionHead.

    Parameters
    ----------
    vocab_size : token ids run in [0, vocab_size)
    sequence_length : fixed row width; SpectreAttention pins its FFT to it
    padding_idx : id whose embedding row is fixed at zero

    Returns
    -------
    Network mapping token ids (batch, sequence_length) to option logits
    """
    net = Network(name="system_one", input_shape=(sequence_length,))
    embedded = net.connect(
        TextEmbedding(vocab_size, HIDDEN_DIM, padding_idx=padding_idx),
        net.input,
        name="embedding",
    )
    attended = net.connect(
        SpectreAttention(sequence_length, HIDDEN_DIM, num_heads=NUM_HEADS),
        embedded,
        name="attention",
    )
    net.output = net.connect(
        DecisionHead(HIDDEN_DIM, HEAD_HIDDEN), attended, name="decision_head"
    )
    return net


def row_correctness(logits, y, kwargs: dict):
    """(batch,) 1.0 where the decoded answer matches the generator's answer."""
    probabilities = masked_softmax(logits, kwargs["token_mask"])
    decisions = decode_decisions(probabilities, kwargs["decisiontypes"])
    return decision_correct(decisions, y, kwargs["decisiontypes"])


def accuracy(logits, y, kwargs: dict) -> float:
    """Fraction of rows decoded correctly against the generator's answers."""
    return float(row_correctness(logits, y, kwargs).mean())


def split_rows(x, y, kwargs: dict, fractions=SPLIT_FRACTIONS) -> list[tuple]:
    """
    Split rows in order into train, calibration and test sets.

    Returns
    -------
    one (x, y, kwargs) tuple per fraction
    """
    edges = np.cumsum([int(len(x) * fraction) for fraction in fractions[:-1]])
    return [
        (x[rows], y[rows], {name: value[rows] for name, value in kwargs.items()})
        for rows in np.split(np.arange(len(x)), edges)
    ]


def fit_calibration(net: Network, x, y, kwargs: dict) -> dict[str, float]:
    """Temperature per (type, option count) bucket, fit on rows the model did not train on."""
    net.eval()
    head = net.node("decision_head").layer
    logits = net.forward(x, **kwargs)
    return fit_temperatures(logits, y, kwargs["token_mask"], kwargs["decisiontypes"], head.TEMPERATURE_GRID)


def answer_nll(probabilities, y) -> float:
    picked = np.take_along_axis(probabilities, y[:, None], axis=-1)[:, 0]
    return float(-np.mean(np.log(np.maximum(picked, EPSILON))))


def rate(values, rows) -> float:
    return float(values[rows].mean()) if rows.any() else float("nan")


def evaluate(net: Network, x, y, kwargs: dict, temperatures: dict, threshold: float = ESCALATION_THRESHOLD) -> dict:
    """
    Decode, score and escalate every row with the head's three outputs.

    Returns
    -------
    per-row arrays (decisions, correct, confidence, act, escalated) and the
    answer NLL before and after calibration
    """
    net.eval()
    head = net.node("decision_head").layer
    token_mask, decisiontypes = kwargs["token_mask"], kwargs["decisiontypes"]
    logits = net.forward(x, **kwargs)
    raw = masked_softmax(logits, token_mask)
    probabilities = calibrated_probabilities(logits, token_mask, decisiontypes, temperatures)
    decisions = decode_decisions(probabilities, decisiontypes)
    return dict(
        decisions=decisions,
        correct=decision_correct(decisions, y, decisiontypes),
        confidence=decision_confidence(probabilities, token_mask),
        act=head.act_probabilities,
        escalated=head.escalate(threshold),
        nll_raw=answer_nll(raw, y),
        nll_calibrated=answer_nll(probabilities, y),
    )


def print_evaluation(results: dict, temperatures: dict) -> None:
    """Print temperatures, calibration effect, confidence and the escalation split."""
    escalated, correct = results["escalated"], results["correct"]
    print("temperatures: " + ", ".join(f"{bucket}={value:.2f}" for bucket, value in temperatures.items()))
    print(f"NLL raw {results['nll_raw']:.3f} -> calibrated {results['nll_calibrated']:.3f}")
    print(f"accuracy {correct.mean():.3f}  mean confidence {results['confidence'].mean():.3f}  mean P(act) {results['act'].mean():.3f}")
    print(f"escalated to system two: {escalated.mean():.3f} of rows")
    print(f"accuracy kept {rate(correct, ~escalated):.3f}  escalated {rate(correct, escalated):.3f}")


def main():
    generator = RandomDatasetGenerator(random_seed=0)
    X, y, meta = generator.generate(
        "decision",
        num_samples=NUM_SAMPLES,
        num_choices=4,
        num_levels=4,
        decision_types=tuple(DECISION_TYPES),
        verbose=False,
    )
    kwargs = dict(
        mask=meta["attention_mask"],
        marker_pos=meta["marker_pos"],
        marker_end=meta["marker_end"],
        token_mask=meta["token_mask"],
        decisiontypes=meta["decisiontypes"],
    )
    train_set, calibration_set, test_set = split_rows(X, y, kwargs)
    x_train, y_train, train_kwargs = train_set
    x_test, y_test, test_kwargs = test_set

    net = build_network(
        vocab_size=meta["vocab_size"],
        sequence_length=X.shape[1],
        padding_idx=meta["pad_id"],
    )

    net.eval()
    print(net.summary())
    print(f"test accuracy before training: {accuracy(net.forward(x_test, **test_kwargs), y_test, test_kwargs):.3f}")

    loss_fn = DecisionLoss(ordinal_weight=0.25)
    optimizer = SGD(LEARNING_RATE)
    head = net.node("decision_head").layer

    net.train()
    for step in range(TRAIN_STEPS):
        net.zero_gradients()
        logits = net.forward(x_train, **train_kwargs)
        loss = loss_fn(logits, y_train, train_kwargs["token_mask"], train_kwargs["decisiontypes"])
        act_loss = head.score_act(row_correctness(logits, y_train, train_kwargs))
        net.backward(loss_fn.backward())
        optimizer.step(net)
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss:.4f}  act loss {act_loss:.4f}")

    temperatures = fit_calibration(net, *calibration_set)
    print("\n--- test rows ---")
    print_evaluation(evaluate(net, x_test, y_test, test_kwargs, temperatures), temperatures)


if __name__ == "__main__":
    main()
