"""
System-one decision example.

Wires the minimal encoder-head setup from decision-head-overview.md --
TextEmbedding -> SpectreAttention -> DecisionHead -- into one NeuralNetwork
graph, then trains it on RandomDatasetGenerator's "decision" task. A single
forward pass answers every question (no chain-of-thought, no decoding loop),
which is the "system one" part: one Spectre mixing pass over the sequence,
then one shared scorer read off the [MARK] positions.

Run: python NNet_system_one_decision_example.py
"""

from polyergalio.generators.data_generators import RandomDatasetGenerator
from polyergalio.models.constants import DECISION_TYPES
from polyergalio.models.embedding.embedding import TextEmbedding
from polyergalio.models.layers.decision_layers import (
    DecisionHead,
    decision_correct,
    decode_decisions,
    masked_softmax,
)
from polyergalio.models.layers.spectre_layers import SpectreAttention
from polyergalio.models.model_loss import DecisionLoss
from polyergalio.models.neural_network import NeuralNetwork
from polyergalio.models.optimizers import SGD

HIDDEN_DIM = 32
HEAD_HIDDEN = 16
NUM_HEADS = 4
LEARNING_RATE = 0.05
TRAIN_STEPS = 200


def build_network(vocab_size: int, sequence_length: int, padding_idx: int) -> NeuralNetwork:
    """
    Encoder-head system-one network: TextEmbedding -> SpectreAttention -> DecisionHead.

    Parameters
    ----------
    vocab_size : token ids run in [0, vocab_size)
    sequence_length : fixed row width; SpectreAttention pins its FFT to it
    padding_idx : id whose embedding row is fixed at zero

    Returns
    -------
    NeuralNetwork mapping token ids (batch, sequence_length) to option logits
    """
    net = NeuralNetwork(name="system_one", input_shape=(sequence_length,))
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


def main():
    generator = RandomDatasetGenerator(random_seed=0)
    X, y, meta = generator.generate(
        "decision",
        num_samples=256,
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

    net = build_network(
        vocab_size=meta["vocab_size"],
        sequence_length=X.shape[1],
        padding_idx=meta["pad_id"],
    )

    net.eval()
    logits = net.forward(X, **kwargs)
    print(net.summary())
    print(f"accuracy before training: {accuracy(logits, y, kwargs):.3f}")

    loss_fn = DecisionLoss(ordinal_weight=0.25)
    optimizer = SGD(LEARNING_RATE)

    head = net.node("decision_head").layer

    net.train()
    for step in range(TRAIN_STEPS):
        net.zero_gradients()
        logits = net.forward(X, **kwargs)
        loss = loss_fn(logits, y, kwargs["token_mask"], kwargs["decisiontypes"])
        act_loss = head.score_act(row_correctness(logits, y, kwargs))
        net.backward(loss_fn.backward())
        optimizer.step(net.layers)
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss:.4f}  act loss {act_loss:.4f}")

    net.eval()
    logits = net.forward(X, **kwargs)
    print(f"accuracy after training: {accuracy(logits, y, kwargs):.3f}")
    escalated = head.escalate()
    print(f"escalated to system two: {escalated.mean():.3f} of rows")


if __name__ == "__main__":
    main()
