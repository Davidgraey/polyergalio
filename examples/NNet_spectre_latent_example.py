"""
Standard Spectre, latent Spectre and latent softmax attention on the same task.

SpectreAttention computes its per-head spectral gate with a dense head and mixes
values through a dense projection; it is compared at several Toeplitz band
radii, from half the sequence length down to none. SpectreLatentAttention
replaces the dense head and projection with LatentHead layers at several ranks.
LatentAttention is position-free softmax attention whose queries, keys and
values are rebuilt from low-rank latents. All layers are built at the same
width, their parameters are counted, and they are trained on identical data for
one of four tasks: shift each sequence right by one position, smooth over a
three-token window, replace every token with the sequence mean, or replace it
with a similarity-weighted average of all tokens. The last two need no position
information, which is all LatentAttention has.

Run: python NNet_spectre_latent_example.py
"""

import numpy as np
from polyergalio.models.layers.latent_attention import LatentAttention
from polyergalio.models.layers.spectre_latent_layers import SpectreLatentAttention
from polyergalio.models.layers.spectre_layers import SpectreAttention
from polyergalio.models.optimizers import Adam

SEQUENCE_LENGTH = 32
HIDDEN_DIM = 16
NUM_HEADS = 2
TRAIN_SAMPLES = 128
TEST_SAMPLES = 128
STEPS = 300
LEARNING_RATE = 3e-3
LATENT_RANKS = (4, 8, 16)
TASKS = ("shift", "smooth", "mean", "similar")


def make_data(task: str, samples: int, sequence_length: int, hidden: int, seed: int):
    """Random sequences and their targets: "shift", "smooth", "mean" or "similar"."""
    x = np.random.default_rng(seed).normal(size=(samples, sequence_length, hidden))
    if task == "shift":
        y = np.zeros_like(x)
        y[:, 1:] = x[:, :-1]
    elif task == "smooth":
        padded = np.pad(x, ((0, 0), (1, 1), (0, 0)))
        y = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3
    elif task == "mean":
        y = np.broadcast_to(x.mean(axis=1, keepdims=True), x.shape).copy()
    elif task == "similar":
        scores = np.einsum("btd,bsd->bts", x, x) / np.sqrt(hidden)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        y = np.einsum("bts,bsd->btd", weights / weights.sum(axis=-1, keepdims=True), x)
    else:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    return x, y


def band_radii(sequence_length: int) -> tuple:
    """Band radii halving from half the sequence length down to 1, then 0."""
    radii, radius = [], sequence_length // 2
    while radius >= 1:
        radii.append(radius)
        radius //= 2
    return tuple(radii) + (0,)


def build_models(
    sequence_length: int, hidden: int, heads: int, ranks=LATENT_RANKS, radii=None, use_wrm: bool = False
) -> dict:
    """Spectre per band radius, LatentSpectre per rank, and LatentAttention per rank."""
    radii = band_radii(sequence_length) if radii is None else radii
    models = {
        f"Spectre band={radius}": SpectreAttention(sequence_length, hidden, num_heads=heads, band_radius=radius, use_wrm=use_wrm)
        for radius in radii
    }
    for rank in ranks:
        models[f"LatentSpectre r={rank}"] = SpectreLatentAttention(
            sequence_length, hidden, num_heads=heads, query_rank=rank, value_rank=hidden, use_wrm=use_wrm
        )
    for rank in ranks:
        models[f"LatentAttention r={rank}"] = LatentAttention(hidden, heads, kv_rank=rank)
    return models


def parameter_counts(models: dict) -> dict:
    """Parameter count per model."""
    return {name: int(layer.num_parameters) for name, layer in models.items()}


def mse(layer, x, y) -> float:
    """Mean squared error of the layer's forward pass."""
    return float(((layer.forward(x) - y) ** 2).mean())


def train(layer, train_set, steps: int = STEPS, learning_rate: float = LEARNING_RATE) -> list:
    """Adam on mean squared error; returns the loss per step."""
    x, y = train_set
    optimizer = Adam(learning_rate)
    losses = []
    layer.train()
    for _ in range(steps):
        layer.zero_gradients()
        difference = layer.forward(x) - y
        losses.append(float((difference**2).mean()))
        layer.backward(2 * difference / difference.size)
        optimizer.step(layer)
    layer.eval()
    return losses


def compare(models: dict, train_set, test_set, steps: int = STEPS, learning_rate: float = LEARNING_RATE):
    """Train every model on the same data; returns (results table rows, losses, outputs on the test set)."""
    counts = parameter_counts(models)
    rows, curves, outputs = [], {}, {}
    for name, layer in models.items():
        curves[name] = train(layer, train_set, steps, learning_rate)
        outputs[name] = layer.forward(test_set[0])
        rows.append(
            {
                "model": name,
                "parameters": counts[name],
                "train mse": curves[name][-1],
                "test mse": mse(layer, *test_set),
            }
        )
    return rows, curves, outputs


def output_agreement(outputs: dict, reference: str) -> dict:
    """Correlation between each model's test outputs and the reference model's."""
    base = outputs[reference].ravel()
    return {name: float(np.corrcoef(base, out.ravel())[0, 1]) for name, out in outputs.items() if name != reference}


def main() -> None:
    models = build_models(SEQUENCE_LENGTH, HIDDEN_DIM, NUM_HEADS)
    reference = "Spectre band=0"
    print(f"sequence {SEQUENCE_LENGTH}, hidden {HIDDEN_DIM}, heads {NUM_HEADS}\n")
    print("--- initialised ---")
    for name, count in parameter_counts(models).items():
        print(f"{name:22s} {count:6d} parameters")

    for task in TASKS:
        train_set = make_data(task, TRAIN_SAMPLES, SEQUENCE_LENGTH, HIDDEN_DIM, seed=0)
        test_set = make_data(task, TEST_SAMPLES, SEQUENCE_LENGTH, HIDDEN_DIM, seed=1)
        models = build_models(SEQUENCE_LENGTH, HIDDEN_DIM, NUM_HEADS)
        rows, _, outputs = compare(models, train_set, test_set)
        agreement = output_agreement(outputs, reference)
        print(f"\n--- task: {task} (correlation is with {reference}) ---")
        for row in rows:
            corr = agreement.get(row["model"])
            tail = "" if corr is None else f", correlation {corr:.3f}"
            print(f"{row['model']:22s} train {row['train mse']:.4f}, test {row['test mse']:.4f}{tail}")


if __name__ == "__main__":
    main()
