import contextlib
import itertools

import numpy as np

from polyergalio.models.layers.spectre_layers import SpectreAttention
from polyergalio.models.optimizers import Adam

SEQUENCE_LENGTH = 16
FEATURES = 16
HEADS = 2
MEMORY_TOKENS = 4
STEPS = 600
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
SEEDS = 8
STUCK_THRESHOLD = 1e-3

CONFIGURATIONS = {
    "plain": {"memory_tokens": 0, "use_wrm": False},
    "memory": {"memory_tokens": MEMORY_TOKENS, "use_wrm": False},
    "memory+wrm": {"memory_tokens": MEMORY_TOKENS, "use_wrm": True},
}


@contextlib.contextmanager
def seeded_layers(seed):
    """Make layer initialisation reproducible by patching RandomState."""
    original = np.random.RandomState
    counter = itertools.count(seed * 1000)
    np.random.RandomState = lambda *args: original(*args) if args else original(next(counter))
    try:
        yield
    finally:
        np.random.RandomState = original


def shift_dataset(samples, rng):
    """Inputs are noise; targets are the inputs delayed by one position."""
    x = rng.normal(size=(samples, SEQUENCE_LENGTH, FEATURES))
    y = np.roll(x, 1, axis=1)
    y[:, 0] = 0.0
    return x, y


def train(seed, memory_tokens, use_wrm):
    """Train one layer and return the loss curve and held-out errors.

    Returns
    -------
    tuple
        Training losses, held-out mse on positions 1+, held-out mse on all positions.
    """
    rng = np.random.default_rng(100 + seed)
    x_train, y_train = shift_dataset(256, rng)
    x_test, y_test = shift_dataset(128, rng)
    with seeded_layers(seed):
        layer = SpectreAttention(
            SEQUENCE_LENGTH, FEATURES, num_heads=HEADS,
            memory_tokens=memory_tokens, use_wrm=use_wrm,
        )
    optimizer = Adam(LEARNING_RATE)
    order = np.random.default_rng(seed)
    losses = []
    layer.train()
    for _ in range(STEPS):
        idx = order.integers(0, len(x_train), BATCH_SIZE)
        layer.zero_gradients()
        output = layer.forward(x_train[idx])
        losses.append(np.mean((output - y_train[idx]) ** 2))
        layer.backward(2 * (output - y_train[idx]) / output.size)
        optimizer.step(layer)
    layer.eval()
    error = (layer.forward(x_test) - y_test) ** 2
    return np.array(losses), error[:, 1:].mean(), error.mean()


def main():
    print(f"{SEEDS} seeds, {STEPS} steps, Adam {LEARNING_RATE}")
    print(f"{'config':12s} {'median':>9s} {'worst':>9s} {'stuck':>6s}")
    for name, settings in CONFIGURATIONS.items():
        errors = np.array([train(seed, **settings)[1] for seed in range(SEEDS)])
        stuck = int(np.sum(errors > STUCK_THRESHOLD))
        print(f"{name:12s} {np.median(errors):9.5f} {errors.max():9.4f} {stuck:>3d}/{SEEDS}")


if __name__ == "__main__":
    main()
