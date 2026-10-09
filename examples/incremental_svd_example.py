"""
Incremental SVD layer example.

IncrementalSVDLayer merges batches of rows into a running SVD, so the eigenvectors of X'X / n and the orthogonalized,
standardized rows

    p, s, q = svd(x); z_std = zscore(p @ q)

are available while data streams in. The layer expects rows that are already normalized: it does not center or scale
them. Parts 1 and 2 use data normalized up front, part 3 puts a StandardizeLayer ahead of it.

1. Batch sizes: the factors match the one-shot decomposition of the full matrix whatever the batch size; the output
   differs slightly because its mean and std are moving statistics.
2. decay_rate: on data whose covariance changes halfway, decay_rate below 1 follows the recent covariance.
3. In a network: StandardizeLayer -> IncrementalSVDLayer -> linear head against StandardizeLayer -> head, with the
   SVD layer at full width and at a reduced output_dimension, which keeps the shape fixed for the connection (the
   target depends on every feature, so the narrow output costs accuracy). Then a serialize / deserialize round trip.
4. Output forms and freezing: the orthogonalized, whitened and projection outputs; eigenvector_change shrinking as
   batches arrive; freeze() so extra epochs over the same rows leave the factors and counts alone.

Run: python incremental_svd_example.py
"""

import numpy as np
from typing import Optional

from polyergalio.composite_model import Composite
from polyergalio.models.layers import FullyConnectedLayer, StandardizeLayer
from polyergalio.models.model_loss import MSELoss
from polyergalio.models.network import Network
from polyergalio.models.optimizers import SGD
from polyergalio.transforms.decompositions import IncrementalSVDLayer
from polyergalio.utilities import standardize_data

NUM_SAMPLES = 3000
NUM_FEATURES = 8
REDUCED_DIMENSION = 4
BATCH_PLANS = ((100, 100), (100, 7), (7, 7), (1, 1))
DECAY_RATES = (1.0, 0.8)
DRIFT_BATCH = 100
EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.05
CHANGE_TOLERANCE = 0.02
EXTRA_EPOCHS = 3
NOISE = 0.1


def make_data(rng: np.random.Generator) -> np.ndarray:
    """Correlated features with a large offset."""
    mixing = rng.normal(size=(NUM_FEATURES, NUM_FEATURES))
    return rng.normal(size=(NUM_SAMPLES, NUM_FEATURES)) @ mixing + 5 * rng.normal(size=NUM_FEATURES)


def one_shot(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    eigenvalues and eigenvectors of x'x / n, largest first, and zscore(p @ q) from the full SVD
    """
    values, vectors = np.linalg.eigh(x.T @ x / len(x))
    p, _, q = np.linalg.svd(x, full_matrices=False)
    return values[::-1], vectors[:, ::-1], standardize_data(p @ q)


def stream(layer: IncrementalSVDLayer, x: np.ndarray, first: int, size: int) -> IncrementalSVDLayer:
    """Feed x through the layer in training mode: one batch of `first` rows, then batches of `size`."""
    layer.forward(x[:first])
    for start in range(first, len(x), size):
        layer.forward(x[start:start + size])
    return layer


def compare_batch_sizes(x: np.ndarray) -> None:
    values, vectors, z_std = one_shot(x)
    print("batch sizes against the one-shot decomposition")
    print(f"{'first':>6} {'batch':>6} {'eigenvalue err':>15} {'min |cos|':>10} {'output diff':>12}")
    for first, size in BATCH_PLANS:
        layer = stream(IncrementalSVDLayer(NUM_FEATURES), x, first, size).eval()
        eigenvalue_error = np.max(np.abs(layer.eigenvalues / values - 1))
        alignment = np.min(np.abs(np.sum(layer.eigenvectors * vectors, axis=0)))
        output_diff = np.mean(np.abs(layer.forward(x) - z_std))
        print(f"{first:>6} {size:>6} {eigenvalue_error:>15.1e} {alignment:>10.6f} {output_diff:>12.4f}")
    print(f"explained variance ratio: {np.round(layer.explained_variance_ratio, 3)}\n")


def compare_decay(rng: np.random.Generator) -> None:
    halves = [
        rng.normal(size=(NUM_SAMPLES // 2, NUM_FEATURES)) @ rng.normal(size=(NUM_FEATURES, NUM_FEATURES))
        for _ in range(2)
    ]
    x = np.vstack(halves)
    recent_top = np.linalg.eigh(halves[1].T @ halves[1])[1][:, -1]
    print("covariance changes halfway; |cos| of the top eigenvector with the recent half's")
    for rate in DECAY_RATES:
        layer = stream(IncrementalSVDLayer(NUM_FEATURES, decay_rate=rate), x, DRIFT_BATCH, DRIFT_BATCH)
        alignment = abs(layer.eigenvectors[:, 0] @ recent_top)
        print(f"  decay_rate {rate}: {alignment:.4f}, effective rows {layer.num_seen_samples:.0f} of {len(x)}")
    print()


def build_network(output_dimension: Optional[int]) -> Network:
    """StandardizeLayer, then an IncrementalSVDLayer of that output width unless None, then a linear head."""
    net = Network(name="standardized" if output_dimension is None else f"svd_{output_dimension}", input_shape=(NUM_FEATURES,))
    node = net.connect(StandardizeLayer(NUM_FEATURES), net.input, name="normalize")
    width = NUM_FEATURES
    if output_dimension is not None:
        form = "orthogonalized" if output_dimension == NUM_FEATURES else "whitened"
        layer = IncrementalSVDLayer(NUM_FEATURES, output_dimension=output_dimension, output_form=form)
        node = net.connect(layer, node, name="svd")
        width = output_dimension
    net.output = net.connect(
        FullyConnectedLayer(width, 1, activation_type="linear", is_output=True), node, name="head"
    )
    return net


def train_head(net: Network, x: np.ndarray, y: np.ndarray, seed: int = 0) -> list[float]:
    """SGD on mean squared error; returns the full-data error after each epoch."""
    loss_fn = MSELoss()
    optimizer = SGD(LEARNING_RATE)
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(EPOCHS):
        net.train()
        for batch in np.array_split(rng.permutation(len(x)), len(x) // BATCH_SIZE):
            net.zero_gradients()
            loss_fn(net.forward(x[batch]), y[batch])
            net.backward(loss_fn.backward())
            optimizer.step(net)
        net.eval()
        errors.append(float(MSELoss()(net.forward(x), y)))
    return errors


def compare_networks(x: np.ndarray, rng: np.random.Generator) -> None:
    y = standardize_data(x) @ rng.normal(size=(NUM_FEATURES, 1)) + NOISE * rng.normal(size=(len(x), 1))
    print(f"linear head, SGD at {LEARNING_RATE}: full-data mean squared error by epoch")
    nets = {}
    for output_dimension in (None, NUM_FEATURES, REDUCED_DIMENSION):
        net = build_network(output_dimension)
        errors = train_head(net, x, y)
        nets[output_dimension] = net
        shape = "" if output_dimension is None else f"  output shape {net.node('svd').layer.shapes['output'][0]}"
        print(f"  {net.name:>12}: {errors[0]:.3f} -> {errors[-1]:.3f}{shape}")

    net = nets[NUM_FEATURES]
    restored = Composite.deserialize(net.serialize()).eval()
    print(f"restored network output matches: {np.allclose(restored.forward(x), net.forward(x))}")


def compare_forms(x: np.ndarray) -> None:
    values = one_shot(x)[0]
    print("output forms, unstandardized: variance of each output column")
    for form in ("orthogonalized", "whitened", "projection"):
        layer = stream(IncrementalSVDLayer(NUM_FEATURES, output_form=form, standardize=False), x, 100, 100).eval()
        print(f"  {form:>14}: {np.round(layer.forward(x).var(axis=0), 3)}")
    print(f"  eigenvalues:    {np.round(values, 3)}\n")


def freeze_when_settled(x: np.ndarray) -> None:
    layer = IncrementalSVDLayer(NUM_FEATURES).train()
    print("eigenvector_change (radians) after each batch of 100 rows")
    for index, start in enumerate(range(0, len(x), BATCH_SIZE)):
        layer.forward(x[start:start + BATCH_SIZE])
        if index in (0, 1, 2, 4, 9, 19) or layer.eigenvector_change < CHANGE_TOLERANCE:
            print(f"  batch {index + 1:>2}: {layer.eigenvector_change:.4f}")
        if layer.eigenvector_change < CHANGE_TOLERANCE:
            break
    layer.freeze()
    seen, values = layer.num_seen_samples, layer.singular_values.copy()
    for _ in range(EXTRA_EPOCHS):
        for start in range(0, len(x), BATCH_SIZE):
            layer.forward(x[start:start + BATCH_SIZE])
    print(f"frozen after {seen:.0f} rows; {EXTRA_EPOCHS} more epochs in training mode")
    print(f"  rows counted unchanged: {layer.num_seen_samples == seen}")
    print(f"  singular values unchanged: {np.array_equal(layer.singular_values, values)}\n")


def main():
    rng = np.random.default_rng(0)
    x = make_data(rng)
    compare_batch_sizes(standardize_data(x))
    compare_decay(rng)
    compare_networks(x, rng)
    compare_forms(standardize_data(x))
    freeze_when_settled(standardize_data(x))


if __name__ == "__main__":
    main()
