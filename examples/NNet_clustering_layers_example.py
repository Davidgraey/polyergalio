"""
Clustering layers example.

Trains the differentiable prototype layers -- CentroidLayer, PLSOMLayer,
GPLSOMLayer and FreePLSOMLayer -- in two ways.

Unsupervised: each layer is the only node of a Network. Every batch runs
forward, then backward(None), so the layer learns from its own clustering
energy alone; end_epoch() lets the growing layers grow, prune and merge, and
anneal() sharpens the assignments.

Supervised: FullyConnectedLayer encoder -> clustering layer -> FullyConnectedLayer
classifier, trained with cross entropy. The downstream loss reaches both the
encoder and the prototypes through the soft assignment. Adam trains the
encoder and classifier; the clustering layer is not adaptive, so Adam gives
its prototypes a plain gradient step at the layer's own learning_rate.

Run: python NNet_clustering_layers_example.py
"""

import numpy as np
from polyergalio.generators.data_generators import RandomDatasetGenerator, to_onehot
from polyergalio.models.clustering.cluster_metrics import homogeneity
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.layers import (
    CentroidLayer,
    FreePLSOMLayer,
    FullyConnectedLayer,
    GPLSOMLayer,
    PLSOMLayer,
)
from polyergalio.models.layers.clustering_layers import ParameterlessLayer
from polyergalio.models.model_loss import CrossEntropyLoss
from polyergalio.models.network import Network
from polyergalio.models.optimizers import SGD, Adam

LAYER_NAMES = ("CentroidLayer", "PLSOMLayer", "GPLSOMLayer", "FreePLSOMLayer")

NUM_SAMPLES = 900
NUM_CLUSTERS = 5
UNSUPERVISED_FEATURES = 2
SUPERVISED_FEATURES = 8

GRID_DIM = 5
START_GRID_DIM = 3
START_NEURONS = 9
SPREAD_FACTOR = 0.95
TEMPERATURE = 0.3
ANNEAL = 0.97
FREEZE_FRACTION = 0.15

UNSUPERVISED_EPOCHS = 60
UNSUPERVISED_BATCH = 100
SOM_RATE = 1.0
CENTROID_RATE = 3.0

HIDDEN_DIM = 8
PROTOTYPE_RATE = 0.05
ENERGY_WEIGHT = 0.1
SUPERVISED_EPOCHS = 60
SUPERVISED_BATCH = 100
LEARNING_RATE = 0.02


def make_data(num_features: int, seed: int = 0):
    """
    Standardized "clustering" data and its true labels.

    Returns
    -------
    features (standardized), integer labels
    """
    x, labels, meta = RandomDatasetGenerator(random_seed=seed).generate(
        task="clustering",
        num_samples=NUM_SAMPLES,
        num_features=num_features,
        num_clusters=NUM_CLUSTERS,
        noise_scale=0.6,
        verbose=False,
    )
    return (x - x.mean(axis=0)) / x.std(axis=0), labels.astype(int)


def build_layer(name: str, input_dim: int, num_clusters: int = NUM_CLUSTERS, output_type: str = "assignment", **options):
    """
    One of LAYER_NAMES with this example's settings; options override them.

    Parameters
    ----------
    name : entry of LAYER_NAMES
    input_dim : width of the vectors being clustered
    num_clusters : centroid count for CentroidLayer
    output_type : assignment, distance, quantized or coordinates (coordinates
        only on the lattice layers)
    """
    common = dict(temperature=TEMPERATURE, output_type=output_type)
    if name == "CentroidLayer":
        return CentroidLayer(num_clusters, input_dim, **{**common, **options})
    if name == "PLSOMLayer":
        return PLSOMLayer(GRID_DIM, GRID_DIM, input_dim, **{**common, **options})
    if name == "GPLSOMLayer":
        return GPLSOMLayer(
            START_GRID_DIM, START_GRID_DIM, input_dim, spread_factor=SPREAD_FACTOR, **{**common, **options}
        )
    if name == "FreePLSOMLayer":
        return FreePLSOMLayer(
            START_NEURONS, input_dim, spread_factor=SPREAD_FACTOR, **{**common, **options}
        )
    raise ValueError(f"name must be one of {LAYER_NAMES}, got {name!r}")


def quantization_error(layer, x: np.ndarray) -> float:
    """Mean distance from each sample to its nearest prototype."""
    return float(np.sqrt(layer.distances(x).min(axis=-1)).mean())


def finish_epoch(layer, epoch: int, epochs: int, anneal: float = ANNEAL) -> None:
    """Freeze the structure late in training, restructure, and anneal, for layers that have an epoch protocol."""
    if isinstance(layer, ParameterlessLayer):
        layer.frozen = epoch >= (1.0 - FREEZE_FRACTION) * epochs
        layer.end_epoch()
    layer.anneal(anneal)


def train_unsupervised(layer, x: np.ndarray, epochs: int = UNSUPERVISED_EPOCHS, batch_size: int = UNSUPERVISED_BATCH,
                       anneal: float = ANNEAL, seed: int = 0) -> list:
    """
    Train a layer on its clustering energy alone.

    Parameters
    ----------
    layer : a prototype layer
    x : standardized data

    Returns
    -------
    per-epoch snapshots: dict(epoch, error, num_prototypes, weights)
    """
    net = Network(name="clustering", input_shape=(x.shape[1],))
    net.output = net.connect(layer, net.input, name="prototypes")
    optimizer = SGD(CENTROID_RATE if isinstance(layer, CentroidLayer) else SOM_RATE)
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(epochs):
        net.train()
        for batch in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch_size)):
            net.zero_gradients()
            net.forward(x[batch])
            net.backward(None)
            optimizer.step(net)
        finish_epoch(layer, epoch, epochs, anneal)
        history.append(
            dict(epoch=epoch, error=quantization_error(layer, x), num_prototypes=layer.num_prototypes, weights=layer.weights.copy())
        )
    return history


def cluster_unsupervised(name: str, x: np.ndarray, labels: np.ndarray, num_clusters: int = NUM_CLUSTERS,
                         epochs: int = UNSUPERVISED_EPOCHS, batch_size: int = UNSUPERVISED_BATCH,
                         anneal: float = ANNEAL, **options) -> dict:
    """
    Build and train one layer, then print how it did.

    Returns
    -------
    dict(layer, history, homogeneity)
    """
    layer = build_layer(name, x.shape[1], num_clusters, **options)
    history = train_unsupervised(layer, x, epochs, batch_size, anneal)
    score = homogeneity(labels, layer.labels(x))
    events = [event for _, event, _ in getattr(layer, "structure_trace", [])]
    print(
        f"  {name:<15s} prototypes={layer.num_prototypes:3d}  "
        f"quantization error {history[0]['error']:.3f} -> {history[-1]['error']:.3f}  "
        f"homogeneity {score:.3f}  "
        f"grow/prune/merge/reinit {events.count('grow')}/{events.count('prune')}/{events.count('merge')}/{events.count('reinit')}"
    )
    return dict(layer=layer, history=history, homogeneity=score)


def build_classifier(name: str, num_features: int, num_classes: int, hidden_dim: int = HIDDEN_DIM,
                     energy_weight: float = ENERGY_WEIGHT, prototype_rate: float = PROTOTYPE_RATE, **options) -> Network:
    """
    Encoder -> clustering layer -> classifier.

    Growing layers emit quantized output, whose width does not change with
    the map, so the classifier can be wired before any growth happens.

    Parameters
    ----------
    energy_weight : weight of the layer's own clustering energy next to the classification loss
    prototype_rate : the layer's learning_rate, its prototype step under Adam
    options : other layer settings, such as temperature
    """
    output_type = "quantized" if name in ("GPLSOMLayer", "FreePLSOMLayer") else "assignment"
    clustering = build_layer(
        name, hidden_dim, num_classes, output_type, energy_weight=energy_weight, learning_rate=prototype_rate, **options
    )
    width = clustering.shapes["output"][0][0] or hidden_dim

    net = Network(name=f"classifier_{name}", input_shape=(num_features,))
    encoded = net.connect(FullyConnectedLayer(num_features, hidden_dim, activation_type="tanh"), net.input, name="encoder")
    clustered = net.connect(clustering, encoded, name="clusters")
    net.output = net.connect(
        FullyConnectedLayer(width, num_classes, activation_type="linear", is_output=True), clustered, name="classifier"
    )
    return net


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of rows whose argmax logit matches the label."""
    return float(np.mean(np.argmax(logits, axis=-1) == labels))


def train_classifier(net: Network, x: np.ndarray, labels: np.ndarray, epochs: int = SUPERVISED_EPOCHS,
                     batch_size: int = SUPERVISED_BATCH, learning_rate: float = LEARNING_RATE,
                     anneal: float = ANNEAL, seed: int = 0) -> dict:
    """
    Train the classifier with cross entropy and Adam, restructuring the
    clustering layer after each epoch.

    Returns
    -------
    dict of accuracy before and after, per-epoch losses and prototype counts, and the mean prototype shift (None when the map grew or shrank)
    """
    layer = net.node("clusters").layer
    targets = to_onehot(labels, int(labels.max()) + 1)
    loss_fn = CrossEntropyLoss(ClassificationTask.MULTINOMIAL)
    optimizer = Adam(learning_rate)
    rng = np.random.default_rng(seed)

    net.eval()
    before = accuracy(net.forward(x), labels)
    start = layer.weights.copy()
    losses, counts = [], []

    for epoch in range(epochs):
        net.train()
        for batch in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch_size)):
            net.zero_gradients()
            loss = loss_fn(net.forward(x[batch]), targets[batch])
            net.backward(loss_fn.backward())
            optimizer.step(net)
        finish_epoch(layer, epoch, epochs, anneal)
        losses.append(float(loss))
        counts.append(layer.num_prototypes)

    net.eval()
    after = accuracy(net.forward(x), labels)
    moved = float(np.abs(layer.weights - start).mean()) if layer.weights.shape == start.shape else None
    return dict(before=before, after=after, losses=losses, counts=counts, moved=moved)


def classify_supervised(name: str, x: np.ndarray, labels: np.ndarray, epochs: int = SUPERVISED_EPOCHS,
                        batch_size: int = SUPERVISED_BATCH, learning_rate: float = LEARNING_RATE,
                        hidden_dim: int = HIDDEN_DIM, energy_weight: float = ENERGY_WEIGHT,
                        prototype_rate: float = PROTOTYPE_RATE, anneal: float = ANNEAL, **options) -> dict:
    """
    Build and train one classifier, then print how it did.

    Returns
    -------
    dict(net, plus the train_classifier results)
    """
    net = build_classifier(name, x.shape[1], int(labels.max()) + 1, hidden_dim, energy_weight, prototype_rate, **options)
    result = train_classifier(net, x, labels, epochs, batch_size, learning_rate, anneal)
    print(
        f"  {name:<15s} accuracy {result['before']:.3f} -> {result['after']:.3f}  "
        f"final loss {result['losses'][-1]:.3f}  prototypes {result['counts'][-1]}  "
        + (f"mean prototype shift {result['moved']:.3f}" if result["moved"] is not None else "map restructured")
    )
    return dict(net=net, **result)


def main():
    x, labels = make_data(UNSUPERVISED_FEATURES)
    print(f"unsupervised: {len(x)} samples, {UNSUPERVISED_FEATURES} features, {NUM_CLUSTERS} true clusters")
    for name in LAYER_NAMES:
        cluster_unsupervised(name, x, labels)

    x, labels = make_data(SUPERVISED_FEATURES)
    print(f"\nsupervised: encoder -> clustering layer -> classifier, {SUPERVISED_FEATURES} features")
    for name in LAYER_NAMES:
        classify_supervised(name, x, labels)


if __name__ == "__main__":
    main()
