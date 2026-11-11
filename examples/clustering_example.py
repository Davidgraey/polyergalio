"""
Clustering example.

Runs every clustering mechanism in this package -- PLSOM, GPLSOM, FreePLSOM,
SelfOrganizingLatentSpace (SOLS) and CentroidNeuralNetwork -- against several RandomDatasetGenerator
"clustering" scenarios, each picked to show off one of the generator's
clustering controls (plain blobs, non-convex shapes, imbalance/separation,
noise/outliers, higher dimensionality).

For each scenario this prints a metrics report per mechanism (including the
Cubic Clustering Criterion), plots the final predicted clusters side by
side, and writes two GIFs per mechanism set, one subplot per mechanism: how
its prototypes or centroids evolve over training/growth (animate_growth),
and how its predicted cluster membership -- i.e. each point's color --
evolves over the same steps (animate_membership). Both are shown alongside
being saved, so they can be reviewed without re-running. Saving uses
matplotlib's own Pillow-backed writer, so no external encoder (e.g. ffmpeg)
is required. By default both animations only keep every 3rd growth step
(--stride), which keeps GIF creation quick; pass --stride 1 for every step.

The plotting and animation support (plot_final_predictions, animate_growth,
animate_membership) lives in polyergalio.visuals.cluster_visuals, reusable
outside this script; the generic GIF thin/save helpers underneath those
live in polyergalio.visuals.animation_visuals.

Run all scenarios:            python clustering_example.py
Run specific scenarios:       python clustering_example.py blobs noisy
Save without showing windows: python clustering_example.py --no-show
Save into a specific folder:  python clustering_example.py --save-dir out
Animate every step (slower):  python clustering_example.py --stride 1
"""

import argparse
import os
import time

import numpy as np
from polyergalio.generators import RandomDatasetGenerator
from polyergalio.models.clustering.centroid_network import CentroidNeuralNetwork
from polyergalio.models.clustering.cluster_metrics import (
    calinski_harabasz_index,
    cubic_clustering_criterion,
    davies_bouldin_index,
    homogeneity,
    silhouette_score,
)
from polyergalio.models.clustering.freeplsom_clustering import FreePLSOM
from polyergalio.models.clustering.gplsom_clustering import GPLSOM
from polyergalio.models.clustering.plsom_clustering import PLSOM
from polyergalio.models.clustering.plsom_latent import SelfOrganizingLatentSpace
from polyergalio.visuals.animation_utils import thin_indices
from polyergalio.visuals.cluster_visuals import (
    animate_growth,
    animate_membership,
    plot_clusters,
    plot_final_predictions,
)

# --------------- Scenarios ---------------
# Each scenario is a RandomDatasetGenerator(task="clustering") config,
# chosen to demonstrate one of the generator's clustering controls.
SCENARIOS = {
    "blobs": dict(
        num_samples=1500,
        num_features=2,
        num_clusters=6,
        noise_scale=0.6,
    ),
    "shapes": dict(
        num_samples=1500,
        num_features=2,
        num_clusters=4,
        noise_scale=0.5,
        cluster_shape=["ring", "moon", "elongated", "disc"],
    ),
    "imbalanced": dict(
        num_samples=1500,
        num_features=2,
        num_clusters=5,
        noise_scale=0.5,
        imbalance=2.5,
        separation=2.0,
    ),
    "noisy": dict(
        num_samples=1500,
        num_features=2,
        num_clusters=5,
        noise_scale=0.5,
        variance_jitter=0.7,
        outlier_fraction=0.08,
    ),
    "high_dim": dict(
        num_samples=1500,
        num_features=8,
        num_clusters=5,
        noise_scale=0.4,
    ),
    "high_dim, noisy": dict(
        num_samples=1500,
        num_features=20,
        num_clusters=9,
        noise_scale=0.77,
    ),
}

MECHANISM_NAMES = ("PLSOM", "GPLSOM", "FreePLSOM", "SOLS", "CentroidNN")
GRID_DIM = 6  # PLSOM's fixed width/height
GROW_EPOCHS = 256  # GPLSOM / FreePLSOM
SOLS_ITERATIONS = 256
CNN_MAX_CLUSTERS = 20
CNN_ITERATIONS = 256


# --------------- Metrics report ---------------


def report(name, x_data, truth, prediction, elapsed):
    print(f"  {name:<11s} {elapsed:6.2f}s  clusters found={len(np.unique(prediction))}")
    print(f"      silhouette (1 best):  {silhouette_score(x_data, prediction):.4f}")
    print(
        f"      CH index (high best): {calinski_harabasz_index(x_data, prediction):.4f}"
    )
    print(f"      DB index (low best):  {davies_bouldin_index(x_data, prediction):.4f}")
    print(
        f"      CCC (>2/3 is good):   {cubic_clustering_criterion(x_data, prediction):.4f}"
    )
    print(f"      homogeneity:          {homogeneity(truth, prediction):.4f}")


# --------------- Per-mechanism runners ---------------
# Each returns (prediction, elapsed, history). history is a list of
# per-step snapshots -- one per training epoch for the PLSOM family, one per
# cluster count grown for CentroidNeuralNetwork -- used by animate_growth and
# animate_membership. Snapshots num_kept by `stride` also carry a cluster
# membership ('prediction' for the PLSOM family, 'labels' for CentroidNN).


def _predict_from_weights(standardize_fn, x_data, weights, k):
    """
    Cluster membership for x_data from one weight snapshot, mirroring
    PLSOM.predict() (nearest prototype, then cluster the prototypes) without
    mutating the live model -- so it can be reused on a past training step.
    """
    _x = standardize_fn(x_data)
    dists = np.linalg.norm(_x[:, np.newaxis, :] - weights[np.newaxis, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    clust_model = CentroidNeuralNetwork(
        max_clusters=k, seed=42, initial_clusters=None, epsilon=1e-4
    )
    clust_model.fit_predict(
        org_x_data=weights,
        num_iterations=100,
        fast_forward=False,
        verbose=False,
        skip_standardize=True,
    )
    _, _, neuron_labels = clust_model.get_optimal()
    return neuron_labels[nearest]


def run_plsom(x_data, k, stride=1):
    som = PLSOM(
        width=GRID_DIM,
        height=GRID_DIM,
        input_dim=x_data.shape[1],
        theta_min=0.01,
        theta_max=GRID_DIM - 0.01,
        lock_seed=42,
        distance="euclidean",
        verbose=False,
    )
    history = []

    def on_epoch(model, step):
        history.append(
            dict(
                shape=tuple(model.network_shape),
                weights=model.weights.copy(),
                background=model.standardize(x_data),
            )
        )

    start = time.time()
    som.fit(x_data, num_iterations=GRID_DIM * 5, on_epoch=on_epoch)
    prediction = som.predict(x=x_data, n_clusters=k)
    for idx in thin_indices(len(history), stride):
        history[idx]["prediction"] = _predict_from_weights(
            som.standardize, x_data, history[idx]["weights"], k
        )
    return prediction, time.time() - start, history


def run_gplsom(x_data, k, stride=1):
    som = GPLSOM(
        width=3,
        height=3,
        input_dim=x_data.shape[1],
        theta_min=0.01,
        theta_max=2.99,
        spread_factor=0.95,
        max_neurons=120,
        verbose=False,
    )
    history = []

    def on_epoch(model, step):
        history.append(
            dict(
                shape=tuple(model.network_shape),
                weights=model.weights.copy(),
                background=model.standardize(x_data),
            )
        )

    start = time.time()
    som.fit(x_data, num_iterations=GROW_EPOCHS, on_epoch=on_epoch)
    prediction = som.predict(x=x_data, n_clusters=k)
    print(
        f"      grew to {som.n_neurons} neurons across {len(som.structure_trace)} structural changes"
    )
    for idx in thin_indices(len(history), stride):
        history[idx]["prediction"] = _predict_from_weights(
            som.standardize, x_data, history[idx]["weights"], k
        )
    return prediction, time.time() - start, history


def run_freeplsom(x_data, k, stride=1):
    som = FreePLSOM(
        n_neurons=9,
        input_dim=x_data.shape[1],
        spread_factor=0.95,
        max_neurons=120,
        verbose=False,
        lock_seed=42,
    )
    history = []

    def on_epoch(model, step):
        history.append(
            dict(
                weights=model.weights.copy(),
                background=model.standardize(x_data),
            )
        )

    start = time.time()
    som.fit(x_data, num_iterations=GROW_EPOCHS, on_epoch=on_epoch)
    prediction = som.predict(x=x_data, n_clusters=k)
    print(
        f"      grew to {som.n_neurons} neurons across {len(som.structure_trace)} structural changes"
    )
    for idx in thin_indices(len(history), stride):
        history[idx]["prediction"] = _predict_from_weights(
            som.standardize, x_data, history[idx]["weights"], k
        )
    return prediction, time.time() - start, history


def run_sols(x_data, k, stride=1):
    sols = SelfOrganizingLatentSpace(
        width=GRID_DIM,
        height=GRID_DIM,
        input_dim=x_data.shape[1],
        lock_seed=42,
    )
    history = []

    def on_epoch(model, step):
        history.append(
            dict(
                shape=tuple(model.network_shape),
                weights=model.weights.copy(),
                background=model.standardize(x_data),
            )
        )

    start = time.time()
    sols.fit(x_data, num_iterations=SOLS_ITERATIONS, on_epoch=on_epoch)
    prediction = sols.predict(x_data, n_clusters=k)
    print(f"      settled after {len(history)} cycles")
    for idx in thin_indices(len(history), stride):
        history[idx]["prediction"] = _predict_from_weights(
            sols.standardize, x_data, history[idx]["weights"], k
        )
    return prediction, time.time() - start, history


def run_centroid_nn(x_data, k, stride=1):
    cnn = CentroidNeuralNetwork(max_clusters=max(k + 3, CNN_MAX_CLUSTERS), epsilon=0.2)
    start = time.time()
    cnn.fit_predict(
        x_data,
        num_iterations=CNN_ITERATIONS,
        mini_batch=False,
        verbose=False,
        fast_forward=False,
    )
    _, _, prediction = cnn.get_optimal()
    history = cnn.growth_history()
    for snap in history:
        snap["background"] = x_data
    return prediction, time.time() - start, history


RUNNERS = dict(
    PLSOM=run_plsom,
    GPLSOM=run_gplsom,
    FreePLSOM=run_freeplsom,
    SOLS=run_sols,
    CentroidNN=run_centroid_nn,
)


# --------------- Scenario driver ---------------


def run_scenario(name, config, seed=123, save_dir=None, show=True, stride=1):
    print(f"\n=== scenario: {name} ===")
    gen = RandomDatasetGenerator(random_seed=seed)
    x_data, truth, meta = gen.generate(task="clustering", verbose=True, **config)
    k = config["num_clusters"]
    if x_data.shape[1] > 2:
        print(
            f"      ({x_data.shape[1]} features -- plots show the first two dimensions only)"
        )

    plot_clusters(x_data, truth, meta.get("centroids"))

    predictions = {}
    histories = {}
    for mech_name in MECHANISM_NAMES:
        prediction, elapsed, history = RUNNERS[mech_name](x_data, k, stride=stride)
        report(mech_name, x_data, truth, prediction, elapsed)
        predictions[mech_name] = prediction
        histories[mech_name] = history

    plot_final_predictions(name, x_data, predictions)
    growth_path = os.path.join(save_dir, f"{name}_growth.gif") if save_dir else None
    animate_growth(name, histories, save_path=growth_path, show=show, stride=stride)
    membership_path = (
        os.path.join(save_dir, f"{name}_membership.gif") if save_dir else None
    )
    animate_membership(
        name, x_data, histories, save_path=membership_path, show=show, stride=stride
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run clustering scenarios and animate model growth."
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        choices=list(SCENARIOS) or None,
        default=list(SCENARIOS.keys()),
        help="which scenarios to run (default: all)",
    )
    parser.add_argument(
        "--save-dir",
        default="animations",
        help="folder to write growth animation GIFs into (default: animations)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save animations without opening interactive windows",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="keep every Nth growth step in the animation, fewer frames means faster, smaller GIFs (default: 3, use 1 for every step)",
    )
    args = parser.parse_args()

    for scenario_name in args.scenarios:
        run_scenario(
            scenario_name,
            SCENARIOS[scenario_name],
            save_dir=args.save_dir,
            show=not args.no_show,
            stride=args.stride,
        )
