"""Clustering tab: PLSOM, GPLSOM, FreePLSOM, SOLS and CentroidNeuralNetwork."""

import numpy as np
import pandas as pd
import streamlit as st
from clustering_example import MECHANISM_NAMES, RUNNERS, SCENARIOS, report
from common import (
    discard_figures,
    edit_table,
    emit_image,
    flush_figures,
    run_panel,
    scratch_path,
    show_diagram,
)
from diagrams import plot_graph
from polyergalio.generators import RandomDatasetGenerator
from polyergalio.visuals.cluster_visuals import (
    animate_growth,
    animate_membership,
    plot_clusters,
    plot_final_predictions,
)

DIAGRAMS = {
    "PLSOM": [
        ("Data", "Standardize"),
        ("Standardize", "Best matching unit on a fixed grid"),
        ("Best matching unit on a fixed grid", "Adaptive neighborhood: epsilon and theta"),
        ("Adaptive neighborhood: epsilon and theta", "Move prototypes toward the sample"),
        ("Move prototypes toward the sample", "Cluster the prototypes"),
        ("Cluster the prototypes", "Label points by nearest prototype"),
    ],
    "GPLSOM": [
        ("Data", "Standardize"),
        ("Standardize", "Best matching unit on a rectangular grid"),
        ("Best matching unit on a rectangular grid", "Adaptive neighborhood: epsilon and theta"),
        ("Adaptive neighborhood: epsilon and theta", "Move prototypes toward the sample"),
        ("Move prototypes toward the sample", "Grow or prune a whole row or column"),
        ("Grow or prune a whole row or column", "Cluster the prototypes"),
        ("Cluster the prototypes", "Label points by nearest prototype"),
    ],
    "FreePLSOM": [
        ("Data", "Standardize"),
        ("Standardize", "Best matching unit among free neurons"),
        ("Best matching unit among free neurons", "Rank neighborhood, as in neural gas"),
        ("Rank neighborhood, as in neural gas", "Move neurons toward the sample"),
        ("Move neurons toward the sample", "Grow, prune or merge single neurons"),
        ("Grow, prune or merge single neurons", "Cluster the neurons"),
        ("Cluster the neurons", "Label points by nearest neuron"),
    ],
    "SOLS": [
        ("Data", "Standardize"),
        ("Standardize", "Soft assignment: softmax over distance and lattice terms"),
        ("Soft assignment: softmax over distance and lattice terms", "Update latent positions and prototypes"),
        ("Update latent positions and prototypes", "Per-sample neighborhood: epsilon, theta and gamma"),
        ("Per-sample neighborhood: epsilon, theta and gamma", "Repeat until the objective settles"),
        ("Repeat until the objective settles", "Cluster the live prototypes"),
        ("Cluster the live prototypes", "Label points by nearest prototype"),
    ],
    "CentroidNN": [
        ("Data", "Standardize"),
        ("Standardize", "Start with two centroids"),
        ("Start with two centroids", "Assign points and update centroids"),
        ("Assign points and update centroids", "Split a cluster to grow k"),
        ("Split a cluster to grow k", "Score every k with cluster metrics"),
        ("Score every k with cluster metrics", "Select the optimal k"),
    ],
}


def generate_frame(config: dict, seed: int):
    """Generator output as (table with F columns and a label column, meta)."""
    x, truth, meta = RandomDatasetGenerator(random_seed=seed).generate(task="clustering", verbose=False, **config)
    frame = pd.DataFrame(x, columns=[f"F{i}" for i in range(x.shape[1])])
    frame["label"] = truth
    return frame, meta


def cluster(x, truth, centroids, k: int, mechanisms: list, stride: int, animate: bool, title: str) -> None:
    """Run each chosen mechanism, print its metrics, plot results and optionally animate."""
    if x.shape[1] > 2:
        print(f"{x.shape[1]} features: plots show the first two dimensions only")
    plot_clusters(x, truth, centroids)
    predictions, histories = {}, {}
    for name in mechanisms:
        prediction, elapsed, history = RUNNERS[name](x, k, stride=stride)
        report(name, x, truth, prediction, elapsed)
        predictions[name], histories[name] = prediction, history
    plot_final_predictions(title, x, predictions)
    flush_figures()
    if animate:
        growth, membership = scratch_path("growth.gif"), scratch_path("membership.gif")
        animate_growth(title, histories, save_path=growth, show=False, stride=stride)
        animate_membership(title, x, histories, save_path=membership, show=False, stride=stride)
        discard_figures()
        emit_image(growth, "Prototype growth")
        emit_image(membership, "Cluster membership")


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "Self-organizing maps, a self-organizing latent space and a centroid network that grows its cluster count, "
            "compared on synthetic scenarios."
        )
        with st.expander("Model structure", expanded=True):
            name = st.selectbox("Mechanism", MECHANISM_NAMES, key="cluster_diagram")
            show_diagram(plot_graph(DIAGRAMS[name], name))
        st.subheader("Settings")
        mechanisms = st.multiselect("Mechanisms", MECHANISM_NAMES, default=list(MECHANISM_NAMES), key="cluster_mechanisms")
        k = int(st.number_input("Clusters to find", 2, 20, 5, key="cluster_k"))
        stride = int(st.number_input("Animation stride", 1, 10, 3, key="cluster_stride"))
        animate = st.checkbox("Render animations", key="cluster_animate")
        st.subheader("Data settings")
        scenario = st.selectbox("Scenario", list(SCENARIOS), key="cluster_scenario")
        preset = SCENARIOS[scenario]
        samples = int(st.number_input("Samples", 100, 5000, 600, step=100, key="cluster_samples"))
        features = int(st.number_input("Features", 2, 30, preset["num_features"], key="cluster_features"))
        clusters = int(st.number_input("Clusters", 2, 20, preset["num_clusters"], key="cluster_count"))
        seed = int(st.number_input("Seed", 0, 9999, 123, key="cluster_seed"))
    config = dict(preset, num_samples=samples, num_features=features, num_clusters=clusters)

    with right:
        st.subheader("Data")
        frame, meta, unchanged = edit_table(
            f"cluster_{scenario}",
            (scenario, samples, features, clusters, seed),
            lambda: generate_frame(config, seed),
        )
        st.caption("Columns F0, F1, ... are features; label is the true cluster (-1 for outliers) and is optional.")

        frame = frame.dropna(subset=[c for c in frame.columns if c != "label"])
        features_only = frame[[c for c in frame.columns if c != "label"]]
        x = features_only.to_numpy(dtype=float)
        truth = frame["label"].fillna(-1).to_numpy(dtype=int) if "label" in frame else np.zeros(len(x), dtype=int)
        centroids = meta.get("centroids") if unchanged else None
        run_panel("cluster", cluster, x, truth, centroids, k, mechanisms, stride, animate, scenario)
