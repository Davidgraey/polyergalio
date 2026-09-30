"""Distributed training tab: an orchestrator, local nodes, and one button per orchestrator process."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from common import show_diagram
from diagrams import plot_cluster
from orchestration_example import KEY, PHASES, STARTED_STAGES, Cluster, build_network, hyperparameters
from polyergalio.visuals.nnet_visuals import plot_network
from supervised_data import select_data

CLUSTER = "orchestration_cluster"
ERROR = "orchestration_error"
NEXT_STEP = {
    "listening": "Discover nodes (optional), then Connect + authenticate",
    "connected": "Initialize training",
    "ready": "Dispatch step",
    "stepped": "Aggregate gradients",
    "aggregated": "Apply gradients",
}
LOG_NAME = "collect.log"


def stop_cluster() -> None:
    cluster = st.session_state.pop(CLUSTER, None)
    if cluster is not None and cluster.stage != "stopped":
        cluster.shutdown()


def act(label: str, action, *args) -> None:
    """Run an orchestrator process, keep any error for the next draw, and redraw with the new stage."""
    st.session_state.pop(ERROR, None)
    try:
        with st.spinner(label):
            action(*args)
    except Exception as error:
        st.session_state[ERROR] = f"{label} failed: {error}"
    st.rerun()


def plot_history(history: list) -> plt.Figure:
    rounds = [entry["round"] for entry in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(rounds, [entry["full_loss"] for entry in history], color="black", label="all data")
    for node_id in history[-1]["node_loss"]:
        axes[0].scatter(rounds, [entry["node_loss"].get(node_id) for entry in history], s=10, alpha=0.6, label=node_id)
    axes[0].set_title("Mean squared error")
    axes[0].legend(fontsize=7)
    for node_id in history[-1]["gradient_norm"]:
        axes[1].plot(rounds, [entry["gradient_norm"].get(node_id) for entry in history], alpha=0.5, label=node_id)
    axes[1].plot(rounds, [entry["pooled_norm"] for entry in history], color="black", linewidth=2, label="pooled")
    axes[1].set_title("Gradient norm")
    axes[1].legend(fontsize=7)
    axes[2].bar(rounds, [entry["weight_gap"] for entry in history], color="#8338ec")
    axes[2].set_title("Largest weight difference between nodes")
    for axis in axes:
        axis.set_xlabel("Round")
    fig.tight_layout()
    return fig


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "An orchestrator dials several node processes on this machine, authenticates with a shared key, sends each "
            "the model and a shard of data, averages the gradients they return, and sends the pool back so every "
            "node applies the same update. Discovery finds nodes over a UDP beacon and records them in a roster file. "
            "A node that drops is redialed, or rediscovered if it moved, and resynced before the next step."
        )
        diagram = st.container()
        st.subheader("Settings")
        num_nodes = int(st.number_input("Nodes", 1, 8, 3, key="orchestration_nodes"))
        hidden = int(st.number_input("Hidden dim", 2, 128, 8, key="orchestration_hidden"))
        activation = st.selectbox("Hidden activation", ["tanh", "relu", "sigmoid"], key="orchestration_activation")
        optimizer = st.selectbox("Optimizer", ["SGD", "Adam"], key="orchestration_optimizer")
        learning_rate = float(st.number_input("Learning rate", 0.001, 1.0, 0.05, step=0.01, format="%.3f", key="orchestration_lr"))
        batch_size = int(st.number_input("Rows per node per step", 1, 512, 16, key="orchestration_batch"))
        rounds = int(st.number_input("Rounds for Run rounds", 1, 500, 20, key="orchestration_rounds"))
        key = st.text_input("Shared key for the nodes", KEY, key="orchestration_key")
        wrong_key = st.checkbox("Orchestrator uses a different key", key="orchestration_wrong_key")

    with right:
        st.subheader("Data")
        data = select_data("orchestration", "regression", samples=600, features=3, settings=left)
        st.subheader("Orchestrator")
        cluster = st.session_state.get(CLUSTER)
        stage = cluster.stage if cluster else "idle"
        node_ids = [node.node_id for node in cluster.nodes] if cluster else []
        chosen = st.multiselect("Nodes to use after discovery", node_ids, default=node_ids, key=f"orchestration_selected_{len(node_ids)}", disabled=cluster is None)
        skip_busy = st.checkbox("Discovery skips busy nodes", key="orchestration_skip_busy")
        target = st.selectbox("Node to disturb", node_ids or ["-"], key=f"orchestration_target_{len(node_ids)}", disabled=cluster is None)
        rows = [st.columns(3) for _ in range(5)]
        start = rows[0][0].button("Start nodes", key="orchestration_start")
        discover = rows[0][1].button("Discover nodes", key="orchestration_discover", disabled=stage not in STARTED_STAGES)
        connect = rows[0][2].button("Connect + authenticate", key="orchestration_connect", disabled=stage not in STARTED_STAGES)
        initialize = rows[1][0].button("Initialize training", key="orchestration_initialize", disabled=stage not in ("connected", "ready", "stepped", "aggregated"))
        dispatch = rows[1][1].button("Dispatch step", key="orchestration_dispatch", disabled=stage != "ready")
        aggregate = rows[1][2].button("Aggregate gradients", key="orchestration_aggregate", disabled=stage != "stepped")
        apply = rows[2][0].button("Apply gradients", key="orchestration_apply", disabled=stage != "aggregated")
        one_pass = rows[2][1].button("Run one pass", key="orchestration_pass", disabled=stage != "ready")
        many = rows[2][2].button("Run rounds", key="orchestration_many", disabled=stage != "ready")
        drop = rows[3][0].button("Drop its connection", key="orchestration_drop", disabled=stage not in ("ready", "stepped", "aggregated"))
        restart = rows[3][1].button("Restart it on a new port", key="orchestration_restart", disabled=stage not in ("ready", "stepped", "aggregated"))
        collect = rows[4][0].button("Collect + save model", key="orchestration_collect", disabled=stage != "ready")
        shutdown = rows[4][1].button("Shut down", key="orchestration_shutdown", disabled=cluster is None)

        settings = hyperparameters(optimizer, learning_rate)
        if start:
            stop_cluster()
            act("Starting nodes", lambda: st.session_state.update({CLUSTER: Cluster(num_nodes, key)}))
        elif shutdown:
            act("Shutting down", stop_cluster)
        elif cluster is not None:
            if discover:
                act("Discovering", cluster.discover, 1.0, skip_busy)
            elif connect:
                act("Connecting", cluster.connect, cluster.key + "-wrong" if wrong_key else None, chosen)
            elif initialize:
                act("Initializing", cluster.initialize, build_network(data.x.shape[1], hidden, activation), settings, data.x, data.y, batch_size)
            elif dispatch:
                act("Dispatching", cluster.dispatch_step, batch_size)
            elif aggregate:
                act("Aggregating", cluster.aggregate)
            elif apply:
                act("Applying", cluster.apply)
            elif one_pass:
                act("Running a pass", cluster.run_pass, batch_size)
            elif many:
                act(f"Running {rounds} rounds", lambda: [cluster.run_pass(batch_size) for _ in range(rounds)])
            elif drop:
                act("Dropping connection", cluster.drop_connection, target)
            elif restart:
                act("Restarting node", cluster.restart_node, target)
            elif collect:
                act("Collecting the model", cluster.collect_model)

        if ERROR in st.session_state:
            st.error(st.session_state[ERROR])
        if cluster is None:
            st.info("Start the nodes to begin.")
        else:
            if cluster.num_nodes != num_nodes or cluster.key != key:
                st.caption("The node count or key changed since the nodes started. Start nodes again to apply it.")
            if cluster.handler.missing_nodes():
                st.warning(f"Dropped: {', '.join(cluster.handler.missing_nodes())}. The next dispatch redials or rediscovers them and resyncs from a healthy node.")
            if cluster.stage in NEXT_STEP:
                st.caption(f"Stage: {cluster.stage}. Next: {NEXT_STEP[cluster.stage]}.")
            snapshot = cluster.snapshot()
            figure = plot_cluster(snapshot["nodes"], PHASES, snapshot["phase"], snapshot["round"], snapshot["pooled_norm"])
            st.pyplot(figure)
            plt.close(figure)
            if cluster.history:
                latest = cluster.history[-1]
                columns = st.columns(4)
                columns[0].metric("Rounds", latest["round"])
                columns[1].metric("Mean squared error", f"{latest['full_loss']:.4f}")
                columns[2].metric("Pooled gradient norm", f"{latest['pooled_norm']:.4f}")
                columns[3].metric("Node weight gap", f"{latest['weight_gap']:.1e}")
                figure = plot_history(cluster.history)
                st.pyplot(figure)
                plt.close(figure)
            if cluster.collected is not None:
                collected = cluster.collected
                message = (
                    f"Collected from {len(collected.deviations)} nodes at {collected.timestamp:%Y-%m-%d %H:%M:%S} UTC. "
                    f"Largest weight deviation {collected.max_deviation:.1e} (tolerance {collected.tolerance:.0e}). "
                    f"Saved to {cluster.checkpoint}"
                )
                (st.error if collected.diverged else st.success)(("Nodes diverged. " if collected.diverged else "") + message)
                log_path = cluster.checkpoint.parent / LOG_NAME
                if log_path.exists():
                    with st.expander("Collect log"):
                        st.code(log_path.read_text())
            if cluster.events:
                with st.expander("Message log"):
                    st.dataframe(pd.DataFrame(cluster.events).iloc[::-1], height=300)

    with diagram:
        with st.expander("Model structure", expanded=True):
            show_diagram(plot_network(build_network(data.x.shape[1], hidden, activation), figsize=(5, 4)).figure)
