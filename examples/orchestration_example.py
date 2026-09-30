"""
Distributed training example.

Starts NodeLauncher servers on localhost, has an Orchestrator dial and authenticate with them, and trains a
two-layer regression network: each round dispatches a data shard to every node, averages the gradients they return,
and broadcasts the pooled gradients back. Cluster exposes each orchestrator process as one method, so a script or
the Streamlit tab can run them one at a time.

Run: python orchestration_example.py
"""

import asyncio
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from polyergalio.generators.data_generators import RandomDatasetGenerator
from polyergalio.models.layers.basal_layers import FullyConnectedLayer
from polyergalio.models.neural_network import NeuralNetwork
from polyergalio.models.training.checkpoint import CollectedModel
from polyergalio.models.training.discovery import NodeRecord, Roster
from polyergalio.models.training.node_launcher import NodeLauncher
from polyergalio.models.training.orchestration import ConnectionHandler, Orchestrator, Task, TrainingData

NUM_SAMPLES = 600
NUM_FEATURES = 3
NUM_NODES = 3
HIDDEN_DIM = 8
ACTIVATION = "tanh"
BATCH_SIZE = 16
LEARNING_RATE = 0.05
NUM_ROUNDS = 40
KEY = "112233"
PHASES = ("discover", "connect", "initialize", "step", "aggregate", "apply", "collect")
STARTED_STAGES = ("listening", "connected", "ready", "stepped", "aggregated")


def build_network(num_features: int, hidden: int, activation: str) -> NeuralNetwork:
    """
    Two-layer regression network.

    Parameters
    ----------
    num_features : width of the input
    hidden : width of the hidden layer
    activation : hidden layer activation name

    Returns
    -------
    NeuralNetwork mapping features to one output
    """
    return NeuralNetwork(
        [
            FullyConnectedLayer(num_features, hidden, activation),
            FullyConnectedLayer(hidden, 1, "linear", is_output=True),
        ],
        name="distributed_regression",
        input_shape=(num_features,),
    )


def hyperparameters(optimizer: str = "SGD", learning_rate: float = LEARNING_RATE) -> dict:
    """Optimizer and loss settings in the form Orchestrator sends to each node."""
    return {
        "optimizer": {"name": optimizer, "params": {"learning_rate": learning_rate}},
        "loss": {"name": "MSELoss"},
    }


def nested_arrays(value) -> list[np.ndarray]:
    """Every array inside nested dicts, lists and tuples, skipping None."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [array for item in value.values() for array in nested_arrays(item)]
    if isinstance(value, (list, tuple)):
        return [array for item in value for array in nested_arrays(item)]
    return [np.asarray(value)]


def payload_kilobytes(value) -> float:
    """Array bytes in a message payload, in kilobytes."""
    return sum(array.nbytes for array in nested_arrays(value)) / 1024


def gradient_norm(gradients: dict) -> float:
    """Euclidean norm over every gradient array."""
    return float(np.sqrt(sum(float(np.sum(np.square(array, dtype=np.float64))) for array in nested_arrays(gradients))))


def weight_vector(network: NeuralNetwork) -> np.ndarray:
    """All of a network's weights as one vector."""
    arrays = nested_arrays([layer.get_weights(for_serialize=True) for layer in network.layers])
    return np.concatenate([array.ravel().astype(np.float64) for array in arrays])


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Cluster:
    """
    Local nodes plus an orchestrator, run on a background event loop and driven one process at a time.

    Stages move listening -> connected -> ready -> stepped -> aggregated -> ready: discover (optional),
    connect, initialize, then repeat dispatch_step, aggregate and apply, and collect_model when done. Each method raises RuntimeError
    when called out of order. Nodes that drop are redialed, or rediscovered if they moved, at the start of
    the next dispatch_step and resynchronized from a healthy node.

    Parameters
    ----------
    num_nodes : node launchers to start
    key : shared key the nodes require
    """

    def __init__(self, num_nodes: int, key: str = KEY):
        self.num_nodes = num_nodes
        self.key = key
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.beacon_ports = {f"node-{i + 1}": free_port() for i in range(num_nodes)}
        self.nodes = [self.launch(node_id) for node_id in self.beacon_ports]
        self.servers = [asyncio.run_coroutine_threadsafe(node.serve(), self.loop) for node in self.nodes]
        self.directory = Path(tempfile.mkdtemp())
        self.roster = Roster(self.directory / "nodes.toml")
        self.orchestrator = Orchestrator(
            None, ConnectionHandler(key.encode(), timeout=5.0), None, {}, roster=self.roster,
            reconnect_attempts=2, reconnect_delay=0.2,
            probe_targets=[f"127.0.0.1:{port}" for port in self.beacon_ports.values()],
            checkpoint_directory=self.directory / "checkpoints",
        )
        self.discovered: list[NodeRecord] = []
        self.collected: Optional[CollectedModel] = None
        self.checkpoint: Optional[Path] = None
        self.stage = "listening"
        self.phase: Optional[str] = None
        self.status = {node.node_id: "listening" for node in self.nodes}
        self.node_info: dict[str, dict] = {node.node_id: {} for node in self.nodes}
        self.events: list[dict] = []
        self.history: list[dict] = []
        self.results: Optional[dict] = None
        self.pooled: Optional[dict] = None
        self.pooled_norm: Optional[float] = None
        self.x_data: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None
        self.execute(asyncio.sleep(0.2))

    @property
    def handler(self) -> ConnectionHandler:
        return self.orchestrator.handler

    @property
    def addresses(self) -> list[tuple[str, int]]:
        return [(node.host, node.port) for node in self.nodes]

    def launch(self, node_id: str) -> NodeLauncher:
        return NodeLauncher(self.key.encode(), node_id, "127.0.0.1", free_port(), beacon_port=self.beacon_ports[node_id])

    def execute(self, coroutine, timeout: float = 120.0):
        """Run a coroutine on the cluster's event loop and wait for its result."""
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    def require(self, *stages: str) -> None:
        if self.stage not in stages:
            raise RuntimeError(f"not available at stage '{self.stage}'")

    def record(self, phase: str, source: str, target: str, message: str, kilobytes: float, started: float) -> None:
        self.events.append(
            {
                "round": len(self.history) + 1,
                "phase": phase,
                "from": source,
                "to": target,
                "message": message,
                "KB": round(kilobytes, 2),
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )

    def connected_nodes(self) -> list[NodeLauncher]:
        return [node for node in self.nodes if node.node_id in self.handler.nodes]

    def wait_for_free_nodes(self, timeout: float = 3.0) -> None:
        deadline = time.perf_counter() + timeout
        while any(node.busy for node in self.nodes) and time.perf_counter() < deadline:
            time.sleep(0.02)

    def discover(self, timeout: float = 1.0, skip_busy: bool = False) -> list[NodeRecord]:
        """
        Probe for nodes over UDP and record the answers in the roster.

        Parameters
        ----------
        timeout : seconds to wait for answers
        skip_busy : leave out nodes already serving an orchestrator

        Returns
        -------
        nodes that answered and were accepted
        """
        self.require(*STARTED_STAGES)
        started = time.perf_counter()
        self.phase = "discover"
        select = (lambda node: not node.busy) if skip_busy else None
        self.discovered = self.execute(self.orchestrator.discover(timeout, select))
        found = {node.node_id for node in self.discovered}
        for node in self.nodes:
            if node.node_id in found:
                self.status[node.node_id] = "discovered"
                self.record("discover", "orchestrator", node.node_id, "probe answered", 0.0, started)
            else:
                self.status[node.node_id] = "listening"
        return self.discovered

    def connect(self, orchestrator_key: Optional[str] = None, node_ids: Optional[list[str]] = None) -> list[str]:
        """
        Dial the discovered nodes, or every node if none were discovered, and authenticate with the orchestrator's key.

        Parameters
        ----------
        orchestrator_key : key the orchestrator signs with, the nodes' own if None
        node_ids : nodes to use; the rest are skipped

        Returns
        -------
        node ids that accepted the connection
        """
        self.require(*STARTED_STAGES)
        started = time.perf_counter()
        self.phase = "connect"
        self.execute(self.handler.stop())
        self.wait_for_free_nodes()
        self.orchestrator.handler = ConnectionHandler((orchestrator_key or self.key).encode(), timeout=5.0)
        known = {record.node_id: record.address for record in self.discovered}
        known = known or {node.node_id: (node.host, node.port) for node in self.nodes}
        chosen = {node_id: address for node_id, address in known.items() if node_ids is None or node_id in node_ids}
        connected = self.execute(self.handler.connect_all(list(chosen.values())))
        self.wait_for_free_nodes()
        for node in self.nodes:
            accepted = node.node_id in connected
            if node.node_id not in chosen:
                self.status[node.node_id] = "skipped"
                continue
            self.status[node.node_id] = "connected" if accepted else "rejected"
            self.record("connect", "orchestrator", node.node_id, "authenticated" if accepted else "rejected: wrong key", 0.0, started)
        self.stage = "connected" if connected else "listening"
        return connected

    def initialize(self, network: NeuralNetwork, settings: dict, x_data: np.ndarray, y_data: np.ndarray,
                   batch_size: int, seed: int = 42) -> None:
        """Send every node the serialized model and hyperparameters, and wait for all to report completed."""
        self.require("connected", "ready", "stepped", "aggregated")
        started = time.perf_counter()
        self.phase = "initialize"
        np.random.seed(seed)
        self.x_data, self.targets = x_data, y_data.reshape(-1, 1)
        self.orchestrator.model, self.orchestrator.hyperparameters = network, settings
        self.orchestrator.training_data = TrainingData(x_data, None, self.targets)
        self.orchestrator.batch_size = batch_size
        self.execute(self.orchestrator.initialize_training())
        kilobytes = payload_kilobytes(network.serialize())
        self.history, self.pooled, self.pooled_norm, self.results = [], None, None, None
        for node in self.connected_nodes():
            self.status[node.node_id] = "initialized"
            self.node_info[node.node_id] = {"sent_kb": kilobytes}
            self.record("initialize", "orchestrator", node.node_id, "model + hyperparameters", kilobytes, started)
            self.record("initialize", node.node_id, "orchestrator", "completed", 0.0, started)
        self.stage = "ready"

    def drop_connection(self, node_id: str) -> None:
        """Cut the orchestrator's connection to one node, as a network failure would; the node keeps running."""
        self.require("ready", "stepped", "aggregated")
        started = time.perf_counter()
        node = next(node for node in self.nodes if node.node_id == node_id)
        self.loop.call_soon_threadsafe(node.active.close)
        self.wait_until(lambda: node_id in self.handler.missing_nodes())
        self.status[node_id] = "dropped"
        self.record("restore", node_id, "orchestrator", "connection dropped", 0.0, started)

    def restart_node(self, node_id: str) -> None:
        """Stop one node and start a fresh one with the same id on a new port, holding no model."""
        self.require("ready", "stepped", "aggregated")
        started = time.perf_counter()
        index = next(index for index, node in enumerate(self.nodes) if node.node_id == node_id)
        old = self.nodes[index]
        if old.active is not None:
            self.loop.call_soon_threadsafe(old.active.close)
        self.servers[index].cancel()
        self.wait_until(lambda: node_id in self.handler.missing_nodes())
        self.nodes[index] = self.launch(node_id)
        self.servers[index] = asyncio.run_coroutine_threadsafe(self.nodes[index].serve(), self.loop)
        self.execute(asyncio.sleep(0.2))
        self.status[node_id] = "restarted"
        self.record("restore", node_id, "orchestrator", f"restarted on port {self.nodes[index].port}", 0.0, started)

    def wait_until(self, condition, timeout: float = 3.0) -> None:
        deadline = time.perf_counter() + timeout
        while not condition() and time.perf_counter() < deadline:
            time.sleep(0.02)

    def restore(self) -> list[str]:
        """Redial or rediscover dropped nodes and resync them; the node ids that came back."""
        started = time.perf_counter()
        missing = self.handler.missing_nodes()
        if not missing:
            return []
        self.execute(self.orchestrator.restore_nodes())
        back = [node_id for node_id in missing if node_id in self.handler.nodes]
        for node_id in back:
            self.status[node_id] = "reconnected"
            self.record("restore", "orchestrator", node_id, "reconnected + resynced", 0.0, started)
        for node_id in set(missing) - set(back):
            self.record("restore", "orchestrator", node_id, "still unreachable", 0.0, started)
        return back

    def dispatch_step(self, batch_size: int) -> None:
        """Restore dropped nodes, send each its own random shard, and collect the gradients and loss that return."""
        self.require("ready")
        self.restore()
        started = time.perf_counter()
        self.phase = "step"
        self.orchestrator.batch_size = batch_size
        node_ids = self.handler.idle_nodes()
        if not node_ids:
            raise RuntimeError("no nodes connected")
        shards = self.orchestrator.batch_data(len(node_ids), batch_size)
        tasks = {node_id: Task("step", shard) for node_id, shard in zip(node_ids, shards)}
        self.results = self.execute(self.orchestrator.collect(tasks))
        if not self.results:
            raise RuntimeError("every node dropped during the step")
        for node_id, shard in zip(node_ids, shards):
            if node_id not in self.results:
                self.status[node_id] = "dropped"
                self.record("step", node_id, "orchestrator", "lost during step", 0.0, started)
                continue
            result = self.results[node_id]
            self.node_info[node_id] = {
                "rows": len(shard["targets"]),
                "loss": result["loss"],
                "gradient_norm": gradient_norm(result["gradients"]),
                "sent_kb": payload_kilobytes(shard),
                "received_kb": payload_kilobytes(result["gradients"]),
            }
            self.status[node_id] = "stepped"
            self.record("step", "orchestrator", node_id, f"shard of {len(shard['targets'])} rows", self.node_info[node_id]["sent_kb"], started)
            self.record("step", node_id, "orchestrator", "gradients + loss", self.node_info[node_id]["received_kb"], started)
        self.stage = "stepped"

    def aggregate(self) -> None:
        """Average the collected gradients."""
        self.require("stepped")
        started = time.perf_counter()
        self.phase = "aggregate"
        self.pooled = self.orchestrator.aggregate_gradients(list(self.results.values()))
        self.pooled_norm = gradient_norm(self.pooled)
        self.record("aggregate", "orchestrator", "orchestrator", f"mean of {len(self.results)} gradients", 0.0, started)
        self.stage = "aggregated"

    def apply(self) -> None:
        """Send the pooled gradients to every node that returned one, which update their local models."""
        self.require("aggregated")
        started = time.perf_counter()
        self.phase = "apply"
        tasks = {node_id: Task("apply_gradients", self.pooled) for node_id in self.results}
        applied = self.execute(self.orchestrator.collect(tasks))
        kilobytes = payload_kilobytes(self.pooled)
        for node_id in self.results:
            if node_id not in applied:
                self.status[node_id] = "dropped"
                self.record("apply", node_id, "orchestrator", "lost during apply", 0.0, started)
                continue
            self.status[node_id] = "updated"
            self.node_info[node_id].update({"sent_kb": kilobytes, "received_kb": 0.0})
            self.record("apply", "orchestrator", node_id, "pooled gradients", kilobytes, started)
            self.record("apply", node_id, "orchestrator", "completed", 0.0, started)
        if not applied:
            raise RuntimeError("every node dropped during apply")
        nodes = [node for node in self.connected_nodes() if node.node_id in applied]
        vectors = [weight_vector(node.model) for node in nodes]
        predictions = nodes[0].model.forward(self.x_data)
        self.history.append(
            {
                "round": len(self.history) + 1,
                "full_loss": float(np.mean((predictions - self.targets) ** 2)),
                "node_loss": {node_id: self.node_info[node_id]["loss"] for node_id in applied},
                "gradient_norm": {node_id: self.node_info[node_id]["gradient_norm"] for node_id in applied},
                "pooled_norm": self.pooled_norm,
                "weight_gap": max(float(np.max(np.abs(vector - vectors[0]))) for vector in vectors),
            }
        )
        self.stage = "ready"

    def collect_model(self) -> CollectedModel:
        """
        Collect the canonical model from the healthy nodes and persist it with a timestamp and a node log.

        Returns
        -------
        collected model, flagged diverged if node weights differ beyond the tolerance
        """
        self.require("ready")
        started = time.perf_counter()
        self.phase = "collect"
        self.collected, self.checkpoint = self.execute(self.orchestrator.save_model(self.orchestrator.checkpoint_directory))
        kilobytes = self.checkpoint.stat().st_size / 1024
        for node_id, gap in self.collected.deviations.items():
            self.status[node_id] = "collected"
            self.node_info[node_id].update({"sent_kb": 0.0, "received_kb": kilobytes})
            self.record("collect", node_id, "orchestrator", f"model, deviation {gap:.1e}", kilobytes, started)
        return self.collected

    def run_pass(self, batch_size: int) -> None:
        """Dispatch, aggregate and apply once."""
        self.dispatch_step(batch_size)
        self.aggregate()
        self.apply()

    def snapshot(self) -> dict:
        """Plain description of the cluster for drawing."""
        return {
            "phase": self.phase,
            "round": len(self.history),
            "pooled_norm": self.pooled_norm,
            "nodes": [{"id": node.node_id, "status": self.status[node.node_id], **self.node_info[node.node_id]} for node in self.nodes],
        }

    def shutdown(self) -> None:
        """Disconnect, stop every node, and end the event loop."""
        self.execute(self.handler.stop())
        for server in self.servers:
            server.cancel()
        self.execute(asyncio.sleep(0.05))
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.stage = "stopped"


def main() -> None:
    x_data, y_data, _ = RandomDatasetGenerator(random_seed=42).generate(
        "regression", num_samples=NUM_SAMPLES, num_features=NUM_FEATURES, noise_scale=0.33, verbose=False
    )
    cluster = Cluster(NUM_NODES)
    try:
        print(f"connected: {cluster.connect()}")
        network = build_network(NUM_FEATURES, HIDDEN_DIM, ACTIVATION)
        print(network.summary())
        cluster.initialize(network, hyperparameters(), x_data, y_data, BATCH_SIZE)
        for _ in range(NUM_ROUNDS):
            cluster.run_pass(BATCH_SIZE)
            entry = cluster.history[-1]
            if entry["round"] % 5 == 0 or entry["round"] == 1:
                print(f"round {entry['round']:3d}  mse {entry['full_loss']:.4f}  weight gap {entry['weight_gap']:.1e}")
        collected = cluster.collect_model()
        print(f"collected from {len(collected.deviations)} nodes, max deviation {collected.max_deviation:.1e}, "
              f"diverged: {collected.diverged}")
        print(f"saved {cluster.checkpoint}")
        print((cluster.checkpoint.parent / "collect.log").read_text())
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()
