"""
Distributed training: an orchestrator dialing two node launchers over localhost, authenticating with a shared
key, and running synchronized training steps on small regression networks.

Every training test replays the same shards through a single-process reference, so the nodes must match it after
each step, not merely agree with each other.
"""

import asyncio
import socket
from contextlib import asynccontextmanager

import numpy as np
import pytest
from polyergalio.models.layers.basal_layers import FullyConnectedLayer
from polyergalio.models.layers.mixture_layers import MixtureOfExperts
from polyergalio.models.model_loss import MSELoss
from polyergalio.models.neural_network import NeuralNetwork
from polyergalio.models.optimizers import SGD
from polyergalio.models.training.node_launcher import NodeLauncher
from polyergalio.models.training.orchestration import ConnectionHandler, Orchestrator, average

KEY = b"112233"
NODE_IDS = ("node-a", "node-b")
LEARNING_RATE = 0.05
SHARD_SIZE = 8
HYPERPARAMETERS = {
    "optimizer": {"name": "SGD", "params": {"learning_rate": LEARNING_RATE}},
    "loss": {"name": "MSELoss"},
}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def build_regression_network() -> NeuralNetwork:
    return NeuralNetwork(
        [
            FullyConnectedLayer(3, 4, "tanh"),
            FullyConnectedLayer(4, 1, "linear", is_output=True),
        ]
    )


def build_moe_network() -> NeuralNetwork:
    return NeuralNetwork(
        [
            MixtureOfExperts(
                input_dim=3,
                upscale_dim=6,
                hidden_dim=3,
                num_shared_experts=1,
                num_routed_experts=4,
                top_k=2,
            ),
            FullyConnectedLayer(3, 1, "linear", is_output=True),
        ]
    )


def flatten(value) -> list[np.ndarray]:
    if isinstance(value, dict):
        return [part for key in sorted(value) for part in flatten(value[key])]
    if isinstance(value, (tuple, list)):
        return [part for item in value for part in flatten(item)]
    return [np.asarray(value, dtype=np.float64).ravel()]


def flat_weights(network: NeuralNetwork) -> np.ndarray:
    return np.concatenate(flatten([layer.get_weights(for_serialize=True) for layer in network.layers]))


def expert_bias(network: NeuralNetwork) -> np.ndarray:
    moe = next(layer for layer in network.layers if isinstance(layer, MixtureOfExperts))
    return moe.gate.expert_bias


def make_shard(x_data: np.ndarray, targets: np.ndarray, start: int, size: int) -> dict:
    rows = slice(start, start + size)
    return {"input_data": {"x_data": x_data[rows]}, "targets": targets[rows]}


def advance_reference(reference: NeuralNetwork, optimizer: SGD, loss: MSELoss, shards: list[dict]) -> None:
    gradients = []
    for shard in shards:
        reference.zero_gradients()
        predictions = reference.forward(**shard["input_data"])
        loss(predictions, shard["targets"])
        reference.backward(loss.backward())
        gradients.append(reference.get_gradients())
    reference.update_weights(average(gradients), optimizer)


@asynccontextmanager
async def running_nodes(key: bytes, node_ids: tuple[str, ...]):
    nodes = [NodeLauncher(key, node_id, "127.0.0.1", free_port()) for node_id in node_ids]
    tasks = [asyncio.create_task(node.serve()) for node in nodes]
    await asyncio.sleep(0.2)
    try:
        yield nodes, [(node.host, node.port) for node in nodes]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def assert_nodes_match(nodes: list[NodeLauncher], reference: NeuralNetwork) -> None:
    expected = flat_weights(reference)
    for node in nodes:
        assert np.allclose(flat_weights(node.model), expected, atol=1e-6)


async def train_in_lockstep(network: NeuralNetwork, passes: list[list[dict]], use_run: bool = False):
    """
    Train live nodes with one orchestrator pass per entry in passes, each entry holding one shard per node.

    Returns the nodes' trained networks and the initial weights.
    """
    initial = flat_weights(network)
    reference = NeuralNetwork.deserialize(network.serialize()).train()
    optimizer, loss = SGD(LEARNING_RATE), MSELoss()
    pending = iter([shard for shards in passes for shard in shards])

    async with running_nodes(KEY, NODE_IDS) as (nodes, addresses):
        handler = ConnectionHandler(KEY)
        orchestrator = Orchestrator(
            network,
            handler,
            addresses,
            HYPERPARAMETERS,
            num_rounds=len(passes),
            batch_size=SHARD_SIZE,
        )
        orchestrator.batch_data = lambda num_shards, batch_size: [next(pending) for _ in range(num_shards)]

        if use_run:
            await orchestrator.run()
            assert handler.nodes == {}
            for shards in passes:
                advance_reference(reference, optimizer, loss, shards)
            assert_nodes_match(nodes, reference)
        else:
            assert set(await handler.connect_all(addresses)) == set(NODE_IDS)
            await orchestrator.initialize_training()
            for node in nodes:
                assert np.array_equal(flat_weights(node.model), initial)

            for shards in passes:
                await orchestrator.run_pass()
                assert handler.idle_nodes() == list(NODE_IDS)
                advance_reference(reference, optimizer, loss, shards)
                assert_nodes_match(nodes, reference)
            await handler.stop()

    return [node.model for node in nodes], initial


@pytest.fixture()
def regression_arrays(regression_dataset):
    x_data, y_data, _ = regression_dataset
    return x_data, y_data.reshape(-1, 1)


def test_wrong_key_is_rejected():
    async def scenario():
        async with running_nodes(KEY, ("node-a",)) as (_, addresses):
            handler = ConnectionHandler(b"445566")
            assert await handler.connect_all(addresses) == []
            assert handler.nodes == {}

    asyncio.run(scenario())


def test_two_synchronized_training_steps_match_single_process(regression_arrays):
    x_data, targets = regression_arrays
    passes = [
        [make_shard(x_data, targets, start, SHARD_SIZE) for start in (0, 8)],
        [make_shard(x_data, targets, start, SHARD_SIZE) for start in (16, 24)],
    ]

    async def scenario():
        return await train_in_lockstep(build_regression_network(), passes)

    models, initial = asyncio.run(scenario())
    assert not np.allclose(flat_weights(models[0]), initial)


def test_run_trains_and_disconnects(regression_arrays):
    x_data, targets = regression_arrays
    passes = [
        [make_shard(x_data, targets, start, SHARD_SIZE) for start in (0, 8)],
        [make_shard(x_data, targets, start, SHARD_SIZE) for start in (16, 24)],
    ]

    async def scenario():
        return await train_in_lockstep(build_regression_network(), passes, use_run=True)

    models, initial = asyncio.run(scenario())
    assert not np.allclose(flat_weights(models[0]), initial)


def test_run_without_reachable_nodes_raises():
    async def scenario():
        handler = ConnectionHandler(KEY, timeout=1.0)
        orchestrator = Orchestrator(
            build_regression_network(), handler, [("127.0.0.1", free_port())], HYPERPARAMETERS
        )
        with pytest.raises(ConnectionError):
            await orchestrator.run()

    asyncio.run(scenario())


def test_uneven_shards_stay_in_sync_with_plain_mean(regression_arrays):
    x_data, targets = regression_arrays
    passes = [
        [make_shard(x_data, targets, 0, 4), make_shard(x_data, targets, 4, 12)],
        [make_shard(x_data, targets, 16, 4), make_shard(x_data, targets, 20, 12)],
    ]

    async def scenario():
        return await train_in_lockstep(build_regression_network(), passes)

    models, initial = asyncio.run(scenario())
    assert not np.allclose(flat_weights(models[0]), initial)


def test_moe_expert_bias_is_pooled_and_stays_in_sync(regression_arrays):
    x_data, targets = regression_arrays
    passes = [
        [make_shard(x_data, targets, start, SHARD_SIZE) for start in (0, 8)],
        [make_shard(x_data, targets, start, SHARD_SIZE) for start in (16, 24)],
    ]

    async def scenario():
        return await train_in_lockstep(build_moe_network(), passes)

    models, initial = asyncio.run(scenario())
    assert np.array_equal(expert_bias(models[0]), expert_bias(models[1]))
    assert np.any(expert_bias(models[0]) != 0.0)
