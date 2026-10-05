import asyncio
import json
import logging
import pickle
import uuid
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from polyergalio.models.training.checkpoint import DEFAULT_TOLERANCE, CollectedModel, collect_weights, persist
from polyergalio.models.training.discovery import BROADCAST, DEFAULT_PORT, NodeRecord, Roster, find_nodes, parse_address
from polyergalio.models.training.transport import receive_frame, receive_message, send_frame, send_message, sign
from polyergalio.models.network import Network

log = logging.getLogger(__name__)

@dataclass
class TrainingData:
    """prepared and preprocessed training data"""
    x_data: NDArray
    x_data_meta: Optional[dict[str, NDArray]]
    target_data: NDArray

    def random_batch_to_message(self, batch_size: int) -> dict:
        idxs = np.random.randint(0, self.x_data.shape[0], batch_size)
        input_data = {"x_data": self.x_data[idxs]}

        if self.x_data_meta:
            for key, data in self.x_data_meta.items():
                input_data.update({key: data[idxs]})

        return {"input_data": input_data, "targets": self.target_data[idxs]}

@dataclass
class Task:
    """A unit of work sent to a node; kind names the operation, payload carries its inputs."""
    kind: str
    payload: Any = None
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_message(self) -> dict:
        return {"task_id": self.task_id, "kind": self.kind, "payload": self.payload}


@dataclass
class NodeSession:
    """A connected, authenticated node and the tasks it has yet to answer."""
    node_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    pending: dict[str, asyncio.Future] = field(default_factory=dict)
    listener: Optional[asyncio.Task] = None

    @property
    def is_idle(self) -> bool:
        return not self.pending


class ConnectionHandler:
    """
    Dials listening nodes, authenticates with them, and routes tasks and results.

    The node issues a JSON challenge and this side signs it with the shared secret; only
    after that frames are read with a restricted unpickler. Run on a trusted network or behind TLS.

    Parameters
    ----------
    secret : shared key used to sign each node's challenge
    timeout : seconds allowed for connecting and for the handshake
    """

    def __init__(self, secret: bytes, timeout: float = 10.0):
        self.secret = secret
        self.timeout = timeout
        self.nodes: dict[str, NodeSession] = {}
        self.addresses: dict[str, tuple[str, int]] = {}
        self.unsynced: set[str] = set()

    async def connect(self, host: str, port: int) -> str:
        """Connect and authenticate to one node; its node_id."""
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), self.timeout)
        try:
            challenge = json.loads(await asyncio.wait_for(receive_frame(reader), self.timeout))
            reply = {"digest": sign(self.secret, challenge["nonce"])}
            await send_frame(writer, json.dumps(reply).encode())
            verdict = json.loads(await asyncio.wait_for(receive_frame(reader), self.timeout))
            if not verdict.get("accepted"):
                raise PermissionError(f"{host}:{port} rejected authentication")
            node_id = str(verdict["node_id"])
            if node_id in self.nodes:
                raise ValueError(f"node {node_id} is already connected")
        except BaseException:
            writer.close()
            raise
        if node_id in self.addresses:
            self.unsynced.add(node_id)
        self.addresses[node_id] = (host, port)
        session = NodeSession(node_id, reader, writer)
        self.nodes[node_id] = session
        session.listener = asyncio.create_task(self.monitor(session))
        log.info("node %s connected at %s:%s", node_id, host, port)
        return node_id

    async def connect_all(self, addresses: list[tuple[str, int]]) -> list[str]:
        """Connect to every address; the node_ids that succeeded."""
        outcomes = await asyncio.gather(*(self.connect(*address) for address in addresses), return_exceptions=True)
        connected = []
        for address, outcome in zip(addresses, outcomes):
            if isinstance(outcome, BaseException):
                log.warning("could not connect to %s: %r", address, outcome)
            else:
                connected.append(outcome)
        return connected

    def missing_nodes(self) -> list[str]:
        """Nodes that were connected and have since dropped."""
        return [node_id for node_id in self.addresses if node_id not in self.nodes]

    async def reconnect(self, node_id: str, attempts: int = 3, delay: float = 1.0) -> bool:
        """Redial a dropped node at its last known address; whether it is connected again."""
        for attempt in range(attempts):
            try:
                host, port = self.addresses[node_id]
                if await self.connect(host, port) == node_id:
                    return True
            except (OSError, asyncio.TimeoutError, PermissionError, ValueError) as error:
                log.warning("reconnect to %s failed: %r", node_id, error)
            await asyncio.sleep(delay)
        return False

    async def stop(self) -> None:
        for session in list(self.nodes.values()):
            session.listener.cancel()
            session.writer.close()
            self.remove_node(session)
        self.addresses.clear()
        self.unsynced.clear()

    async def monitor(self, session: NodeSession) -> None:
        try:
            await self.listen(session)
        except (asyncio.IncompleteReadError, ConnectionError, ValueError, pickle.UnpicklingError):
            pass
        finally:
            self.remove_node(session)
            session.writer.close()

    async def listen(self, session: NodeSession) -> None:
        """Resolve pending task futures as the node returns results."""
        while True:
            message = await receive_message(session.reader)
            future = session.pending.pop(message["task_id"], None)
            if future is None or future.done():
                continue
            if message.get("error"):
                future.set_exception(RuntimeError(message["error"]))
            else:
                future.set_result(message["result"])

    def remove_node(self, session: NodeSession) -> None:
        self.nodes.pop(session.node_id, None)
        for future in session.pending.values():
            if not future.done():
                future.set_exception(ConnectionError(f"node {session.node_id} disconnected"))
        session.pending.clear()

    def idle_nodes(self) -> list[str]:
        return [node_id for node_id, session in self.nodes.items() if session.is_idle]

    async def assign_task(self, node_id: str, task: Task) -> asyncio.Future:
        """Send a task to a node; the returned future resolves with the node's result."""
        session = self.nodes.get(node_id)
        if session is None:
            raise ConnectionError(f"node {node_id} is not connected")
        future = asyncio.get_running_loop().create_future()
        session.pending[task.task_id] = future
        await send_message(session.writer, task.to_message())
        return future


def average(values: list):
    """Elementwise mean of arrays, or of nested dicts of arrays sharing the same keys."""
    first = values[0]
    if isinstance(first, dict):
        return {key: average([value[key] for value in values]) for key in first}
    if first is None:
        return None
    return sum(values) / len(values)


class Orchestrator:
    """
    Control plane: shards data across nodes, pools their gradients, and sends the pool back.

    Nodes keep the model. Training is initialize_training once, then repeated
    step (nodes return gradients), average, and apply_gradients (nodes update).

    Parameters
    ----------
    model : model each node instantiates and keeps
    handler : connection handler that owns the nodes
    node_addresses : (host, port) or "host[:port]" of each listening node; None reads the roster
    hyperparameters : {"optimizer": {"name", "params"}, "loss": {"name", "params"}} sent to each node
    data : training data to shard across nodes. Tokenized or preprocessed, ready for the forward pass and loss calc.
        List of already processed dicts
        Requires nested dicts for x data, since a foward pass might make use of masks:
            {"input_data":
                {"x_data": [array], **kwargs "attention_mask: [array]}},
            "targets": [array] }
    num_rounds : steps to run
    batch_size : samples per shard
    roster : roster file updated by discover and read when node_addresses is None
    reconnect_attempts : tries to redial a dropped node before rediscovering it
    reconnect_delay : seconds between tries
    beacon_port : UDP port discovery probes are sent to
    checkpoint_directory : when set, run() collects and saves the trained model here on completion
    tolerance : weight deviation between nodes above which the collected model is flagged as diverged
    probe_targets : addresses to probe, as "host" or "host:beacon_port"; None broadcasts and probes roster hosts

    A node that drops mid-run is redialed, then rediscovered by node_id if its address changed,
    and resynchronized from a healthy node before the next pass. Until then the pass continues
    on the remaining nodes.
    """

    def __init__(self, model: Network,
                 handler: ConnectionHandler,
                 node_addresses: Optional[list[Union[str, tuple[str, int]]]],
                 hyperparameters: dict,
                 training_data: TrainingData = None,
                 num_rounds: int = 1,
                 batch_size: int = 1,
                 roster: Optional[Roster] = None,
                 reconnect_attempts: int = 3,
                 reconnect_delay: float = 1.0,
                 beacon_port: int = DEFAULT_PORT,
                 probe_targets: Optional[list[str]] = None,
                 checkpoint_directory: Optional[Union[str, Path]] = None,
                 tolerance: float = DEFAULT_TOLERANCE):
        self.model = model
        self.handler = handler
        self.node_addresses = node_addresses
        self.hyperparameters = hyperparameters
        self.training_data = training_data
        self.num_rounds = num_rounds
        self.batch_size = batch_size
        self.roster = roster or Roster()
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.beacon_port = beacon_port
        self.probe_targets = probe_targets
        self.checkpoint_directory = checkpoint_directory
        self.tolerance = tolerance

    async def discover(self, timeout: float = 2.0,
                       select: Optional[Callable[[NodeRecord], bool]] = None) -> list[NodeRecord]:
        """
        Probe the network, record the nodes that answer in the roster, and return the ones select accepts.

        Parameters
        ----------
        timeout : seconds to listen for answers
        select : predicate on NodeRecord, such as ``lambda node: not node.busy``; all if None

        Returns
        -------
        live nodes; use ``[node.address for node in nodes]`` as node_addresses
        """
        targets = self.probe_targets
        if targets is None:
            targets = [BROADCAST, *sorted({record.host for record in self.roster.load()})]
        found = await find_nodes(timeout, self.beacon_port, targets)
        self.roster.update(found)
        return [node for node in found if select is None or select(node)]

    async def restore_nodes(self) -> None:
        """Bring dropped nodes back, rediscovering any that moved, and resync them."""
        for node_id in self.handler.missing_nodes():
            if await self.handler.reconnect(node_id, self.reconnect_attempts, self.reconnect_delay):
                continue
            for node in await self.discover(select=lambda node: node.node_id == node_id):
                self.handler.addresses[node_id] = node.address
                await self.handler.reconnect(node_id, attempts=1, delay=0)
        await self.resync_nodes()

    async def resync_nodes(self) -> None:
        """Copy model and optimizer state from a node that never dropped onto reconnected ones."""
        stale = self.handler.unsynced & set(self.handler.nodes)
        if not stale:
            return
        sources = [node_id for node_id in self.handler.idle_nodes() if node_id not in stale]
        if not sources:
            raise ConnectionError("no synchronized node left to resync from")
        state = (await self.collect({sources[0]: Task("export_state")}))[sources[0]]
        payload = {**self.hyperparameters, **state}
        await self.run_tasks({node_id: Task("initialize_training", payload) for node_id in stale})
        self.handler.unsynced -= stale
        log.info("resynchronized %s from %s", sorted(stale), sources[0])

    async def collect_model(self) -> CollectedModel:
        """
        Gather the serialized model from every healthy node and make the first of them canonical.

        Nodes that dropped and have not been resynced are left out. The canonical model replaces self.model,
        and the result is flagged diverged if any node's weights differ from it by more than the tolerance.
        """
        healthy = [node_id for node_id in self.handler.idle_nodes() if node_id not in self.handler.unsynced]
        states = await self.collect({node_id: Task("export_model") for node_id in healthy})
        if not states:
            raise ConnectionError("no healthy node to collect weights from")
        collected = collect_weights({node_id: state["model"] for node_id, state in states.items()},
                                    type(self.model), self.tolerance)
        self.model = collected.model
        return collected

    async def save_model(self, directory: Union[str, Path]) -> tuple[CollectedModel, Path]:
        """Collect the canonical model, serialize it to a timestamped file in directory, and log the nodes."""
        collected = await self.collect_model()
        return collected, persist(collected, directory, dict(self.handler.addresses))

    async def run(self) -> None:
        try:
            addresses = self.roster.addresses() if self.node_addresses is None else self.node_addresses
            if not await self.handler.connect_all([parse_address(address) for address in addresses]):
                raise ConnectionError("no nodes reachable")
            await self.initialize_training()
            # run through num_rounds of training ---
            for round_index in range(self.num_rounds):
                log.info("round %s", round_index)
                # run_pass does a training step; a forward / backward pass and aggregated GRADIENTS from each source.
                # Then broadcasts the aggregation back out to each processor
                await self.run_pass()
            if self.checkpoint_directory is not None:
                await self.save_model(self.checkpoint_directory)
        finally:
            await self.handler.stop()

    async def run_tasks(self, tasks: dict[str, Task]) -> list:
        """Send each node its task and wait for every result."""
        futures = [await self.handler.assign_task(node_id, task) for node_id, task in tasks.items()]
        return await asyncio.gather(*futures)

    async def collect(self, tasks: dict[str, Task]) -> dict[str, Any]:
        """Like run_tasks, but nodes that drop meanwhile are left out; results keyed by node_id."""

        async def one_task(node_id: str, task: Task):
            return await (await self.handler.assign_task(node_id, task))

        outcomes = await asyncio.gather(*(one_task(*item) for item in tasks.items()), return_exceptions=True)
        results = {}
        for node_id, outcome in zip(tasks, outcomes):
            if isinstance(outcome, ConnectionError):
                log.warning("lost %s: %s", node_id, outcome)
            elif isinstance(outcome, BaseException):
                raise outcome
            else:
                results[node_id] = outcome
        return results

    async def initialize_training(self) -> None:
        """Give every node the model and hyperparameters, and wait until all report completed."""
        payload = {"model": self.model.serialize(), **self.hyperparameters}
        results = await self.run_tasks(
            { node_id: Task("initialize_training", payload) for node_id in self.handler.idle_nodes() })
        if not all(result["completed"] for result in results):
            raise RuntimeError("a node failed to initialize training")

    async def run_pass(self) -> None:
        """One synchronized step: restore dropped nodes, shard, collect gradients, average, and apply everywhere."""
        await self.restore_nodes()
        node_ids = self.handler.idle_nodes()
        if not node_ids:
            raise ConnectionError("no nodes connected")

        sharded_data = self.batch_data(len(node_ids), self.batch_size)

        results = await self.collect({node_id: Task("step", shard) for node_id, shard in zip(node_ids, sharded_data)})
        if not results:
            raise ConnectionError("every node dropped during the step")

        pooled = self.aggregate_gradients(list(results.values()))

        await self.collect({node_id: Task("apply_gradients", pooled) for node_id in results})

    def batch_data(self, num_shards: int, batch_size: Optional[int]) -> list[dict]:
        """Split data into one shard per node, each a structured dict for dipatch"""
        return [self.training_data.random_batch_to_message(batch_size) for _ in range(num_shards)]


    def aggregate_gradients(self, results: list[dict]) -> dict:
        """Average the per-node gradients."""
        log.info("")
        return average([result["gradients"] for result in results])
