import asyncio
import hmac
import json
import logging
import os
import pickle
import secrets
import socket
from typing import Callable, Optional

import numpy as np

from polyergalio.models.training.discovery import DEFAULT_PORT, announce
from polyergalio.models.training.orchestration import Task
from polyergalio.models.training.transport import receive_frame, receive_message, send_frame, send_message, sign
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.model_loss import (
    CosineLoss, CrossEntropyLoss, DifferenceLoss, Loss, MAELoss, MSELoss, RMSELoss, SSELoss,
)
from polyergalio.models.network import Network
from polyergalio.models.optimizers import Adam, Optimizer, SGD

log = logging.getLogger(__name__)

OPTIMIZERS = {"SGD": SGD, "Adam": Adam}
LOSSES = {
    loss.__name__: loss
    for loss in (CosineLoss, CrossEntropyLoss, DifferenceLoss, MAELoss, MSELoss, RMSELoss, SSELoss)
}


class NodeLauncher:
    """
    Worker process: listens for an orchestrator, authenticates it, and executes its tasks.

    Serves one orchestrator at a time. Each task kind maps to a handler taking the task
    payload and returning a picklable result; handler errors are sent back to the orchestrator.

    Parameters
    ----------
    secret : shared key the orchestrator must sign the challenge with
    node_id : name reported to the orchestrator, defaults to the hostname
    host : interface to listen on
    port : TCP port to listen on
    timeout : seconds the orchestrator has to complete the handshake
    beacon_port : UDP port to answer discovery probes on, None to stay undiscoverable

    A newly authenticated orchestrator replaces the current one, so a dropped
    connection never locks the node out.
    """

    def __init__(self, secret: bytes, node_id: Optional[str] = None, host: str = "0.0.0.0",
                 port: int = DEFAULT_PORT, timeout: float = 10.0,
                 beacon_port: Optional[int] = DEFAULT_PORT):
        self.secret = secret
        self.node_id = node_id or socket.gethostname()
        self.host = host
        self.port = port
        self.timeout = timeout
        self.beacon_port = beacon_port
        self.handlers: dict[str, Callable] = {
            "ping": lambda payload: payload,
            "initialize_training": self.initialize_training,
            "export_model": self.export_model,
            "export_state": self.export_state,
            "step": self.step,
            "apply_gradients": self.apply_gradients,
        }
        self.active: Optional[asyncio.StreamWriter] = None
        self.model: Optional[Network] = None
        self.optimizer: Optional[Optimizer] = None
        self.loss_fn: Optional[Loss] = None

    @property
    def busy(self) -> bool:
        return self.active is not None

    def register(self, kind: str, handler: Callable) -> None:
        """Handle tasks of this kind with handler(payload)."""
        self.handlers[kind] = handler

    def initialize_training(self, payload: dict) -> dict:
        """Keep the serialized model, optimizer and loss described by the payload for the run."""
        self.model = Network.deserialize(payload["model"]).train()
        self.optimizer = OPTIMIZERS[payload["optimizer"]["name"]](**payload["optimizer"].get("params", {}))
        loss_params = dict(payload["loss"].get("params", {}))
        if "task" in loss_params:
            loss_params["task"] = ClassificationTask[loss_params["task"]]
        self.loss_fn = LOSSES[payload["loss"]["name"]](**loss_params)
        if "optimizer_state" in payload:
            self.optimizer.set_state(payload["optimizer_state"])
        return {"completed": True}

    def export_model(self, payload) -> dict:
        """The serialized model, for collecting trained weights."""
        return {"model": self.model.serialize()}

    def export_state(self, payload) -> dict:
        """Model and optimizer state, for resynchronizing a node that dropped out."""
        return {"model": self.model.serialize(), "optimizer_state": self.optimizer.get_state()}

    def step(self, payload: dict) -> dict:
        """Forward and backward on one shard; the local gradients and loss."""
        self.model.zero_gradients()
        predictions = self.model.forward(**payload["input_data"])
        loss = self.loss_fn(predictions, payload["targets"])

        self.model.backward(self.loss_fn.backward())

        return {"gradients": self.model.get_gradients(), "loss": float(np.mean(loss))}

    def apply_gradients(self, payload: dict) -> dict:
        """Update the local model with gradients pooled across nodes."""
        self.optimizer.step(self.model, payload)
        return {"completed": True}

    async def serve(self) -> None:
        server = await asyncio.start_server(self.handle_connection, self.host, self.port)
        log.info("node %s listening on %s:%s", self.node_id, self.host, self.port)
        beacon = await self.start_beacon()
        try:
            async with server:
                await server.serve_forever()
        finally:
            if beacon:
                beacon.close()

    async def start_beacon(self):
        """Answer discovery probes; a busy beacon port only costs discoverability."""
        if self.beacon_port is None:
            return None
        try:
            return await announce(self.node_id, self.port, lambda: self.busy, self.beacon_port)
        except OSError as error:
            log.warning("node %s is not discoverable: %s", self.node_id, error)
            return None

    async def authenticate(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
        nonce = secrets.token_hex(16)
        await send_frame(writer, json.dumps({"nonce": nonce}).encode())
        reply = json.loads(await asyncio.wait_for(receive_frame(reader), self.timeout))
        accepted = hmac.compare_digest(str(reply.get("digest", "")), sign(self.secret, nonce))
        await send_frame(writer, json.dumps({"accepted": accepted, "node_id": self.node_id}).encode())
        return accepted

    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if not await self.authenticate(reader, writer):
                log.warning("rejected %s", writer.get_extra_info("peername"))
                writer.close()
                return
            if self.active is not None:
                self.active.close()
            self.active = writer
            log.info("orchestrator connected")
            while True:
                task = Task(**await receive_message(reader))
                await send_message(writer, await self.execute(task))
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError, ValueError, TypeError,
                pickle.UnpicklingError):
            pass
        finally:
            if self.active is writer:
                self.active = None
            writer.close()
            log.info("orchestrator disconnected")

    async def execute(self, task: Task) -> dict:
        """Run the task's handler off the event loop; the reply message."""
        try:
            handler = self.handlers[task.kind]
            result = await asyncio.get_running_loop().run_in_executor(None, handler, task.payload)
            return {"task_id": task.task_id, "result": result}
        except Exception as error:
            return {"task_id": task.task_id, "error": repr(error)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    launcher = NodeLauncher(secret=os.environ["POLYERGALIO_SECRET"].encode())
    asyncio.run(launcher.serve())
