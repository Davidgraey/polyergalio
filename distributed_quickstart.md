# Distributed training quickstart
### check the streamlit app in app/streamlit_app.py for some more details and diagrams.

One of the core tennants of Polyergalio is to build, train, and predict with models transparently, and at a wide-system level compatability. 
Avoiding large packages, minimizing dependnecies, and supporting small, task-focused models.
One of the goals, early on, was to support a multi-device distributed training mechanic. Use your old machines, your edge devices, anything that you can run python, numpy, and open a port on can be used to help train.


One or more **nodes** hold a copy of the model and compute gradients on their own shard of the data. One **orchestrator** finds the nodes, connects to them, averages their gradients, and sends the pool back so every node applies the same update. These are the same steps the Streamlit tab ("Neural networks" -> "Distributed training") walks through with buttons.

```
orchestrator                              node (one per machine)
  discover        -- UDP probe            answers with its id and port
  connect         -- TCP + HMAC           authenticates the shared secret
  initialize      -- model + settings     builds the model, optimizer, loss
  step            -- data shard           forward, backward, returns gradients
  aggregate       (mean of the gradients)
  apply           -- pooled gradients     every node updates identically
  collect         <- serialized model     canonical model saved, deviation checked
```

The default port is **31352** for both protocols. TCP (training) and UDP (discovery).
Override it with `POLYERGALIO_PORT` on both sides.

## 1. Start a node

On each worker machine, install the package (same version as the orchestrator), set the shared secret, and run the launcher:

```bash
pip install polyergalio
export POLYERGALIO_SECRET="PLACEHOLDER SHARED SECRET"
python -m polyergalio.models.training.node_launcher
```

It logs `node <hostname> listening on 0.0.0.0:31352` and waits. The node id is the machine's hostname. 
You may need to open **TCP and UDP 31352** inbound in the machine's firewall

A node serves one orchestrator at a time; a newly authenticated orchestrator replaces the previous one, so a dropped connection shouldn't stop / disable a node from contributing

## 2. Build the model, data and settings

On the orchestrator machine: create the model - Network, Layers, and set up the data and training task.

```python
import numpy as np
from polyergalio.models.layers.basal_layers import FullyConnectedLayer
from polyergalio.models.neural_network import NeuralNetwork
from polyergalio.models.training.orchestration import ConnectionHandler, Orchestrator, Task, TrainingData

SECRET = b"change-me"

network = NeuralNetwork(
    [FullyConnectedLayer(3, 8, "tanh"), FullyConnectedLayer(8, 1, "linear", is_output=True)],
    input_shape=(3,),
)

rng = np.random.default_rng(0)
x = rng.normal(size=(600, 3))
y = (x @ np.array([1.0, -2.0, 0.5])).reshape(-1, 1)
data = TrainingData(x_data=x, x_data_meta=None, target_data=y)

settings = {
    "optimizer": {"name": "Adam", "params": {"learning_rate": 0.01}},
    "loss": {"name": "MSELoss"},
}
```

`x_data_meta` carries extra forward arguments per row (eg an attention mask) as `{name: array}`.
(confirm this with the package version you have installed, there may be new unlisted options)
Optimizers available on nodes: `SGD`, `Adam`.
Losses: `CosineLoss`, `CrossEntropyLoss`, `DifferenceLoss`, `MAELoss`, `MSELoss`, `RMSELoss`, `SSELoss`.

## 3. Build the orchestrator

```python
orchestrator = Orchestrator(
    network,
    ConnectionHandler(SECRET),
    None,
    settings,
    data,
    num_rounds=50,
    batch_size=16,
    checkpoint_directory="checkpoints",
)
handler = orchestrator.handler
```

`None` for the addresses means "use the existing roster file" (`nodes.toml` in the working directory, or `POLYERGALIO_ROSTER`).
You can instead pass `["system_name", "10.0.0.7:31352"]`.
`checkpoint_directory` makes `run()` save the trained model when it finishes.

## 4. Step by step

Every step below is one `await`, so run them inside an `async def main()` and `asyncio.run(main())`.

### Discover

```python
found = await orchestrator.discover(timeout=2.0, select=lambda node: not node.busy)
orchestrator.node_addresses = [node.address for node in found]
```

This broadcasts a UDP probe (and probes every host already in the roster), writes whoever answers into `nodes.toml`, and returns the `NodeRecord`s that pass `select`.
The roster is the source of truth. you can also write it by hand as a toml.

```toml
[[node]]
node_id = "mini-box"
host = "10.0.0.7"
port = 31352
```

Broadcast does not cross subnets or VPNs without multicast; list those hosts in the roster and they are probed directly.

### Connect and authenticate

```python
connected = await handler.connect_all(orchestrator.node_addresses)
print(connected)
```

Each node sends a challenge; the orchestrator answers with an HMAC of it under the shared secret. A wrong secret is rejected and that node is left out of `connected`.

### Initialize training

```python
await orchestrator.initialize_training()
```

Every connected node receives the serialized model and the optimizer and loss settings, builds them, and reports `completed`.

### Step, aggregate, apply

```python
await orchestrator.restore_nodes()
node_ids = handler.idle_nodes()

shards = orchestrator.batch_data(len(node_ids), batch_size=16)
results = await orchestrator.collect({n: Task("step", s) for n, s in zip(node_ids, shards)})

pooled = orchestrator.aggregate_gradients(list(results.values()))

await orchestrator.collect({n: Task("apply_gradients", pooled) for n in results})
```

- `step` returns each node's gradients and loss on its own random shard,
- `aggregate_gradients` takes their mean, 
- `apply_gradients` sends the pool back so all nodes take the identical optimizer step. 
- `collect` tolerates nodes that drop mid-call and leaves them out of the result. 
- All three can be called individually, or wrapped as a single training step ('run pass')

```python
for _ in range(50):
    await orchestrator.run_pass()
```

### Collect and save

```python
collected, path = await orchestrator.save_model("checkpoints")
print(collected.diverged, collected.max_deviation, path)
```

Healthy nodes each return their serialized model. The lowest node id becomes the canonical one, `orchestrator.model` is replaced by it, and `collected.diverged` is `True` if any node's weights differ from it by more than `1e-4` (`Orchestrator(tolerance=...)`). The model is written to `checkpoints/model-<UTC timestamp>.pkl`, and a line is appended to `checkpoints/collect.log`:

```
2026-09-30T17:58:45+00:00 collected from 3 of 3 nodes: mini-box@10.0.0.7:31352 ... canonical=gpu-box max_deviation=0 diverged=False file=model-20260930T175845112127Z.pkl
```

Load it later with `NeuralNetwork.deserialize("checkpoints/model-....pkl")`.

### Disconnect

```python
await handler.stop()
```

## 5. The short version

```python
async def main():
    await orchestrator.discover(timeout=2.0)
    await orchestrator.run()

asyncio.run(main())
```

`run()` connects to the roster's nodes, initializes, runs `num_rounds` passes, saves to `checkpoint_directory`, and disconnects.

## When a node drops

Before each pass, `run_pass` redials any node that dropped. If its address changed, the orchestrator probes for it by node id and redials at the new address. A returning node is resynchronized (model and optimizer state copied from a node that never dropped) before it takes part again. The pass runs on the remaining nodes meanwhile, and collection leaves out nodes that are not healthy, marking them `(unavailable)` in the log. If no synchronized node is left to copy from, the run raises `ConnectionError`.

## Security

The shared secret authenticates the orchestrator to each node, but traffic is not encrypted.
Payloads after authentication are pickles read through a restricted unpickler (containers, numbers and NumPy arrays only).
Run on a trusted network or behind a VPN or TLS tunnel. It is NOT robust or comprehensive

## Try it locally first

`python examples/orchestration_example.py` starts three nodes on localhost and trains a small regression network; `streamlit run app/streamlit_app.py` shows the same steps with buttons, failure injection and a diagram.
