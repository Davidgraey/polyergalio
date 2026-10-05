"""
A container for wiring estimators into a network.

Layers know how to transform an array and how to push a gradient back through
themselves. What they do not know is what feeds them. This module owns that.

Connections are object references, not names. `connect` returns the node it
created, and you pass that node in as the source of the next one, so an edge is
a pointer from consumer to producer rather than a string resolved later. Three
things follow from that:

  * a mistyped source is a NameError where you wrote it, not a wrong edge
  * a cycle cannot be built, since a node can only reference nodes that already
    exist, so there is no forward reference to close a loop with
  * insertion order is therefore already a topological order, and the forward
    pass is a walk down the list

Every node holds its own estimator, and the node name is the key everything
else uses: gradients are `gradient_<node name>`, saved nodes are keyed by name.

Training mode is one operation at every level. `train()` and `eval()` on the
network cascade to its nodes, a node passes them to its layer, and a layer
passes them to its own sublayers.
"""
from __future__ import annotations

import inspect
import time
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from polyergalio.base_model import BasalEstimator
from polyergalio.models.constants import ANY_SHAPE
from polyergalio.utilities import shape_conflict
from polyergalio.composite_model import Composite, CompositeNode

INPUT_NAME = "input"
GRADIENT_PREFIX = "gradient_"


def forward_options(layer: BasalEstimator) -> tuple[frozenset[str], bool]:
    """Names of the optional forward parameters, and whether forward accepts **kwargs."""
    parameters = inspect.signature(layer.forward).parameters
    optional = frozenset(
        name for name, parameter in parameters.items() if parameter.default is not inspect.Parameter.empty
    )
    return optional, any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values())


class Node(CompositeNode):
    """
    One vertex in the graph: an estimator, and references (edges) to the nodes feeding it.

    Nodes are compared and hashed by identity, so the same node passed to two consumers is
    one shared producer, which is what makes a fan-out visible in the code that builds it.

    Shapes, logic, and processes are held in the estimator that the Node wraps.

        Network
            Node(Layer) -- Node(Layer) --> Node(Layer)

    A layer node's `in_shape` is the layer's declared input shapes, one per source. Its
    `out_shape` is what it produces for downstream edges, resolved from the sources it was
    wired to. That is what the next node's check is made against, so an array width flows
    down the graph as it is wired and checked on connect.
    """

    def __init__(
        self,
        name: str,
        layer: Optional[BasalEstimator] = None,
        sources: tuple = (),
        shape: tuple = ANY_SHAPE,
    ):
        """
        Parameters
        ----------
        name : label, and the key the network stores this node's gradients under
        layer : the estimator to run, None for the graph's input node
        sources : nodes feeding this one, in the order the layer's forward takes them
        shape : trailing axes of the data entering the graph, for the input node only
        """
        super().__init__(name, layer, sources)

        if layer is None:
            self.in_shape = shape
            self.out_shape = shape
            self.forward_kwargs: frozenset[str] = frozenset()
            self.accepts_any_kwarg = False
        else:
            self.in_shape = layer.shapes["input"]
            self.forward_kwargs, self.accepts_any_kwarg = forward_options(layer)
            self.resolve_shapes()

        for source in sources:
            source.consumers.append(self)

    def resolve_shapes(self) -> None:
        """
        Work out the shape this node produces from what its sources produce
        now; a node with no sources yields its layer's declared output.
        """
        incoming = tuple(source.out_shape for source in self.sources)
        resolved = self.layer.infer_output_shapes(incoming) if incoming else self.layer.shapes["output"]
        if len(resolved) != 1:
            raise ValueError(
                f"{self.layer.__class__.__name__}.infer_output_shapes returned "
                f"{len(resolved)} shapes for one node. A node holds one "
                "value, so it must yield one shape"
            )
        self.out_shape = resolved[0]

    @property
    def layer(self) -> Optional[BasalEstimator]:
        return self.component

    @property
    def shapes(self) -> dict[str, tuple]:
        return {"input": self.in_shape, "output": self.out_shape}

    @property
    def training(self) -> bool:
        """The layer's mode; the input node has no layer and reports True."""
        return True if self.layer is None else self.layer.training

    def train(self, mode: bool = True) -> Node:
        """Pass the mode to the layer, which passes it to its sublayers."""
        if self.layer is not None:
            self.layer.train(mode)
        return self

    def eval(self) -> Node:
        return self.train(False)

    def __repr__(self):
        if self.is_source:
            return f"Node({self.name}, graph input)"
        feeding = ", ".join(source.name for source in self.sources)
        return f"Node({self.name} <- {feeding})"

    def __str__(self):
        return f"{self.__repr__()} producing {self.out_shape}"


class Network(Composite):
    """
    A directed acyclic graph of layer connections D(AG)

    connect the DAG by passing nodes:

        net = Network()
        audio = net.input
        # fanning out to multiple outputs
        amp = net.connect(amplitude_fc, audio)
        freq = net.connect(frequency_fft, audio)
        merged = net.connect(LatentStack(), amp, freq)
        net.output = net.connect(head, merged)

    or sequentially, when there is nothing to branch::

        net = Network([layer_a, layer_b, layer_c])

    Each node owns its layer: connecting a layer that already runs at another node is an error.

    Gradients and weights are keyed by node name:

        gradients = net.get_gradients()      # {"gradient_<node name>": {...}}
        net -= gradients                     # or net.update_weights(**gradients)

    Swapping a trained network's head means connecting a new node to an
    already-trained node and retargeting the output -- not editing an
    existing edge. Node sources are fixed at construction, which is what
    keeps the graph acyclic and insertion order a valid topological order;
    nothing rewires a node once it exists.

        new_head = net.connect(NewHead(...), net.node("encoder_out"))
        net.output = new_head

    The old head is now unreachable from `net.output` and stops training
    (`backward()` only walks nodes reachable from the output), though it
    still runs forward every pass until pruned; `net.validate()` flags it.

    To freeze the encoder while fine-tuning only the new head, control
    which layers the optimizer touches: `optimizer.step()` takes whatever
    layer list you hand it, so `optimizer.step([new_head_layer])` instead
    of `optimizer.step(net.layers)` trains only the head. `backward()`
    still has to flow gradient through the encoder's layers to reach it
    (that's unavoidable), but nothing forces you to APPLY those encoder
    gradients.
    """

    component_family = BasalEstimator

    def __init__(
        self,
        layers: Optional[Iterable[BasalEstimator]] = None,
        name: Optional[str] = None,
        input_shape: tuple = ANY_SHAPE,
    ):
        """
        Parameters
        ----------
        layers : optional layers to chain end to end, for a graph with nothing
            to branch
        name : label for summaries
        input_shape : trailing axes of the data the graph will be fed, with
            None for any axis that varies. Given, the first edge is checked
            like every other one; left out, the first layer is taken on trust
            until data arrives.
        """
        super().__init__(name)
        self._output = None
        self.training = True
        self.activations = {}
        self._timings = {}

        self._input = Node(INPUT_NAME, shape=tuple(input_shape))
        self._input_shape = tuple(input_shape)
        self._nodes.append(self._input)

        if layers is not None:
            self.extend(layers)

    # ------------- connecting
    @property
    def input(self) -> Node:
        """the graph's source node; pass it as an input source to the first layer"""
        return self._input

    def connect(
        self, layer: BasalEstimator, *sources: Node, name: Optional[str] = None, strict: bool = True
    ) -> Node:
        """
        Place a layer in the graph, fed by the given nodes, and return its node.

        Parameters
        ----------
        layer : the layer to run at this node; it may not already run at another node
        sources : the nodes whose outputs feed it, in the order the layer's
            forward takes them. Passing one node to two different calls is how
            a fan-out is expressed.
        name : optional label. Defaults to the layer's class name with a
            counter, and is the key its gradients are stored under.
        strict : False places the layer without requiring its inputs, as an
            unwired node that reconnect() can feed later; an unwired node is
            skipped by forward.

        Returns
        -------
        the new node, to pass as a source to whatever comes next
        """
        return super().connect(layer, *sources, name=name, strict=strict)

    def reconnect(
        self,
        node: Node | str,
        upstream_edge: Node | str | Sequence[Node | str] | None = None,
        downstream_edge: Node | str | None = None,
    ) -> Node:
        """
        Place a node already in the graph between an upstream and a downstream
        node, with the checks connect() makes plus two that only apply once a
        graph exists: no cycle may form, and every node downstream is
        re-checked against the shapes it now receives. Nothing changes if a
        check fails. Execution order is recomputed.

            net.reconnect(net.node("head"), upstream_edge=net.node("encoder"), downstream_edge=net.node("loss_head"))

        Parameters
        ----------
        node : the node, or its label
        upstream_edge : the node (or nodes, in the order the layer's forward
            takes them) that feeds it from now on; None keeps its inputs
        downstream_edge : a node that should read this one. It takes this node
            in place of the upstream node it read, else as an added input;
            None leaves the readers as they are
        """
        return super().reconnect(node, upstream_edge, downstream_edge)

    def disconnect(self, node: Node | str, *sources: Node | str) -> Node:
        """
        Break connections. With sources, removes just those from the node's
        inputs. With none, cuts the node off completely: all its inputs, and
        the node as a source of every node that reads it.

        A node left with fewer sources than its layer needs is unwired:
        forward skips it and everything that depends on it, and validate()
        reports it. Reconnect it, delete() it, or let prune() drop it once
        the output no longer depends on it.

        Parameters
        ----------
        node : the node, or its label
        sources : nodes (or labels) to remove as sources, none for all
        """
        return super().disconnect(node, *sources)

    def delete(self, node: Node | str) -> BasalEstimator:
        """
        Remove a node and its layer from the network, after disconnect().

        Halts, changing nothing, while the node still has inputs or is read
        by another node, or if it is the network's output (retarget the
        output first).

            net.disconnect(net.node("mlm_head"))
            net.delete(net.node("mlm_head"))

        Parameters
        ----------
        node : the node, or its label

        Returns
        -------
        the layer that was removed
        """
        return super().delete(node)

    def check_delete(self, node: Node) -> None:
        if node is self._output:
            raise ValueError(
                f"{node.name!r} is the network's output; set another output before deleting it"
            )

    def after_delete(self, node: Node) -> None:
        self.activations.pop(node.name, None)
        self._timings.pop(node.name, None)

    def entry_point(self, layer: BasalEstimator):
        return layer.forward

    def check_connection(
        self, layer: BasalEstimator, sources: tuple, strict: bool = True, replacing=None
    ) -> None:
        if not isinstance(layer, BasalEstimator):
            raise TypeError(f"a node runs a BasalEstimator, got {type(layer).__name__}")

        holder = next((node for node in self._nodes if node.layer is layer), None)
        if holder is not None and holder is not replacing:
            raise ValueError(
                f"{layer.__class__.__name__} already runs at node {holder.name!r}; "
                "every node needs its own layer"
            )

        if strict and not sources:
            raise ValueError(
                f"{layer.__class__.__name__} needs at least one source. Pass "
                "net.input for the first layer in a graph."
            )

        for position, source in enumerate(sources):
            if not isinstance(source, Node):
                raise TypeError(
                    f"source {position} is {type(source).__name__}, expected a "
                    "Node. Use the value returned by connect(), or net.input."
                )
            if not any(known is source for known in self._nodes):
                raise ValueError(
                    f"source {source.name!r} belongs to a different network"
                )

        if strict:
            self.check_graph(layer, len(sources))
            self.check_shapes(layer, sources)

    def check_rewire(self, node: Node, sources: tuple) -> None:
        downstream = self.descendants(node)
        for source in sources:
            if source is node or source in downstream:
                raise ValueError(
                    f"connecting {source.name!r} to {node.name!r} would form a cycle"
                )

    def after_rewire(self, node: Node, strict: bool) -> None:
        for member in self._nodes:
            member.consumers = []
        for member in self._nodes:
            for source in member.sources:
                source.consumers.append(member)
        self.sort_nodes()
        for member in [node, *self.descendants(node)]:
            member.resolve_shapes()
            if strict and member is not node:
                self.check_shapes(member.layer, member.sources)

    def descendants(self, node: Node) -> list[Node]:
        """every node that depends on this one, in execution order"""
        seen, stack = set(), list(node.consumers)
        while stack:
            member = stack.pop()
            if member not in seen:
                seen.add(member)
                stack.extend(member.consumers)
        return [member for member in self._nodes if member in seen]

    def ancestors(self, node: Node) -> list[Node]:
        """every node this one depends on, in execution order"""
        seen, stack = set(), list(node.sources)
        while stack:
            member = stack.pop()
            if member not in seen:
                seen.add(member)
                stack.extend(member.sources)
        return [member for member in self._nodes if member in seen]

    def sort_nodes(self) -> None:
        """Put the nodes back in an order where every node follows its sources, keeping the current order where free."""
        placed, ordered, pending = set(), [], list(self._nodes)
        while pending:
            for candidate in pending:
                if all(source in placed for source in candidate.sources):
                    break
            else:
                raise ValueError("the graph contains a cycle")
            ordered.append(candidate)
            placed.add(candidate)
            pending.remove(candidate)
        self._nodes[:] = ordered

    def make_node(self, name: str, layer: BasalEstimator, sources: tuple) -> Node:
        return Node(name, layer, sources)

    def after_connect(self, node: Node) -> None:
        node.train(self.training)
        self._output = node

    def check_graph(self, layer: BasalEstimator, given: int) -> None:
        """
        Compare the edge count against the layer's forward signature and declared shapes.

        Three counts have to agree
        - how many sources were passed
        - how many args forward takes
        - how many input shapes the layer declares

        Raises only; designed to fail when constructing, not when passing data.
        """
        name = layer.__class__.__name__
        shapes = layer.shapes
        emitted = len(shapes["output"])
        if emitted != 1:
            raise ValueError(
                f"{name} declares {emitted} outputs. A node carries one value"
            )

        required, maximum = self.positional_counts(layer.forward)
        if given < required or (maximum is not None and given > maximum):
            raise ValueError(
                f"{name}.forward takes {required} to {maximum} inputs, got {given}"
            )

        declared = len(shapes["input"])
        if given > declared:
            raise ValueError(
                f"{name} was given {given} sources but declares {declared} "
                "input shapes, so the extra ones would go unchecked. Declare "
                "one shape per input!"
            )

    def check_shapes(self, layer: BasalEstimator, sources: tuple[Node, ...]) -> None:
        """
        Compare what each source produces against what the layer says it takes,
        pairing them by position.

        Position matters: a merge takes its sources in a fixed order, and
        checking every one against the first declaration would pass a graph
        whose inputs are transposed.

        The source's resolved out_shape is used, not its declared output. A
        shape-preserving layer declares no width of its own, so reading its
        declaration would report None and silently pass every edge below it --
        the resolved shape is the one that carries the width down the graph.
        """
        expected = layer.shapes["input"]

        for position, source in enumerate(sources):
            wanted = expected[position] if position < len(expected) else ANY_SHAPE
            conflict = shape_conflict(source.shapes["output"], wanted)

            if conflict:
                raise ValueError(
                    f"{layer.__class__.__name__} cannot be fed by "
                    f"{source.name!r} at position {position}: {conflict}"
                )

    def extend(self, layers: Iterable[BasalEstimator]) -> Node:
        """chain layers end to end"""
        node = self._output or self._input
        for layer in layers:
            node = self.connect(layer, node)
        return node

    # ------------- the output
    @property
    def output(self) -> Node:
        if self._output is None:
            raise ValueError("the network has no layers")
        return self._output

    @output.setter
    def output(self, node: Node) -> None:
        if not isinstance(node, Node):
            raise TypeError("the output must be a Node() returned by connect()")
        if not any(known is node for known in self._nodes):
            raise ValueError(f"node {node.name!r} belongs to a different network")
        self._output = node

    # ------------- the passes
    def forward(self, x_data: NDArray, **kwargs) -> NDArray:
        """
        forward pass -- taking the insertion order or navigating the node-to-node process

        kwargs : offered to every node, and picked up only by the layers
            that declare a matching optional parameter -- mask, forced_activation,
            or anything a layer adds later. A layer that doesn't declare the
            name never receives it, so unused kwargs are silently ignored.
            Mode is not a kwarg: layers read their own `training`, set by train() / eval().
        """
        output = self.output
        values = {self._input: x_data}

        for node in self._nodes:
            if node.is_source or not self.is_wired(node):
                continue
            if any(source not in values for source in node.sources):
                continue
            arguments = [values[source] for source in node.sources]
            passthrough = (
                kwargs
                if node.accepts_any_kwarg
                else {key: value for key, value in kwargs.items() if key in node.forward_kwargs}
            )
            start = time.perf_counter()
            try:
                values[node] = node.layer.forward(*arguments, **passthrough)
            except Exception as error:
                shapes = ", ".join(str(np.shape(argument)) for argument in arguments)
                raise RuntimeError(
                    f"{node.name} ({node.layer.__class__.__name__}).forward failed "
                    f"on input shape(s) {shapes}: {error}"
                ) from error
            self.record_timing(node.name, "forward", time.perf_counter() - start)

        self.activations = {node.name: value for node, value in values.items()}
        if output not in values:
            raise ValueError(
                f"the output node {output.name!r} was not run: it, or a node it "
                "depends on, is unwired. Reconnect it or set another output"
            )
        return values[output]

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        """
        Navigate the gradient back through the graph
        """
        gradients = {self.output: incoming_gradient}

        for node in reversed(self._nodes):
            if node.is_source or node not in gradients:
                continue

            incoming = gradients.pop(node)
            start = time.perf_counter()
            try:
                returned = node.layer.backward(incoming)
            except Exception as error:
                raise RuntimeError(
                    f"{node.name} ({node.layer.__class__.__name__}).backward failed "
                    f"on gradient shape {np.shape(incoming)}: {error}"
                ) from error
            self.record_timing(node.name, "backward", time.perf_counter() - start)
            parts = returned if len(node.sources) > 1 else (returned,)

            if len(parts) != len(node.sources):
                raise ValueError(
                    f"node {node.name!r} has {len(node.sources)} sources but "
                    f"its backward returned {len(parts)} gradients"
                )

            for source, part in zip(node.sources, parts):
                if source in gradients:
                    if gradients[source].shape != part.shape:
                        raise ValueError(
                            f"{node.name!r} sent a {part.shape} gradient to "
                            f"{source.name!r}, but another consumer already "
                            f"sent {gradients[source].shape} -- every consumer "
                            "of a shared source has to agree on its shape"
                        )
                    gradients[source] = gradients[source] + part
                else:
                    gradients[source] = part

        return gradients.get(self._input)

    def __call__(self, x_data: NDArray, **kwargs) -> NDArray:
        return self.forward(x_data, **kwargs)

    def record_timing(self, name: str, phase: str, elapsed: float) -> None:
        entry = self._timings.setdefault(
            name,
            {
                "forward_total": 0.0,
                "forward_calls": 0,
                "backward_total": 0.0,
                "backward_calls": 0,
            },
        )
        entry[f"{phase}_total"] += elapsed
        entry[f"{phase}_calls"] += 1

    def reset_timings(self) -> None:
        """clear accumulated per-node timing, e.g. between epochs"""
        self._timings = {}

    def timing_summary(self, top: Optional[int] = None) -> str:
        """
        Per-node timing, sorted by total forward+backward time descending.

        Parameters
        ----------
        top : only show this many nodes; None shows every timed node
        """
        rows = [
            (
                name,
                entry["forward_total"],
                entry["forward_calls"],
                entry["backward_total"],
                entry["backward_calls"],
            )
            for name, entry in self._timings.items()
        ]
        rows.sort(key=lambda row: row[1] + row[3], reverse=True)
        if top is not None:
            rows = rows[:top]

        width = max((len(name) for name, *_ in rows), default=4)
        lines = [f"{self.name}: per-node timing"]
        lines.append(
            f"  {'node'.ljust(width)}  {'fwd avg (ms)':>13} {'fwd total (s)':>14} "
            f"{'bwd avg (ms)':>13} {'bwd total (s)':>14}"
        )
        for name, fwd_total, fwd_calls, bwd_total, bwd_calls in rows:
            fwd_avg = (fwd_total / fwd_calls * 1000) if fwd_calls else 0.0
            bwd_avg = (bwd_total / bwd_calls * 1000) if bwd_calls else 0.0
            lines.append(
                f"  {name.ljust(width)}  {fwd_avg:13.3f} {fwd_total:14.4f} "
                f"{bwd_avg:13.3f} {bwd_total:14.4f}"
            )
        return "\n".join(lines)

    # ------------- inspection
    def validate(self) -> list[str]:
        """
        Check for structural problem
        """
        problems = []
        output = self._output

        for node in self._nodes:
            if node.is_source:
                continue
            if not self.is_wired(node):
                problems.append(
                    f"{node.name} has {len(node.sources)} of {self.required_sources(node.layer)} "
                    "inputs connected, so forward skips it"
                )
            if node is output:
                continue
            if not node.consumers:
                problems.append(
                    f"{node.name} feeds nothing and is not the output, so it "
                    "runs forward but never trains"
                )
        return problems

    def prune(self) -> list[str]:
        """
        Drop nodes that are no longer ancestors of the current output.

        A branch left behind by retargeting `net.output` elsewhere (e.g.
        swapping a trained network's head) keeps running forward every
        pass and cluttering `layers`, `summary()`, and serialization,
        even though `backward()` already ignores it -- `validate()` flags
        it but leaves it in place. This removes such nodes outright, by
        walking `.sources` back from the output and dropping anything
        that walk never reaches. It only ever removes nodes; edges among
        the ones that remain are untouched.

        Returns
        -------
        names of the nodes removed, in their original graph order
        """
        reachable = {self._input}
        frontier = [self.output]
        while frontier:
            node = frontier.pop()
            if node in reachable:
                continue
            reachable.add(node)
            frontier.extend(node.sources)

        kept = [node for node in self._nodes if node in reachable]
        dropped = [node for node in self._nodes if node not in reachable]

        for node in kept:
            node.consumers = [
                consumer for consumer in node.consumers if consumer in reachable
            ]

        self._nodes = kept
        for node in dropped:
            self.after_delete(node)
        return [node.name for node in dropped]

    # ------------- serialization
    def get_config(self) -> dict:
        return {"name": self.name, "input_shape": self._input_shape}

    def resolve_source(self, name: str) -> Node:
        try:
            return self.node(name)
        except KeyError as error:
            raise KeyError(
                f"no node named {name!r} to use as a source; when restoring a "
                "saved network, the saved nodes may be out of order"
            ) from error

    def extra_state(self) -> dict:
        return {"output": self.output.name}

    def restore_extras(self, state: dict) -> None:
        self._output = self.node(state["output"])

    # ------------- mode
    def train(self, mode: bool = True) -> Network:
        """Switch the network, and every node and layer in it, between training and inference."""
        self.training = mode
        for node in self._nodes:
            node.train(mode)
        return self

    def eval(self) -> Network:
        """Switch to inference."""
        return self.train(False)

    # ------------- parameters, keyed by node name
    @property
    def layers(self) -> list[BasalEstimator]:
        """every node's layer, in graph order"""
        return [node.layer for node in self._nodes if not node.is_source]

    @property
    def num_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self.layers)

    def purge(self) -> None:
        """Clear every layer's forward-pass caches and the stored activations."""
        for layer in self.layers:
            layer.purge()
        self.activations = {}

    def zero_gradients(self) -> None:
        for layer in self.layers:
            layer.zero_gradients()

    def get_gradients(self) -> dict[str, dict]:
        """`gradient_<node name>` for every node whose layer reports gradients."""
        gradients = {}
        for node in self._nodes:
            if node.is_source:
                continue
            node_gradients = node.layer.get_gradients()
            if node_gradients:
                gradients[f"{GRADIENT_PREFIX}{node.name}"] = node_gradients
        return gradients

    def update_weights(self, **gradients: dict) -> None:
        """
        Subtract each node's gradients from its layer: `layer -= gradient`.

        Parameters
        ----------
        gradients : `gradient_<node name>` as returned by get_gradients(), already scaled
        """
        layers = {node.name: node.layer for node in self._nodes if not node.is_source}
        unexpected = [key for key in gradients if key.removeprefix(GRADIENT_PREFIX) not in layers]
        if unexpected:
            raise ValueError(f"{self.name} has no nodes for gradients {sorted(unexpected)}")
        for key, node_gradients in gradients.items():
            if not node_gradients:
                continue
            layer = layers[key.removeprefix(GRADIENT_PREFIX)]
            layer -= node_gradients

    def __isub__(self, gradients: dict) -> Network:
        """`net -= gradients` applies a get_gradients() dict through update_weights()."""
        if not isinstance(gradients, dict):
            raise TypeError("can only subtract a gradient dict from a Network")
        self.update_weights(**gradients)
        return self

    def node_shapes(self) -> dict[str, dict[str, tuple]]:
        """
        every node's declared input and output shapes, keyed by node name
        """
        return {
            node.name: {**node.layer.shapes, "resolved": node.out_shape}
            for node in self._nodes
            if not node.is_source
        }

    def summary(self, x_data: Optional[NDArray] = None) -> str:
        """
        Summary of the network -- shapshot view of the setup
        """
        if x_data is not None:
            self.forward(x_data)
        shapes = {name: np.shape(value) for name, value in self.activations.items()}

        listed = [node for node in self._nodes if not node.is_source]
        width = max((len(node.name) for node in listed), default=4)

        lines = [f"{self.name}: {len(listed)} nodes, {self.num_parameters} parameters"]
        lines.append(f"  {'node'.ljust(width)}  {'sources':<26} shape")
        for node in listed:
            marker = " <- output" if node is self._output else ""
            lines.append(
                f"  {node.name.ljust(width)}  "
                f"{','.join(source.name for source in node.sources):<26} "
                f"{shapes.get(node.name, node.out_shape)}{marker}"
            )

        for problem in self.validate():
            lines.append(f"  warning: {problem}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self)} nodes, "
            f"{self.num_parameters} parameters)"
        )
