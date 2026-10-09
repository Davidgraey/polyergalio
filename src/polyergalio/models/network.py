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
from itertools import islice
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from polyergalio.base_model import BasalEstimator, resolve_data_types
from polyergalio.composite_model import Composite, CompositeNode
from polyergalio.models.constants import ANY_SHAPE
from polyergalio.utilities import shape_conflict

INPUT_NAME = "input"
GRADIENT_PREFIX = "gradient_"


def describe_shape(value) -> tuple:
    """Shape of an array, or the shapes of the arrays in a tuple."""
    if isinstance(value, (tuple, list)):
        return tuple(np.shape(member) for member in value)
    return np.shape(value)


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
            Node(Layer) -[edge]-> Node(Layer) -[edge]-> Node(Layer)

    A layer node's `in_shape` is the layer's declared input shapes, one per source. Its
    `out_shape` is what it produces for downstream edges, resolved from the sources it was
    wired to. That is what the next node's check is made against, so an array width flows
    down the graph as it is wired and checked on connect().
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

        try:
            for position, source in enumerate(sources):
                source.add_consumer(self, position)
            if layer is not None:
                self.resolve_shapes()
        except Exception:
            for source in sources:
                source.remove_consumer(self)
            raise

    def add_consumer(self, consumer: Node, position: int) -> None:
        """Record that consumer reads this node as its argument at position."""
        self.consumers.append(consumer)

    def remove_consumer(self, consumer: Node) -> None:
        """Forget every edge from this node to consumer."""
        self.consumers = [member for member in self.consumers if member is not consumer]

    def shape_for(self, consumer: Node, position: int) -> tuple:
        """Trailing shape of what the edge to consumer at position carries."""
        return self.out_shape

    def candidate_shape(self, taken: int = 0, replacing: Optional[Node] = None) -> tuple:
        """Trailing shape a new edge would carry; taken counts earlier new edges from this node in the same connection."""
        return self.out_shape

    def read(self, values: dict, consumer: Node, position: int):
        """The value the edge to consumer at position carries, from the forward values."""
        return values[self]

    def receive(self, gradients: dict, consumer: Node, position: int, part: NDArray) -> None:
        """Add the gradient consumer sent back along its edge at position."""
        if self in gradients:
            if gradients[self].shape != part.shape:
                raise ValueError(
                    f"{consumer.name!r} sent a {part.shape} gradient to "
                    f"{self.name!r}, but another consumer already "
                    f"sent {gradients[self].shape} -- every consumer "
                    "of a shared source has to agree on its shape"
                )
            gradients[self] = gradients[self] + part
        else:
            gradients[self] = part

    def take(self, gradients: dict, values: dict):
        """The gradient with respect to this node's output, removed from gradients; values are the forward activations by name."""
        return gradients.pop(self)

    def resolve_shapes(self) -> None:
        """
        Work out the shape this node produces from what its sources produce
        now; a node with no sources yields its layer's declared output.
        """
        incoming = tuple(source.shape_for(self, position) for position, source in enumerate(self.sources))
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


class RoutingNode(Node):
    """
    A node whose layer returns several outputs (the layer sets multi_output), and which decides which edges read which.

    It owns the routing, output index -> the edges reading that output, each edge a (consumer node, argument position).
    A new connection takes the lowest output with no edges, so consumers receive output 0, 1, 2 in the order they connect.
    assign() routes an edge to a chosen output, and several edges may share one.

        Network
            Node(Layer) -[edge]-> RoutingNode(Layer)--output 0  -[edge]-> Node(Layer)
                                                    --output 1  -[edge]-> Node(Layer)

    A layer that declares n output shapes has n outputs; one that declares a single shape has any number, all of that
    shape, checked against what forward returns. Outputs with no edge are dropped, with zero gradient.
    """

    def __init__(self, name: str, layer: BasalEstimator, sources: tuple = ()):
        """
        Parameters
        ----------
        name : label, and the key the network stores this node's gradients under
        layer : the estimator to run; its multi_output is True
        sources : nodes feeding this one, in the order the layer's forward takes them
        """
        self.routing: dict[int, list[tuple[Node, int]]] = {}
        super().__init__(name, layer, sources)

    @property
    def num_outputs(self) -> Optional[int]:
        """Number of outputs the layer declares, None when it is open-ended."""
        declared = len(self.layer.shapes["output"])
        return declared if declared > 1 else None

    def resolve_shapes(self) -> None:
        incoming = tuple(source.shape_for(self, position) for position, source in enumerate(self.sources))
        resolved = self.layer.infer_output_shapes(incoming) if incoming else self.layer.shapes["output"]
        if len(resolved) != 1 and len(resolved) != self.num_outputs:
            raise ValueError(
                f"{self.layer.__class__.__name__}.infer_output_shapes returned {len(resolved)} shapes, "
                f"but the layer declares {self.num_outputs or 1}"
            )
        self.out_shapes = tuple(resolved)
        self.out_shape = resolved[0]

    def output_shape(self, output: int) -> tuple:
        """Trailing shape of one output."""
        return self.out_shapes[output] if len(self.out_shapes) > 1 else self.out_shapes[0]

    def copy_routing(self) -> dict[int, list[tuple[Node, int]]]:
        """A copy of the routing; changes replace the dict, so a rolled-back snapshot gets the old one."""
        return {output: list(edges) for output, edges in self.routing.items()}

    def free_outputs(self, replacing: Optional[Node] = None):
        """
        Iterate the outputs with no edges, lowest first. Edges of replacing, a consumer being rewired, do not count.
        """
        output = 0
        while self.num_outputs is None or output < self.num_outputs:
            if all(consumer is replacing for consumer, _ in self.routing.get(output, ())):
                yield output
            output += 1

    def unrouted_outputs(self) -> list[int]:
        """Outputs with no edge, for a layer that declares its count."""
        return [output for output in range(self.num_outputs or 0) if output not in self.routing]

    def output_for(self, consumer: Node, position: int) -> int:
        """The output the edge-to-consumer at [position] reads."""
        for output, edges in self.routing.items():
            if any(member is consumer and place == position for member, place in edges):
                return output
        raise ValueError(f"{consumer.name!r} reads {self.name!r} at position {position}, but no output is routed to it")

    def shape_for(self, consumer: Node, position: int) -> tuple:
        return self.output_shape(self.output_for(consumer, position))

    def candidate_shape(self, taken: int = 0, replacing: Optional[Node] = None) -> tuple:
        output = next(islice(self.free_outputs(replacing), taken, taken + 1), None)
        if output is None:
            raise ValueError(self.full_message())
        return self.output_shape(output)

    def full_message(self) -> str:
        return f"every output of {self.name!r} already feeds an edge; assign() routes another edge onto one"

    def add_consumer(self, consumer: Node, position: int) -> None:
        """
        adding the mapping or routing for this node's odwnstreams. updates `self.routing` object
        ---inplace---

        Parameters
        ----------
        consumer: the downstream node to build an edge FROM self TO consumer
        position: index of the consumer node
        """
        if not any(member is consumer and place == position for edges in self.routing.values() for member, place in edges):
            output = next(self.free_outputs(), None)
            if output is None:
                raise ValueError(self.full_message())
            routing = self.copy_routing()
            routing[output] = [(consumer, position)]
            self.routing = routing
        super().add_consumer(consumer, position)

    def remove_consumer(self, consumer: Node) -> None:
        """
        Remove the downstream node from THIS node's connected outputs. updates `self.routing` object
        ---inplace---

        Parameters
        ----------
        consumer: the downstream (consumer) node
        """
        routing = {}
        for output, edges in self.routing.items():
            kept = [(member, place) for member, place in edges if member is not consumer]
            if kept:
                routing[output] = kept
        self.routing = routing
        super().remove_consumer(consumer)

    def sync(self, live) -> None:
        """
        Drop edges whose consumer is not in live or no longer reads this node at that position. Edges that remain keep
        their output, so rewiring one consumer does not move the others.
        ---inplace---

        Parameters
        ----------
        live : the nodes still in the graph
        """
        routing = {}
        for output, edges in self.routing.items():
            kept = [
                (member, place) for member, place in edges
                if member in live and place < len(member.sources) and member.sources[place] is self
            ]
            if kept:
                routing[output] = kept
        self.routing = routing

    def assign(self, output_index: int, consumer: Node, position: Optional[int] = None) -> None:
        """
        Route an edge to an output, moving it from wherever it was. Changes the routing only.

        Parameters
        ----------
        output_index : the output the consumer should read
        consumer : a node that reads this one
        position : the consumer's argument position; None when it reads this node once

        Raises
        ------
        ValueError, with nothing changed, if the output does not exist, the consumer does not read this node at that
        position, or the output's shape cannot feed that argument
        """
        positions = [place for place, source in enumerate(consumer.sources) if source is self]
        if position is None:
            if len(positions) != 1:
                raise ValueError(
                    f"{consumer.name!r} reads {self.name!r} at {len(positions)} positions; say which with position"
                )
            position = positions[0]
        elif position not in positions:
            raise ValueError(f"{consumer.name!r} does not read {self.name!r} at position {position}")

        count = self.num_outputs
        if output_index < 0 or (count is not None and output_index >= count):
            raise ValueError(f"{self.name!r} has no output {output_index}; it has {count if count is not None else 'any number of'}")

        expected = consumer.in_shape
        wanted = expected[position] if position < len(expected) else ANY_SHAPE
        conflict = shape_conflict(self.output_shape(output_index), wanted)
        if conflict:
            raise ValueError(f"output {output_index} of {self.name!r} cannot feed {consumer.name!r} at position {position}: {conflict}")

        routing = {}
        for output, edges in self.routing.items():
            kept = [(member, place) for member, place in edges if not (member is consumer and place == position)]
            if kept:
                routing[output] = kept
        routing.setdefault(output_index, []).append((consumer, position))
        self.routing = routing

    def read(self, values: dict, consumer: Node, position: int):
        output = self.output_for(consumer, position)
        produced = values[self]
        if not isinstance(produced, (tuple, list)):
            raise ValueError(f"{self.name!r} ({self.layer.__class__.__name__}) must return a tuple of outputs")
        if output >= len(produced):
            raise ValueError(
                f"{self.name!r} ({self.layer.__class__.__name__}) returned {len(produced)} outputs, "
                f"but output {output} feeds {consumer.name!r}"
            )
        return produced[output]

    def receive(self, gradients: dict, consumer: Node, position: int, part: NDArray) -> None:
        output = self.output_for(consumer, position)
        parts = gradients.setdefault(self, {})
        if output in parts:
            if parts[output].shape != part.shape:
                raise ValueError(
                    f"{consumer.name!r} sent a {part.shape} gradient to output {output} of "
                    f"{self.name!r}, but another consumer already sent {parts[output].shape} -- "
                    "every consumer of a shared output has to agree on its shape"
                )
            parts[output] = parts[output] + part
        else:
            parts[output] = part

    def take(self, gradients: dict, values: dict) -> tuple:
        parts = gradients.pop(self)
        return tuple(
            parts[output] if output in parts else np.zeros_like(produced)
            for output, produced in enumerate(values[self.name])
        )

    def routing_text(self) -> str:
        """The routing as one line: output -> consumers."""
        return "; ".join(
            f"{output} -> {', '.join(consumer.name for consumer, _ in edges)}"
            for output, edges in sorted(self.routing.items())
        )


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
        data_type=np.float64,
        complex_data_type=None,
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
        data_type : numpy floating type the network's real arrays and inputs use; see recast()
        complex_data_type : numpy complex type for complex arrays; defaults to the precision matching data_type
        """
        super().__init__(name)
        self.data_type, self.complex_data_type = resolve_data_types(data_type, complex_data_type)
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
            self.check_shapes(layer, sources, replacing=replacing)

    def check_rewire(self, node: Node, sources: tuple) -> None:
        downstream = self.descendants(node)
        for source in sources:
            if source is node or source in downstream:
                raise ValueError(
                    f"connecting {source.name!r} to {node.name!r} would form a cycle"
                )

    def after_rewire(self, node: Node, strict: bool) -> None:
        live = set(self._nodes)
        for member in self._nodes:
            if isinstance(member, RoutingNode):
                member.sync(live)
            member.consumers = []
        for member in self._nodes:
            for position, source in enumerate(member.sources):
                source.add_consumer(member, position)
        self.sort_nodes()
        for member in [node, *self.descendants(node)]:
            member.resolve_shapes()
            if strict and member is not node:
                self.check_shapes(member.layer, member.sources, consumer=member)

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
        node_class = RoutingNode if layer.multi_output else Node
        return node_class(name, layer, sources)

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
        if emitted != 1 and not layer.multi_output:
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

    def check_shapes(
        self,
        layer: BasalEstimator,
        sources: tuple[Node, ...],
        consumer: Optional[Node] = None,
        replacing: Optional[Node] = None,
    ) -> None:
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

        A source with several outputs is compared on the output the edge reads:
        for a node already in the graph (consumer), the one it is routed to; for
        a new connection, the output it would take, skipping the edges of
        replacing, the node being rewired.

        Parameters
        ----------
        consumer : the node whose existing edges are being checked, None for a new connection
        replacing : the node whose sources are being replaced
        """
        expected = layer.shapes["input"]
        taken = {}

        for position, source in enumerate(sources):
            wanted = expected[position] if position < len(expected) else ANY_SHAPE
            if consumer is not None:
                produced = source.shape_for(consumer, position)
            else:
                produced = source.candidate_shape(taken.get(source, 0), replacing)
                taken[source] = taken.get(source, 0) + 1
            conflict = shape_conflict(produced, wanted)

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

    # ---------------- OUTPUTS ---------------
    @property
    def output(self) -> Node:
        if self._output is None:
            raise ValueError("the network has no layers")
        return self._output

    @output.setter
    def output(self, node: Node) -> None:
        if not isinstance(node, Node):
            raise TypeError("the output must be a Node() returned by connect()")
        if isinstance(node, RoutingNode):
            raise ValueError(f"{node.name!r} is a RoutingNode with several outputs; the network's output is a single value")
        if not any(known is node for known in self._nodes):
            raise ValueError(f"node {node.name!r} belongs to a different network")
        self._output = node

    # ---------------- Passes - forward/ backward -----------------------------
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
        if isinstance(x_data, np.ndarray):
            if x_data.dtype.kind == "f":
                x_data = x_data.astype(self.data_type, copy=False)
            elif x_data.dtype.kind == "c":
                x_data = x_data.astype(self.complex_data_type, copy=False)
        values = {self._input: x_data}

        for node in self._nodes:
            if node.is_source or not self.is_wired(node):
                continue
            if any(source not in values for source in node.sources):
                continue
            arguments = [source.read(values, node, position) for position, source in enumerate(node.sources)]
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

            incoming = node.take(gradients, self.activations)
            start = time.perf_counter()
            try:
                returned = node.layer.backward(incoming)
            except Exception as error:
                raise RuntimeError(
                    f"{node.name} ({node.layer.__class__.__name__}).backward failed "
                    f"on gradient shape {describe_shape(incoming)}: {error}"
                ) from error
            self.record_timing(node.name, "backward", time.perf_counter() - start)
            parts = returned if len(node.sources) > 1 else (returned,)

            if len(parts) != len(node.sources):
                raise ValueError(
                    f"node {node.name!r} has {len(node.sources)} sources but "
                    f"its backward returned {len(parts)} gradients"
                )

            for position, (source, part) in enumerate(zip(node.sources, parts)):
                source.receive(gradients, node, position, part)

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
            if isinstance(node, RoutingNode):
                for unrouted in node.unrouted_outputs():
                    problems.append(f"{node.name} output {unrouted} feeds nothing, its slice is dropped")
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
        ***** Be careful with this process *****

        A branch left behind by retargeting `net.output` elsewhere (e.g. swapping a trained network's head) keeps
        running forward every pass and cluttering `layers`, `summary()`, and serialization, even though `backward()`
        will ignore such unconnected nodes

        `validate()` flags disconnected nodes, but prune() removes these nodes outright, by walking `.sources` back
        from the output and dropping anything that walk never reaches.
        It only ever removes nodes; edges among intact nodes that remain are not modified.

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

        live = set(kept)
        for node in kept:
            if isinstance(node, RoutingNode):
                node.sync(live)

        self._nodes = kept
        for node in dropped:
            self.after_delete(node)

        return [node.name for node in dropped]

    # -------------------------- serialization --------------------------
    def get_config(self) -> dict:
        return {
            "name": self.name,
            "input_shape": self._input_shape,
            "data_type": self.data_type,
            "complex_data_type": self.complex_data_type,
        }

    def resolve_source(self, name: str) -> Node:
        try:
            return self.node(name)
        except KeyError as error:
            raise KeyError(
                f"no node named {name!r} to use as a source; when restoring a "
                "saved network, the saved nodes may be out of order"
            ) from error

    def extra_state(self) -> dict:
        state = {"output": self.output.name}
        routing = {
            node.name: [
                [output, consumer.name, position]
                for output, edges in sorted(node.routing.items())
                for consumer, position in edges
            ]
            for node in self._nodes
            if isinstance(node, RoutingNode)
        }
        if routing:
            state["routing"] = routing
        return state

    def restore_extras(self, state: dict) -> None:
        self._output = self.node(state["output"])
        for name, entries in state.get("routing", {}).items():
            node = self.node(name)
            node.routing = {}
            for output, consumer_name, position in entries:
                node.assign(output, self.node(consumer_name), position)
            wanted = sum(1 for consumer in self._nodes for source in consumer.sources if source is node)
            if sum(len(edges) for edges in node.routing.values()) != wanted:
                raise ValueError(f"the saved routing of {name!r} does not cover every edge that reads it")
        self.recast(self.data_type, self.complex_data_type)

    # ------------------------ precision and data types --------------------------
    def recast(self, data_type=np.float64, complex_data_type=None) -> Network:
        """
        Change the network's precision, and every layer's with it.

        Layers connected afterwards keep their own dtype until the next recast. Gradients are reset to zeros and
        stored activations are cleared; recast before creating an optimizer, whose state is keyed to the gradients.

        Parameters
        ----------
        data_type : numpy floating type, e.g. np.float32
        complex_data_type : numpy complex type; defaults to the precision matching data_type

        Returns
        -------
        self, so calls chain
        """
        self.data_type, self.complex_data_type = resolve_data_types(data_type, complex_data_type)
        for layer in self.layers:
            layer.recast(self.data_type, self.complex_data_type)
        self.activations = {}
        return self

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
        shapes = {name: describe_shape(value) for name, value in self.activations.items()}

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

        for node in listed:
            if isinstance(node, RoutingNode):
                lines.append(f"  routing {node.name}: {node.routing_text()}")

        for problem in self.validate():
            lines.append(f"  warning: {problem}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self)} nodes, "
            f"{self.num_parameters} parameters)"
        )
