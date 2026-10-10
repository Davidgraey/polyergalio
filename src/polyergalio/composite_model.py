"""
Containers of named components, each fed by sources.

A Composite orders its nodes and saves them by name, nesting each component's own serialized payload. It is a
Serializable family of its own (NeuralNetwork, UniversalPipeline) and does not share the concerns of the components it
holds.
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Optional

from polyergalio.base_model import Serializable


class CompositeNode:
    """One vertex of a Composite Model: a component and what feeds it."""

    def __init__(self, name: str, component=None, sources: tuple = ()):
        self.name = name
        self.component = component
        self.sources = tuple(sources)
        self.consumers: list = []
        self.required: Optional[int] = None

    @property
    def is_source(self) -> bool:
        return self.component is None

    @property
    def source_names(self) -> list[str]:
        return [getattr(source, "name", source) for source in self.sources]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class Composite(Serializable, ABC):
    """
    An ordered container of named components, each fed by sources.

    Insertion order is execution order
    source definitions can be different for each container type

    Serialization nests each component's own payload, so a subclass sets component_family to the Serializable family
    those components belong to.
    """

    component_family: type

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._nodes: list[CompositeNode] = []

    @property
    def nodes(self) -> list[CompositeNode]:
        """every node, including any source node, in insertion order"""
        return list(self._nodes)

    @property
    def components(self) -> list:
        """the component of every node that holds one, in insertion order"""
        return [node.component for node in self._nodes if not node.is_source]

    def node(self, name: str) -> CompositeNode:
        """fetch a node by label"""
        for node in self._nodes:
            if node.name == name:
                return node
        raise KeyError(f"no node named {name!r}. Known nodes: {[node.name for node in self._nodes]}")

    def __getitem__(self, name: str):
        return self.node(name).component

    def __len__(self) -> int:
        return sum(1 for node in self._nodes if not node.is_source)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)} nodes)"

    def edges(self) -> list[tuple[str, str]]:
        """every (source, node) name pair"""
        return [(source, node.name) for node in self._nodes for source in node.source_names]

    def auto_name(self, component) -> str:
        """class name with the first free counter"""
        stem = component.__class__.__name__
        taken = {node.name for node in self._nodes}
        index = 0
        while f"{stem}_{index}" in taken:
            index += 1
        return f"{stem}_{index}"

    def default_name(self, component, sources: tuple) -> str:
        return self.auto_name(component)

    def connect(self, component, *sources, name: Optional[str] = None, strict: bool = True) -> CompositeNode:
        """
        Place a component in the container, fed by the given sources, and
        return its node.

        Parameters
        ----------
        component : member of component_family
        sources : what feeds it, in the order the component takes them
        name : optional label; defaults to default_name(), must be unique
        strict : False places the node without requiring all the sources it
            needs, leaving it unwired until reconnect() supplies them
        """
        self.check_connection(component, sources, strict=strict)
        label = name or self.default_name(component, sources)
        if any(node.name == label for node in self._nodes):
            raise ValueError(f"node name {label!r} is already taken")
        node = self.make_node(label, component, tuple(sources))
        self._nodes.append(node)
        self.after_connect(node)
        return node

    def reconnect(self, node, upstream_edge=None, downstream_edge=None) -> CompositeNode:
        """
        Place an existing node between an upstream and a downstream node, with
        the checks connect() makes. All or nothing.

        Parameters
        ----------
        node : the node, or its label
        upstream_edge : what feeds the node from now on (one source or a
            sequence of them, in the order the component takes them); None
            keeps its current sources
        downstream_edge : a node that should read this one. It takes this node
            in place of the node's upstream source when it read that, and
            otherwise gains it as a further source; None leaves readers as is
        """
        node = self.resolve_node(node)
        saved = self.snapshot()
        try:
            if upstream_edge is not None:
                upstream = upstream_edge if isinstance(upstream_edge, (tuple, list)) else (upstream_edge,)
                self.set_sources(node, self.resolve_sources(upstream), strict=True)
            if downstream_edge is not None:
                reader = self.resolve_node(downstream_edge)
                sources = list(reader.sources)
                for position, existing in enumerate(sources):
                    if any(existing is feed or existing == feed for feed in node.sources):
                        sources[position] = node
                        break
                else:
                    sources.append(node)
                self.set_sources(reader, tuple(sources), strict=True)
        except Exception:
            self.restore(saved)
            raise
        return node

    def disconnect(self, node, *sources) -> CompositeNode:
        """
        Break connections. With sources, removes just those from the node's
        inputs. With none, cuts the node off completely: every input, and the
        node as a source of every node that reads it. Nodes left with too few
        sources to run are unwired, and everything stays reconnectable.

        Parameters
        ----------
        node : the node, or its label
        sources : sources to remove, each as it was connected or by label
        """
        node = self.resolve_node(node)
        if sources:
            remaining = list(node.sources)
            for source in self.resolve_sources(sources):
                for position, existing in enumerate(remaining):
                    if existing is source or existing == source:
                        del remaining[position]
                        break
                else:
                    raise ValueError(f"{source!r} is not a source of {node.name!r}")
            return self.set_sources(node, tuple(remaining), strict=False)

        saved = self.snapshot()
        try:
            for consumer in self.consumers_of(node):
                self.set_sources(consumer, tuple(source for source in consumer.sources if source is not node), strict=False)
            return self.set_sources(node, (), strict=False)
        except Exception:
            self.restore(saved)
            raise

    def consumers_of(self, node: CompositeNode) -> list[CompositeNode]:
        """the nodes that read this node, in order"""
        return [member for member in self._nodes if any(source is node for source in member.sources)]

    def delete(self, node):
        """
        Remove a node and its component from the container. Refuses while the
        node is still connected, in either direction, so disconnect it first.

        Parameters
        ----------
        node : the node, or its label

        Returns
        -------
        the component that was removed
        """
        node = self.resolve_node(node)
        if node.is_source:
            raise ValueError(f"{node.name!r} is a source node and cannot be deleted")
        readers = [consumer.name for consumer in self.consumers_of(node)]
        if node.sources or readers:
            raise ValueError(
                f"{node.name!r} is still connected (reads from {node.source_names}, "
                f"read by {readers}); disconnect it first"
            )
        self.check_delete(node)
        self._nodes.remove(node)
        self.after_delete(node)
        return node.component

    def set_sources(self, node, sources: tuple, strict: bool = True) -> CompositeNode:
        """
        Replace a node's sources in one step. Every check runs before the
        change, and a failure afterwards restores the container exactly.

        Parameters
        ----------
        strict : as in connect(); reconnect() is strict, disconnect() is not
        """
        node = self.resolve_node(node)
        if node.is_source:
            raise ValueError(f"{node.name!r} is a source node and takes no inputs")
        sources = tuple(sources)
        self.check_connection(node.component, sources, strict=strict, replacing=node)
        self.check_rewire(node, sources)
        saved = self.snapshot()
        node.sources = sources
        try:
            self.after_rewire(node, strict)
        except Exception:
            self.restore(saved)
            raise
        return node

    def resolve_node(self, node) -> CompositeNode:
        """a node of this container, given as the node or its label"""
        if isinstance(node, str):
            return self.node(node)
        if not any(known is node for known in self._nodes):
            raise ValueError(f"node {getattr(node, 'name', node)!r} belongs to a different {self.__class__.__name__}")
        return node

    def resolve_sources(self, sources: tuple) -> tuple:
        """sources given as labels turned into what connect() takes; anything else unchanged"""
        return tuple(self.resolve_source(source) if isinstance(source, str) else source for source in sources)

    def snapshot(self) -> tuple:
        """the node order and every node's fields, for restore()"""
        return list(self._nodes), [(node, dict(node.__dict__)) for node in self._nodes]

    def restore(self, saved: tuple) -> None:
        """put back what snapshot() returned"""
        order, states = saved
        self._nodes[:] = order
        for node, state in states:
            node.__dict__.clear()
            node.__dict__.update(state)

    @staticmethod
    def positional_counts(function) -> tuple[int, Optional[int]]:
        """(required, maximum) positional arguments of a callable; maximum is None with *args"""
        parameters = list(inspect.signature(function).parameters.values())
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            and parameter.name != "self"
        ]
        required = sum(1 for parameter in positional if parameter.default is parameter.empty)
        variadic = any(parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters)
        return required, None if variadic else len(positional)

    @abstractmethod
    def entry_point(self, component):
        """the callable whose positional arguments the sources feed"""

    def required_sources(self, component) -> int:
        """how many sources a component needs to run"""
        return self.positional_counts(self.entry_point(component))[0]

    def is_wired(self, node) -> bool:
        """whether the node has every source its component needs"""
        if node.is_source:
            return True
        if node.required is None:
            node.required = self.required_sources(node.component)
        return len(node.sources) >= node.required

    @property
    def active_nodes(self) -> list[CompositeNode]:
        """the wired, non-source nodes in order: the ones that run"""
        return [node for node in self._nodes if not node.is_source and self.is_wired(node)]

    @abstractmethod
    def check_connection(self, component, sources: tuple, strict: bool = True, replacing=None) -> None:
        """
        raise if this component cannot take these sources

        Parameters
        ----------
        strict : also require the sources the component needs and that they fit
        replacing : the node whose sources are being replaced, when reconnecting
        """

    def check_delete(self, node: CompositeNode) -> None:
        """raise if something other than an edge still depends on this node; nothing by default"""

    def after_delete(self, node: CompositeNode) -> None:
        """release whatever the container held for a node that has been removed; nothing by default"""

    def check_rewire(self, node: CompositeNode, sources: tuple) -> None:
        """raise if these sources would make the container inconsistent; nothing by default"""

    def after_rewire(self, node: CompositeNode, strict: bool) -> None:
        """bring the container up to date after a node's sources changed; nothing by default"""

    @abstractmethod
    def make_node(self, name: str, component, sources: tuple) -> CompositeNode:
        """build the node for one connection"""

    def after_connect(self, node: CompositeNode) -> None:
        """runs once a node has been appended"""

    def resolve_source(self, name: str):
        """turn a saved source name back into a source; the name itself by default"""
        return name

    def extra_state(self) -> dict:
        """container state beyond the nodes, saved with them"""
        return {}

    def restore_extras(self, state: dict) -> None:
        """counterpart of extra_state, run after every node is reconnected"""

    def validate(self) -> list[str]:
        """structural problems as readable strings; none by default"""
        return []

    def summary(self) -> str:
        lines = [f"{self.name}: {len(self)} nodes"]
        for node in self._nodes:
            if not node.is_source:
                lines.append(f"  {node.name}  <- {', '.join(str(name) for name in node.source_names)}")
        return "\n".join(lines)

    def get_config(self) -> dict:
        return {"name": self.name}

    def get_state(self) -> dict:
        """Every node keyed by name: its source names and its component's serialized payload, plus extra_state()."""
        nodes = {
            node.name: {"sources": node.source_names, "component": node.component.serialize()}
            for node in self._nodes
            if not node.is_source
        }
        return {"nodes": nodes, **self.extra_state()}

    def set_state(self, state: dict) -> None:
        raise NotImplementedError("a composite is restored by rebuild(), which reconnects every node")

    @classmethod
    def rebuild(cls, config: dict, state: dict) -> Composite:
        """Construct from config, then deserialize and reconnect every node in saved order."""
        composite = cls(**config)
        for name, entry in state["nodes"].items():
            component = cls.component_family.deserialize(entry["component"])
            sources = tuple(composite.resolve_source(source) for source in entry["sources"])
            wired = len(sources) >= composite.required_sources(component)
            composite.connect(component, *sources, name=name, strict=wired)
        composite.restore_extras(state)
        return composite
