"""
Typing complex objects & definitions for __package__
"""
import copy
import inspect
import logging
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import (
    Optional,
)

import numpy as np
from numpy.typing import NDArray

from polyergalio import __version__
from polyergalio.models.constants import EPSILON

log = logging.getLogger(__name__)


# -------------- Persistence mixin  --------------
def write_serialized(payload: dict, path: str | os.PathLike) -> None:
    """Pickle a serialize() payload to a single file."""
    

    with open(path, mode="wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.DEFAULT_PROTOCOL)


def read_serialized(source: dict | str | os.PathLike) -> dict:
    """A serialize() payload, given directly or as the path of a file written by serialize(path)."""
    if isinstance(source, (str, os.PathLike)):
        with open(source, mode="rb") as handle:
            return pickle.load(handle)
    return source


class Serializable:
    """
    Persistence convention shared by layers, models, transforms and processors.

    A class that lists Serializable as a direct base starts its own registry; every class below it registers under
    its registry_key
    State is the fields named in _structural_state_keys, unless a subclass overrides get_weights and set_weights

    serialize() packages {"type", "config", "weights"}; the config is what has to keep working if a constructor changes,
    and the weights are plain arrays with no dependency on the class.
    """

    registry_name: Optional[str] = None
    snapshot_weights: bool = False
    _structural_state_keys: tuple[str, ...] = ()
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if Serializable in cls.__bases__:
            cls._registry = {}
            return
        name = cls.registry_key()
        known = cls._registry.get(name)
        if known is not None and known is not cls:
            warnings.warn(f"{name} redefined; keeping the latest class")
        cls._registry[name] = cls

    @classmethod
    def registry_key(cls) -> str:
        """Name this class is registered under: its own registry_name, else the class name."""
        return cls.__dict__.get("registry_name") or cls.__name__

    def get_config(self) -> dict:
        """Constructor hyperparameters, read from the class signature by name."""
        parameters = inspect.signature(self.__class__.__init__).parameters
        return {
            name: getattr(self, name)
            for name in parameters
            if name != "self" and hasattr(self, name)
        }

    def get_weights(self, for_serialize: bool = False) -> dict:
        """
        Fields named in _structural_state_keys, as live references.

        Parameters
        ----------
        for_serialize : lets a subclass return a persistence-specific view
        """
        return {key: getattr(self, key) for key in self._structural_state_keys}

    def set_weights(self, weights: dict) -> None:
        """Restore fields from get_weights() output."""
        for key, value in weights.items():
            setattr(self, key, value)

    def serialize(self, path: Optional[str | os.PathLike] = None) -> dict:
        """
        Package type, version, config and weights into a plain dict; config
        and weights are deep copied.

        Parameters
        ----------
        path : also pickle the dict to this file
        """
        weights = self.get_weights(for_serialize=True)
        payload = {
            "type": self.registry_key(),
            "version": __version__,
            "config": copy.deepcopy(self.get_config()),
            "weights": weights if self.snapshot_weights else copy.deepcopy(weights),
        }
        if path is not None:
            write_serialized(payload, path)
        return payload

    @classmethod
    def rebuild(cls, config: dict, weights: dict) -> "Serializable":
        """
        Construct from config, then restore weights. Override where the
        constructor needs more than config, such as an injected object.
        """
        instance = cls(**config)
        try:
            instance.set_weights(weights)
        except (ValueError, NotImplementedError) as error:
            warnings.warn(f"{cls.__name__}: could not restore weights -- {error}")
        return instance

    @classmethod
    def deserialize(cls, source: dict | str | os.PathLike) -> "Serializable":
        """
        Rebuild from serialize() output, or from a file written by
        serialize(path). Saved config is filtered against the current
        constructor, and dropped keys raise a warning.
        """
        serialized_dict = read_serialized(source)
        target = cls._registry.get(serialized_dict["type"])
        if target is None:
            raise KeyError(
                f"nothing registered as {serialized_dict['type']!r}. Known: {sorted(cls._registry)}"
            )

        accepted = inspect.signature(target.__init__).parameters
        config = {k: v for k, v in serialized_dict["config"].items() if k in accepted}
        dropped = set(serialized_dict["config"]) - set(config)
        if dropped:
            warnings.warn(f"{serialized_dict['type']}: dropping saved config keys: {sorted(dropped)}")
        return target.rebuild(config, serialized_dict["weights"])


# -------------- Composite containers  --------------
class CompositeNode:
    """One vertex of a Composite: a component and what feeds it."""

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
    source definition is different for each container type.
    (nodes for a network, input column names for a pipeline)

    Serialization nests each component's own payload, so a subclass sets component_family to the Serializable family
    those components belong to.
    """

    component_family: type
    snapshot_weights = True

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

    def extra_weights(self) -> dict:
        """container state beyond the nodes, saved with them"""
        return {}

    def restore_extras(self, weights: dict) -> None:
        """counterpart of extra_weights, run after every node is reconnected"""

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

    def get_weights(self, for_serialize: bool = False) -> dict:
        nodes = [
            {"name": node.name, "sources": node.source_names, "component": node.component.serialize()}
            for node in self._nodes
            if not node.is_source
        ]
        return {"nodes": nodes, **self.extra_weights()}

    def set_weights(self, weights: dict) -> None:
        raise NotImplementedError("a composite is restored by rebuild(), which reconnects every node")

    @classmethod
    def rebuild(cls, config: dict, weights: dict) -> "Composite":
        """Construct from config, then deserialize and reconnect every node in saved order."""
        composite = cls(**config)
        for entry in weights["nodes"]:
            component = cls.component_family.deserialize(entry["component"])
            sources = tuple(composite.resolve_source(name) for name in entry["sources"])
            wired = len(sources) >= composite.required_sources(component)
            composite.connect(component, *sources, name=entry["name"], strict=wired)
        composite.restore_extras(weights)
        return composite


# -------------- Transform Base Class  --------------
class BasalTransform(Serializable, ABC):
    """
    Base for stateful transforms: things that learn a mapping from data and
    then apply it. Deliberately narrower than BasalModel -- a transform has no
    forward pass and no loss to minimise on its own behalf, so it only
    promises fit, predict and the two combined.
    """

    _structural_state_keys: tuple[str, ...] = ()

    @abstractmethod
    def fit(self, x_data: NDArray, **kwargs):
        """learn the transform's parameters from data"""

    @abstractmethod
    def predict(self, x_data: NDArray, **kwargs):
        """apply the fitted transform"""

    @abstractmethod
    def fit_predict(self, x_data: NDArray, **kwargs):
        """fit then apply, in one call"""

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "_is_fitted", False)

    @property
    def info(self) -> dict:
        info = {"self": self.__class__}
        info.update(self.__dict__)
        return info

    def set_weights(self, weights: dict) -> None:
        """Restore fields from get_weights() output and mark the transform fitted."""
        super().set_weights(weights)
        self._is_fitted = True


# -------------- Model Base Class  --------------
def count_array_elements(value) -> int:
    """Total size of every array nested in a dict, list or tuple."""
    if isinstance(value, np.ndarray):
        return value.size
    if isinstance(value, dict):
        return sum(count_array_elements(member) for member in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count_array_elements(member) for member in value)
    return 0


class BasalModel(Serializable, ABC):
    _structural_state_keys: tuple[str, ...] = ("x_means", "x_stds", "num_seen_samples")
    training: bool = True

    def __init__(self,
                 input_dimension: int = 1,
                 output_dimension: int = 1,
                 seed: int = 42):
        self.seed = seed
        self.RNG = np.random.default_rng(seed)

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.x_means = None
        self.x_stds = None
        self.num_seen_samples: int = 0
        self._is_fitted = False

    @abstractmethod
    def forward(self, x_data: NDArray, **kwargs):
        pass

    @abstractmethod
    def predict(self, x_data: NDArray, **kwargs):
        pass

    @abstractmethod
    def fit(self, x_data: NDArray, **kwargs):
        pass

    @abstractmethod
    def fit_predict(self, x_data: NDArray, **kwargs):
        pass

    @abstractmethod
    def calculate_loss(self, **kwargs) -> NDArray:
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def info(self):
        return ""

    @property
    def num_parameters(self) -> int:
        learned = {
            key: getattr(self, key)
            for key in self._structural_state_keys
            if key not in BasalModel._structural_state_keys
        }
        return count_array_elements(learned)

    def train(self, mode: bool = True) -> "BasalModel":
        """
        Switch between training and inference.

        Returns
        -------
        self, so calls chain: model.eval().predict(x)
        """
        self.training = mode
        return self

    def eval(self) -> "BasalModel":
        """Switch to inference, see train()."""
        return self.train(False)

    def set_weights(self, weights: dict) -> None:
        """Restore fields from get_weights() output and mark the model fitted."""
        super().set_weights(weights)
        self._is_fitted = True

    # --------------- Standardization / Normalize ---------------
    def update_running_standardize(self, new_data_mean, new_data_std, new_data_count) -> None:
        """
        proportionally update the running mean and standard deviation for standardization processes
        Parameters
        ----------
        new_data_mean : mean of the new observations or samples under considerations
        new_data_std : standard deviation of the new observations or samples under considerations
        new_data_count : the number of new samples (for proportionally weighting)

        Returns
        -------
        No returns - we update the self. params with the updated mean of the MEAN and STD DEV
        """
        full_count = self.num_seen_samples + new_data_count
        full_mean = (self.num_seen_samples  * self.x_means + new_data_count * new_data_mean) / full_count
        var1 = self.x_stds ** 2
        var2 = new_data_std ** 2

        # error sum of squares
        sum_square_errors = var1 * (self.num_seen_samples  - 1) + var2 * (new_data_count - 1)
        # total group sum of squares
        sum_squares = (self.x_means - full_mean) ** 2 * self.num_seen_samples  + (new_data_mean - full_mean) ** 2 * new_data_count
        full_var = (sum_square_errors + sum_squares) / (full_count - 1)
        full_std = np.sqrt(full_var)

        self.x_means = full_mean
        self.x_stds = full_std

    def standardize(self, data_array: NDArray) -> NDArray:
        """
        Standardize our data array
        Parameters
        ----------
        data_array : numpy array of x-variable

        Returns
        -------
        the mean and standard deviation of the data array
        """
        if self.x_means is None or data_array.shape[-1] != self.x_means.shape[0]:
            log.error("initialize process hasn't been done yet!")

        return (data_array - self.x_means) / (self.x_stds + EPSILON)

    def unstandardize(self, data_array: NDArray) -> NDArray:
        """
        unstandardize the data -> convert back into unit space
        Parameters
        ----------
        data_array : numpy array of x-variable

        Returns
        -------
        the data, transformed back into the original unit space
        """
        assert data_array.shape[-1] == self.x_means.shape[0]

        return data_array * (self.x_stds + EPSILON) + self.x_means

    def init_standardize(self, x_data: NDArray) -> None:
        """
        initalize the standardize variables for tracking, or update them if
        we're adjusting an already fitted model
        Parameters
        ----------
        x_data : NDArray
        """
        if (self.x_means is None) or (self.num_seen_samples == 0):
            # set up the initial values for the new incoming data
            self.num_seen_samples = x_data.shape[0]
            self.x_means = np.mean(x_data, axis=0)
            self.x_stds = np.std(x_data, axis=0)
        else:
            # update the running standardization parameters with proportional weighting
            self.update_running_standardize(
                new_data_mean=np.mean(x_data, axis=0),
                new_data_std=np.std(x_data, axis=0),
                new_data_count=x_data.shape[0])
        pass
