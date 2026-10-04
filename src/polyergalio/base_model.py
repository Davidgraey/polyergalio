"""
The two classes split by concern.
Serializable is the base structural elements of an object to serialize and reinstantiate
BasalEstimator holds behaviors, parameters, operations and what the object does.

Concerns
--------
    concern      declared by                       read             write                         owner
    config       constructor arguments             get_config       the constructor               Serializable
    state        state_names                       get_state        set_state                     Serializable
    parameters   parameter_names                   get_weights      set_weights, update_weights   BasalEstimator
    gradients    gradient_<name> attributes        get_gradients    zero_gradients                BasalEstimator
    caches       set during forward                -                purge                         BasalEstimator
    mode         training flag                     training         train, eval                   BasalEstimator

config
    The constructor arguments. Each is read back from the attribute of the same name, so an argument without a
    matching attribute is an error. Config is fixed at construction; to change it, build a new object.

state
    Everything besides config that is needed to restore a fitted object: fitted statistics, counters, learned
    values. A member that is itself Serializable contributes its own state. set_state requires exactly the names
    that get_state returns.

parameters
    The trainable subset of state. A parameter is an array, a scalar or another BasalEstimator. backward() sets
    gradient_<name> for each one, and update_weights applies it as `member -= gradient`, the same statement for an
    array and for a sublayer. Parameters are part of the state, so they are never listed twice.

caches and mode
    Caches are forward-pass values kept for backward(). They are cleared by purge() and never persisted. Mode is
    the training flag, which train() and eval() set on an estimator and every sublayer it holds.

Serialization
    serialize() packages {"type", "version", "config", "state"}. Every subclass registers under its class name, and
    each family (a class that lists Serializable as a direct base) has its own registry, so deserialize() resolves
    the saved type, constructs it from config, then restores state.

Writing a subclass
    Declare parameter_names and state_names with only the names the class adds; they accumulate down the
    hierarchy. Name every constructor argument after an attribute. Override get_state or set_state only when state
    is not a flat list of names, and update_weights only when an update is not a subtraction.
"""
import copy
import inspect
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio import __version__
from polyergalio.models.constants import ANY_SHAPE
from polyergalio.utilities import count_elements


# ===================================================================================================
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


def read_members(owner: Serializable, names: tuple[str, ...], read: Callable) -> dict:
    """
    Map each name to the owner's attribute. A member that is itself Serializable is replaced by read(member).
    Arrays are active / live references.
    """
    members = {name: getattr(owner, name) for name in names}
    return {
        name: read(member) if isinstance(member, Serializable) else member
        for name, member in members.items()
    }


def write_members(owner: Serializable, names: tuple[str, ...], values: dict, write: Callable) -> None:
    """
    Restore the owner's attributes from {name: value}, which must hold exact names.
    A Serializable member receives write(member, value); an array is copied and keeps its dtype;
    anything else is assigned.
    Nothing is written when the names do not match.
    """
    kind = owner.__class__.__name__
    if not isinstance(values, dict):
        raise ValueError(f"{kind}: expected a dict, got {type(values).__name__}")
    unexpected = set(values) - set(names)
    missing = set(names) - set(values)
    if unexpected or missing:
        raise ValueError(f"{kind}: unexpected names {sorted(unexpected)}, missing names {sorted(missing)}")
    for name, value in values.items():
        current = getattr(owner, name)
        if isinstance(current, Serializable):
            write(current, value)
        elif isinstance(current, np.ndarray):
            setattr(owner, name, np.array(value, dtype=current.dtype))
        else:
            setattr(owner, name, value)


# ===================================================================================================
class Serializable:
    """
    How an object is described and rebuilt. Knows nothing about training.

    A class that lists Serializable as a direct base starts its own registry; every class below it registers under
    its registry_key.

    Config is the constructor arguments, read from attributes of the same name. State is everything else needed to
    restore a fitted object: the attributes named in state_names, where a Serializable member contributes its own
    state
    Subclasses list only the names they add
    declarations accumulate down the class hierarchy.

    serialize() packages {"type", "version", "config", "state"}. The config is what has to keep working if a
    constructor changes; the state is bare values (arrays or scalars) with no dependency on the class.
    """
    registry_name: Optional[str] = None
    state_names: tuple[str, ...] = ()
    declarations: tuple[str, ...] = ("state_names",)
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attribute in cls.declarations:
            inherited = tuple(name for base in cls.__bases__ for name in getattr(base, attribute, ()))
            own = cls.__dict__.get(attribute, ())
            setattr(cls, attribute, tuple(dict.fromkeys(inherited + own)))
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

    @classmethod
    def lookup(cls, type_name: str) -> type:
        """The class registered under type_name in this family."""
        target = cls._registry.get(type_name)
        if target is None:
            raise KeyError(f"nothing registered as {type_name!r}. Known: {sorted(cls._registry)}")
        return target

    @classmethod
    def config_names(cls) -> tuple[str, ...]:
        """Constructor argument names."""
        variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        parameters = inspect.signature(cls.__init__).parameters
        return tuple(
            name for name, parameter in parameters.items() if name != "self" and parameter.kind not in variadic
        )

    def get_config(self) -> dict:
        """Constructor arguments as {name: value}. Every argument needs an attribute of the same name."""
        names = self.config_names()
        missing = [name for name in names if not hasattr(self, name)]
        if missing:
            raise AttributeError(f"{self.__class__.__name__}: constructor arguments without an attribute: {missing}")
        return {name: getattr(self, name) for name in names}

    def persisted_names(self) -> tuple[str, ...]:
        """Attribute names that make up the state. Subclasses with more kinds of state extend this."""
        return self.state_names

    def get_state(self) -> dict:
        """Persisted attributes as {name: value}; a Serializable member contributes its own get_state()."""
        return read_members(self, self.persisted_names(), lambda member: member.get_state())

    def set_state(self, state: dict) -> None:
        """Restore from get_state() output, which must hold exactly persisted_names()."""
        write_members(self, self.persisted_names(), state, lambda member, value: member.set_state(value))

    def serialize(self, path: Optional[str | os.PathLike] = None) -> dict:
        """
        Package type, version, config and state into a plain dict; config and state are deep-copied.

        Parameters
        ----------
        path : also pickle the dict to this file
        """
        payload = {
            "type": self.registry_key(),
            "version": __version__,
            "config": copy.deepcopy(self.get_config()),
            "state": copy.deepcopy(self.get_state()),
        }
        if path is not None:
            write_serialized(payload, path)
        return payload

    @classmethod
    def rebuild(cls, config: dict, state: dict) -> Serializable:
        """
        Construct from config, then restore state. Override where the constructor needs more than config, such as
        an injected object.
        """
        instance = cls(**config)
        try:
            instance.set_state(state)
        except (ValueError, NotImplementedError) as error:
            warnings.warn(f"{cls.__name__}: could not restore state -- {error}")
        return instance

    @classmethod
    def deserialize(cls, source: dict | str | os.PathLike) -> Serializable:
        """
        Rebuild from serialize() output, or from a file written by serialize(path). Saved config is filtered
        against the current constructor, and dropped keys raise a warning.
        """
        serialized_dict = read_serialized(source)
        target = cls.lookup(serialized_dict["type"])
        accepted = target.config_names()
        config = {key: value for key, value in serialized_dict["config"].items() if key in accepted}
        dropped = set(serialized_dict["config"]) - set(config)
        if dropped:
            warnings.warn(f"{serialized_dict['type']}: dropping saved config keys: {sorted(dropped)}")
        return target.rebuild(config, serialized_dict["state"])


# ===================================================================================================
class BasalEstimator(Serializable, ABC):
    """
    How an object operates. Foundational base for layers and models.

    Trainable members are named in parameter_names. Each is an array, a scalar or another BasalEstimator.
    Every parameter `name` has a gradient attribute `gradient_<name>` that backward() sets; a sublayer reports its
    own. zero_gradients() creates those attributes, so a subclass calls it at the end of __init__.

    A weight update is `member -= gradient`, which is the same statement for an array and for a sublayer:
    BasalEstimator.__isub__ hands the gradient dict to update_weights.

    *** KEY *** parameters are part of the state. Persisted attributes that are not trained go in state_names.
    """
    adaptive: bool = True
    preserves_shape: bool = False
    training: bool = True
    parameter_names: tuple[str, ...] = ()
    declarations: tuple[str, ...] = Serializable.declarations + ("parameter_names",)

    def __init__(self):
        super().__init__()
        self.declare_shapes()

    def persisted_names(self) -> tuple[str, ...]:
        """Parameters followed by the declared state names."""
        return self.parameter_names + self.state_names

    def sublayers(self) -> list[BasalEstimator]:
        """Estimators held as attributes, directly or inside a tuple, list or dict."""
        found = []
        for value in vars(self).values():
            if isinstance(value, dict):
                members = value.values()
            elif isinstance(value, (tuple, list)):
                members = value
            else:
                members = (value,)
            found.extend(member for member in members if isinstance(member, BasalEstimator) and member is not self)
        return found

    def train(self, mode: bool = True) -> BasalEstimator:
        """
        Switch this estimator, and every sublayer it holds, between training and inference.

        Returns
        -------
        self, so calls chain: layer.eval().forward(x)
        """
        self.training = mode
        for sublayer in self.sublayers():
            sublayer.train(mode)
        return self

    def eval(self) -> BasalEstimator:
        """Switch to inference, see train()."""
        return self.train(False)

    def declare_shapes(self, inputs=(ANY_SHAPE,), outputs=(ANY_SHAPE,)) -> None:
        """
        Record what this estimator accepts and emits, in trailing-axis form.
        Used to validate incoming and outgoing connections.
        """
        self._in_shapes = tuple(inputs)
        self._out_shapes = tuple(outputs)

    @property
    def shapes(self) -> dict[str, tuple[tuple, ...]]:
        """The declared (not observed) input and output shapes."""
        return {"input": self._in_shapes, "output": self._out_shapes}

    def infer_output_shapes(self, input_shapes: tuple[tuple, ...]) -> tuple[tuple, ...]:
        """Resolve the output shapes given what actually arrives."""
        if self.preserves_shape:
            return (input_shapes[0],)
        return self._out_shapes

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def backward(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def purge(self) -> None:
        """Clear forward-pass caches. Gradients belong to zero_gradients()."""

    def get_weights(self) -> dict:
        """Trainable members as {name: value}. A sublayer's value is its own get_weights() dict."""
        return read_members(self, self.parameter_names, lambda member: member.get_weights())

    def set_weights(self, weights: dict) -> None:
        """Restore trainable members from get_weights() output, which must hold exactly parameter_names."""
        write_members(self, self.parameter_names, weights, lambda member, value: member.set_weights(value))

    def get_gradients(self) -> dict:
        """
        Gradients to apply in update_weights, keyed `gradient_<name>`. A sublayer's value is its own
        get_gradients() dict. Empty when there are no parameters.
        """
        gradients = {}
        for name in self.parameter_names:
            member = getattr(self, name)
            if isinstance(member, BasalEstimator):
                gradients[f"gradient_{name}"] = member.get_gradients()
            else:
                gradients[f"gradient_{name}"] = getattr(self, f"gradient_{name}")
        return gradients

    def update_weights(self, **gradients) -> None:
        """
        Subtract each gradient from its parameter. A missing or None gradient leaves the parameter alone.
        Override for update rules that are not a subtraction.

        Parameters
        ----------
        gradients : `gradient_<name>` for each parameter, already scaled by the optimizer
        """
        expected = {f"gradient_{name}" for name in self.parameter_names}
        unexpected = set(gradients) - expected
        if unexpected:
            raise ValueError(f"{self.__class__.__name__} has no gradients named {sorted(unexpected)}")
        for name in self.parameter_names:
            gradient = gradients.get(f"gradient_{name}")
            if gradient is None:
                continue
            member = getattr(self, name)
            # ====---- actual direct operation ----====
            member -= gradient
            setattr(self, name, member)

    def __isub__(self, gradients: dict) -> BasalEstimator:
        """`sublayer -= gradients` applies a get_gradients() dict through update_weights()."""
        if not isinstance(gradients, dict):
            raise TypeError(f"can only subtract a gradient dict from {self.__class__.__name__}")
        self.update_weights(**gradients)
        return self

    def zero_gradients(self) -> None:
        """Reset every parameter's gradient to zeros, and every sublayer's."""
        for name in self.parameter_names:
            member = getattr(self, name)
            if isinstance(member, BasalEstimator):
                member.zero_gradients()
            else:
                setattr(self, f"gradient_{name}", np.zeros_like(member))

    @property
    def num_parameters(self) -> int:
        """Count of trainable values, sublayers included."""
        return count_elements(self.get_weights())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
