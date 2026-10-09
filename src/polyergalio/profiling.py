"""
Timing and memory profiling for any call, block, or model, in one pass.

    profiler = Profiler()

    result = profiler.call(train_step, batch, num_loops=10)

    with profiler.measure("evaluate"):
        evaluate(model, data)

    timed_step = profiler.wrap(train_step, label="step", num_loops=10)

    with profiler.instrument(network):
        network.forward(x)
        network.backward(grad)

    print(profiler.summary())

Memory is Python allocated while the block ran, as seen by tracemalloc, which covers numpy buffers.
Use `with Profiler() as profiler:` to keep tracing on across several measurements, so arrays freed inside a block that
were allocated before it are still counted.
"""
from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any, Callable, Iterator, Optional

from polyergalio.base_model import BasalEstimator
from polyergalio.composite_model import Composite

MEGABYTE = 1e6


@dataclass
class Measurement:
    """
    Accumulated cost of one label, over every call recorded under it.

    Attributes
    ----------
    calls : number of times the label ran
    total_time : seconds summed over calls, including nested measurements
    min_time, max_time : seconds of the fastest and slowest call
    self_time : seconds summed over calls, excluding nested measurements
    net_bytes : memory held after a call minus before it, summed over calls; negative when calls free more than
        they allocate
    peak_total : each call's peak above the level at entry, including nested measurements, summed over calls
    min_peak, max_peak : smallest and largest single-call peak
    """
    label: str
    calls: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    self_time: float = 0.0
    net_bytes: int = 0
    peak_total: int = 0
    min_peak: int = 0
    max_peak: int = 0

    @property
    def mean_time(self) -> float:
        return self.total_time / self.calls if self.calls else 0.0

    @property
    def mean_net_bytes(self) -> float:
        return self.net_bytes / self.calls if self.calls else 0.0

    @property
    def mean_peak(self) -> float:
        return self.peak_total / self.calls if self.calls else 0.0

    def add(self, elapsed: float, self_elapsed: float, net: int, peak: int) -> None:
        """Fold one call into the totals."""
        if self.calls == 0:
            self.min_peak = peak
        self.calls += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.self_time += self_elapsed
        self.net_bytes += net
        self.peak_total += peak
        self.min_peak = min(self.min_peak, peak)
        self.max_peak = max(self.max_peak, peak)


@dataclass
class Frame:
    """Open measurement on the stack: what its nested measurements have used so far."""
    peak: int = 0
    child_time: float = 0.0


def named_estimators(model: Any, path: Optional[str] = None) -> list[tuple[str, Any]]:
    """
    Every estimator reachable from a model, with a label built from the path to it.

    A Composite's nodes are labelled by node name; sublayers by class name. A shared estimator appears once,
    under the first path that reaches it.

    Parameters
    ----------
    model : an estimator or a Composite such as a Network
    path : label for the model itself; defaults to its class name or Composite name
    """
    found, seen = [], set()

    def visit(member: Any, label: str) -> None:
        if id(member) in seen:
            return
        seen.add(id(member))
        found.append((label, member))
        if isinstance(member, Composite):
            for node in member.nodes:
                if not node.is_source:
                    visit(node.component, f"{label}/{node.name}")
        elif isinstance(member, BasalEstimator):
            for sublayer in member.sublayers():
                visit(sublayer, f"{label}/{type(sublayer).__name__}")

    visit(model, path or getattr(model, "name", None) or type(model).__name__)
    return found


class Profiler:
    """Collects wall time and allocated memory per label, for calls, blocks and models."""

    def __init__(self, track_memory: bool = True):
        """
        Parameters
        ----------
        track_memory : record memory with tracemalloc; False times only, which avoids its overhead
        """
        self.track_memory = track_memory
        self.measurements: dict[str, Measurement] = {}
        self.stack: list[Frame] = []
        self.owns_tracing = False

    def start(self) -> Profiler:
        """Begin tracing allocations, if tracing is not already on."""
        if self.track_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            self.owns_tracing = True
        return self

    def stop(self) -> None:
        """End tracing, if this profiler started it."""
        if self.owns_tracing:
            tracemalloc.stop()
            self.owns_tracing = False

    def __enter__(self) -> Profiler:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def reset(self) -> None:
        """Discard everything measured so far."""
        self.measurements = {}

    @contextmanager
    def measure(self, label: str) -> Iterator[None]:
        """Time and memory-profile the enclosed block under a label; blocks can nest."""
        temporary = self.track_memory and not tracemalloc.is_tracing()
        if temporary:
            tracemalloc.start()

        start_bytes = 0
        if self.track_memory:
            start_bytes, peak_so_far = tracemalloc.get_traced_memory()
            if self.stack:
                self.stack[-1].peak = max(self.stack[-1].peak, peak_so_far)
            tracemalloc.reset_peak()
        frame = Frame(peak=start_bytes)
        self.stack.append(frame)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stack.pop()
            end_bytes = 0
            if self.track_memory:
                end_bytes, peak_now = tracemalloc.get_traced_memory()
                frame.peak = max(frame.peak, peak_now)
                tracemalloc.reset_peak()
            if self.stack:
                parent = self.stack[-1]
                parent.child_time += elapsed
                parent.peak = max(parent.peak, frame.peak)

            self.measurements.setdefault(label, Measurement(label)).add(
                elapsed, elapsed - frame.child_time, end_bytes - start_bytes, frame.peak - start_bytes
            )
            if temporary:
                tracemalloc.stop()

    def call(self, func: Callable, *args, label: Optional[str] = None, num_loops: int = 1, **kwargs) -> Any:
        """
        Run func(*args, **kwargs) under measurement, num_loops times, and return the last result.

        Every loop is recorded as its own call under the label, so the report gives their mean, min and max. func
        really runs each time, so anything it changes is changed num_loops times.

        Parameters
        ----------
        label : name in the report; defaults to the function's qualified name
        num_loops : how many times to run func
        """
        return self.wrap(func, label=label, num_loops=num_loops)(*args, **kwargs)

    def wrap(self, func: Optional[Callable] = None, *, label: Optional[str] = None, num_loops: int = 1) -> Callable:
        """
        Return func measured on every call. Works as a decorator, `@profiler.wrap` or `@profiler.wrap(label="x")`.

        Parameters
        ----------
        label : name in the report; defaults to the function's qualified name
        num_loops : times each call of the wrapped function runs func, each recorded separately; the last result is
            returned
        """
        assert num_loops >= 1, f"num_loops must be at least 1, got {num_loops}"
        if func is None:
            return lambda target: self.wrap(target, label=label, num_loops=num_loops)

        name = label or getattr(func, "__qualname__", repr(func))

        @wraps(func)
        def measured(*args, **kwargs):
            for _ in range(num_loops):
                with self.measure(name):
                    result = func(*args, **kwargs)
            return result

        return measured

    @contextmanager
    def instrument(self, model: Any, phases: tuple[str, ...] = ("forward", "backward")) -> Iterator[None]:
        """
        Measure every estimator in a model, per phase, for the duration of the block.

        Labels are `<path>.<phase>`, e.g. `net/ffn_1.forward`. Each estimator's own method is restored on exit.

        Parameters
        ----------
        model : an estimator or Composite
        phases : method names to wrap on every estimator that has them
        """
        restore = []
        for label, estimator in named_estimators(model):
            for phase in phases:
                original = getattr(estimator, phase, None)
                if not callable(original):
                    continue
                restore.append((estimator, phase, vars(estimator).get(phase)))
                setattr(estimator, phase, self.wrap(original, label=f"{label}.{phase}"))
        try:
            yield
        finally:
            for estimator, phase, previous in reversed(restore):
                if previous is None:
                    delattr(estimator, phase)
                else:
                    setattr(estimator, phase, previous)

    def rows(self) -> list[dict]:
        """Every measurement as a flat dict, ready for a DataFrame; times in seconds, memory in bytes."""
        return [
            {**asdict(entry), "mean_time": entry.mean_time, "mean_net_bytes": entry.mean_net_bytes,
             "mean_peak": entry.mean_peak}
            for entry in self.measurements.values()
        ]

    def summary(self, sort_by: str = "self_time", top: Optional[int] = None) -> str:
        """
        Table of every label, largest first. Time and peak memory are per call: mean, min and max over the calls.

        Parameters
        ----------
        sort_by : a Measurement field or mean: self_time, total_time, mean_time, max_time, calls, net_bytes,
            mean_peak, max_peak
        top : only the first this many rows; None shows all
        """
        entries = sorted(self.measurements.values(), key=lambda entry: getattr(entry, sort_by), reverse=True)
        total_self = sum(entry.self_time for entry in self.measurements.values()) or 1.0
        if top is not None:
            entries = entries[:top]

        width = max((len(entry.label) for entry in entries), default=5)
        header = (
            f"{'label':{width}s} {'calls':>6s} {'mean ms':>9s} {'min ms':>9s} {'max ms':>9s} "
            f"{'self ms':>10s} {'self %':>7s} {'net MB':>8s} {'peak MB':>8s} {'min MB':>8s} {'max MB':>8s}"
        )
        lines = [header, "-" * len(header)]
        for entry in entries:
            lines.append(
                f"{entry.label:{width}s} {entry.calls:6d} {entry.mean_time * 1e3:9.2f} "
                f"{entry.min_time * 1e3:9.2f} {entry.max_time * 1e3:9.2f} {entry.self_time * 1e3:10.2f} "
                f"{100 * entry.self_time / total_self:7.1f} {entry.mean_net_bytes / MEGABYTE:8.2f} "
                f"{entry.mean_peak / MEGABYTE:8.2f} {entry.min_peak / MEGABYTE:8.2f} {entry.max_peak / MEGABYTE:8.2f}"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    import numpy as np

    from polyergalio.models.layers.basic_layers import FullyConnectedLayer

    layer = FullyConnectedLayer(input_dimension=64, output_dimension=256, activation_type="swish")
    inputs = np.random.default_rng(0).normal(size=(2048, 64))

    profiler = Profiler()
    with profiler:
        output = profiler.call(layer.forward, inputs, label="forward x5", num_loops=5)
        with profiler.measure("block"):
            layer.backward(np.ones_like(output))
        with profiler.instrument(layer):
            layer.backward(layer.forward(inputs))
    print(profiler.summary())
