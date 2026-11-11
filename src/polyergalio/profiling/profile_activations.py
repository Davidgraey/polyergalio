"""
BENCHMARK
      python <this file> [--dtype float32|float64|both] [--shapes 64x128,1024x256] [--only elu,softmax]
                         [--baseline path/to/older_copy_of_this_module.py]

      µs      best of 5 repeats, microseconds per call (inputs are never mutated, so repeats are fair)
      peak x  tracemalloc peak during one call, output included, as a multiple of the call's input array:
              1.00x = one new array-sized buffer, 2.25x = two buffers plus a bool mask, and so on
      foo'    is the derivative of foo; masked_softmax uses ~20% random padding and temperature 0.8
"""
import argparse
import importlib.util
import sys
import time
import timeit
import tracemalloc
import numpy as np
from polyergalio.models.activations import *


parser = argparse.ArgumentParser(description="Benchmark activations, derivatives, masked_softmax and modReLU.")
parser.add_argument("--dtype", default="both", choices=("float32", "float64", "both"))
parser.add_argument("--shapes", default="64x128,1024x256,4096x1024", help="comma separated ROWSxCOLS")
parser.add_argument("--only", default="", help="comma separated substrings; run matching cases only")
parser.add_argument("--baseline", default=None, help="older copy of this module to compare against")
args = parser.parse_args()

def timed(fn):
    fn()  # warm-up
    t0 = time.perf_counter()
    fn()
    number = max(1, int(0.02 / max(time.perf_counter() - t0, 1e-7)))  # ~20 ms per repeat
    return min(timeit.repeat(fn, repeat=5, number=number)) / number * 1e6

def peak_multiple(fn, nbytes):
    fn()  # allocator warm-up, so one-off caches are not counted
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    result = fn()
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    del result
    return peak / nbytes

def load_module(path):
    spec = importlib.util.spec_from_file_location("baseline_activations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def build_cases(mod, x, upstream, mask, z, dout, beta):
    """{case name: (zero-argument callable, input nbytes)} for one copy of the module."""
    cases = {}
    for name, forward in mod.activation_dictionary.items():
        if name == "mod_relu":
            continue  # complex input, benchmarked separately below
        output = forward(x)
        cases[name] = (lambda forward=forward: forward(x), x.nbytes)
        backward = mod.derivative_dictionary.get(name)
        if backward is not None:
            cases[name + "'"] = (lambda backward=backward, output=output: backward(output, x, upstream), x.nbytes)
    if hasattr(mod, "masked_softmax"):
        cases["masked_softmax"] = (lambda: mod.masked_softmax(x, mask, 0.8), x.nbytes)
    cases["mod_relu (complex)"] = (lambda: mod.mod_relu(z, -0.2), z.nbytes)
    cases["mod_relu' (complex)"] = (lambda: mod.mod_relu_derivative(z, beta, dout), z.nbytes)
    return cases

current = sys.modules[__name__]
baseline = load_module(args.baseline) if args.baseline else None
wanted = [s for s in args.only.split(",") if s]
rng = np.random.default_rng(0)
dtype_names = ("float32", "float64") if args.dtype == "both" else (args.dtype,)

print(f"\nnumpy {np.__version__}" + (f"   baseline: {args.baseline}" if baseline else ""))
for dtype_name in dtype_names:
    dtype = np.dtype(dtype_name)
    complex_dtype = np.dtype("complex64" if dtype_name == "float32" else "complex128")
    for shape_text in args.shapes.split(","):
        rows, cols = (int(n) for n in shape_text.lower().split("x"))
        shape = (rows, cols)
        data = (rng.standard_normal(shape) * 3).astype(dtype)
        grad = rng.standard_normal(shape).astype(dtype)
        mask = rng.random(shape) > 0.2
        mask[:, 0] = True  # every row keeps at least one real option
        z = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(complex_dtype)
        dout = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(complex_dtype)
        beta = rng.uniform(-1.0, 0.0, cols).astype(dtype)
        inputs = (data, grad, mask, z, dout, beta)

        new_cases = build_cases(current, *inputs)
        old_cases = build_cases(baseline, *inputs) if baseline else {}

        print(f"\n== {dtype_name} {shape}  ({data.nbytes / 1e6:.2f} MB per array) " + "=" * 20)
        if baseline:
            print(f"{'case':22s} {'old µs':>11s} {'new µs':>11s} {'speedup':>8s}   {'old peak':>9s} {'new peak':>9s}")
        else:
            print(f"{'case':22s} {'µs':>11s}   {'peak x':>7s}")
        for case_name, (fn, nbytes) in new_cases.items():
            if wanted and not any(s in case_name for s in wanted):
                continue
            t_new, m_new = timed(fn), peak_multiple(fn, nbytes)
            if baseline and case_name in old_cases:
                old_fn, old_nbytes = old_cases[case_name]
                t_old, m_old = timed(old_fn), peak_multiple(old_fn, old_nbytes)
                print(f"{case_name:22s} {t_old:11.1f} {t_new:11.1f} {t_old / t_new:7.2f}x   {m_old:8.2f}x {m_new:8.2f}x")
            elif baseline:
                print(f"{case_name:22s} {'--':>11s} {t_new:11.1f} {'--':>8s}   {'--':>9s} {m_new:8.2f}x")
            else:
                print(f"{case_name:22s} {t_new:11.1f}   {m_new:6.2f}x")