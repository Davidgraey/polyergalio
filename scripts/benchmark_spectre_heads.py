"""
Times the Spectre layers with the batched-matmul DenseHead against the einsum DenseHead it replaced.
"""
import time

import numpy as np

from polyergalio.models.layers import SpectreAttention, SpectrePooledAttention
from polyergalio.models.layers.attention_heads import DenseHead, merge_heads, split_heads
from polyergalio.models.layers.spectre_layers import SpectreDecoderAttention
# (8, 64, 128, 4), (16, 128, 256, 4), (16, 128, 256, 16), (4, 512, 512, 8),
CONFIGS = [(2, 512, 1200, 20), (8, 512, 1200, 20), (16, 512, 1200, 20)]
DTYPES = ("float64", "float32")
REPEATS = 10

MATMUL = (DenseHead.pre_activation, DenseHead.forward, DenseHead.backward)


def einsum_pre_activation(self, input_data):
    heads = split_heads(input_data, self.num_heads)
    return np.einsum("...hi,hio->...ho", heads, self.weights, optimize=True) + self.bias


def einsum_forward(self, input_data):
    self.input = input_data
    self.z = einsum_pre_activation(self, input_data)
    self.output = self.activation_function(self.z)
    return merge_heads(self.output)


def einsum_backward(self, incoming_gradient):
    delta = self.activation_derivative(self.output, self.z, split_heads(incoming_gradient, self.num_heads))
    rows = split_heads(self.input, self.num_heads).reshape(-1, self.num_heads, self.input_dimension)
    deltas = delta.reshape(-1, self.num_heads, self.output_dimension)
    self.gradient_weights = np.einsum("nhi,nho->hio", rows, deltas, optimize=True)
    self.gradient_bias = deltas.sum(axis=0)
    return merge_heads(np.einsum("...ho,hio->...hi", delta, self.weights, optimize=True))


EINSUM = (einsum_pre_activation, einsum_forward, einsum_backward)

LAYERS = {
    "Spectre": lambda length, hidden, heads: SpectreAttention(length, hidden, num_heads=heads, use_wrm=False),
    "Spectre+WRM": lambda length, hidden, heads: SpectreAttention(length, hidden, num_heads=heads, use_wrm=True),
    "SpectrePooled": lambda length, hidden, heads: SpectrePooledAttention(length, hidden, num_heads=heads),
    "SpectreDecoder c=16": lambda length, hidden, heads: SpectreDecoderAttention(
        length, hidden, num_heads=heads, chunk_size=16
    ),
}


def median_ms(function):
    function()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1e3


def use(implementation):
    DenseHead.pre_activation, DenseHead.forward, DenseHead.backward = implementation


def time_layer(make, implementation, shape, dtype):
    batch, length, hidden, heads = shape
    use(implementation)
    layer = make(length, hidden, heads)
    if dtype == "float32":
        layer.recast(np.float32)
    layer.train(True)
    data = np.random.default_rng(0).normal(size=(batch, length, hidden)).astype(dtype)
    mask = np.ones((batch, length), dtype=dtype)
    gradient = np.ones_like(layer.forward(data, mask=mask))
    forward = median_ms(lambda: layer.forward(data, mask=mask))
    step = median_ms(lambda: (layer.forward(data, mask=mask), layer.backward(gradient)))
    return forward, step


def describe_build():
    print(f"numpy {np.__version__}")
    blas = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    print(f"blas {blas.get('name')} {blas.get('version')}")


if __name__ == "__main__":
    describe_build()
    print(f"{'shape (B, T, H, model_heads)':26s}{'dtype':9s}{'layer':22s}{'einsum fwd/step':>18s}{'matmul fwd/step':>18s}{'step speedup':>14s}")
    for shape in CONFIGS:
        for dtype in DTYPES:
            for name, make in LAYERS.items():
                old_forward, old_step = time_layer(make, EINSUM, shape, dtype)
                new_forward, new_step = time_layer(make, MATMUL, shape, dtype)
                print(
                    f"{str(shape):26s}{dtype:9s}{name:22s}"
                    f"{old_forward:8.1f}/{old_step:<8.1f} {new_forward:8.1f}/{new_step:<8.1f} {old_step / new_step:12.2f}x",
                    flush=True,
                )
    use(MATMUL)
