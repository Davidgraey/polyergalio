"""
SpectreAttention and SpectreDecoderAttention: exact gradients in every
configuration, and the decoder's causal training forward against its
time-domain definition and against prefill / decode_step.
"""

import numpy as np
import pytest
from conftest import numeric_gradient, relative_error
from polyergalio.encoders.text_encoders import TextProcessor
from polyergalio.encoders.tokenizer import SentencePieceTokenizer, fit_tokenizer
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.embedding.embedding import TextEmbedding
from polyergalio.models.embedding.positional import RopeEmbedding
from polyergalio.models.layers.basic_layers import FullyConnectedLayer, RMSNormLayer
from polyergalio.models.layers.operator_layers import LatentSum, MaskGather
from polyergalio.models.layers.spectre_layers import (
    DenseHead,
    HeadGate,
    HeadProjection,
    SpectreAttention,
    SpectreDecoderAttention,
)
from polyergalio.models.model_loss import CrossEntropyLoss
from polyergalio.models.network import Network as NeuralNetwork
from polyergalio.models.optimizers import Adam

SEQUENCE = 8
HIDDEN = 8

CONFIGURATIONS = {
    "plain": {},
    "memory": {"memory_tokens": 3},
    "band": {"band_radius": 1},
    "band_and_memory": {"band_radius": 2, "memory_tokens": 2},
    "odd_length": {"sequence_length": 7},
}


def build(layer_class, config: dict):
    config = dict(config)
    sequence = config.pop("sequence_length", SEQUENCE)
    layer = layer_class(sequence, HIDDEN, num_heads=2, **config)
    rng = np.random.default_rng(0)
    if layer.band_radius:
        taps = layer.head_gate.band_taps
        layer.head_gate.band_taps = 0.3 * (rng.normal(size=taps.shape) + 1j * rng.normal(size=taps.shape))
    layer.head_gate.output_layer.weights = rng.normal(size=layer.head_gate.output_layer.weights.shape)
    x = rng.normal(size=(3, sequence, HIDDEN))
    upstream = rng.normal(size=x.shape)
    mask = np.ones((3, sequence))
    mask[:, -2:] = 0
    return layer, x, upstream, mask


def complex_numeric_gradient(loss, tensor, step=1e-6):
    """dL/dRe + i dL/dIm, the convention the layer's complex gradients follow"""
    gradient = np.zeros_like(tensor)
    for index in np.ndindex(tensor.shape):
        original = tensor[index]
        for direction in (1, 1j):
            tensor[index] = original + step * direction
            up = loss()
            tensor[index] = original - step * direction
            down = loss()
            tensor[index] = original
            gradient[index] += direction * (up - down) / (2 * step)
    return gradient


@pytest.mark.parametrize("layer_class", [SpectreAttention, SpectreDecoderAttention], ids=lambda c: c.__name__)
@pytest.mark.parametrize("config", CONFIGURATIONS.values(), ids=CONFIGURATIONS.keys())
def test_gradients_match_finite_differences(layer_class, config):
    layer, x, upstream, mask = build(layer_class, config)

    def loss():
        return float(np.sum(layer.forward(x, mask=mask, training_now=False) * upstream))

    def analytic():
        loss()
        layer.zero_gradients()
        dx = layer.backward(upstream)
        return dx, layer.get_gradients()

    dx, gradients = analytic()
    assert relative_error(dx, numeric_gradient(loss, x)) < 1e-6

    for projection in ("query_projection", "value_projection"):
        for name in ("weights", "bias"):
            tensor = getattr(getattr(layer, projection), name)
            analytic_projection = gradients[projection]["gradient_" + name]
            assert relative_error(analytic_projection, numeric_gradient(loss, tensor)) < 1e-6, (projection, name)
    for name in ("gamma", "beta", "activation_bias"):
        analytic_gate = gradients["head_gate"]["gradient_" + name]
        assert relative_error(analytic_gate, numeric_gradient(loss, getattr(layer.head_gate, name))) < 1e-6, name
    for sublayer in ("hidden_layer", "output_layer"):
        for name in ("weights", "bias"):
            tensor = getattr(getattr(layer.head_gate, sublayer), name)
            analytic_mlp = gradients["head_gate"][sublayer]["gradient_" + name]
            assert relative_error(analytic_mlp, numeric_gradient(loss, tensor)) < 1e-6, (sublayer, name)
    if layer.memory_tokens:
        memory = gradients["persistent_memory"]["gradient_memory"]
        assert relative_error(memory, numeric_gradient(loss, layer.memory.memory)) < 1e-6
    if layer.band_radius:
        numeric_band = complex_numeric_gradient(loss, layer.head_gate.band_taps)
        assert relative_error(gradients["head_gate"]["gradient_band_taps"], numeric_band) < 1e-6


def test_wrm_gradients_match_finite_differences():
    layer, x, upstream, mask = build(SpectreAttention, {"use_wrm": True})

    def loss():
        return float(np.sum(layer.forward(x, mask=mask, training_now=False) * upstream))

    loss()
    layer.zero_gradients()
    assert relative_error(layer.backward(upstream), numeric_gradient(loss, x)) < 1e-6


def test_the_layer_learns_token_mixing():
    """each position copies its predecessor -- a target only the spectral gate can express"""
    rng = np.random.default_rng(0)
    layer = SpectreAttention(SEQUENCE, HIDDEN, num_heads=2)
    x = rng.normal(size=(16, SEQUENCE, HIDDEN))
    target = np.roll(x, 1, axis=1)
    optimizer = Adam(1e-2)
    for _ in range(200):
        layer.zero_gradients()
        output = layer.forward(x, training_now=True)
        layer.backward(2 * (output - target) / output.size)
        optimizer.step([layer])
    held_out = rng.normal(size=(16, SEQUENCE, HIDDEN))
    error = np.mean((layer.forward(held_out, training_now=False) - np.roll(held_out, 1, axis=1)) ** 2)
    assert error < 0.1


# -------------    decoding    ----------------------------
def causal_reference(layer, x, mask=None):
    """
    The decoder's definition, written directly in time: position p mixes values at positions <= p
    with the filter from its chunk anchor's pooled query.
    """
    memory, batch = layer.memory_tokens, x.shape[0]
    mask = np.ones(x.shape[:2]) if mask is None else mask
    combined = x
    if memory:
        bank = np.broadcast_to(layer.memory.get_memory()[None], (batch, memory, layer.input_dimension))
        combined = np.concatenate([bank, x], axis=1)
    queries = layer.query_projection.project(combined)
    values = layer.value_projection.project(combined)
    values[:, memory:] *= mask[..., None]

    output = np.zeros_like(x)
    for position in range(x.shape[1]):
        anchor = (position // layer.chunk_size) * layer.chunk_size
        query_sum = queries[:, :memory].sum(axis=1) + (queries[:, memory: memory + anchor + 1] * mask[:, : anchor + 1, None]).sum(axis=1)
        counts = np.maximum(mask[:, : anchor + 1].sum(axis=1, keepdims=True), 1.0) + memory
        filters = np.fft.irfft(layer.gate_from_pooled_sum(query_sum, counts), n=combined.shape[1], axis=1)
        row = memory + position
        output[:, position] = sum(filters[:, row - j] * values[:, j] for j in range(row + 1))
    return output


CHUNK_SIZES = [1, 3, SEQUENCE]

DECODER_CONFIGURATIONS = {
    "plain": {},
    "memory": {"memory_tokens": 2},
    "band_and_memory": {"memory_tokens": 2, "band_radius": 1},
    "full_reconstruction": {"memory_tokens": 2, "use_positional_phase": False},
}


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("config", DECODER_CONFIGURATIONS.values(), ids=DECODER_CONFIGURATIONS.keys())
def test_the_training_forward_matches_the_causal_definition(config, chunk_size):
    layer, x, _, mask = build(SpectreDecoderAttention, {**config, "chunk_size": chunk_size})
    np.testing.assert_allclose(layer.forward(x, mask=mask), causal_reference(layer, x, mask), atol=1e-10)


@pytest.mark.parametrize("chunk_size", [3, SEQUENCE])
def test_chunked_gradients_match_finite_differences(chunk_size):
    layer, x, upstream, mask = build(SpectreDecoderAttention, {"memory_tokens": 2, "band_radius": 1, "chunk_size": chunk_size})

    def loss():
        return float(np.sum(layer.forward(x, mask=mask) * upstream))

    loss()
    layer.zero_gradients()
    assert relative_error(layer.backward(upstream), numeric_gradient(loss, x)) < 1e-6
    gradients = layer.get_gradients()
    query_weights = layer.query_projection.weights
    assert relative_error(gradients["query_projection"]["gradient_weights"], numeric_gradient(loss, query_weights)) < 1e-6
    hidden_weights = layer.head_gate.hidden_layer.weights
    assert relative_error(gradients["head_gate"]["hidden_layer"]["gradient_weights"], numeric_gradient(loss, hidden_weights)) < 1e-6
    assert relative_error(gradients["persistent_memory"]["gradient_memory"], numeric_gradient(loss, layer.memory.memory)) < 1e-6


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("config", DECODER_CONFIGURATIONS.values(), ids=DECODER_CONFIGURATIONS.keys())
def test_prefill_and_decode_match_the_training_forward(config, chunk_size):
    layer, x, _, _ = build(SpectreDecoderAttention, {**config, "chunk_size": chunk_size})
    trained = layer.forward(x)
    for prompt in (1, 2, 4):
        outputs = [layer.prefill(x[:, :prompt])] + [layer.decode_step(x[:, t]) for t in range(prompt, SEQUENCE)]
        for position, output in zip(range(prompt - 1, SEQUENCE), outputs):
            np.testing.assert_allclose(output, trained[:, position], atol=1e-10)


@pytest.mark.parametrize("config", DECODER_CONFIGURATIONS.values(), ids=DECODER_CONFIGURATIONS.keys())
def test_decoding_past_the_window_matches_a_forward_over_the_last_window(config):
    """chunk_size=1 only: larger chunks' anchors stop lining up with a fresh window once it slides"""
    layer, _, _, _ = build(SpectreDecoderAttention, config)
    x = np.random.default_rng(2).normal(size=(2, 2 * SEQUENCE + 3, HIDDEN))
    outputs = [layer.prefill(x[:, :1])] + [layer.decode_step(x[:, t]) for t in range(1, x.shape[1])]
    for t in range(SEQUENCE - 1, x.shape[1]):
        window = x[:, t - SEQUENCE + 1: t + 1]
        np.testing.assert_allclose(outputs[t], layer.forward(window, training_now=False)[:, -1], atol=1e-10)


# -------------    head layers    --------------------------
def test_head_projection_gradients_match_finite_differences():
    projection = HeadProjection(num_heads=2, head_dimension=3)
    rng = np.random.default_rng(6)
    x = rng.normal(size=(2, 5, 6))
    upstream = rng.normal(size=x.shape)

    def loss():
        return float(np.sum(projection.forward(x) * upstream))

    loss()
    dx = projection.backward(upstream)
    assert relative_error(dx, numeric_gradient(loss, x)) < 1e-6
    assert relative_error(projection.gradient_weights, numeric_gradient(loss, projection.weights)) < 1e-6
    assert relative_error(projection.gradient_bias, numeric_gradient(loss, projection.bias)) < 1e-6


@pytest.mark.parametrize("band_radius", [0, 2])
def test_head_gate_gradients_match_finite_differences(band_radius):
    gate = HeadGate(num_heads=2, head_dimension=3, num_frequencies=5, gate_dimension=4, band_radius=band_radius)
    rng = np.random.default_rng(8)
    gate.output_layer.weights = rng.normal(size=gate.output_layer.weights.shape)
    if band_radius:
        gate.band_taps = 0.3 * (rng.normal(size=gate.band_taps.shape) + 1j * rng.normal(size=gate.band_taps.shape))
    pooled = rng.normal(size=(3, 6))
    upstream = rng.normal(size=(3, 2, 5)) + 1j * rng.normal(size=(3, 2, 5))
    descriptor_upstream = rng.normal(size=(3, 6))

    def loss():
        activated = gate.forward(pooled)
        return float(np.sum(activated.real * upstream.real + activated.imag * upstream.imag)
                     + np.sum(gate.descriptor * descriptor_upstream))

    loss()
    dpooled = gate.backward(upstream, descriptor_upstream)
    assert relative_error(dpooled, numeric_gradient(loss, pooled)) < 1e-6
    for name in ("gamma", "beta", "activation_bias"):
        assert relative_error(getattr(gate, "gradient_" + name), numeric_gradient(loss, getattr(gate, name))) < 1e-6, name
    for sublayer in (gate.hidden_layer, gate.output_layer):
        assert relative_error(sublayer.gradient_weights, numeric_gradient(loss, sublayer.weights)) < 1e-6
        assert relative_error(sublayer.gradient_bias, numeric_gradient(loss, sublayer.bias)) < 1e-6
    if band_radius:
        assert relative_error(gate.gradient_band_taps, complex_numeric_gradient(loss, gate.band_taps)) < 1e-6


@pytest.mark.parametrize("activation_type", ["linear", "relu", "tanh", "softmax"])
def test_dense_head_gradients_match_finite_differences(activation_type):
    layer = DenseHead(num_heads=3, input_dimension=2, output_dimension=4, activation_type=activation_type)
    rng = np.random.default_rng(9)
    layer.bias = rng.normal(size=layer.bias.shape)
    x = rng.normal(size=(2, 5, 6))
    upstream = rng.normal(size=(2, 5, 12))

    def loss():
        return float(np.sum(layer.forward(x) * upstream))

    loss()
    dx = layer.backward(upstream)
    assert relative_error(dx, numeric_gradient(loss, x)) < 1e-6
    assert relative_error(layer.gradient_weights, numeric_gradient(loss, layer.weights)) < 1e-6
    assert relative_error(layer.gradient_bias, numeric_gradient(loss, layer.bias)) < 1e-6


# -------------    stacked encoder    ----------------------------
def test_a_multi_block_encoder_overfits_masked_language_modelling(tmp_path):
    """two Spectre blocks memorise one cloze batch of 50 sentences: tokenizer, processor, embedding, RoPE, norms, attention, ffn, residuals, gather and head train together"""
    length, hidden, blocks, cases = 20, 32, 2, 50
    rng = np.random.default_rng(0)

    sentences = [
        f"{det} {adjective} {noun} {verb} {adverb}."
        for det in ("the", "a")
        for adjective in ("small", "large", "quiet", "loud", "red", "blue")
        for noun in ("cat", "dog", "bird", "fish", "fox", "owl")
        for verb in ("runs", "jumps", "sleeps", "swims", "climbs", "watches")
        for adverb in ("fast", "slowly", "quietly", "today")
    ]
    texts = [sentences[i] for i in rng.choice(len(sentences), size=cases, replace=False)]

    model_path = fit_tokenizer(texts, str(tmp_path / "tokenizer"), vocab_size=80, hard_vocab_limit=False)
    processor = TextProcessor(SentencePieceTokenizer(model_path), max_length=length, mask_prob=0.25, random_seed=0)
    batch = processor.distort_batch(texts, "cloze")
    ids, attention_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]
    vocab, padding_id = processor.vocab_size, processor.special.PAD
    labels = batch["labels"][target_mask]
    targets = np.eye(vocab)[labels]

    net = NeuralNetwork(name="spectre_mlm", input_shape=(length,))
    stream = net.connect(TextEmbedding(vocab, hidden, padding_idx=padding_id), net.input, name="embedding")
    stream = net.connect(RopeEmbedding(length, hidden), stream, name="positional")
    for block in range(blocks):
        normed = net.connect(RMSNormLayer(hidden), stream, name=f"prenorm_{block}")
        mixed = net.connect(
            SpectreAttention(length, hidden, num_heads=4, memory_tokens=2, use_wrm=True), normed, name=f"attention_{block}"
        )
        mixed = net.connect(LatentSum(), mixed, stream, name=f"attention_residual_{block}")
        expanded = net.connect(FullyConnectedLayer(hidden, 2 * hidden, "swish"), mixed, name=f"ffn_1_{block}")
        projected = net.connect(FullyConnectedLayer(2 * hidden, hidden, "linear"), expanded, name=f"ffn_2_{block}")
        stream = net.connect(LatentSum(), projected, mixed, name=f"ffn_residual_{block}")
    stream = net.connect(RMSNormLayer(hidden), stream, name="final_norm")
    gathered = net.connect(MaskGather(), stream, name="mask_gather")
    net.output = net.connect(FullyConnectedLayer(hidden, vocab, "linear", is_output=True), gathered, name="mlm_head")

    loss_fn = CrossEntropyLoss(task=ClassificationTask.MULTINOMIAL)
    optimizer = Adam(3e-3)
    net.train()
    losses = []
    for _ in range(500):
        batch = processor.distort_batch(texts, "cloze")
        ids, attention_mask, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_mask"]
        labels = batch["labels"][target_mask]
        targets = np.eye(vocab)[labels]
        net.zero_gradients()
        logits = net.forward(ids, mask=attention_mask, target_mask=target_mask)
        losses.append(loss_fn(logits, targets))
        net.backward(loss_fn.backward())
        optimizer.step(net.layers)

    net.eval()
    logits = net.forward(ids, mask=attention_mask, target_mask=target_mask)
    accuracy = np.mean(np.argmax(logits, axis=-1) == labels)
    assert np.all(np.isfinite(losses))
    assert losses[-1] < 0.1 * losses[0]
    assert accuracy > 0.95
