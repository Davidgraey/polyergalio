"""
Embeddings: token lookup tables and positional encodings.
"""

import numpy as np
import pytest

from conftest import GRADIENT_TOLERANCE, input_gradient_error, numeric_gradient, relative_error
from polyergalio.models.embedding.embedding import TextEmbedding
from polyergalio.models.embedding.positional import RopeEmbedding, SinusoidEmbedding
from polyergalio.models.optimizers import Adam

SEQUENCE = 8
DIMENSION = 6
VOCABULARY = 10


@pytest.fixture()
def rope():
    return RopeEmbedding(sequence_length=SEQUENCE, embedding_dimension=DIMENSION)


@pytest.fixture()
def embedded_batch():
    rng = np.random.default_rng(0)
    return rng.normal(size=(3, SEQUENCE, DIMENSION))


# -------------    rotary    ---------------------------------------
def test_rope_preserves_norms(rope, embedded_batch):
    rotated = rope.forward(embedded_batch)
    assert np.allclose(
        np.linalg.norm(embedded_batch, axis=-1), np.linalg.norm(rotated, axis=-1)
    )


def test_rope_inner_product_depends_only_on_separation(rope):
    """q at position i against k at position j depends on i - j alone"""
    rng = np.random.default_rng(0)
    query = rng.normal(size=DIMENSION)
    key = rng.normal(size=DIMENSION)

    def rotated_pair(first, second):
        block = np.zeros((1, SEQUENCE, DIMENSION))
        block[0, first] = query
        rotated_query = rope.forward(block)[0, first]
        block = np.zeros((1, SEQUENCE, DIMENSION))
        block[0, second] = key
        rotated_key = rope.forward(block)[0, second]
        return float(rotated_query @ rotated_key)

    separation_two = [rotated_pair(index, index + 2) for index in range(SEQUENCE - 2)]
    assert np.allclose(separation_two, separation_two[0], atol=1e-8)


def test_rope_gradient(rope, embedded_batch):
    assert input_gradient_error(rope, embedded_batch) < GRADIENT_TOLERANCE


# -------------    sinusoid    -------------------------------------
def test_sinusoid_is_purely_additive(embedded_batch):
    """the offset depends on position alone, whatever the content"""
    sinusoid = SinusoidEmbedding(sequence_length=SEQUENCE, embedding_dimension=DIMENSION)
    offset = sinusoid.forward(embedded_batch) - embedded_batch
    assert np.allclose(offset, sinusoid.sinusoid_array)
    assert np.allclose(offset[0], offset[1])


# -------------    token lookup table    ---------------------------
def test_text_embedding_weight_gradient():
    layer = TextEmbedding(num_embeddings=VOCABULARY, embedding_dim=DIMENSION)
    token_ids = np.random.default_rng(0).integers(0, VOCABULARY, size=(3, SEQUENCE))
    upstream = np.random.default_rng(1).normal(size=token_ids.shape + (DIMENSION,))
    layer.forward(token_ids)
    layer.backward(upstream)
    numeric = numeric_gradient(
        lambda: float((layer.forward(token_ids) * upstream).sum()), layer.weights
    )
    assert relative_error(layer.gradient_weights, numeric) < GRADIENT_TOLERANCE


def test_text_embedding_padding_row_is_zero_and_frozen():
    layer = TextEmbedding(VOCABULARY, DIMENSION, padding_idx=0)
    ids = np.array([[0, 1, 0, 2]])
    assert not layer.forward(ids)[0, [0, 2]].any()
    optimizer = Adam(0.1)
    for _ in range(3):
        layer.forward(ids)
        layer.backward(np.ones((1, 4, DIMENSION)))
        optimizer.step([layer])
    assert not layer.weights[0].any()
    assert layer.weights[1].any()


@pytest.mark.parametrize("bad_ids", (np.array([VOCABULARY]), np.array([-1])))
def test_text_embedding_rejects_ids_outside_the_table(bad_ids):
    with pytest.raises(IndexError):
        TextEmbedding(VOCABULARY, DIMENSION).forward(bad_ids)
