"""
Spectre encoder-decoder example on a synthetic text corpus, in three stages.

  1. Encoder, masked language modelling. TextProcessor (polyergalio.encoders)
     tokenizes and applies BERT-style cloze masking; a Spectre encoder block
     learns to fill the masks. The block also pools to a sentence vector.
  2. Decoder training. The encoder's layers are reused in a second network
     that adds a causal SpectreDecoderAttention block. The pooled sentence
     vector conditions the decoder, which is taught to reproduce the sentence
     one token ahead of itself (teacher forcing through ShiftRight).
  3. Generation. A sentence is encoded once, then tokens are sampled one at a
     time through the decoder's Prefix-FFT cache (prefill, then decode_step),
     checked against a full causal forward over the tokens it produced.

Sentences are built from a small grammar. Held-out sentences are combinations
of words never seen together in training, so reconstructing them shows the
sentence vector carries content rather than memorised strings.

Run: python NNet_spectre_example.py
"""

import tempfile
from pathlib import Path

import numpy as np
from polyergalio.encoders.text_encoders import TextProcessor
from polyergalio.encoders.tokenizer import SentencePieceTokenizer, fit_tokenizer
from polyergalio.generators.data_generators import token_accuracy
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.embedding.embedding import TextEmbedding
from polyergalio.models.embedding.positional import RopeEmbedding
from polyergalio.models.layers.basic_layers import (
    DropoutLayer,
    FullyConnectedLayer,
    RMSNormLayer,
)
from polyergalio.models.layers.mixture_layers import PoolingLayer
from polyergalio.models.layers.operator_layers import LatentSum, MaskGather, ShiftRight
from polyergalio.models.layers.spectre_layers import (
    SpectreAttention,
    SpectreDecoderAttention,
)
from polyergalio.models.model_loss import CrossEntropyLoss
from polyergalio.models.network import Network
from polyergalio.models.optimizers import Adam

TARGET_VOCAB_SIZE = 150
SEQUENCE_LENGTH = 24
HIDDEN_DIM = 64
FFN_HIDDEN = 128
NUM_HEADS = 4
MEMORY_TOKENS = 4
DROPOUT_PROB = 0.1
MASK_PROB = 0.25
BATCH_SIZE = 32
MLM_STEPS = 300
DECODER_STEPS = 400
MLM_LEARNING_RATE = 3e-3
DECODER_LEARNING_RATE = 3e-3
HELD_OUT_FRACTION = 0.2
NUM_SHOWN = 6
LOG_EVERY = 50

DETERMINERS = ("the", "a")
ADJECTIVES = ("small", "large", "quiet", "loud", "red", "blue")
NOUNS = ("cat", "dog", "bird", "fish", "fox", "owl")
VERBS = ("runs", "jumps", "sleeps", "swims", "climbs", "watches")
ADVERBS = ("fast", "slowly", "quietly", "today")


def generate_corpus(rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """
    Every sentence the grammar can form, split into training and held-out.

    Parameters
    ----------
    rng : source of randomness for the split

    Returns
    -------
    train, held_out : sentences of the form "the red cat runs fast."
    """
    sentences = [
        f"{det} {adj} {noun} {verb} {adverb}."
        for det in DETERMINERS
        for adj in ADJECTIVES
        for noun in NOUNS
        for verb in VERBS
        for adverb in ADVERBS
    ]
    order = rng.permutation(len(sentences))
    cut = int(len(sentences) * (1 - HELD_OUT_FRACTION))
    return [sentences[i] for i in order[:cut]], [sentences[i] for i in order[cut:]]


def fit_text_tokenizer(corpus: list[str]) -> SentencePieceTokenizer:
    """Fit a SentencePiece tokenizer on the corpus and load it."""
    with tempfile.TemporaryDirectory() as workdir:
        model_path = fit_tokenizer(
            corpus,
            model_prefix=str(Path(workdir) / "spectre_tokenizer"),
            vocab_size=TARGET_VOCAB_SIZE,
            hard_vocab_limit=False,
        )
        return SentencePieceTokenizer(model_path)


def sample_texts(rng: np.random.Generator, texts: list[str]) -> list[str]:
    return [texts[i] for i in rng.integers(len(texts), size=BATCH_SIZE)]


# -------------    networks    ------------------------------------------------
def build_encoder(vocab_size: int, padding_idx: int) -> Network:
    """
    TextEmbedding -> RoPE -> Spectre block -> [pooled, MLM head].

    Parameters
    ----------
    vocab_size : the tokenizer's fitted vocabulary size
    padding_idx : id whose embedding row is fixed at zero

    Returns
    -------
    Network whose output is the MLM head, (masked positions, vocab_size)
    """
    net = Network(name="spectre_encoder", input_shape=(SEQUENCE_LENGTH,))
    embedding = net.connect(
        TextEmbedding(vocab_size, HIDDEN_DIM, padding_idx=padding_idx),
        net.input,
        name="embedding",
    )
    positional = net.connect(
        RopeEmbedding(SEQUENCE_LENGTH, HIDDEN_DIM), embedding, name="positional_emb"
    )
    prenorm = net.connect(RMSNormLayer(HIDDEN_DIM), positional, name="prenorm")
    attention = net.connect(
        SpectreAttention(
            SEQUENCE_LENGTH,
            HIDDEN_DIM,
            num_heads=NUM_HEADS,
            memory_tokens=MEMORY_TOKENS,
            use_wrm=True,
        ),
        prenorm,
        name="attention",
    )
    ffn_1 = net.connect(
        FullyConnectedLayer(HIDDEN_DIM, FFN_HIDDEN, "swish"), attention, name="ffn_1"
    )
    ffn_2 = net.connect(
        FullyConnectedLayer(FFN_HIDDEN, HIDDEN_DIM, "swish"), ffn_1, name="ffn_2"
    )
    postnorm = net.connect(RMSNormLayer(HIDDEN_DIM), ffn_2, name="postnorm")
    dropped = net.connect(DropoutLayer(DROPOUT_PROB), postnorm, name="dropout")
    block_out = net.connect(LatentSum(), dropped, embedding, name="residual")

    net.connect(PoolingLayer(), block_out, name="pooled")
    gathered = net.connect(MaskGather(), block_out, name="mask_gather")
    net.output = net.connect(
        FullyConnectedLayer(HIDDEN_DIM, vocab_size, "linear", is_output=True),
        gathered,
        name="mlm_head",
    )
    return net


ENCODER_CHAIN = ("positional_emb", "prenorm", "attention", "ffn_1", "ffn_2", "postnorm", "dropout")


def build_seq2seq(encoder: Network, vocab_size: int) -> Network:
    """
    The trained encoder's layers plus a causal decoder, in one graph.

    The encoder reads the sentence, and its pooled vector is projected and
    added to every decoder input. The decoder reads the same sentence shifted
    one position right, so position t sees only tokens before t plus the
    sentence vector, and predicts token t.

    Parameters
    ----------
    encoder : network from build_encoder; its layers are shared, not copied
    vocab_size : the tokenizer's fitted vocabulary size

    Returns
    -------
    Network whose output is (batch, sequence, vocab_size) logits
    """
    net = Network(name="spectre_seq2seq", input_shape=(SEQUENCE_LENGTH,))
    embedding = net.connect(encoder.node("embedding").layer, net.input, name="embedding")

    stream = embedding
    for name in ENCODER_CHAIN:
        stream = net.connect(encoder.node(name).layer, stream, name=name)
    block_out = net.connect(
        encoder.node("residual").layer, stream, embedding, name="residual"
    )
    pooled = net.connect(encoder.node("pooled").layer, block_out, name="pooled")

    shifted = net.connect(ShiftRight(HIDDEN_DIM), embedding, name="shift_right")
    latent = net.connect(
        FullyConnectedLayer(HIDDEN_DIM, HIDDEN_DIM, "linear"), pooled, name="latent_projection"
    )
    conditioned = net.connect(LatentSum(), shifted, latent, name="conditioned")

    prenorm = net.connect(RMSNormLayer(HIDDEN_DIM), conditioned, name="decoder_prenorm")
    attention = net.connect(
        SpectreDecoderAttention(
            SEQUENCE_LENGTH, HIDDEN_DIM, num_heads=NUM_HEADS, memory_tokens=MEMORY_TOKENS
        ),
        prenorm,
        name="decoder_attention",
    )
    ffn_1 = net.connect(
        FullyConnectedLayer(HIDDEN_DIM, FFN_HIDDEN, "swish"), attention, name="decoder_ffn_1"
    )
    ffn_2 = net.connect(
        FullyConnectedLayer(FFN_HIDDEN, HIDDEN_DIM, "swish"), ffn_1, name="decoder_ffn_2"
    )
    postnorm = net.connect(RMSNormLayer(HIDDEN_DIM), ffn_2, name="decoder_postnorm")
    dropped = net.connect(DropoutLayer(DROPOUT_PROB), postnorm, name="decoder_dropout")
    hidden = net.connect(LatentSum(), dropped, conditioned, name="decoder_residual")
    net.output = net.connect(
        FullyConnectedLayer(HIDDEN_DIM, vocab_size, "linear", is_output=True),
        hidden,
        name="lm_head",
    )
    return net


# -------------    stage 1: encoder MLM    ------------------------------------
def mlm_accuracy(
    net: Network, processor: TextProcessor, texts: list[str]
) -> float:
    """Accuracy at masked positions of a cloze batch, with the network in eval mode."""
    batch = processor.distort_batch(texts, "cloze")
    net.eval()
    logits = net.forward(
        batch["input_ids"], mask=batch["attention_mask"], target_mask=batch["target_mask"]
    )
    return token_accuracy(np.argmax(logits, axis=-1), batch["labels"][batch["target_mask"]])


def train_mlm(
    net: Network,
    processor: TextProcessor,
    texts: list[str],
    held_out: list[str],
    rng: np.random.Generator,
) -> None:
    """Masked language modelling with a fresh cloze distortion every step."""
    vocab_size = processor.vocab_size
    loss_fn = CrossEntropyLoss(task=ClassificationTask.MULTINOMIAL)
    optimizer = Adam(MLM_LEARNING_RATE)

    print(f"masked-token accuracy before training: {mlm_accuracy(net, processor, held_out[:BATCH_SIZE]):.3f}")
    net.train()
    for step in range(MLM_STEPS):
        batch = processor.distort_batch(sample_texts(rng, texts), "cloze")
        labels = batch["labels"][batch["target_mask"]]
        net.zero_gradients()
        logits = net.forward(
            batch["input_ids"], mask=batch["attention_mask"], target_mask=batch["target_mask"]
        )
        loss = loss_fn(logits, np.eye(vocab_size)[labels])
        net.backward(loss_fn.backward())
        optimizer.step(net)
        if step % LOG_EVERY == 0:
            print(f"mlm step {step:4d}  loss {loss:.4f}")
    print(f"masked-token accuracy after training:  {mlm_accuracy(net, processor, held_out[:BATCH_SIZE]):.3f}")


# -------------    stage 2: decoder training    -------------------------------
def encode_batch(processor: TextProcessor, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """[CLS] text [SEP] ids and their attention mask, padded to SEQUENCE_LENGTH."""
    encoded = processor.encode(texts)
    return encoded[processor.target], encoded[f"{processor.target}_attention_mask"]


def decoder_accuracy(
    net: Network, processor: TextProcessor, texts: list[str]
) -> float:
    """Teacher-forced next-token accuracy over real positions, in eval mode."""
    ids, attention_mask = encode_batch(processor, texts)
    net.eval()
    logits = net.forward(ids, mask=attention_mask)
    return token_accuracy(np.argmax(logits, axis=-1), ids, attention_mask.astype(bool))


def train_decoder(
    net: Network,
    processor: TextProcessor,
    texts: list[str],
    held_out: list[str],
    rng: np.random.Generator,
) -> None:
    """
    Next-token training of the decoder through the sentence vector. The whole
    graph trains, so the encoder is fine-tuned by the decoder's gradient.
    """
    vocab_size = processor.vocab_size
    loss_fn = CrossEntropyLoss(task=ClassificationTask.MULTINOMIAL)
    optimizer = Adam(DECODER_LEARNING_RATE)

    print(f"decoder accuracy before training (held out): {decoder_accuracy(net, processor, held_out[:BATCH_SIZE]):.3f}")
    net.train()
    for step in range(DECODER_STEPS):
        ids, attention_mask = encode_batch(processor, sample_texts(rng, texts))
        net.zero_gradients()
        logits = net.forward(ids, mask=attention_mask)
        loss = loss_fn(logits, np.eye(vocab_size)[ids], attention_mask)
        net.backward(loss_fn.backward())
        optimizer.step(net)
        if step % LOG_EVERY == 0:
            print(f"decoder step {step:4d}  loss {loss:.4f}")
    print(f"decoder accuracy after training (held out):  {decoder_accuracy(net, processor, held_out[:BATCH_SIZE]):.3f}")


# -------------    stage 3: generation    -------------------------------------
def decoder_layers(net: Network) -> dict:
    """The layers generation drives by hand, by node name."""
    names = (
        "embedding", "shift_right", "latent_projection", "decoder_prenorm",
        "decoder_attention", "decoder_ffn_1", "decoder_ffn_2", "decoder_postnorm", "lm_head",
    )
    return {name: net.node(name).layer for name in names}


def decoder_step(layers: dict, step_input: np.ndarray, position: int, valid: np.ndarray) -> np.ndarray:
    """
    One token through the decoder block, the attention reading and extending
    its Prefix-FFT cache.

    Parameters
    ----------
    layers : from decoder_layers
    step_input : (batch, hidden) previous token's embedding plus the sentence vector
    position : 0 fills the cache with the first token, later positions append to it
    valid : (batch,) False for rows that already finished

    Returns
    -------
    (batch, vocab_size) logits for this position
    """
    normed = layers["decoder_prenorm"].forward(step_input)
    attention = layers["decoder_attention"]
    if position == 0:
        mixed = attention.prefill(normed[:, None, :])
    else:
        mixed = attention.decode_step(normed, valid=valid)
    hidden = layers["decoder_ffn_2"].forward(layers["decoder_ffn_1"].forward(mixed))
    hidden = layers["decoder_postnorm"].forward(hidden) + step_input
    return layers["lm_head"].forward(hidden)


def teacher_forced_logits(layers: dict, latent: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """The training-time decoder forward over a whole token sequence, for checking the cache."""
    shifted = layers["shift_right"].forward(layers["embedding"].forward(tokens))
    conditioned = shifted + latent
    mask = (tokens != 0).astype(float)
    mixed = layers["decoder_attention"].forward(
        layers["decoder_prenorm"].forward(conditioned), mask=mask
    )
    hidden = layers["decoder_ffn_2"].forward(layers["decoder_ffn_1"].forward(mixed))
    hidden = layers["decoder_postnorm"].forward(hidden) + conditioned
    return layers["lm_head"].forward(hidden)


def sample_tokens(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> np.ndarray:
    """Greedy at temperature 0, otherwise a draw from softmax(logits / temperature)."""
    if temperature <= 0:
        return np.argmax(logits, axis=-1)
    scaled = logits / temperature
    probabilities = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.array([rng.choice(row.size, p=row) for row in probabilities])


def generate(
    encoder: Network,
    seq2seq: Network,
    processor: TextProcessor,
    texts: list[str],
    temperature: float = 0.0,
    rng: np.random.Generator = None,
) -> dict:
    """
    Encode each text once, then generate tokens one at a time from the
    sentence vector until [SEP] or SEQUENCE_LENGTH.

    Parameters
    ----------
    encoder : network from build_encoder
    seq2seq : network from build_seq2seq, sharing the encoder's layers
    processor : TextProcessor for the tokenizer's special ids
    texts : sentences to encode
    temperature : 0 for greedy decoding
    rng : source of randomness when temperature > 0

    Returns
    -------
    dict with tokens (batch, SEQUENCE_LENGTH), lengths (batch,), latent
    (batch, hidden), the per-position logits and the maximum difference
    between them and a full causal forward over the generated tokens
    """
    layers = decoder_layers(seq2seq)
    special = processor.special
    ids, attention_mask = encode_batch(processor, texts)
    batch = ids.shape[0]

    encoder.eval()
    seq2seq.eval()
    encoder.forward(ids, mask=attention_mask, target_mask=attention_mask)
    latent = layers["latent_projection"].forward(encoder.activations["pooled"])

    start = layers["shift_right"].start_token[0, 0]
    step_input = np.broadcast_to(start, (batch, HIDDEN_DIM)) + latent[:, 0]
    tokens = np.full((batch, SEQUENCE_LENGTH), special.PAD)
    finished = np.zeros(batch, dtype=bool)
    lengths = np.full(batch, SEQUENCE_LENGTH)
    logits_seen = []

    for position in range(SEQUENCE_LENGTH):
        logits = decoder_step(layers, step_input, position, ~finished)
        logits_seen.append(logits)
        chosen = sample_tokens(logits, temperature, rng)
        chosen[finished] = special.PAD
        tokens[:, position] = chosen
        just_finished = ~finished & (chosen == special.SEP)
        lengths[just_finished] = position + 1
        finished |= just_finished
        if finished.all():
            break
        step_input = layers["embedding"].forward(chosen) + latent[:, 0]

    cached = np.stack(logits_seen, axis=1)
    full = teacher_forced_logits(layers, latent, tokens)[:, : cached.shape[1]]
    live = np.arange(cached.shape[1])[None, :] < lengths[:, None]
    difference = np.abs(cached - full)[live].max()
    return {
        "tokens": tokens,
        "lengths": lengths,
        "latent": latent[:, 0],
        "logits": cached,
        "cache_difference": difference,
    }


def show_generation(processor: TextProcessor, texts: list[str], result: dict) -> None:
    """Prints each sentence with what was generated from its vector, and the first one's token trace."""
    tokenizer = processor.tokenizer
    exact = 0
    for row, text in enumerate(texts):
        produced = tokenizer.decode(result["tokens"][row, : result["lengths"][row]].tolist())
        exact += produced == text
        print(f"  {'ok ' if produced == text else 'off'} in:  {text}\n      out: {produced}")
    print(f"exact reconstructions: {exact}/{len(texts)}")

    print("\ntoken by token for the first sentence (piece, probability):")
    logits = result["logits"][0]
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    for position in range(result["lengths"][0]):
        token = int(result["tokens"][0, position])
        piece = tokenizer.decode([token], skip_special=False)
        print(f"  {position:2d}  {piece:<8s} {probabilities[position, token]:.3f}")


def main():
    rng = np.random.default_rng(0)
    train_texts, held_out = generate_corpus(rng)
    print(f"{len(train_texts)} training sentences, {len(held_out)} held out")

    tokenizer = fit_text_tokenizer(train_texts)
    vocab_size = tokenizer.get_vocab_size()
    print(f"requested vocab_size={TARGET_VOCAB_SIZE}, corpus fit {vocab_size}")

    processor = TextProcessor(tokenizer, max_length=SEQUENCE_LENGTH, tasks=("cloze",), mask_prob=MASK_PROB, random_seed=0)
    processor.fit(train_texts)

    encoder = build_encoder(vocab_size, padding_idx=tokenizer.special_tokens.PAD)
    print(encoder.summary())

    print("\n--- stage 1: encoder, masked language modelling ---")
    train_mlm(encoder, processor, train_texts, held_out, rng)

    seq2seq = build_seq2seq(encoder, vocab_size)
    print("\n--- stage 2: decoder training ---")
    print(seq2seq.summary())
    train_decoder(seq2seq, processor, train_texts, held_out, rng)

    print("\n--- stage 3: generation from held-out sentences ---")
    shown = held_out[:NUM_SHOWN]
    result = generate(encoder, seq2seq, processor, shown)
    show_generation(processor, shown, result)
    print(f"\ncached decoding vs full causal forward, max logit difference: {result['cache_difference']:.2e}")

    print("\nsampled at temperature 0.8 from the first sentence's vector:")
    repeated = [shown[0]] * 4
    sampled = generate(encoder, seq2seq, processor, repeated, temperature=0.8, rng=rng)
    for row in range(len(repeated)):
        print(f"  {tokenizer.decode(sampled['tokens'][row, : sampled['lengths'][row]].tolist())}")


if __name__ == "__main__":
    main()
