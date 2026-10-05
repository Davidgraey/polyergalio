"""
Spectre transformer encoder into a DecisionHead, on real text.

No `SpectreTransformer` class exists in the codebase yet -- "the Spectre
transformer" here means the same hand-assembled block
`NNet_spectre_example.py` builds: TextEmbedding, RoPE, a SpectreAttention
mixing pass, a feed-forward, and a residual, wired through Network's
graph API. Two stages:

  1. Encoder pretraining. TextProcessor (polyergalio.encoders.text_encoders)
     applies BERT-style cloze masking on a small real-text corpus; the block
     learns to fill the masks.
  2. Decision training. The pretrained block's layers are reused (shared,
     not copied) in a second network with a DecisionHead in place of the MLM
     head. Real-text BINARY / CHOICE / SCORE questions about a short passage
     -- does it mention a word, which of these words appears, how many times
     does a word appear -- are laid out by tokenizer.py's
     TokenSequenceBuilder and tokenized by the same fitted SentencePiece
     tokenizer the encoder was pretrained with. Questions are drawn as train,
     calibration and test sets: the head trains on the first (main loss plus
     the act branch), temperatures are fit on the second, and the third is
     decoded with calibrated probabilities, scored for confidence and
     escalated by the act branch.

Run: python NNet_spectre_decision_example.py
"""

import tempfile
from pathlib import Path

import numpy as np

from NNet_system_one_decision_example import evaluate, fit_calibration, print_evaluation
from polyergalio.encoders.text_encoders import TextProcessor
from polyergalio.encoders.tokenizer import SentencePieceTokenizer, TokenSequenceBuilder, fit_tokenizer
from polyergalio.generators.data_generators import token_accuracy
from polyergalio.models.constants import DECISION_TYPES, ClassificationTask
from polyergalio.models.embedding.embedding import TextEmbedding
from polyergalio.models.embedding.positional import RopeEmbedding
from polyergalio.models.layers.basic_layers import DropoutLayer, FullyConnectedLayer, RMSNormLayer
from polyergalio.models.layers.decision_layers import (
    DecisionHead,
    decision_correct,
    decode_decisions,
    masked_softmax,
)
from polyergalio.models.layers.operator_layers import LatentSum, MaskGather
from polyergalio.models.layers.spectre_layers import SpectreAttention
from polyergalio.models.model_loss import CrossEntropyLoss, DecisionLoss
from polyergalio.models.network import Network
from polyergalio.models.optimizers import SGD, Adam

TARGET_VOCAB_SIZE = 150
SEQUENCE_LENGTH = 40
HIDDEN_DIM = 32
FFN_HIDDEN = 64
NUM_HEADS = 4
HEAD_HIDDEN = 16
DROPOUT_PROB = 0.1
MASK_PROB = 0.25
CORPUS_SIZE = 400
HELD_OUT_FRACTION = 0.2
BATCH_SIZE = 32
MLM_STEPS = 200
MLM_LEARNING_RATE = 3e-3
NUM_CHOICES = 4
NUM_LEVELS = 4
TRAIN_SAMPLES = 256
CALIBRATION_SAMPLES = 128
TEST_SAMPLES = 128
DECISION_STEPS = 500
DECISION_LEARNING_RATE = 0.05
LOG_EVERY = 50

ITEMS = ("apple", "cat", "dog", "fish", "fox", "leaf", "moon", "rose")
DECISION_TYPE_VALUES = [kind.value for kind in DECISION_TYPES]

INSTRUCTION_EXAMPLES = (
    *(f"does the list mention {item}?" for item in ITEMS),
    *(f"how many times does {item} appear in the list?" for item in ITEMS),
    "which of these words is in the list?",
)


def render_passage(words: list[str]) -> str:
    """A word list as one plain-English sentence, the corpus's only shape."""
    return "the list today is: " + ", ".join(words) + "."


def make_passages(rng: np.random.Generator, count: int, min_len: int = 4, max_len: int = 10) -> list[str]:
    """`count` passages of random ITEMS words, for tokenizer fitting and MLM."""
    return [
        render_passage(list(rng.choice(ITEMS, size=int(rng.integers(min_len, max_len + 1)))))
        for _ in range(count)
    ]


def fit_text_tokenizer(corpus: list[str]) -> SentencePieceTokenizer:
    """Fit a SentencePiece tokenizer on the corpus and load it."""
    with tempfile.TemporaryDirectory() as workdir:
        model_path = fit_tokenizer(
            corpus,
            model_prefix=str(Path(workdir) / "decision_tokenizer"),
            vocab_size=TARGET_VOCAB_SIZE,
            hard_vocab_limit=False,
        )
        return SentencePieceTokenizer(model_path)


def sample_texts(rng: np.random.Generator, texts: list[str]) -> list[str]:
    return [texts[i] for i in rng.integers(len(texts), size=BATCH_SIZE)]


# -------------    networks    ------------------------------------------------
def build_encoder(vocab_size: int, padding_idx: int) -> Network:
    """
    TextEmbedding -> RoPE -> Spectre transformer block -> MLM head.

    Parameters
    ----------
    vocab_size : the tokenizer's fitted vocabulary size
    padding_idx : id whose embedding row is fixed at zero

    Returns
    -------
    Network whose output is the MLM head, (masked positions, vocab_size)
    """
    net = Network(name="spectre_transformer", input_shape=(SEQUENCE_LENGTH,))
    embedding = net.connect(
        TextEmbedding(vocab_size, HIDDEN_DIM, padding_idx=padding_idx), net.input, name="embedding"
    )
    positional = net.connect(RopeEmbedding(SEQUENCE_LENGTH, HIDDEN_DIM), embedding, name="positional_emb")
    prenorm = net.connect(RMSNormLayer(HIDDEN_DIM), positional, name="prenorm")
    attention = net.connect(
        SpectreAttention(SEQUENCE_LENGTH, HIDDEN_DIM, num_heads=NUM_HEADS), prenorm, name="attention"
    )
    ffn_1 = net.connect(FullyConnectedLayer(HIDDEN_DIM, FFN_HIDDEN, "swish"), attention, name="ffn_1")
    ffn_2 = net.connect(FullyConnectedLayer(FFN_HIDDEN, HIDDEN_DIM, "swish"), ffn_1, name="ffn_2")
    postnorm = net.connect(RMSNormLayer(HIDDEN_DIM), ffn_2, name="postnorm")
    dropped = net.connect(DropoutLayer(DROPOUT_PROB), postnorm, name="dropout")
    block_out = net.connect(LatentSum(), dropped, embedding, name="residual")

    gathered = net.connect(MaskGather(), block_out, name="mask_gather")
    net.output = net.connect(
        FullyConnectedLayer(HIDDEN_DIM, vocab_size, "linear", is_output=True), gathered, name="mlm_head"
    )
    return net


ENCODER_CHAIN = ("positional_emb", "prenorm", "attention", "ffn_1", "ffn_2", "postnorm", "dropout")


def build_decision_network(encoder: Network) -> Network:
    """
    The pretrained Spectre transformer block's layers plus a DecisionHead,
    replacing the MLM head.

    Parameters
    ----------
    encoder : network from build_encoder, already trained; its layers are
        shared with this new graph, not copied

    Returns
    -------
    Network mapping token ids (batch, SEQUENCE_LENGTH) to option logits
    """
    net = Network(name="spectre_decision", input_shape=(SEQUENCE_LENGTH,))
    embedding = net.connect(encoder.node("embedding").layer, net.input, name="embedding")
    stream = embedding
    for name in ENCODER_CHAIN:
        stream = net.connect(encoder.node(name).layer, stream, name=name)
    block_out = net.connect(encoder.node("residual").layer, stream, embedding, name="residual")
    net.output = net.connect(DecisionHead(HIDDEN_DIM, HEAD_HIDDEN), block_out, name="decision_head")
    return net


# -------------    stage 1: encoder MLM    ------------------------------------
def mlm_accuracy(net: Network, processor: TextProcessor, texts: list[str]) -> float:
    """Accuracy at masked positions of a cloze batch, with the network in eval mode."""
    batch = processor.distort_batch(texts, "cloze")
    net.eval()
    logits = net.forward(batch["input_ids"], mask=batch["attention_mask"], target_mask=batch["target_mask"])
    return token_accuracy(np.argmax(logits, axis=-1), batch["labels"][batch["target_mask"]])


def train_mlm(
    net: Network, processor: TextProcessor, texts: list[str], held_out: list[str], rng: np.random.Generator
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
        logits = net.forward(batch["input_ids"], mask=batch["attention_mask"], target_mask=batch["target_mask"])
        loss = loss_fn(logits, np.eye(vocab_size)[labels])
        net.backward(loss_fn.backward())
        optimizer.step(net)
        if step % LOG_EVERY == 0:
            print(f"mlm step {step:4d}  loss {loss:.4f}")
    print(f"masked-token accuracy after training:  {mlm_accuracy(net, processor, held_out[:BATCH_SIZE]):.3f}")


# -------------    stage 2: real-text decision questions    -------------------
def make_decision_sample(rng: np.random.Generator, kind: DECISION_TYPES, length: int):
    """
    One real-text question: instructions, its options (None for BINARY and
    SCORE, which TokenSequenceBuilder fills in itself), the passage state
    text, and the answer option index.

    CHOICE plants exactly one of `NUM_CHOICES` candidate words in the
    passage; BINARY and SCORE plant a query word 0..1 or 0..NUM_LEVELS-1
    times, so the answer is exact by construction.
    """
    if kind == DECISION_TYPES.CHOICE:
        candidates = list(rng.choice(ITEMS, size=NUM_CHOICES, replace=False))
        answer = int(rng.integers(NUM_CHOICES))
        pool = [item for item in ITEMS if item not in candidates]
        filler = list(rng.choice(pool, size=length))
        filler.insert(int(rng.integers(len(filler) + 1)), candidates[answer])
        return "which of these words is in the list?", candidates, render_passage(filler), answer

    query = str(rng.choice(ITEMS))
    pool = [item for item in ITEMS if item != query]
    filler = list(rng.choice(pool, size=length))
    if kind == DECISION_TYPES.SCORE:
        answer = int(rng.integers(NUM_LEVELS))
        instructions = f"how many times does {query} appear in the list?"
    else:
        answer = int(rng.integers(2))
        instructions = f"does the list mention {query}?"
    for _ in range(answer):
        filler.insert(int(rng.integers(len(filler) + 1)), query)
    return instructions, None, render_passage(filler), answer


def make_decision_batch(
    rng: np.random.Generator,
    builder: TokenSequenceBuilder,
    tokenizer: SentencePieceTokenizer,
    count: int,
    min_len: int = 4,
    max_len: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    `count` real-text decision questions, laid out by `builder` and padded
    to SEQUENCE_LENGTH and the batch's widest option count.

    Returns
    -------
    X : (count, SEQUENCE_LENGTH) padded token ids
    y : (count,) answer option index
    kwargs : DecisionHead's forward kwargs (marker_pos, marker_end,
        token_mask, decisiontypes) plus `mask` for the encoder
    """
    kinds = rng.choice(DECISION_TYPE_VALUES, size=count)
    rows, markers, ends, answers, decisiontypes = [], [], [], [], []
    for kind_value in kinds:
        kind = DECISION_TYPES(int(kind_value))
        length = int(rng.integers(min_len, max_len + 1))
        instructions, options, state, answer = make_decision_sample(rng, kind, length)
        if kind == DECISION_TYPES.BINARY:
            ids, positions, span_ends = builder.build_binary_decision(instructions, state, max_length=SEQUENCE_LENGTH)
        elif kind == DECISION_TYPES.CHOICE:
            ids, positions, span_ends = builder.build_choice_decision(instructions, options, state, max_length=SEQUENCE_LENGTH)
        else:
            ids, positions, span_ends = builder.build_score_decision(instructions, NUM_LEVELS, state, max_length=SEQUENCE_LENGTH)
        rows.append(ids)
        markers.append(positions)
        ends.append(span_ends)
        answers.append(answer)
        decisiontypes.append(int(kind_value))

    num_options = max(len(positions) for positions in markers)
    pad_id = tokenizer.special_tokens.PAD
    X = np.full((count, SEQUENCE_LENGTH), pad_id, dtype=int)
    marker_pos = np.zeros((count, num_options), dtype=int)
    marker_end = np.zeros((count, num_options), dtype=int)
    token_mask = np.zeros((count, num_options), dtype=bool)
    for i, (ids, positions, span_ends) in enumerate(zip(rows, markers, ends)):
        X[i, : len(ids)] = ids
        marker_pos[i, : len(positions)] = positions
        marker_end[i, : len(span_ends)] = span_ends
        token_mask[i, : len(positions)] = True

    kwargs = dict(
        mask=(X != pad_id),
        marker_pos=marker_pos,
        marker_end=marker_end,
        token_mask=token_mask,
        decisiontypes=np.array(decisiontypes, dtype=int),
    )
    return X, np.array(answers, dtype=int), kwargs


def row_correctness(logits: np.ndarray, y: np.ndarray, kwargs: dict) -> np.ndarray:
    """(batch,) 1.0 where the decoded answer matches the planted answer."""
    probabilities = masked_softmax(logits, kwargs["token_mask"])
    decisions = decode_decisions(probabilities, kwargs["decisiontypes"])
    return decision_correct(decisions, y, kwargs["decisiontypes"])


def decision_accuracy(logits: np.ndarray, y: np.ndarray, kwargs: dict) -> float:
    return float(row_correctness(logits, y, kwargs).mean())


def accuracy_by_type(logits: np.ndarray, y: np.ndarray, kwargs: dict) -> dict[str, float]:
    """Decision accuracy broken out per DECISION_TYPES member present in the batch."""
    correctness = row_correctness(logits, y, kwargs)
    return {
        kind.name: float(correctness[rows].mean())
        for kind in DECISION_TYPES
        if (rows := kwargs["decisiontypes"] == kind.value).any()
    }


def train_decision(net: Network, train_set: tuple, calibration_set: tuple, test_set: tuple) -> None:
    """
    SGD over the train questions with DecisionLoss plus the act branch, then
    temperatures fit on the calibration questions and calibrated decoding,
    confidence and escalation reported on the test questions.

    Parameters
    ----------
    net : encoder plus DecisionHead network
    train_set, calibration_set, test_set : (X, y, kwargs) from make_decision_batch
    """
    X, y, kwargs = train_set
    x_test, y_test, test_kwargs = test_set
    net.eval()
    logits = net.forward(x_test, **test_kwargs)
    print(f"test accuracy before training: {decision_accuracy(logits, y_test, test_kwargs):.3f}")

    loss_fn = DecisionLoss(ordinal_weight=0.25)
    optimizer = SGD(DECISION_LEARNING_RATE)
    head = net.node("decision_head").layer

    net.train()
    for step in range(DECISION_STEPS):
        net.zero_gradients()
        logits = net.forward(X, **kwargs)
        loss = loss_fn(logits, y, kwargs["token_mask"], kwargs["decisiontypes"])
        act_loss = head.score_act(row_correctness(logits, y, kwargs))
        net.backward(loss_fn.backward())
        optimizer.step(net)
        if step % LOG_EVERY == 0:
            print(f"decision step {step:4d}  loss {loss:.4f}  act loss {act_loss:.4f}")

    temperatures = fit_calibration(net, *calibration_set)
    print("\n--- test questions ---")
    results = evaluate(net, x_test, y_test, test_kwargs, temperatures)
    print_evaluation(results, temperatures)
    for name, value in accuracy_by_type(net.forward(x_test, **test_kwargs), y_test, test_kwargs).items():
        print(f"  {name}: {value:.3f}")


def main():
    rng = np.random.default_rng(0)
    passages = make_passages(rng, CORPUS_SIZE)
    order = rng.permutation(len(passages))
    cut = int(len(passages) * (1 - HELD_OUT_FRACTION))
    train_texts = [passages[i] for i in order[:cut]] + list(INSTRUCTION_EXAMPLES)
    held_out_texts = [passages[i] for i in order[cut:]]
    print(f"{len(train_texts)} training texts, {len(held_out_texts)} held out")

    tokenizer = fit_text_tokenizer(train_texts)
    vocab_size = tokenizer.get_vocab_size()
    print(f"requested vocab_size={TARGET_VOCAB_SIZE}, corpus fit {vocab_size}")

    processor = TextProcessor(tokenizer, max_length=SEQUENCE_LENGTH, tasks=("cloze",), mask_prob=MASK_PROB, random_seed=0)
    processor.fit(train_texts)

    encoder = build_encoder(vocab_size, padding_idx=tokenizer.special_tokens.PAD)
    print(encoder.summary())

    print("\n--- stage 1: Spectre transformer encoder, masked language modelling ---")
    train_mlm(encoder, processor, train_texts, held_out_texts, rng)

    decision_net = build_decision_network(encoder)
    print("\n--- stage 2: DecisionHead, real-text BINARY / CHOICE / SCORE questions ---")
    print(decision_net.summary())

    builder = TokenSequenceBuilder(tokenizer)
    train_decision(
        decision_net,
        make_decision_batch(rng, builder, tokenizer, TRAIN_SAMPLES),
        make_decision_batch(rng, builder, tokenizer, CALIBRATION_SAMPLES),
        make_decision_batch(rng, builder, tokenizer, TEST_SAMPLES),
    )


if __name__ == "__main__":
    main()
