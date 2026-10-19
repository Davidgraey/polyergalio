"""
Spectre pooled-attention example: classify a sequence from one summary vector, and profile the cost.

SpectrePooledAttention returns (batch, 1, hidden) from (batch, sequence, hidden): a spectral gate followed by a
learned positional readout. It is compared with SpectreAttention
followed by mean pooling, and for the text task also with reading the [CLS] position as BERT does.

Tasks
    next sentence       Generated stories (a subject carrying out one of four six-step routines). TextProcessor builds
                        [CLS] A [SEP] B [SEP]; B is A's successor or a random corpus sentence. Needs which tokens
                        belong to A and B and the routine's order. Held-out stories use unseen subjects.
    marker position     Random tokens, one marker; the label is whether it lies in the first half of the real tokens.
    marker, fixed       The same without padding. Mean pooling of a Spectre output keeps only the gate's zero
                        frequency there, so SpectreAttention + mean pool stays at chance.

Profile
    forward and backward time and memory of the mixers alone, on random input, with the Profiler.

Run: python NNet_spectre_pooled_example.py
"""

import gc
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from polyergalio.encoders.text_encoders import TextProcessor
from polyergalio.encoders.tokenizer import SentencePieceTokenizer, fit_tokenizer
from polyergalio.models.embedding.embedding import TextEmbedding
from polyergalio.models.layers.basic_layers import FullyConnectedLayer, RMSNormLayer
from polyergalio.models.layers.mixture_layers import PoolingLayer
from polyergalio.models.layers.operator_layers import MaskGather
from polyergalio.models.layers.spectre_layers import SpectreAttention
from polyergalio.models.layers import SpectrePooledAttention
from polyergalio.models.model_loss import SparseCrossEntropyLoss
from polyergalio.models.network import Network
from polyergalio.models.optimizers import Adam
from polyergalio.profiling import Profiler

TARGET_VOCAB_SIZE = 200
SEQUENCE_LENGTH = 32
HIDDEN_DIM = 32
FFN_HIDDEN = 64
NUM_HEADS = 4
BATCH_SIZE = 32
STEPS = 800
LEARNING_RATE = 3e-3
HELD_OUT_FRACTION = 0.25
EVAL_PAIRS = 1024
SEEDS = (0, 1, 2)
MARKER_VALUE = 4.0

PROFILE_LENGTH = 500
PROFILE_HIDDEN = 960
PROFILE_HEADS = 12
PROFILE_BATCH = 4

DETERMINERS = ("the", "a")
ADJECTIVES = ("small", "large", "quiet", "loud", "red", "blue")
NOUNS = ("cat", "dog", "bird", "fish", "fox", "owl")
ROUTINES = (
    ("wakes up", "eats breakfast", "walks to the market", "buys some bread", "returns home", "goes to sleep"),
    ("opens the shop", "sweeps the floor", "greets a customer", "counts the coins", "closes the shop", "walks home"),
    ("finds a seed", "digs a hole", "plants the seed", "waters the ground", "waits for rain", "picks a flower"),
    ("hears a noise", "looks around", "climbs the tree", "spots a nest", "feeds the chicks", "flies away"),
)

TASKS = ("next sentence", "marker position", "marker position, fixed length")
MODELS = {
    "spectre_mean": "Spectre + mean pool",
    "spectre_cls": "Spectre + CLS",
    "pooled": "SpectrePooled",
}


def models_for(name: str) -> dict:
    """The models that apply to a task; the [CLS] read needs a [CLS] token, which only the text task has."""
    if name == "next sentence":
        return dict(MODELS)
    return {kind: label for kind, label in MODELS.items() if kind != "spectre_cls"}


@dataclass
class Task:
    """What a comparison needs: the models that apply, how to build one, and where batches come from."""

    kinds: dict
    build: Callable
    draw: Callable
    eval_batch: dict


# -------------    data    ----------------------------------------------------
def generate_stories(rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """
    One story per subject and routine, split by subject.

    Returns
    -------
    train, held_out : stories whose sentences are "the red cat wakes up." joined by spaces
    """
    subjects = [f"{det} {adj} {noun}" for det in DETERMINERS for adj in ADJECTIVES for noun in NOUNS]
    order = rng.permutation(len(subjects))
    cut = int(len(subjects) * (1 - HELD_OUT_FRACTION))

    def tell(chosen: np.ndarray) -> list[str]:
        return [" ".join(f"{subjects[i]} {step}." for step in routine) for i in chosen for routine in ROUTINES]

    return tell(order[:cut]), tell(order[cut:])


def fit_text_tokenizer(corpus: list[str]) -> SentencePieceTokenizer:
    """Fit a SentencePiece tokenizer on the corpus and load it."""
    with tempfile.TemporaryDirectory() as workdir:
        model_path = fit_tokenizer(
            corpus,
            model_prefix=str(Path(workdir) / "pooled_tokenizer"),
            vocab_size=TARGET_VOCAB_SIZE,
            hard_vocab_limit=False,
        )
        return SentencePieceTokenizer(model_path)


def sentence_batch(processor: TextProcessor, texts: list[str]) -> dict:
    """A next_sentence batch as inputs, masks and is-next labels."""
    batch = processor.distort_batch(texts, "next_sentence")
    return {
        "inputs": batch["input_ids"],
        "mask": batch["attention_mask"],
        "target_mask": batch["target_mask"],
        "labels": batch["labels"][:, 0],
    }


def marker_batch(rng: np.random.Generator, count: int, length: int, hidden: int, fixed_length: bool) -> dict:
    """
    Random tokens with one marker; the label is 1 if the marker is in the first half of the real tokens.

    Parameters
    ----------
    count : sequences in the batch
    length, hidden : sequence length and feature width
    fixed_length : no padding if True, otherwise lengths between 5/8 of length and length
    """
    x = rng.normal(size=(count, length, hidden))
    x[:, :, 0] *= 0.5
    low = max(2, int(length * 5 / 8))
    lengths = np.full(count, length) if fixed_length else rng.integers(low, length + 1, size=count)
    mask = (np.arange(length)[None] < lengths[:, None]).astype(float)
    position = (rng.random(count) * lengths).astype(int)
    x[np.arange(count), position, 0] = MARKER_VALUE
    return {
        "inputs": x * mask[..., None],
        "mask": mask,
        "target_mask": None,
        "labels": (position < lengths / 2).astype(int),
    }


# -------------    networks    ------------------------------------------------
def build_mixer(
    net: Network, kind: str, source, length: int, hidden: int, heads: int
):
    """Connect the mixer for kind to source and return the node holding one summary vector."""
    if kind in ("spectre_mean", "spectre_cls"):
        mixed = net.connect(
            SpectreAttention(length, hidden, num_heads=heads, use_wrm=False), source, name="mixer"
        )
        summary_layer = PoolingLayer() if kind == "spectre_mean" else MaskGather()
        return net.connect(summary_layer, mixed, name="summary")
    return net.connect(
        SpectrePooledAttention(length, hidden, num_heads=heads), source, name="mixer"
    )


def build_head(net: Network, summary, hidden: int):
    """Two-layer is-next / is-first-half classifier on the summary vector."""
    ffn = net.connect(FullyConnectedLayer(hidden, FFN_HIDDEN, "swish"), summary, name="hidden")
    net.output = net.connect(FullyConnectedLayer(FFN_HIDDEN, 2, "linear", is_output=True), ffn, name="classifier")
    return net


def build_text_model(
    kind: str, vocab_size: int, padding_idx: int, length: int = SEQUENCE_LENGTH, hidden: int = HIDDEN_DIM,
    heads: int = NUM_HEADS,
) -> Network:
    """
    TextEmbedding -> RMSNorm -> mixer -> summary vector -> classifier.

    Parameters
    ----------
    kind : a key of MODELS
    vocab_size : the tokenizer's fitted vocabulary size
    padding_idx : id whose embedding row is fixed at zero

    Returns
    -------
    Network mapping token ids (batch, length) to is-next logits
    """
    net = Network(name=kind, input_shape=(length,))
    embedding = net.connect(TextEmbedding(vocab_size, hidden, padding_idx=padding_idx), net.input, name="embedding")
    normed = net.connect(RMSNormLayer(hidden), embedding, name="prenorm")
    return build_head(net, build_mixer(net, kind, normed, length, hidden, heads), hidden)


def build_marker_model(kind: str, length: int, hidden: int, heads: int) -> Network:
    """mixer -> summary vector -> classifier, on continuous token features."""
    net = Network(name=kind, input_shape=(length, hidden))
    return build_head(net, build_mixer(net, kind, net.input, length, hidden, heads), hidden)


# -------------    training    ------------------------------------------------
def run_batch(net: Network, batch: dict) -> tuple[np.ndarray, np.ndarray]:
    """Logits and labels for a batch, shaped alike."""
    extra = {} if batch["target_mask"] is None else {"target_mask": batch["target_mask"]}
    logits = net.forward(batch["inputs"], mask=batch["mask"], **extra)
    return logits, batch["labels"].reshape(logits.shape[:-1])


def accuracy(net: Network, batch: dict) -> float:
    net.eval()
    logits, labels = run_batch(net, batch)
    net.train()
    return float((np.argmax(logits, axis=-1) == labels).mean())


def train(
    net: Network, task: Task, rng: np.random.Generator, steps: int, learning_rate: float, checkpoints: tuple
) -> tuple[dict, list]:
    """
    Train on fresh batches every step.

    Returns
    -------
    held-out accuracy at each checkpoint, and the loss at every step
    """
    loss_fn = SparseCrossEntropyLoss()
    optimizer = Adam(learning_rate)
    curve, losses = {}, []
    net.train()
    for step in range(1, steps + 1):
        net.zero_gradients()
        logits, labels = run_batch(net, task.draw(rng))
        losses.append(float(loss_fn.forward(logits, labels)))
        net.backward(loss_fn.backward())
        optimizer.step(net)
        if step in checkpoints:
            curve[step] = accuracy(net, task.eval_batch)
    return curve, losses


def checkpoints_for(steps: int) -> tuple:
    """Five checkpoints, the last at steps."""
    return tuple(sorted({max(1, int(round(steps * fraction))) for fraction in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1)}))


def prepare_task(
    name: str, hidden: int = HIDDEN_DIM, heads: int = NUM_HEADS, length: int = SEQUENCE_LENGTH,
    batch_size: int = BATCH_SIZE, seed: int = 0,
) -> Task:
    """
    Data, models and a fixed held-out batch for one task.

    Parameters
    ----------
    name : one of TASKS
    hidden, heads, length : model and sequence size
    batch_size : training batch size
    seed : data seed
    """
    if name == "next sentence":
        stories, held_out = generate_stories(np.random.default_rng(seed))
        tokenizer = fit_text_tokenizer(stories)
        padding_idx = tokenizer.special_tokens.PAD
        vocab_size = tokenizer.get_vocab_size()
        train_processor = TextProcessor(tokenizer, max_length=length, tasks=("next_sentence",), random_seed=seed)
        train_processor.fit(stories)
        eval_processor = TextProcessor(tokenizer, max_length=length, tasks=("next_sentence",), random_seed=seed + 1)
        eval_processor.fit(held_out)
        picks = np.random.default_rng(seed + 2).integers(len(held_out), size=EVAL_PAIRS)

        def draw(rng: np.random.Generator) -> dict:
            return sentence_batch(train_processor, [stories[i] for i in rng.integers(len(stories), size=batch_size)])

        return Task(
            kinds=models_for(name),
            build=lambda kind: build_text_model(kind, vocab_size, padding_idx, length, hidden, heads),
            draw=draw,
            eval_batch=sentence_batch(eval_processor, [held_out[i] for i in picks]),
        )

    fixed_length = name == "marker position, fixed length"
    eval_rng = np.random.default_rng(seed + 1)
    return Task(
        kinds=models_for(name),
        build=lambda kind: build_marker_model(kind, length, hidden, heads),
        draw=lambda rng: marker_batch(rng, batch_size, length, hidden, fixed_length),
        eval_batch=marker_batch(eval_rng, EVAL_PAIRS, length, hidden, fixed_length),
    )


def compare_models(
    task: Task, kinds: list, steps: int, learning_rate: float, seeds: tuple, checkpoints: tuple
) -> tuple[dict, dict, dict]:
    """
    Train each kind on the task for every seed.

    Returns
    -------
    parameter counts, accuracy at each checkpoint (kind -> list over seeds of step -> accuracy), and loss curves
    (kind -> list over seeds)
    """
    parameters, accuracies, losses = {}, {}, {}
    for kind in kinds:
        accuracies[kind], losses[kind] = [], []
        for seed in seeds:
            net = task.build(kind)
            curve, loss_curve = train(net, task, np.random.default_rng(100 + seed), steps, learning_rate, checkpoints)
            accuracies[kind].append(curve)
            losses[kind].append(loss_curve)
        parameters[kind] = net.num_parameters
    return parameters, accuracies, losses


def accuracy_table(task: Task, parameters: dict, accuracies: dict, checkpoints: tuple) -> list[dict]:
    """One row per model: parameters and mean held-out accuracy at each checkpoint."""
    rows = []
    for kind, curves in accuracies.items():
        row = {"model": task.kinds[kind], "parameters": parameters[kind]}
        row.update({f"step {step}": float(np.mean([curve[step] for curve in curves])) for step in checkpoints})
        rows.append(row)
    return rows


# -------------    profile    -------------------------------------------------
def profile_mixers(
    length: int = PROFILE_LENGTH, hidden: int = PROFILE_HIDDEN, heads: int = PROFILE_HEADS,
    batch: int = PROFILE_BATCH, memory_tokens: int = 0, repeats: int = 3,
) -> list[dict]:
    """
    Time and memory of SpectreAttention (with its mean pool's gradient) and SpectrePooledAttention, float32.

    Time is the median of repeats untraced runs; memory comes from the Profiler: peak above the starting level
    during forward, memory the forward pass keeps, and peak during backward.

    Returns
    -------
    one row per layer
    """
    makers = {
        "Spectre": lambda: SpectreAttention(length, hidden, num_heads=heads, memory_tokens=memory_tokens, use_wrm=False),
        "SpectrePooled": lambda: SpectrePooledAttention(length, hidden, num_heads=heads, memory_tokens=memory_tokens),
    }
    rng = np.random.default_rng(0)
    x = rng.normal(size=(batch, length, hidden)).astype(np.float32)
    mask = np.ones((batch, length), dtype=np.float32)
    counts = mask.sum(axis=1, keepdims=True)[..., None]
    rows = []
    for name, make in makers.items():
        layer = make().recast(np.float32, np.complex64)
        layer.train()
        pooled = name != "Spectre"

        def gradient_for(output: np.ndarray) -> np.ndarray:
            if pooled:
                return np.ones_like(output)
            return (np.broadcast_to(np.ones((batch, 1, hidden), np.float32) / counts, output.shape) * mask[..., None]).astype(np.float32)

        def run() -> tuple[float, float]:
            layer.zero_gradients()
            start = time.perf_counter()
            output = layer.forward(x, mask=mask)
            middle = time.perf_counter()
            layer.backward(gradient_for(output))
            return middle - start, time.perf_counter() - middle

        run()
        timings = [run() for _ in range(repeats)]
        layer.purge()
        gc.collect()
        profiler = Profiler()
        with profiler:
            layer.zero_gradients()
            with profiler.measure("forward"):
                output = layer.forward(x, mask=mask)
            with profiler.measure("backward"):
                layer.backward(gradient_for(output))
        forward, backward = profiler.measurements["forward"], profiler.measurements["backward"]
        rows.append(
            {
                "layer": name,
                "parameters": layer.num_parameters,
                "forward ms": float(np.median([t[0] for t in timings]) * 1e3),
                "backward ms": float(np.median([t[1] for t in timings]) * 1e3),
                "forward peak MB": forward.max_peak / 1e6,
                "kept MB": forward.net_bytes / 1e6,
                "backward peak MB": backward.max_peak / 1e6,
            }
        )
        layer.purge()
        del layer, output
        gc.collect()
    return rows


# -------------    script    --------------------------------------------------
def print_accuracy(title: str, rows: list[dict]) -> None:
    columns = [key for key in rows[0] if key.startswith("step")]
    print(f"\n{title}: held-out accuracy, mean over {len(SEEDS)} seeds")
    print(f"{'model':26s} {'params':>7s}  " + "  ".join(f"{column:>9s}" for column in columns))
    for row in rows:
        print(f"{row['model']:26s} {row['parameters']:7d}  " + "  ".join(f"{row[column]:9.3f}" for column in columns))


def main():
    checkpoints = checkpoints_for(STEPS)
    for name in TASKS:
        task = prepare_task(name)
        parameters, accuracies, _ = compare_models(task, list(task.kinds), STEPS, LEARNING_RATE, SEEDS, checkpoints)
        print_accuracy(name, accuracy_table(task, parameters, accuracies, checkpoints))

    print(f"\nprofile: length {PROFILE_LENGTH}, hidden {PROFILE_HIDDEN}, {PROFILE_HEADS} heads, batch {PROFILE_BATCH}")
    print(f"{'layer':26s} {'params':>9s} {'fwd ms':>8s} {'bwd ms':>8s} {'fwd MB':>8s} {'kept MB':>8s} {'bwd MB':>8s}")
    for row in profile_mixers():
        print(
            f"{row['layer']:26s} {row['parameters']:9d} {row['forward ms']:8.1f} {row['backward ms']:8.1f} "
            f"{row['forward peak MB']:8.1f} {row['kept MB']:8.1f} {row['backward peak MB']:8.1f}"
        )


if __name__ == "__main__":
    main()
