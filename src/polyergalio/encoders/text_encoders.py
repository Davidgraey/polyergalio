"""
------------------------ Text preprocessing / self-supervised distortions ------------------------------

TextProcessor tokenizes raw strings with a SentencePieceTokenizer and builds
corrupted training views for pretraining text encoders and embedding models.

Every distortion returns the same per-token fields, so tasks can be swapped
without changing the training loop:

    input_ids       (sequence,) distorted token ids fed to the network
    attention_mask  (sequence,) 1 for real content, 0 for padding
    segment_ids     (sequence,) 0 for the first segment, 1 for the second
    target_mask     (sequence,) True at the positions the task scores
    labels          (sequence,) per-position target, read only where target_mask

target_mask is always a subset of attention_mask, and pairs directly with
MaskGather: gather the encoder output with it, and score against
labels[target_mask].
"""
import re
from dataclasses import asdict
from enum import Enum
from typing import Iterable, Optional

import numpy as np

from polyergalio.encoders.encoders import Processor
from polyergalio.encoders.tokenizer import SentencePieceTokenizer, Tokenizer


class DistortionTask(Enum):
    CLOZE = "cloze"
    NEXT_SENTENCE = "next_sentence"
    SPAN_BOUNDARY = "span_boundary"
    REPLACED_TOKEN = "replaced_token"
    SENTENCE_ORDER = "sentence_order"
    SENTENCE_BOUNDARY = "sentence_boundary"
    TOKEN_DELETION = "token_deletion"
    TEXT_INFILLING = "text_infilling"
    SENTENCE_PERMUTATION = "sentence_permutation"
    DOCUMENT_ROTATION = "document_rotation"
    CONTRASTIVE_VIEW = "contrastive_view"


IMPLEMENTED_TASKS = (
    DistortionTask.CLOZE,
    DistortionTask.NEXT_SENTENCE,
    DistortionTask.SPAN_BOUNDARY,
    DistortionTask.REPLACED_TOKEN,
    DistortionTask.TOKEN_DELETION,
    DistortionTask.TEXT_INFILLING,
)

PAD_VALUES = {
    "attention_mask": 0,
    "segment_ids": 0,
    "target_mask": False,
    "labels": 0,
    "span_left": -1,
    "span_right": -1,
    "span_offset": -1,
}

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """split on whitespace following ., ! or ?"""
    return [sentence.strip() for sentence in SENTENCE_END.split(text.strip()) if sentence.strip()]


class TextProcessor(Processor):
    state_names = ("sentence_pool", "token_counts")

    def __init__(
        self,
        tokenizer: SentencePieceTokenizer,
        target: str = "text",
        max_length: int = 128,
        tasks: Iterable[DistortionTask | str] = IMPLEMENTED_TASKS,
        mask_prob: float = 0.15,
        replace_prob: float = 0.15,
        span_geometric_p: float = 0.2,
        max_span_length: int = 10,
        negative_sentence_prob: float = 0.5,
        delete_prob: float = 0.15,
        infill_prob: float = 0.3,
        infill_poisson_lambda: float = 3.0,
        random_seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        tokenizer : fitted SentencePieceTokenizer
        target : name of the text column
        max_length : padded sequence length, special tokens included
        tasks : distortions distort() and distort_batch() draw from
        mask_prob : fraction of content tokens masked by CLOZE and SPAN_BOUNDARY
        replace_prob : fraction of content tokens replaced by REPLACED_TOKEN
        span_geometric_p : p of the geometric distribution span lengths are drawn from
        max_span_length : longest span SPAN_BOUNDARY masks
        negative_sentence_prob : chance NEXT_SENTENCE pairs a random sentence
        delete_prob : fraction of content tokens TOKEN_DELETION removes
        infill_prob : fraction of content TEXT_INFILLING covers with spans
        infill_poisson_lambda : mean of the Poisson distribution TEXT_INFILLING draws span lengths from
        random_seed : seed for every random draw
        """
        super().__init__(target)
        self.tokenizer = tokenizer
        self.special = tokenizer.special_tokens
        self.vocab_size = tokenizer.get_vocab_size()
        if self.vocab_size <= self.special.TOKEN_OFFSET:
            raise ValueError(
                f"vocab_size {self.vocab_size} leaves no content tokens past "
                f"TOKEN_OFFSET {self.special.TOKEN_OFFSET}"
            )
        if max_length < 5:
            raise ValueError("max_length must leave room for [CLS] A [SEP] B [SEP]")

        self.max_length = max_length
        self.tasks = [DistortionTask(task) for task in tasks]
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.span_geometric_p = span_geometric_p
        self.max_span_length = max_span_length
        self.negative_sentence_prob = negative_sentence_prob
        self.delete_prob = delete_prob
        self.infill_prob = infill_prob
        self.infill_poisson_lambda = infill_poisson_lambda
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.special_ids = np.array(
            [value for name, value in asdict(self.special).items() if name != "TOKEN_OFFSET"]
        )
        self.sentence_pool: list[str] = []
        self.token_counts = np.zeros(self.vocab_size)

        self.distortions = {
            DistortionTask.CLOZE: self.cloze,
            DistortionTask.NEXT_SENTENCE: self.next_sentence,
            DistortionTask.SPAN_BOUNDARY: self.span_boundary,
            DistortionTask.REPLACED_TOKEN: self.replaced_token,
            DistortionTask.SENTENCE_ORDER: self.sentence_order,
            DistortionTask.SENTENCE_BOUNDARY: self.sentence_boundary,
            DistortionTask.TOKEN_DELETION: self.token_deletion,
            DistortionTask.TEXT_INFILLING: self.text_infilling,
            DistortionTask.SENTENCE_PERMUTATION: self.sentence_permutation,
            DistortionTask.DOCUMENT_ROTATION: self.document_rotation,
            DistortionTask.CONTRASTIVE_VIEW: self.contrastive_view,
        }

    # ------------- persistence
    def get_config(self) -> dict:
        config = super().get_config()
        del config["tokenizer"]
        return config

    def get_state(self) -> dict:
        state = super().get_state()
        state["rng_state"] = self.rng.bit_generator.state
        state["tokenizer"] = self.tokenizer.serialize()
        return state

    def set_state(self, state: dict) -> None:
        super().set_state({key: value for key, value in state.items() if key not in ("tokenizer", "rng_state")})
        self.rng.bit_generator.state = state["rng_state"]

    @classmethod
    def rebuild(cls, config: dict, state: dict) -> "TextProcessor":
        """The constructor needs the tokenizer, so it is restored first from its own payload."""
        processor = cls(Tokenizer.deserialize(state["tokenizer"]), **config)
        processor.set_state(state)
        return processor

    # ------------- Processor interface
    def fit(self, values: Iterable[str]) -> bool:
        """
        Collect the sentence pool NEXT_SENTENCE draws negatives from and the unigram counts REPLACED_TOKEN samples
        replacements from. The tokenizer itself is fitted separately, see fit_tokenizer.
        """
        for text in values:
            if not isinstance(text, str):
                continue
            self.sentence_pool.extend(split_sentences(text))
            np.add.at(self.token_counts, self.tokenizer.encode(text), 1)
        self.fitted = True
        return True

    def encode(self, values: Iterable[str]) -> dict:
        """[CLS] text [SEP], undistorted and padded to max_length"""
        samples = [self.blank_sample(self.single_sequence(text)) for text in values]
        batch = self.collate(samples)
        return {
            f"{self.target}": batch["input_ids"],
            f"{self.target}_attention_mask": batch["attention_mask"],
        }

    def fit_encode(self, values: Iterable[str]) -> dict:
        values = list(values)
        assert self.fit(values) is True
        return self.encode(values)

    def inverse(self, values: Iterable) -> dict:
        """decode token id rows back to strings, special tokens dropped"""
        texts = [self.tokenizer.decode(np.asarray(row).tolist()) for row in values]
        return {f"{self.target}": np.array(texts, dtype=object)}

    @property
    def metadata(self):
        return {
            str(self.target): {
                "vocab_size": self.vocab_size,
                "output_dimension": self.max_length,
                "output_type": "int",
                "enc_type": "text",
                "tasks": [task.value for task in self.tasks],
                "special_tokens": asdict(self.special),
                "variable_names": [f"{self.target}"],
            }
        }

    # ------------- sequence helpers
    def single_sequence(self, text: str) -> np.ndarray:
        ids = self.tokenizer.encode(text)[: self.max_length - 2]
        return np.array([self.special.CLS, *ids, self.special.SEP], dtype=int)

    def content_positions(self, input_ids: np.ndarray) -> np.ndarray:
        return np.flatnonzero(~np.isin(input_ids, self.special_ids))

    def blank_sample(self, input_ids: np.ndarray, segment_ids: Optional[np.ndarray] = None) -> dict:
        size = input_ids.size
        return {
            "input_ids": input_ids,
            "attention_mask": np.ones(size, dtype=int),
            "segment_ids": np.zeros(size, dtype=int) if segment_ids is None else segment_ids,
            "target_mask": np.zeros(size, dtype=bool),
            "labels": np.zeros(size, dtype=int),
        }

    def choose_positions(self, input_ids: np.ndarray, probability: float) -> np.ndarray:
        """content positions kept with the given probability, at least one if any exist"""
        candidates = self.content_positions(input_ids)
        chosen = candidates[self.rng.random(candidates.size) < probability]
        if chosen.size == 0 and candidates.size:
            chosen = self.rng.choice(candidates, size=1)
        return chosen

    def random_content_tokens(self, size: int) -> np.ndarray:
        return self.rng.integers(self.special.TOKEN_OFFSET, self.vocab_size, size=size)

    def unigram_content_tokens(self, size: int) -> np.ndarray:
        """draws from the fitted unigram counts, uniform before fit()"""
        counts = self.token_counts[self.special.TOKEN_OFFSET:]
        if counts.sum() == 0:
            return self.random_content_tokens(size)
        return self.rng.choice(
            np.arange(self.special.TOKEN_OFFSET, self.vocab_size), size=size, p=counts / counts.sum()
        )

    def adjacent_segments(self, text: str) -> tuple[list[int], list[int]]:
        """two consecutive sentences, or one sentence split at a random token"""
        sentences = split_sentences(text)
        if len(sentences) >= 2:
            index = self.rng.integers(0, len(sentences) - 1)
            return self.tokenizer.encode(sentences[index]), self.tokenizer.encode(sentences[index + 1])
        ids = self.tokenizer.encode(text)
        if len(ids) < 2:
            raise ValueError("next_sentence needs at least two tokens of text")
        cut = self.rng.integers(1, len(ids))
        return ids[:cut], ids[cut:]

    def pool_sentence(self, exclude: list[int]) -> list[int]:
        if not self.sentence_pool:
            raise ValueError("next_sentence draws negatives from the pool fit() collects; call fit() first")
        for _ in range(10):
            ids = self.tokenizer.encode(self.sentence_pool[self.rng.integers(len(self.sentence_pool))])
            if ids != exclude:
                return ids
        return ids

    def pair_sequence(self, first: list[int], second: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """[CLS] first [SEP] second [SEP], trimming the longer segment to fit"""
        first, second = list(first), list(second)
        while len(first) + len(second) > self.max_length - 3:
            (first if len(first) >= len(second) else second).pop()
        input_ids = np.array(
            [self.special.CLS, *first, self.special.SEP, *second, self.special.SEP], dtype=int
        )
        segment_ids = np.zeros(input_ids.size, dtype=int)
        segment_ids[len(first) + 2:] = 1
        return input_ids, segment_ids

    # ------------- implemented distortions
    def cloze(self, text: str) -> dict:
        """
        BERT masked language modeling. Of the content positions chosen, 80%
        become [MASK], 10% a random token, 10% are left as they were; labels
        hold the original ids.
        """
        original = self.single_sequence(text)
        sample = self.blank_sample(original.copy())
        chosen = self.choose_positions(original, self.mask_prob)

        roll = self.rng.random(chosen.size)
        sample["input_ids"][chosen[roll < 0.8]] = self.special.MASK
        swapped = chosen[(roll >= 0.8) & (roll < 0.9)]
        sample["input_ids"][swapped] = self.random_content_tokens(swapped.size)

        sample["target_mask"][chosen] = True
        sample["labels"] = original
        return sample

    def replaced_token(self, text: str) -> dict:
        """
        ELECTRA replaced-token detection. Replacements come from the fitted
        unigram distribution rather than a trained generator, which makes
        them less plausible and the task easier than ELECTRA's. Every content
        position is scored; label 1 where the token changed, so a draw that
        happens to equal the original is labelled 0.
        """
        original = self.single_sequence(text)
        sample = self.blank_sample(original.copy())
        chosen = self.choose_positions(original, self.replace_prob)
        sample["input_ids"][chosen] = self.unigram_content_tokens(chosen.size)

        sample["target_mask"][self.content_positions(original)] = True
        sample["labels"] = (sample["input_ids"] != original).astype(int)
        return sample

    def span_boundary(self, text: str) -> dict:
        """
        SpanBERT span masking with the span boundary objective. Contiguous
        spans, geometric lengths clipped to max_span_length, are masked up to
        mask_prob of the content, with 80/10/10 [MASK]/random/unchanged
        decided per span. Spans are kept one token apart so both boundary
        tokens are always observed. Spans are over subword pieces, not
        whole words.

        Extra per-token fields, -1 outside spans:
            span_left, span_right : positions of the tokens just outside the span
            span_offset : 1-based position inside the span
        """
        original = self.single_sequence(text)
        sample = self.blank_sample(original.copy())
        size = original.size
        span_left = np.full(size, -1, dtype=int)
        span_right = np.full(size, -1, dtype=int)
        span_offset = np.full(size, -1, dtype=int)

        candidates = self.content_positions(original)
        budget = max(1, int(round(self.mask_prob * candidates.size))) if candidates.size else 0
        covered = sample["target_mask"]

        for _ in range(10 * budget):
            remaining = budget - int(covered.sum())
            if remaining <= 0:
                break
            length = min(int(self.rng.geometric(self.span_geometric_p)), self.max_span_length, remaining)
            if length > candidates.size:
                continue
            start = self.rng.integers(0, candidates.size - length + 1)
            positions = candidates[start: start + length]
            if positions[-1] - positions[0] != length - 1:
                continue
            if covered[positions[0] - 1: positions[-1] + 2].any():
                continue

            covered[positions] = True
            span_left[positions] = positions[0] - 1
            span_right[positions] = positions[-1] + 1
            span_offset[positions] = np.arange(1, length + 1)

            roll = self.rng.random()
            if roll < 0.8:
                sample["input_ids"][positions] = self.special.MASK
            elif roll < 0.9:
                sample["input_ids"][positions] = self.random_content_tokens(length)

        sample["labels"] = original
        sample["span_left"] = span_left
        sample["span_right"] = span_right
        sample["span_offset"] = span_offset
        return sample

    def next_sentence(self, text: str) -> dict:
        """
        BERT next sentence prediction, [CLS] A [SEP] B [SEP]. B is A's true
        successor (label 1) or, with negative_sentence_prob, a sentence from
        the pool fit() collects (label 0). Scored at [CLS] only.
        """
        first, second = self.adjacent_segments(text)
        is_next = self.rng.random() >= self.negative_sentence_prob
        if not is_next:
            second = self.pool_sentence(exclude=second)

        input_ids, segment_ids = self.pair_sequence(first, second)
        sample = self.blank_sample(input_ids, segment_ids)
        sample["target_mask"][0] = True
        sample["labels"][0] = int(is_next)
        return sample

    def token_deletion(self, text: str) -> dict:
        """
        BART token deletion, encoder-only form. Content tokens are removed
        outright, with no [MASK] left behind, at delete_prob. Every
        remaining position after [CLS] is scored: label 1 where a token
        was deleted immediately before it ([SEP] catches a deletion at the
        end), so a run of several deletions in a row is still one label,
        carried by the position right after the run.

        original_ids (padded, undistorted) is included alongside the
        shared fields, for a future reconstruction decoder -- BART's
        actual objective.
        """
        original = self.single_sequence(text)
        content = self.content_positions(original)
        deleted = content[self.rng.random(content.size) < self.delete_prob]

        deleted_before = np.zeros(original.size, dtype=bool)
        deleted_before[deleted + 1] = True

        kept = np.flatnonzero(~np.isin(np.arange(original.size), deleted))
        sample = self.blank_sample(original[kept])
        sample["target_mask"][1:] = True
        sample["labels"][1:] = deleted_before[kept][1:].astype(int)
        sample["original_ids"] = original
        return sample

    def text_infilling(self, text: str) -> dict:
        """
        BART text infilling, encoder-only form. Spans over content
        positions -- lengths drawn from Poisson(infill_poisson_lambda),
        clipped to max_span_length, covering up to infill_prob of the
        content -- are each collapsed to a single [MASK]; a 0-length span
        inserts a [MASK] where nothing was missing. Spans stay one token
        apart. Scored at each [MASK]; label is the span's original
        length, so the head is a (max_span_length + 1)-way classifier.

        original_ids (padded, undistorted) is included alongside the
        shared fields, for a future reconstruction decoder -- BART's
        actual objective. Reconstruction itself needs a decoder; this
        encoder-only variant predicts span length instead.
        """
        original = self.single_sequence(text)
        content = self.content_positions(original)
        budget = max(1, int(round(self.infill_prob * content.size))) if content.size else 0
        covered = np.zeros(original.size, dtype=bool)

        spans: list[tuple[int, int]] = []
        used_starts: set[int] = set()
        for _ in range(10 * (budget + 1)):
            if spans and int(covered.sum()) >= budget:
                break
            length = min(int(self.rng.poisson(self.infill_poisson_lambda)), self.max_span_length)

            if length == 0:
                if not content.size:
                    continue
                anchor = int(self.rng.choice(content))
                if covered[anchor] or anchor in used_starts:
                    continue
                used_starts.add(anchor)
                spans.append((anchor, 0))
                continue

            remaining = budget - int(covered.sum())
            length = min(length, remaining, content.size)
            if length <= 0:
                continue
            start = self.rng.integers(0, content.size - length + 1)
            positions = content[start: start + length]
            if covered[positions[0] - 1: positions[-1] + 2].any() or int(positions[0]) in used_starts:
                continue
            covered[positions] = True
            used_starts.add(int(positions[0]))
            spans.append((int(positions[0]), length))

        spans.sort()
        new_ids, target_mask, labels = [], [], []
        cursor = 0
        for start, length in spans:
            new_ids.extend(original[cursor:start])
            target_mask.extend([False] * (start - cursor))
            labels.extend([0] * (start - cursor))

            new_ids.append(self.special.MASK)
            target_mask.append(True)
            labels.append(length)
            cursor = start + length

        new_ids.extend(original[cursor:])
        target_mask.extend([False] * (original.size - cursor))
        labels.extend([0] * (original.size - cursor))

        sample = self.blank_sample(np.array(new_ids, dtype=int))
        sample["target_mask"] = np.array(target_mask, dtype=bool)
        sample["labels"] = np.array(labels, dtype=int)
        sample["original_ids"] = original
        return sample

    # ------------- stubbed distortions
    def sentence_order(self, text: str) -> dict:
        """
        ALBERT sentence-order prediction: two adjacent segments, swapped half
        the time; label whether they are in their original order. Negatives
        share a topic, so unlike NEXT_SENTENCE it cannot be solved by topic
        alone. Scored at [CLS].
        """
        raise NotImplementedError("sentence_order is stubbed")

    def sentence_boundary(self, text: str) -> dict:
        """
        Sentence boundary detection: concatenate sentences with their end
        punctuation stripped; label each token 1 where a new sentence begins.
        """
        raise NotImplementedError("sentence_boundary is stubbed")

    def sentence_permutation(self, text: str) -> dict:
        """
        BART sentence permutation: shuffle sentence order; predict each
        sentence's original index.
        """
        raise NotImplementedError("sentence_permutation is stubbed")

    def document_rotation(self, text: str) -> dict:
        """
        BART document rotation: rotate the sequence to start at a random
        token; predict the position of the true start.
        """
        raise NotImplementedError("document_rotation is stubbed")

    def contrastive_view(self, text: str) -> dict:
        """
        Two independently distorted views of the same text as a positive
        pair, with the rest of the batch as negatives (SimCSE / InfoNCE).
        The objective that trains sentence embeddings directly; returns a
        pair of samples rather than one.
        """
        raise NotImplementedError("contrastive_view is stubbed")

    # ------------- dispatch and batching
    def distort(self, text: str, task: Optional[DistortionTask | str] = None) -> dict:
        """one distorted sample; task drawn from self.tasks if not given"""
        task = DistortionTask(task) if task is not None else self.tasks[self.rng.integers(len(self.tasks))]
        sample = self.distortions[task](text)
        sample["task"] = task.value
        return sample

    def distort_batch(self, texts: Iterable[str], task: Optional[DistortionTask | str] = None) -> dict:
        """
        One task per batch, since each task feeds its own head. Every
        per-token field is padded to (batch, max_length).
        """
        task = DistortionTask(task) if task is not None else self.tasks[self.rng.integers(len(self.tasks))]
        batch = self.collate([self.distortions[task](text) for text in texts])
        batch["task"] = task.value
        return batch

    def collate(self, samples: list[dict]) -> dict:
        batch = {}
        fields = [name for name, value in samples[0].items() if isinstance(value, np.ndarray)]
        for name in fields:
            fill = self.special.PAD if name in ("input_ids", "original_ids") else PAD_VALUES[name]
            dtype = bool if name == "target_mask" else int
            padded = np.full((len(samples), self.max_length), fill, dtype=dtype)
            for row, sample in enumerate(samples):
                values = sample[name]
                padded[row, : values.size] = values
            batch[name] = padded
        return batch


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from polyergalio.encoders.tokenizer import fit_tokenizer

    corpus = [
        "The river runs past the old mill. Children fish there in the morning. "
        "By evening the water turns gold.",
        "A small dog waits by the door. It hears footsteps on the street. "
        "The door opens and the dog runs out.",
    ] * 200

    with tempfile.TemporaryDirectory() as workdir:
        model_path = fit_tokenizer(corpus, str(Path(workdir) / "demo"), vocab_size=200, hard_vocab_limit=False)
        tokenizer = SentencePieceTokenizer(model_path)

    processor = TextProcessor(tokenizer, max_length=32, random_seed=0)
    processor.fit(corpus)
    for task in IMPLEMENTED_TASKS:
        batch = processor.distort_batch(corpus[:4], task)
        print(task.value, {name: value.shape for name, value in batch.items() if name != "task"})
