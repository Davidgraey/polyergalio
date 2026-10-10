"""
TextProcessor distortions, against a word-level stand-in tokenizer so no
SentencePiece model has to be trained.
"""

import numpy as np
import pytest

pytest.importorskip("sentencepiece")

from polyergalio.encoders.text_encoders import (
    DistortionTask,
    TextProcessor,
)
from polyergalio.encoders.tokenizer import SpecialTokens

MAX_LENGTH = 32

CORPUS = [
    "the river runs past the old mill. children fish there in the morning. "
    "by evening the water turns gold.",
    "a small dog waits by the door. it hears footsteps on the street. "
    "the door opens and the dog runs out.",
    "rain fell all night. the roads were quiet by dawn.",
]


class WordTokenizer:
    """one id per whitespace-separated word, assigned on first sight"""

    def __init__(self, vocab_size: int = 200):
        self.special_tokens = SpecialTokens()
        self.vocab_size = vocab_size
        self.ids: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        for word in text.split():
            self.ids.setdefault(word, self.special_tokens.TOKEN_OFFSET + len(self.ids))
        return [self.ids[word] for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        words = {value: key for key, value in self.ids.items()}
        return " ".join(words[i] for i in ids if i in words)

    def get_vocab_size(self) -> int:
        return self.vocab_size


@pytest.fixture
def processor():
    processor = TextProcessor(WordTokenizer(), max_length=MAX_LENGTH, random_seed=0)
    processor.fit(CORPUS)
    return processor


def structural_positions(processor, input_ids):
    return np.isin(input_ids, [processor.special.CLS, processor.special.SEP])


@pytest.mark.parametrize("task", ids=lambda task: task.value)
def test_targets_never_fall_on_padding(processor, task):
    batch = processor.distort_batch(CORPUS, task)
    assert not (batch["target_mask"] & (batch["attention_mask"] == 0)).any()


# -------------    cloze    ----------------------------
def test_cloze_labels_hold_the_original_ids(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.CLOZE)
    original = processor.single_sequence(CORPUS[0])
    np.testing.assert_array_equal(sample["labels"], original)
    assert sample["target_mask"].any()


def test_cloze_leaves_untargeted_positions_alone(processor):
    sample = processor.distort(CORPUS[1], DistortionTask.CLOZE)
    untouched = ~sample["target_mask"]
    np.testing.assert_array_equal(sample["input_ids"][untouched], sample["labels"][untouched])


# -------------    replaced token detection    ----------------------------
def test_replaced_token_labels_mark_changed_positions(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.REPLACED_TOKEN)
    original = processor.single_sequence(CORPUS[0])
    np.testing.assert_array_equal(sample["labels"], (sample["input_ids"] != original).astype(int))


# -------------    span boundary    ----------------------------
def test_span_tokens_point_at_observed_boundaries(processor):
    sample = processor.distort(CORPUS[0], DistortionTask.SPAN_BOUNDARY)
    targets = np.flatnonzero(sample["target_mask"])
    assert targets.size

    for position in targets:
        left, right = sample["span_left"][position], sample["span_right"][position]
        assert left < position < right
        assert not sample["target_mask"][left] and not sample["target_mask"][right]
        assert sample["span_offset"][position] == position - left


# -------------    next sentence    ----------------------------
def test_a_positive_pair_keeps_the_true_successor(processor):
    processor.negative_sentence_prob = 0.0
    sample = processor.distort("alpha beta. gamma delta.", DistortionTask.NEXT_SENTENCE)
    expected, _ = processor.pair_sequence(
        processor.tokenizer.encode("alpha beta."), processor.tokenizer.encode("gamma delta.")
    )
    np.testing.assert_array_equal(sample["input_ids"], expected)
    assert sample["labels"][0] == 1


# -------------    processor interface    ----------------------------
def test_inverse_recovers_the_text(processor):
    encoded = processor.encode(["rain fell all night."])
    assert processor.inverse(encoded["text"])["text"][0] == "rain fell all night."
