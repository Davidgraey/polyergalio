"""
Text distortion example.

Fits a small SentencePiece tokenizer and a TextProcessor
(polyergalio.encoders) on a synthetic corpus, then applies every
implemented self-supervised distortion -- CLOZE, REPLACED_TOKEN,
SPAN_BOUNDARY, TOKEN_DELETION, TEXT_INFILLING, NEXT_SENTENCE -- to the
same sentences, printing the original once and each distortion's output
beneath it.

Run: python text_distortions_example.py
"""

import tempfile
from pathlib import Path

from polyergalio.encoders.text_encoders import DistortionTask, TextProcessor
from polyergalio.encoders.tokenizer import SentencePieceTokenizer, fit_tokenizer

VOCAB_SIZE = 2000
MAX_LENGTH = 32
RANDOM_SEED = 0
NUM_SHOWN = 3

CORPUS = open("examples/text_reference.txt", "r").readlines()

TASKS = (
    DistortionTask.CLOZE,
    DistortionTask.REPLACED_TOKEN,
    DistortionTask.SPAN_BOUNDARY,
    DistortionTask.TOKEN_DELETION,
    DistortionTask.TEXT_INFILLING,
    DistortionTask.NEXT_SENTENCE,
)


def fit_text_tokenizer(corpus: list[str]) -> SentencePieceTokenizer:
    """Fit a SentencePiece tokenizer on the corpus and load it."""
    with tempfile.TemporaryDirectory() as workdir:
        model_path = fit_tokenizer(
            corpus,
            model_prefix=str(Path(workdir) / "distortions_tokenizer"),
            vocab_size=VOCAB_SIZE,
            hard_vocab_limit=False,
        )
        return SentencePieceTokenizer(model_path)


def piece_line(tokenizer: SentencePieceTokenizer, ids) -> str:
    """Space-joined piece string for a sequence of ids, special tokens included."""
    return " ".join(tokenizer.id_to_piece(int(token_id)) for token_id in ids)


def original_line(processor: TextProcessor, tokenizer: SentencePieceTokenizer, text: str) -> str:
    """The undistorted [CLS] text [SEP] sequence, as one readable line."""
    return piece_line(tokenizer, processor.single_sequence(text))


def distorted_tokens(processor: TextProcessor, tokenizer: SentencePieceTokenizer, task: DistortionTask, text: str) -> list[tuple[str, bool]]:
    """
    One distortion's output sequence as (piece, is_target) pairs -- is_target
    marks the positions the task scores (target_mask), i.e. what to
    highlight as "distorted": a [MASK], a replaced or deleted token, an
    infilled span, or NEXT_SENTENCE's [CLS].
    """
    sample = processor.distort(text, task)
    pieces = [tokenizer.id_to_piece(int(token_id)) for token_id in sample["input_ids"]]
    return list(zip(pieces, sample["target_mask"].astype(bool).tolist()))


def distorted_line(processor: TextProcessor, tokenizer: SentencePieceTokenizer, task: DistortionTask, text: str) -> str:
    """One distortion's output sequence, as one readable line."""
    return " ".join(piece for piece, _ in distorted_tokens(processor, tokenizer, task, text))


def main():
    tokenizer = fit_text_tokenizer(CORPUS)
    processor = TextProcessor(tokenizer, max_length=MAX_LENGTH, tasks=TASKS, random_seed=RANDOM_SEED)
    processor.fit(CORPUS)

    sentences = CORPUS[:NUM_SHOWN]
    print("original")
    for i, text in enumerate(sentences, start=1):
        print(f"  {i}. {original_line(processor, tokenizer, text)}")

    for task in TASKS:
        print(f"\n{task.value}")
        for i, text in enumerate(sentences, start=1):
            print(f"  {i}. {distorted_line(processor, tokenizer, task, text)}")


if __name__ == "__main__":
    main()
