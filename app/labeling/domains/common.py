"""Shared helpers for the multilingual decision datasets."""

import math
import random

CHANNELS = ["email", "chat", "web"]


def pick(rng: random.Random, items: list):
    return items[rng.randrange(len(items))]


def weighted(rng: random.Random, weights: list[float]) -> int:
    return rng.choices(range(len(weights)), weights=weights)[0]


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def quotas(weights: list[float], size: int) -> list[int]:
    """Whole-number class counts summing to size, proportional to weights (largest remainder)."""
    total = sum(weights)
    exact = [size * w / total for w in weights]
    counts = [int(value) for value in exact]
    order = sorted(range(len(weights)), key=lambda i: exact[i] - counts[i], reverse=True)
    for index in order[: size - sum(counts)]:
        counts[index] += 1
    return counts


def spec(name, language, kind, instructions, options, weights, id_field, id_prefix, make) -> dict:
    """One dataset: metadata plus make(rng, want) -> (record, answer index)."""
    assert len(options) == len(weights) <= 20
    return dict(
        name=name, language=language, kind=kind, instructions=instructions, options=options,
        weights=weights, id_field=id_field, id_prefix=id_prefix, make=make,
    )


def slot_values(rng: random.Random, slots: dict) -> dict:
    return {key: value(rng) if callable(value) else pick(rng, value) for key, value in slots.items()}


def text_maker(templates: list[list[str]], slots: dict, field: str, greetings=("",), closings=("",), joiner=" ", extra=None):
    """
    make(rng, want) for text tasks: a template of the wanted class, its slots
    filled, wrapped in an optional greeting and closing. extra(rng) adds
    more fields to the record.
    """

    def make(rng: random.Random, want: int):
        body = pick(rng, templates[want]).format(**slot_values(rng, slots))
        parts = [pick(rng, list(greetings)), body, pick(rng, list(closings))]
        record = {field: joiner.join(part for part in parts if part).strip()}
        record.update(extra(rng) if extra else {})
        return record, want

    return make


def date_2026(rng: random.Random) -> str:
    return f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}"


def number_between(low: int, high: int):
    return lambda rng: str(rng.randint(low, high))


def clock(rng: random.Random) -> str:
    return f"{rng.randint(7, 18)}:{pick(rng, ['00', '15', '30', '45'])}"
