"""
Builds the multilingual Decision datasets in the final export format.

Run: python app/labeling/make_domain_data.py

Twenty datasets of 500 records each, from ten tasks in eight languages.
Class shares follow each spec's weights, so the answers are imbalanced the
way real data is. Every file is written through the app's own export and
read back to confirm it re-imports exactly.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE)]

import decision_tab  # noqa: E402
from domains import (  # noqa: E402
    candidate_screening,
    comment_moderation,
    crop_advice,
    email_triage,
    er_triage,
    loan_decision,
    machine_maintenance,
    news_topic,
    restaurant_rating,
    support_intent,
)
from domains.common import quotas  # noqa: E402
from label_common import export_text  # noqa: E402
from make_decision_data import DATA_DIR, SIZE, check  # noqa: E402

SEED = 4100
MAX_TRIES = 200000
MODULES = (
    support_intent, news_topic, email_triage, er_triage, loan_decision,
    machine_maintenance, restaurant_rating, candidate_screening, comment_moderation, crop_advice,
)


def draw_rows(spec: dict, seed: int) -> list[tuple[dict, int]]:
    """Rows for each class in proportion to the spec weights, shuffled."""
    rng = random.Random(seed)
    rows = []
    for want, count in enumerate(quotas(spec["weights"], SIZE)):
        for _ in range(count):
            for _ in range(MAX_TRIES):
                record, answer = spec["make"](rng, want)
                if answer == want:
                    break
            else:
                raise RuntimeError(f"{spec['name']}: could not produce class {want}")
            rows.append((record, want))
    rng.shuffle(rows)
    return rows


def build(spec: dict, seed: int) -> Path:
    rows = draw_rows(spec, seed)
    records = [
        {spec["id_field"]: f"{spec['id_prefix']}{index:05d}", **record} for index, (record, _) in enumerate(rows, 1)
    ]
    config = {
        **decision_tab.DEFAULT_CONFIG, "kind": spec["kind"], "instructions": spec["instructions"],
        "options_text": "\n".join(spec["options"]), "levels": len(spec["options"]),
        "columns": decision_tab.record_columns(records),
    }
    q = decision_tab.question(config)
    assert q["options"] == spec["options"], f"{spec['name']}: options rejected by the app's question builder"
    labels = {
        index: {
            "record": record, **q, "state": decision_tab.record_state(record, q["columns"]),
            "answer": answer, "option": q["options"][answer],
        }
        for index, (record, (_, answer)) in enumerate(zip(records, rows))
    }
    space = dict(
        records=records, ids=list(range(SIZE)), labels=labels, position=0, config=config, view={}, drafts={}, notice=None,
    )
    path = DATA_DIR / f"decision_{spec['name']}.jsonl"
    path.write_text(export_text(decision_tab, space), encoding="utf-8")
    check(path, records, labels)
    return path


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    specs = [spec for module in MODULES for spec in module.SPECS]
    assert len(specs) == 20 and len({spec["name"] for spec in specs}) == 20
    for offset, spec in enumerate(specs):
        path = build(spec, SEED + offset)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        shares = [sum(row["answer"] == i for row in rows) for i in range(len(spec["options"]))]
        print(f"{path.name}: {len(rows)} rows, {dict(zip(spec['options'], shares))}")


if __name__ == "__main__":
    main()
