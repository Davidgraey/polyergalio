"""
Builds one labeled dataset per Decision example task, in the final export format.

Run: python app/labeling/make_decision_data.py

Every dataset holds 500 records with balanced answer classes. Labels come from
explicit rules over the record fields (or, for text tasks, from the class the
text was written for), and each file is written through the app's own export
so it re-imports exactly into the labeling app.
"""

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE)]

import decision_tab  # noqa: E402
from label_common import export_text, import_labeled  # noqa: E402
from label_tasks import DECISION_TASKS  # noqa: E402

SIZE = 500
SEED = 2026
DATA_DIR = HERE.parents[1] / "data"


def pick(rng: random.Random, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


# -------------    urgent ticket    ------------------------------
TICKET_SUBJECTS = {
    "critical": [
        "Site down for all users", "Production outage in {region}", "Data loss after migration",
        "Security breach suspected on our account", "Entire team locked out of login",
        "Payment processing failing for every customer",
    ],
    "high": [
        "Payment failed, account locked", "API returning errors intermittently",
        "Data export returns empty file", "Cannot reach billing portal before renewal",
        "Report generation times out",
    ],
    "medium": [
        "Dashboard loads slowly", "Email notifications delayed", "{tool} integration keeps disconnecting",
        "Wrong amount on last invoice", "Two-factor codes not arriving",
    ],
    "low": [
        "How do I change my avatar?", "Feature request: dark mode", "Typo on the pricing page",
        "Question about export formats", "Can I rename my workspace?", "Where is the mobile app?",
    ],
}
REGIONS = ["EU", "US-East", "APAC", "US-West"]
TOOLS = ["Slack", "Salesforce", "Zapier", "Jira", "HubSpot"]


def ticket_urgent(severity: str, tier: str, hours_open: int) -> bool:
    if severity == "critical":
        return True
    if severity == "high":
        return tier != "free" or hours_open >= 24
    if severity == "medium":
        return tier == "enterprise" and hours_open >= 24
    return False


def urgent_ticket(rng: random.Random, number: int):
    severity = pick(rng, {"critical": 2, "high": 3, "medium": 3, "low": 3})
    tier = pick(rng, {"free": 4, "pro": 4, "enterprise": 2})
    hours = min(168, int(rng.expovariate(1 / 30)) + 1)
    subject = rng.choice(TICKET_SUBJECTS[severity]).format(region=rng.choice(REGIONS), tool=rng.choice(TOOLS))
    record = {
        "ticket_id": f"T-{10000 + number}", "subject": subject, "hours_open": hours,
        "customer_tier": tier, "previous_tickets": min(12, int(rng.expovariate(1 / 3))),
    }
    return record, int(ticket_urgent(severity, tier, hours))


# -------------    flag transaction    ---------------------------
COUNTRIES = ["US", "GB", "FR", "DE", "ES", "NG", "RO", "BR", "IN", "JP", "MX", "PL"]
CATEGORIES = {
    "grocery": (45, 0.9), "restaurant": (40, 0.8), "fuel": (60, 0.9), "retail": (90, 0.6),
    "travel": (450, 0.3), "electronics": (350, 0.4), "subscriptions": (18, 0.0),
    "gift_cards": (150, 0.0), "crypto": (700, 0.0), "wire_transfer": (1200, 0.0),
}
RISKY_CATEGORIES = {"gift_cards", "crypto", "wire_transfer"}


def transaction_risk(record: dict) -> int:
    score = 0
    score += 2 if record["country"] != record["home_country"] else 0
    score += 2 if record["amount"] > 1000 else 1 if record["amount"] > 500 else 0
    score += 1 if record["hour"] < 6 else 0
    score += 2 if record["merchant_category"] in RISKY_CATEGORIES else 0
    score += 1 if not record["card_present"] else 0
    return score


def flag_transaction(rng: random.Random, number: int):
    category = pick(rng, {name: 1 for name in CATEGORIES})
    typical, in_person = CATEGORIES[category]
    home = rng.choice(COUNTRIES)
    abroad = rng.random() < 0.25
    record = {
        "txn_id": f"X{700000 + number}",
        "amount": round(typical * math.exp(rng.gauss(0, 0.9)), 2),
        "country": rng.choice([c for c in COUNTRIES if c != home]) if abroad else home,
        "home_country": home,
        "hour": rng.choice(range(24)) if rng.random() < 0.7 else rng.choice(range(0, 6)),
        "merchant_category": category,
        "card_present": rng.random() < in_person,
    }
    return record, int(transaction_risk(record) >= 4)


# -------------    route ticket    -------------------------------
ROUTES = {
    "billing": (
        ["Charged twice this month", "Invoice needs a VAT number", "Refund for order {n}",
         "Wrong amount on last invoice", "Update credit card on file", "Discount code was not applied"],
        ["I see two identical charges of ${amount} on my card.", "Please reissue invoice {n} with our VAT id.",
         "The order was cancelled but the ${amount} has not come back.", "My card expired and the renewal failed.",
         "The promo code SPRING was rejected at checkout."],
    ),
    "technical": (
        ["App crashes on launch", "API returns {code}", "Sync fails after login",
         "Export button does nothing", "Webhook is not firing"],
        ["Since the last update it closes immediately.", "POST /v2/orders fails intermittently with {code}.",
         "Files stay on pending and never finish uploading.", "Clicking export shows a spinner forever.",
         "No events reach our endpoint even though it is reachable."],
    ),
    "account": (
        ["Cannot reset password", "Change the email on my account", "Add a teammate to the workspace",
         "Delete my account", "Two-factor locked me out"],
        ["The reset email never arrives.", "I no longer have access to my old address.",
         "Please give jamie@example.com editor access.", "I want my data removed under GDPR.",
         "I lost my phone and cannot get a code."],
    ),
    "other": (
        ["Do you sponsor events?", "Partnership inquiry", "Press request", "Feedback on the new logo", "Job openings?"],
        ["We are organizing a meetup in June.", "We would like to explore a joint webinar.",
         "I am writing an article about your industry.", "It looks great but the blue is hard to read.",
         "Are you hiring engineers in Europe?"],
    ),
}
CLOSERS = ["", "", "Thanks in advance.", "Please advise.", "This is the second time I am asking."]


def route_ticket(rng: random.Random, number: int):
    team = pick(rng, {"billing": 1, "technical": 1, "account": 1, "other": 1})
    subjects, bodies = ROUTES[team]
    fill = dict(n=rng.randint(1000, 9999), amount=rng.choice([9, 19, 29, 49, 99, 199]), code=rng.choice([500, 502, 503, 429]))
    body = f"{rng.choice(bodies)} {rng.choice(CLOSERS)}".strip().format(**fill)
    record = {"ticket_id": f"T-{20000 + number}", "subject": rng.choice(subjects).format(**fill), "body": body}
    return record, list(ROUTES).index(team)


# -------------    inventory action    ---------------------------
PERISHABLE = ["Yogurt 500g", "Fresh basil", "Whole milk 1L", "Sourdough loaf", "Salad kit", "Chicken breast 1kg", "Strawberries 250g"]
DURABLE = [
    "USB-C cable", "Desk lamp", "Notebook A5", "Phone case", "Water bottle", "Wireless mouse",
    "Backpack", "Bluetooth speaker", "HDMI cable", "Yoga mat", "Coffee grinder", "Umbrella",
]
ACTIONS = ["reorder", "discount", "hold", "discontinue"]


def inventory_action_for(record: dict) -> str:
    cover_weeks = record["stock"] / max(record["weekly_sales"], 0.5)
    expiry = record["days_to_expiry"]
    if expiry is not None and expiry <= 7 and cover_weeks * 7 > expiry:
        return "discount"
    if record["weekly_sales"] <= 2 and record["stock"] >= 50:
        return "discontinue"
    if cover_weeks < 2:
        return "reorder"
    return "hold"


def inventory_item(rng: random.Random, number: int):
    perishable = rng.random() < 0.4
    weekly = rng.choice([0, 1, 2, 3, 5, 8, 12, 20, 35, 60]) if not perishable else rng.choice([10, 20, 30, 45, 60])
    record = {
        "sku": f"SKU-{3000 + number}",
        "item": rng.choice(PERISHABLE if perishable else DURABLE),
        "stock": int(log_uniform(rng, 2, 400)),
        "weekly_sales": weekly,
        "days_to_expiry": rng.randint(1, 21) if perishable else None,
    }
    return record, ACTIONS.index(inventory_action_for(record))


# -------------    review sentiment    ---------------------------
PRODUCTS = ["Kettle", "Headphones", "Backpack", "Blender", "Monitor", "Mouse", "Vacuum", "Keyboard", "Desk lamp", "Speaker", "Router", "Sneakers"]
ASPECTS = ["battery", "sound", "build quality", "size", "price", "setup", "packaging", "noise level"]
MAIN = {
    0: ["Stopped working after {days} days. Total waste of money.", "Terrible quality, it broke within the first week.",
        "Awful. I returned it immediately.", "Do not buy this, it failed after {days} days."],
    1: ["Disappointing. It works but feels flimsy and overpriced.", "Not what I expected and several things annoy me.",
        "Mediocre at best, I would not buy it again."],
    2: ["It is a {product}. Does what it says.", "Average. Nothing special, nothing terrible.", "Okay for the price, no strong feelings."],
    3: ["Good value and it works well.", "Happy with it overall.", "Solid product, would recommend."],
    4: ["Absolutely love it, best purchase this year!", "Fantastic quality and it works perfectly.",
        "Exceeded my expectations, highly recommend!"],
}
POSITIVE = ["The {aspect} is great.", "Great {aspect} for the money.", "The {aspect} really impressed me."]
NEGATIVE = ["The {aspect} is disappointing.", "The {aspect} feels cheap.", "Sadly the {aspect} is a letdown."]


def review_text(rng: random.Random, level: int, product: str) -> str:
    fill = dict(days=rng.randint(2, 20), product=product.lower())
    sentences = [rng.choice(MAIN[level]).format(**fill)]
    aspect = rng.choice(ASPECTS)
    if level in (3, 4) and rng.random() < 0.7:
        sentences.append(rng.choice(POSITIVE).format(aspect=aspect))
    if level == 3 and rng.random() < 0.6:
        sentences.append(rng.choice(NEGATIVE).format(aspect=rng.choice([a for a in ASPECTS if a != aspect])))
    if level in (0, 1) and rng.random() < 0.7:
        sentences.append(rng.choice(NEGATIVE).format(aspect=aspect))
    if level == 1 and rng.random() < 0.4:
        sentences.append(rng.choice(POSITIVE).format(aspect=rng.choice([a for a in ASPECTS if a != aspect])))
    if level == 2 and rng.random() < 0.6:
        sentences = [rng.choice(POSITIVE).format(aspect=aspect), rng.choice(NEGATIVE).format(aspect=rng.choice([a for a in ASPECTS if a != aspect]))]
    return " ".join(sentences)


def review_sentiment(rng: random.Random, number: int):
    level = pick(rng, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
    product = rng.choice(PRODUCTS)
    return {"review_id": f"R-{50000 + number}", "product": product, "review": review_text(rng, level, product)}, level


# -------------    lead priority    ------------------------------
INDUSTRIES = ["software", "retail", "healthcare", "logistics", "education", "finance", "manufacturing"]
SOURCES = ["referral", "webinar", "inbound_form", "trade_show", "cold_outreach"]


def lead_priority_for(record: dict) -> int:
    points = 0
    points += 2 if record["company_size"] >= 500 else 1 if record["company_size"] >= 50 else 0
    points += 3 if record["budget_usd"] >= 100000 else 2 if record["budget_usd"] >= 25000 else 1 if record["budget_usd"] >= 5000 else 0
    points += 1 if record["last_contact_days"] <= 7 else -1 if record["last_contact_days"] > 90 else 0
    points += 2 if record["requested_demo"] else 0
    points += 1 if record["source"] == "referral" else 0
    return 0 if points <= 1 else 1 if points <= 3 else 2 if points <= 5 else 3


def lead(rng: random.Random, number: int):
    record = {
        "lead_id": f"L-{8000 + number}",
        "industry": rng.choice(INDUSTRIES),
        "source": rng.choice(SOURCES),
        "company_size": int(log_uniform(rng, 2, 5000)),
        "budget_usd": 0 if rng.random() < 0.15 else int(round(log_uniform(rng, 500, 300000), -2)),
        "last_contact_days": rng.randint(0, 240),
        "requested_demo": rng.random() < 0.35,
    }
    return record, lead_priority_for(record)


# -------------    assembly    -----------------------------------
GENERATORS = {
    "binary: urgent ticket": urgent_ticket,
    "binary: flag transaction": flag_transaction,
    "choice: route ticket": route_ticket,
    "choice: inventory action": inventory_item,
    "score: review sentiment": review_sentiment,
    "score: lead priority": lead,
}


def balanced_sample(generate, classes: int, seed: int) -> list[tuple[dict, int]]:
    """SIZE rows with each class within one row of an equal share, in shuffled order."""
    rng = random.Random(seed)
    quota = {index: SIZE // classes + (1 if index < SIZE % classes else 0) for index in range(classes)}
    rows, number = [], 0
    while any(quota.values()):
        number += 1
        if number > SIZE * 500:
            raise RuntimeError("could not fill every class; the label rule never produces one")
        record, answer = generate(rng, number)
        if quota[answer] > 0:
            quota[answer] -= 1
            rows.append((record, answer))
    rng.shuffle(rows)
    return rows


def slug(name: str) -> str:
    return name.replace(": ", "_").replace(" ", "_")


def build(name: str, seed: int) -> Path:
    task = DECISION_TASKS[name]
    classes = len(task["options"])
    rows = balanced_sample(GENERATORS[name], classes, seed)
    records = [record for record, _ in rows]
    config = decision_tab.config_for_example(task, records, decision_tab.DEFAULT_CONFIG)
    q = decision_tab.question(config)
    labels = {
        index: {
            "record": record, **q, "state": decision_tab.record_state(record, q["columns"]),
            "answer": answer, "option": q["options"][answer],
        }
        for index, (record, answer) in enumerate(rows)
    }
    space = dict(records=records, ids=list(range(len(rows))), labels=labels, position=0, config=config, view={}, drafts={}, notice=None)
    text = export_text(decision_tab, space)
    path = DATA_DIR / f"decision_{slug(name)}.jsonl"
    path.write_text(text, encoding="utf-8")
    check(path, records, labels)
    return path


def check(path: Path, records: list[dict], labels: dict) -> None:
    """Read the file back and require the same 500 samples and labels."""
    parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(parsed) == SIZE, f"{path.name}: {len(parsed)} rows"
    back_records, ids, back_labels = import_labeled(decision_tab, parsed)
    assert back_records == records and ids == list(range(SIZE)), f"{path.name}: samples changed"
    assert list(back_labels.values()) == list(labels.values()), f"{path.name}: labels changed"
    assert len({json.dumps(record, sort_keys=True) for record in records}) == SIZE, f"{path.name}: duplicate records"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for offset, name in enumerate(DECISION_TASKS):
        path = build(name, SEED + offset)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        counts = {option: sum(row["option"] == option for row in rows) for option in rows[0]["options"]}
        print(f"{path.relative_to(HERE.parents[1])}: {len(rows)} rows, {counts}")


if __name__ == "__main__":
    main()
