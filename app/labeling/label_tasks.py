"""Example labeling tasks: two per decision type, two token tasks and two sequence tasks."""

DECISION_TASKS = {
    "binary: urgent ticket": {
        "kind": "BINARY",
        "instructions": "Is this support ticket urgent?",
        "options": ["false", "true"],
        "records": [
            {"subject": "Site down for all users", "hours_open": 1, "customer_tier": "enterprise"},
            {"subject": "How do I change my avatar?", "hours_open": 30, "customer_tier": "free"},
            {"subject": "Payment failed, account locked", "hours_open": 6, "customer_tier": "pro"},
            {"subject": "Feature request: dark mode", "hours_open": 72, "customer_tier": "pro"},
            {"subject": "Data export returns empty file", "hours_open": 48, "customer_tier": "enterprise"},
            {"subject": "Typo on pricing page", "hours_open": 12, "customer_tier": "free"},
        ],
    },
    "binary: flag transaction": {
        "kind": "BINARY",
        "instructions": "Should this transaction be flagged for manual review?",
        "options": ["false", "true"],
        "records": [
            {"amount": 4.50, "country": "US", "home_country": "US", "hour": 8},
            {"amount": 2900.00, "country": "RO", "home_country": "US", "hour": 3},
            {"amount": 65.20, "country": "US", "home_country": "US", "hour": 19},
            {"amount": 1500.00, "country": "US", "home_country": "US", "hour": 2},
            {"amount": 18.99, "country": "FR", "home_country": "FR", "hour": 13},
            {"amount": 940.00, "country": "NG", "home_country": "GB", "hour": 23},
        ],
    },
    "choice: route ticket": {
        "kind": "CHOICE",
        "instructions": "Which team should handle this ticket?",
        "options": ["billing", "technical", "account", "other"],
        "records": [
            {"subject": "Charged twice this month", "body": "I see two identical charges on my card."},
            {"subject": "App crashes on launch", "body": "Since the last update it closes immediately."},
            {"subject": "Cannot reset password", "body": "The reset email never arrives."},
            {"subject": "Invoice needs a VAT number", "body": "Please reissue invoice 4411 with our VAT id."},
            {"subject": "Do you sponsor events?", "body": "We are organizing a meetup in June."},
            {"subject": "API returns 500", "body": "POST /v2/orders fails intermittently."},
        ],
    },
    "choice: inventory action": {
        "kind": "CHOICE",
        "instructions": "What should be done with this inventory item?",
        "options": ["reorder", "discount", "hold", "discontinue"],
        "records": [
            {"item": "USB-C cable", "stock": 4, "weekly_sales": 40, "days_to_expiry": None},
            {"item": "Yogurt 500g", "stock": 120, "weekly_sales": 30, "days_to_expiry": 5},
            {"item": "Desk lamp", "stock": 60, "weekly_sales": 8, "days_to_expiry": None},
            {"item": "Phone case (old model)", "stock": 300, "weekly_sales": 1, "days_to_expiry": None},
            {"item": "Notebook A5", "stock": 15, "weekly_sales": 20, "days_to_expiry": None},
            {"item": "Fresh basil", "stock": 50, "weekly_sales": 45, "days_to_expiry": 2},
        ],
    },
    "score: review sentiment": {
        "kind": "SCORE",
        "instructions": "Rate the sentiment of this review: 0 very negative, 2 neutral, 4 very positive.",
        "options": ["0", "1", "2", "3", "4"],
        "records": [
            {"product": "Kettle", "review": "Stopped working after two days. Total waste of money."},
            {"product": "Headphones", "review": "Sound is great and the battery lasts all week."},
            {"product": "Backpack", "review": "It is a backpack. Does what it says."},
            {"product": "Blender", "review": "Powerful, but loud and a pain to clean."},
            {"product": "Monitor", "review": "Absolutely love it, best purchase this year!"},
            {"product": "Mouse", "review": "Works fine, though the scroll wheel feels cheap."},
        ],
    },
    "score: lead priority": {
        "kind": "SCORE",
        "instructions": "Rate the sales priority of this lead: 0 ignore, 1 low, 2 medium, 3 high.",
        "options": ["0", "1", "2", "3"],
        "records": [
            {"company_size": 5, "budget_usd": 2000, "last_contact_days": 90, "requested_demo": False},
            {"company_size": 800, "budget_usd": 150000, "last_contact_days": 2, "requested_demo": True},
            {"company_size": 40, "budget_usd": 12000, "last_contact_days": 14, "requested_demo": True},
            {"company_size": 1200, "budget_usd": 0, "last_contact_days": 200, "requested_demo": False},
            {"company_size": 150, "budget_usd": 40000, "last_contact_days": 7, "requested_demo": False},
            {"company_size": 12, "budget_usd": 5000, "last_contact_days": 30, "requested_demo": True},
        ],
    },
}

TOKEN_TASKS = {
    "token: part of speech": {
        "tags": ["X", "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "PUNCT"],
        "records": [
            {"id": 1, "text": "The quick fox jumps over the lazy dog."},
            {"id": 2, "text": "She quietly reads three old books."},
            {"id": 3, "text": "We walked to the river at dawn."},
            {"id": 4, "text": "Prices rose sharply in March."},
            {"id": 5, "text": "He never answers his phone!"},
            {"id": 6, "text": "Two small birds sang in the garden."},
        ],
    },
    "token: recipe ingredients": {
        "tags": ["O", "QTY", "UNIT", "FOOD", "PREP"],
        "records": [
            {"id": 1, "text": "2 cups finely chopped onions"},
            {"id": 2, "text": "1 tbsp olive oil, plus extra for frying"},
            {"id": 3, "text": "500 g chicken thighs, skinless and diced"},
            {"id": 4, "text": "3 cloves garlic, minced"},
            {"id": 5, "text": "a pinch of sea salt"},
            {"id": 6, "text": "1/2 cup fresh basil leaves, torn"},
        ],
    },
}

SEQUENCE_TASKS = {
    "sequence: support tickets": {
        "span_types": ["PRODUCT", "ORDER_ID", "DATE", "PERSON"],
        "labels": ["refund", "bug", "question", "complaint"],
        "records": [
            {"id": 1, "text": "Order 88213 arrived on Monday with a cracked Aero kettle, I want my money back."},
            {"id": 2, "text": "The Pulse app crashes every time I open settings since the June update."},
            {"id": 3, "text": "Does the Nimbus backpack fit a 16 inch laptop?"},
            {"id": 4, "text": "Maria from support promised a callback on Friday and nobody called."},
            {"id": 5, "text": "Please refund order 90417, the Lumen desk lamp never shipped."},
            {"id": 6, "text": "Sync fails on the Pulse web dashboard after login, error 500."},
        ],
    },
    "sequence: news headlines": {
        "span_types": ["PER", "ORG", "LOC"],
        "labels": ["business", "politics", "sports", "science"],
        "records": [
            {"id": 1, "text": "Acme Corp shares jump after record quarter in Europe"},
            {"id": 2, "text": "Senator Alvarez unveils housing plan in Denver"},
            {"id": 3, "text": "Lina Park scores twice as Rovers beat United in Leeds"},
            {"id": 4, "text": "NASA probe finds water ice near the lunar south pole"},
            {"id": 5, "text": "Helix Bank to cut 2,000 jobs across Asia"},
            {"id": 6, "text": "Geneva talks stall as Minister Okafor walks out"},
        ],
    },
}
