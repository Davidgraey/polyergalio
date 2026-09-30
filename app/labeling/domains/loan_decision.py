"""Loan application decision (binary), Norwegian and Spanish; about two thirds are approved."""

from domains.common import clamp, log_uniform, pick, spec

WEIGHTS = [32, 68]
EMPLOYMENT = {
    "no": ["fast ansatt", "midlertidig ansatt", "selvstendig næringsdrivende", "arbeidsledig", "pensjonist"],
    "es": ["contrato indefinido", "contrato temporal", "autónomo", "desempleado", "jubilado"],
}
PURPOSE = {
    "no": ["bolig", "bil", "refinansiering", "studier", "oppussing", "forbruk"],
    "es": ["vivienda", "coche", "refinanciación", "estudios", "reforma", "consumo"],
}
EMPLOYMENT_WEIGHTS = [55, 12, 13, 8, 12]
CURRENCY_SCALE = {"no": 11, "es": 1}


def approve(income: float, amount: float, debt: float, score: int, years: float, employment: int) -> bool:
    ratio = (debt + amount) / max(income, 1)
    if employment == 3:
        return score >= 750 and ratio <= 1.0
    if score >= 680 and ratio <= 2.5:
        return True
    return score >= 620 and ratio <= 1.5 and years >= 2


def maker(language: str):
    scale = CURRENCY_SCALE[language]

    def make(rng, want):
        employment = rng.choices(range(5), weights=EMPLOYMENT_WEIGHTS)[0]
        base_income = log_uniform(rng, 14000, 90000)
        income = int(round(base_income * scale, -2))
        amount = int(round(log_uniform(rng, 1500, 220000) * scale, -2))
        debt = int(round(base_income * clamp(rng.gauss(0.6, 0.6), 0, 3.5) * scale, -2))
        score = int(clamp(rng.gauss(690, 70), 450, 850))
        years = 0.0 if employment in (3, 4) else round(clamp(rng.expovariate(1 / 6), 0, 35), 1)
        record = {
            "income_yearly": income, "loan_amount": amount, "existing_debt": debt, "credit_score": score,
            "employment": EMPLOYMENT[language][employment], "years_employed": years, "purpose": pick(rng, PURPOSE[language]),
        }
        return record, int(approve(income, amount, debt, score, years, employment))

    return make


SPECS = [
    spec(
        "loan_decision_no", "no", "BINARY", "Bør denne lånesøknaden godkjennes?",
        ["false", "true"], WEIGHTS, "application_id", "NO-", maker("no"),
    ),
    spec(
        "loan_decision_es", "es", "BINARY", "¿Se debe aprobar esta solicitud de préstamo?",
        ["false", "true"], WEIGHTS, "application_id", "ES-", maker("es"),
    ),
]
