"""Field action for a crop from sensor readings, Dutch and Italian; waiting is the most common answer."""

from domains.common import clamp, pick, spec

WEIGHTS = [45, 22, 13, 12, 8]
CROPS = {
    "nl": [("tarwe", 150), ("aardappelen", 130), ("suikerbieten", 180), ("mais", 140), ("uien", 120)],
    "it": [("grano", 150), ("patate", 130), ("pomodori", 110), ("mais", 140), ("cipolle", 120)],
}


def advice(days: int, maturity: int, moisture: float, rain: float, nitrogen: float, pests: int) -> int:
    if days >= maturity:
        return 4
    if pests >= 15:
        return 3
    if moisture < 22 and rain < 5:
        return 1
    if nitrogen < 40 and days < maturity - 20:
        return 2
    return 0


def maker(language: str):
    def make(rng, want):
        crop, maturity = pick(rng, CROPS[language])
        ripe = rng.random() < 0.1
        days = int(rng.uniform(maturity - 3, maturity + 12)) if ripe else int(rng.uniform(5, maturity - 4))
        dry = rng.random() < 0.3
        moisture = round(clamp(rng.gauss(19 if dry else 31, 5 if dry else 6), 6, 55), 1)
        rain = round(clamp(rng.expovariate(1 / 5), 0, 40), 1)
        low_n = rng.random() < 0.25
        nitrogen = round(clamp(rng.gauss(32 if low_n else 62, 8 if low_n else 14), 5, 100), 0)
        infested = rng.random() < 0.15
        pests = int(clamp(rng.gauss(20 if infested else 4, 5 if infested else 3), 0, 60))
        record = {
            "crop": crop, "days_since_planting": days, "days_to_maturity": maturity, "soil_moisture_pct": moisture,
            "rain_forecast_mm": rain, "leaf_nitrogen_index": int(nitrogen), "pests_per_trap": pests,
        }
        return record, advice(days, maturity, moisture, rain, nitrogen, pests)

    return make


SPECS = [
    spec(
        "crop_advice_nl", "nl", "CHOICE", "Welke actie moet de teler nu nemen op dit perceel?",
        ["afwachten", "water geven", "bemesten", "plaagbestrijding", "oogsten"],
        WEIGHTS, "field_id", "NL-", maker("nl"),
    ),
    spec(
        "crop_advice_it", "it", "CHOICE", "Quale intervento va eseguito ora su questo appezzamento?",
        ["attendere", "irrigare", "concimare", "trattamento antiparassitario", "raccogliere"],
        WEIGHTS, "field_id", "IT-", maker("it"),
    ),
]
