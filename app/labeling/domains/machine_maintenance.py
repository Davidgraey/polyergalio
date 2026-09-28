"""Machine maintenance action from sensor readings, Dutch and English; most machines need nothing."""

from domains.common import clamp, pick, spec

WEIGHTS = [70, 17, 9, 4]
TYPES = {
    "nl": ["compressor", "hydraulische pers", "transportband", "pomp", "generator"],
    "en": ["compressor", "hydraulic press", "conveyor", "pump", "generator"],
}


def action(vibration, temperature, hours, pressure, errors) -> int:
    if vibration >= 11 or temperature >= 105 or pressure < 1.0:
        return 3
    if hours >= 1800 or vibration >= 7 or temperature >= 95 or errors >= 6:
        return 2
    if errors >= 2 or vibration >= 4.5 or temperature >= 85 or hours >= 1200:
        return 1
    return 0


def poisson(rng, mean: float) -> int:
    limit, count, product = 2.718281828 ** -mean, 0, rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def maker(language: str):
    def make(rng, want):
        stress = rng.random() < 0.3
        vibration = round(clamp(rng.gauss(4.5 if stress else 2.6, 2.2 if stress else 1.0), 0.3, 16), 1)
        temperature = round(clamp(rng.gauss(88 if stress else 71, 10 if stress else 7), 40, 120), 1)
        hours = int(rng.uniform(0, 2300))
        pressure = round(clamp(rng.gauss(3.6 if stress else 4.1, 1.0 if stress else 0.4), 0.4, 6), 2)
        errors = poisson(rng, 2.5 if stress else 0.6)
        record = {
            "machine_type": pick(rng, TYPES[language]), "vibration_mm_s": vibration, "temperature_c": temperature,
            "hours_since_service": hours, "oil_pressure_bar": pressure, "error_codes_last_week": errors,
        }
        return record, action(vibration, temperature, hours, pressure, errors)

    return make


SPECS = [
    spec(
        "machine_maintenance_nl", "nl", "CHOICE", "Welke onderhoudsactie is nu nodig voor deze machine?",
        ["geen actie", "inspecteren", "binnenkort onderhoud", "direct stilleggen"],
        WEIGHTS, "machine_id", "NL-", maker("nl"),
    ),
    spec(
        "machine_maintenance_en", "en", "CHOICE", "Which maintenance action does this machine need now?",
        ["no action", "inspect", "schedule service", "shut down"],
        WEIGHTS, "machine_id", "EN-", maker("en"),
    ),
]
