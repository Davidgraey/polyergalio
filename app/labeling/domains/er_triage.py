"""Emergency department triage level (ESI style), Italian and Japanese; most patients are level 3 or 4."""

from domains.common import clamp, pick, spec, weighted

WEIGHTS = [2, 10, 38, 34, 16]

COMPLAINTS = [
    ("cardiac_arrest", 0.4, 0, False, "arresto cardiaco", "心停止"),
    ("chest_pain", 4, 2, True, "dolore toracico intenso", "強い胸の痛み"),
    ("stroke", 2.5, 2, True, "debolezza al braccio e difficoltà a parlare", "腕の脱力と言葉のもつれ"),
    ("breathing", 3, 2, True, "grave difficoltà respiratoria", "ひどい呼吸困難"),
    ("abdominal", 9, 2, False, "dolore addominale", "腹痛"),
    ("fracture", 7, 2, False, "sospetta frattura del polso", "手首の骨折の疑い"),
    ("cut", 7, 1, False, "taglio profondo alla mano", "手の深い切り傷"),
    ("sprain", 11, 1, False, "distorsione alla caviglia", "足首の捻挫"),
    ("urinary", 6, 1, False, "dolore alla minzione", "排尿時の痛み"),
    ("migraine", 7, 1, False, "forte mal di testa", "強い頭痛"),
    ("cold", 14, 0, False, "tosse e raffreddore", "咳と鼻水"),
    ("rash", 8, 0, False, "lieve eruzione cutanea", "軽い発疹"),
    ("throat", 8, 0, False, "mal di gola", "のどの痛み"),
    ("refill", 6, 0, False, "richiesta di ricetta", "処方箋の再発行"),
]
ARRIVAL = {"it": ["autonomo", "ambulanza", "trasferito"], "ja": ["自力来院", "救急搬送", "転院搬送"]}


def triage_level(complaint, vitals) -> int:
    code, _, resources, high_risk = complaint[:4]
    heart_rate, systolic, spo2, temperature, pain = vitals
    if code == "cardiac_arrest" or spo2 < 85 or systolic < 70 or heart_rate > 150:
        return 0
    if high_risk and (pain >= 7 or code != "chest_pain") or spo2 < 92 or heart_rate > 130 or systolic < 90 or temperature >= 40:
        return 1
    danger = heart_rate > 100 or spo2 < 95 or temperature >= 38.5 or pain >= 8
    if resources >= 2:
        return 2
    if resources == 1:
        return 2 if danger else 3
    return 3 if danger else 4


def maker(language: str):
    index = 4 if language == "it" else 5
    weights = [entry[1] for entry in COMPLAINTS]

    def make(rng, want):
        complaint = COMPLAINTS[weighted(rng, weights)]
        unwell = rng.random() < 0.25
        extreme = rng.random() < 0.02
        vitals = (
            int(clamp(rng.gauss(96 if unwell else 78, 14), 40, 190 if extreme else 165)),
            int(clamp(rng.gauss(118 if not unwell else 108, 16), 60, 200)),
            int(clamp(rng.gauss(96.5 if not unwell else 93, 2.2), 70, 100)),
            round(clamp(rng.gauss(37.0 if not unwell else 38.2, 0.6), 35, 41.5), 1),
            int(clamp(rng.gauss(3 if not unwell else 6, 2.2), 0, 10)),
        )
        if extreme:
            vitals = (int(rng.choice([155, 160, 45])), int(rng.choice([65, 68, 85])), int(rng.choice([80, 84, 90])), vitals[3], vitals[4])
        record = {
            "age": int(clamp(rng.gauss(46, 22), 1, 97)), "complaint": complaint[index], "arrival": pick(rng, ARRIVAL[language]),
            "heart_rate": vitals[0], "systolic_bp": vitals[1], "spo2": vitals[2], "temperature_c": vitals[3], "pain_score": vitals[4],
        }
        return record, triage_level(complaint, vitals)

    return make


SPECS = [
    spec(
        "er_triage_it", "it", "CHOICE", "Quale livello di triage va assegnato a questo paziente?",
        ["1 - rianimazione", "2 - emergenza", "3 - urgenza", "4 - urgenza minore", "5 - non urgente"],
        WEIGHTS, "patient_id", "IT-", maker("it"),
    ),
    spec(
        "er_triage_ja", "ja", "CHOICE", "この患者にはどのトリアージレベルを割り当てますか？",
        ["1 蘇生", "2 緊急", "3 準緊急", "4 低緊急", "5 非緊急"],
        WEIGHTS, "patient_id", "JA-", maker("ja"),
    ),
]
