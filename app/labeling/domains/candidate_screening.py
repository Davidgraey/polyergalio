"""Candidate screening outcome, Korean and English; most applications are rejected at the first pass."""

from domains.common import clamp, pick, spec

WEIGHTS = [55, 25, 15, 5]
ROLES = {
    "ko": ["백엔드 개발자", "데이터 분석가", "마케팅 매니저", "UX 디자이너", "영업 담당자"],
    "en": ["backend developer", "data analyst", "marketing manager", "UX designer", "sales representative"],
}
EDUCATION = {"ko": ["고졸", "학사", "석사", "박사"], "en": ["high school", "bachelor's", "master's", "doctorate"]}
CITIES = {"ko": ["서울", "부산", "인천", "대전", "광주"], "en": ["London", "Manchester", "Leeds", "Bristol", "Glasgow"]}


def outcome(skills: float, years: float, education: int, salary_ratio: float, referral: bool) -> int:
    if skills < 35:
        return 0
    score = skills * 0.6 + min(years, 10) * 3 + (8 if referral else 0) + (4 if education >= 2 else 0) - (10 if salary_ratio > 1.2 else 0)
    return 0 if score < 50 else 1 if score < 65 else 2 if score < 80 else 3


def maker(language: str):
    def make(rng, want):
        strong = rng.random() < 0.25
        skills = round(clamp(rng.gauss(74 if strong else 45, 14 if strong else 18), 5, 100), 0)
        years = round(clamp(rng.expovariate(1 / (6 if strong else 3.5)), 0, 30), 1)
        education = rng.choices(range(4), weights=[10, 55, 28, 7])[0]
        salary_ratio = round(clamp(rng.gauss(1.0, 0.18), 0.6, 1.8), 2)
        referral = rng.random() < 0.12
        record = {
            "role": pick(rng, ROLES[language]), "years_experience": years, "skills_match_pct": int(skills),
            "education": EDUCATION[language][education], "location": pick(rng, CITIES[language]),
            "salary_expectation_ratio": salary_ratio, "notice_period_weeks": pick(rng, [0, 2, 4, 4, 8, 12]), "referral": referral,
        }
        return record, outcome(skills, years, education, salary_ratio, referral)

    return make


SPECS = [
    spec(
        "candidate_screening_ko", "ko", "CHOICE", "이 지원자를 서류 심사에서 어떻게 처리해야 하나요?",
        ["서류 탈락", "전화 스크리닝", "면접 진행", "우선 진행 후보"],
        WEIGHTS, "candidate_id", "KO-", maker("ko"),
    ),
    spec(
        "candidate_screening_en", "en", "CHOICE", "How should this candidate be handled in the first screening?",
        ["reject", "phone screen", "interview", "fast-track"],
        WEIGHTS, "candidate_id", "EN-", maker("en"),
    ),
]
