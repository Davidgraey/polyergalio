"""Forum comment moderation, German and Norwegian; most comments are fine, spam and insults are rare."""

from domains.common import number_between, pick, spec, text_maker

WEIGHTS = [62, 12, 10, 9, 7]

GERMAN = [
    [
        "Danke für die Infos zu {topic}, das war sehr hilfreich.",
        "Ich beschäftige mich auch mit {topic} und habe eine Frage zum Einstieg.",
        "Bei mir hat das genauso funktioniert, danke für den Tipp!",
        "Ich sehe das etwas anders, aber deine Erfahrung ist interessant.",
        "Kann jemand erklären, wie man bei {topic} am besten anfängt?",
        "Interessanter Beitrag, vor allem der Teil über {topic}.",
        "Hat das schon jemand länger ausprobiert? Ich bin gespannt auf eure Erfahrungen.",
        "Ich habe ähnliche Erfahrungen bei {topic} gemacht, es lohnt sich.",
        "Nach einem Jahr {topic} kann ich das bestätigen.",
        "Guter Hinweis, das werde ich am Wochenende ausprobieren.",
        "Gibt es dazu eine einfache Anleitung für Anfänger?",
    ],
    [
        "GEWINNSPIEL!!! Klick hier: bit.example/{code} und gewinne {amount} Euro",
        "Verdiene {amount} Euro am Tag von zu Hause, kein Risiko! Jetzt anklicken!!!",
        "Gratis-Geschenk für dich, gib einfach deine Kartennummer ein: bit.example/{code}",
    ],
    [
        "Hat jemand einen guten Tipp für einen Steuerberater in Köln?",
        "Wer schaut heute Abend das Fußballspiel?",
        "Weiß jemand, wo ich günstige Flüge nach Spanien finde?",
        "Meine Waschmaschine leckt, hat jemand eine Idee?",
    ],
    [
        "Wir verkaufen hochwertige {product} zu Top-Preisen, besuchen Sie unseren Shop!",
        "Mein Kanal hat ein neues Video, schaut vorbei und abonniert!",
        "Jetzt bei uns bestellen und {pct} Prozent Rabatt sichern.",
    ],
    [
        "Was für ein dummer Beitrag, du hast keine Ahnung.",
        "Halt einfach den Mund, du Idiot.",
        "Leute wie du sollten das Forum verlassen.",
        "Du Trottel, lies erst mal nach, bevor du schreibst.",
    ],
]
NORWEGIAN = [
    [
        "Takk for tipsene om {topic}, veldig nyttig!",
        "Jeg holder også på med {topic} og lurer på hvordan man kommer i gang.",
        "Det samme skjedde med meg, så takk for at du delte.",
        "Jeg ser det litt annerledes, men erfaringen din er interessant.",
        "Kan noen forklare hva som er best å starte med innen {topic}?",
        "Interessant innlegg, spesielt delen om {topic}.",
        "Har noen prøvd dette lenge? Jeg er nysgjerrig på erfaringene deres.",
        "Jeg har hatt lignende erfaringer med {topic}, det er verdt det.",
        "Etter et år med {topic} kan jeg bekrefte dette.",
        "Bra poeng, det skal jeg prøve i helgen.",
        "Finnes det en enkel veiledning for nybegynnere?",
    ],
    [
        "VINN {amount} kroner nå!!! Trykk her: bit.example/{code}",
        "Tjen {amount} kr om dagen hjemmefra, ingen risiko! Klikk her!!!",
        "Gratis gave til deg, bare legg inn kortnummeret ditt: bit.example/{code}",
    ],
    [
        "Er det noen som vet om en god rørlegger i Bergen?",
        "Hvem ser fotballkampen i kveld?",
        "Noen som har tips til billige flyreiser til Spania?",
        "Vaskemaskinen min lekker, noen som har en idé?",
    ],
    [
        "Vi selger {product} til gode priser, sjekk nettbutikken vår!",
        "Kanalen min har ny video, kom innom og abonner!",
        "Bestill hos oss i dag og få {pct} % rabatt.",
    ],
    [
        "For et dumt innlegg, du har ikke peiling.",
        "Hold kjeft, din idiot.",
        "Folk som deg burde forlate forumet.",
        "Din tosk, les før du skriver.",
    ],
]
TOPICS = {
    "de": ["Radfahren", "Gartenarbeit", "Fotografie", "Kochen", "Wandern"],
    "no": ["sykling", "hagearbeid", "fotografering", "matlaging", "fjellturer"],
}
PRODUCTS = {"de": ["Sneaker", "Kaffeemaschinen", "Handyhüllen"], "no": ["joggesko", "kaffemaskiner", "mobildeksler"]}


FRAMES = {
    "de": (("", "", "Hallo!", "Hallo zusammen!"), ("", "", "Gruß", "Danke!", "LG")),
    "no": (("", "", "Hei!", "Hallo alle sammen!"), ("", "", "Hilsen", "Takk!", "Mvh")),
}


def maker(language: str, templates: list):
    slots = dict(
        topic=TOPICS[language], product=PRODUCTS[language], amount=number_between(100, 900),
        code=lambda rng: "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(6)),
        pct=[10, 20, 30, 50],
    )
    inner_slots = {key: value for key, value in slots.items()}

    def make(rng, want):
        def extra(rng):
            spam_like = want in (1, 3)
            return {
                "thread_topic": pick(rng, TOPICS[language]),
                "account_age_days": rng.randint(0, 40) if spam_like and rng.random() < 0.7 else int(rng.expovariate(1 / 400)) + 1,
                "previous_flags": rng.choice([1, 2, 3, 5]) if want != 0 and rng.random() < 0.4 else 0,
            }

        greetings, closings = FRAMES[language]
        return text_maker(templates, inner_slots, "comment", greetings=greetings, closings=closings, extra=extra)(rng, want)

    return make


SPECS = [
    spec(
        "comment_moderation_de", "de", "CHOICE", "Wie soll dieser Forumskommentar moderiert werden?",
        ["in Ordnung", "Spam", "Thema verfehlt", "Werbung", "Beleidigung"],
        WEIGHTS, "comment_id", "DE-", maker("de", GERMAN),
    ),
    spec(
        "comment_moderation_no", "no", "CHOICE", "Hvordan skal denne kommentaren i forumet modereres?",
        ["greit", "spam", "utenfor tema", "reklame", "krenkelse"],
        WEIGHTS, "comment_id", "NO-", maker("no", NORWEGIAN),
    ),
]
