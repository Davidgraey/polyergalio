"""Inbox triage, German and Italian; newsletters and spam make up half of the mail."""

from domains.common import clock, pick, spec

WEIGHTS = [8, 18, 14, 26, 12, 22]

GERMAN = [
    [
        ("Dringend: Antwort bis heute {time} Uhr erforderlich", "Bitte bestätigen Sie sofort die Freigabe, sonst verzögert sich die Lieferung an unseren Kunden."),
        ("Produktionsstopp in Halle {n} – bitte sofort melden", "Die Anlage steht seit {time} Uhr, wir brauchen eine Entscheidung vom Bereichsleiter."),
        ("Eskalation: Kunde droht mit Kündigung", "Der Kunde hat heute Morgen erneut angerufen. Bitte kümmern Sie sich umgehend darum."),
        ("Wichtig: Frist läuft heute ab", "Die Angebotsfrist endet um {time} Uhr. Es fehlen noch zwei Unterschriften."),
        ("Sicherheitsvorfall: Passwort zurücksetzen", "Bitte ändern Sie noch heute Ihr Passwort, ein Konto wurde kompromittiert."),
    ],
    [
        ("Einladung: Projektbesprechung am {weekday}", "Hallo zusammen, wir treffen uns am {weekday} um {time} Uhr im Besprechungsraum {n}."),
        ("Terminverschiebung: Jour fixe", "Der Jour fixe wird auf {weekday}, {time} Uhr verschoben. Bitte Kalender aktualisieren."),
        ("Können wir einen Termin für nächste Woche finden?", "Ich würde gern die Roadmap mit Ihnen durchgehen. Passt Ihnen {weekday} um {time} Uhr?"),
        ("Agenda für das Quartalsgespräch", "Anbei die Agenda für {weekday}. Bitte ergänzen Sie Ihre Punkte bis morgen."),
    ],
    [
        ("Rechnung Nr. {n} vom {month}", "Anbei erhalten Sie die Rechnung über {amount} Euro. Zahlungsziel: 14 Tage."),
        ("Zahlungserinnerung: Rechnung {n}", "Leider konnten wir bisher keinen Zahlungseingang über {amount} Euro feststellen."),
        ("Ihre Abrechnung für {month} ist verfügbar", "Der Betrag von {amount} Euro wird in den nächsten Tagen abgebucht."),
        ("Gutschrift zu Rechnung {n}", "Wir haben Ihnen {amount} Euro gutgeschrieben. Die Gutschrift finden Sie im Anhang."),
    ],
    [
        ("Unsere Neuheiten im {month}", "Entdecken Sie die neuen Produkte und sichern Sie sich {pct} Prozent Rabatt auf Ihre nächste Bestellung."),
        ("Newsletter: Die besten Tipps für den Alltag", "In dieser Ausgabe: Rezepte, Reisetipps und ein Gewinnspiel. Jetzt lesen."),
        ("Ihr Wochenrückblick", "Die wichtigsten Nachrichten der Woche kompakt zusammengefasst. Zum Abbestellen hier klicken."),
        ("{pct} % Rabatt nur an diesem Wochenende", "Nur bis Sonntag: Sonderaktion in unserem Onlineshop. Abmelden ist jederzeit möglich."),
    ],
    [
        ("Wochenende?", "Hast du am {weekday} Zeit? Wir könnten grillen, wenn das Wetter mitspielt."),
        ("Fotos vom Urlaub", "Ich habe dir die Bilder von unserer Reise geschickt. Sag mir, welche dir gefallen."),
        ("Geburtstag von Oma", "Wir wollen zusammen etwas schenken. Bist du dabei?"),
        ("Danke für gestern Abend!", "Es war ein toller Abend. Lass uns das bald wiederholen."),
    ],
    [
        ("Sie haben gewonnen! Jetzt Preis abholen", "Herzlichen Glückwunsch, Sie wurden ausgewählt. Klicken Sie hier und geben Sie Ihre Daten ein."),
        ("Ihr Konto wurde gesperrt – sofort handeln", "Bestätigen Sie Ihre Bankdaten über den folgenden Link, um Ihr Konto zu entsperren."),
        ("Günstige Medikamente ohne Rezept", "Bestellen Sie diskret und bequem. Nur heute {pct} Prozent Rabatt."),
        ("Schnell reich werden mit Kryptowährung", "Verdienen Sie täglich {amount} Euro von zu Hause. Kein Risiko, garantiert."),
    ],
]
ITALIAN = [
    [
        ("Urgente: risposta entro le {time}", "Ti chiediamo di confermare subito l'approvazione, altrimenti la consegna al cliente subirà un ritardo."),
        ("Blocco della produzione nel reparto {n}", "L'impianto è fermo dalle {time}. Serve una decisione dal responsabile."),
        ("Escalation: il cliente minaccia di disdire", "Il cliente ha richiamato stamattina. Occupatevene subito, per favore."),
        ("Scadenza oggi: firme mancanti", "Il termine dell'offerta scade alle {time}. Mancano ancora due firme."),
        ("Incidente di sicurezza: cambia la password", "Cambia la password oggi stesso, un account è stato compromesso."),
    ],
    [
        ("Invito: riunione di progetto {weekday}", "Ciao a tutti, ci vediamo {weekday} alle {time} nella sala riunioni {n}."),
        ("Spostamento della riunione settimanale", "La riunione slitta a {weekday} alle {time}. Aggiornate il calendario, grazie."),
        ("Troviamo un momento la prossima settimana?", "Vorrei rivedere la roadmap con te. Ti va bene {weekday} alle {time}?"),
        ("Ordine del giorno del colloquio trimestrale", "In allegato l'ordine del giorno di {weekday}. Aggiungete i vostri punti entro domani."),
    ],
    [
        ("Fattura n. {n} di {month}", "In allegato la fattura di {amount} euro. Pagamento a 30 giorni."),
        ("Sollecito di pagamento: fattura {n}", "Non risulta ancora il pagamento di {amount} euro."),
        ("Il tuo estratto conto di {month} è disponibile", "L'importo di {amount} euro sarà addebitato nei prossimi giorni."),
        ("Nota di credito relativa alla fattura {n}", "Abbiamo accreditato {amount} euro. Trovi il documento in allegato."),
    ],
    [
        ("Le novità di {month}", "Scopri i nuovi prodotti e ottieni il {pct}% di sconto sul prossimo ordine."),
        ("Newsletter: i migliori consigli per la casa", "In questo numero ricette, idee di viaggio e un concorso. Leggi ora."),
        ("Il tuo riepilogo settimanale", "Le notizie più importanti della settimana in breve. Per annullare l'iscrizione clicca qui."),
        ("Sconto del {pct}% solo questo weekend", "Offerta valida fino a domenica nel nostro negozio online. Puoi disiscriverti in qualsiasi momento."),
    ],
    [
        ("Weekend?", "Sei libero {weekday}? Potremmo fare una grigliata se il tempo regge."),
        ("Foto della vacanza", "Ti ho mandato le foto del viaggio. Dimmi quali ti piacciono."),
        ("Compleanno della nonna", "Vorremmo farle un regalo insieme. Ci stai?"),
        ("Grazie per ieri sera!", "È stata una serata bellissima. Rifacciamola presto."),
    ],
    [
        ("Hai vinto! Ritira subito il premio", "Congratulazioni, sei stato selezionato. Clicca qui e inserisci i tuoi dati."),
        ("Il tuo conto è stato bloccato: agisci ora", "Conferma i dati bancari tramite il link per sbloccare il conto."),
        ("Farmaci economici senza ricetta", "Ordina in modo discreto. Solo oggi sconto del {pct}%."),
        ("Diventa ricco con le criptovalute", "Guadagna {amount} euro al giorno da casa. Nessun rischio, garantito."),
    ],
]
SENDERS = {
    "de": [
        ["chef@firma-beispiel.example", "leitung@werk-nord.example", "support@kunde-ag.example"],
        ["kalender@firma-beispiel.example", "projekt@firma-beispiel.example", "anna.keller@firma-beispiel.example"],
        ["buchhaltung@lieferant.example", "rechnung@stadtwerke.example", "abrechnung@telekom-dienst.example"],
        ["news@shop-mode.example", "info@reisewelt.example", "wochenblick@presse.example"],
        ["lena.meier@mail.example", "opa.horst@mail.example", "jonas.schmitt@mail.example"],
        ["gewinn@aktion-preis.example", "sicherheit@bank-check.example", "angebot@pharma-shop24.example"],
    ],
    "it": [
        ["direzione@azienda-esempio.example", "stabilimento@fabbrica-nord.example", "assistenza@cliente-spa.example"],
        ["calendario@azienda-esempio.example", "progetto@azienda-esempio.example", "marco.rossi@azienda-esempio.example"],
        ["amministrazione@fornitore.example", "fatture@energia-italia.example", "estratto@telefonia.example"],
        ["news@negozio-moda.example", "info@viaggi-mondo.example", "settimana@giornale.example"],
        ["giulia.bianchi@posta.example", "nonno.carlo@posta.example", "luca.ferrari@posta.example"],
        ["premio@promo-vincita.example", "sicurezza@banca-verifica.example", "offerte@farmacia-web24.example"],
    ],
}
SLOTS = {
    "de": dict(
        weekday=["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"],
        month=["Januar", "Februar", "März", "April", "Mai", "Juni"],
    ),
    "it": dict(
        weekday=["lunedì", "martedì", "mercoledì", "giovedì", "venerdì"],
        month=["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno"],
    ),
}


def maker(language: str, templates: list):
    def make(rng, want):
        subject, snippet = pick(rng, templates[want])
        values = dict(
            time=clock(rng), n=str(rng.randint(2, 9999)), amount=f"{rng.randint(20, 4800)},{rng.randint(0, 99):02d}",
            pct=str(pick(rng, [10, 15, 20, 30, 40, 50])), **{key: pick(rng, options) for key, options in SLOTS[language].items()},
        )
        hour = rng.randint(0, 5) if want == 5 and rng.random() < 0.4 else rng.randint(6, 23)
        record = {
            "sender": pick(rng, SENDERS[language][want]), "subject": subject.format(**values),
            "preview": snippet.format(**values), "received_hour": hour, "has_attachment": want == 2 or rng.random() < 0.1,
        }
        return record, want

    return make


SPECS = [
    spec(
        "email_triage_de", "de", "CHOICE", "Wie soll diese E-Mail im Posteingang einsortiert werden?",
        ["dringend", "Besprechung", "Rechnung", "Newsletter", "privat", "Spam"],
        WEIGHTS, "email_id", "DE-", maker("de", GERMAN),
    ),
    spec(
        "email_triage_it", "it", "CHOICE", "In quale categoria va classificata questa email?",
        ["urgente", "riunione", "fattura", "newsletter", "personale", "spam"],
        WEIGHTS, "email_id", "IT-", maker("it", ITALIAN),
    ),
]
