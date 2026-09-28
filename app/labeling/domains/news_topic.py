"""News headline topic, Japanese and Dutch; politics, economy and sports dominate."""

from domains.common import date_2026, number_between, pick, spec, text_maker

WEIGHTS = [22, 20, 18, 12, 9, 8, 6, 5]

JAPANESE = [
    [
        "{party}が{policy}法案を国会に提出",
        "首相、{country}との首脳会談で{topic}について協議",
        "{city}市長選、現職が3選を果たす",
        "内閣支持率が{approval}%に下落　世論調査",
    ],
    [
        "日経平均が{index}円台を回復　{cause}を好感",
        "{company}、通期の営業利益が{pct}%増　過去最高を更新",
        "円相場、1ドル={usd}円台で推移",
        "{industry}の倒産件数が前年比{pct}%増加",
    ],
    [
        "{team}が{score}で{opp}を破り{wins}連勝",
        "{player}選手が今季{hr}号本塁打",
        "{event}で日本勢が金メダルを獲得",
        "{team}、主将の{player}選手が負傷で離脱",
    ],
    [
        "{tech_company}、新型{device}を発表　{feature}を搭載",
        "生成AIの利用に関する新ガイドラインを政府が公表",
        "5G通信網が{region}で運用開始",
        "{tech_company}が{nm}ナノメートルの半導体の量産を開始",
    ],
    [
        "{disease}の患者数が{wins}週連続で増加　厚生労働省まとめ",
        "{pharma}が新型ワクチンの承認を申請",
        "{food}の摂取と{effect}の関連を確認　大学の研究チーム",
    ],
    [
        "人気{genre}『{title}』の実写映画化が決定",
        "{artist}が全国{wins}公演のツアーを発表",
        "『{title}』が{award}で作品賞を受賞",
    ],
    [
        "{region}で記録的な{weather}　気象庁が警戒呼びかけ",
        "政府、{year}年までにCO2排出量を{pct}%削減する新目標を発表",
        "{animal}の生息数が回復　{region}の保護区で確認",
    ],
    [
        "{city}で{crime}の疑い　{age}歳の男を逮捕",
        "特殊詐欺グループの{wins}人を逮捕　被害総額{amount}億円",
        "{city}市内で連続{crime}事件　警察がパトロール強化",
    ],
]
JAPANESE_SLOTS = dict(
    tech_company=["北斗エレクトロニクス", "青空デバイス", "ミライテック", "ソラリス電機"], pharma=["みどり製薬", "北斗ファーマ", "あおば薬品"],
    party=["与党", "野党", "連立与党"], policy=["税制改正", "教育無償化", "防衛費増額", "年金改革", "子育て支援"],
    country=["米国", "韓国", "フランス", "インド", "オーストラリア"], topic=["貿易", "安全保障", "エネルギー", "気候変動"],
    city=["横浜", "札幌", "福岡", "仙台", "神戸"], approval=number_between(28, 58),
    index=lambda rng: f"{rng.randint(30, 45)},000", cause=["好決算", "円安", "米株高"],
    company=["大和精機", "北斗ホールディングス", "青空電機", "みどり食品", "光洋運輸"], pct=number_between(3, 25),
    usd=number_between(135, 165), industry=["建設業", "飲食業", "小売業", "運輸業"],
    team=["東京ブレイブス", "大阪サンライズ", "名古屋ファルコンズ", "福岡スターズ"],
    opp=["神戸ウェーブ", "広島フェニックス", "仙台ナイツ"], score=["3-1", "2-0", "4-2", "1-0"],
    wins=number_between(3, 9), player=["山本", "佐藤", "田中", "鈴木", "高橋"], hr=number_between(10, 45),
    event=["世界選手権", "アジア大会", "全国大会"], device=["スマートフォン", "ノートパソコン", "スマートウォッチ"],
    feature=["AI機能", "高速充電", "折りたたみ式ディスプレイ"], region=["東北地方", "九州全域", "北海道"],
    nm=["2", "3", "5"], disease=["インフルエンザ", "ノロウイルス", "溶連菌感染症"], food=["納豆", "緑茶", "魚"],
    effect=["血圧の低下", "認知症リスクの低下", "睡眠の質の改善"], genre=["漫画", "小説", "アニメ"],
    title=["月影の旅人", "星降る街", "青い季節", "海辺のカフェ"], artist=["ミナト", "サクラ・バンド", "ハルカ"],
    award=["国際映画祭", "日本映画大賞"], weather=["猛暑", "大雨", "大雪"], year=["2030", "2035", "2040"],
    animal=["トキ", "ニホンカモシカ", "ウミガメ"], crime=["空き巣", "車上荒らし", "自転車盗"],
    age=number_between(19, 64), amount=number_between(2, 30),
)

DUTCH = [
    [
        "Kabinet dient wetsvoorstel over {policy} in bij de Tweede Kamer",
        "Premier bespreekt {topic} met {leader} tijdens top in {city}",
        "De {party} verliest zetels bij gemeenteraadsverkiezingen in {city}",
        "Peiling: steun voor de coalitie daalt naar {approval} procent",
    ],
    [
        "AEX sluit {pct_small} procent hoger na {cause}",
        "{company} verhoogt winstverwachting na sterk kwartaal",
        "Inflatie in {month} gedaald naar {pct_small} procent",
        "Aantal faillissementen in de {sector} stijgt met {pct} procent",
    ],
    [
        "{team} wint met {score} van {opp} en blijft koploper",
        "{player} scoort twee keer bij zege van {team}",
        "Nederlandse schaatsers pakken goud op de {event}",
        "Trainer van {team} stapt op na {wins} nederlagen op rij",
    ],
    [
        "{tech_company} onthult nieuwe {device} met {feature}",
        "Nieuwe richtlijnen voor het gebruik van kunstmatige intelligentie gepresenteerd",
        "5G-netwerk in {region} nu volledig in gebruik",
        "Chipfabrikant begint met productie van {nm}-nanometerchips",
    ],
    [
        "RIVM: aantal gevallen van {disease} stijgt voor de {wins}e week op rij",
        "Nieuw medicijn tegen {disease} goedgekeurd",
        "Onderzoek: {food} verkleint de kans op {condition}",
        "Huisartsen waarschuwen voor drukte door {disease}",
    ],
    [
        "Succesvolle serie '{title}' krijgt een tweede seizoen",
        "{artist} kondigt tournee aan met {wins} concerten",
        "'{title}' wint prijs voor beste film op festival in {city}",
    ],
    [
        "Recordhitte in {region}: KNMI geeft code oranje af",
        "Kabinet wil CO2-uitstoot in {year} met {pct} procent verlagen",
        "Aantal {animal_plural} in {region} weer toegenomen",
    ],
    [
        "Politie arresteert verdachte van {crime} in {city}",
        "Oplichters bellen ouderen: schade van {amount} miljoen euro",
        "Reeks {crime_plural} in {city}: politie verhoogt toezicht",
    ],
]


def dutch_slots(rng):
    crime, crime_plural = pick(rng, [("inbraak", "inbraken"), ("autokraak", "autokraken"), ("fietsendiefstal", "fietsendiefstallen")])
    animal_plural = pick(rng, ["wolven", "zeehonden", "bevers"])
    return {"crime": crime, "crime_plural": crime_plural, "animal_plural": animal_plural}


DUTCH_SLOTS = dict(
    tech_company=["Nova Elektronica", "Delta Devices", "Hollandia Tech", "Noordzee Chips"],
    policy=["belastinghervorming", "huurbeleid", "pensioenen", "stikstofbeleid", "onderwijs"],
    topic=["handel", "veiligheid", "migratie", "energie"],
    leader=["de Duitse bondskanselier", "de Franse president", "de Belgische premier"],
    city=["Utrecht", "Rotterdam", "Groningen", "Eindhoven", "Zwolle"],
    party=["regeringspartij", "oppositiepartij"], approval=number_between(28, 52),
    pct_small=lambda rng: f"{rng.randint(1, 30) / 10:.1f}".replace(".", ","),
    cause=["sterke bedrijfscijfers", "een renteverlaging", "goede banencijfers"], company=["Delta Energie", "Hollandia Logistiek", "Vlaamse Staal", "Noordzee Techniek"],
    month=["januari", "februari", "maart", "april", "mei", "juni"], sector=["bouw", "horeca", "detailhandel"], pct=number_between(3, 25),
    team=["FC Noordwijk", "Rotterdam Rangers", "Utrecht Stars", "Groningen Wolves"], opp=["Zwolle Eagles", "Eindhoven Lions", "Haarlem FC"],
    score=["3-1", "2-0", "4-2", "1-0"], player=["De Vries", "Bakker", "Jansen", "Visser", "Smit"],
    event=["WK afstanden", "Europese kampioenschappen", "wereldbeker"], wins=number_between(3, 9),
    device=["smartphone", "laptop", "smartwatch"], feature=["AI-functies", "snelladen", "een opvouwbaar scherm"],
    region=["Zeeland", "Limburg", "Friesland", "de Randstad"], nm=["2", "3", "5"],
    disease=["griep", "RS-virus", "norovirus"], food=["noten", "vis", "volkorenbrood"], condition=["hartziekten", "diabetes", "dementie"],
    title=["De Laatste Dijk", "Zomer aan Zee", "Het Verborgen Huis"], artist=["Lotte Marlies", "De Nachtreizigers", "Sanne Vos"],
    year=["2030", "2035", "2040"], amount=number_between(2, 30),
)


def japanese_maker():
    inner = text_maker(JAPANESE, JAPANESE_SLOTS, "headline", joiner="")

    def make(rng, want):
        record, answer = inner(rng, want)
        record.update({"source": pick(rng, ["東都日報", "みらいニュース", "日本経済通信", "週刊ヘッドライン"]), "published": date_2026(rng)})
        return record, answer

    return make


def dutch_maker():
    def make(rng, want):
        slots = {**DUTCH_SLOTS, **{k: [v] for k, v in dutch_slots(rng).items()}}
        record, answer = text_maker(DUTCH, slots, "headline")(rng, want)
        record.update({"source": pick(rng, ["De Noordkrant", "NieuwsNu", "Hollands Dagblad", "Landelijk Journaal"]), "published": date_2026(rng)})
        return record, answer

    return make


SPECS = [
    spec(
        "news_topic_ja", "ja", "CHOICE", "このニュース見出しの主なトピックは何ですか？",
        ["政治", "経済", "スポーツ", "テクノロジー", "健康", "エンタメ", "環境", "事件・事故"],
        WEIGHTS, "article_id", "JA-", japanese_maker(),
    ),
    spec(
        "news_topic_nl", "nl", "CHOICE", "Wat is het hoofdonderwerp van deze nieuwskop?",
        ["politiek", "economie", "sport", "technologie", "gezondheid", "entertainment", "milieu", "misdaad"],
        WEIGHTS, "article_id", "NL-", dutch_maker(),
    ),
]
