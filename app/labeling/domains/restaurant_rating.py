"""Restaurant review rating on a 0 to 4 scale, Japanese and Spanish; ratings skew positive with a small tail of very bad."""

from domains.common import pick, spec

WEIGHTS = [12, 8, 15, 27, 38]
OPTIONS = ["0", "1", "2", "3", "4"]

JAPANESE = [
    ["最悪でした。{dish}に髪の毛が入っていて、対応も不誠実でした。", "二度と行きません。注文が来るまで{n}分も待たされました。", "{dish}が生ぬるく、店員の態度も最悪でした。"],
    ["{dish}はいまいちでした。期待していたほどではなかったです。", "接客が雑で、料理も冷めていました。", "値段の割に量が少なく、正直がっかりしました。"],
    ["普通でした。{dish}は悪くないけれど特別ではありません。", "可もなく不可もなく、という感じです。", "味は普通で、待ち時間が少し長かったです。"],
    ["{dish}が美味しかったです。少し混んでいましたが、また行きたいです。", "味は良かったです。値段も手頃でおすすめです。", "店員さんが親切で、{dish}も満足でした。"],
    ["最高でした！{dish}が絶品で、店員さんの対応も素晴らしかったです。", "また絶対に来たいです。{dish}は必食です！", "雰囲気もよく、{dish}がとても美味しかったです。大満足です。"],
]
SPANISH = [
    ["Pésima experiencia. Encontramos un pelo en el {dish} y nadie se disculpó.", "No volveremos jamás, esperamos {n} minutos por la comida.", "El {dish} llegó frío y el camarero fue muy maleducado."],
    ["El {dish} estaba flojo y el servicio fue lento.", "Decepcionante, la comida llegó fría.", "Las raciones son pequeñas para lo que cobran, esperaba mucho más."],
    ["Normal. El {dish} no estuvo mal, pero nada especial.", "Ni fu ni fa, esperaba algo más.", "La comida correcta, aunque tardaron bastante en atendernos."],
    ["Muy buena comida, aunque había bastante gente. Repetiremos.", "El {dish} estaba rico y el precio es razonable.", "Personal amable y buen {dish}, lo recomiendo."],
    ["¡Excelente! El {dish} estaba espectacular y el servicio fue impecable.", "Volveremos seguro, el {dish} es imprescindible.", "Un sitio con encanto y un {dish} buenísimo. Muy recomendable."],
]
SUPPLEMENTS = {
    "ja": [
        ["店内も不潔で、衛生面に不安を感じました。", "謝罪の一言もなく、本当に残念です。", "値段に見合わないひどい味でした。", "予約したのに席に案内されるまで長く待たされました。"],
        ["味付けが濃すぎて、最後まで食べきれませんでした。", "店内が騒がしく、落ち着いて食事できませんでした。", "もう少し工夫があれば良いと思います。", "ドリンクの提供も遅かったです。"],
        ["雰囲気はまあまあです。", "駅から近いのは便利です。", "次は別のメニューを試してみようと思います。", "ランチの時間帯は少し混んでいました。"],
        ["デザートも美味しかったです。", "駅から近くて便利でした。", "次回は友人を誘って来たいです。", "メニューが豊富で選ぶのが楽しかったです。"],
        ["家族みんなが大満足でした。", "予約して行くことをおすすめします。", "デザートまで完璧でした。", "接客も丁寧で、気持ちよく過ごせました。"],
    ],
    "es": [
        ["El local estaba sucio y la higiene deja mucho que desear.", "Ni una disculpa, una lástima.", "Precio absurdo para una comida tan mala.", "Teníamos reserva y aun así esperamos muchísimo."],
        ["Estaba demasiado salado y no pude terminarlo.", "Mucho ruido, no se puede hablar con tranquilidad.", "Con un poco más de cuidado sería mejor.", "Las bebidas tardaron una eternidad."],
        ["El ambiente está bien.", "Está bien situado, cerca del centro.", "Quizá probemos otro plato la próxima vez.", "A la hora de comer había bastante gente."],
        ["El postre también estaba muy bueno.", "Está bien ubicado y es fácil aparcar.", "Traeremos a unos amigos la próxima vez.", "La carta es amplia y cuesta elegir."],
        ["Toda la familia quedó encantada.", "Mejor reservar con antelación.", "Hasta el postre fue perfecto.", "Trato cercano y muy profesional."],
    ],
}
NAMES = {
    "ja": ["麺屋 青葉", "寿司 ほたる", "居酒屋 ふくろう", "カレー工房 まる", "天ぷら 一心"],
    "es": ["Casa Marisol", "El Rincón de Pepe", "Bodega La Ola", "Taberna del Puerto", "Mesón Los Olivos"],
}
PRICES = {"ja": ["¥", "¥¥", "¥¥¥"], "es": ["€", "€€", "€€€"]}
DISHES = {
    "ja": ["ラーメン", "餃子", "天ぷら", "寿司", "カレー", "焼き鳥", "うどん"],
    "es": ["cocido", "arroz", "pulpo", "jamón", "bacalao", "chuletón"],
}


def maker(language: str, templates: list, joiner: str):
    def make(rng, want):
        n = rng.randint(30, 90)
        fill = dict(dish=pick(rng, DISHES[language]), n=n)
        sentences = [pick(rng, templates[want]).format(**fill)]
        if rng.random() < 0.6:
            sentences.append(pick(rng, SUPPLEMENTS[language][want]).format(**fill))
        record = {
            "restaurant": pick(rng, NAMES[language]), "price_level": pick(rng, PRICES[language]),
            "review": joiner.join(sentences),
        }
        return record, want

    return make


SPECS = [
    spec(
        "restaurant_rating_ja", "ja", "SCORE", "このレストランのレビューを評価してください: 0 = 非常に悪い、2 = 普通、4 = 非常に良い。",
        OPTIONS, WEIGHTS, "review_id", "JA-", maker("ja", JAPANESE, ""),
    ),
    spec(
        "restaurant_rating_es", "es", "SCORE", "Valora esta reseña de restaurante: 0 = muy mala, 2 = normal, 4 = excelente.",
        OPTIONS, WEIGHTS, "review_id", "ES-", maker("es", SPANISH, " "),
    ),
]
