"""Customer message intent, Spanish and Korean; skewed toward shipping and refund questions."""

from domains.common import CHANNELS, number_between, pick, spec, text_maker

WEIGHTS = [30, 18, 15, 10, 9, 7, 7, 4]


def channel(rng):
    return {"channel": pick(rng, CHANNELS)}


SPANISH = [
    [
        "¿Dónde está mi pedido {order}? Lleva {days} días sin llegar.",
        "¿Me pueden decir cuándo llegará el paquete {order}?",
        "El seguimiento de mi pedido {order} no se actualiza desde hace {days} días.",
        "Todavía no he recibido el {product}. ¿Hay algún retraso con el envío?",
        "Quisiera saber el estado del envío del pedido {order}.",
    ],
    [
        "Quiero que me devuelvan el dinero de mi pedido {order}.",
        "El {product} llegó defectuoso y solicito un reembolso.",
        "Hice una devolución hace {days} días y todavía no veo el reembolso.",
        "¿Cómo puedo pedir la devolución del importe del pedido {order}?",
        "Necesito que me reembolsen el cobro del {product}.",
    ],
    [
        "La aplicación se cierra sola cada vez que la abro.",
        "No puedo iniciar sesión: dice que la contraseña es incorrecta, pero es la correcta.",
        "El {product} no se conecta al wifi desde la última actualización.",
        "La página de pago da error y no puedo completar la compra.",
        "El código de verificación nunca llega a mi teléfono.",
    ],
    [
        "Necesito la factura del pedido {order} a nombre de mi empresa.",
        "La factura tiene un importe distinto al que pagué.",
        "¿Pueden enviarme de nuevo la factura de {month}?",
        "Falta mi número de identificación fiscal en la factura del pedido {order}.",
    ],
    [
        "Quiero dar de baja mi suscripción.",
        "Por favor, cancelen mi plan premium antes de la próxima renovación.",
        "¿Cómo cancelo la suscripción? No encuentro la opción en mi cuenta.",
        "No quiero que me sigan cobrando la suscripción mensual.",
    ],
    [
        "Me equivoqué de dirección en el pedido {order}. ¿Puedo cambiarla?",
        "Me he mudado y necesito actualizar mi dirección de envío.",
        "¿Se puede enviar el paquete a otra dirección? Estaré fuera esta semana.",
    ],
    [
        "¿El {product} viene con garantía?",
        "¿Tienen el {product} en otros colores?",
        "¿Cuáles son las medidas del {product}?",
        "¿El {product} es compatible con mi modelo anterior?",
    ],
    [
        "Estoy muy molesto con el servicio, es la tercera vez que me pasa.",
        "Nadie me responde y llevo {days} días esperando. Es inaceptable.",
        "El trato que recibí en la tienda fue pésimo y quiero presentar una queja formal.",
    ],
]
KOREAN = [
    [
        "주문번호 {order} 배송이 언제 도착하나요?",
        "{days}일째 택배가 오지 않았어요. 배송 상태를 확인해 주세요.",
        "{product} 주문 건이 아직 도착하지 않았습니다.",
        "배송 조회가 {days}일 동안 업데이트되지 않아요.",
        "배송이 지연되고 있는 건가요? 주문번호는 {order}입니다.",
    ],
    [
        "주문번호 {order} 환불해 주세요.",
        "{product} 주문 건이 불량이라 환불을 요청합니다.",
        "반품한 지 {days}일이 지났는데 아직 환불이 안 됐어요.",
        "결제 금액을 환불받고 싶습니다. 주문번호는 {order}입니다.",
    ],
    [
        "앱이 실행하자마자 자꾸 꺼집니다.",
        "비밀번호가 맞는데 로그인이 되지 않아요.",
        "{product}: 최근 업데이트 이후 연결이 계속 끊깁니다.",
        "결제 페이지에서 오류가 나서 구매를 완료할 수 없어요.",
        "인증번호 문자가 오지 않습니다.",
    ],
    [
        "주문번호 {order} 세금계산서를 발행해 주세요.",
        "영수증에 적힌 금액이 실제 결제 금액과 달라요.",
        "{month} 결제분 영수증을 다시 보내 주실 수 있나요?",
        "사업자등록번호가 빠진 채로 계산서가 발행되었습니다.",
    ],
    [
        "구독을 해지하고 싶습니다.",
        "다음 결제일 전에 프리미엄 요금제를 해지해 주세요.",
        "구독 해지 메뉴를 찾을 수 없어요. 어떻게 해지하나요?",
        "월 정기 결제가 더 이상 청구되지 않게 해 주세요.",
    ],
    [
        "주문번호 {order} 배송지를 잘못 입력했어요. 변경할 수 있을까요?",
        "이사를 해서 배송 주소를 바꾸고 싶습니다.",
        "이번 주에는 집에 없어서 다른 주소로 받고 싶어요.",
    ],
    [
        "{product} 보증 기간이 어떻게 되나요?",
        "{product} 다른 색상도 있나요?",
        "{product} 크기가 어떻게 되는지 알려 주세요.",
        "{product} 이전 모델과 호환되나요?",
    ],
    [
        "서비스가 너무 불만스럽습니다. 벌써 세 번째예요.",
        "{days}일째 답변이 없습니다. 정말 받아들일 수 없어요.",
        "매장 직원의 응대가 너무 불친절해서 정식으로 항의하고 싶습니다.",
    ],
]

SPECS = [
    spec(
        "support_intent_es", "es", "CHOICE", "¿Cuál es la intención principal del cliente en este mensaje?",
        ["estado del envío", "reembolso", "problema técnico", "factura", "cancelar suscripción",
         "cambio de dirección", "información del producto", "queja"],
        WEIGHTS, "ticket_id", "ES-",
        text_maker(
            SPANISH,
            dict(
                order=number_between(100000, 999999), days=number_between(2, 14),
                product=["altavoz", "teléfono", "portátil", "ventilador", "reloj inteligente", "aspirador"],
                month=["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto"],
            ),
            "message", greetings=("", "Hola.", "Buenos días.", "Buenas tardes."),
            closings=("", "Gracias.", "Saludos.", "Quedo atento."), extra=channel,
        ),
    ),
    spec(
        "support_intent_ko", "ko", "CHOICE", "이 고객 메시지의 주된 문의 의도는 무엇인가요?",
        ["배송 상태", "환불", "기술 문제", "영수증/계산서", "구독 해지", "배송지 변경", "상품 문의", "불만 접수"],
        WEIGHTS, "ticket_id", "KO-",
        text_maker(
            KOREAN,
            dict(
                order=number_between(100000, 999999), days=number_between(2, 14),
                product=["무선 이어폰", "공기청정기", "전기포트", "노트북 가방", "스마트워치", "블렌더"],
                month=["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월"],
            ),
            "message", greetings=("", "안녕하세요.", "문의드립니다."),
            closings=("", "감사합니다.", "빠른 답변 부탁드립니다."), extra=channel,
        ),
    ),
]
