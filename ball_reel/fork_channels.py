"""Поток D форка: ДВА РАЗНЫХ КАНАЛА условий — лицо отдельно, тело отдельно.

ЗАЧЕМ ЭТОТ МОДУЛЬ СУЩЕСТВУЕТ. В вендорском графе Wan-Animate мимика и поза
приходят в модель РАЗНЫМИ входами: в канале позы лица нет, в канале лица нет
тела. Из этого следует продуктовое обещание — губы и мимика берутся из
драйвинга независимо от движения, — и следствие это не архитектурная догадка, а
свойство весов: в модели лежат отдельные `face_adapter` (1.5628 ГБ),
`face_encoder`, `motion_encoder` и собственное `pose_patch_embedding`.

Значит и производить условия надо ДВУМЯ отрисовками, а не одной. Смешать их —
значит отдать модели картинку, для которой у неё нет входа.

---

ПЕРВОЕ ДЕЙСТВИЕ ПОТОКА БЫЛО ВОПРОСОМ, И ОТВЕТ СНЯТ ЗАМЕРОМ, А НЕ ПАМЯТЬЮ.

Спрашивалось: декодирует ли наш `ball_reel/dwpose.py` 68 точек лица, или из 133
берутся только тело и кисти. Прогон `dw-ll_ucoco_384.onnx` на `demo/hero.png`
(команда и вывод — `docs/FORK_NUMBERS.md`):

    simcc_x (1, 133, 576)   simcc_y (1, 133, 768)
    суставов на выходе модели: 133
    декодирует dwpose.py:       17

То есть модель отдаёт 133 сустава, а `dwpose._decode_simcc` обходит
`WHOLEBODY_INDEX` — 17 имён тела — и остальные 116 выходов выбрасывает.

    точек лица, доезжающих до кода сейчас:  0
    точек кистей, доезжающих до кода сейчас: 0

ГЛАВНОЕ СЛЕДСТВИЕ, ради которого вопрос и задавался: **канал лица существует и
новых весов не требует.** Раскладка COCO-WholeBody — 17 тело, 6 стопы, 68 лицо,
21 левая кисть, 21 правая = 133; лицо лежит на индексах 23..90. Нужно
декодировать выход, который уже считается и уже оплачен, а не искать другую
модель. Это разворачивает поток из «есть ли у нас канал лица» в «дописать
декодер», то есть из открытого риска в работу.

---

ЧЕГО ЗДЕСЬ НЕТ И ПОЧЕМУ.

`ball_reel/dwpose.py` НЕ ПРАВИТСЯ — один писатель на модуль (Ц2), и он чужой:
на нём стоит сданный пайплайн. Раскладка и декодер живут здесь, а из `dwpose`
берутся веса, нормализация и геометрия бокса — импортом, не копией (Е1).

Отрисовка НЕ КОПИРУЕТ палитру OpenPose из `skeleton.py`: там она — контракт с
ControlNet, обученным на openpose-аннотаторе, а здесь другой потребитель и
другой контракт. Смешивать два контракта в одной константе значит получить
дефект, который проявится ровно один раз и не там, где его будут искать.
"""

from __future__ import annotations

from pathlib import Path

from . import dwpose

#: Раскладка COCO-WholeBody: имя группы -> (первый индекс, последний+1).
#: ВЫВЕДЕНО арифметикой по замеренному выходу модели (133), а не переписано из
#: статьи: 17 + 6 + 68 + 21 + 21 = 133, и сумма сходится ровно. Тест сторожит
#: и сумму, и непересечение групп — раскладка, где две группы делят индекс,
#: молча смешала бы лицо с кистью.
WHOLEBODY_GROUPS = {
    "body": (0, 17),
    "feet": (17, 23),
    "face": (23, 91),
    "l_hand": (91, 112),
    "r_hand": (112, 133),
}

#: Сколько суставов отдаёт модель. ИЗМЕРЕНО (прогон dw-ll_ucoco_384.onnx на
#: demo/hero.png, вывод в docs/FORK_NUMBERS.md), а не взято из документации:
#: по документации у нас уже было три тихих расхождения подряд (см. докстринг
#: `dwpose`), и все три не падали, а выдавали правдоподобный мусор.
WHOLEBODY_JOINTS = 133

#: Сколько точек в лицевой части. Отдельной константой, потому что именно это
#: число называют, когда спрашивают «есть ли канал лица».
FACE_POINTS = 68

#: Имена каналов. Два, и третьего не предполагается: у модели два входа.
FACE_CHANNEL, BODY_CHANNEL = "face", "body"
CHANNELS = (FACE_CHANNEL, BODY_CHANNEL)

#: Что попадает в КАЖДЫЙ канал. Тело идёт С КИСТЯМИ и стопами, но БЕЗ лица;
#: лицо идёт одно. Пересечения быть не должно, и это проверяется тестом, а не
#: соблюдением при чтении.
#:
#: «БЕЗ ЛИЦА» ЗНАЧИТ «без 68 лицевых точек», а не «без головы», и уточнение
#: это стоило одного взгляда на артефакт (П3): в канале тела рисуется голова —
#: нос, глаза, уши, индексы 0..4. Они принадлежат ТЕЛУ в раскладке COCO, и
#: openpose-скелет вендора их несёт; выкинуть их значило бы отдать модели
#: обезглавленный скелет и получить голову, поставленную наугад. Мимики в них
#: нет — пять точек не кодируют выражение, — поэтому канал лица они не
#: дублируют и обещание «мимика приходит отдельным каналом» не нарушают.
CHANNEL_GROUPS = {
    FACE_CHANNEL: ("face",),
    BODY_CHANNEL: ("body", "feet", "l_hand", "r_hand"),
}

#: Ниже этого балла точка не рисуется. НЕ СВОЙ порог: берётся из `dwpose`,
#: потому что балл — его шкала, и второе значение для одной величины разъедется
#: при первой правке (Е1).
MIN_SCORE = dwpose.MIN_SCORE

#: Радиус точки при отрисовке, px. ВЫБРАНО: 2 px на кадре порядка 512-1024 —
#: точка видна и не сливается с соседней. Калибровать не на чем, пока нет
#: генерации; когда появится, число двигать здесь, а не по месту.
DOT_RADIUS = 2

#: Толщина линии скелета, px. ВЫБРАНО по той же причине.
LIMB_WIDTH = 3

#: Цвет условий. ВЫБРАНО белым на чёрном: у канала лица нет вендорской палитры,
#: которую можно было бы соблюсти, а выдумать цветовое кодирование и подать его
#: как контракт — это ровно тот сорт правдоподобной выдумки, который потом
#: невозможно отличить от измеренного.
INK = (255, 255, 255)
GROUND = (0, 0, 0)

#: Рёбра тела, по индексам COCO-WholeBody (первые 17 — тело в порядке COCO).
#: Своё, а не `skeleton.LIMBS18`: там нумерация OpenPose COCO-18 с
#: синтезированной шеей на позиции 1, здесь — сырая нумерация модели. Две
#: разные нумерации под одним именем — тот самый второй способ узнать
#: известное, который Е1 называет дефектом.
BODY_EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10),        # руки
    (5, 6), (5, 11), (6, 12), (11, 12),     # плечи и таз
    (11, 13), (13, 15), (12, 14), (14, 16),  # ноги
    (0, 1), (0, 2), (1, 3), (2, 4),          # голова: нос, глаза, уши
)

#: Рёбра кисти: пять пальцев от запястья, по 4 фаланги. Индексы ЛОКАЛЬНЫЕ
#: внутри группы кисти (0 — запястье), прибавляется начало группы.
HAND_EDGES = tuple(
    (0, base) if step == 0 else (base + step - 1, base + step)
    for base in (1, 5, 9, 13, 17) for step in range(4)
)


def group_indices(group: str) -> range:
    """Индексы одной группы COCO-WholeBody.

    Отдельной функцией, а не срезом по месту: срез, повторённый в пяти местах,
    разъедется в пяти местах — и разъезд будет тихим, потому что точки всё
    равно нарисуются, просто не те.
    """
    if group not in WHOLEBODY_GROUPS:
        raise ValueError(
            f"нет такой группы COCO-WholeBody: {group!r}. "
            f"Есть: {', '.join(WHOLEBODY_GROUPS)}")
    lo, hi = WHOLEBODY_GROUPS[group]
    return range(lo, hi)


def channel_indices(channel: str) -> tuple:
    """Индексы, попадающие в канал. Отсортированы — порядок отрисовки стабилен."""
    if channel not in CHANNEL_GROUPS:
        raise ValueError(
            f"нет такого канала: {channel!r}. Есть: {', '.join(CHANNELS)}")
    out: list[int] = []
    for group in CHANNEL_GROUPS[channel]:
        out.extend(group_indices(group))
    return tuple(sorted(out))


def decode_wholebody(x_logits, y_logits, box, input_size=dwpose.POSE_INPUT) -> list:
    """Полный выход SimCC -> 133 точки `(x, y, балл)` в координатах кадра.

    Это `dwpose._decode_simcc`, у которого снят потолок в 17 имён: та же
    арифметика SimCC (argmax по бину, делённый на коэффициент разбиения), но
    по ВСЕМ суставам, что отдала модель, и с выходом списком, а не словарём
    имён — имён у 68 точек лица нет и придумывать их незачем.

    Число суставов НЕ хардкодится в цикле: берётся из формы выхода. Модель,
    отдавшая другое число, обязана быть замечена вызывающим, а не молча
    обрезана до нашего представления о ней.
    """
    import numpy as np

    x_logits = np.asarray(x_logits)
    y_logits = np.asarray(y_logits)
    if x_logits.ndim != 2 or y_logits.ndim != 2:
        raise ValueError(
            f"ожидались логиты SimCC формы (суставы, бины), пришло "
            f"{x_logits.shape} и {y_logits.shape}")
    if x_logits.shape[0] != y_logits.shape[0]:
        raise ValueError(
            f"по x и по y разное число суставов: {x_logits.shape[0]} и "
            f"{y_logits.shape[0]} — это не одна и та же модель")

    x_locs = np.argmax(x_logits, axis=-1)
    y_locs = np.argmax(y_logits, axis=-1)
    x_conf = dwpose._peak(x_logits)
    y_conf = dwpose._peak(y_logits)
    split_x = x_logits.shape[-1] / input_size[0]
    split_y = y_logits.shape[-1] / input_size[1]

    x0, y0, x1, y1 = box
    sx = (x1 - x0) / input_size[0]
    sy = (y1 - y0) / input_size[1]

    return [(float(x0 + (x_locs[i] / split_x) * sx),
             float(y0 + (y_locs[i] / split_y) * sy),
             float(min(x_conf[i], y_conf[i])))
            for i in range(x_logits.shape[0])]


def wholebody_points(path: str | Path) -> list | None:
    """133 точки одного кадра. Требует весов DWPose; None — человека не нашли.

    НЕ ПОКРЫТО ОФЛАЙН-ТЕСТОМ ЦЕЛИКОМ: это обёртка над двумя ONNX-сессиями.
    Проверяемая часть — `decode_wholebody` — вынесена наружу намеренно (Т5),
    потому что развилка внутри функции, требующей 350 МБ весов, недостижима
    для теста и деградирует молча.
    """
    import numpy as np
    from PIL import Image

    det_path, pose_path = dwpose._model_paths()
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    det = dwpose._session(det_path)
    canvas, scale = dwpose._letterbox(rgb, (640, 640))
    blob = canvas.transpose(2, 0, 1)[None].astype("float32")
    raw = det.run(None, {det.get_inputs()[0].name: blob})[0]
    boxes, scores = dwpose.decode_yolox(raw, scale)
    person = dwpose._largest_person(boxes, scores, scale)
    if person is None:
        return None

    person = dwpose.expand_box(person)
    x0, y0, x1, y1 = person
    crop = Image.fromarray(rgb).crop((int(x0), int(y0), int(x1), int(y1)))
    arr = np.asarray(crop.resize(dwpose.POSE_INPUT, Image.BILINEAR),
                     dtype="float32")
    arr = (arr - np.array(dwpose.PIXEL_MEAN)) / np.array(dwpose.PIXEL_STD)
    inp = arr.transpose(2, 0, 1)[None].astype("float32")
    x_logits, y_logits = dwpose._session(pose_path).run(
        None, {dwpose._session(pose_path).get_inputs()[0].name: inp})
    return decode_wholebody(x_logits[0], y_logits[0], person)


def visible(points: list, channel: str, *, min_score: float = MIN_SCORE) -> int:
    """Сколько точек канала наблюдаемо. ЧИСЛО, а не флаг (Е3).

    Флаг «канал снят» на месте этого числа читался бы как полная работа при
    одной уцелевшей точке из шестидесяти восьми.
    """
    n = len(points)
    return sum(1 for i in channel_indices(channel)
               if i < n and points[i][2] >= min_score)


def render(points: list, channel: str, width: int, height: int,
           *, min_score: float = MIN_SCORE):
    """Одна режимная отрисовка канала. Возвращает изображение PIL.

    Канал `face` рисует ТОЛЬКО точки лица, канал `body` — тело, стопы и кисти
    и НИ ОДНОЙ точки лица. Это не оформительское решение: у модели два входа, и
    картинка, где смешано, не подходит ни к одному.

    Точки ниже порога не рисуются вовсе. Не «рисуются бледнее» — условие либо
    поставлено, либо нет, а полутон в условии модель прочтёт как факт.
    """
    from PIL import Image, ImageDraw

    if channel not in CHANNEL_GROUPS:
        raise ValueError(
            f"нет такого канала: {channel!r}. Есть: {', '.join(CHANNELS)}")
    if width <= 0 or height <= 0:
        raise ValueError(f"негодный размер холста: {width}x{height}")

    img = Image.new("RGB", (width, height), GROUND)
    draw = ImageDraw.Draw(img)
    n = len(points)

    def ok(i: int) -> bool:
        return i < n and points[i][2] >= min_score

    if channel == BODY_CHANNEL:
        for a, b in BODY_EDGES:
            if ok(a) and ok(b):
                draw.line([points[a][:2], points[b][:2]],
                          fill=INK, width=LIMB_WIDTH)
        for group in ("l_hand", "r_hand"):
            base = WHOLEBODY_GROUPS[group][0]
            for a, b in HAND_EDGES:
                if ok(base + a) and ok(base + b):
                    draw.line([points[base + a][:2], points[base + b][:2]],
                              fill=INK, width=1)

    for i in channel_indices(channel):
        if not ok(i):
            continue
        x, y = points[i][0], points[i][1]
        draw.ellipse([x - DOT_RADIUS, y - DOT_RADIUS,
                      x + DOT_RADIUS, y + DOT_RADIUS], fill=INK)
    return img


def render_pair(points: list, width: int, height: int,
                *, min_score: float = MIN_SCORE) -> dict:
    """Обе отрисовки разом: `{"face": img, "body": img}`.

    Парой, а не двумя вызовами по месту: два канала одного кадра обязаны иметь
    один холст и один порог, а розданные по вызывающим они разъедутся.
    """
    return {c: render(points, c, width, height, min_score=min_score)
            for c in CHANNELS}


def render_sequence(frame_paths, out_dir: str | Path,
                    *, min_score: float = MIN_SCORE) -> dict:
    """Обе последовательности условий по кадрам драйвинга.

    Возвращает три исхода на кадр, не два (Р1): кадр либо снят, либо человека
    в нём не нашли, либо лицо в нём нечитаемо — и последнее НЕ ТО ЖЕ САМОЕ, что
    «лица нет». Итог печатается числами (`снято N из M`), а не флагом (Е3, Р2).
    """
    from PIL import Image

    out = Path(out_dir)
    (out / FACE_CHANNEL).mkdir(parents=True, exist_ok=True)
    (out / BODY_CHANNEL).mkdir(parents=True, exist_ok=True)

    frames = [Path(p) for p in frame_paths]
    done, no_person, face_unreadable = [], [], []
    for i, p in enumerate(frames):
        pts = wholebody_points(p)
        if pts is None:
            no_person.append(p.name)
            continue
        with Image.open(p) as im:
            w, h = im.size
        if visible(pts, FACE_CHANNEL, min_score=min_score) == 0:
            face_unreadable.append(p.name)
        for channel, img in render_pair(pts, w, h, min_score=min_score).items():
            img.save(out / channel / f"{i:05d}.png")
        done.append(p.name)

    total = len(frames)
    return {
        "total": total,
        "rendered": len(done),
        "no_person": no_person,
        "face_unreadable": face_unreadable,
        "dir": {c: str(out / c) for c in CHANNELS},
        "note": (f"условия сняты на {len(done)} из {total} кадров; "
                 f"{len(no_person)} без человека, "
                 f"{len(face_unreadable)} с нечитаемым лицом "
                 f"(это НЕ «лица нет» — это «судить нечем»)"),
    }
