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

---

ЧТО ДЕЛАЕТ С НАШИМИ КАНАЛАМИ САМА НОДА. Прочитано в `comfy_extras/nodes_wan.py`,
класс `WanAnimateToVideo`, а не припомнено. Номера строк — по копии файла,
скачанной для этого спринта (sha256 dcd8b81d1225d84e…8962002); сам файл в наш
репозиторий не входит, поэтому ключевое утверждение проверено ВТОРЫМ
источником, который в репозитории есть:
`workflows/upstream/WanAnimateToVideo.doc.md`, «Parameter Constraints» —
«`face_video` is automatically resized to 512x512 resolution».

    face_video  →  common_upscale(..., 512, 512, "area", "center") * 2 - 1
                   →  conditioning["face_video_pixels"]        (nodes_wan.py:1208)
    pose_video  →  vae.encode(...) → conditioning["pose_video_latent"] (:1192)
    character_mask → common_upscale(..., "nearest-exact") до латентной сетки,
                   а она width//8 × height//8              (:1236, :1152–1153)
                   → mask_refmotion.view(1, T//4, 4, H, W).transpose(1,2) (:1243)

Три следствия, и они разные для двух каналов:

1. **Лицевой канал в пикселях, без VAE.** `face_video_pixels` уходит в модель
   покадрово и в полном разрешении 512×512 — никакого латентного ужатия на нём
   нет. Поэтому мелкая геометрия губ здесь доезжает, и поэтому же 512 не
   декоративное число: нода приводит вход к 512 САМА, и приводит его
   `crop="center"`, то есть сначала режет по центру до квадрата. Кадр 480×848,
   поданный целиком, будет обрезан до центральных 480×480 — а голова при
   кадрировке во весь рост лежит выше этой полосы, то есть лицевое условие
   уедет из картинки целиком и молча. Отсюда `face_box` + `render_face` ниже:
   квадрат вокруг лица мы задаём сами, и «center» ноды попадает ровно в него.
2. **Канал позы идёт через VAE:** 8 px по стороне и 4 кадра по времени в один
   латентный кадр. Точность позы там и теряется — но это цена самого канала, а
   не наша, и подать позу иначе нода не даёт.
3. **Квантование маски на наши каналы НЕ ВЛИЯЕТ, и это проверено, а не
   предположено.** Маска ужимается пространственно до `width//8 × height//8`
   (`nodes_wan.py:1236` в сетку из `:1152`), а с патчем `(1,2,2)` — размеры весов, см.
   `docs/FORK_HARDWARE.md` — один токен трансформера кроет 8·2 = **16 px**
   кадра. По времени `view(1, T//4, 4, H, W).transpose(1,2)` — это НЕ усреднение
   четвёрки, а раскладка четырёх кадров по каналам одного латентного кадра:
   покадровая маска сохраняется, просто лежит в четырёх каналах. Проверено
   чтением строки, потому что «квантуется по времени группами по 4» звучит как
   потеря, а в коде стоит перестановка.
   Ни `face_video_pixels`, ни `pose_video_latent` через эту ветку не проходят —
   у них свои ключи обусловливания. Значит грубость маски ограничивает, ГДЕ
   модель перерисовывает, и не ограничивает, ЧТО мы говорим о мимике: условие
   лица покадровое и пиксельное. Разница масштабов при этом настоящая и её надо
   держать в голове — 16 px токена против 512-го лицевого кадра, где вся мимика
   умещается в габарит около 340 px.
"""

from __future__ import annotations

from pathlib import Path

from . import dwpose
from .fork_identity import FAIL, PASS, UNMEASURED

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
#:
#: НИ ОДНА функция ниже не ставит эту константу умолчанием прямо в сигнатуре —
#: только `= None` и подстановка в теле. Причина найдена мутацией (Т1) и стоила
#: одного ложно-зелёного теста: умолчание в сигнатуре вычисляется один раз при
#: импорте, поэтому подмена константы на модуле до него не доходит, и сторож
#: константы проходит, ничего не сторожа. По И7 форма выгреблена целиком, а не
#: в месте находки: 7 сигнатур (`min_score` — 5, `length` в
#: `check_face_geometry`, `input_size` в `decode_wholebody`) плюс `side` и
#: `margin` лицевого канала.
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


#: Требования к размерности, снятые с контракта ноды (ХЭНДОФ §3.2), а не
#: выдуманные. Нарушение каждого — ТИХОЕ: Comfy либо откажет уже на карте,
#: после того как веса загружены, либо молча подгонит и сместит всё условие.
#: Дешевле проверить здесь, на CPU, за микросекунды (П2).
#: ИСТОЧНИК — `workflows/upstream/WanAnimateToVideo.doc.md`, строки 18–20, а не
#: пересказ. Дословно: `width` «default: 832, step: 16», `height` «default: 480,
#: step: 16», `length` «default: 77, step: 4», диапазон длины «1 to
#: MAX_RESOLUTION».
SIDE_MULTIPLE = 16          # ширина и высота кратны 16

#: РАСХОЖДЕНИЕ, НАЙДЕННОЕ ПРОГОНОМ И РАЗРЕШЁННОЕ В ПОЛЬЗУ ФАЙЛА. Хэндоф §3.2
#: говорит «длина кратна 4», но §3 и §3.1 того же хэндофа задают **77 кадров**,
#: а 77 на 4 не делится. Оба утверждения одновременно верными быть не могут.
#:
#: Первоисточник разрешает спор: у `length` не «кратность 4», а **шаг 4 от
#: минимума 1** — то есть допустимы 1, 5, 9, …, 77, 81. Вендорский темплейт
#: ставит ровно 77 (`widgets_values` узла `WanAnimateToVideo`), и это его
#: собственное умолчание.
#:
#: Первая редакция этой проверки читала «кратна 4» буквально и **отвергала 77**
#: — то есть забраковала бы штатную геометрию стека. Поймано тем, что число из
#: §3 не прошло проверку, написанную по §3.2.
LENGTH_STEP = 4             # шаг 4
LENGTH_MIN = 1              # от 1: годны длины вида 1 + 4k

#: Сторона лицевого канала. НЕ НАШ ВЫБОР и не пожелание: нода приводит
#: `face_video` к 512×512 сама — `common_upscale(..., 512, 512, "area",
#: "center")`, `nodes_wan.py:1208`, то же сказано в документации ноды в
#: репозитории. Отдавать другое разрешение бессмысленно: его всё равно
#: перемасштабируют, а перед этим ОБРЕЖУТ по центру до квадрата.
#:
#: ~~Первая редакция объявляла эту константу и нигде не использовала~~ — снято
#: тем, что `render_face` теперь единственный способ получить лицевой канал, а
#: тест сторожит и сторону, и её кратность 16.
FACE_SIDE = 512

#: Запас вокруг габарита 68 точек, долей от большей стороны габарита, с каждой
#: стороны. ВЫБРАНО 0.25 по замеру на `demo/hero.png` (прогон
#: `wholebody_points`, 68 из 68 точек наблюдаемы, габарит лица 130.1×137.4 px,
#: минимальное расстояние между соседними точками лица 2.45 px кадра):
#:
#:     запас   сторона кропа   масштаб   соседние точки в кадре 512
#:      0.10       164.9 px     ×3.10           7.62 px
#:      0.25       206.2 px     ×2.48           6.10 px
#:      0.35       233.7 px     ×2.19           5.38 px
#:      0.50       274.9 px     ×1.86           4.57 px
#:     без кропа (весь кадр в 512)                1.64 px
#:
#: Нижняя граница выбора — диаметр точки: при `DOT_RADIUS = 2` он 5 px, и запас
#: 0.50 уже кладёт соседей ближе диаметра, то есть контур губ слипается в
#: сплошной штрих и мимика в нём перестаёт читаться. Верхняя граница — то, ради
#: чего запас вообще нужен: 68 точек не содержат лба и волос, а при общем боксе
#: на всю последовательность (см. `render_sequence`) в него должен помещаться
#: поворот головы между кадрами. 0.25 держит соседей на 6.10 px — больше
#: диаметра — и оставляет четверть габарита кругом.
#:
#: Строка «без кропа» — то, что было до этой правки: лицо на холсте кадра.
#: 1.64 px между точками означает, что все 68 сливаются в пятно, а при
#: продуктовой кадрировке во весь рост (лицо 63–80 px, ХЭНДОФ §4) от лицевого
#: условия не осталось бы ничего.
FACE_MARGIN = 0.25

#: Сколько точек лица должно быть наблюдаемо, чтобы бокс вообще строился.
#: ВЫБРАНО 17 — четверть от 68. Ниже этого габарит собран по случайным
#: уцелевшим точкам, и кроп по нему промахнётся мимо лица; промах хуже отказа,
#: потому что чёрный кадр модель читает как «условия нет», а кроп мимо лица —
#: как «лицо вот такое». Отказ здесь третий исход (Р1), а не провал.
FACE_BOX_MIN_POINTS = 17


def check_geometry(width: int, height: int, length: int) -> dict:
    """Годится ли геометрия для `WanAnimateToVideo`. Числа в отчёт, не флаг.

    Отдельной функцией, а не проверкой внутри отрисовки (Т5): развилка,
    спрятанная в вызове, который требует весов и сотни кадров, недостижима для
    теста и деградирует молча.
    """
    problems = []
    for name, value in (("ширина", width), ("высота", height)):
        if value <= 0:
            problems.append(f"{name} {value} — не размер")
        elif value % SIDE_MULTIPLE:
            problems.append(
                f"{name} {value} не кратна {SIDE_MULTIPLE} "
                f"(ближайшие {value // SIDE_MULTIPLE * SIDE_MULTIPLE} и "
                f"{(value // SIDE_MULTIPLE + 1) * SIDE_MULTIPLE})")
    if length < LENGTH_MIN:
        problems.append(f"длина {length} меньше минимума {LENGTH_MIN}")
    elif (length - LENGTH_MIN) % LENGTH_STEP:
        below = length - (length - LENGTH_MIN) % LENGTH_STEP
        problems.append(
            f"длина {length} не лежит на шаге {LENGTH_STEP} от {LENGTH_MIN} "
            f"(ближайшие {below} и {below + LENGTH_STEP})")
    return {
        "ok": not problems, "problems": problems,
        "width": width, "height": height, "length": length,
        "note": (f"геометрия {width}x{height}, {length} кадров: "
                 + ("годна" if not problems else "; ".join(problems))),
    }


def check_face_geometry(side: int | None = None,
                        length: int | None = None) -> dict:
    """Годится ли лицевой канал по размерности. Той же проверкой, что и кадр.

    Отдельной функцией, а не отдельной арифметикой: сторона лицевого канала
    подчиняется тому же контракту ноды, что ширина и высота кадра, и второй
    способ проверить кратность разъехался бы с первым (Е1). 512 = 32·16 —
    кратность выполняется, но это выполняется ЧИСЛОМ, а не удачей, и потому
    проверяется, а не читается глазами.
    """
    side = FACE_SIDE if side is None else side   # см. face_box: не в сигнатуре
    length = LENGTH_MIN if length is None else length
    got = check_geometry(side, side, length)
    got["note"] = "лицевой канал: " + got["note"]
    return got


def face_box(points: list, *, min_score: float | None = None,
             margin: float | None = None) -> tuple | None:
    """Квадрат вокруг 68 точек лица в координатах кадра. None — судить нечем.

    КВАДРАТ, а не габарит лица: нода режет `face_video` `crop="center"` до
    квадрата прежде, чем масштабировать (`nodes_wan.py:1208`). Отдав
    прямоугольник, мы отдали бы решение о том, что именно отрезать, коду,
    который про лицо ничего не знает.

    Бокс НЕ ПРИЖИМАЕТСЯ к границам кадра. Прижатый перестал бы быть квадратом,
    и его снова обрезала бы нода; лицо у края кадра честнее дорисовать фоном —
    на канале условий фон чёрный, и чёрное поле означает ровно «здесь точек
    нет».

    None возвращается, когда наблюдаемых точек меньше `FACE_BOX_MIN_POINTS`
    или когда они совпали в одну точку: это третий исход, «не смогли», и
    сворачивать его в пустой кадр обязан вызывающий явно, а не эта функция
    молча (Р1).
    """
    # Умолчания берутся ВНУТРИ, а не в сигнатуре: значение по умолчанию в
    # сигнатуре привязывается один раз при импорте, и подмена константы (Т1)
    # до него не доходит — тест на страже константы прошёл бы, не проверив её.
    # Поймано ровно так: мутация FACE_MARGIN не сдвинула ни одного пикселя.
    margin = FACE_MARGIN if margin is None else margin
    min_score = MIN_SCORE if min_score is None else min_score
    n = len(points)
    seen = [points[i][:2] for i in group_indices("face")
            if i < n and points[i][2] >= min_score]
    if len(seen) < FACE_BOX_MIN_POINTS:
        return None
    xs = [p[0] for p in seen]
    ys = [p[1] for p in seen]
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1.0 + 2.0 * margin)
    if side <= 0:
        return None
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def union_box(boxes) -> tuple | None:
    """Один квадрат на всю последовательность. None — ни одного бокса не было.

    ЗАЧЕМ ОБЩИЙ, а не свой на каждый кадр: покадровый бокс пересчитывается по
    точкам этого кадра, поэтому при неподвижной голове он всё равно дышит на
    величину дрожания детектора, и в 512-м кадре это дыхание превращается в
    наезд камеры, которого в драйвинге нет. Условие обязано нести мимику, а не
    движение кропа. Цена — лицо в кадре мельче на величину хода головы, и она
    названа.
    """
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


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


def decode_wholebody(x_logits, y_logits, box, input_size=None) -> list:
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

    input_size = dwpose.POSE_INPUT if input_size is None else input_size
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


def visible(points: list, channel: str, *,
            min_score: float | None = None) -> int:
    """Сколько точек канала наблюдаемо. ЧИСЛО, а не флаг (Е3).

    Флаг «канал снят» на месте этого числа читался бы как полная работа при
    одной уцелевшей точке из шестидесяти восьми.
    """
    min_score = MIN_SCORE if min_score is None else min_score
    n = len(points)
    return sum(1 for i in channel_indices(channel)
               if i < n and points[i][2] >= min_score)


def render(points: list, channel: str, width: int, height: int,
           *, min_score: float | None = None):
    """Одна режимная отрисовка канала в ЗАДАННЫХ координатах. Изображение PIL.

    Низкий уровень: рисует там, где лежат точки, и ничего не кропает. Канал
    тела уходит в ноду прямо отсюда — он и должен быть в геометрии кадра. Для
    лицевого канала прямой вызов НЕ ГОДИТСЯ: нужен кроп и 512, этим занят
    `render_face`, который сюда же и сводится после переноса координат.

    Канал `face` рисует ТОЛЬКО точки лица, канал `body` — тело, стопы и кисти
    и НИ ОДНОЙ точки лица. Это не оформительское решение: у модели два входа, и
    картинка, где смешано, не подходит ни к одному.

    Точки ниже порога не рисуются вовсе. Не «рисуются бледнее» — условие либо
    поставлено, либо нет, а полутон в условии модель прочтёт как факт.
    """
    from PIL import Image, ImageDraw

    min_score = MIN_SCORE if min_score is None else min_score
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


def blank_face(side: int | None = None):
    """Пустой лицевой кадр. Отдельным именем, чтобы «условия нет» было решением.

    Нужен там, где лицо нечитаемо: пропустить кадр нельзя — `face_video` идёт
    покадрово рядом с `pose_video`, и выпавший кадр сдвинет всю мимику
    относительно движения. Чёрный кадр — честное «на этом кадре сказать нечего»;
    он обязан быть посчитан вызывающим, поэтому и не прячется внутрь отрисовки.
    """
    from PIL import Image

    side = FACE_SIDE if side is None else side   # см. face_box: не в сигнатуре
    got = check_face_geometry(side)
    if not got["ok"]:
        raise ValueError(got["note"])
    return Image.new("RGB", (side, side), GROUND)


def render_face(points: list, *, box: tuple | None = None,
                side: int | None = None, min_score: float | None = None):
    """Лицевой канал: только 68 точек, кроп по лицу, ровно `side`×`side`.

    ЭТО и есть то, что уходит в `face_video`. `render(points, "face", w, h)`
    ниже по стеку остаётся сырой отрисовкой в заданных координатах и в ноду не
    годится: лицо на холсте кадра нода сожмёт вместе с кадром, и от 68 точек
    останется пятно — замерено, 1.64 px между соседними точками против 6.10 px
    после кропа (числа и условия у `FACE_MARGIN`).

    Бокс можно передать снаружи — так `render_sequence` даёт всей
    последовательности один кроп. Не передан — считается по этому кадру.
    Лицо нечитаемо — ValueError с числом наблюдаемых точек, а не тихий чёрный
    кадр: за чёрный кадр отвечает `blank_face`, и решение принимает вызывающий.
    """
    side = FACE_SIDE if side is None else side   # см. face_box: не в сигнатуре
    min_score = MIN_SCORE if min_score is None else min_score
    got = check_face_geometry(side)
    if not got["ok"]:
        raise ValueError(got["note"])
    if box is None:
        box = face_box(points, min_score=min_score)
    if box is None:
        raise ValueError(
            f"лицевой бокс не строится: наблюдаемо "
            f"{visible(points, FACE_CHANNEL, min_score=min_score)} точек из "
            f"{FACE_POINTS} при минимуме {FACE_BOX_MIN_POINTS} — это «судить "
            f"нечем», см. blank_face()")

    x0, y0, x1, y1 = box
    k = side / (x1 - x0)
    moved = [((x - x0) * k, (y - y0) * k, s) for x, y, s in points]
    return render(moved, FACE_CHANNEL, side, side, min_score=min_score)


def render_pair(points: list, width: int, height: int,
                *, box: tuple | None = None,
                min_score: float | None = None) -> dict:
    """Обе отрисовки разом: `{"face": 512×512, "body": width×height}`.

    Парой, а не двумя вызовами по месту: два канала одного кадра обязаны иметь
    один порог и один набор точек, а розданные по вызывающим они разъедутся.

    Холст у них РАЗНЫЙ, и это не небрежность: тело нода берёт в геометрии
    кадра и гонит через VAE, лицо — в 512×512 и в пикселях. Один холст на оба
    канала был бы удобнее ровно до первого прогона.

    Нечитаемое лицо даёт чёрный лицевой кадр: последовательности обязаны
    остаться одной длины. Сколько таких кадров — считает `render_sequence`, и
    число это в отчёте, а не флаг (Е3).
    """
    min_score = MIN_SCORE if min_score is None else min_score
    if box is None:
        box = face_box(points, min_score=min_score)
    face = (blank_face() if box is None
            else render_face(points, box=box, min_score=min_score))
    return {FACE_CHANNEL: face,
            BODY_CHANNEL: render(points, BODY_CHANNEL, width, height,
                                 min_score=min_score)}


def render_sequence(frame_paths, out_dir: str | Path,
                    *, min_score: float | None = None) -> dict:
    """Обе последовательности условий по кадрам драйвинга.

    Возвращает три исхода на кадр, не два (Р1): кадр либо снят, либо человека
    в нём не нашли, либо лицо в нём нечитаемо — и последнее НЕ ТО ЖЕ САМОЕ, что
    «лица нет». Итог печатается числами (`снято N из M`), а не флагом (Е3, Р2),
    а вердикт всей последовательности берётся из общего словаря исходов
    (`fork_identity`), чтобы «не смогли» читалось одинаково во всех потоках.

    ДВА ПРОХОДА, а не один. Первый снимает точки, второй рисует: лицевой кроп
    у всей последовательности ОБЩИЙ (`union_box`), иначе бокс дышит по кадрам и
    в 512-м кадре это выглядит наездом камеры, которого в драйвинге не было.
    Дорогая часть — ONNX — всё равно делается по разу на кадр.

    КАДР БЕЗ ЧЕЛОВЕКА ТОЖЕ ЗАПИСЫВАЕТСЯ, чёрным. ~~Прежняя редакция такой кадр
    пропускала~~ — снято: `face_video`, `pose_video` и кадры драйвинга нода
    режет по общей длине (`[:length]`, `nodes_wan.py:1207,1190`) и сопоставляет
    по порядку, поэтому выпавший кадр сдвинул бы всю мимику относительно
    движения на всём остатке ролика. Пропуск ловился бы только глазами на
    выходе, то есть после часа счёта.
    """
    from PIL import Image

    min_score = MIN_SCORE if min_score is None else min_score
    out = Path(out_dir)
    (out / FACE_CHANNEL).mkdir(parents=True, exist_ok=True)
    (out / BODY_CHANNEL).mkdir(parents=True, exist_ok=True)

    frames = [Path(p) for p in frame_paths]
    shot = []
    for p in frames:
        with Image.open(p) as im:
            size = im.size
        shot.append((p, size, wholebody_points(p)))

    box = union_box(face_box(pts, min_score=min_score)
                    for _, _, pts in shot if pts is not None)

    done, no_person, face_unreadable = [], [], []
    for i, (p, (w, h), pts) in enumerate(shot):
        if pts is None:
            no_person.append(p.name)
            blank_face().save(out / FACE_CHANNEL / f"{i:05d}.png")
            Image.new("RGB", (w, h), GROUND).save(
                out / BODY_CHANNEL / f"{i:05d}.png")
            continue
        if face_box(pts, min_score=min_score) is None:
            face_unreadable.append(p.name)
        pair = render_pair(pts, w, h, box=box, min_score=min_score)
        for channel, img in pair.items():
            img.save(out / channel / f"{i:05d}.png")
        done.append(p.name)

    total = len(frames)
    if total == 0 or not done or len(face_unreadable) == total:
        outcome = UNMEASURED
    elif no_person or face_unreadable:
        outcome = FAIL
    else:
        outcome = PASS
    return {
        "total": total,
        "rendered": len(done),
        "no_person": no_person,
        "face_unreadable": face_unreadable,
        "outcome": outcome,
        "face_box": box,
        "face_side": FACE_SIDE,
        "dir": {c: str(out / c) for c in CHANNELS},
        "note": (f"{outcome}: условия сняты на {len(done)} из {total} кадров; "
                 f"{len(no_person)} без человека, "
                 f"{len(face_unreadable)} с нечитаемым лицом "
                 f"(это НЕ «лица нет» — это «судить нечем»); "
                 f"лицевой канал {FACE_SIDE}x{FACE_SIDE}, "
                 f"общий кроп {box}"),
    }
