"""Калибровочный стенд для метрики примет: разделяет ли она чистую кожу и тату.

ЗАЧЕМ ЭТОТ МОДУЛЬ СУЩЕСТВУЕТ. `marks.distinctiveness` отдаёт два канала —
массовый (`score`, среднее отклонение от окрестной кожи) и частотный
(`texture`, высокочастотная энергия против локального баланса). Пороги под них
(`MIN_REFERENCE_CONTRAST`, `MIN_TEXTURE_CONTRAST`) помечены в `marks.py` как
ВЫБРАННЫЕ, и стояли они на четырёх числах, снятых руками. Четыре числа — это не
калибровка, а анекдот: первая версия частотного порога была поставлена по одной
паре и на втором же образце пропустила дракона.

Здесь тот разовый эксперимент превращён в стенд, который можно перезапустить.

ЧТО ИМЕННО МЕРЯЕТСЯ, И ПОЧЕМУ ИМЕННО ПАРАМИ. Абсолютный уровень «приметности»
на живом кадре задаётся не приметой, а телом: собственная светотень круглой руки
даёт 0.31 массового контраста там, где никакой татуировки нет. Поэтому
одиночный замер ничего не значит, и стенд строит ПАРЫ: одна и та же фотография,
отличающаяся РОВНО приметой. Композит — умножением по скелету, ограниченный
маской кожи; всё остальное в двух кадрах совпадает пиксель в пиксель.

Умножение, а не вставка — потому что вставка даёт ровное чёрное пятно, на
котором провалиться невозможно, а умножение сохраняет светотень: чернила темнеют
там, где темна кожа под ними. Метод унаследован от `marktest/make_inked_reference.py`
и здесь обобщён: рисунок строится процедурно, значит его размер и тип задаются
параметром, а не файлом.

ЧТО ВАРЬИРУЕТСЯ, И ПОЧЕМУ БЕЗ ЭТОГО КАЛИБРОВКИ НЕ БЫВАЕТ

* **тип рисунка** — контурный (тонкие линии) и сплошной (заливка). Каналы ловят
  их по-разному: сплошное пятно внутри себя высоких частот НЕ ИМЕЕТ, только на
  краю, а контур почти не сдвигает среднее по окну. Стенд на одном типе
  откалибровал бы половину прибора;
* **размер** — вдоль кости, в долях её длины. На этом проекте уже дважды горели
  на синтетике, стоявшей в одной точке;
* **место** — кость и положение вдоль неё;
* **разрешение** — тот же кадр, уменьшенный. Это единственный способ отделить
  «мало пикселей» от «другое тело»: тело то же самое, свет тот же, меняется
  ровно число пикселей чернил.

КАК СЧИТАЕТСЯ РАЗДЕЛИМОСТЬ, И ЧЕГО ЗДЕСЬ НАМЕРЕННО НЕТ. Здесь НЕ подбирается
порог. Подобранный по выборке порог — это её пересказ, а не измерение. Считается
другое: перекрываются ли распределения чистых и приметных значений, насколько
широка полоса перекрытия и какую долю пар правильно разложит ЛУЧШИЙ ВОЗМОЖНЫЙ
порог. «Лучший возможный даёт 71%» — честная оценка потолка канала; «порог
0.104» — иллюзия точности.

Отдельно считается ПАРНАЯ разделимость: доля пар, где приметное значение выше
чистого. Это другой вопрос и другая задача пайплайна: проверка ПЕРЕНОСА
сравнивает два измерения ОДНОЙ приметы, и там работает отношение, а не
абсолютный уровень. Абсолютный порог нужен только проверке ВХОДА.

Работает на CPU, ничего не генерирует, в сеть не ходит. Требует скелета (DWPose)
и — желательно — сегментации кожи; без сегментации примета ограничивается
капсулой вокруг кости, и это помечается в отчёте, а не замалчивается.

    python3 -m ball_reel.calibrate_marks --out evidence/marks_calibration.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import marks

#: Во сколько раз чернила темнее кожи под ними. Множитель, а не цвет: 0.30
#: значит «в этой точке кожа стала втрое темнее себя же», и потому образец
#: остаётся живым при любом загаре и любом свете.
#:
#: ВЫБРАНО. Настоящая свежая татуировка на светлой коже даёт примерно такое
#: отношение; выцветшая — светлее. Число сознательно НЕ нулевое: чёрная заливка
#: (множитель 0) дала бы образец, на котором метрика не может провалиться.
INK_VALUE = 0.30

#: Чернила ВЫЦВЕТШЕЙ приметы: старая татуировка, светлый шрам, родинка на
#: смуглой коже. Отличие от кожи всего на треть, и это тот конец диапазона, где
#: метрика обязана начать ошибаться. Без него стенд мерил бы только лёгкий
#: случай и отчитывался бы красиво. ВЫБРАНО.
FADED_INK = 0.70

#: Ниже этого множителя пиксель считается ЧЕРНИЛАМИ при подсчёте размера
#: приметы. Не 1.0 намеренно: у повёрнутого и растянутого рисунка край
#: сглажен, и пиксели, потемневшие на процент, — это не чернила, а
#: интерполяция. ВЫБРАНО.
INK_PIXEL_MAX = 0.9

#: Толщина линии контурного рисунка в долях его ПОПЕРЕЧНОГО размера, и шаг
#: между линиями там же. ВЫБРАНО так, чтобы чернила заняли около четверти
#: площади: это и есть контур — рисунок, который глазу виден отлично, а среднее
#: по окну сдвигает слабо. Если шаг сделать равным толщине, линии сольются и
#: «контурный» образец станет сплошным — то есть стенд потеряет половину.
CONTOUR_LINE_FRACTION = 0.07
CONTOUR_PITCH_FRACTION = 0.24

#: Насколько рисунок уже полуширины конечности. Чуть уже намеренно: рисунок,
#: доходящий до силуэта, проверял бы обрезку по краю, а не перенос.
#: Унаследовано из `marktest/make_inked_reference.py`. ВЫБРАНО.
ACROSS_FRACTION = 0.85

#: Размеры приметы вдоль кости, в долях её длины. ДИАПАЗОН, а не точка — это
#: правило проекта, оплаченное дважды: вся первая синтетика заливала окно на
#: 100%, то есть стояла в единственной точке, где дефект не проявлялся.
#: ВЫБРАНО: от 0.15 (пятно размером с четверть предплечья) до 0.60 (рисунок в
#: полруки), то есть шестнадцатикратный разброс по площади чернил.
DEFAULT_SPANS = (0.15, 0.25, 0.40, 0.60)

#: Во сколько раз уменьшается кадр в разрешающем прогоне. Тело, свет и рисунок
#: те же самые — меняется ровно число пикселей чернил. ВЫБРАНО.
DEFAULT_SCALES = (1.0, 0.7, 0.5, 0.35)

#: Половина стороны измерительного окна в долях размера приметы. 0.5 значит
#: «окно ровно по примете»: сторона окна равна её длине вдоль кости.
#:
#: Это не украшение, а условие корректности сравнения размеров. `marks.py`
#: прямо предупреждает: метрика — среднее по окну, значит она разбавляется
#: чистой кожей, и примета в заведомо большом окне уйдёт под порог. Если окно
#: держать постоянным, а примету уменьшать, то «зависимость от размера»
#: окажется зависимостью от доли заполнения окна — то есть мы измерим свою же
#: разметку. ВЫБРАНО.
RADIUS_TO_SPAN = 0.5

#: Меньше этого числа пикселей чернил — пара НЕ СУДИТСЯ. Это отдельный исход
#: («не смогли измерить»), а не ноль и не провал: на сотне пикселей отличие
#: неотличимо от шума пережатия. ВЫБРАНО по аналогии с `marks.MIN_MARK_PX`
#: (24 px по стороне, то есть ~576 px площади сплошного пятна; контур от того
#: же окна оставляет около четверти) — калибровать сам этот порог не на чем.
MIN_INK_PIXELS = 150

#: Меньше этого числа пар о разделимости НЕ ГОВОРЯТ ВОВСЕ. Именно так и был
#: поставлен предыдущий порог — по одной паре, — и он пропустил дракона на
#: втором же образце. ВЫБРАНО: 6 пар — всё ещё мало для статистики, но уже
#: достаточно, чтобы увидеть перекрытие распределений, если оно есть.
MIN_PAIRS_FOR_SEPARABILITY = 6

#: Во сколько раз обязан различаться размер чернил внутри прогона, чтобы о
#: зависимости от размера вообще можно было говорить. ВЫБРАНО: тройка. Меньший
#: разброс — это одна точка с шумом, и «гипотеза не проверена» тогда честнее,
#: чем коэффициент корреляции по слипшемуся облаку.
SIZE_HYPOTHESIS_MIN_SPREAD = 3.0

#: Каналы метрики, по которым считается разделимость. Имена — ключи из
#: `marks.distinctiveness`.
CHANNELS = ("score", "texture")


@dataclass(frozen=True)
class Recipe:
    """Что именно наносим: тип рисунка, его размер, место и сила чернил."""

    name: str
    kind: str = "contour"          # "contour" | "solid" | "art"
    span: float = 0.40             # доля длины кости вдоль неё
    centre: float = 0.5            # где середина приметы на кости
    ink: float = INK_VALUE         # множитель чернил; больше — бледнее
    art: str | None = None         # файл рисунка для kind="art"


@dataclass(frozen=True)
class Sample:
    """На чём наносим: кадр, скелет, кость.

    `points` и `skin` можно не задавать — тогда они будут сняты DWPose и
    сегментацией. Задать их нужно там, где весов нет: тест обязан работать без
    моделей, иначе он превращается в проверку наличия файлов.
    """

    name: str
    image: object                  # путь или массив 0..1
    bone: str = "l_upperarm"
    points: dict | None = None
    skin: object | None = None     # булева маска размером с кадр


#: Набор по умолчанию — те кадры проекта, на которых кость видна и есть кожа.
#: Это НЕ отобранные «удачные» образцы: сюда входит всё, что вообще пригодно,
#: включая мелкие кадры 360x639, где примета еле проходит порог размера.
DEFAULT_SAMPLES = (
    ("kit/face.jpg", "l_upperarm"),
    ("kit/face.jpg", "l_forearm"),
    ("marktest_ref.png", "l_forearm"),
    ("marktest_ref.png", "l_upperarm"),
    ("marktest_ref.png", "r_shin"),
    ("attrtest/bodyref.png", "l_shin"),
    ("attrtest/bodyref.png", "l_upperarm"),
    ("attrtest/bodyref.png", "r_upperarm"),
    ("newref/01.png", "l_upperarm"),
    ("newref/02.png", "l_shin"),
    ("newref/03.png", "l_upperarm"),
    ("newref/03.png", "r_upperarm"),
    ("newref/04.png", "r_shin"),
)

#: Рецепты по умолчанию: два типа рисунка, три размера, два места на кости и
#: два уровня чернил. Ни одна из этих осей не убирается: тип решает, какой
#: канал сработает; размер уже дважды ловил стенд, стоявший в одной точке;
#: место меняет фон под приметой; сила чернил задаёт тот конец диапазона, где
#: метрика обязана ошибаться.
DEFAULT_RECIPES = (
    Recipe("contour-s", "contour", 0.25, 0.5),
    Recipe("contour-m", "contour", 0.40, 0.5),
    Recipe("contour-hi", "contour", 0.40, 0.32),
    Recipe("contour-pale", "contour", 0.40, 0.5, ink=FADED_INK),
    Recipe("solid-s", "solid", 0.25, 0.5),
    Recipe("solid-m", "solid", 0.40, 0.5),
    Recipe("solid-hi", "solid", 0.40, 0.32),
    Recipe("solid-pale", "solid", 0.40, 0.5, ink=FADED_INK),
)

#: Настоящий рисунок, которым ставился первоначальный ручной эксперимент. Он не
#: заменяет процедурные образцы (у них задан диапазон, а у него — одна точка),
#: но связывает стенд с теми семью числами, ради которых он написан.
DEFAULT_ART = "marktest/dragon_raw.png"


def default_recipes(root=None):
    """Рецепты по умолчанию плюс настоящий дракон, если файл на месте."""
    from pathlib import Path

    base = Path(root) if root else Path(__file__).resolve().parent.parent
    art = base / DEFAULT_ART
    if not art.exists():
        return list(DEFAULT_RECIPES)
    return list(DEFAULT_RECIPES) + [
        Recipe("art-m", "art", 0.40, 0.5, art=str(art)),
        Recipe("art-pale", "art", 0.40, 0.5, ink=FADED_INK, art=str(art)),
    ]


# --------------------------------------------------------------------------
# Рисунок
# --------------------------------------------------------------------------

def design(kind: str, along_px: int, across_px: int, *, ink: float = INK_VALUE,
           art: str | None = None):
    """Множитель чернил в системе КОСТИ. 1.0 — кожа не тронута.

    Рисунок строится длинной стороной вдоль кости и поворачивается в кадр уже
    потом. Обратный порядок — повернуть, затем растянуть — исказил бы пропорции.

    Три типа, и они не декоративные:

    * ``solid`` — сплошной эллипс. Внутри него высоких частот нет, только на
      краю: это образец, на котором частотный канал обязан быть слабее;
    * ``contour`` — эллипс, заполненный тонкими линиями. Чернил вчетверо
      меньше, среднее по окну почти не сдвигается — это образец, на котором
      обязан работать частотный канал, а массовый слепнет;
    * ``art`` — настоящий рисунок из файла. Одна точка, зато та самая, на
      которой ставился первоначальный ручной эксперимент.

    `ink` — во сколько раз чернила темнее кожи: 0.30 свежая татуировка, 0.70
    выцветшая. Ось силы такая же обязательная, как ось размера.
    """
    import numpy as np

    along_px, across_px = max(1, int(along_px)), max(1, int(across_px))
    layer = np.ones((along_px, across_px), dtype=np.float64)

    if kind == "art":
        from PIL import Image

        if art is None:
            raise ValueError("для kind='art' нужен файл рисунка (Recipe.art)")
        with Image.open(art) as im:
            drawn = np.asarray(
                im.convert("L").resize((across_px, along_px), Image.LANCZOS),
                dtype=np.float64) / 255.0
        # Чёрное на белом -> множитель от `ink` (чернила) до 1.0 (пусто).
        return ink + (1.0 - ink) * drawn

    yy, xx = np.mgrid[0:along_px, 0:across_px]
    ry, rx = along_px / 2.0, across_px / 2.0
    body = (((yy - ry + 0.5) / max(ry, 1e-6)) ** 2
            + ((xx - rx + 0.5) / max(rx, 1e-6)) ** 2) <= 1.0

    if kind == "solid":
        layer[body] = ink
        return layer
    if kind != "contour":
        raise ValueError(f"неизвестный тип рисунка {kind!r}: "
                         f"есть 'contour', 'solid' и 'art'")

    line = max(1, int(round(across_px * CONTOUR_LINE_FRACTION)))
    pitch = max(line + 1, int(round(across_px * CONTOUR_PITCH_FRACTION)))
    strokes = np.zeros_like(body)
    for start in range(0, across_px, pitch):
        strokes[:, start:start + line] = True
    for start in range(0, along_px, pitch):
        strokes[start:start + line, :] = True
    # Обводка: край эллипса всегда прорисован, иначе рисунок теряет силуэт.
    edge = body & ~(
        (((yy - ry + 0.5) / max(ry - line, 1e-6)) ** 2
         + ((xx - rx + 0.5) / max(rx - line, 1e-6)) ** 2) <= 1.0)
    layer[body & (strokes | edge)] = ink
    return layer


def _load(image):
    """Картинка -> массив 0..1. Путь или уже готовый массив.

    Своя копия, а не `marks._load`: тот приватный и принадлежит другому файлу.
    Стенд обязан переживать переименование внутренностей измеряемого модуля —
    иначе он мерил бы не метрику, а её реализацию.
    """
    import numpy as np
    from PIL import Image

    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        with Image.open(image) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    arr = np.asarray(image, dtype=np.float64)
    return arr / 255.0 if arr.max() > 1.0 else arr


def _bone_geometry(points: dict, bone: str):
    """(угол в градусах, длина в пикселях, центр кости в пикселях) или None.

    Своя копия арифметики, а не `marks._bone_frame`: приватная функция чужого
    модуля — не интерфейс, а `marks.BONES` и `attach_size` — да.
    """
    pair = marks.BONES.get(bone)
    if pair is None:
        return None
    a, b = points.get(pair[0]), points.get(pair[1])
    if not a or not b:
        return None
    size = points.get("__size__")
    if not size:
        return None
    w, h = float(size[0]), float(size[1])
    ax, ay, bx, by = a[0] * w, a[1] * h, b[0] * w, b[1] * h
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx)), length, (ax, ay, bx, by)


def _limb_mask(points: dict, bone: str, shape):
    """Капсула вокруг кости как запасная «кожа», когда сегментации нет.

    Это ХУЖЕ сегментации и помечается как таковое: капсула не отличает
    предплечье от рукава. Но она удерживает примету на конечности, и без неё
    стенд на машине без весов рисовал бы по фону.
    """
    import numpy as np

    h, w = shape[:2]
    gy, gx = np.mgrid[0:h, 0:w]
    uv = marks.limb_uv(points, bone, gx.astype(float), gy.astype(float))
    if uv is None:
        return np.zeros((h, w), dtype=bool)
    return uv[2]


def ink_pair(sample: Sample, recipe: Recipe) -> dict:
    """Построить пару «чисто / с приметой». Отличие ровно в примете.

    Возвращает словарь с обоими кадрами, окном измерения и числом пикселей
    чернил. Если наносить некуда (кость не видна, окно мельче порога, кожи в
    окне не осталось) — это ОТДЕЛЬНЫЙ ИСХОД `state="not_measured"` с причиной,
    а не пара с нулевым отличием.
    """
    import numpy as np
    from PIL import Image

    base = _load(sample.image)
    h, w = base.shape[:2]
    points = sample.points or _points_for(sample.image, (w, h))
    if points is None:
        return {"state": "not_measured", "note": "скелет не снят с кадра"}
    geom = _bone_geometry(points, sample.bone)
    if geom is None:
        return {"state": "not_measured",
                "note": f"кость {sample.bone} не видна на кадре"}
    angle, length, (ax, ay, bx, by) = geom

    mark = marks.Mark(bone=sample.bone, along=recipe.centre, across=0.0,
                      radius=recipe.span * RADIUS_TO_SPAN, name=recipe.name)
    box = marks.locate(points, mark)
    if box is None:
        return {"state": "not_measured",
                "note": (f"окно приметы {recipe.span * length:.0f} px мельче "
                         f"{marks.MIN_MARK_PX} px — судить не о чем"),
                "bone_px": round(length, 1)}

    along_px = int(length * recipe.span)
    across_px = int(length * marks.half_width_for(sample.bone)
                    * 2 * ACROSS_FRACTION)
    drawn = design(recipe.kind, along_px, across_px, ink=recipe.ink,
                   art=recipe.art)
    layer = Image.fromarray(drawn.astype(np.float32), mode="F").rotate(
        -(angle - 90.0), resample=Image.BILINEAR, expand=True, fillcolor=1.0)
    layer = np.asarray(layer, dtype=np.float64)

    cx = int((ax + bx) / 2 + (bx - ax) * (recipe.centre - 0.5))
    cy = int((ay + by) / 2 + (by - ay) * (recipe.centre - 0.5))
    ih, iw = layer.shape
    x0, y0 = cx - iw // 2, cy - ih // 2

    skin, skin_source = _skin_for(sample, points, base.shape)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(w, x0 + iw), min(h, y0 + ih)
    if xs1 <= xs0 or ys1 <= ys0:
        return {"state": "not_measured", "note": "примета вышла за край кадра"}

    sub = layer[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    inked = base.copy()
    region = inked[ys0:ys1, xs0:xs1]
    paint = skin[ys0:ys1, xs0:xs1] & (sub < 1.0)
    region[paint] *= sub[paint][:, None]
    inked[ys0:ys1, xs0:xs1] = region

    # Размер приметы — это ЧЕРНИЛА, а не задетые интерполяцией пиксели: край
    # повёрнутого рисунка сглажен, и считать его частью рисунка значит
    # завышать размер тем сильнее, чем мельче примета.
    ink_px = int((paint & (sub <= INK_PIXEL_MAX)).sum())
    changed = int((np.abs(inked - base).sum(axis=-1) > 0.01).sum())
    row = {
        "state": "ok", "clean": base, "inked": inked, "box": box, "mark": mark,
        "points": points, "bone_px": round(length, 1), "ink_px": ink_px,
        "changed_px": changed, "skin_source": skin_source,
        "design_px": (int(along_px), int(across_px)),
        "note": "",
    }
    if ink_px < MIN_INK_PIXELS:
        row["state"] = "not_measured"
        row["note"] = (f"чернил {ink_px} px < {MIN_INK_PIXELS}: примета почти "
                       f"не попала на кожу (одежда, край кадра или слишком "
                       f"мелкий кадр) — судить не о чем")
    return row


#: Скелет и маска кожи снимаются с ФАЙЛА, значит зависят только от его пути:
#: инференс на один кадр повторяется столько раз, сколько рецептов, а это
#: минуты на пустом месте. Кэш живёт в процессе и не переживает перезапуск.
_POINTS_CACHE: dict = {}
_SKIN_CACHE: dict = {}


def _points_for(image, size):
    """Скелет для кадра. None — снять не удалось (нет весов или нет человека)."""
    try:
        from . import dwpose
    except ImportError:                                   # pragma: no cover
        return None
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        key = (str(image), size)
        if key not in _POINTS_CACHE:
            pts = dwpose.pose_points(image) if dwpose.available() else None
            _POINTS_CACHE[key] = (marks.attach_size(pts, size) if pts else None)
        return _POINTS_CACHE[key]
    return None


def _skin_for(sample: Sample, points: dict, shape):
    """Маска «сюда можно рисовать» и её происхождение.

    Порядок именно такой: переданная маска, потом сегментация, потом капсула.
    Капсула — честный запасной вариант, а не эквивалент: она не знает про
    одежду, и в отчёте это видно словом `limb-capsule`.
    """
    import numpy as np

    if sample.skin is not None:
        return np.asarray(sample.skin, dtype=bool), "given"
    if isinstance(sample.image, (str, bytes)) or hasattr(sample.image, "__fspath__"):
        try:
            from . import bodyparts

            if bodyparts.available():
                key = str(sample.image)
                if key not in _SKIN_CACHE:
                    _SKIN_CACHE[key] = np.asarray(
                        bodyparts.skin_mask(sample.image), dtype=bool)
                mask = _SKIN_CACHE[key]
                if mask.shape[:2] == shape[:2]:
                    return mask, "segmentation"
        except Exception:                                 # pragma: no cover
            pass
    return _limb_mask(points, sample.bone, shape), "limb-capsule"


# --------------------------------------------------------------------------
# Замер
# --------------------------------------------------------------------------

def measure_pair(sample: Sample, recipe: Recipe) -> dict:
    """Одна строка таблицы: чистое, приметное и отношения по обоим каналам."""
    pair = ink_pair(sample, recipe)
    row = {"sample": sample.name, "bone": sample.bone, "recipe": recipe.name,
           "kind": recipe.kind, "span": recipe.span, "centre": recipe.centre,
           "state": pair["state"], "note": pair.get("note", "")}
    for key in ("bone_px", "ink_px", "skin_source"):
        if key in pair:
            row[key] = pair[key]
    if pair["state"] != "ok":
        return row

    clean = marks.distinctiveness(pair["clean"], pair["box"],
                                  points=pair["points"], bone=sample.bone)
    inked = marks.distinctiveness(pair["inked"], pair["box"],
                                  points=pair["points"], bone=sample.bone)
    if clean is None or inked is None:
        row["state"] = "not_measured"
        row["note"] = ("метрика отказалась судить окно: опорной кожи вокруг "
                       "приметы не осталось (окно шире конечности или примета "
                       "у края кадра)")
        return row

    row["state"] = "measured"
    for ch in CHANNELS:
        row[f"clean_{ch}"] = clean[ch]
        row[f"inked_{ch}"] = inked[ch]
        row[f"ratio_{ch}"] = round(inked[ch] / max(clean[ch], 1e-6), 3)
    row["clean_coverage"] = clean["coverage"]
    row["inked_coverage"] = inked["coverage"]
    row["baseline"] = inked["baseline"]
    row["baseline_suspect"] = inked["baseline_suspect"]
    return row


# --------------------------------------------------------------------------
# Разделимость
# --------------------------------------------------------------------------

def best_threshold(clean: list, inked: list) -> dict:
    """Потолок канала: лучший ВОЗМОЖНЫЙ порог и что он даёт.

    Здесь не подбирается рабочая константа. Подобранный по выборке порог — это
    её пересказ, и на новом теле он поедет. Считается верхняя граница: сколько
    вообще можно вытянуть из этого канала, если знать ответы заранее. Если даже
    так разложить не получается — канал не разделяет, и никакая настройка не
    поможет.

    Ниже порога — «чистая кожа», на пороге и выше — «примета». Возвращаются
    также границы полосы перекрытия: именно в ней живут все ошибки.

    ПОЧЕМУ ВЫБИРАЕТСЯ СБАЛАНСИРОВАННАЯ ТОЧНОСТЬ, а не общая. Классы здесь
    неравны по размеру: чистых окон меньше, чем приметных (на одном окне
    ставится несколько рисунков). Порог, выбранный по общей доле верных, в
    таком наборе выгодно сдвинуть в сторону большего класса — и получится
    «96% верных» у прибора, который всё подряд называет приметой. Поэтому
    максимизируется среднее двух долей, и обе они возвращаются по отдельности.
    """
    if not clean or not inked:
        return {"state": "not_measured",
                "note": "одна из сторон пуста — разделять нечего"}
    values = sorted(set(clean) | set(inked))
    cuts = [values[0] - 1.0]
    cuts += [(a + b) / 2.0 for a, b in zip(values, values[1:])]
    cuts.append(values[-1] + 1.0)

    total = len(clean) + len(inked)
    scored = []
    for t in cuts:
        tn = sum(1 for v in clean if v < t)
        tp = sum(1 for v in inked if v >= t)
        scored.append(((tn / len(clean) + tp / len(inked)) / 2.0, tn, tp, t))
    balanced, tn, tp, cut = max(scored, key=lambda s: (s[0], -s[3]))

    lo, hi = min(inked), max(clean)
    overlap = hi >= lo
    return {
        "state": "measured",
        "threshold": round(cut, 4),
        "balanced": round(balanced, 3),
        "accuracy": round((tn + tp) / total, 3),
        "correct": tn + tp, "total": total,
        "clean_correct": tn, "clean_n": len(clean),
        "inked_correct": tp, "inked_n": len(inked),
        "clean_min": round(min(clean), 4), "clean_max": round(hi, 4),
        "inked_min": round(lo, 4), "inked_max": round(max(inked), 4),
        "overlap": bool(overlap),
        # Ширина полосы перекрытия в долях полного размаха значений: 0 — классы
        # не касаются, 1 — они лежат друг в друге целиком.
        "overlap_width": (round((hi - lo) / max(max(clean + inked)
                                                - min(clean + inked), 1e-6), 3)
                          if overlap else 0.0),
    }


def paired_ratios(rows: list, channel: str) -> dict:
    """Парная разделимость: доля пар, где приметное значение выше чистого.

    Это ДРУГОЙ вопрос, чем абсолютный порог, и другая задача пайплайна. Проверка
    ПЕРЕНОСА сравнивает два измерения ОДНОЙ приметы — там отношение и работает.
    Проверке ВХОДА пары взять неоткуда, ей нужен абсолютный уровень.
    """
    ratios = [r[f"ratio_{channel}"] for r in rows if r["state"] == "measured"]
    if not ratios:
        return {"state": "not_measured", "note": "нет измеренных пар"}
    ratios.sort()
    mid = len(ratios) // 2
    median = (ratios[mid] if len(ratios) % 2 else
              (ratios[mid - 1] + ratios[mid]) / 2.0)
    return {
        "state": "measured", "pairs": len(ratios),
        "min": round(ratios[0], 3), "median": round(median, 3),
        "max": round(ratios[-1], 3),
        "grew": sum(1 for r in ratios if r > 1.0),
        "share_grew": round(sum(1 for r in ratios if r > 1.0) / len(ratios), 3),
    }


def _pearson(xs: list, ys: list) -> float | None:
    """Линейная корреляция. None — считать не на чем."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-12 or syy < 1e-12:
        return None
    return sxy / math.sqrt(sxx * syy)


def size_hypothesis(rows: list, channel: str = "texture") -> dict:
    """Зависит ли отношение внутри пары от размера приметы в пикселях.

    Гипотеза родилась на семи разрозненных точках, где менялось всё сразу —
    тело, свет, кость, размер. На таком материале «корреляция с размером» и
    «корреляция с телом» неразличимы. Поэтому проверять её положено на строках
    ОДНОГО кадра: тогда единственное, что меняется, — число пикселей чернил.

    Разброс размеров меньше `SIZE_HYPOTHESIS_MIN_SPREAD` — это отказ считать, а
    не слабая корреляция: облако из одной точки коэффициент всё равно выдаст.
    """
    good = [r for r in rows if r["state"] == "measured" and r.get("ink_px")]
    if len(good) < 3:
        return {"state": "not_measured",
                "note": f"измеренных точек {len(good)} < 3 — корреляции нет"}
    px = [float(r["ink_px"]) for r in good]
    spread = max(px) / max(min(px), 1e-6)
    if spread < SIZE_HYPOTHESIS_MIN_SPREAD:
        return {"state": "not_measured", "spread": round(spread, 2),
                "note": (f"размер чернил меняется всего в {spread:.1f}x при "
                         f"требуемых {SIZE_HYPOTHESIS_MIN_SPREAD}x — это одна "
                         f"точка с шумом, а не зависимость")}
    ratios = [float(r[f"ratio_{channel}"]) for r in good]
    r_lin = _pearson(px, ratios)
    r_log = _pearson([math.log(v) for v in px], ratios)
    return {
        "state": "measured", "points": len(good), "channel": channel,
        "spread": round(spread, 2),
        "ink_px_min": int(min(px)), "ink_px_max": int(max(px)),
        "ratio_at_min": good[px.index(min(px))][f"ratio_{channel}"],
        "ratio_at_max": good[px.index(max(px))][f"ratio_{channel}"],
        "pearson": None if r_lin is None else round(r_lin, 3),
        "pearson_log": None if r_log is None else round(r_log, 3),
    }


def separability(rows: list) -> dict:
    """Сводка по обоим каналам: абсолютная и парная разделимость.

    Пар меньше `MIN_PAIRS_FOR_SEPARABILITY` — вердикта НЕТ. Это тот самый
    отдельный исход «не смогли измерить»: предыдущий порог был поставлен по
    одной паре и пропустил дракона на втором образце.
    """
    measured = [r for r in rows if r["state"] == "measured"]
    out = {"pairs": len(measured), "rows_total": len(rows),
           "not_measured": len(rows) - len(measured)}
    if len(measured) < MIN_PAIRS_FOR_SEPARABILITY:
        out["state"] = "not_measured"
        out["note"] = (f"измеренных пар {len(measured)} < "
                       f"{MIN_PAIRS_FOR_SEPARABILITY}: о разделимости "
                       f"распределений на таком материале не говорят")
        return out
    out["state"] = "measured"

    # ЧИСТЫХ ОКОН МЕНЬШЕ, ЧЕМ ПАР: на одном и том же окне ставится несколько
    # рисунков, а чистое измерение у них общее — оно зависит только от кадра,
    # кости и геометрии окна. Считать его столько раз, сколько было рецептов,
    # значит размножить один и тот же замер и выдать копии за наблюдения.
    seen, clean_rows = set(), []
    for r in measured:
        key = (r["sample"], r["bone"], r["centre"], r["span"])
        if key not in seen:
            seen.add(key)
            clean_rows.append(r)
    out["clean_windows"] = len(clean_rows)
    for ch in CHANNELS:
        out[ch] = {
            "absolute": best_threshold([r[f"clean_{ch}"] for r in clean_rows],
                                       [r[f"inked_{ch}"] for r in measured]),
            "paired": paired_ratios(measured, ch),
        }
    for kind in sorted({r["kind"] for r in measured}):
        subset = [r for r in measured if r["kind"] == kind]
        out.setdefault("by_kind", {})[kind] = {
            ch: paired_ratios(subset, ch) for ch in CHANNELS}
    return out


# --------------------------------------------------------------------------
# Прогоны
# --------------------------------------------------------------------------

def calibrate(samples=None, recipes=None) -> dict:
    """Полная таблица: каждый образец на каждом рецепте, плюс разделимость."""
    samples = list(samples if samples is not None else default_samples())
    recipes = list(recipes if recipes is not None else default_recipes())
    rows = [measure_pair(s, r) for s in samples for r in recipes]
    return {"rows": rows, "separability": separability(rows),
            "samples": len(samples), "recipes": len(recipes)}


def size_sweep(sample: Sample, kind: str = "contour",
               spans=None) -> dict:
    """Один кадр, одна кость, РАЗНЫЙ размер приметы.

    Отвечает на вопрос, который таблица по разным телам ответить не может:
    падает ли отношение от того, что чернил мало, или от того, что тело другое.
    Здесь тело одно.
    """
    spans = tuple(spans if spans is not None else DEFAULT_SPANS)
    rows = [measure_pair(sample, Recipe(f"{kind}-{s:.2f}", kind, s))
            for s in spans]
    return {"rows": rows, "spans": spans, "sample": sample.name,
            "hypothesis": {ch: size_hypothesis(rows, ch) for ch in CHANNELS}}


def resolution_sweep(sample: Sample, recipe: Recipe | None = None,
                     scales=None) -> dict:
    """Тот же кадр в разном разрешении: чистая проверка «мало пикселей».

    Скелет нормирован в 0..1, поэтому он переносится на уменьшенный кадр без
    повторного инференса — меняется ровно число пикселей, а не геометрия.
    """
    import numpy as np
    from PIL import Image

    recipe = recipe or Recipe("contour-m", "contour", 0.40, 0.5)
    scales = tuple(scales if scales is not None else DEFAULT_SCALES)
    base = _load(sample.image)
    h, w = base.shape[:2]
    points = sample.points or _points_for(sample.image, (w, h))
    if points is None:
        return {"rows": [], "hypothesis": {ch: {"state": "not_measured",
                                                "note": "скелет не снят"}
                                           for ch in CHANNELS}}
    skin, skin_source = _skin_for(sample, points, base.shape)

    rows = []
    for scale in scales:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = np.asarray(
            Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
            .resize((nw, nh), Image.LANCZOS), dtype=np.float64) / 255.0
        mask = np.asarray(
            Image.fromarray(skin.astype(np.uint8) * 255, mode="L")
            .resize((nw, nh), Image.NEAREST), dtype=np.uint8) > 127
        small = Sample(f"{sample.name}@{scale:.2f}", img, sample.bone,
                       points=marks.attach_size(points, (nw, nh)), skin=mask)
        row = measure_pair(small, recipe)
        row["scale"] = scale
        row["skin_source"] = skin_source
        rows.append(row)
    return {"rows": rows, "scales": scales, "sample": sample.name,
            "hypothesis": {ch: size_hypothesis(rows, ch) for ch in CHANNELS}}


def default_samples(root=None):
    """Образцы проекта, которые ФИЗИЧЕСКИ ЕСТЬ рядом. Нет файла — нет строки."""
    from pathlib import Path

    base = Path(root) if root else Path(__file__).resolve().parent.parent
    out = []
    for rel, bone in DEFAULT_SAMPLES:
        path = base / rel
        if path.exists():
            out.append(Sample(rel, str(path), bone))
    return out


# --------------------------------------------------------------------------
# Вывод
# --------------------------------------------------------------------------

def _as_run(sweep: dict, why: str) -> dict:
    """Прогон — не выборка: разделимость по нему НЕ считается, и это говорится."""
    return {**sweep, "separability": {"state": "not_measured", "note": why}}


def format_table(result: dict) -> str:
    """Таблица для человека. Неизмеренные строки НЕ прячутся."""
    head = (f"{'образец':<21}{'кость':<11}{'рецепт':<13}{'черн.px':>8}  "
            f"{'score: чисто -> с тату':<26}{'texture: чисто -> с тату':<26}")
    lines = [head, "-" * len(head)]
    for r in result["rows"]:
        left = f"{r['sample']:<21}{r['bone']:<11}{r['recipe']:<13}"
        if r["state"] != "measured":
            lines.append(f"{left}{r.get('ink_px', 0):>8}  "
                         f"НЕ ИЗМЕРЕНО: {r['note'][:58]}")
            continue
        lines.append(
            f"{left}{r['ink_px']:>8}  "
            f"{r['clean_score']:.4f} -> {r['inked_score']:.4f} "
            f"{r['ratio_score']:>5.2f}x   "
            f"{r['clean_texture']:.4f} -> {r['inked_texture']:.4f} "
            f"{r['ratio_texture']:>5.2f}x")
    sep = result["separability"]
    lines.append("")
    if sep["state"] != "measured":
        lines.append(f"РАЗДЕЛИМОСТЬ НЕ ИЗМЕРЕНА: {sep['note']}")
        return "\n".join(lines)
    lines.append(f"измерено пар: {sep['pairs']} из {sep['rows_total']} "
                 f"({sep['not_measured']} не измерено); "
                 f"чистых окон: {sep['clean_windows']}")
    for ch in CHANNELS:
        a, p = sep[ch]["absolute"], sep[ch]["paired"]
        lines.append(
            f"{ch}: чистое {a['clean_min']}..{a['clean_max']}, "
            f"с приметой {a['inked_min']}..{a['inked_max']}; "
            f"перекрытие {'ДА' if a['overlap'] else 'нет'} "
            f"(ширина {a['overlap_width']})")
        lines.append(
            f"    ЛУЧШИЙ ВОЗМОЖНЫЙ порог {a['threshold']}: сбалансированная "
            f"точность {a['balanced']:.0%} — чистых узнано "
            f"{a['clean_correct']}/{a['clean_n']}, приметных "
            f"{a['inked_correct']}/{a['inked_n']}")
        lines.append(
            f"    парно: отношение {p['min']}..{p['max']} "
            f"(медиана {p['median']}), выросло у {p['grew']}/{p['pairs']} пар")
    for kind, per in sorted(sep.get("by_kind", {}).items()):
        lines.append(
            f"    по типу {kind}: score x{per['score']['median']} медиана, "
            f"texture x{per['texture']['median']} медиана "
            f"({per['score']['pairs']} пар)")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="ball_reel.calibrate_marks",
        description="калибровка метрики примет на управляемых парах")
    ap.add_argument("--json", help="куда сложить полную таблицу")
    ap.add_argument("--out", help="куда сложить отчёт в markdown")
    ap.add_argument("--sweep", action="store_true",
                    help="плюс прогоны по размеру и по разрешению")
    args = ap.parse_args(argv)

    samples = default_samples()
    if not samples:
        print("нет ни одного образца рядом с пакетом — калибровать не на чем")
        return 1
    result = calibrate(samples)
    print(format_table(result))

    if args.sweep or args.out:
        biggest = max(samples, key=lambda s: _bone_px(s))
        result["size_sweep"] = size_sweep(biggest)
        result["resolution_sweep"] = resolution_sweep(biggest)
        print("\nразмер приметы на ОДНОМ кадре:")
        print(format_table(_as_run(result["size_sweep"], "прогон по размеру")))
        print(f"  гипотеза о размере (texture): "
              f"{result['size_sweep']['hypothesis']['texture']}")
        print("\nтот же кадр в разном разрешении:")
        print(format_table(_as_run(result["resolution_sweep"], "прогон по разрешению")))
        print(f"  гипотеза о размере (texture): "
              f"{result['resolution_sweep']['hypothesis']['texture']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in result.items() if k != "images"},
                      fh, ensure_ascii=False, indent=1, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(as_markdown(result))
        print(f"\nотчёт: {args.out}")
    return 0


def _bone_px(sample: Sample) -> float:
    """Длина кости образца в пикселях. 0 — снять не удалось."""
    base = _load(sample.image)
    h, w = base.shape[:2]
    points = sample.points or _points_for(sample.image, (w, h))
    if points is None:
        return 0.0
    geom = _bone_geometry(points, sample.bone)
    return 0.0 if geom is None else geom[1]


def as_markdown(result: dict) -> str:
    """Отчёт в evidence/. Пишется машиной, чтобы его нельзя было приукрасить."""
    sep = result["separability"]
    out = ["# Калибровка метрики примет", "",
           f"Построено пар: {sep['pairs']} измеренных из "
           f"{sep['rows_total']} запрошенных "
           f"({sep['not_measured']} не измерено).", "",
           "```", format_table(result), "```", ""]
    for key, title in (("size_sweep", "Размер приметы на одном кадре"),
                       ("resolution_sweep", "Тот же кадр в разном разрешении")):
        if key not in result:
            continue
        out += [f"## {title}", "", "```",
                format_table(_as_run(result[key], "прогон, не выборка")),
                "", f"гипотеза (texture): {result[key]['hypothesis']['texture']}",
                f"гипотеза (score):   {result[key]['hypothesis']['score']}",
                "```", ""]
    return "\n".join(out)


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
