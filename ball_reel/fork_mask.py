"""Поток C форка: ПОСЛЕДОВАТЕЛЬНОСТЬ масок персонажа, с избирательным расширением.

ЧТО ЗДЕСЬ ПРОВЕРЯЕТСЯ, И ЭТО СКАЗАНО ВСЛУХ, ЧТОБЫ НЕ ПРОЧЛИ ИНАЧЕ: проверяются
МАСКИ, а не генерация. Ни одна строка этого модуля не говорит, что модель на
расширенную маску отзовётся другим телосложением. Гипотеза «телосложение —
расширением маски, а не деформацией тела» здесь ПОДАЁТСЯ, а подтверждается
только генерацией, которой в спринте нет.

---

ЗАЧЕМ ВООБЩЕ РАСШИРЯТЬ МАСКУ. В режиме Mix маска очерчивает персонажа: внутри
модель перерисовывает по референсу, снаружи кадры драйвинга проходят
нетронутыми. Значит маска — единственный рычаг, которым можно попросить тело
шире, чем у драйвинга. Все найденные модели прямого изменения фигуры отпали по
лицензиям (FBBR non-commercial, AGGN без лицензии, Odo на SMPL), и к тому же
против них есть геометрическое возражение: палка, которая длиннее тела, в поле
смещений изогнётся. Ширина маски таким возражением не страдает.

---

ГЛАВНОЕ ЧИСЛО МОДУЛЯ — 32, И ОНО МЕНЯЕТ СМЫСЛ ВСЕГО ОСТАЛЬНОГО.

В вендорском графе последним по маске стоит `BlockifyMask[32]` (проверено в
`workflows/upstream/video_wan2_2_14B_animate.json`, узел 276). То есть маска
квантуется до блоков 32x32, и **расширение силуэта меньше 32 px для модели
просто не существует**.

Без проверки на этот предел поток отдал бы честное «расширил на 5%», модель не
увидела бы ничего, а разница между «гипотеза не работает» и «мы её не подали»
стала бы неразличимой — то есть эксперимент потерял бы способность отвечать.
Поэтому `growth_verdict` отвергает расширение, не изменившее НИ ОДНОГО блока,
и отвергает его как НЕ СМОГЛИ ПОДАТЬ, а не как «не сработало» (Р1).

---

ЧТО ЗАМОРОЖЕНО И ПОЧЕМУ ИМЕННО ЭТО.

Кисти и голова не расширяются никогда. Не из осторожности: кисть на мяче — это
и есть то взаимодействие с предметом, которое продукт обещает НЕ СЛОМАТЬ, а
раздутая маска кисти отдаёт модели область, где мяч, и разрешает его
перерисовать. Голова заморожена по той же логике с другой стороны: там личность,
и менять её ширину — значит менять лицо, за которое отвечает совсем другой
канал.

---

НЕПРОВЕРЕНО (Ц4): семантика вендорского `BlockifyMask` прочитана по имени узла и
его единственному параметру, исходник ноды не читался — он в кастомном наборе,
который мы намеренно не тянем (см. `fork_comfy`). Здесь блок считается
занятым, если в нём есть хоть один пиксель маски. Это ОСТОРОЖНАЯ сторона: так
блокификация только расширяет маску, и предел 32 px остаётся нижней оценкой
видимости, а не завышенной.
"""

from __future__ import annotations

from pathlib import Path

from . import bodyparts, fork_channels

#: Всё, что НЕ фон, — это персонаж. Классы берутся у `bodyparts` по индексам
#: его собственного `LABELS`, а не переписываются числами: список читан из
#: метаданных модели, и второй его экземпляр разъедется при обновлении весов.
PERSON_CLASSES = tuple(i for i, name in enumerate(bodyparts.LABELS)
                       if name != "background")

#: Сторона блока, до которого вендор квантует маску. ИЗМЕРЕНО чтением графа:
#: узел 276, `BlockifyMask`, единственный виджет — 32.
BLOCK = 32

#: На сколько пикселей вендор растит маску до блокификации. ИЗМЕРЕНО там же:
#: узел 274, `GrowMask`, виджеты [10, True]. Наше расширение — ДРУГАЯ величина
#: и складывается с этой; держится рядом, чтобы обе были видны разом.
VENDOR_GROW = 10

#: Радиус заморозки вокруг кисти, в долях диагонали кадра. ВЫБРАНО: кисть с
#: пальцами занимает единицы процентов кадра, 0.06 накрывает её с запасом на
#: смаз. Калибровать не на чем, пока нет генерации; когда появится — двигать
#: здесь, а не по месту.
HAND_FREEZE = 0.06

#: То же вокруг головы. Больше, чем у кисти: голова крупнее, и захватить надо
#: вместе с волосами, которые сегментация относит к персонажу.
HEAD_FREEZE = 0.10

#: Три исхода (Р1). Берутся у потока A, а не заводятся заново: это одни и те же
#: три слова, и разъехавшись, они дадут два несравнимых отчёта (Е1).
from .fork_identity import FAIL, PASS, UNMEASURED  # noqa: E402


def person_mask(image, *, model=None):
    """Силуэт персонажа: True там, где не фон.

    Тонкая обёртка над `bodyparts.category_mask` — но обёртка нужная: «маска
    персонажа» и «маска кожи» это разные вопросы к одной модели, и
    вызывающему, который спутает их, никто не скажет.
    """
    import numpy as np

    cat = bodyparts.category_mask(image, model=model)
    return np.isin(np.asarray(cat), PERSON_CLASSES)


def frozen_zones(points, width: int, height: int) -> list:
    """Круги, внутри которых расширять нельзя: кисти и голова.

    Возвращает `[(cx, cy, r), ...]` в пикселях. Точки берутся из раскладки
    `fork_channels`, а не из своей копии индексов (Е1): раскладка там уже
    проверена тестом на непересечение и полноту.
    """
    diag = (width ** 2 + height ** 2) ** 0.5
    out = []
    if not points:
        return out

    def centre(indices, min_score=fork_channels.MIN_SCORE):
        seen = [points[i] for i in indices
                if i < len(points) and points[i][2] >= min_score]
        if not seen:
            return None
        return (sum(p[0] for p in seen) / len(seen),
                sum(p[1] for p in seen) / len(seen))

    for group, share in (("l_hand", HAND_FREEZE), ("r_hand", HAND_FREEZE)):
        c = centre(fork_channels.group_indices(group))
        if c is not None:
            out.append((c[0], c[1], diag * share))
    # Голова — по точкам головы COCO (нос, глаза, уши), а не по 68 лицевым:
    # лицевые кучнее и дают круг меньше самой головы.
    head = centre(range(0, 5))
    if head is not None:
        out.append((head[0], head[1], diag * HEAD_FREEZE))
    return out


def _disc(radius: int):
    """Круглый структурный элемент. Круг, а не квадрат: квадрат растит углы
    сильнее сторон, и силуэт получает плечи, которых у человека нет."""
    import numpy as np

    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def dilate(mask, radius: int):
    """Расширение маски на `radius` пикселей. Без scipy — своим проходом.

    scipy в зависимостях проекта нет, а тащить его ради одной морфологии
    значило бы добавить колесо в установку, которая и так собирается в три
    приёма из двух индексов.
    """
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return m.copy()
    disc = _disc(radius)
    h, w = m.shape
    pad = int(radius)
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    padded[pad:pad + h, pad:pad + w] = m
    out = np.zeros_like(m)
    dh, dw = disc.shape
    for dy in range(dh):
        for dx in range(dw):
            if disc[dy, dx]:
                out |= padded[dy:dy + h, dx:dx + w]
    return out


def grow_selective(mask, points, grow_px: int, *,
                   width: int | None = None, height: int | None = None):
    """Расширить силуэт ВЕЗДЕ, КРОМЕ кистей и головы.

    Порядок именно такой — сначала расширить целиком, потом отнять прирост в
    замороженных зонах, — а не «расширить только разрешённое». Второй способ
    рвёт силуэт на границе зоны: там, где предплечье входит в круг кисти,
    расширенная часть обрывается ступенькой, и модель получает маску с уступом.
    Отнимая ПРИРОСТ, мы оставляем исходный силуэт целым везде.
    """
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    width = w if width is None else width
    height = h if height is None else height

    grown = dilate(m, grow_px)
    added = grown & ~m
    for cx, cy, r in frozen_zones(points, width, height):
        yy, xx = np.ogrid[:h, :w]
        inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        added &= ~inside
    return m | added


def blockify(mask, block: int = BLOCK):
    """Квантовать маску до блоков `block`x`block`, как это делает вендор.

    Блок занят, если в нём есть хоть один пиксель маски — осторожная сторона,
    см. пометку НЕПРОВЕРЕНО в докстринге модуля.
    """
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    if block <= 0:
        raise ValueError(f"сторона блока должна быть положительной, дано {block}")
    h, w = m.shape
    bh = -(-h // block)
    bw = -(-w // block)
    out = np.zeros_like(m)
    for by in range(bh):
        for bx in range(bw):
            y0, y1 = by * block, min((by + 1) * block, h)
            x0, x1 = bx * block, min((bx + 1) * block, w)
            if m[y0:y1, x0:x1].any():
                out[y0:y1, x0:x1] = True
    return out


def blocks_changed(before, after, block: int = BLOCK) -> int:
    """Сколько блоков ПОЯВИЛОСЬ после расширения. Это и есть то, что видит модель.

    Считаются именно блоки, а не пиксели: пиксели, не дожившие до блока, для
    модели не существуют, и отчёт в пикселях врал бы ровно про то, ради чего
    поток заведён.
    """
    import numpy as np

    b = blockify(before, block)
    a = blockify(after, block)
    gained = np.asarray(a) & ~np.asarray(b)
    h, w = gained.shape
    n = 0
    for by in range(-(-h // block)):
        for bx in range(-(-w // block)):
            tile = gained[by * block:(by + 1) * block,
                          bx * block:(bx + 1) * block]
            if tile.any():
                n += 1
    return n


def growth_verdict(before, after, *, block: int = BLOCK) -> dict:
    """Дошло ли расширение до модели. Три исхода, и третий — главный.

    «Расширили, но ни один блок не изменился» — это НЕ «расширение не помогло».
    Это «расширение не подано». Свернуть первое во второе значит объявить
    гипотезу опровергнутой на прогоне, где её не проверяли.
    """
    import numpy as np

    b = np.asarray(before, dtype=bool)
    a = np.asarray(after, dtype=bool)
    if b.shape != a.shape:
        return {"outcome": UNMEASURED, "blocks": 0, "pixels": 0,
                "note": f"маски разного размера: {b.shape} и {a.shape}"}
    pixels = int((a & ~b).sum())
    if pixels == 0:
        return {"outcome": UNMEASURED, "blocks": 0, "pixels": 0,
                "note": ("расширение не добавило ни одного пикселя: подавать "
                         "модели нечего, а «не сработало» тут сказать не о чем")}
    n = blocks_changed(b, a, block)
    held = _blocks_occupied(b, block)
    share = round(n / held, 4) if held else 0.0
    if n == 0:
        return {
            "outcome": UNMEASURED, "blocks": 0, "pixels": pixels,
            "held": held, "share": 0.0,
            "note": (f"прирост {pixels} px НЕ ПЕРЕЖИЛ блокификацию до {block}: "
                     f"изменилось 0 блоков. Для модели этого расширения не "
                     f"существует — это «не смогли подать», а НЕ «гипотеза не "
                     f"работает». Расширять минимум на {block} px."),
        }
    return {"outcome": PASS, "blocks": n, "pixels": pixels,
            "held": held, "share": share,
            "note": (f"прирост {pixels} px пережил блокификацию: {n} блок(ов) "
                     f"{block}x{block} добавлено к {held} занятым "
                     f"({share:.1%}) — модель это увидит")}


def _blocks_occupied(mask, block: int = BLOCK) -> int:
    """Сколько блоков занимает исходный силуэт. Знаменатель для доли прироста.

    ЗАЧЕМ ДОЛЯ, А НЕ ТОЛЬКО «НОЛЬ ИЛИ НЕ НОЛЬ». Найдено живьём (П3), синтетика
    этого показать не могла: на РВАНОМ настоящем силуэте край пересекает сетку
    блоков почти везде, и даже рост на 5 px занимает несколько новых блоков —
    то есть формально «доходит». Замер на `demo/hero.png`:

        рост  5 px ->  5 блоков      рост 32 px -> 50 блоков
        рост 16 px -> 19 блоков      рост 64 px -> 92 блока

    Ноль блоков — крайний случай, а не единственный способ подать невидимое:
    пять блоков против пятисот занятых модель тоже не отличит от шума.
    Порога на долю здесь НЕТ намеренно — вывести его можно только генерацией,
    которой в спринте нет, — но число печатается, чтобы «расширил» и
    «расширил на 1%» нельзя было спутать.
    """
    import numpy as np

    # Считается перебором плиток, а НЕ делением суммы пикселей на площадь
    # блока: у кадра, не кратного 32, краевые блоки урезаны, и деление дало бы
    # дробное число блоков там, где их целое количество.
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    return sum(1
               for by in range(-(-h // block))
               for bx in range(-(-w // block))
               if m[by * block:(by + 1) * block,
                    bx * block:(bx + 1) * block].any())


def sequence(frame_paths, out_dir: str | Path, *, grow_px: int = BLOCK,
             block: int = BLOCK, model=None) -> dict:
    """Последовательность масок по кадрам — ровно то, что ждёт `character_mask`.

    Отчёт числами (Е3, Р2): сколько кадров снято, на скольких расширение дошло
    до модели, на скольких не дошло. Агрегатный флаг здесь читался бы как
    полная работа при одном удавшемся кадре из ста.
    """
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = [Path(p) for p in frame_paths]
    delivered, invisible, no_person = [], [], []
    for i, p in enumerate(frames):
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            base = person_mask(rgb, model=model)
        if not base.any():
            no_person.append(p.name)
            continue
        points = fork_channels.wholebody_points(p)
        grown = grow_selective(base, points, grow_px, width=w, height=h)
        verdict = growth_verdict(base, grown, block=block)
        (delivered if verdict["outcome"] == PASS else invisible).append(p.name)
        final = blockify(grown, block)
        Image.fromarray((final * 255).astype("uint8"), mode="L").save(
            out / f"{i:05d}.png")

    total = len(frames)
    return {
        "total": total, "written": total - len(no_person),
        "delivered": delivered, "invisible": invisible, "no_person": no_person,
        "grow_px": grow_px, "block": block, "dir": str(out),
        "outcome": (UNMEASURED if not delivered and not invisible else
                    PASS if not invisible else FAIL),
        "note": (f"масок записано {total - len(no_person)} из {total}; "
                 f"расширение {grow_px}px дошло до модели на "
                 f"{len(delivered)}, не дошло на {len(invisible)}, "
                 f"персонаж не найден на {len(no_person)}. "
                 f"ПРОВЕРЕНЫ МАСКИ, НЕ ГЕНЕРАЦИЯ."),
    }
