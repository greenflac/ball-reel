"""Отличительные приметы: доехала ли татуировка до результата.

Это измеритель заявленного дифференциатора. `PRODUCT.md` формулирует его так:
если приметы не переносятся, зритель не отличит наш результат от face swap.
Значит утверждение «приметы на месте» обязано быть проверяемым, иначе оно
остаётся лозунгом — а лозунг на аудите стоит меньше, чем честное «не умеем».

Канала переноса примет у нас ещё НЕТ. Метрика пишется первой намеренно: ручка
без счётчика уже один раз обошлась дорого — позу впрыскивали и не знали, что
она не доезжает, пока не посмотрели на картинку глазами.

ЧТО ЗДЕСЬ УСТОЙЧИВО, А ЧТО НЕТ

Примета переживает смену позы, света и масштаба — а вот её пиксели не
переживают ничего. Поэтому сравнивать напрямую бессмысленно, и метрика стоит на
двух свойствах, которые остаются:

**1. Место на ТЕЛЕ, а не в кадре.** Татуировка на предплечье сидит в
фиксированной точке относительно кости «локоть-запястье»: столько-то вдоль,
столько-то поперёк, в долях длины кости. Рука может уехать куда угодно — примета
поедет с ней. Поэтому координаты здесь костные, а не пиксельные, и переводятся в
пиксели уже по найденному скелету. Скелет у нас и так есть на обоих концах: на
фото и в результате.

**2. Контраст с ОКРЕСТНОЙ КОЖЕЙ, а не абсолютный цвет.** Тот же приём, что в
`garment.py`: сравнение с локальной опорой, а не с эталонным оттенком. Иначе
метрика будет мерить освещение. Примета — это участок, который отличается от
кожи вокруг него; насколько отличается, столько в ней и «приметности».

ЧЕГО ЭТА МЕТРИКА НЕ УМЕЕТ, И ЭТО НАДО ЗНАТЬ

Она надёжно ловит **ПОТЕРЮ** приметы и лишь слабо — **ПОДМЕНУ**. «Здесь что-то
контрастное того же масштаба» и «здесь тот самый дракон» — разные утверждения,
и второе она не делает. Полное сличение рисунка требует корреляции шаблона,
а шаблон разваливается при смене ракурса и освещения быстрее, чем успевает
помочь.

Это честный размен, и он в нашу пользу: наблюдаемый отказ — именно потеря.
Генератор не рисует чужую татуировку вместо вашей, он не рисует никакой.

Работает на CPU, ничего не генерирует, в сеть не ходит.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Кости, вдоль которых можно закрепить примету. Пары точек COCO-18, то есть
#: ровно то, что отдают и DWPose, и MediaPipe.
BONES = {
    "l_forearm": ("l_elbow", "l_wrist"),
    "r_forearm": ("r_elbow", "r_wrist"),
    "l_upperarm": ("l_shoulder", "l_elbow"),
    "r_upperarm": ("r_shoulder", "r_elbow"),
    "l_thigh": ("l_hip", "l_knee"),
    "r_thigh": ("r_hip", "r_knee"),
    "l_shin": ("l_knee", "l_ankle"),
    "r_shin": ("r_knee", "r_ankle"),
    "torso": ("neck", "hip_c"),
    "neck": ("neck", "nose"),
}

#: Меньше этого по стороне — судить не о чем. Та же природа, что у порога лица
#: и у порога области жидкости: на пятне 20x20 контраст с окрестностью
#: неотличим от шума сжатия. Число ВЫБРАНО по аналогии с MIN_FACE_PX, а не
#: измерено — калибровать его не на чем, пока нет ни одного прогона с приметой.
MIN_MARK_PX = 24

#: Во сколько раз кольцо окрестности шире самой приметы. Кольцо должно быть
#: заметно больше пятна (иначе в «кожу» попадёт сама примета) и заметно меньше
#: конечности (иначе в опору попадёт фон). 2.0 — середина этого коридора.
SURROUND_SCALE = 2.0

#: Ниже этого контраста с кожей область НЕ СЧИТАЕТСЯ приметой вовсе. Это
#: проверка ВХОДА, а не результата: если на референсе в указанном месте ровная
#: кожа, то переносить нечего, и сказать об этом надо сразу, а не отчитаться
#: потом «примета потеряна». ВЫБРАНО: 0.08 отделяет заметное глазу пятно от
#: неоднородности освещения на коже.
MIN_REFERENCE_CONTRAST = 0.08

#: Какая доля исходной «приметности» должна дожить до результата, чтобы примета
#: считалась перенесённой. ВЫБРАНО и заведомо мягко: генератор перерисовывает,
#: а не копирует, поэтому требовать сохранения контраста один в один — значит
#: браковать всё. Половина — это «след явно виден», а не «пиксельное совпадение».
PRESENT_RATIO = 0.5


@dataclass(frozen=True)
class Mark:
    """Примета в КОСТНЫХ координатах — поэтому она переживает смену позы.

    `along` — доля длины кости от её начала (0 у сустава-родителя, 1 у дочернего).
    `across` — смещение поперёк кости, тоже в долях её длины; знак задаёт сторону.
    `radius` — половина стороны окна, в долях длины кости.
    """

    bone: str
    along: float
    across: float = 0.0
    radius: float = 0.15
    name: str = "mark"


def locate(points: dict, mark: Mark, *, min_visibility: float = 0.5):
    """Костные координаты -> прямоугольник в пикселях. None, если судить нельзя.

    None здесь означает «кость не видна», а не «приметы нет». Разница
    принципиальная: рука, отвернувшаяся от камеры, — это отсутствие
    наблюдения, и записывать его как потерю приметы значит штрафовать
    генератор за геометрию.
    """
    pair = BONES.get(mark.bone)
    if pair is None:
        return None
    a, b = points.get(pair[0]), points.get(pair[1])
    if not a or not b:
        return None
    if min(a[2], b[2]) < min_visibility:
        return None
    w, h, _ = points.get("__size__", (0.0, 0.0, 1.0))
    if not w or not h:
        return None

    ax, ay, bx, by = a[0] * w, a[1] * h, b[0] * w, b[1] * h
    dx, dy = bx - ax, by - ay
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return None
    ux, uy = dx / length, dy / length          # вдоль кости
    px, py = -uy, ux                           # поперёк неё

    cx = ax + ux * length * mark.along + px * length * mark.across
    cy = ay + uy * length * mark.along + py * length * mark.across
    half = length * mark.radius
    if half * 2 < MIN_MARK_PX:
        return None
    return (int(cx - half), int(cy - half), int(cx + half), int(cy + half))


def _median_stats(arr):
    """Медианная яркость и насыщенность массива пикселей 0..1."""
    import numpy as np

    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
    hi, lo = arr.max(axis=-1), arr.min(axis=-1)
    sat = np.where(hi > 1e-6, (hi - lo) / np.maximum(hi, 1e-6), 0.0)
    return float(np.median(lum)), float(np.median(sat))


def distinctiveness(image, box) -> dict | None:
    """Насколько область отличается от кожи ВОКРУГ неё. None — судить нельзя.

    Считается против кольца-окрестности, а не против абсолютного оттенка:
    так метрика не превращается в измеритель освещения. Ровно тот же приём, что
    спас метрику одежды, где сравнение по среднему RGB давало ложную тревогу
    0.265 на съёмке с заведомо одной одеждой.
    """
    import numpy as np
    from PIL import Image

    x0, y0, x1, y1 = box
    if min(x1 - x0, y1 - y0) < MIN_MARK_PX:
        return None
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        with Image.open(image) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    else:
        arr = np.asarray(image, dtype=np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0
    h, w = arr.shape[:2]

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < MIN_MARK_PX or y1 - y0 < MIN_MARK_PX:
        return None
    inner = arr[y0:y1, x0:x1]

    # Кольцо: тот же центр, сторона в SURROUND_SCALE раз больше, минус середина.
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = int((x1 - x0) * SURROUND_SCALE / 2)
    sx0, sy0 = max(0, cx - half), max(0, cy - half)
    sx1, sy1 = min(w, cx + half), min(h, cy + half)
    ring = arr[sy0:sy1, sx0:sx1].copy()
    iy0, ix0 = y0 - sy0, x0 - sx0
    mask = np.ones(ring.shape[:2], dtype=bool)
    mask[max(0, iy0):max(0, iy0) + (y1 - y0),
         max(0, ix0):max(0, ix0) + (x1 - x0)] = False
    if mask.sum() < 16:
        return None
    skin = ring[mask]

    in_lum, in_sat = _median_stats(inner)
    sk_lum, sk_sat = _median_stats(skin.reshape(-1, 1, 3))
    base = max(sk_lum, 1e-3)
    return {
        # Знак сохраняем: тату темнее кожи, шрам светлее. Направление отличия —
        # часть подписи приметы, и терять его в модуле значит склеить два
        # разных объекта.
        "luma_contrast": round((in_lum - sk_lum) / base, 4),
        "sat_contrast": round(in_sat - sk_sat, 4),
        "score": round(abs(in_lum - sk_lum) / base + abs(in_sat - sk_sat), 4),
        "pixels": int((x1 - x0) * (y1 - y0)),
    }


def mark_transferred(reference: dict | None, produced: dict | None, *,
                     ratio: float = PRESENT_RATIO,
                     min_reference: float = MIN_REFERENCE_CONTRAST) -> dict:
    """Дожила ли примета от референса до результата.

    Три разных исхода, и путать их дорого:

    * **на референсе приметы нет** — это претензия ко ВХОДУ. Переносить нечего,
      и сообщить надо сразу, иначе потом отчитаемся «потеряна» о том, чего не
      было;
    * **не наблюдалось** — кость не видна или область мала. Это отсутствие
      измерения, а не потеря;
    * **потеряна / на месте** — собственно вердикт.
    """
    if reference is None:
        return {"verdict": None, "state": "not_observed",
                "note": "на референсе область не измерена: кость не видна или "
                        f"пятно мельче {MIN_MARK_PX}px — переносить нечего "
                        f"и судить не о чем"}
    if reference["score"] < min_reference:
        return {"verdict": None, "state": "no_mark_on_reference",
                "reference": reference["score"],
                "note": (f"контраст на референсе {reference['score']} < "
                         f"{min_reference}: в этом месте ровная кожа, приметы "
                         f"нет. Это про ВХОД, а не про результат.")}
    if produced is None:
        return {"verdict": None, "state": "not_observed",
                "reference": reference["score"],
                "note": "в результате кость не видна или область мала — "
                        "примета НЕ НАБЛЮДАЛАСЬ. Это не то же самое, что "
                        "потеряна: снимать ближе или брать кадр, где эта "
                        "часть тела видна."}
    kept = produced["score"] / max(reference["score"], 1e-6)
    # ПОРЯДОК ПРОВЕРОК ВАЖЕН, и первая версия его перепутала.
    #
    # Сначала «сколько дожило», и только потом «то ли это». На ровной коже
    # контраст равен нулю, а ноль формально «неотрицателен» — поэтому проверка
    # знака впереди объявляла потерю приметы ПЕРЕВОРОТОМ контраста и отправляла
    # чинить не то. Ноль — это отсутствие, а не переворот, и различие тут не
    # педантичное: «примета исчезла» и «на её месте что-то другое» чинятся
    # разными концами пайплайна.
    #
    # Знак поэтому судится только когда контраст ЕСТЬ, то есть когда в
    # результате действительно нашлось что-то приметное.
    lost = kept < ratio
    has_contrast = produced["score"] >= min_reference
    same_side = ((reference["luma_contrast"] >= 0)
                 == (produced["luma_contrast"] >= 0))
    ok = bool(not lost and (same_side or not has_contrast))
    if lost:
        return {"verdict": False, "state": "measured", "kept": round(kept, 4),
                "reference": reference["score"], "produced": produced["score"],
                "note": (f"ПРИМЕТА ПОТЕРЯНА: сохранилось {kept:.0%} контраста "
                         f"при пороге {ratio:.0%} — в результате там ровная "
                         f"кожа")}
    if has_contrast and not same_side:
        why = (f"контраст перевернулся ({reference['luma_contrast']} -> "
               f"{produced['luma_contrast']}): на месте приметы что-то другое")
    elif ok:
        why = f"примета на месте: сохранилось {kept:.0%} исходного контраста"
    else:
        why = (f"на месте приметы посторонний контраст ({produced['score']}) "
               f"при исходном {reference['score']}")
    return {"verdict": ok, "state": "measured", "kept": round(kept, 4),
            "reference": reference["score"], "produced": produced["score"],
            "note": why}


#: Ширина мягкого края при вклейке, в долях стороны окна. Резкий край читается
#: как наклейка мгновенно, даже когда цвет подобран идеально.
FEATHER = 0.18

#: Полуширина конечности в долях длины кости — граница, за которую вклейка не
#: выходит. Примета наносится не в КВАДРАТ, а в капсулу вдоль кости.
#:
#: Здесь стоял цветовой признак «похоже на кожу», и замер его опроверг. Взяты
#: два окна на одном кадре: плечо (кожа) дало разброс p50 0.063 / p90 0.586,
#: бедро в леггинсах (ткань) — p50 0.021 / p90 0.031. ОДЕЖДА ОКАЗАЛАСЬ
#: ОДНОРОДНЕЕ КОЖИ, потому что в окно на руке попадает силуэт и светотень.
#: Признак назывался «кожа», а мерил однородность, и отсекал ровно ту кожу,
#: на которую надо рисовать.
#:
#: Геометрия честнее: рука — это капсула вокруг кости, и за её пределами
#: заведомо фон. Это решает «не рисовать мимо конечности». Задачу «не рисовать
#: поверх одежды» она НЕ решает — для неё нужна сегментация, которой у нас нет,
#: и пока примету размещает оператор, который знает, где у него татуировка.
#: ВЫБРАНО 0.22: типичная рука примерно вчетверо длиннее своей ширины.
LIMB_HALF_WIDTH = 0.22


def _load(image):
    """Картинка -> массив 0..1. Принимает путь или уже готовый массив."""
    import numpy as np
    from PIL import Image

    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        with Image.open(image) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    arr = np.asarray(image, dtype=np.float64)
    return arr / 255.0 if arr.max() > 1.0 else arr


def _bone_frame(points: dict, bone: str):
    """(угол кости, её длина в пикселях) или None."""
    import math

    pair = BONES.get(bone)
    if pair is None:
        return None
    a, b = points.get(pair[0]), points.get(pair[1])
    if not a or not b:
        return None
    w, h, _ = points.get("__size__", (0.0, 0.0, 1.0))
    dx, dy = (b[0] - a[0]) * w, (b[1] - a[1]) * h
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx)), length


def transfer(source_image, source_points: dict, target_image,
             target_points: dict, mark: Mark, *, feather: float = FEATHER):
    """Перенести примету с референса в результат. (картинка, отчёт).

    ПЕРЕНОСЯТСЯ НЕ ПИКСЕЛИ, А ОТКЛОНЕНИЕ ОТ ОКРЕСТНОЙ КОЖИ. Это единственная
    часть замысла, которую стоит запомнить.

    Вклеить пиксели напрямую нельзя: на референсе другой свет, другой загар,
    другая экспозиция — заплатка будет видна как заплатка. Поэтому с референса
    снимается ОТНОШЕНИЕ каждого пикселя приметы к медианной коже вокруг неё, а
    в результат это отношение применяется к ЕГО коже. Тёмная линия татуировки
    остаётся «в 0.4 раза темнее окружающей кожи» — и на светлой коже, и на
    смуглой, и в тени.

    Ровно тот же принцип, на котором стоит метрика этого модуля, и та же
    причина: локальная опора вместо абсолютного цвета.

    Геометрия берётся из скелета: угол и длина кости известны на обоих концах,
    значит заплатка поворачивается и масштабируется под конечность в кадре.

    ПОЧЕМУ ВООБЩЕ ВКЛЕЙКА, А НЕ ГЕНЕРАЦИЯ. Правило продукта запрещает
    выдумывать: любой генеративный способ примету ПЕРЕРИСУЕТ, то есть выдаст
    похожую вместо той самой. Композит — единственный способ гарантировать, что
    на теле именно та татуировка, что на фото.

    ЧЕГО НЕ ДЕЛАЕТ: не рисует примету там, где кость не видна (вернёт кадр без
    изменений и скажет об этом), не исправляет ракурс сложнее плоского поворота,
    не знает про перекрытие рукой.
    """
    import numpy as np
    from PIL import Image

    report = {"name": mark.name, "applied": False, "note": ""}
    src_box = locate(source_points, mark)
    dst_box = locate(target_points, mark)
    target = _load(target_image)
    if src_box is None:
        report["note"] = "на референсе кость не видна — переносить нечего"
        return target, report
    if dst_box is None:
        report["note"] = ("в результате кость не видна — примета НЕ вклеена. "
                          "Это отказ от вмешательства, а не потеря: рисовать "
                          "примету на невидимой конечности значит выдумывать.")
        return target, report

    source = _load(source_image)
    sx0, sy0, sx1, sy1 = src_box
    patch = source[max(0, sy0):sy1, max(0, sx0):sx1]
    if patch.size == 0:
        report["note"] = "область приметы вышла за край референса"
        return target, report

    # Опора: медианная кожа вокруг приметы НА РЕФЕРЕНСЕ.
    src_stats = distinctiveness(source, src_box)
    if src_stats is None or src_stats["score"] < MIN_REFERENCE_CONTRAST:
        report["note"] = (f"на референсе в этом месте ровная кожа "
                          f"(контраст {None if src_stats is None else src_stats['score']}) "
                          f"— приметы нет, переносить нечего")
        return target, report

    src_skin = _skin_median(source, src_box)
    ratio = patch / np.maximum(src_skin, 1e-3)          # отклонение от кожи

    # Геометрия: повернуть и растянуть под кость в результате.
    src_frame = _bone_frame(source_points, mark.bone)
    dst_frame = _bone_frame(target_points, mark.bone)
    if src_frame and dst_frame:
        turn = dst_frame[0] - src_frame[0]
        img = Image.fromarray(np.clip(ratio / 4.0, 0, 1).astype(np.float32),
                              mode="F") if ratio.ndim == 2 else None
        # Поворот делаем поканально, чтобы не терять цвет отношения.
        chans = []
        for c in range(3):
            layer = Image.fromarray(ratio[..., c].astype(np.float32), mode="F")
            layer = layer.rotate(-turn, resample=Image.BILINEAR, fillcolor=1.0)
            chans.append(np.asarray(layer, dtype=np.float64))
        ratio = np.stack(chans, axis=-1)

    dx0, dy0, dx1, dy1 = dst_box
    dh, dw = dy1 - dy0, dx1 - dx0
    chans = []
    for c in range(3):
        layer = Image.fromarray(ratio[..., c].astype(np.float32), mode="F")
        layer = layer.resize((dw, dh), Image.BILINEAR)
        chans.append(np.asarray(layer, dtype=np.float64))
    ratio = np.stack(chans, axis=-1)

    dst_skin = _skin_median(target, dst_box)
    painted = np.clip(ratio * dst_skin, 0.0, 1.0)

    # Мягкий край: резкая граница читается как наклейка даже при точном цвете.
    yy, xx = np.mgrid[0:dh, 0:dw]
    edge = np.minimum.reduce([xx, yy, dw - 1 - xx, dh - 1 - yy]).astype(float)
    width = max(1.0, min(dh, dw) * feather)
    alpha = np.clip(edge / width, 0.0, 1.0)[..., None]

    out = target.copy()
    region = out[max(0, dy0):dy1, max(0, dx0):dx1]

    # НЕ ВЫХОДИТЬ ЗА КОНЕЧНОСТЬ. Найдено визуальным аудитом: без ограничения
    # крест лёг поверх лямки топа и фона, а метрика отрапортовала честные 74%
    # сохранившегося контраста — число было правдой, картинка нет.
    #
    # Ограничение геометрическое, а не цветовое: рука — капсула вокруг кости.
    # Цветовой признак «похоже на кожу» здесь уже пробовался и был опровергнут
    # замером (см. LIMB_HALF_WIDTH).
    if dst_frame:
        _, bone_len = dst_frame
        limb = bone_len * LIMB_HALF_WIDTH
        gy, gx = np.mgrid[0:dh, 0:dw]
        px_x = max(0, dx0) + gx
        px_y = max(0, dy0) + gy
        ax_, ay_ = target_points[BONES[mark.bone][0]][:2]
        bx_, by_ = target_points[BONES[mark.bone][1]][:2]
        W, H, _ = target_points["__size__"]
        ax_, ay_, bx_, by_ = ax_ * W, ay_ * H, bx_ * W, by_ * H
        vx, vy = bx_ - ax_, by_ - ay_
        vlen2 = max(vx * vx + vy * vy, 1e-6)
        s = np.clip(((px_x - ax_) * vx + (px_y - ay_) * vy) / vlen2, 0.0, 1.0)
        dist = np.hypot(px_x - (ax_ + s * vx), px_y - (ay_ + s * vy))
        inside = np.clip(1.0 - (dist - limb * 0.7) / max(limb * 0.3, 1e-6),
                         0.0, 1.0)
        alpha = alpha * inside[..., None]
        report["on_limb"] = round(float(inside.mean()), 3)
    if region.shape[:2] != painted.shape[:2]:
        painted = painted[:region.shape[0], :region.shape[1]]
        alpha = alpha[:region.shape[0], :region.shape[1]]
    out[max(0, dy0):dy1, max(0, dx0):dx1] = (
        region * (1 - alpha) + painted * alpha)
    report.update(applied=True,
                  note=(f"примета перенесена отклонением от кожи "
                        f"(контраст на референсе {src_stats['score']}). "
                        f"ПЛОСКО: без обёртывания по кривизне конечности и без "
                        f"перспективного сжатия к силуэту"))
    return out, report


def _skin_median(arr, box):
    """Медианный цвет кожи ВОКРУГ окна, каналами. Опора для переноса."""
    import numpy as np

    x0, y0, x1, y1 = box
    h, w = arr.shape[:2]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = int((x1 - x0) * SURROUND_SCALE / 2)
    sx0, sy0 = max(0, cx - half), max(0, cy - half)
    sx1, sy1 = min(w, cx + half), min(h, cy + half)
    ring = arr[sy0:sy1, sx0:sx1]
    mask = np.ones(ring.shape[:2], dtype=bool)
    iy0, ix0 = max(0, y0 - sy0), max(0, x0 - sx0)
    mask[iy0:iy0 + (y1 - y0), ix0:ix0 + (x1 - x0)] = False
    skin = ring[mask] if mask.sum() >= 16 else ring.reshape(-1, 3)
    return np.median(skin.reshape(-1, 3), axis=0)


def marks_report(ref_image, ref_points: dict, out_image, out_points: dict,
                 marks) -> dict:
    """Все приметы разом. Сводка для гейта и для отчёта."""
    rows = []
    for mark in marks:
        r = distinctiveness(ref_image, locate(ref_points, mark) or (0, 0, 0, 0))
        p = distinctiveness(out_image, locate(out_points, mark) or (0, 0, 0, 0))
        got = mark_transferred(r, p)
        got["name"] = mark.name
        rows.append(got)
    measured = [r for r in rows if r["state"] == "measured"]
    lost = [r["name"] for r in measured if not r["verdict"]]
    unobserved = [r["name"] for r in rows if r["state"] == "not_observed"]
    return {
        "marks": rows,
        "measured": len(measured),
        "lost": lost,
        "unobserved": unobserved,
        # Вердикт выносится ТОЛЬКО по измеренным. Ненаблюдавшаяся примета не
        # провал и не успех — она отдельной строкой, чтобы её не приняли ни за
        # то, ни за другое.
        "all_transferred": bool(measured) and not lost,
        "note": (f"{len(measured) - len(lost)}/{len(measured)} примет на месте"
                 + (f"; потеряны: {', '.join(lost)}" if lost else "")
                 + (f"; не наблюдались: {', '.join(unobserved)}"
                    if unobserved else "")
                 if measured else
                 "ни одной приметы не измерено — судить не о чем"),
    }
