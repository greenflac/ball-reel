"""Датасет для LoRA личности из ОДНОЙ фотографии. Сбор, отбор, подписи.

ЗАЧЕМ ЭТО, ЕСЛИ ЕСТЬ FaceID. FaceID переносит лицо эмбеддингом ArcFace — и тем
же ArcFace мы потом проверяем результат. Генератор оптимизируется ровно к той
величине, которой его судят, то есть правило «судья независим от судимого»
нарушено в самом центре пайплайна. Заметно это стало не сразу: канал писался
раньше, чем гейт.

LoRA закрывает эту дыру, если — и только если — **ArcFace не участвует в
порождении датасета**. Иначе круг замкнётся через посредника: модель выучит «то,
что нравится ArcFace», просто на шаг дальше от глаз.

Отсюда разделение ролей, и оно здесь главное:

    порождение   шлюз (своим механизмом референса)   ArcFace НЕ участвует
    отбор        независимый распознаватель A        ArcFace НЕ участвует
    обучение     LoRA
    суд          ArcFace (B)                         независим от обоих

ЧЕГО ЭТОТ МОДУЛЬ НЕ ОБЕЩАЕТ. Одна фотография содержит ровно столько информации о
человеке, сколько содержит; ни аугментация, ни синтетика её не добавляют. По
литературе синтетика без реальной опоры уплывает от целевого распределения, и
поэтому реальное фото остаётся в наборе якорем с повышенным весом.

Более того: измеренный дрейф шлюзовых моделей на нашей рефке — 0.345 у лучшей
(seedream5) при баре «тот же человек» 0.35. То есть датасет по построению
состоит из ПОХОЖЕГО человека, и обученная на нём LoRA не может оказаться ближе
к оригиналу, чем её материал. Ожидать выигрыша по дистанции ArcFace нельзя, и
цель не в нём: цель — личность без ArcFace в контуре.

Работает на CPU. Генерацию не делает сам — только говорит, что генерировать, и
решает, что оставить.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Сколько изображений просить у генератора на одно выжившее. ВЫБРАНО по
#: измеренному дрейфу шлюза: у лучшей модели 0.345 и 0.367 на двух пробах, то
#: есть около половины кадров ложится по ту сторону бара. Берём с запасом.
OVERSHOOT = 2.5

#: Ниже этого числа изображений обучать бессмысленно — LoRA переобучится на
#: один ракурс. ВЫБРАНО по практике обучения персонажных LoRA для SD1.5, где
#: обычный набор 15-30 кадров; нижняя граница взята с запасом вниз.
MIN_DATASET = 12

#: Во сколько раз реальное фото весит больше синтетического. Литература прямо
#: говорит: без реальной опоры синтетика уплывает от целевого распределения.
#: ВЫБРАНО: якорь должен быть заметен, но не подавлять разнообразие.
REAL_ANCHOR_REPEATS = 5

#: Что МЕНЯЕТСЯ от кадра к кадру. Это и есть ось разнообразия: без неё LoRA
#: выучит не человека, а одну конкретную фотографию вместе с её светом и фоном.
VARIATIONS = {
    "framing": ("waist-up portrait", "full body standing",
                "three-quarter view from the knees up", "head and shoulders"),
    "angle": ("facing the camera", "turned slightly to the left",
              "turned slightly to the right", "seen from a low angle"),
    "light": ("soft daylight from a window", "bright overhead studio light",
              "warm evening light", "overcast diffuse light"),
    "place": ("in a bright gym", "against a plain grey wall",
              "in a home living room", "outdoors in a park"),
}

#: Что НЕ МЕНЯЕТСЯ и потому в подписи НЕ УПОМИНАЕТСЯ. Правило подписи короткое:
#: описывай то, что должно варьироваться, и молчи о том, что должно прилипнуть
#: к триггеру. Написав в каждой подписи «женщина с драконом на плече», мы учим
#: модель, что дракон — часть слова-триггера, и потом не сможем его убрать.
CONSTANT = ("лицо", "телосложение", "приметы", "цвет волос", "цвет глаз")


@dataclass
class Sample:
    """Один кадр набора вместе с его происхождением.

    Происхождение хранится не для порядка: набор смешанный, и вопрос «сколько
    здесь реального» — первый, который задаст любой, кто будет разбираться,
    почему LoRA выучила именно это.
    """

    path: str
    origin: str                      # "real" | "augmented" | "generated"
    prompt: str = ""
    caption: str = ""
    repeats: int = 1
    score: float | None = None       # дистанция независимого распознавателя
    kept: bool | None = None
    note: str = ""


def variations(limit: int | None = None) -> list:
    """Комбинации осей разнообразия. Порядок — по одной оси за раз.

    ПОЧЕМУ НЕ ДЕКАРТОВО ПРОИЗВЕДЕНИЕ. Оно даёт 256 комбинаций, из которых
    большинство различается мелочью, а стоит каждая одинаково. Меняя по одной
    оси от базового набора, получаем максимум РАЗЛИЧИЯ на кадр — а именно
    различие и покупается генерацией.
    """
    keys = list(VARIATIONS)
    base = {k: VARIATIONS[k][0] for k in keys}
    out = [dict(base)]
    for k in keys:
        for value in VARIATIONS[k][1:]:
            row = dict(base)
            row[k] = value
            out.append(row)
    return out[:limit] if limit else out


def prompt_for(row: dict, subject: str = "a woman") -> str:
    """Комбинация осей -> текст запроса. Личность НЕ описывается словами."""
    return (f"{subject}, {row['framing']}, {row['angle']}, "
            f"{row['light']}, {row['place']}, photographic, natural skin "
            f"texture, sharp focus on the face")


def caption_for(row: dict, trigger: str) -> str:
    """Подпись для обучения: триггер плюс ТОЛЬКО то, что меняется.

    Всё постоянное (лицо, телосложение, приметы) намеренно не упоминается —
    именно оно должно прилипнуть к триггеру. Это единственное правило подписей,
    которое действительно важно, и нарушить его легче всего из лучших
    побуждений: подробная подпись кажется полезнее короткой.
    """
    return (f"{trigger}, {row['framing']}, {row['angle']}, "
            f"{row['light']}, {row['place']}")


def augmentations(*, allow_mirror: bool = False) -> list:
    """Преобразования одного реального кадра. Зеркало по умолчанию ЗАПРЕЩЕНО.

    ЗЕРКАЛО — САМАЯ ДОРОГАЯ ОШИБКА В ЭТОЙ ЗАДАЧЕ, и она общепринятая: в
    руководствах по обучению LoRA отражение стоит первым в списке дешёвых
    аугментаций. Для персонажа с ПРИМЕТАМИ оно ядовито: татуировка с левого
    предплечья переезжает на правое, и модель учится, что она бывает и там, и
    там. Ровно то, что наш продукт обязан не допускать — приметы переносятся
    на СВОЁ место, иначе мы не лучше face swap.

    То же касается несимметричных черт: пробора, шрама, родинки.

    Оставлены преобразования, не меняющие сторону: масштаб, сдвиг кадра,
    небольшой поворот, мягкая правка яркости и температуры.
    """
    rows = [
        ("crop_tight", "кадр плотнее на 10% — учит, что человек может быть ближе"),
        ("crop_wide", "кадр шире на 10%"),
        ("rotate_ccw", "поворот на 4 градуса против часовой"),
        ("rotate_cw", "поворот на 4 градуса по часовой"),
        ("bright_up", "ярче на 8% — свет меняется, человек нет"),
        ("bright_down", "темнее на 8%"),
        ("warm", "теплее по цветовой температуре"),
        ("cool", "холоднее по цветовой температуре"),
    ]
    if allow_mirror:
        rows.append(("mirror", "ЗЕРКАЛО: приметы меняют сторону — включать "
                               "только если у субъекта нет несимметричных примет"))
    return rows


def plan(target: int = 20, *, price_per_image: float = 0.035,
         overshoot: float = OVERSHOOT) -> dict:
    """Сколько генерировать и почём, чтобы после отбора осталось `target`.

    Считается ДО траты. Дрейф шлюза измерен (0.345 и 0.367 у лучшей модели при
    баре 0.35), то есть примерно половина кадров отсеется; запас берётся с
    поправкой на это.
    """
    to_generate = int(round(target * overshoot))
    return {
        "target": target,
        "generate": to_generate,
        "augmented_from_real": len(augmentations()),
        "real_anchor_repeats": REAL_ANCHOR_REPEATS,
        "pollen": round(to_generate * price_per_image, 3),
        "note": (f"сгенерировать {to_generate}, чтобы после отбора осталось "
                 f"~{target}; плюс {len(augmentations())} аугментаций и одно "
                 f"реальное фото с весом {REAL_ANCHOR_REPEATS}. "
                 f"Оценка {round(to_generate * price_per_image, 3)} pollen"),
    }


def select(samples: list, *, bar: float, min_kept: int = MIN_DATASET) -> dict:
    """Отбор по НЕЗАВИСИМОМУ распознавателю. Три исхода, а не два.

    `bar` — порог дистанции того распознавателя, которым отбираем. Он ОБЯЗАН
    отличаться от того, которым потом судят обученную LoRA: отобрав по ArcFace,
    мы выберем «то, что нравится ArcFace», и независимость суда потеряем ровно
    там, где собирались её получить.

    Кадр без оценки не отбрасывается молча: «не смогли измерить» — отдельное
    состояние, и оно попадает в отчёт, потому что большая доля неизмеримых
    означает, что отбор вообще не состоялся.
    """
    kept, dropped, unscored = [], [], []
    for s in samples:
        if s.origin == "real":
            s.kept, s.repeats = True, REAL_ANCHOR_REPEATS
            s.note = "реальный якорь: в отбор не входит и не может"
            kept.append(s)
            continue
        if s.score is None:
            s.kept, s.note = None, "НЕ ИЗМЕРЕНО: лицо не найдено или мельче порога"
            unscored.append(s)
            continue
        s.kept = s.score <= bar
        (kept if s.kept else dropped).append(s)
        s.note = (f"дистанция {s.score:.3f} "
                  f"{'<=' if s.kept else '>'} бар {bar}")

    enough = len(kept) >= min_kept
    return {
        "kept": kept, "dropped": dropped, "unscored": unscored,
        "enough": enough,
        "real": sum(1 for s in kept if s.origin == "real"),
        "generated": sum(1 for s in kept if s.origin == "generated"),
        "augmented": sum(1 for s in kept if s.origin == "augmented"),
        "note": (f"оставлено {len(kept)} из {len(samples)} "
                 f"(отсеяно {len(dropped)}, не измерено {len(unscored)})"
                 + ("" if enough else
                    f"; МАЛО: нужно хотя бы {min_kept}, иначе LoRA "
                    f"переобучится на один ракурс")),
    }


def independence_report(*, generator: str, selector: str, judge: str) -> dict:
    """Проверка того, что круг не замкнулся. Главная гарантия этого модуля.

    Утверждение «LoRA даёт личность независимо от ArcFace» проверяемо ровно
    одним способом: посмотреть, не участвует ли судья в порождении или отборе.
    Здесь это делается кодом, а не обещанием в презентации.
    """
    same_family = {"arcface", "buffalo_l", "buffalo_s", "antelopev2",
                   "auraface", "insightface", "ip-adapter-faceid"}

    def fam(name: str) -> bool:
        return any(k in name.lower() for k in same_family)

    problems = []
    if fam(generator) and fam(judge):
        problems.append("судья участвует в ПОРОЖДЕНИИ: обученная модель "
                        "оптимизируется к той же величине, которой её судят")
    if fam(selector) and fam(judge):
        problems.append("судья участвует в ОТБОРЕ: набор выбран по тому, что "
                        "нравится судье, и оценка завышена")
    return {
        "generator": generator, "selector": selector, "judge": judge,
        "independent": not problems,
        "problems": problems,
        "note": ("судья независим от порождения и отбора" if not problems
                 else "; ".join(problems)),
    }


def manifest(samples: list, *, trigger: str, judge: str,
             selector: str, generator: str) -> dict:
    """Полное происхождение набора. То, что кладётся рядом с весами LoRA.

    Без него через неделю невозможно ответить, почему LoRA выучила именно это:
    сколько в наборе было реального, чем отбирали и что судило.
    """
    kept = [s for s in samples if s.kept]
    return {
        "trigger": trigger,
        "size": len(kept),
        "steps_equivalent": sum(s.repeats for s in kept),
        "by_origin": {o: sum(1 for s in kept if s.origin == o)
                      for o in ("real", "augmented", "generated")},
        "independence": independence_report(generator=generator,
                                            selector=selector, judge=judge),
        "mirror_used": any("mirror" in s.note for s in samples),
        "samples": [{"path": s.path, "origin": s.origin, "repeats": s.repeats,
                     "score": s.score, "caption": s.caption, "note": s.note}
                    for s in kept],
    }
