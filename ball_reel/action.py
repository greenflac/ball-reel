"""Совершил ли сгенерированный человек ЗАЯВЛЕННОЕ движение.

Дыра, ради которой написан модуль, уже сработала. Живой прогон дал кадр с
лучшим за день дрейфом лица (0.21), на котором человек в бальном платье СИДИТ
в позе с фотографии. Все двенадцать осей гейта (`produce.CHECK_ORDER`) его
пропустили — и это не случайность, а следствие того, ЧТО каждая из них меряет:

* `pose_wander` сверяет позу с РЕФЕРЕНСОМ и ПОКАДРОВО. Сидящий человек сидит
  ровно в референсной позе, поэтому расстояние маленькое, и ось довольна.
  Референс задаёт стартовую конфигурацию, а не движение: driving-кадры в это
  сравнение вообще не входят.
* `motion_amount` и `motion_physical` меряют, что хоть что-то шевелится и что
  шевелится непрерывно. Разница соседних кадров в пикселях не знает, ЧЬЁ это
  движение: дыхание, качание камеры и прыжок для неё одинаково «движение».
* `garment` меряет СТАБИЛЬНОСТЬ одежды между кадрами, то есть «одна и та же
  ли она», а не «та ли, что заказана».
* `anatomy` (`limb_consistency`) меряет постоянство длин костей. У сидящего
  человека кости идеально постоянны — это лучший возможный балл.
* `loop` меряет стык, идентичность — лицо.

Ни одна не задаёт вопрос: движение, которое совершил сгенерированный человек,
— ТО ЖЕ САМОЕ, что совершил человек на driving-видео? Он и есть содержание
заказа: условия ControlNet построены покадрово по driving-кадрам, и кадр i
обязан повторить кадр i донора.

## Что сравнивается

Не позы, а ТРАЕКТОРИИ. Для донорской и для сгенерированной последовательности
считаются изменения НОРМИРОВАННОЙ позы (`pose._normalise`: центрирование на
бёдрах, деление на длину торса — она и делает «снято ближе» ≠ «другая поза»),
после чего сравниваются сами последовательности изменений. Кадры соответствуют
один к одному (условие i порождено driving-кадром i, карта — в
`kit/conditions/manifest.json`, поле `driving_frames`), поэтому выравнивание
уже есть и DTW не нужен.

ДВА ЧИСЛА, И СМЕШИВАТЬ ИХ НЕЛЬЗЯ — это измерено, а не выбрано по вкусу.
Каждое из них слепо ровно там, где видит другое (числа — из таблицы ниже):

    случай                            amplitude   direction   ловит
    донор против себя                    1.00        1.00     — (эталон)
    приглушённое x0.5 («в такт, вяло»)   0.50        1.00     amplitude
    ТОЧНО замороженный кадр x16          0.00        н/д      amplitude
    замороженный + дрожь детектора       0.14-1.43   0.015    direction
    перетасованный                       1.42        -0.16    direction
    обращённый во времени                1.00        +0.04    direction
    другой фрагмент того же видео        6.03        +0.15    direction

Одно число из двух дало бы либо «сидящего человека с дрожащим детектором»
как норму (по amplitude), либо «вялое, но верное движение» как норму (по
direction). Корреляция ВЕЛИЧИН по кадрам («движется ли в такт») проверялась
третьим каналом и выброшена: на 66 парах реальных окон она давала от -0.66 до
+0.77 при том, что честный сдвиг на один кадр давал +0.41 — распределения
перекрываются полностью, порога не существует, число только создаёт видимость
измерения.

## Почему шаг в ТРИ кадра, а не в один

Соседние кадры отличаются на 0.050 длины торса (медиана по 71 реальному
кадру), и на этом масштабе направление шага задаёт дрожь детектора, а не
движение. Проверка: клип, делающий ТО ЖЕ движение с опозданием на один кадр —
то есть заведомо годный, — на шаге 1 получает direction +0.02, то есть
объявляется браком. На шаге 3 он получает +0.78, а весь брак остаётся ниже
+0.40. Шаг 3 — самый короткий, при котором это так (шаг 2 даёт +0.56 при браке
до +0.30; шаги 4-5 дают ещё немного запаса, но съедают сравнения: на клипе в
16 кадров шаг 5 оставляет 11 шагов вместо 13).

## Три исхода, и третий обязан браковать замороженный клип

`same` / `different` / `not_measurable`. Третий — не «плохо», а честный отказ:
позы не нашлись, кадров меньше минимума, донор сам не движется, карты на
driving нет. Но клип из 16 ОДИНАКОВЫХ кадров этим исходом закрываться не
должен: проект уже ловил ровно эту ошибку — такой клип не проваливал ни одной
проверки, потому что все отвечали «не смог», и ноль на выходе читался как
успех. Здесь замороженный клип идёт по amplitude = 0.00 в `different`, а не в
`not_measurable`, и порядок проверок в `compare_trajectories` выбран так
намеренно.

## Чего ось НЕ видит (говорится прямо, чтобы на неё не полагались шире)

* НЕ ТОТ ОБЪЕКТ И НЕ ТА ОДЕЖДА. Ось геометрическая: человек в бальном платье и
  человек в спортивной форме, совершающие одно движение, для неё неразличимы.
  Это ловится только семантикой кадра (CLIP/VLM), и ни одна ось гейта сегодня
  этого не делает.
* ПЕРЕНОС ТЕЛА ЦЕЛИКОМ. `_normalise` центрирует на бёдрах, поэтому подъём и
  опускание всего тела без смены конфигурации для оси — ноль. Замерено на
  доноре: перенос таза 0.031 длины торса за кадр против 0.050 смены
  конфигурации, то есть примерно 0.6 движения донора — вне поля зрения этой
  оси (её видит `motion_amount`, в пикселях).
* ЗЕРКАЛО. Левая и правая рука не проверяются на подмену: направления обеих
  входят в одну сумму.
* СКОРОСТЬ в абсолютных единицах: сравнение всегда с донором, а не с секундами.

## Как воспроизвести числа

    python3 -m ball_reel.action --controls kit/driving

Пять контрольных случаев строятся из настоящих driving-кадров, а не из
синтетики; синтетика живёт в тестах, где проверяется арифметика.

numpy + `pose.landmarks` (MediaPipe, локально). Ни сети, ни платных вызовов.
"""

from __future__ import annotations

import json
from pathlib import Path

from .pose import BODY_POINTS, MIN_VISIBILITY, _normalise, landmarks

#: Шаг сравнения в кадрах. ИЗМЕРЕН (см. докстринг, раздел «Почему шаг в ТРИ»):
#: на шаге 1 годный клип с опозданием на кадр получает direction +0.02, на
#: шаге 2 — +0.56, на шаге 3 — +0.78 при браке не выше +0.40. Меньше 3 нельзя,
#: больше 3 — можно, но каждая единица стоит одного сравнения.
STEP_STRIDE = 3

#: Ниже этого числа кадров вердикт не выносится. ИЗМЕРЕН на реальном доноре:
#: разрыв между «то же движение, исполненное неточно» и браком держится только
#: на длинных окнах (минимум годного / максимум брака):
#:      4 кадра   0.83 / 0.91   разрыва НЕТ, ось бессмысленна
#:      6         0.82 / 0.74
#:      8         0.76 / 0.59
#:     10         0.71 / 0.54
#:     12         0.71 / 0.44   первое окно, где брак ушёл ниже DIRECTION_MIN
#:     16         0.68 / 0.26
#: На 10 кадрах и короче два РАЗНЫХ куска одного и того же движения набирают
#: больше 0.50, то есть проходят как «то же движение». Продуктовый клип — 16
#: кадров, так что 12 не мешает работе и оставляет запас.
MIN_TRAJECTORY_FRAMES = 12

#: Совпадение НАПРАВЛЕНИЙ изменения: взвешенный по величине косинус между
#: шагами донора и клипа, -1..+1. ВЫБРАН серединой измеренного разрыва.
#: Измерены обе границы разрыва, и это разные по надёжности замеры:
#:   верх брака 0.40 — на настоящих данных (66 пар окон донора + перетасовка,
#:                     обращение времени, чужой фрагмент);
#:   низ годного 0.68 — на настоящих данных, ИСПОРЧЕННЫХ синтетически
#:                     (дрожь 0.05 торса на сустав, опоздание на кадр).
#: Настоящего сгенерированного клипа против этого донора на диске нет, поэтому
#: НЕПРОВЕРЕНО, где на самом деле лежит годная генерация; первый живой прогон
#: обязан перепроверить это число, а не унаследовать его.
DIRECTION_MIN = 0.50

#: Доля движения донора, воспроизведённая клипом (медиана величин шагов клипа
#: к медиане донора). ВЫБРАН: измеренных случаев между 0.20 и 0.50 нет вообще,
#: опорные точки — точная заморозка 0.00, приглушение x0.2 = 0.20, x0.5 = 0.50,
#: честный сдвиг на кадр 0.99. 0.35 — это решение «меньше трети движения донора
#: не считается тем же движением», а не замер. Верхнего порога сознательно нет:
#: дрожь детектора поднимает amplitude произвольно высоко (замерено: 1.43 на
#: ЗАМОРОЖЕННОМ клипе с дрожью 0.05), и «слишком много движения» надёжно ловит
#: direction, а не это число.
AMPLITUDE_MIN = 0.35

#: Если сам донор на этом окне двигается меньше — судить нечего: и отношение
#: величин, и косинус считаются от шума. ВЫБРАН с опорой на замеры: медиана
#: шага донора на шаге 3 равна 0.098 длины торса, десятый процентиль 0.059, то
#: есть 0.03 вдвое ниже самого спокойного реального участка; при этом дрожь
#: 0.01 торса на сустав, влитая в замороженный клип, даёт шаги того же порядка.
MIN_DONOR_STEP = 0.03

#: Столько суставов должно быть видно В ОБОИХ кадрах, иначе шаг не считается.
#: То же правило, что в `pose.pose_delta` (там оно записано числом в коде):
#: по трём точкам «конфигурация тела» — это уже не измерение.
MIN_SHARED_JOINTS = 4

_JOINTS = tuple(sorted(BODY_POINTS))


def _steps(poses, stride: int) -> list:
    """Изменения нормированной позы через `stride` кадров.

    На входе — последовательность landmark-словарей (как их отдаёт
    `pose.landmarks`), где None означает «тела в кадре не нашли». На выходе —
    список длиной ``len(poses) - stride``: для каждого шага словарь
    ``{сустав: вектор смещения}`` в длинах торса, либо None, если шаг измерить
    не удалось.
    """
    import numpy as np

    normed = [None if p is None else _normalise(p) for p in poses]
    out: list = []
    for i in range(len(normed) - stride):
        a, b = normed[i], normed[i + stride]
        if a is None or b is None:
            out.append(None)
            continue
        vec = {j: np.asarray(b[j][0]) - np.asarray(a[j][0]) for j in _JOINTS
               if j in a and j in b
               and a[j][1] >= MIN_VISIBILITY and b[j][1] >= MIN_VISIBILITY}
        out.append(vec if len(vec) >= MIN_SHARED_JOINTS else None)
    return out


def compare_trajectories(driving, generated, *,
                         stride: int = STEP_STRIDE,
                         direction_min: float = DIRECTION_MIN,
                         amplitude_min: float = AMPLITUDE_MIN,
                         min_frames: int = MIN_TRAJECTORY_FRAMES,
                         min_donor_step: float = MIN_DONOR_STEP) -> dict:
    """Сравнить две траектории поз. Чистая арифметика, без картинок и модели.

    `driving` и `generated` — РАВНОЙ ДЛИНЫ последовательности landmark-словарей
    (None там, где тела не нашли), выровненные один к одному: кадр i клипа
    порождён driving-кадром i.

    Возвращает словарь с `verdict` ("same" / "different" / "not_measurable"),
    числами `amplitude` и `direction`, ключом `follows` (True только при
    "same") и `note` человеческим текстом. `follows` намеренно False и при
    "not_measurable": ось, которая не смогла измерить, не имеет права
    выглядеть как пройденная — на этом проект уже обжигался.
    """
    import numpy as np

    if len(driving) != len(generated):
        raise ValueError(
            f"compare_trajectories: длины не совпали — {len(driving)} "
            f"driving-кадр(ов) против {len(generated)} кадр(ов) клипа. Кадры "
            f"должны соответствовать один к одному (условие i порождено "
            f"driving-кадром i). Возьмите карту из поля driving_frames в "
            f"manifest.json рядом с условиями — см. driving_frames_for().")

    # `follows` НЕ False, а None: у вердикта три исхода, и булев флаг рядом с
    # ним схлопывает их обратно в два. Читатель, взявший `follows` вместо
    # `verdict`, получил бы «движение не то» там, где мы просто не смогли
    # измерить — а это разные решения: первое бракует клип, второе требует
    # разобраться, почему нечем судить. Ровно эта ошибка уже ловилась в позе,
    # в жидкости и в предполёте.
    empty = {"verdict": "not_measurable", "follows": None, "amplitude": None,
             "direction": None, "steps": 0, "frames": len(driving),
             "donor_step": None, "check": None}
    if len(driving) < min_frames:
        return {**empty, "note": (
            f"движение НЕ ИЗМЕРЕНО: кадров {len(driving)}, нужно хотя бы "
            f"{min_frames} — на более коротких окнах два разных куска одного "
            f"движения набирают выше порога и проходят как то же самое.")}

    d_steps, g_steps = _steps(driving, stride), _steps(generated, stride)
    mag_d, mag_g, dots, prods = [], [], [], []
    for d, g in zip(d_steps, g_steps):
        if d is None or g is None:
            continue
        shared = sorted(set(d) & set(g))
        if len(shared) < MIN_SHARED_JOINTS:
            continue
        D = np.concatenate([d[j] for j in shared])
        G = np.concatenate([g[j] for j in shared])
        nd, ng = float(np.linalg.norm(D)), float(np.linalg.norm(G))
        k = np.sqrt(len(shared))
        mag_d.append(nd / k)
        mag_g.append(ng / k)
        if nd > 1e-9 and ng > 1e-9:
            dots.append(float(D @ G))
            prods.append(nd * ng)

    needed = min_frames - stride
    if len(mag_d) < needed:
        return {**empty, "steps": len(mag_d), "note": (
            f"движение НЕ ИЗМЕРЕНО: измеримых шагов {len(mag_d)}, нужно "
            f"{needed} (кадров {len(driving)}, шаг {stride}). Обычная причина "
            f"— поза не нашлась на части кадров: сверять надо с driving-"
            f"кадрами, а не с условиями ControlNet (на рисунке скелета "
            f"детектор поз не находит ничего).")}

    donor = float(np.median(mag_d))
    if donor < min_donor_step:
        return {**empty, "steps": len(mag_d), "donor_step": round(donor, 4),
                "note": (
            f"движение НЕ ИЗМЕРЕНО: сам донор почти не движется "
            f"({donor:.4f} длины торса за {stride} кадра при пороге "
            f"{min_donor_step}). Воспроизводить нечего, и отношение величин "
            f"здесь считалось бы от шума детектора.")}

    amplitude = round(float(np.median(mag_g)) / donor, 4)
    direction = round(sum(dots) / sum(prods), 4) if prods else None
    base = {"amplitude": amplitude, "direction": direction,
            "steps": len(mag_d), "frames": len(driving),
            "donor_step": round(donor, 4)}

    # ПОРЯДОК ВАЖЕН. Сначала величина: у ТОЧНО замороженного клипа направление
    # не определено вовсе (нулевые векторы), и проверка направления первой
    # закрыла бы его исходом «не смогли измерить» — той самой ошибкой, из-за
    # которой клип из 16 одинаковых кадров однажды не провалил ни одной оси.
    if amplitude < amplitude_min:
        return {**base, "verdict": "different", "follows": False,
                "check": "action_amplitude", "note": (
            f"движение НЕ ТО: клип воспроизвёл {amplitude:.2f} движения донора "
            f"при пороге {amplitude_min}"
            + (" — кадры практически неподвижны, человек стоит или сидит там, "
               "где донор движется." if amplitude < 0.05 else
               " — движется в такт, но вяло."))}
    if direction is None:
        return {**empty, **base, "verdict": "not_measurable", "note": (
            "движение НЕ ИЗМЕРЕНО: ни на одном шаге обе траектории не имели "
            "ненулевого смещения.")}
    if direction < direction_min:
        return {**base, "verdict": "different", "follows": False,
                "check": "action_direction", "note": (
            f"движение НЕ ТО: направления совпадают на {direction:+.2f} при "
            f"пороге {direction_min} — клип движется, но не туда, куда донор "
            f"(перепутанный порядок, обращённое время или вовсе другое "
            f"действие). Амплитуда при этом {amplitude:.2f}, то есть «шевелится"
            f" достаточно» — на неё и опирался бы обманутый гейт.")}
    return {**base, "verdict": "same", "follows": True, "check": None,
            "note": (f"движение ТО ЖЕ: направление {direction:+.2f} (порог "
                     f"{direction_min}), доля движения донора {amplitude:.2f} "
                     f"(порог {amplitude_min}) по {len(mag_d)} шагам.")}


def action_match(generated_frames, driving_frames, **kw) -> dict:
    """То же сравнение, но по путям к картинкам: читает позы и сравнивает.

    Требует MediaPipe (`pose.landmarks`). Вся арифметика вынесена в
    `compare_trajectories` именно для того, чтобы её можно было проверить без
    модели — как в `pose.py`.
    """
    gen = [str(p) for p in generated_frames]
    drv = [str(p) for p in driving_frames]
    if len(gen) != len(drv):
        raise ValueError(
            f"action_match: {len(gen)} кадр(ов) клипа против {len(drv)} "
            f"driving-кадров. Нужна карта один к одному — driving_frames_for().")
    res = compare_trajectories([landmarks(p) for p in drv],
                               [landmarks(p) for p in gen], **kw)
    return {**res, "generated": gen, "driving": drv}


def driving_frames_for(manifest, conditions=None, *, root=None) -> list:
    """Достать driving-кадры из manifest.json — карту, а не догадку.

    `manifest` — путь к manifest.json или уже разобранный словарь. `conditions`
    — условия, которые реально пошли в генерацию (пути или их имена); если не
    задать, берутся все ключи карты по порядку.

    Пути в манифесте записаны относительно корня репозитория, поэтому `root`
    (по умолчанию — папка на два уровня выше манифеста) достраивает их до
    существующих файлов.
    """
    if isinstance(manifest, (str, Path)):
        mpath = Path(manifest)
        data = json.loads(mpath.read_text())
        base = Path(root) if root else mpath.parent.parent.parent
    else:
        data = manifest
        base = Path(root) if root else Path(".")
    table = data.get("driving_frames")
    if not table:
        raise ValueError(
            "в манифесте нет поля driving_frames: карты условий на "
            "driving-кадры нет, а без неё сравнивать траектории не с чем. "
            "Условия, собранные без этой карты, для оси движения непригодны.")
    keys = ([Path(str(c)).stem for c in conditions] if conditions is not None
            else sorted(table))
    missing = [k for k in keys if k not in table]
    if missing:
        raise ValueError(
            f"в карте driving_frames нет условий {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''} — манифест не от этого "
            f"набора условий.")
    return [str(base / table[k]) if not Path(table[k]).is_absolute()
            else table[k] for k in keys]


def render(res: dict) -> str:
    """Одна читаемая строка отчёта."""
    v = {"same": "ДВИЖЕНИЕ ТО ЖЕ", "different": "ДВИЖЕНИЕ НЕ ТО",
         "not_measurable": "НЕ СМОГЛИ ИЗМЕРИТЬ"}[res["verdict"]]
    amp = "-" if res.get("amplitude") is None else f"{res['amplitude']:.2f}"
    dr = "-" if res.get("direction") is None else f"{res['direction']:+.2f}"
    return (f"{v}  amplitude={amp}  direction={dr}  "
            f"шагов={res.get('steps', 0)}\n  {res.get('note', '')}")


# --------------------------------------------------------------------------
# Контрольные случаи на НАСТОЯЩИХ driving-кадрах.
# Числа в докстринге получены отсюда: python3 -m ball_reel.action --controls
# --------------------------------------------------------------------------

def _shuffled(seq, seed: int = 0):
    import random

    out = list(seq)
    random.Random(seed).shuffle(out)
    return out


def _damped(seq, factor: float):
    """Та же траектория, но с амплитудой в `factor` раз: «в такт, но вяло»."""
    import numpy as np

    base = seq[0]
    out = []
    for p in seq:
        out.append({j: (base[j][0] + factor * (p[j][0] - base[j][0]),
                        base[j][1] + factor * (p[j][1] - base[j][1]), p[j][2])
                    for j in p})
    return out


def _jittered(seq, sd: float, seed: int = 0):
    """Та же траектория плюс дрожь детектора: модель «сидит, но не статуя»."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return [{j: (p[j][0] + float(rng.normal(0, sd)),
                 p[j][1] + float(rng.normal(0, sd)), p[j][2]) for j in p}
            for p in seq]


def controls(driving_dir: str | Path, *, window: int = 16) -> list:
    """Пять контрольных случаев из настоящих кадров + три поясняющих.

    Возвращает список (имя, результат). Дрожь и приглушение задаются в ДОЛЯХ
    ТОРСА уже после нормировки — поэтому строятся не из сырых landmark'ов, а
    поверх них, чтобы `compare_trajectories` видела ровно тот же вход.
    """
    files = sorted(Path(driving_dir).glob("*.jpg")) or sorted(
        Path(driving_dir).glob("*.png"))
    poses = [landmarks(p) for p in files]
    n = len(poses)
    if n < window * 2:
        raise ValueError(f"нужно хотя бы {window * 2} driving-кадров, есть {n}")
    head = poses[:window]
    tail = poses[n - window:]
    out = [
        ("1 донор против себя", head),
        ("2 перетасованная копия", _shuffled(head, 0)),
        ("3 замороженный кадр x%d" % window, [head[0]] * window),
        ("4 обращённая во времени копия", head[::-1]),
        (f"5 окно {n - window}..{n - 1} против 0..{window - 1}", tail),
        ("+ приглушённое x0.5 (в такт, вяло)", _damped(head, 0.5)),
        ("+ замороженный с дрожью 0.02", _jittered([head[0]] * window, 0.02)),
        ("+ сдвиг на кадр (то же движение, позже)", poses[1:window + 1]),
    ]
    return [(name, compare_trajectories(head, seq)) for name, seq in out]


def main(argv: list) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="ball_reel.action",
        description="выполняет ли клип то же движение, что driving-видео")
    ap.add_argument("--controls", metavar="DRIVING_DIR",
                    help="прогнать контрольные случаи на настоящих кадрах")
    ap.add_argument("--frames", nargs="*", default=None,
                    help="кадры сгенерированного клипа")
    ap.add_argument("--manifest", help="kit/conditions/manifest.json")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="условия, по которым сгенерирован клип (по порядку)")
    args = ap.parse_args(argv)

    if args.controls:
        rows = controls(args.controls)
        print(f"{'случай':<40}{'amplitude':>10}{'direction':>11}  вердикт")
        print("-" * 78)
        for name, r in rows:
            amp = "-" if r["amplitude"] is None else f"{r['amplitude']:.3f}"
            dr = "-" if r["direction"] is None else f"{r['direction']:+.3f}"
            print(f"{name:<40}{amp:>10}{dr:>11}  {r['verdict']}")
        return 0

    if not args.frames or not args.manifest:
        ap.error("нужны --frames и --manifest (или --controls)")
    drv = driving_frames_for(args.manifest, args.conditions)
    if args.conditions is None:
        drv = drv[:len(args.frames)]
    print(render(action_match(args.frames, drv)))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
