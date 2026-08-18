"""Разметка ПРЕДМЕТОВ и её место в маске персонажа.

ЗАЧЕМ ЭТО ВООБЩЕ ЕСТЬ. Сегментатор тела знает шесть классов, и пятый из них —
`others`, который ВХОДИТ в персонажа (`fork_mask.PERSON_CLASSES`). Значит
предмет в руке с большой вероятностью уже размечен как часть человека, и модель
получает право его перерисовать — не по нашему решению, а по умолчанию чужой
модели. Здесь это решение становится НАШИМ и явным: каждый предмет либо
сохраняется, либо отдаётся модели, и то и другое печатается числом.

ГДЕ ЭТО ЖИВЁТ В ПРОДУКТЕ. На сборке ТЕМПЛЕЙТА, один раз, руками оператора — и
никогда на клиенте. Драйвинг у всех клиентов один, поэтому маска предмета
снимается однажды и переиспользуется всеми. Требование «без ручных шагов»
относится к пути клиента; LoRA мы тоже обучаем руками, и это никого не смущает.

ПОЧЕМУ БЕЗ НОДЫ `PointsEditor`. Её исходник прочитан (kjnodes, `curve_nodes.py`,
тело sha256 27024564bd835c64..., класс на строке 1403). Собственная шапка ноды
говорит: «# WORK IN PROGRESS / Do not count on this as part of your workflow
yet, probably contains lots of bugs and stability is not guaranteed!!» — это
слова её автора, не мои. А главное, весь её ВЫЧИСЛИТЕЛЬНЫЙ вклад — разбор
JSON-строки: `coordinates` приходит текстом, `json.loads`, округление,
необязательная нормировка, сборка bbox-маски. Редактор живёт в браузере, нода
лишь принимает результат. Значит нам нужен не узел, а ЕГО КОНТРАКТ: файл с
точками. Тогда ручная разметка остаётся, а путь клиента не тянет ни
экспериментальную ноду, ни четвёртый лицензионный набор.

ТРИ ИСХОДА, КАК ВЕЗДЕ (Р1). И третий здесь не формальность: предмет тоньше
блока блокификации СОХРАНИТЬ НЕЛЬЗЯ, и честный ответ на такую разметку —
«не смогли», а не молчаливое «сохранили».
"""

from __future__ import annotations

import json
from pathlib import Path

from . import fork_mask
from .fork_identity import FAIL, PASS, UNMEASURED

#: Что делать с размеченным предметом. Два намерения, и они противоположны:
#: `KEEP` вычитает предмет из маски — модель его не трогает; `REPLACE`
#: добавляет — модель рисует там заново. Умолчания нет НАРОЧНО: намерение
#: обязано быть названо, потому что цена ошибки разная в обе стороны.
KEEP = "сохранить"
REPLACE = "заменить"
ROLES = (KEEP, REPLACE)

#: Формат файла разметки — тот же, что отдаёт `PointsEditor`, чтобы разметку
#: можно было снять и его редактором тоже. Проверено по исходнику: точки едут
#: списком словарей `{"x": int, "y": int}`, бокс — четвёркой в порядке `xyxy`.
POINTS_SCHEMA = "fork-props/1"

#: Наименьший предмет, который МОЖНО обещать сохранить.
#:
#: ИЗМЕРЕНО НЕ ЗДЕСЬ, А ВЗЯТО ИЗ УЖЕ ЗАМЕРЕННОГО (Е1): маска рубится на блоки
#: `fork_mask.BLOCK`, и блок считается занятым, если в нём есть хоть один
#: пиксель персонажа. Значит вычесть предмет из маски можно только целыми
#: блоками: предмет уже блока затягивается обратно соседними пикселями руки.
#: Собственный предел модели вдвое мельче (`MODEL_TOKEN_PX`), но он здесь ни при
#: чём — ограничение наше, и оно грубее.
MIN_KEEPABLE_PX = fork_mask.BLOCK

#: Доля кадра, выше которой маска перестаёт что-либо гарантировать. ВЫБРАНО по
#: замеру потока C на живом кадре: силуэт как есть даёт 65.6% после блоков,
#: расширение на 64 px — 89.6%, и там вне маски остаётся десятая часть кадра.
#: Планка стоит между ними и ближе к верхней: она ловит не расширение рук, ради
#: которого замер делался, а добавление предметов поверх него.
GUARDED_MIN_SHARE = 0.15


def refuse_per_client(marking: dict) -> None:
    """Уронить прогон, если разметку пытаются снять на клиента.

    Не совет и не предупреждение. Разметка, привязанная к клиенту, означает
    оператора на каждый заказ — то есть отмену продуктового обещания, ради
    которого весь форк и делается. Один раз мы уже получили дефект того же
    рода: медоид, поданный как якорь личности, читался как успех, пока не был
    запрещён кодом. Здесь то же лекарство.
    """
    if marking.get("client") or marking.get("client_photo"):
        raise ValueError(
            "разметка помечена клиентом — она снимается ОДИН РАЗ на темплейт "
            "и переиспользуется всеми клиентами. Разметка на клиента означает "
            "оператора на каждый заказ")
    if not marking.get("template"):
        raise ValueError(
            "разметка без имени темплейта: непонятно, к какому драйвингу она "
            "относится, а драйвинг у темплейта один и тот же для всех")


def load_marking(path: str | Path) -> dict:
    """Файл разметки. Схема проверяется, а не предполагается."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"нет файла разметки {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != POINTS_SCHEMA:
        raise ValueError(
            f"схема {data.get('schema')!r} вместо {POINTS_SCHEMA!r}: файл "
            f"разметки не тот или устарел")
    refuse_per_client(data)
    # Роли проверяются по ТОЙ ЖЕ схеме, по которой файл потом читается —
    # покадрово. Сначала здесь стоял обход `data["objects"]`, ключа, которого в
    # формате нет: проверка обходила ноль записей и всегда возвращала «чисто»
    # (Р2 ровно про это). Поймано тестом на отсутствие умолчания у роли.
    seen = 0
    for frame, objects in (data.get("frames") or {}).items():
        for i, obj in enumerate(objects):
            seen += 1
            if obj.get("role") not in ROLES:
                raise ValueError(
                    f"кадр {frame}, предмет #{i} ({obj.get('name', '?')}): "
                    f"роль {obj.get('role')!r} не из {ROLES}. Умолчания нет: "
                    f"намерение обязано быть названо")
            if "bbox" not in obj:
                raise ValueError(
                    f"кадр {frame}, предмет #{i} ({obj.get('name', '?')}): "
                    f"нет поля bbox — размечать нечего")
    data["_objects_checked"] = seen
    return data


def bbox_mask(bbox, width: int, height: int):
    """Прямоугольник в маску. Порядок `xyxy` — как у прочитанной ноды."""
    import numpy as np

    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
    m = np.zeros((height, width), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def keepable(mask) -> dict:
    """Можно ли ОБЕЩАТЬ сохранить этот предмет. Три исхода, и третий главный.

    Предмет уже блока блокификации сохранить нельзя ничем: соседние пиксели
    руки затянут его блок в маску обратно. Сказать про такую разметку
    «сохранено» — соврать ровно там, где оператор потратил время, поэтому здесь
    `не смогли` с числами, а не `годно`.
    """
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {"outcome": UNMEASURED, "width_px": 0, "height_px": 0,
                "note": "разметка пуста — сохранять нечего"}
    ys, xs = np.where(m)
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    thin = min(w, h)
    if thin < MIN_KEEPABLE_PX:
        return {
            "outcome": UNMEASURED, "width_px": w, "height_px": h,
            "note": (f"предмет {w}x{h} px, узкая сторона {thin} меньше блока "
                     f"{MIN_KEEPABLE_PX}: вычесть его из маски нельзя — "
                     f"блокификация затянет его обратно соседними пикселями. "
                     f"Сохранить не обещаем, и это не отказ прибора, а предел "
                     f"тракта"),
        }
    return {"outcome": PASS, "width_px": w, "height_px": h,
            "note": f"предмет {w}x{h} px, узкая сторона {thin} — целые блоки "
                    f"есть, вычитание доедет"}


def apply(person, objects, *, block: int | None = None) -> dict:
    """Свести маску персонажа с размеченными предметами.

    Возвращает маску И ЧИСЛА: сколько площади ушло модели, сколько отобрано
    обратно, сколько кадра осталось под охраной. Голая маска здесь была бы
    дефектом Е3 — «получилось» без указания, ЧЕГО получилось.
    """
    import numpy as np

    block = fork_mask.BLOCK if block is None else block
    base = np.asarray(person, dtype=bool)
    total = base.size
    out = base.copy()
    rows = []

    for obj in objects:
        role = obj.get("role")
        if role not in ROLES:
            raise ValueError(f"роль {role!r} не из {ROLES}")
        m = np.asarray(obj["mask"], dtype=bool)
        if m.shape != base.shape:
            raise ValueError(
                f"маска предмета {obj.get('name', '?')} размера {m.shape} "
                f"против маски персонажа {base.shape}")
        row = {"name": obj.get("name", "?"), "role": role,
               "share": round(float(m.sum()) / total, 6)}
        if role == KEEP:
            verdict = keepable(m)
            row.update(verdict)
            # Вычитаем ВСЕГДА, даже когда обещать сохранение нельзя: вычитание
            # безвредно и иногда сработает. Врать в отчёте нельзя — а вычесть
            # можно. Исход остаётся `не смогли`, и это видно.
            out &= ~m
        else:
            out |= m
            row["outcome"] = PASS
            row["note"] = "отдано модели: она нарисует здесь заново"
        rows.append(row)

    blocked = fork_mask.blockify(out, block=block)
    blocked = np.asarray(blocked, dtype=bool)
    guarded = 1.0 - float(blocked.sum()) / total

    problems = [r for r in rows if r["outcome"] == FAIL]
    unmeasured = [r for r in rows if r["outcome"] == UNMEASURED]
    if guarded < GUARDED_MIN_SHARE:
        # Строка кладётся В СПИСОК ПРЕДМЕТОВ, а не только в счётчик провалов:
        # сначала причина провала никуда не доезжала — отчёт печатал «не годно»
        # и ни слова о том, почему. Вердикт без причины заставляет следующего
        # человека воспроизводить прогон, чтобы узнать то, что уже было
        # известно приборам.
        rows.append({
            "name": "охраняемая площадь", "role": "—", "outcome": FAIL,
            "share": round(guarded, 6),
            "note": (f"вне маски осталось {guarded:.1%} кадра при планке "
                     f"{GUARDED_MIN_SHARE:.0%} — ось протечки больше почти "
                     f"ничего не сторожит")})
        problems.append(rows[-1])

    outcome = (FAIL if problems else UNMEASURED if unmeasured else PASS)
    return {
        "mask": blocked,
        "objects": rows,
        "problems": problems,
        "outcome": outcome,
        "share_before": round(float(fork_mask.blockify(base, block=block)
                                    .astype(bool).sum()) / total, 4),
        "share_after": round(float(blocked.sum()) / total, 4),
        "guarded_share": round(guarded, 4),
        "kept": sum(1 for r in rows if r["role"] == KEEP),
        "replaced": sum(1 for r in rows if r["role"] == REPLACE),
        "guard_problem": next((r["note"] for r in problems
                               if r["name"] == "охраняемая площадь"), None),
        "unmeasured": len(unmeasured),
        "note": (f"предметов {len(rows)}: сохраняем {sum(1 for r in rows if r['role'] == KEEP)}, "
                 f"отдаём модели {sum(1 for r in rows if r['role'] == REPLACE)}, "
                 f"не смогли обещать {len(unmeasured)}. Маска была "
                 f"{round(float(fork_mask.blockify(base, block=block).astype(bool).sum()) / total * 100, 1)}% кадра, "
                 f"стала {round(float(blocked.sum()) / total * 100, 1)}%, "
                 f"под охраной осталось {guarded:.1%}"
                 + ("" if guarded >= GUARDED_MIN_SHARE else
                    f" — при планке {GUARDED_MIN_SHARE:.0%} ось протечки "
                    f"больше почти ничего не сторожит")),
    }


def _as_mask(item):
    """Маска массивом ИЛИ путём к файлу.

    Принимать оба вида нужно потому, что `fork_mask.sequence` кладёт маски на
    диск и массивов не возвращает — стык, на котором первый вариант этого
    модуля падал в `KeyError` и печатал провал вместо работы. Загрузка здесь, в
    одном месте: разбросанный по вызовам `Image.open` разъедется по порогу
    бинаризации.
    """
    import numpy as np

    if isinstance(item, (str, Path)):
        from PIL import Image

        with Image.open(item) as im:
            return np.asarray(im.convert("L")) > 127
    return np.asarray(item, dtype=bool)


def sequence(marking_path: str | Path, person_masks, *,
             block: int | None = None) -> dict:
    """Разметка по всей последовательности кадров.

    Числами (Е3): на скольких кадрах разметка применилась, на скольких предмет
    оказался неудерживаемым, на скольких её просто нет. Агрегатный флаг здесь
    читался бы как полная работа при одном удавшемся кадре из ста.
    """
    import numpy as np

    marking = load_marking(marking_path)
    per_frame = {int(k): v for k, v in (marking.get("frames") or {}).items()}
    applied, skipped, weak = [], [], []
    masks = []

    for i, person in enumerate(person_masks):
        base = _as_mask(person)
        spec = per_frame.get(i)
        if spec is None:
            masks.append(np.asarray(
                fork_mask.blockify(base, block=block or fork_mask.BLOCK),
                dtype=bool))
            skipped.append(i)
            continue
        objects = []
        for obj in spec:
            objects.append({
                "name": obj.get("name", "?"), "role": obj.get("role"),
                "mask": bbox_mask(obj["bbox"], base.shape[1], base.shape[0])})
        got = apply(base, objects, block=block)
        masks.append(got["mask"])
        (weak if got["outcome"] == UNMEASURED else applied).append(i)

    outcome = (UNMEASURED if not applied and not weak else
               UNMEASURED if weak else PASS)
    return {
        "masks": masks, "applied": len(applied), "weak": len(weak),
        "no_marking": len(skipped), "frames": len(masks),
        "outcome": outcome,
        "template": marking.get("template"),
        "note": (f"кадров {len(masks)}: разметка применена на {len(applied)}, "
                 f"предмет неудерживаем на {len(weak)}, разметки нет на "
                 f"{len(skipped)}. Разметка темплейта "
                 f"{marking.get('template')!r} — она одна на всех клиентов"),
    }
