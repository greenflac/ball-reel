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

#: Роли, которые ПРАВЯТ МАСКУ. Только эти две доходят до `apply`: остальные
#: ничего в маске не двигают, и пускать их туда значило бы дать им тихо
#: подействовать не тем способом, каким они объявлены.
MASK_ROLES = (KEEP, REPLACE)

#: Протагонист — НЕ предмет и не операция над маской. Это ВЫБОР: кого из
#: нескольких людей в кадре мы ведём. Роль отдельная от `KEEP`/`REPLACE`
#: нарочно: вычесть протагониста из маски или отдать его модели целиком — обе
#: операции бессмысленны, а роль, допускающая бессмыслицу, однажды будет
#: поставлена по ошибке.
#:
#: ЗАЧЕМ ЭТО ВООБЩЕ НУЖНО. `kijai/ComfyUI-WanAnimatePreprocess : nodes.py:120-124`
#: берёт на КАЖДОМ кадре первую рамку, какую вернул детектор:
#: `bboxes.append(detector(...)[0][0]["bbox"])`. Ни выбора, ни трекинга. При
#: двух-трёх людях порядок детекций между кадрами меняется, и скелет
#: перепрыгивает с человека на человека посреди клипа — поэтому вендор и пишет
#: «single-person videos ONLY». Здесь протагонист называется ОДИН РАЗ на сборке
#: темплейта и ведётся сквозь кадры теми же ключевыми кадрами, что и предметы.
PROTAGONIST = "протагонист"

#: Второстепенный человек — часть ОКРУЖЕНИЯ, то есть заведомо вне маски. Его
#: область объявляется не ради маски, а ради АДРЕСА: без имени протечка
#: печатается одним числом по всему кадру, и «модель тронула прохожего» не
#: отличается от «модель перекрасила стену».
BYSTANDER = "второстепенный"

#: Роли, у которых есть только рамка и тождество между кадрами, но нет
#: операции над маской.
TRACKED_ROLES = (PROTAGONIST, BYSTANDER)

ROLES = MASK_ROLES + TRACKED_ROLES

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

#: ТРИ ИСХОДА РАМКИ НА КАДРЕ — не служебные строки, а ровно то, ради чего
#: заведены ключевые кадры. `размечена` — оператор поставил её руками;
#: `выведена` — она получена интерполяцией между двумя ключевыми;
#: `вне размеченного диапазона` — рамки НЕТ, и выдумать её нельзя.
FROM_KEYFRAME = "размечена"
FROM_INTERPOLATION = "выведена"
OUT_OF_RANGE = "вне размеченного диапазона"
BOX_SOURCES = (FROM_KEYFRAME, FROM_INTERPOLATION, OUT_OF_RANGE)

#: Смещение центра рамки за один кадр, выше которого мы считаем, что ведём
#: РАЗНЫХ людей (или разные предметы), а не одного.
#:
#: ВЫБРАНО (мной, 2026-08-18, из ничего измеренного). Мера безразмерная —
#: доля средней стороны самой рамки, — потому что абсолютные пиксели зависят
#: от разрешения, а прыжок «на полкорпуса» не зависит. 0.5 = центр съехал на
#: половину собственного размера объекта за 1/30 секунды. Число обязано быть
#: заменено первым же замером на настоящем многолюдном драйвинге; до тех пор
#: оно ловит подмену человека, а не мерит скорость.
JUMP_CENTER_SHARE = 0.5

#: Во сколько раз за один кадр может измениться площадь рамки, прежде чем это
#: перестанет быть приближением/удалением и станет другим объектом.
#: ВЫБРАНО там же и по той же причине: удвоение площади за кадр — это скачок
#: масштаба в 1.41 раза по стороне, чего у одного тела за 1/30 с не бывает.
JUMP_AREA_RATIO = 2.0

#: Какая доля рамки второстепенного человека может перекрываться с рамкой
#: протагониста, прежде чем режиссёрское правило считается нарушенным.
#:
#: ВЫБРАНО. Правило владельца — «второстепенные на фоне, без контакта с
#: протагонистом и без прохода перед ним» — это требование к СЪЁМКЕ, кодом оно
#: не решается. Здесь оно только ОТРАЖЕНО в проверке: рамки прямоугольные и
#: захватывают воздух, поэтому ноль был бы ложной тревогой на каждом касании
#: краёв, а 0.05 отделяет касание от прохода перед протагонистом.
BYSTANDER_OVERLAP_MAX = 0.05

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
    roles_by_name: dict[str, str] = {}
    for frame, objects in (data.get("frames") or {}).items():
        try:
            int(frame)
        except (TypeError, ValueError):
            raise ValueError(
                f"ключ кадра {frame!r} — не номер кадра. Ключевые кадры "
                f"нумеруются, иначе между ними нечего интерполировать")
        for i, obj in enumerate(objects):
            seen += 1
            name = obj.get("name")
            if obj.get("role") not in ROLES:
                raise ValueError(
                    f"кадр {frame}, предмет #{i} ({name or '?'}): "
                    f"роль {obj.get('role')!r} не из {ROLES}. Умолчания нет: "
                    f"намерение обязано быть названо")
            if "bbox" not in obj:
                raise ValueError(
                    f"кадр {frame}, предмет #{i} ({name or '?'}): "
                    f"нет поля bbox — размечать нечего")
            if len(tuple(obj["bbox"])) != 4:
                raise ValueError(
                    f"кадр {frame}, предмет #{i} ({name or '?'}): bbox из "
                    f"{len(tuple(obj['bbox']))} чисел вместо четырёх (xyxy)")
            # ИМЯ — ЭТО ТОЖДЕСТВО, а не подпись для отчёта. По нему рамка с
            # одного ключевого кадра склеивается с рамкой на другом; без имени
            # склеивать нечего, и интерполяция соединит мяч с прохожим.
            if not name:
                raise ValueError(
                    f"кадр {frame}, предмет #{i}: нет имени. Имя — это "
                    f"тождество между ключевыми кадрами, по нему рамки и "
                    f"склеиваются; безымянную рамку не с чем связать")
            if roles_by_name.setdefault(name, obj["role"]) != obj["role"]:
                raise ValueError(
                    f"{name!r} размечен и как {roles_by_name[name]!r}, и как "
                    f"{obj['role']!r}: одно имя — один объект с одной ролью, "
                    f"иначе интерполяция ведёт то предмет, то человека")

    leads = sorted(n for n, r in roles_by_name.items() if r == PROTAGONIST)
    if len(leads) > 1:
        # Р1/Е2: выбор обязан быть детерминированным. Два протагониста — это
        # ровно тот дефект вендора, от которого мы лечимся: кого вести, решал
        # бы порядок обхода словаря.
        raise ValueError(
            f"протагонистов размечено {len(leads)} ({', '.join(leads)}): вести "
            f"можно ровно одного, иначе выбор снова достаётся порядку "
            f"перебора — тому самому дефекту, ради которого роль и заведена")

    data["_objects_checked"] = seen
    data["_names"] = roles_by_name
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
        if role not in MASK_ROLES:
            # Именно MASK_ROLES, а не ROLES: протагонист и второстепенный —
            # это выбор и адрес, а не операция над маской. Пустив их сюда, мы
            # дали бы им подействовать не тем способом, каким они объявлены.
            raise ValueError(
                f"роль {role!r} не из {MASK_ROLES}: маску правят только эти "
                f"две роли, {TRACKED_ROLES} ничего в ней не двигают")
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




# ─────────────────────────────────────────────────────────────────────────────
# КЛЮЧЕВЫЕ КАДРЫ: рамка ВЫВОДИТСЯ, а не размечается триста раз
#
# Прежняя версия читала рамку на КАЖДЫЙ кадр (`per_frame.get(i)`). На шаблоне
# в 300 кадров это 300 рамок руками, то есть механизм, которым никто не
# воспользуется. Здесь оператор ставит КЛЮЧЕВЫЕ кадры, между ними рамка
# интерполируется линейно, а ЗА ПРЕДЕЛАМИ размеченного диапазона честно
# получается «не смогли»: экстраполяция — это выдуманная рамка, а выдуманная
# рамка вычтет из маски не тот кусок и сделает это молча.
# ─────────────────────────────────────────────────────────────────────────────

def tracks(marking: dict) -> dict:
    """Разметка, перевёрнутая с «кадр → объекты» на «объект → ключевые кадры».

    Ключ — ИМЯ, потому что имя здесь и есть тождество: по нему рамка с одного
    ключевого кадра склеивается с рамкой на другом. Роль хранится при объекте,
    а не при кадре, — она у объекта одна на весь клип (проверено в
    `load_marking`).
    """
    out: dict = {}
    for frame, objects in (marking.get("frames") or {}).items():
        f = int(frame)
        for obj in objects:
            name = obj.get("name")
            if not name:
                raise ValueError(
                    f"кадр {frame}: рамка без имени — склеивать её между "
                    f"ключевыми кадрами не с чем")
            t = out.setdefault(name, {"name": name, "role": obj.get("role"),
                                      "keys": {}})
            t["keys"][f] = [float(v) for v in obj["bbox"]]
    return out


def _center_size(bbox):
    """Центр, стороны и площадь рамки. Одно место, где это считается (Е1)."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = abs(x2 - x1), abs(y2 - y1)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h, w * h


def box_at(track: dict, frame: int) -> dict:
    """Рамка объекта на кадре. ТРИ ИСХОДА, и третий — весь смысл функции.

        размечена                   оператор поставил рамку руками на этом кадре
        выведена                    интерполяция между двумя ключевыми
        вне размеченного диапазона  рамки НЕТ — и её нельзя придумать

    Экстраполяция запрещена не из осторожности. За последним ключевым кадром
    объект мог уйти из кадра, остановиться или закрыться рукой, и продолженная
    рамка вычтет из маски пустое место, оставив предмет внутри маски. Это
    провал, который печатается как успех, — та самая форма дефекта, из-за
    которой в проекте вообще заведён третий исход.
    """
    keys = (track or {}).get("keys") or {}
    if not keys:
        return {"outcome": UNMEASURED, "bbox": None, "source": OUT_OF_RANGE,
                "keyframes": 0,
                "note": f"{track.get('name', '?')!r}: ключевых кадров нет — "
                        f"выводить нечего и не из чего"}
    lo_bound, hi_bound = min(keys), max(keys)
    if frame in keys:
        return {"outcome": PASS, "bbox": list(keys[frame]),
                "source": FROM_KEYFRAME, "keyframes": len(keys),
                "note": f"кадр {frame}: рамка размечена руками"}
    lower = [f for f in keys if f < frame]
    upper = [f for f in keys if f > frame]
    if not lower or not upper:
        return {
            "outcome": UNMEASURED, "bbox": None, "source": OUT_OF_RANGE,
            "keyframes": len(keys), "range": (lo_bound, hi_bound),
            "note": (f"кадр {frame} вне размеченного диапазона "
                     f"{lo_bound}..{hi_bound}: экстраполировать НЕЛЬЗЯ — за "
                     f"последним ключевым кадром рамки не существует, а "
                     f"продолженная вычтет из маски пустое место и оставит "
                     f"объект внутри маски"),
        }
    lo, hi = max(lower), min(upper)
    t = (frame - lo) / float(hi - lo)
    a, b = keys[lo], keys[hi]
    bbox = [a[k] + (b[k] - a[k]) * t for k in range(4)]
    return {"outcome": PASS, "bbox": bbox, "source": FROM_INTERPOLATION,
            "keyframes": len(keys), "between": (lo, hi),
            "note": (f"кадр {frame}: рамка выведена между ключевыми {lo} и "
                     f"{hi} (доля {t:.3f})")}


# ─────────────────────────────────────────────────────────────────────────────
# СТОРОЖ «РАМКА ПРЫГНУЛА»
#
# Это проверка ТОЖДЕСТВА, а не плавности. Если между соседними кадрами центр
# уехал на полразмера объекта или площадь изменилась вдвое, то самое вероятное
# объяснение — мы ведём ДРУГОГО человека (или другой предмет), а не того же
# самого. Ровно это и происходит у вендора: `detector(...)[0][0]["bbox"]`
# берёт первую детекцию, порядок детекций между кадрами меняется, и рамка
# перепрыгивает. Здесь прыжок ловится числом.
# ─────────────────────────────────────────────────────────────────────────────

def jump_verdict(boxes, *, center_share: float | None = None,
                 area_ratio: float | None = None) -> dict:
    """Прыгала ли рамка. На вход — пары `(кадр, bbox|None)`.

    Р2: печатаются ТРИ числа — сколько пар проверено, сколько нарушений и
    сколько пар проверить не смогли. Ноль нарушений при нуле проверенных пар
    успехом не считается и отдаёт «не смогли».
    """
    import math

    center_share = (JUMP_CENTER_SHARE if center_share is None
                    else center_share)
    area_ratio = JUMP_AREA_RATIO if area_ratio is None else area_ratio

    checked, gaps, jumps = 0, 0, []
    prev = None
    for frame, bbox in boxes:
        if bbox is None:
            gaps += 1
            # Через дыру сравнивать нельзя: в ней рамка могла смениться на
            # чужую, и «плавный» переход через пропуск — это не наблюдение.
            prev = None
            continue
        cx, cy, w, h, area = _center_size(bbox)
        if area <= 0:
            gaps += 1
            prev = None
            continue
        if prev is not None:
            pf, pcx, pcy, pw, ph, parea = prev
            dt = frame - pf
            if dt <= 0:
                gaps += 1
                prev = (frame, cx, cy, w, h, area)
                continue
            checked += 1
            side = (pw + ph + w + h) / 4.0
            shift = math.hypot(cx - pcx, cy - pcy) / dt
            rel = shift / side if side > 0 else float("inf")
            grow = (max(area, parea) / min(area, parea)) ** (1.0 / dt)
            if rel > center_share or grow > area_ratio:
                jumps.append({
                    "from": pf, "to": frame,
                    "center_shift_share": round(rel, 4),
                    "area_ratio": round(grow, 4),
                    "note": (f"кадры {pf}→{frame}: центр съехал на "
                             f"{rel:.2f} собственного размера за кадр "
                             f"(планка {center_share}), площадь изменилась в "
                             f"{grow:.2f} раза за кадр (планка {area_ratio})"),
                })
        prev = (frame, cx, cy, w, h, area)

    outcome = (FAIL if jumps else UNMEASURED if checked == 0 else PASS)
    return {
        "outcome": outcome, "checked": checked, "violations": len(jumps),
        "unmeasured": gaps, "jumps": jumps,
        "center_share": center_share, "area_ratio": area_ratio,
        "note": (f"проверено пар соседних кадров {checked}, нарушений "
                 f"{len(jumps)}, не смогли {gaps}."
                 + (" Пар для сравнения не набралось — это НЕ «прыжков нет»."
                    if checked == 0 else
                    f" Рамка прыгала: ведём, судя по всему, разные объекты — "
                    f"{jumps[0]['note']}" if jumps else
                    f" Рамка идёт непрерывно при планках центра "
                    f"{center_share} и площади {area_ratio}.")),
    }


def track_report(track: dict, frames: int) -> dict:
    """Один объект на всей последовательности: откуда взялась каждая рамка."""
    counts = {FROM_KEYFRAME: 0, FROM_INTERPOLATION: 0, OUT_OF_RANGE: 0}
    boxes = []
    for i in range(frames):
        got = box_at(track, i)
        counts[got["source"]] += 1
        boxes.append((i, got["bbox"]))
    jump = jump_verdict(boxes)
    resolved = counts[FROM_KEYFRAME] + counts[FROM_INTERPOLATION]
    outcome = (FAIL if jump["outcome"] == FAIL else
               UNMEASURED if resolved == 0 or counts[OUT_OF_RANGE] else PASS)
    return {
        "name": track.get("name"), "role": track.get("role"),
        "outcome": outcome, "boxes": boxes, "jump": jump,
        "keyframes": len(track.get("keys") or {}),
        "keyed": counts[FROM_KEYFRAME],
        "interpolated": counts[FROM_INTERPOLATION],
        "out_of_range": counts[OUT_OF_RANGE],
        "note": (f"{track.get('name')!r} ({track.get('role')}): кадров "
                 f"{frames} — размечено руками {counts[FROM_KEYFRAME]}, "
                 f"выведено {counts[FROM_INTERPOLATION]}, вне размеченного "
                 f"диапазона {counts[OUT_OF_RANGE]}. {jump['note']}"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ПРОТАГОНИСТ: выбор один раз, тождество — сквозь кадры
# ─────────────────────────────────────────────────────────────────────────────

def protagonist_name(marking: dict) -> str | None:
    """Кого ведём. `None` — никого не назвали, и это не то же, что «одного»."""
    for name, track in tracks(marking).items():
        if track["role"] == PROTAGONIST:
            return name
    return None


def protagonist_track(marking: dict, frames: int) -> dict:
    """Рамка протагониста на всех кадрах плюс проверка тождества.

    Третий исход обязателен и здесь: разметки протагониста нет — это НЕ
    «людей в кадре один». Это «мы не знаем, сколько их», и в многолюдной сцене
    именно так выглядит вендорский дефект, из-за которого скелет прыгает.
    """
    name = protagonist_name(marking)
    if name is None:
        return {
            "outcome": UNMEASURED, "name": None, "boxes": [],
            "checked": 0, "violations": 0,
            "note": (f"протагонист не размечен (роль {PROTAGONIST!r} в файле "
                     f"не встретилась). Если человек в кадре один — это верно "
                     f"и ничего не стоит; если их несколько, детектор возьмёт "
                     f"первую попавшуюся рамку на каждом кадре, и скелет "
                     f"перепрыгнет с человека на человека посреди клипа"),
        }
    report = track_report(tracks(marking)[name], frames)
    return {
        "outcome": report["outcome"], "name": name,
        "boxes": report["boxes"], "jump": report["jump"],
        "checked": report["jump"]["checked"],
        "violations": report["jump"]["violations"],
        "keyed": report["keyed"], "interpolated": report["interpolated"],
        "out_of_range": report["out_of_range"],
        "note": f"протагонист {report['note']}",
    }


def directing_rule(marking: dict, frames: int) -> dict:
    """Режиссёрское правило, отражённое в проверке.

    Правило владельца: второстепенные люди — на фоне, без контакта с
    протагонистом и без прохода перед ним. Кодом это НЕ решается — это
    требование к съёмке. Но наблюдаемо оно вполне: если рамка второстепенного
    накрывает рамку протагониста, значит он прошёл перед ним, и дальше уже
    неважно, что мы решили, — детектор с сегментатором получат двух слипшихся
    людей.
    """
    all_tracks = tracks(marking)
    lead = protagonist_name(marking)
    others = [t for t in all_tracks.values() if t["role"] == BYSTANDER]
    if lead is None or not others:
        missing = ("протагонист не размечен" if lead is None
                   else "второстепенные не размечены")
        return {"outcome": UNMEASURED, "checked": 0, "violations": 0,
                "unmeasured": frames,
                "note": (f"{missing}: правило «второстепенные на фоне» "
                         f"проверить не на чем. Ноль нарушений при нуле "
                         f"проверок — не успех")}

    checked, unmeasured, bad = 0, 0, []
    for i in range(frames):
        lead_box = box_at(all_tracks[lead], i)["bbox"]
        if lead_box is None:
            unmeasured += 1
            continue
        lx1, lx2 = min(lead_box[0], lead_box[2]), max(lead_box[0], lead_box[2])
        ly1, ly2 = min(lead_box[1], lead_box[3]), max(lead_box[1], lead_box[3])
        for track in others:
            box = box_at(track, i)["bbox"]
            if box is None:
                unmeasured += 1
                continue
            checked += 1
            bx1, bx2 = min(box[0], box[2]), max(box[0], box[2])
            by1, by2 = min(box[1], box[3]), max(box[1], box[3])
            inter = (max(0.0, min(lx2, bx2) - max(lx1, bx1))
                     * max(0.0, min(ly2, by2) - max(ly1, by1)))
            area = (bx2 - bx1) * (by2 - by1)
            share = inter / area if area > 0 else 0.0
            if share > BYSTANDER_OVERLAP_MAX:
                bad.append({
                    "frame": i, "name": track["name"],
                    "overlap": round(share, 4),
                    "note": (f"кадр {i}: {track['name']!r} перекрывает "
                             f"протагониста на {share:.1%} своей рамки при "
                             f"планке {BYSTANDER_OVERLAP_MAX:.0%} — это "
                             f"проход перед протагонистом, а он запрещён "
                             f"правилом съёмки")})
    outcome = (FAIL if bad else UNMEASURED if checked == 0 else PASS)
    return {
        "outcome": outcome, "checked": checked, "violations": len(bad),
        "unmeasured": unmeasured, "overlaps": bad,
        "bar": BYSTANDER_OVERLAP_MAX,
        "note": (f"правило «второстепенные на фоне»: проверено пар "
                 f"{checked}, нарушений {len(bad)}, не смогли {unmeasured}."
                 + (f" {bad[0]['note']}" if bad else
                    " Ни одной пары не набралось — не успех."
                    if checked == 0 else
                    f" Никто не проходит перед протагонистом при планке "
                    f"{BYSTANDER_OVERLAP_MAX:.0%}.")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ИМЕНОВАННЫЕ ОБЛАСТИ: чтобы протечка была АДРЕСНОЙ
# ─────────────────────────────────────────────────────────────────────────────

def leak_areas(marking: dict, frame: int, width: int, height: int) -> dict:
    """Области второстепенных людей на кадре — по одной маске на имя.

    Ось протечки (`fork_leak`) меряет всё, что вне маски, одним числом. Пока
    второстепенный человек безымянная часть фона, «модель шевельнула прохожего»
    не отличается от «модель перекрасила стену». Имя превращает число в адрес.
    """
    rows, missing = [], []
    for name, track in tracks(marking).items():
        if track["role"] != BYSTANDER:
            continue
        got = box_at(track, frame)
        if got["bbox"] is None:
            missing.append({"name": name, "source": got["source"],
                            "note": got["note"]})
            continue
        m = bbox_mask(got["bbox"], width, height)
        rows.append({"name": name, "mask": m, "source": got["source"],
                     "share": round(float(m.sum()) / (width * height), 6)})
    outcome = (UNMEASURED if not rows else
               UNMEASURED if missing else PASS)
    return {
        "outcome": outcome, "areas": rows, "missing": missing,
        "named": len(rows), "unmeasured": len(missing), "frame": frame,
        "note": (f"кадр {frame}: именованных областей {len(rows)}, без рамки "
                 f"{len(missing)}."
                 + (" Второстепенные не размечены — протечка останется одним "
                    "числом по всему кадру, без адреса." if not rows
                    and not missing else "")),
    }


def address_leak(output, driving, areas) -> dict:
    """Протечка по каждой именованной области, числом на имя.

    Прибор НЕ СВОЙ (Е1): считает `fork_leak.outside_divergence`, тот же, что
    судит протечку по кадру целиком. Хитрость одна и она объявлена: функция
    меряет там, где маски НЕТ, поэтому область подаётся ей инверсией — тогда
    «вне маски» совпадает с «внутри области».
    """
    import numpy as np

    from . import fork_leak

    rows, blind = [], []
    for area in areas or []:
        m = np.asarray(area["mask"], dtype=bool)
        got = fork_leak.outside_divergence(output, driving, ~m)
        row = {"name": area.get("name", "?"), "mean": got.get("mean"),
               "max": got.get("max"), "share": got.get("outside_share"),
               "note": got.get("note")}
        if got.get("mean") is None:
            row["outcome"] = UNMEASURED
            blind.append(row)
        else:
            row["outcome"] = PASS
            rows.append(row)
    measured = rows
    outcome = (UNMEASURED if not measured else
               UNMEASURED if blind else PASS)
    return {
        "outcome": outcome, "areas": measured + blind,
        "measured": len(measured), "unmeasured": len(blind),
        "note": (f"именованных областей {len(measured) + len(blind)}: "
                 f"измерено {len(measured)}, не смогли {len(blind)}"
                 + ("".join(f"; {r['name']} среднее {r['mean']:.6f}, "
                            f"максимум {r['max']:.6f}" for r in measured))),
    }


def sequence(marking_path: str | Path, person_masks, *,
             block: int | None = None) -> dict:
    """Разметка по всей последовательности кадров.

    Числами (Е3): на скольких кадрах разметка применилась, на скольких предмет
    оказался неудерживаемым, на скольких она ВЫВЕДЕНА между ключевыми и на
    скольких её нет вовсе. Агрегатный флаг здесь читался бы как полная работа
    при одном удавшемся кадре из ста.

    Разница с прежней версией, и она смысловая: «на этом кадре разметки нет»
    больше не одно состояние, а два. Либо в файле вообще нет предметов —
    применять нечего; либо предмет размечен, но кадр лежит ЗА пределами его
    ключевых кадров — и тогда это «не смогли», а не «нечего делать».

    ЧТО НЕ ТОПИТ ВЕРДИКТ И ПОЧЕМУ. Неразмеченный протагонист оставляет вердикт
    кадров как есть: одиночная сцена — обычный и правильный случай, и топить на
    ней каждый прогон значило бы приучить читать «не смогли» как шум. Своё
    «не смогли» протагонист печатает отдельным ключом `protagonist` со своим
    объяснением, и оно доезжает в `note`. А вот ПРЫЖОК рамки вердикт топит: он
    наблюдаем, и он означает, что мы вели разные объекты.
    """
    import numpy as np

    marking = load_marking(marking_path)
    all_tracks = tracks(marking)
    mask_tracks = {n: t for n, t in all_tracks.items()
                   if t["role"] in MASK_ROLES}

    person_masks = list(person_masks)
    frames = len(person_masks)
    applied, weak, broken, outside, skipped = [], [], [], [], []
    sources = {FROM_KEYFRAME: 0, FROM_INTERPOLATION: 0, OUT_OF_RANGE: 0}
    masks = []

    for i, person in enumerate(person_masks):
        base = _as_mask(person)
        objects = []
        for name, track in mask_tracks.items():
            got = box_at(track, i)
            sources[got["source"]] += 1
            if got["bbox"] is None:
                continue
            objects.append({
                "name": name, "role": track["role"], "source": got["source"],
                "mask": bbox_mask(got["bbox"], base.shape[1], base.shape[0])})
        if not objects:
            masks.append(np.asarray(
                fork_mask.blockify(base, block=block or fork_mask.BLOCK),
                dtype=bool))
            (outside if mask_tracks else skipped).append(i)
            continue
        got = apply(base, objects, block=block)
        masks.append(got["mask"])
        (broken if got["outcome"] == FAIL else
         weak if got["outcome"] == UNMEASURED else applied).append(i)

    reports = {n: track_report(t, frames) for n, t in all_tracks.items()}
    jumped = [n for n, r in reports.items() if r["jump"]["outcome"] == FAIL]
    lead = protagonist_track(marking, frames)
    directing = directing_rule(marking, frames)

    outcome = (FAIL if broken or jumped or lead["outcome"] == FAIL
               or directing["outcome"] == FAIL else
               UNMEASURED if weak or outside or skipped or not applied
               else PASS)
    return {
        "masks": masks, "applied": len(applied), "weak": len(weak),
        "failed": len(broken), "outside_range": len(outside),
        "no_marking": len(skipped), "frames": frames,
        "keyed": sources[FROM_KEYFRAME],
        "interpolated": sources[FROM_INTERPOLATION],
        "out_of_range": sources[OUT_OF_RANGE],
        "tracks": reports, "jumped": jumped,
        "protagonist": lead, "directing": directing,
        "outcome": outcome,
        "template": marking.get("template"),
        "note": (f"кадров {frames}: разметка применена на {len(applied)}, "
                 f"предмет неудерживаем на {len(weak)}, вне размеченного "
                 f"диапазона {len(outside)}, разметки нет на {len(skipped)}. "
                 f"Рамок: размечено руками {sources[FROM_KEYFRAME]}, выведено "
                 f"между ключевыми {sources[FROM_INTERPOLATION]}, вывести "
                 f"нельзя {sources[OUT_OF_RANGE]}. "
                 + (f"РАМКА ПРЫГНУЛА у {', '.join(jumped)}: ведём разные "
                    f"объекты, а не один. " if jumped else "")
                 + f"{lead['note'].split('.')[0]}. "
                 + f"Разметка темплейта {marking.get('template')!r} — она "
                   f"одна на всех клиентов"),
    }
