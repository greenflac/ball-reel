"""Поток A форка: ось личности, у которой якорь — СЫРАЯ ФОТОГРАФИЯ, и только.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ, КОГДА `identity_arcface` УЖЕ ЕСТЬ. Прибор тот же,
негодным был не он, а ЯКОРЬ. Замерено на `demo/lora_dataset` и записано в
манифест самим набором:

    против медоида порождённых   медиана 0.2579,  19/21 в баре 0.35
    против сырой фотографии      медиана 0.5067,  0/21  в баре

Одни и те же кадры, один и тот же прибор, противоположные выводы. Первая строка
читается как успех и однажды уже была прочитана как успех; вторая говорит, что
по собственной планке проекта это НЕ ТОТ ЧЕЛОВЕК. Разница только в том, с чем
сравнивают, и якорь на медоид — это сравнение порождённого с порождённым.

Поэтому здесь медоид ЗАПРЕЩЁН КОДОМ, а не соглашением: соглашение уже было, и
оно продержалось до первого удобного случая. `refuse_derived_anchor` роняет
прогон, если якорем подан кадр из того же набора, что судится.

---

ТРИ ЧИСЛА, И ТРЕТЬЕ — НЕГАТИВНЫЙ КОНТРОЛЬ (И5).

    d_raw   до сырой фотографии          — ВЕРДИКТ выносится по нему и только
    d_ref   до канонического референса   — диагностика: что дал референсный шаг
    d_neg   до заведомо чужого человека  — контроль: прибор обязан сказать «нет»

Без d_neg прибор может отдать правдоподобное число, ничего не измеряя: на этом
проекте метрика уже давала 0.3106 и 0.3072 на кадрах, отличавшихся на 37%
пикселей. Контроль стоит одного вызова и отличает «сходство не найдено» от
«прибор молчит».

d_ref НЕ ВЕРДИКТ. Он отвечает на другой вопрос — «референсный шаг приблизил или
нет», — и подменять им d_raw значит вернуться ровно к дефекту медоида.

---

СОСТОЯНИЕ ЗАМЕРОВ НА ДЕНЬ НАПИСАНИЯ. Воспроизведены не все три строки, и это
записано числом, а не сглажено (И6, Ц4). Полностью — `docs/FORK_NUMBERS.md`.

    d_ref (медоид)      ВОСПРОИЗВЕДЕНО ТОЧНО: медиана 0.2579, 19/21 в баре
    d_neg (чужой)       0.7007, 0/21 — направление верное, величина НЕ ТА
    d_raw (сырое фото)  НЕ СМОГЛИ ИЗМЕРИТЬ: фотографии нет в репозитории

Про d_raw важно, что это не забывчивость смены: сырая фотография была
намеренно исключена из набора владельцем продукта, и манифест это прямо
говорит — «Загруженная снята со спины... ЦЕНА: верифицируется человек, каким
его рисует генератор, а не буквально человек с загруженной фотографии».
Число 0.5067 снято на ней тогда, когда она была. Значит ГЛАВНАЯ строка приёмки
этого потока не закрыта, и модуль обязан говорить это вслух, а не молчать.

Про d_neg: 0.96–1.05 снималось на фотографии, которой в дереве тоже нет.
Самый далёкий чужой человек, найденный ПРИБОРОМ среди того, что есть
(66 кадров просмотрено), стоит на 0.7007. Это выше `HARD_DRIFT_MAX` 0.6, то
есть контроль работает — прибор говорит «другой человек», — но полосу 0.96–1.05
он не воспроизводит, и выдавать 0.70 за неё нельзя.
"""

from __future__ import annotations

import json
from pathlib import Path

from .identity_arcface import HARD_DRIFT_MAX, SAME_PERSON_MAX

#: Прибор по умолчанию — ПАРАМЕТР, а не вшитая зависимость. Смена прибора
#: обнуляет все ранее снятые числа, поэтому она обязана быть видимым решением
#: вызывающего, а не следствием правки импорта.
DEFAULT_INSTRUMENT = "identity_arcface"

#: ЛИЦЕНЗИОННЫЙ БЛОКЕР РЕЛИЗА, а не примечание. Веса `buffalo_l`, на которых
#: сняты все числа проекта, идут под non-commercial условиями InsightFace, а
#: продукт коммерческий. Цена замены прибора известна и велика: ПЕРЕСЧЁТ ВСЕХ
#: ПОРОГОВ, потому что 0.35 и 0.6 откалиброваны на этой шкале и на другой не
#: значат ничего. Строка живёт в коде, чтобы попадать в отчёт вместе с числами,
#: а не теряться в документе.
INSTRUMENT_LICENCE = {
    "identity_arcface": ("buffalo_l / InsightFace — NON-COMMERCIAL. "
                         "Блокер релиза. Цена замены: пересчёт всех порогов."),
}

#: Три исхода вместо двух (Р1). «Не смогли» не сворачивается ни в один из двух.
PASS, FAIL, UNMEASURED = "годно", "не годно", "не смогли проверить"

#: Какую долю кадров надо суметь измерить, чтобы вердикт вообще что-то значил.
#: НЕ СВОЯ константа: берётся у прибора, потому что это его свойство (Е1).
from .identity_arcface import MIN_COVERAGE  # noqa: E402


class DerivedAnchor(ValueError):
    """Якорем подан кадр из судимого набора. Это и есть дефект медоида."""


def _samples(manifest_path: str | Path) -> list:
    """Пути всех кадров набора, относительно каталога манифеста."""
    p = Path(manifest_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    root = p.parent
    return [(root / s["path"]).resolve() for s in data.get("samples", [])]


def refuse_derived_anchor(anchor: str | Path, frames, *,
                          manifest: str | Path | None = None) -> None:
    """Уронить прогон, если якорь взят из того, что судят. Медоид — этот случай.

    Три проверки, потому что медоид умеет прятаться тремя способами:

    1. Якорь буквально в списке судимых кадров.
    2. Якорь — кадр НАБОРА (манифест перечисляет его среди `samples`), даже
       если в судимый список он не попал. Ровно так и вышло на живом наборе:
       `real_0000.png` лежит в наборе, судятся 21 порождённый, и якорь
       формально «не среди них» — а по происхождению среди них.
    3. Ни того ни другого, но манифест сам называет якорь производным.

    Проверка по ПУТИ, а не по эмбеддингу: эмбеддинг-якорь, совпавший с кадром,
    даёт расстояние 0, и по нулю это тоже видно, — но кадр, порождённый ТЕМ ЖЕ
    генератором из ТОГО ЖЕ условия, нулю не равен и по эмбеддингу не ловится.
    Происхождение — свойство пути и манифеста, а не пикселей.
    """
    a = Path(anchor).resolve()
    if a in {Path(f).resolve() for f in frames}:
        raise DerivedAnchor(
            f"якорем подан кадр из судимого набора: {a.name}. Это сравнение "
            f"порождённого с порождённым — ровно тот дефект, из-за которого "
            f"0.2579 однажды прочли как успех при 0.5067 на настоящем якоре.")
    if manifest is None:
        return
    if a in _samples(manifest):
        raise DerivedAnchor(
            f"якорь {a.name} перечислен в наборе {Path(manifest).name} среди "
            f"samples: по происхождению он производный, даже если в судимый "
            f"список не попал. Якорем может быть только ЗАГРУЖЕННАЯ "
            f"фотография.")
    text = Path(manifest).read_text(encoding="utf-8")
    if "МЕДОИД" in text and a.name in text:
        raise DerivedAnchor(
            f"манифест {Path(manifest).name} сам называет {a.name} медоидом.")


def _instrument(name: str):
    """Прибор по имени. Только известные — опечатка не должна давать заглушку."""
    if name != DEFAULT_INSTRUMENT:
        raise ValueError(
            f"неизвестный прибор личности: {name!r}. Известен "
            f"{DEFAULT_INSTRUMENT!r}. Смена прибора обнуляет все снятые числа "
            f"и делается решением, а не опечаткой.")
    from . import identity_arcface

    return identity_arcface


def distances(frames, anchor: str | Path, *,
              instrument: str = DEFAULT_INSTRUMENT,
              min_face_px: int | None = None) -> dict:
    """Расстояния от каждого кадра до якоря. Три исхода на кадр, не два.

    `min_face_px=None` значит «не отсеивать по размеру лица» и это ОСОЗНАННОЕ
    умолчание, а не упущение. Отсев по размеру — свойство судейства ВИДЕО, где
    мелкое лицо означает «нечем судить»; здесь судится НАБОР НЕПОДВИЖНЫХ
    кадров, и отсев смещает саму величину, которую воспроизводим: на живом
    наборе он выкинул 2 кадра из 21 и превратил «19/21 в баре» в «17/19».
    Оба числа верны, и это разные числа — поэтому режим прибора всегда стоит
    рядом с результатом в `note`.
    """
    mod = _instrument(instrument)
    a = mod.face_detail(anchor)
    empty = {"per_frame": {}, "face_px": {}, "no_face": [], "too_small": [],
             "median": None, "min": None, "max": None,
             "inside": 0, "judged": 0, "total": 0,
             "coverage": 0.0, "outcome": UNMEASURED}
    if a is None:
        return {**empty,
                "note": f"на якоре {Path(anchor).name} лица нет: мерить не от чего"}

    per_frame, face_px, no_face, too_small = {}, {}, [], []
    total = 0
    for p in frames:
        total += 1
        name = Path(p).name
        d = mod.face_detail(p)
        if d is None:
            no_face.append(name)
            continue
        face_px[name] = d["face_px"]
        if min_face_px is not None and d["face_px"] < min_face_px:
            too_small.append(name)
            continue
        per_frame[name] = mod.cosine_distance(a["embedding"], d["embedding"])

    if not per_frame:
        return {**empty, "total": total, "face_px": face_px,
                "no_face": no_face, "too_small": too_small,
                "note": (f"судить нечего: из {total} кадров {len(no_face)} без "
                         f"лица, {len(too_small)} с лицом мельче "
                         f"{min_face_px}px. Это НЕ «другой человек».")}

    vals = sorted(per_frame.values())
    inside = sum(1 for v in vals if v <= SAME_PERSON_MAX)
    coverage = round(len(vals) / total, 3)
    return {
        "per_frame": per_frame, "face_px": face_px,
        "no_face": no_face, "too_small": too_small,
        "median": round(mod._quantile(vals, 0.5), 4),
        "min": round(vals[0], 4), "max": round(vals[-1], 4),
        "inside": inside, "judged": len(vals), "total": total,
        "coverage": coverage,
        "outcome": UNMEASURED if coverage < MIN_COVERAGE else (
            PASS if inside * 2 > len(vals) else FAIL),
        "note": (f"{instrument}: медиана "
                 f"{round(mod._quantile(vals, 0.5), 4)}, "
                 f"в баре {SAME_PERSON_MAX}: {inside} из {len(vals)} судимых "
                 f"(всего {total}; отсев по размеру лица: "
                 f"{'выключен' if min_face_px is None else str(min_face_px) + 'px'})"),
    }


def axis(frames, *, raw_photo: str | Path,
         reference: str | Path | None = None,
         foreign: str | Path | None = None,
         manifest: str | Path | None = None,
         instrument: str = DEFAULT_INSTRUMENT,
         min_face_px: int | None = None) -> dict:
    """Три числа разом, с вердиктом ПО СЫРОЙ ФОТОГРАФИИ и ни по чему другому.

    `raw_photo` обязателен и позиционно недоступен: якорь — то место, где уже
    один раз подменили величину, и подменить его случайным аргументом не
    должно быть возможно.

    Негативный контроль (`foreign`) не обязателен, но его отсутствие ПОПАДАЕТ
    В ОТЧЁТ как отдельное состояние. Прогон без контроля — не то же самое, что
    прогон, где контроль сработал, и сливать их в один «успех» нельзя.
    """
    # ОБА якоря проверяются ДО первого обращения к прибору (П2). Сначала здесь
    # стоял только `raw_photo`, а референс проверялся перед своим замером —
    # то есть после того, как d_raw уже посчитан. Тест это уронил: негодный
    # референс отказывался лишь потратив весь дорогой проход. Отказ за
    # микросекунды не должен стоить прогона по набору.
    refuse_derived_anchor(raw_photo, frames, manifest=manifest)
    if reference is not None:
        refuse_derived_anchor(reference, frames, manifest=manifest)

    out = {
        "instrument": instrument,
        "licence": INSTRUMENT_LICENCE.get(instrument, "лицензия не проверена"),
        "bar": SAME_PERSON_MAX,
        "hard_bar": HARD_DRIFT_MAX,
        "d_raw": distances(frames, raw_photo, instrument=instrument,
                           min_face_px=min_face_px),
        "d_ref": None,
        "d_neg": None,
        "control": "НЕ СТАВИЛСЯ",
    }
    if reference is not None:
        out["d_ref"] = distances(frames, reference, instrument=instrument,
                                 min_face_px=min_face_px)
    if foreign is not None:
        out["d_neg"] = distances(frames, foreign, instrument=instrument,
                                 min_face_px=min_face_px)
        out["control"] = control_verdict(out["d_neg"])

    out["verdict"] = out["d_raw"]["outcome"]
    out["note"] = _note(out)
    return out


def control_verdict(d_neg: dict) -> str:
    """Сработал ли негативный контроль. Тоже три исхода.

    Контроль ОБЯЗАН сказать «другой человек». Если чужого человека прибор
    принял за своего, все остальные числа этого прогона недействительны — не
    «хуже», а НЕДЕЙСТВИТЕЛЬНЫ, потому что мерили не то.
    """
    if d_neg.get("median") is None:
        return f"{UNMEASURED}: контроль не дал ни одного судимого кадра"
    if d_neg["inside"] > 0:
        return (f"{FAIL}: прибор принял чужого человека за своего на "
                f"{d_neg['inside']} кадре(ах) — числа прогона недействительны")
    if d_neg["median"] < HARD_DRIFT_MAX:
        return (f"{UNMEASURED}: чужой стоит на {d_neg['median']}, ниже "
                f"{HARD_DRIFT_MAX} — контроль слабый, полосу «заведомо чужой» "
                f"он не показывает")
    return f"{PASS}: чужой на {d_neg['median']}, ни одного кадра в баре"


def _note(out: dict) -> str:
    """Отчёт числами (Р2): проверено N, в баре M, не смогли K."""
    raw = out["d_raw"]
    head = (f"ВЕРДИКТ ПО СЫРОЙ ФОТОГРАФИИ: {raw['outcome']}. "
            f"медиана {raw['median']}, в баре {raw['inside']} из "
            f"{raw['judged']} судимых, не смогли "
            f"{len(raw['no_face']) + len(raw['too_small'])} из {raw['total']}.")
    if out["d_ref"] is not None:
        ref = out["d_ref"]
        head += (f" СПРАВОЧНО, НЕ ВЕРДИКТ — до референса: медиана "
                 f"{ref['median']}, в баре {ref['inside']} из {ref['judged']}.")
    head += f" КОНТРОЛЬ: {out['control']}."
    head += f" ЛИЦЕНЗИЯ ПРИБОРА: {out['licence']}"
    return head
