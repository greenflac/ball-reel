"""Стенд: прогнать пайплайн N раз и сказать, что и как часто ломается.

    python3 -m ball_reel.bench --face face.jpg --sessions 15 --attempts 2

Отвечает на три вопроса, которые задаёт нанимающий, и ни на один больше:

  1. сколько попыток гейт бракует и сколько сессий доходит до годного клипа;
  2. ЧТО ломается первым — распределение по проверкам;
  3. сколько pollen стоит один ПРИНЯТЫЙ клип (а не один вызов).

Четыре решения, которые здесь зафиксированы.

**Пишем построчно, а не в конце.** Каждая сессия ложится в JSONL сразу, как
закончилась. Стенд живёт на чужом API, у которого кончаются деньги, случаются
таймауты и 400-е; прогон, теряющий всё собранное при падении на четырнадцатой
сессии из пятнадцати, — это не стенд, а лотерея. Заодно это делает наблюдаемым
само поведение при отказе API, что для продукта на внешнем шлюзе часть
контракта, а не досадная мелочь.

**Считаем ШТУКИ, а не проценты.** «Годность 67%» на трёх сессиях — это 2 из 3,
и доверительный интервал там от 20% до 95%. Процент подразумевает точность,
которой нет. Поэтому наружу идут счётчики, а процент — только вместе с
интервалом Уилсона, чтобы читатель видел, насколько числу можно верить.

**Отделяем брак ПОПЫТОК от выхода СЕССИЙ.** Это два разных вопроса: первый про
качество генератора, второй про качество продукта, который умеет повторить.
Смешать их — значит либо польстить себе ретраями, либо занизить результат,
выбросив то, ради чего ретраи и написаны.

**Упавшая сессия — не пропуск строки, а строка отчёта.** Деньги тратит вызов, а
не удачный исход: сессия, оборвавшаяся на 429 после платной картинки и платного
видео, стоила ровно столько же, сколько дошедшая до вердикта. Считать расход по
дошедшим — занижать цену именно в том сценарии, ради которого стенд написан
(измерено на бумаге: 4 дошедших + 2 оборвавшихся по 0.08 — это 0.48 pollen, а
«по дошедшим» выходит 0.32, занижение ровно в 1.5 раза). Поэтому расход
считается по СОВЕРШЁННЫМ вызовам, а оборвавшиеся сессии видны отдельными
строками и в сводке, и в печати.

Ничего не чинит и не переспрашивает. Тратит деньги — ровно на то, что напечатал
перед стартом.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

#: Сколько независимых сессий по умолчанию. Не «побольше»: при 15 сессиях
#: интервал Уилсона для доли ещё широк, и стенд об этом честно скажет, но
#: распределение «что ломается первым» уже читается.
DEFAULT_SESSIONS = 15
#: Попыток внутри сессии. Это и есть ретраи боевого пути.
DEFAULT_ATTEMPTS = 2

#: Ниже этого числа сессий доля НЕ подаётся как результат.
#:
#: Порог появился из мутационного аудита: там выжила мутация на DEFAULT_SESSIONS,
#: и разбор показал, что я добавил её по шаблону — дефолт не порог, сдвинуть его
#: не значит сломать свойство. Но за ним стояла настоящая дыра: ничто не мешало
#: напечатать «1/1 прошло, 100%» и получить скриншот, который живёт своей
#: жизнью. Интервал Уилсона там честно скажет 5%..100%, но заголовок прочтут
#: раньше подписи.
#:
#: **ИЗМЕРЕН** — счётом по той же `wilson`, что печатается наружу. ВЫБРАН здесь
#: только критерий; число из него следует, а не наоборот.
#:
#: Критерий: доля имеет право называться результатом, когда интервал вокруг неё
#: у́же самой доли, то есть при худшем (самом широком) случае p=0.5 ширина
#: интервала меньше 50 процентных пунктов. Ширина берётся максимальная по всем
#: исходам k при данном n: порог обязан держать худший расклад, а не средний.
#:
#: Таблица (n → ширина интервала Уилсона в п.п., z=1.96, максимум по k):
#:
#:      n:    3     6     8     9    10    11    12    13    15    20
#:      pp: 73.1  62.4  57.0  54.4  52.6  50.7  49.2  47.7  45.1  40.2
#:
#: Первое n, где ширина уходит ниже 50 п.п. — ДВЕНАДЦАТЬ. Прежняя восьмёрка
#: стояла на арифметической ошибке в этом самом комментарии: там было написано
#: «~40 п.п. при n=8», а на деле при n=8 интервал 57 п.п. — шире самой доли,
#: то есть ровно тот случай, который порог и должен отсекать. 40 п.п. наступают
#: только к n=20. Число исправлено по расчёту, текст — под расчёт.
#:
#: Числом это всё ещё грубо (49 п.п. — не точность, а «хотя бы не
#: бессмыслица»), и стенд говорит об этом вслух рядом с каждой долей.
MIN_SESSIONS_FOR_YIELD = 12

#: Цена, pollen за единицу. Снято с /image/models и /video/models
#: [проверено live 2026-08-13]. Держится здесь, чтобы стенд печатал смету ДО
#: траты, а не выяснял стоимость по факту.
POLLEN = {"kontext": 0.04, "flux": 0.002, "nanobanana": 0.00003,
          "seedream5": 0.035, "seedream5-pro": 0.09,
          "wan-fast": 0.01, "veo": 0.08, "wan-pro": 0.1, "seedance-2.0": 0.18}


def estimate(sessions: int, attempts: int, seconds: int,
             start_model: str, video_model: str) -> float:
    """Смета до старта. Верхняя граница: считает, что все ретраи израсходуются
    И что каждая попытка дошла до видео."""
    per_attempt = POLLEN.get(start_model, 0.0) + POLLEN.get(video_model, 0.0) * seconds
    return round(per_attempt * attempts * sessions, 3)


def attempt_cost(check: str | None, seconds: int, start_model: str,
                 video_model: str) -> float:
    """Сколько стоила ОДНА попытка на самом деле.

    Попытка, отсеянная экраном старт-кадра, не доходит до видео-вызова и стоит
    только картинку. Считать ей полную цену — значит завышать себестоимость в
    разы и заодно прятать главное достоинство дешёвого экрана: он бракует ДО
    того, как потрачены основные деньги.

    Замерено живьём: на сложной рефке (профиль) экран отсеял 15 сессий из 15,
    ни один видео-вызов не состоялся, и смета в 2.4 pollen обернулась расходом
    примерно в 1.2 — ровно вдвое меньше, потому что дорогая половина каждой
    попытки не случилась.
    """
    from .produce import PRE_VIDEO_CHECKS

    cost = POLLEN.get(start_model, 0.0)
    if check not in PRE_VIDEO_CHECKS:
        cost += POLLEN.get(video_model, 0.0) * seconds
    return round(cost, 4)


def calls_made(session_dir) -> dict:
    """Сколько ПЛАТНЫХ вызовов реально состоялось — по артефактам на диске.

    Нужно там, где спросить больше некого: сессия оборвалась исключением, и
    списка попыток от неё не осталось. Файл в каталоге сессии — единственный
    свидетель, который не зависит от того, как именно порвалось.

    Допущение записано явно: считается состоявшимся тот вызов, который оставил
    файл. Вызов, оборвавшийся до файла, здесь не виден — значит цена оборванной
    сессии это НИЖНЯЯ граница расхода, а не точное число. Занижение на один
    незавершённый вызов честнее, чем выдумать вызовы, которых не было.

    `video_NN_loop.mp4` не считается: луп подрезается локально, ffmpeg денег не
    берёт. Считать его вторым видео-вызовом — удваивать самую дорогую строку
    сметы на каждом лупе.
    """
    d = Path(session_dir)
    if not d.is_dir():
        return {"image": 0, "video": 0}
    return {"image": len(list(d.glob("start_[0-9][0-9].png"))),
            "video": len(list(d.glob("video_[0-9][0-9].mp4")))}


def session_cost(rec: dict, seconds: int, start_model: str,
                 video_model: str) -> float:
    """Цена ОДНОЙ записи журнала — по совершённым вызовам, а не по исходу.

    У дошедшей до вердикта сессии есть список попыток, и он знает больше диска:
    по имени несработавшей проверки видно, дошла ли попытка до видео-вызова.
    У оборвавшейся списка нет — там меряем по артефактам (`calls_made`).
    """
    attempts = rec.get("attempts") or []
    if attempts:
        return round(sum(attempt_cost(a.get("check"), seconds, start_model,
                                      video_model) for a in attempts), 4)
    calls = rec.get("calls")
    if calls is None:
        calls = calls_made(rec.get("dir") or "")
    per_video = POLLEN.get(video_model, 0.0) * seconds
    return round(POLLEN.get(start_model, 0.0) * calls.get("image", 0)
                 + per_video * calls.get("video", 0), 4)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple:
    """Интервал Уилсона для доли. Возвращает (низ, верх) в долях единицы.

    Обычная формула `p ± z*sqrt(p(1-p)/n)` на малых выборках врёт особенно
    грубо: при 3 успехах из 3 она даёт интервал нулевой ширины, то есть
    «уверены на 100%» по трём наблюдениям. Уилсон на краях ведёт себя разумно,
    и именно он нужен, когда сессий пятнадцать, а не пятнадцать тысяч.
    """
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / d
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def _made_no_paid_call(rec: dict) -> bool:
    """Известно ли ТОЧНО, что запись не стоила ни одного вызова.

    Молчание (`calls` нет — старый журнал) толкуется не в свою пользу: считаем,
    что вызовы были. Иначе любой пробел в журнале начинает улучшать отчёт.
    """
    calls = rec.get("calls")
    return calls is not None and not (calls.get("image", 0)
                                      + calls.get("video", 0))


def summarise(records: list) -> dict:
    """Свести сессии в числа, которые читает нанимающий.

    ЗНАМЕНАТЕЛЬ ВЫХОДА ГОДНОГО — решение, принятое явно.

    В знаменатель идут все НАЧАТЫЕ сессии, включая оборвавшиеся на исключении.
    Обоснование: измеритель (гейт) в момент отказа работал — сломалось изделие,
    а не прибор. Отказ шлюза (429, таймаут, модерация) — это свойство продукта,
    который на этом шлюзе живёт, и выброшенная из знаменателя авария значила бы
    отчёт «выход годного при условии, что API отвечает». Нанимателю сдаётся
    конвейер целиком, вместе с его внешней зависимостью.

    Ровно одно исключение, и оно противоположного смысла: если про сессию
    ИЗВЕСТНО, что платных вызовов в ней не случилось вовсе (`calls` пустой), то
    порвалось до продукта — упал сам стенд, не нашёлся файл, не собрался бриф.
    Такие считаются отдельно (`aborted_before_any_call`) и в знаменатель не
    идут: иначе опечатка в стенде обваливает выход годного продукта, и число
    начинает мерить не то, что подписано.

    Альтернатива («крах — это не измерение, значит вне знаменателя, как
    отдельный вердикт „не смогли измерить“») имеет право на жизнь и здесь
    отклонена сознательно: она даёт долю, которую нельзя предъявить как
    надёжность продукта, а именно за этим числом сюда и приходят. Обе величины
    в сводке есть — `sessions` (дошли до вердикта) и `sessions_denominator`, —
    так что читатель волен посчитать и по-другому, но по умолчанию отчёт не
    льстит себе.
    """
    sessions = [r for r in records if r.get("kind") == "session"]
    attempts = [a for r in sessions for a in r.get("attempts", [])]
    crashed = [r for r in records if r.get("kind") == "crash"]
    aborted = [r for r in crashed if _made_no_paid_call(r)]
    burned = [r for r in crashed if not _made_no_paid_call(r)]

    passed_sessions = [r for r in sessions if r.get("passed")]
    passed_attempts = [a for a in attempts if a.get("passed")]

    from .produce import PRE_VIDEO_CHECKS

    first_fail: dict = {}
    pre_video = post_video = 0
    for a in attempts:
        if a.get("passed"):
            continue
        name = a.get("check") or "error"
        first_fail[name] = first_fail.get(name, 0) + 1
        if name in PRE_VIDEO_CHECKS:
            pre_video += 1
        else:
            post_video += 1

    # Деньги считаются по ВСЕМ записям: вызов уже оплачен независимо от того,
    # чем кончилась сессия. Раньше здесь стояли только `sessions`, и расход
    # оборвавшихся сессий исчезал из отчёта вместе с ними.
    spent = round(sum(r.get("pollen", 0.0) for r in records), 4)
    on_crashes = round(sum(r.get("pollen", 0.0) for r in crashed), 4)
    denominator = len(sessions) + len(burned)
    per_accepted = (round(spent / len(passed_sessions), 4)
                    if passed_sessions else None)
    durations = sorted(r["seconds"] for r in sessions if r.get("seconds"))

    # СКОЛЬКО ПОПЫТОК ОСЬ ОДЕЖДЫ НЕ СМОГЛА ПОСУДИТЬ. Веса CLIP (~600 МБ) на
    # машине могут отсутствовать, и гейт на этом не падает — исход честно
    # `not_measurable`. Но тогда «одежда проверена» становится неправдой, а по
    # сводке этого не видно: непроверенное неотличимо от прошедшего.
    #
    # Это та же беда, что уже стоила проекту неверного вывода: поле
    # `first_failing_check` свело два разных исхода к слову `error`, и по нему
    # ДВАЖДЫ был сделан вывод «всё упало на вызовах API», хотя отказов API не
    # было ни одного. Поэтому непроверенное считается отдельным числом, а не
    # выводится вычитанием.
    judged = [a for a in attempts if a.get("semantic")]
    unmeasured = [a for a in judged
                  if (a["semantic"] or {}).get("verdict") == "not_measurable"]

    return {
        "semantic_judged": len(judged),
        "semantic_not_measurable": len(unmeasured),
        "semantic_note": (
            f"ось одежды не смогла посудить {len(unmeasured)} из {len(judged)} "
            f"попыток, до которых дошла" if judged else
            "ось одежды не запускалась ни разу: до неё не дошла ни одна "
            "попытка, либо весов CLIP нет на машине"),
        "sessions": len(sessions),
        "sessions_denominator": denominator,
        "sessions_passed": len(passed_sessions),
        "sessions_ci": wilson(len(passed_sessions), denominator),
        "attempts": len(attempts),
        "attempts_passed": len(passed_attempts),
        "attempts_ci": wilson(len(passed_attempts), len(attempts)),
        "first_failing_check": dict(sorted(first_fail.items(),
                                           key=lambda kv: -kv[1])),
        # Отсев ДО видео-вызова — это сэкономленные деньги, а не просто брак.
        "rejected_before_video": pre_video,
        "rejected_after_video": post_video,
        "pollen_spent": spent,
        "pollen_on_crashes": on_crashes,
        "pollen_per_accepted_clip": per_accepted,
        "seconds_median": (durations[len(durations) // 2] if durations
                           else None),
        "crashes": len(crashed),
        # Оборвалось ПОСЛЕ платных вызовов — это отказ продукта, он в
        # знаменателе. Оборвалось ДО — отказ стенда, он вне знаменателя.
        "crashed_after_paying": len(burned),
        "aborted_before_any_call": len(aborted),
    }


def yield_is_reportable(sessions: int,
                        floor: int = MIN_SESSIONS_FOR_YIELD) -> tuple:
    """Можно ли вообще называть долю результатом. (можно, почему нет).

    Формулировка нарочно про «наблюдения», а не про «сессии»: тот же порог
    сторожит и строку попыток, и подписывать её словом «сессий» — врать в
    мелочи там, где весь смысл функции в том, чтобы не врать.
    """
    if sessions >= floor:
        return True, ""
    return False, (f"{sessions} наблюдени(й) — доля НЕ отчитывается как "
                   f"результат "
                   f"(нужно от {floor}). Смотреть счётчики и распределение "
                   f"отказов: они информативны и на малой выборке, в отличие "
                   f"от процента.")


def render(s: dict) -> str:
    """Профиль стенда таблицей. Ведём со штук, процент — только с интервалом."""
    # Знаменатель — начатые сессии, включая оборвавшиеся (см. summarise).
    # Пусто он или нет, решает именно он: прогон, где все сессии упали, — это
    # не «сессий не было», а результат, и молчать о нём нельзя.
    total = s.get("sessions_denominator", s.get("sessions", 0))
    if not total and not s.get("aborted_before_any_call"):
        return "сессий не было — судить не о чем"
    if not total:
        return (f"  ни одна сессия не дошла до платного вызова: "
                f"{s['aborted_before_any_call']} — это отказ стенда, "
                f"судить продукт не по чему")
    lo, hi = s["sessions_ci"]
    alo, ahi = s["attempts_ci"]
    ok_yield, why = yield_is_reportable(total)
    share = (f"доля {s['sessions_passed'] / total:.0%}, "
             f"но истинная лежит между {lo:.0%} и {hi:.0%} — выборка мала"
             if ok_yield else why)
    lines = [
        f"  сессий:   {s['sessions_passed']}/{total} прошло  ({share})",
        # Тот же порог и для попыток: «1/1, 100%» здесь читается ровно так же,
        # и защищать только строку сессий значит оставить дверь рядом открытой.
        (f"  попыток:  {s['attempts_passed']}/{s['attempts']} прошло  "
         + (f"(доля {s['attempts_passed'] / s['attempts']:.0%}, "
            f"интервал {alo:.0%}..{ahi:.0%})"
            if yield_is_reportable(s["attempts"])[0]
            else f"({yield_is_reportable(s['attempts'])[1]})")
         ) if s["attempts"] else "",
        f"  потрачено: {s['pollen_spent']} pollen; на ПРИНЯТЫЙ клип "
        + (f"{s['pollen_per_accepted_clip']}" if s["pollen_per_accepted_clip"]
           else "— (принятых нет, делить не на что)"),
        f"  медиана сессии: {s['seconds_median']} c",
    ]
    # Обрыв — отдельная строка, а не примечание мелким шрифтом: он и в
    # знаменателе, и в деньгах, и читатель обязан видеть обе стороны.
    if s.get("crashed_after_paying"):
        lines.append(
            f"  оборвалось на отказе API: {s['crashed_after_paying']} "
            f"(деньги потрачены — {s.get('pollen_on_crashes', 0.0)} pollen, "
            f"клипа нет; сидят в знаменателе доли)")
    if s.get("aborted_before_any_call"):
        lines.append(
            f"  не дошло до первого платного вызова: "
            f"{s['aborted_before_any_call']} (отказ стенда, а не продукта — "
            f"из знаменателя исключены)")
    if s.get("rejected_before_video"):
        lines.append(
            f"  отсеяно ДО видео-вызова: {s['rejected_before_video']} "
            f"(дешёвый экран), ПОСЛЕ: {s.get('rejected_after_video', 0)}")
    if s["first_failing_check"]:
        lines.append("  что ломается ПЕРВЫМ:")
        for name, n in s["first_failing_check"].items():
            lines.append(f"      {name:<18} {n}")
    return "\n".join(ln for ln in lines if ln)


def _session(face: str, n: int, out_dir: Path, *, attempts: int,
             start_model: str, video_model: str, brief=None,
             subject=None) -> dict:
    """Одна независимая сессия боевого пути. Возвращает запись для журнала."""
    from . import pollinations
    from .produce import produce

    started = time.perf_counter()
    sdir = out_dir / f"s{n:02d}"
    rec: dict = {"kind": "session", "n": n, "face": face, "dir": str(sdir)}
    try:
        kwargs = {"attempts": attempts, "start_model": start_model,
                  "video_model": video_model,
                  "out_dir": str(sdir)}
        if brief is not None:
            kwargs["brief"] = brief
        if subject is not None:
            kwargs["subject"] = subject
        res = produce(face, **kwargs)
        rec.update({
            "passed": bool(res.passed),
            "note": res.note,
            "clip": res.clip_path,
            "attempts": [{
                "n": a.n, "passed": bool(a.passed), "reason": a.reason,
                "check": getattr(a, "check", None),
                "identity": a.worst_identity_drift, "motion": a.motion,
                "loop_ratio": a.loop_ratio, "worst_jump": a.worst_jump,
                "pose_distance": a.pose_distance,
            } for a in res.attempts],
        })
    except Exception as e:  # noqa: BLE001 — стенд обязан пережить отказ API
        rec.update({"kind": "crash", "passed": False,
                    "error": f"{type(e).__name__}: {e}"[:400], "attempts": []})
    rec["seconds"] = round(time.perf_counter() - started, 1)
    # Свидетель платных вызовов. Пишется ВСЕГДА, а не только при обрыве:
    # у оборвавшейся сессии это единственный источник цены, а у дошедшей —
    # независимая от списка попыток проверка, что счёт сошёлся.
    rec["calls"] = calls_made(sdir)
    usage = getattr(pollinations, "LAST_VIDEO_USAGE", None) or {}
    rec["usage"] = usage
    return rec


def main(argv: list) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="ball_reel.bench",
        description="N живых сессий -> yield, распределение отказов, цена")
    ap.add_argument("--face", required=True)
    ap.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    ap.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    ap.add_argument("--start-model", default="kontext")
    ap.add_argument("--video-model", default="wan-fast")
    ap.add_argument("--seconds", type=int, default=4)
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--budget", type=float, default=None,
                    help="потолок в pollen; стенд остановится, не превысив его")
    ap.add_argument("--dry-run", action="store_true",
                    help="только смета, ничего не тратить")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    journal = out / "sessions.jsonl"

    plan = estimate(args.sessions, args.attempts, args.seconds,
                    args.start_model, args.video_model)
    print(f"смета (верхняя граница, все ретраи израсходованы): {plan} pollen")
    print(f"  {args.sessions} сессий x {args.attempts} попыток; "
          f"старт {args.start_model} {POLLEN.get(args.start_model)} + "
          f"видео {args.video_model} {POLLEN.get(args.video_model)}/с "
          f"x {args.seconds} c")
    if args.budget is not None and plan > args.budget:
        print(f"смета выше потолка {args.budget} — уменьшить --sessions "
              f"или взять модель дешевле")
        return 1
    if args.dry_run:
        print("--dry-run: ничего не потрачено")
        return 0

    print(f"журнал пишется построчно в {journal} — падение API не унесёт "
          f"собранное\n")
    records = []
    for n in range(1, args.sessions + 1):
        rec = _session(args.face, n, out, attempts=args.attempts,
                       start_model=args.start_model,
                       video_model=args.video_model)
        # Цену считаем по факту вызовов этой сессии, а не по смете, — и у
        # оборвавшейся тоже: она успела потратить до того, как упала.
        rec["pollen"] = session_cost(rec, args.seconds, args.start_model,
                                     args.video_model)
        records.append(rec)
        with journal.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mark = "OK  " if rec.get("passed") else "FAIL"
        why = (rec.get("error") or
               (rec.get("attempts") or [{}])[-1].get("check") or
               rec.get("note", ""))
        print(f"  {mark} сессия {n:>2}/{args.sessions}  {rec['seconds']:>5.1f} c  "
              f"{str(why)[:80]}")

    s = summarise(records)
    (out / "summary.json").write_text(
        json.dumps(s, indent=2, ensure_ascii=False))
    print("\n--- итог ---")
    print(render(s))
    print(f"\nжурнал: {journal}\nсводка: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
