"""Точка входа форка: сквозной путь от фотографии до кадров условий, НА МОКЕ.

ЧТО ЭТО И ЧЕМ ОНО НЕ ЯВЛЯЕТСЯ. Здесь собирается путь — приём входов, условия,
маски, граф, приёмочные оси — и НЕ ЗАПУСКАЕТСЯ генерация. Генерации в спринте
нет: железо не выбрано, ComfyUI не установлен. Модуль отдаёт то, что можно
получить без карты: посчитанные последовательности, произведённый граф и отчёт
о том, какие проверки прошли, какие провалились и какие НЕ СМОГЛИ отработать.

ЗАЧЕМ ОН ВООБЩЕ НУЖЕН, КРОМЕ УДОБСТВА. `ball_reel/tests/test_reachable.py`
краснеет на модуле, который написан и не подключён ни к чему, — и он покраснел
на `fork_comfy`, `fork_leak`, `fork_mask` ровно тогда, когда они были дописаны.
Это правильное поведение сторожа: модуль, которого нет в конвейере, но есть в
отчёте и в справочнике, опаснее ненаписанного. Собрать их в путь — верный ответ
на это, а вписать в список самостоятельных — способ заглушить.

ТРИ ИСХОДА НА КАЖДОМ ШАГЕ (Р1), и итог не сворачивается в булев флаг: шаг,
который не смог отработать, отличается и от пройденного, и от провалившегося.
Печатается числами (Р2): проверено N, провалено M, не смогли K.
"""

from __future__ import annotations

from pathlib import Path

from . import fork_channels, fork_comfy, fork_leak, fork_mask
from .fork_identity import FAIL, PASS, UNMEASURED

#: Порядок шагов. Дешёвое раньше дорогого (П2): отсутствующий вход ловится за
#: миллисекунды, а поиск его после снятия условий по сотне кадров стоил бы
#: всего прогона. Длительность каждого шага печатается.
STEPS = ("входы", "условия", "маски", "граф", "протечка")


def _step(name: str, outcome: str, note: str, seconds: float) -> dict:
    return {"step": name, "outcome": outcome, "note": note,
            "seconds": round(seconds, 3)}


def run(photo: str | Path, driving_frames, out_dir: str | Path, *,
        grow_px: int = fork_mask.BLOCK,
        mask_model=None) -> dict:
    """Сквозной путь на моке. Возвращает отчёт по шагам, а не «получилось».

    `photo` — ЗАГРУЖЕННАЯ фотография, она же якорь оси личности. Медоид сюда
    подать нельзя: `fork_identity.refuse_derived_anchor` уронит прогон, и это
    единственный способ не повторить дефект, из-за которого 0.2579 однажды
    прочли как успех.
    """
    import time

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [Path(p) for p in driving_frames]
    steps: list[dict] = []

    t = time.perf_counter()
    missing = [str(p) for p in [Path(photo), *frames] if not p.exists()]
    steps.append(_step(
        "входы", FAIL if missing else PASS,
        f"нет файлов: {missing}" if missing else
        f"фотография и {len(frames)} кадров драйвинга на месте",
        time.perf_counter() - t))
    if missing:
        return _report(steps, out)

    t = time.perf_counter()
    if fork_channels.dwpose.available():
        cond = fork_channels.render_sequence(frames, out / "cond")
        steps.append(_step(
            "условия",
            PASS if cond["rendered"] == cond["total"] else
            UNMEASURED if cond["rendered"] else FAIL,
            cond["note"], time.perf_counter() - t))
    else:
        cond = None
        steps.append(_step("условия", UNMEASURED,
                           fork_channels.dwpose.why_unavailable().split(".")[0],
                           time.perf_counter() - t))

    t = time.perf_counter()
    from . import bodyparts

    if bodyparts.available():
        masks = fork_mask.sequence(frames, out / "mask", grow_px=grow_px,
                                   model=mask_model)
        steps.append(_step("маски", masks["outcome"], masks["note"],
                           time.perf_counter() - t))
    else:
        masks = None
        steps.append(_step("маски", UNMEASURED,
                           bodyparts.why_unavailable().split(".")[0],
                           time.perf_counter() - t))

    t = time.perf_counter()
    try:
        derived = fork_comfy.derive()
        audit = fork_comfy.audit(derived)
        fork_comfy.write(derived, out / "fork_graph.json")
        steps.append(_step("граф", audit["outcome"], audit["note"],
                           time.perf_counter() - t))
    except (OSError, ValueError) as exc:
        steps.append(_step("граф", UNMEASURED, str(exc)[:200],
                           time.perf_counter() - t))

    # Протечка на моке НЕ ИЗМЕРЯЕТСЯ, и это не пропуск: сравнивать выход с
    # драйвингом можно только когда выход есть, а генерации в спринте нет.
    # Планка шума при этом замеряема уже сейчас — она про кодек, не про модель.
    t = time.perf_counter()
    floor = fork_leak.noise_floor(frames)
    steps.append(_step(
        "протечка", UNMEASURED,
        (f"выхода нет — мерить нечего; планка шума: {floor['note']}"),
        time.perf_counter() - t))

    return _report(steps, out)


def _report(steps: list, out: Path) -> dict:
    passed = sum(1 for s in steps if s["outcome"] == PASS)
    failed = sum(1 for s in steps if s["outcome"] == FAIL)
    unmeasured = sum(1 for s in steps if s["outcome"] == UNMEASURED)
    return {
        "steps": steps, "dir": str(out),
        "passed": passed, "failed": failed, "unmeasured": unmeasured,
        "outcome": FAIL if failed else (UNMEASURED if unmeasured else PASS),
        "note": (f"шагов {len(steps)}: пройдено {passed}, провалено {failed}, "
                 f"не смогли {unmeasured}. ГЕНЕРАЦИИ НЕ БЫЛО — путь собран на "
                 f"моке, продуктовое заявление им не проверяется."),
    }
