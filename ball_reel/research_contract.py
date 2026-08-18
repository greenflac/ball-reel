"""Контракт отчёта ресёрч-агента: три исхода вместо двух, проверяемые машиной.

    python3 -m ball_reel.research_contract отчёт.txt

Зачем это здесь. Ресёрч-агент ходит в сеть, а сеть отвечает тремя разными
вещами: «вот ответ», «ответа нет» и «меня не пустили». Третье — не второе.
Агент, у которого нет места под третий исход, кладёт блокировку в «не найдено»,
и дальше по цепочке 403 от WAF читается как установленный факт отсутствия.
Это самая дорогая ошибка ресёрча: она не выглядит ошибкой.

Словами это не держится: инструкция в промпте деградирует молча. Поэтому
контракт разбирается кодом, а код сторожится тестами и мутационным аудитом.

Модуль ничего не генерирует, В СЕТЬ НЕ ХОДИТ и ничего не скачивает: на вход
подаётся текст отчёта, на выходе — вердикт. Это позволяет гонять его в CI.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

#: Исходы самого ресёрча. Ровно три, и третий не сворачивается в первые два.
OUTCOME_FOUND = "найдено"
OUTCOME_NOT_FOUND = "не найдено"
OUTCOME_FAILED = "не смогли"
OUTCOMES = (OUTCOME_FOUND, OUTCOME_NOT_FOUND, OUTCOME_FAILED)

#: Исходы ПРОВЕРКИ отчёта. Тоже три: разобранный и годный, разобранный и
#: негодный, и неразобранный — последний не равен «негодному», потому что
#: про неразобранный текст мы не знаем ничего, включая то, плох ли он.
VERDICT_OK = "годно"
VERDICT_BAD = "не годно"
VERDICT_UNPARSED = "не смогли разобрать"

#: Коды, которые означают «нас не пустили», а не «там пусто». 403 и 407 —
#: политика и прокси, 401/402 — доступ по подписке, 429 — троттлинг,
#: 451 — блокировка по праву. Ни один из них НЕ ЯВЛЯЕТСЯ свидетельством
#: отсутствия информации, и именно на этом месте ломается вся цепочка.
BLOCKED_STATUS_CODES = frozenset({401, 402, 403, 407, 429, 451})

#: Сколько источников обязано стоять за исходом «найдено». Ноль означал бы,
#: что «нашёл» можно сказать, не показав ничего.
MIN_SOURCES_FOR_FOUND = 1

_URL = re.compile(r"https?://\S+")
_BLOCK = re.compile(r"(\d{3})\s+(\S+)")


@dataclass
class Report:
    """Разобранный отчёт. Числа — как их назвал агент, без исправлений."""

    outcome: str
    checked: int
    failed: int
    sources: list = field(default_factory=list)
    blocked: list = field(default_factory=list)   # [(код, хост), ...]


@dataclass
class Verdict:
    """Вердикт проверки. `status` — один из VERDICT_*, нарушения — списком."""

    status: str
    violations: list = field(default_factory=list)
    report: object = None

    @property
    def exit_code(self) -> int:
        """0 годно, 1 нарушения, 2 не разобрали.

        Три кода, а не два: «не смогли разобрать» обязано отличаться от
        «нарушений нет» ровно так же, как в самом проверяемом контракте.
        """
        return {VERDICT_OK: 0, VERDICT_BAD: 1, VERDICT_UNPARSED: 2}[self.status]


def expected_outcome(checked: int, failed: int, sources, blocked) -> str:
    """Какой исход СЛЕДУЕТ из свидетельства, независимо от того, что написал агент.

    Порядок веток — это и есть правило:

    1. ноль проверенных источников — «не смогли», даже если нарушений нет:
       ноль нарушений при нуле проверок не бывает успехом;
    2. есть чем подтвердить — «найдено», а недоехавшие источники раскрываются
       счётчиком, а не молчанием;
    3. подтвердить нечем И что-то не доехало — «не смогли»;
    4. подтвердить нечем, при этом доехало всё — только тогда «не найдено».

    Пункт 4 узкий намеренно: «не найдено» — сильное утверждение, оно требует,
    чтобы обыск действительно состоялся целиком.

    Считаются только коды из BLOCKED_STATUS_CODES: в списке блокировок может
    стоять и 404, а 404 — это как раз честное отсутствие страницы, а не отказ
    пустить. Ловушка поймана негативным контролем в тестах: без фильтра
    прибор объявлял «не смогли» там, где обыск состоялся полностью.
    """
    if checked <= 0:
        return OUTCOME_FAILED
    if len(sources) >= MIN_SOURCES_FOR_FOUND:
        return OUTCOME_FOUND
    if failed > 0 or any(code in BLOCKED_STATUS_CODES for code, _ in blocked):
        return OUTCOME_FAILED
    return OUTCOME_NOT_FOUND


def parse_report(text: str):
    """Разобрать отчёт. Вернуть Report либо None, если формата нет.

    None — это не «плохой отчёт», а «не отчёт»: см. VERDICT_UNPARSED.
    """
    def field_of(name: str):
        m = re.search(rf"^\s*{name}\s*:\s*(.*)$", text, re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip() if m else None

    raw_outcome = field_of("ИСХОД")
    raw_checked = field_of("ПРОВЕРЕНО")
    raw_failed = field_of("НЕ СМОГЛИ")
    if raw_outcome is None or raw_checked is None or raw_failed is None:
        return None

    outcome = raw_outcome.lower().strip(" .")
    if outcome not in OUTCOMES:
        return None
    try:
        checked = int(re.search(r"-?\d+", raw_checked).group())
        failed = int(re.search(r"-?\d+", raw_failed).group())
    except (AttributeError, ValueError):
        return None

    raw_blocked = field_of("БЛОКИРОВКИ") or ""
    blocked = [(int(code), host.rstrip(",;"))
               for code, host in _BLOCK.findall(raw_blocked)]

    sources_part = text.split("ИСТОЧНИКИ", 1)[1] if "ИСТОЧНИКИ" in text else ""
    sources = _URL.findall(sources_part)
    return Report(outcome=outcome, checked=checked, failed=failed,
                  sources=sources, blocked=blocked)


def verify(text: str) -> Verdict:
    """Проверить отчёт на соответствие контракту."""
    report = parse_report(text)
    if report is None:
        return Verdict(VERDICT_UNPARSED,
                       ["формат отчёта не распознан: нужны строки "
                        "ИСХОД / ПРОВЕРЕНО / НЕ СМОГЛИ"])

    v = []
    if report.checked < 0 or report.failed < 0:
        v.append("счётчики отрицательные")
    if report.checked < len(report.sources):
        v.append(f"источников {len(report.sources)} больше, чем проверено "
                 f"{report.checked}")

    # Свидетельство важнее ярлыка: исход выводится из того, что исполнилось.
    should = expected_outcome(report.checked, report.failed,
                              report.sources, report.blocked)
    if report.outcome != should:
        v.append(f"исход «{report.outcome}» не следует из свидетельства: "
                 f"проверено {report.checked}, не смогли {report.failed}, "
                 f"источников {len(report.sources)}, "
                 f"блокировок {len(report.blocked)} — следует «{should}»")

    # Отдельной строкой, потому что это тот самый дефект, ради которого
    # написан модуль: блокировка, поданная как установленное отсутствие.
    blocking = [c for c, _ in report.blocked if c in BLOCKED_STATUS_CODES]
    if blocking and report.outcome == OUTCOME_NOT_FOUND:
        v.append(f"коды блокировки {sorted(set(blocking))} выданы за «не найдено»")

    if report.outcome == OUTCOME_FOUND and len(report.sources) < MIN_SOURCES_FOR_FOUND:
        v.append("«найдено» без источников")

    return Verdict(VERDICT_BAD if v else VERDICT_OK, v, report)


def main(argv: list) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    verdict = verify(open(argv[1], encoding="utf-8").read())
    r = verdict.report
    if r is not None:
        # Числа печатаются рядом с вердиктом: агрегатное «годно» без них
        # читается как полная работа, даже когда половина не доехала.
        print(f"проверено {r.checked}, не смогли {r.failed}, "
              f"источников {len(r.sources)}, блокировок {len(r.blocked)}")
    print(f"вердикт: {verdict.status}, нарушений {len(verdict.violations)}")
    for line in verdict.violations:
        print(f"  - {line}")
    return verdict.exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
