"""Тесты контракта отчёта ресёрч-агента.

Стережём здесь одну вещь: чтобы «нас не пустили» нельзя было подать как
«там ничего нет». Всё остальное — обвязка вокруг этого.

Ожидаемые значения — литералы, а не импорт из проверяемого модуля: константа,
взятая из того же файла, поедет вместе с ним и промолчит.
"""

import unittest

from ball_reel.research_contract import (
    expected_outcome, parse_report, verify,
    VERDICT_OK, VERDICT_BAD, VERDICT_UNPARSED,
)


def отчёт(исход, проверено, не_смогли, источники=(), блокировки=""):
    строки = [
        f"ИСХОД: {исход}",
        f"ПРОВЕРЕНО: {проверено}",
        f"НЕ СМОГЛИ: {не_смогли}",
    ]
    if блокировки:
        строки.append(f"БЛОКИРОВКИ: {блокировки}")
    строки.append("ИСТОЧНИКИ:")
    строки += [f"- {u}" for u in источники]
    return "\n".join(строки)


ССЫЛКА = "https://example.org/doc"
ВТОРАЯ = "https://example.net/page"


class ТриИсхода(unittest.TestCase):
    """expected_outcome: исход выводится из свидетельства, а не из намерения."""

    def test_ноль_проверенных_это_не_смогли(self):
        # Ноль нарушений при нуле проверок не бывает успехом — и не бывает
        # доказанным отсутствием.
        self.assertEqual(expected_outcome(0, 0, [], []), "не смогли")

    def test_есть_источник_это_найдено(self):
        self.assertEqual(expected_outcome(3, 0, [ССЫЛКА], []), "найдено")

    def test_нечем_подтвердить_и_что_то_не_доехало_это_не_смогли(self):
        self.assertEqual(expected_outcome(3, 1, [], []), "не смогли")

    def test_блокировка_без_источников_это_не_смогли(self):
        self.assertEqual(expected_outcome(3, 0, [], [(403, "example.org")]),
                         "не смогли")

    def test_полный_обыск_без_находок_это_не_найдено(self):
        # Единственная ветка, где «не найдено» законно: доехало всё.
        self.assertEqual(expected_outcome(5, 0, [], []), "не найдено")

    def test_середина_диапазона_часть_доехала(self):
        # Половина источников дошла, половина нет: находка есть, и она
        # перевешивает — недоехавшее раскрывается счётчиком, а не исходом.
        self.assertEqual(expected_outcome(4, 2, [ССЫЛКА, ВТОРАЯ], [(429, "a.io")]),
                         "найдено")


class БлокировкаНеОтсутствие(unittest.TestCase):
    """Тот самый дефект: 403 подан как установленный факт «ничего нет»."""

    def test_403_выданный_за_не_найдено_ловится(self):
        v = verify(отчёт("не найдено", 3, 0, блокировки="403 example.org"))
        self.assertEqual(v.status, VERDICT_BAD)
        self.assertTrue(any("403" in s for s in v.violations), v.violations)

    def test_429_тоже_блокировка(self):
        v = verify(отчёт("не найдено", 2, 0, блокировки="429 example.org"))
        self.assertEqual(v.status, VERDICT_BAD)

    def test_404_не_блокировка_а_отсутствие(self):
        # Негативный контроль прибора: на входе, где он обязан молчать,
        # он обязан молчать. 404 — это действительно «нет страницы».
        v = verify(отчёт("не найдено", 2, 0, блокировки="404 example.org"))
        self.assertEqual(v.status, VERDICT_OK, v.violations)

    def test_блокировка_при_честном_исходе_проходит(self):
        v = verify(отчёт("не смогли", 2, 1, блокировки="403 example.org"))
        self.assertEqual(v.status, VERDICT_OK, v.violations)


class Счётчики(unittest.TestCase):

    def test_найдено_без_источников_не_годно(self):
        v = verify(отчёт("найдено", 3, 0))
        self.assertEqual(v.status, VERDICT_BAD)
        self.assertTrue(any("источник" in s for s in v.violations), v.violations)

    def test_источников_больше_чем_проверено(self):
        v = verify(отчёт("найдено", 1, 0, [ССЫЛКА, ВТОРАЯ]))
        self.assertEqual(v.status, VERDICT_BAD)

    def test_отрицательный_счётчик(self):
        self.assertEqual(verify(отчёт("не смогли", -1, 0)).status, VERDICT_BAD)

    def test_годный_отчёт_проходит(self):
        # Второй край негативного контроля: вход, на котором прибор обязан
        # шевельнуться в сторону «годно», иначе он просто всегда говорит «нет».
        v = verify(отчёт("найдено", 4, 1, [ССЫЛКА, ВТОРАЯ], "403 blocked.io"))
        self.assertEqual(v.status, VERDICT_OK, v.violations)
        self.assertEqual(v.violations, [])


class Разбор(unittest.TestCase):

    def test_не_отчёт_это_третий_исход_а_не_нарушение(self):
        v = verify("Я посмотрел пару сайтов, там ничего интересного.")
        self.assertEqual(v.status, VERDICT_UNPARSED)
        self.assertIsNone(v.report)

    def test_неизвестный_исход_не_разбирается(self):
        self.assertEqual(verify(отчёт("возможно", 1, 0)).status, VERDICT_UNPARSED)

    def test_источники_и_блокировки_разобраны(self):
        r = parse_report(отчёт("найдено", 2, 1, [ССЫЛКА], "403 a.io, 429 b.io"))
        self.assertEqual(r.sources, [ССЫЛКА])
        self.assertEqual(r.blocked, [(403, "a.io"), (429, "b.io")])
        self.assertEqual((r.checked, r.failed), (2, 1))


class КодыВозврата(unittest.TestCase):
    """Три кода, а не два: неразобранное отличается от нарушений."""

    def test_коды(self):
        self.assertEqual(verify(отчёт("найдено", 1, 0, [ССЫЛКА])).exit_code, 0)
        self.assertEqual(verify(отчёт("найдено", 1, 0)).exit_code, 1)
        self.assertEqual(verify("мусор").exit_code, 2)


if __name__ == "__main__":
    unittest.main()
