"""Число, повторённое в восьми документах, разойдётся в восьми документах.

НАЙДЕНО ЗАМЕРОМ, а не рассуждением. На один и тот же момент документы
утверждали:

    README.md              Ran 1336 tests
    NEW_SESSION_PROMPT.md  1283 теста, 119 мутаций
    LAUNCH.md              1192 теста, 118 порогов
    GUIDE.md               824 теста, 77 мутаций — и 56 порогов абзацем выше
    REFERENCE.md           801 тест, 27 файлов, 56 порогов
    GPU_RUNBOOK.md         801 тест, 56 мутаций
    TEST_KIT.md            Ran 801 tests in 88s
    DEMO_RUNBOOK.md        527

Разброс по тестам — от 527 до 1336 при действительных 1345, по мутациям — от 56
до 119 при действительных 119. GUIDE противоречил сам себе на расстоянии
четырёхсот строк. `CLAUDE.md` требует сверять хэндоф прогоном именно из-за
такого: смена, спланированная по числу «801», начинается с неверной картины.

ЛЕЧЕНИЕ РАЗНОЕ ДЛЯ ДВУХ ЧИСЕЛ, и это не прихоть.

Число мутаций ВЫВОДИТСЯ ИЗ КОДА — `len(codeaudit.MUTATIONS)`, — поэтому его
можно не запрещать, а сверять: пусть стоит хоть во всех документах, лишь бы
совпадало.

Число тестов вывести нельзя: его можно только прогнать. Значит любое место, где
оно записано, — это копия, которая устареет молча. Поэтому оно разрешено ровно
в одном файле, `docs/NUMBERS.md`, рядом с командой, которой получено.

ИСТОРИЧЕСКИЕ УПОМИНАНИЯ РАЗРЕШЕНЫ И НУЖНЫ. «Здесь стояло 249», «~~801~~» — это
история находок, часть ценности репозитория на аудите; `CLAUDE.md` прямо
требует перечёркивать, а не стирать. Отличаются они пометкой при себе, и
детектор смотрит именно на пометку, а не на число.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Единственный файл, которому можно называть размер набора.
HOME = "docs/NUMBERS.md"

#: Пометки, превращающие утверждение в исторический факт. Строка с любой из них
#: говорит о прошлом, и запрещать ей числа значило бы запрещать историю.
HISTORICAL = ("~~", "стояло", "устарел", "ранее", "раньше", "прежн",
              "было верным", "более ранн", "первой редакции", "выросло",
              "концу", "в тот же момент", "разошлись", "утверждало")

#: `Ran 1345 tests` — форма однозначная, ни с чем не путается.
RAN = re.compile(r"Ran\s+(\d+)\s+tests")

#: «119 мутаций», «ломает 119 порогов», «100% (119/119)». Только эти три
#: оборота: слово «порог» в документах чаще значит число вроде 0.30, и ловить
#: его целиком означало бы завалить проверку ложными тревогами. Ложная тревога
#: тут дороже пропуска — тест, кричащий на исправном тексте, выключают.
MUTATIONS = (re.compile(r"(\d+)\s+мутаци"),
             re.compile(r"ломает\s+\*{0,2}(\d+)\*{0,2}\s+порог"),
             re.compile(r"портит\s+\*{0,2}(\d+)\*{0,2}\s+порог"),
             re.compile(r"покрытие:?\s*100%\s*\((\d+)/(\d+)\)"))


def _docs() -> list:
    out = [ROOT / "README.md"]
    out += sorted((ROOT / "docs").glob("*.md"))
    return [p for p in out if p.exists()]


def _historical(line: str) -> bool:
    low = line.lower()
    return any(m in low for m in HISTORICAL)


class TheSizeOfTheSuiteLivesInExactlyOnePlace(unittest.TestCase):

    def test_no_other_document_states_it(self):
        offenders = []
        for p in _docs():
            rel = p.relative_to(ROOT).as_posix()
            if rel == HOME:
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if RAN.search(line) and not _historical(line):
                    offenders.append(f"{rel}:{i}: {line.strip()[:90]}")
        self.assertEqual(
            offenders, [],
            f"размер набора записан вне {HOME} и без пометки, что это прошлое. "
            f"Его нельзя вывести из кода — только прогнать, — поэтому всякая "
            f"копия устареет молча, и уже устаревала восемь раз: {offenders}")

    def test_the_one_place_exists_and_carries_the_command(self):
        p = ROOT / HOME
        self.assertTrue(p.exists(), f"{HOME} исчез — числу негде жить")
        text = p.read_text(encoding="utf-8")
        self.assertIn("unittest discover", text,
                      "число есть, а команды рядом нет — пересчитать нечем")

    def test_the_detector_would_have_caught_the_defect_it_was_written_for(self):
        """Сторож, который не умеет краснеть, — украшение."""
        self.assertTrue(RAN.search("PASS  тесты          Ran 1336 tests"))
        self.assertFalse(_historical("PASS  тесты          Ran 1336 tests"))

    def test_it_lets_the_history_through(self):
        for line in ("Здесь стояло «Ran 249 tests»; в день правки прогон дал 631",
                     "~~Ran 1283 tests~~ — из первой редакции памятки",
                     "к концу той же смены — `Ran 824 tests`"):
            with self.subTest(line=line[:40]):
                self.assertTrue(_historical(line))


class TheMutationCountAgreesWithTheCode(unittest.TestCase):
    """Это число ВЫВОДИТСЯ, поэтому не запрещается, а сверяется."""

    def setUp(self):
        from ball_reel import codeaudit

        self.n = len(codeaudit.MUTATIONS)

    def test_every_document_that_names_it_names_the_right_one(self):
        offenders = []
        for p in _docs():
            rel = p.relative_to(ROOT).as_posix()
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if _historical(line):
                    continue
                for rx in MUTATIONS:
                    for m in rx.finditer(line):
                        for got in m.groups():
                            if int(got) != self.n:
                                offenders.append(
                                    f"{rel}:{i}: {got} вместо {self.n} — "
                                    f"{line.strip()[:70]}")
        self.assertEqual(
            offenders, [],
            f"мутаций в коде {self.n}, а документы говорят иное. Это число "
            f"выводится из `len(codeaudit.MUTATIONS)` и расходиться с ним не "
            f"имеет права: {offenders}")

    def test_at_least_one_document_actually_names_it(self):
        """Иначе предыдущий тест зеленеет на пустом множестве.

        Форма частая и незаметная: проверка «все найденные верны» проходит,
        когда не найдено ничего, и молчит ровно тогда, когда сломался поиск.
        """
        seen = 0
        for p in _docs():
            for line in p.read_text(encoding="utf-8").splitlines():
                if _historical(line):
                    continue
                seen += sum(1 for rx in MUTATIONS if rx.search(line))
        self.assertGreater(seen, 2,
                           "ни один документ не называет число мутаций — "
                           "скорее всего сломались выражения, а не документы")

    def test_the_detector_catches_a_wrong_number(self):
        line = "ломает **56 порогов** в копии исходника"
        found = [int(g) for rx in MUTATIONS for m in rx.finditer(line)
                 for g in m.groups()]
        self.assertEqual(found, [56])
        self.assertNotEqual(found[0], self.n)


if __name__ == "__main__":
    unittest.main()
