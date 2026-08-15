"""Аудит проверяет тесты — значит кто-то должен проверять аудит.

Дефект, ради которого файл появился, был тихим и дорогим: мутации гоняются не в
рабочем дереве, а в КОПИИ пакета, и данные, лежащие в корне репозитория, в
копию не попадали. Тесты, работающие на настоящих кадрах, там просто
пропускались — то есть во время каждой мутации не сторожили ничего. ИЗМЕРЕНО:
копия пропускала 8 тестов против 1 в дереве, и все семь замолчавших бежали по
живым данным.

Направление ошибки было безопасным (лишние ВЫЖИВШИЕ, а не лишние убитые), но
аудит при этом печатал «покрытие 100%», описывая охрану, которой не было.

ЗДЕСЬ НЕ ВЫЗЫВАЕТСЯ `copy_is_as_strong_as_the_tree`: она гоняет весь набор
дважды, а мы внутри этого набора — вышла бы рекурсия и минуты вместо секунд.
Проверяется то, из чего она состоит.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TheCopyUnderMutationIsNotWeakerThanTheTree(unittest.TestCase):
    def setUp(self):
        from ball_reel import codeaudit

        self.c = codeaudit
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_the_data_the_live_tests_need_is_linked_into_the_copy(self):
        """`kit/` — не украшение: на нём стоят тесты по настоящим кадрам."""
        self.assertIn("kit", self.c.DATA_LINKS)
        pkg = self.c._stage_copy(self.dir)
        self.assertTrue(pkg.is_dir())
        for name in self.c.DATA_LINKS:
            if Path(name).exists():
                with self.subTest(link=name):
                    self.assertTrue((self.dir / name).exists(),
                                    f"{name} не доехал до копии")

    def test_fixtures_are_linked_rather_than_copied(self):
        # Копировать их дорого, а без них офлайн-тесты молчат.
        pkg = self.c._stage_copy(self.dir)
        self.assertTrue((pkg / "fixtures").exists())

    def test_root_level_experiment_frames_reach_the_copy(self):
        # `marktest_ref.png` и соседи лежат россыпью в корне, а не в папке;
        # именно на них стоит тест, отличающий одетую ногу от голой руки.
        pkg = self.c._stage_copy(self.dir)
        self.assertTrue(pkg.exists())
        for png in list(Path(".").glob("*.png"))[:3]:
            with self.subTest(png=png.name):
                self.assertTrue((self.dir / png.name).exists())

    def test_a_mutation_run_stops_at_the_first_red_test(self):
        """failfast — условие того, что аудит вообще запускают.

        ИЗМЕРЕНО: убитый мутант с полным прогоном стоит 16.2 с, с failfast —
        0.3 с. При 88 мутациях и 921 тесте полный прогон не влезал в
        собственный таймаут, а аудит, который перестают запускать, не сторожит
        ничего. Информации не теряется: «убит» значит «покраснел хотя бы один»,
        и первого достаточно.
        """
        import inspect

        src = inspect.getsource(self.c.run_mutation)
        self.assertIn('"-f"', src)

    def test_the_self_check_stops_the_audit_instead_of_warning(self):
        # Предупреждение в потоке из девяноста строк не читают. Слабая копия
        # обязана останавливать аудит, а не сопровождать его.
        import inspect

        src = inspect.getsource(self.c.main)
        self.assertIn("copy_is_as_strong_as_the_tree", src)
        self.assertIn("ОСТАНОВЛЕНО", src)


class TheCloneIsNotWeakerThanTheAuthorsDisk(unittest.TestCase):
    """Второй дефект той же формы, и копия для мутаций его не ловит.

    Три живых теста ходили в каталог `ref_frames`, которого в индексе git нет
    ни одним файлом. У автора он лежал на диске, тесты были зелёными; в клоне
    пропускались. ИЗМЕРЕНО: 22 пропуска против 7 — пятнадцать сторожей спали у
    всех, кроме одного человека, а прогон при этом печатал `OK`.

    Проверка копии для мутаций этого не видит ПО УСТРОЙСТВУ: `_stage_copy`
    линкует данные из рабочего дерева, то есть воспроизводит диск автора.
    Вопроса два — «доехали ли данные до копии» и «есть ли они в индексе», — и
    отвечать на них надо порознь.

    Полный `clone_is_as_strong_as_the_tree` здесь не зовётся по той же причине,
    что и его сосед: он гоняет весь набор, а мы внутри набора.
    """

    def setUp(self):
        from ball_reel import codeaudit

        self.c = codeaudit
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _staged(self) -> bool:
        self.c._stage_clone(self.dir)
        return (self.dir / "ball_reel" / "codeaudit.py").exists()

    def test_the_staged_tree_holds_what_the_index_holds(self):
        if not self._staged():
            self.skipTest("не рабочее дерево git")
        self.assertTrue((self.dir / "demo" / "bench" / "pose_96.json").exists(),
                        "стенд лупа не доехал — значит его нет в индексе")

    def test_the_staged_tree_does_NOT_hold_what_only_the_disk_holds(self):
        """Суть проверки. Если сюда попадает неотслеживаемое — она бесполезна.

        Неотслеживаемый файл СОЗДАЁТСЯ здесь же, а не берётся из дерева. Первая
        редакция смотрела на `ref_frames` — тот самый каталог, из-за которого
        всё завелось, — и пропускалась там, где его нет. То есть в клоне. То
        есть добавляла восьмой пропуск к семи и ломала ту самую сверку, ради
        которой написана: `clone_is_as_strong_as_the_tree` вернула
        `в клоне пропущено 8 против 7`. Сторож, роняющий собственную меру,
        хуже отсутствующего.
        """
        import os

        if not self._staged():
            self.skipTest("не рабочее дерево git")
        root = Path(__file__).resolve().parents[2]
        probe = root / "codeaudit_clone_probe.tmp"
        probe.write_text("не в индексе\n", encoding="utf-8")
        self.addCleanup(lambda: probe.unlink(missing_ok=True))
        again = Path(self.tmp.name) / "second"
        again.mkdir()
        cwd = os.getcwd()
        try:
            os.chdir(root)
            self.c._stage_clone(again)
        finally:
            os.chdir(cwd)
        self.assertTrue((again / "ball_reel" / "codeaudit.py").exists())
        self.assertFalse((again / probe.name).exists(),
                         "в клон попало то, чего нет в индексе — проверка "
                         "сравнивает диск автора с диском автора")

    def test_the_clone_gets_an_index_of_its_own(self):
        """Без индекса проверка врёт В СВОЮ ПОЛЬЗУ.

        Тесты, читающие состав репозитория, вне рабочего дерева git
        пропускаются — ровно два лишних пропуска, измерено. Голый каталог с
        файлами показал бы 9 против 7 там, где настоящий клон даёт 7, и аудит
        останавливался бы на дефекте, которого нет.
        """
        if not self._staged():
            self.skipTest("не рабочее дерево git")
        self.assertTrue((self.dir / ".git").exists(),
                        "у клона нет индекса — проверка насчитает лишние "
                        "пропуски и остановит аудит без причины")

    def test_the_self_check_stops_the_audit_instead_of_warning(self):
        import inspect

        src = inspect.getsource(self.c.main)
        self.assertIn("clone_is_as_strong_as_the_tree", src)
        i = src.index("clone_is_as_strong_as_the_tree")
        self.assertIn("ОСТАНОВЛЕНО", src[i:],
                      "слабый клон только печатается, но аудит не роняет")

    def test_it_says_what_to_do_and_forbids_the_easy_way_out(self):
        """Самый вероятный «ремонт» — смягчить тест, и он же самый вредный.

        Читается исходник, а не результат вызова: вызов гоняет весь набор
        дважды, и звать его изнутри набора значит устроить рекурсию.
        """
        import inspect

        src = inspect.getsource(self.c.clone_is_as_strong_as_the_tree)
        fail = src[src.index("return False"):]
        self.assertIn("git add", fail, "отказ не говорит, чем лечить")
        self.assertIn("Смягчать", fail,
                      "отказ не запрещает самый вероятный ложный ремонт")


class TheExemptionListIsNotAPlaceToHideThings(unittest.TestCase):
    """`TREE_ONLY_SKIPS` вычитается из числа пропусков — значит он ослабляет.

    Ослабление тут законное: три-четыре проверки читают состав репозитория, и в
    копии без индекса им нечего делать ни при какой мутации. Но список
    вычитается ПО ДЛИНЕ, а имена никто не сверял, и устаревшее имя молча
    ослабляло бы самопроверку ровно на единицу — навсегда и незаметно.
    """

    def setUp(self):
        from ball_reel import codeaudit

        self.c = codeaudit
        self.src = "\n".join(
            p.read_text(encoding="utf-8")
            for p in Path(__file__).resolve().parent.glob("test_*.py"))

    def test_every_exempted_name_is_a_test_that_exists(self):
        missing = [n for n in self.c.TREE_ONLY_SKIPS
                   if f"def test_{n}" not in self.src]
        self.assertEqual(
            missing, [],
            "имя в списке исключений ни на что не указывает — вычитание "
            f"осталось, а сторожа нет: {missing}")

    def test_every_exemption_explains_itself(self):
        for name, why in self.c.TREE_ONLY_SKIPS.items():
            with self.subTest(name=name):
                self.assertGreater(
                    len(why), 40,
                    "попасть в список — это заявление «мутация этот тест не "
                    "затрагивает», и оно должно стоить осознанной строки")


class TheMutationTableIsWellFormed(unittest.TestCase):
    def setUp(self):
        from ball_reel import codeaudit

        self.c = codeaudit

    def test_no_two_entries_mutate_the_same_constant_to_the_same_value(self):
        # Дубликат гоняется дважды и стоит вдвое, ничего не проверяя сверх.
        seen = [(m, n, repr(v)) for m, n, v, _ in self.c.MUTATIONS]
        self.assertEqual(len(seen), len(set(seen)))

    def test_every_entry_names_what_breaks_in_plain_words(self):
        # Строка «ВЫЖИЛА порог X» без объяснения не говорит, что чинить.
        for module, name, _, meaning in self.c.MUTATIONS:
            with self.subTest(const=f"{module}.{name}"):
                self.assertTrue(meaning and len(meaning) > 20)

    def test_the_modules_named_in_the_table_actually_exist(self):
        # Опечатка в имени модуля даёт «нет файла» — то есть мутация, которая
        # не мутирует, и она молча считается ВЫЖИВШЕЙ.
        import importlib

        for module, _, _, _ in self.c.MUTATIONS:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_the_constants_named_in_the_table_actually_exist(self):
        import importlib

        for module, name, _, _ in self.c.MUTATIONS:
            with self.subTest(const=f"{module}.{name}"):
                self.assertTrue(hasattr(importlib.import_module(module), name),
                                f"{name} нет в {module}: мутация вхолостую")


class TheSkipCounterReadsTheSummaryNotTheLastLine(unittest.TestCase):
    """Ложная самопроверка выключает аудит так же надёжно, как сломанная.

    ИЗМЕРЕНО: полный прогон аудита остановился на строке «в копии пропущено 4
    тестов против 0 в дереве» — и до мутаций не дошёл вовсе. Расхождения не
    было: итог unittest (`OK (skipped=4)`) не является последней строкой
    stderr. После него печатают mediapipe, absl и TensorFlow Lite, а один раз
    ещё и traceback из `PoseLandmarker.__del__` при сборке мусора. У дерева
    хвост оказался длиннее, чем у копии, — отсюда 0 против 4.
    """

    def setUp(self):
        import importlib

        self.c = importlib.import_module("ball_reel.codeaudit")

    def test_plain_summary(self):
        self.assertEqual(self.c._skipped("Ran 10 tests\n\nOK (skipped=4)\n"), 4)

    def test_noise_after_the_summary_does_not_hide_it(self):
        """Ровно тот случай, на котором аудит встал."""
        out = ("OK (skipped=4)\n"
               "I0000 00:00 migration_state_tracking.cc:24] Migration not enabled\n"
               "Exception ignored in: <function PoseLandmarker.__del__>\n"
               "TypeError: 'NoneType' object is not callable\n")
        self.assertEqual(self.c._skipped(out), 4)

    def test_summary_without_skips_is_zero_not_a_miss(self):
        self.assertEqual(self.c._skipped("OK\nI0000 шум\nещё шум\n"), 0)

    def test_failed_summary_is_read_too(self):
        self.assertEqual(self.c._skipped("FAILED (errors=1, skipped=2)\nшум\n"), 2)
        self.assertEqual(self.c._skipped("FAILED (skipped=3, errors=1)\nшум\n"), 3)

    def test_no_summary_at_all_is_zero(self):
        self.assertEqual(self.c._skipped(""), 0)
        self.assertEqual(self.c._skipped("всё сломалось до запуска"), 0)

    def test_the_tree_and_the_copy_are_counted_by_the_same_function(self):
        """Два счётчика для одного числа разошлись бы снова, и снова молча."""
        import inspect

        src = inspect.getsource(self.c)
        self.assertEqual(src.count("def _skipped"), 1)
