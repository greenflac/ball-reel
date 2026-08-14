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
