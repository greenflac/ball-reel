"""Сквозной путь форка: отчёт по шагам, а не «получилось».

Главное, что здесь сторожится, — что путь НЕ ОБЪЯВЛЯЕТ УСПЕХ, когда половина
шагов не смогла отработать. Прогон без весов, без ffmpeg и без генерации — это
нормальное состояние спринта, и он обязан читаться как «не смогли», а не как
зелёный сквозной путь.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ball_reel import fork_run
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED


class MissingInputsAreCaughtFirstAndCheaply(unittest.TestCase):
    """П2: отсутствующий файл ловится за миллисекунды, а не после условий."""

    def test_a_missing_photo_fails_the_first_step_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.run(Path(tmp) / "нет.png", [], tmp)
        self.assertEqual(got["steps"][0]["step"], "входы")
        self.assertEqual(got["steps"][0]["outcome"], FAIL)
        self.assertEqual(len(got["steps"]), 1,
                         "путь пошёл дальше по отсутствующему входу — значит "
                         "дорогие шаги оплачиваются до дешёвой проверки")

    def test_the_failing_step_names_the_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.run(Path(tmp) / "нет.png", [], tmp)
        self.assertIn("нет.png", got["steps"][0]["note"])

    def test_every_step_reports_its_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.run(Path(tmp) / "нет.png", [], tmp)
        for s in got["steps"]:
            with self.subTest(step=s["step"]):
                self.assertIsInstance(s["seconds"], float)


class TheReportIsNumbersNotAFlag(unittest.TestCase):

    def _report(self, outcomes):
        steps = [{"step": f"ш{i}", "outcome": o, "note": "", "seconds": 0.0}
                 for i, o in enumerate(outcomes)]
        return fork_run._report(steps, Path("."))

    def test_a_failed_step_makes_the_whole_run_failed(self):
        got = self._report([PASS, FAIL, UNMEASURED])
        self.assertEqual(got["outcome"], FAIL)

    def test_an_unmeasured_step_is_not_swallowed_by_passes(self):
        got = self._report([PASS, PASS, UNMEASURED])
        self.assertEqual(got["outcome"], UNMEASURED,
                         "«не смогли» свёрнуто в успех — Р1 нарушено")

    def test_all_passing_is_the_only_way_to_pass(self):
        self.assertEqual(self._report([PASS, PASS])["outcome"], PASS)

    def test_the_counts_are_printed(self):
        got = self._report([PASS, FAIL, UNMEASURED, UNMEASURED])
        self.assertEqual((got["passed"], got["failed"], got["unmeasured"]),
                         (1, 1, 2))
        for piece in ("пройдено 1", "провалено 1", "не смогли 2"):
            self.assertIn(piece, got["note"])

    def test_the_note_says_out_loud_there_was_no_generation(self):
        got = self._report([PASS])
        self.assertIn("ГЕНЕРАЦИИ НЕ БЫЛО", got["note"])
        self.assertIn("продуктовое заявление им не проверяется", got["note"])


class TheStepOrderPutsCheapBeforeExpensive(unittest.TestCase):

    def test_inputs_come_first_and_leak_last(self):
        self.assertEqual(fork_run.STEPS[0], "входы")
        self.assertEqual(fork_run.STEPS[-1], "протечка")

    def test_conditions_and_masks_come_before_the_graph(self):
        order = list(fork_run.STEPS)
        self.assertLess(order.index("условия"), order.index("граф"))
        self.assertLess(order.index("маски"), order.index("граф"))


class TheModulesAreActuallyWiredIn(unittest.TestCase):
    """Смысл существования этого файла для `test_reachable`.

    Импорт ради галочки прошёл бы проверку достижимости и оставил модуль
    мёртвым. Здесь проверяется, что путь их ЗОВЁТ.
    """

    def test_the_entry_point_calls_each_fork_module(self):
        import ast

        src = Path(fork_run.__file__).read_text(encoding="utf-8")
        called = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Attribute) and isinstance(
                    node.value, ast.Name):
                called.add(node.value.id)
        for module in ("fork_channels", "fork_comfy", "fork_leak", "fork_mask"):
            with self.subTest(module=module):
                self.assertIn(module, called,
                              f"{module} импортирован, но не вызван — импорт "
                              f"ради проверки достижимости оставляет модуль "
                              f"мёртвым")


if __name__ == "__main__":
    unittest.main()
