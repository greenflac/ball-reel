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

from ball_reel import fork_comfy, fork_run
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED


class MissingInputsAreCaughtFirstAndCheaply(unittest.TestCase):
    """П2: отсутствующий файл ловится за миллисекунды, а не после условий."""

    def test_a_missing_photo_fails_the_first_step_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.run(Path(tmp) / "нет.png", [], tmp)
        self.assertEqual(got["steps"][1]["step"], "входы")
        self.assertEqual(got["steps"][1]["outcome"], FAIL)
        self.assertEqual(len(got["steps"]), 2,
                         "путь пошёл дальше по отсутствующему входу — значит "
                         "дорогие шаги оплачиваются до дешёвой проверки")

    def test_the_failing_step_names_the_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.run(Path(tmp) / "нет.png", [], tmp)
        self.assertIn("нет.png", got["steps"][1]["note"])

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

    def test_the_preflight_comes_first_of_all(self):
        """Самая дешёвая проверка и та, что решает, поедет ли вообще что-то."""
        self.assertEqual(fork_run.STEPS[0], "предполёт")

    def test_inputs_come_before_any_real_work(self):
        order = list(fork_run.STEPS)
        self.assertLess(order.index("входы"), order.index("условия"))

    def test_conditions_and_masks_come_before_the_graph(self):
        order = list(fork_run.STEPS)
        self.assertLess(order.index("условия"), order.index("граф"))
        self.assertLess(order.index("маски"), order.index("граф"))

    def test_the_bucket_is_chosen_before_anything_expensive(self):
        """Корзина решает, какой темплейт поедет. Узнать это после снятия
        условий по сотне кадров значит, возможно, снять их зря."""
        order = list(fork_run.STEPS)
        self.assertLess(order.index("корзина"), order.index("условия"))
        self.assertLess(order.index("корзина"), order.index("маски"))

    def test_the_axes_that_need_a_real_output_come_last(self):
        order = list(fork_run.STEPS)
        for axis in ("протечка", "шов"):
            with self.subTest(axis=axis):
                self.assertGreater(order.index(axis), order.index("граф"))


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
        for module in ("fork_build_route", "fork_channels", "fork_comfy",
                       "fork_leak", "fork_lora_dataset", "fork_mask",
                       "fork_preflight", "fork_seam"):
            with self.subTest(module=module):
                self.assertIn(module, called,
                              f"{module} импортирован, но не вызван — импорт "
                              f"ради проверки достижимости оставляет модуль "
                              f"мёртвым")


class EveryForkFunctionTheSweepCallsActuallyExists(unittest.TestCase):
    """Дыра, найденная потоком B, а не мной, и закрытая здесь.

    `fork_run` звал `fork_leak.noise_floor`, которую поток B удалил по §1a.
    Все три теста выше падают на отсутствующем фото и ДО этой строки не
    доходят — то есть сводящий проход мог звать несуществующую функцию, а
    сьют оставался зелёным. Это ровно тот дефект, ради которого в проекте
    заведено правило про мёртвый код: он есть в отчёте, а в конвейере его нет.

    Сквозной прогон закрыл бы дыру честнее, но он требует весов DWPose,
    сегментации и десятков секунд на кадр. Разбор дерева стоит миллисекунды и
    ловит ТОТ ЖЕ класс: имя, которого больше нет. Меньше, чем прогон, но не
    ноль — а ноль здесь и был.
    """

    def _called_attributes(self) -> dict:
        import ast
        from collections import defaultdict

        src = Path(fork_run.__file__).read_text(encoding="utf-8")
        out = defaultdict(set)
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Attribute) and isinstance(node.value,
                                                              ast.Name):
                if node.value.id.startswith("fork_"):
                    out[node.value.id].add(node.attr)
        return out

    def test_no_call_points_at_a_name_that_was_removed(self):
        import importlib

        missing = []
        for module_name, attrs in self._called_attributes().items():
            module = importlib.import_module(f"ball_reel.{module_name}")
            for attr in sorted(attrs):
                if not hasattr(module, attr):
                    missing.append(f"{module_name}.{attr}")
        self.assertEqual(
            missing, [],
            f"сводящий проход зовёт то, чего в модуле нет: {missing}. "
            f"Так уже было с fork_leak.noise_floor после того, как §1a снял "
            f"кодековую планку — и сьют этого не заметил.")

    def test_the_check_would_have_caught_the_defect_it_was_written_for(self):
        """Сторож, который не умеет краснеть, — украшение."""
        import importlib

        module = importlib.import_module("ball_reel.fork_leak")
        self.assertFalse(hasattr(module, "noise_floor"),
                         "кодековая планка вернулась в fork_leak — тогда этот "
                         "тест больше не про тот дефект")
        self.assertTrue(hasattr(module, "FLOOR_FULL_LOOP"),
                        "имя, которым сводящий проход теперь пользуется, "
                        "исчезло — проверка ищет не то")

    def test_it_looks_at_more_than_one_module(self):
        """Иначе проверка зеленела бы, найдя один модуль и пропустив восемь."""
        seen = set(self._called_attributes())
        self.assertGreaterEqual(len(seen), 6, f"разобрано слишком мало: {seen}")

    def test_the_weight_audit_is_actually_called_by_the_sweep(self):
        """Проверка, не подключённая к пути, — мёртвый код в отчёте.

        `audit_weights` написана после того, как расхождение графа с локом
        прошло ОБА прежних аудита и полный прогон с мутациями. Если она снова
        окажется вне сводящего прохода, дефект вернётся тем же способом.
        """
        self.assertIn("audit_weights",
                      self._called_attributes().get("fork_comfy", set()))


class TheGraphStepJudgesWeightsAndStructureTogether(unittest.TestCase):
    """Структурно чистый граф с разъехавшимися весами — не «граф готов»."""

    def _graph_step(self, quant):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp) / "p.png"
            Image.fromarray(
                (np.random.rand(64, 64, 3) * 255).astype("uint8")).save(photo)
            got = fork_run.run(photo, [], Path(tmp) / "out", quant=quant)
        return next(s for s in got["steps"] if s["step"] == "граф")

    def test_the_open_fork_makes_the_graph_step_fail(self):
        self.assertEqual(self._graph_step(fork_comfy.QUANT_TEMPLATE)["outcome"], FAIL,
                         "шаг «граф» зелен при графе, который просит не те "
                         "файлы, что качает лок")

    def test_the_decided_fork_makes_it_pass(self):
        self.assertEqual(self._graph_step("gguf")["outcome"], PASS)

    def test_both_notes_reach_the_report(self):
        note = self._graph_step("gguf")["note"]
        self.assertIn("ЧИСТО", note, "потерян отчёт структурной проверки")
        self.assertIn("ВЕСА:", note, "потерян отчёт проверки весов")
        self.assertIn("разобрано загрузчиков", note)


if __name__ == "__main__":
    unittest.main()
