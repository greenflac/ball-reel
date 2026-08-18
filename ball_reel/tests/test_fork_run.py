"""Сквозной путь форка: отчёт по шагам, а не «получилось».

Главное, что здесь сторожится, — что путь НЕ ОБЪЯВЛЯЕТ УСПЕХ, когда половина
шагов не смогла отработать. Прогон без весов, без ffmpeg и без генерации — это
нормальное состояние спринта, и он обязан читаться как «не смогли», а не как
зелёный сквозной путь.
"""

from __future__ import annotations

import json
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
        прошло ОБА прежних аудита и полный прогон с мутациями. Разбор дерева
        её больше НЕ ВИДИТ: она переехала внутрь `audit_wrapper`, и текст
        `fork_run` её не упоминает. Поэтому здесь проверяется свидетельство, а
        не намерение (Е2) — что вложенный аудит зовётся и его числа доезжают
        до отчёта. Прямая проверка того же на вердикте шага —
        `test_a_lock_naming_another_file_makes_the_step_fail`.
        """
        from ball_reel import fork_comfy

        self.assertIn("audit_wrapper",
                      self._called_attributes().get("fork_comfy", set()))
        got = fork_comfy.audit_wrapper(fork_comfy.derive_wrapper())
        self.assertIn("weights", got,
                      "аудит обёртки больше не включает проверку весов — "
                      "тогда её в конвейере нет вовсе")


def _photo(dir_path: Path, name: str = "p.png") -> Path:
    import numpy as np
    from PIL import Image

    path = Path(dir_path) / name
    Image.fromarray(
        (np.random.rand(64, 64, 3) * 255).astype("uint8")).save(path)
    return path


class TheGraphStepProducesTheGraphThatWillActuallyRun(unittest.TestCase):
    """Два производителя графа в модуле, один конвейер — Е1.

    Пока сводящий проход звал `derive` (штатные ноды), решение владельца про
    обёртку лежало в документах, а поедет по нему было бы нечему: цикл по
    окнам у штатных нод надо строить в графе руками, и мы его не строили.
    Пять секунд при 30 к/с — 150 кадров, то есть КАЖДЫЙ ролик длиннее окна 77.
    """

    def _graph_step(self, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.run(_photo(tmp), [], Path(tmp) / "out", **kw)
        return next(s for s in got["steps"] if s["step"] == "граф")

    def test_the_graph_is_the_wrapper_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            fork_run.run(_photo(tmp), [], Path(tmp) / "out")
            written = json.loads(
                (Path(tmp) / "out" / "fork_graph.json").read_text(
                    encoding="utf-8"))
        # ТИПЫ УЗЛОВ, А НЕ ПОДСТРОКА В ФАЙЛЕ. Первая версия искала имя по
        # всему тексту и покраснела на СПРАВКЕ о происхождении: граф несёт в
        # себе цитаты из исходников, и в одной из них штатное имя упомянуто
        # честно. Проверка по подстроке отвечала не на тот вопрос.
        types = {n.get("type") for n in written["nodes"]}
        # Литералы, не импорт из проверяемого модуля (Т2).
        self.assertIn("WanVideoAnimateEmbeds", types,
                      "сводящий проход собрал не тот граф, который решено "
                      "гнать на карте")
        self.assertNotIn("WanAnimateToVideo", types,
                         "в конвейере остался граф на штатных нодах")

    def test_a_clean_run_passes(self):
        self.assertEqual(self._graph_step()["outcome"], PASS)

    def test_a_lock_naming_another_file_makes_the_step_fail(self):
        """Негативный контроль (И5) на месте прежней проверки весов.

        Структурно чистый граф с разъехавшимися весами — не «граф готов».
        Раньше это сторожил отдельный вызов `audit_weights` из этого файла;
        теперь он живёт ВНУТРИ `audit_wrapper`, и проверять надо не текст
        исходника, а то, что расхождение доезжает до вердикта шага (Е2).
        """
        from ball_reel import fork_comfy

        lock = fork_comfy.load_lock()
        broken = json.loads(json.dumps(lock))
        for entry in broken.get("files", broken.get("weights", [])):
            if str(entry.get("path", entry.get("name", ""))).endswith(".gguf"):
                key = "path" if "path" in entry else "name"
                entry[key] = str(entry[key]) + ".ПОДМЕНЕНО"
                break
        else:
            self.skipTest("в локе не нашлось .gguf — проверка ищет не то")
        self.assertEqual(self._graph_step(lock=broken)["outcome"], FAIL)

    def test_the_weight_and_format_numbers_reach_the_report(self):
        note = self._graph_step()["note"]
        for piece in ("весов разобрано", "форматов проверено", "окнами по"):
            with self.subTest(piece=piece):
                self.assertIn(piece, note)

    def test_the_length_the_owner_asked_for_reaches_the_graph(self):
        """5 с при 30 к/с — 150 кадров, и это больше одного окна."""
        note = self._graph_step()["note"]
        self.assertIn("150 кадров", note)
        self.assertIn("окон 2", note,
                      "ролик короче окна — значит окна опять не считаются")


class ZeroChecksIsNotSuccess(unittest.TestCase):
    """Р2, найденное прогоном этого же прохода, а не рассуждением.

    Шаг условий печатал «годно» под текстом «снято 0 из 0 кадров»: сравнение
    `rendered == total` на пустом списке даёт истину. Ровно та ошибка, ради
    которой в проекте заведено правило про ноль проверок.
    """

    def _step(self, name, frames):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.run(_photo(tmp), frames, Path(tmp) / "out")
        return next((s for s in got["steps"] if s["step"] == name), None)

    def test_no_driving_frames_makes_conditions_unmeasured(self):
        got = self._step("условия", [])
        self.assertEqual(got["outcome"], UNMEASURED,
                         "ноль снятых условий прочтено как успех")

    def test_the_note_says_the_zero_out_loud(self):
        self.assertIn("0 из 0", self._step("условия", [])["note"])


class InputsThatExistButCannotBeRead(unittest.TestCase):
    """Файл есть, а изображения в нём нет. П2: ловить это на входах.

    Испытательное описание кладёт заглушку с расширением `.png`. Проверка
    существования её пропускала, и путь падал трассировкой на шаге корзины —
    дорогой шаг оплачивался ради ошибки, читаемой за миллисекунду.
    """

    def _inputs(self, payload: bytes):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "photo.png"
            bad.write_bytes(payload)
            got = fork_run.run(bad, [], Path(tmp) / "out")
        return got["steps"][1]

    def test_a_stub_with_a_png_name_fails_the_inputs_step(self):
        got = self._inputs(b"SYNTHETIC-NOT-A-PHOTO")
        self.assertEqual(got["step"], "входы")
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("не читаются", got["note"])

    def test_the_run_stops_right_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "photo.png"
            bad.write_bytes(b"SYNTHETIC-NOT-A-PHOTO")
            got = fork_run.run(bad, [], Path(tmp) / "out")
        self.assertEqual(len(got["steps"]), 2,
                         "путь поехал дальше по нечитаемому входу")

    def test_a_real_image_passes_the_same_step(self):
        """Негативный контроль (И5): проверка обязана уметь и пропускать."""
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.run(_photo(tmp), [], Path(tmp) / "out")
        self.assertEqual(got["steps"][1]["outcome"], PASS)


class TheForksTakenOutOfTheEntryPoint(unittest.TestCase):
    """Т5. Обе развилки внутри `run` пережили мутацию, и это не случайность.

    Чтобы достать их прогоном, нужны сотни настоящих кадров и работающий
    детектор позы. Такого прогона в сьюте нет и не будет — значит развилки
    были недостижимы, а «покрыты» они выглядели.
    """

    def test_thirty_and_twentyfour_give_different_lengths(self):
        """Мутация частоты пережила весь сьют, пока это считалось внутри run."""
        self.assertEqual(fork_run.seconds_for(240, fps=30), 8.0)
        self.assertEqual(fork_run.seconds_for(240, fps=24), 10.0)

    def test_the_default_rate_is_the_one_the_owner_chose(self):
        # Литерал, не импорт (Т2): 24 к/с забракованы владельцем за реализм.
        self.assertEqual(fork_run.seconds_for(240), 8.0)

    def test_a_short_driving_is_pulled_up_to_the_floor(self):
        self.assertEqual(fork_run.seconds_for(30), 5.0)

    def test_a_long_driving_is_cut_to_the_ceiling(self):
        self.assertEqual(fork_run.seconds_for(3000), 10.0)

    def test_no_frames_at_all_is_the_floor_not_zero(self):
        self.assertEqual(fork_run.seconds_for(0), 5.0)

    def test_conditions_can_pass_and_that_side_is_checked_too(self):
        """Обратная сторона Р2 (И5): прибор обязан уметь и говорить «годно»."""
        self.assertEqual(
            fork_run.conditions_verdict({"rendered": 150, "total": 150}), PASS)

    def test_conditions_partially_taken_are_unmeasured(self):
        self.assertEqual(
            fork_run.conditions_verdict({"rendered": 3, "total": 150}),
            UNMEASURED)

    def test_conditions_taken_on_nothing_are_a_failure(self):
        self.assertEqual(
            fork_run.conditions_verdict({"rendered": 0, "total": 150}), FAIL)

    def test_more_rendered_than_given_is_a_broken_counter_not_a_pass(self):
        """Единственная мутация, пережившая сьют, — и она была не пустой."""
        self.assertEqual(
            fork_run.conditions_verdict({"rendered": 151, "total": 150}),
            UNMEASURED)

    def test_zero_out_of_zero_is_never_success(self):
        self.assertEqual(
            fork_run.conditions_verdict({"rendered": 0, "total": 0}),
            UNMEASURED)


class TheOperatorEntryPoint(unittest.TestCase):
    """Ради чего писался слой оператора: новый драйвинг без входа в код."""

    def _bench(self, tmp):
        from ball_reel import fork_template

        return fork_template.make_bench(Path(tmp) / "bench")

    def _bench_with_a_real_photo(self, tmp):
        """Стенд кладёт заглушку `SYNTHETIC-NOT-A-PHOTO`, и путь честно на ней
        останавливается. Чтобы дойти до масок, фотографию надо подменить
        настоящей — заглушка тут не годится по построению, а не по недосмотру.
        """
        card = self._bench(tmp)
        _photo(card.parent, "photo.png")
        return card

    def test_a_broken_description_comes_back_as_a_report_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "нет.json"
            got = fork_run.from_template(bad, Path(tmp) / "out")
        self.assertEqual(got["steps"][0]["step"], "описание")
        self.assertEqual(got["outcome"], FAIL)

    def test_the_description_step_comes_before_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.from_template(self._bench(tmp), Path(tmp) / "out")
        self.assertEqual([s["step"] for s in got["steps"]][:2],
                         ["описание", "плечо"])

    def test_the_arm_named_in_the_card_is_the_one_used(self):
        """Иначе в карточке стоит одно, а маску расширяет роутер по-своему."""
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.from_template(self._bench_with_a_real_photo(tmp),
                                         Path(tmp) / "out")
        arm = next(s for s in got["steps"] if s["step"] == "плечо")
        self.assertEqual(arm["outcome"], PASS)
        self.assertIn("narrow", arm["note"])
        self.assertIn("роутер не спрашивался", arm["note"])
        # И ЧТО ЭТО ЧИСЛО ДОЕХАЛО ДО МАСКИ. Без этой строки мутация
        # «плечо из карточки игнорируется» переживала весь сьют: шаг «плечо»
        # печатал правильное имя, а маску расширял кто хотел. У `narrow`
        # расширение 0 px, у умолчания — блок 32, и они различимы в отчёте.
        mask = next(s for s in got["steps"] if s["step"] == "маски")
        self.assertIn("расширение 0px", mask["note"],
                      "в карточке narrow, а маска расширена не на ноль")

    def test_a_bench_card_is_refused_in_production(self):
        """Негативный контроль: прогнать клиента по заглушкам — это провал."""
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.from_template(self._bench(tmp), Path(tmp) / "out",
                                         production=True)
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(len(got["steps"]), 1,
                         "боевой прогон поехал по испытательному описанию")

    def test_the_same_card_is_allowed_when_production_is_not_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_run.from_template(self._bench(tmp), Path(tmp) / "out")
        self.assertEqual(got["steps"][0]["outcome"], PASS)


if __name__ == "__main__":
    unittest.main()
