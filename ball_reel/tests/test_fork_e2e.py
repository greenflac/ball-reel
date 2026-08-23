"""Сквозной стенд: весь путь на подставных функциях, без сети и без денег.

ЧТО ЗДЕСЬ ГЛАВНОЕ. Не «функции работают по отдельности», а три свойства
прогона, каждое из которых уже стоило нам прогона или денег:

  1. ВЕСЬ путь идёт на подставных функциях. Сеть в этом файле физически
     перекрыта (`_no_network`), а не обещана комментарием (Т4): иначе первый же
     недосмотр в умолчании увёл бы тест в платный вызов.
  2. Упавший внешний вызов даёт `не смогли проверить`, а НЕ `не годно`. Это
     ровно то место, где третий исход схлопывали в обе стороны (Р1), и оба раза
     это стоило прогонов.
  3. Ступени печатаются ПО ХОДУ. Проверяется наблюдаемо: подставной стилизатор
     смотрит на журнал в момент своего вызова и обязан увидеть там первую
     ступень. Прогон, печатающий всё в конце, здесь краснеет.

Ожидаемые числа — ЛИТЕРАЛЫ (Т2): 0.35, 3.0, 0.05, 0/1/2. Импортированное
ожидание уехало бы вместе с кодом и промолчало.
"""

from __future__ import annotations

import io
import socket
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ball_reel import fork_e2e as E
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED


class _Blocked(RuntimeError):
    """Сеть в тестах этого файла запрещена раннером, а не договорённостью."""


def _no_network():
    def deny(*a, **k):
        raise _Blocked("сеть в тестах запрещена: подставь функцию, а не ходи наружу")
    return mock.patch.object(socket, "socket", deny)


def _files(root: Path) -> dict:
    """Три входа: настоящие файлы на диске, но крошечные и без содержимого.

    Содержимое не нужно: все приборы в этих тестах подставные. Файлы нужны,
    потому что ступень 1 меряет именно наличие и размер.
    """
    paths = {}
    for name in ("client.png", "style.png", "driving.mp4"):
        p = root / name
        p.write_bytes(b"\x00" * 64)
        paths[name.split(".")[0]] = p
    return paths


def _probe_ok(path):
    return {"outcome": PASS, "fps": 30.0, "frames": 373, "width": 960,
            "height": 960, "note": "подставной опрос"}


def _cutter_ok(src, dst):
    Path(dst).write_bytes(b"\x00" * 32)
    return {"path": str(dst), "frames": 100}


def _decode_ok(video, out_dir):
    return {"paths": [f"{out_dir}/{i:05d}.png" for i in range(99)],
            "note": "подставная раскладка"}


def _distances_ok(frames, anchor, **kw):
    return {"outcome": PASS, "median": 0.0652, "inside": len(frames),
            "judged": len(frames), "note": "подставной прибор личности"}


def _cuts_ok(paths, **kw):
    return {"outcome": PASS, "cuts": [], "note": "подставной прибор резов"}


def _similarity_ok(a, b):
    # Пол (стиль против НЕстилизованного) и попадание различаются по ВТОРОМУ
    # аргументу: так подставной прибор повторяет устройство настоящего.
    return 0.8801 if "styled" in str(b) else 0.6409


def _upload_ok(path):
    return f"https://example.invalid/{Path(path).name}"


def _kling_ok(*, video_url, image_url, character_orientation, out_path):
    Path(out_path).write_bytes(b"\x00" * 128)
    return str(out_path)


def _intake_ok(*, client_photo, style_ref, driving):
    return {"outcome": PASS, "note": "подставной приём"}


def _finish_ok(*, driving_path, kling_path, out_path, window):
    # Сигнатура повторяет НАСТОЯЩУЮ у `fork_finish.finish`: подставная функция
    # с удобной сигнатурой зеленела бы на контракте, которого нет.
    Path(out_path).write_bytes(b"\x00" * 64)
    return {"outcome": PASS, "path": str(out_path),
            "note": f"подставная сборка, окно {window}"}


def _stylize_ok(*, person, style, prompt, out_path):
    Path(out_path).write_bytes(b"\x00" * 64)
    return str(out_path)


class _PlanOk:
    """Подставной сосед-план: НЕ ХОДИТ НА ДИСК и не тащит PIL (Т4).

    Сигнатура повторяет настоящую `fork_plan.to_plan`: подставка с удобной
    сигнатурой зеленела бы на контракте, которого нет.
    """

    @staticmethod
    def to_plan(src, dst, **kw):
        Path(dst).write_bytes(b"\x00" * 64)
        return {"outcome": PASS, "checked": 1, "violations": 0,
                "unmeasured": 0, "path": str(dst),
                "note": "подставной план 9:16"}

    @staticmethod
    def extend_to_plan(src, dst, **kw):
        Path(dst).write_bytes(b"\x00" * 64)
        return {"outcome": PASS, "checked": 1, "violations": 0,
                "unmeasured": 0, "path": str(dst), "extended": True,
                "note": "подставная дорисовка полей"}


def _run(root: Path, log, **over):
    f = _files(root)
    kw = dict(client_photo=f["client"], style_ref=f["style"],
              driving=f["driving"], first=100, last=199,
              out_dir=root / "out", intake=_intake_ok, stylize=_stylize_ok,
              similarity=_similarity_ok, distances=_distances_ok,
              probe=_probe_ok, cutter=_cutter_ok, decode=_decode_ok,
              cuts=_cuts_ok, upload=_upload_ok, kling=_kling_ok,
              finish=_finish_ok, plan=_PlanOk, log=log)
    kw.update(over)
    return E.run(**kw)


class WholePathOnFakes(unittest.TestCase):
    def test_every_stage_passes_and_nothing_touches_the_network(self):
        log = io.StringIO()
        with TemporaryDirectory() as td, _no_network():
            got = _run(Path(td), log)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["exit_code"], 0)
        # Восемь ступеней, все до одной: путь пройден целиком, а не до середины.
        self.assertEqual(len(got["stages"]), 8)
        self.assertEqual([s["outcome"] for s in got["stages"]], ["годно"] * 8)
        self.assertEqual(got["totals"]["stages_passed"], 7)
        self.assertEqual(got["totals"]["violations"], 0)
        self.assertEqual(got["totals"]["unmeasured"], 0)

    def test_the_stage_names_are_the_declared_order(self):
        log = io.StringIO()
        with TemporaryDirectory() as td, _no_network():
            got = _run(Path(td), log)
        self.assertEqual([s["stage"] for s in got["stages"]],
                         ["1 приём трёх входов",
                          "2 стилизация фото клиента",
                          "3 приёмка стилизованного фото",
                          "4 окно драйвинга и нарезка",
                          "5 загрузка входов и вызов Kling",
                          "6 приёмка выхода",
                          "7 финальная сборка",
                          "8 отчёт"])

    def test_stages_are_printed_while_the_run_is_still_going(self):
        """Печать ПО ХОДУ — наблюдаемо, а не на слово.

        Молчащий 25-минутный прогон уже уносил всё измеренное. Здесь
        подставной стилизатор (ступень 2) смотрит в журнал и обязан увидеть
        там строку ступени 1.
        """
        log = io.StringIO()
        seen = {}

        def watching_stylize(*, person, style, prompt, out_path):
            seen["log"] = log.getvalue()
            return _stylize_ok(person=person, style=style, prompt=prompt,
                               out_path=out_path)

        with TemporaryDirectory() as td, _no_network():
            _run(Path(td), log, stylize=watching_stylize)
        self.assertIn("1 приём трёх входов", seen["log"])
        self.assertNotIn("6 приёмка выхода", seen["log"])


class ThirdOutcomeIsNotCollapsed(unittest.TestCase):
    def test_a_falling_kling_is_unmeasured_and_not_a_defect(self):
        def falling(*, video_url, image_url, character_orientation, out_path):
            raise RuntimeError("очередь fal вернула 503")

        log = io.StringIO()
        with TemporaryDirectory() as td, _no_network():
            got = _run(Path(td), log, kling=falling)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertEqual(got["exit_code"], 2)
        self.assertEqual(got["stopped_at"], "5 загрузка входов и вызов Kling")
        self.assertEqual(got["stopped_index"], 5)
        # Именно НЕ «не годно»: снимается это другим способом.
        self.assertNotEqual(got["outcome"], "не годно")

    def test_a_real_defect_is_a_defect_and_stops_the_run_naming_the_stage(self):
        def stranger(frames, anchor, **kw):
            # 1.0217 — измеренное расстояние до ЧУЖОГО человека.
            return {"outcome": FAIL, "median": 1.0217, "inside": 0,
                    "judged": len(frames), "note": "чужой человек"}

        log = io.StringIO()
        with TemporaryDirectory() as td, _no_network():
            got = _run(Path(td), log, distances=stranger)
        self.assertEqual(got["outcome"], "не годно")
        self.assertEqual(got["exit_code"], 1)
        self.assertEqual(got["stopped_at"], "3 приёмка стилизованного фото")
        # Ступени 4..7 не выполнялись, а отчёт всё равно напечатан.
        self.assertEqual([s["stage"] for s in got["stages"]],
                         ["1 приём трёх входов", "2 стилизация фото клиента",
                          "3 приёмка стилизованного фото", "8 отчёт"])
        self.assertIn("ИТОГ: не годно на ступени «3 приёмка стилизованного фото»",
                      log.getvalue())

    def test_the_three_exit_codes_are_deliberately_different(self):
        self.assertEqual([E.EXIT_BY_OUTCOME["годно"],
                          E.EXIT_BY_OUTCOME["не годно"],
                          E.EXIT_BY_OUTCOME["не смогли проверить"]], [0, 1, 2])

    def test_zero_checks_is_not_a_success(self):
        # Р2: ноль нарушений при нуле отработавших проверок — не «годно».
        self.assertEqual(E.verdict(0, 0, 0), "не смогли проверить")
        self.assertEqual(E.verdict(1, 0, 0), "годно")
        self.assertEqual(E.verdict(1, 1, 0), "не годно")
        self.assertEqual(E.verdict(1, 0, 1), "не смогли проверить")
        # Нарушение перебивает «не смогли»: найденное не перестаёт быть найденным.
        self.assertEqual(E.verdict(2, 1, 1), "не годно")


class IdentityBarIsGuarded(unittest.TestCase):
    """Планка личности — константа-решение. Мутация в ОБЕ стороны."""

    def _acceptance(self, median):
        def d(frames, anchor, **kw):
            return {"outcome": PASS if median <= 0.35 else FAIL,
                    "median": median, "inside": 0, "judged": 1, "note": ""}
        return E.stage_style_acceptance(styled="styled.png", style_ref="s.png",
                                        client_photo="c.png",
                                        similarity=_similarity_ok, distances=d)

    def test_just_inside_the_bar_passes_and_just_outside_is_UNMEASURED(self):
        # ПЕРЕПИСАН 22.08 под решение владельца: за планкой теперь НЕ «не
        # годно», а «не смогли» — там начинается средняя полоса лестницы, где
        # лицо закрыто аксессуаром и ArcFace не судья. «Не годно» переехало за
        # ступень «другой человек» 0.7137 и проверяется отдельным классом.
        self.assertEqual(self._acceptance(0.34)["outcome"], "годно")
        self.assertEqual(self._acceptance(0.36)["outcome"], "не смогли проверить")
        self.assertEqual(self._acceptance(0.80)["outcome"], "не годно")

    def test_the_bar_itself_moved_flips_the_verdict_both_ways(self):
        # Мутация планки в обе стороны по-прежнему видна, только нижний исход
        # теперь «не смогли», а не «не годно».
        with mock.patch.object(E, "SAME_PERSON_MAX", 0.30):
            self.assertEqual(self._acceptance(0.32)["outcome"],
                             "не смогли проверить")
        with mock.patch.object(E, "SAME_PERSON_MAX", 0.40):
            self.assertEqual(self._acceptance(0.32)["outcome"], "годно")

    def test_an_unmeasured_identity_is_not_a_defect(self):
        def d(frames, anchor, **kw):
            return {"outcome": UNMEASURED, "median": None,
                    "note": "лица на кадре нет"}
        got = E.stage_style_acceptance(styled="styled.png", style_ref="s.png",
                                       client_photo="c.png",
                                       similarity=_similarity_ok, distances=d)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertEqual((got["checked"], got["violations"], got["unmeasured"]),
                         (1, 0, 1))


class StyleFloorIsTheNegativeControl(unittest.TestCase):
    """Пол считается НА МЕСТЕ, и без него ступень не выносит вердикта."""

    def _acceptance(self, hit, floor):
        def sim(a, b):
            return hit if "styled" in str(b) else floor
        return E.stage_style_acceptance(styled="styled.png", style_ref="s.png",
                                        client_photo="c.png", similarity=sim,
                                        distances=_distances_ok)

    def test_the_measured_winner_passes_and_the_rejected_text_route_fails(self):
        # ИЗМЕРЕНО: картинкой 0.8801 при поле 0.6409; текстом 0.6773 — шум.
        self.assertEqual(self._acceptance(0.8801, 0.6409)["outcome"], "годно")
        self.assertEqual(self._acceptance(0.6773, 0.6409)["outcome"], "не годно")

    def test_the_margin_constant_is_guarded_in_both_directions(self):
        with mock.patch.object(E, "STYLE_MARGIN_MIN", 0.02):
            # Слабее планка — шумовой путь проходит, и это видно.
            self.assertEqual(self._acceptance(0.6773, 0.6409)["outcome"], "годно")
        with mock.patch.object(E, "STYLE_MARGIN_MIN", 0.30):
            # Строже планка — измеренный победитель не проходит.
            self.assertEqual(self._acceptance(0.8801, 0.6409)["outcome"], "не годно")

    def test_without_a_floor_the_stage_says_it_could_not_measure(self):
        def sim(a, b):
            return None
        got = E.stage_style_acceptance(styled="styled.png", style_ref="s.png",
                                       client_photo="c.png", similarity=sim,
                                       distances=_distances_ok)
        self.assertEqual(got["checks"][0]["outcome"], "не смогли проверить")
        self.assertEqual(got["outcome"], "не смогли проверить")


class PaletteInstrumentHasBothControls(unittest.TestCase):
    """У прибора есть вход, где он обязан сказать «нет», и где обязан шевельнуться."""

    def _png(self, root: Path, name: str, colour) -> Path:
        from PIL import Image

        p = root / name
        Image.new("RGB", (64, 64), colour).save(p)
        return p

    def test_same_image_is_one_and_a_different_palette_is_far_below(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            blue = self._png(root, "blue.png", (40, 120, 220))
            red = self._png(root, "red.png", (220, 60, 40))
            self.assertEqual(E.palette_similarity(blue, blue), 1.0)
            self.assertEqual(E.palette_similarity(blue, red), 0.0)

    def test_an_unreadable_input_gives_none_and_not_a_number(self):
        with TemporaryDirectory() as td:
            broken = Path(td) / "broken.png"
            broken.write_bytes(b"not a picture")
            self.assertIsNone(E.palette_similarity(broken, broken))


class SceneLengthIsGuarded(unittest.TestCase):
    """3 секунды — критерий приёма окна, а не пожелание (гейт Kling)."""

    def _window(self, first, last, cutter=None):
        return E.stage_window(driving="d.mp4", first=first, last=last,
                              out_path="w.mp4", probe=_probe_ok,
                              cutter=cutter or (lambda s, d: {"path": "w.mp4",
                                                              "frames": last - first + 1}))

    def test_ninety_frames_at_thirty_fps_pass_and_eighty_nine_fail(self):
        # 90/30 = 3.0 с ровно; 89/30 = 2.967 с.
        with mock.patch.object(E, "file_fact",
                               lambda p, w: (w, PASS, "подставная проверка файла")):
            self.assertEqual(self._window(0, 89)["outcome"], "годно")
            self.assertEqual(self._window(0, 88)["outcome"], "не годно")

    def test_the_threshold_moved_flips_the_verdict_both_ways(self):
        with mock.patch.object(E, "file_fact",
                               lambda p, w: (w, PASS, "подставная проверка файла")):
            with mock.patch.object(E, "MIN_SCENE_S", 4.0):
                self.assertEqual(self._window(0, 89)["outcome"], "не годно")
            with mock.patch.object(E, "MIN_SCENE_S", 2.0):
                self.assertEqual(self._window(0, 88)["outcome"], "годно")

    def test_a_window_outside_the_clip_is_a_defect(self):
        got = self._window(300, 500)
        self.assertEqual(got["outcome"], "не годно")
        self.assertIn("373", got["checks"][1]["note"])

    def test_a_cut_that_returned_the_wrong_frame_count_is_a_defect(self):
        # ИЗМЕРЕНО: ffmpeg с точкой реза за концом молча отдаёт файл ЦЕЛИКОМ.
        with mock.patch.object(E, "file_fact",
                               lambda p, w: (w, PASS, "подставная проверка файла")):
            got = self._window(100, 199,
                               cutter=lambda s, d: {"path": "w.mp4", "frames": 373})
        self.assertEqual(got["outcome"], "не годно")
        self.assertIn("373 при заказанных 100",
                      [c["note"] for c in got["checks"]][-2])

    def test_a_cut_that_could_not_be_counted_is_not_a_defect(self):
        with mock.patch.object(E, "file_fact",
                               lambda p, w: (w, PASS, "подставная проверка файла")):
            got = self._window(100, 199,
                               cutter=lambda s, d: {"path": "w.mp4", "frames": None})
        self.assertEqual(got["outcome"], "не смогли проверить")

    def test_the_cut_command_counts_frames_and_not_seconds(self):
        argv = E.cut_argv("in.mp4", "out.mp4", first=100, last=199, fps=30.0,
                          exe="ffmpeg")
        self.assertIn("-frames:v", argv)
        self.assertEqual(argv[argv.index("-frames:v") + 1], "100")
        self.assertEqual(argv[argv.index("-ss") + 1], "3.333333")
        self.assertNotIn("-t", argv)


class MoneyGuards(unittest.TestCase):
    def test_pro_is_refused_before_any_money_is_spent(self):
        with self.assertRaises(ValueError) as e:
            E.refuse_pro("fal-ai/kling-video/v2.6/pro/motion-control")
        self.assertIn("12.8", str(e.exception))
        # Негативный контроль сторожа: боевой эндпоинт он пропускает.
        self.assertIsNone(E.refuse_pro("fal-ai/kling-video/v2.6/standard/motion-control"))
        # И не срабатывает на слово внутри другого слова.
        self.assertIsNone(E.refuse_pro("fal-ai/proxy-kling/standard/motion-control"))

    def test_a_pro_endpoint_stops_the_stage_before_the_upload(self):
        called = []
        got = E.stage_kling(styled="s.png", window="w.mp4", out_path="o.mp4",
                            upload=lambda p: called.append(p),
                            kling=_kling_ok,
                            endpoint="fal-ai/kling-video/v2.6/pro/motion-control")
        self.assertEqual(got["outcome"], "не годно")
        self.assertEqual(called, [])

    def test_the_payload_has_exactly_the_three_measured_fields(self):
        payload = E.kling_payload(video_url="v", image_url="i")
        self.assertEqual(sorted(payload),
                         ["character_orientation", "image_url", "video_url"])
        self.assertEqual(payload["character_orientation"], "video")

    def test_an_orientation_outside_the_two_measured_values_is_refused(self):
        with self.assertRaises(ValueError):
            E.kling_payload(video_url="v", image_url="i",
                            character_orientation="auto")
        # Оба измеренных значения принимаются — негативный контроль сторожа.
        for value in ("image", "video"):
            self.assertEqual(
                E.kling_payload(video_url="v", image_url="i",
                                character_orientation=value)["character_orientation"],
                value)

    def test_a_field_added_to_the_payload_reddens_the_stage(self):
        with mock.patch.object(E, "kling_payload",
                               lambda **kw: {"video_url": "v", "image_url": "i",
                                             "character_orientation": "video",
                                             "prompt": "лишнее поле"}):
            got = E.stage_kling(styled="s.png", window="w.mp4", out_path="o.mp4",
                                upload=_upload_ok, kling=_kling_ok)
        self.assertEqual(got["outcome"], "не годно")
        self.assertIn("лишние ['prompt']",
                      [c["note"] for c in got["checks"]][-1])

    def test_a_failed_upload_never_reaches_the_paid_call(self):
        ordered = []

        def bad_upload(path):
            raise OSError("сеть недоступна")

        def counting_kling(**kw):
            ordered.append(kw)
            return "o.mp4"

        got = E.stage_kling(styled="s.png", window="w.mp4", out_path="o.mp4",
                            upload=bad_upload, kling=counting_kling)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertEqual(ordered, [])


class OutputAcceptance(unittest.TestCase):
    def _accept(self, **over):
        kw = dict(produced="o.mp4", client_photo="c.png", frames_dir="f",
                  probe=_probe_ok, decode=_decode_ok, distances=_distances_ok,
                  cuts=_cuts_ok)
        kw.update(over)
        return E.stage_output_acceptance(**kw)

    def test_any_vertical_geometry_passes_and_landscape_is_a_defect(self):
        # ПЕРЕПИСАН 22.08. Прежняя редакция требовала совпадения с
        # `KLING_OUT_SIZE` и забраковала боевой выход 816x1104 — вертикаль,
        # которой мы добивались весь день. Теперь сторожится СВОЙСТВО: 720x1280
        # обязано ПРОХОДИТЬ (это вертикаль), а горизонталь — падать.
        self.assertEqual(self._accept()["outcome"], "годно")
        vertical = lambda p: {"outcome": PASS, "fps": 30.0, "frames": 99,
                              "width": 720, "height": 1280, "note": ""}
        self.assertEqual(self._accept(probe=vertical)["outcome"], "годно")
        landscape = lambda p: {"outcome": PASS, "fps": 30.0, "frames": 99,
                               "width": 1280, "height": 720, "note": ""}
        self.assertEqual(self._accept(probe=landscape)["outcome"], "не годно")

    def test_the_ratio_ceiling_moved_flips_the_verdict_both_ways(self):
        # Мутация НОВОЙ константы-решения в обе стороны. `KLING_OUT_SIZE`
        # больше не решает ничего — он остался историей восьми заказов.
        square = lambda p: {"outcome": PASS, "fps": 30.0, "frames": 99,
                            "width": 960, "height": 960, "note": ""}
        with mock.patch.object(E, "OUT_RATIO_MAX", 0.9):
            self.assertEqual(self._accept(probe=square)["outcome"], "не годно")
        with mock.patch.object(E, "OUT_RATIO_MAX", 1.5):
            self.assertEqual(self._accept(probe=square)["outcome"], "годно")

    def test_a_single_cut_on_the_output_is_a_defect(self):
        one = lambda paths, **kw: {"outcome": PASS, "cuts": [37], "note": ""}
        self.assertEqual(self._accept(cuts=one)["outcome"], "не годно")
        with mock.patch.object(E, "MAX_CUTS_OUT", 1):
            self.assertEqual(self._accept(cuts=one)["outcome"], "годно")

    def test_cuts_that_could_not_be_looked_for_are_not_zero_cuts(self):
        blind = lambda paths, **kw: {"outcome": UNMEASURED, "cuts": [],
                                     "note": "типичный скачок равен нулю"}
        got = self._accept(cuts=blind)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIsNone(got["numbers"]["cuts"])

    def test_no_frames_decoded_is_unmeasured_and_stops_before_judging(self):
        empty = lambda v, d: {"paths": [], "note": "раскладка пуста"}
        got = self._accept(decode=empty)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertEqual([c["name"] for c in got["checks"]],
                         ["геометрия выхода", "раскладка на кадры"])


class NeighbourModulesAreSoft(unittest.TestCase):
    """Соседа нет — «не смогли проверить», и сказано, кого именно нет."""

    def test_a_missing_intake_module_is_unmeasured_not_a_defect(self):
        with TemporaryDirectory() as td:
            f = _files(Path(td))
            with mock.patch.object(E, "soft_import",
                                   lambda name: (None, f"модуля ball_reel.{name} нет")):
                got = E.stage_intake(client_photo=f["client"],
                                     style_ref=f["style"], driving=f["driving"])
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIn("fork_intake", got["note"])
        self.assertEqual((got["checked"], got["violations"], got["unmeasured"]),
                         (3, 0, 1))

    def test_a_missing_input_file_is_a_defect_even_without_the_neighbour(self):
        with TemporaryDirectory() as td:
            f = _files(Path(td))
            f["driving"].unlink()
            with mock.patch.object(E, "soft_import",
                                   lambda name: (None, "соседа нет")):
                got = E.stage_intake(client_photo=f["client"],
                                     style_ref=f["style"], driving=f["driving"])
        self.assertEqual(got["outcome"], "не годно")

    def test_a_neighbour_that_raises_is_unmeasured(self):
        def boom(**kw):
            raise KeyError("ещё не написан")

        with TemporaryDirectory() as td:
            f = _files(Path(td))
            got = E.stage_intake(client_photo=f["client"], style_ref=f["style"],
                                 driving=f["driving"], intake=boom)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIn("KeyError", got["checks"][-1]["note"])

    def test_a_neighbour_answering_without_a_verdict_is_not_a_success(self):
        got = E.outcome_of("готово", what="fork_finish")
        self.assertEqual(got[0], "не смогли проверить")
        self.assertEqual(E.outcome_of({"outcome": "годно"}, what="x")[0], "годно")

    def test_a_neighbour_taking_positional_arguments_is_still_called(self):
        seen = []

        def positional(a, b, c):
            seen.append((a, b, c))
            return {"outcome": PASS, "note": "позиционно"}

        with TemporaryDirectory() as td:
            f = _files(Path(td))
            got = E.stage_intake(client_photo=f["client"], style_ref=f["style"],
                                 driving=f["driving"], intake=positional)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(len(seen), 1)

    def test_the_entry_point_refusal_names_what_was_tried(self):
        class Empty:
            __name__ = "ball_reel.fork_finish"

        fn, name, why = E.entry_point(Empty(), ("finish", "assemble"))
        self.assertIsNone(fn)
        self.assertIn("['finish', 'assemble']", why)


class FinishSeam(unittest.TestCase):
    """Сосед-сборщик берёт драйвинг ПЕРВЫМ. Перепутанный порядок — не деталь."""

    def test_the_neighbour_gets_the_driving_first_and_the_window_inclusive(self):
        seen = {}

        def finish(*, driving_path, kling_path, out_path, window):
            seen.update(driving_path=driving_path, kling_path=kling_path,
                        window=window)
            Path(out_path).write_bytes(b"\x00")
            return {"outcome": PASS, "path": out_path, "note": ""}

        with TemporaryDirectory() as td:
            got = E.stage_finish(produced="kling.mp4", driving="drv.mp4",
                                 out_path=Path(td) / "final.mp4",
                                 window=(100, 199), finish=finish)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(seen["driving_path"], "drv.mp4")
        self.assertEqual(seen["kling_path"], "kling.mp4")
        self.assertEqual(seen["window"], (100, 199))

    def test_a_neighbour_verdict_of_unmeasured_is_carried_through(self):
        def finish(**kw):
            return {"outcome": UNMEASURED, "note": "длительность не прочиталась"}

        got = E.stage_finish(produced="k.mp4", driving="d.mp4",
                             out_path="f.mp4", window=(0, 99), finish=finish)
        self.assertEqual(got["outcome"], "не смогли проверить")


class IntakeSeam(unittest.TestCase):
    """У соседа-приёмщика ТРИ функции, а не одна. Форма измерена, не угадана."""

    def test_the_three_intake_functions_are_all_called(self):
        seen = []

        class Trio:
            __name__ = "ball_reel.fork_intake"

            @staticmethod
            def photo_intake(path, **kw):
                seen.append(("photo", path))
                return {"outcome": PASS, "checked": 3, "violations": 0,
                        "unmeasured": 0, "note": "лицо одно"}

            @staticmethod
            def style_intake(path, **kw):
                seen.append(("style", path))
                return {"outcome": PASS, "checked": 1, "violations": 0,
                        "unmeasured": 0, "note": "карточка читается"}

            @staticmethod
            def driving_intake(path, frames=None, **kw):
                seen.append(("driving", path))
                return {"outcome": PASS, "checked": 5, "violations": 0,
                        "unmeasured": 0, "note": "склеек 0"}

        with TemporaryDirectory() as td:
            f = _files(Path(td))
            with mock.patch.object(E, "soft_import", lambda n: (Trio(), None)):
                got = E.stage_intake(client_photo=f["client"],
                                     style_ref=f["style"], driving=f["driving"])
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual([kind for kind, _ in seen], ["photo", "style", "driving"])
        self.assertEqual(got["checked"], 6)
        self.assertIn("проверено 5, нарушений 0, не смогли 0",
                      got["checks"][-1]["note"])

    def test_one_refused_input_reddens_the_stage_and_the_others_still_ran(self):
        class Trio:
            __name__ = "ball_reel.fork_intake"

            @staticmethod
            def photo_intake(path, **kw):
                return {"outcome": FAIL, "checked": 3, "violations": 1,
                        "unmeasured": 0, "note": "два лица на фото"}

            @staticmethod
            def style_intake(path, **kw):
                return {"outcome": PASS, "checked": 1, "violations": 0,
                        "unmeasured": 0, "note": ""}

            @staticmethod
            def driving_intake(path, frames=None, **kw):
                return {"outcome": UNMEASURED, "checked": 1, "violations": 0,
                        "unmeasured": 4, "note": "кадров не подали"}

        with TemporaryDirectory() as td:
            f = _files(Path(td))
            with mock.patch.object(E, "soft_import", lambda n: (Trio(), None)):
                got = E.stage_intake(client_photo=f["client"],
                                     style_ref=f["style"], driving=f["driving"])
        self.assertEqual(got["outcome"], "не годно")
        self.assertEqual((got["checked"], got["violations"], got["unmeasured"]),
                         (5, 1, 1))


    def test_the_card_reader_is_handed_to_the_style_intake_only(self):
        """Без читателя карточки сосед по стилю честно встаёт: это ИЗМЕРЕНО."""
        seen = {}

        class Trio:
            __name__ = "ball_reel.fork_intake"

            @staticmethod
            def photo_intake(path, **kw):
                seen["photo_kw"] = kw
                return {"outcome": PASS, "checked": 1, "violations": 0,
                        "unmeasured": 0, "note": ""}

            @staticmethod
            def style_intake(path, card_reader=None, **kw):
                seen["reader"] = card_reader
                return {"outcome": PASS if card_reader else UNMEASURED,
                        "checked": 1 if card_reader else 0, "violations": 0,
                        "unmeasured": 0 if card_reader else 1, "note": ""}

            @staticmethod
            def driving_intake(path, frames=None, **kw):
                seen["frames"] = frames
                return {"outcome": PASS, "checked": 1, "violations": 0,
                        "unmeasured": 0, "note": ""}

        reader = lambda p: {"card": {}}
        with TemporaryDirectory() as td:
            f = _files(Path(td))
            with mock.patch.object(E, "soft_import", lambda n: (Trio(), None)):
                with_reader = E.stage_intake(
                    client_photo=f["client"], style_ref=f["style"],
                    driving=f["driving"], card_reader=reader,
                    driving_frames=["a.png"])
                without = E.stage_intake(client_photo=f["client"],
                                         style_ref=f["style"],
                                         driving=f["driving"])
        self.assertEqual(with_reader["outcome"], "годно")
        self.assertEqual(without["outcome"], "не смогли проверить")
        self.assertEqual(seen["photo_kw"], {})
        self.assertEqual(seen["frames"], None)


class BrandBanIsInThePrompt(unittest.TestCase):
    def test_the_prompt_carries_the_ban_and_the_roles(self):
        built = E.style_prompt("style.png", card_reader=lambda p: {})
        # ПЕРЕПИСАН 22.08: владелец разрешил называть марки словами и оставил
        # запрет только на НАРИСОВАННЫЙ знак. Прежний литерал сторожил решение,
        # которого больше нет.
        self.assertIn("no logo", built["prompt"])
        self.assertNotIn("no brand names", built["prompt"])
        self.assertIn("FIRST image", built["prompt"])
        self.assertIn("SECOND image", built["prompt"])

    def test_a_readable_style_card_adds_words_but_the_ban_stays(self):
        # Словарь карточки — чужой (`fork_style_prompt`), и значения берутся
        # из него: "mid"/"saturated" — те самые, на которых снят промт стиля.
        card = {"colours": ["sky blue", "chocolate", "blue"],
                "value_key": "mid", "saturation": "saturated",
                "texture": "clean flat surfaces"}
        built = E.style_prompt("style.png", card_reader=lambda p: card)
        self.assertIn("sky blue", built["prompt"])
        # ПЕРЕПИСАН 22.08: владелец разрешил называть марки словами и оставил
        # запрет только на НАРИСОВАННЫЙ знак. Прежний литерал сторожил решение,
        # которого больше нет.
        self.assertIn("no logo", built["prompt"])
        self.assertNotIn("no brand names", built["prompt"])

    def test_a_prompt_without_the_ban_reddens_the_stage(self):
        """Негативный контроль сторожа: без него проверка всегда зелена."""
        with TemporaryDirectory() as td:
            got = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                                  out_path=Path(td) / "styled.png",
                                  stylize=_stylize_ok, plan=_PlanOk,
                                  prompt="just make it look nice")
            self.assertEqual(got["outcome"], "не годно")
            self.assertEqual(got["checks"][0]["outcome"], "не годно")
            # И вход, на котором сторож обязан молчать (И5).
            ok = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                                 out_path=Path(td) / "styled.png",
                                 stylize=_stylize_ok, plan=_PlanOk,
                                 prompt="a look, " + E.NO_BRANDS_CLAUSE)
            self.assertEqual(ok["outcome"], "годно")

    def test_the_ban_text_itself_is_a_decision_constant(self):
        # Мутация константы: сторож ищет ИМЕННО её, а не любое слово про бренды.
        with mock.patch.object(E, "NO_BRANDS_CLAUSE", "no logos whatsoever"):
            with TemporaryDirectory() as td:
                got = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                                      out_path=Path(td) / "styled.png",
                                      stylize=_stylize_ok, plan=_PlanOk,
                                      prompt="a look, no brand names, no logos")
        self.assertEqual(got["outcome"], "не годно")

    def test_a_stylizer_that_fell_is_unmeasured(self):
        def boom(**kw):
            raise RuntimeError("HTTP 524")

        got = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                              out_path="styled.png", stylize=boom,
                              card_reader=lambda p: {})
        self.assertEqual(got["outcome"], "не смогли проверить")


class WindowArgument(unittest.TestCase):
    def test_a_window_is_parsed_and_garbage_is_refused(self):
        self.assertEqual(E.parse_window("100:199"), (100, 199))
        for bad in ("100", "100:", "a:b", "199:100", "100:199:2"):
            with self.assertRaises(ValueError, msg=bad):
                E.parse_window(bad)


class ReportIsAlwaysWritten(unittest.TestCase):
    def test_the_report_survives_an_early_stop_and_carries_numbers(self):
        import json

        def falling(**kw):
            raise RuntimeError("503")

        log = io.StringIO()
        with TemporaryDirectory() as td, _no_network():
            got = _run(Path(td), log, kling=falling)
            data = json.loads(Path(got["report"]).read_text(encoding="utf-8"))
        self.assertEqual(got["stages"][-1]["stage"], "8 отчёт")
        self.assertEqual(len(data["stages"]), 5)
        self.assertEqual(data["unmeasured"], 1)
        self.assertEqual(data["violations"], 0)


if __name__ == "__main__":
    unittest.main()


class TheStyliserWasChosenByEyeNotByNumber(unittest.TestCase):
    """Решение владельца 22.08: `nanobanana-2`, ВОПРЕКИ числу.

    Гейт стоит на самом решении, а не на его следствиях: замер награждал
    перерисовку, и модель с бОльшим числом утащила из референса одежду и позу.
    Если кто-то вернёт `gpt-image-2` обратно «потому что 0.8801 больше», этот
    тест покраснеет и заставит прочитать, почему так делать нельзя.
    """

    def test_the_chosen_styliser_is_the_one_the_owner_picked(self):
        # Литерал, а не импорт из проверяемого модуля (Т2).
        self.assertEqual(E.STYLE_MODEL, "nanobanana-2")

    def test_the_rejected_styliser_scored_HIGHER_and_is_still_rejected(self):
        # Негативный контроль решения: отвергнутый обязан быть ВЫШЕ по числу,
        # иначе история «выбрали вопреки мере» не воспроизводится и правило
        # выглядит произволом.
        self.assertGreater(E.STYLE_HIT_REJECTED,
                           E.STYLE_HIT_REFERENCE)
        self.assertNotEqual(E.STYLE_MODEL, "gpt-image-2")

    def test_the_chosen_styliser_still_beats_the_floor(self):
        # Выбор глазами не отменяет требования: стиль обязан доехать.
        self.assertGreater(E.STYLE_HIT_REFERENCE,
                           E.STYLE_FLOOR_REFERENCE)

    def test_the_text_route_stays_below_the_floor_margin(self):
        # Текстовый путь отвергнут числом, и это по-прежнему верно.
        self.assertLess(E.STYLE_TEXT_ROUTE_REFERENCE
                        - E.STYLE_FLOOR_REFERENCE, 0.05)


class TheStyleReferenceLeaksAppearanceAndItIsGuarded(unittest.TestCase):
    """Боевой прогон 22.08: стилизация надела на клиента ОЧКИ с референса.

    ArcFace дал 0.3928 при планке 0.35 и стенд встал, не потратив денег.
    Диагноз в отчёте был неверный: не «личность потеряна», а ЛИЦО ЗАКРЫТО —
    ArcFace опирается на область глаз. Гейт сторожит запрет, а не последствие.
    """

    def test_the_prompt_forbids_copying_eyewear_from_the_reference(self):
        built = E.style_prompt("любой.png", card_reader=lambda p: None)
        # Литералы, а не импорт из проверяемого модуля (Т2).
        for word in ("eyewear", "accessory", "garment", "pose"):
            with self.subTest(word=word):
                self.assertIn(word, built["prompt"])

    def test_the_role_clause_names_what_to_KEEP_not_only_what_to_take(self):
        built = E.style_prompt("любой.png", card_reader=lambda p: None)
        for word in ("same clothing", "same pose", "same accessories"):
            with self.subTest(word=word):
                self.assertIn(word, built["prompt"])

    def test_the_two_bans_are_separate_constants_with_separate_histories(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ склейки: слипнись они в одну строку, вынуть
        # можно было бы только обе сразу, и история каждой потерялась бы.
        self.assertNotEqual(E.NO_BRANDS_CLAUSE, E.NO_LOOK_TRANSFER_CLAUSE)
        self.assertNotIn(E.NO_BRANDS_CLAUSE, E.NO_LOOK_TRANSFER_CLAUSE)

    def test_removing_the_look_ban_is_visible_in_the_prompt(self):
        # Мутация в слабую сторону: без запрета промт обязан стать другим.
        built = E.style_prompt("любой.png", card_reader=lambda p: None)
        self.assertIn(E.NO_LOOK_TRANSFER_CLAUSE, built["prompt"])


class TheIdentityAxisHasAMiddleBandAndAnOperatorOverride(unittest.TestCase):
    """Решение владельца 22.08: очки со стиля — не баг, а фича.

    Планку НЕ подняли: поднятая перестала бы ловить настоящую подмену.
    Вместо этого средняя полоса лестницы стала третьим исходом, а проход по
    ней — ЯВНЫМ допуском оператора, который виден в отчёте.
    """

    def _stage(self, median, **kw):
        return E.stage_style_acceptance(
            styled="s.png", style_ref="r.png", client_photo="p.png",
            similarity=lambda a, b: 0.9 if "s.png" in str(b) else 0.2,
            distances=lambda fr, an: {"outcome": E.PASS, "median": median},
            **kw)

    def _axis(self, res):
        return [c for c in res["checks"] if "личность" in c["name"]][0]

    def test_below_the_bar_is_plainly_good(self):
        self.assertEqual(self._axis(self._stage(0.0652))["outcome"], E.PASS)

    def test_the_middle_band_is_UNMEASURED_not_failed(self):
        # 0.3928 — ровно тот боевой случай с очками.
        self.assertEqual(self._axis(self._stage(0.3928))["outcome"], E.UNMEASURED)

    def test_the_middle_band_passes_only_with_an_explicit_operator_flag(self):
        got = self._axis(self._stage(0.3928, operator_ok_identity=True))
        self.assertEqual(got["outcome"], E.PASS)
        self.assertIn("ДОПУЩЕНО ОПЕРАТОРОМ", got["note"])

    def test_above_the_other_person_rung_stays_FAILED_even_for_the_operator(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ допуска: он не должен уметь пропустить подмену.
        got = self._axis(self._stage(0.80, operator_ok_identity=True))
        self.assertEqual(got["outcome"], E.FAIL)

    def test_the_ladder_numbers_are_the_measured_ones(self):
        # Литералы (Т2).
        self.assertEqual(E.LADDER_SAME, 0.0652)
        self.assertEqual(E.LADDER_REJECTED, 0.7137)
        self.assertEqual(E.LADDER_STRANGER, 1.0217)


class TheGeometryCheckGuardsVerticalityNotExactNumbers(unittest.TestCase):
    """Боевой прогон 22.08 вернул 816x1104 — ВЕРТИКАЛЬ, и прибор её забраковал.

    Он сверял выход с измеренными 960x960 и завернул самый желанный исход.
    Теперь сторожится СВОЙСТВО (вертикаль или квадрат), а не число.
    """

    def _geom(self, w, h, fps=30.0, **kw):
        res = E.stage_output_acceptance(
            produced="p.mp4", client_photo="c.png", frames_dir="d",
            probe=lambda p: {"width": w, "height": h, "fps": fps, "frames": 99},
            decode=_decode_ok,
            distances=lambda fr, an: {"outcome": E.PASS, "median": 0.20,
                                      "inside": 99, "judged": 99},
            cuts=lambda p: {"outcome": E.PASS, "cuts": [], "note": ""}, **kw)
        return [c for c in res["checks"] if "геометрия" in c["name"]][0]

    def test_the_new_vertical_output_passes(self):
        got = self._geom(816, 1104)
        self.assertEqual(got["outcome"], E.PASS)
        self.assertIn("НОВАЯ геометрия", got["note"])

    def test_the_old_square_output_still_passes(self):
        self.assertEqual(self._geom(960, 960)["outcome"], E.PASS)

    def test_a_landscape_output_is_a_defect(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ: прибор обязан уметь сказать «нет».
        self.assertEqual(self._geom(1104, 816)["outcome"], E.FAIL)

    def test_a_wrong_fps_is_UNMEASURED_not_failed(self):
        # Сборка звука считает кадры по 30: другая частота — судить нечем.
        self.assertEqual(self._geom(816, 1104, fps=24.0)["outcome"], E.UNMEASURED)

    def test_the_ratio_ceiling_is_the_chosen_one(self):
        self.assertEqual(E.OUT_RATIO_MAX, 1.0)


class TheOutputIdentityUsesTheSameLadderAsTheStyledPhoto(unittest.TestCase):
    """Одно знание — одно место: лестница на выходе та же, что на фото."""

    def _axis(self, median, **kw):
        res = E.stage_output_acceptance(
            produced="p.mp4", client_photo="c.png", frames_dir="d",
            probe=lambda p: {"width": 816, "height": 1104, "fps": 30.0,
                             "frames": 99},
            decode=_decode_ok,
            distances=lambda fr, an: {"outcome": E.PASS, "median": median,
                                      "inside": 0, "judged": 99},
            cuts=lambda p: {"outcome": E.PASS, "cuts": [], "note": ""}, **kw)
        return [c for c in res["checks"] if "личность" in c["name"]][0]

    def test_the_measured_occluded_case_is_UNMEASURED(self):
        # 0.5109 — ровно боевой случай с очками.
        self.assertEqual(self._axis(0.5109)["outcome"], E.UNMEASURED)

    def test_the_operator_can_let_the_occluded_case_through(self):
        got = self._axis(0.5109, operator_ok_identity=True)
        self.assertEqual(got["outcome"], E.PASS)
        self.assertIn("ДОПУЩЕНО ОПЕРАТОРОМ", got["note"])

    def test_a_real_swap_is_failed_even_for_the_operator(self):
        self.assertEqual(self._axis(0.90, operator_ok_identity=True)["outcome"],
                         E.FAIL)


class TheDeliverableIsBuiltEvenWhenIdentityCannotBeMeasured(unittest.TestCase):
    """Ступень 7 механическая, и «не смогли» на ступени 6 её НЕ отменяет.

    ЗАЧЕМ СТОРОЖ. Гейт `r6 == PASS` означал: заплатили за Kling, ArcFace не
    сумел судить мелкое лицо — и оператор не получил файла, КОТОРЫМ он как раз
    и должен судить. ИЗМЕРЕНО, что это не редкость: приём driving_b4 даёт лицо
    31..98 px и 161 кадр из 450 вообще без лица.
    """

    # Прибор личности ОДИН на две ступени: ступень 3 судит стилизованное фото
    # (один путь), ступень 6 — кадры выхода (много). Подставной прибор обязан
    # различать их так же, иначе он уронит ступень 3 и до ступени 6 дело не
    # дойдёт — на этом первая редакция теста и покраснела.
    @staticmethod
    def _on_output(median):
        def distances(frames, anchor, **kw):
            if len(frames) == 1:                 # стилизованное фото: чисто
                return {"outcome": PASS, "median": 0.0652, "inside": 1,
                        "judged": 1, "note": "подставной прибор личности"}
            return {"outcome": PASS, "median": median, "inside": 0,
                    "judged": len(frames), "note": "кадры выхода"}
        return distances

    @property
    def _band(self):
        """Средняя полоса лестницы: 0.5109 — БОЕВОЕ измерение 22.08."""
        return self._on_output(0.5109)

    @property
    def _swap(self):
        """Выше ступени «другой человек» 0.7137 — настоящая подмена."""
        return self._on_output(0.9)

    def test_an_unmeasurable_identity_still_yields_the_final_file(self):
        log = io.StringIO()
        with TemporaryDirectory() as td, _no_network():
            root = Path(td)
            got = _run(root, log, distances=self._band)
            final = root / "out" / "final_9x16.mp4"
            self.assertTrue(final.exists(), "финального файла нет — судить нечем")
        names = [s["stage"] for s in got["stages"]]
        self.assertIn(E.STAGES[6], names)

    def test_but_the_verdict_is_NOT_whitewashed_into_good(self):
        # Вторая сторона: доделали работу — не значит переписали оценку (Р1).
        log = io.StringIO()
        with TemporaryDirectory() as td, _no_network():
            got = _run(Path(td), log, distances=self._band)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual(got["exit_code"], 2)
        self.assertEqual(got["stopped_at"], E.STAGES[5])

    def test_a_real_swap_still_stops_before_the_finish(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ гейта: на «не годно» ступень 7 обязана НЕ идти,
        # иначе гейт снят целиком, а не смягчён.
        log = io.StringIO()
        with TemporaryDirectory() as td, _no_network():
            root = Path(td)
            got = _run(root, log, distances=self._swap)
            self.assertFalse((root / "out" / "final_9x16.mp4").exists())
        self.assertEqual(got["outcome"], FAIL)
        self.assertNotIn(E.STAGES[6], [s["stage"] for s in got["stages"]])

    def test_the_operator_flag_reaches_run_from_the_command_line(self):
        # Канал операторского вердикта обязан быть ПРОВОДИМ до `run`: флаг,
        # который разбирается и теряется, выглядит рабочим до первого прогона.
        seen = {}

        def fake_run(**kw):
            seen.update(kw)
            return {"exit_code": 0}

        with mock.patch.object(E, "run", fake_run):
            E.main(["--client", "c.png", "--style", "s.png", "--driving",
                    "d.mp4", "--window", "100:199", "--operator-ok-identity"])
        self.assertIs(seen["operator_ok_identity"], True)
        seen.clear()
        with mock.patch.object(E, "run", fake_run):
            E.main(["--client", "c.png", "--style", "s.png", "--driving",
                    "d.mp4", "--window", "100:199"])
        self.assertIs(seen["operator_ok_identity"], False)


class TheFramesChannelReachesRunFromTheCommandLine(unittest.TestCase):
    """Канал, который разбирается и теряется, выглядит рабочим до прогона.

    ИЗМЕРЕНО 22.08: без него оба боевых прогона (b2 и b4) встали на ступени 1
    с «приём драйвинга — не смогли, не смогли 3», хотя тот же приёмщик с теми
    же кадрами отдельно давал «годно, проверено 887, нарушений 0».
    """

    def _seen(self, argv):
        seen = {}

        def fake_run(**kw):
            seen.update(kw)
            return {"exit_code": 0}

        with mock.patch.object(E, "run", fake_run):
            E.main(["--client", "c.png", "--style", "s.png", "--driving",
                    "d.mp4", "--window", "100:199", *argv])
        return seen

    def test_the_frames_arrive_sorted_and_whole(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            for i in (3, 1, 2):                  # вперемешку: порядок задаём мы
                (root / f"f{i:05d}.png").write_bytes(b"\x00")
            got = self._seen(["--frames", str(root)])
        self.assertEqual([Path(p).name for p in got["driving_frames"]],
                         ["f00001.png", "f00002.png", "f00003.png"])

    def test_without_the_flag_the_frames_are_None_not_an_empty_list(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ: «кадров не просили» и «кадров нет» — разные
        # вещи, и приёмщик отвечает на них по-разному.
        self.assertIsNone(self._seen([])["driving_frames"])

    def test_a_missing_directory_is_refused_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            E.frame_paths("нет-такого-каталога")

    def test_an_empty_directory_is_refused_not_silently_ignored(self):
        with TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                E.frame_paths(td)

    def test_non_frames_do_not_count_as_frames(self):
        # Каталог с одним отчётом — это ПУСТОЙ каталог кадров, а не каталог
        # с одним кадром: иначе приёмщик получит на вход json.
        with TemporaryDirectory() as td:
            (Path(td) / "report.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                E.frame_paths(td)


class ThePriceIsPerSecondNotPerCall(unittest.TestCase):
    """Вторая ИЗМЕРЕННАЯ цена исправляет первую, и число сторожится литералом.

    ИЗМЕРЕНО по балансу fal 22.08: 10.8490375 -> 10.1490375 за два
    пятисекундных вызова, то есть $0.35 за вызов, а не $0.21. Прежние $0.21
    были измерены на ТРЁХСЕКУНДНЫХ заказах. Оба делятся на длину в одно
    число, и «цена вызова» без длины оказалась величиной без смысла.
    """

    def test_the_measured_numbers_are_the_ones_shipped(self):
        # Литералы, а не арифметика из проверяемого модуля (Т2).
        self.assertEqual(E.KLING_PRICE_PER_SECOND_USD, 0.07)
        self.assertEqual(E.PRODUCT_SECONDS, 5.0)
        self.assertEqual(E.KLING_PRICE_USD, 0.35)
        self.assertEqual(E.KLING_PRICE_3S_USD, 0.21)

    def test_both_measurements_land_on_the_same_per_second_price(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ гипотезы «берут посекундно»: если бы она была
        # неверна, два независимых замера не сошлись бы в одну точку.
        self.assertEqual(E.kling_price(3), 0.21)
        self.assertEqual(E.kling_price(5), 0.35)

    def test_the_owner_matrix_now_costs_seven_dollars_not_four_twenty(self):
        # Ради чего правка: 20 ячеек по старой цене считались как $4.20.
        self.assertEqual(round(20 * E.KLING_PRICE_USD, 2), 7.0)
        self.assertEqual(round(20 * E.KLING_PRICE_3S_USD, 2), 4.2)

    def test_a_nonsense_duration_is_refused_not_guessed(self):
        for bad in (0, -5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                E.kling_price(bad)
        for bad in ("5", None, True):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                E.kling_price(bad)


class TheStandActuallyCallsItsNeighbours(unittest.TestCase):
    """Соседи `fork_aesthetic` и `fork_plan` ПОЗВАНЫ, а не просто написаны.

    Репозиторий ловит это отдельным правилом (`test_reachable`), и он прав:
    модуль, живущий только в справочнике, в конвейере не работает. Здесь —
    проверка не факта импорта, а того, что вызов ВЛИЯЕТ на исход.
    """

    def test_the_neighbours_are_imported_for_real_not_by_string(self):
        # `soft_import` по строке не считается: правило смотрит на настоящий
        # импорт, и правильно делает — по строке связь не видна ничему.
        self.assertTrue(hasattr(E._default_aesthetic(), "gender_of"))
        self.assertTrue(hasattr(E._default_plan(), "to_plan"))

    def test_the_plan_step_changes_which_file_goes_on(self):
        # Сторож дефекта: план, посчитанный и выброшенный, выглядит рабочим.
        with TemporaryDirectory() as td:
            got = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                                  out_path=Path(td) / "styled.png",
                                  stylize=_stylize_ok, plan=_PlanOk,
                                  prompt="a look, " + E.NO_BRANDS_CLAUSE)
        # ПЕРЕПИСАН: за планом теперь идёт ДОРИСОВКА ПОЛЕЙ, и дальше едет её
        # файл. Сторожим то же самое — что доделка меняет путь, а не считается
        # и выбрасывается, — но на конце цепочки, а не на середине.
        self.assertTrue(got["styled"].endswith("_9x16_full.png"), got["styled"])

    def test_a_plan_that_could_not_be_laid_is_UNMEASURED_not_a_defect(self):
        class Broken:
            @staticmethod
            def to_plan(src, dst, **kw):
                raise OSError("картинка не открылась")

        with TemporaryDirectory() as td:
            got = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                                  out_path=Path(td) / "styled.png",
                                  stylize=_stylize_ok, plan=Broken,
                                  prompt="a look, " + E.NO_BRANDS_CLAUSE)
        self.assertEqual(got["outcome"], UNMEASURED)
        # И файл остаётся ПРЕЖНИЙ, а не выдуманный: врать про путь нельзя.
        self.assertTrue(got["styled"].endswith("styled.png"))


class TheGenderGateStopsTheRunBeforeAnyGeneration(unittest.TestCase):
    """ИЗМЕРЕНО, чем кончается его отсутствие: клиент-мужчина с женской
    эстетикой получил юбку, и ВСЕ приборы при этом были зелёными."""

    class _A:
        """Подставной сосед-эстетика с настоящей сигнатурой."""

        calls = []

        @staticmethod
        def gender_of(aid):
            return "f"

        @staticmethod
        def pair_check(*, client_gender, aesthetic_gender):
            ok = client_gender == aesthetic_gender
            return {"outcome": PASS if ok else FAIL, "checked": 1,
                    "violations": 0 if ok else 1, "unmeasured": 0,
                    "note": "подставной гейт"}

        @staticmethod
        def aesthetic_file(aid):
            return f"assets/aesthetics/{aid}_f.png"

        @staticmethod
        def compose(aid):
            return {"prompt": "эстетика словами"}

        @staticmethod
        def assemble_prompt():
            return "роли, " + E.NO_BRANDS_CLAUSE

    def test_a_mismatched_gender_stops_before_the_styliser_is_called(self):
        seen = []

        def counting(**kw):
            seen.append(kw)
            return kw["out_path"]

        with TemporaryDirectory() as td:
            got = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                                  out_path=Path(td) / "styled.png",
                                  stylize=counting, plan=_PlanOk,
                                  aesthetic="y2k", client_gender="m",
                                  aesthetic_mod=self._A)
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(seen, [], "стилизатор позван при разъехавшемся поле")
        self.assertIn("генерация не запускалась", got["note"])

    def test_a_matching_gender_goes_through_and_uses_the_aesthetic_file(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ гейта: он обязан уметь пропускать, и вторая
        # картинка обязана стать ЭСТЕТИКОЙ, а не прежним референсом.
        seen = {}

        def counting(**kw):
            seen.update(kw)
            Path(kw["out_path"]).write_bytes(b"\x00" * 64)
            return kw["out_path"]

        with TemporaryDirectory() as td:
            got = E.stage_stylize(client_photo="c.png", style_ref="ЧУЖОЙ.png",
                                  out_path=Path(td) / "styled.png",
                                  stylize=counting, plan=_PlanOk,
                                  aesthetic="y2k", client_gender="f",
                                  aesthetic_mod=self._A)
        self.assertEqual(got["outcome"], PASS)
        self.assertIn("y2k_f.png", seen["style"])
        self.assertNotIn("ЧУЖОЙ", seen["style"])
        self.assertIn("эстетика словами", seen["prompt"])


class TheTemplateFlagsReachRunFromTheCommandLine(unittest.TestCase):
    """Флаг, который разбирается и теряется, выглядит рабочим до прогона."""

    def _seen(self, argv):
        seen = {}

        def fake_run(**kw):
            seen.update(kw)
            return {"exit_code": 0}

        with mock.patch.object(E, "run", fake_run):
            E.main(argv)
        return seen

    def test_the_aesthetic_and_gender_travel_to_run(self):
        got = self._seen(["--client", "c.png", "--driving", "d.mp4",
                          "--window", "100:199", "--aesthetic", "y2k",
                          "--client-gender", "f"])
        self.assertEqual(got["aesthetic"], "y2k")
        self.assertEqual(got["client_gender"], "f")

    def test_an_aesthetic_without_a_gender_is_refused(self):
        # Пол — гейт, а не удобство: без него шаблон уедет чужому клиенту.
        with self.assertRaises(SystemExit):
            self._seen(["--client", "c.png", "--driving", "d.mp4",
                        "--window", "100:199", "--aesthetic", "y2k"])

    def test_neither_style_nor_aesthetic_is_refused(self):
        with self.assertRaises(SystemExit):
            self._seen(["--client", "c.png", "--driving", "d.mp4",
                        "--window", "100:199"])

    def test_the_old_style_path_still_works_without_an_aesthetic(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ: прежний путь не сломан новым флагом.
        got = self._seen(["--client", "c.png", "--style", "s.png",
                          "--driving", "d.mp4", "--window", "100:199"])
        self.assertIsNone(got["aesthetic"])
        self.assertEqual(got["style_ref"], "s.png")


class TheOutpaintFixesTheLetterboxWithoutLosingTheRun(unittest.TestCase):
    """Поля плана видны полосами. Прибор говорит «годно»: он проверяет
    соотношение сторон и НЕ МОЖЕТ проверить, выглядит ли дополненная область
    продолжением. Увидел глаз.

    ЦЕНА ИЗМЕРЕНА НА ЧЕТЫРЁХ ТОЧКАХ, и первая была выбросом: -0.0046, потом
    +0.0636, +0.0659, +0.0786. Дорисовка стоит около +0.065 расстояния до
    клиента. По одному замеру было записано «не трогает» — неверно.
    """

    class _PlanNoExtend:
        @staticmethod
        def to_plan(src, dst, **kw):
            Path(dst).write_bytes(b"\x00" * 64)
            return {"outcome": PASS, "checked": 1, "violations": 0,
                    "unmeasured": 0, "path": str(dst), "note": "план"}

        @staticmethod
        def extend_to_plan(src, dst, *, extender=None, **kw):
            return {"outcome": UNMEASURED, "checked": 0, "violations": 0,
                    "unmeasured": 1, "path": str(src), "extended": False,
                    "note": "дорисовщик не ответил"}

    def test_a_failed_outpaint_does_NOT_sink_the_stage(self):
        # «Не смогли дорисовать» и «рефки нет» — разные события. Картинка с
        # полями хуже, но она есть, и прогон обязан идти на ней.
        with TemporaryDirectory() as td:
            got = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                                  out_path=Path(td) / "styled.png",
                                  stylize=_stylize_ok, plan=self._PlanNoExtend,
                                  prompt="a look, " + E.NO_BRANDS_CLAUSE)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertTrue(got["styled"].endswith("_9x16.png"), got["styled"])

    def test_the_extend_prompt_forbids_redrawing_the_person(self):
        # Сторож ГЛАВНОЙ строки: без неё модель понимает «дорисуй кадр» как
        # «нарисуй заново», и личность уезжает вместе с фоном.
        from ball_reel import fork_plan

        self.assertIn("do not move, rescale, recrop or alter the person",
                      fork_plan.extend_prompt())
        self.assertIn("no logo", fork_plan.extend_prompt())

    def test_removing_the_keep_clause_is_visible_in_the_prompt(self):
        from ball_reel import fork_plan

        with mock.patch.object(fork_plan, "KEEP_SUBJECT_CLAUSE", ""):
            self.assertNotIn("alter the person", fork_plan.extend_prompt())


class ThePrintedPriceFollowsTheWindowLength(unittest.TestCase):
    """ИЗМЕРЕНО 22.08: на десятисекундном прогоне стенд напечатал «$0.35», а
    счёт списал $0.70 (баланс 10.1490375 -> 9.4490375). Константа была зашита
    под пять секунд и на другой длине соврала.

    Цифра в отчёте, которой нельзя верить, ХУЖЕ отсутствующей: по ней принимают
    решения о батче.
    """

    @staticmethod
    def _probe(frames, fps):
        def prober(path):
            return {"width": 816, "height": 1104, "fps": fps, "frames": frames,
                    "note": "подставной опрос"}
        return prober

    def test_ten_seconds_is_seventy_cents(self):
        self.assertEqual(E._window_seconds("w.mp4",
                                           prober=self._probe(300, 30)), 10.0)
        self.assertEqual(E.kling_price(10.0), 0.7)

    def test_five_seconds_is_thirty_five_cents(self):
        self.assertEqual(E._window_seconds("w.mp4",
                                           prober=self._probe(150, 30)), 5.0)
        self.assertEqual(E.kling_price(5.0), 0.35)

    def test_the_stage_prints_the_price_it_will_actually_cost(self):
        with TemporaryDirectory() as td:
            got = E.stage_kling(styled="s.png", window="w.mp4",
                                out_path=Path(td) / "out.mp4",
                                upload=_upload_ok, kling=_kling_ok,
                                probe=self._probe(300, 30))
        self.assertEqual(got["numbers"]["price_usd"], 0.7)
        self.assertEqual(got["numbers"]["seconds"], 10.0)
        self.assertTrue(any("$0.7" in str(c["note"]) for c in got["checks"]),
                        [c["note"] for c in got["checks"]])

    def test_an_unmeasurable_window_falls_back_and_does_NOT_guess(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ: без длины цена неизвестна, и подставлять догадку
        # нельзя — печатается продуктовая ставка, а длина честно None.
        def broken(path):
            raise OSError("файла нет")

        with TemporaryDirectory() as td:
            got = E.stage_kling(styled="s.png", window="w.mp4",
                                out_path=Path(td) / "out.mp4",
                                upload=_upload_ok, kling=_kling_ok,
                                probe=broken)
        self.assertIsNone(got["numbers"]["seconds"])
        self.assertEqual(got["numbers"]["price_usd"], E.KLING_PRICE_USD)
