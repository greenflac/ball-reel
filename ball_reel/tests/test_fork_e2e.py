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


def _run(root: Path, log, **over):
    f = _files(root)
    kw = dict(client_photo=f["client"], style_ref=f["style"],
              driving=f["driving"], first=100, last=199,
              out_dir=root / "out", intake=_intake_ok, stylize=_stylize_ok,
              similarity=_similarity_ok, distances=_distances_ok,
              probe=_probe_ok, cutter=_cutter_ok, decode=_decode_ok,
              cuts=_cuts_ok, upload=_upload_ok, kling=_kling_ok,
              finish=_finish_ok, log=log)
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

    def test_the_measured_geometry_passes_and_anything_else_is_a_defect(self):
        self.assertEqual(self._accept()["outcome"], "годно")
        other = lambda p: {"outcome": PASS, "fps": 30.0, "frames": 99,
                           "width": 720, "height": 1280, "note": ""}
        self.assertEqual(self._accept(probe=other)["outcome"], "не годно")

    def test_the_geometry_constant_moved_flips_the_verdict(self):
        with mock.patch.object(E, "KLING_OUT_SIZE", (720, 1280)):
            self.assertEqual(self._accept()["outcome"], "не годно")

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
        self.assertIn("no brand names, no logos", built["prompt"])
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
        self.assertIn("no brand names, no logos", built["prompt"])

    def test_a_prompt_without_the_ban_reddens_the_stage(self):
        """Негативный контроль сторожа: без него проверка всегда зелена."""
        with TemporaryDirectory() as td:
            got = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                                  out_path=Path(td) / "styled.png",
                                  stylize=_stylize_ok,
                                  prompt="just make it look nice")
            self.assertEqual(got["outcome"], "не годно")
            self.assertEqual(got["checks"][0]["outcome"], "не годно")
            # И вход, на котором сторож обязан молчать (И5).
            ok = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                                 out_path=Path(td) / "styled.png",
                                 stylize=_stylize_ok,
                                 prompt="a look, " + E.NO_BRANDS_CLAUSE)
            self.assertEqual(ok["outcome"], "годно")

    def test_the_ban_text_itself_is_a_decision_constant(self):
        # Мутация константы: сторож ищет ИМЕННО её, а не любое слово про бренды.
        with mock.patch.object(E, "NO_BRANDS_CLAUSE", "no logos whatsoever"):
            with TemporaryDirectory() as td:
                got = E.stage_stylize(client_photo="c.png", style_ref="s.png",
                                      out_path=Path(td) / "styled.png",
                                      stylize=_stylize_ok,
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
