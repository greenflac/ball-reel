"""Склейка петли: стыковой кадр не дублируется, длина ложится на шаг, три исхода.

ПОЧЕМУ ЗДЕСЬ НЕТ КАРТИНОК И FFMPEG (Т4). Склейка не смотрит внутрь кадра: она
раскладывает ФАЙЛЫ по порядку. Поэтому «кадр» здесь — текстовый файл с
расширением `.png`, в котором лежит собственный номер, и порядок склейки
проверяется ЧТЕНИЕМ СОДЕРЖИМОГО, а не длиной списка: список нужной длины можно
получить и перепутав порядок. Пиксельная проверка стыка на боевом материале —
дело прогона (П3), и её числа лежат в отчёте смены, а не здесь.

ОЖИДАЕМОЕ — ЛИТЕРАЛ (Т2). Ни одно число и ни одна строка вердикта не берутся
из проверяемого модуля: 145 написано цифрами, «годно» написано словом. Импорт
`fork_splice.LENGTH_SLACK` в ожидание поехал бы вместе с константой и промолчал
ровно тогда, когда её и надо сторожить.

НЕГАТИВНЫЙ КОНТРОЛЬ С ОБЕИХ СТОРОН (И5) у каждого прибора: вход, где он обязан
сказать «не годно» (петля за краем материала, петля из одного кадра, длина не
набирается), и вход, где он обязан пропустить (боевая петля 49 кадров на 24 к/с).
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from .. import fork_splice as fs

# Боевые числа материала (`assets/README.md`), проставлены руками:
# 720x1278, 24 к/с, 362 кадра; лучшая петля 114..162 — 49 кадров.
DRIVING_FPS = 24
DRIVING_FRAMES = 362
LOOP_I, LOOP_J = 114, 162


def make_frames(root, n, *, start=0):
    """`n` файлов-кадров; в каждом лежит его собственный номер."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for k in range(start, start + n):
        p = root / f"{k:05d}.png"
        p.write_text(f"кадр {k}", encoding="utf-8")
        out.append(p)
    return out


def read_written(out_dir):
    """Что реально легло, по порядку имён: список исходных номеров."""
    return [int(p.read_text(encoding="utf-8").split()[1])
            for p in sorted(Path(out_dir).glob("*.png"))]


class TheSeamFrameIsNeverDuplicated(unittest.TestCase):
    """N*(L-1)+1, а не N*L. Дубль — заедание на каждом стыке."""

    def test_the_battle_loop_of_49_frames_gives_145_and_not_147(self):
        got = fs.sequence_indices(LOOP_I, LOOP_J, 3)
        self.assertEqual(len(got), 145)
        self.assertNotEqual(len(got), 147)   # 3*49 — цена дубля, в кадрах

    def test_the_order_is_spelled_out_frame_by_frame(self):
        # Петля [0..4] — пять кадров, шаг склейки четыре. Кадр 4 — тот же
        # момент, что кадр 0, поэтому в теле его нет, а в конце он один.
        self.assertEqual(fs.sequence_indices(0, 4, 2),
                         [0, 1, 2, 3, 0, 1, 2, 3, 4])
        self.assertEqual(fs.sequence_indices(0, 4, 1), [0, 1, 2, 3, 4])

    def test_no_two_neighbours_are_the_same_frame(self):
        seq = fs.sequence_indices(LOOP_I, LOOP_J, 4)
        self.assertEqual([k for k in range(len(seq) - 1)
                          if seq[k] == seq[k + 1]], [])

    def test_the_closing_frame_appears_exactly_once(self):
        seq = fs.sequence_indices(LOOP_I, LOOP_J, 3)
        self.assertEqual(seq.count(LOOP_J), 1)
        self.assertEqual(seq.count(LOOP_I), 3)   # ровно по разу на повтор

    def test_one_repeat_is_the_loop_itself(self):
        self.assertEqual(len(fs.sequence_indices(LOOP_I, LOOP_J, 1)), 49)

    def test_nonsense_arguments_are_refused_not_guessed(self):
        for args in [(0, 0, 1), (5, 3, 1), (-1, 4, 1)]:
            with self.assertRaises(ValueError):
                fs.sequence_indices(*args)
        with self.assertRaises(ValueError):
            fs.sequence_indices(0, 4, 0)
        with self.assertRaises(TypeError):
            fs.sequence_indices(0, 4, True)


class TheLoopMustLieInsideTheMaterial(unittest.TestCase):
    def test_the_battle_loop_passes(self):
        got = fs.loop_bounds_ok(LOOP_I, LOOP_J, DRIVING_FRAMES)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["frames"], 49)

    def test_the_last_frame_of_the_material_is_still_inside(self):
        self.assertEqual(fs.loop_bounds_ok(0, 361, 362)["outcome"], "годно")

    def test_one_frame_past_the_end_is_refused_with_the_number(self):
        got = fs.loop_bounds_ok(320, 362, DRIVING_FRAMES)
        self.assertEqual(got["outcome"], "не годно")
        self.assertIn("362", got["note"])

    def test_a_loop_longer_than_the_material_is_refused(self):
        self.assertEqual(fs.loop_bounds_ok(0, 500, 100)["outcome"], "не годно")

    def test_a_loop_of_one_frame_is_not_a_loop(self):
        got = fs.loop_bounds_ok(7, 7, 100)
        self.assertEqual(got["outcome"], "не годно")

    def test_a_loop_of_two_frames_is_admitted_by_bounds(self):
        # Границы её пропускают: годность по ДЛИНЕ судит план повторов, и
        # свернуть два разных вердикта в один значило бы потерять причину.
        self.assertEqual(fs.loop_bounds_ok(7, 8, 100)["outcome"], "годно")


class SecondsAreCountedAtTheSourceRate(unittest.TestCase):
    """145 кадров — это 6.04 с на 24 к/с и 4.83 на 30. Разница в четверть."""

    def test_twenty_four_is_inherited(self):
        got = fs.playback_fps(24)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["fps"], 24)

    def test_thirty_is_ours_and_passes(self):
        self.assertEqual(fs.playback_fps(30)["outcome"], "годно")

    def test_sixty_is_refused_and_sends_to_the_decoder(self):
        got = fs.playback_fps(60)
        self.assertEqual(got["outcome"], "не годно")
        self.assertIn("fork_video", got["note"])

    def test_unknown_rate_is_the_third_outcome_not_thirty(self):
        got = fs.playback_fps(None)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIsNone(got["fps"])

    def test_zero_and_text_are_reported_not_raised(self):
        for bad in (0, -24, "24"):
            got = fs.playback_fps(bad)
            self.assertEqual(got["outcome"], "не годно")

    def test_the_plan_of_the_battle_loop_is_six_oh_four_not_four_eight_three(self):
        got = fs.choose_repeats(49, 6.04, fps=24)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["plan"]["seconds"], 6.04)
        self.assertNotEqual(got["plan"]["seconds"], 4.83)


class TheLengthMustLandOnTheWrapperStep(unittest.TestCase):
    """Прижатие вниз молчит, и прижатая склейка обрывает последний повтор."""

    def test_the_battle_loop_loses_nothing_to_the_step(self):
        got = fs.admissible_plans(49, fps=24)
        self.assertEqual([p["frames"] for p in got["kept"]], [145, 193])
        self.assertEqual(got["dropped_step"], [])

    def test_a_loop_of_43_frames_loses_the_odd_repeats(self):
        # Шаг склейки 42: длина 4k+1 выходит только при ЧЁТНОМ числе повторов.
        got = fs.admissible_plans(43, fps=24)
        self.assertEqual([p["frames"] for p in got["kept"]], [169])
        self.assertEqual([p["frames"] for p in got["dropped_step"]], [127, 211])

    def test_every_kept_length_is_of_the_form_four_k_plus_one(self):
        for length in (41, 43, 45, 49, 53, 61):
            for p in fs.admissible_plans(length, fps=24)["kept"]:
                self.assertEqual((p["frames"] - 1) % 4, 0,
                                 f"петля {length}: {p['frames']} не 4k+1")


class TheOrderIsEitherFilledOrRefusedWithNumbers(unittest.TestCase):
    def test_six_oh_four_out_of_the_battle_loop_is_three_repeats(self):
        got = fs.choose_repeats(49, 6.04, fps=24)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["plan"]["repeats"], 3)
        self.assertEqual(got["plan"]["frames"], 145)

    def test_eight_seconds_is_four_repeats(self):
        got = fs.choose_repeats(49, 8.04, fps=24)
        self.assertEqual(got["plan"]["repeats"], 4)
        self.assertEqual(got["plan"]["frames"], 193)

    def test_ten_seconds_out_of_a_two_second_loop_is_not_filled_silently(self):
        # Пятый повтор — 241 кадр, 10.04 с, за потолком. Ближайшее 8.04 с.
        got = fs.choose_repeats(49, 10.0, fps=24)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIsNone(got["plan"])
        self.assertIn("193", got["note"])
        self.assertIn("8.04", got["note"])

    def test_a_reachable_order_from_the_43_frame_loop_is_filled(self):
        got = fs.choose_repeats(43, 7.0, fps=24)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["plan"]["frames"], 169)

    def test_an_unreachable_order_from_the_43_frame_loop_says_what_is_reachable(self):
        got = fs.choose_repeats(43, 5.3, fps=24)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIn("169", got["note"])

    def test_the_slack_is_four_frames_and_it_is_a_boundary(self):
        # ОБЕ СТОРОНЫ ГРАНИЦЫ, а не одна (И5), и шаг между ними ровно четыре:
        # заказ и склейка оба лежат на сетке 4k+1, поэтому промах всегда
        # кратен четырём, и допуск в 4 кадра — это «ровно один шаг обёртки».
        # 7.00 с при 24 к/с = 168 кадров, обёртка прижмёт к 165; 169-165 = +4:
        self.assertEqual(fs.choose_repeats(43, 7.0, fps=24)["outcome"], "годно")
        # 7.40 с = 178 -> 177; 169-177 = -8, два шага — это уже другая длина:
        self.assertEqual(fs.choose_repeats(43, 7.4, fps=24)["outcome"],
                         "не смогли проверить")
        # 5.30 с = 127 -> 125; ближайшее 169, промах +44 — далеко за допуском.
        self.assertEqual(fs.choose_repeats(43, 5.3, fps=24)["outcome"],
                         "не смогли проверить")

    def test_a_rate_we_do_not_know_gives_no_plan(self):
        got = fs.choose_repeats(49, 6.04, fps=None)
        self.assertEqual(got["outcome"], "не смогли проверить")

    def test_a_loop_of_one_frame_gives_no_plan(self):
        got = fs.choose_repeats(1, 6.0, fps=24)
        self.assertEqual(got["outcome"], "не годно")

    def test_an_order_outside_the_product_band_is_refused(self):
        for seconds in (4.0, 11.0, 0.0):
            got = fs.choose_repeats(49, seconds, fps=24)
            self.assertEqual(got["outcome"], "не годно", seconds)

    def test_a_loop_whose_every_length_is_off_the_step_gives_nothing(self):
        # Петля 200 кадров при 24 к/с: один повтор — 200 кадров (8.33 с), в
        # полосе, но обёртка прижмёт к 197; два повтора — 399 кадров (16.6 с),
        # за потолком. Не остаётся НИ ОДНОЙ длины, и это третий исход.
        got = fs.choose_repeats(200, 6.0, fps=24)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIn("5.0-10.0", got["note"])
        self.assertIn("197", got["note"])

    def test_the_same_loop_at_a_slower_rate_is_filled(self):
        # Негативный контроль к предыдущему (И5): прибор обязан шевельнуться.
        # 200 кадров при 30 к/с — 6.67 с, и уже два повтора влезают: 399/30.
        self.assertEqual(fs.choose_repeats(201, 6.7, fps=30)["outcome"], "годно")


class TheTwoPlacesThatKnowTheLengthAreCompared(unittest.TestCase):
    """Е1: расхождение формулы и построения ловит машина, а не глаз."""

    def test_agreeing_plan_returns_the_length(self):
        self.assertEqual(fs._agree({"repeats": 3, "frames": 145}, 114, 162), 145)

    def test_a_plan_that_says_147_falls(self):
        with self.assertRaises(AssertionError):
            fs._agree({"repeats": 3, "frames": 147}, 114, 162)


class FramesArePlacedWithoutCopyingBytes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = make_frames(self.root / "src", 10)
        self.addCleanup(self.tmp.cleanup)

    def test_a_hard_link_is_the_same_inode(self):
        dst = self.root / "one.png"
        mode = fs.place(self.src[3], dst, prefer="жёсткая ссылка")
        self.assertEqual(mode, "жёсткая ссылка")
        self.assertEqual(os.stat(dst).st_ino, os.stat(self.src[3]).st_ino)

    def test_a_copy_is_a_different_inode(self):
        dst = self.root / "two.png"
        mode = fs.place(self.src[3], dst, prefer="копия")
        self.assertEqual(mode, "копия")
        self.assertNotEqual(os.stat(dst).st_ino, os.stat(self.src[3]).st_ino)
        self.assertEqual(dst.read_text(encoding="utf-8"), "кадр 3")

    def test_when_the_hard_link_is_impossible_it_falls_back_and_says_so(self):
        real = os.link

        def refuse(*a, **k):
            raise OSError("разные файловые системы")

        os.link = refuse
        try:
            mode = fs.place(self.src[3], self.root / "three.png")
        finally:
            os.link = real
        self.assertEqual(mode, "символическая")

    def test_the_sequence_lands_in_the_right_order(self):
        out = self.root / "out"
        got = fs.write_sequence(self.src, fs.sequence_indices(0, 4, 2), out)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["written"], 9)
        self.assertEqual(read_written(out), [0, 1, 2, 3, 0, 1, 2, 3, 4])

    def test_the_names_are_the_ones_the_decoder_writes(self):
        out = self.root / "out"
        fs.write_sequence(self.src, [0, 1, 2], out)
        self.assertEqual([p.name for p in sorted(out.glob("*.png"))],
                         ["00001.png", "00002.png", "00003.png"])

    def test_hard_links_cost_no_bytes_and_copies_do(self):
        linked = fs.write_sequence(self.src, [0, 1, 2], self.root / "a")
        copied = fs.write_sequence(self.src, [0, 1, 2], self.root / "b",
                                   prefer="копия")
        self.assertEqual(linked["bytes"], 0)
        self.assertGreater(copied["bytes"], 0)

    def test_a_busy_directory_is_not_overwritten_silently(self):
        out = self.root / "out"
        fs.write_sequence(self.src, [0, 1, 2], out)
        again = fs.write_sequence(self.src, [4, 5], out)
        self.assertEqual(again["outcome"], "не смогли проверить")
        self.assertEqual(read_written(out), [0, 1, 2])

    def test_overwrite_leaves_no_foreign_frames_behind(self):
        out = self.root / "out"
        fs.write_sequence(self.src, [0, 1, 2, 3, 4], out)
        fs.write_sequence(self.src, [7, 8], out, overwrite=True)
        self.assertEqual(read_written(out), [7, 8])

    def test_a_file_in_place_of_the_directory_is_refused(self):
        busy = self.root / "busy.png"
        busy.write_text("не каталог", encoding="utf-8")
        got = fs.write_sequence(self.src, [0], busy)
        self.assertEqual(got["outcome"], "не годно")


class TheWholeSpliceOnADirectoryOfFrames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_frames(self.root / "src", DRIVING_FRAMES)
        self.addCleanup(self.tmp.cleanup)

    def splice(self, **kw):
        kw.setdefault("fps", DRIVING_FPS)
        return fs.splice(self.root / "src", (LOOP_I, LOOP_J), kw.pop("seconds", 6.04),
                         self.root / "out", **kw)

    def test_the_battle_order_lands_as_145_frames_of_6_04_seconds(self):
        rep = self.splice()
        self.assertEqual(rep["outcome"], "годно")
        self.assertEqual(rep["frames"], 145)
        self.assertEqual(rep["repeats"], 3)
        self.assertEqual(rep["seconds"], 6.04)
        self.assertEqual(len(list((self.root / "out").glob("*.png"))), 145)

    def test_what_landed_is_the_loop_repeated_without_the_duplicate(self):
        self.splice()
        got = read_written(self.root / "out")
        body = list(range(LOOP_I, LOOP_J))
        self.assertEqual(got, body * 3 + [LOOP_J])
        self.assertEqual(got[47], 161)
        self.assertEqual(got[48], 114)   # стык: не 162, иначе дубль момента

    def test_the_steps_are_named_and_ordered_cheap_first(self):
        rep = self.splice()
        self.assertEqual([s["step"] for s in rep["steps"]],
                         ["кадры", "частота", "петля", "план", "запись"])

    def test_a_loop_outside_the_material_writes_nothing(self):
        rep = fs.splice(self.root / "src", (350, 400), 6.04, self.root / "out",
                        fps=DRIVING_FPS)
        self.assertEqual(rep["outcome"], "не годно")
        self.assertFalse((self.root / "out").exists())

    def test_an_unreachable_length_writes_nothing(self):
        rep = self.splice(seconds=10.0)
        self.assertEqual(rep["outcome"], "не смогли проверить")
        self.assertFalse((self.root / "out").exists())

    def test_an_unknown_rate_writes_nothing(self):
        rep = fs.splice(self.root / "src", (LOOP_I, LOOP_J), 6.04,
                        self.root / "out", fps=None)
        self.assertEqual(rep["outcome"], "не смогли проверить")
        self.assertFalse((self.root / "out").exists())

    def test_an_empty_source_is_refused(self):
        (self.root / "empty").mkdir()
        rep = fs.splice(self.root / "empty", (0, 4), 6.04, self.root / "out",
                        fps=24)
        self.assertEqual(rep["outcome"], "не годно")

    def test_a_source_that_is_neither_directory_nor_file(self):
        rep = fs.splice(self.root / "нет-такого", (0, 4), 6.04,
                        self.root / "out", fps=24)
        self.assertEqual(rep["outcome"], "не смогли проверить")

    def test_a_video_source_is_decoded_by_the_injected_decoder(self):
        video = self.root / "driving.mp4"
        video.write_text("не видео, но файл", encoding="utf-8")
        paths = sorted((self.root / "src").glob("*.png"))

        def decoder(src, dst, **kw):
            return {"outcome": "годно", "paths": paths, "fps_in": 24.0,
                    "fps_out": 24.0, "note": "подменённый раскодировщик"}

        rep = fs.splice(video, (LOOP_I, LOOP_J), 6.04, self.root / "out",
                        decode=decoder)
        self.assertEqual(rep["outcome"], "годно")
        self.assertEqual(rep["frames"], 145)
        self.assertEqual(rep["fps"], 24.0)   # снята с файла, а не наша 30

    def test_a_decoder_that_failed_stops_the_run(self):
        video = self.root / "driving.mp4"
        video.write_text("не видео", encoding="utf-8")

        def decoder(src, dst, **kw):
            return {"outcome": "не годно", "paths": [],
                    "note": "файл не видео"}

        rep = fs.splice(video, (LOOP_I, LOOP_J), 6.04, self.root / "out",
                        decode=decoder)
        self.assertEqual(rep["outcome"], "не годно")


class TheEntryPoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_frames(self.root / "src", DRIVING_FRAMES)
        self.addCleanup(self.tmp.cleanup)

    def run_main(self, argv):
        """Точка входа печатает — в тесте её вывод ловится, а не льётся."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = fs.main(argv)
        return code, buf.getvalue()

    def test_the_loop_argument_is_parsed(self):
        self.assertEqual(fs.parse_loop("114..162"), (114, 162))

    def test_a_broken_loop_argument_is_refused(self):
        for bad in ("114-162", "114..", "..", "a..b", "114..162..3"):
            with self.assertRaises(ValueError):
                fs.parse_loop(bad)

    def test_the_command_returns_zero_on_the_battle_order(self):
        code, text = self.run_main(
            [str(self.root / "src"), "--loop", "114..162", "--seconds", "6.04",
             "--out", str(self.root / "out"), "--fps", "24"])
        self.assertEqual(code, 0)
        self.assertIn("145", text)
        self.assertIn("6.04", text)
        self.assertEqual(len(list((self.root / "out").glob("*.png"))), 145)

    def test_the_command_returns_one_on_a_loop_outside_the_material(self):
        code, _ = self.run_main(
            [str(self.root / "src"), "--loop", "350..400", "--seconds", "6.04",
             "--out", str(self.root / "out"), "--fps", "24"])
        self.assertEqual(code, 1)

    def test_the_command_returns_two_when_the_length_is_unreachable(self):
        code, _ = self.run_main(
            [str(self.root / "src"), "--loop", "114..162", "--seconds", "10",
             "--out", str(self.root / "out"), "--fps", "24"])
        self.assertEqual(code, 2)

    def test_the_command_returns_one_on_a_broken_loop_argument(self):
        code, _ = self.run_main(
            [str(self.root / "src"), "--loop", "114-162", "--seconds", "6.04",
             "--out", str(self.root / "out")])
        self.assertEqual(code, 1)

    def test_the_three_exit_codes_are_distinct(self):
        self.assertEqual(sorted(fs.EXIT_BY_OUTCOME.values()), [0, 1, 2])


class WritingNeverEatsTheSource(unittest.TestCase):
    """Дефект, найденный прогоном: назначение = источник -> материала нет.

    ИЗМЕРЕНО до починки на 6 кадрах: остаётся 121 файл, читается 0, вердикт
    «годно», код 0, в отчёте «своих байт на диске 0» — читается как экономия
    на жёстких ссылках. Стирание идёт ДО чтения источника, а `os.symlink`
    удаётся на несуществующую цель, поэтому склейка ещё и рапортует «легло».
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_the_source_directory_is_refused_as_destination(self):
        src = make_frames(self.root / "src", 6)
        rep = fs.write_sequence(src, [0, 1, 2, 0], self.root / "src",
                                overwrite=True)
        self.assertEqual(rep["outcome"], "не годно")
        self.assertEqual(rep["written"], 0)

    def test_the_source_frames_are_still_on_disk_and_readable(self):
        src = make_frames(self.root / "src", 6)
        before = [Path(p).read_text() for p in src]
        fs.write_sequence(src, [0, 1, 2, 0], self.root / "src", overwrite=True)
        after = sorted((self.root / "src").glob("*.png"))
        self.assertEqual(len(after), 6)
        self.assertEqual([p.read_text() for p in after], before)

    def test_the_refusal_names_how_many_frames_clashed(self):
        src = make_frames(self.root / "src", 6)
        rep = fs.write_sequence(src, [0, 1], self.root / "src", overwrite=True)
        self.assertIn("6 из 6", rep["note"])

    def test_a_different_directory_still_writes(self):
        # Негативный контроль (И5): проверка обязана пропускать нормальный ход.
        src = make_frames(self.root / "src", 6)
        rep = fs.write_sequence(src, [0, 1, 2, 0], self.root / "out")
        self.assertEqual(rep["outcome"], "годно")
        self.assertEqual(rep["written"], 4)

    def test_a_dangling_symlink_is_not_reported_as_placed(self):
        # Е2: отчёт о том, что ИСПОЛНИЛОСЬ. Ссылка на несуществующий файл
        # создаётся успешно; «символическая» про неё — ложь в вердикте.
        missing = self.root / "нет-такого.png"
        dst = self.root / "куда.png"
        with self.assertRaises(OSError):
            fs.place(missing, dst, prefer="символическая")
        self.assertFalse(dst.is_symlink())
        self.assertFalse(dst.exists())


class TheModuleDoesNotPromiseSeamlessness(unittest.TestCase):
    """Слова, которых у склейки нет права говорить: она не судит стык."""

    def test_no_verdict_calls_the_seam_seamless(self):
        rep = fs.choose_repeats(49, 6.04, fps=24)
        self.assertNotIn("бесшов", rep["note"].lower())


if __name__ == "__main__":
    unittest.main()
