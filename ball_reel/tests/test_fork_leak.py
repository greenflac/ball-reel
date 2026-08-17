"""Поток B: ось протечки обязана КРАСНЕТЬ на нарисованном пятне вне маски.

Тест, ради которого написан модуль, один: берём кадр, копируем, ставим пятно
СНАРУЖИ маски — ось обязана это увидеть. И зеркальный к нему: то же пятно
ВНУТРИ маски ось видеть НЕ должна, иначе она меряет не границу, а картинку
целиком, и вся затея с Mix теряет смысл.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ball_reel import fork_leak as fl


def _frame(h=64, w=64, value=0.5):
    return np.full((h, w, 3), value, dtype="float64")


def _mask(h=64, w=64, box=(16, 16, 48, 48)):
    m = np.zeros((h, w), dtype=bool)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = True
    return m


class TheAxisFindsASpotOutsideTheMask(unittest.TestCase):
    """Сторож, который не умеет краснеть, — украшение."""

    def test_a_painted_spot_outside_shows_up(self):
        driving = _frame()
        output = driving.copy()
        output[2:6, 2:6] = 1.0                      # пятно вне маски
        got = fl.outside_divergence(output, driving, _mask())
        self.assertGreater(got["max"], 0.4,
                           "ось не увидела пятно вне маски — она не сторожит "
                           "ту единственную поверхность, ради которой заведена")
        self.assertGreater(got["mean"], 0.0)

    def test_the_same_spot_inside_the_mask_is_invisible_to_the_axis(self):
        """Зеркальный тест. Внутри маски модель имеет право на всё."""
        driving = _frame()
        output = driving.copy()
        output[20:44, 20:44] = 1.0                  # пятно ЦЕЛИКОМ внутри
        got = fl.outside_divergence(output, driving, _mask())
        self.assertEqual(got["max"], 0.0,
                         "ось увидела перерисовку ВНУТРИ маски — тогда она "
                         "меряет кадр целиком, а не границу")

    def test_identical_frames_read_as_zero(self):
        """Негативный контроль (И5): вход, где прибор обязан сказать «ничего»."""
        driving = _frame()
        got = fl.outside_divergence(driving.copy(), driving, _mask())
        self.assertEqual((got["mean"], got["max"]), (0.0, 0.0))

    def test_a_faint_wide_shift_is_caught_by_the_mean_when_the_max_is_small(self):
        driving = _frame()
        output = driving.copy()
        output[:16, :] += 0.02                      # слабый сдвиг широкой полосой
        got = fl.outside_divergence(output, driving, _mask())
        self.assertLess(got["max"], 0.05, "фикстура задумана слабой")
        self.assertGreater(got["mean"], 0.0,
                           "среднее слепо к равномерному сдвигу — тогда "
                           "перекрашенный фон пройдёт")

    def test_a_tiny_spot_is_caught_by_the_max_when_the_mean_is_small(self):
        driving = _frame()
        output = driving.copy()
        output[1, 1] = 1.0                          # один пиксель
        got = fl.outside_divergence(output, driving, _mask())
        self.assertLess(got["mean"], 0.001, "фикстура задумана точечной")
        self.assertEqual(got["max"], 0.5,
                         "максимум слеп к точечному пятну — тогда стёртый мяч "
                         "пройдёт, он занимает проценты площади")


class ThereAreThreeOutcomesNotTwo(unittest.TestCase):

    def test_a_mask_that_eats_the_frame_is_unmeasured_not_leak_free(self):
        driving = _frame()
        full = np.ones((64, 64), dtype=bool)
        got = fl.outside_divergence(driving.copy(), driving, full)
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("НЕ «протечки нет»", got["note"])

    def test_the_share_bar_is_guarded_in_both_directions(self):
        """Т1: константа-решение подменяется строже и слабее."""
        driving = _frame()
        m = np.ones((64, 64), dtype=bool)
        m[:2, :] = False                     # ровно 3.125% вне маски
        original = fl.MIN_OUTSIDE_SHARE
        try:
            fl.MIN_OUTSIDE_SHARE = 0.01
            self.assertIsNotNone(
                fl.outside_divergence(driving.copy(), driving, m)["mean"],
                "порог ослаблен, а измерение всё равно отказано")
            fl.MIN_OUTSIDE_SHARE = 0.5
            self.assertEqual(
                fl.outside_divergence(driving.copy(), driving, m)["outcome"],
                fl.UNMEASURED,
                "порог поднят выше доступного, а измерение всё равно прошло — "
                "значит порог ни на что не влияет")
        finally:
            fl.MIN_OUTSIDE_SHARE = original

    def test_mismatched_frames_are_unmeasured(self):
        got = fl.outside_divergence(_frame(32, 32), _frame(64, 64), _mask())
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("разного размера", got["note"])

    def test_a_mask_of_the_wrong_shape_is_unmeasured(self):
        got = fl.outside_divergence(_frame(), _frame(), _mask(32, 32))
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("не по кадру", got["note"])


class TheVerdictRefusesToJudgeWithoutAMeasuredFloor(unittest.TestCase):
    """Планка выдумана — вердикта нет. Это не строгость, это честность."""

    def test_no_floor_means_unmeasured_even_when_divergence_is_tiny(self):
        got = fl.verdict({"mean": 0.000001}, {"floor": None, "note": "нет ffmpeg"})
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("сколько из него стоит кодек", got["note"])

    def test_no_divergence_means_unmeasured_too(self):
        got = fl.verdict({"mean": None, "note": "маска съела кадр"},
                         {"floor": 0.001})
        self.assertEqual(got["outcome"], fl.UNMEASURED)

    def test_a_zero_floor_is_refused_as_impossible(self):
        got = fl.verdict({"mean": 0.01}, {"floor": 0.0})
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("сам с собой", got["note"])

    def test_divergence_under_the_multiplier_is_the_codec(self):
        got = fl.verdict({"mean": 0.0015}, {"floor": 0.001})
        self.assertEqual(got["outcome"], fl.PASS)
        self.assertEqual(got["ratio"], 1.5)

    def test_divergence_over_the_multiplier_is_a_leak(self):
        got = fl.verdict({"mean": 0.005}, {"floor": 0.001})
        self.assertEqual(got["outcome"], fl.FAIL)
        self.assertIn("ПРОТЕЧКА", got["note"])

    def test_the_multiplier_is_guarded_in_both_directions(self):
        """Т1 на LEAK_OVER_FLOOR."""
        d, f = {"mean": 0.0015}, {"floor": 0.001}
        self.assertEqual(fl.verdict(d, f, over=1.2)["outcome"], fl.FAIL,
                         "порог ужесточён, а вердикт не изменился")
        self.assertEqual(fl.verdict({"mean": 0.005}, f, over=99.0)["outcome"],
                         fl.PASS,
                         "порог снят, а протечка всё равно объявлена")

    def test_the_note_carries_both_numbers_and_the_ratio(self):
        got = fl.verdict({"mean": 0.005}, {"floor": 0.001})
        for piece in ("0.005", "0.001", "5.0x"):
            self.assertIn(piece, got["note"])


class TheLocalChannelIsJudgedByItsOwnFloor(unittest.TestCase):
    """Найдено замером: локальный шум кодека в 16 раз выше среднего.

    На настоящих кадрах при crf 18 среднее 0.004513, максимум 0.071895. Судить
    максимум средней планкой значило бы объявлять протечкой каждый блок сжатия;
    не судить его вовсе — терять канал, ради которого он заведён.
    """

    FLOOR = {"floor": 0.004513, "floor_max": 0.071895}

    def test_a_spot_within_the_local_floor_is_the_codec(self):
        got = fl.verdict({"mean": 0.005, "max": 0.10}, self.FLOOR)
        self.assertEqual(got["outcome"], fl.PASS,
                         "локальное отличие 0.10 при локальной планке 0.072 — "
                         "это блок сжатия, а не стёртый мяч")

    def test_a_spot_far_over_the_local_floor_is_a_leak_even_when_the_mean_is_fine(self):
        got = fl.verdict({"mean": 0.005, "max": 0.6}, self.FLOOR)
        self.assertEqual(got["outcome"], fl.FAIL,
                         "точечная протечка прошла: среднее её не видит, а "
                         "локального порога нет")
        self.assertGreater(got["spot_ratio"], fl.LEAK_OVER_FLOOR)

    def test_judging_the_max_by_the_mean_floor_would_have_failed_the_codec(self):
        """Прямая проверка того, что новая планка НУЖНА, а не украшает.

        0.10 против средней планки — это 22x, то есть по старому правилу
        протечка. Против своей — 1.4x, то есть кодек.
        """
        self.assertGreater(0.10 / self.FLOOR["floor"], fl.LEAK_OVER_FLOOR)
        self.assertLess(0.10 / self.FLOOR["floor_max"], fl.LEAK_OVER_FLOOR)

    def test_a_missing_local_floor_says_so_instead_of_judging_silently(self):
        got = fl.verdict({"mean": 0.005, "max": 0.9}, {"floor": 0.004513})
        self.assertIsNone(got["spot_ratio"])
        self.assertIn("не судится", got["note"])
        self.assertEqual(got["outcome"], fl.PASS,
                         "без локальной планки максимум обязан молчать, а не "
                         "решать")


class TheFloorIsMeasuredNotAssumed(unittest.TestCase):

    def test_too_few_frames_is_unmeasured(self):
        got = fl.noise_floor(["a.png"])
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("по одной точке планку не ставят", got["note"])

    def test_the_frame_count_bar_is_guarded(self):
        original = fl.MIN_FLOOR_FRAMES
        try:
            fl.MIN_FLOOR_FRAMES = 99
            got = fl.noise_floor(["a.png"] * 10)
            self.assertEqual(got["outcome"], fl.UNMEASURED)
            self.assertIn("99", got["note"])
        finally:
            fl.MIN_FLOOR_FRAMES = original

    @unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg не в PATH")
    def test_a_real_roundtrip_gives_a_floor_above_zero(self):
        """Главный смысл планки: ноль недостижим, и это ИЗМЕРЯЕТСЯ."""
        from PIL import Image

        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(4):
                arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
                p = Path(tmp) / f"{i:03d}.png"
                Image.fromarray(arr).save(p)
                paths.append(p)
            got = fl.noise_floor(paths)
        self.assertEqual(got["outcome"], fl.PASS, got["note"])
        self.assertGreater(got["floor"], 0.0,
                           "перекодирование не изменило ни пикселя — такого не "
                           "бывает, значит замер сравнивал файл сам с собой")
        self.assertIn("ИЗМЕРЕНА", got["note"])

    @unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg не в PATH")
    def test_a_harsher_codec_gives_a_higher_floor(self):
        """Негативный контроль к планке (И5): она обязана шевелиться.

        Планка, не реагирующая на смену сжатия, меряет не сжатие.
        """
        from PIL import Image

        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(4):
                arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
                p = Path(tmp) / f"{i:03d}.png"
                Image.fromarray(arr).save(p)
                paths.append(p)
            gentle = fl.noise_floor(paths, crf=10)
            harsh = fl.noise_floor(paths, crf=45)
        self.assertGreater(harsh["floor"], gentle["floor"],
                           f"crf 45 дал {harsh['floor']}, crf 10 — "
                           f"{gentle['floor']}: планка не следит за кодеком")


if __name__ == "__main__":
    unittest.main()
