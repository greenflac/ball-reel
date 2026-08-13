"""Сегментация кожи и ткани: проверяется решение, а не «функция не упала».

Основная часть тестов НЕ ТРЕБУЕТ ВЕСОВ: маска подаётся руками. Так и надо —
вопрос здесь не «хорошо ли сегментирует MediaPipe» (это её забота, не наша), а
«правильно ли мы поступаем с тем, что она вернула». Для второго синтетическая
маска даже лучше: в ней доля кожи задана нами, а не угадывается.

Два теста весов всё-таки требуют и пропускаются без них. Первый сверяет имена
классов с метаданными самого файла модели — потому что на этом проекте уже был
случай, когда имя весов оказалось выдуманным, и код писался против
несуществующего API. Второй прогоняет настоящий кадр.
"""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

FRAME = Path(__file__).resolve().parents[2] / "marktest_ref.png"


def _mask(size=100, skin_box=None):
    """Булева маска кожи: True там, где кожа. Доля задана нами, а не моделью."""
    import numpy as np

    m = np.zeros((size, size), dtype=bool)
    if skin_box:
        x0, y0, x1, y1 = skin_box
        m[y0:y1, x0:x1] = True
    return m


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class AbsentWeightsAreAStateNotACrash(unittest.TestCase):
    """Отсутствие сегментации — это отсутствие СВЕДЕНИЙ, а не запрет."""

    def setUp(self):
        from ball_reel import bodyparts

        self.b = bodyparts

    def test_a_missing_model_is_reported_with_the_command_to_fix_it(self):
        why = self.b.why_unavailable("/nowhere/nothing.tflite")
        self.assertIn("curl", why)
        self.assertIn(self.b.MODEL_URL, why)
        self.assertFalse(self.b.available("/nowhere/nothing.tflite"))

    def test_no_model_is_neither_yes_nor_no(self):
        # Самое важное решение модуля. Молча вернуть True значит выдать
        # незнание за разрешение и нарисовать примету поверх рукава; молча
        # вернуть False значит отключить фичу на любой машине без весов.
        got = self.b.paintable("frame.png", (0, 0, 10, 10),
                               model="/nowhere/nothing.tflite")
        self.assertIsNone(got["ok"])
        self.assertEqual(got["state"], "no_model")


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheVerdictOnAWindowIsAboutSKINSHARE(unittest.TestCase):
    def setUp(self):
        from ball_reel import bodyparts

        self.b = bodyparts

    def test_a_window_fully_on_skin_is_paintable(self):
        got = self.b.paintable(None, (20, 20, 40, 40),
                               mask=_mask(skin_box=(0, 0, 100, 100)))
        self.assertTrue(got["ok"])
        self.assertEqual(got["skin_share"], 1.0)

    def test_a_window_on_fabric_is_refused_and_the_reason_is_named(self):
        # Ровно тот отказ, ради которого модуль написан. Замерено на живом
        # кадре: в окне на голени кожи 0%, потому что на человеке леггинсы.
        # Без этой проверки примета легла бы на ткань, а метрика отрапортовала
        # бы честный контраст — число правда, картинка нет.
        got = self.b.paintable(None, (20, 20, 40, 40), mask=_mask())
        self.assertFalse(got["ok"])
        self.assertEqual(got["skin_share"], 0.0)
        self.assertIn("одета", got["note"])

    def test_the_bar_is_read_from_the_module_and_bites_from_both_sides(self):
        # Входы литеральные, порог из модуля: тест, берущий вход из константы,
        # которую сторожит, двигается вместе с ней и никогда не падает.
        bar = self.b.MIN_SKIN_SHARE
        self.assertGreater(bar, 0.0)
        self.assertLess(bar, 1.0)
        # Окно 100x10; кожей закрашено столько столбцов, чтобы доля легла по
        # обе стороны порога.
        below = int((bar - 0.05) * 100)
        above = int((bar + 0.05) * 100)
        self.assertFalse(self.b.paintable(
            None, (0, 0, 100, 10), mask=_mask(skin_box=(0, 0, below, 100)))["ok"])
        self.assertTrue(self.b.paintable(
            None, (0, 0, 100, 10), mask=_mask(skin_box=(0, 0, above, 100)))["ok"])

    def test_a_window_off_the_frame_is_not_measured_rather_than_refused(self):
        got = self.b.paintable(None, (200, 200, 260, 260), mask=_mask())
        self.assertIsNone(got["ok"])
        self.assertEqual(got["state"], "outside_frame")

    def test_a_window_is_clipped_to_the_frame_not_wrapped_around(self):
        # Отрицательные координаты приходят от `locate`, когда примета у края
        # кадра. Питоновский срез с отрицательным началом молча взял бы
        # ПРОТИВОПОЛОЖНЫЙ край картинки, и доля кожи посчиталась бы по чужому
        # месту.
        m = _mask(skin_box=(0, 0, 100, 20))          # кожа только сверху
        got = self.b.paintable(None, (-30, -30, 30, 10), mask=m)
        self.assertEqual(got["state"], "measured")
        self.assertEqual(got["skin_share"], 1.0)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheMaskIsCoarseAndSaysSoOutLoud(unittest.TestCase):
    """Оговорка, которую нельзя оставлять в голове у автора."""

    def setUp(self):
        from ball_reel import bodyparts

        self.b = bodyparts

    def test_the_coarseness_scales_with_the_frame(self):
        # Модель решает на стороне 256. На кадре 720x1278 это ровно 5 px на
        # одно решение — то самое число, которым потом задаётся смягчение края.
        small = self.b.edge_coarseness((256, 256))["pixels_per_decision"]
        real = self.b.edge_coarseness((720, 1278))["pixels_per_decision"]
        self.assertEqual(small, 1.0)
        self.assertAlmostEqual(real, 1278 / 256, places=2)
        self.assertGreater(real, small)

    def test_the_native_side_is_read_from_the_module(self):
        self.assertEqual(
            self.b.edge_coarseness((self.b.NATIVE_SIDE, self.b.NATIVE_SIDE))
            ["pixels_per_decision"], 1.0)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class SkinAndFabricAreNotTheSameCLASSES(unittest.TestCase):
    def setUp(self):
        from ball_reel import bodyparts

        self.b = bodyparts

    def test_hair_and_accessories_are_not_skin(self):
        # Татуировка под дужкой очков — не примета на коже, и волосы тоже не
        # холст. Разделение намеренное, а не следствие невнимательности.
        for i in self.b.COVERED_CLASSES:
            self.assertNotIn(i, self.b.SKIN_CLASSES)
        self.assertEqual(sorted(self.b.SKIN_CLASSES + self.b.COVERED_CLASSES),
                         [1, 2, 3, 4, 5])          # всё, кроме фона, разобрано

    def test_the_class_names_match_the_MODEL_FILE_not_an_article(self):
        # ПРОТИВ ГЛАВНОЙ ОШИБКИ ЭТОГО ПРОЕКТА. Имена классов взяты из labels.txt
        # ВНУТРИ .tflite, а не из документации: один раз имя весов уже оказалось
        # выдуманным, и код писался против несуществующего API.
        import zipfile

        path = self.b.model_path()
        if not path.exists():
            self.skipTest(f"нет весов {path} — см. bodyparts.why_unavailable()")
        with zipfile.ZipFile(path) as z:
            labels = z.read("labels.txt").decode().split()
        self.assertEqual(tuple(labels), self.b.LABELS)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class OnARealFrame(unittest.TestCase):
    """Единственное место, где модель действительно исполняется."""

    def setUp(self):
        from ball_reel import bodyparts

        self.b = bodyparts
        if not self.b.available():
            self.skipTest("нет весов сегментации")
        if not FRAME.exists():
            self.skipTest(f"нет кадра {FRAME}")

    def test_a_dressed_leg_and_a_bare_arm_are_told_apart(self):
        # Живой замер, ради которого всё это: на этом кадре человек в леггинсах
        # и топе. Предплечье — 91% кожи, голень — 0%. Геометрическая капсула
        # различить их не могла в принципе, она не знает, что такое ткань.
        from ball_reel import dwpose, marks
        from PIL import Image

        if not dwpose.available():
            self.skipTest("нет весов DWPose")
        with Image.open(FRAME) as im:
            size = im.size
        pts = marks.attach_size(dwpose.pose_points(str(FRAME)), size)
        mask = self.b.skin_mask(str(FRAME))
        arm = self.b.paintable(None, marks.locate(
            pts, marks.Mark(bone="l_forearm", along=0.5, radius=0.15)),
            mask=mask)
        leg = self.b.paintable(None, marks.locate(
            pts, marks.Mark(bone="l_shin", along=0.5, radius=0.15)), mask=mask)
        self.assertTrue(arm["ok"], arm)
        self.assertFalse(leg["ok"], leg)
        self.assertGreater(arm["skin_share"], leg["skin_share"] + 0.5)


if __name__ == "__main__":
    unittest.main()
