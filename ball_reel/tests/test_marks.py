"""Приметы — измеритель заявленного дифференциатора, и судится как таковой.

Проверяется не «функция не упала», а четыре решения, на которых стоит вердикт:
координаты костные (значит поза не ломает измерение), опора локальная (значит
свет не ломает), «не наблюдалось» отделено от «потеряно», и перевёрнутый
контраст не считается половиной успеха.

Синтетика построена руками, потому что вопрос сейчас другой, чем при
калибровке: не «какие числа у настоящей татуировки», а «меряет ли метрика то,
что заявлено». Для этого нужны образцы, где искомое свойство задано нами.
Калибровки на настоящих приметах здесь НЕТ и она этими тестами не подменяется.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def _skin(size=200, tone=0.72):
    import numpy as np

    a = np.full((size, size, 3), tone, dtype=np.float64)
    a[..., 2] *= 0.82          # кожа теплее по красному, холоднее по синему
    return a


def _with_patch(size=200, box=(80, 80, 120, 120), value=0.25, sat=None):
    """Кожа с тёмным пятном — модель татуировки."""
    a = _skin(size)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = value
    if sat is not None:
        a[y0:y1, x0:x1, 2] = value * (1 - sat)
    return a


def _points(size=200, elbow=(0.5, 0.2), wrist=(0.5, 0.8), vis=0.9):
    """Скелет с одной костью «локоть-запястье» по вертикали."""
    return {
        "l_elbow": (elbow[0], elbow[1], vis),
        "l_wrist": (wrist[0], wrist[1], vis),
        "__size__": (float(size), float(size), 1.0),
    }


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheMarkTravelsWithTheLimb(unittest.TestCase):
    """Координаты костные, поэтому смена позы не ломает измерение."""

    def setUp(self):
        from ball_reel import marks

        self.m = marks
        self.mark = marks.Mark(bone="l_forearm", along=0.5, radius=0.2)

    def test_the_box_lands_in_the_middle_of_the_bone(self):
        box = self.m.locate(_points(), self.mark)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        self.assertAlmostEqual(cx, 100, delta=2)   # кость по x=0.5
        self.assertAlmostEqual(cy, 100, delta=2)   # середина между 0.2 и 0.8

    def test_the_same_mark_follows_a_rotated_limb(self):
        # Кость повернули на 90 градусов — примета обязана уехать вместе с ней,
        # а не остаться там, где была в кадре. Ради этого координаты и костные.
        turned = _points(elbow=(0.2, 0.5), wrist=(0.8, 0.5))
        box = self.m.locate(turned, self.mark)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        self.assertAlmostEqual(cx, 100, delta=2)
        self.assertAlmostEqual(cy, 100, delta=2)

    def test_an_offset_mark_sits_to_the_side_of_the_bone(self):
        side = self.m.Mark(bone="l_forearm", along=0.5, across=0.2, radius=0.1)
        box = self.m.locate(_points(), side)
        cx = (box[0] + box[2]) / 2
        self.assertNotAlmostEqual(cx, 100, delta=5)

    def test_an_invisible_bone_is_not_observed_rather_than_missing(self):
        # Рука отвернулась от камеры. Это отсутствие наблюдения, а не потеря
        # приметы, и штрафовать за это генератор было бы враньём.
        self.assertIsNone(self.m.locate(_points(vis=0.2), self.mark))

    def test_an_unknown_bone_is_refused(self):
        self.assertIsNone(
            self.m.locate(_points(), self.m.Mark(bone="tail", along=0.5)))

    def test_a_limb_too_small_in_frame_is_refused(self):
        # Далёкая рука: окно приметы выходит мельче порога, и мерить нечего.
        tiny = _points(elbow=(0.50, 0.50), wrist=(0.52, 0.56))
        self.assertIsNone(self.m.locate(tiny, self.mark))


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class DistinctivenessIsMeasuredAgainstNeighbouringSkin(unittest.TestCase):
    """Опора локальная, иначе метрика мерила бы освещение."""

    def setUp(self):
        from ball_reel import marks

        self.m = marks
        self.box = (80, 80, 120, 120)

    def test_a_dark_patch_on_skin_scores_high(self):
        got = self.m.distinctiveness(_with_patch(), self.box)
        self.assertGreater(got["score"], self.m.MIN_REFERENCE_CONTRAST)
        self.assertLess(got["luma_contrast"], 0)   # тату темнее кожи

    def test_plain_skin_scores_near_zero(self):
        got = self.m.distinctiveness(_skin(), self.box)
        self.assertLess(got["score"], self.m.MIN_REFERENCE_CONTRAST)

    def test_a_lighter_mark_keeps_the_opposite_sign(self):
        # Шрам светлее кожи. Знак — часть подписи приметы, и терять его значит
        # склеить два разных объекта в один.
        got = self.m.distinctiveness(_with_patch(value=0.95), self.box)
        self.assertGreater(got["luma_contrast"], 0)

    def test_changing_the_light_does_not_change_the_verdict(self):
        # Тот же кадр, вся сцена темнее вдвое. Абсолютные значения уедут,
        # относительный контраст — нет. Ровно это спасло метрику одежды, где
        # сравнение по среднему RGB давало ложную тревогу на одной и той же
        # одежде.
        # Именно ТА ЖЕ картинка, умноженная на 0.5. В первой версии этого
        # теста я домножил ещё и пятно отдельно и получил ДРУГУЮ примету, а не
        # ту же при другом свете — тест проверял бы не то, что заявлено.
        bright = self.m.distinctiveness(_with_patch(), self.box)
        dim = self.m.distinctiveness(_with_patch() * 0.5, self.box)
        self.assertAlmostEqual(bright["luma_contrast"], dim["luma_contrast"],
                               delta=0.15)

    def test_a_region_below_the_floor_is_refused(self):
        self.assertIsNone(self.m.distinctiveness(_with_patch(), (80, 80, 90, 90)))

    def test_the_floor_is_read_from_the_module_with_literal_sizes(self):
        floor = self.m.MIN_MARK_PX
        self.assertIsNone(
            self.m.distinctiveness(_with_patch(), (80, 80, 80 + floor - 1,
                                                   80 + floor - 1)))
        self.assertIsNotNone(
            self.m.distinctiveness(_with_patch(), (80, 80, 80 + floor,
                                                   80 + floor)))
        self.assertGreaterEqual(floor, 16)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class LostIsDistinguishedFromNotObserved(unittest.TestCase):
    """Три исхода, и путать их дорого."""

    def setUp(self):
        from ball_reel import marks

        self.m = marks
        self.box = (80, 80, 120, 120)
        self.ref = self.m.distinctiveness(_with_patch(), self.box)

    def test_a_mark_that_survived_passes(self):
        # Генератор перерисовывает, а не копирует: контраст просел, но след
        # явно виден.
        produced = self.m.distinctiveness(_with_patch(value=0.40), self.box)
        got = self.m.mark_transferred(self.ref, produced)
        self.assertTrue(got["verdict"], got["note"])
        self.assertEqual(got["state"], "measured")

    def test_plain_skin_where_a_mark_was_is_named_LOST(self):
        produced = self.m.distinctiveness(_skin(), self.box)
        got = self.m.mark_transferred(self.ref, produced)
        self.assertFalse(got["verdict"])
        self.assertIn("ПОТЕРЯНА", got["note"])

    def test_an_unobserved_limb_is_not_counted_as_loss(self):
        # Самое важное различение модуля: кость не видна — это отсутствие
        # измерения. Записать его потерей значит наказать генератор за то,
        # что рука повернулась.
        got = self.m.mark_transferred(self.ref, None)
        self.assertIsNone(got["verdict"])
        self.assertEqual(got["state"], "not_observed")
        self.assertIn("НЕ НАБЛЮДАЛАСЬ", got["note"])

    def test_no_mark_on_the_reference_is_a_complaint_about_the_INPUT(self):
        # Переносить нечего. Сказать это надо сразу, иначе потом отчитаемся
        # «потеряна» о том, чего не было.
        flat = self.m.distinctiveness(_skin(), self.box)
        got = self.m.mark_transferred(flat, flat)
        self.assertIsNone(got["verdict"])
        self.assertEqual(got["state"], "no_mark_on_reference")
        self.assertIn("ВХОД", got["note"])

    def test_an_inverted_contrast_is_not_half_a_success(self):
        # На месте тёмной татуировки светлое пятно того же масштаба. По одному
        # только модулю контраста это выглядело бы как «сохранилось 100%».
        produced = self.m.distinctiveness(_with_patch(value=0.98), self.box)
        got = self.m.mark_transferred(self.ref, produced)
        self.assertFalse(got["verdict"])
        self.assertIn("перевернулся", got["note"])

    def test_the_ratio_is_read_from_the_module_with_literal_inputs(self):
        ratio = self.m.PRESENT_RATIO
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)
        strong = self.m.distinctiveness(_with_patch(value=0.30), self.box)
        faint = self.m.distinctiveness(_with_patch(value=0.70), self.box)
        self.assertTrue(self.m.mark_transferred(self.ref, strong)["verdict"])
        self.assertFalse(self.m.mark_transferred(self.ref, faint)["verdict"])


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheReportJudgesOnlyWhatWasMeasured(unittest.TestCase):
    def setUp(self):
        from ball_reel import marks

        self.m = marks

    def test_an_unobserved_mark_does_not_make_the_report_pass_or_fail(self):
        ref_pts = _points()
        out_pts = _points(vis=0.2)          # в результате рука не видна
        mark = self.m.Mark(bone="l_forearm", along=0.5, radius=0.2, name="tattoo")
        got = self.m.marks_report(_with_patch(), ref_pts, _skin(), out_pts,
                                  [mark])
        self.assertEqual(got["measured"], 0)
        self.assertEqual(got["unobserved"], ["tattoo"])
        self.assertFalse(got["all_transferred"])   # не успех
        self.assertEqual(got["lost"], [])          # но и не потеря
        self.assertIn("судить не о чем", got["note"])

    def test_a_lost_mark_is_named_in_the_report(self):
        pts = _points()
        mark = self.m.Mark(bone="l_forearm", along=0.5, radius=0.2, name="tattoo")
        got = self.m.marks_report(_with_patch(), pts, _skin(), pts, [mark])
        self.assertEqual(got["lost"], ["tattoo"])
        self.assertFalse(got["all_transferred"])

    def test_a_transferred_mark_passes(self):
        pts = _points()
        mark = self.m.Mark(bone="l_forearm", along=0.5, radius=0.2, name="tattoo")
        got = self.m.marks_report(_with_patch(), pts,
                                  _with_patch(value=0.35), pts, [mark])
        self.assertTrue(got["all_transferred"], got["note"])
        self.assertEqual(got["lost"], [])


if __name__ == "__main__":
    unittest.main()
