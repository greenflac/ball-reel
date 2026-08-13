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


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheMarkIsCarriedAsDeviationNotAsPixels(unittest.TestCase):
    """Переносится отношение к коже, поэтому свет и загар не ломают перенос.

    ОГОВОРКА О КРУГОВОЙ ПОРУКЕ, и её надо держать в голове: перенос и метрика
    стоят на одном принципе (локальная опора), поэтому метрика ОБЯЗАНА
    признавать всё, что перенос вклеил. Это не доказательство качества вклейки,
    и выдавать одно за другое нельзя.

    Что эти тесты действительно проверяют: геометрию (примета едет с костью),
    независимость от освещения, и отказ вмешиваться там, где кости не видно.
    """

    def setUp(self):
        from ball_reel import marks

        self.m = marks
        self.mark = marks.Mark(bone="l_forearm", along=0.5, radius=0.2,
                               name="tattoo")

    def test_a_mark_lands_on_plain_skin_where_the_bone_is(self):
        out, rep = self.m.transfer(_with_patch(), _points(), _skin(),
                                   _points(), self.mark)
        self.assertTrue(rep["applied"], rep["note"])
        box = self.m.locate(_points(), self.mark)
        got = self.m.distinctiveness(out, box)
        self.assertGreater(got["score"], self.m.MIN_REFERENCE_CONTRAST)
        self.assertLess(got["luma_contrast"], 0)   # тату осталась ТЕМНЕЕ кожи

    def test_the_mark_follows_a_rotated_limb(self):
        # Кость в результате повёрнута на 90 градусов. Примета обязана
        # оказаться на ней, а не там, где была в кадре референса.
        turned = _points(elbow=(0.2, 0.5), wrist=(0.8, 0.5))
        out, rep = self.m.transfer(_with_patch(), _points(), _skin(),
                                   turned, self.mark)
        self.assertTrue(rep["applied"])
        on_bone = self.m.distinctiveness(out, self.m.locate(turned, self.mark))
        self.assertGreater(on_bone["score"], self.m.MIN_REFERENCE_CONTRAST)

    def test_darker_target_skin_keeps_the_mark_relative_not_absolute(self):
        # Смуглая кожа в результате. Вклеенные ПИКСЕЛИ выглядели бы светлым
        # пятном; перенесённое ОТНОШЕНИЕ остаётся темнее своей кожи.
        dark = _skin(tone=0.34)
        out, _ = self.m.transfer(_with_patch(), _points(), dark, _points(),
                                 self.mark)
        got = self.m.distinctiveness(out, self.m.locate(_points(), self.mark))
        self.assertLess(got["luma_contrast"], 0)
        self.assertGreater(got["score"], self.m.MIN_REFERENCE_CONTRAST)

    def test_an_invisible_bone_leaves_the_frame_untouched(self):
        # Рисовать примету на невидимой конечности значит выдумывать — ровно
        # то, что правило продукта запрещает.
        import numpy as np

        target = _skin()
        out, rep = self.m.transfer(_with_patch(), _points(), target,
                                   _points(vis=0.2), self.mark)
        self.assertFalse(rep["applied"])
        self.assertIn("НЕ вклеена", rep["note"])
        self.assertTrue(np.allclose(out, target))

    def test_plain_skin_on_the_reference_is_refused(self):
        import numpy as np

        target = _skin()
        out, rep = self.m.transfer(_skin(), _points(), target, _points(),
                                   self.mark)
        self.assertFalse(rep["applied"])
        self.assertIn("переносить нечего", rep["note"])
        self.assertTrue(np.allclose(out, target))

    def test_the_edges_are_feathered_not_cut(self):
        # Резкий край читается как наклейка даже при точном цвете. У края
        # результат обязан быть ближе к коже, чем в середине приметы.
        out, _ = self.m.transfer(_with_patch(), _points(), _skin(), _points(),
                                 self.mark)
        x0, y0, x1, y1 = self.m.locate(_points(), self.mark)
        mid = out[(y0 + y1) // 2, (x0 + x1) // 2].mean()
        edge = out[y0 + 1, (x0 + x1) // 2].mean()
        self.assertGreater(edge, mid)

    def test_the_paste_does_not_leave_the_limb(self):
        # Найдено визуальным аудитом: без ограничения крест лёг поверх лямки
        # топа и фона. Ограничение геометрическое — рука это капсула вокруг
        # кости, и за ней заведомо не тело.
        import numpy as np

        target = _skin()
        out, rep = self.m.transfer(_with_patch(), _points(), target,
                                   _points(), self.mark)
        far = int(200 * 0.5 + 200 * self.m.LIMB_HALF_WIDTH * 1.6)
        self.assertTrue(np.allclose(out[100, far:], target[100, far:]),
                        "вклейка вышла за конечность")

    def test_the_limb_width_is_read_from_the_module(self):
        half = self.m.LIMB_HALF_WIDTH
        self.assertGreater(half, 0.05)
        self.assertLess(half, 0.5)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheSurfaceGivesMaterialCoordinates(unittest.TestCase):
    """(u, v) — это точка НА ТЕЛЕ, одна и та же во всех кадрах.

    Ценность этой параметризации не во вклейке татуировки: замер показал, что
    против плоской вклейки цилиндр выигрывает 0.0001 по среднему. Ценность в
    том, что появляется материальная привязка, которой скелет не даёт, — на ней
    строится вопрос «едет ли ткань вместе с телом или скользит по нему».
    """

    def setUp(self):
        from ball_reel import marks

        self.m = marks

    def test_the_centre_of_the_visible_side_is_v_zero(self):
        import numpy as np

        u, v, inside = self.m.limb_uv(_points(), "l_forearm",
                                      np.array([100.0]), np.array([100.0]))
        self.assertAlmostEqual(float(v[0]), 0.0, places=3)
        self.assertAlmostEqual(float(u[0]), 0.5, places=2)
        self.assertTrue(bool(inside[0]))

    def test_beyond_the_silhouette_there_is_no_surface(self):
        import numpy as np

        far = 100.0 + 200 * self.m.LIMB_HALF_WIDTH * 1.5
        _, _, inside = self.m.limb_uv(_points(), "l_forearm",
                                      np.array([far]), np.array([100.0]))
        self.assertFalse(bool(inside[0]))

    def test_equal_steps_on_the_surface_shrink_toward_the_silhouette(self):
        # Суть цилиндра: равные шаги ПО ПОВЕРХНОСТИ дают неравные шаги В КАДРЕ.
        # Замерено на живом кадре: 4.91 px в центре против 0.77 у силуэта.
        import numpy as np

        pts = _points()
        centre = self.m._uv_to_pixels(pts, "l_forearm", np.array([0.5]),
                                      np.array([0.0]))[0][0]
        near_c = self.m._uv_to_pixels(pts, "l_forearm", np.array([0.5]),
                                      np.array([0.10]))[0][0]
        edge = self.m._uv_to_pixels(pts, "l_forearm", np.array([0.5]),
                                    np.array([0.85]))[0][0]
        near_e = self.m._uv_to_pixels(pts, "l_forearm", np.array([0.5]),
                                      np.array([0.95]))[0][0]
        self.assertGreater(abs(near_c - centre), abs(near_e - edge) * 3)

    def test_the_same_body_point_maps_to_both_frames(self):
        # Материальная привязка: одна и та же (u, v) в двух РАЗНЫХ позах даёт
        # разные пиксели, и оба лежат на своей кости.
        import numpy as np

        a = _points()
        b = _points(elbow=(0.2, 0.5), wrist=(0.8, 0.5))
        ua, va = np.array([0.5]), np.array([0.3])
        xa, ya = self.m._uv_to_pixels(a, "l_forearm", ua, va)
        xb, yb = self.m._uv_to_pixels(b, "l_forearm", ua, va)
        self.assertNotAlmostEqual(float(xa[0]), float(xb[0]), delta=5)
        for pts, x, y in ((a, xa, ya), (b, xb, yb)):
            _, _, inside = self.m.limb_uv(pts, "l_forearm", x, y)
            self.assertTrue(bool(inside[0]))

    def test_an_unknown_bone_has_no_surface(self):
        import numpy as np

        self.assertIsNone(self.m.limb_uv(_points(), "tail",
                                         np.array([1.0]), np.array([1.0])))
