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


def _partly_inked(fill, size=200, box=(60, 60, 140, 140), value=0.05):
    """Кожа с приметой, закрывающей ДОЛЮ окна. Ключевое слово — долю.

    Из-за отсутствия ровно этой функции модуль полгода жил со ступенькой:
    все синтетические образцы заливали окно на 100%, то есть стояли в
    единственной точке, где дефект не проявляется.
    """
    a = _skin(size)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x0 + int((x1 - x0) * fill)] = value
    return a


def _contour_mark(size=200, box=(60, 60, 140, 140), pitch=6, width=2):
    """Контурная татуировка: тонкие линии, чернил мало, глазу видно отлично."""
    a = _skin(size)
    x0, y0, x1, y1 = box
    for x in range(x0, x1, pitch):
        a[y0:y1, x:x + width] = 0.05
    return a


def _limb_over_background(size=300, bone_px=200.0, inked=False):
    """Рука поперёк ТЁМНОГО ФОНА — то, чего синтетика раньше не давала.

    Прежние образцы состояли из кожи целиком, поэтому вопрос «а что попадает
    в опорное кольцо» в них не мог возникнуть в принципе. На живом кадре он
    возник сразу: опорой стал тёмный топ, и ровная кожа получила приметность
    1.411.
    """
    import numpy as np
    from ball_reel import marks

    scene = np.zeros((size, size, 3), dtype=np.float64)
    scene[..., 2] = 0.18                       # тёмно-синий фон
    ax, ay = size / 2.0, size * 0.17
    half = bone_px * marks.half_width_for("l_forearm")
    gy, gx = np.mgrid[0:size, 0:size]
    on_arm = ((np.abs(gx - ax) <= half) & (gy >= ay) & (gy <= ay + bone_px))
    scene[on_arm] = [0.72, 0.72, 0.72 * 0.82]
    points = {
        "l_elbow": (ax / size, ay / size, 0.9),
        "l_wrist": (ax / size, (ay + bone_px) / size, 0.9),
        "__size__": (float(size), float(size), 1.0),
    }
    if inked:
        x0, y0, x1, y1 = marks.locate(points, marks.Mark(
            bone="l_forearm", along=0.5, radius=0.15))
        for x in range(x0, x1, 6):
            scene[y0:y1, x:x + 2] = 0.05
    return scene, points


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

    def test_the_score_grows_with_how_much_of_the_window_is_inked(self):
        # ДЕФЕКТ, КОТОРЫЙ ЭТОТ ТЕСТ ЛОВИТ. Первая версия брала медиану окна,
        # а медиана двухмодальной выборки — это её большинство. Замерено на
        # старом коде: заливка 10 / 30 / 49% давала score ровно 0.0000,
        # заливка 51% — 0.5546, а 80 и 100% — одинаковые 1.1091. Метрика была
        # ступенькой на половине и не отличала ни тонкую примету от кожи, ни
        # половину рисунка от целого.
        box = (60, 60, 140, 140)
        scores = [self.m.distinctiveness(_partly_inked(f), box)["score"]
                  for f in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)]
        self.assertEqual(scores[0], 0.0)               # чистая кожа
        for lo, hi in zip(scores, scores[1:]):
            self.assertGreater(hi, lo + 0.05, f"нет хода: {scores}")
        # И отдельно — что дефект именно этой формы: ниже половины больше не ноль.
        self.assertGreater(scores[1], self.m.MIN_REFERENCE_CONTRAST)

    def test_a_thin_contour_tattoo_is_a_mark_not_bare_skin(self):
        # Самый дорогой частный случай ступеньки: контурная тату занимает
        # чернилами меньше половины окна, и старая метрика объявляла её
        # отсутствием приметы НА ВХОДЕ — то есть отказывалась переносить ровно
        # то, ради чего модуль написан.
        got = self.m.distinctiveness(_contour_mark(), (60, 60, 140, 140))
        self.assertGreater(got["score"], self.m.MIN_REFERENCE_CONTRAST)
        self.assertLess(got["luma_contrast"], 0)
        verdict = self.m.mark_transferred(got, got)
        self.assertEqual(verdict["state"], "measured")

    def test_coverage_tells_the_operator_the_window_is_oversized(self):
        # Диагностика, а не вердикт: одно и то же число score получается у
        # сплошного слабого пятна и у тонкого контура. Различает их coverage,
        # и без него оператор не поймёт, почему примету «не видно».
        wide = self.m.distinctiveness(_partly_inked(0.9), (60, 60, 140, 140))
        sparse = self.m.distinctiveness(_partly_inked(0.1), (60, 60, 140, 140))
        self.assertGreater(wide["coverage"], 0.8)
        self.assertLess(sparse["coverage"], 0.2)

    def test_soft_shading_is_not_counted_as_ink(self):
        # Мутационный аудит показал, что NOISE_FLOOR не сторожил никто: тест
        # выше гоняет чистую кожу без светотени, а на ней порог шума не влияет
        # ни на что — с ним и без него coverage одинаковый.
        #
        # Здесь кожа с мягким градиентом ±3% яркости, как на круглой руке, и
        # настоящие чернила на 12.5% окна. С порогом coverage = 0.125, то есть
        # ровно доля чернил; со снятым порогом = 1.0, то есть «чернила везде».
        # Диагностика «окно шире приметы» при этом перестаёт работать вовсе.
        import numpy as np

        a = _skin()
        a = np.clip(a + np.linspace(-0.03, 0.03, 200)[None, :, None], 0, 1)
        a[60:140, 60:70] = 0.05
        got = self.m.distinctiveness(a, (60, 60, 140, 140))
        self.assertLess(got["coverage"], 0.3)
        self.assertGreater(got["coverage"], 0.05)

    def test_peak_separates_a_faded_mark_from_a_shrunken_one(self):
        # Две разные поломки с одинаковой средней: примета уменьшилась вдвое,
        # либо выцвела вдвое. Чинятся они разным, и слепить их в одно число
        # значит отправить чинить не тот конец.
        box = (60, 60, 140, 140)
        shrunk = self.m.distinctiveness(_partly_inked(0.5, value=0.05), box)
        faded = self.m.distinctiveness(_partly_inked(1.0, value=0.40), box)
        self.assertAlmostEqual(shrunk["score"], faded["score"], delta=0.12)
        self.assertGreater(shrunk["peak"], faded["peak"] * 1.5)

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

    def test_a_weak_mark_replaced_by_its_opposite_is_still_a_failure(self):
        # ДЫРА, КОТОРУЮ ЭТОТ ТЕСТ ЗАКРЫВАЕТ. Проверка знака была обусловлена
        # тем, что контраст РЕЗУЛЬТАТА дотягивает до порога входа. Поэтому в
        # окне «референс слабоват» (0.08..0.16) перевёрнутый результат проходил
        # как «примета на месте»: величина сохранилась, а направление не
        # смотрели.
        #
        # Числа тут литеральные и намеренно лежат в этом окне: тёмная примета
        # 0.1386 на референсе, светлое пятно 0.0783 в результате — сохранилось
        # 56%, порог величины взят, а примета противоположная.
        box = (60, 60, 140, 140)
        ref = self.m.distinctiveness(_partly_inked(0.125, value=0.05), box)
        produced = self.m.distinctiveness(_partly_inked(0.1375, value=0.98), box)
        self.assertLess(produced["score"], self.m.MIN_REFERENCE_CONTRAST)
        self.assertGreater(produced["score"] / ref["score"],
                           self.m.PRESENT_RATIO)
        got = self.m.mark_transferred(ref, produced)
        self.assertFalse(got["verdict"], got["note"])
        self.assertIn("перевернулся", got["note"])

    def test_a_colour_mark_without_a_luma_direction_says_so_instead_of_guessing(self):
        # Обратная сторона той же проверки: цветная татуировка той же светлоты,
        # что кожа, отличается насыщенностью. Знак её яркости — шум, и судить
        # по нему нельзя. Молчаливое «прошло» и молчаливое «перевернулось» тут
        # одинаково нечестны, поэтому отказ проверять знак произносится вслух.
        import numpy as np

        ref_img, out_img = _skin(), _skin()
        # Пятно той же яркости, но насыщенное: меняем только распределение
        # каналов, сохраняя взвешенную сумму.
        ref_img[80:120, 80:120] = [0.95, 0.66, 0.30]
        out_img[80:120, 80:120] = [0.30, 0.76, 0.95]
        box = (80, 80, 120, 120)
        ref = self.m.distinctiveness(ref_img, box)
        produced = self.m.distinctiveness(out_img, box)
        self.assertLess(abs(ref["luma_contrast"]), self.m.SIGN_MIN)
        got = self.m.mark_transferred(ref, produced)
        self.assertTrue(got["verdict"], got["note"])
        self.assertIn("Направление НЕ проверялось", got["note"])

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


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheBaselineStaysOnTheLIMB(unittest.TestCase):
    """Опорой обязана быть кожа этой конечности, а не то, что рядом в кадре.

    Дефект был арифметический и от синтетики скрытый. Половина опорного кольца
    равна `radius * SURROUND_SCALE` длины кости, то есть при значениях по
    умолчанию 0.15 и 2.0 составляет 0.30 длины — против полуширины предплечья
    0.15. Кольцо вдвое шире руки, значит на любом настоящем кадре половина
    «кожи» — это фон.

    Все прежние образцы состояли из кожи целиком, поэтому вопрос не мог даже
    возникнуть. Здесь рука лежит поперёк тёмного фона.
    """

    def setUp(self):
        from ball_reel import marks

        self.m = marks
        self.mark = marks.Mark(bone="l_forearm", along=0.5, radius=0.15,
                               name="tattoo")

    def test_bare_skin_over_a_dark_background_is_not_a_mark(self):
        scene, pts = _limb_over_background()
        box = self.m.locate(pts, self.mark)
        got = self.m.distinctiveness(scene, box, points=pts, bone="l_forearm")
        self.assertLess(got["score"], self.m.MIN_REFERENCE_CONTRAST)
        self.assertEqual(got["baseline"], "limb")
        self.assertEqual(self.m.mark_transferred(got, got)["state"],
                         "no_mark_on_reference")

    def test_a_real_tattoo_on_that_same_arm_is_still_found(self):
        # Обратная сторона: починка, которая просто глушит число, — регресс.
        scene, pts = _limb_over_background(inked=True)
        box = self.m.locate(pts, self.mark)
        got = self.m.distinctiveness(scene, box, points=pts, bone="l_forearm")
        self.assertGreater(got["score"], self.m.MIN_REFERENCE_CONTRAST)
        self.assertLess(got["luma_contrast"], 0)

    def test_without_a_skeleton_the_contaminated_baseline_is_flagged(self):
        # Скелета может не быть на входе. Тогда честно не «число», а «число и
        # предупреждение»: молча выдать приметность ровной коже нельзя.
        scene, pts = _limb_over_background()
        box = self.m.locate(pts, self.mark)
        blind = self.m.distinctiveness(scene, box)
        self.assertTrue(blind["baseline_suspect"])
        self.assertEqual(blind["baseline"], "ring")
        self.assertGreater(blind["baseline_spread"], self.m.MAX_BASELINE_SPREAD)

    def test_the_report_passes_the_skeleton_through(self):
        # Сводка знает оба скелета, значит отговорок «геометрии не было» у неё
        # нет. Раньше она их не прокидывала, и весь путь целиком судил по фону.
        scene, pts = _limb_over_background()
        got = self.m.marks_report(scene, pts, scene, pts, [self.mark])
        self.assertEqual(got["measured"], 0)
        self.assertEqual(got["marks"][0]["state"], "no_mark_on_reference")

    def test_a_window_wider_than_the_limb_measures_only_the_limb(self):
        # Найдено этим же тестовым классом уже ПОСЛЕ первой правки, и это
        # показательно: чистили опору, а грязным остался второй конец сравнения.
        # Окно с radius 0.45 шире руки втрое, опора при этом бралась правильная
        # — и ровная кожа получала score 1.184, потому что «приметой» работал
        # фон, попавший в окно. Чистить надо оба конца.
        scene, pts = _limb_over_background()
        fat = self.m.Mark(bone="l_forearm", along=0.5, radius=0.45)
        box = self.m.locate(pts, fat)
        blind = self.m.distinctiveness(scene, box)
        got = self.m.distinctiveness(scene, box, points=pts, bone="l_forearm")
        self.assertGreater(blind["score"], 1.0)          # что было
        self.assertLess(got["score"], self.m.MIN_REFERENCE_CONTRAST)
        self.assertLess(got["pixels"], (box[2] - box[0]) * (box[3] - box[1]))

    def test_no_supporting_skin_is_refused_rather_than_guessed(self):
        # Опоры может не остаться вовсе — тогда это ОТКАЗ ИЗМЕРЯТЬ. Прежняя
        # версия в этом случае тихо брала кольцо ЦЕЛИКОМ, вместе с самой
        # приметой: опора тем хуже, чем сильнее её не хватает, и молча.
        import numpy as np

        tiny = _skin(size=40)
        whole = (0, 0, 40, 40)               # окно занимает весь кадр
        self.assertIsNone(self.m._ring_pixels(tiny, whole)[0])
        self.assertIsNone(self.m._skin_median(tiny, whole))
        self.assertIsNone(self.m.distinctiveness(tiny, whole))


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class LimbWidthIsPerBone(unittest.TestCase):
    """Одного числа на все кости не бывает, и обоснование обязано сходиться.

    Здесь стояло единственное значение 0.22 с пояснением «конечность вчетверо
    длиннее своей ширины». Из этого пояснения следует 0.125 — оно противоречило
    собственному числу почти вдвое, и капсула получалась шире руки.
    """

    def setUp(self):
        from ball_reel import marks

        self.m = marks

    def test_a_thigh_is_relatively_thicker_than_a_shin(self):
        # Не «число в диапазоне», а проверяемое утверждение об анатомии.
        self.assertGreater(self.m.half_width_for("l_thigh"),
                           self.m.half_width_for("l_shin"))
        self.assertGreater(self.m.half_width_for("r_upperarm"),
                           self.m.half_width_for("r_forearm"))

    def test_an_unlisted_bone_falls_back_to_the_default(self):
        self.assertEqual(self.m.half_width_for("tail"), self.m.LIMB_HALF_WIDTH)

    def test_an_explicit_override_beats_the_table(self):
        self.assertEqual(self.m.half_width_for("l_thigh", 0.33), 0.33)

    def test_the_capsule_actually_clips_at_the_declared_width(self):
        # Прежний тест на эту константу утверждал лишь, что она между 0.05 и
        # 0.5, то есть не сторожил ничего: любая правка числа его проходила.
        # Здесь проверяется, что объявленное число — это ГРАНИЦА ПОВЕРХНОСТИ.
        import numpy as np

        pts = _points()                        # кость 120 px по вертикали
        half = self.m.half_width_for("l_forearm") * 120.0
        inside = 100.0 + half * 0.9
        outside = 100.0 + half * 1.1
        _, _, got_in = self.m.limb_uv(pts, "l_forearm", np.array([inside]),
                                      np.array([100.0]))
        _, _, got_out = self.m.limb_uv(pts, "l_forearm", np.array([outside]),
                                       np.array([100.0]))
        self.assertTrue(bool(got_in[0]))
        self.assertFalse(bool(got_out[0]))

    def test_a_thigh_mark_uses_the_thigh_width_not_the_forearm_one(self):
        # Таблица должна доезжать до геометрии, а не украшать модуль.
        import numpy as np

        pts = {"l_hip": (0.5, 0.2, 0.9), "l_knee": (0.5, 0.8, 0.9),
               "__size__": (200.0, 200.0, 1.0)}
        edge = 100.0 + self.m.half_width_for("l_forearm") * 120.0 * 1.2
        _, _, on_thigh = self.m.limb_uv(pts, "l_thigh", np.array([edge]),
                                        np.array([100.0]))
        _, _, on_arm = self.m.limb_uv(_points(), "l_forearm", np.array([edge]),
                                      np.array([100.0]))
        self.assertTrue(bool(on_thigh[0]))     # бедро сюда достаёт
        self.assertFalse(bool(on_arm[0]))      # предплечье — нет


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheRealPoseProducerMustNotFailSILENTLY(unittest.TestCase):
    """Самый дорогой отказ — тот, что выглядит как работа.

    `pose.landmarks` отдаёт точки, нормированные в 0..1, и размера кадра не
    несёт: ему он не нужен, там всё сравнивается в долях. Этот модуль считает в
    пикселях. Стыка между ними не было, и `locate` на настоящем выходе позы
    возвращал None — то есть весь модуль примет отвечал «кость не видна» на
    любой живой вход, и отвечал бы так молча и вечно.

    Синтетика этого поймать не могла: `_points` здесь всегда клала `__size__`
    сама, то есть тесты кормили модуль не тем, чем его кормит пайплайн.
    """

    def setUp(self):
        from ball_reel import marks

        self.m = marks
        self.mark = marks.Mark(bone="l_forearm", along=0.5, radius=0.15)

    def _as_pose_returns_it(self):
        """Ровно та форма, что у `pose.landmarks`: имя -> (x, y, visibility)."""
        return {"l_elbow": (0.5, 0.2, 0.9), "l_wrist": (0.5, 0.8, 0.9)}

    def test_a_skeleton_without_a_frame_size_is_an_ERROR_not_a_shrug(self):
        with self.assertRaises(ValueError) as cm:
            self.m.locate(self._as_pose_returns_it(), self.mark)
        self.assertIn("attach_size", str(cm.exception))

    def test_attach_size_makes_the_real_producer_usable(self):
        import numpy as np

        pts = self.m.attach_size(self._as_pose_returns_it(), (200, 200))
        box = self.m.locate(pts, self.mark)
        self.assertIsNotNone(box)
        self.assertAlmostEqual((box[0] + box[2]) / 2, 100, delta=2)
        # И то же самое от картинки, а не от пары чисел: вызывающему обычно
        # проще подать кадр, который у него и так в руках.
        from_image = self.m.attach_size(self._as_pose_returns_it(), _skin(200))
        self.assertEqual(self.m.locate(from_image, self.mark), box)

    def test_every_pixel_facing_entry_point_refuses_the_same_way(self):
        # Не только locate: любая дверь в модуль, считающая в пикселях, обязана
        # сказать одно и то же. Иначе дыру заткнут в одном месте и оставят в
        # трёх.
        import numpy as np

        bare = self._as_pose_returns_it()
        for call in (
            lambda: self.m.locate(bare, self.mark),
            lambda: self.m.limb_uv(bare, "l_forearm", np.array([1.0]),
                                   np.array([1.0])),
            lambda: self.m._bone_frame(bare, "l_forearm"),
            lambda: self.m._uv_to_pixels(bare, "l_forearm", np.array([0.5]),
                                         np.array([0.0])),
        ):
            with self.assertRaises(ValueError):
                call()
