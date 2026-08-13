"""Прилегание одежды — метрика, у которой есть заведомо известный ответ.

В отличие от «реалистичности», скольжение ткани можно ЗАДАТЬ. Синтетика здесь
не иллюстрация, а якорь: узор наносится в координатах поверхности (u, v), и
сдвиг узора между кадрами известен до измерения с точностью до числа. Если
метрика не отличает ткань, едущую с телом, от ткани, скользящей по нему, она
бесполезна, и это обязан обнаружить тест, а не надежда.

Проверяются пять решений, на которых стоит вердикт, а не то, что функция не
падает:

* поверхность — та же самая, что параметризует `marks.limb_uv` (круговой
  прогон, чтобы формулы не разошлись тихо);
* меряется ПОЛОЖЕНИЕ совпадения, а не похожесть: свет двигает значения и не
  должен двигать вердикт;
* светотень, приклеенная к ТЕЛУ, не выдаётся за прилегание ткани;
* «не смогли измерить» отделено и от «прилегает», и от «скользит» — тремя
  разными исходами;
* пороги читаются из модуля, а входы литеральные: тест, берущий вход из
  сторожимой константы, поедет вместе с ней и не упадёт никогда.

КАЛИБРОВКИ ЗДЕСЬ НЕТ. Опоры для баров измерены на настоящих кадрах отдельно и
записаны в комментариях к константам модуля; эти тесты их не подменяют.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

SIZE = 400

#: Апериодический узор: полтора десятка синусоид со случайными, но
#: ЗАФИКСИРОВАННЫМИ частотами и фазами. Периодический узор (полоска, клетка)
#: совпадает сам с собой через период и даёт неверный сдвиг — это настоящее
#: ограничение метрики, оно записано в докстринге модуля, и якорем такой узор
#: служить не может.
_PATTERN = None


def _pattern_terms():
    global _PATTERN
    if _PATTERN is None:
        rng = np.random.default_rng(7)
        _PATTERN = (rng.uniform(0.4, 1.0, 14), rng.uniform(6, 26, 14),
                    rng.uniform(-4, 4, 14), rng.uniform(0, 6.283, 14))
    return _PATTERN


def _pattern(u, v, slip_u=0.0):
    amp, fu, fv, phase = _pattern_terms()
    uu = u - slip_u
    out = 0.0
    for a, f, g, p in zip(amp, fu, fv, phase):
        out = out + a * np.sin(f * uu + g * v + p)
    return out / np.sqrt((amp ** 2).sum() / 2)


def _points(elbow=(0.30, 0.15), wrist=(0.50, 0.85), vis=0.9, size=SIZE):
    """Скелет с одной костью «локоть-запястье»."""
    return {"l_elbow": (elbow[0], elbow[1], vis),
            "l_wrist": (wrist[0], wrist[1], vis),
            "__size__": (float(size), float(size), 1.0)}


#: Вторая поза: кость и повернулась, и сместилась, и удлинилась. Ткань,
#: прилегающая к телу, обязана пережить это без единицы скольжения.
MOVED = {"elbow": (0.20, 0.25), "wrist": (0.62, 0.78)}


def _frame(points, *, slip_u=0.0, smooth=False, shading=0.25, tone=0.55,
           noise=0.0, seed=0):
    """Кадр с конечностью, узор нанесён в координатах ПОВЕРХНОСТИ.

    Именно поэтому якорь честный: при `slip_u=0` одна и та же материальная
    точка ткани оказывается над одной и той же точкой тела в любой позе, а при
    `slip_u=s` — ровно на `s` длины кости дальше. Ответ известен заранее.
    """
    from ball_reel import marks

    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    u, v, inside = marks.limb_uv(points, "l_forearm", xs, ys)
    img = np.full((SIZE, SIZE, 3), 0.18)
    p = np.zeros_like(u) if smooth else _pattern(u, v, slip_u)
    # Светотень зависит ТОЛЬКО от обхвата: она принадлежит телу, а не ткани,
    # и обязана быть выброшена метрикой.
    lit = (1.0 - shading) + shading * np.cos(v * (np.pi / 2))
    val = np.clip(tone * lit + 0.10 * p, 0.0, 1.0)
    for c in range(3):
        img[..., c] = np.where(inside, val * (1.0 - 0.12 * c), img[..., c])
    if noise:
        img = np.clip(img + np.random.default_rng(seed).normal(
            0, noise, img.shape), 0.0, 1.0)
    return img


def _unwrap(points, **kw):
    from ball_reel import garment_fit as gf

    return gf.unwrap(_frame(points, **kw), points, "l_forearm")


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheSurfaceIsTheOneMarksParameterises(unittest.TestCase):
    """Круговой прогон через настоящий `marks.limb_uv`.

    Обратное преобразование живёт в `garment_fit`, а не берётся из приватных
    имён `marks`: у модуля один писатель, и опираться на его внутренности
    значит подписаться на чужие правки. Цена такого решения — риск, что формулы
    разойдутся молча. Этот тест и есть плата: разойдутся — упадёт здесь.
    """

    def setUp(self):
        from ball_reel import garment_fit, marks

        self.gf, self.marks = garment_fit, marks

    def test_a_surface_point_maps_back_to_itself(self):
        pts = _points()
        u_in = np.array([0.2, 0.5, 0.8])
        v_in = np.array([-0.5, 0.0, 0.4])
        xs, ys = self.gf.surface_to_pixel(pts, "l_forearm", u_in, v_in)
        u_out, v_out, inside = self.marks.limb_uv(pts, "l_forearm", xs, ys)
        self.assertTrue(bool(np.all(inside)))
        self.assertTrue(bool(np.allclose(u_in, u_out, atol=1e-6)), u_out)
        self.assertTrue(bool(np.allclose(v_in, v_out, atol=1e-6)), v_out)

    def test_the_round_trip_holds_on_a_bone_of_a_DIFFERENT_thickness(self):
        # Тест выше гонял только предплечье, и этого было достаточно ровно до
        # тех пор, пока полуширина была одна на все кости. Как только она стала
        # табличной (бедро толще предплечья и в долях собственной длины),
        # жёсткое число в обратном преобразовании начало сходиться ТОЛЬКО на
        # предплечье — а тест продолжал бы зеленеть, потому что другой кости не
        # видел. Проверять круговой прогон на одной кости из десяти значит
        # проверять совпадение констант, а не совпадение формул.
        pts = {"l_hip": (0.5, 0.2, 0.9), "l_knee": (0.5, 0.8, 0.9),
               "__size__": (200.0, 200.0, 1.0)}
        self.assertNotEqual(self.marks.half_width_for("l_thigh"),
                            self.marks.half_width_for("l_forearm"))
        u_in, v_in = np.array([0.3, 0.6]), np.array([-0.4, 0.5])
        xs, ys = self.gf.surface_to_pixel(pts, "l_thigh", u_in, v_in)
        u_out, v_out, inside = self.marks.limb_uv(pts, "l_thigh", xs, ys)
        self.assertTrue(bool(np.all(inside)))
        self.assertTrue(bool(np.allclose(u_in, u_out, atol=1e-6)), u_out)
        self.assertTrue(bool(np.allclose(v_in, v_out, atol=1e-6)), v_out)

    def test_the_same_surface_point_moves_with_the_bone(self):
        # Материальная точка тела: кость повернулась — точка уехала В КАДРЕ,
        # но осталась той же точкой ТЕЛА. На этом стоит вся метрика.
        a = self.gf.surface_to_pixel(_points(), "l_forearm", 0.5, 0.0)
        b = self.gf.surface_to_pixel(_points(**MOVED), "l_forearm", 0.5, 0.0)
        self.assertGreater(abs(a[0] - b[0]) + abs(a[1] - b[1]), 5.0)

    def test_an_invisible_bone_yields_no_surface(self):
        self.assertIsNone(
            self.gf.surface_to_pixel(_points(vis=0.2), "l_forearm", 0.5, 0.0))


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class FabricRidingWithTheBodyIsSeparatedFromFabricSlidingOverIt(
        unittest.TestCase):
    """Главный вопрос модуля. Оба случая построены руками, ответ известен."""

    def setUp(self):
        from ball_reel import garment_fit

        self.gf = garment_fit
        self.before = _unwrap(_points())

    def test_fabric_that_rides_with_the_body_reads_as_no_slip(self):
        # Поза изменилась сильно, узор нанесён на ТУ ЖЕ поверхность.
        got = self.gf.surface_slip(self.before, _unwrap(_points(**MOVED)))
        self.assertEqual(got["state"], "measured", got["note"])
        self.assertTrue(got["held"], got["note"])
        self.assertLess(abs(got["slip_u"]), 0.005, got["note"])

    def test_fabric_that_slides_is_caught_and_the_amount_is_right(self):
        # Узор сдвинут на 0.10 длины кости. Метрика обязана не только сказать
        # «скользит», но и назвать сколько: иначе это детектор, а не измеритель.
        got = self.gf.surface_slip(self.before,
                                   _unwrap(_points(**MOVED), slip_u=0.10))
        self.assertEqual(got["state"], "measured", got["note"])
        self.assertFalse(got["held"], got["note"])
        self.assertAlmostEqual(got["slip_u"], 0.10, delta=0.01)

    def test_the_sign_says_which_way_the_fabric_went(self):
        back = self.gf.surface_slip(self.before,
                                    _unwrap(_points(**MOVED), slip_u=-0.10))
        self.assertAlmostEqual(back["slip_u"], -0.10, delta=0.01)

    def test_a_ladder_of_known_slips_is_read_monotonically(self):
        read = [self.gf.surface_slip(
            self.before, _unwrap(_points(**MOVED), slip_u=s))["slip_u"]
            for s in (0.0, 0.02, 0.05, 0.08, 0.12)]
        self.assertEqual(read, sorted(read), read)
        for want, got in zip((0.0, 0.02, 0.05, 0.08, 0.12), read):
            self.assertAlmostEqual(got, want, delta=0.01)

    def test_the_bar_sits_between_the_two_measured_anchors(self):
        # Вход литеральный, бар читается из модуля. Опоры взяты из замера на
        # настоящих кадрах (см. комментарий к SLIP_MAX): пол настоящей съёмки
        # 0.0175, заведомое скольжение читается как 0.046. Бар обязан лежать
        # между ними, иначе он либо ругает настоящую одежду, либо пропускает
        # скольжение.
        bar = self.gf.SLIP_MAX
        self.assertGreater(bar, 0.0175)
        self.assertLess(bar, 0.046)
        small = self.gf.surface_slip(self.before,
                                     _unwrap(_points(**MOVED), slip_u=0.01))
        big = self.gf.surface_slip(self.before,
                                   _unwrap(_points(**MOVED), slip_u=0.10))
        self.assertTrue(small["held"], small["note"])
        self.assertFalse(big["held"], big["note"])


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class PositionIsMeasured_NotSimilarity(unittest.TestCase):
    """Свет двигает значения. Вердикт он двигать не должен."""

    def setUp(self):
        from ball_reel import garment_fit

        self.gf = garment_fit
        self.before = _unwrap(_points())

    def test_halving_the_light_does_not_move_the_answer(self):
        dim = self.gf.unwrap(_frame(_points(**MOVED)) * 0.5,
                             _points(**MOVED), "l_forearm")
        got = self.gf.surface_slip(self.before, dim)
        self.assertTrue(got["held"], got["note"])
        self.assertLess(abs(got["slip_u"]), 0.005)

    def test_halving_the_light_does_not_hide_a_slide_either(self):
        # Симметричная проверка: устойчивость к свету не должна быть куплена
        # слепотой. Тот же тусклый кадр, но ткань уехала — вердикт обязан
        # смениться.
        dim = self.gf.unwrap(_frame(_points(**MOVED), slip_u=0.10) * 0.5,
                             _points(**MOVED), "l_forearm")
        got = self.gf.surface_slip(self.before, dim)
        self.assertFalse(got["held"], got["note"])
        self.assertAlmostEqual(got["slip_u"], 0.10, delta=0.015)

    def test_sensor_noise_lowers_confidence_but_not_the_position(self):
        noisy = _unwrap(_points(**MOVED), noise=0.03, seed=3)
        got = self.gf.surface_slip(self.before, noisy)
        self.assertEqual(got["state"], "measured", got["note"])
        self.assertLess(abs(got["slip_u"]), 0.01)
        self.assertLess(got["peak"], 0.99)      # похожесть просела


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class ShadingBelongsToTheBodyAndMustNotCountAsFit(unittest.TestCase):
    """Самая дорогая ошибка, которую этот модуль мог бы сделать.

    Светотень на конечности приклеена к ТЕЛУ: она стоит на месте, пока ткань
    едет. Метрика, оставившая её в данных, приклеила бы максимум корреляции к
    нулю и отчиталась бы «прилегает» о чём угодно — то есть всегда давала бы
    ответ «всё хорошо». Ровно так `garment.py` однажды объявил эталон плывущим,
    только с обратным знаком: там опора мерила освещение, здесь освещение
    подменило бы измерение.
    """

    def setUp(self):
        from ball_reel import garment_fit

        self.gf = garment_fit

    def test_a_slide_under_strong_shading_is_still_seen(self):
        # Светотень вчетверо сильнее узора и одинакова в обоих кадрах.
        before = _unwrap(_points(), shading=0.40)
        after = _unwrap(_points(**MOVED), slip_u=0.10, shading=0.40)
        got = self.gf.surface_slip(before, after)
        self.assertEqual(got["state"], "measured", got["note"])
        self.assertFalse(got["held"], got["note"])
        self.assertAlmostEqual(got["slip_u"], 0.10, delta=0.02)

    def test_shading_alone_carries_no_texture_at_all(self):
        # Гладкая ткань со светотенью: после высокочастотного остатка и
        # вычитания строчного среднего мерить нечего, и модуль обязан это
        # сказать, а не выдать «прилегает».
        flat = self.gf.unwrap(_frame(_points(), smooth=True, shading=0.40),
                              _points(), "l_forearm")
        self.assertLess(flat["contrast"], self.gf.MIN_TEXTURE)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class WhatCannotBeMeasuredIsSaidOutLoud(unittest.TestCase):
    """Три отказа, и ни один из них не «прилегает» и не «скользит»."""

    def setUp(self):
        from ball_reel import garment_fit

        self.gf = garment_fit
        self.before = _unwrap(_points())

    def test_smooth_fabric_is_refused_rather_than_praised(self):
        # Однотонное полотно: скользящее выглядит ровно как прилегающее.
        # Назвать это «прилегает» значит соврать в самую выгодную сторону.
        a = self.gf.unwrap(_frame(_points(), smooth=True), _points(),
                           "l_forearm")
        b = self.gf.unwrap(_frame(_points(**MOVED), smooth=True),
                           _points(**MOVED), "l_forearm")
        got = self.gf.surface_slip(a, b)
        self.assertIsNone(got["held"])
        self.assertEqual(got["state"], "no_texture")
        self.assertIn("НЕНАБЛЮДАЕМО", got["note"])

    def test_an_invisible_bone_is_not_observed_rather_than_bad(self):
        self.assertIsNone(_unwrap(_points(vis=0.2)))
        got = self.gf.surface_slip(self.before, None)
        self.assertIsNone(got["held"])
        self.assertEqual(got["state"], "not_observed")

    def test_a_limb_too_small_in_frame_is_refused(self):
        # Дальний план: кость короче порога, и сетка начала бы интерполировать
        # вместо того, чтобы читать пиксели.
        tiny = _points(elbow=(0.50, 0.50), wrist=(0.54, 0.62))
        self.assertIsNone(self.gf.unwrap(_frame(tiny), tiny, "l_forearm"))

    def test_a_limb_hanging_off_the_frame_is_refused(self):
        # Половина поверхности за краем кадра — сравнивать было бы нечего,
        # кроме продлённого края.
        out = _points(elbow=(0.90, 0.15), wrist=(1.25, 0.85))
        self.assertIsNone(self.gf.unwrap(_frame(out), out, "l_forearm"))

    def test_an_unrecognisable_area_is_refused_rather_than_judged(self):
        # Между кадрами нет НИКАКОГО соответствия: вместо узора независимый
        # шум. Сдвиг не определён, и выдавать любой найденный максимум за
        # измерение нельзя.
        #
        # Что этот отказ ловит, а что нет: только ПОЛНУЮ потерю соответствия.
        # Перерисованная тем же материалом область (та же ткань, другие
        # складки) даёт пик выше бара — на такой паре модуль назовёт сдвиг
        # уверенно и, возможно, неверно. Это записано в докстринге модуля.
        a = self.gf.unwrap(_frame(_points(), smooth=True, noise=0.05, seed=1),
                           _points(), "l_forearm")
        b = self.gf.unwrap(_frame(_points(**MOVED), smooth=True, noise=0.05,
                                  seed=2), _points(**MOVED), "l_forearm")
        self.assertGreater(a["contrast"], self.gf.MIN_TEXTURE)
        got = self.gf.surface_slip(a, b)
        self.assertIsNone(got["held"], got["note"])
        self.assertEqual(got["state"], "weak_match")

    def test_comparing_two_different_bones_is_an_error_not_a_number(self):
        a = _unwrap(_points())
        b = _unwrap(_points(**MOVED))
        b["bone"] = "l_thigh"
        with self.assertRaises(ValueError):
            self.gf.surface_slip(a, b)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheBarsAreReadFromTheModuleWithLiteralInputs(unittest.TestCase):
    """Пороги сторожатся литералами, иначе тест поедет вместе с константой."""

    def setUp(self):
        from ball_reel import garment_fit

        self.gf = garment_fit

    def test_the_peak_bar_sits_between_noise_and_real_tracking(self):
        # Опоры замерены отдельно (см. комментарий к MIN_PEAK): случайные
        # текстуры дают пик 0.058..0.078, настоящие соседние кадры — p10 0.605.
        self.assertGreater(self.gf.MIN_PEAK, 0.078)
        self.assertLess(self.gf.MIN_PEAK, 0.605)

    def test_two_independent_textures_do_not_clear_the_peak_bar(self):
        rng = np.random.default_rng(0)
        shape = (3, self.gf.GRID_ACROSS, self.gf.GRID_ALONG)
        peaks = [self.gf._correlate(rng.normal(0, 1, shape),
                                    rng.normal(0, 1, shape), 12, 4)[0]
                 for _ in range(5)]
        self.assertLess(max(peaks), self.gf.MIN_PEAK)

    def test_the_texture_bar_sits_below_the_faintest_real_fabric(self):
        # Самая гладкая выкройка на настоящем отрезке дала 0.0040.
        self.assertLess(self.gf.MIN_TEXTURE, 0.0040)
        self.assertGreater(self.gf.MIN_TEXTURE, 0.0)

    def test_the_limb_floor_keeps_the_grid_from_inventing_detail(self):
        # Сетка не должна быть плотнее пикселей: узлов вдоль кости не больше,
        # чем пикселей на измеряемом куске самой короткой допустимой кости.
        span = self.gf.MIN_LIMB_PX * (1.0 - 2.0 * self.gf.U_MARGIN)
        self.assertGreaterEqual(span, self.gf.GRID_ALONG)

    def test_the_search_range_covers_the_bar_with_room(self):
        # Искать надо заметно дальше, чем бар, иначе скольжение упрётся в
        # границу поиска и прочтётся как «чуть выше бара».
        self.assertGreater(self.gf.SEARCH_ALONG, 2 * self.gf.SLIP_MAX)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheClipVerdictJudgesTheTypicalFrame(unittest.TestCase):
    """Клип целиком: вердикт, отказ судить и осознанный выбор медианы."""

    def setUp(self):
        from ball_reel import garment_fit

        self.gf = garment_fit
        self.poses = [_points(elbow=(0.30 + 0.01 * i, 0.15),
                              wrist=(0.50, 0.85 - 0.004 * i))
                      for i in range(9)]

    def _clip(self, slips):
        return ([_frame(p, slip_u=s) for p, s in zip(self.poses, slips)],
                list(self.poses))

    def test_a_clip_where_the_fabric_holds_passes(self):
        frames, poses = self._clip([0.0] * 9)
        got = self.gf.garment_fit(frames, poses, bones=("l_forearm",))
        self.assertTrue(got["fits"], got["note"])
        self.assertEqual(got["refused"]["not_observed"], 0)
        self.assertIn("держится", got["note"])

    def test_a_clip_where_the_fabric_slides_fails(self):
        # Ткань уезжает на 0.06 длины кости за кадр — сползает по руке.
        frames, poses = self._clip([0.06 * i for i in range(9)])
        got = self.gf.garment_fit(frames, poses, bones=("l_forearm",))
        self.assertFalse(got["fits"], got["note"])
        self.assertIn("СКОЛЬЗИТ", got["note"])
        self.assertGreater(got["bones"]["l_forearm"]["p50"], self.gf.SLIP_MAX)

    def test_the_signed_drift_shows_which_way_the_fabric_crept(self):
        frames, poses = self._clip([0.06 * i for i in range(9)])
        got = self.gf.garment_fit(frames, poses, bones=("l_forearm",))
        self.assertGreater(got["bones"]["l_forearm"]["drift"], 0.3)
        back, poses = self._clip([-0.06 * i for i in range(9)])
        got = self.gf.garment_fit(back, poses, bones=("l_forearm",))
        self.assertLess(got["bones"]["l_forearm"]["drift"], -0.3)

    def test_one_bad_pair_does_not_condemn_the_clip(self):
        # ОСОЗНАННЫЙ РАЗМЕН, а не недосмотр: хвост на настоящей съёмке
        # принадлежит оценщику позы (p90 0.055 при медиане 0.017), и метрика,
        # судящая по хвосту, судила бы DWPose. Скользящая ткань скользит в
        # большинстве кадров, поэтому вердикт по медиане. Цена: одиночный
        # рывок ткани клип не заваливает — и это записано здесь честно, а не
        # спрятано.
        slips = [0.0] * 9
        slips[4] = 0.12
        frames, poses = self._clip(slips)
        got = self.gf.garment_fit(frames, poses, bones=("l_forearm",))
        self.assertTrue(got["fits"], got["note"])
        self.assertGreater(got["bones"]["l_forearm"]["p90"], self.gf.SLIP_MAX)

    def test_too_few_measured_pairs_is_not_a_verdict(self):
        # Кость видна лишь в паре кадров: это отсутствие измерения, а не
        # «плохо» и не «хорошо».
        frames, poses = self._clip([0.0] * 9)
        blind = [_points(vis=0.2) for _ in poses]
        blind[0], blind[1] = poses[0], poses[1]
        got = self.gf.garment_fit(frames, blind, bones=("l_forearm",))
        self.assertIsNone(got["fits"])
        self.assertIn("НЕ ПРОВЕРЕНО", got["note"])
        self.assertGreater(got["refused"]["not_observed"], 0)

    def test_a_clip_of_smooth_fabric_is_refused_not_passed(self):
        frames = [_frame(p, smooth=True) for p in self.poses]
        got = self.gf.garment_fit(frames, list(self.poses),
                                  bones=("l_forearm",))
        self.assertIsNone(got["fits"])
        self.assertEqual(got["refused"]["no_texture"], len(self.poses) - 1)

    def test_mismatched_frames_and_poses_are_refused_loudly(self):
        frames, poses = self._clip([0.0] * 9)
        with self.assertRaises(ValueError):
            self.gf.garment_fit(frames, poses[:-1])


if __name__ == "__main__":
    unittest.main()
