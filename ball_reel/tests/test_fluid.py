"""Метрика жидкости проверяется на синтетике с известными свойствами.

Синтетика здесь не «за неимением лучшего», а потому что она отвечает на нужный
вопрос. Реальные фото сказали бы «метрика откалибрована», а нам сначала надо
знать другое: **меряет ли она то, что заявлено**. Для этого нужны образцы, у
которых искомое свойство задано нами, а не угадывается глазом.

Поэтому якоря построены руками: «зеркальный» — редкие крошечные экстремумы,
как у мокрой изогнутой поверхности; «матовый» — широкое мягкое пятно той же
средней яркости, как у сглаженной генерации. Если метрика их не различает, она
бесполезна независимо от того, что покажет на фотографиях.

Калибровки на реальных фото ЗДЕСЬ НЕТ, и она не подменяется этими тестами.

ПОЧЕМУ ЯКОРЯ ПАРАМЕТРИЗОВАНЫ, А НЕ ЗАФИКСИРОВАНЫ

Один образец проверяет одну точку. У нас это уже стоило дефекта в соседнем
модуле: все тесты строили пятно с заливкой 100%, и ошибка вида «ступенька на
50%» жила незамеченной, потому что 50% никто не подавал. Поэтому здесь у
каждого якоря есть ручки (число пятен, их радиус, яркость блика, сила глянца,
наклон фона), и проверки идут ДИАПАЗОНОМ: монотонность обязана держаться на
всём пробеге, а не в окрестности одной точки.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def _canvas(size=96, base=0.35):
    import numpy as np

    return np.full((size, size, 3), base, dtype=np.float64)


def _specular(size=96, spots=6, radius=1, highlight=1.0, base=0.35):
    """Редкие крошечные очень яркие пятна — зеркальный блик на жидкости.

    Блик БЕЛЫЙ: зеркальное отражение несёт цвет источника, а не материала.
    Фон при этом окрашен, чтобы обесцвечивание блика было измеримым.

    Ручки: `spots` и `radius` двигают долю площади под бликом (замерено:
    0.00065 при radius=0 ... 0.0788 при radius=5), `highlight` гасит блик от
    зеркального 1.0 до уровня вещества 0.44 — это пробег «настоящая жидкость →
    сухая кожа», по которому проверяется монотонность.
    """
    import numpy as np

    a = _canvas(size, base)
    a[..., 0] *= 1.25          # тёплый цвет самого вещества
    a[..., 2] *= 0.7
    rng = [(size // (spots + 1) * (i + 1)) for i in range(spots)]
    for i, cx in enumerate(rng):
        cy = size // 3 + (i % 3) * size // 4
        a[max(0, cy - radius):cy + radius + 1,
          max(0, cx - radius):cx + radius + 1] = highlight
    return np.clip(a, 0, 1)


def _matte(size=96, strength=1.0):
    """Широкое мягкое пятно той же средней яркости — сглаженный глянец.

    Пятно сохраняет цвет вещества (не белеет) и не даёт мелкой детали: ровно
    то, что генератор рисует вместо блика. `strength` пробегает силу глянца от
    нуля (голое вещество) до пересвета.
    """
    import numpy as np

    a = _canvas(size)
    a[..., 0] *= 1.25
    a[..., 2] *= 0.7
    y, x = np.mgrid[0:size, 0:size]
    d = ((x - size / 2) ** 2 + (y - size / 2) ** 2) ** 0.5
    glow = np.clip(1.0 - d / (size * 0.55), 0, 1) ** 2
    a += glow[..., None] * 0.45 * a / max(a.max(), 1e-6) * 2 * strength
    return np.clip(a, 0, 1)


def _bare_skin(size=96, base=0.5, slope=0.02):
    """Кожа БЕЗ жидкости: плавный градиент, ни одного блика, нулевая деталь.

    Это «жидкости на кадре нет вообще» — самый далёкий от эталона вход, какой
    вообще бывает. Градиент, а не заливка: у настоящей кожи всегда есть мягкий
    перепад освещения, и на нём MAD не вырождается — то есть образец идёт по
    обычной ветке порога, а не по аварийной.
    """
    import numpy as np

    y, _x = np.mgrid[0:size, 0:size]
    g = base + slope * (y / size)
    return np.clip(np.dstack([g * 1.02, g, g * 0.99]), 0, 1)


def _milky_gel(size=96, light=(1.0, 1.0, 1.0), tint=1.002, base=0.5):
    """Белёсый гель: вещество почти без цвета, блик нейтральный.

    Такой гель существует (вазелин, прозрачный гель, вода), и он неудобен
    ровно тем, что нужно: обесцвечивание блика у него околонулевое, потому что
    обесцвечивать нечего. `light` — цвет источника: (1,1,1) нейтральный,
    (1,1,0.996) — на 0.4% теплее. Две картинки различаются не более чем на
    0.004 по каналу, то есть это ОДИН И ТОТ ЖЕ гель, снятый дважды.
    """
    import numpy as np

    a = _canvas(size, base)
    a[..., 0] *= tint
    for i in range(6):
        cx = size // 7 * (i + 1)
        cy = size // 3 + (i % 3) * size // 4
        a[cy - 1:cy + 2, cx - 1:cx + 2] = light
    return np.clip(a, 0, 1)


#: Опорный набор координат для проверок нормировки. Числа — с зеркального
#: якоря, но выписаны ЛИТЕРАЛАМИ намеренно: тест, который берёт вход из того
#: же места, что и проверяемый код, едет вместе с ним и не падает никогда.
_GEL_LIKE = {"highlight_share": 0.006, "highlight_ratio": 2.7,
             "highlight_desaturation": 0.44, "detail": 0.21}


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheMetricSeparatesSpecularFromMatte(unittest.TestCase):
    """Если она не различает эти два якоря — она не меряет ничего."""

    def setUp(self):
        from ball_reel import fluid

        self.f = fluid
        self.spec = self.f.fluid_stats(_specular())
        self.matte = self.f.fluid_stats(_matte())

    def test_both_anchors_are_measurable(self):
        self.assertIsNotNone(self.spec)
        self.assertIsNotNone(self.matte)

    def test_specular_highlights_are_far_brighter_than_the_median(self):
        # Настоящий блик отрывается от фона в разы, мягкий глянец — на проценты.
        self.assertGreater(self.spec["highlight_ratio"],
                           self.matte["highlight_ratio"])

    def test_specular_highlights_are_whiter_than_the_material(self):
        # Зеркальное отражение несёт цвет ИСТОЧНИКА: насыщенность в блике
        # падает. У сглаженного пятна цвет вещества остаётся.
        self.assertGreater(self.spec["highlight_desaturation"],
                           self.matte["highlight_desaturation"])
        self.assertGreater(self.spec["highlight_desaturation"], 0.0)

    def test_specular_keeps_fine_detail_and_matte_loses_it(self):
        self.assertGreater(self.spec["detail"], self.matte["detail"])

    def test_the_two_anchors_are_far_apart_in_the_metric(self):
        # Расстояние между ними должно быть заметным, иначе сдвиг от LoRA
        # утонет в шуме измерения.
        self.assertGreater(self.f._distance(self.spec, self.matte), 0.5)

    def test_a_real_highlight_covers_a_fraction_of_a_percent_not_a_half(self):
        # Ловит съехавший порог отрыва: если «бликом» объявить всё, что выше
        # медианы, доля хвоста прыгает к 0.5 и метрика начинает мерить фон.
        # Замерено на якорях: зеркальный 0.0007-0.016, широкий глянец 0.055-0.074.
        self.assertLess(self.spec["highlight_share"], 0.05)
        self.assertLess(self.matte["highlight_share"], 0.2)

    def test_skin_without_fluid_has_no_highlight_at_all(self):
        # Тот же отказ с другой стороны: на голой коже блик РОВНО нулевой.
        # Градиент кожи идёт по обычной ветке порога (MAD не вырожден),
        # поэтому здесь проверяется именно порог отрыва, а не аварийная ветка.
        got = self.f.fluid_stats(_bare_skin())
        self.assertEqual(got["highlight_share"], 0.0)
        self.assertEqual(got["highlight_ratio"], 1.0)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class ATinyRegionIsRefusedNotGuessedAt(unittest.TestCase):
    """Кисть на общем плане — это 30x40 пикселей, и судить по ним нельзя."""

    def setUp(self):
        from ball_reel import fluid

        self.f = fluid

    def test_a_region_below_the_floor_returns_nothing(self):
        # Замерено на нашем driving-отрезке: плечи 181 px при кадре 720,
        # значит кисть около 50 px, а окно жидкости и того меньше. Отсюда
        # режиссёрское следствие — снимать крупным планом.
        self.assertIsNone(self.f.fluid_stats(_specular(size=32)))

    def test_the_floor_is_read_from_the_module_with_literal_sizes(self):
        floor = self.f.MIN_REGION_PX
        self.assertIsNone(self.f.fluid_stats(_specular(size=floor - 1)))
        self.assertIsNotNone(self.f.fluid_stats(_specular(size=floor)))
        self.assertGreaterEqual(floor, 32)

    def test_a_flat_array_is_not_an_image(self):
        self.assertIsNone(self.f.fluid_stats(np.zeros((64, 64))))

    def test_bytes_and_floats_are_read_the_same_way(self):
        a = _specular()
        as_bytes = (a * 255).astype("uint8")
        self.assertAlmostEqual(self.f.fluid_stats(a)["highlight_ratio"],
                               self.f.fluid_stats(as_bytes)["highlight_ratio"],
                               places=1)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheNormalisationMustNotInvertTheVerdict(unittest.TestCase):
    """Главный дефект: нормировка «на себя» переворачивала вердикт.

    Каждая ось делилась на max(|x|, |y|), то есть на сам эталон. Следствия:
    ось с нулевым эталоном давала вклад ровно 1.0 при ЛЮБОМ отличии, а на
    знаковой оси противоположные знаки давали до 4.0. В сумме «жидкости нет
    вообще» оказывалось БЛИЖЕ к эталону, чем «тот же самый гель»: замерено
    2.0 против 1.8027 на белёсом геле, снятом дважды.
    """

    def setUp(self):
        from ball_reel import fluid

        self.f = fluid

    def test_an_empty_frame_is_farther_from_the_gel_than_the_same_gel(self):
        # Ловит переворот вердикта. Эталон — белёсый гель; «двойник» — он же,
        # снятый при источнике на 0.4% теплее (максимум расхождения по каналу
        # 0.004); «пусто» — кожа без жидкости вообще.
        ref = self.f.fluid_stats(_milky_gel())
        twin = self.f.fluid_stats(_milky_gel(light=(1.0, 1.0, 0.996)))
        empty = self.f.fluid_stats(_bare_skin())
        d_twin = self.f._distance(twin, ref)
        d_empty = self.f._distance(empty, ref)
        print(f"\n  эталон       {ref}"
              f"\n  тот же гель  {twin}  -> d={d_twin}"
              f"\n  пустой кадр  {empty}  -> d={d_empty}")
        self.assertLess(d_twin, d_empty)
        # И не «на волосок»: пустой кадр обязан быть далеко, а двойник — рядом.
        self.assertLess(d_twin, 0.2)
        self.assertGreater(d_empty, 1.0)

    def test_the_same_verdict_flip_seen_through_the_public_api(self):
        # То же самое, но так, как это увидит пайплайн: «было пусто, стало
        # гелем» обязано читаться как улучшение.
        ref = self.f.fluid_stats(_milky_gel())
        twin = self.f.fluid_stats(_milky_gel(light=(1.0, 1.0, 0.996)))
        empty = self.f.fluid_stats(_bare_skin())
        got = self.f.realism_shift(empty, twin, reference=ref)
        self.assertTrue(got["verdict"], got["note"])

    def test_an_axis_whose_reference_is_zero_still_measures_size(self):
        # Ловит «эталон равен нулю — любое отличие даёт ровно 1.0». Белёсый
        # гель даёт обесцвечивание 0.0, и на такой оси отличие на 0.002 и
        # отличие на 0.40 обязаны различаться на порядок, а не совпадать.
        ref = dict(_GEL_LIKE, highlight_desaturation=0.0)
        near = dict(ref, highlight_desaturation=0.002)
        far = dict(ref, highlight_desaturation=0.40)
        d_near = self.f._distance(near, ref)
        d_far = self.f._distance(far, ref)
        print(f"\n  эталон обесцвечивания 0.0: отличие 0.002 -> {d_near}, "
              f"отличие 0.40 -> {d_far}")
        self.assertGreater(d_far, 20 * max(d_near, 1e-9))

    def test_opposite_signs_are_not_worth_four_axes(self):
        # Ловит «на знаковой оси смена знака даёт до 4.0». Обесцвечивание
        # +0.001 против -0.001 — это разница в 0.002 единицы насыщенности,
        # то есть шум, и стоить она должна почти ничего.
        a = dict(_GEL_LIKE, highlight_desaturation=0.001)
        b = dict(_GEL_LIKE, highlight_desaturation=-0.001)
        self.assertLess(self.f._distance(a, b), 0.05)

    def test_ten_times_farther_is_ten_times_farther_not_six_percent(self):
        # Ловит насыщение: при делении на эталон вклад оси упирается в 1.0,
        # и «доля блика 0.02» почти неотличима от «доля блика 0.2».
        ref = dict(_GEL_LIKE)
        near = dict(ref, highlight_share=0.02)
        far = dict(ref, highlight_share=0.20)
        d_near = self.f._distance(near, ref)
        d_far = self.f._distance(far, ref)
        print(f"\n  доля блика 0.006 -> 0.02 даёт {d_near}, "
              f"0.006 -> 0.20 даёт {d_far}")
        self.assertGreater(d_far, 4 * d_near)

    def test_every_axis_grows_without_a_ceiling(self):
        # Монотонность на ВСЁМ пробеге каждой оси, а не вблизи эталона.
        # Шаги выписаны литералами и уходят далеко за правдоподобный диапазон
        # намеренно: потолок нормировки виден только там.
        sweeps = {
            "highlight_share": [0.006, 0.01, 0.03, 0.08, 0.2, 0.6],
            "highlight_ratio": [2.7, 3.5, 6.0, 15.0, 60.0, 250.0],
            "highlight_desaturation": [0.44, 0.3, 0.1, 0.0, -0.2, -0.6],
            "detail": [0.21, 0.3, 0.6, 1.5, 5.0, 20.0],
        }
        ref = dict(_GEL_LIKE)
        for axis, steps in sweeps.items():
            with self.subTest(axis=axis):
                seen = [self.f._distance(dict(ref, **{axis: v}), ref)
                        for v in steps]
                for prev, cur in zip(seen, seen[1:]):
                    self.assertGreater(cur, prev, f"{axis}: {seen}")

    def test_a_fading_highlight_moves_away_from_the_gel_step_by_step(self):
        # Монотонность на КАРТИНКАХ, а не на выдуманных координатах: блик
        # гасится от зеркального 1.0 до уровня вещества, и каждый шаг обязан
        # уводить дальше от эталона. Это тот самый пробег, на котором ловится
        # «дефект живёт между двумя проверенными точками».
        ref = self.f.fluid_stats(_specular())
        seen = []
        for h in (1.0, 0.95, 0.85, 0.75, 0.65, 0.55):
            seen.append(self.f._distance(
                self.f.fluid_stats(_specular(highlight=h)), ref))
        for prev, cur in zip(seen, seen[1:]):
            self.assertGreater(cur, prev, f"пробег яркости блика: {seen}")

    def test_fluid_vanishing_from_the_frame_grows_the_distance_at_every_step(
            self):
        # Пробег «жидкость исчезает»: кадр линейно перетекает от зеркального
        # геля к голой коже. Расстояние обязано расти на КАЖДОМ шаге, а не
        # только на краях — именно так ловится метрика, которая упирается в
        # потолок где-то посередине и перестаёт отличать плохое от ужасного.
        ref = self.f.fluid_stats(_specular())
        gel, skin = _specular(), _bare_skin()
        seen = []
        for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0):
            blend = np.clip((1.0 - t) * gel + t * skin, 0, 1)
            seen.append(self.f._distance(self.f.fluid_stats(blend), ref))
        for prev, cur in zip(seen, seen[1:]):
            self.assertGreater(cur, prev, f"пробег исчезновения: {seen}")

    def test_a_glaze_of_any_strength_stays_far_from_a_real_highlight(self):
        # Диапазоном, а не одной точкой: сглаженный глянец любой силы обязан
        # остаться далеко от зеркального якоря. Монотонности здесь НЕ
        # утверждаем — с ростом силы глянец подтягивается к гелю по отрыву
        # хвоста и расходится по мелкой детали, и требовать монотонность
        # значило бы вписать в тест неверную физику.
        ref = self.f.fluid_stats(_specular())
        for s in (0.25, 0.5, 0.75, 1.0, 1.5):
            with self.subTest(strength=s):
                got = self.f._distance(
                    self.f.fluid_stats(_matte(strength=s)), ref)
                self.assertGreater(got, 1.0)

    def test_the_axis_scales_are_the_declared_ones(self):
        # Сторожит сами масштабы: отличие ровно в одну шкалу по одной оси даёт
        # расстояние около единицы. Числа литеральные — если масштаб в модуле
        # молча сдвинут, тест обязан покраснеть, а не поехать вместе с ним.
        ref = dict(_GEL_LIKE)
        one_scale = {"highlight_share": 0.006 + 0.05,
                     "highlight_ratio": 2.7 + 1.0,
                     "highlight_desaturation": 0.44 - 0.25,
                     "detail": 0.21 + 0.25}
        for axis, value in one_scale.items():
            with self.subTest(axis=axis):
                got = self.f._distance(dict(ref, **{axis: value}), ref)
                self.assertAlmostEqual(got, 1.0, places=2)

    def test_the_two_anchors_stay_far_apart_after_rescaling(self):
        # Нормировка чинится не за счёт чувствительности: зеркальный и матовый
        # якоря обязаны остаться разделёнными с запасом.
        spec = self.f.fluid_stats(_specular())
        matte = self.f.fluid_stats(_matte())
        self.assertGreater(self.f._distance(spec, matte), 1.0)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class NothingUnmeasurableTurnsIntoAVerdict(unittest.TestCase):
    """Краевые случаи. Ни один вход не отдаёт NaN и не проходит за «хорошо»."""

    def setUp(self):
        from ball_reel import fluid

        self.f = fluid

    def _all_finite(self, stats):
        import math

        for k, v in stats.items():
            self.assertTrue(math.isfinite(v), f"{k} = {v}")

    def test_a_nan_in_the_region_is_refused_not_propagated(self):
        # Ловит утечку NaN наружу: сейчас проверка «значения в 0..255» на NaN
        # даёт False, масштабирование не срабатывает, и NaN доезжает до detail.
        a = _specular()
        a[5, 5, 0] = np.nan
        self.assertIsNone(self.f.fluid_stats(a))

    def test_an_infinity_in_the_region_is_refused(self):
        a = _specular()
        a[7, 7, 1] = np.inf
        self.assertIsNone(self.f.fluid_stats(a))

    def test_a_uniform_patch_reports_no_fluid_rather_than_full_coverage(self):
        # Однотонная заливка — обычно признак того, что окно уехало мимо кадра.
        # Порог отрыва на ней вырождается, и «бликом» рискует стать всё сразу.
        for base in (0.0, 0.25, 0.5, 1.0):
            with self.subTest(base=base):
                got = self.f.fluid_stats(np.full((96, 96, 3), base))
                self._all_finite(got)
                self.assertEqual(got["highlight_share"], 0.0)
                self.assertEqual(got["highlight_ratio"], 1.0)
                self.assertEqual(got["detail"], 0.0)

    def test_an_almost_black_region_does_not_explode_the_ratio(self):
        # Ловит деление на ноль: медиана чёрной области равна нулю, и отрыв
        # хвоста улетал в 1000000.0 — число, которое потом задавит все оси.
        a = np.zeros((96, 96, 3))
        a[10:12, 10:12] = 1.0
        got = self.f.fluid_stats(a)
        self._all_finite(got)
        print(f"\n  почти чёрная область: highlight_ratio="
              f"{got['highlight_ratio']}")
        self.assertLess(got["highlight_ratio"], 1000.0)

    def test_a_hair_dimmer_gel_is_still_a_gel(self):
        # Ловит ножевой край в аварийной ветке порога: она считала отсечку как
        # med + сигмы * ((max-med)/сигмы), и округление плавающей точки то
        # включало самые яркие пиксели, то выбрасывало их. Замерено: при
        # общем множителе 0.99 блик исчезал целиком, при 0.98 и 0.999 — нет.
        for gain in (1.0, 0.999, 0.99, 0.98, 0.95, 0.9):
            with self.subTest(gain=gain):
                got = self.f.fluid_stats(np.clip(_specular() * gain, 0, 1))
                self.assertGreater(got["highlight_share"], 0.0)
                self.assertGreater(got["highlight_ratio"], 1.5)

    def test_stats_are_finite_across_the_whole_synthetic_range(self):
        # Диапазоном, а не одной точкой: провал обычно сидит на краю ручки.
        regions = ([_specular(spots=s) for s in (1, 3, 6, 12)]
                   + [_specular(radius=r) for r in (0, 2, 5)]
                   + [_specular(highlight=h) for h in (0.5, 0.75, 1.0)]
                   + [_matte(strength=s) for s in (0.0, 0.5, 1.0, 1.5)]
                   + [_bare_skin(slope=s) for s in (0.0, 0.001, 0.05, 0.4)]
                   + [np.full((96, 96, 3), 1.0), np.zeros((96, 96, 3))])
        for i, region in enumerate(regions):
            with self.subTest(i=i):
                got = self.f.fluid_stats(region)
                self.assertIsNotNone(got)
                self._all_finite(got)

    def test_a_region_thinner_than_the_operator_is_refused(self):
        # Лапласиан берёт трёх соседей: на полосе в два пикселя срез пустой и
        # среднее по нему — NaN. Порог области это обычно закрывает, но
        # закрывать обязан сам расчёт, а не константа рядом.
        self.assertIsNone(self.f.fluid_stats(np.zeros((2, 200, 3))))
        self.assertIsNone(self.f.fluid_stats(np.zeros((200, 2, 3))))

    def test_coordinates_without_axes_are_not_a_zero_distance(self):
        # Ловит самый тихий из отказов: _distance по двум наборам без общих
        # осей давал 0.0, то есть «идентичны», хотя не сравнилось ничего.
        self.assertIsNone(self.f._distance({}, {}))
        self.assertIsNone(self.f._distance({"pixels": 10}, {"pixels": 10}))

    def test_a_non_numeric_coordinate_is_not_silently_skipped(self):
        self.assertIsNone(self.f._distance(dict(_GEL_LIKE),
                                           dict(_GEL_LIKE,
                                                detail=float("nan"),
                                                highlight_share=None,
                                                highlight_ratio=None,
                                                highlight_desaturation=None)))

    def test_an_unmeasurable_result_is_not_reported_as_an_improvement(self):
        # Ловит молчаливое «стало лучше»: если после генерации область
        # неизмерима, расстояние выходило 0.0 и вердикт становился ИСТИНОЙ —
        # 1.8034 -> 0.0, «ближе к настоящему». Замерено на текущем коде.
        ref = self.f.fluid_stats(_specular())
        before = self.f.fluid_stats(_bare_skin())
        got = self.f.realism_shift(before, {"pixels": 10}, reference=ref)
        print(f"\n  неизмеримый результат -> {got}")
        self.assertIsNone(got["verdict"])
        self.assertEqual(got["state"], "not_measured")

    def test_nan_coordinates_never_leave_the_module(self):
        import math

        ref = self.f.fluid_stats(_specular())
        broken = dict(ref, detail=float("nan"))
        got = self.f.realism_shift(ref, broken, reference=ref)
        self.assertIsNone(got["verdict"])
        self.assertEqual(got["state"], "not_measured")
        for v in got.values():
            self.assertFalse(isinstance(v, float) and math.isnan(v), got)

    def test_the_three_outcomes_are_told_apart(self):
        # «Не смогли измерить» — отдельный исход, отличимый и от «хорошо», и
        # от «плохо». Без этого неизмеримое сливается с одним из вердиктов.
        spec = self.f.fluid_stats(_specular())
        matte = self.f.fluid_stats(_matte())
        good = self.f.realism_shift(matte, spec, reference=spec)
        bad = self.f.realism_shift(spec, matte, reference=spec)
        none = self.f.realism_shift(None, spec, reference=spec)
        blind = self.f.realism_shift(matte, spec)
        self.assertEqual((good["verdict"], good["state"]), (True, "measured"))
        self.assertEqual((bad["verdict"], bad["state"]), (False, "measured"))
        self.assertEqual((none["verdict"], none["state"]),
                         (None, "not_measured"))
        self.assertEqual((blind["verdict"], blind["state"]),
                         (None, "no_reference"))
        self.assertEqual(len({good["state"], bad["state"], none["state"],
                              blind["state"]}), 3)

    def test_standing_still_is_not_reported_as_moving_away(self):
        # «Не сдвинулось» — не улучшение, вердикт остаётся ложью; но и не
        # «ДАЛЬШЕ», и написать так в отчёте значит соврать о направлении.
        spec = self.f.fluid_stats(_specular())
        got = self.f.realism_shift(spec, spec, reference=spec)
        self.assertFalse(got["verdict"])
        self.assertEqual(got["state"], "measured")
        self.assertNotIn("ДАЛЬШЕ", got["note"])
        self.assertIn("НЕ СДВИНУЛОСЬ", got["note"])

    def test_an_unmeasurable_reference_is_refused_too(self):
        # Эталон тоже бывает не снят: судить «ближе/дальше» не от чего.
        spec = self.f.fluid_stats(_specular())
        got = self.f.realism_shift(spec, spec, reference={"pixels": 10})
        self.assertIsNone(got["verdict"])
        self.assertEqual(got["state"], "not_measured")


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class WithoutAReferenceItRefusesToCallAnythingRealistic(unittest.TestCase):
    """«Стало иначе» и «стало ближе к настоящему» — разные утверждения."""

    def setUp(self):
        from ball_reel import fluid

        self.f = fluid
        self.spec = self.f.fluid_stats(_specular())
        self.matte = self.f.fluid_stats(_matte())

    def test_no_reference_means_no_verdict_only_a_delta(self):
        got = self.f.realism_shift(self.matte, self.spec)
        self.assertIsNone(got["verdict"])
        self.assertGreater(got["changed"], 0)
        self.assertIn("БЕЗ ЭТАЛОНА", got["note"])

    def test_moving_toward_the_reference_is_a_pass(self):
        # Эталон — «зеркальный» якорь. Генерация ушла от матовой к нему.
        got = self.f.realism_shift(self.matte, self.spec, reference=self.spec)
        self.assertTrue(got["verdict"])
        self.assertLess(got["distance_after"], got["distance_before"])

    def test_moving_away_from_the_reference_is_named_a_failure(self):
        # LoRA может и ухудшить — метрика обязана это сказать, а не сгладить.
        got = self.f.realism_shift(self.spec, self.matte, reference=self.spec)
        self.assertFalse(got["verdict"])
        self.assertIn("ДАЛЬШЕ", got["note"])

    def test_an_unmeasurable_region_says_shoot_closer(self):
        got = self.f.realism_shift(None, self.spec)
        self.assertIsNone(got["verdict"])
        self.assertIn("крупнее", got["note"])


if __name__ == "__main__":
    unittest.main()
