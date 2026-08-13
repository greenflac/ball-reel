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


def _specular(size=96, spots=6, radius=1):
    """Редкие крошечные очень яркие пятна — зеркальный блик на жидкости.

    Блик БЕЛЫЙ: зеркальное отражение несёт цвет источника, а не материала.
    Фон при этом окрашен, чтобы обесцвечивание блика было измеримым.
    """
    import numpy as np

    a = _canvas(size)
    a[..., 0] *= 1.25          # тёплый цвет самого вещества
    a[..., 2] *= 0.7
    rng = [(size // (spots + 1) * (i + 1)) for i in range(spots)]
    for i, cx in enumerate(rng):
        cy = size // 3 + (i % 3) * size // 4
        a[max(0, cy - radius):cy + radius + 1,
          max(0, cx - radius):cx + radius + 1] = 1.0
    return a


def _matte(size=96):
    """Широкое мягкое пятно той же средней яркости — сглаженный глянец.

    Пятно сохраняет цвет вещества (не белеет) и не даёт мелкой детали: ровно
    то, что генератор рисует вместо блика.
    """
    import numpy as np

    a = _canvas(size)
    a[..., 0] *= 1.25
    a[..., 2] *= 0.7
    y, x = np.mgrid[0:size, 0:size]
    d = ((x - size / 2) ** 2 + (y - size / 2) ** 2) ** 0.5
    glow = np.clip(1.0 - d / (size * 0.55), 0, 1) ** 2
    a += glow[..., None] * 0.45 * a / max(a.max(), 1e-6) * 2
    return np.clip(a, 0, 1)


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
