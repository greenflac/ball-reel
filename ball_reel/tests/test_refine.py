"""Доводка лица: проверяется всё, что проверяемо БЕЗ карты и БЕЗ весов.

Карты в среде разработки нет, поэтому граница проведена так: геометрия,
арифметика памяти, отказы предполёта и РАЗБОР АРГУМЕНТОВ проверяются целиком, а
сам вызов пайплайна — только через подставной объект. Отдельно сверяется, что
ключи вызова существуют в НАСТОЯЩЕЙ сигнатуре diffusers: на этом проекте уже
писали код против выдуманного API, и повторять это дважды дорого.

Синтетика намеренно покрывает ДИАПАЗОН размеров лица — 12, 20, 38, 57, 72, 78,
150 и 400 px, — а не одно удобное значение. Причина измеренная: проект уже
получил слепую метрику из-за фикстур, заливавших окно на 100%. Крайние случаи
(лицо у самой рамки кадра, лицо мельче ячейки латента, лицо во весь кадр) —
это и есть места, где геометрия ломается.

Числа во входах записаны литералами. Тест, берущий вход из константы, которую
сторожит, не падает никогда — сдвинь константу, и «ожидаемое» уедет вместе с ней.
"""

from __future__ import annotations

import unittest


def _box(cx, cy, size):
    """Квадратный бокс лица со стороной `size` вокруг (cx, cy)."""
    half = size / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def _jitter(cx, cy, size, n=16, wobble=6.0):
    """Дрожащий покадровый бокс — ровно то, что отдаёт детектор на 16 кадрах."""
    out = []
    for i in range(n):
        dx = wobble * ((i % 4) - 1.5) / 1.5
        dy = wobble * ((i % 3) - 1.0)
        out.append(_box(cx + dx, cy + dy, size))
    return out


class TheBoxIsFixedOnceForTheWholeSequence(unittest.TestCase):
    """Покадровый бокс дёргается, и модуль движения борется с дрожью субъекта.

    Фиксированный бокс — это «камера», внутри которой лицо двигается само. Всё,
    что здесь проверяется, — свойства этой камеры: она накрывает лицо на ВСЕХ
    кадрах, она квадратная и она внутри кадра.
    """

    def setUp(self):
        from ball_reel import refine

        self.r = refine

    def test_the_union_covers_every_frame_box_not_just_the_first(self):
        boxes = _jitter(256, 120, 78)
        got, rep = self.r.union_box(boxes, size=(512, 768))
        self.assertTrue(rep["ok"], rep)
        x0, y0, x1, y1 = got
        for b in boxes:
            self.assertLessEqual(x0, b[0])
            self.assertLessEqual(y0, b[1])
            self.assertGreaterEqual(x1, b[2])
            self.assertGreaterEqual(y1, b[3])

    def test_the_box_is_square_across_the_whole_range_of_face_sizes(self):
        # Несквадратный кроп, растянутый в 512x512, деформирует лицо — эта
        # ошибка на проекте уже растягивала фигуру в 2.7 раза.
        for face in (38, 57, 72, 78, 150, 400):
            with self.subTest(face=face):
                got, rep = self.r.union_box([_box(256, 200, face)],
                                            size=(512, 768))
                self.assertIsNotNone(got, rep)
                self.assertEqual(got[2] - got[0], got[3] - got[1])

    def test_a_rectangular_face_box_still_comes_back_square(self):
        # Детектор отдаёт вытянутый прямоугольник постоянно: лицо выше, чем шире.
        got, _ = self.r.union_box([(200.0, 100.0, 260.0, 190.0)],
                                  size=(512, 768))
        self.assertEqual(got[2] - got[0], got[3] - got[1])
        self.assertGreaterEqual(got[2] - got[0], 90)

    def test_the_box_stays_inside_the_frame_even_at_the_corners(self):
        for cx, cy in ((5, 5), (507, 5), (5, 763), (507, 763), (256, 384)):
            with self.subTest(centre=(cx, cy)):
                got, rep = self.r.union_box([_box(cx, cy, 78)],
                                            size=(512, 768))
                self.assertIsNotNone(got, rep)
                x0, y0, x1, y1 = got
                self.assertGreaterEqual(x0, 0)
                self.assertGreaterEqual(y0, 0)
                self.assertLessEqual(x1, 512)
                self.assertLessEqual(y1, 768)
                self.assertEqual(x1 - x0, y1 - y0)

    def test_a_face_bigger_than_the_frame_shrinks_the_side_not_the_shape(self):
        # Проверяются СВОЙСТВА, которые названы в имени теста, а не конкретный
        # кортеж. Первая редакция ждала (0, 0, 512, 512) — то есть требовала
        # прижать бокс к верху кадра и увести лицо (центр y=384) из центра
        # кропа. Тест, зафиксировавший случайный кортеж вместо правила,
        # краснеет на верном коде и зеленеет на неверном.
        got, rep = self.r.union_box([_box(256, 384, 900)], size=(512, 768))
        x0, y0, x1, y1 = got
        self.assertEqual(x1 - x0, y1 - y0, "форма не ломается — квадрат")
        self.assertEqual(x1 - x0, 512, "сторона ужата по узкой стороне кадра")
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 512)
        self.assertLessEqual(y1, 768)
        # И лицо остаётся в центре кропа, а не уезжает под край.
        self.assertAlmostEqual((y0 + y1) / 2, 384, delta=1)
        self.assertTrue(rep["ok"], rep)

    def test_the_margin_makes_the_box_visibly_bigger_than_the_face(self):
        # Детектор отдаёт рамку ЛИЦА: волосы, подбородок и кольцо пера лежат
        # снаружи. Запас по умолчанию обязан их вместить.
        _, rep = self.r.union_box([_box(256, 200, 100)], size=(512, 768))
        self.assertGreaterEqual(rep["box_px"], 1.3 * rep["face_px"])

    def test_no_faces_at_all_is_a_separate_outcome_from_a_small_face(self):
        got, rep = self.r.union_box([None] * 16, size=(512, 768))
        self.assertIsNone(got)
        self.assertIsNone(rep["ok"])
        self.assertEqual(rep["outcome"], "no_faces")
        self.assertEqual(rep["with_face"], 0)

    def test_a_face_below_the_measurable_floor_is_REFUSED_with_the_reason(self):
        # 12 px лица дают кроп ~20 px: в нём меньше пикселей, чем ячеек в
        # латенте 64x64, и «доводка» выродилась бы в рисование с нуля.
        got, rep = self.r.union_box([_box(256, 200, 12)] * 16, size=(512, 768))
        self.assertIsNone(got)
        self.assertIs(rep["ok"], False)
        self.assertEqual(rep["outcome"], "too_small")
        self.assertIn("латент", rep["reason"])

    def test_the_refusal_and_the_unmeasurable_case_do_not_share_a_verdict(self):
        _, small = self.r.union_box([_box(256, 200, 12)], size=(512, 768))
        _, none = self.r.union_box([None], size=(512, 768))
        self.assertIsNot(small["ok"], none["ok"])

    def test_heavy_upsampling_is_named_generation_not_refinement(self):
        # 57 px — самая мелкая ИЗМЕРЕННАЯ в проекте доля лица (широкая стойка).
        _, tight = self.r.union_box([_box(256, 200, 57)], size=(512, 768))
        self.assertTrue(tight["heavy_upsample"], tight)
        self.assertIn("ПОРОЖДАЕТСЯ", tight["reason"])
        # 150 px увеличиваются вдвое — это ещё доводка.
        _, roomy = self.r.union_box([_box(256, 300, 150)], size=(512, 768))
        self.assertFalse(roomy["heavy_upsample"], roomy)

    def test_a_box_seen_on_a_handful_of_frames_is_flagged_not_trusted(self):
        boxes = [_box(256, 200, 78)] * 3 + [None] * 13
        _, rep = self.r.union_box(boxes, size=(512, 768))
        self.assertTrue(rep["ok"])
        self.assertTrue(rep["low_coverage"], rep)
        self.assertEqual(rep["coverage"], 0.188)

    def test_a_degenerate_box_is_an_error_not_a_silently_empty_crop(self):
        with self.assertRaises(ValueError) as cm:
            self.r.union_box([(100.0, 100.0, 100.0, 180.0)], size=(512, 768))
        self.assertIn("x0", str(cm.exception))

    def test_a_zero_sized_frame_is_refused(self):
        with self.assertRaises(ValueError):
            self.r.union_box([_box(10, 10, 8)], size=(0, 768))


class TheDecisionIsMadeBeforeTheGpuMinutesAreSpent(unittest.TestCase):
    """`refine_plan` — тот же жанр, что `run_local.identity_forecast`.

    Дорогой шаг оплачивается только после того, как сказано, что он даст. И
    сказано трёхзначно: «не смогли посчитать» — отдельный исход.
    """

    def setUp(self):
        from ball_reel import refine

        self.r = refine

    def test_the_answer_has_the_same_shape_as_the_other_forecasts(self):
        got = self.r.refine_plan(78, 768)
        for key in ("label", "ok", "value", "note"):
            self.assertIn(key, got)

    def test_the_measured_face_clears_the_bar_after_the_upscale(self):
        # 78 px -> x2 = 156 при баре 100.
        got = self.r.refine_plan(78, 768)
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["face_px_out"], 156.0)

    def test_even_the_most_pessimistic_measured_face_clears_the_bar(self):
        # 57 px — широкая стойка, самое мелкое измеренное по реальным условиям.
        got = self.r.refine_plan(57, 768)
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["face_px_out"], 114.0)

    def test_a_face_that_stays_below_the_bar_is_told_what_would_fix_it(self):
        # 40 px -> x2 = 80 < 100: доводка сделает картинку, но не измеримость.
        got = self.r.refine_plan(40, 768)
        self.assertIs(got["ok"], False)
        self.assertIn("НЕ ПРОВЕРЕНО", got["note"])
        self.assertIn("50", got["note"])           # нужно лицо от 50 px

    def test_a_face_with_nothing_to_refine_is_refused_before_the_upscale(self):
        # 20 px лица -> кроп 34 px, ниже поля измеримости 64.
        got = self.r.refine_plan(20, 768)
        self.assertIs(got["ok"], False)
        self.assertIn("доводить нечем", got["note"])

    def test_an_unmeasured_face_is_a_forecast_and_says_so_loudly(self):
        got = self.r.refine_plan(None, 768)
        self.assertIsNone(got["ok"])
        self.assertFalse(got["measured"])
        self.assertIn("НЕ ИЗМЕРЕН", got["note"])

    def test_the_bar_comes_from_the_gate_not_from_a_copy(self):
        from ball_reel.identity_arcface import MIN_FACE_PX

        self.assertEqual(self.r.refine_plan(78, 768)["bar"], MIN_FACE_PX)

    def test_the_typical_share_comes_from_animate_not_from_a_copy(self):
        from ball_reel.animate import FULL_BODY_FACE_SHARE

        got = self.r.refine_plan(None, 1000)
        self.assertAlmostEqual(got["face_px_in"], FULL_BODY_FACE_SHARE * 1000,
                               places=1)

    def test_every_verdict_repeats_that_the_arcface_number_will_be_biased(self):
        # Это не украшение отчёта: число, поданное без этой оговорки, читается
        # как доказательство личности, которым оно не является.
        for face in (None, 78, 57):
            with self.subTest(face=face):
                got = self.r.refine_plan(face, 768)
                self.assertIn("СМЕЩЕНО", got["note"])
                self.assertTrue(got["biased"])

    def test_a_nonsense_upscale_is_refused_rather_than_forecast(self):
        with self.assertRaises(ValueError):
            self.r.refine_plan(78, 768, upscale=0)
        with self.assertRaises(ValueError):
            self.r.refine_plan(78, 0)


class TheSeamMustNotBeVisible(unittest.TestCase):
    """Жёсткий край читается как наклейка мгновенно — даже при точном цвете."""

    def setUp(self):
        from ball_reel import refine

        self.r = refine

    def test_the_default_mask_has_a_real_gradient_not_two_values(self):
        mask = self.r.feather_mask((0, 0, 128, 128), 128)
        middle = [v for v in mask[64].tolist() if 0.0 < v < 1.0]
        self.assertGreater(len(middle), 4, mask[64].tolist()[:20])

    def test_the_mask_is_zero_on_the_border_and_one_in_the_middle(self):
        mask = self.r.feather_mask((0, 0, 200, 200), 200)
        self.assertEqual(mask.shape, (200, 200))
        self.assertEqual(mask[0].max(), 0.0)
        self.assertEqual(mask[-1].max(), 0.0)
        self.assertEqual(mask[:, 0].max(), 0.0)
        self.assertEqual(mask[:, -1].max(), 0.0)
        self.assertEqual(mask[100][100], 1.0)

    def test_the_mask_never_decreases_towards_the_centre(self):
        mask = self.r.feather_mask((0, 0, 100, 100), 100)
        row = mask[50].tolist()[:50]
        for a, b in zip(row, row[1:]):
            self.assertLessEqual(a, b)

    def test_a_zero_falloff_gives_the_hard_edge_it_promises(self):
        # Контрольный случай: так вклеивать не надо, но исход обязан быть
        # отличим от мягкого, иначе «перо работает» ничем не подтверждено.
        mask = self.r.feather_mask((0, 0, 64, 64), 64, falloff=0.0)
        self.assertEqual(sorted(set(mask.ravel().tolist())), [0.0, 1.0])

    def test_a_non_square_box_is_refused_rather_than_stretched(self):
        with self.assertRaises(ValueError) as cm:
            self.r.feather_mask((0, 0, 120, 90), 120)
        self.assertIn("квадрат", str(cm.exception))

    def test_the_feather_width_is_imported_from_marks_not_copied(self):
        from ball_reel.marks import FEATHER

        self.assertEqual(self.r.FEATHER, FEATHER)


class PastingBackIsArithmeticAndIsCheckedAsSuch(unittest.TestCase):
    """Всё, что может сдвинуться на пиксель, обязано падать громко.

    Сдвиг вклейки на 16 кадрах читается зрителем как дрожь лица, а в логе
    выглядит как успешный прогон.
    """

    def setUp(self):
        import numpy as np

        from ball_reel import refine

        self.np = np
        self.r = refine

    def _frame(self, value=10, w=64, h=96):
        return self.np.full((h, w, 3), value, dtype="uint8")

    def test_a_mask_of_ones_replaces_and_a_mask_of_zeros_does_not(self):
        frame = self._frame(10)
        patch = self.np.full((16, 16, 3), 200, dtype="uint8")
        ones = self.np.ones((16, 16))
        got = self.r.paste_back(frame, patch, (8, 8, 24, 24), ones)
        self.assertEqual(int(got[16, 16, 0]), 200)
        self.assertEqual(int(got[0, 0, 0]), 10)
        same = self.r.paste_back(frame, patch, (8, 8, 24, 24),
                                 self.np.zeros((16, 16)))
        self.assertTrue((same == frame).all())

    def test_a_half_mask_blends_rather_than_switches(self):
        frame = self._frame(0)
        patch = self.np.full((16, 16, 3), 100, dtype="uint8")
        half = self.np.full((16, 16), 0.5)
        got = self.r.paste_back(frame, patch, (8, 8, 24, 24), half)
        self.assertEqual(int(got[16, 16, 0]), 50)

    def test_the_input_frame_is_not_modified_in_place(self):
        frame = self._frame(10)
        before = frame.copy()
        self.r.paste_back(frame, self.np.full((16, 16, 3), 200, dtype="uint8"),
                          (8, 8, 24, 24), self.np.ones((16, 16)))
        self.assertTrue((frame == before).all())

    def test_the_dtype_survives_the_blend(self):
        frame = self._frame(10)
        got = self.r.paste_back(frame, self.np.full((16, 16, 3), 255,
                                                    dtype="uint8"),
                                (8, 8, 24, 24), self.np.ones((16, 16)))
        self.assertEqual(got.dtype, frame.dtype)
        self.assertEqual(int(got[16, 16, 0]), 255)

    def test_a_patch_of_the_wrong_size_names_both_numbers(self):
        with self.assertRaises(ValueError) as cm:
            self.r.paste_back(self._frame(), self.np.zeros((10, 10, 3),
                                                           dtype="uint8"),
                              (8, 8, 24, 24), self.np.ones((16, 16)))
        self.assertIn("10", str(cm.exception))
        self.assertIn("16", str(cm.exception))

    def test_a_box_outside_the_frame_is_refused_not_clipped(self):
        with self.assertRaises(ValueError) as cm:
            self.r.paste_back(self._frame(), self.np.zeros((16, 16, 3),
                                                           dtype="uint8"),
                              (56, 8, 72, 24), self.np.ones((16, 16)))
        self.assertIn("за кадр", str(cm.exception))

    def test_a_mask_outside_zero_to_one_is_refused(self):
        with self.assertRaises(ValueError):
            self.r.paste_back(self._frame(), self.np.zeros((16, 16, 3),
                                                           dtype="uint8"),
                              (8, 8, 24, 24), self.np.full((16, 16), 1.5))

    def test_the_feathered_seam_leaves_the_border_pixel_untouched(self):
        # Прямая проверка утверждения «маска не выходит за бокс»: на самой
        # рамке кадр обязан остаться исходным.
        frame = self._frame(10)
        patch = self.np.full((32, 32, 3), 200, dtype="uint8")
        mask = self.r.feather_mask((8, 8, 40, 40), 32)
        got = self.r.paste_back(frame, patch, (8, 8, 40, 40), mask)
        self.assertEqual(int(got[8, 8, 0]), 10)
        self.assertEqual(int(got[24, 24, 0]), 200)


class TheCropsAreOneStableCamera(unittest.TestCase):
    def setUp(self):
        from ball_reel import refine

        self.r = refine

    def _frames(self, n=4, value=30):
        from PIL import Image

        return [Image.new("RGB", (512, 768), (value, value, value))
                for _ in range(n)]

    def test_every_crop_comes_out_at_the_refine_side(self):
        got = self.r.crop_series(self._frames(4), (100, 100, 228, 228),
                                 side=64)
        self.assertEqual(len(got), 4)
        for im in got:
            self.assertEqual(im.size, (64, 64))

    def test_a_non_square_box_is_refused_before_any_resize(self):
        with self.assertRaises(ValueError) as cm:
            self.r.crop_series(self._frames(1), (100, 100, 228, 200), side=64)
        self.assertIn("квадрат", str(cm.exception))

    def test_the_round_trip_puts_the_patch_where_the_box_was(self):
        from PIL import Image

        frames = self._frames(3, value=30)
        patches = [Image.new("RGB", (512, 512), (200, 200, 200))
                   for _ in range(3)]
        got = self.r.paste_series(frames, patches, (100, 100, 228, 228))
        self.assertEqual(len(got), 3)
        px = got[0].load()
        self.assertEqual(px[164, 164][0], 200)     # середина бокса
        self.assertEqual(px[10, 10][0], 30)        # снаружи кадр не тронут

    def test_a_mismatched_number_of_patches_is_refused(self):
        with self.assertRaises(ValueError):
            self.r.paste_series(self._frames(3), self._frames(2),
                                (100, 100, 228, 228))


class MemoryIsCALCULATEDAndTheAnswerSaysSo(unittest.TestCase):
    """Расчёт и замер — разные вещи, и здесь только расчёт.

    Пометка не косметическая: число из этой функции попадёт в отчёт, и если
    пометка не приедет вместе с ним, читатель примет арифметику за измерение.
    """

    def setUp(self):
        from ball_reel import refine

        self.r = refine

    def test_the_square_crop_is_two_thirds_of_the_tall_frame(self):
        got = self.r.vram_delta(512, 768, side=512, frames=16)
        self.assertAlmostEqual(got["activation_ratio"], 0.667, delta=0.002)

    def test_a_stage_that_is_narrower_adds_nothing_to_the_peak(self):
        got = self.r.vram_delta(512, 768, side=512, frames=16)
        self.assertEqual(got["added_peak_gb"], 0.0)
        self.assertEqual(got["added_weights_gb"], 0.0)

    def test_a_stage_that_is_wider_refuses_to_promise_zero(self):
        got = self.r.vram_delta(384, 576, side=512, frames=16)
        self.assertGreater(got["activation_ratio"], 1.0)
        self.assertIsNone(got["added_peak_gb"])
        self.assertIn("ШИРЕ", got["note"])

    def test_the_answer_calls_itself_a_calculation_out_loud(self):
        got = self.r.vram_delta(512, 768)
        self.assertIn("РАСЧЁТ", got["note"])
        self.assertIn("не замер", got["note"])
        self.assertFalse(got["measured"])

    def test_the_only_absolute_number_is_labelled_a_lower_bound(self):
        got = self.r.vram_delta(512, 768, side=512, frames=16)
        # 64x64 ячеек x 16 кадров x 4 канала x 2 байта = 524288 байт.
        self.assertAlmostEqual(got["latent_gb_lower_bound"], 0.000524,
                               places=5)
        self.assertIn("НИЖНЯЯ ГРАНИЦА", got["note"])

    def test_the_idle_controlnet_is_counted_from_animate_not_guessed(self):
        from ball_reel.animate import WEIGHTS_GB

        got = self.r.vram_delta(512, 768)
        self.assertEqual(got["controlnet_idle_gb"], WEIGHTS_GB["controlnet"])

    def test_nonsense_sizes_are_refused(self):
        with self.assertRaises(ValueError):
            self.r.vram_delta(512, 768, frames=0)


class TheExpensiveStepIsRefusedEarly(unittest.TestCase):
    """Отказ обязан случиться до загрузки весов, а не в середине генерации."""

    def setUp(self):
        from ball_reel import refine

        self.r = refine

    def _crops(self, n=16, side=512):
        from PIL import Image

        return [Image.new("RGB", (side, side)) for _ in range(n)]

    def test_a_single_frame_instead_of_a_stack_is_refused_with_the_reason(self):
        from ball_reel.animate import CONTEXT_FRAMES

        with self.assertRaises(ValueError) as cm:
            self.r.refine_preflight(self._crops(1), face_embeds=[0.1])
        self.assertIn("мерцание", str(cm.exception))
        self.assertIn(str(CONTEXT_FRAMES), str(cm.exception))

    def test_a_missing_face_embedding_is_refused_because_it_draws_a_stranger(self):
        with self.assertRaises(ValueError) as cm:
            self.r.refine_preflight(self._crops(), face_embeds=None)
        self.assertIn("ПОСТОРОННЕГО", str(cm.exception))

    def test_a_denoise_that_repaints_rather_than_refines_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self.r.refine_preflight(self._crops(), face_embeds=[0.1],
                                    denoise=0.9)
        self.assertIn("перерисовка", str(cm.exception))

    def test_a_denoise_outside_the_open_unit_interval_is_refused(self):
        for bad in (0.0, -0.1, 1.5):
            with self.subTest(denoise=bad):
                with self.assertRaises(ValueError):
                    self.r.refine_preflight(self._crops(), face_embeds=[0.1],
                                            denoise=bad)

    def test_a_setting_that_would_run_zero_steps_is_refused_not_run(self):
        # int(10 * 0.05) = 0: diffusers вернёт вход нетронутым, и ступень будет
        # ВЫГЛЯДЕТЬ отработавшей. Худший вид отказа — похожий на работу.
        with self.assertRaises(ValueError) as cm:
            self.r.refine_preflight(self._crops(), face_embeds=[0.1],
                                    denoise=0.05, steps=10)
        self.assertIn("НЕТРОНУТЫМ", str(cm.exception))

    def test_a_non_square_crop_is_refused_before_the_pipeline_sees_it(self):
        from PIL import Image

        crops = self._crops(15) + [Image.new("RGB", (512, 400))]
        with self.assertRaises(ValueError) as cm:
            self.r.refine_preflight(crops, face_embeds=[0.1])
        self.assertIn("деформирует", str(cm.exception))

    def test_a_pipeline_that_still_carries_controlnet_is_refused(self):
        class Pipe:
            controlnet = object()

        with self.assertRaises(ValueError) as cm:
            self.r.refine_preflight(self._crops(), face_embeds=[0.1],
                                    pipe=Pipe())
        self.assertIn("ControlNet", str(cm.exception))

    def test_the_preflight_reports_the_steps_that_will_actually_run(self):
        # diffusers берёт int(steps * strength) — умолчание, меняющее смысл
        # параметра, и потому напечатанное, а не оставленное в уме.
        got = self.r.refine_preflight(self._crops(), face_embeds=[0.1],
                                      denoise=0.4, steps=30)
        self.assertEqual(got["effective_steps"], 12)
        self.assertIn("12", got["note"])

    def test_too_few_effective_steps_are_flagged_even_when_allowed(self):
        got = self.r.refine_preflight(self._crops(), face_embeds=[0.1],
                                      denoise=0.2, steps=10)
        self.assertEqual(got["effective_steps"], 2)
        self.assertIn("мало", got["note"])


class TheArgumentsAreCheckedAgainstTheRealPipeline(unittest.TestCase):
    """Против выдуманного API на этом проекте уже писали. Второй раз — нет.

    Здесь ключи вызова сверяются с НАСТОЯЩЕЙ сигнатурой установленного
    diffusers, а не с памятью автора.
    """

    def setUp(self):
        from ball_reel import refine

        self.r = refine

    def _crops(self, n=16, side=512):
        from PIL import Image

        return [Image.new("RGB", (side, side)) for _ in range(n)]

    def _signature(self):
        try:
            from diffusers import AnimateDiffVideoToVideoPipeline
        except ImportError:
            self.skipTest("diffusers не установлен")
        import inspect

        return AnimateDiffVideoToVideoPipeline, inspect.signature(
            AnimateDiffVideoToVideoPipeline.__call__).parameters

    def test_every_key_we_pass_exists_in_the_pipeline_signature(self):
        _, params = self._signature()
        kwargs = self.r.refine_call_kwargs(self._crops(), face_embeds=[0.1],
                                           prompt="portrait")
        for key in kwargs:
            with self.subTest(key=key):
                self.assertIn(key, params)

    def test_the_clip_length_comes_from_the_video_not_from_num_frames(self):
        # Отличие от первой ступени, которое легко проспать: `num_frames` у
        # vid2vid НЕТ, длину задаёт длина `video`.
        _, params = self._signature()
        self.assertNotIn("num_frames", params)
        kwargs = self.r.refine_call_kwargs(self._crops(), face_embeds=[0.1],
                                           prompt="portrait")
        self.assertEqual(len(kwargs["video"]), 16)

    def test_controlnet_conditions_are_not_even_accepted_here(self):
        # Ещё одна причина снять ControlNet, а не «просто не передавать условия».
        _, params = self._signature()
        self.assertNotIn("conditioning_frames", params)

    def test_the_denoise_is_passed_under_the_name_the_pipeline_uses(self):
        _, params = self._signature()
        self.assertIn("strength", params)
        kwargs = self.r.refine_call_kwargs(self._crops(), face_embeds=[0.1],
                                           prompt="p", denoise=0.3)
        self.assertEqual(kwargs["strength"], 0.3)
        self.assertNotIn("denoise", kwargs)

    def test_the_pipeline_can_be_rebuilt_from_the_loaded_one(self):
        cls, _ = self._signature()
        self.assertTrue(hasattr(cls, "from_pipe"))

    def test_the_face_embedding_is_wrapped_in_a_list_as_the_adapter_expects(self):
        kwargs = self.r.refine_call_kwargs(self._crops(), face_embeds="EMB",
                                           prompt="p")
        self.assertEqual(kwargs["ip_adapter_image_embeds"], ["EMB"])

    def test_the_crops_are_sent_square_at_the_side_they_were_made(self):
        kwargs = self.r.refine_call_kwargs(self._crops(side=512),
                                           face_embeds=[0.1], prompt="p")
        self.assertEqual((kwargs["width"], kwargs["height"]), (512, 512))


class TheThinWrapperIsStillCheckable(unittest.TestCase):
    """Сам вызов без карты не исполняется — но подставной пайплайн исполняется."""

    def setUp(self):
        from ball_reel import refine

        self.r = refine

    def _crops(self, n=16, side=512):
        from PIL import Image

        return [Image.new("RGB", (side, side)) for _ in range(n)]

    def _pipe(self):
        class Out:
            frames = [["FRAME"] * 16]

        class Pipe:
            controlnet = None

            def __init__(self):
                self.scale = None
                self.kwargs = None

            def set_ip_adapter_scale(self, v):
                self.scale = v

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return Out()

        return Pipe()

    def test_the_scale_is_set_before_the_call_and_the_frames_come_back(self):
        pipe = self._pipe()
        got = self.r.refine(pipe, self._crops(), face_embeds=[0.1],
                            prompt="portrait", ip_adapter_scale=0.8)
        self.assertEqual(pipe.scale, 0.8)
        self.assertEqual(len(got), 16)
        self.assertEqual(pipe.kwargs["prompt"], "portrait")

    def test_the_refusal_happens_before_the_pipeline_is_touched(self):
        pipe = self._pipe()
        with self.assertRaises(ValueError):
            self.r.refine(pipe, self._crops(3), face_embeds=[0.1], prompt="p")
        self.assertIsNone(pipe.scale)
        self.assertIsNone(pipe.kwargs)


class WhatThisStageDoesNotProveIsWrittenDownNotHidden(unittest.TestCase):
    """Оговорка про смещение обязана ездить вместе с числом.

    Она живёт в докстринге модуля, и это проверяется: докстринг здесь —
    не украшение, а часть протокола демо. Число ArcFace после доводки будет
    произнесено вслух, и рядом обязано прозвучать, чего оно не доказывает.
    """

    def setUp(self):
        from ball_reel import refine

        self.r = refine

    def test_the_module_says_the_face_is_generated_not_restored(self):
        doc = self.r.__doc__
        self.assertIn("НЕ ВОССТАНАВЛИВАЕТ", doc)
        self.assertIn("ПОРОЖДАЕТ", doc)

    def test_the_module_names_what_the_stage_does_not_prove(self):
        doc = self.r.__doc__
        self.assertIn("ЧЕГО ДОВОДКА НЕ ДОКАЗЫВАЕТ", doc)
        # Пробелы схлопываются: фраза в докстринге перенесена по строкам, и
        # тест, ищущий её подстрокой, краснел на верном тексте — то есть
        # сторожил ширину абзаца вместо смысла.
        flat = " ".join(doc.split())
        self.assertIn("Судья зависит от судимого", flat)

    def test_the_module_names_both_face_share_numbers_instead_of_choosing(self):
        # Расхождение между документами — находка, а не мелочь: 0.094 в коде
        # против 0.102 в постановке задачи, и оба названы вслух.
        doc = self.r.__doc__
        self.assertIn("0.094", doc)
        self.assertIn("0.102", doc)

    def test_the_upscale_is_named_as_someone_elses_module(self):
        self.assertIn("upscale.py", self.r.__doc__)


if __name__ == "__main__":
    unittest.main()
