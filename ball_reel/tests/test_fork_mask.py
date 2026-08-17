"""Поток C: главный тест — про предел 32 px, и он обязан краснеть.

Расширение, не пережившее блокификацию, — это НЕ «гипотеза не сработала», а «мы
её не подали». Если этот тест зелёный на маске, выросшей на 5 px, весь поток
отдаёт число, которое ничего не значит.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ball_reel import fork_mask as fm


def _blob(h=256, w=256, box=(64, 64, 192, 192)):
    """Прямоугольный «силуэт». Прямоугольник, а не круг: границы блоков тогда
    считаются в уме, и ожидаемое в тесте можно написать литералом (Т2)."""
    m = np.zeros((h, w), dtype=bool)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = True
    return m


class TheThirtyTwoPixelFloorIsTheWholePoint(unittest.TestCase):
    """Главная проверка потока."""

    def test_a_growth_that_survives_blockification_is_delivered(self):
        base = _blob()
        grown = fm.dilate(base, fm.BLOCK)
        got = fm.growth_verdict(base, grown)
        self.assertEqual(got["outcome"], fm.PASS)
        self.assertGreater(got["blocks"], 0)

    def test_a_growth_that_dies_in_blockification_is_refused_as_unmeasured(self):
        """Силуэт по границам блоков: прирост в 5 px не занимает нового блока.

        Бокс (64,64,192,192) выровнен по сетке 32. Рост на 5 px выходит за него
        на 5 px — то есть внутрь СОСЕДНИХ блоков... поэтому силуэт берётся
        с запасом внутрь: (69,69,187,187), и рост на 5 остаётся в своих блоках.
        """
        base = _blob(box=(69, 69, 187, 187))
        grown = fm.dilate(base, 5)
        got = fm.growth_verdict(base, grown)
        self.assertGreater(got["pixels"], 0, "фикстура не выросла вовсе")
        self.assertEqual(got["blocks"], 0)
        self.assertEqual(got["outcome"], fm.UNMEASURED,
                         "невидимое расширение выдано за результат")

    def test_the_refusal_does_not_read_as_the_hypothesis_failing(self):
        base = _blob(box=(69, 69, 187, 187))
        got = fm.growth_verdict(base, fm.dilate(base, 5))
        self.assertNotEqual(got["outcome"], fm.FAIL)
        self.assertIn("не смогли подать", got["note"].lower())
        self.assertIn("НЕ «гипотеза не", got["note"])

    def test_the_refusal_names_the_number_to_grow_by(self):
        base = _blob(box=(69, 69, 187, 187))
        self.assertIn("32", fm.growth_verdict(base, fm.dilate(base, 5))["note"])

    def test_mutating_the_block_size_flips_the_verdict_both_ways(self):
        """Т1: константа-решение подменяется в ОБЕ стороны.

        Мельче блок — то же расширение становится видимым; крупнее — перестаёт.
        Если ни то ни другое не меняет исход, блок не участвует в решении.
        """
        base = _blob(box=(69, 69, 187, 187))
        grown = fm.dilate(base, 5)
        self.assertEqual(fm.growth_verdict(base, grown, block=32)["outcome"],
                         fm.UNMEASURED)
        self.assertEqual(fm.growth_verdict(base, grown, block=1)["outcome"],
                         fm.PASS, "блок в 1 px обязан пропустить любой прирост")

        big = fm.dilate(_blob(box=(69, 69, 187, 187)), 40)
        self.assertEqual(fm.growth_verdict(base, big, block=32)["outcome"],
                         fm.PASS)
        self.assertEqual(fm.growth_verdict(base, big, block=1024)["outcome"],
                         fm.UNMEASURED,
                         "блок больше кадра обязан съесть любое расширение")

    def test_no_growth_at_all_is_unmeasured_and_says_a_different_thing(self):
        base = _blob()
        got = fm.growth_verdict(base, base)
        self.assertEqual(got["outcome"], fm.UNMEASURED)
        self.assertEqual(got["pixels"], 0)
        self.assertIn("ни одного пикселя", got["note"])

    def test_mismatched_shapes_are_unmeasured_not_a_crash(self):
        got = fm.growth_verdict(_blob(64, 64), _blob(128, 128))
        self.assertEqual(got["outcome"], fm.UNMEASURED)
        self.assertIn("разного размера", got["note"])

    def test_the_share_of_gained_blocks_is_reported_not_only_the_count(self):
        """Найдено живьём: на рваном силуэте даже 5 px занимает пару блоков.

        «Дошло» и «дошло на 1%» — разные вещи, и число обязано их различать.
        """
        base = _blob(box=(64, 64, 192, 192))          # ровно 4x4 = 16 блоков
        got = fm.growth_verdict(base, fm.dilate(base, fm.BLOCK))
        self.assertEqual(got["held"], 16)
        self.assertGreater(got["share"], 0.0)
        self.assertAlmostEqual(got["share"], got["blocks"] / got["held"],
                               places=4)

    def test_the_occupied_count_survives_a_frame_that_is_not_a_multiple(self):
        """Краевой блок урезан; делить сумму пикселей на 32*32 здесь нельзя."""
        m = np.zeros((40, 40), dtype=bool)
        m[35, 35] = True
        self.assertEqual(fm._blocks_occupied(m, 32), 1)

    def test_the_vendor_numbers_are_the_ones_read_from_the_graph(self):
        """Т2: литералы, сверенные с файлом графа, а не импорт из кода."""
        self.assertEqual(fm.BLOCK, 32)
        self.assertEqual(fm.VENDOR_GROW, 10)


class BlockificationBehavesLikeQuantisation(unittest.TestCase):

    def test_a_single_pixel_claims_its_whole_block(self):
        m = np.zeros((64, 64), dtype=bool)
        m[10, 10] = True
        got = fm.blockify(m, 32)
        self.assertTrue(got[0:32, 0:32].all())
        self.assertFalse(got[32:, :].any())
        self.assertEqual(int(got.sum()), 32 * 32)

    def test_an_empty_mask_stays_empty(self):
        """Негативный контроль (И5): квантователь не должен выдумывать маску."""
        self.assertFalse(fm.blockify(np.zeros((64, 64), dtype=bool), 32).any())

    def test_a_ragged_edge_block_is_still_filled(self):
        m = np.zeros((40, 40), dtype=bool)
        m[35, 35] = True
        got = fm.blockify(m, 32)
        self.assertTrue(got[32:40, 32:40].all())

    def test_a_nonpositive_block_is_refused(self):
        for bad in (0, -32):
            with self.subTest(block=bad):
                with self.assertRaises(ValueError):
                    fm.blockify(_blob(), bad)

    def test_blockification_only_ever_adds(self):
        m = _blob()
        self.assertTrue((fm.blockify(m, 32) | m == fm.blockify(m, 32)).all(),
                        "квантование стёрло часть маски — тогда предел 32 px "
                        "перестаёт быть нижней оценкой")


class HandsAndHeadAreFrozen(unittest.TestCase):
    """Кисть на мяче — это то самое взаимодействие, которое нельзя ломать."""

    def _points(self, hand_at, head_at):
        pts = [(0.0, 0.0, 0.0)] * fork_joints()
        for i in range(0, 5):
            pts[i] = (head_at[0], head_at[1], 0.9)
        lo, hi = _groups()["l_hand"]
        for i in range(lo, hi):
            pts[i] = (hand_at[0], hand_at[1], 0.9)
        return pts

    def test_growth_is_suppressed_inside_the_hand_circle(self):
        base = _blob(box=(64, 64, 192, 192))
        pts = self._points(hand_at=(64.0, 128.0), head_at=(400.0, 400.0))
        grown = fm.grow_selective(base, pts, 16, width=256, height=256)
        free = fm.grow_selective(base, [], 16, width=256, height=256)
        added_frozen = int((grown & ~base).sum())
        added_free = int((free & ~base).sum())
        self.assertLess(added_frozen, added_free,
                        "заморозка кисти ничего не изменила — маска кисти "
                        "растёт, и мяч попадает под перерисовку")
        # Слева от силуэта, ровно на кисти, прироста быть не должно вовсе.
        self.assertFalse(grown[126:130, 60:64].any(),
                         "прирост лёг прямо на кисть")

    def test_the_free_run_does_grow_there_which_proves_the_test_can_tell(self):
        """Негативный контроль к предыдущему (И5)."""
        base = _blob(box=(64, 64, 192, 192))
        free = fm.grow_selective(base, [], 16, width=256, height=256)
        self.assertTrue(free[126:130, 60:64].any(),
                        "без заморозки прироста тоже нет — тест меряет не то")

    def test_the_original_silhouette_is_never_eaten_by_freezing(self):
        base = _blob(box=(64, 64, 192, 192))
        pts = self._points(hand_at=(128.0, 128.0), head_at=(128.0, 128.0))
        grown = fm.grow_selective(base, pts, 16, width=256, height=256)
        self.assertTrue((grown | base == grown).all(),
                        "заморозка отняла часть ИСХОДНОГО силуэта, а должна "
                        "отнимать только прирост — иначе в маске дыра")

    def test_frozen_zones_are_empty_without_points(self):
        self.assertEqual(fm.frozen_zones([], 256, 256), [])

    def test_frozen_zones_ignore_unobservable_points(self):
        pts = [(10.0, 10.0, 0.0)] * fork_joints()
        self.assertEqual(fm.frozen_zones(pts, 256, 256), [],
                         "точки ниже порога наблюдаемости задали зону "
                         "заморозки — заморозим по шуму")

    def test_the_head_zone_is_wider_than_the_hand_zone(self):
        self.assertGreater(fm.HEAD_FREEZE, fm.HAND_FREEZE)


class DilationIsRoundAndBounded(unittest.TestCase):

    def test_zero_radius_changes_nothing(self):
        m = _blob()
        self.assertTrue((fm.dilate(m, 0) == m).all())

    def test_the_grown_box_is_exactly_radius_wider_on_each_side(self):
        m = np.zeros((64, 64), dtype=bool)
        m[30:34, 30:34] = True
        got = fm.dilate(m, 3)
        ys, xs = np.nonzero(got)
        self.assertEqual((int(ys.min()), int(ys.max())), (27, 36))
        self.assertEqual((int(xs.min()), int(xs.max())), (27, 36))

    def test_the_corner_is_rounded_off(self):
        """Круг, а не квадрат: квадратный элемент даёт плечи, которых нет."""
        m = np.zeros((64, 64), dtype=bool)
        m[32, 32] = True
        got = fm.dilate(m, 5)
        self.assertTrue(got[32, 37], "по стороне не выросло на радиус")
        self.assertFalse(got[27, 27], "угол квадратный — элемент не круглый")

    def test_growth_does_not_wrap_around_the_frame(self):
        m = np.zeros((32, 32), dtype=bool)
        m[0, 0] = True
        got = fm.dilate(m, 4)
        self.assertFalse(got[-1, -1], "расширение обернулось через край кадра")


class PersonClassesComeFromTheModelsOwnLabels(unittest.TestCase):
    """Е1: список классов читан из метаданных весов, копии быть не должно."""

    def test_background_is_the_only_thing_excluded(self):
        from ball_reel import bodyparts

        self.assertEqual(
            fm.PERSON_CLASSES,
            tuple(i for i, n in enumerate(bodyparts.LABELS) if n != "background"))
        self.assertNotIn(bodyparts.LABELS.index("background"),
                         fm.PERSON_CLASSES)

    def test_clothes_and_hair_count_as_the_character(self):
        from ball_reel import bodyparts

        for name in ("clothes", "hair"):
            with self.subTest(name=name):
                self.assertIn(bodyparts.LABELS.index(name), fm.PERSON_CLASSES,
                              "одежда или волосы объявлены фоном — силуэт "
                              "персонажа окажется голым телом")


class TheSequenceReportsNumbersNotAFlag(unittest.TestCase):

    def test_an_empty_run_is_unmeasured_and_says_zero_of_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fm.sequence([], tmp)
        self.assertEqual(got["outcome"], fm.UNMEASURED)
        self.assertIn("0 из 0", got["note"])

    def test_the_note_says_out_loud_that_masks_were_checked_not_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fm.sequence([], tmp)
        self.assertIn("ПРОВЕРЕНЫ МАСКИ, НЕ ГЕНЕРАЦИЯ", got["note"])

    def test_the_report_carries_the_settings_it_ran_with(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fm.sequence([], tmp, grow_px=48, block=16)
        self.assertEqual((got["grow_px"], got["block"]), (48, 16))


def fork_joints() -> int:
    from ball_reel import fork_channels

    return fork_channels.WHOLEBODY_JOINTS


def _groups() -> dict:
    from ball_reel import fork_channels

    return fork_channels.WHOLEBODY_GROUPS


if __name__ == "__main__":
    unittest.main()
