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

    def test_the_refusal_does_not_blame_the_model_for_our_own_floor(self):
        """Закрытый дефект: докстринг звал 32 px пределом МОДЕЛИ.

        Предел модели 16 px (латент /8 x патч 2), 32 — наш `BlockifyMask`.
        Отказ обязан называть предел своим, иначе число станет неприкасаемым.
        """
        base = _blob(box=(69, 69, 187, 187))
        note = fm.growth_verdict(base, fm.dilate(base, 5))["note"]
        self.assertIn("НАШУ блокификацию", note)
        self.assertIn("16", note, "предел модели в отказе не назван")
        self.assertNotIn("для модели этого расширения не существует",
                         note.lower(),
                         "вернулась формулировка, объявлявшая наш предел "
                         "свойством весов")


class TheModelsOwnFloorIsSixteenNotThirtyTwo(unittest.TestCase):
    """Числа прочитаны в первоисточнике, а не вспомнены (И4, §9 хэндофа).

    `comfy_extras/nodes_wan.py`: `latent_width = width // 8`, и
    `character_mask` уводится `common_upscale` ровно в `concat_latent_image`.
    `config.json` весов Wan2.2-Animate-14B: `patch_size = [1, 2, 2]`.
    """

    def test_the_two_factors_are_the_ones_read_from_the_sources(self):
        self.assertEqual(fm.LATENT_DOWNSCALE, 8)
        self.assertEqual(fm.PATCH, 2)

    def test_the_model_floor_is_derived_not_copied(self):
        """Е1: 16 не лежит третьим числом — оно произведение двух первых."""
        self.assertEqual(fm.MODEL_TOKEN_PX, 16)
        self.assertEqual(fm.MODEL_TOKEN_PX, fm.LATENT_DOWNSCALE * fm.PATCH)

    def test_our_floor_is_coarser_than_the_models_and_that_is_the_finding(self):
        self.assertGreater(fm.BLOCK, fm.MODEL_TOKEN_PX,
                           "наш блокификатор перестал быть грубее модели — "
                           "тогда весь разбор про 16 против 32 неверен")
        self.assertEqual(fm.BLOCK % fm.MODEL_TOKEN_PX, 0,
                         "наш блок не кратен токену модели — блокификация "
                         "перестанет быть чистым огрублением")

    def test_a_growth_invisible_to_us_can_still_be_visible_at_the_models_floor(self):
        """Негативный контроль к главному тесту (И5): разница НАБЛЮДАЕМА.

        Тот же прирост, который наш блок 32 съедает, при 16 px доезжает.
        Если бы не доезжал, различать 16 и 32 было бы не на чем.
        """
        # Бокс выровнен по сетке 16, но НЕ по сетке 32: рост на 12 px выходит
        # в соседний блок 16 и остаётся внутри своего блока 32.
        base = _blob(box=(80, 80, 176, 176))
        grown = fm.dilate(base, 12)
        self.assertEqual(fm.growth_verdict(base, grown, block=32)["outcome"],
                         fm.UNMEASURED)
        self.assertEqual(
            fm.growth_verdict(base, grown, block=fm.MODEL_TOKEN_PX)["outcome"],
            fm.PASS,
            "при пределе модели прирост тоже невидим — тогда 32 и 16 "
            "неразличимы и находка пуста")

    def test_the_share_of_the_frame_claimed_is_reported(self):
        """§1a: расширение съедает окружение, и это обязано быть числом.

        Одна маска на 16 блоков из 64 — 25% кадра; после роста больше.
        """
        base = _blob(box=(64, 64, 192, 192))          # 4x4 блока в кадре 8x8
        got = fm.growth_verdict(base, fm.dilate(base, fm.BLOCK))
        self.assertEqual(got["frame"], 64)
        self.assertAlmostEqual(got["claimed"], 36 / 64, places=4)
        self.assertGreater(got["claimed"], 16 / 64,
                           "доля кадра не выросла после расширения")


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


class TheMaskIsQuantisedInTimeToo(unittest.TestCase):
    """Найдено чтением ноды: `view(1, T//4, 4, H, W).transpose(1, 2)`.

    Четыре соседних кадра — четыре канала ОДНОЙ латентной ячейки. Прибор
    отвечает на вопрос «выразимо ли то, что мы отдаём, в этой решётке», и у
    него двусторонний контроль (И5): различие внутри четвёрки роняет, различие
    ровно на границе четвёрки — нет.
    """

    def test_the_group_size_is_the_one_read_from_the_node(self):
        self.assertEqual(fm.TEMPORAL_GROUP, 4)

    def test_identical_frames_are_expressible(self):
        got = fm.temporal_report([_blob()] * 8)
        self.assertEqual(got["outcome"], fm.PASS)
        self.assertEqual((got["groups"], got["varying"]), (2, 0))

    def test_a_change_inside_the_cell_is_not_expressible(self):
        frames = [_blob()] * 4 + [_blob()] * 3 + [_blob(box=(64, 64, 224, 224))]
        got = fm.temporal_report(frames)
        self.assertEqual(got["outcome"], fm.FAIL)
        self.assertEqual(got["varying"], 1, "различие внутри второй четвёрки "
                                            "не найдено")

    def test_a_change_exactly_on_the_cell_boundary_is_fine(self):
        """Негативный контроль: прибор обязан молчать, когда терять нечего."""
        frames = [_blob()] * 4 + [_blob(box=(64, 64, 224, 224))] * 4
        got = fm.temporal_report(frames)
        self.assertEqual(got["outcome"], fm.PASS,
                         "сдвиг на границе ячейки объявлен невыразимым — "
                         "прибор меряет не то")
        self.assertEqual(got["varying"], 0)

    def test_a_difference_too_small_to_survive_blockification_is_not_counted(self):
        """Считаются блокифицированные кадры: то, что не доедет, не потеряно."""
        a = _blob(box=(69, 69, 187, 187))
        b = a.copy()
        b[70, 70] = False                     # блок остаётся занятым
        self.assertEqual(fm.temporal_report([a, b, a, b])["outcome"], fm.PASS)

    def test_mutating_the_group_size_flips_the_verdict_both_ways(self):
        """Т1: константа-решение подменяется в ОБЕ стороны."""
        frames = [_blob()] * 4 + [_blob(box=(64, 64, 224, 224))] * 4
        self.assertEqual(fm.temporal_report(frames, group=4)["outcome"],
                         fm.PASS)
        self.assertEqual(fm.temporal_report(frames, group=8)["outcome"],
                         fm.FAIL,
                         "ячейка в 8 кадров обязана поглотить сдвиг на 5-м")
        self.assertEqual(fm.temporal_report(frames, group=1)["outcome"],
                         fm.PASS,
                         "ячейка в один кадр не может ничего потерять")

    def test_too_few_frames_is_unmeasured_not_a_pass(self):
        got = fm.temporal_report([_blob()] * 3)
        self.assertEqual(got["outcome"], fm.UNMEASURED)
        self.assertEqual(got["groups"], 0)

    def test_mismatched_frames_are_unmeasured_not_a_crash(self):
        got = fm.temporal_report([_blob(64, 64)] * 3 + [_blob(128, 128)])
        self.assertEqual(got["outcome"], fm.UNMEASURED)
        self.assertIn("разного размера", got["note"])

    def test_the_tail_outside_the_cells_is_named(self):
        got = fm.temporal_report([_blob()] * 6)
        self.assertEqual((got["groups"], got["tail"]), (1, 2))


class TheThreeArmsAreNamedNotRemembered(unittest.TestCase):
    """§8 просит три плеча по ширине. Числа ВЫБРАНЫ, обоснование — в модуле."""

    def test_all_three_arms_exist_and_are_ordered(self):
        self.assertEqual(sorted(fm.ARMS), ["narrow", "wide", "wider"])
        self.assertEqual(fm.arm("narrow"), 0)
        self.assertEqual(fm.arm("wide"), 32)
        self.assertEqual(fm.arm("wider"), 64)

    def test_every_arm_carries_its_reason(self):
        for name, spec in fm.ARMS.items():
            with self.subTest(name=name):
                self.assertGreater(len(spec["why"]), 40,
                                   "плечо без обоснования — число по памяти")

    def test_the_wide_arms_clear_our_own_floor(self):
        """Плечо, не переживающее блокификацию, — не плечо, а пустой прогон."""
        for name in ("wide", "wider"):
            with self.subTest(name=name):
                self.assertGreaterEqual(fm.arm(name), fm.BLOCK)
                self.assertGreaterEqual(fm.arm(name), fm.MODEL_TOKEN_PX)

    def test_the_arms_are_actually_distinguishable_after_blockification(self):
        """Плечи обязаны разойтись НА БЛОКАХ, а не только в пикселях."""
        base = _blob(box=(64, 64, 192, 192))
        counts = {n: fm.growth_verdict(base, fm.dilate(base, fm.arm(n)))
                  ["blocks"] for n in ("narrow", "wide", "wider")}
        self.assertEqual(counts["narrow"], 0)
        self.assertLess(counts["wide"], counts["wider"],
                        f"широкое и ещё более широкое дали одно и то же: "
                        f"{counts}")

    def test_the_narrow_arm_is_not_the_bare_silhouette_and_says_so(self):
        """Найдено глазами (П3): блокификация сама даёт лестницу.

        На demo/hero.png силуэт 55.3% кадра, после блоков 65.6% — +40 блоков
        при +50 у плеча `wide`. Плечо `narrow` обязано быть измерено ПОСЛЕ
        блокификации, иначе разрыв между плечами читается завышенным.
        """
        base = _blob(box=(70, 70, 190, 190))          # край не на сетке 32
        got = fm.growth_verdict(base, fm.dilate(base, fm.arm("narrow")))
        self.assertEqual(got["outcome"], fm.UNMEASURED)
        raw_share = base.sum() / base.size
        self.assertGreater(got["claimed"], raw_share,
                           "маска после блоков не шире сырого силуэта — "
                           "лестницы нет, и замер на hero.png не объясним")

    def test_an_unknown_arm_raises_instead_of_defaulting(self):
        with self.assertRaises(KeyError):
            fm.arm("широкая")

    def test_naming_an_arm_and_a_width_at_once_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                fm.sequence([], tmp, grow_px=48, arm_name="wide")

    def test_the_arm_name_reaches_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fm.sequence([], tmp, arm_name="wider")
        self.assertEqual((got["arm"], got["grow_px"]), ("wider", 64))


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

    def test_the_note_never_lets_the_temporal_grid_go_unsaid(self):
        """Молчание о шаге 4 читалось бы как покадровое управление силуэтом."""
        with tempfile.TemporaryDirectory() as tmp:
            got = fm.sequence([], tmp)
        self.assertIn("покадрового управления силуэтом нет", got["note"])
        self.assertIn("4 кадра", got["note"])

    def test_the_temporal_verdict_is_a_separate_field_not_the_outcome(self):
        """Два разных вопроса — два поля. Слить их значило бы соврать обоими."""
        with tempfile.TemporaryDirectory() as tmp:
            got = fm.sequence([], tmp)
        self.assertIn("temporal", got)
        self.assertIsNot(got["temporal"], got["outcome"])
        self.assertEqual(got["temporal"]["outcome"], fm.UNMEASURED)

    def test_the_note_says_whose_floor_the_block_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fm.sequence([], tmp)
        self.assertIn("НАШ предел", got["note"])
        self.assertIn("16", got["note"])


def fork_joints() -> int:
    from ball_reel import fork_channels

    return fork_channels.WHOLEBODY_JOINTS


def _groups() -> dict:
    from ball_reel import fork_channels

    return fork_channels.WHOLEBODY_GROUPS


if __name__ == "__main__":
    unittest.main()
