"""Поток B: ось протечки обязана КРАСНЕТЬ на нарисованном пятне вне маски.

Тест, ради которого написан модуль, один: берём кадр, копируем, ставим пятно
СНАРУЖИ маски — ось обязана это увидеть. И зеркальный к нему: то же пятно
ВНУТРИ маски ось видеть НЕ должна, иначе она меряет не границу, а картинку
целиком, и вся затея с Mix теряет смысл.

Второй пласт тестов появился после §1a хэндофа, снявшего утверждение «вне маски
пиксели копируются». Раз копирования нет и весь кадр идёт через VAE и сэмплер,
то ~~кодековая планка~~ снята, а на её место встают два пола, и тесты обязаны
сторожить главное свойство новой конструкции: **планка с чужим клеймом не
судит**. Ошибка была не в числе 0.004513, а в том, что число из одного прогона
подставили туда, где нужен совсем другой, — от этого и сторожим.
"""

from __future__ import annotations

import unittest

import numpy as np

from ball_reel import fork_leak as fl


def _frame(h=64, w=64, value=0.5):
    return np.full((h, w, 3), value, dtype="float64")


def _mask(h=64, w=64, box=(16, 16, 48, 48)):
    m = np.zeros((h, w), dtype=bool)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def _loop_floor(floor=0.01, floor_max=0.09):
    """Пол (б) с настоящим клеймом — то, чем ось имеет право судить."""
    return fl.declared_full_loop_floor(floor=floor, floor_max=floor_max,
                                       frames=13, measured_by="фикстура теста")


class TheAxisFindsASpotOutsideTheMask(unittest.TestCase):
    """Сторож, который не умеет краснеть, — украшение."""

    def test_a_painted_spot_outside_shows_up(self):
        driving = _frame()
        output = driving.copy()
        output[2:6, 2:6] = 1.0                      # пятно вне маски
        got = fl.outside_divergence(output, driving, _mask())
        self.assertGreater(got["max"], 0.4,
                           "ось не увидела пятно вне маски — она не сторожит "
                           "ту единственную поверхность, ради которой заведена")
        self.assertGreater(got["mean"], 0.0)

    def test_the_same_spot_inside_the_mask_is_invisible_to_the_axis(self):
        """Зеркальный тест. Внутри маски модель имеет право на всё."""
        driving = _frame()
        output = driving.copy()
        output[20:44, 20:44] = 1.0                  # пятно ЦЕЛИКОМ внутри
        got = fl.outside_divergence(output, driving, _mask())
        self.assertEqual(got["max"], 0.0,
                         "ось увидела перерисовку ВНУТРИ маски — тогда она "
                         "меряет кадр целиком, а не границу")

    def test_identical_frames_read_as_zero(self):
        """Негативный контроль (И5): вход, где прибор обязан сказать «ничего»."""
        driving = _frame()
        got = fl.outside_divergence(driving.copy(), driving, _mask())
        self.assertEqual((got["mean"], got["max"]), (0.0, 0.0))

    def test_a_faint_wide_shift_is_caught_by_the_mean_when_the_max_is_small(self):
        driving = _frame()
        output = driving.copy()
        output[:16, :] += 0.02                      # слабый сдвиг широкой полосой
        got = fl.outside_divergence(output, driving, _mask())
        self.assertLess(got["max"], 0.05, "фикстура задумана слабой")
        self.assertGreater(got["mean"], 0.0,
                           "среднее слепо к равномерному сдвигу — тогда "
                           "перекрашенный фон пройдёт")

    def test_a_tiny_spot_is_caught_by_the_max_when_the_mean_is_small(self):
        driving = _frame()
        output = driving.copy()
        output[1, 1] = 1.0                          # один пиксель
        got = fl.outside_divergence(output, driving, _mask())
        self.assertLess(got["mean"], 0.001, "фикстура задумана точечной")
        self.assertEqual(got["max"], 0.5,
                         "максимум слеп к точечному пятну — тогда стёртый мяч "
                         "пройдёт, он занимает проценты площади")


class ThereAreThreeOutcomesNotTwo(unittest.TestCase):

    def test_a_mask_that_eats_the_frame_is_unmeasured_not_leak_free(self):
        driving = _frame()
        full = np.ones((64, 64), dtype=bool)
        got = fl.outside_divergence(driving.copy(), driving, full)
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("НЕ «протечки нет»", got["note"])

    def test_the_share_bar_is_guarded_in_both_directions(self):
        """Т1: константа-решение подменяется строже и слабее."""
        driving = _frame()
        m = np.ones((64, 64), dtype=bool)
        m[:2, :] = False                     # ровно 3.125% вне маски
        original = fl.MIN_OUTSIDE_SHARE
        try:
            fl.MIN_OUTSIDE_SHARE = 0.01
            self.assertIsNotNone(
                fl.outside_divergence(driving.copy(), driving, m)["mean"],
                "порог ослаблен, а измерение всё равно отказано")
            fl.MIN_OUTSIDE_SHARE = 0.5
            self.assertEqual(
                fl.outside_divergence(driving.copy(), driving, m)["outcome"],
                fl.UNMEASURED,
                "порог поднят выше доступного, а измерение всё равно прошло — "
                "значит порог ни на что не влияет")
        finally:
            fl.MIN_OUTSIDE_SHARE = original

    def test_mismatched_frames_are_unmeasured(self):
        got = fl.outside_divergence(_frame(32, 32), _frame(64, 64), _mask())
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("разного размера", got["note"])

    def test_a_mask_of_the_wrong_shape_is_unmeasured(self):
        got = fl.outside_divergence(_frame(), _frame(), _mask(32, 32))
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("не по кадру", got["note"])


class TheCodecFloorIsGoneAndCannotComeBackThroughTheSideDoor(unittest.TestCase):
    """Главный тест новой конструкции, и он про §1a, а не про арифметику.

    ~~`noise_floor` на ffmpeg, планка 0.004513~~ — удалена. Проверяем две вещи
    сразу: что её больше нет в модуле, и что число такого происхождения нельзя
    внести в вердикт «сбоку», просто передав словарь с ключом `floor`.
    """

    def test_the_codec_floor_function_is_gone_from_the_module(self):
        self.assertFalse(hasattr(fl, "noise_floor"),
                         "функция кодековой планки вернулась в модуль: пометки "
                         "недостаточно, заниженная планка будет использована")

    def test_a_bare_number_without_provenance_cannot_judge(self):
        got = fl.verdict({"mean": 0.03, "max": 0.4},
                         {"floor": 0.004513, "floor_max": 0.071895})
        self.assertEqual(got["outcome"], fl.UNMEASURED,
                         "планка без клейма происхождения вынесла вердикт — "
                         "ровно так кодековое число и судило весь тракт")
        self.assertIn("судить протечку не вправе", got["note"])

    def test_the_vae_floor_alone_may_not_judge_either(self):
        """VAE-пол занижен на весь вклад сэмплера — он не планка протечки."""
        vae = {"outcome": fl.PASS, "kind": fl.FLOOR_VAE,
               "floor": 0.0026, "floor_max": 0.05}
        got = fl.verdict({"mean": 0.03, "max": 0.4}, vae)
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn(fl.FLOOR_FULL_LOOP, got["note"])

    def test_the_full_loop_floor_does_judge(self):
        """Зеркало к двум предыдущим: правильное клеймо обязано ПРОПУСКАТЬ.

        Без этой половины предыдущие тесты проходили бы и на верdict'е,
        который отказывает всегда.
        """
        got = fl.verdict({"mean": 0.03, "max": 0.4}, _loop_floor())
        self.assertIn(got["outcome"], (fl.PASS, fl.FAIL))
        self.assertEqual(got["floor_kind"], fl.FLOOR_FULL_LOOP)

    def test_the_list_of_judging_kinds_is_guarded_in_both_directions(self):
        """Т1 на константе-решении `FLOOR_KINDS_THAT_MAY_JUDGE`."""
        vae = {"outcome": fl.PASS, "kind": fl.FLOOR_VAE,
               "floor": 0.0026, "floor_max": 0.05}
        original = fl.FLOOR_KINDS_THAT_MAY_JUDGE
        try:
            fl.FLOOR_KINDS_THAT_MAY_JUDGE = (fl.FLOOR_FULL_LOOP, fl.FLOOR_VAE)
            self.assertIn(fl.verdict({"mean": 0.03}, vae)["outcome"],
                          (fl.PASS, fl.FAIL),
                          "список ослаблен, а VAE-пол всё равно отвергнут — "
                          "значит вердикт смотрит не на список")
            fl.FLOOR_KINDS_THAT_MAY_JUDGE = ()
            self.assertEqual(
                fl.verdict({"mean": 0.03}, _loop_floor())["outcome"],
                fl.UNMEASURED,
                "список пуст, а вердикт всё равно вынесен")
        finally:
            fl.FLOOR_KINDS_THAT_MAY_JUDGE = original


class TheFullLoopFloorRefusesToBeAssumed(unittest.TestCase):

    def test_a_floor_declared_without_a_source_is_refused(self):
        """И4: число-решение без происхождения не принимается вовсе."""
        got = fl.declared_full_loop_floor(floor=0.01, floor_max=0.09,
                                          frames=13, measured_by="  ")
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIsNone(got["floor"])
        self.assertEqual(fl.verdict({"mean": 0.03}, got)["outcome"],
                         fl.UNMEASURED)

    def test_a_floor_declared_with_a_source_carries_it(self):
        got = fl.declared_full_loop_floor(floor=0.01, floor_max=0.09, frames=13,
                                          measured_by="прогон на A16, 2026-..-..")
        self.assertEqual(got["outcome"], fl.PASS)
        self.assertIn("A16", got["note"])

    def test_a_renderer_that_blows_up_gives_unmeasured_not_zero(self):
        def broken(frames, mask):
            raise RuntimeError("Comfy не поднялся")

        got = fl.full_loop_floor([_frame()] * 5, broken)
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIsNone(got["floor"])
        self.assertIn("пустой маской пол не подменяется", got["note"].lower())

    def test_a_renderer_returning_nothing_gives_unmeasured(self):
        got = fl.full_loop_floor([_frame()] * 5, lambda f, m: None)
        self.assertEqual(got["outcome"], fl.UNMEASURED)

    def test_the_renderer_is_handed_an_all_zero_mask(self):
        """Смысл пола (б) целиком в нулевой маске. Проверяем, что она нулевая.

        Если сюда однажды приедет настоящая маска, число всё равно посчитается
        и будет выглядеть полом — а будет полом «протечка при нашей маске»,
        то есть тем самым, что мы собирались измерять.
        """
        seen = {}

        def spy(frames, mask):
            seen["mask"] = np.asarray(mask)
            return frames

        fl.full_loop_floor([_frame()] * 5, spy)
        self.assertEqual(seen["mask"].shape, (5, 64, 64))
        self.assertEqual(float(seen["mask"].max()), 0.0,
                         "исполнителю подали НЕ нулевую маску — тогда это не "
                         "пол, а замер протечки под видом пола")

    def test_an_honest_loop_that_changes_nothing_is_a_zero_floor_and_is_refused(self):
        """Негативный контроль (И5): вход, на котором вердикт обязан сказать «нет».

        Исполнитель-тождество даёт пол 0.0. При сэмплировании из шума такого
        не бывает — значит мерили не то, и судить по этому полу нельзя.
        """
        got = fl.full_loop_floor([_frame()] * 5, lambda f, m: f)
        self.assertEqual(got["floor"], 0.0)
        self.assertEqual(fl.verdict({"mean": 0.03}, got)["outcome"],
                         fl.UNMEASURED)

    def test_a_loop_that_changes_frames_gives_a_floor_above_zero(self):
        def noisy(frames, mask):
            return [f + 0.02 for f in frames]

        got = fl.full_loop_floor([_frame()] * 5, noisy)
        self.assertEqual(got["outcome"], fl.PASS)
        self.assertAlmostEqual(got["floor"], 0.02, places=6)
        self.assertEqual(got["kind"], fl.FLOOR_FULL_LOOP)

    def test_the_frame_count_bar_is_guarded(self):
        original = fl.MIN_FLOOR_FRAMES
        try:
            fl.MIN_FLOOR_FRAMES = 99
            got = fl.full_loop_floor([_frame()] * 10, lambda f, m: f)
            self.assertEqual(got["outcome"], fl.UNMEASURED)
            self.assertIn("99", got["note"])
            fl.MIN_FLOOR_FRAMES = 2
            self.assertEqual(
                fl.full_loop_floor([_frame()] * 10, lambda f, m: f)["outcome"],
                fl.PASS, "порог ослаблен, а измерение всё равно отказано")
        finally:
            fl.MIN_FLOOR_FRAMES = original


class TheSamplerContributionIsTheDifferenceOfTwoFloors(unittest.TestCase):
    """Ради этой разности полов и два, а не один (§1a)."""

    VAE = {"outcome": "годно", "kind": fl.FLOOR_VAE,
           "floor": 0.0026, "floor_max": 0.05}

    def test_the_difference_is_reported(self):
        got = fl.sampler_contribution(self.VAE, _loop_floor(floor=0.01))
        self.assertAlmostEqual(got["delta"], 0.0074, places=6)
        self.assertEqual(got["ratio"], 3.846)

    def test_a_full_floor_far_above_the_vae_floor_is_called_out(self):
        got = fl.sampler_contribution(self.VAE, _loop_floor(floor=0.01))
        self.assertIn("нестабильна сама по себе", got["note"])

    def test_a_full_floor_close_to_the_vae_floor_is_not_called_out(self):
        """Зеркало: предупреждение обязано МОЛЧАТЬ, когда сэмплер тих."""
        got = fl.sampler_contribution(self.VAE, _loop_floor(floor=0.003))
        self.assertNotIn("нестабильна", got["note"])

    def test_the_alarm_threshold_is_guarded_in_both_directions(self):
        """Т1 на SAMPLER_FLOOR_ALARM: 3.85x — та же пара, оба исхода."""
        loud = _loop_floor(floor=0.01)
        original = fl.SAMPLER_FLOOR_ALARM
        try:
            fl.SAMPLER_FLOOR_ALARM = 10.0
            self.assertNotIn("нестабильна",
                             fl.sampler_contribution(self.VAE, loud)["note"],
                             "порог поднят выше замера, а тревога всё равно "
                             "объявлена — значит порог ни на что не влияет")
            fl.SAMPLER_FLOOR_ALARM = 1.1
            self.assertIn("нестабильна",
                          fl.sampler_contribution(
                              self.VAE, _loop_floor(floor=0.003))["note"],
                          "порог опущен ниже замера, а тревоги нет")
        finally:
            fl.SAMPLER_FLOOR_ALARM = original

    def test_a_missing_full_floor_is_unmeasured_not_zero_contribution(self):
        empty = {"outcome": fl.UNMEASURED, "kind": fl.FLOOR_FULL_LOOP,
                 "floor": None}
        got = fl.sampler_contribution(self.VAE, empty)
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIsNone(got["delta"])

    def test_the_floors_may_not_be_swapped(self):
        """Порядок важен: разность со знаком, и перепутанные полы соврут."""
        got = fl.sampler_contribution(_loop_floor(), self.VAE)
        self.assertEqual(got["outcome"], fl.UNMEASURED)


class TheVerdictRefusesToJudgeWithoutAMeasuredFloor(unittest.TestCase):
    """Планка выдумана — вердикта нет. Это не строгость, это честность."""

    def test_no_floor_means_unmeasured_even_when_divergence_is_tiny(self):
        empty = {"kind": fl.FLOOR_FULL_LOOP, "floor": None,
                 "note": "Comfy не поднялся"}
        got = fl.verdict({"mean": 0.000001}, empty)
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("сколько из него стоит реконструкция", got["note"])

    def test_no_divergence_means_unmeasured_too(self):
        got = fl.verdict({"mean": None, "note": "маска съела кадр"},
                         _loop_floor(floor=0.001))
        self.assertEqual(got["outcome"], fl.UNMEASURED)

    def test_a_zero_floor_is_refused_as_impossible(self):
        got = fl.verdict({"mean": 0.01}, _loop_floor(floor=0.0))
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("сам с собой", got["note"])

    def test_divergence_under_the_multiplier_is_the_floor_of_the_pipeline(self):
        got = fl.verdict({"mean": 0.0015}, _loop_floor(floor=0.001))
        self.assertEqual(got["outcome"], fl.PASS)
        self.assertEqual(got["ratio"], 1.5)

    def test_divergence_over_the_multiplier_is_a_leak(self):
        got = fl.verdict({"mean": 0.005}, _loop_floor(floor=0.001))
        self.assertEqual(got["outcome"], fl.FAIL)
        self.assertIn("ПРОТЕЧКА", got["note"])

    def test_the_multiplier_is_guarded_in_both_directions(self):
        """Т1 на LEAK_OVER_FLOOR."""
        f = _loop_floor(floor=0.001)
        self.assertEqual(fl.verdict({"mean": 0.0015}, f, over=1.2)["outcome"],
                         fl.FAIL, "порог ужесточён, а вердикт не изменился")
        self.assertEqual(fl.verdict({"mean": 0.005}, f, over=99.0)["outcome"],
                         fl.PASS, "порог снят, а протечка всё равно объявлена")

    def test_the_note_carries_both_numbers_and_the_ratio(self):
        got = fl.verdict({"mean": 0.005}, _loop_floor(floor=0.001))
        for piece in ("0.005", "0.001", "5.0x"):
            self.assertIn(piece, got["note"])


class TheLocalChannelIsJudgedByItsOwnFloor(unittest.TestCase):
    """Локальный пол всегда много выше среднего — это свойство тракта.

    Замер VAE-круга: среднее 0.007308, максимум 0.338698 — в 46 раз, потому
    что реконструкция врёт на краях объектов, а не равномерно. Судить максимум
    средней планкой значило бы
    объявлять протечкой каждый контур; не судить его вовсе — терять канал, ради
    которого он заведён (стёртый мяч занимает проценты площади).
    """

    FLOOR = _loop_floor(floor=0.01, floor_max=0.09)

    def test_a_spot_within_the_local_floor_is_the_pipeline(self):
        got = fl.verdict({"mean": 0.012, "max": 0.13}, self.FLOOR)
        self.assertEqual(got["outcome"], fl.PASS,
                         "локальное отличие 0.13 при локальном поле 0.09 — "
                         "это край объекта, а не стёртый мяч")

    def test_a_spot_far_over_the_local_floor_is_a_leak_even_when_the_mean_is_fine(self):
        got = fl.verdict({"mean": 0.012, "max": 0.6}, self.FLOOR)
        self.assertEqual(got["outcome"], fl.FAIL,
                         "точечная протечка прошла: среднее её не видит, а "
                         "локального порога нет")
        self.assertGreater(got["spot_ratio"], fl.LEAK_OVER_FLOOR)

    def test_judging_the_max_by_the_mean_floor_would_have_failed_the_pipeline(self):
        """Прямая проверка того, что отдельный локальный пол НУЖЕН."""
        self.assertGreater(0.13 / self.FLOOR["floor"], fl.LEAK_OVER_FLOOR)
        self.assertLess(0.13 / self.FLOOR["floor_max"], fl.LEAK_OVER_FLOOR)

    def test_a_missing_local_floor_says_so_instead_of_judging_silently(self):
        bare = fl.declared_full_loop_floor(floor=0.01, floor_max=0.0, frames=13,
                                           measured_by="фикстура")
        got = fl.verdict({"mean": 0.012, "max": 0.9}, bare)
        self.assertIsNone(got["spot_ratio"])
        self.assertIn("не судится", got["note"])
        self.assertEqual(got["outcome"], fl.PASS,
                         "без локального пола максимум обязан молчать, а не "
                         "решать")


class TheVaeRoundtripFloorIsMeasuredNotAssumed(unittest.TestCase):
    """Пол (а). Дорогой прогон, поэтому здесь — только его отказы и клеймо.

    Само число снято отдельной командой и лежит в `VAE_ROUNDTRIP_MEASURED`;
    гонять 0.5 ГБ весов внутри `unittest` нельзя (Т4: тест не ходит в сеть).
    """

    def test_too_few_frames_is_unmeasured(self):
        got = fl.vae_roundtrip_floor([_frame()])
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("сжатие по времени 4x", got["note"])

    def test_frames_of_different_sizes_are_unmeasured(self):
        got = fl.vae_roundtrip_floor([_frame(64, 64), _frame(32, 32),
                                      _frame(64, 64), _frame(64, 64)])
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIn("разного размера", got["note"])

    def test_missing_weights_are_unmeasured_and_say_so_loudly(self):
        got = fl.vae_roundtrip_floor([_frame()] * 5,
                                     vae_dir="/nope/no/such/vae")
        self.assertEqual(got["outcome"], fl.UNMEASURED)
        self.assertIsNone(got["floor"])
        self.assertIn("НЕ ноль протечки", got["note"])
        self.assertNotIn("huggingface", got["note"].lower(),
                         "загрузчик ушёл в сеть за несуществующей папкой — "
                         "такой тест краснеет от чужой аварии (Т4)")

    def test_the_measured_vae_floor_is_above_the_removed_codec_bar(self):
        """Замер, а не довод: снятая планка была НИЖЕ неустранимого пола.

        0.004513 — литерал (Т2), а не импорт: он нарочно приколочен здесь,
        чтобы при любой правке замера было видно, разошлись они или нет.
        """
        if fl.VAE_ROUNDTRIP_MEASURED is None:
            self.skipTest("VAE-круг в этой сборке не снят")
        self.assertGreater(fl.VAE_ROUNDTRIP_MEASURED["floor"], 0.004513,
                           "VAE-пол оказался ниже кодековой планки — тогда "
                           "весь довод §1a о заниженной планке неверен, и это "
                           "надо разбирать, а не молча оставлять")

    def test_the_local_vae_floor_is_far_above_the_mean_one(self):
        """Почему локальный канал судится отдельной планкой — числом."""
        if fl.VAE_ROUNDTRIP_MEASURED is None:
            self.skipTest("VAE-круг в этой сборке не снят")
        m = fl.VAE_ROUNDTRIP_MEASURED
        self.assertGreater(m["floor_max"] / m["floor"], 10.0,
                           "локальный пол сравнялся со средним — тогда вторая "
                           "планка не нужна, и её надо убрать, а не носить")

    def test_the_measured_number_carries_the_command_that_produced_it(self):
        """И4: замеренное число без условий замера — то же выдуманное число."""
        if fl.VAE_ROUNDTRIP_MEASURED is None:
            self.skipTest("VAE-круг в этой сборке не снят")
        self.assertEqual(fl.VAE_ROUNDTRIP_MEASURED["kind"], fl.FLOOR_VAE)
        self.assertGreater(fl.VAE_ROUNDTRIP_MEASURED["floor"], 0.0,
                           "VAE-круг не изменил ни пикселя — такого не бывает, "
                           "значит замер сравнивал массив сам с собой")
        self.assertIn("python3", fl.VAE_ROUNDTRIP_NOTE)


if __name__ == "__main__":
    unittest.main()
