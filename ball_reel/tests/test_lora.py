"""LoRA личности: сторожатся замысел, расчёт памяти и ПРИЁМКА.

ПРАВИЛО, КОТОРОЕ ЗДЕСЬ СОБЛЮДАЕТСЯ БУКВАЛЬНО: тест не берёт вход из константы,
которую сторожит. Сравнение с константой едет вместе с ней и мутацию не ловит —
на этом проекте на таком уже попадались, и не однажды. Поэтому все входы ниже
литеральные, а ожидания записаны числами или свойствами («ранг стал меньше
запрошенного», «без чекпоинтов дороже, чем с ними»), а не ссылками на модуль.

Карты, весов и сети не требуется ничему из перечисленного.
"""

from __future__ import annotations

import unittest


def _trials(cls, keys, values):
    return [cls(key=k, distance=v) for k, v in zip(keys, values)]


class TheCircleMustNotCloseOnTheJUDGE(unittest.TestCase):
    """Первая проверка приёмки, и она не про арифметику.

    Если ArcFace участвовал в порождении или отборе, любая цифра ниже — цифра о
    самой себе. Проверка обязана стоять ДО подсчётов и обесценивать их целиком,
    а не приписываться примечанием к красивому результату.
    """

    def setUp(self):
        from ball_reel import lora

        self.l = lora
        self.keys = [f"k{i}" for i in range(12)]

    def _pair(self, generator, selector, judge):
        base = [0.30] * 12
        cand = [0.20] * 12          # заведомый и крупный «выигрыш»
        return self.l.accept(
            _trials(self.l.Trial, self.keys, base),
            _trials(self.l.Trial, self.keys, cand),
            generator=generator, selector=selector, judge=judge)

    def test_arcface_in_the_generator_voids_the_comparison(self):
        got = self._pair("ip-adapter-faceid", "facenet", "arcface")
        self.assertEqual(got["verdict"], "НЕ ИЗМЕРЕНО")
        self.assertIn("круг замкнут", got["reason"])

    def test_arcface_in_the_selector_voids_it_too(self):
        # Тоньше и опаснее: порождение честное, а набор выбран по тому, что
        # нравится судье, и «выигрыш» получен авансом.
        got = self._pair("seedream5", "buffalo_l", "arcface")
        self.assertEqual(got["verdict"], "НЕ ИЗМЕРЕНО")

    def test_a_clean_split_lets_the_verdict_through(self):
        got = self._pair("seedream5", "facenet", "arcface")
        self.assertEqual(got["verdict"], "ЛУЧШЕ")

    def test_the_check_runs_before_the_arithmetic_not_after(self):
        # Замкнутый круг обязан съесть даже безупречные числа: 12 пар, полное
        # покрытие, огромный эффект. Если проверка стоит после подсчёта и лишь
        # дописывает примечание — этот тест покраснеет.
        got = self._pair("insightface", "facenet", "arcface")
        self.assertEqual(got["verdict"], "НЕ ИЗМЕРЕНО")
        self.assertIsNone(got["effect"])
        self.assertEqual(got["pairs"], 0)


class AcceptanceHasTHREEOutcomes(unittest.TestCase):
    """Лучше, хуже, не смогли измерить. Все входы литеральные."""

    def setUp(self):
        from ball_reel import lora

        self.l = lora
        self.clean = dict(generator="seedream5", selector="facenet",
                          judge="arcface")
        self.keys = [f"k{i}" for i in range(12)]

    def _accept(self, base, cand, **kw):
        return self.l.accept(_trials(self.l.Trial, self.keys[:len(base)], base),
                             _trials(self.l.Trial, self.keys[:len(cand)], cand),
                             **self.clean, **kw)

    def test_a_clear_improvement_is_called_better(self):
        got = self._accept([0.30] * 12, [0.25] * 12)
        self.assertEqual(got["verdict"], "ЛУЧШЕ")
        self.assertGreater(got["effect"], 0)

    def test_a_clear_regression_is_called_worse(self):
        got = self._accept([0.30] * 12, [0.35] * 12)
        self.assertEqual(got["verdict"], "ХУЖЕ")
        self.assertLess(got["effect"], 0)

    def test_a_difference_below_the_instruments_resolution_is_NOT_a_verdict(self):
        # Сдвиг на 0.001 при собственном разбросе прибора 0.04 — это не
        # «не стало лучше», а «не смогли отличить». Вход литеральный: 0.001
        # заведомо мельче любого разумного порога различимости.
        got = self._accept([0.30] * 12, [0.301] * 12)
        self.assertEqual(got["verdict"], "НЕ ИЗМЕРЕНО")
        self.assertIn("не смогли отличить", got["reason"])

    def test_a_big_mean_built_on_a_huge_spread_is_not_a_verdict_either(self):
        # Среднее +0.03 выглядит как эффект, но собрано из качелей -0.50/+0.56.
        # Порог различимости обязан подниматься за разбросом, иначе шум
        # объявляется результатом.
        base = [0.50, 0.50] * 6
        cand = [1.00, -0.06] * 6          # разности -0.50 и +0.56
        got = self._accept(base, cand)
        self.assertEqual(got["verdict"], "НЕ ИЗМЕРЕНО")
        self.assertIn("разрешающей способности", got["reason"])

    def test_too_few_pairs_is_not_a_verdict_however_clear_the_effect(self):
        # Пять пар с огромным эффектом. Средняя разница на таком числе
        # определяется одним неудачным кадром.
        got = self._accept([0.40] * 5, [0.10] * 5)
        self.assertEqual(got["verdict"], "НЕ ИЗМЕРЕНО")
        self.assertIn("пар", got["reason"])

    def test_unjudgeable_frames_lower_coverage_and_stop_the_verdict(self):
        # Гейт не смог судить большинство кадров. Это диагноз кадрировке и
        # разрешению, а не модели, и вердикт выносить не на чем.
        base = [0.30] + [None] * 11
        cand = [0.25] + [None] * 11
        got = self._accept(base, cand)
        self.assertEqual(got["verdict"], "НЕ ИЗМЕРЕНО")
        self.assertIsNotNone(got["coverage"])

    def test_the_pairing_is_by_key_so_different_prompts_never_get_compared(self):
        # Разброс между сценами больше измеряемого эффекта. Сравнив средние по
        # разным промтам, мы измерим, какие сцены попались.
        T = self.l.Trial
        base = [T(key=f"a{i}", distance=0.30) for i in range(12)]
        cand = [T(key=f"b{i}", distance=0.10) for i in range(12)]
        got = self.l.accept(base, cand, **self.clean)
        self.assertEqual(got["verdict"], "НЕ ИЗМЕРЕНО")
        self.assertIn("общей пары", got["reason"])

    def test_the_sign_count_is_reported_alongside_the_mean(self):
        # Знаковый счёт не зависит от среднего и потому ловит эффект, который
        # держится на выбросах.
        got = self._accept([0.30] * 12, [0.25] * 12)
        self.assertEqual(got["wins"], 12)
        self.assertEqual(got["losses"], 0)

    def test_a_mean_that_disagrees_with_the_sign_count_is_flagged(self):
        # Семь пар чуть хуже, пять заметно лучше. Среднее говорит «лучше», а
        # большинство пар проиграло: эффект держится не на всём наборе.
        # Вердикт при этом выносится — прятать его нельзя, — но рядом обязана
        # стоять строка, отправляющая смотреть на кадры глазами.
        got = self._accept([0.30] * 12, [0.31] * 7 + [0.10] * 5)
        self.assertEqual(got["verdict"], "ЛУЧШЕ")
        self.assertLess(got["wins"], got["losses"])
        self.assertTrue(any("глазами" in n.lower() for n in got["notes"]),
                        got["notes"])

    def test_a_single_huge_outlier_cannot_manufacture_a_verdict(self):
        # Одиннадцать пар чуть хуже, одна — невероятно лучше. Среднее уезжает
        # в плюс, но вместе с ним уезжает и разброс, поэтому порог различимости
        # поднимается выше эффекта. Проверяется здесь именно это: выброс не
        # покупает вердикт, он его отменяет.
        got = self._accept([0.30] * 12, [0.31] * 11 + [-2.0])
        self.assertEqual(got["verdict"], "НЕ ИЗМЕРЕНО")
        self.assertGreater(got["effect"], 0)


class LeakageVoidsTheComparison(unittest.TestCase):
    """Промт из обучения проверяет память, а не перенос личности."""

    def setUp(self):
        from ball_reel import lora

        self.l = lora

    def test_a_training_prompt_in_the_probe_set_stops_the_verdict(self):
        keys = [f"k{i}" for i in range(12)]
        got = self.l.accept(
            _trials(self.l.Trial, keys, [0.30] * 12),
            _trials(self.l.Trial, keys, [0.10] * 12),
            generator="seedream5", selector="facenet", judge="arcface",
            trained_keys={"k3"})
        self.assertEqual(got["verdict"], "НЕ ИЗМЕРЕНО")
        self.assertIn("утечка", got["reason"])

    def test_without_leakage_the_same_numbers_produce_a_verdict(self):
        # Тот же вход, только без пересечения — иначе тест доказывал бы лишь
        # то, что функция умеет говорить «не измерено».
        keys = [f"k{i}" for i in range(12)]
        got = self.l.accept(
            _trials(self.l.Trial, keys, [0.30] * 12),
            _trials(self.l.Trial, keys, [0.10] * 12),
            generator="seedream5", selector="facenet", judge="arcface",
            trained_keys={"совсем другой промт"})
        self.assertEqual(got["verdict"], "ЛУЧШЕ")


class ANegativeResultIsPresentedAsARESULT(unittest.TestCase):
    """Главное требование к приёмке, и оно не косметическое.

    Честно измеренный проигрыш — сдаваемый результат. Отчёт обязан показывать,
    что исход был ПРЕДСКАЗАН до обучения и что куплено вместо метрики; иначе
    отрицательный результат читается как провал и его начинают прятать.
    """

    def setUp(self):
        from ball_reel import lora

        self.l = lora
        keys = [f"k{i}" for i in range(12)]
        self.worse = lora.accept(
            _trials(lora.Trial, keys, [0.29] * 12),
            _trials(lora.Trial, keys, [0.35] * 12),
            generator="seedream5", selector="facenet", judge="arcface")

    def test_the_regression_is_still_a_verdict_not_an_error(self):
        self.assertEqual(self.worse["verdict"], "ХУЖЕ")
        self.assertIsNotNone(self.worse["effect"])
        self.assertEqual(self.worse["pairs"], 12)

    def test_the_prediction_was_registered_BEFORE_training_not_after(self):
        # Предсказание живёт в константе модуля, а не дописывается в отчёт по
        # факту. Предсказание, сочинённое после прогона, ничего не стоит.
        self.assertTrue(self.worse["matches_prediction"])
        self.assertIn("ArcFace", self.worse["predicted"])

    def test_the_report_says_what_was_bought_instead_of_the_metric(self):
        text = self.l.render_acceptance(self.worse)
        self.assertIn("ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ", text)
        self.assertIn("ПРЕДСКАЗАН", text)
        self.assertIn("независим", text)

    def test_the_report_names_what_to_do_next(self):
        # Отрицательный результат без следующего шага — это жалоба.
        text = self.l.render_acceptance(self.worse)
        self.assertIn("ДАЛЬШЕ", text)
        self.assertIn("МАТЕРИАЛА", text)

    def test_an_unexpected_WIN_is_treated_with_suspicion_not_celebration(self):
        # Выигрыш противоречит предсказанию, а значит первым делом ищут утечку.
        keys = [f"k{i}" for i in range(12)]
        won = self.l.accept(
            _trials(self.l.Trial, keys, [0.35] * 12),
            _trials(self.l.Trial, keys, [0.20] * 12),
            generator="seedream5", selector="facenet", judge="arcface")
        self.assertFalse(won["matches_prediction"])
        self.assertIn("утечку", self.l.render_acceptance(won))

    def test_cannot_measure_is_rendered_as_a_statement_about_the_INSTRUMENT(self):
        keys = [f"k{i}" for i in range(12)]
        flat = self.l.accept(
            _trials(self.l.Trial, keys, [0.30] * 12),
            _trials(self.l.Trial, keys, [0.3005] * 12),
            generator="seedream5", selector="facenet", judge="arcface")
        text = self.l.render_acceptance(flat)
        self.assertIn("НЕ ИЗМЕРЕНО", text)
        self.assertIn("ПРИБОРЕ", text)


class MemoryIsCountedBeforeLaunch(unittest.TestCase):
    """Расчёт до запуска, как `animate.plan`. И три исхода, а не два."""

    def setUp(self):
        from ball_reel import lora

        self.l = lora

    def test_an_unknown_card_yields_NEITHER_fits_nor_does_not_fit(self):
        got = self.l.memory_plan(vram_gb=None)
        self.assertIsNone(got.fits)
        self.assertIn("НЕ СМОГЛИ СКАЗАТЬ", got.render())

    def test_a_tiny_card_is_told_it_does_not_fit(self):
        self.assertFalse(self.l.memory_plan(vram_gb=1.0).fits)

    def test_a_big_card_fits(self):
        self.assertTrue(self.l.memory_plan(vram_gb=24.0).fits)

    def test_fitting_and_fitting_WITH_ROOM_are_different_statements(self):
        # Половина расчёта — оценки, помеченные НЕПРОВЕРЕНО. «Влезает впритык»
        # на оценённых числах значит «влезет, если мы угадали точно».
        snug = self.l.memory_plan(vram_gb=4.42, gradient_checkpointing=True)
        self.assertTrue(snug.fits)
        self.assertFalse(snug.roomy)
        self.assertIn("ВПРИТЫК", snug.render())

    def test_calculated_and_ESTIMATED_parts_are_reported_separately(self):
        # Смешав их, мы подадим угаданное как посчитанное — ровно та ошибка,
        # против которой в этом проекте помечают каждое число.
        got = self.l.memory_plan(vram_gb=6.0)
        self.assertTrue(got.measured)
        self.assertTrue(got.estimated)
        self.assertFalse(set(got.measured) & set(got.estimated))
        self.assertIn("НЕПРОВЕРЕНО", got.render())

    def test_the_weight_figures_agree_with_the_generation_module(self):
        # ВТОРОЙ способ узнать размер весов — источник расхождений, который на
        # этом проекте кусался трижды за день. Здесь он сторожится сличением.
        from ball_reel.animate import WEIGHTS_GB

        got = self.l.memory_plan(vram_gb=6.0, cache_latents=False)
        self.assertAlmostEqual(got.measured["unet fp16"],
                               WEIGHTS_GB["unet"], places=1)
        self.assertAlmostEqual(got.measured["text_encoder fp16"],
                               WEIGHTS_GB["text_encoder"], places=1)
        self.assertAlmostEqual(got.measured["vae fp16"],
                               WEIGHTS_GB["vae"], places=2)

    def test_gradient_checkpointing_saves_a_LOT_not_a_little(self):
        # Литеральный запас 0.5 ГБ: если множитель снят, разница схлопнется.
        with_ck = self.l.memory_plan(vram_gb=6.0, gradient_checkpointing=True)
        without = self.l.memory_plan(vram_gb=6.0, gradient_checkpointing=False)
        self.assertGreater(without.total_gb, with_ck.total_gb + 0.5)

    def test_resolution_costs_more_than_rank_which_is_the_whole_point(self):
        # Открытие расчёта: на 6 ГБ ранг не решает. Если бы решал, лестница
        # экономии начиналась бы не с того конца.
        rank_jump = (self.l.memory_plan(rank=32, vram_gb=6.0).total_gb
                     - self.l.memory_plan(rank=8, vram_gb=6.0).total_gb)
        res_jump = (self.l.memory_plan(resolution=768, vram_gb=6.0).total_gb
                    - self.l.memory_plan(resolution=512, vram_gb=6.0).total_gb)
        self.assertGreater(res_jump, rank_jump * 3)

    def test_activations_are_not_free_so_resolution_actually_costs(self):
        big = self.l.memory_plan(resolution=768, vram_gb=6.0).total_gb
        small = self.l.memory_plan(resolution=384, vram_gb=6.0).total_gb
        self.assertGreater(big, small + 0.5)

    def test_a_card_that_also_draws_the_desktop_is_charged_for_it(self):
        # Самая частая причина расхождения «по расчёту влезало» с отказом по
        # памяти на ноутбуке.
        on = self.l.memory_plan(vram_gb=6.0, drives_display=True).total_gb
        off = self.l.memory_plan(vram_gb=6.0, drives_display=False).total_gb
        self.assertGreater(on, off + 0.2)

    def test_a_positive_allowance_for_fragmentation_is_always_present(self):
        # Свободного суммарно хватает, а непрерывного куска нет — отказ при
        # живых сотнях мегабайт. Запас на это обязан существовать.
        got = self.l.memory_plan(vram_gb=6.0)
        frag = [v for k, v in got.estimated.items() if "фрагмент" in k]
        self.assertEqual(len(frag), 1)
        self.assertGreater(frag[0], 0.0)

    def test_a_positive_allowance_for_the_cuda_context_is_always_present(self):
        got = self.l.memory_plan(vram_gb=6.0)
        ctx = [v for k, v in got.estimated.items() if "CUDA" in k]
        self.assertEqual(len(ctx), 1)
        self.assertGreater(ctx[0], 0.0)

    def test_the_total_is_the_sum_of_the_parts_and_hides_nothing(self):
        got = self.l.memory_plan(vram_gb=6.0)
        self.assertAlmostEqual(got.total_gb, sum(got.parts.values()), places=2)


class TheLoRASizeMatchesTheREALWorld(unittest.TestCase):
    """Арифметика проверена наблюдаемой величиной, а не собственной верой."""

    def setUp(self):
        from ball_reel import lora

        self.l = lora

    def test_a_rank_32_lora_weighs_about_the_37_MB_that_real_ones_weigh(self):
        # Реальные ранг-32 LoRA для SD1.5 весят примерно 37 МБ. Если бы
        # коэффициенты были выдуманы, сойтись с этим числом они не могли.
        self.assertAlmostEqual(self.l.lora_file_mb(32), 37.7, delta=1.5)

    def test_the_size_grows_linearly_with_rank(self):
        self.assertAlmostEqual(self.l.lora_file_mb(16),
                               self.l.lora_file_mb(8) * 2, delta=0.2)

    def test_dropping_the_text_encoder_makes_the_file_smaller(self):
        full = self.l.trainable_params(16, train_text_encoder=True)
        unet = self.l.trainable_params(16, train_text_encoder=False)
        self.assertLess(unet, full)


class TheLadderIsAPPLIEDNotAdvised(unittest.TestCase):
    """Совет, который надо исполнять руками, исполняется не всегда и не так."""

    def setUp(self):
        from ball_reel import lora

        self.l = lora

    def test_six_gigabytes_turns_gradient_checkpointing_ON(self):
        # Это и есть заявленный режим ноутбучной 3050. Он должен получаться
        # расчётом, а не оказываться в примечании как пожелание.
        self.assertTrue(self.l.config(6.0).gradient_checkpointing)

    def test_six_gigabytes_keeps_the_native_training_resolution(self):
        # Ронять разрешение на 6 ГБ не требуется — это стоит проверять, потому
        # что лестница легко съезжает на ступень ниже нужной.
        self.assertEqual(self.l.config(6.0).resolution, 512)

    def test_a_headless_six_gigabyte_card_still_keeps_checkpointing(self):
        # Освободившиеся полгигабайта не должны отменять чекпоинты: запас на
        # ошибку оценки активаций важнее нескольких процентов скорости.
        self.assertTrue(self.l.config(6.0, drives_display=False)
                        .gradient_checkpointing)

    def test_a_roomy_card_is_not_slowed_down_for_nothing(self):
        # Обратная сторона: на 24 ГБ чекпоинты — это потерянное время даром.
        self.assertFalse(self.l.config(24.0).gradient_checkpointing)

    def test_a_chosen_mode_leaves_real_room_not_a_sliver(self):
        # Литеральные 0.5 ГБ: режим, выбранный с остатком в сотые доли, влезает
        # только при условии, что оценённые активации угаданы точно.
        mem = self.l.config(6.0).memory
        self.assertGreater(mem.headroom_gb, 0.5)

    def test_a_tiny_card_gives_up_the_text_encoder_before_the_resolution(self):
        # Порядок лестницы: разрешение бьёт по лицу, ради которого всё
        # затевается, и потому снимается последним.
        cfg = self.l.config(3.6)
        self.assertFalse(cfg.train_text_encoder)

    def test_alpha_never_drifts_away_from_rank(self):
        # Разъехавшись, эти два числа молча меняют эффективную скорость
        # обучения — самая тихая из ловушек тренера.
        for vram in (4.0, 6.0, 12.0, 24.0):
            cfg = self.l.config(vram)
            self.assertEqual(cfg.alpha, cfg.rank, vram)

    def test_an_oversized_rank_is_cut_back(self):
        # Ограничение по ПЕРЕОБУЧЕНИЮ, а не по памяти: памяти хватило бы.
        # Ожидание записано свойством, а не числом: «дали 128 — получили меньше».
        self.assertLess(self.l.config(6.0, rank=128).rank, 128)

    def test_the_learning_rate_is_orders_above_full_finetuning(self):
        # Умолчание тренера 2e-6 — это скорость ПОЛНОГО дообучения. Приехав с
        # ней, LoRA обучится в ничто, и ни одна ошибка не всплывёт.
        self.assertGreater(self.l.config(6.0).learning_rate, 1e-5)

    def test_a_silly_step_budget_is_clamped_from_both_sides(self):
        self.assertGreater(self.l.config(6.0, steps=1).max_train_steps, 100)
        self.assertLess(self.l.config(6.0, steps=999999).max_train_steps, 5000)


class TheScheduleCountsTheHEAVIESTFrame(unittest.TestCase):
    """Переобучение начинается с якоря, а не со среднего кадра."""

    def setUp(self):
        from ball_reel import lora

        self.l = lora

    def test_epochs_are_derived_from_a_step_budget_not_the_other_way(self):
        # Задав эпохи, мы получили бы длительность, зависящую от размера
        # набора, и перестали бы сравнивать прогоны между собой.
        small = self.l.schedule(20, target_steps=1200)
        big = self.l.schedule(60, target_steps=1200)
        self.assertGreater(small["epochs"], big["epochs"])
        for got in (small, big):
            self.assertLess(abs(got["steps"] - 1200), 200, got)

    def test_the_anchors_repeats_multiply_its_exposure(self):
        # Якорь с весом 5 при 60 эпохах показывается 300 раз. Считая по
        # эпохам, эту границу не увидишь никогда.
        flat = self.l.schedule(20, target_steps=1200, max_repeats=1)
        anchored = self.l.schedule(20, target_steps=1200, max_repeats=5)
        self.assertEqual(anchored["views_of_heaviest"],
                         flat["views_of_heaviest"] * 5)

    def test_an_overexposed_anchor_is_called_out(self):
        got = self.l.schedule(20, target_steps=1200, max_repeats=5)
        self.assertTrue(any("перестаёт быть набором" in n
                            for n in got["notes"]), got["notes"])

    def test_a_comfortable_set_produces_no_overexposure_warning(self):
        # Иначе тест доказывал бы только то, что функция умеет ругаться.
        got = self.l.schedule(400, target_steps=1200, max_repeats=1)
        self.assertFalse(any("перестаёт быть набором" in n
                             for n in got["notes"]), got["notes"])

    def test_gradient_checkpointing_costs_time(self):
        # Сделка «память за время» должна быть видна в оценке, иначе её цена
        # не обсуждается.
        with_ck = self.l.schedule(30, gradient_checkpointing=True)
        without = self.l.schedule(30, gradient_checkpointing=False)
        self.assertGreater(with_ck["minutes"], without["minutes"])

    def test_a_guessed_step_time_is_MARKED_as_guessed(self):
        got = self.l.schedule(30)
        self.assertFalse(got["sec_per_step_measured"])
        self.assertIn("ОЦЕНКА", got["note"])
        self.assertTrue(any("НЕПРОВЕРЕНО" in n for n in got["notes"]))

    def test_a_measured_step_time_is_used_and_marked_as_measured(self):
        got = self.l.schedule(30, sec_per_step=0.5)
        self.assertTrue(got["sec_per_step_measured"])
        self.assertNotIn("ОЦЕНКА", got["note"])
        self.assertAlmostEqual(got["sec_per_step"], 0.5, places=3)

    def test_intermediate_checkpoints_are_saved_so_ACCEPT_has_candidates(self):
        # Промежуточные веса нужны не для страховки, а для приёмки: «на каком
        # шаге стало хуже» — измеримый вопрос только если есть что мерить.
        got = self.l.schedule(30, target_steps=1200)
        self.assertGreaterEqual(got["checkpoints"], 1)


class TheDatasetLayoutKeepsTheANCHORHeavy(unittest.TestCase):
    """Тренер задаёт повторы ПАПКОЙ. Свалив всё в одну, вес якоря теряют молча."""

    def setUp(self):
        from ball_reel import dataset, lora

        self.l = lora
        self.d = dataset

    def _samples(self):
        rows = [self.d.Sample(path="real.jpg", origin="real"),
                self.d.Sample(path="g0.png", origin="generated", score=0.2),
                self.d.Sample(path="g1.png", origin="generated", score=0.2),
                self.d.Sample(path="bad.png", origin="generated", score=0.9)]
        self.d.select(rows, bar=0.5, min_kept=1)
        return rows

    def test_frames_with_different_weights_land_in_different_folders(self):
        groups = self.l.layout(self._samples())
        self.assertGreater(len(groups), 1)
        self.assertEqual(len({g["num_repeats"] for g in groups}), len(groups))

    def test_the_real_anchor_keeps_a_heavier_folder_than_the_synthetic(self):
        groups = self.l.layout(self._samples())
        real = next(g for g in groups if "real" in g["origins"])
        synth = [g for g in groups if "real" not in g["origins"]]
        self.assertTrue(synth)
        for g in synth:
            self.assertLess(g["num_repeats"], real["num_repeats"])

    def test_rejected_frames_never_reach_the_layout(self):
        paths = [p for g in self.l.layout(self._samples()) for p in g["paths"]]
        self.assertNotIn("bad.png", paths)

    def test_the_repeat_count_reaches_the_generated_config(self):
        cfg = self.l.config(6.0)
        toml = self.l.dataset_toml(self._samples(), root="/data",
                                   cfg=cfg, trigger="ohwx_person")
        for group in self.l.layout(self._samples()):
            self.assertIn(f"num_repeats = {group['num_repeats']}", toml)

    def test_the_generated_config_never_enables_MIRRORING(self):
        # Зеркало переносит приметы на другую сторону. Это не настройка, а
        # условие продукта — см. `dataset.augmentations`.
        toml = self.l.dataset_toml(self._samples(), root="/data",
                                   cfg=self.l.config(6.0), trigger="ohwx")
        self.assertIn("flip_aug = false", toml)
        self.assertNotIn("flip_aug = true", toml)

    def test_the_trigger_token_is_pinned_to_the_front_of_the_caption(self):
        # Перемешав подпись, мы приклеим к триггеру случайную ось вместо
        # человека.
        toml = self.l.dataset_toml(self._samples(), root="/data",
                                   cfg=self.l.config(6.0), trigger="ohwx")
        self.assertIn("shuffle_caption = false", toml)
        self.assertIn("keep_tokens = 1", toml)


class TheCommandMatchesTheLiveTRAINER(unittest.TestCase):
    """Имена флагов сверены с живым апстримом. Здесь сторожится их наличие."""

    def setUp(self):
        from ball_reel import lora

        self.l = lora
        self.cfg = lora.config(6.0)
        self.argv = lora.command(self.cfg, trainer_dir="/opt/sd-scripts",
                                 dataset_config="/data/ds.toml",
                                 output_dir="/out", output_name="ohwx")

    def test_the_network_module_and_rank_are_passed(self):
        self.assertIn("--network_module", self.argv)
        self.assertIn("--network_dim", self.argv)

    def test_alpha_is_ALWAYS_passed_explicitly(self):
        # Умолчание тренера — 1. При ранге 8 это делит поправку на 8, то есть
        # роняет скорость обучения в 8 раз, и обучение проходит без ошибок.
        self.assertIn("--network_alpha", self.argv)
        self.assertEqual(self.argv[self.argv.index("--network_alpha") + 1],
                         str(self.cfg.alpha))

    def test_the_learning_rate_is_passed_explicitly(self):
        # Умолчание тренера 2e-6 — скорость полного дообучения.
        self.assertIn("--learning_rate", self.argv)

    def test_the_command_is_a_LIST_so_paths_with_spaces_survive(self):
        argv = self.l.command(self.cfg, trainer_dir="/opt/sd scripts",
                              dataset_config="/my data/ds.toml",
                              output_dir="/out dir", output_name="ohwx")
        self.assertIn("/my data/ds.toml", argv)

    def test_the_low_ram_trap_is_never_set(self):
        # Флаг делает обратное ожидаемому: грузит модели в VRAM вместо RAM.
        self.assertNotIn("--lowram", self.argv)

    def test_clip_skip_stays_at_one_for_a_photographic_base(self):
        self.assertEqual(self.argv[self.argv.index("--clip_skip") + 1], "1")

    def test_the_base_model_is_the_SAME_one_generation_uses(self):
        # LoRA, обученная для одной базы и применённая к другой, проявляется
        # только тем, что «что-то не то с лицом».
        from ball_reel.animate import BASE_MODEL

        self.assertIn(BASE_MODEL, self.argv)

    def test_checkpointing_and_latent_cache_reach_the_command_line(self):
        self.assertIn("--gradient_checkpointing", self.argv)
        self.assertIn("--cache_latents", self.argv)

    def test_dropping_the_text_encoder_changes_the_command_accordingly(self):
        cfg = self.l.config(3.6)
        argv = self.l.command(cfg, trainer_dir="/opt/sd-scripts",
                              dataset_config="/d.toml", output_dir="/o",
                              output_name="x")
        self.assertIn("--network_train_unet_only", argv)
        self.assertNotIn("--text_encoder_lr", argv)


class ProbePromptsMeasureGENERALISATION(unittest.TestCase):
    """Спрашивать модель о том, на чём её учили, — это спросить генератор."""

    def setUp(self):
        from ball_reel import dataset, lora

        self.l = lora
        self.d = dataset

    def test_no_probe_scene_appears_among_the_training_axes(self):
        trained = {v.lower() for values in self.d.VARIATIONS.values()
                   for v in values}
        for row in self.l.probe_prompts("ohwx"):
            self.assertNotIn(row["key"].lower(), trained)

    def test_the_probe_carries_the_trigger(self):
        for row in self.l.probe_prompts("ohwx_person"):
            self.assertTrue(row["prompt"].startswith("ohwx_person"))

    def test_the_probe_set_is_large_enough_to_be_judged(self):
        # Меньше восьми промтов — и приёмка вернёт «не измерено» из-за числа
        # пар, сколько бы ни старалось обучение.
        self.assertGreaterEqual(len(self.l.probe_prompts("ohwx")), 8)

    def test_the_probe_never_describes_the_person_in_words(self):
        # Тот же запрет, что в `dataset`: личность приходит весами, а не
        # текстом. Описав её словами, мы измерим текст, а не LoRA.
        for row in self.l.probe_prompts("ohwx"):
            low = row["prompt"].lower()
            for word in ("tattoo", "blonde", "blue eyes", "woman", "man"):
                self.assertNotIn(word, low)


class PreflightRefusesEARLYAndNamesTheCure(unittest.TestCase):
    """Порядок проверок — по стоимости отказа, и первая из них не про железо."""

    def setUp(self):
        from ball_reel import lora

        self.l = lora

    def _manifest(self, *, generator, selector, judge, size=20):
        from ball_reel.dataset import independence_report

        return {"size": size,
                "independence": independence_report(
                    generator=generator, selector=selector, judge=judge)}

    def test_independence_is_checked_FIRST_of_all(self):
        # Если круг замкнут, обучать нечего при любом состоянии железа, и
        # узнать это надо до того, как человек полчаса ставил bitsandbytes.
        rep = self.l.preflight(self._manifest(
            generator="ip-adapter-faceid", selector="facenet", judge="arcface"))
        self.assertEqual(rep["checks"][0]["name"], "независимость")
        self.assertFalse(rep["checks"][0]["ok"])
        self.assertFalse(rep["ok"])

    def test_a_clean_manifest_passes_the_independence_check(self):
        rep = self.l.preflight(self._manifest(
            generator="seedream5", selector="facenet", judge="arcface"))
        self.assertTrue(rep["checks"][0]["ok"])

    def test_a_short_dataset_is_refused_with_the_number_named(self):
        rep = self.l.preflight(self._manifest(
            generator="seedream5", selector="facenet", judge="arcface", size=3))
        check = next(c for c in rep["checks"] if c["name"] == "размер набора")
        self.assertFalse(check["ok"])
        self.assertTrue(check["cure"])

    def test_a_missing_manifest_is_SKIPPED_not_passed(self):
        # «Проверить нечем» и «проверено, всё хорошо» — разные утверждения.
        rep = self.l.preflight(None)
        self.assertIsNone(rep["checks"][0]["ok"])

    def test_a_missing_trainer_names_the_clone_command(self):
        rep = self.l.preflight(
            self._manifest(generator="seedream5", selector="facenet",
                           judge="arcface"),
            trainer_dir="/nowhere/at/all")
        check = next(c for c in rep["checks"] if c["name"] == "тренер")
        self.assertFalse(check["ok"])
        self.assertIn("git clone", check["cure"])

    def test_every_failed_check_carries_a_cure(self):
        rep = self.l.preflight(
            self._manifest(generator="ip-adapter-faceid", selector="facenet",
                           judge="arcface", size=2),
            trainer_dir="/nowhere")
        failed = [c for c in rep["checks"] if c["ok"] is False]
        self.assertTrue(failed)
        for c in failed:
            self.assertTrue(c["cure"], c["name"])

    def test_the_dataset_floor_comes_from_the_dataset_module_not_a_copy(self):
        # Второй способ узнать одно и то же — источник расхождений, который на
        # этом проекте кусался трижды за день.
        from ball_reel.dataset import MIN_DATASET

        ok = self.l.preflight(self._manifest(
            generator="seedream5", selector="facenet", judge="arcface",
            size=MIN_DATASET))
        short = self.l.preflight(self._manifest(
            generator="seedream5", selector="facenet", judge="arcface",
            size=MIN_DATASET - 1))
        name = "размер набора"
        self.assertTrue(next(c for c in ok["checks"] if c["name"] == name)["ok"])
        self.assertFalse(
            next(c for c in short["checks"] if c["name"] == name)["ok"])


class TheTrainerIsNamedAndItsTrapsAreWrittenDown(unittest.TestCase):
    """Свой цикл обучения не пишем; чужой — называем точно."""

    def setUp(self):
        from ball_reel import lora

        self.l = lora

    def test_the_trainer_names_a_concrete_script_not_a_project(self):
        self.assertTrue(self.l.TRAINER["script"].endswith(".py"))
        self.assertIn("github.com", self.l.TRAINER["repo"])

    def test_the_licence_is_recorded_before_the_build_not_after(self):
        # Правило репозитория: лицензии сторонних весов и кода проверяются до
        # сборки. Продукт коммерческий.
        for trainer in (self.l.TRAINER, self.l.FALLBACK_TRAINER):
            self.assertTrue(trainer["license"])

    def test_a_fallback_trainer_exists(self):
        self.assertNotEqual(self.l.FALLBACK_TRAINER["repo"],
                            self.l.TRAINER["repo"])

    def test_the_silent_traps_are_written_down_with_their_consequence(self):
        # Ловушка, о которой знает только автор, — это не знание проекта.
        self.assertGreaterEqual(len(self.l.TRAINER_TRAPS), 4)
        for name, why in self.l.TRAINER_TRAPS:
            self.assertTrue(name and why)

    def test_the_reduction_ladder_puts_resolution_LAST(self):
        # Разрешение бьёт по лицу, ради которого всё затевается.
        steps = [name for name, _, _ in self.l.reduction_ladder()]
        res = [i for i, n in enumerate(steps) if "resolution" in n]
        self.assertTrue(res)
        self.assertEqual(max(res), len(steps) - 1)

    def test_the_cheapest_step_of_the_ladder_comes_first(self):
        first = self.l.reduction_ladder()[0]
        self.assertIn("бесплатно", first[1])


if __name__ == "__main__":
    unittest.main()
