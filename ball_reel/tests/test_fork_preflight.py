"""Предполёт: подбирает конфигурацию, а не отвергает машину.

Два требования несут файл. Первое — арифметика ступеней обязана краснеть, если
её подделать. Второе, важнее: **пропускная способность без замера не
подставляется из паспорта**. У A16 TFLOPS в даташите нет вообще, и функция,
которая всё-таки выдала бы минуты на ролик, выдала бы их из воздуха.
"""

from __future__ import annotations

import unittest

from ball_reel import fork_preflight as fp
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED

SMI = ("NVIDIA A16, 16376 MiB, 8.6, 550.90.07\n"
       "NVIDIA A16, 16376 MiB, 8.6, 550.90.07\n")


class TheSmiOutputIsParsedWithoutACard(unittest.TestCase):
    """Т5: разбор вынесен из функции, которой нужна видеокарта."""

    def test_every_row_becomes_a_gpu(self):
        got = fp.parse_smi(SMI)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["count"], 2)

    def test_mib_is_converted_to_gigabytes(self):
        got = fp.parse_smi(SMI)
        self.assertAlmostEqual(got["gpus"][0]["memory_gb"], 15.99, places=2)

    def test_ampere_reads_as_native_bf16(self):
        self.assertTrue(fp.parse_smi(SMI)["gpus"][0]["bf16_native"])

    def test_turing_reads_as_not_native_bf16(self):
        """Негативный контроль (И5): признак обязан уметь говорить «нет»."""
        got = fp.parse_smi("NVIDIA T4, 15360 MiB, 7.5, 550.90.07\n")
        self.assertFalse(got["gpus"][0]["bf16_native"])

    def test_the_capability_bar_is_guarded_in_both_directions(self):
        """Т1: подмена константы-решения строже и слабее."""
        original = fp.BF16_MIN_CAPABILITY
        try:
            fp.BF16_MIN_CAPABILITY = 7.0
            self.assertTrue(
                fp.parse_smi("T4, 15360 MiB, 7.5, 550\n")["gpus"][0]["bf16_native"])
            fp.BF16_MIN_CAPABILITY = 9.0
            self.assertFalse(fp.parse_smi(SMI)["gpus"][0]["bf16_native"])
        finally:
            fp.BF16_MIN_CAPABILITY = original

    def test_empty_output_is_unmeasured_not_zero_cards(self):
        got = fp.parse_smi("\n\n")
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_a_missing_utility_is_not_the_same_as_no_cards(self):
        """Разные состояния: одно чинится драйвером, другое — другой машиной."""
        import shutil

        original = shutil.which
        shutil.which = lambda name: None
        try:
            got = fp.gpus()
        finally:
            shutil.which = original
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("НЕ «карт нет»", got["note"])

    def test_an_unparseable_capability_does_not_crash(self):
        got = fp.parse_smi("NVIDIA A16, 16376 MiB, N/A, 550\n")
        self.assertIsNone(got["gpus"][0]["compute_cap"])
        self.assertIsNone(got["gpus"][0]["bf16_native"])


class TheBudgetPicksAStepInsteadOfRejectingTheCard(unittest.TestCase):
    """Т2: числа — литералы из ХЭНДОФ §3.1, не пересчёт формулой модуля."""

    def test_sixteen_gigabytes_at_seventy_seven_frames_chooses_q3(self):
        """Не «что влезает», а «что влезает С ЗАПАСОМ».

        Первая редакция брала Q4_K_M: 15.97 из 16.0 арифметически влезает.
        ХЭНДОФ §3.1 называет это «впритык» и берёт Q3_K_M, потому что ECC
        включён по умолчанию и забирает НЕИЗМЕРЕННУЮ часть 16 ГБ. Остаток
        0.03 ГБ против неизвестного расхода — совпадение, а не запас.
        """
        got = fp.budget(16.0, length=77)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["chosen"], "Q3_K_M")

    def test_the_step_that_only_just_fits_is_called_tight_not_fitting(self):
        got = fp.budget(16.0, length=77)
        self.assertEqual(got["tight"], ["Q4_K_M"])
        self.assertNotIn("Q4_K_M", got["fits"])
        self.assertIn("впритык", got["note"])
        self.assertIn("ECC", got["note"])

    def test_the_headroom_bar_is_guarded_in_both_directions(self):
        """Т1: сняв запас, обязаны получить Q4; задрав — ни одной ступени."""
        original = fp.MIN_HEADROOM_GB
        try:
            fp.MIN_HEADROOM_GB = 0.0
            self.assertEqual(fp.budget(16.0, length=77)["chosen"], "Q4_K_M")
            fp.MIN_HEADROOM_GB = 10.0
            self.assertIsNone(fp.budget(16.0, length=77)["chosen"])
        finally:
            fp.MIN_HEADROOM_GB = original

    def test_the_arithmetic_matches_the_handoff_to_the_hundredth(self):
        rows = {r["step"]: r for r in fp.budget(16.0, length=77)["rows"]}
        self.assertAlmostEqual(rows["Q3_K_M"]["total_gb"], 13.30, places=2)
        self.assertAlmostEqual(rows["Q4_K_M"]["total_gb"], 15.97, places=2)
        self.assertAlmostEqual(rows["Q3_K_M"]["headroom_gb"], 2.70, places=2)

    def test_forty_nine_frames_matches_the_handoff_arithmetic(self):
        rows = {r["step"]: r for r in fp.budget(16.0, length=49)["rows"]}
        self.assertAlmostEqual(rows["Q4_K_M"]["total_gb"], 15.26, places=2)

    def test_the_report_shows_the_rejected_step_too(self):
        """«А почему не Q4» должно отвечаться тем же отчётом."""
        got = fp.budget(16.0, length=77)
        self.assertIn("Q4_K_M", got["note"])
        self.assertEqual(len(got["rows"]), 2)

    def test_a_tiny_card_fails_and_names_the_levers(self):
        got = fp.budget(8.0, length=77)
        self.assertEqual(got["outcome"], FAIL)
        self.assertIsNone(got["chosen"])
        self.assertIn("ECC", got["note"])
        self.assertIn("49", got["note"])

    def test_a_big_card_takes_the_bigger_step(self):
        """Негативный контроль: подбор умеет и повышать, а не только резать."""
        self.assertEqual(fp.budget(24.0, length=77)["chosen"], "Q4_K_M")

    def test_an_unknown_length_is_unmeasured_rather_than_extrapolated(self):
        got = fp.budget(16.0, length=61)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("выдуманный расход", got["note"])

    def test_unknown_memory_is_unmeasured(self):
        self.assertEqual(fp.budget(None)["outcome"], UNMEASURED)

    def test_the_step_weights_are_guarded(self):
        """Т1: раздув вес ступени, обязаны потерять её из годных."""
        original = dict(fp.STEPS_GB)
        try:
            fp.STEPS_GB["Q3_K_M"] = 99.0
            got = fp.budget(16.0, length=77)
            self.assertNotIn("Q3_K_M", got["fits"],
                             "ступень весом 99 ГБ влезла в 16 — вес не "
                             "участвует в расчёте")
        finally:
            fp.STEPS_GB.clear()
            fp.STEPS_GB.update(original)


class ThroughputRefusesToGuessFromASpecSheet(unittest.TestCase):
    """Главный тест файла. У A16 TFLOPS в даташите нет вообще."""

    def test_without_a_measurement_it_says_unmeasured(self):
        got = fp.throughput()
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["minutes"])

    def test_it_names_why_the_spec_sheet_is_not_used(self):
        got = fp.throughput()
        self.assertIn("НЕ ЗАМЕРЕНА", got["note"])
        self.assertIn("TFLOPS в даташите нет", got["note"])

    def test_the_flops_per_clip_is_still_reported(self):
        """Что посчитать можно — считается: 1.54 PFLOPs x 6 шагов."""
        self.assertAlmostEqual(fp.throughput()["pflops_per_clip"], 9.24,
                               places=2)

    def test_with_a_measurement_it_gives_minutes(self):
        got = fp.throughput(measured_tflops=5.0)
        self.assertEqual(got["outcome"], PASS)
        self.assertAlmostEqual(got["minutes"], 30.8, places=1)

    def test_a_faster_measurement_gives_fewer_minutes(self):
        slow = fp.throughput(measured_tflops=5.0)["minutes"]
        fast = fp.throughput(measured_tflops=10.0)["minutes"]
        self.assertLess(fast, slow)
        self.assertAlmostEqual(fast * 2, slow, places=1)

    def test_a_nonsense_measurement_is_refused(self):
        for bad in (0.0, -3.0):
            with self.subTest(value=bad):
                self.assertEqual(fp.throughput(measured_tflops=bad)["outcome"],
                                 UNMEASURED)


class TheReportCountsThreeOutcomes(unittest.TestCase):

    def test_it_runs_without_a_card_and_says_what_it_could_not_do(self):
        got = fp.report()
        self.assertIn(got["outcome"], (UNMEASURED, FAIL))
        self.assertGreater(got["unmeasured"], 0)

    def test_it_says_out_loud_that_licences_are_not_checked(self):
        self.assertIn("ЛИЦЕНЗИИ НЕ ПРОВЕРЯЮТСЯ", fp.report()["note"])

    def test_the_counts_add_up_to_the_number_of_checks(self):
        got = fp.report()
        self.assertEqual(got["passed"] + got["failed"] + got["unmeasured"],
                         len(got["checks"]))

    def test_disk_is_checked_and_is_cheap(self):
        got = fp.disk()
        self.assertIn(got["outcome"], (PASS, FAIL, UNMEASURED))
        self.assertEqual(got["needed_gb"], 40)


if __name__ == "__main__":
    unittest.main()
