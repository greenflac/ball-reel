"""Арифметика стенда — без сети и без единого потраченного pollen.

Стенд считает числа, которые пойдут нанимающему в отчёт, и потому обязан
считать их правильно там, где ошибиться проще всего: на малой выборке. Именно
малая выборка здесь и проверяется — пятнадцать сессий, а не пятнадцать тысяч.

Живые вызовы заглушены не для удобства: `_session` — единственный шов, за
которым лежат деньги, и всё остальное ниже по потоку это настоящий код.
"""

from __future__ import annotations

import unittest


class CountsBeatPercentagesOnSmallSamples(unittest.TestCase):
    """Процент на трёх наблюдениях подразумевает точность, которой нет."""

    def setUp(self):
        from ball_reel import bench

        self.b = bench

    def test_two_of_three_carries_an_interval_from_a_fifth_to_almost_all(self):
        # Ровно тот случай, ради которого интервал и считается: «годность 67%»
        # звучит как знание, а на деле истинная доля может быть и 20%, и 95%.
        lo, hi = self.b.wilson(2, 3)
        self.assertLess(lo, 0.3)
        self.assertGreater(hi, 0.9)

    def test_a_perfect_run_is_not_certainty(self):
        # Наивная формула p ± z*sqrt(p(1-p)/n) на 3 из 3 даёт интервал нулевой
        # ширины — «уверены на 100%» по трём наблюдениям. Уилсон так не делает.
        lo, hi = self.b.wilson(3, 3)
        self.assertLess(lo, 1.0)
        self.assertEqual(hi, 1.0)

    def test_zero_of_zero_is_total_ignorance_not_zero_percent(self):
        self.assertEqual(self.b.wilson(0, 0), (0.0, 1.0))

    def test_the_interval_narrows_as_evidence_grows(self):
        narrow = self.b.wilson(140, 200)
        wide = self.b.wilson(7, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])


class AttemptsAndSessionsAreCountedApart(unittest.TestCase):
    """Качество генератора и качество продукта — разные вопросы."""

    def setUp(self):
        from ball_reel import bench

        self.b = bench

    @staticmethod
    def _session(n, passed, attempts):
        return {"kind": "session", "n": n, "passed": passed, "seconds": 10.0,
                "pollen": 0.16,
                "attempts": [{"n": i + 1, "passed": p, "check": c}
                             for i, (p, c) in enumerate(attempts)]}

    def test_a_retry_that_rescues_the_session_shows_in_both_numbers(self):
        # Сессия прошла со второй попытки. По попыткам это 1 из 2, по сессиям
        # 1 из 1. Смешать — значит либо польстить себе ретраями, либо выбросить
        # то, ради чего ретраи написаны.
        s = self.b.summarise([
            self._session(1, True, [(False, "identity_median"), (True, None)])])
        self.assertEqual((s["attempts_passed"], s["attempts"]), (1, 2))
        self.assertEqual((s["sessions_passed"], s["sessions"]), (1, 1))

    def test_the_histogram_counts_only_failures_and_ranks_them(self):
        s = self.b.summarise([
            self._session(1, False, [(False, "identity_median"),
                                     (False, "loop")]),
            self._session(2, False, [(False, "identity_median"),
                                     (False, "identity_median")]),
            self._session(3, True, [(True, None)]),
        ])
        self.assertEqual(list(s["first_failing_check"]),
                         ["identity_median", "loop"])   # по убыванию
        self.assertEqual(s["first_failing_check"]["identity_median"], 3)
        self.assertNotIn(None, s["first_failing_check"])

    def test_a_crashed_session_is_counted_not_silently_dropped(self):
        # Отказ API — это факт о стенде, а не пропуск строки.
        s = self.b.summarise([
            self._session(1, True, [(True, None)]),
            {"kind": "crash", "n": 2, "error": "HTTPError: 429"},
        ])
        self.assertEqual(s["crashes"], 1)
        self.assertEqual(s["sessions"], 1)

    def test_an_error_without_a_named_check_is_bucketed_not_lost(self):
        s = self.b.summarise([self._session(1, False, [(False, None)])])
        self.assertEqual(s["first_failing_check"], {"error": 1})


class CostIsPerAcceptedClipNotPerCall(unittest.TestCase):
    """Нанимающего интересует цена годного результата, а не цена вызова."""

    def setUp(self):
        from ball_reel import bench

        self.b = bench

    def _runs(self, passes):
        return [{"kind": "session", "n": i, "passed": p, "seconds": 5.0,
                 "pollen": 0.1, "attempts": [{"n": 1, "passed": p,
                                              "check": None if p else "loop"}]}
                for i, p in enumerate(passes)]

    def test_the_brak_is_paid_for_by_the_accepted_clips(self):
        # Четыре сессии по 0.1, прошла одна: принятый клип стоит 0.4, а не 0.1.
        # Это и есть настоящая юнит-экономика, и она в четыре раза хуже
        # наивной «цены вызова».
        s = self.b.summarise(self._runs([False, False, False, True]))
        self.assertAlmostEqual(s["pollen_spent"], 0.4, places=4)
        self.assertAlmostEqual(s["pollen_per_accepted_clip"], 0.4, places=4)

    def test_with_nothing_accepted_the_price_is_none_not_zero(self):
        # Делить не на что. Ноль здесь означал бы «бесплатно», что ровно
        # наоборот: деньги потрачены, годного нет.
        s = self.b.summarise(self._runs([False, False]))
        self.assertIsNone(s["pollen_per_accepted_clip"])
        self.assertGreater(s["pollen_spent"], 0)

    def test_a_rejection_before_the_video_call_costs_only_the_image(self):
        # Главное достоинство дешёвого экрана: он бракует ДО того, как
        # потрачены основные деньги. Если считать такой попытке полную цену,
        # себестоимость завышается в разы, а достоинство становится невидимым.
        cheap = self.b.attempt_cost("start_identity", 4, "kontext", "wan-fast")
        self.assertAlmostEqual(cheap, 0.04, places=4)

    def test_a_rejection_after_the_video_call_costs_both(self):
        full = self.b.attempt_cost("loop", 4, "kontext", "wan-fast")
        self.assertAlmostEqual(full, 0.04 + 0.04, places=4)

    def test_a_passing_attempt_costs_the_full_price_too(self):
        # check=None означает «прошла», а прошедшая попытка видео вызывала.
        self.assertAlmostEqual(
            self.b.attempt_cost(None, 4, "kontext", "wan-fast"), 0.08, places=4)

    def test_the_estimate_is_an_upper_bound_and_names_its_parts(self):
        # Смета печатается ДО траты и считает, что все ретраи израсходуются.
        got = self.b.estimate(sessions=10, attempts=2, seconds=4,
                              start_model="kontext", video_model="wan-fast")
        self.assertAlmostEqual(got, (0.04 + 0.01 * 4) * 2 * 10, places=4)

    def test_the_expensive_model_costs_what_the_contract_says(self):
        # seedance дороже wan-fast в 18 раз — на этом стоит вся стратегия
        # «сначала связка на дешёвой, потом качество на дорогой».
        cheap = self.b.estimate(1, 1, 4, "kontext", "wan-fast")
        dear = self.b.estimate(1, 1, 4, "kontext", "seedance-2.0")
        self.assertGreater(dear - 0.04, (cheap - 0.04) * 17)


class TheReportRefusesToOverclaim(unittest.TestCase):
    def setUp(self):
        from ball_reel import bench

        self.b = bench

    def test_an_empty_run_says_so_instead_of_printing_zeros(self):
        self.assertIn("судить не о чем", self.b.render(self.b.summarise([])))

    def test_every_percentage_is_printed_with_its_interval(self):
        s = self.b.summarise([
            {"kind": "session", "n": 1, "passed": True, "seconds": 5.0,
             "pollen": 0.1, "attempts": [{"n": 1, "passed": True,
                                          "check": None}]}])
        text = self.b.render(s)
        self.assertIn("1/1", text)
        self.assertIn("между", text)   # интервал рядом с долей


if __name__ == "__main__":
    unittest.main()
