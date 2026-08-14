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


class ACrashedSessionSpentRealMoney(unittest.TestCase):
    """Сессия, оборвавшаяся на отказе API, успела сделать платные вызовы.

    Ловим отказ, который дороже всех остальных: стоимость, посчитанная ТОЛЬКО
    по дошедшим до вердикта сессиям. Отказ API — единственный сценарий, ради
    которого стенд и написан, и ровно в нём цена занижалась.
    """

    def setUp(self):
        from ball_reel import bench

        self.b = bench

    @staticmethod
    def _ok(n, passed):
        # Одна попытка, дошедшая до видео: kontext 0.04 + wan-fast 0.04.
        return {"kind": "session", "n": n, "passed": passed, "seconds": 20.0,
                "pollen": 0.08,
                "attempts": [{"n": 1, "passed": passed,
                              "check": None if passed else "loop"}]}

    @staticmethod
    def _crash(n, pollen):
        # Оборвалась ПОСЛЕ платных вызовов: картинка и видео уже оплачены.
        return {"kind": "crash", "n": n, "passed": False, "seconds": 12.0,
                "error": "HTTPError: 429 Too Many Requests", "attempts": [],
                "calls": {"image": 1, "video": 1}, "pollen": pollen}

    def _series(self):
        # 6 начатых сессий: 4 дошли до вердикта (одна прошла), 2 оборвались.
        # Руками: 4 x 0.08 + 2 x 0.08 = 0.48 pollen. Считая только дошедшие,
        # получим 0.32 — занижение ровно в 1.5 раза.
        return [self._ok(1, False), self._ok(2, False), self._ok(3, False),
                self._ok(4, True), self._crash(5, 0.08), self._crash(6, 0.08)]

    def test_the_money_burned_by_a_crashed_session_is_still_money(self):
        s = self.b.summarise(self._series())
        self.assertAlmostEqual(s["pollen_spent"], 0.48, places=4)

    def test_the_accepted_clip_carries_the_crashes_it_took_to_get_it(self):
        # Принятый клип оплачивает ВЕСЬ прогон, включая сгоревшее на отказах:
        # 0.48, а не 0.32. Именно это число идёт в отчёт как юнит-экономика.
        s = self.b.summarise(self._series())
        self.assertAlmostEqual(s["pollen_per_accepted_clip"], 0.48, places=4)

    def test_the_crash_money_is_shown_apart_and_not_folded_into_the_total(self):
        # Отдельной строкой — иначе «потрачено 0.48» не отличить от прогона,
        # где всё дошло до вердикта, а это разные истории для нанимающего.
        s = self.b.summarise(self._series())
        self.assertAlmostEqual(s["pollen_on_crashes"], 0.16, places=4)

    def test_a_crash_is_visible_in_the_printed_report(self):
        # Сводка может быть правильной, а печать — молчать; наниматель читает
        # печать. Обе строки: сколько оборвалось и сколько на этом сгорело.
        text = self.b.render(self.b.summarise(self._series()))
        self.assertIn("оборв", text)
        self.assertIn("0.16", text)

    def test_a_crash_counts_in_the_yield_denominator(self):
        # Решение (см. комментарий в summarise): упавшая сессия — несостоявшаяся
        # доставка продукта, а не поломка измерителя. 1 из 6, не 1 из 4.
        s = self.b.summarise(self._series())
        self.assertEqual(s["sessions_denominator"], 6)
        self.assertEqual(s["sessions_passed"], 1)
        self.assertEqual(s["sessions"], 4)      # дошли до вердикта

    def test_a_crash_before_the_first_paid_call_leaves_the_denominator(self):
        # Обратная сторона того же решения: если платных вызовов не было
        # вовсе, сломался стенд, а не продукт. Такую сессию в знаменатель
        # ставить нельзя — иначе опечатка в bench обваливает yield продукта.
        s = self.b.summarise([
            self._ok(1, True),
            {"kind": "crash", "n": 2, "passed": False, "attempts": [],
             "calls": {"image": 0, "video": 0}, "pollen": 0.0,
             "error": "FileNotFoundError: face.jpg"},
        ])
        self.assertEqual(s["sessions_denominator"], 1)
        self.assertEqual(s["aborted_before_any_call"], 1)
        self.assertAlmostEqual(s["pollen_spent"], 0.08, places=4)

    def test_a_crash_of_unknown_depth_is_assumed_to_have_spent(self):
        # Старый журнал без поля calls: неизвестно, дошло ли до вызова.
        # Молчание толкуем не в свою пользу — сессия остаётся в знаменателе.
        s = self.b.summarise([
            self._ok(1, True),
            {"kind": "crash", "n": 2, "passed": False, "attempts": [],
             "pollen": 0.04, "error": "Timeout"},
        ])
        self.assertEqual(s["sessions_denominator"], 2)
        self.assertEqual(s["aborted_before_any_call"], 0)


class SpendingIsMeasuredByCallsThatHappened(unittest.TestCase):
    """Цена записи журнала считается по совершённым вызовам, а не по исходу."""

    def setUp(self):
        from ball_reel import bench

        self.b = bench

    def _dir(self, names):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, d, True)
        for name in names:
            (d / name).write_bytes(b"x")
        return d

    def test_artifacts_on_disk_are_the_witness_of_a_paid_call(self):
        # У упавшей сессии списка попыток нет — считать не по чему, кроме
        # того, что реально легло на диск.
        got = self.b.calls_made(self._dir(
            ["start_00.png", "start_01.png", "video_00.mp4"]))
        self.assertEqual(got, {"image": 2, "video": 1})

    def test_a_locally_trimmed_loop_is_not_a_second_video_call(self):
        # video_NN_loop.mp4 режется у нас на машине, денег не стоит. Считать
        # его вызовом — завышать цену видео вдвое на каждом лупе.
        got = self.b.calls_made(
            self._dir(["video_00.mp4", "video_00_loop.mp4"]))
        self.assertEqual(got["video"], 1)

    def test_a_session_directory_that_never_appeared_costs_nothing(self):
        self.assertEqual(self.b.calls_made("/nonexistent/s01"),
                         {"image": 0, "video": 0})

    def test_a_crash_after_the_image_but_before_the_video_pays_for_one(self):
        # Гейт старт-кадра падает уже после платной картинки, но до видео:
        # 0.04, а не 0.08 и не 0.
        rec = {"kind": "crash", "attempts": [],
               "calls": {"image": 1, "video": 0}}
        self.assertAlmostEqual(
            self.b.session_cost(rec, 4, "kontext", "wan-fast"), 0.04, places=4)

    def test_a_crash_after_the_video_pays_for_both(self):
        rec = {"kind": "crash", "attempts": [],
               "calls": {"image": 1, "video": 1}}
        self.assertAlmostEqual(
            self.b.session_cost(rec, 4, "kontext", "wan-fast"), 0.08, places=4)

    def test_a_crash_that_never_called_anything_costs_zero(self):
        rec = {"kind": "crash", "attempts": [],
               "calls": {"image": 0, "video": 0}}
        self.assertEqual(
            self.b.session_cost(rec, 4, "kontext", "wan-fast"), 0.0)

    def test_a_crashing_session_records_the_calls_it_managed_to_make(self):
        # Шов, за которым лежат деньги: если `_session` не запишет свидетеля,
        # чинить арифметику в сводке будет не из чего — журнал уже потерян.
        # Заглушка тратит картинку и видео, а потом падает, как падает шлюз.
        import tempfile
        import unittest.mock as mock
        from pathlib import Path

        from ball_reel import produce as produce_mod

        out = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, out, True)

        def spend_then_die(face, **kw):
            d = Path(kw["out_dir"])
            d.mkdir(parents=True, exist_ok=True)
            (d / "start_00.png").write_bytes(b"x")
            (d / "video_00.mp4").write_bytes(b"x")
            raise RuntimeError("HTTPError: 429 Too Many Requests")

        with mock.patch.object(produce_mod, "produce", spend_then_die):
            rec = self.b._session("face.jpg", 1, out, attempts=2,
                                  start_model="kontext", video_model="wan-fast")
        self.assertEqual(rec["kind"], "crash")
        self.assertEqual(rec["calls"], {"image": 1, "video": 1})
        self.assertAlmostEqual(
            self.b.session_cost(rec, 4, "kontext", "wan-fast"), 0.08, places=4)

    def test_a_finished_session_is_priced_by_its_attempts(self):
        # Две попытки: первую отсеял дешёвый экран (0.04), вторая дошла до
        # видео (0.08). Руками 0.12.
        rec = {"kind": "session", "attempts": [
            {"n": 1, "passed": False, "check": "start_identity"},
            {"n": 2, "passed": True, "check": None}]}
        self.assertAlmostEqual(
            self.b.session_cost(rec, 4, "kontext", "wan-fast"), 0.12, places=4)


class TheReportRefusesToOverclaim(unittest.TestCase):
    """Сколько бы ни было сессий, доля не печатается раньше, чем заслужена."""

    def setUp(self):
        from ball_reel import bench

        self.b = bench

    @staticmethod
    def _run(n_sessions, passed=True):
        # Числа литеральные и от порога не зависят: тест, берущий вход из
        # константы, которую сторожит, проходит при любом значении константы.
        return [{"kind": "session", "n": i, "passed": passed, "seconds": 5.0,
                 "pollen": 0.1, "attempts": [{"n": 1, "passed": passed,
                                              "check": None if passed
                                              else "loop"}]}
                for i in range(n_sessions)]

    def test_a_tiny_run_refuses_to_headline_a_percentage(self):
        # «1/1 прошло, 100%» — заголовок, который живёт своей жизнью, даже
        # когда рядом честно написан интервал 5%..100%. Подпись читают позже
        # заголовка, если читают вообще.
        text = self.b.render(self.b.summarise(self._run(1)))
        self.assertIn("1/1", text)          # счётчик остаётся
        self.assertNotIn("100%", text)      # доля — нет
        self.assertIn("НЕ отчитывается", text)

    def test_eleven_sessions_still_refuse_the_percentage(self):
        # Одиннадцать — последнее n, где интервал Уилсона (50.7 п.п.) ещё шире
        # самой доли. Литерал взят из расчёта в комментарии к порогу, а не из
        # самого порога: если порог тихо опустят, этот тест покраснеет.
        text = self.b.render(self.b.summarise(self._run(11)))
        self.assertIn("11/11", text)
        self.assertNotIn("100%", text)
        self.assertIn("НЕ отчитывается", text)

    def test_twelve_sessions_earn_the_percentage_with_its_interval(self):
        # Двенадцать — первое n, где ширина уходит ниже 50 п.п. Здесь доля
        # печатается, но только вместе с интервалом.
        text = self.b.render(self.b.summarise(self._run(12)))
        self.assertIn("12/12", text)
        self.assertIn("между", text)

    def test_the_attempts_line_refuses_on_the_same_terms(self):
        # Здесь однажды выжил мутант: страж стоял на строке сессий, а строка
        # попыток продолжала печатать «100%». Проверяем ОБЕ строки поимённо.
        text = self.b.render(self.b.summarise(self._run(3)))
        sessions_line = next(ln for ln in text.splitlines() if "сессий:" in ln)
        attempts_line = next(ln for ln in text.splitlines() if "попыток:" in ln)
        self.assertNotIn("100%", sessions_line)
        self.assertNotIn("100%", attempts_line)
        self.assertIn("НЕ отчитывается", attempts_line)

    def test_no_percentage_leaks_anywhere_in_the_report_on_a_small_run(self):
        # Строки в отчёте прибавляются (обрывы, отсев, гистограмма), и каждая
        # новая — потенциальная вторая дверь. Поэтому проверка не по строкам, а
        # по всему тексту: на малой выборке процента не должно быть НИГДЕ.
        s = self.b.summarise([
            *self._run(2),
            {"kind": "crash", "n": 3, "passed": False, "attempts": [],
             "calls": {"image": 1, "video": 1}, "pollen": 0.08,
             "error": "HTTPError: 500"},
        ])
        text = self.b.render(s)
        self.assertNotIn("%", text)     # ни одной доли, ни в одной строке
        self.assertIn("2/3", text)      # счётчики на месте

    def test_the_refusal_points_at_what_IS_informative(self):
        # Отказ обязан сказать, на что смотреть вместо доли: счётчики и
        # распределение отказов работают и на малой выборке.
        _, why = self.b.yield_is_reportable(2)
        self.assertIn("распределение отказов", why)

    def test_an_empty_run_says_so_instead_of_printing_zeros(self):
        self.assertIn("судить не о чем", self.b.render(self.b.summarise([])))

    def test_a_run_with_no_paid_call_is_not_a_zero_percent(self):
        # Все сессии упали до первого вызова: знаменатель пуст. Напечатать
        # «0/0, 0%» значило бы обвинить продукт в отказе стенда.
        s = self.b.summarise([
            {"kind": "crash", "n": i, "passed": False, "attempts": [],
             "calls": {"image": 0, "video": 0}, "pollen": 0.0,
             "error": "FileNotFoundError: face.jpg"} for i in range(3)])
        text = self.b.render(s)
        self.assertNotIn("%", text)
        self.assertIn("отказ стенда", text)


class TheYieldFloorMatchesItsOwnArithmetic(unittest.TestCase):
    """Порог обязан совпадать с расчётом, которым он обоснован.

    Ловим отказ, который уже случался в этом файле: комментарий обосновывал
    восьмёрку шириной интервала «~40 п.п. при n=8», хотя на деле там 57 п.п.
    Обоснование и число разъехались, и заметить это можно было только счётом.
    """

    def setUp(self):
        from ball_reel import bench

        self.b = bench

    def _widest(self, n):
        # Ширина интервала при самом неудачном исходе: порог должен держать
        # худший расклад, а не средний.
        return max(hi - lo for lo, hi in (self.b.wilson(k, n)
                                          for k in range(n + 1)))

    def test_the_floor_is_the_first_n_with_an_interval_under_the_share(self):
        # Критерий из комментария к порогу: ширина < 50 п.п. (интервал у́же той
        # доли, вокруг которой он построен). Ответ считается здесь, независимо
        # от значения константы, и только потом сверяется с ней.
        computed = next(n for n in range(1, 200) if self._widest(n) < 0.5)
        self.assertEqual(self.b.MIN_SESSIONS_FOR_YIELD, computed)

    def test_the_numbers_quoted_in_the_comment_are_what_wilson_gives(self):
        # Таблица в комментарии — не украшение: по ней принято решение.
        for n, pp in ((3, 73.1), (8, 57.0), (11, 50.7), (12, 49.2), (20, 40.2)):
            self.assertAlmostEqual(
                self._widest(n) * 100, pp, places=1,
                msg=f"строка таблицы n={n} разошлась с wilson")

    def test_the_default_run_is_big_enough_to_report_a_share(self):
        # Иначе стенд по умолчанию печатает отказ вместо результата, и порог
        # надо либо опускать, либо поднимать число сессий — но не молчать.
        self.assertTrue(
            self.b.yield_is_reportable(self.b.DEFAULT_SESSIONS)[0])


if __name__ == "__main__":
    unittest.main()


class UnmeasuredIsCountedNotSubtracted(unittest.TestCase):
    """«Ось не говорила» обязано быть видно отдельно от «ось сказала да».

    Веса CLIP (~600 МБ) на машине могут отсутствовать, и гейт на этом не
    падает: исход честно `not_measurable`. Но тогда фраза «одежда проверена»
    становится неправдой, а по сводке этого не видно — непроверенное
    неотличимо от прошедшего.

    Проект уже платил за ровно эту ошибку: поле `first_failing_check` свело
    два разных исхода к слову `error`, и по нему ДВАЖДЫ был сделан вывод «всё
    упало на вызовах API», хотя отказов API не было ни одного.
    """

    def setUp(self):
        from ball_reel import bench

        self.b = bench

    def _session(self, *attempts):
        return [{"kind": "session", "passed": False, "attempts": list(attempts)}]

    def test_unmeasured_attempts_are_counted(self):
        got = self.b.summarise(self._session(
            {"passed": True, "semantic": {"verdict": "matches"}},
            {"passed": True, "semantic": {"verdict": "not_measurable"}}))
        self.assertEqual(got["semantic_judged"], 2)
        self.assertEqual(got["semantic_not_measurable"], 1)

    def test_attempts_that_never_reached_the_axis_are_not_in_the_denominator(self):
        # Попытка, отсеянная раньше по цене, ось не запускала вовсе — считать
        # её «непроверенной одеждой» значило бы врать в другую сторону.
        got = self.b.summarise(self._session(
            {"passed": False, "check": "identity_median"},
            {"passed": True, "semantic": {"verdict": "matches"}}))
        self.assertEqual(got["semantic_judged"], 1)

    def test_a_run_where_the_axis_never_spoke_says_so_in_words(self):
        got = self.b.summarise(self._session({"passed": False,
                                              "check": "motion_amount"}))
        self.assertEqual(got["semantic_judged"], 0)
        self.assertIn("не запускалась", got["semantic_note"])

    def test_the_share_is_counted_not_derived_by_subtraction(self):
        # Вычитание «судила минус прошла» смешало бы непроверенное с
        # забракованным: оба не «matches», а чинятся по-разному.
        got = self.b.summarise(self._session(
            {"passed": False, "check": "semantic",
             "semantic": {"verdict": "mismatched"}},
            {"passed": True, "semantic": {"verdict": "not_measurable"}}))
        self.assertEqual(got["semantic_judged"], 2)
        self.assertEqual(got["semantic_not_measurable"], 1)
