"""Счётчик времени судится теми же правилами, что и остальные измерители.

Тайминги — это метрика, а метрика в этом проекте обязана отвечать не только
«сколько», но и «на основании чего» и «а можно ли вообще судить». Поэтому здесь
проверяется не то, что секундомер тикает, а три решения, которые он кодирует:
холодный старт отделён, судим по хвосту, а не по середине, и этап, замеров по
которому нет, не выдаётся за нулевой.

Время не измеряется настоящими часами: тест на sleep(0.05) — это тест на
планировщик, он мигает на загруженной машине и ничего не сторожит. Замеры
подаются через `record()`, то есть проверяется арифметика и вердикт, а не то,
умеет ли ОС спать.
"""

from __future__ import annotations

import unittest


class ColdStartIsNotPartOfTheProfile(unittest.TestCase):
    """Первый вызов — это загрузка модели, а не работа этапа."""

    def setUp(self):
        from ball_reel import timing

        self.t = timing

    def test_the_first_call_goes_to_cold_and_leaves_the_quantiles_alone(self):
        # Живой масштаб: ArcFace грузится сотнями миллисекунд, а работает
        # десятками. Смешать их — получить число, которое не описывает ни
        # прогрев, ни работу.
        got = self.t.Timings()
        for ms in (820.0, 31.0, 33.0, 30.0, 35.0):
            got.record("arcface", ms, per=self.t.PER_FRAME)
        s = got.summary()["arcface"]
        self.assertEqual(s["cold_ms"], 820.0)
        self.assertEqual(s["calls"], 4)
        self.assertLess(s["p95_ms"], 100.0)

    def test_a_stage_called_once_reports_no_quantiles_rather_than_zero(self):
        # «Замеров не было» и «занимает ноль миллисекунд» — разные утверждения,
        # и второе неправда всегда.
        got = self.t.Timings()
        got.record("dwpose", 500.0, per=self.t.PER_FRAME)
        s = got.summary()["dwpose"]
        self.assertIsNone(s["p50_ms"])
        self.assertIsNone(s["p95_ms"])
        self.assertEqual(s["calls"], 0)
        self.assertEqual(s["cold_ms"], 500.0)

    def test_the_stopwatch_records_even_when_the_stage_raises(self):
        # Этап, падающий через десять секунд, стоит этих десяти секунд.
        got = self.t.Timings()
        with self.assertRaises(RuntimeError):
            with got.stage("boom", per=self.t.PER_FRAME):
                raise RuntimeError("х")
        self.assertIn("boom", got.summary())


class TheTailIsWhatGetsJudged(unittest.TestCase):
    """Бюджет, который держится в половине случаев, — не бюджет."""

    def setUp(self):
        from ball_reel import timing

        self.t = timing

    def _profile(self, name, samples, per=None):
        got = self.t.Timings()
        got.record(name, 999.0, per=per or self.t.PER_FRAME)  # холодный
        for ms in samples:
            got.record(name, ms)
        return got.summary()

    def test_a_stage_fast_in_the_median_but_slow_in_the_tail_is_refused(self):
        # Медиана 4 мс, то есть «влезаем». Но каждый десятый вызов — 40 мс,
        # вчетверо дороже бюджета, и хвост это показывает, а среднее спрятало
        # бы (среднее тут 7.6 — почти в бюджете).
        s = self._profile("spiky", [4.0] * 36 + [40.0] * 4)
        self.assertLessEqual(s["spiky"]["p50_ms"], 10.0)
        self.assertGreater(s["spiky"]["p95_ms"], 10.0)
        v = self.t.latency_verdict(s)
        self.assertNotIn("spiky", v["online"])

    def test_a_genuinely_fast_stage_passes(self):
        s = self._profile("cheap", [1.0, 1.2, 0.9, 1.1, 1.3] * 8)
        v = self.t.latency_verdict(s)
        self.assertEqual(v["online"], ["cheap"])

    def test_one_slow_call_in_twenty_is_p100_not_p95(self):
        # Граничный случай, на котором я сам ошибся, когда писал этот файл:
        # один медленный вызов из двадцати — это сотая перцентиль, а не 95-я,
        # и p95 обязан её НЕ видеть. Иначе метрика превращается в «максимум»
        # и любая случайная задержка отправляет этап в безнадёжные.
        s = self._profile("blip", [4.0] * 19 + [40.0])
        self.assertLess(s["blip"]["p95_ms"], 10.0)

    def test_quantiles_interpolate(self):
        q = self.t._quantile
        self.assertEqual(q([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertEqual(q([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertEqual(q([5.0], 0.95), 5.0)
        self.assertIsNone(q([], 0.5))


class HopelessIsDistinguishedFromFixable(unittest.TestCase):
    """Промах вдвое и промах в сто раз чинятся по-разному."""

    def setUp(self):
        from ball_reel import timing

        self.t = timing

    def _one(self, name, ms):
        got = self.t.Timings()
        got.record(name, 999.0, per=self.t.PER_FRAME)
        # Замеров с запасом над MIN_TAIL_SAMPLES: иначе вердикт справедливо
        # откажется судить, и тест проверял бы не ту корзину.
        for _ in range(40):
            got.record(name, ms)
        return got.summary()

    def test_a_near_miss_is_worth_optimising(self):
        # 30 мс при бюджете 10: разрешение, батчинг, потоки могут вытянуть.
        v = self.t.latency_verdict(self._one("pose", 30.0))
        self.assertEqual(v["optimise"], ["pose"])
        self.assertEqual(v["precompute"], [])

    def test_a_stage_off_by_orders_is_not_an_optimisation_task(self):
        # 4 секунды при бюджете 10 мс — это не «медленно», это другая
        # архитектура. Потратить день на настройки здесь — потерянный день.
        v = self.t.latency_verdict(self._one("diffusion", 4000.0))
        self.assertEqual(v["precompute"], ["diffusion"])
        self.assertEqual(v["optimise"], [])
        self.assertIn("ЗАРАНЕЕ", v["note"])

    def test_the_boundary_is_read_from_the_module_with_literal_inputs(self):
        # Порог берётся из модуля, входы литеральные: тест не поедет вместе
        # с константой, но покраснеет, если её сдвинуть или снять.
        budget, factor = self.t.REALTIME_BUDGET_MS, self.t.HOPELESS_FACTOR
        just_over = self.t.latency_verdict(self._one("a", budget * 2))
        self.assertEqual(just_over["optimise"], ["a"])
        far_over = self.t.latency_verdict(self._one("b", budget * factor * 2))
        self.assertEqual(far_over["precompute"], ["b"])
        self.assertEqual(budget, 10.0)
        self.assertGreaterEqual(factor, 2.0)


class ThinEvidenceIsRefusedNotGuessedAt(unittest.TestCase):
    """Мало замеров — это отказ судить, а не вердикт «медленно»."""

    def setUp(self):
        from ball_reel import timing

        self.t = timing

    def _profile(self, name, samples):
        got = self.t.Timings()
        got.record(name, 999.0, per=self.t.PER_FRAME)
        for ms in samples:
            got.record(name, ms)
        return got.summary()

    def test_five_samples_do_not_condemn_a_fast_stage(self):
        # На пяти замерах один случайный выброс весит 80% ответа: p95 выходит
        # 32.8 при реальной медиане 4. Без этой корзины этап уехал бы в
        # «безнадёжно» и отправил переделывать архитектуру там, где всё цело.
        s = self._profile("pose", [4.0, 4.0, 4.0, 4.0, 40.0])
        self.assertGreater(s["pose"]["p95_ms"], 10.0)   # число врёт
        v = self.t.latency_verdict(s)
        self.assertEqual(v["thin"], ["pose"])           # вердикта не выносим
        self.assertEqual(v["precompute"], [])
        self.assertEqual(v["online"], [])
        self.assertIn("СУДИТЬ РАНО", v["note"])

    def test_enough_samples_restore_the_verdict(self):
        s = self._profile("pose", [4.0] * 39 + [40.0])
        v = self.t.latency_verdict(s)
        self.assertEqual(v["online"], ["pose"])
        self.assertEqual(v["thin"], [])

    def test_the_sample_floor_is_read_from_the_module(self):
        # Порог из модуля, размеры выборок литеральные.
        floor = self.t.MIN_TAIL_SAMPLES
        self.assertGreaterEqual(floor, 20)
        below = self._profile("a", [1.0] * (floor - 1))
        self.assertEqual(self.t.latency_verdict(below)["thin"], ["a"])
        at = self._profile("b", [1.0] * floor)
        self.assertEqual(self.t.latency_verdict(at)["online"], ["b"])


class AVerdictThatFlipsBetweenRunsIsNotAVerdict(unittest.TestCase):
    """Этап на самой черте бюджета обязан признаться, а не выбрать сторону."""

    def setUp(self):
        from ball_reel import timing

        self.t = timing

    def _at(self, ms):
        got = self.t.Timings()
        got.record("draw", 99.0, per=self.t.PER_FRAME)
        for _ in range(40):
            got.record("draw", ms)
        return self.t.latency_verdict(got.summary())

    def test_both_real_measurements_of_draw_land_in_the_same_bucket(self):
        # Настоящий случай, ради которого корзина и появилась: skeleton.draw
        # дал p95 9.90 в нашем профиле и 10.49 в независимом замере на той же
        # машине. Два честных прогона — по разные стороны черты. Пока вердикт
        # был бинарным, он бы отрапортовал «влезает» и «не влезает» на одних
        # и тех же данных.
        self.assertEqual(self._at(9.90)["borderline"], ["draw"])
        self.assertEqual(self._at(10.49)["borderline"], ["draw"])
        self.assertEqual(self._at(9.90)["online"], [])
        self.assertEqual(self._at(10.49)["optimise"], [])

    def test_clearly_fast_and_clearly_slow_still_get_a_verdict(self):
        # Полоса не должна съесть решительность: то, что явно быстрее или
        # явно медленнее, обязано получить ответ.
        self.assertEqual(self._at(1.0)["online"], ["draw"])
        self.assertEqual(self._at(50.0)["optimise"], ["draw"])
        self.assertEqual(self._at(1.0)["borderline"], [])

    def test_the_band_is_read_from_the_module_with_literal_inputs(self):
        band = self.t.BORDERLINE_BAND
        self.assertGreater(band, 0.0)
        self.assertLess(band, 0.5)
        edge = self.t.REALTIME_BUDGET_MS * (1 + band) * 0.99
        self.assertEqual(self._at(edge)["borderline"], ["draw"])
        clear = self.t.REALTIME_BUDGET_MS * (1 + band) * 1.5
        self.assertEqual(self._at(clear)["borderline"], [])

    def test_the_note_tells_the_operator_what_to_do_about_it(self):
        note = self._at(10.0)["note"]
        self.assertIn("НА ГРАНИЦЕ", note)
        self.assertIn("мерить дольше", note)


class PerRunWorkIsNotJudgedByARealtimeBudget(unittest.TestCase):
    """Этап, случающийся раз за прогон, не стоит латентности запроса."""

    def setUp(self):
        from ball_reel import timing

        self.t = timing

    def test_a_once_per_run_stage_is_left_out_of_the_verdict(self):
        # Загрузка пайплайна занимает минуту и не имеет никакого отношения
        # к времени отклика: она случается до запросов. Мерить её бюджетом
        # реального времени — категориальная ошибка, и вердикт, который так
        # делает, будет вечно кричать о безнадёжности там, где всё в порядке.
        got = self.t.Timings()
        got.record("load_pipeline", 1.0, per=self.t.PER_RUN)
        for _ in range(40):
            got.record("load_pipeline", 60000.0)
        v = self.t.latency_verdict(got.summary())
        self.assertNotIn("load_pipeline", v["precompute"])
        self.assertNotIn("load_pipeline", v["online"])

    def test_with_nothing_per_item_the_verdict_says_so(self):
        got = self.t.Timings()
        got.record("only_run", 1.0, per=self.t.PER_RUN)
        got.record("only_run", 2.0)
        v = self.t.latency_verdict(got.summary())
        self.assertIn("судить не о чем", v["note"])


class TimeToFirstResponseIsADifferentMeasurement(unittest.TestCase):
    """Вехи — про накопленное время от запроса, а не про длительность этапа.

    Разделение не педантичное. Профиль этапов может выглядеть прилично, пока
    время до первого отклика ужасно: очередь, холодный старт и сеть в
    длительности этапа не видны вообще. Пользователь чувствует именно
    накопленное время от своего клика до первого пикселя.
    """

    def setUp(self):
        from ball_reel import timing

        self.t = timing

    def test_the_verdict_judges_only_the_first_rung(self):
        # Лестница прогрессивной отдачи: заглушка мгновенно, вердикт intake
        # через треть секунды, первый кейфрейм через секунды, финал через
        # минуты. Требовать 10 мс от финала бессмысленно — вердикт был бы
        # вечно красным на исправной системе.
        ladder = [("blurred_upload", 4.0), ("intake_verdict", 310.0),
                  ("first_keyframe", 9000.0), ("final_clip", 120000.0)]
        v = self.t.first_paint_verdict(ladder)
        self.assertTrue(v["ok"])
        self.assertEqual(v["first"], "blurred_upload")
        self.assertIn("first_keyframe 9000 мс", v["note"])

    def test_a_first_rung_that_needs_the_server_misses_and_says_why(self):
        # Один round-trip до бэкенда — это уже десятки миллисекунд, и никакая
        # оптимизация инференса этого не меняет. Вердикт обязан называть
        # причину, а не просто «медленно».
        v = self.t.first_paint_verdict([("server_placeholder", 85.0)])
        self.assertFalse(v["ok"])
        self.assertIn("не требует обращения к серверу", v["note"])

    def test_no_milestones_is_reported_not_passed(self):
        v = self.t.first_paint_verdict([])
        self.assertFalse(v["ok"])
        self.assertIsNone(v["first"])
        self.assertIn("не измерялось", v["note"])

    def test_milestones_are_cumulative_and_ordered(self):
        got = self.t.Timings()
        a = got.milestone("first")
        b = got.milestone("second")
        self.assertLessEqual(a, b)
        self.assertEqual([n for n, _ in got.milestones()], ["first", "second"])

    def test_a_milestone_is_not_a_stage(self):
        # Веха не должна попадать в профиль этапов: это разные величины, и
        # смешать их — значит сложить длительности с моментами времени.
        got = self.t.Timings()
        got.milestone("shown")
        self.assertEqual(got.summary(), {})


class TheProfilePrintsReadably(unittest.TestCase):
    def setUp(self):
        from ball_reel import timing

        self.t = timing

    def test_an_empty_profile_says_so_instead_of_printing_a_blank_table(self):
        self.assertIn("нет", self.t.render({}))

    def test_the_slowest_stage_comes_first(self):
        got = self.t.Timings()
        for name, ms in (("fast", 1.0), ("slow", 900.0)):
            got.record(name, 0.5, per=self.t.PER_FRAME)
            for _ in range(40):
                got.record(name, ms)
        text = self.t.render(got.summary())
        self.assertLess(text.index("slow"), text.index("fast"))


if __name__ == "__main__":
    unittest.main()
