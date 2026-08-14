"""Калибровочный стенд примет: проверяется АРИФМЕТИКА вердикта, не картинки.

Стенд строит контролируемые пары (одно и то же фото ± примета) и отвечает на
один вопрос: разделяет ли метрика примету и её отсутствие. Ответ у него
получился отрицательный — распределения перекрываются, — и именно поэтому его
собственная арифметика обязана быть сторожена: отрицательный результат,
посчитанный неправильно, хуже отсутствия результата.

Картинки здесь не строятся намеренно. Нужны веса DWPose и сегментации, а прогон
всего набора занимает минуты; это работа `main()`, а не набора тестов. Здесь
проверяется то, что остаётся, когда числа уже получены: как из них делается
вывод.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def _rows(pairs, channel="texture", kind="contour"):
    """Строки замера В ТОЙ ФОРМЕ, что отдаёт `measure_pair`.

    Форма взята из кода, а не придумана: первая версия этого помощника
    описывала строку как `{"measured": True, ...}`, тогда как модуль кладёт
    `{"state": "measured", ...}` и готовые отношения `ratio_<канал>`. Тест,
    построенный на выдуманной форме, проверяет выдумку.
    """
    out = []
    for n, (c, i) in enumerate(pairs):
        # `sample`/`bone`/`centre`/`span` образуют ключ окна: несколько рисунков
        # ставятся на ОДНО окно, и чистое значение у них общее. Без этих полей
        # сводка не соберётся — форму диктует код, а не удобство теста.
        other = "score" if channel == "texture" else "texture"
        row = {"state": "measured", "kind": kind, "ink_px": 400,
               "sample": f"s{n}", "bone": "l_upperarm",
               "centre": 0.5, "span": 0.3,
               f"clean_{channel}": c, f"inked_{channel}": i,
               f"ratio_{channel}": (i / c if c else 0.0),
               f"clean_{other}": 0.1, f"inked_{other}": 0.1,
               f"ratio_{other}": 1.0}
        out.append(row)
    return out


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class OverlapIsReportedNotHidden(unittest.TestCase):
    """Перекрытие распределений — это ОТВЕТ, а не неудача измерения.

    На 69 живых парах стенд показал именно его: чистая кожа доходит до 0.4428,
    примета начинается с 0.2094, полосы пересекаются на ширине 0.377. Из этого
    следует, что единого абсолютного порога не существует, и подать такой порог
    как рабочий было бы враньём.
    """

    def setUp(self):
        from ball_reel import calibrate_marks

        self.c = calibrate_marks

    def test_perfectly_separated_classes_report_no_overlap(self):
        got = self.c.best_threshold([0.10, 0.12, 0.15], [0.40, 0.45, 0.50])
        self.assertFalse(got["overlap"])
        self.assertAlmostEqual(got["balanced"], 1.0, places=3)

    def test_overlapping_classes_are_named_with_the_width(self):
        # Чистое доходит до 0.30, приметное начинается с 0.20 — полосы
        # пересекаются, и ширина пересечения обязана быть в отчёте.
        got = self.c.best_threshold([0.10, 0.20, 0.30], [0.20, 0.35, 0.50])
        self.assertTrue(got["overlap"])
        self.assertGreater(got["overlap_width"], 0.0)
        self.assertLess(got["balanced"], 1.0)

    def test_the_best_possible_threshold_is_reported_not_invented(self):
        # «Лучший возможный порог даёт N%» полезнее подобранной константы:
        # он говорит, чего от метрики в принципе можно ждать.
        got = self.c.best_threshold([0.10, 0.20, 0.30], [0.20, 0.35, 0.50])
        self.assertIn("threshold", got)
        self.assertGreater(got["balanced"], 0.5)

    def test_a_class_with_no_samples_is_not_measured_rather_than_perfect(self):
        # Пустая сторона даёт формально безупречное разделение. Это ловушка:
        # «не на чем проверить» и «разделяется идеально» — разные утверждения.
        got = self.c.best_threshold([], [0.4, 0.5])
        self.assertEqual(got["state"], "not_measured")


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class ThePairedRatioIsTheStableQuantity(unittest.TestCase):
    """Внутри пары отношение устойчиво, абсолютный уровень — нет.

    Это главный вывод стенда, и он определяет, как метрикой можно пользоваться:
    как проверкой ПЕРЕНОСА (сравниваются два измерения одной приметы) — можно,
    как проверкой ВХОДА на незнакомом фото — нельзя.
    """

    def setUp(self):
        from ball_reel import calibrate_marks

        self.c = calibrate_marks

    def test_the_ratio_counts_how_many_pairs_actually_grew(self):
        # Доля выросших пар — честнее среднего отношения: одна пара с
        # десятикратным ростом вытянула бы среднее, ничего не доказав.
        got = self.c.paired_ratios(
            _rows([(0.10, 0.20), (0.10, 0.30), (0.10, 0.05)]), "texture")
        self.assertEqual(got["grew"], 2)
        self.assertEqual(got["pairs"], 3)

    def test_the_median_ratio_is_not_dragged_by_one_outlier(self):
        got = self.c.paired_ratios(
            _rows([(0.10, 0.15), (0.10, 0.16), (0.10, 5.00)]), "texture")
        self.assertLess(got["median"], 2.0, got)

    def test_a_channel_with_no_rows_is_refused(self):
        got = self.c.paired_ratios([], "texture")
        self.assertEqual(got["state"], "not_measured")


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class EachInkTypeIsCountedSEPARATELY(unittest.TestCase):
    """Ради чего в метрике два канала, и почему их нельзя сливать.

    Живой прогон на 69 парах: у контурной приметы частотный канал даёт медиану
    2.51x, а массовый только 1.54x; у сплошной наоборот — массовый 1.97x,
    частотный 1.44x. Один усреднённый ответ спрятал бы ровно это.
    """

    def setUp(self):
        from ball_reel import calibrate_marks

        self.c = calibrate_marks

    def test_types_are_not_averaged_into_one_number(self):
        # Пар не меньше MIN_PAIRS_FOR_SEPARABILITY: ниже этого числа модуль
        # намеренно ОТКАЗЫВАЕТСЯ выносить вердикт, и брать вход из константы,
        # которую сторожишь, нельзя — поэтому здесь литеральные восемь.
        rows = (_rows([(0.10, 0.30), (0.10, 0.28), (0.10, 0.31),
                       (0.10, 0.29)], kind="contour")
                + _rows([(0.10, 0.12), (0.10, 0.13), (0.10, 0.11),
                         (0.10, 0.12)], kind="solid"))
        got = self.c.separability(rows)
        self.assertEqual(got["state"], "measured", got)
        by_kind = got["by_kind"]
        self.assertIn("contour", by_kind)
        self.assertIn("solid", by_kind)
        self.assertGreater(by_kind["contour"]["texture"]["median"],
                           by_kind["solid"]["texture"]["median"] * 1.5)

    def test_unmeasured_rows_do_not_enter_the_statistics(self):
        # «Не смогли измерить» — отдельный исход. Затесавшись в статистику, он
        # тихо сдвинул бы вывод о разделимости.
        rows = _rows([(0.10, 0.30)] * 7) + [{"state": "no_ink",
                                            "kind": "contour",
                                            "note": "чернил 0 px"}]
        got = self.c.separability(rows)
        self.assertEqual(got["pairs"], 7)
        self.assertEqual(got["not_measured"], 1)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class TheSizeHypothesisIsTestedNotAssumed(unittest.TestCase):
    """Гипотеза «чем крупнее примета, тем лучше видно» проверяется, а не верится.

    Она возникла на семи точках, где отношение падало вместе с числом пикселей
    чернил. Семь точек — не проверка, поэтому стенд варьирует размер приметы на
    ОДНОМ теле: так «мало пикселей» отделяется от «другое тело».
    """

    def setUp(self):
        from ball_reel import calibrate_marks

        self.c = calibrate_marks

    def test_a_clear_trend_is_reported_with_its_correlation(self):
        rows = [{"state": "measured", "kind": "contour", "ink_px": px,
                 "ratio_texture": r, "clean_texture": 0.10,
                 "inked_texture": 0.10 * r}
                for px, r in ((100, 1.1), (400, 1.6), (900, 2.2), (1600, 3.0))]
        got = self.c.size_hypothesis(rows, "texture")
        self.assertEqual(got["state"], "measured", got)
        self.assertGreater(got["pearson"], 0.8, got)

    def test_too_narrow_a_range_of_sizes_refuses_to_conclude(self):
        # Разброс размеров втрое — минимум, ниже которого корреляция говорит о
        # шуме, а не о зависимости. Отказ здесь честнее числа.
        rows = [{"state": "measured", "kind": "contour", "ink_px": px,
                 "ratio_texture": r, "clean_texture": 0.10,
                 "inked_texture": 0.10 * r}
                for px, r in ((400, 1.5), (420, 1.7), (450, 1.4))]
        got = self.c.size_hypothesis(rows, "texture")
        self.assertEqual(got["state"], "not_measured", got)


if __name__ == "__main__":
    unittest.main()
