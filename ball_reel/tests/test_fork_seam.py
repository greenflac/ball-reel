"""Шовная ось: порог обязан стоять ВНУТРИ разрыва, а перекрытие — быть отказом.

Два теста несут весь смысл файла:

* ось обязана КРАСНЕТЬ, когда внутри маски фактура подменена — иначе гипотеза
  о LoRA неопровержима;
* калибровка обязана ОТКАЗАТЬ на перекрывшихся облаках, а не выдать число
  «примерно посередине». Порог вне разрыва выглядит калибровкой и ею не
  является — это худший исход из трёх.
"""

from __future__ import annotations

import unittest

import numpy as np

from ball_reel import fork_seam as fs
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED


def _mask(h=200, w=200, box=(50, 50, 150, 150)):
    m = np.zeros((h, w), dtype=bool)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def _textured(h=200, w=200, amp=0.25, seed=3):
    rng = np.random.default_rng(seed)
    return np.clip(0.5 + rng.normal(0, amp, (h, w, 3)), 0, 1)


class TheAxisSeesAFakedTextureInsideTheMask(unittest.TestCase):
    """Сторож, который не умеет краснеть, — украшение."""

    def test_a_uniformly_textured_frame_reads_near_one(self):
        got = fs.seam(_textured(), _mask())
        self.assertIsNotNone(got["ratio"], got["note"])
        self.assertAlmostEqual(got["ratio"], 1.0, delta=0.25,
                               msg=f"однородная фактура дала {got['ratio']}")

    def test_smoothing_the_inside_drops_the_ratio_well_below_one(self):
        img = _textured()
        m = _mask()
        img[m] = 0.5                     # внутри — идеально гладко
        got = fs.seam(img, m)
        self.assertLess(got["ratio"], 0.5,
                        "ось не увидела, что внутри маски всё размыто — "
                        "ровно тот дефект, ради которого она написана")

    def test_over_texturing_the_inside_lifts_the_ratio_above_one(self):
        """Ось двусторонняя: дорисованная фактура — тоже шов."""
        img = _textured(amp=0.05)
        m = _mask()
        rng = np.random.default_rng(9)
        img[m] = np.clip(0.5 + rng.normal(0, 0.4, (int(m.sum()), 3)), 0, 1)
        got = fs.seam(img, m)
        self.assertGreater(got["ratio"], 1.5,
                           "перерисованная фактура не поймана")

    def test_it_measures_at_the_seam_and_not_across_the_whole_frame(self):
        """Далёкий от шва угол не должен влиять на результат.

        Иначе мерилась бы разница СЮЖЕТОВ, а не шва: фон вдалеке отличается
        от персонажа законно.
        """
        m = _mask()
        base = _textured()
        far = base.copy()
        far[0:20, 0:20] = 0.0            # угол, далёкий от границы маски
        self.assertAlmostEqual(fs.seam(base, m)["ratio"],
                               fs.seam(far, m)["ratio"], places=6)

    def test_the_ring_width_is_guarded_in_both_directions(self):
        """Т1: константа-решение подменяется строже и слабее.

        Фикстура НЕ «внутри всё гладко»: при идеально гладкой середине медиана
        градиента внутри равна нулю при любой ширине кольца, отношение выходит
        0.0 в обоих случаях, и мутация ничего не показывает. Поймано прогоном —
        первая редакция теста именно так и была написана и покраснела не по
        делу.

        Здесь фактура МЕНЯЕТСЯ С ГЛУБИНОЙ: у самой границы она сохранена, а
        ядро выглажено. Тогда узкое кольцо видит одно, широкое — другое, и
        ширина участвует в решении.
        """
        img = _textured()
        m = _mask()
        core = _mask(box=(60, 60, 140, 140))     # ядро глубже границы на 10 px
        img[core] = 0.5
        narrow = fs.seam(img, m, width=4)
        wide = fs.seam(img, m, width=40)
        self.assertNotAlmostEqual(narrow["ratio"], wide["ratio"], places=3,
                                  msg="ширина кольца ни на что не влияет")


class ThereAreThreeOutcomesNotTwo(unittest.TestCase):

    def test_a_mask_of_the_wrong_shape_is_unmeasured(self):
        got = fs.seam(_textured(), _mask(100, 100))
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("не по кадру", got["note"])

    def test_too_few_pixels_at_the_seam_is_unmeasured_not_seamless(self):
        got = fs.seam(_textured(), _mask(box=(99, 99, 101, 101)))
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("НЕ «шва нет»", got["note"])

    def test_a_flat_background_is_unmeasured_rather_than_infinite(self):
        img = np.full((200, 200, 3), 0.5)
        m = _mask()
        rng = np.random.default_rng(1)
        img[m] = np.clip(0.5 + rng.normal(0, 0.3, (int(m.sum()), 3)), 0, 1)
        got = fs.seam(img, m)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("залитый фон", got["note"])


class TheBarComesFromAMeasuredGapOrNotAtAll(unittest.TestCase):
    """Главное решение калибровки."""

    def test_separated_clouds_give_a_bar_inside_the_gap(self):
        seamless = [1.00, 0.98, 1.03, 0.99]
        seamed = [0.55, 1.60, 0.40]
        got = fs.bar_from_clouds(seamless, seamed)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertGreater(got["bar"], got["seamless_max"])
        self.assertLess(got["bar"], got["seamed_min"])
        self.assertIn("ВНУТРЬ разрыва", got["note"])

    def test_overlapping_clouds_are_refused_rather_than_split_down_the_middle(self):
        seamless = [1.00, 1.30, 0.70]
        seamed = [1.10, 1.50, 0.60]
        got = fs.bar_from_clouds(seamless, seamed)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["bar"])
        self.assertIn("ПЕРЕКРЫЛИСЬ", got["note"])
        self.assertIn("украшением", got["note"])

    def test_tiny_clouds_are_refused(self):
        got = fs.bar_from_clouds([1.0, 1.0], [2.0, 2.0])
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("По двум точкам разрыва нет", got["note"])

    def test_the_bar_is_two_sided_and_catches_both_kinds_of_seam(self):
        """Шов бывает и глаже, и детальнее. Обе стороны обязаны ловиться."""
        bar = fs.bar_from_clouds([1.0, 1.01, 0.99], [0.5, 1.5, 0.45])
        smoother = fs.verdict({"ratio": 0.5}, bar)
        rougher = fs.verdict({"ratio": 1.5}, bar)
        self.assertEqual(smoother["outcome"], FAIL)
        self.assertEqual(rougher["outcome"], FAIL)
        self.assertIn("глаже", smoother["note"])
        self.assertIn("детальнее", rougher["note"])

    def test_a_clean_frame_passes_against_the_same_bar(self):
        """Негативный контроль: планка обязана кого-то пропускать."""
        bar = fs.bar_from_clouds([1.0, 1.01, 0.99], [0.5, 1.5, 0.45])
        self.assertEqual(fs.verdict({"ratio": 1.005}, bar)["outcome"], PASS)


class TheVerdictRefusesToJudgeWithoutCalibration(unittest.TestCase):

    def test_no_bar_means_unmeasured_even_for_an_extreme_ratio(self):
        got = fs.verdict({"ratio": 0.01}, {"bar": None, "note": "облака не сняты"})
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("само по себе ничего не значит", got["note"])

    def test_no_measurement_means_unmeasured(self):
        got = fs.verdict({"ratio": None, "note": "мало пикселей"}, {"bar": 0.2})
        self.assertEqual(got["outcome"], UNMEASURED)


class TheModuleDoesNotTouchStyle(unittest.TestCase):
    """Решение владельца: `style.py` не используется ни исполнителем, ни прибором."""

    def test_style_is_not_imported_anywhere_in_the_fork(self):
        import ast
        from pathlib import Path

        pkg = Path(fs.__file__).resolve().parent
        offenders = []
        for p in sorted(pkg.glob("fork_*.py")):
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    if node.module == "style" or any(
                            a.name == "style" for a in node.names):
                        offenders.append(p.name)
                elif isinstance(node, ast.Import):
                    if any(a.name.endswith(".style") or a.name == "style"
                           for a in node.names):
                        offenders.append(p.name)
        self.assertEqual(offenders, [],
                         f"style.py вернулся в форк через {offenders} — "
                         f"решение владельца нарушено")


if __name__ == "__main__":
    unittest.main()
