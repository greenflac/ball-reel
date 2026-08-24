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
from pathlib import Path
from unittest import mock

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


def _almost_flat_outside(h=200, w=200):
    """Снаружи фактура НЕНУЛЕВАЯ, но ниже одного кода яркости; внутри — обычная.

    Именно этот случай ловил старый гвард `d_out == 0` и не поймал на настоящем
    кадре: ноль был не строгим нулём, а 0.000022.
    """
    # Пологий пандус, а не чередование: у чередования период 2, и центральная
    # разность даёт РОВНО ноль — первая редакция фикстуры так и была написана и
    # проверяла не то (поймано прогоном). Наклон 1.5e-5 на пиксель — ниже
    # кванта 0.00196, то есть «плоско до предела представления».
    img = np.full((h, w, 3), 0.5)
    img += np.arange(w)[None, :, None] * 1.5e-5
    m = _mask(h, w)
    rng = np.random.default_rng(7)
    img[m] = np.clip(0.5 + rng.normal(0, 0.25, (int(m.sum()), 3)), 0, 1)
    return img


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

    def test_a_near_flat_background_is_unmeasured_rather_than_a_huge_ratio(self):
        """Поймано ГЛАЗАМИ на настоящем кадре, а не рассуждением.

        chain_frames/0049.png: внешнее кольцо легло на выбитую в белое штору,
        d_out = 0.000022, и ось выдала 27.1818 на кадре, где шва нет по
        построению. Строгий ноль такое не ловит — нужен пол.
        """
        img = _almost_flat_outside()
        got = fs.seam(img, _mask())
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("пола", got["note"])

    def test_the_detail_floor_is_guarded_in_both_directions(self):
        """Т1: константа подменяется строже и слабее, оба раза видно."""
        near_flat, plain = _almost_flat_outside(), _textured()
        m = _mask()
        with mock.patch.object(fs, "DETAIL_FLOOR", 0.0):    # слабее
            loosened = fs.seam(near_flat, m)
        with mock.patch.object(fs, "DETAIL_FLOOR", 1.0):    # строже
            tightened = fs.seam(plain, m)
        self.assertIsNotNone(loosened["ratio"],
                             "ослабленный пол ничего не пропустил — значит не он решает")
        self.assertGreater(loosened["ratio"], 10.0)
        self.assertEqual(tightened["outcome"], UNMEASURED,
                         "ужесточённый пол не отверг обычный кадр — "
                         "константу никто не сторожит")
        self.assertIsNotNone(fs.seam(plain, m)["ratio"],
                             "при настоящем поле обычный кадр обязан меряться")

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
        self.assertGreater(got["bar"].value, got["seamless_max"])
        self.assertLess(got["bar"].value, got["seamed_min"])
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
        real = fs.bar_from_clouds([1.0, 1.01, 0.99], [0.5, 1.5, 0.45])
        got = fs.verdict({"ratio": None, "note": "мало пикселей"}, real)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("ось не измерена", got["note"])


class TheBarCannotBeSetAroundTheCalibration(unittest.TestCase):
    """Правило «порог только внутрь разрыва» обязан держать код, а не привычка.

    До этой правки `verdict` принимал любой словарь, и `{"bar": 0.2}`, набранный
    руками, судил кадры наравне с калиброванным — то есть главное решение модуля
    держалось соглашением.
    """

    def test_a_hand_written_bar_dict_is_refused(self):
        got = fs.verdict({"ratio": 0.5}, {"bar": 0.2})
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("В ОБХОД", got["note"])

    def test_a_bare_number_is_not_a_bar_either(self):
        self.assertEqual(fs.verdict({"ratio": 0.5}, 0.2)["outcome"], UNMEASURED)

    def test_the_bar_class_refuses_to_be_built_by_hand(self):
        with self.assertRaises(TypeError) as e:
            fs.Bar("не тот ключ", 0.2, 0.1, 0.05, 0.3, "из головы")
        self.assertIn("bar_from_clouds", str(e.exception))

    def test_a_minted_bar_does_judge(self):
        """Негативный контроль И5: сторож обязан кого-то и пропускать."""
        bar = fs.bar_from_clouds([1.0, 1.01, 0.99], [0.5, 1.5, 0.45])["bar"]
        self.assertIsInstance(bar, fs.Bar)
        self.assertEqual(fs.verdict({"ratio": 0.5}, bar)["outcome"], FAIL)
        self.assertEqual(fs.verdict({"ratio": 1.0}, bar)["outcome"], PASS)


class TheCloudCountsRefusalsSeparately(unittest.TestCase):
    """Р2: «облако из N значений» без числа отказов — ноль нарушений при нуле проверок."""

    def test_measured_and_unmeasured_are_two_numbers(self):
        img, m = _textured(), _mask()
        flat = np.full((200, 200, 3), 0.5)      # снаружи пусто -> отказ
        rng = np.random.default_rng(1)
        flat[m] = np.clip(0.5 + rng.normal(0, 0.3, (int(m.sum()), 3)), 0, 1)
        got = fs.cloud([(img, m), (flat, m), (img, _mask(100, 100))])
        self.assertEqual(got["measured"], 1)
        self.assertEqual(got["unmeasured"], 2)
        self.assertEqual(len(got["ratios"]), 1)


class TheAxisOnRealRepositoryFrames(unittest.TestCase):
    """Облако «бесшовных» снимается на настоящих кадрах, а не на синтетике.

    Числа — литералы (Т2): импортируй ожидаемое из прибора, и оно поедет вместе
    с ним. Получены прогоном `python3 -m ball_reel.fork_seam` 2026-08-17.
    """

    FRAME = Path(fs.__file__).resolve().parents[1] / "demo/kit/driving/0000.jpg"

    def test_a_real_frame_reproduces_the_measured_ratio(self):
        self.assertTrue(self.FRAME.exists(), f"кадр пропал: {self.FRAME}")
        g = fs._gray(self.FRAME)
        m = fs.arbitrary_masks(*g.shape)["ellipse"]
        self.assertAlmostEqual(fs.seam(self.FRAME, m)["ratio"], 1.1509, places=4)

    def test_the_seamless_cloud_on_one_real_frame_is_wide_not_tight(self):
        """Отрицательный результат И6: на бесшовном кадре ось далека от 1.0.

        Три произвольные маски на одном настоящем кадре дают отношения, которые
        расходятся больше, чем хотелось бы от «шума». Это измеренная граница
        применимости, и она записана числом, а не забыта.
        """
        g = fs._gray(self.FRAME)
        got = fs.seamless_cloud([self.FRAME])
        self.assertEqual(got["measured"], 3, got["note"])
        self.assertGreater(got["off_max"], 0.4,
                           "разброс вдруг стал маленьким — либо кадр подменён, "
                           "либо ось перестала мерить положение маски")
        self.assertEqual(g.shape, (1278, 720))


class TheModuleReadsTheRatioTheWayParagraph1aRequires(unittest.TestCase):
    """§1a: снаружи маски НЕ копия драйвинга, а такая же реконструкция.

    Проверено самостоятельно 2026-08-17 по исходнику ноды: у `WanAnimateToVideo`
    (строка 1113) нет `noise_mask` — единственный в файле принадлежит
    `Wan22ImageToVideoLatent` (строка 1414), — и слова `composite` в файле нет.
    Значит отношение 1.0 означает «две области реконструированы одинаково», а
    НЕ «шва нет, потому что снаружи оригинал». Тест сторожит именно смысл: он
    краснеет, если прежняя формулировка вернётся не перечёркнутой.
    """

    def test_the_old_copy_paste_reading_survives_only_struck_through(self):
        import re

        alive = re.sub(r"~~.*?~~", "", fs.__doc__, flags=re.S)
        for dead in ("скопирован", "копия драйвинга"):
            self.assertNotIn(dead, alive,
                             f"«{dead}» вернулось в живой текст: §1a снял это "
                             f"утверждение, перечёркивать, а не стирать")
        self.assertIn("реконструкц", alive)
        self.assertIn("~~", fs.__doc__, "старое стёрли вместо того чтобы зачеркнуть")

    def test_the_verdict_note_does_not_promise_anything_about_the_background(self):
        bar = fs.bar_from_clouds([1.0, 1.01, 0.99], [0.5, 1.5, 0.45])
        note = fs.verdict({"ratio": 1.0}, bar)["note"]
        self.assertIn("шва по этой оси не видно", note)
        for overclaim in ("фон", "оригинал", "цел"):
            self.assertNotIn(overclaim, note,
                             "вердикт обещает больше, чем меряет: целость фона "
                             "мерит ось протечки, а не эта")


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
