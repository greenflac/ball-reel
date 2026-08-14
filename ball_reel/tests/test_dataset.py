"""Датасет для LoRA: проверяется то, из-за чего обучение может выйти впустую.

Три решения несут всю ценность модуля, и каждое легко нарушить из лучших
побуждений: не замыкать круг на ArcFace, не отражать несимметричные приметы, и
не описывать в подписи то, что должно прилипнуть к триггеру.
"""

from __future__ import annotations

import unittest


class TheCircleMustNotCloseOnTheJUDGE(unittest.TestCase):
    """Главная гарантия модуля, и она проверяется кодом, а не обещанием.

    LoRA затевается ради независимости от ArcFace. Если ArcFace порождает
    датасет или отбирает его, независимости нет — она только выглядит
    полученной, что хуже, чем её отсутствие.
    """

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_arcface_in_the_generator_is_named_as_a_problem(self):
        got = self.d.independence_report(
            generator="ip-adapter-faceid", selector="facenet", judge="arcface")
        self.assertFalse(got["independent"])
        self.assertIn("ПОРОЖДЕНИИ", got["note"])

    def test_arcface_in_the_selector_is_named_too(self):
        # Тоньше первого и потому опаснее: порождение честное, а набор выбран
        # по тому, что нравится судье, и оценка завышена.
        got = self.d.independence_report(
            generator="seedream5", selector="buffalo_l", judge="arcface")
        self.assertFalse(got["independent"])
        self.assertIn("ОТБОРЕ", got["note"])

    def test_the_clean_split_is_declared_independent(self):
        got = self.d.independence_report(
            generator="seedream5", selector="facenet", judge="arcface")
        self.assertTrue(got["independent"], got["note"])
        self.assertEqual(got["problems"], [])

    def test_the_family_is_recognised_by_name_not_by_exact_match(self):
        # AuraFace и antelopev2 — та же школа, что buffalo_l. Считать их
        # независимыми значит поменять вывеску и объявить задачу решённой.
        for name in ("auraface", "antelopev2", "insightface/buffalo_s"):
            got = self.d.independence_report(
                generator="seedream5", selector=name, judge="arcface")
            self.assertFalse(got["independent"], name)


class MirrorAugmentationIsPOISONHere(unittest.TestCase):
    """Общепринятая дешёвая аугментация, ядовитая именно для нашей задачи."""

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_mirroring_is_off_by_default(self):
        names = [n for n, _ in self.d.augmentations()]
        self.assertNotIn("mirror", names)

    def test_turning_it_on_carries_the_warning_about_marks(self):
        rows = dict(self.d.augmentations(allow_mirror=True))
        self.assertIn("mirror", rows)
        self.assertIn("приметы", rows["mirror"])

    def test_every_default_augmentation_keeps_the_side(self):
        # Смысл списка: свет и кадр меняются, СТОРОНА — нет. Татуировка на
        # левом предплечье обязана остаться на левом во всех кадрах набора.
        for name, _ in self.d.augmentations():
            self.assertNotIn("mirror", name)
            self.assertNotIn("flip", name)


class CaptionsDescribeWhatVARIES(unittest.TestCase):
    """Единственное правило подписей, которое действительно важно."""

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_the_caption_carries_the_trigger_and_the_varying_axes(self):
        row = self.d.variations()[0]
        cap = self.d.caption_for(row, trigger="ohwx_person")
        self.assertTrue(cap.startswith("ohwx_person"))
        for axis in ("framing", "angle", "light", "place"):
            self.assertIn(row[axis], cap)

    def test_the_caption_never_describes_the_person(self):
        # Написав в каждой подписи «женщина с драконом на плече», мы учим
        # модель, что дракон — часть триггера, и потом не сможем его убрать.
        cap = self.d.caption_for(self.d.variations()[0], trigger="ohwx_person")
        for word in ("tattoo", "dragon", "blonde", "face", "eyes"):
            self.assertNotIn(word, cap.lower())

    def test_the_prompt_does_not_describe_the_person_either(self):
        # Тот же запрет на стороне ПОРОЖДЕНИЯ: личность приходит референсом, а
        # не словами. Описав её текстом, мы получим похожего человека вместо
        # того же самого — и не заметим этого, потому что похожий тоже красив.
        p = self.d.prompt_for(self.d.variations()[0])
        for word in ("tattoo", "blonde", "blue eyes", "freckles"):
            self.assertNotIn(word, p.lower())


class TheAxesActuallyVARY(unittest.TestCase):
    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_each_row_differs_from_the_base_in_exactly_one_axis(self):
        rows = self.d.variations()
        base = rows[0]
        for row in rows[1:]:
            differing = [k for k in base if row[k] != base[k]]
            self.assertEqual(len(differing), 1, row)

    def test_all_four_axes_are_exercised(self):
        rows = self.d.variations()
        for axis in self.d.VARIATIONS:
            self.assertGreater(len({r[axis] for r in rows}), 1, axis)

    def test_a_limit_does_not_silently_drop_an_axis_entirely(self):
        # Обрезав список до четырёх, легко остаться с одной осью и решить, что
        # разнообразие есть. Тест сторожит именно это.
        rows = self.d.variations(limit=4)
        self.assertEqual(len(rows), 4)


class SelectionHasTHREEOutcomes(unittest.TestCase):
    def setUp(self):
        from ball_reel import dataset

        self.d = dataset
        self.S = dataset.Sample

    def _batch(self, scores):
        return [self.S(path=f"g{i}.png", origin="generated", score=s)
                for i, s in enumerate(scores)]

    def test_unscored_frames_are_not_silently_dropped(self):
        # «Лицо не найдено» и «лицо непохоже» — разные вещи. Слить их значит
        # не заметить, что отбор вообще не состоялся.
        got = self.d.select(self._batch([0.2, None, 0.9]), bar=0.5, min_kept=1)
        self.assertEqual(len(got["kept"]), 1)
        self.assertEqual(len(got["dropped"]), 1)
        self.assertEqual(len(got["unscored"]), 1)
        self.assertIn("не измерено", got["note"])

    def test_the_real_photo_is_an_anchor_and_never_filtered_out(self):
        # Реальное фото — единственная опора набора. Отсеять его по метрике,
        # которая мерит его же, было бы абсурдом.
        rows = self._batch([0.9, 0.9]) + [self.S(path="real.jpg", origin="real")]
        got = self.d.select(rows, bar=0.1, min_kept=1)
        self.assertEqual(got["real"], 1)
        real = next(s for s in got["kept"] if s.origin == "real")
        # Вес якоря сверяется НЕ с константой, которую сторожит, а с весом
        # синтетики: утверждение здесь — «реальное весит БОЛЬШЕ», и оно должно
        # падать при снятии веса до единицы. Мутационный аудит показал, что
        # сравнение с константой выживает, потому что едет вместе с ней.
        self.assertGreaterEqual(real.repeats, 3,
                                "реальный якорь должен весить заметно больше "
                                "синтетики, иначе набор уплывает")
        others = [s.repeats for s in got["kept"] if s.origin != "real"]
        for r in others:
            self.assertLess(r, real.repeats)

    def test_too_small_a_set_is_declared_not_quietly_accepted(self):
        got = self.d.select(self._batch([0.1, 0.1]), bar=0.5)
        self.assertFalse(got["enough"])
        self.assertIn("переобучится", got["note"])

    def test_the_bar_bites_from_both_sides_with_literal_scores(self):
        # Входы литеральные: тест, берущий порог из константы, которую
        # сторожит, едет вместе с ней и никогда не падает.
        got = self.d.select(self._batch([0.30, 0.31]), bar=0.30, min_kept=1)
        self.assertEqual(len(got["kept"]), 1)
        self.assertEqual(len(got["dropped"]), 1)


class ThePlanIsMadeBeforeSpending(unittest.TestCase):
    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_the_overshoot_accounts_for_measured_gateway_drift(self):
        # Дрейф шлюза измерен 0.345/0.367 при баре 0.35 — отсеется около
        # половины. Запрашивать ровно столько, сколько нужно, значит получить
        # вдвое меньше и доплачивать вторым заходом.
        got = self.d.plan(target=20)
        self.assertGreater(got["generate"], 20)
        self.assertGreaterEqual(got["generate"] / 20, 2.0)

    def test_the_cost_is_stated_in_pollen_before_anything_is_spent(self):
        got = self.d.plan(target=20, price_per_image=0.035)
        self.assertAlmostEqual(got["pollen"], got["generate"] * 0.035, places=3)
        self.assertIn("pollen", got["note"])


class TheManifestExplainsWhatTheLoRALearned(unittest.TestCase):
    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_the_mix_of_real_and_synthetic_is_recorded(self):
        # Первый вопрос любого, кто будет разбираться: сколько здесь реального.
        rows = [self.d.Sample(path="real.jpg", origin="real"),
                self.d.Sample(path="g0.png", origin="generated", score=0.2),
                self.d.Sample(path="a0.png", origin="augmented", score=0.1)]
        self.d.select(rows, bar=0.5, min_kept=1)
        got = self.d.manifest(rows, trigger="ohwx", judge="arcface",
                              selector="facenet", generator="seedream5")
        self.assertEqual(got["by_origin"]["real"], 1)
        self.assertEqual(got["by_origin"]["generated"], 1)
        self.assertTrue(got["independence"]["independent"])

    def test_repeats_are_summed_so_the_real_anchor_is_visible_in_steps(self):
        rows = [self.d.Sample(path="real.jpg", origin="real"),
                self.d.Sample(path="g0.png", origin="generated", score=0.2)]
        self.d.select(rows, bar=0.5, min_kept=1)
        got = self.d.manifest(rows, trigger="ohwx", judge="arcface",
                              selector="facenet", generator="seedream5")
        self.assertEqual(got["steps_equivalent"],
                         self.d.REAL_ANCHOR_REPEATS + 1)


if __name__ == "__main__":
    unittest.main()
