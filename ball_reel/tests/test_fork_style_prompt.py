"""Гейты адаптера стиля. Каждый сторожит дефект, а не строчку кода."""

import unittest

from ball_reel import fork_style_prompt as sp
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED

LIGHT = {"colours": ["black", "beige", "camel"], "value_key": "mid",
         "saturation": "moderate",
         "texture": "clean flat surfaces with smooth, untextured colour"}
DARK = {"colours": ["maroon", "charcoal", "rust"], "value_key": "dark",
        "saturation": "saturated",
        "texture": "visible grain and tactile surface texture"}


class TheAdapterIsReproducible(unittest.TestCase):
    """Один референс — один промт. Иначе два прогона стиля несравнимы."""

    def test_the_same_card_gives_the_same_prompt(self):
        self.assertEqual(sp.compose(LIGHT)["prompt"], sp.compose(LIGHT)["prompt"])

    def test_different_cards_give_different_prompts(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ адаптера: прибор, отвечающий одним промтом на
        # любой вход, выглядит рабочим до первого сравнения стилей.
        self.assertEqual(sp.differ(LIGHT, DARK)["outcome"], PASS)

    def test_the_differ_instrument_catches_a_pair_that_does_not_differ(self):
        # Вторая сторона того же контроля: одинаковые карточки обязаны дать
        # «не годно», иначе `differ` не измеряет ничего.
        self.assertEqual(sp.differ(LIGHT, dict(LIGHT))["outcome"], FAIL)


class ThreeOutcomesNotTwo(unittest.TestCase):
    """«Карточку не прочитали» никогда не превращается в «стиля нет»."""

    def test_a_missing_field_is_unmeasured_not_failed(self):
        for field in ("colours", "value_key", "saturation", "texture"):
            with self.subTest(field=field):
                card = dict(LIGHT); card.pop(field)
                out = sp.compose(card)
                self.assertEqual(out["outcome"], UNMEASURED)
                self.assertIn(field, out["note"])

    def test_an_unknown_value_key_is_unmeasured_not_silently_defaulted(self):
        card = dict(LIGHT); card["value_key"] = "twilight"
        out = sp.compose(card)
        self.assertEqual(out["outcome"], UNMEASURED)
        self.assertIsNone(out["prompt"])

    def test_an_unknown_saturation_is_unmeasured(self):
        card = dict(LIGHT); card["saturation"] = "neon"
        self.assertEqual(sp.compose(card)["outcome"], UNMEASURED)

    def test_a_non_dict_is_unmeasured(self):
        self.assertEqual(sp.compose(None)["outcome"], UNMEASURED)

    def test_differ_is_unmeasured_when_one_side_will_not_read(self):
        self.assertEqual(sp.differ(LIGHT, {})["outcome"], UNMEASURED)


class TheProductBoundaryIsGuarded(unittest.TestCase):
    """Промт стиля описывает вид, а не персонажа: персонаж идёт из фото."""

    def test_a_subject_word_makes_the_prompt_not_good(self):
        card = dict(LIGHT); card["texture"] = "a woman with long hair"
        out = sp.compose(card)
        self.assertEqual(out["outcome"], FAIL)
        self.assertIn("woman", out["leak"])

    def test_a_clean_prompt_is_not_accused(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ запретного списка: он обязан молчать на чистом.
        self.assertEqual(sp.compose(LIGHT)["leak"], [])

    def test_a_word_that_merely_contains_a_forbidden_one_is_not_a_leak(self):
        # `bodysuit` содержит `body`, но нарушением не является: ищем по
        # границе слова, иначе список запретов начнёт ловить невиновных.
        self.assertEqual(sp.subject_leak("soft bodysuit texture"), [])
        self.assertEqual(sp.subject_leak("a body in frame"), ["body"])


class TheShapeComesFromTheCorpusNotFromTaste(unittest.TestCase):
    """Числа формы ИЗМЕРЕНЫ по 522 карточкам. Тест сторожит их значением."""

    def test_the_measured_corpus_numbers_are_the_ones_shipped(self):
        # Литералы, а не импорт из проверяемого модуля (Т2): иначе тест поедет
        # вместе с кодом и промолчит.
        self.assertEqual(sp.WORDS_TARGET, 24)
        self.assertEqual((sp.WORDS_MIN, sp.WORDS_MAX), (9, 67))
        self.assertEqual(sp.CLAUSES_TARGET, 5)
        self.assertEqual((sp.CLAUSES_MIN, sp.CLAUSES_MAX), (1, 13))
        self.assertEqual(sp.CLAUSES_MOST_COMMON, 7)

    def test_the_built_prompt_lands_inside_the_measured_band(self):
        out = sp.compose(LIGHT)
        self.assertEqual(out["outcome"], PASS)
        self.assertTrue(9 <= out["words"] <= 67, out["words"])
        # Клауз ровно столько, сколько даёт словарь: 7 — самое частое
        # значение корпуса. Точечное равенство здесь было бы подгонкой.
        self.assertTrue(1 <= out["clauses"] <= 13, out["clauses"])
        self.assertEqual(out["clauses"], sp.CLAUSES_MOST_COMMON)

    def test_a_prompt_outside_the_band_is_not_good(self):
        # Мутация полосы В СТРОГУЮ сторону: если планка ничего не сторожит,
        # этот тест не покраснеет.
        card = dict(LIGHT); card["texture"] = "grain " * 80
        self.assertEqual(sp.compose(card)["outcome"], FAIL)

    def test_too_many_clauses_is_not_good(self):
        # Мутация полосы клауз В СЛАБУЮ сторону: без верхней границы этот
        # промт прошёл бы, и «стиль» превратился бы в перечисление.
        card = dict(LIGHT); card["texture"] = ", ".join(["grain"] * 20)
        self.assertEqual(sp.compose(card)["outcome"], FAIL)

    def test_the_palette_width_is_guarded(self):
        # Мутация ширины палитры в обе стороны видна в числе слов и в тексте.
        many = dict(LIGHT)
        many["colours"] = ["black", "beige", "camel", "rust", "slate grey"]
        self.assertNotIn("rust", sp.compose(many)["prompt"])
        one = dict(LIGHT); one["colours"] = ["black"]
        self.assertIn("a palette of black", sp.compose(one)["prompt"])


class TheReaderIsAnInjectionPoint(unittest.TestCase):
    """Без внешнего пакета и без диска прибор обязан проверяться (Т4)."""

    def test_from_image_uses_the_injected_reader(self):
        out = sp.from_image("нет-такого-файла.png", reader=lambda p: LIGHT)
        self.assertEqual(out["outcome"], PASS)
        self.assertEqual(out["source"], "нет-такого-файла.png")

    def test_a_reader_that_raises_is_unmeasured_not_failed(self):
        def broken(_):
            raise OSError("файла нет")
        out = sp.from_image("x.png", reader=broken)
        self.assertEqual(out["outcome"], UNMEASURED)
        self.assertIn("OSError", out["note"])


if __name__ == "__main__":
    unittest.main()
