"""Гейты шага составителя. Числа-ожидания — ЛИТЕРАЛЫ (Т2)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ball_reel import fork_aesthetic as A
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED

DEMO = "assets/fork_plan_woman_fullbody.png"


def base_with(*aesthetics):
    return {"aesthetics": list(aesthetics)}


PLAIN = {"id": "чисто", "kind": "scene", "prompt": "a woman in a red coat"}
BRANDED = {"id": "сбрендом", "kind": "scene",
           "prompt": "a woman in a Balenciaga trench and Adidas sneakers"}


class TheOwnersBaseIsShippedWhole(unittest.TestCase):
    """База — материал владельца. Модуль её читает, а не пересказывает."""

    def test_all_six_aesthetics_are_present_by_name(self):
        self.assertEqual(sorted(A.ids()),
                         ["country", "fisheye", "icecream", "midcentury",
                          "tomatoes", "y2k"])

    def test_the_prompts_are_stored_verbatim_and_not_trimmed(self):
        # Сторож дефекта: промт, ужатый «для порядка», меняет шаблон владельца
        # молча. Длины ИЗМЕРЕНЫ по поданному тексту.
        self.assertEqual(len(A.load("y2k")["prompt"].split()), 211)
        self.assertEqual(len(A.load("fisheye")["prompt"].split()), 49)
        self.assertIn("Adidas", A.load("y2k")["prompt"])

    def test_the_two_kinds_are_marked(self):
        # `transform` сам велит анализировать поданную картинку, `scene` — нет.
        # Смешать их значило бы подать промт не тем способом.
        self.assertEqual(A.load("y2k")["kind"], "scene")
        self.assertEqual(A.load("icecream")["kind"], "transform")

    def test_an_unknown_name_is_refused_with_the_list_of_what_exists(self):
        with self.assertRaises(KeyError) as e:
            A.load("нетакой")
        self.assertIn("y2k", str(e.exception))

    def test_a_missing_base_is_refused_not_silently_empty(self):
        with TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                A.load_base(Path(td) / "нет.json")

    def test_an_empty_base_is_refused_not_treated_as_no_aesthetics(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "пусто.json"
            p.write_text(json.dumps({"aesthetics": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                A.load_base(p)


class TheBrandConflictIsReportedNotResolved(unittest.TestCase):
    """Владелец назвал бренды в своих промтах, а проект их запрещает."""

    def test_a_branded_prompt_is_the_THIRD_outcome_not_a_defect(self):
        # Ни «годно» (запрет нарушен), ни «не годно» (промт рабочий).
        got = A.brand_conflict(BRANDED)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual(sorted(got["brands"]), ["Adidas", "Balenciaga"])

    def test_a_clean_prompt_is_NOT_accused(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ: прибор, кричащий всегда, не значит ничего.
        got = A.brand_conflict(PLAIN)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["brands"], [])

    def test_the_shipped_base_really_carries_the_conflict(self):
        self.assertEqual(A.brand_conflict(A.load("y2k"))["brands"], ["Adidas"])
        self.assertEqual(A.brand_conflict(A.load("fisheye"))["brands"],
                         ["Balenciaga"])
        self.assertEqual(A.brand_conflict(A.load("country"))["brands"], [])

    def test_the_prompt_itself_is_never_edited_by_the_check(self):
        before = A.load("y2k")["prompt"]
        A.brand_conflict(A.load("y2k"))
        self.assertEqual(A.load("y2k")["prompt"], before)


class TheIdentityClauseResolvesTheConflictExplicitly(unittest.TestCase):
    """Промты описывают ЧУЖУЮ внешность; личность обязана прийти с картинки."""

    def test_the_owner_prompt_comes_first_and_whole(self):
        got = A.compose(PLAIN)
        self.assertEqual(got["outcome"], PASS)
        self.assertTrue(got["prompt"].startswith("a woman in a red coat"),
                        got["prompt"][:80])

    def test_the_identity_clause_is_in_and_names_what_wins(self):
        got = A.compose(PLAIN)["prompt"]
        self.assertIn("the input image wins on identity", got)
        self.assertIn("same face", got)
        self.assertIn("same hair colour", got)

    def test_hairstyling_and_wardrobe_are_left_to_the_prompt(self):
        # Раздел проведён по ИЗМЕРИМОСТИ: лицо и цвет волос судит ArcFace,
        # причёску и одежду не судит никто, значит их решает составитель.
        got = A.compose(PLAIN)["prompt"]
        self.assertIn("hairstyling", got)
        self.assertIn("wardrobe", got)

    def test_removing_the_identity_clause_is_visible_in_the_prompt(self):
        with mock.patch.object(A, "IDENTITY_CLAUSE", ""):
            self.assertNotIn("wins on identity", A.compose(PLAIN)["prompt"])

    def test_the_lettering_ban_comes_from_the_stand_not_a_local_copy(self):
        # Е1: одно решение — одно место.
        from ball_reel import fork_e2e

        with mock.patch.object(fork_e2e, "NO_BRANDS_CLAUSE", "ЗАПРЕТ-ПОДМЕНА"):
            self.assertIn("ЗАПРЕТ-ПОДМЕНА", A.compose(PLAIN)["prompt"])

    def test_turning_the_ban_off_is_LOUD_in_the_note(self):
        # Отключение гейта обязано быть видно, а не тихо (Ц7).
        got = A.compose(PLAIN, with_ban=False)
        self.assertNotIn("no logos", got["prompt"])
        self.assertIn("ОТКЛЮЧЁН", got["note"])

    def test_an_aesthetic_without_a_prompt_is_UNMEASURED_not_failed(self):
        for bad in ({"id": "пусто"}, None, "нетакой-как-строка-не-в-базе"):
            with self.subTest(bad=bad):
                if isinstance(bad, str):
                    with self.assertRaises(KeyError):
                        A.compose(bad)
                else:
                    self.assertEqual(A.compose(bad)["outcome"], UNMEASURED)

    def test_composing_by_name_reads_the_shipped_base(self):
        got = A.compose("country")
        self.assertEqual(got["outcome"], PASS)
        self.assertIn("Scottish landscape", got["prompt"])


class TheOnlyMeasurableAxisIsTheDemoIdentity(unittest.TestCase):
    """Осталась ли на эстетике НАША демо-личность. Лестница одна на проект."""

    @staticmethod
    def _at(median, outcome=PASS):
        def distances(frames, anchor, **kw):
            return {"outcome": outcome, "median": median, "inside": 1,
                    "judged": 1, "note": "подставной прибор"}
        return distances

    def test_the_demo_survived_is_plainly_good(self):
        got = A.accept(made="э.png", demo=DEMO, distances=self._at(0.0652))
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["median"], 0.0652)

    def test_the_middle_band_is_UNMEASURED_not_failed(self):
        # Та же средняя полоса, что везде: прибор не судья, судит глаз.
        got = A.accept(made="э.png", demo=DEMO, distances=self._at(0.5))
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("НЕ СУДЬЯ", got["note"])

    def test_a_repainted_person_is_a_real_defect(self):
        # Выше ступени «другой человек»: промт нарисовал не нашу демо-личность,
        # и «эстетика с нашей демо личностью» не получилась.
        got = A.accept(made="э.png", demo=DEMO, distances=self._at(0.9))
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("ПЕРЕРИСОВАЛ", got["note"])

    def test_the_bar_is_the_project_one_and_not_a_copy(self):
        from ball_reel import fork_identity

        self.assertEqual(fork_identity.SAME_PERSON_MAX, 0.35)
        self.assertIs(A.SAME_PERSON_MAX, fork_identity.SAME_PERSON_MAX)

    def test_mutating_the_bar_both_ways_turns_the_verdict(self):
        """Т1: планка строже и слабее на ИЗМЕРЕННОМ значении 0.2753."""
        was = A.SAME_PERSON_MAX
        try:
            A.SAME_PERSON_MAX = 0.1
            self.assertEqual(
                A.accept(made="э.png", demo=DEMO,
                         distances=self._at(0.2753))["outcome"], UNMEASURED)
            A.SAME_PERSON_MAX = 0.5
            self.assertEqual(
                A.accept(made="э.png", demo=DEMO,
                         distances=self._at(0.2753))["outcome"], PASS)
        finally:
            A.SAME_PERSON_MAX = was
        self.assertEqual(A.SAME_PERSON_MAX, 0.35)

    def test_an_instrument_that_fell_is_UNMEASURED_not_failed(self):
        def broken(*a, **k):
            raise RuntimeError("модель не загрузилась")

        got = A.accept(made="э.png", demo=DEMO, distances=broken)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("RuntimeError", got["note"])

    def test_the_verdict_says_out_loud_that_taste_is_not_measured(self):
        # Сторож честности: модуль обязан признавать, что «попало ли в
        # эстетику» он не судит. Прибор, молчащий об этом, читается как судья.
        got = A.accept(made="э.png", demo=DEMO, distances=self._at(0.0652))
        self.assertIn("СУДИТ СОСТАВИТЕЛЬ", got["note"])

    def test_the_plan_is_explicitly_NOT_required_here(self):
        got = A.accept(made="э.png", demo=DEMO, distances=self._at(0.0652))
        self.assertIn("НЕ ТРЕБУЕТСЯ", got["note"])


if __name__ == "__main__":
    unittest.main()
