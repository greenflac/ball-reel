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


class TheBrandListIsASpravkaNotAGate(unittest.TestCase):
    """ПЕРЕПИСАН 22.08: владелец решил — «бренды пусть остаются». Конфликта
    больше нет, и прибор разжалован из гейта в справку.

    Гейт, докладывающий о решённом как о нерешённом, — ложная тревога, и она
    обесценивает настоящие.
    """

    def test_a_branded_prompt_is_no_longer_a_third_outcome(self):
        got = A.brand_conflict(BRANDED)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(sorted(got["brands"]), ["Adidas", "Balenciaga"])
        self.assertIn("РАЗРЕШЕНО", got["note"])

    def test_the_note_says_out_loud_that_the_mark_is_judged_by_eye(self):
        # Прибора, читающего надпись НА КАРТИНКЕ, у нас нет, и молчать об
        # этом значило бы выдать справку за проверку.
        self.assertIn("СУДИТ ГЛАЗ", A.brand_conflict(BRANDED)["note"])

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

    def test_the_owner_prompt_comes_first(self):
        """ПЕРЕПИСАН под решение «антропометрию вырезаем»: материал владельца
        по-прежнему идёт ПЕРВЫМ, но пол из него теперь уносится, поэтому
        дословного совпадения быть не может. Сторожим порядок, а не буквы."""
        got = A.compose(PLAIN)
        self.assertEqual(got["outcome"], PASS)
        self.assertTrue(got["prompt"].startswith("A person in a red coat"),
                        got["prompt"][:80])
        # Приписки идут ПОСЛЕ, а не до: ведущие токены весят больше.
        self.assertLess(got["prompt"].index("red coat"),
                        got["prompt"].index("wins on identity"))

    def test_the_owner_prompt_survives_whole_when_the_cut_is_off(self):
        got = A.compose(PLAIN, cut_body=False)
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


class TheAnthropometryIsCutOutAndTheCutIsReadable(unittest.TestCase):
    """Решение владельца 22.08: «антропометрию мы всю вырезаем».

    Резак — прибор, и он обязан проходить оба контроля: резать виновное и
    МОЛЧАТЬ на невиновном. Первая редакция резала по голому слову и унесла
    три невиновных оборота; эти тесты сторожат именно тот дефект.
    """

    def test_a_body_clause_is_carried_away_whole(self):
        got = A.strip_anthropometry(A.load("y2k")["prompt"])
        cut = [d["clause"] for d in got["dropped"]]
        self.assertIn("she has warm tanned skin with visible freckles", cut)
        self.assertIn("green eyes", cut)
        self.assertNotIn("freckles", got["prompt"])
        self.assertNotIn("green eyes", got["prompt"])

    def test_the_POSE_survives_even_though_it_names_a_body_part(self):
        # СТОРОЖ ИЗМЕРЕННОГО ДЕФЕКТА: этот оборот — сердце эстетики y2k, и он
        # был унесён первой редакцией из-за слова «lips».
        got = A.strip_anthropometry(A.load("y2k")["prompt"])
        self.assertIn("lip gloss applicator", got["prompt"])
        self.assertIn("holding the gloss bottle", got["prompt"])

    def test_the_RENDER_QUALITY_clauses_survive_the_word_skin(self):
        # Те же два невиновных: «skin» здесь про текстуру, а не про человека.
        got = A.strip_anthropometry(A.load("y2k")["prompt"])["prompt"]
        self.assertIn("natural skin texture", got)
        self.assertIn("textures of fabric skin and accessories", got)

    def test_the_SCENE_survives_when_only_an_adjective_was_anthropometric(self):
        # «extremely beautiful woman seated in a minimal armchair placed in a
        # vast Scottish landscape» — оборот несёт ВСЮ сцену. Уносится слово.
        got = A.strip_anthropometry(A.load("country")["prompt"])["prompt"]
        self.assertNotIn("beautiful", got)
        self.assertIn("Scottish landscape", got)
        self.assertIn("grazing cows", got)
        self.assertIn("minimal armchair", got)

    def test_hair_COLOUR_goes_but_hair_STYLING_stays(self):
        # Цвет волос ArcFace судит, укладку не судит никто: укладка — стайлинг
        # и принадлежит шаблону владельца.
        got = A.strip_anthropometry(A.load("y2k")["prompt"])["prompt"]
        self.assertNotIn("brunette", got)
        self.assertIn("messy bun", got)
        self.assertIn("soft bangs", got)

    def test_the_gender_is_neutralised_because_the_client_may_be_anyone(self):
        got = A.strip_anthropometry(A.load("y2k")["prompt"])["prompt"]
        self.assertNotIn(" woman ", got)
        self.assertNotIn(" she ", got)
        self.assertIn("the person leaning forward", got)

    def test_ethnicity_goes_too(self):
        got = A.strip_anthropometry(A.load("fisheye")["prompt"])["prompt"]
        self.assertNotIn("Slavic", got)
        self.assertNotIn("platinum", got)
        self.assertIn("14mm lens", got)
        self.assertIn("lavender and golden hues", got)

    def test_a_clean_prompt_comes_back_BYTE_FOR_BYTE(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ резака, и он обязателен: резак, что-то меняющий
        # всегда, неотличим от резака, который режет наугад. ИЗМЕРЕНО: два
        # промта владельца из шести антропометрии не содержат.
        for aid in ("tomatoes", "icecream"):
            with self.subTest(aid=aid):
                was = A.load(aid)["prompt"]
                got = A.strip_anthropometry(was)
                self.assertEqual(got["prompt"], was)
                self.assertEqual(got["dropped"], [])
                self.assertEqual(got["cut_share"], 0.0)

    def test_the_cut_is_reported_in_numbers_not_only_done(self):
        got = A.strip_anthropometry(A.load("y2k")["prompt"])
        self.assertEqual(len(got["dropped"]), 2)
        self.assertIn("оборотов унесено 2", got["note"])
        self.assertGreater(got["cut_share"], 0)

    def test_the_traces_of_the_operation_are_cleaned_up(self):
        # Каждый след наблюдался на боевом промте владельца и читается моделью
        # как значащий.
        country = A.strip_anthropometry(A.load("country")["prompt"])["prompt"]
        self.assertIn("of a person seated", country)
        self.assertNotIn("an person", country)
        fisheye = A.strip_anthropometry(A.load("fisheye")["prompt"])["prompt"]
        self.assertIn("lens. Person with", fisheye)
        self.assertNotIn("  ", fisheye)
        self.assertNotIn(" ,", fisheye)

    def test_the_owners_base_on_disk_is_NEVER_touched_by_the_cut(self):
        # Ц2 по смыслу: материал владельца режется НА ЛЕТУ, а не в файле.
        before = A.load("y2k")["prompt"]
        A.strip_anthropometry(before)
        self.assertEqual(A.load("y2k")["prompt"], before)
        self.assertIn("brunette", A.load("y2k")["prompt"])

    def test_an_empty_prompt_is_UNMEASURED_not_an_empty_result(self):
        for bad in ("", "   ", None, 42):
            with self.subTest(bad=bad):
                got = A.strip_anthropometry(bad)
                self.assertEqual(got["outcome"], UNMEASURED)
                self.assertIsNone(got["prompt"])

    def test_mutating_the_clause_list_both_ways_moves_what_is_cut(self):
        """Т1: список образцов строже и слабее."""
        was = A.ANTHROPOMETRY_CLAUSES
        try:
            A.ANTHROPOMETRY_CLAUSES = ()
            self.assertEqual(
                A.strip_anthropometry(A.load("y2k")["prompt"])["dropped"], [])
            A.ANTHROPOMETRY_CLAUSES = (r"\bcamera\b",)
            got = A.strip_anthropometry(A.load("y2k")["prompt"])
            self.assertGreater(len(got["dropped"]), 0)
        finally:
            A.ANTHROPOMETRY_CLAUSES = was
        self.assertEqual(
            len(A.strip_anthropometry(A.load("y2k")["prompt"])["dropped"]), 2)


class TheCutIsWiredIntoTheComposedPrompt(unittest.TestCase):
    """Резак, который написан, но не позван, выглядит рабочим до прогона."""

    def test_the_composed_prompt_carries_no_anthropometry_by_default(self):
        got = A.compose("y2k")["prompt"]
        self.assertNotIn("brunette", got)
        self.assertNotIn("freckles", got)
        self.assertNotIn("green eyes", got)
        self.assertIn("messy bun", got)
        self.assertIn("lip gloss applicator", got)

    def test_the_identity_clause_KEEPS_its_own_body_words(self):
        # СТОРОЖ ПОРЯДКА: рез идёт ДО приписок. Пройдись он по ним — унёс бы
        # «same face» и «same skin tone», то есть ровно ту строку, ради
        # которой всё и делается.
        got = A.compose("y2k")["prompt"]
        self.assertIn("same face", got)
        self.assertIn("same skin tone", got)
        self.assertIn("same hair colour", got)

    def test_turning_the_cut_off_is_LOUD_and_restores_the_owner_text(self):
        got = A.compose("y2k", cut_body=False)
        self.assertIn("brunette", got["prompt"])
        self.assertIn("РЕЗ ОТКЛЮЧЁН", got["note"])

    def test_the_cut_report_travels_with_the_prompt(self):
        got = A.compose("y2k")
        self.assertEqual(len(got["cut"]["dropped"]), 2)
        self.assertIn("оборотов унесено 2", got["note"])


class TheAssembledReferenceTakesTheLookButNeverTheFace(unittest.TestCase):
    """Ролевая строка под эстетику ОБРАТНА строке стенда, и это намеренно."""

    def test_the_new_clause_asks_for_the_wardrobe_the_old_one_forbade(self):
        new = A.assemble_prompt()
        self.assertIn("take the wardrobe", new)
        self.assertIn("accessories", new)
        self.assertIn("pose", new)

    def test_the_face_stays_forbidden_in_BOTH_editions(self):
        # Единственная ось, пережившая смену модели: лицо со второй картинки
        # нельзя никогда, потому что цена ошибки — чужой человек в ролике.
        self.assertIn("never copy the face", A.assemble_prompt())
        self.assertIn("same face", A.assemble_prompt(legacy=True))

    def test_the_legacy_clause_is_the_stand_one_and_not_a_copy(self):
        # Е1 плюс негативный контроль: старый вариант обязан БЫТЬ старым.
        from ball_reel import fork_e2e

        legacy = A.assemble_prompt(legacy=True)
        self.assertIn(fork_e2e.ROLE_CLAUSE, legacy)
        self.assertIn(fork_e2e.NO_LOOK_TRANSFER_CLAUSE, legacy)

    def test_the_two_editions_really_differ(self):
        # Иначе «мы поменяли строку» — слова о правке, а не правка.
        self.assertNotEqual(A.assemble_prompt(), A.assemble_prompt(legacy=True))


class TheLeakIsMeasuredFromBOTHSides(unittest.TestCase):
    """Мера похожести умеет сказать «похоже», но не «похоже на ЭТОГО».

    Поэтому меряются оба расстояния сразу — до клиента и до демо, — и судит
    их взаимное положение, а не одно число.
    """

    CLIENT = "assets/fork_plan_man_fullbody.png"
    DEMO_W = "assets/fork_plan_woman_fullbody.png"

    @classmethod
    def _pair(cls, to_client, to_demo):
        # Различаем якоря ПОЛНЫМ ПУТЁМ, а не подстрокой. Первая редакция
        # подставки писала `"man" in anchor` и падала на том, что «woman»
        # содержит «man»: подставной прибор отвечал за клиента на оба вопроса.
        def distances(frames, anchor, **kw):
            median = {cls.CLIENT: to_client, cls.DEMO_W: to_demo}[str(anchor)]
            return {"outcome": PASS, "median": median, "inside": 1,
                    "judged": 1, "note": "подставной прибор"}
        return distances

    def _run(self, to_client, to_demo):
        return A.leak_verdict(made="р.png", client=self.CLIENT,
                              demo=self.DEMO_W,
                              distances=self._pair(to_client, to_demo))

    def test_the_measured_good_case_is_good(self):
        # ИЗМЕРЕНО 22.08 на старых ролях: 0.2506 против 0.9436.
        got = self._run(0.2506, 0.9436)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["gap"], 0.693)

    def test_a_leaked_demo_is_a_REAL_defect_not_a_third_outcome(self):
        # Единственный случай на проекте, где «не годно» стоит без колебаний:
        # цена — чужой человек в ролике клиента.
        got = self._run(0.8, 0.1)
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("ПРОТЕКЛА", got["note"])
        self.assertEqual(got["gap"], -0.7)

    def test_the_measured_middle_case_is_UNMEASURED(self):
        # ИЗМЕРЕНО 22.08 на новых ролях: 0.3727 против 0.9258 — клиент за
        # планкой, хотя глазом это он. Прибор честно говорит «не различаю».
        got = self._run(0.3727, 0.9258)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("СУДИТ ОПЕРАТОР", got["note"])

    def test_both_close_is_UNMEASURED_because_the_gap_means_nothing_then(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ разности: она значима, только когда стороны
        # планки разные. Два близких числа — это «прибор не различает».
        got = self._run(0.20, 0.30)
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_a_missing_distance_is_UNMEASURED_not_a_pass(self):
        def half(frames, anchor, **kw):
            return {"outcome": PASS,
                    "median": None if str(anchor) == self.DEMO_W else 0.2,
                    "inside": 1, "judged": 1, "note": "полприбора"}

        got = A.leak_verdict(made="р.png", client=self.CLIENT,
                             demo=self.DEMO_W, distances=half)
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_an_instrument_that_fell_is_UNMEASURED(self):
        def broken(*a, **k):
            raise RuntimeError("модель не загрузилась")

        got = A.leak_verdict(made="р.png", client="м.png", demo="ж.png",
                             distances=broken)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("RuntimeError", got["note"])


class TheGenderPairIsAMachineGateNotANote(unittest.TestCase):
    """Решение владельца 22.08: «верно, эстетики по полу».

    Правило, записанное словами, живёт до первой спешки (Ц7). Здесь оно
    роняет пару в «не годно» ДО всякой генерации.
    """

    def test_both_demos_are_the_universal_plan_assets(self):
        self.assertEqual(sorted(A.GENDERS), ["f", "m"])
        self.assertEqual(A.demo_for("m"), "assets/fork_plan_man_fullbody.png")
        self.assertEqual(A.demo_for("f"), "assets/fork_plan_woman_fullbody.png")

    def test_an_unknown_gender_is_refused_not_defaulted(self):
        # Молчаливое умолчание подсунуло бы женскую эстетику мужчине — ровно
        # тот дефект, ради которого правило заведено.
        for bad in ("", "ж", None, "x"):
            with self.subTest(bad=bad), self.assertRaises(KeyError):
                A.demo_for(bad)

    def test_the_gender_lives_in_the_file_name(self):
        # Реестр на стороне может отстать от файла; имя — не может.
        self.assertTrue(str(A.aesthetic_file("y2k", "m")).endswith("y2k_m.png"))
        self.assertTrue(str(A.aesthetic_file("y2k", "f")).endswith("y2k_f.png"))

    def test_a_matching_pair_passes(self):
        got = A.pair_check(client_gender="m", aesthetic_gender="m")
        self.assertEqual(got["outcome"], PASS)

    def test_a_mismatched_pair_is_a_REAL_defect(self):
        got = A.pair_check(client_gender="m", aesthetic_gender="f")
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("РАЗЪЕХАЛСЯ", got["note"])

    def test_an_unnamed_gender_is_NOT_permission_to_continue(self):
        got = A.pair_check(client_gender="", aesthetic_gender="m")
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("НЕ разрешение", got["note"])

    def test_the_gate_is_case_and_space_insensitive(self):
        self.assertEqual(
            A.pair_check(client_gender=" M ", aesthetic_gender="m")["outcome"],
            PASS)


class TheGenderSplitDidNotFixTheWardrobeAndItIsRecorded(unittest.TestCase):
    """ИЗМЕРЕННЫЙ ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ (И6), а не забытая догадка.

    Разделение эстетик по полу СДЕЛАНО и работает как гейт, но заявленную
    задачу НЕ РЕШИЛО: y2k на мужской демо-личности всё равно надел на мужчину
    мини-юбку. Причина: юбка стоит В САМОМ ПРОМТЕ владельца, а не в демо, и
    пол демо на гардероб не влияет.

    Сторож стоит здесь, чтобы следующая сессия не переставила ту же ручку.
    """

    def test_the_owner_prompt_itself_names_the_gendered_garment(self):
        prompt = A.load("y2k")["prompt"]
        self.assertIn("denim mini skirt", prompt)
        # И он переживает рез антропометрии — потому что одежда НЕ
        # антропометрия и уноситься не должна.
        self.assertIn("denim mini skirt", A.compose("y2k")["prompt"])

    def test_the_cut_deliberately_leaves_the_wardrobe_alone(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ ошибочной догадки: если бы гардероб резался,
        # эстетика перестала бы быть шаблоном.
        got = A.strip_anthropometry(A.load("y2k")["prompt"])
        self.assertIn("sweatshirt", got["prompt"])
        self.assertIn("mini skirt", got["prompt"])


class TheTemplateGenderIsTheAestheticGender(unittest.TestCase):
    """Решение владельца 22.08: «не меняем пол в промтах, просто сами шаблоны
    собираем в соответствии с полом; какая стилевая рефка по полу такой и пол
    шаблона».

    ЭТО ОТМЕНЯЕТ прежнюю попытку собирать каждую эстетику в двух полах: она
    ИЗМЕРЕНО не работала — y2k на мужской демо-личности всё равно надел на
    мужчину мини-юбку, потому что юбка стоит в промте. Теперь y2k просто
    ЖЕНСКИЙ шаблон, и мужчине он не подаётся вовсе.
    """

    def test_every_shipped_aesthetic_names_its_gender(self):
        # Без этого поля клиенту можно подсунуть шаблон чужого пола.
        for aid in A.ids():
            with self.subTest(aid=aid):
                self.assertIn(A.gender_of(aid), A.GENDERS)

    def test_the_shipped_assignment_is_the_one_chosen(self):
        # Литералы (Т2): происхождение каждого — гардероб в промте владельца.
        self.assertEqual(A.gender_of("y2k"), "f")          # мини-юбка, блеск
        self.assertEqual(A.gender_of("midcentury"), "f")   # юбка, сапоги
        self.assertEqual(A.gender_of("fisheye"), "m")      # тренч, нейтрален
        self.assertEqual(A.gender_of("tomatoes"), "m")     # transform

    def test_an_aesthetic_without_a_gender_is_refused_not_defaulted(self):
        with self.assertRaises(KeyError) as e:
            A.gender_of({"id": "безпола", "prompt": "x"})
        self.assertIn("безпола", str(e.exception))

    def test_the_file_name_follows_the_base_without_being_told(self):
        self.assertTrue(str(A.aesthetic_file("y2k")).endswith("y2k_f.png"))
        self.assertTrue(str(A.aesthetic_file("fisheye")).endswith("fisheye_m.png"))

    def test_a_client_of_the_other_gender_is_stopped_before_generation(self):
        # Гейт, ради которого всё: мужчина и женский шаблон не встречаются.
        got = A.pair_check(client_gender="m",
                           aesthetic_gender=A.gender_of("y2k"))
        self.assertEqual(got["outcome"], FAIL)

    def test_a_client_of_the_same_gender_passes(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ гейта: он обязан уметь пропускать.
        got = A.pair_check(client_gender="m",
                           aesthetic_gender=A.gender_of("fisheye"))
        self.assertEqual(got["outcome"], PASS)


class TheBrandBanWasNarrowedByTheOwner(unittest.TestCase):
    """«Бренды пусть остаются, просто добавляем no logo во все промты стилей».

    Прежняя редакция запрещала НАЗВАНИЯ МАРОК и этим воевала с промтами самого
    владельца. ИЗМЕРЕНО, чем кончалась война: на y2k_f знака не было, на y2k_m
    проступил читаемый «adidas» — исход решался случаем, а не запретом.
    """

    def test_the_ban_forbids_the_drawn_mark_not_the_word(self):
        from ball_reel import fork_e2e

        ban = fork_e2e.NO_BRANDS_CLAUSE
        self.assertIn("no logo", ban)
        self.assertIn("no lettering", ban)
        self.assertNotIn("no brand names", ban)

    def test_the_owners_own_brand_words_now_survive_into_the_prompt(self):
        # Ровно то, ради чего правка: промт владельца больше не спорит сам с
        # собой в одной строке.
        built = A.compose("y2k")["prompt"]
        self.assertIn("Adidas", built)
        self.assertIn("no logo", built)

    def test_the_ban_reaches_every_aesthetic_prompt(self):
        # «Во все промты стилей» — проверяем ВСЕ, а не один.
        for aid in A.ids():
            with self.subTest(aid=aid):
                self.assertIn("no logo", A.compose(aid)["prompt"])
