"""Маршрутизатор: подделанные пропорции обязаны уводить решение в другой исход,
а портрет — давать «не уверен», а не тихое умолчание.

Оба требования взяты из приёмки ХЭНДОФ §6 F дословно. Плюс то, что появилось
вместе с §2a: исходы называются «LoRA не нужна» / «LoRA полная 40+», полоса
неуверенности НЕСИММЕТРИЧНА, и её асимметрия сама сторожится мутацией — иначе
её можно было бы обнулить, не покраснев ни разу.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from ball_reel import fork_build_route as fbr
from ball_reel.fork_identity import PASS, UNMEASURED

ROOT = Path(__file__).resolve().parents[2]
HERO = ROOT / "demo" / "hero.png"
PORTRAIT = ROOT / "demo" / "lora_dataset" / "img" / "real_0000.png"


def _p(value):
    return {fbr.KEY: value, "shoulder_width": 0.6, "hip_width": 0.4}


class FakedProportionsMoveTheDecision(unittest.TestCase):
    """Приёмка §6 F: тест, который умеет краснеть."""

    def test_a_narrow_shoulder_figure_asks_for_the_lora(self):
        self.assertEqual(fbr.route(proportions=_p(1.30))["bucket"],
                         fbr.BUCKET_LORA_FULL_W40)

    def test_a_broad_shoulder_figure_needs_no_lora(self):
        self.assertEqual(fbr.route(proportions=_p(1.90))["bucket"],
                         fbr.BUCKET_NO_LORA)

    def test_moving_the_number_across_the_band_moves_the_outcome(self):
        low, high = fbr.bounds()
        below = fbr.route(proportions=_p(low - 0.01))
        above = fbr.route(proportions=_p(high + 0.01))
        self.assertEqual(below["bucket"], fbr.BUCKET_LORA_FULL_W40)
        self.assertEqual(above["bucket"], fbr.BUCKET_NO_LORA)
        self.assertNotEqual(below["bucket"], above["bucket"])

    def test_the_split_is_guarded_in_both_directions(self):
        """Т1: подмена константы-решения строже и слабее."""
        value = _p(1.30)
        self.assertEqual(fbr.route(proportions=value, split=1.53)["bucket"],
                         fbr.BUCKET_LORA_FULL_W40)
        self.assertEqual(fbr.route(proportions=value, split=1.00)["bucket"],
                         fbr.BUCKET_NO_LORA,
                         "точка раздела сдвинута ниже значения, а исход не "
                         "изменился — значит она ни на что не влияет")

    def test_the_lora_side_margin_is_guarded_in_both_directions(self):
        """Т1 по `MARGIN_TO_LORA`: шире — перестаём включать, уже — включаем."""
        value = _p(1.50)  # между 1.43 и 1.53, то есть внутри полосы по умолчанию
        self.assertEqual(fbr.route(proportions=value)["bucket"], fbr.UNSURE)
        self.assertEqual(
            fbr.route(proportions=value, margin_to_lora=0.01)["bucket"],
            fbr.BUCKET_LORA_FULL_W40,
            "запас со стороны LoRA сжат почти в ноль, а LoRA всё равно не "
            "включилась — значит запас ни на что не влияет")
        self.assertEqual(
            fbr.route(proportions=_p(1.20), margin_to_lora=5.0)["bucket"],
            fbr.UNSURE,
            "запас шире всего диапазона, а LoRA всё равно включена")

    def test_the_no_lora_side_margin_is_guarded_in_both_directions(self):
        """Т1 по `MARGIN_TO_NO_LORA`. Он маленький, и тем важнее, что он живой."""
        value = _p(1.535)  # выше раздела 1.53, но в пределах запаса 0.01
        self.assertEqual(fbr.route(proportions=value)["bucket"], fbr.UNSURE)
        self.assertEqual(
            fbr.route(proportions=value, margin_to_no_lora=0.0)["bucket"],
            fbr.BUCKET_NO_LORA,
            "запас снят, а решение не принято")
        self.assertEqual(
            fbr.route(proportions=_p(1.90), margin_to_no_lora=5.0)["bucket"],
            fbr.UNSURE,
            "запас шире всего диапазона, а исход всё равно выбран")


class TheBandLeansTowardsTheSafeOutcome(unittest.TestCase):
    """§2: первый исход — ветвь по умолчанию, то есть безопасная.

    «Ошибка скорее недодаст эффект, чем навяжет чужую фигуру». Это утверждение
    про ЦЕНУ ДВУХ РАЗНЫХ ОШИБОК, и в коде оно обязано быть числом, а не словом
    в комментарии: полоса неуверенности шире со стороны LoRA.
    """

    def test_the_safe_outcome_is_the_one_without_the_lora(self):
        self.assertEqual(fbr.SAFE_BUCKET, fbr.BUCKET_NO_LORA)

    def test_the_margin_towards_the_expensive_error_is_the_wider_one(self):
        self.assertGreater(fbr.MARGIN_TO_LORA, fbr.MARGIN_TO_NO_LORA,
                           "полоса симметрична — значит код молча утверждает, "
                           "что навязать чужую фигуру стоит столько же, "
                           "сколько недодать эффект")
        self.assertEqual((fbr.MARGIN_TO_LORA, fbr.MARGIN_TO_NO_LORA),
                         (0.10, 0.01),
                         "Т2: числа ВЫБРАНЫ (0.10 — наблюдавшийся разброс "
                         "1.48..1.58, 0.01 — на порядок меньше как запись "
                         "асимметрии); менять их вместе с этим литералом")

    def test_two_mirrored_deviations_do_not_get_mirrored_answers(self):
        """Настоящая проверка асимметрии: одинаковый отход, разные исходы."""
        step = 0.05
        below = fbr.route(proportions=_p(fbr.SPLIT - step))
        above = fbr.route(proportions=_p(fbr.SPLIT + step))
        self.assertEqual(below["bucket"], fbr.UNSURE,
                         "отход вниз на столько же уже даёт LoRA — полоса "
                         "перестала защищать дорогую сторону")
        self.assertEqual(above["bucket"], fbr.BUCKET_NO_LORA)

    def test_making_the_band_symmetric_changes_the_answer(self):
        """Т1 по самой асимметрии: симметричная полоса обязана дать другое."""
        value = _p(fbr.SPLIT - 0.05)
        self.assertEqual(fbr.route(proportions=value)["bucket"], fbr.UNSURE)
        symmetric = fbr.route(proportions=value,
                              margin_to_lora=fbr.MARGIN_TO_NO_LORA)
        self.assertEqual(symmetric["bucket"], fbr.BUCKET_LORA_FULL_W40,
                         "при симметричной полосе исход не изменился — значит "
                         "асимметрия ничего не делает и является украшением")

    def test_the_note_names_the_asymmetry_so_a_report_carries_it(self):
        note = fbr.route(proportions=_p(1.90))["note"]
        self.assertIn("несимметрична", note)
        self.assertIn("чужую фигуру", note)

    def test_the_bounds_are_the_ones_the_decision_actually_uses(self):
        """Е1: края читаются оттуда же, откуда их берёт решение."""
        low, high = fbr.bounds()
        self.assertAlmostEqual(low, 1.43, places=6)
        self.assertAlmostEqual(high, 1.54, places=6)
        got = fbr.route(proportions=_p(1.90))
        self.assertAlmostEqual(got["low"], low, places=6)
        self.assertAlmostEqual(got["high"], high, places=6)


class TheOutcomesAreNamedSoAReportReadsWithoutDecoding(unittest.TestCase):
    """§2a: обучается ОДНА LoRA, второе плечо — её отсутствие."""

    def test_the_outcome_names_are_the_handoff_wording(self):
        self.assertEqual(fbr.BUCKET_NO_LORA, "LoRA не нужна")
        self.assertEqual(fbr.BUCKET_LORA_FULL_W40, "LoRA полная 40+")
        self.assertEqual(fbr.UNSURE, "не уверен")

    def test_no_outcome_is_named_after_the_driving_figure_any_more(self):
        """Имя «как в драйвинге» читалось как ВТОРАЯ LoRA. Снято §2a."""
        for name in fbr.BUCKETS:
            self.assertNotIn("драйвинг", name.lower(),
                             "исход снова назван фигурой драйвинга — это "
                             "отсутствие LoRA, а не вторая LoRA")

    def test_there_are_exactly_three_outcomes(self):
        self.assertEqual(len(fbr.BUCKETS), 3)
        self.assertIn(fbr.UNSURE, fbr.BUCKETS)
        self.assertIn(fbr.SAFE_BUCKET, fbr.BUCKETS)


class TheThirdOutcomeIsRealAndNotDecorative(unittest.TestCase):

    def test_a_value_inside_the_band_is_unsure(self):
        got = fbr.route(proportions=_p(fbr.SPLIT))
        self.assertEqual(got["bucket"], fbr.UNSURE)
        self.assertIn("нет оснований выбирать", got["note"])

    def test_unsure_says_it_falls_back_to_a_human(self):
        got = fbr.route(proportions=_p(fbr.SPLIT))
        self.assertIn("ручной выбор", got["note"])

    def test_missing_proportions_are_unsure_not_the_safe_default(self):
        """Безопасная ветвь — про пограничное ИЗМЕРЕНИЕ, а не про его отсутствие."""
        got = fbr.route(proportions={})
        self.assertEqual(got["bucket"], fbr.UNSURE)
        self.assertNotEqual(got["bucket"], fbr.SAFE_BUCKET)
        self.assertFalse(got["measured"])
        self.assertIn("измерения нет вовсе", got["note"])

    def test_proportions_without_the_key_are_unsure_and_say_why(self):
        got = fbr.route(proportions={"shoulder_width": 0.6})
        self.assertEqual(got["bucket"], fbr.UNSURE)
        self.assertIn("не смогли", got["note"].lower())

    def test_it_refuses_to_be_called_with_nothing(self):
        with self.assertRaises(ValueError):
            fbr.route()

    def test_the_assumed_body_fallback_is_deliberately_not_repeated(self):
        """intake откатывается на типовое тело; маршрутизатор — нет.

        Типовое тело годится, чтобы не отказать человеку в приёме, и негодно,
        чтобы выбрать ему ветвь.
        """
        from ball_reel.metrics import ASSUMED_PROPORTIONS

        self.assertIn("shoulder_to_hip", ASSUMED_PROPORTIONS,
                      "форма ASSUMED_PROPORTIONS изменилась — проверить, не "
                      "начал ли маршрутизатор случайно им пользоваться")
        got = fbr.route(proportions={})
        self.assertEqual(got["bucket"], fbr.UNSURE)
        self.assertIsNone(got["value"])

    def test_the_typical_body_would_not_even_decide(self):
        """Т2: 1.49 — литерал, сверенный с `metrics.ASSUMED_PROPORTIONS`."""
        from ball_reel.metrics import ASSUMED_PROPORTIONS

        self.assertEqual(ASSUMED_PROPORTIONS["shoulder_to_hip"], 1.49)
        self.assertEqual(fbr.route(proportions=_p(1.49))["bucket"], fbr.UNSURE)


class AgreementCountsThreeNumbersNotOne(unittest.TestCase):
    """Маршрутизатор, отдающий человеку всё, согласуется идеально и не работает."""

    def test_full_agreement_on_decided_examples(self):
        got = fbr.agreement([(_p(1.20), fbr.BUCKET_LORA_FULL_W40),
                             (_p(1.95), fbr.BUCKET_NO_LORA)])
        self.assertEqual((got["agreed"], got["disagreed"]), (2, 0))
        self.assertEqual(got["accuracy_on_decided"], 1.0)

    def test_a_disagreement_is_reported_with_what_was_chosen(self):
        got = fbr.agreement([(_p(1.20), fbr.BUCKET_NO_LORA)])
        self.assertEqual(got["disagreed"], 1)
        self.assertEqual(got["disagreements"][0][2], fbr.BUCKET_LORA_FULL_W40)

    def test_the_expensive_direction_of_a_disagreement_is_counted_apart(self):
        """Навязанная LoRA и недоданный эффект — разные новости, не одна."""
        imposed = fbr.agreement([(_p(1.20), fbr.BUCKET_NO_LORA)])
        cheap = fbr.agreement([(_p(1.95), fbr.BUCKET_LORA_FULL_W40)])
        self.assertEqual((imposed["disagreed"], imposed["imposed_lora"]), (1, 1))
        self.assertEqual((cheap["disagreed"], cheap["imposed_lora"]), (1, 0))
        self.assertIn("навязанной LoRA", imposed["note"])

    def test_deferring_everything_is_not_reported_as_success(self):
        got = fbr.agreement([(_p(fbr.SPLIT), fbr.BUCKET_LORA_FULL_W40),
                             (_p(fbr.SPLIT), fbr.BUCKET_NO_LORA)])
        self.assertEqual(got["deferred"], 2)
        self.assertIsNone(got["accuracy_on_decided"])
        self.assertIn("не сделал ничего", got["note"])

    def test_the_deferral_share_is_printed_next_to_the_accuracy(self):
        got = fbr.agreement([(_p(1.20), fbr.BUCKET_LORA_FULL_W40),
                             (_p(fbr.SPLIT), fbr.BUCKET_LORA_FULL_W40)])
        self.assertEqual(got["deferral_share"], 0.5)
        self.assertEqual(got["accuracy_on_decided"], 1.0)

    def test_an_empty_handful_is_unmeasured(self):
        got = fbr.agreement([])
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_the_note_admits_the_handful_does_not_exist_yet(self):
        got = fbr.agreement([(_p(1.20), fbr.BUCKET_LORA_FULL_W40)])
        self.assertEqual(got["outcome"], PASS)
        self.assertIn("НЕПРОВЕРЕНО", got["note"])


class TheNumbersBehindTheSplitAreTheOnesThatWereRead(unittest.TestCase):
    """Т2: литералы, сверенные с кодом и прогоном, а не импорт из модуля."""

    def test_the_split_sits_between_the_two_measured_people(self):
        from ball_reel.metrics import ASSUMED_PROPORTIONS

        self.assertEqual(fbr.SPLIT, 1.53)
        self.assertGreater(fbr.SPLIT, 1.48)
        self.assertLess(fbr.SPLIT, 1.58)
        self.assertIsNotNone(ASSUMED_PROPORTIONS.get("shoulder_to_hip"))

    def test_the_lora_side_margin_covers_the_spread_we_have_seen(self):
        self.assertGreaterEqual(fbr.MARGIN_TO_LORA, 0.10,
                                "запас со стороны LoRA уже наблюдавшегося "
                                "разброса 1.48..1.58 — значит он утверждает "
                                "уверенность, которой у нас нет, и утверждает "
                                "её на дорогой стороне")


@unittest.skipUnless(HERO.exists() and PORTRAIT.exists(),
                     "нет демо-изображений")
class OnRealPhotographs(unittest.TestCase):
    """Живьём: портрет обязан дать «не уверен», кадр с телом — число."""

    def test_a_portrait_routes_to_unsure_by_construction(self):
        got = fbr.route(PORTRAIT)
        self.assertEqual(got["bucket"], fbr.UNSURE)
        self.assertFalse(got["measured"])
        self.assertIn("портрет", got["note"])

    def test_a_full_body_frame_actually_produces_a_number(self):
        """Иначе предыдущий тест зеленел бы просто потому, что ничего не работает."""
        got = fbr.route(HERO)
        self.assertTrue(got["measured"],
                        f"тела не нашли и на demo/hero.png: {got['note']}")
        self.assertIsNotNone(got["value"])

    def test_the_measured_value_matches_what_was_recorded(self):
        """Т2: 1.5511 снято прогоном, команда в docs/FORK_LORA.md."""
        got = fbr.route(HERO)
        self.assertAlmostEqual(got["value"], 1.5511, places=3)

    def test_hero_lands_in_no_lora_and_only_because_of_the_asymmetry(self):
        """Живой замер: 1.5511 при симметричной полосе был «не уверен».

        1.5511 стоит на 0.0211 выше раздела 1.53. Прежняя симметричная полоса
        0.08 накрывала его, и живой кадр не решался вовсе. Несимметричная
        полоса решает его в безопасную сторону с запасом 0.0111 до края 1.54 —
        небольшим, и это честно: кадр действительно пограничный, просто
        пограничность на дешёвой стороне не стоит вопроса человеку.
        """
        got = fbr.route(HERO)
        self.assertEqual(got["bucket"], fbr.BUCKET_NO_LORA)
        self.assertEqual(got["bucket"], fbr.SAFE_BUCKET)
        symmetric = fbr.route(HERO, margin_to_no_lora=fbr.MARGIN_TO_LORA)
        self.assertEqual(symmetric["bucket"], fbr.UNSURE,
                         "при симметричной полосе hero тоже решается — значит "
                         "этот тест не про асимметрию")


if __name__ == "__main__":
    unittest.main()
