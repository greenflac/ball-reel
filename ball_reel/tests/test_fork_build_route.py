"""Маршрутизатор: подделанные пропорции обязаны уводить решение в другую корзину,
а портрет — давать «не уверен», а не тихое умолчание.

Оба требования взяты из приёмки §3b дословно. Третий тест — про то, что
маршрутизатор, отдающий человеку всё, не считается согласившимся с человеком.
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
    """Приёмка §3b: тест, который умеет краснеть."""

    def test_a_narrow_shoulder_figure_goes_to_A(self):
        self.assertEqual(fbr.route(proportions=_p(1.30))["bucket"], fbr.BUCKET_A)

    def test_a_broad_shoulder_figure_goes_to_B(self):
        self.assertEqual(fbr.route(proportions=_p(1.90))["bucket"], fbr.BUCKET_B)

    def test_moving_the_number_across_the_split_moves_the_bucket(self):
        low = fbr.route(proportions=_p(fbr.SPLIT - fbr.UNSURE_BAND - 0.01))
        high = fbr.route(proportions=_p(fbr.SPLIT + fbr.UNSURE_BAND + 0.01))
        self.assertEqual(low["bucket"], fbr.BUCKET_A)
        self.assertEqual(high["bucket"], fbr.BUCKET_B)
        self.assertNotEqual(low["bucket"], high["bucket"])

    def test_the_split_is_guarded_in_both_directions(self):
        """Т1: подмена константы-решения строже и слабее."""
        value = _p(1.30)
        self.assertEqual(fbr.route(proportions=value, split=1.53)["bucket"],
                         fbr.BUCKET_A)
        self.assertEqual(fbr.route(proportions=value, split=1.00)["bucket"],
                         fbr.BUCKET_B,
                         "точка раздела сдвинута ниже значения, а корзина не "
                         "изменилась — значит она ни на что не влияет")

    def test_the_band_is_guarded_in_both_directions(self):
        value = _p(1.55)
        self.assertEqual(fbr.route(proportions=value, band=0.0)["bucket"],
                         fbr.BUCKET_B, "полоса снята, а решение не принято")
        self.assertEqual(fbr.route(proportions=value, band=5.0)["bucket"],
                         fbr.UNSURE,
                         "полоса шире всего диапазона, а корзина всё равно "
                         "выбрана")


class TheThirdOutcomeIsRealAndNotDecorative(unittest.TestCase):

    def test_a_value_inside_the_band_is_unsure(self):
        got = fbr.route(proportions=_p(fbr.SPLIT))
        self.assertEqual(got["bucket"], fbr.UNSURE)
        self.assertIn("нет оснований выбирать", got["note"])

    def test_unsure_says_it_falls_back_to_a_human(self):
        got = fbr.route(proportions=_p(fbr.SPLIT))
        self.assertIn("ручной выбор", got["note"])

    def test_missing_proportions_are_unsure_not_a_default_bucket(self):
        got = fbr.route(proportions=None if False else {})
        self.assertEqual(got["bucket"], fbr.UNSURE)
        self.assertFalse(got["measured"])
        self.assertIn("наугад", got["note"])

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
        чтобы выбрать ему корзину.
        """
        from ball_reel.metrics import ASSUMED_PROPORTIONS

        self.assertIn("shoulder_to_hip", ASSUMED_PROPORTIONS,
                      "форма ASSUMED_PROPORTIONS изменилась — проверить, не "
                      "начал ли маршрутизатор случайно им пользоваться")
        got = fbr.route(proportions={})
        self.assertEqual(got["bucket"], fbr.UNSURE)
        self.assertIsNone(got["value"])


class AgreementCountsThreeNumbersNotOne(unittest.TestCase):
    """Маршрутизатор, отдающий человеку всё, согласуется идеально и не работает."""

    def test_full_agreement_on_decided_examples(self):
        got = fbr.agreement([(_p(1.20), fbr.BUCKET_A),
                             (_p(1.95), fbr.BUCKET_B)])
        self.assertEqual((got["agreed"], got["disagreed"]), (2, 0))
        self.assertEqual(got["accuracy_on_decided"], 1.0)

    def test_a_disagreement_is_reported_with_what_was_chosen(self):
        got = fbr.agreement([(_p(1.20), fbr.BUCKET_B)])
        self.assertEqual(got["disagreed"], 1)
        self.assertEqual(got["disagreements"][0][2], fbr.BUCKET_A)

    def test_deferring_everything_is_not_reported_as_success(self):
        got = fbr.agreement([(_p(fbr.SPLIT), fbr.BUCKET_A),
                             (_p(fbr.SPLIT), fbr.BUCKET_B)])
        self.assertEqual(got["deferred"], 2)
        self.assertIsNone(got["accuracy_on_decided"])
        self.assertIn("не сделал ничего", got["note"])

    def test_the_deferral_share_is_printed_next_to_the_accuracy(self):
        got = fbr.agreement([(_p(1.20), fbr.BUCKET_A),
                             (_p(fbr.SPLIT), fbr.BUCKET_A)])
        self.assertEqual(got["deferral_share"], 0.5)
        self.assertEqual(got["accuracy_on_decided"], 1.0)

    def test_an_empty_handful_is_unmeasured(self):
        got = fbr.agreement([])
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_the_note_admits_the_handful_does_not_exist_yet(self):
        got = fbr.agreement([(_p(1.20), fbr.BUCKET_A)])
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

    def test_the_band_covers_the_spread_we_have_actually_seen(self):
        self.assertGreaterEqual(fbr.UNSURE_BAND * 2, 0.10,
                                "полоса уже наблюдавшегося разброса 1.48..1.58 "
                                "— значит она утверждает уверенность, которой "
                                "у нас нет")

    def test_there_are_exactly_three_outcomes(self):
        self.assertEqual(len(fbr.BUCKETS), 3)
        self.assertIn(fbr.UNSURE, fbr.BUCKETS)


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


if __name__ == "__main__":
    unittest.main()
