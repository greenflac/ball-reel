"""Gates. Each asserts a PROPERTY and is written to fail on a real mutation.

These are the "does it bite" tests: break the code the docstring names and the
gate goes red, not "the function did not crash".
"""

from __future__ import annotations

import unittest
from pathlib import Path

from ball_reel import identity
from ball_reel.critic import DEFAULT_BAR
from ball_reel.pipeline import run
from ball_reel.tools import make_fixtures

ROOT = Path(identity.__file__).resolve().parent


def _ensure_fixtures():
    if not (ROOT / "fixtures" / "index.json").exists():
        make_fixtures.main()


class IdentityDriftMeasuresIdentity(unittest.TestCase):
    """Property: identical frames drift 0; unrelated frames drift high.

    Break `_agreement` to always return 1.0 and this goes red — the measure
    would then say every frame is the same person.
    """

    def setUp(self):
        _ensure_fixtures()

    def test_identical_is_zero_drift(self):
        f = str(ROOT / "fixtures" / "start" / "energy_led.png")
        d = identity.identity_drift([f], f)
        self.assertEqual(d["per_frame"][Path(f).name], 0.0)

    def test_unrelated_drifts_past_the_bar(self):
        ref = str(ROOT / "fixtures" / "start" / "product_led.png")
        frames = sorted(str(p) for p in
                        (ROOT / "fixtures" / "video" / "product_led").glob("*.png"))
        d = identity.identity_drift(frames, ref)
        # the composition changes character across the clip: its worst frame
        # drifts past the bar, which is what rejects it end to end.
        self.assertGreater(d["worst"][1], DEFAULT_BAR.max_identity_drift)


class MotionPresenceRejectsAFrozenClip(unittest.TestCase):
    """Property: six identical frames read as not moving.

    Break `motion_presence` to floor at a constant and the frozen clip would
    pass; this holds the floor.
    """

    def setUp(self):
        _ensure_fixtures()

    def test_frozen_is_not_moving(self):
        frames = sorted(str(p) for p in
                        (ROOT / "fixtures" / "video" / "routine_led").glob("*.png"))
        self.assertFalse(identity.motion_presence(frames)["moving"])

    def test_shifting_clip_moves(self):
        frames = sorted(str(p) for p in
                        (ROOT / "fixtures" / "video" / "energy_led").glob("*.png"))
        self.assertTrue(identity.motion_presence(frames)["moving"])


class TheRunAcceptsOnlyTheGoodClip(unittest.TestCase):
    """End to end: exactly the consistent, moving clip ships; the other two are
    rejected on the two different computed gates.

    This is the whole point of the critic: a beautiful frozen clip and a lively
    but drifting one both fail, for stated and different reasons.
    """

    def setUp(self):
        _ensure_fixtures()

    def test_one_accepted_for_the_right_reasons(self):
        result = run(ROOT)
        by_id = {s.strategy_id: s for s in result["ranked"]}
        self.assertTrue(by_id["energy_led"].accepted)
        self.assertFalse(by_id["product_led"].accepted)
        self.assertIn("identity drift", by_id["product_led"].reason)
        self.assertFalse(by_id["routine_led"].accepted)
        self.assertIn("motion", by_id["routine_led"].reason)
        self.assertEqual(result["accepted"], 1)

    def test_offline_run_is_flagged_synthetic(self):
        self.assertTrue(run(ROOT)["synthetic"])


if __name__ == "__main__":
    unittest.main()


class APartialRetargetIsWorseThanNone(unittest.TestCase):
    """Масштабировать одну кость конечности значит сломать конечность.

    ИЗМЕРЕНО на живом ките: у цели (портретное фото) измерены бёдра и НЕ
    измерены голени — коленей и стоп в кадре нет. Прежний код ретаргетил то,
    что смог:

        нога слева : бедро/голень 0.899 -> 0.630 (множители 0.701 и 1.000)
        нога справа: бедро/голень 0.960 -> 0.617 (множители 0.643 и 1.000)

    Бедро сжималось на треть, голень оставалась донорской. Такого тела нет ни
    у кого, и ControlNet честно рисовал деформацию. Поза потом судилась против
    НАСТОЯЩЕГО driving-кадра и мазала системно: 0.26 при баре 0.15, ноль кадров
    из шестнадцати в баре. Дефект читался как «генератор не слушается позы».

    Руки при этом были целы: там измерены обе кости. Разница между двумя
    случаями и есть правило — конечность целиком или никак.
    """

    DONOR = {"l_shoulder->l_elbow": 0.585, "l_elbow->l_wrist": 0.5371,
             "r_shoulder->r_elbow": 0.6109, "r_elbow->r_wrist": 0.5649,
             "l_hip->l_knee": 0.8513, "l_knee->l_ankle": 0.9467,
             "r_hip->r_knee": 0.9274, "r_knee->r_ankle": 0.9664}

    def setUp(self):
        from ball_reel import skeleton

        self.s = skeleton

    def test_a_leg_with_an_unmeasured_shin_is_not_retargeted_at_all(self):
        target = {"l_hip->l_knee": 0.5967}          # голени нет
        factors, origin = self.s.retarget_plan(target, self.DONOR)
        self.assertNotIn("l_hip->l_knee", factors,
                         "бедро сжали, голень оставили — нога сломана")
        self.assertIn("снят", origin["l_hip->l_knee"])

    def test_an_arm_with_both_bones_measured_is_retargeted(self):
        target = {"l_shoulder->l_elbow": 0.6997, "l_elbow->l_wrist": 0.5751}
        factors, _ = self.s.retarget_plan(target, self.DONOR)
        self.assertIn("l_shoulder->l_elbow", factors)
        self.assertIn("l_elbow->l_wrist", factors)

    def test_the_inner_proportion_of_every_limb_survives(self):
        """Главное свойство: отношение костей внутри конечности не уезжает."""
        target = {"l_shoulder->l_elbow": 0.6997, "l_elbow->l_wrist": 0.5751,
                  "l_hip->l_knee": 0.5967}
        factors, _ = self.s.retarget_plan(target, self.DONOR)
        for chain in self.s.LIMB_CHAINS:
            a, b = chain
            if a not in self.DONOR or b not in self.DONOR:
                continue
            fa, fb = factors.get(a, 1.0), factors.get(b, 1.0)
            before = self.DONOR[a] / self.DONOR[b]
            after = (self.DONOR[a] * fa) / (self.DONOR[b] * fb)
            with self.subTest(limb=a):
                self.assertLess(
                    abs(after / before - 1), 0.25,
                    f"{a}: пропорция внутри конечности уехала "
                    f"{before:.3f} -> {after:.3f}")

    def test_the_broken_case_from_the_kit_is_a_regression(self):
        """Ровно те числа, что стояли в манифесте живого кита."""
        target = {"l_hip->l_knee": 0.5967, "r_hip->r_knee": 0.5967}
        factors, _ = self.s.retarget_plan(target, self.DONOR)
        for k in ("l_hip->l_knee", "r_hip->r_knee"):
            self.assertNotIn(k, factors)

    def test_a_whole_limb_missing_from_the_donor_is_not_a_partial_case(self):
        """Нет донора — нечего и снимать: список множителей и так пуст."""
        donor = {k: v for k, v in self.DONOR.items() if "hip" not in k
                 and "knee" not in k}
        target = {"l_shoulder->l_elbow": 0.6997, "l_elbow->l_wrist": 0.5751}
        factors, origin = self.s.retarget_plan(target, donor)
        self.assertIn("l_shoulder->l_elbow", factors)
        self.assertIn("донор не измерен", origin["l_hip->l_knee"])

    def test_every_chain_is_a_real_pair_of_bones(self):
        for chain in self.s.LIMB_CHAINS:
            for bone in chain:
                with self.subTest(bone=bone):
                    self.assertIn(bone, self.s.BONE_TO_PROPORTION.values())
