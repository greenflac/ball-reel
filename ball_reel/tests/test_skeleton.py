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
