"""Вердикт гейта — по одному тесту на каждую причину отказа.

Эти тесты написаны не «на всякий случай», а по результату мутационного аудита:
снятый `MIN_COVERAGE` не ронял ни одного из 120 тестов, потому что вердикт жил
внутри платного цикла генерации и проверить его было нечем. Порог, который
нечем проверить, — это порог, который завтра сдвинут молча.

Каждый тест ломает ровно одну составляющую и требует, чтобы вердикт назвал
именно её. Проверяется не только «не прошло», но и ПРИЧИНА: гейт, который
заворачивает клип с неправильным объяснением, бесполезен для отладки и
опасен — по нему будут чинить не то.
"""

from __future__ import annotations

import unittest


def _drift(median=0.20, p90=0.30, coverage=1.0):
    return {"median": median, "p90": p90, "coverage": coverage,
            "note": "identity note"}


def _good(**over):
    base = dict(drift=_drift(), motion=0.10,
                quality={"smooth": True, "note": "continuous"},
                seam={"seamless": True, "note": "seamless"},
                limbs={"anatomical": True, "note": "stable"},
                wander=None, bar=0.35, min_motion=0.02, loop=True)
    base.update(over)
    return base


class VerdictPassesOnlyWhenEverythingHolds(unittest.TestCase):
    def setUp(self):
        from ball_reel.produce import verdict

        self.v = verdict

    def test_all_clear_passes_and_scores_the_median(self):
        passed, score, reason = self.v(**_good())
        self.assertTrue(passed)
        self.assertEqual(score, 0.20)
        self.assertEqual(reason, "")

    def test_nothing_judgeable_is_not_a_pass(self):
        # The mutation that survived: with MIN_COVERAGE removed, a clip nobody
        # could verify would sail through as if it had been checked.
        passed, score, reason = self.v(**_good(drift=_drift(coverage=0.1)))
        self.assertFalse(passed)
        self.assertEqual(score, 1.0)
        self.assertIn("not verifiable", reason)

    def test_an_unmeasurable_median_is_not_a_pass(self):
        passed, _, reason = self.v(**_good(drift=_drift(median=None)))
        self.assertFalse(passed)
        self.assertIn("not verifiable", reason)

    def test_identity_drift_past_the_bar_is_named(self):
        passed, _, reason = self.v(**_good(drift=_drift(median=0.50)))
        self.assertFalse(passed)
        self.assertIn("identity drift", reason)

    def test_identity_that_falls_apart_late_is_caught_by_p90(self):
        passed, _, reason = self.v(**_good(drift=_drift(median=0.20, p90=0.90)))
        self.assertFalse(passed)
        self.assertIn("p90", reason)

    def test_a_frozen_clip_is_rejected_for_motion(self):
        passed, _, reason = self.v(**_good(motion=0.0))
        self.assertFalse(passed)
        self.assertIn("motion", reason)

    def test_teleporting_motion_is_rejected_as_not_physical(self):
        passed, _, reason = self.v(
            **_good(quality={"smooth": False, "note": "limbs teleport"}))
        self.assertFalse(passed)
        self.assertIn("not physical", reason)

    def test_rubber_limbs_are_rejected_as_not_anatomical(self):
        passed, _, reason = self.v(
            **_good(limbs={"anatomical": False, "note": "body is stretching"}))
        self.assertFalse(passed)
        self.assertIn("not anatomical", reason)

    def test_a_wandering_pose_is_rejected_when_a_reference_was_given(self):
        passed, _, reason = self.v(
            **_good(wander={"held": False, "note": "DRIFTED"}))
        self.assertFalse(passed)
        self.assertIn("pose wandered", reason)

    def test_no_pose_reference_means_no_pose_complaint(self):
        self.assertTrue(self.v(**_good(wander=None))[0])

    def test_a_visible_seam_is_rejected_when_a_loop_was_asked_for(self):
        passed, _, reason = self.v(
            **_good(seam={"seamless": False, "note": "visible cut"}))
        self.assertFalse(passed)
        self.assertIn("does not loop", reason)

    def test_the_same_seam_is_fine_when_no_loop_was_asked_for(self):
        self.assertTrue(self.v(**_good(
            seam={"seamless": False, "note": "visible cut"}, loop=False))[0])

    def test_identity_is_reported_before_the_cosmetic_failures(self):
        # Everything is broken at once. The reason must be the one to fix
        # first — a clip of the wrong person is not a looping problem.
        passed, _, reason = self.v(**_good(
            drift=_drift(median=0.9), motion=0.0,
            quality={"smooth": False, "note": "x"},
            seam={"seamless": False, "note": "y"},
            limbs={"anatomical": False, "note": "z"}))
        self.assertFalse(passed)
        self.assertIn("identity drift", reason)


class MotionQualitySeesNearlyStaticClips(unittest.TestCase):
    """Второй выживший мутант: STILL_MIN ничем не сторожился.

    Полностью замерший клип ловился ранним возвратом (медианный шаг ровно 0),
    а вот «почти не двигается» — тот случай, ради которого порог и существует, —
    не проверялся ни одним тестом.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        from ball_reel import motion

        self.m = motion
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _frames(self, values):
        from PIL import Image

        out = []
        for i, v in enumerate(values):
            p = self.dir / f"{i:02d}.png"
            Image.new("L", (96, 96), v).save(p)
            out.append(str(p))
        return out

    def test_a_barely_moving_clip_is_not_moving(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("numpy/Pillow not installed")
        # Bright frames, drifting by one grey level: there IS a nonzero step,
        # so the early return does not fire, and STILL_MIN is what decides.
        q = self.m.motion_quality(self._frames([200, 201, 200, 201, 200, 201]))
        self.assertFalse(q["moving"])

    def test_a_clearly_moving_clip_is_moving(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("numpy/Pillow not installed")
        q = self.m.motion_quality(self._frames([100, 130, 160, 190, 160, 130]))
        self.assertTrue(q["moving"])


if __name__ == "__main__":
    unittest.main()
