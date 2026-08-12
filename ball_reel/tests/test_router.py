"""The stack decision, offline.

The two errors this guards against are not symmetric. Sending an easy job to a
GPU wastes money loudly. Asking the public gateway for a trajectory it cannot
honour is quiet: it returns a confident clip of the wrong motion, and nothing
in the output says so. The second is what most of these tests are about.
"""

from __future__ import annotations

import unittest


class _Spec:
    """Stand-in for a DrivingSpec: only what the router reads."""

    def __init__(self, coverage=1.0, problems=()):
        self.pose_coverage = coverage
        self._problems = list(problems)

    def validate(self):
        return self._problems


def _subject(face_px=148, has_body=True, turned=False):
    return {
        "face": {"identity": {"face_px": face_px} if face_px else {}},
        "body": {"proportions": {"shoulder_width": 0.64} if has_body else {},
                 "turned": turned},
        "missing": [],
    }


class TheCheapStackIsChosenWhenItSuffices(unittest.TestCase):
    def setUp(self):
        from ball_reel import router

        self.r = router

    def test_plain_identity_job_stays_on_the_public_gateway(self):
        d = self.r.choose(("identity",), subject=_subject(), video_model="veo")
        self.assertEqual(d.stack, "pollinations")
        self.assertTrue(d.runnable)

    def test_cost_is_estimated_from_the_live_price_list(self):
        d = self.r.choose(("identity",), subject=_subject(),
                          video_model="wan-fast", seconds=5)
        self.assertAlmostEqual(d.estimated_pollen, 0.05, places=3)

    def test_a_loop_on_a_model_without_end_frame_is_blocked_before_spending(self):
        d = self.r.choose(("identity", "loop"), subject=_subject(),
                          video_model="seedance-pro", loop=True)
        self.assertFalse(d.runnable)
        self.assertTrue(any("end keyframe" in b for b in d.blockers))


class TrajectoryForcesTheExpensiveStack(unittest.TestCase):
    def setUp(self):
        from ball_reel import router

        self.r = router

    def test_pose_trajectory_never_routes_to_the_public_gateway(self):
        # Measured: multi-reference transfers face and build but not pose, and
        # the gateway has no per-frame conditioning at all. Routing this to the
        # API would return a plausible clip of the wrong motion.
        d = self.r.choose(("identity", "pose_trajectory"), subject=_subject(),
                          spec=_Spec())
        self.assertEqual(d.stack, "gpu")
        self.assertIsNone(d.estimated_pollen)

    def test_object_fidelity_also_forces_gpu(self):
        d = self.r.choose(("object_fidelity",), subject=_subject(), spec=_Spec())
        self.assertEqual(d.stack, "gpu")

    def test_trajectory_without_a_driving_spec_is_blocked(self):
        d = self.r.choose(("pose_trajectory",), subject=_subject(), spec=None)
        self.assertFalse(d.runnable)
        self.assertTrue(any("no trajectory" in b for b in d.blockers))

    def test_a_holey_track_cannot_drive_a_trajectory(self):
        d = self.r.choose(("pose_trajectory",), subject=_subject(),
                          spec=_Spec(coverage=0.6))
        self.assertFalse(d.runnable)
        self.assertTrue(any("gaps" in b for b in d.blockers))


class InputsAreCheckedBeforeAnythingIsSpent(unittest.TestCase):
    def setUp(self):
        from ball_reel import router

        self.r = router

    def test_no_face_blocks_the_job(self):
        d = self.r.choose(("identity",), subject=_subject(face_px=None))
        self.assertFalse(d.runnable)

    def test_a_small_source_face_warns_about_the_output_gate(self):
        # Not a blocker: it will generate. The point is that the verdict will
        # likely be "not verifiable", which is worth knowing beforehand.
        d = self.r.choose(("identity",), subject=_subject(face_px=72), loop=False)
        self.assertTrue(d.runnable)
        self.assertTrue(any("not verifiable" in w for w in d.warnings))

    def test_build_without_a_body_is_blocked(self):
        d = self.r.choose(("identity", "build"),
                          subject=_subject(has_body=False))
        self.assertFalse(d.runnable)

    def test_a_turned_subject_is_a_warning_not_a_blocker(self):
        # 3D landmarks rescued build; identity across viewpoint is still harder.
        d = self.r.choose(("identity", "build"), subject=_subject(turned=True),
                          loop=False)
        self.assertTrue(d.runnable)
        self.assertTrue(any("turned" in w for w in d.warnings))

    def test_unmeasured_inputs_are_flagged_rather_than_assumed_fine(self):
        d = self.r.choose(("identity",))
        self.assertTrue(any("not measured" in w for w in d.warnings))

    def test_a_missing_expression_does_not_block_a_job_that_never_claimed_it(self):
        spec = _Spec(problems=["face readable in only 0% of frames: expression "
                               "is NOT specified by this video — do not claim it is."])
        d = self.r.choose(("identity",), subject=_subject(), spec=spec,
                          loop=False)
        self.assertTrue(d.runnable)
        self.assertTrue(any("expression is NOT specified" in w for w in d.warnings))

    def test_the_defaults_do_not_block_themselves(self):
        # Regression: looping was on by default while the default model could
        # not take an end keyframe, so the out-of-the-box call blocked itself.
        d = self.r.choose(("identity",), subject=_subject())
        self.assertTrue(d.runnable, d.blockers)

    def test_an_unknown_capability_is_refused_loudly(self):
        with self.assertRaises(ValueError):
            self.r.choose(("teleportation",))


if __name__ == "__main__":
    unittest.main()
