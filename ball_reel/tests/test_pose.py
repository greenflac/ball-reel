"""Pose arithmetic, without MediaPipe.

`landmarks()` needs the model; everything the verdict rests on — normalisation,
distance, limb variation — is plain geometry and is checked here on synthetic
skeletons. The properties that matter are the ones a sceptic would attack:
that moving the camera does not read as moving the body, and that an occluded
joint contributes nothing rather than inventing disagreement.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def _skeleton(dx=0.0, dy=0.0, scale=1.0, **moved):
    """A plain standing skeleton, optionally shifted, resized, or bent.

    Coordinates are image-normalised like MediaPipe's. `moved` overrides any
    joint with an (x, y) or (x, y, visibility) tuple.
    """
    base = {
        "l_shoulder": (0.45, 0.30), "r_shoulder": (0.55, 0.30),
        "l_elbow": (0.42, 0.40), "r_elbow": (0.58, 0.40),
        "l_wrist": (0.40, 0.50), "r_wrist": (0.60, 0.50),
        "l_hip": (0.46, 0.55), "r_hip": (0.54, 0.55),
        "l_knee": (0.45, 0.70), "r_knee": (0.55, 0.70),
        "l_ankle": (0.44, 0.85), "r_ankle": (0.56, 0.85),
    }
    out = {}
    for name, (x, y) in base.items():
        out[name] = ((x - 0.5) * scale + 0.5 + dx, (y - 0.5) * scale + 0.5 + dy, 1.0)
    for name, val in moved.items():
        out[name] = val if len(val) == 3 else (val[0], val[1], 1.0)
    return out


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class PoseDistanceMeasuresConfigurationNotFraming(unittest.TestCase):
    def setUp(self):
        from ball_reel import pose

        self.p = pose

    def test_identical_poses_are_zero(self):
        self.assertEqual(self.p.pose_distance(_skeleton(), _skeleton()), 0.0)

    def test_moving_the_subject_across_frame_is_not_a_new_pose(self):
        # Same body, shifted: normalisation centres on the hips, so this must
        # not register — otherwise every reframe would read as a pose change.
        d = self.p.pose_distance(_skeleton(), _skeleton(dx=0.2, dy=-0.1))
        self.assertLess(d, 1e-6)

    def test_framing_closer_is_not_a_new_pose(self):
        # Scaled body: normalisation divides by torso length, so a closer crop
        # must not register either.
        d = self.p.pose_distance(_skeleton(), _skeleton(scale=1.8))
        self.assertLess(d, 1e-6)

    def test_a_bent_arm_is_diluted_by_the_mean_but_caught_by_the_worst_joint(self):
        # The exact failure mode that made `worst` necessary: one limb moved
        # somewhere else — "hands on hips" vs "arms down" — is 2 joints out of
        # 12, so the mean stays low while the wrist itself has travelled far.
        # If this ever flips, the gate has quietly stopped catching arm changes.
        bent = _skeleton(l_wrist=(0.30, 0.28), l_elbow=(0.35, 0.34))
        d = self.p.pose_delta(_skeleton(), bent)
        self.assertLess(d["mean"], self.p.SAME_POSE_MAX)
        self.assertGreater(d["worst"], self.p.WORST_JOINT_MAX)
        self.assertEqual(d["worst_joint"], "l_wrist")

    def test_invisible_joints_are_skipped_not_scored(self):
        # An occluded wrist carries no information. Hiding it in BOTH poses must
        # not change the answer for the joints that were actually seen.
        hidden_a = _skeleton(l_wrist=(0.40, 0.50, 0.1))
        hidden_b = _skeleton(l_wrist=(0.99, 0.99, 0.1))
        self.assertEqual(self.p.pose_distance(hidden_a, hidden_b), 0.0)

    def test_a_pose_without_hips_cannot_be_normalised(self):
        no_hips = _skeleton(l_hip=(0.46, 0.55, 0.0), r_hip=(0.54, 0.55, 0.0))
        self.assertIsNone(self.p.pose_distance(_skeleton(), no_hips))

    def test_the_two_bars_are_ordered_and_straddle_real_motion(self):
        # Calibrated live: motion within one clip reaches 0.30, a genuinely
        # different pose is 0.67. The still bar must sit below that motion and
        # the wander bar above it, or one of the two checks is meaningless.
        self.assertLess(self.p.SAME_POSE_MAX, 0.30)
        self.assertGreater(self.p.POSE_WANDER_MAX, 0.30)
        self.assertLess(self.p.POSE_WANDER_MAX, 0.67)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class LimbConsistencyDetectsRubberBodies(unittest.TestCase):
    def setUp(self):
        from ball_reel import pose

        self.p = pose
        self.frames = []

        def fake_landmarks(path):
            return self.frames[int(path)]

        self._real = pose.landmarks
        pose.landmarks = fake_landmarks
        self.addCleanup(setattr, pose, "landmarks", self._real)

    def _run(self):
        return self.p.limb_consistency([str(i) for i in range(len(self.frames))])

    def test_a_body_that_keeps_its_proportions_is_anatomical(self):
        # Same skeleton, moving across the frame and closer to camera: limb
        # lengths in torso units are unchanged, which is what a real body does.
        self.frames = [_skeleton(dx=i * 0.02, scale=1 + i * 0.05) for i in range(6)]
        r = self._run()
        self.assertTrue(r["anatomical"])
        self.assertLess(r["worst"][1], self.p.LIMB_WOBBLE_MAX)

    def test_a_stretching_forearm_is_caught(self):
        # The forearm grows frame by frame while everything else holds — the
        # signature of a generator losing the body.
        self.frames = [_skeleton(l_wrist=(0.40 - i * 0.06, 0.50 + i * 0.06))
                       for i in range(6)]
        r = self._run()
        self.assertFalse(r["anatomical"])
        self.assertIn("l_elbow->l_wrist", r["unstable"])
        self.assertIn("stretching", r["note"])

    def test_too_few_frames_is_not_verifiable_rather_than_pass(self):
        self.frames = [_skeleton()]
        r = self._run()
        self.assertFalse(r["anatomical"])
        self.assertIn("NOT VERIFIABLE", r["note"])


if __name__ == "__main__":
    unittest.main()
