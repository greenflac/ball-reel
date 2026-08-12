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



@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class BuildIsMeasuredIn3DNotProjection(unittest.TestCase):
    """Regression: build measured in image space inverted on a turned photo.

    Live, the same measurement gave shoulder width 0.143 on a three-quarter
    photo and 1.019 on a frontal one — a sevenfold disagreement produced purely
    by camera angle. In 3D the same two photos, of two DIFFERENT people, gave
    0.638 and 0.643. Users rarely send front-on photos, so the 3D path is the
    one that has to work.
    """

    def setUp(self):
        from ball_reel import pose

        self.p = pose
        self._real = pose.world_landmarks
        self.addCleanup(setattr, pose, "world_landmarks", self._real)

    def _stub(self, points):
        self.p.world_landmarks = lambda _p: points

    def _body(self, shoulder_half=0.20, hip_half=0.13, torso=0.50):
        # A skeleton in metres: shoulders and hips spread on x, torso along y.
        pts = {}
        for side, sign in (("l", -1), ("r", 1)):
            pts[f"{side}_shoulder"] = (sign * shoulder_half, -torso / 2, 0.0, 1.0)
            pts[f"{side}_hip"] = (sign * hip_half, torso / 2, 0.0, 1.0)
            pts[f"{side}_elbow"] = (sign * shoulder_half, -torso / 6, 0.0, 1.0)
            pts[f"{side}_wrist"] = (sign * shoulder_half, torso / 6, 0.0, 1.0)
            pts[f"{side}_knee"] = (sign * hip_half, torso, 0.0, 1.0)
            pts[f"{side}_ankle"] = (sign * hip_half, torso * 1.5, 0.0, 1.0)
        return pts

    def test_proportions_are_expressed_in_torso_lengths(self):
        self._stub(self._body())
        got = self.p.world_proportions("x")
        self.assertAlmostEqual(got["shoulder_width"], 0.40 / 0.50, places=3)
        self.assertAlmostEqual(got["hip_width"], 0.26 / 0.50, places=3)
        self.assertAlmostEqual(got["shoulder_to_hip"], 0.40 / 0.26, places=3)

    def test_the_same_body_further_from_camera_measures_the_same(self):
        # 3D landmarks are metric, so a scaled skeleton (the model's estimate of
        # a smaller or more distant person) yields identical ratios.
        self._stub(self._body())
        near = self.p.world_proportions("x")
        self._stub(self._body(shoulder_half=0.10, hip_half=0.065, torso=0.25))
        far = self.p.world_proportions("x")
        self.assertAlmostEqual(near["shoulder_to_hip"], far["shoulder_to_hip"],
                               places=3)

    def test_a_broader_build_reads_as_broader(self):
        self._stub(self._body(shoulder_half=0.20))
        lean = self.p.world_proportions("x")["shoulder_to_hip"]
        self._stub(self._body(shoulder_half=0.30))
        broad = self.p.world_proportions("x")["shoulder_to_hip"]
        self.assertGreater(broad, lean)

    def test_invisible_joints_are_left_out(self):
        pts = self._body()
        pts["l_knee"] = (*pts["l_knee"][:3], 0.0)
        self._stub(pts)
        got = self.p.world_proportions("x")
        self.assertNotIn("l_hip->l_knee", got)
        self.assertIn("r_hip->r_knee", got)

    def test_no_body_gives_nothing_rather_than_zeros(self):
        self._stub(None)
        self.assertIsNone(self.p.world_proportions("x"))


if __name__ == "__main__":
    unittest.main()
