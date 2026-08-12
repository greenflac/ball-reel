"""The identity VERDICT logic, tested without the 300 MB model.

`test_arcface_math` pins the cosine distance; this pins what the pipeline does
with a whole clip of them — which is where the real calls live: what counts as
judgeable, what the verdict rests on, and what "cannot verify" means.

The model is stubbed, not mocked away for convenience: `face_detail` is the one
seam that needs weights, so it is replaced with synthetic faces whose sizes and
embeddings reproduce profiles actually measured on live clips (recorded in
POLLINATIONS_CONTRACT.md). Everything downstream is the real code.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def _emb(drift: float):
    """A 3-D unit vector at cosine distance `drift` from the reference [1,0,0]."""
    import numpy as np

    cos = 1.0 - drift
    return np.array([cos, (1.0 - cos ** 2) ** 0.5, 0.0])


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class VerdictRestsOnTheJudgeableMass(unittest.TestCase):
    """Live-measured clip profiles must produce the verdicts we reasoned to."""

    def setUp(self):
        from ball_reel import identity_arcface

        self.mod = identity_arcface
        self._real = identity_arcface.face_detail
        self.faces: dict[str, tuple[int, float]] = {}

        def fake_face_detail(path):
            key = str(path)
            if key not in self.faces:
                return None
            px, drift = self.faces[key]
            return {"embedding": _emb(drift), "face_px": px, "det_score": 0.87}

        identity_arcface.face_detail = fake_face_detail
        self.addCleanup(setattr, identity_arcface, "face_detail", self._real)
        self.faces["ref"] = (147, 0.0)

    def _clip(self, profile):
        for i, (px, drift) in enumerate(profile):
            self.faces[f"f{i}"] = (px, drift)
        return [f"f{i}" for i in range(len(profile))]

    def test_blur_spike_does_not_sink_a_true_identity_clip(self):
        # Live clip A: the real person, faces 111-114px, but one frame at the top
        # of the jump blurs to 0.71. Judging by the worst frame would reject the
        # clip; the median is what carries identity.
        frames = self._clip([(112, 0.176), (112, 0.177), (112, 0.205),
                             (113, 0.219), (111, 0.362), (112, 0.714),
                             (114, 0.426), (114, 0.221)])
        d = self.mod.arcface_drift(frames, "ref")
        self.assertEqual(d["coverage"], 1.0)
        self.assertLessEqual(d["median"], self.mod.SAME_PERSON_MAX)
        self.assertGreater(d["worst"][1], self.mod.SAME_PERSON_MAX)  # the trap
        self.assertLessEqual(d["p90"], self.mod.HARD_DRIFT_MAX)

    def test_small_faces_are_not_verifiable_rather_than_drifted(self):
        # Live clip B: faces 64-86px. "Too small to tell" is a different claim
        # from "different person", and must not be reported as drift.
        frames = self._clip([(66, 0.63), (73, 0.64), (67, 0.44), (68, 0.68)])
        d = self.mod.arcface_drift(frames, "ref")
        self.assertIsNone(d["median"])
        self.assertEqual(d["coverage"], 0.0)
        self.assertEqual(len(d["too_small"]), 4)
        self.assertEqual(d["drifted"], [])
        self.assertIn("NOT VERIFIABLE", d["note"])

    def test_a_clip_that_turns_into_someone_else_is_caught_by_p90(self):
        # The median alone would forgive a clip whose back half is another
        # person; p90 is what stops that.
        frames = self._clip([(112, 0.18)] * 6 + [(112, 0.75)] * 3)
        d = self.mod.arcface_drift(frames, "ref")
        self.assertLessEqual(d["median"], self.mod.SAME_PERSON_MAX)
        self.assertGreater(d["p90"], self.mod.HARD_DRIFT_MAX)

    def test_start_frame_floor_accepts_the_known_good_still(self):
        # Regression: the start frame that produced a fully passing clip had a
        # 91px face. Judging stills at the video floor rejected it, which cost
        # two live attempts before it was caught.
        self.faces["start"] = (91, 0.1386)
        at_video_floor = self.mod.arcface_drift(["start"], "ref")
        self.assertIsNone(at_video_floor["median"])  # what went wrong
        at_still_floor = self.mod.arcface_drift(
            ["start"], "ref", min_face_px=self.mod.START_MIN_FACE_PX)
        self.assertIsNotNone(at_still_floor["median"])
        self.assertLessEqual(at_still_floor["median"], self.mod.SAME_PERSON_MAX)

    def test_a_reference_photo_too_small_to_identify_from_says_so(self):
        self.faces["ref"] = (40, 0.0)
        d = self.mod.arcface_drift(self._clip([(112, 0.18)]), "ref")
        self.assertIsNone(d["median"])
        self.assertIn("too small", d["note"])

    def test_no_face_anywhere_is_reported_not_scored(self):
        d = self.mod.arcface_drift(["missing_a", "missing_b"], "ref")
        self.assertEqual(len(d["no_face"]), 2)
        self.assertIsNone(d["median"])
        self.assertIn("NOT VERIFIABLE", d["note"])


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class QuantileIsRight(unittest.TestCase):
    def setUp(self):
        from ball_reel import identity_arcface

        self.q = identity_arcface._quantile

    def test_median_of_odd_and_even_counts(self):
        self.assertEqual(self.q([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertEqual(self.q([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)

    def test_single_value_is_every_quantile(self):
        self.assertEqual(self.q([0.42], 0.5), 0.42)
        self.assertEqual(self.q([0.42], 0.9), 0.42)

    def test_empty_is_max_distance_not_a_free_pass(self):
        self.assertEqual(self.q([], 0.5), 1.0)


if __name__ == "__main__":
    unittest.main()
