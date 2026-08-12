"""The driving-spec arithmetic and its honesty, offline.

Video decoding and landmark detection need ffmpeg and a model; the parts that
decide what the spec CLAIMS — cadence, hip height, validation — are arithmetic
and are checked here. The validation tests matter most: a spec that quietly
reports an unmeasured expression as neutral is worse than no spec, so the tests
pin that it refuses to.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class CadenceCountsRepetitions(unittest.TestCase):
    def setUp(self):
        from ball_reel import driving

        self.d = driving

    def _sine(self, hz, seconds, fps=12):
        import math

        return [math.sin(2 * math.pi * hz * i / fps)
                for i in range(int(seconds * fps))]

    def test_a_short_window_under_reports_by_half_a_cycle(self):
        # Documented bias, pinned rather than papered over: N cycles show
        # 2N-1 crossings, so 1 Hz over 2 s reads 0.75. Correcting it would
        # invent a rhythm in data that merely trends.
        self.assertAlmostEqual(self.d._cadence(self._sine(1, 2), 12), 0.75, places=2)

    def test_a_longer_window_is_close_to_the_truth(self):
        # The error is bounded, not monotonic: the count depends on the phase
        # the window happens to end on, so 6 s can land exactly on 1.00 while
        # 12 s reads 0.96. What holds is that a short window is badly off and a
        # long one is not — which is the reason the docstring says to measure
        # over a longer segment.
        self.assertGreater(abs(self.d._cadence(self._sine(1, 2), 12) - 1.0), 0.2)
        for seconds in (6, 8, 12):
            self.assertLess(abs(self.d._cadence(self._sine(1, seconds), 12) - 1.0),
                            0.1, f"{seconds}s window drifted too far")

    def test_faster_motion_reads_faster(self):
        slow = self.d._cadence(self._sine(1, 6), 12)
        fast = self.d._cadence(self._sine(2, 6), 12)
        self.assertGreater(fast, slow * 1.8)

    def test_a_monotonic_trend_is_not_reported_as_a_rhythm(self):
        # A ramp crosses the mean exactly once. It must not become a cadence.
        ramp = [i / 24 for i in range(24)]
        self.assertLessEqual(self.d._cadence(ramp, 12), 0.3)

    def test_a_flat_series_has_no_cadence_rather_than_noise(self):
        # Without the variance floor, rounding jitter around the mean would be
        # counted as crossings and invent a cadence out of a static clip.
        self.assertEqual(self.d._cadence([0.5] * 24, 12), 0.0)

    def test_too_few_samples_is_unmeasurable(self):
        self.assertIsNone(self.d._cadence([0.1, 0.2], 12))


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class HeightUsesOneScalePerSegment(unittest.TestCase):
    """Regression: a per-frame scale exploded on real footage.

    Dividing each frame's hip height by THAT frame's torso length looks
    scale-invariant, but projected torso length collapses when the person bends
    or lies down. On a real stretching video that produced an amplitude of 5.1
    torso-lengths, which is not a thing a body does. The scale is now one
    constant per segment.
    """

    def setUp(self):
        from ball_reel import driving

        self.d = driving

    def _pts(self, hip_y, torso=0.25):
        return {"l_hip": (0.46, hip_y, 1.0), "r_hip": (0.54, hip_y, 1.0),
                "l_shoulder": (0.45, hip_y - torso, 1.0),
                "r_shoulder": (0.55, hip_y - torso, 1.0)}

    def test_raw_reports_height_and_torso_without_dividing(self):
        hip_y, torso = self.d._hip_raw(self._pts(0.60, torso=0.25))
        self.assertAlmostEqual(hip_y, 0.60, places=6)
        self.assertAlmostEqual(torso, 0.25, places=6)

    def test_missing_hips_is_unmeasurable(self):
        pts = self._pts(0.5)
        pts["l_hip"] = (0.46, 0.5, 0.0)
        self.assertIsNone(self.d._hip_raw(pts))

    def test_a_foreshortened_frame_cannot_set_the_scale(self):
        # One bent frame with a near-zero torso among upright ones: the median
        # must ignore it. Under the old per-frame division that single frame
        # produced the impossible amplitude.
        torsos = [0.25, 0.25, 0.26, 0.24, 0.001, 0.25]
        self.assertAlmostEqual(self.d._segment_scale(torsos), 0.25, places=2)

    def test_scale_tracks_camera_distance(self):
        near = self.d._segment_scale([0.40] * 5)
        far = self.d._segment_scale([0.20] * 5)
        # Same posture at two camera distances: dividing by each segment's own
        # scale puts both back on the same footing.
        self.assertAlmostEqual((0.60 / near) , (0.30 / far), places=6)

    def test_too_few_usable_frames_gives_no_scale(self):
        self.assertIsNone(self.d._segment_scale([0.25, 0.0]))


class SpecRefusesToOverclaim(unittest.TestCase):
    """The spec must name what it did NOT measure."""

    def setUp(self):
        from ball_reel import driving

        self.d = driving

    def _spec(self, **kw):
        s = self.d.DrivingSpec(source="x", fps=12, frames=kw.pop("frames", 60),
                               duration=5.0)
        s.pose_coverage = kw.pop("pose_coverage", 1.0)
        s.expression.coverage = kw.pop("expression_coverage", 1.0)
        s.motion.cadence_hz = kw.pop("cadence", 1.0)
        s.motion.amplitude = kw.pop("amplitude", 0.5)
        return s

    def test_an_unread_face_is_reported_not_assumed_neutral(self):
        problems = self._spec(expression_coverage=0.0).validate()
        self.assertTrue(any("expression is NOT specified" in p for p in problems))

    def test_a_holey_pose_track_is_flagged(self):
        problems = self._spec(pose_coverage=0.5).validate()
        self.assertTrue(any("motion track has holes" in p for p in problems))

    def test_no_periodic_motion_is_flagged(self):
        problems = self._spec(cadence=None).validate()
        self.assertTrue(any("no periodic motion" in p for p in problems))

    def test_a_short_clip_is_flagged(self):
        problems = self._spec(frames=3).validate()
        self.assertTrue(any("too short" in p for p in problems))

    def test_camera_motion_caveat_is_always_carried(self):
        # Not conditional: nothing in the pipeline separates camera from
        # subject, so every spec must say so, including the clean ones.
        s = self._spec()
        s.warnings.append("camera motion is not separated from subject motion: "
                          "amplitude and cadence assume a locked-off camera.")
        self.assertTrue(any("camera motion" in p for p in s.validate()))

    def test_to_prompt_omits_what_was_not_measured(self):
        s = self._spec(expression_coverage=0.0)
        text = self.d.to_prompt(s)
        self.assertIn("bounces per second", text)
        for invented in ("smil", "eyebrow"):
            self.assertNotIn(invented, text.lower())

    def test_to_prompt_is_empty_when_nothing_was_measured(self):
        s = self._spec(cadence=None, amplitude=None, expression_coverage=0.0)
        self.assertEqual(self.d.to_prompt(s), "")

    def test_the_track_can_be_dropped_for_a_compact_spec(self):
        s = self._spec()
        s.track = [{"i": 0}]
        self.assertIn("track", s.to_dict())
        self.assertNotIn("track", s.to_dict(with_track=False))



class PromptRendersOnlyMeasuredExpression(unittest.TestCase):
    """Ветка мимики в to_prompt: описывать только то, что реально измерено."""

    def setUp(self):
        from ball_reel import driving

        self.d = driving

    def _spec(self, coverage, peaks):
        s = self.d.DrivingSpec(source="x", fps=12, frames=60, duration=5.0)
        s.motion.cadence_hz, s.motion.amplitude = 1.0, 0.5
        s.motion.velocity, s.motion.peak_velocity = 1.0, 1.5
        s.expression.coverage, s.expression.peaks = coverage, peaks
        return s

    def test_a_broad_smile_is_described(self):
        text = self.d.to_prompt(self._spec(1.0, {"mouthSmileLeft": 0.7}))
        self.assertIn("smiling broadly", text)

    def test_a_faint_smile_is_described_as_faint(self):
        text = self.d.to_prompt(self._spec(1.0, {"mouthSmileRight": 0.25}))
        self.assertIn("slight smile", text)

    def test_raised_brows_are_described(self):
        text = self.d.to_prompt(self._spec(1.0, {"browInnerUp": 0.7}))
        self.assertIn("eyebrows raised", text)

    def test_an_unread_face_contributes_nothing(self):
        # Low coverage means the video did not specify the expression; the
        # prompt must not invent one from peaks measured on a couple of frames.
        text = self.d.to_prompt(self._spec(0.1, {"mouthSmileLeft": 0.9}))
        self.assertNotIn("smil", text.lower())

    def test_an_even_pace_reads_as_even(self):
        s = self._spec(1.0, {})
        s.motion.peak_velocity = 1.2      # ratio 1.2 -> not "sharp"
        self.assertIn("even pace", self.d.to_prompt(s))


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class SurveyFindsWhereTheMovementIs(unittest.TestCase):
    """Обзор длинного ролика: где какое движение, до дорогой нарезки.

    Реальный референс — программа из разных упражнений: на живом 112-секундном
    ролике размах по окнам разошёлся от 0.12 до 1.11. Один extract() на всю
    длину усреднил бы их в число, не описывающее ни одно.
    """

    def setUp(self):
        from ball_reel import driving

        self.d = driving
        self.heights = []
        self._saved = (driving._sample_frames, None)
        driving._sample_frames = lambda v, d, **kw: [
            str(i) for i in range(len(self.heights))]

        import ball_reel.pose as pose

        self._real_landmarks = pose.landmarks
        pose.landmarks = lambda p: (None if self.heights[int(p)] is None
                                    else {"h": self.heights[int(p)]})
        driving._hip_raw = lambda pts: (None if pts is None
                                        else (pts["h"], 0.25))

        def restore():
            driving._sample_frames = self._saved[0]
            pose.landmarks = self._real_landmarks

        self.addCleanup(restore)
        self._real_hip_raw = driving._hip_raw

    def test_a_lively_window_reads_higher_than_a_quiet_one(self):
        # окно 0: качается; окно 1: почти стоит
        self.heights = [0.2, 0.6, 0.2, 0.6] + [0.4, 0.41, 0.4, 0.41]
        rows = self.d.survey("v.mp4", fps=2, window=2.0)
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[0]["hip_range"], rows[1]["hip_range"] * 5)

    def test_windows_are_stamped_with_their_start_time(self):
        self.heights = [0.3] * 8
        rows = self.d.survey("v.mp4", fps=2, window=2.0)
        self.assertEqual([r["start"] for r in rows], [0.0, 2.0])

    def test_frames_without_a_body_lower_coverage(self):
        self.heights = [0.3, None, 0.5, 0.4]
        rows = self.d.survey("v.mp4", fps=2, window=2.0)
        self.assertLess(rows[0]["coverage"], 1.0)

    def test_a_video_with_no_body_at_all_surveys_to_nothing(self):
        self.heights = [None] * 6
        self.assertEqual(self.d.survey("v.mp4", fps=2, window=2.0), [])


if __name__ == "__main__":
    unittest.main()
