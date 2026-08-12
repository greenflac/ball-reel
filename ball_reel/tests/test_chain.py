"""Keyframe selection and chain guards, offline.

Rendering and video generation need the network; what decides WHERE the motion
gets pinned is arithmetic, and that is what is checked here. Picking the wrong
frames is the failure that would quietly make the whole scheme decorative — a
chain whose nodes all sit on the way up constrains nothing.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def _track(heights):
    return [{"hip_height": h} for h in heights]


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (live extra)")
class KeyframesLandOnTheTurningPoints(unittest.TestCase):
    def setUp(self):
        from ball_reel import chain

        self.c = chain

    def test_extremes_of_a_bounce_are_chosen(self):
        # Two full bounces: the tops and bottoms are what define the movement,
        # and evenly spaced sampling could miss every one of them.
        chosen = self.c.pick_keyframe_times(
            _track([0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]), 5)
        for turning_point in (2, 4, 6):
            self.assertIn(turning_point, chosen)
        self.assertEqual(chosen[0], 0)  # so the chain can close onto the start

    def test_a_monotonic_track_falls_back_to_even_spacing(self):
        chosen = self.c.pick_keyframe_times(_track([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), 4)
        self.assertGreaterEqual(len(chosen), 2)
        self.assertEqual(chosen, sorted(set(chosen)))

    def test_the_count_is_respected(self):
        chosen = self.c.pick_keyframe_times(
            _track([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]), 4)
        self.assertLessEqual(len(chosen), 4)

    def test_frames_without_a_measurable_hip_are_never_chosen(self):
        track = _track([0.0, 0.5, 1.0, 0.5, 0.0])
        for i in (1, 3):
            track[i]["hip_height"] = None
        chosen = self.c.pick_keyframe_times(track, 5)
        for i in chosen:
            self.assertIsNotNone(track[i]["hip_height"])

    def test_an_unmeasurable_track_yields_nothing_rather_than_guesses(self):
        self.assertEqual(self.c.pick_keyframe_times([{"hip_height": None}] * 5, 4), [])


class ChainRefusesToPretend(unittest.TestCase):
    def setUp(self):
        from ball_reel import chain

        self.c = chain

    def test_a_model_without_end_frame_is_rejected(self):
        # Such a model ignores the second keyframe silently, degrading the
        # chain to ordinary unconstrained generation while looking fine.
        from ball_reel.chain import Keyframe

        kfs = [Keyframe(0, 0.0, "a", url="u1", accepted=True),
               Keyframe(1, 1.0, "b", url="u2", accepted=True)]
        with self.assertRaises(ValueError) as ctx:
            self.c.generate_chain(kfs, "x", "out", model="seedance-pro")
        self.assertIn("end_frame", str(ctx.exception))

    def test_too_few_accepted_keyframes_produces_no_clip(self):
        from ball_reel.chain import Keyframe

        kfs = [Keyframe(0, 0.0, "a", url="u1", accepted=True),
               Keyframe(1, 1.0, "b", accepted=False, reason="pose off")]
        res = self.c.generate_chain(kfs, "x", "out", model="wan-fast")
        self.assertEqual(res.clip_path, "")
        self.assertIn("nothing to chain", res.note)

    def test_every_listed_model_actually_supports_end_frame(self):
        # Pinned against the live capability listing; if this list drifts, the
        # constraint silently stops being one.
        self.assertEqual(set(self.c.END_FRAME_MODELS),
                         {"wan-fast", "veo", "wan-pro", "seedance-2.0"})



class GPUPlanFitsTheCard(unittest.TestCase):
    """The 4 GB configuration, checkable without a GPU."""

    def setUp(self):
        from ball_reel import gpu_keyframes

        self.g = gpu_keyframes

    def test_the_default_plan_fits_four_gigabytes(self):
        p = self.g.plan(vram_gb=4.0)
        self.assertLess(p.estimated_vram_gb, 4.0)
        # Every saving must be on: at this size they are not optional tuning.
        for opt in ("attention_slicing", "vae_slicing", "vae_tiling",
                    "model_cpu_offload"):
            self.assertIn(opt, p.optimisations)

    def test_a_smaller_card_is_told_so_rather_than_left_to_crash(self):
        p = self.g.plan(vram_gb=3.0)
        self.assertTrue(any("below" in n for n in p.notes))

    def test_a_bigger_card_is_offered_continuous_control(self):
        # Keyframes are a workaround for small VRAM; on a card that fits a video
        # model the honest advice is to stop working around it.
        p = self.g.plan(vram_gb=12.0)
        self.assertTrue(any("CONTINUOUS" in n for n in p.notes))
        self.assertGreater(p.width, 512)

    def test_every_plan_admits_it_has_never_been_executed(self):
        # This module was written without a GPU. That has to travel with it.
        self.assertTrue(any("UNVERIFIED" in n for n in self.g.plan().notes))

    def test_keyframe_count_is_reported_as_the_accuracy_dial(self):
        p = self.g.plan(keyframes=9)
        self.assertTrue(any("9 keyframes" in n for n in p.notes))


if __name__ == "__main__":
    unittest.main()
