"""Loop/motion arithmetic and subject rendering, offline.

The motion measures are ratios against a clip's own median step, so they can be
checked exactly on synthetic frame sequences — no model, no network, no video.
Frames are written as tiny PNGs, which is what the real code reads anyway.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _write(dirpath: Path, name: str, value: int, side: int = 96) -> str:
    """A flat grey frame — brightness stands in for "where the subject is"."""
    from PIL import Image

    p = dirpath / name
    Image.new("L", (side, side), value).save(p)
    return str(p)


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class LoopAndMotionMeasures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        from ball_reel import motion

        self.m = motion

    def _seq(self, values):
        return [_write(self.dir, f"{i:02d}.png", v) for i, v in enumerate(values)]

    def test_a_clip_that_returns_to_its_start_is_seamless(self):
        # out and back: the last frame equals the first, so the repeat is invisible
        frames = self._seq([100, 110, 120, 130, 120, 110, 100])
        r = self.m.loop_seam(frames)
        self.assertEqual(r["seam"], 0.0)
        self.assertTrue(r["seamless"])

    def test_a_clip_that_ends_far_from_its_start_is_not(self):
        frames = self._seq([100, 105, 110, 115, 120, 125, 130])
        r = self.m.loop_seam(frames)
        self.assertGreater(r["ratio"], self.m.SEAMLESS_MAX)
        self.assertFalse(r["seamless"])
        self.assertIn("visible cut", r["note"])

    def test_a_teleport_is_caught_even_though_the_clip_loops(self):
        # Ends where it began, but jumps hugely in the middle: a loop that is
        # still garbage. The two measures must not cover for each other.
        # Enough ordinary frames to set the median — a teleport makes TWO large
        # steps (in and out), so on a very short sequence it inflates the very
        # median it is measured against and hides itself. Real clips run 24+.
        frames = self._seq([100, 102, 104, 106, 108, 250,
                            108, 106, 104, 102, 100])
        self.assertTrue(self.m.loop_seam(frames)["seamless"])
        q = self.m.motion_quality(frames)
        self.assertFalse(q["smooth"])
        self.assertGreater(q["worst_jump"], self.m.JUMP_MAX)

    def test_smooth_motion_has_no_jumps(self):
        q = self.m.motion_quality(self._seq([100, 110, 120, 130, 140]))
        self.assertTrue(q["smooth"])
        self.assertEqual(q["jumps"], [])

    def test_a_static_clip_is_reported_as_static_not_smooth(self):
        q = self.m.motion_quality(self._seq([100] * 6))
        self.assertFalse(q["moving"])
        self.assertFalse(q["smooth"])
        self.assertIn("static", q["note"])

    def test_best_loop_cut_finds_the_return_point(self):
        # rises, comes back to the start at index 6, then wanders off again;
        # cutting at 6 is the loop, keeping the tail is not.
        frames = self._seq([100, 120, 140, 160, 140, 120, 100, 160, 180, 200])
        cut = self.m.best_loop_cut(frames)
        self.assertEqual(cut["cut_at"], 6)
        self.assertTrue(cut["seamless"])

    def test_best_loop_cut_never_returns_a_worse_clip(self):
        frames = self._seq([100, 110, 120, 130, 140, 150, 160, 170])
        cut = self.m.best_loop_cut(frames)
        self.assertLessEqual(cut["cut_at"], len(frames) - 1)
        self.assertGreaterEqual(cut["cut_at"], 2)


class SubjectRendersWhatWasSpecified(unittest.TestCase):
    """A subject must never invent an attribute nobody asked for."""

    def setUp(self):
        from ball_reel import subject

        self.s = subject

    def test_an_empty_subject_says_nothing(self):
        self.assertEqual(self.s.UNSPECIFIED.to_prompt(), "")
        self.assertEqual(self.s.UNSPECIFIED.specified, ())

    def test_only_set_fields_appear(self):
        sub = self.s.Subject(gender="woman", outfit="pink crop top")
        text = sub.to_prompt()
        self.assertIn("woman", text)
        self.assertIn("pink crop top", text)
        for absent in ("hair", "Skin", "Posture"):
            self.assertNotIn(absent, text)
        self.assertEqual(sub.specified, ("gender", "outfit"))

    def test_reference_images_pick_the_multi_reference_model(self):
        plain = self.s.Subject(build="heavy-set")
        self.assertEqual(plain.start_model(), self.s.SINGLE_REF_MODEL)
        with_body = self.s.Subject(body_ref="body.png")
        self.assertEqual(with_body.start_model(), self.s.MULTI_REF_MODEL)

    def test_reference_roles_are_ordered_face_first(self):
        sub = self.s.Subject(body_ref="b.png", pose_ref="p.png")
        roles = sub.reference_roles
        self.assertEqual(len(roles), 3)
        self.assertEqual(roles[0][0], "__face__")
        self.assertEqual(roles[1][0], "b.png")
        self.assertEqual(roles[2][0], "p.png")

    def test_reference_clause_names_each_image_by_position(self):
        clause = self.s.Subject(body_ref="b.png", pose_ref="p.png").reference_clause()
        self.assertIn("FIRST", clause)
        self.assertIn("SECOND", clause)
        self.assertIn("THIRD", clause)
        # the face must be pinned to image one, or a pose reference leaks a face
        self.assertIn("face must come only from the first image", clause)

    def test_no_clause_when_there_is_only_a_face(self):
        self.assertEqual(self.s.Subject().reference_clause(), "")

    def test_age_bands_read_as_decades(self):
        self.assertEqual(self.s._age_band(43), "in their early 40s")
        self.assertEqual(self.s._age_band(38), "in their late 30s")
        self.assertEqual(self.s._age_band(35), "in their mid 30s")


if __name__ == "__main__":
    unittest.main()
