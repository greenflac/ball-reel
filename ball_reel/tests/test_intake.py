"""Input validation, offline.

Intake decides what a photo licenses the pipeline to claim. The failure worth
guarding against is not a wrong verdict but a SILENT one: a photo that cannot
support a capability being waved through, so the gap surfaces later as a
confident-looking clip nobody can verify.
"""

from __future__ import annotations

import unittest


def _pkg(face_px=148, score=0.87, geometry=True, expression=True,
         has_body=True, reliable=True, turned=False):
    face = {"identity": {"face_px": face_px, "detector_score": score,
                         "embedding_dim": 512} if face_px else {},
            "geometry": {"face_width_to_height": 0.78} if geometry else {},
            "notes": []}
    if expression:
        face["expression"] = {"count": 52, "top": {}, "all": {"mouthSmileLeft": 0.5}}
    body = {"proportions": {"shoulder_to_hip": 1.5} if has_body else {},
            "projected": {}, "reliable": reliable, "turned": turned, "notes": []}
    return {"face": face, "body": body, "missing": []}


class IntakeSaysWhatAPhotoCanSupport(unittest.TestCase):
    def setUp(self):
        from ball_reel import intake

        self.i = intake
        self._real = intake.__dict__.get("subject_package")
        self.pkg = _pkg()
        import ball_reel.metrics as metrics

        self._real_metrics = metrics.subject_package
        metrics.subject_package = lambda p, **kw: self.pkg
        self.addCleanup(setattr, metrics, "subject_package", self._real_metrics)

    def test_a_good_photo_supports_everything(self):
        got = self.i.inspect("x.jpg")
        for cap in ("identity", "face_geometry", "expression", "build"):
            self.assertTrue(got.can(cap), cap)
        self.assertTrue(got.usable)

    def test_no_face_blocks_identity_and_names_the_fix(self):
        self.pkg = _pkg(face_px=None)
        got = self.i.inspect("x.jpg")
        self.assertFalse(got.usable)
        # The message must tell the sender what to do, not just what is wrong.
        self.assertIn("Send a photo", got.blocked["identity"])

    def test_a_tiny_face_is_blocked_rather_than_quietly_accepted(self):
        self.pkg = _pkg(face_px=40)
        got = self.i.inspect("x.jpg")
        self.assertFalse(got.can("identity"))
        self.assertIn("crop", got.blocked["identity"])

    def test_a_borderline_face_is_accepted_with_the_consequence_spelled_out(self):
        # 94px generated fine and then could not be verified — the case that
        # makes "accept, but warn" the right answer instead of pass or fail.
        self.pkg = _pkg(face_px=94)
        got = self.i.inspect("x.jpg")
        self.assertTrue(got.can("identity"))
        self.assertTrue(any("not verifiable" in w for w in got.warnings))

    def test_a_portrait_gets_an_assumed_build_rather_than_a_rejection(self):
        # Product decision: only a missing FACE stops a job. Most people have
        # portraits, not full-length photos, and a typical body passes unnoticed
        # where a wrong face never would.
        self.pkg = _pkg(has_body=False)
        got = self.i.inspect("x.jpg")
        self.assertTrue(got.can("build"))
        self.assertIn("build", got.assumed)
        self.assertEqual(got.measurements["build_source"], "assumed")
        self.assertTrue(any("assumed" in w for w in got.warnings))

    def test_an_assumed_build_is_never_shown_as_measured(self):
        self.pkg = _pkg(has_body=False)
        self.assertIn("build (assumed)", self.i.inspect("x.jpg").render())
        self.pkg = _pkg()
        got = self.i.inspect("x.jpg")
        self.assertEqual(got.assumed, [])
        self.assertEqual(got.measurements["build_source"], "measured")

    def test_a_turned_subject_still_supplies_build(self):
        # 3D landmarks survive the turn; blocking here would throw away a photo
        # that measures fine.
        self.pkg = _pkg(turned=True)
        got = self.i.inspect("x.jpg")
        self.assertTrue(got.can("build"))
        self.assertTrue(any("turned" in w for w in got.warnings))

    def test_a_mostly_hidden_body_also_falls_back_rather_than_blocking(self):
        self.pkg = _pkg(reliable=False)
        got = self.i.inspect("x.jpg")
        self.assertTrue(got.can("build"))
        self.assertIn("build", got.assumed)

    def test_no_mesh_means_no_expression_claim(self):
        self.pkg = _pkg(geometry=False, expression=False)
        got = self.i.inspect("x.jpg")
        self.assertFalse(got.can("expression"))
        self.assertFalse(got.can("face_geometry"))
        self.assertTrue(got.can("identity"))  # ArcFace still had the face

    def test_low_detector_confidence_is_surfaced(self):
        self.pkg = _pkg(score=0.4)
        got = self.i.inspect("x.jpg")
        self.assertTrue(any("confidence" in w for w in got.warnings))


class ReportPicksTheBestPhotoPerCapability(unittest.TestCase):
    def setUp(self):
        from ball_reel import intake
        import ball_reel.metrics as metrics

        self.i = intake
        self.by_path = {}
        self._real = metrics.subject_package
        metrics.subject_package = lambda p, **kw: self.by_path[str(p)]
        self.addCleanup(setattr, metrics, "subject_package", self._real)

    def test_the_sharpest_face_wins_identity_even_if_it_lacks_a_body(self):
        # The real case: the best face photo is often a portrait, and the body
        # has to come from a different picture.
        self.by_path = {"portrait.jpg": _pkg(face_px=300, has_body=False),
                        "fullbody.jpg": _pkg(face_px=90)}
        text = self.i.report(["portrait.jpg", "fullbody.jpg"])
        self.assertIn("identity <- portrait.jpg", text)
        self.assertIn("build <- fullbody.jpg", text)

    def test_an_unusable_set_says_so(self):
        # Only a missing face makes a photo unusable now, so this is the case.
        self.by_path = {"a.jpg": _pkg(face_px=None, has_body=False)}
        text = self.i.report(["a.jpg"])
        self.assertIn("cannot identity", text)


if __name__ == "__main__":
    unittest.main()
