"""Condition rendering and retargeting, offline.

These conditions are what the generator is forced to obey, so an error here does
not surface as a crash — it surfaces as a confident clip of the wrong body doing
the wrong thing. The aspect-ratio regression below was found by LOOKING at an
image, which is exactly the kind of check that does not survive contact with a
deadline; it belongs in a test.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np  # noqa: F401
    from PIL import Image  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _points(**over):
    """A plain upright skeleton in normalised coords, source 1000x1000."""
    pts = {
        "nose": (0.50, 0.10, 1.0), "l_eye": (0.52, 0.09, 1.0),
        "r_eye": (0.48, 0.09, 1.0), "l_ear": (0.54, 0.10, 1.0),
        "r_ear": (0.46, 0.10, 1.0),
        "l_shoulder": (0.58, 0.25, 1.0), "r_shoulder": (0.42, 0.25, 1.0),
        "l_elbow": (0.62, 0.40, 1.0), "r_elbow": (0.38, 0.40, 1.0),
        "l_wrist": (0.64, 0.55, 1.0), "r_wrist": (0.36, 0.55, 1.0),
        "l_hip": (0.55, 0.55, 1.0), "r_hip": (0.45, 0.55, 1.0),
        "l_knee": (0.56, 0.75, 1.0), "r_knee": (0.44, 0.75, 1.0),
        "l_ankle": (0.57, 0.92, 1.0), "r_ankle": (0.43, 0.92, 1.0),
        "__size__": (1000.0, 1000.0, 1.0),
    }
    pts["neck"] = (0.50, 0.25, 1.0)
    pts["hip_c"] = (0.50, 0.55, 1.0)
    pts.update(over)
    return pts


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class RetargetingChangesLengthsNotDirections(unittest.TestCase):
    def setUp(self):
        from ball_reel import skeleton

        self.s = skeleton

    def _bone(self, pts, a, b):
        import numpy as np

        return float(np.hypot(pts[b][0] - pts[a][0], pts[b][1] - pts[a][1]))

    def test_a_longer_target_upper_arm_lengthens_that_bone(self):
        pts = _points()
        torso = self._bone(pts, "hip_c", "neck")
        before = self._bone(pts, "l_shoulder", "l_elbow")
        out = self.s.retarget(pts, {"l_shoulder->l_elbow": before / torso * 2})
        self.assertAlmostEqual(self._bone(out, "l_shoulder", "l_elbow"),
                               before * 2, places=3)

    def test_the_rest_of_the_limb_travels_with_it(self):
        # If only the elbow moved, the forearm would stretch to compensate and
        # the hand would stay put — a detached limb, not a longer arm.
        pts = _points()
        torso = self._bone(pts, "hip_c", "neck")
        forearm = self._bone(pts, "l_elbow", "l_wrist")
        out = self.s.retarget(
            pts, {"l_shoulder->l_elbow":
                  self._bone(pts, "l_shoulder", "l_elbow") / torso * 1.5})
        self.assertAlmostEqual(self._bone(out, "l_elbow", "l_wrist"),
                               forearm, places=3)

    def test_direction_is_preserved(self):
        import numpy as np

        pts = _points()
        torso = self._bone(pts, "hip_c", "neck")
        before = np.array([pts["l_elbow"][0] - pts["l_shoulder"][0],
                           pts["l_elbow"][1] - pts["l_shoulder"][1]])
        out = self.s.retarget(
            pts, {"l_shoulder->l_elbow":
                  self._bone(pts, "l_shoulder", "l_elbow") / torso * 1.7})
        after = np.array([out["l_elbow"][0] - out["l_shoulder"][0],
                          out["l_elbow"][1] - out["l_shoulder"][1]])
        cos = float(np.dot(before, after)
                    / (np.linalg.norm(before) * np.linalg.norm(after)))
        self.assertAlmostEqual(cos, 1.0, places=5)

    def test_no_proportions_is_a_no_op(self):
        pts = _points()
        self.assertEqual(self.s.retarget(pts, {}), dict(pts))

    def test_an_invisible_joint_is_not_moved(self):
        pts = _points(l_elbow=(0.62, 0.40, 0.1))
        out = self.s.retarget(pts, {"l_shoulder->l_elbow": 5.0})
        self.assertEqual(out["l_elbow"], pts["l_elbow"])


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class ConditionsKeepBodyProportions(unittest.TestCase):
    """Regression: a landscape source drawn onto a portrait canvas.

    Live, a 1280x720 driving frame rendered onto 512x768 stretched the figure
    vertically about 2.7x, because normalised coordinates were multiplied by
    canvas width and height independently. A stretched skeleton teaches the
    generator a body nobody has — and nothing crashes, so only a measurement
    catches it.
    """

    def setUp(self):
        from ball_reel import skeleton

        self.s = skeleton
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _drawn_ratio(self, pts, width, height):
        """Height/width of the drawn figure's bounding box, in canvas pixels."""
        import numpy as np
        from PIL import Image

        out = self.s.draw(pts, self.dir / "c.png", width=width, height=height)
        arr = np.asarray(Image.open(out).convert("L"))
        ys, xs = np.nonzero(arr)
        self.assertTrue(len(xs) > 0, "nothing was drawn")
        return (ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1)

    def test_a_landscape_source_is_not_stretched_onto_a_portrait_canvas(self):
        # The SAME PERSON, same pixel size, filmed on a 1:1 and on a 16:10
        # sensor. "Same person" has to be built in pixels, not in normalised
        # coordinates: squeezing x in normalised space would describe a
        # genuinely narrower body, and the test would then be asserting that
        # two different bodies draw alike.
        square = _points()                      # body 280 x 830 px in 1000x1000
        wide = _points()
        wide["__size__"] = (1600.0, 1000.0, 1.0)
        # 280 px of 1600 is 0.175 of the frame, against 0.28 of the square one.
        for k, v in list(wide.items()):
            if k != "__size__":
                wide[k] = (0.5 + (v[0] - 0.5) * (0.175 / 0.28), v[1], v[2])

        from_square = self._drawn_ratio(square, 512, 768)
        from_wide = self._drawn_ratio(wide, 512, 768)
        self.assertAlmostEqual(from_square, from_wide, delta=0.12)

    def test_the_canvas_shape_does_not_change_the_figure_shape(self):
        pts = _points()
        tall = self._drawn_ratio(pts, 512, 768)
        square = self._drawn_ratio(pts, 512, 512)
        self.assertAlmostEqual(tall, square, delta=0.12)

    def test_the_subject_fills_the_frame_rather_than_sitting_in_a_band(self):
        # Cropping, not letterboxing: conditioning resolution spent on black
        # bars is resolution not spent on the body.
        import numpy as np
        from PIL import Image

        out = self.s.draw(_points(), self.dir / "c.png", width=512, height=768)
        arr = np.asarray(Image.open(out).convert("L"))
        ys, _ = np.nonzero(arr)
        self.assertGreater((ys.max() - ys.min()) / 768, 0.5)


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class ADrawingNeverInventsAJoint(unittest.TestCase):
    def setUp(self):
        from ball_reel import skeleton

        self.s = skeleton
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_invisible_joints_are_omitted_not_guessed(self):
        import numpy as np
        from PIL import Image

        full = np.asarray(Image.open(
            self.s.draw(_points(), self.dir / "a.png")).convert("L"))
        hidden = _points(l_wrist=(0.64, 0.55, 0.0), l_elbow=(0.62, 0.40, 0.0))
        part = np.asarray(Image.open(
            self.s.draw(hidden, self.dir / "b.png")).convert("L"))
        self.assertLess(int(np.count_nonzero(part)), int(np.count_nonzero(full)))

    def test_a_sequence_reports_frames_it_could_not_read(self):
        from ball_reel import skeleton

        real = skeleton.pose_points
        calls = {"n": 0}

        def flaky(_p):
            calls["n"] += 1
            return None if calls["n"] == 2 else _points()

        skeleton.pose_points = flaky
        self.addCleanup(setattr, skeleton, "pose_points", real)
        # Экстрактор задан явно: с умолчанием этот тест зависел бы от того,
        # лежат ли на машине веса DWPose, и «пропуск кадра» подменялся бы
        # «нет весов» — два разных отказа под одним красным тестом.
        m = self.s.render_sequence(["a", "b", "c"], self.dir / "seq",
                                   source=flaky)
        self.assertEqual(m["missing_frames"], [1])
        self.assertAlmostEqual(m["coverage"], 2 / 3, places=3)
        self.assertTrue(any("unconstrained" in w for w in m["warnings"]))

    def test_an_unretargeted_sequence_says_whose_body_it_carries(self):
        m = self.s.render_sequence(["a"], self.dir / "seq2",
                                   source=lambda _p: _points())
        self.assertTrue(any("DRIVING person's proportions" in w
                            for w in m["warnings"]))

    def _without_dwpose(self):
        """Машина без весов DWPose — самый частый случай на чужом ноутбуке."""
        from ball_reel import dwpose, skeleton

        real_why, real_points = dwpose.why_unavailable, skeleton.pose_points
        dwpose.why_unavailable = lambda: (
            "нет весов DWPose: ~/.dwpose/yolox_l.onnx. Скачать один раз: "
            "curl -sSLO .../yolox_l.onnx")
        skeleton.pose_points = lambda _p: _points()
        self.addCleanup(setattr, dwpose, "why_unavailable", real_why)
        self.addCleanup(setattr, skeleton, "pose_points", real_points)

    def test_conditioning_from_the_verifier_is_declared(self):
        # Снимать условия тем же экстрактором, что проверяет, — ослабленный
        # режим: гейт перестаёт ловить его собственные ошибки. Он разрешён,
        # потому что работает без установки весов, но никогда не молча.
        self._without_dwpose()
        m = self.s.render_sequence(["a"], self.dir / "seq3")
        self.assertEqual(m["source"], "mediapipe")
        self.assertTrue(any("also the verifier" in w for w in m["warnings"]))
        # И сразу сказано, чем это лечится — иначе предупреждение бесполезно.
        self.assertTrue(any("yolox_l.onnx" in w for w in m["warnings"]))

    def test_the_default_conditioner_is_NOT_the_verifier(self):
        # Заявление «условия снимает DWPose, проверяет MediaPipe» долго было
        # только заявлением: умолчанием стоял MediaPipe, то есть судья и
        # судимый совпадали, а независимость держалась на предупреждении.
        from ball_reel import dwpose, skeleton

        real = dwpose.why_unavailable
        dwpose.why_unavailable = lambda: ""
        self.addCleanup(setattr, dwpose, "why_unavailable", real)
        extract, name = skeleton._extractor()
        self.assertEqual(name, "dwpose")
        self.assertIs(extract, dwpose.pose_points)
        self.assertIsNot(extract, skeleton.pose_points)

    def test_the_manifest_cannot_LIE_about_who_extracted(self):
        # ДЕФЕКТ, КОТОРЫЙ ЭТОТ ТЕСТ ЗАКРЫВАЕТ. Экстрактор выбирался параметром
        # `source`, а подпись в манифесте — отдельным флагом `from_mediapipe`,
        # независимо от того, кто отработал. То есть можно было получить точки
        # MediaPipe и записать «dwpose». Манифест лежит рядом с условиями и
        # читается на другой машине через часы; подпись, способная соврать о
        # происхождении условий, хуже отсутствующей.
        import json

        m = self.s.render_sequence(["a"], self.dir / "seq4",
                                   source=lambda _p: _points())
        saved = json.loads((self.dir / "seq4" / "manifest.json").read_text())
        self.assertEqual(saved["source"], m["source"])
        self.assertNotIn(saved["source"], ("dwpose", "mediapipe"))


if __name__ == "__main__":
    unittest.main()
