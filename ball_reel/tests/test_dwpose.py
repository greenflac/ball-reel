"""DWPose: всё, что можно проверить без весов.

Инференс требует ONNX-моделей, которых в этой среде нет (`huggingface.co`
закрыт egress-политикой). Но веса нужны ровно двум строчкам — прогону сессий;
всё, что решает, ЧТО получится на выходе, — геометрия, выбор человека,
декодирование SimCC и приведение к общему контракту — обычная арифметика, и
она проверяется здесь.

Главное свойство под охраной: выход DWPose должен быть неотличим по форме от
выхода MediaPipe, иначе `skeleton.render_sequence` начнёт вести себя по-разному
в зависимости от источника, и сравнить два источника станет нельзя.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np  # noqa: F401
    from PIL import Image  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


class OutputShapeMatchesTheMediaPipeContract(unittest.TestCase):
    def setUp(self):
        from ball_reel import dwpose, skeleton

        self.d = dwpose
        self.s = skeleton

    def _pixels(self, conf=0.9):
        return {name: (100.0 + i * 5, 200.0 + i * 7, conf)
                for i, name in enumerate(self.d.WHOLEBODY_INDEX)}

    def test_every_joint_the_renderer_draws_is_produced(self):
        # Если DWPose не отдаст сустав, который умеет рисовать skeleton, этот
        # сустав молча исчезнет из условий — и никто не заметит.
        out = self.d.to_normalised(self._pixels(), 720, 1280)
        for name in self.s.COCO18:
            self.assertIn(name, out, name)

    def test_synthetic_joints_are_added(self):
        # neck и hip_c нет ни в COCO-WholeBody, ни в MediaPipe: их синтезируют
        # оба источника, и на них держатся нормировка и ретаргетинг.
        out = self.d.to_normalised(self._pixels(), 720, 1280)
        self.assertIn("neck", out)
        self.assertIn("hip_c", out)

    def test_the_frame_size_travels_with_the_points(self):
        # Без __size__ рендер растянет фигуру — ровно тот баг, который уже
        # ловился на MediaPipe-пути.
        out = self.d.to_normalised(self._pixels(), 720, 1280)
        self.assertEqual(out["__size__"][:2], (720.0, 1280.0))

    def test_coordinates_are_normalised_to_the_frame(self):
        out = self.d.to_normalised({"nose": (360.0, 640.0, 0.9)}, 720, 1280)
        self.assertAlmostEqual(out["nose"][0], 0.5, places=6)
        self.assertAlmostEqual(out["nose"][1], 0.5, places=6)

    def test_a_synthetic_joint_inherits_the_weaker_confidence(self):
        # Шея не может быть увереннее плеча, из которого получена, иначе
        # ненаблюдаемое плечо протащит нарисованную шею.
        pts = {"l_shoulder": (100.0, 200.0, 0.9),
               "r_shoulder": (140.0, 200.0, 0.1)}
        out = self.d.to_normalised(pts, 720, 1280)
        self.assertAlmostEqual(out["neck"][2], 0.1, places=6)


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class DetectionPicksTheRightPerson(unittest.TestCase):
    def setUp(self):
        from ball_reel import dwpose

        self.d = dwpose

    def test_the_largest_person_wins_not_the_first(self):
        # В кадре бывает отражение в зеркале или прохожий; условие должно
        # строиться по тому, кого снимают.
        boxes = [[0, 0, 20, 40], [100, 100, 300, 500]]
        got = self.d._largest_person(boxes, [0.9, 0.8], scale=1.0)
        self.assertEqual(got, (100.0, 100.0, 300.0, 500.0))

    def test_low_confidence_detections_are_ignored(self):
        boxes = [[0, 0, 400, 800], [100, 100, 200, 200]]
        got = self.d._largest_person(boxes, [0.05, 0.9], scale=1.0)
        self.assertEqual(got, (100.0, 100.0, 200.0, 200.0))

    def test_nothing_confident_means_no_person(self):
        self.assertIsNone(
            self.d._largest_person([[0, 0, 10, 10]], [0.01], scale=1.0))

    def test_boxes_come_back_in_source_coordinates(self):
        # Детектор работает на уменьшенном полотне; координаты обязаны
        # вернуться в систему исходного кадра, иначе кроп уедет.
        got = self.d._largest_person([[50, 50, 150, 250]], [0.9], scale=0.5)
        self.assertEqual(got, (100.0, 100.0, 300.0, 500.0))


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class LetterboxKeepsProportions(unittest.TestCase):
    def setUp(self):
        from ball_reel import dwpose

        self.d = dwpose

    def test_a_landscape_frame_is_fitted_not_stretched(self):
        import numpy as np

        img = np.zeros((720, 1280, 3), dtype="uint8")
        canvas, scale = self.d._letterbox(img, (640, 640))
        self.assertEqual(canvas.shape[:2], (640, 640))
        self.assertAlmostEqual(scale, 640 / 1280, places=6)

    def test_the_scale_is_the_smaller_of_the_two(self):
        import numpy as np

        img = np.zeros((1280, 720, 3), dtype="uint8")
        _, scale = self.d._letterbox(img, (640, 640))
        self.assertAlmostEqual(scale, 640 / 1280, places=6)


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class SimccDecodesToTheBox(unittest.TestCase):
    def setUp(self):
        from ball_reel import dwpose

        self.d = dwpose

    def _logits(self, peak_x, peak_y, split=2, size=(192, 256)):
        import numpy as np

        n = len(self.d.WHOLEBODY_INDEX)
        x = np.zeros((n, size[0] * split), dtype="float32")
        y = np.zeros((n, size[1] * split), dtype="float32")
        x[:, peak_x] = 1.0
        y[:, peak_y] = 1.0
        return x, y

    def test_a_centred_peak_lands_in_the_middle_of_the_box(self):
        x, y = self._logits(192, 256)          # split=2 -> середина
        got = self.d._decode_simcc(x, y, (0.0, 0.0, 192.0, 256.0), (192, 256))
        self.assertAlmostEqual(got["nose"][0], 96.0, places=3)
        self.assertAlmostEqual(got["nose"][1], 128.0, places=3)

    def test_points_are_offset_by_the_box_origin(self):
        # Точки предсказываются внутри кропа; забыть смещение — значит собрать
        # скелет в углу кадра.
        x, y = self._logits(192, 256)
        got = self.d._decode_simcc(x, y, (100.0, 50.0, 292.0, 306.0), (192, 256))
        self.assertAlmostEqual(got["nose"][0], 196.0, places=3)
        self.assertAlmostEqual(got["nose"][1], 178.0, places=3)

    def test_confidence_is_the_weaker_of_the_two_axes(self):
        import numpy as np

        x, y = self._logits(192, 256)
        x[0, 192] = 0.8
        y[0, 256] = 0.4
        got = self.d._decode_simcc(x, y, (0.0, 0.0, 192.0, 256.0), (192, 256))
        self.assertAlmostEqual(got["nose"][2], 0.4, places=6)


class AnEmptyFrameIsRejected(unittest.TestCase):
    def setUp(self):
        from ball_reel import dwpose

        self.d = dwpose

    def test_a_frame_where_nothing_was_seen_is_not_a_pose(self):
        # Порог 0.3 из практики DWPose; литерал, а не MIN_SCORE + delta —
        # тест, выводящий вход из охраняемой константы, движется вместе с ней
        # и никогда не падает.
        self.assertFalse(self.d.usable({"nose": (1.0, 2.0, 0.05),
                                        "l_wrist": (3.0, 4.0, 0.1)}))

    def test_one_confident_joint_is_enough_to_keep_the_frame(self):
        self.assertTrue(self.d.usable({"nose": (1.0, 2.0, 0.9),
                                       "l_wrist": (3.0, 4.0, 0.05)}))

    def test_a_borderline_frame_is_kept(self):
        self.assertTrue(self.d.usable({"nose": (1.0, 2.0, 0.35)}))


class WeightsAreRequiredLoudly(unittest.TestCase):
    def setUp(self):
        from ball_reel import dwpose

        self.d = dwpose

    def test_missing_weights_name_the_fix_and_refuse_to_fall_back(self):
        import os

        os.environ[self.d.DET_ENV] = "/nonexistent/yolox.onnx"
        os.environ[self.d.POSE_ENV] = "/nonexistent/dw.onnx"
        self.addCleanup(os.environ.pop, self.d.DET_ENV, None)
        self.addCleanup(os.environ.pop, self.d.POSE_ENV, None)
        self.assertFalse(self.d.available())
        with self.assertRaises(RuntimeError) as ctx:
            self.d._model_paths()
        msg = str(ctx.exception)
        self.assertIn("curl", msg)          # говорит, ЧТО сделать
        # и почему нельзя тихо откатиться на MediaPipe
        self.assertIn("верификатор", msg)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class TheCropKeepsRoomForLimbs(unittest.TestCase):
    """Запас вокруг бокса — не косметика.

    Детектор обводит человека вплотную, и кисти со стопами регулярно оказываются
    на самом краю. Кроп без запаса срезает их, а срезанный сустав — это сустав,
    которого не будет в условии, то есть неограниченная конечность в кадре.
    """

    def setUp(self):
        from ball_reel import dwpose

        self.d = dwpose

    def test_the_box_grows_by_the_padding(self):
        # Литерал 1.25, не BOX_PADDING: иначе тест поедет вместе с константой.
        x0, y0, x1, y1 = self.d.expand_box((0, 0, 300, 400), aspect=0.75, pad=1.25)
        self.assertAlmostEqual(x1 - x0, 300 * 1.25, places=3)
        self.assertAlmostEqual(y1 - y0, 400 * 1.25, places=3)

    def test_the_default_leaves_a_real_but_bounded_margin(self):
        # Три теста рядом дёргают pad явным аргументом, и поэтому ни один из
        # них не трогает саму константу: мутационный аудит снял BOX_PADDING
        # до 1.0, и всё осталось зелёным. Здесь pad НЕ передаётся — судится
        # значение по умолчанию, то есть то, с которым поедет живой прогон.
        # Границы литеральные и широкие: запас обязан быть заметным (иначе
        # детектор отрежет кисти и стопы у края бокса) и не обязан быть
        # ровно 1.25 — подкрутить его в этих пределах можно, выключить нельзя.
        x0, _, x1, _ = self.d.expand_box((0, 0, 300, 400), aspect=0.75)
        self.assertGreater(x1 - x0, 300 * 1.15)
        self.assertLess(x1 - x0, 300 * 1.60)

    def test_no_padding_leaves_the_box_touching_the_body(self):
        x0, y0, x1, y1 = self.d.expand_box((0, 0, 300, 400), aspect=0.75, pad=1.0)
        self.assertAlmostEqual(x1 - x0, 300.0, places=3)

    def test_the_box_is_reshaped_to_the_model_aspect(self):
        # Плоский resize прямоугольника в 288x384 сжал бы одну ось и сместил
        # все точки; поэтому бокс сначала приводится к пропорциям входа.
        x0, y0, x1, y1 = self.d.expand_box((0, 0, 400, 400), aspect=0.75, pad=1.0)
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 0.75, places=3)

    def test_the_centre_does_not_move(self):
        for pad in (1.0, 1.25, 2.0):
            x0, y0, x1, y1 = self.d.expand_box((100, 50, 400, 450),
                                               aspect=0.75, pad=pad)
            self.assertAlmostEqual((x0 + x1) / 2, 250.0, places=3)
            self.assertAlmostEqual((y0 + y1) / 2, 250.0, places=3)
