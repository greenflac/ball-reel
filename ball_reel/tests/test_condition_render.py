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

    #: Телосложение донора: все восемь костей, каждая по половине торса.
    DONOR = {k: 0.5 for k in
             ("l_shoulder->l_elbow", "l_elbow->l_wrist",
              "r_shoulder->r_elbow", "r_elbow->r_wrist",
              "l_hip->l_knee", "l_knee->l_ankle",
              "r_hip->r_knee", "r_knee->r_ankle")}

    def test_a_longer_target_upper_arm_lengthens_that_bone(self):
        pts = _points()
        before = self._bone(pts, "l_shoulder", "l_elbow")
        out = self.s.retarget(pts, {"l_shoulder->l_elbow": 1.0, "l_elbow->l_wrist": 0.5},
                              driving=self.DONOR)
        # Цель вдвое длиннее донора (1.0 против 0.5) -> кость вдвое длиннее В
        # КАДРЕ, какой бы она в кадре ни была.
        self.assertAlmostEqual(self._bone(out, "l_shoulder", "l_elbow"),
                               before * 2, places=3)

    def test_a_limb_with_one_bone_measured_is_left_alone(self):
        """Частичный ретаргет ломает пропорцию внутри конечности.

        ИЗМЕРЕНО на живом ките: бедро x0.70, голень донорская -> отношение
        бедро/голень 0.899 -> 0.630, и ControlNet рисует ногу, которой не
        бывает. Правило и тест — в `test_skeleton.APartialRetargetIsWorseThanNone`.
        """
        pts = _points()
        out = self.s.retarget(pts, {"l_hip->l_knee": 1.0}, driving=self.DONOR)
        self.assertAlmostEqual(self._bone(out, "l_hip", "l_knee"),
                               self._bone(pts, "l_hip", "l_knee"), places=6)

    def test_the_rest_of_the_limb_travels_with_it(self):
        # If only the elbow moved, the forearm would stretch to compensate and
        # the hand would stay put — a detached limb, not a longer arm.
        pts = _points()
        forearm = self._bone(pts, "l_elbow", "l_wrist")
        out = self.s.retarget(pts, {"l_shoulder->l_elbow": 0.75, "l_elbow->l_wrist": 0.5},
                              driving=self.DONOR)
        self.assertAlmostEqual(self._bone(out, "l_elbow", "l_wrist"),
                               forearm, places=3)

    def test_direction_is_preserved(self):
        import numpy as np

        pts = _points()
        before = np.array([pts["l_elbow"][0] - pts["l_shoulder"][0],
                           pts["l_elbow"][1] - pts["l_shoulder"][1]])
        out = self.s.retarget(pts, {"l_shoulder->l_elbow": 0.85, "l_elbow->l_wrist": 0.5},
                              driving=self.DONOR)
        after = np.array([out["l_elbow"][0] - out["l_shoulder"][0],
                          out["l_elbow"][1] - out["l_shoulder"][1]])
        cos = float(np.dot(before, after)
                    / (np.linalg.norm(before) * np.linalg.norm(after)))
        self.assertAlmostEqual(cos, 1.0, places=5)

    def test_no_proportions_is_a_no_op(self):
        pts = _points()
        self.assertEqual(self.s.retarget(pts, {}, driving=self.DONOR),
                         dict(pts))

    def test_an_invisible_joint_is_not_moved(self):
        pts = _points(l_elbow=(0.62, 0.40, 0.1))
        out = self.s.retarget(pts, {"l_shoulder->l_elbow": 1.0, "l_elbow->l_wrist": 0.5},
                              driving=self.DONOR)
        self.assertEqual(out["l_elbow"], pts["l_elbow"])

    def test_the_same_body_leaves_the_pose_untouched(self):
        """Донор и цель совпали -> условия обязаны не измениться ВОВСЕ.

        Самая дешёвая проверка того, что ретаргет переносит ТЕЛО, а не
        перерисовывает позу. Прежний код это заваливал: он ставил кости в
        абсолютную длину `пропорция * торс`, и одинаковые тела всё равно
        получали выпрямленную, лишённую ракурса фигуру.
        """
        # Нога уходит в камеру: в кадре она втрое короче анатомической.
        pts = _points(l_knee=(0.56, 0.62, 1.0), l_ankle=(0.57, 0.68, 1.0))
        out = self.s.retarget(pts, dict(self.DONOR), driving=self.DONOR)
        for joint in ("l_knee", "l_ankle", "l_wrist", "r_wrist"):
            self.assertAlmostEqual(out[joint][0], pts[joint][0], places=9)
            self.assertAlmostEqual(out[joint][1], pts[joint][1], places=9)

    def test_foreshortening_survives_retargeting(self):
        """Ракурс — это движение, и ретаргет обязан его сохранить.

        ИЗМЕРЕНО на живом отрезке: длина левого бедра донора гуляет от 0.102 до
        1.485 длины торса — это нога то в камеру, то вбок. Прежний ретаргет
        сводил её к 0.597..0.601, то есть к константе, и ControlNet получал
        «нога отведена вбок» там, где человек шагал вперёд.
        """
        near = _points(l_knee=(0.56, 0.60, 1.0), l_ankle=(0.57, 0.64, 1.0))
        side = _points(l_knee=(0.75, 0.60, 1.0), l_ankle=(0.90, 0.64, 1.0))
        target = {"l_hip->l_knee": 0.6}
        got = []
        for pts in (near, side):
            out = self.s.retarget(pts, target, driving=self.DONOR)
            got.append(self._bone(out, "l_hip", "l_knee"))
        # Тот же множитель на обоих кадрах, а не одна и та же длина.
        self.assertNotAlmostEqual(got[0], got[1], places=3)
        self.assertAlmostEqual(got[0] / self._bone(near, "l_hip", "l_knee"),
                               got[1] / self._bone(side, "l_hip", "l_knee"),
                               places=6)

    def test_an_unmeasured_side_is_mirrored_not_left_to_the_donor(self):
        """Портретная рефка даёт одну сторону — вторая обязана взяться зеркалом.

        ИЗМЕРЕНО на `kit/face.jpg`: измеримы 3 кости из 8. Прежний код молча
        пропускал остальные пять, и в кадре оказывалось тело, у которого левая
        половина от клиента, а правая от донора. Асимметрия бедра при этом
        росла с 0.214 до 0.759 доли длины — такого человека не существует.
        """
        factors, origin = self.s.retarget_plan(
            {"l_hip->l_knee": 0.75, "l_knee->l_ankle": 0.75}, self.DONOR)
        self.assertAlmostEqual(factors["l_hip->l_knee"], 1.5, places=6)
        self.assertAlmostEqual(factors["r_hip->r_knee"], 1.5, places=6)
        self.assertEqual(origin["l_hip->l_knee"], "измерено")
        self.assertEqual(origin["r_hip->r_knee"], "зеркало")

    def test_mirroring_is_not_reported_as_measurement(self):
        # Допущение о симметрии годится для картинки и не годится для отчёта.
        _, origin = self.s.retarget_plan({"l_elbow->l_wrist": 0.6}, self.DONOR)
        self.assertNotEqual(origin["r_elbow->r_wrist"], "измерено")

    def test_without_donor_proportions_nothing_is_scaled(self):
        """Не с чем сравнивать -> не ретаргетим и не притворяемся.

        Прежнее поведение (поставить абсолютную длину) молча утверждало, что
        длина кости в кадре и есть анатомическая, то есть что ракурса не
        бывает. Это хуже отсутствия ретаргета, потому что выглядит как работа.
        """
        pts = _points()
        self.assertEqual(self.s.retarget(pts, {"l_hip->l_knee": 0.9}), dict(pts))
        factors, origin = self.s.retarget_plan({"l_hip->l_knee": 0.9}, None)
        self.assertEqual(factors, {})
        self.assertEqual(set(origin.values()), {"донор не измерен"})

    def test_a_bone_the_donor_lacks_stays_the_donors(self):
        # Донор не знает ГОЛЕНЕЙ. Тогда пара «бедро+голень» неполна не по вине
        # цели, и ретаргет ноги не применяется целиком — иначе бедро сжалось бы
        # при донорской голени. Рука, у которой обе кости есть у обоих,
        # ретаргетится как прежде.
        donor = {k: v for k, v in self.DONOR.items() if "knee->" not in k}
        factors, origin = self.s.retarget_plan(
            {"l_knee->l_ankle": 0.9, "l_hip->l_knee": 0.6,
             "l_shoulder->l_elbow": 0.6, "l_elbow->l_wrist": 0.6}, donor)
        self.assertNotIn("l_knee->l_ankle", factors)
        self.assertEqual(origin["l_knee->l_ankle"], "донор не измерен")
        self.assertNotIn("l_hip->l_knee", factors)
        self.assertIn("l_shoulder->l_elbow", factors)

    def test_an_absurd_factor_is_refused_rather_than_drawn(self):
        # Множитель 6 — это не телосложение, а промах детектора на одном из
        # двух тел. Растянутая втрое нога учит генератор телу, которого нет.
        factors, origin = self.s.retarget_plan({"l_hip->l_knee": 3.0},
                                               self.DONOR)
        self.assertNotIn("l_hip->l_knee", factors)
        self.assertIn("отброшен", origin["l_hip->l_knee"])
        self.assertGreater(self.s.MAX_RETARGET_FACTOR, 1.0)
        self.assertLess(self.s.MIN_RETARGET_FACTOR, 1.0)

    def test_a_bone_collapsing_to_a_point_is_refused_too(self):
        """Нижняя граница обязана кусаться так же, как верхняя.

        НАЙДЕНО МУТАЦИОННЫМ АУДИТОМ: `MIN_RETARGET_FACTOR = 0.0` пережил все
        тесты. Верхнюю границу сторожил случай с множителем 3, а про нижнюю
        утверждалось лишь `MIN < 1.0` — что при нуле ИСТИНА. То есть кость
        могла схлопнуться в точку, и ни один тест бы не возразил: конечность
        просто исчезла бы из кадра, а условия выглядели бы нормально.

        Ровно тот случай, ради которого аудит и написан: порог, который никто
        не сторожит, завтра сдвинут незаметно.
        """
        # Цель в двадцать раз короче донора — это не телосложение.
        factors, origin = self.s.retarget_plan({"l_hip->l_knee": 0.025},
                                               self.DONOR)
        self.assertNotIn("l_hip->l_knee", factors)
        self.assertIn("отброшен", origin["l_hip->l_knee"])
        # И зеркало не должно протащить отвергнутое на другую сторону.
        self.assertNotIn("r_hip->r_knee", factors)

    def test_a_plausibly_shorter_bone_still_passes(self):
        # Обратная сторона: граница, бракующая всё короткое, — не защита, а
        # регресс. На нашей рефке измеренный множитель бедра 0.701.
        factors, _ = self.s.retarget_plan(
            {"l_hip->l_knee": 0.35, "l_knee->l_ankle": 0.35}, self.DONOR)
        self.assertAlmostEqual(factors["l_hip->l_knee"], 0.7, places=6)

    def test_mirror_key_swaps_both_ends_and_leaves_centre_bones(self):
        self.assertEqual(self.s.mirror_key("l_hip->l_knee"), "r_hip->r_knee")
        self.assertEqual(self.s.mirror_key("r_elbow->r_wrist"),
                         "l_elbow->l_wrist")
        self.assertEqual(self.s.mirror_key("hip_c->neck"), "hip_c->neck")


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


def _upper_only(**over):
    """Тот же скелет, но ног не видно.

    Нужен там, где меряется ФОРМА нарисованной фигуры: в кадрировке по пояс
    голени уходят за нижний край холста, обрезанный габарит формы уже не
    описывает, и тест мерил бы обрезку, а не пропорции.
    """
    pts = _points(**over)
    for name in ("l_knee", "r_knee", "l_ankle", "r_ankle"):
        x, y, _ = pts[name]
        pts[name] = (x, y, 0.0)
    return pts


def _canon(src=(1000.0, 2000.0)):
    """Фигура в КАНОНИЧЕСКИХ пропорциях человека, стоя, в долях роста.

    Взято из антропометрии («рост = 7.5 голов»), а не из `ball_reel`: нос
    0.935 роста, шея 0.82, таз 0.53, колено 0.285, щиколотка 0.039, полуширина
    плеч 0.115. Именно на таком теле снимались живые 9.4% и 19%, поэтому только
    на нём и осмысленно сверять геометрию с измерением. `_points()` для этого
    не годится — у него ноги короче человеческих (37% видимой фигуры против
    55%), и выигрыш кадрировки там честно меньше.
    """
    def y(frac):
        return (src[1] - frac * src[1] / 2) / src[1]

    def x(dx):
        return 0.5 + dx * src[1] / 2 / src[0]

    pts = {"nose": (x(0), y(0.935), 1.0), "l_eye": (x(0.015), y(0.945), 1.0),
           "r_eye": (x(-0.015), y(0.945), 1.0),
           "l_ear": (x(0.04), y(0.94), 1.0), "r_ear": (x(-0.04), y(0.94), 1.0),
           "l_shoulder": (x(0.115), y(0.82), 1.0),
           "r_shoulder": (x(-0.115), y(0.82), 1.0),
           "l_elbow": (x(0.13), y(0.63), 1.0),
           "r_elbow": (x(-0.13), y(0.63), 1.0),
           "l_wrist": (x(0.13), y(0.485), 1.0),
           "r_wrist": (x(-0.13), y(0.485), 1.0),
           "l_hip": (x(0.05), y(0.53), 1.0), "r_hip": (x(-0.05), y(0.53), 1.0),
           "l_knee": (x(0.05), y(0.285), 1.0),
           "r_knee": (x(-0.05), y(0.285), 1.0),
           "l_ankle": (x(0.05), y(0.039), 1.0),
           "r_ankle": (x(-0.05), y(0.039), 1.0),
           "neck": (x(0), y(0.82), 1.0), "hip_c": (x(0), y(0.53), 1.0),
           "__size__": (src[0], src[1], 1.0)}
    return pts


def _bodies():
    """ДИАПАЗОН фигур, а не одна. Здесь на этом горели дважды.

    Меняется то, от чего кадрировка по пояс зависит по построению: длина торса,
    размах рук, положение кистей, наклон. Одна точка проверки показала бы, что
    формула считает, а не что она считает ВЕРНО.
    """
    return {
        "канон": _points(),
        "короткий торс": _points(neck=(0.50, 0.35, 1.0),
                                 l_shoulder=(0.58, 0.35, 1.0),
                                 r_shoulder=(0.42, 0.35, 1.0)),
        "длинный торс": _points(hip_c=(0.50, 0.62, 1.0),
                                l_hip=(0.55, 0.62, 1.0),
                                r_hip=(0.45, 0.62, 1.0)),
        "руки в стороны": _points(l_elbow=(0.72, 0.26, 1.0),
                                  r_elbow=(0.28, 0.26, 1.0),
                                  l_wrist=(0.86, 0.27, 1.0),
                                  r_wrist=(0.14, 0.27, 1.0)),
        "кисти ниже таза": _points(l_wrist=(0.64, 0.70, 1.0),
                                   r_wrist=(0.36, 0.70, 1.0)),
        "наклон вбок": _points(nose=(0.40, 0.12, 1.0),
                               neck=(0.44, 0.26, 1.0),
                               l_shoulder=(0.52, 0.26, 1.0),
                               r_shoulder=(0.36, 0.26, 1.0)),
    }


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class WaistUpFramingIsWhatMakesIdentityJUDGEABLE(unittest.TestCase):
    """Почему этот класс вообще существует.

    Потолок карты 6 ГБ — 512x768. На полном росте лицо занимает 9.4% высоты
    (ИЗМЕРЕНО живьём), то есть ~72 px, а ArcFace судит видео от 100 px и
    старт-кадр от 70. Значит гейт по идентичности возвращает не «похож» и не
    «не похож», а «не смог» — и главный критерий демо подтвердить нечем.
    Кадрировка по пояс удваивает долю лица (19%, ИЗМЕРЕНО) и на тех же 512x768
    даёт ~146 px. Всё, что ниже, сторожит именно этот переход.

    Ни один тест здесь не берёт вход из константы, которую сторожит: фигуры
    синтетические, а ожидания выведены из ИЗМЕРЕННОГО удвоения и из инвариантов
    геометрии, а не из `FACE_TO_TORSO`, `WAIST_UP_MARGIN` и порогов ArcFace.
    """

    def setUp(self):
        from ball_reel import skeleton

        self.s = skeleton
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _win(self, pts, width=512, height=768, framing="waist_up"):
        """Окно кадрировки в ИСХОДНЫХ пикселях."""
        sw, sh, _ = pts["__size__"]
        return self.s._window({k: (v[0] * sw, v[1] * sh, v[2])
                               for k, v in pts.items() if k != "__size__"},
                              aspect=width / height, framing=framing)

    def _shares(self, pts, width=512, height=768):
        return tuple(self.s.face_share(pts, width=width, height=height,
                                       framing=f)
                     for f in ("full_body", "waist_up"))

    def test_on_a_canonical_body_waist_up_DOUBLES_the_face(self):
        # ИЗМЕРЕНО живьём: 9.4% -> 19%, то есть ровно вдвое. Полоса 1.8..2.2 —
        # это и есть проверка, что геометрия здесь считает ту же величину, что
        # мерили на кадрах, а не похожую.
        full, waist = self._shares(_canon())
        self.assertGreater(waist / full, 1.8)
        self.assertLess(waist / full, 2.2)

    def test_the_geometry_reproduces_the_LIVE_measurement(self):
        # Самая дорогая проверка в этом файле: число, посчитанное по скелету,
        # против числа, снятого с живых кадров. Фигура задана антропометрией, а
        # не константами `ball_reel`, и на 512x768 обязана дать те самые
        # ~72 px на полном росте и ~146 px по пояс, из-за которых кадрировка и
        # появилась. Разойдись эти два числа — и планировать по манифесту
        # нельзя, он будет обещать лицо, которого не будет.
        from ball_reel.animate import FULL_BODY_FACE_SHARE, WAIST_UP_FACE_SHARE

        # Полоса 5% ВЫБРАНА не для красоты: при 10% мутация запаса кадрировки
        # (0.25 -> 0.35, доля падает до 0.178) ВЫЖИВАЛА — проверено запуском
        # тестов с подменённой константой. Тест, который не отличает
        # откалиброванный запас от произвольного, ничего не сторожит.
        full, waist = self._shares(_canon())
        self.assertAlmostEqual(full, FULL_BODY_FACE_SHARE,
                               delta=0.05 * FULL_BODY_FACE_SHARE)
        self.assertAlmostEqual(waist, WAIST_UP_FACE_SHARE,
                               delta=0.05 * WAIST_UP_FACE_SHARE)
        self.assertAlmostEqual(full * 768, 72, delta=5)
        self.assertAlmostEqual(waist * 768, 146, delta=8)

    def test_the_gain_survives_every_body_in_the_range(self):
        # Диапазон, а не точка. Порог 1.25 ВЫБРАН низким намеренно: выигрыш
        # кадрировки ЗАВИСИТ от тела и обязан зависеть — она обрезает ноги, и
        # чем короче ноги, тем меньше можно выиграть (фигура «длинный торс»
        # даёт 1.49, и это правда, а не дефект). Сторожит этот порог другое:
        # обрушение выигрыша до нуля. Так уже было — разведённые в стороны руки
        # растягивали окно, и 2.0x превращалось в 1.08x.
        for name, pts in _bodies().items():
            with self.subTest(тело=name):
                full, waist = self._shares(pts)
                self.assertIsNotNone(full)
                self.assertIsNotNone(waist)
                self.assertGreater(waist / full, 1.25, f"{name}: выигрыш съеден")
                self.assertLess(waist / full, 2.6, f"{name}: слишком много")

    def test_arms_spread_wide_do_not_eat_the_gain(self):
        # Тот самый случай, отдельной строкой, потому что он стоил правки в
        # `_window`: на вертикальном холсте вместить размах рук и сохранить
        # крупное лицо нельзя, и выбор — что резать. Режется размах.
        wide = _points(l_elbow=(0.72, 0.26, 1.0), r_elbow=(0.28, 0.26, 1.0),
                       l_wrist=(0.92, 0.27, 1.0), r_wrist=(0.08, 0.27, 1.0))
        self.assertAlmostEqual(self._shares(wide)[1], self._shares(_points())[1],
                               delta=0.02)

    def test_the_crop_stops_at_the_pelvis_not_at_the_ankles(self):
        # «По пояс» проводится по СУСТАВАМ: низ окна — таз, а не доля кадра.
        for name, pts in _bodies().items():
            with self.subTest(тело=name):
                _, wy, _, wh = self._win(pts)
                sh = pts["__size__"][1]
                hip = max(pts[k][1] for k in ("l_hip", "r_hip")) * sh
                ankle = min(pts[k][1] for k in ("l_ankle", "r_ankle")) * sh
                self.assertGreater(wy + wh, hip, f"{name}: таз обрезан")
                self.assertLess(wy + wh, ankle, f"{name}: доехало до щиколоток")

    def test_hanging_wrists_do_not_pull_the_crop_back_to_full_body(self):
        # Опущенные руки — самый частый кадр в фитнес-референсе. Если низ окна
        # брать по габариту верхних суставов, кисть ниже бедра растянет окно, и
        # кадрировка молча перестанет работать там, где нужнее всего.
        plain = _points()
        hanging = _points(l_wrist=(0.64, 0.72, 1.0), r_wrist=(0.36, 0.72, 1.0))
        a, b = self._win(plain)[3], self._win(hanging)[3]
        self.assertAlmostEqual(a, b, delta=0.03 * a)
        # А на полном росте кисти на габарит влияют — там это и правильно.
        self.assertGreaterEqual(self._win(hanging, framing="full_body")[3],
                                self._win(plain, framing="full_body")[3])

    def test_raised_arms_stay_inside_the_frame(self):
        # Руки над головой поднимают верх окна сами: иначе половина упражнений
        # получит условие с обрезанными кистями.
        pts = _points(l_wrist=(0.60, 0.02, 1.0), r_wrist=(0.40, 0.02, 1.0),
                      l_elbow=(0.62, 0.14, 1.0), r_elbow=(0.38, 0.14, 1.0))
        wy = self._win(pts)[1]
        self.assertLess(wy, 0.02 * pts["__size__"][1])

    def test_the_top_of_the_head_is_not_cut_off(self):
        # В COCO-18 макушки нет, есть нос и уши. Окно, построенное по ним,
        # срезает верх головы — то есть ровно то, ради чего кадрировка и
        # делается. Проверяется по картинке: над самым верхним нарисованным
        # пикселем обязан остаться воздух.
        import numpy as np
        from PIL import Image

        for name, pts in _bodies().items():
            with self.subTest(тело=name):
                out = self.s.draw(pts, self.dir / "w.png", width=512,
                                  height=768, framing="waist_up")
                arr = np.asarray(Image.open(out).convert("L"))
                ys, _ = np.nonzero(arr)
                gap = int(ys.min()) / 768
                self.assertGreater(gap, 0.03, f"{name}: голова под срез")
                self.assertLess(gap, 0.30, f"{name}: полкадра пустого неба")

    def test_a_landscape_source_is_not_stretched_in_waist_up_either(self):
        # Та же регрессия, что и на полном росте: 1280x720 на 512x768 растянуло
        # фигуру по вертикали примерно в 2.7 раза, потому что нормированные
        # координаты умножались на ширину и высоту независимо. Новая кадрировка
        # меняет ГРАНИЦЫ окна, а не масштабы по осям, и обязана быть к этому
        # так же невосприимчива.
        square = _upper_only()
        wide = _upper_only()
        wide["__size__"] = (1600.0, 1000.0, 1.0)
        for k, v in list(wide.items()):
            if k != "__size__":
                wide[k] = (0.5 + (v[0] - 0.5) * (0.175 / 0.28), v[1], v[2])
        self.assertAlmostEqual(self._drawn_ratio(square, 512, 768),
                               self._drawn_ratio(wide, 512, 768), delta=0.12)

    def test_the_canvas_shape_does_not_change_the_figure_shape_in_waist_up(self):
        pts = _upper_only()
        for w, h in ((512, 512), (512, 1024), (768, 512)):
            with self.subTest(холст=f"{w}x{h}"):
                self.assertAlmostEqual(self._drawn_ratio(pts, 512, 768),
                                       self._drawn_ratio(pts, w, h), delta=0.12)

    def _drawn_ratio(self, pts, width, height):
        import numpy as np
        from PIL import Image

        out = self.s.draw(pts, self.dir / "r.png", width=width, height=height,
                          framing="waist_up")
        arr = np.asarray(Image.open(out).convert("L"))
        ys, xs = np.nonzero(arr)
        self.assertTrue(len(xs) > 0, "nothing was drawn")
        return (ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1)

    def test_the_share_is_a_property_of_the_ASPECT_not_of_the_pixel_count(self):
        # Доля высоты не может зависеть от того, во сколько пикселей мы потом
        # рисуем: 512x768 и 1024x1536 — один и тот же кадр. Если зависит,
        # значит где-то опять смешаны пиксели и доли.
        for framing in ("full_body", "waist_up"):
            got = [self.s.face_share(_points(), width=w, height=int(w * 1.5),
                                     framing=framing)
                   for w in (256, 512, 1024, 1536)]
            with self.subTest(кадрировка=framing):
                self.assertAlmostEqual(min(got), max(got), places=6)

    def test_a_longer_torso_means_a_bigger_face_in_the_same_frame(self):
        # Диапазон, а не точка: лицо отсчитывается от торса, значит доля обязана
        # расти монотонно вместе с ним. Ступенька или плато здесь означали бы,
        # что число берётся не из геометрии этого скелета.
        got = []
        for hip in (0.50, 0.58, 0.66, 0.74, 0.82):
            pts = _upper_only(hip_c=(0.50, hip, 1.0),
                              l_hip=(0.55, hip, 1.0), r_hip=(0.45, hip, 1.0))
            got.append(self.s.face_share(pts, width=512, height=768,
                                         framing="full_body"))
        self.assertEqual(got, sorted(got))
        self.assertGreater(got[-1] / got[0], 1.2)

    def test_a_misspelled_framing_falls_over_instead_of_going_full_body(self):
        # Молчаливый откат к полному росту здесь — худший из возможных отказов:
        # условия приедут полноростовые, а манифест будет обещать лицо по пояс,
        # и разойдутся они уже на карте.
        for call in (lambda: self.s.face_share(_points(), framing="waistup"),
                     lambda: self.s.draw(_points(), self.dir / "x.png",
                                         framing="Waist_Up"),
                     lambda: self.s.render_sequence(
                         ["a"], self.dir / "bad", framing="по пояс",
                         source=lambda _p: _points())):
            with self.assertRaises(ValueError):
                call()

    def test_an_unmeasurable_face_is_the_THIRD_outcome(self):
        # «Не смогли измерить» — не «мелкое лицо» и не «крупное». Если это
        # свернуть в False, вызывающий поднимет разрешение и всё равно ничего
        # не получит, потому что мерить было не от чего.
        blind = _points()
        for name in ("neck", "hip_c", "l_hip", "r_hip"):
            x, y, _ = blind[name]
            blind[name] = (x, y, 0.0)
        for framing in ("full_body", "waist_up"):
            with self.subTest(кадрировка=framing):
                self.assertIsNone(self.s.face_share(blind, framing=framing))
        m = self.s.render_sequence(["a"], self.dir / "blind",
                                   source=lambda _p: blind)
        self.assertIsNone(m["face_share"])
        self.assertIsNone(m["identity_judgeable"])
        self.assertTrue(any("НЕ УДАЛОСЬ ИЗМЕРИТЬ" in w for w in m["warnings"]))


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class TheManifestSaysWhetherTheGateWillSeeAFace(unittest.TestCase):
    """Решение «хватит ли лица гейту» принимается ДО генерации.

    Иначе оно принимается после — потраченными GPU-минутами и вердиктом «не
    смог», который на демо неотличим от провала.
    """

    def setUp(self):
        from ball_reel import skeleton

        self.s = skeleton
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _run(self, framing, height=768, name="seq"):
        return self.s.render_sequence(
            ["a", "b"], self.dir / f"{name}_{framing}_{height}",
            width=512, height=height, framing=framing,
            source=lambda _p: _points())

    def test_full_body_on_the_6gb_ceiling_declares_the_gate_blind(self):
        m = self._run("full_body")
        self.assertFalse(m["identity_judgeable"])
        blind = [w for w in m["warnings"] if "ОСЛЕПНЕТ" in w]
        self.assertEqual(len(blind), 1)
        # Предупреждение без лечения — это жалоба. Здесь названы обе развилки:
        # другая кадрировка и нужная высота холста.
        self.assertIn("waist_up", blind[0])
        self.assertRegex(blind[0], r"высота холста от \d+ px")

    def test_waist_up_at_the_same_size_turns_the_gate_back_on(self):
        m = self._run("waist_up")
        self.assertTrue(m["identity_judgeable"])
        self.assertFalse([w for w in m["warnings"] if "ОСЛЕПНЕТ" in w])
        self.assertGreater(m["face_px"], self._run("full_body")["face_px"])

    def test_both_framings_are_priced_before_either_is_chosen(self):
        m = self._run("full_body")
        both = m["face_share_by_framing"]
        self.assertEqual(sorted(both), ["full_body", "waist_up"])
        self.assertEqual(m["face_share"], both["full_body"])
        # Цена выбора видна, не считая её самому: во сколько раз мельче лицо.
        self.assertGreater(both["waist_up"], both["full_body"])

    def test_judgeability_flips_ONCE_as_the_canvas_grows(self):
        # Порог не подставляется в тест (иначе тест сторожил бы сам себя), а
        # проверяется его СВОЙСТВО: по высоте холста «судимо» обязано включаться
        # ровно один раз и больше не выключаться. Диапазон, а не точка.
        flags = [self._run("full_body", h)["identity_judgeable"]
                 for h in (384, 576, 768, 1024, 1280, 1536, 2048)]
        self.assertIn(False, flags)
        self.assertIn(True, flags)
        self.assertEqual(flags, sorted(flags, key=bool))

    def test_the_manifest_on_disk_carries_it_too(self):
        # Условия рендерятся дома, а читаются на другой машине через часы:
        # число, оставшееся только в возвращённом словаре, к тому моменту
        # потеряно, и папка с png снова ничего о себе не говорит.
        import json

        m = self._run("waist_up", name="disk")
        saved = json.loads(
            (self.dir / "disk_waist_up_768" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["framing"], "waist_up")
        self.assertEqual(saved["face_px"], m["face_px"])
        self.assertEqual(saved["identity_judgeable"], m["identity_judgeable"])

    def test_the_geometry_is_checked_against_the_LIVE_measurement(self):
        # Число, посчитанное по скелету, и число, снятое с живых кадров, обязаны
        # сойтись — иначе одно из них про другую задачу. Расхождение на типовой
        # фигуре не должно порождать предупреждение, а на растянутой позе —
        # должно: молчаливое расхождение хуже громкого.
        from ball_reel.animate import FULL_BODY_FACE_SHARE, WAIST_UP_FACE_SHARE

        for framing, measured in (("full_body", FULL_BODY_FACE_SHARE),
                                  ("waist_up", WAIST_UP_FACE_SHARE)):
            with self.subTest(кадрировка=framing):
                m = self._run(framing)
                self.assertAlmostEqual(m["face_share"], measured,
                                       delta=0.35 * measured)
                self.assertFalse([w for w in m["warnings"]
                                  if "расхождение" in w])
        # Фигура-каракатица: торс вдвое короче нормального. Геометрия честно
        # даёт другую долю, и об этом надо сказать вслух.
        odd = _points(neck=(0.50, 0.43, 1.0), l_shoulder=(0.58, 0.43, 1.0),
                      r_shoulder=(0.42, 0.43, 1.0))
        m = self.s.render_sequence(["a"], self.dir / "odd", framing="full_body",
                                   source=lambda _p: odd)
        self.assertTrue(any("расхождение" in w for w in m["warnings"]))


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

    #: Донор со всеми восемью костями — чтобы измерять поведение ЦЕЛИ, а не
    #: полноту донора.
    DONOR = {k: 0.5 for k in
             ("l_shoulder->l_elbow", "l_elbow->l_wrist",
              "r_shoulder->r_elbow", "r_elbow->r_wrist",
              "l_hip->l_knee", "l_knee->l_ankle",
              "r_hip->r_knee", "r_knee->r_ankle")}

    def test_the_manifest_names_bones_that_stayed_the_donors(self):
        """«retargeted: true» — недостаточное признание.

        На живой рефке этот флаг стоял в true, будучи правдой ровно на три
        кости из восьми: остальные пять несли тело человека из видео. Флаг
        читался как «тело перенесено» и потому был опаснее прочерка.
        """
        # Цель знает только руки; ноги обязаны остаться донорскими и названы.
        m = self.s.render_sequence(
            ["a"], self.dir / "seq3", source=lambda _p: _points(),
            proportions={"l_shoulder->l_elbow": 0.6, "l_elbow->l_wrist": 0.55},
            donor={k: v for k, v in self.DONOR.items() if "hip->" in k
                   or "shoulder->" in k or "elbow->" in k})
        kept = [k for k in self.DONOR if k not in m["retarget_factors"]]
        self.assertTrue(kept)
        note = " ".join(m["warnings"])
        self.assertIn("ДОНОРСКИМИ", note)
        for bone in kept:
            self.assertIn(bone, note)

    def test_the_manifest_separates_mirrored_from_measured(self):
        m = self.s.render_sequence(
            ["a"], self.dir / "seq4", source=lambda _p: _points(),
            proportions={"l_hip->l_knee": 0.6, "l_knee->l_ankle": 0.6},
            donor=self.DONOR)
        self.assertEqual(m["retarget_origin"]["l_hip->l_knee"], "измерено")
        self.assertEqual(m["retarget_origin"]["r_hip->r_knee"], "зеркало")
        self.assertTrue(any("ЗЕРКАЛОМ" in w for w in m["warnings"]))

    def test_retargeted_is_false_when_no_bone_was_actually_scaled(self):
        # Телосложение задано, донор неизмерим -> ни одна кость не тронута, и
        # флаг обязан это признать, а не отчитаться о намерении.
        m = self.s.render_sequence(
            ["a"], self.dir / "seq5", source=lambda _p: _points(),
            proportions={"l_hip->l_knee": 0.6}, donor={})
        self.assertFalse(m["retargeted"])
        self.assertEqual(m["retarget_factors"], {})
        self.assertTrue(any("НЕ ПРИМЕНЕНО" in w for w in m["warnings"]))

    def test_the_donor_body_is_measured_from_too_few_frames_honestly(self):
        # Медиана по трём кадрам — это не медиана. Порог назван числом.
        got, used = self.s.driving_proportions(["a", "b", "c"])
        self.assertEqual(got, {})
        self.assertLess(used, self.s.MIN_DRIVING_FRAMES)

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
        saved = json.loads((self.dir / "seq4" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["source"], m["source"])
        self.assertNotIn(saved["source"], ("dwpose", "mediapipe"))


if __name__ == "__main__":
    unittest.main()

    def test_a_donor_body_handed_in_without_its_frame_count_is_flagged(self):
        """Урезанный кит — ловушка, и манифест обязан её называть.

        ИЗМЕРЕНО: телосложение, снятое только с 16-кадрового окна вместо всех
        71, теряет ТРИ кости целиком (предплечья не видны в достаточном числе
        кадров, `MIN_DRIVING_FRAMES` их отбрасывает) и уводит бедро на 10%.
        Кит при этом выглядит полным. Значит донор для нарезки берётся снаружи,
        по всей последовательности, — а без счётчика кадров манифест не отличит
        честный замер от неизвестно чего.
        """
        m = self.s.render_sequence(
            ["a"], self.dir / "seq6", source=lambda _p: _points(),
            proportions={"l_hip->l_knee": 0.6}, donor=self.DONOR)
        self.assertTrue(any("ЗАДАНО вызывающим" in w for w in m["warnings"]))

    def test_a_donor_measured_wider_than_the_cut_is_declared_normal(self):
        m = self.s.render_sequence(
            ["a"], self.dir / "seq7", source=lambda _p: _points(),
            proportions={"l_hip->l_knee": 0.6}, donor=self.DONOR,
            donor_frames=71)
        self.assertEqual(m["donor_frames_measured"], 71)
        note = " ".join(m["warnings"])
        self.assertNotIn("ЗАДАНО вызывающим", note)
        self.assertIn("шире этой нарезки", note)
