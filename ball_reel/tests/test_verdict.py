"""Вердикт гейта — по одному тесту на каждую причину отказа.

Эти тесты написаны не «на всякий случай», а по результату мутационного аудита:
снятый `MIN_COVERAGE` не ронял ни одного из 120 тестов, потому что вердикт жил
внутри платного цикла генерации и проверить его было нечем. Порог, который
нечем проверить, — это порог, который завтра сдвинут молча.

Каждый тест ломает ровно одну составляющую и требует, чтобы вердикт назвал
именно её. Проверяется не только «не прошло», но и ПРИЧИНА: гейт, который
заворачивает клип с неправильным объяснением, бесполезен для отладки и
опасен — по нему будут чинить не то.
"""

from __future__ import annotations

import unittest


def _drift(median=0.20, p90=0.30, coverage=1.0):
    return {"median": median, "p90": p90, "coverage": coverage,
            "note": "identity note"}


def _good(**over):
    base = dict(drift=_drift(), motion=0.10,
                quality={"smooth": True, "note": "continuous"},
                seam={"seamless": True, "note": "seamless"},
                limbs={"anatomical": True, "note": "stable"},
                wander=None, bar=0.35, min_motion=0.02, loop=True)
    base.update(over)
    return base


class VerdictPassesOnlyWhenEverythingHolds(unittest.TestCase):
    def setUp(self):
        from ball_reel.produce import verdict

        self.v = verdict

    def test_all_clear_passes_and_scores_the_median(self):
        passed, score, reason = self.v(**_good())
        self.assertTrue(passed)
        self.assertEqual(score, 0.20)
        self.assertEqual(reason, "")

    def test_nothing_judgeable_is_not_a_pass(self):
        # The mutation that survived: with MIN_COVERAGE removed, a clip nobody
        # could verify would sail through as if it had been checked.
        passed, score, reason = self.v(**_good(drift=_drift(coverage=0.1)))
        self.assertFalse(passed)
        self.assertEqual(score, 1.0)
        self.assertIn("not verifiable", reason)

    def test_an_unmeasurable_median_is_not_a_pass(self):
        passed, _, reason = self.v(**_good(drift=_drift(median=None)))
        self.assertFalse(passed)
        self.assertIn("not verifiable", reason)

    def test_identity_drift_past_the_bar_is_named(self):
        passed, _, reason = self.v(**_good(drift=_drift(median=0.50)))
        self.assertFalse(passed)
        self.assertIn("identity drift", reason)

    def test_identity_that_falls_apart_late_is_caught_by_p90(self):
        passed, _, reason = self.v(**_good(drift=_drift(median=0.20, p90=0.90)))
        self.assertFalse(passed)
        self.assertIn("p90", reason)

    def test_a_frozen_clip_is_rejected_for_motion(self):
        passed, _, reason = self.v(**_good(motion=0.0))
        self.assertFalse(passed)
        self.assertIn("motion", reason)

    def test_teleporting_motion_is_rejected_as_not_physical(self):
        passed, _, reason = self.v(
            **_good(quality={"smooth": False, "note": "limbs teleport"}))
        self.assertFalse(passed)
        self.assertIn("not physical", reason)

    def test_rubber_limbs_are_rejected_as_not_anatomical(self):
        passed, _, reason = self.v(
            **_good(limbs={"anatomical": False, "note": "body is stretching"}))
        self.assertFalse(passed)
        self.assertIn("not anatomical", reason)

    def test_a_wandering_pose_is_rejected_when_a_reference_was_given(self):
        passed, _, reason = self.v(
            **_good(wander={"held": False, "note": "DRIFTED"}))
        self.assertFalse(passed)
        self.assertIn("pose wandered", reason)

    def test_no_pose_reference_means_no_pose_complaint(self):
        self.assertTrue(self.v(**_good(wander=None))[0])

    def test_a_visible_seam_is_rejected_when_a_loop_was_asked_for(self):
        passed, _, reason = self.v(
            **_good(seam={"seamless": False, "note": "visible cut"}))
        self.assertFalse(passed)
        self.assertIn("does not loop", reason)

    def test_the_same_seam_is_fine_when_no_loop_was_asked_for(self):
        self.assertTrue(self.v(**_good(
            seam={"seamless": False, "note": "visible cut"}, loop=False))[0])

    def test_identity_is_reported_before_the_cosmetic_failures(self):
        # Everything is broken at once. The reason must be the one to fix
        # first — a clip of the wrong person is not a looping problem.
        passed, _, reason = self.v(**_good(
            drift=_drift(median=0.9), motion=0.0,
            quality={"smooth": False, "note": "x"},
            seam={"seamless": False, "note": "y"},
            limbs={"anatomical": False, "note": "z"}))
        self.assertFalse(passed)
        self.assertIn("identity drift", reason)


class MotionQualitySeesNearlyStaticClips(unittest.TestCase):
    """Второй выживший мутант: STILL_MIN ничем не сторожился.

    Полностью замерший клип ловился ранним возвратом (медианный шаг ровно 0),
    а вот «почти не двигается» — тот случай, ради которого порог и существует, —
    не проверялся ни одним тестом.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        from ball_reel import motion

        self.m = motion
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _frames(self, values):
        from PIL import Image

        out = []
        for i, v in enumerate(values):
            p = self.dir / f"{i:02d}.png"
            Image.new("L", (96, 96), v).save(p)
            out.append(str(p))
        return out

    def test_a_barely_moving_clip_is_not_moving(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("numpy/Pillow not installed")
        # Bright frames, drifting by one grey level: there IS a nonzero step,
        # so the early return does not fire, and STILL_MIN is what decides.
        q = self.m.motion_quality(self._frames([200, 201, 200, 201, 200, 201]))
        self.assertFalse(q["moving"])

    def test_a_clearly_moving_clip_is_moving(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("numpy/Pillow not installed")
        q = self.m.motion_quality(self._frames([100, 130, 160, 190, 160, 130]))
        self.assertTrue(q["moving"])



class GarmentDriftSeesClothesNotLighting(unittest.TestCase):
    """Одежду задаёт промт кейфрейма, поэтому она может плыть между узлами.

    Метрика калибрована на двух реальных опорах: одна съёмка (одежда заведомо
    постоянна) дала 0.0245, смесь кадров двух разных съёмок — 0.1799.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        from ball_reel import garment

        self.g = garment
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _pose(self):
        return {"l_shoulder": (0.35, 0.30, 1.0), "r_shoulder": (0.65, 0.30, 1.0),
                "l_hip": (0.40, 0.60, 1.0), "r_hip": (0.60, 0.60, 1.0),
                "l_knee": (0.40, 0.85, 1.0), "r_knee": (0.60, 0.85, 1.0)}

    def _frames(self, colours):
        from PIL import Image

        out = []
        for i, c in enumerate(colours):
            p = self.dir / f"{i:02d}.png"
            Image.new("RGB", (200, 300), c).save(p)
            out.append(str(p))
        return out

    def _skip_without_deps(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("numpy/Pillow not installed")

    def test_the_same_outfit_across_frames_is_stable(self):
        self._skip_without_deps()
        frames = self._frames([(40, 60, 180)] * 5)
        r = self.g.garment_drift(frames, [self._pose()] * 5)
        self.assertTrue(r["stable"], r["note"])

    def test_brightness_changes_alone_do_not_read_as_a_new_outfit(self):
        self._skip_without_deps()
        # Тот же цвет, разная освещённость — ровно тот случай, на котором
        # сырой RGB объявлял эталон плывущим.
        frames = self._frames([(20, 30, 90), (40, 60, 180), (60, 90, 255),
                               (30, 45, 135), (50, 75, 225)])
        r = self.g.garment_drift(frames, [self._pose()] * 5)
        self.assertTrue(r["stable"], r["note"])

    def test_a_changed_colour_is_caught(self):
        self._skip_without_deps()
        frames = self._frames([(40, 60, 180), (40, 60, 180), (180, 40, 40),
                               (180, 40, 40), (40, 60, 180)])
        r = self.g.garment_drift(frames, [self._pose()] * 5)
        self.assertFalse(r["stable"])
        self.assertIn("ПЛЫВЁТ", r["note"])

    def test_frames_without_a_torso_are_not_judged(self):
        self._skip_without_deps()
        hidden = {**self._pose(), "l_shoulder": (0.35, 0.30, 0.0)}
        frames = self._frames([(40, 60, 180)] * 4)
        r = self.g.garment_drift(frames, [self._pose(), hidden,
                                          self._pose(), self._pose()])
        self.assertEqual(r["unmeasured"], 1)

    def test_too_little_evidence_is_not_verified_rather_than_stable(self):
        self._skip_without_deps()
        frames = self._frames([(40, 60, 180)] * 2)
        r = self.g.garment_drift(frames, [self._pose()] * 2)
        self.assertFalse(r["stable"])
        self.assertIn("НЕ ПРОВЕРЕНА", r["note"])

    def test_mismatched_inputs_are_refused(self):
        self._skip_without_deps()
        with self.assertRaises(ValueError):
            self.g.garment_drift(self._frames([(1, 2, 3)] * 3), [self._pose()])


if __name__ == "__main__":
    unittest.main()


class VerdictAlsoJudgesTheGarment(unittest.TestCase):
    def setUp(self):
        from ball_reel.produce import verdict

        self.v = verdict

    def test_a_stable_garment_does_not_block(self):
        self.assertTrue(self.v(**_good(garment={"stable": True, "note": "ok"}))[0])

    def test_no_garment_check_does_not_block(self):
        self.assertTrue(self.v(**_good(garment=None))[0])

    def test_a_drifting_garment_is_named(self):
        passed, _, reason = self.v(
            **_good(garment={"stable": False, "note": "ПЛЫВЁТ на «torso»"}))
        self.assertFalse(passed)
        self.assertIn("garment drifts", reason)

    def test_identity_still_outranks_the_garment(self):
        _, _, reason = self.v(**_good(drift=_drift(median=0.9),
                                      garment={"stable": False, "note": "x"}))
        self.assertIn("identity drift", reason)


class GarmentIgnoresTheBackgroundAtTheEdges(unittest.TestCase):
    """Область одежды сжимается к центру: край ловит фон и руки.

    Без сжатия цвет «одежды» смешивается со стеной, и метрика начинает мерить
    комнату. Мутационный аудит поймал это: снятое CORE_FRACTION не роняло
    тестов, потому что все они шли на одноцветных картинках.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        from ball_reel import garment

        self.g = garment
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _pose(self):
        return {"l_shoulder": (0.30, 0.20, 1.0), "r_shoulder": (0.70, 0.20, 1.0),
                "l_hip": (0.30, 0.80, 1.0), "r_hip": (0.70, 0.80, 1.0)}

    def _frame(self, name, garment_rgb, edge_rgb):
        """Одежда в центре области, чужой цвет по её краям."""
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (200, 300), edge_rgb)
        d = ImageDraw.Draw(im)
        d.rectangle([70, 100, 130, 200], fill=garment_rgb)
        p = self.dir / name
        im.save(p)
        return str(p)

    def test_a_changing_background_is_not_a_changing_outfit(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("numpy/Pillow not installed")
        # Одежда неизменна, фон вокруг неё гуляет. Сжатие к центру обязано
        # оставить вердикт «держится».
        frames = [self._frame(f"{i}.png", (30, 40, 200), edge)
                  for i, edge in enumerate([(220, 30, 30), (30, 220, 30),
                                            (220, 220, 30), (30, 30, 30),
                                            (220, 30, 220)])]
        r = self.g.garment_drift(frames, [self._pose()] * 5)
        self.assertTrue(r["stable"], r["note"])


class AnAxisDeclaredMustAlsoBeCALLED(unittest.TestCase):
    """Ось была объявлена и мертва, и ни один тест этого не заметил.

    Что было сделано: `"semantic"` попало в `CHECK_ORDER`, у `_verdict`
    появился параметр, условие написано верно. Чего не было: НИКТО НЕ ЗВАЛ
    `semantic_match`. Параметр всегда оставался `None`, и строка
    `semantic is None or ...` выходила тождественно истинной — ось значилась в
    гейте и пропускала всё, включая кадр в бальном платье, ради которого её
    писали.

    Почему не поймали. Тесты оси сторожили её АРИФМЕТИКУ, тесты вердикта —
    его логику; связку между ними не сторожил никто. Мутационный аудит при
    этом показывал 100%: мутации констант оси убивались её собственными
    тестами независимо от того, вызывает ли её конвейер. Ровно тот случай, о
    котором предупреждает докстринг `codeaudit` — порог, который никуда не
    подключён.

    Отсюда правило, которое эти тесты и закрепляют: у оси в `CHECK_ORDER`
    обязан быть не только параметр, но и ВЫЗОВ, и вызов проверяется отдельно.
    """

    def setUp(self):
        from ball_reel import produce

        self.p = produce

    def test_every_axis_in_the_order_is_reachable_from_the_verdict(self):
        import inspect

        src = inspect.getsource(self.p._verdict)
        # Клиповые оси судятся здесь; стартовые — до входа в `_verdict`.
        start = {"start_not_verifiable", "start_identity", "start_pose",
                 "not_verifiable"}
        for name in self.p.CHECK_ORDER:
            if name in start:
                continue
            with self.subTest(axis=name):
                self.assertIn(f'"{name}"', src,
                              f"ось {name} объявлена в CHECK_ORDER, но в "
                              f"_verdict её нет")

    def test_the_semantic_axis_is_actually_invoked_by_the_pipeline(self):
        """Главная проверка: ось ЗОВУТ, а не только принимают её ответ.

        Ищется ВЫЗОВ в дереве разбора, а не подстрока в тексте. Первая
        редакция этого теста искала `"semantic_match" in src` — и НЕ ПОЙМАЛА
        нарочно внесённую поломку: строка `from .semantic import
        semantic_match` осталась на месте, подстрока нашлась, тест позеленел.
        Проверено удалением вызова: подстрочная версия — OK, эта — красная.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(self.p.produce)))
        called = {
            node.func.id if isinstance(node.func, ast.Name) else
            getattr(node.func, "attr", "")
            for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        self.assertIn("semantic_match", called,
                      "semantic_match нигде не ВЫЗЫВАЕТСЯ — ось мертва, и "
                      "условие в _verdict тождественно истинно")

    def test_the_semantic_answer_is_kept_whole_not_reduced_to_a_flag(self):
        # Третий исход («нет весов CLIP») обязан быть виден в отчёте: доля
        # непроверенного — это мера того, насколько вообще судили одежду.
        from dataclasses import fields

        names = [f.name for f in fields(self.p.Attempt)]
        self.assertIn("semantic", names)

    def test_a_mismatched_clip_fails_by_name(self):
        got = self.p._verdict(**self._good(), semantic={
            "verdict": "mismatched", "note": "вечернее платье",
            "check": "semantic_clothing"})
        self.assertFalse(got[0])
        self.assertEqual(got[3], "semantic")
        self.assertIn("платье", got[2])

    def test_an_unmeasurable_semantic_does_not_block(self):
        # Веса CLIP ~600 МБ; гейт, падающий на их отсутствии, выключил бы
        # выпуск везде, где их нет.
        got = self.p._verdict(**self._good(),
                              semantic={"verdict": "not_measurable"})
        self.assertTrue(got[0], got[2])

    def test_the_axis_is_last_because_it_is_the_most_expensive(self):
        # Замерено: CLIP 4 с на загрузку плюс 0.04-0.11 с на кадр, остальные
        # оси — доли секунды в numpy.
        self.assertEqual(self.p.CHECK_ORDER[-1], "semantic")

    @staticmethod
    def _good():
        return dict(drift={"median": 0.20, "p90": 0.30, "coverage": 1.0},
                    motion=1.0, quality={"smooth": True},
                    seam={"seamless": True}, limbs={"anatomical": True},
                    wander=None, bar=0.35, min_motion=0.15, loop=True)
