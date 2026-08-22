"""Гейты универсального плана. Каждый сторожит дефект, а не строчку кода.

Числа-ожидания — ЛИТЕРАЛЫ (Т2). Импорт из проверяемого модуля поехал бы вместе
с кодом и промолчал.
"""

import unittest
from unittest import mock

from ball_reel import fork_plan as P
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED

# Поза «в плане»: полный рост, по центру. Значения — доли кадра.
GOOD = {"l_shoulder": (0.58, 0.32, 0.99), "r_shoulder": (0.42, 0.32, 0.99),
        "l_elbow": (0.63, 0.48, 0.98), "r_elbow": (0.37, 0.48, 0.98),
        "l_wrist": (0.66, 0.62, 0.97), "r_wrist": (0.34, 0.62, 0.97),
        "l_hip": (0.55, 0.60, 0.99), "r_hip": (0.45, 0.60, 0.99),
        "l_knee": (0.55, 0.75, 0.98), "r_knee": (0.45, 0.75, 0.98),
        "l_ankle": (0.55, 0.92, 0.96), "r_ankle": (0.45, 0.92, 0.96)}


def shifted(pose, dx=0.0, dy=0.0):
    return {k: (v[0] + dx, v[1] + dy, v[2]) for k, v in pose.items()}


class ThePlanNumbersComeFromTheMeasuredDrivings(unittest.TestCase):
    """Полосы выведены из планов драйвингов, а не из вкуса."""

    def test_the_shipped_numbers_are_the_chosen_ones(self):
        self.assertEqual(P.PLAN_RATIO, 0.5625)
        self.assertEqual(P.SHOULDERS_BAND, (0.20, 0.42))
        self.assertEqual(P.ANKLES_BAND, (0.86, 0.99))
        self.assertEqual(P.CENTRE_TOL, 0.08)
        self.assertEqual(P.WIDTH_MAX, 0.72)

    def test_the_face_bar_is_imported_not_copied(self):
        # Е1: планка размера лица одна на проект. Копия разъехалась бы молча.
        from ball_reel import fork_intake

        self.assertIs(P.MIN_FACE_PX, fork_intake.MIN_FACE_PX)

    def test_the_measured_styliser_size_is_recorded(self):
        # ИЗМЕРЕНО дважды с негативным контролем: стилизатор отвечает этим на
        # ЛЮБОЙ вход. Если однажды заговорит иначе — этот тест покраснеет.
        self.assertEqual(P.STYLED_SIZE_MEASURED, (896, 1200))

    def test_the_measured_driving_plans_land_where_the_bands_say(self):
        """НЕГАТИВНЫЙ КОНТРОЛЬ полос: они обязаны РАЗЛИЧАТЬ драйвинги.

        ИЗМЕРЕНО (медиана по кадрам): b2 плечи 0.375 щиколотки 0.940 — в плане;
        b4 плечи 0.531 — ниже полосы; b5 щиколотки 0.625 — ноги не влезли.
        Полоса, которая пропускает всё, не измеряет ничего.
        """
        lo, hi = P.SHOULDERS_BAND
        self.assertTrue(lo <= 0.375 <= hi, "b2 обязан попадать")
        self.assertFalse(lo <= 0.531 <= hi, "b4 обязан НЕ попадать")
        a_lo, a_hi = P.ANKLES_BAND
        self.assertTrue(a_lo <= 0.940 <= a_hi, "b2 обязан попадать")
        self.assertFalse(a_lo <= 0.625 <= a_hi, "b5 обязан НЕ попадать")


class ThePersonBoxIgnoresUnconfidentJoints(unittest.TestCase):
    """Дефект, а не строчка: mediapipe достраивает суставы ЗА обрезом кадра."""

    def test_a_confident_pose_gives_a_box(self):
        box = P.person_box(GOOD)
        self.assertEqual(box["outcome"], PASS)
        self.assertEqual(box["centre"], 0.5)
        self.assertEqual(box["shoulders"], 0.32)
        self.assertEqual(box["ankles"], 0.92)
        self.assertEqual(box["joints"], 12)

    def test_a_joint_beyond_the_frame_does_not_stretch_the_box(self):
        # ИЗМЕРЕНО на b4: неуверенная точка дала левый край -490 px.
        # Здесь та же форма: точка на -0.6 с уверенностью 0.03.
        dirty = dict(GOOD, l_ankle=(-0.6, 1.7, 0.03))
        self.assertEqual(P.person_box(dirty)["x0"], 0.34)

    def test_no_confident_joints_is_UNMEASURED_not_failed(self):
        faint = {k: (v[0], v[1], 0.01) for k, v in GOOD.items()}
        got = P.person_box(faint)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("12", got["note"])

    def test_no_pose_at_all_is_UNMEASURED(self):
        for empty in ({}, None, "поза"):
            with self.subTest(empty=empty):
                self.assertEqual(P.person_box(empty)["outcome"], UNMEASURED)

    def test_mutating_the_visibility_bar_both_ways_moves_the_box(self):
        """Т1: планка видимости строже и слабее."""
        dirty = dict(GOOD, l_ankle=(-0.6, 1.7, 0.4))
        loose = P.person_box(dirty, min_visibility=0.3)
        self.assertEqual(loose["x0"], -0.6)
        strict = P.person_box(dirty, min_visibility=0.5)
        self.assertEqual(strict["x0"], 0.34)


class TheCanvasAlwaysPadsAndNeverCrops(unittest.TestCase):
    """Обрезка — это ровно тот дефект, ради которого модуль написан."""

    def test_the_measured_styled_size_becomes_nine_by_sixteen(self):
        got = P.canvas_for(896, 1200)
        # 896 / 0.5625 = 1593.0 -> чётное 1594. Ширина НЕ ТРОНУТА.
        self.assertEqual((got["width"], got["height"]), (896, 1594))
        self.assertAlmostEqual(got["width"] / got["height"], 0.5625, places=2)

    def test_nothing_is_ever_lost(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ: канвас обязан быть НЕ МЕНЬШЕ картинки по обеим
        # сторонам. Обрезка прошла бы здесь и нигде больше.
        for w, h in ((896, 1200), (1024, 1024), (620, 1104), (1920, 1080)):
            with self.subTest(size=(w, h)):
                got = P.canvas_for(w, h)
                self.assertGreaterEqual(got["width"], w)
                self.assertGreaterEqual(got["height"], h)
                self.assertGreaterEqual(got["left"], 0)
                self.assertGreaterEqual(got["top"], 0)

    def test_an_image_already_in_plan_is_left_almost_alone(self):
        got = P.canvas_for(1080, 1920)
        self.assertEqual((got["width"], got["height"]), (1080, 1920))
        self.assertEqual(got["added_share"], 0.0)

    def test_a_landscape_image_grows_in_height_not_shrinks_in_width(self):
        got = P.canvas_for(1920, 1080)
        self.assertEqual(got["width"], 1920)
        self.assertEqual(got["height"], 3414)

    def test_the_sides_are_even_because_h264_refuses_odd_ones(self):
        for w, h in ((897, 1201), (895, 1199), (333, 777)):
            with self.subTest(size=(w, h)):
                got = P.canvas_for(w, h)
                self.assertEqual(got["width"] % 2, 0)
                self.assertEqual(got["height"] % 2, 0)

    def test_nonsense_sizes_are_refused_not_guessed(self):
        for bad in ((0, 100), (100, -1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                P.canvas_for(*bad)
        for bad in (("896", 1200), (896.0, 1200)):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                P.canvas_for(*bad)

    def test_mutating_the_plan_ratio_both_ways_moves_the_canvas(self):
        """Т1: PLAN_RATIO строже (уже) и слабее (шире)."""
        was = P.PLAN_RATIO
        try:
            P.PLAN_RATIO = 0.5
            self.assertEqual(P.canvas_for(896, 1200)["height"], 1792)
            # 896/1200 = 0.7467 УЖЕ плана 0.75, значит растёт ШИРИНА,
            # а высота остаётся: 1200 * 0.75 = 900.
            P.PLAN_RATIO = 0.75
            self.assertEqual(P.canvas_for(896, 1200)["width"], 900)
        finally:
            P.PLAN_RATIO = was
        self.assertEqual(P.PLAN_RATIO, 0.5625)


class TheVerdictHasThreeOutcomesOnEveryAxis(unittest.TestCase):

    def test_a_photo_in_plan_is_plainly_good(self):
        got = P.plan_verdict(width=1080, height=1920, points=GOOD, face_px=140)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["violations"], 0)

    def test_the_wrong_canvas_is_a_defect(self):
        # Ровно наш случай: 896x1200 после стилизатора.
        got = P.plan_verdict(width=896, height=1200, points=GOOD, face_px=140)
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual([a["outcome"] for a in got["axes"] if a["name"] == "канвас"],
                         [FAIL])

    def test_a_portrait_crop_fails_on_the_ankles(self):
        # Наши боевые фото: портрет по грудь, щиколоток нет вовсе.
        waist_up = {k: v for k, v in GOOD.items() if "ankle" not in k
                    and "knee" not in k}
        got = P.plan_verdict(width=1080, height=1920, points=waist_up,
                             face_px=300)
        names = {a["name"]: a["outcome"] for a in got["axes"]}
        self.assertEqual(names["щиколотки"], UNMEASURED)
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_an_off_centre_person_is_a_defect(self):
        got = P.plan_verdict(width=1080, height=1920,
                             points=shifted(GOOD, dx=0.2), face_px=140)
        self.assertEqual(got["outcome"], FAIL)

    def test_a_person_filling_the_width_is_a_defect(self):
        wide = dict(GOOD, l_wrist=(0.95, 0.62, 0.97), r_wrist=(0.05, 0.62, 0.97))
        got = P.plan_verdict(width=1080, height=1920, points=wide, face_px=140)
        self.assertEqual(got["outcome"], FAIL)

    def test_a_small_face_only_warns_and_does_not_sink_the_verdict(self):
        # Решение владельца 22.08, та же ось, что в приёме драйвинга: полный
        # рост делает лицо мельче по устройству, и ронять на этом вердикт
        # значило бы запретить сам выбранный план.
        got = P.plan_verdict(width=1080, height=1920, points=GOOD, face_px=61)
        self.assertEqual(got["outcome"], PASS)
        self.assertIn("ПРЕДУПРЕЖДЕНИЕ", got["note"])

    def test_a_big_face_gets_NO_warning(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ предупреждения: оно обязано молчать на годном.
        got = P.plan_verdict(width=1080, height=1920, points=GOOD, face_px=140)
        self.assertNotIn("ПРЕДУПРЕЖДЕНИЕ", got["note"])

    def test_a_face_never_asked_about_is_UNMEASURED_not_absent(self):
        got = P.plan_verdict(width=1080, height=1920, points=GOOD)
        names = {a["name"]: a["outcome"] for a in got["axes"]}
        self.assertEqual(names["лицо"], UNMEASURED)

    def test_a_missing_size_is_UNMEASURED_not_failed(self):
        got = P.plan_verdict(points=GOOD, face_px=140)
        names = {a["name"]: a["outcome"] for a in got["axes"]}
        self.assertEqual(names["канвас"], UNMEASURED)
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_mutating_each_band_both_ways_turns_the_verdict(self):
        """Т1 по каждой полосе: строже роняет годное, слабее пропускает брак."""
        ok = dict(width=1080, height=1920, points=GOOD, face_px=140)
        wide = dict(GOOD, l_wrist=(0.95, 0.62, 0.97), r_wrist=(0.05, 0.62, 0.97))
        # У КАЖДОЙ полосы свой брак, нарушающий ТОЛЬКО её. Общий сдвинутый
        # вход нарушал сразу несколько осей, и «слабее» никогда не проходило —
        # тест мерил бы не ту полосу, которую мутирует.
        # У CENTRE_TOL своё «годное»: GOOD стоит РОВНО по центру, отклонение
        # 0.0, и никакая строгая планка на нём не покраснеет. Мутацию видно
        # только на входе, который допуском пользуется.
        off_centre = dict(ok, points=shifted(GOOD, dx=0.05))
        cases = (
            ("SHOULDERS_BAND", (0.40, 0.42), (0.0, 1.0), ok,
             dict(ok, points=dict(GOOD, l_shoulder=(0.58, 0.55, 0.99),
                                  r_shoulder=(0.42, 0.55, 0.99)))),
            ("ANKLES_BAND", (0.95, 0.99), (0.0, 1.0), ok,
             dict(ok, points=dict(GOOD, l_ankle=(0.55, 0.62, 0.96),
                                  r_ankle=(0.45, 0.62, 0.96)))),
            ("CENTRE_TOL", 0.0001, 0.5, off_centre,
             dict(ok, points=shifted(GOOD, dx=0.2))),
            ("WIDTH_MAX", 0.05, 1.0, ok, dict(ok, points=wide)),
        )
        for name, strict, loose, good, bad in cases:
            was = getattr(P, name)
            try:
                setattr(P, name, was)
                self.assertEqual(P.plan_verdict(**good)["outcome"], PASS,
                                 f"{name}: годное обязано проходить НА БОЕВОЙ полосе")
                setattr(P, name, strict)
                self.assertEqual(P.plan_verdict(**good)["outcome"], FAIL,
                                 f"{name} строже не покраснел")
                setattr(P, name, was)
                self.assertEqual(P.plan_verdict(**bad)["outcome"], FAIL,
                                 f"{name}: брак обязан краснеть НА БОЕВОЙ полосе")
                setattr(P, name, loose)
                self.assertEqual(P.plan_verdict(**bad)["outcome"], PASS,
                                 f"{name} слабее не пропустил")
            finally:
                setattr(P, name, was)


class TheFullBodyPromptIsADecisionNotAString(unittest.TestCase):

    def test_the_prompt_asks_for_head_to_feet_and_keeps_the_person(self):
        got = P.full_body_prompt()
        self.assertIn("FULL HEIGHT", got)
        self.assertIn("head to feet", got)
        self.assertIn("same person", got)

    def test_removing_the_identity_clause_is_visible_in_the_prompt(self):
        # Сторож дефекта: без этой строки НБ2 отдавал похожего человека, а не
        # того же. Мутация видна в тексте, значит и в отчёте.
        with mock.patch.object(P, "KEEP_IDENTITY_CLAUSE", ""):
            self.assertNotIn("same person", P.full_body_prompt())

    def test_the_brand_ban_is_in_the_prompt_and_is_not_a_copy(self):
        # СТОРОЖ НАБЛЮДЕННОГО ДЕФЕКТА: без запрета первый прогон дорисовал на
        # майке читаемую надпись «NYCM MARATHON». Прибор её не увидел.
        from ball_reel import fork_e2e

        self.assertIn(fork_e2e.NO_BRANDS_CLAUSE, P.full_body_prompt())
        self.assertIn("no logos", P.full_body_prompt())

    def test_the_ban_comes_from_the_stand_not_from_a_local_copy(self):
        # Е1: одно решение — одно место. Если запрет здесь скопирован, эта
        # подмена его не сдвинет, и тест покраснеет.
        from ball_reel import fork_e2e

        with mock.patch.object(fork_e2e, "NO_BRANDS_CLAUSE", "ЗАПРЕТ-ПОДМЕНА"):
            self.assertIn("ЗАПРЕТ-ПОДМЕНА", P.full_body_prompt())

    def test_extra_words_are_appended_not_substituted(self):
        got = P.full_body_prompt(extra="plain grey studio background")
        self.assertIn("FULL HEIGHT", got)
        self.assertIn("grey studio", got)


class TheDiskIsAnInjectionPoint(unittest.TestCase):
    """Без PIL и без диска модуль обязан проверяться (Т4)."""

    def test_an_unopenable_image_is_UNMEASURED_not_failed(self):
        def broken(_):
            raise OSError("файла нет")

        got = P.to_plan("нет.png", "out.png", opener=broken)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("OSError", got["note"])
        self.assertIsNone(got["path"])


if __name__ == "__main__":
    unittest.main()
