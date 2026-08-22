"""Сторожа приёма входов. Каждый тест ловит ДЕФЕКТ, а не строчку.

ЧТО ЗДЕСЬ ЕСТЬ, КРОМЕ ПРОВЕРОК ПОВЕДЕНИЯ:

    мутация констант-решений В ОБЕ СТОРОНЫ (Т1) — строже и слабее, для
        MIN_SCENE_SECONDS, ORPHAN_WRIST_WARN, MIN_FACE_PX, MIN_VISIBILITY,
        FRAME_COUNT_EXACT, PHOTO_PEOPLE_EXPECTED;
    негативный контроль у КАЖДОГО прибора (И5) — вход, на котором прибор
        обязан сказать «нет», рядом с входом, на котором он обязан шевельнуться;
    третий исход (Р1) — отдельными тестами на то, что «не смогли» не свернулось
        ни в «годно», ни в «не годно».

ОЖИДАЕМОЕ ВЕЗДЕ ЛИТЕРАЛ (Т2): `3.0`, `100`, `0.5` написаны здесь руками. Если
завтра планку тронут в модуле, эти тесты обязаны покраснеть — импортированное
ожидание уехало бы вместе с кодом и промолчало.

СЕТИ И ДИСКА ЗДЕСЬ НЕТ, и это обеспечено конструкцией, а не договорённостью
(Т4): каждый прибор получает свою точку внедрения списком-заглушкой.
"""

from __future__ import annotations

import unittest

from ball_reel import fork_intake as fi
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED


# --------------------------------------------------------------------------
# Заглушки точек внедрения
# --------------------------------------------------------------------------

PROBE_JSON = ('{"programs":[],"streams":[{"avg_frame_rate":"30/1",'
              '"duration":"10.166667","nb_read_frames":"305"}]}')

#: Настоящий хвост stderr от ffmpeg -stats, с возвратами каретки и первым
#: вхождением `frame= 0`. Записан с живого прогона, а не сочинён.
FFMPEG_STATS = ("frame=    0 fps=0.0 q=-0.0 size=       0kB time=00:00:00.00 "
                "bitrate=N/A speed=   0x    \rframe=  307 fps=0.0 q=-0.0 "
                "Lsize=N/A time=00:00:10.20 bitrate=N/A speed=37.6x    ")


def probe_stub(text=PROBE_JSON, ran=True, why=""):
    def prober(path):
        return {"ran": ran, "code": 0, "out": text, "err": "", "why": why}
    return prober


def decode_stub(plain: int, fixed: int):
    def decoder(path, *, vsync0):
        n = fixed if vsync0 else plain
        return {"ran": True, "code": 0, "out": "",
                "err": f"frame=    0 ...\rframe= {n} Lsize=N/A", "why": ""}
    return decoder


def pose_point(x, y, vis):
    return (x, y, vis)


CLEAN_POSE = {
    "l_shoulder": pose_point(0.4, 0.3, 0.9), "r_shoulder": pose_point(0.6, 0.3, 0.9),
    "l_elbow": pose_point(0.35, 0.5, 0.9), "r_elbow": pose_point(0.65, 0.5, 0.9),
    "l_wrist": pose_point(0.3, 0.7, 0.9), "r_wrist": pose_point(0.7, 0.7, 0.9),
}

#: Кисть видна, локоть той же руки — нет. Ровно определение сироты.
ORPHAN_POSE = {**CLEAN_POSE, "l_elbow": pose_point(0.35, 0.5, 0.2)}


# --------------------------------------------------------------------------
# Ось 1: дырки во временных метках
# --------------------------------------------------------------------------

class Timestamps(unittest.TestCase):

    def test_the_selfie_numbers_are_called_a_defect(self):
        """ИЗМЕРЕНО на driving_selfie: 305 / 307 / 305. Это «не годно»."""
        v = fi.timestamp_verdict(305, 307, 305)
        self.assertEqual(v["outcome"], "не годно")
        self.assertEqual((v["checked"], v["violations"], v["unmeasured"]),
                         (1, 1, 0))
        self.assertEqual(v["gap"], 2)
        self.assertIn("-vsync 0", v["advice"])

    def test_the_arms_numbers_are_clean(self):
        """НЕГАТИВНЫЙ КОНТРОЛЬ прибора (И5): на здоровом файле он молчит.

        Без этого теста «не годно» на всём подряд выглядело бы как работа.
        ИЗМЕРЕНО на driving_arms: 373 / 373 / 373.
        """
        v = fi.timestamp_verdict(373, 373, 373)
        self.assertEqual(v["outcome"], "годно")
        self.assertEqual(v["violations"], 0)
        self.assertEqual(v["advice"], "")

    def test_a_missing_counter_is_the_third_outcome(self):
        for probed, plain, fixed in ((None, 307, 305), (305, None, 305),
                                     (305, 307, None)):
            with self.subTest(triple=(probed, plain, fixed)):
                v = fi.timestamp_verdict(probed, plain, fixed)
                self.assertEqual(v["outcome"], "не смогли проверить")
                self.assertEqual(v["checked"], 0)
                self.assertEqual(v["violations"], 0)

    def test_vsync_that_does_not_heal_is_said_out_loud(self):
        v = fi.timestamp_verdict(305, 307, 306)
        self.assertEqual(v["outcome"], "не годно")
        self.assertIn("НЕ ЛЕЧИТ", v["note"])

    def test_mutating_the_tolerance_both_ways_turns_a_verdict(self):
        """Т1: FRAME_COUNT_EXACT сторожится в обе стороны.

        Слабее (допуск 2) — дефект 305/307 обязан ПЕРЕСТАТЬ ловиться;
        строже он быть уже не может (0 — минимум), поэтому вторая сторона
        мутируется на здоровом файле через отрицательный допуск.
        """
        was = fi.FRAME_COUNT_EXACT
        try:
            fi.FRAME_COUNT_EXACT = 2
            self.assertEqual(fi.timestamp_verdict(305, 307, 305)["outcome"],
                             "годно", "допуск 2 обязан пропустить дефект")
            fi.FRAME_COUNT_EXACT = -1
            self.assertEqual(fi.timestamp_verdict(373, 373, 373)["outcome"],
                             "не годно", "отрицательный допуск обязан ронять "
                                         "даже точное совпадение")
        finally:
            fi.FRAME_COUNT_EXACT = was
        self.assertEqual(fi.FRAME_COUNT_EXACT, 0)


class ParsingTheInstruments(unittest.TestCase):

    def test_count_frames_json_is_parsed_from_a_recorded_answer(self):
        r = fi.parse_count_frames(PROBE_JSON)
        self.assertTrue(r["ok"])
        self.assertEqual(r["frames"], 305)
        self.assertEqual(r["fps"], 30.0)

    def test_garbage_is_not_a_frame_count(self):
        """НЕГАТИВНЫЙ КОНТРОЛЬ разбора: мусор обязан дать «не смогли»."""
        for text in ("", "не json", "{}", '{"streams":[{}]}',
                     '{"streams":[{"nb_read_frames":"N/A"}]}',
                     '{"streams":[{"nb_read_frames":"0"}]}'):
            with self.subTest(text=text):
                r = fi.parse_count_frames(text)
                self.assertFalse(r["ok"])
                self.assertIsNone(r["frames"])
                self.assertTrue(r["why"])

    def test_the_last_frame_number_wins_not_the_first(self):
        """Дефект, ради которого тест написан: первое вхождение — `frame= 0`."""
        r = fi.parse_decoded_frames(FFMPEG_STATS)
        self.assertTrue(r["ok"])
        self.assertEqual(r["frames"], 307)

    def test_stats_without_a_frame_line_is_unmeasured(self):
        r = fi.parse_decoded_frames("Output file is empty, nothing was encoded")
        self.assertFalse(r["ok"])
        self.assertIsNone(r["frames"])


# --------------------------------------------------------------------------
# Оси 2 и 3: швы и длина сцены
# --------------------------------------------------------------------------

class Scenes(unittest.TestCase):

    def test_no_cuts_is_one_scene_over_the_whole_clip(self):
        self.assertEqual(fi.scenes(373, []),
                         [{"start": 0, "end": 372, "frames": 373}])

    def test_a_cut_after_frame_k_splits_between_k_and_k_plus_one(self):
        self.assertEqual(fi.scenes(10, [3]),
                         [{"start": 0, "end": 3, "frames": 4},
                          {"start": 4, "end": 9, "frames": 6}])

    def test_two_cuts_give_three_scenes_and_the_frames_add_up(self):
        got = fi.scenes(100, [29, 59])
        self.assertEqual([s["frames"] for s in got], [30, 30, 40])
        self.assertEqual(sum(s["frames"] for s in got), 100)

    def test_scene_shorter_than_three_seconds_is_refused(self):
        """КРИТЕРИЙ ПРИЁМА. 89 кадров при 30 к/с — это 2.967 с."""
        v = fi.scene_length_verdict(fi.scenes(200, [88]), 30.0)
        self.assertEqual(v["outcome"], "не годно")
        self.assertEqual(v["short"], [0])
        self.assertEqual(v["seconds"][0], 2.967)
        self.assertEqual((v["checked"], v["violations"]), (2, 1))

    def test_exactly_three_seconds_passes(self):
        """Граница включительна: 90 кадров при 30 к/с — ровно 3.0 с."""
        v = fi.scene_length_verdict(fi.scenes(200, [89]), 30.0)
        self.assertEqual(v["outcome"], "годно")
        self.assertEqual(v["seconds"][0], 3.0)

    def test_without_fps_the_length_is_the_third_outcome(self):
        v = fi.scene_length_verdict(fi.scenes(200, [88]), None)
        self.assertEqual(v["outcome"], "не смогли проверить")
        self.assertEqual(v["checked"], 0)
        self.assertEqual(v["violations"], 0)
        v0 = fi.scene_length_verdict(fi.scenes(200, [88]), 0)
        self.assertEqual(v0["outcome"], "не смогли проверить")

    def test_mutating_the_scene_bar_both_ways_turns_the_verdict(self):
        """Т1: MIN_SCENE_SECONDS сторожится строже и слабее.

        Сцена 3.0 с. Планка 3.5 — обязана краснеть; планка 2.5 — обязана
        зеленеть. Значит константа действительно решает, а не украшает.
        """
        scene_list = fi.scenes(200, [89])
        self.assertEqual(fi.scene_length_verdict(scene_list, 30.0,
                                                 min_seconds=3.5)["outcome"],
                         "не годно")
        self.assertEqual(fi.scene_length_verdict(scene_list, 30.0,
                                                 min_seconds=2.5)["outcome"],
                         "годно")
        was = fi.MIN_SCENE_SECONDS
        try:
            fi.MIN_SCENE_SECONDS = 3.5
            self.assertEqual(fi.scene_length_verdict(scene_list, 30.0)["outcome"],
                             "не годно", "подмена самой константы обязана "
                                         "менять вердикт, а не только параметр")
            fi.MIN_SCENE_SECONDS = 2.5
            self.assertEqual(fi.scene_length_verdict(scene_list, 30.0)["outcome"],
                             "годно")
        finally:
            fi.MIN_SCENE_SECONDS = was
        self.assertEqual(fi.MIN_SCENE_SECONDS, 3.0)

    def test_the_bar_is_the_owners_three_seconds(self):
        self.assertEqual(fi.MIN_SCENE_SECONDS, 3.0)


# --------------------------------------------------------------------------
# Ось 4: сиротские кисти — МЯГКАЯ
# --------------------------------------------------------------------------

class OrphanWrists(unittest.TestCase):

    def test_a_visible_wrist_without_its_elbow_is_an_orphan(self):
        self.assertIs(fi.is_orphan_wrist(ORPHAN_POSE), True)

    def test_a_whole_arm_is_not_an_orphan(self):
        """НЕГАТИВНЫЙ КОНТРОЛЬ (И5): прибор, кричащий всегда, ничего не мерит."""
        self.assertIs(fi.is_orphan_wrist(CLEAN_POSE), False)

    def test_an_invisible_wrist_is_not_an_orphan(self):
        """Сирота — про ВИДНУЮ кисть. Невидимая кисть нарушением не является."""
        pts = {**CLEAN_POSE, "l_wrist": pose_point(0.3, 0.7, 0.1),
               "l_elbow": pose_point(0.35, 0.5, 0.1)}
        self.assertIs(fi.is_orphan_wrist(pts), False)

    def test_a_wrist_outside_the_frame_is_not_counted(self):
        pts = {**CLEAN_POSE, "l_wrist": pose_point(1.4, 0.7, 0.9),
               "l_elbow": pose_point(0.35, 0.5, 0.1)}
        self.assertIs(fi.is_orphan_wrist(pts), False)

    def test_a_missing_shoulder_orphans_the_wrist_too(self):
        pts = {**CLEAN_POSE, "r_shoulder": pose_point(0.6, 0.3, 0.2)}
        self.assertIs(fi.is_orphan_wrist(pts), True)

    def test_no_pose_is_the_third_outcome_not_false(self):
        self.assertIsNone(fi.is_orphan_wrist(None))
        self.assertIsNone(fi.is_orphan_wrist({}))

    def test_mutating_visibility_both_ways_changes_who_is_an_orphan(self):
        """Т1: MIN_VISIBILITY сторожится в обе стороны.

        Локоть подан с видимостью 0.2. Планка 0.1 — локоть считается видным,
        сироты нет; планка 0.95 — не видно уже ничего, и сирота исчезает
        по другой причине (кисть тоже перестала быть видной). Обе стороны
        обязаны менять ответ, иначе константу никто не сторожит.
        """
        was = fi.MIN_VISIBILITY
        try:
            fi.MIN_VISIBILITY = 0.1
            self.assertIs(fi.is_orphan_wrist(ORPHAN_POSE), False,
                          "слабая планка обязана перестать видеть сироту")
            fi.MIN_VISIBILITY = 0.95
            self.assertIs(fi.is_orphan_wrist(ORPHAN_POSE), False,
                          "строгая планка обязана погасить и кисть")
        finally:
            fi.MIN_VISIBILITY = was
        self.assertEqual(fi.MIN_VISIBILITY, 0.5)

    def test_the_soft_axis_never_says_not_good(self):
        """ГЛАВНОЕ свойство этой оси: она НЕ критерий отказа.

        Владелец посмотрел выход с 21% сирот и назвал кисти правильными.
        """
        for share in (0.0, 0.04, 0.21, 0.99, 1.0):
            with self.subTest(share=share):
                v = fi.orphan_verdict(share, 100, 0)
                self.assertNotEqual(v["outcome"], "не годно")
                self.assertEqual(v["violations"], 0)

    def test_the_measured_share_warns_and_a_small_one_does_not(self):
        self.assertTrue(fi.orphan_verdict(0.21, 99, 0)["warn"])
        self.assertFalse(fi.orphan_verdict(0.04, 373, 0)["warn"])

    def test_mutating_the_warning_bar_both_ways_turns_the_warning(self):
        """Т1: ORPHAN_WRIST_WARN, строже и слабее, на ИЗМЕРЕННЫХ точках 4% и 21%."""
        was = fi.ORPHAN_WRIST_WARN
        try:
            fi.ORPHAN_WRIST_WARN = 0.30
            self.assertFalse(fi.orphan_verdict(0.21, 99, 0)["warn"],
                             "планка 30% обязана снять предупреждение с 21%")
            fi.ORPHAN_WRIST_WARN = 0.01
            self.assertTrue(fi.orphan_verdict(0.04, 373, 0)["warn"],
                            "планка 1% обязана поднять предупреждение на 4%")
        finally:
            fi.ORPHAN_WRIST_WARN = was
        self.assertEqual(fi.ORPHAN_WRIST_WARN, 0.10)

    def test_no_pose_anywhere_is_unmeasured_not_zero_orphans(self):
        v = fi.orphan_verdict(None, 0, 12)
        self.assertEqual(v["outcome"], "не смогли проверить")
        self.assertIsNone(v["share"])
        self.assertFalse(v["warn"])


# --------------------------------------------------------------------------
# Ось 5: размер лица
# --------------------------------------------------------------------------

class FaceSize(unittest.TestCase):

    def test_the_selfie_range_passes_the_bar(self):
        """ИЗМЕРЕНО: driving_selfie 234..369 px — планка не мешает."""
        v = fi.face_size_verdict([234, 300, 369], 0, 0)
        self.assertEqual(v["outcome"], "годно")
        self.assertEqual((v["checked"], v["violations"]), (3, 0))

    def test_the_yogaball_range_is_thrown_out_whole(self):
        """ИЗМЕРЕНО: driving_yogaball 87..96 px — планка выбрасывает всё."""
        v = fi.face_size_verdict([87, 90, 96], 0, 0)
        self.assertEqual(v["outcome"], "не годно")
        self.assertEqual(v["small"], 3)

    def test_a_frame_without_a_face_is_a_violation_not_an_excuse(self):
        v = fi.face_size_verdict([234], 5, 0)
        self.assertEqual(v["outcome"], "не годно")
        self.assertEqual(v["violations"], 5)

    def test_a_detector_that_could_not_be_asked_is_the_third_outcome(self):
        v = fi.face_size_verdict([], 0, 7)
        self.assertEqual(v["outcome"], "не смогли проверить")
        self.assertEqual(v["checked"], 0)
        self.assertEqual(v["violations"], 0)

    def test_mutating_the_face_bar_both_ways_turns_the_verdict(self):
        """Т1: MIN_FACE_PX строже и слабее, на измеренных 87..96 и 234..369."""
        self.assertEqual(fi.face_size_verdict([87, 96], 0, 0,
                                              min_face_px=80)["outcome"],
                         "годно")
        self.assertEqual(fi.face_size_verdict([234, 369], 0, 0,
                                              min_face_px=400)["outcome"],
                         "не годно")
        was = fi.MIN_FACE_PX
        try:
            fi.MIN_FACE_PX = 80
            self.assertEqual(fi.face_size_verdict([87, 96], 0, 0)["outcome"],
                             "годно")
            fi.MIN_FACE_PX = 400
            self.assertEqual(fi.face_size_verdict([234, 369], 0, 0)["outcome"],
                             "не годно")
        finally:
            fi.MIN_FACE_PX = was
        self.assertEqual(fi.MIN_FACE_PX, 100)


# --------------------------------------------------------------------------
# Выбор окна
# --------------------------------------------------------------------------

class Window(unittest.TestCase):

    def test_the_window_is_in_frame_numbers_and_sits_in_the_middle(self):
        scene_list = [{"start": 0, "end": 199, "frames": 200}]
        v = fi.window(scene_list, 5.0, 30.0)
        self.assertEqual(v["outcome"], "годно")
        self.assertEqual((v["start"], v["end"], v["frames"]), (25, 174, 150))
        self.assertEqual(v["scene"], 0)

    def test_the_longest_scene_wins(self):
        scene_list = [{"start": 0, "end": 99, "frames": 100},
                      {"start": 100, "end": 399, "frames": 300}]
        v = fi.window(scene_list, 5.0, 30.0)
        self.assertEqual(v["scene"], 1)
        self.assertEqual((v["start"], v["end"]), (175, 324))

    def test_a_window_that_does_not_fit_is_refused_not_shrunk(self):
        scene_list = [{"start": 0, "end": 89, "frames": 90}]
        v = fi.window(scene_list, 5.0, 30.0)
        self.assertEqual(v["outcome"], "не годно")
        self.assertIsNone(v["start"])
        self.assertEqual(v["violations"], 1)

    def test_without_fps_the_window_is_unmeasured_not_thirty(self):
        v = fi.window([{"start": 0, "end": 199, "frames": 200}], 5.0, None)
        self.assertEqual(v["outcome"], "не смогли проверить")
        self.assertIsNone(v["start"])

    def test_without_a_markup_the_window_is_unmeasured(self):
        v = fi.window([], 5.0, 30.0)
        self.assertEqual(v["outcome"], "не смогли проверить")

    def test_the_command_carries_setpts_and_frame_numbers(self):
        argv = fi.window_argv("in.mp4", "out.mp4", 25, 174)
        self.assertIn("-vf", argv)
        expr = argv[argv.index("-vf") + 1]
        self.assertEqual(expr,
                         "select='between(n\\,25\\,174)',setpts=N/30/TB")
        self.assertIn("-an", argv)
        self.assertEqual(argv[-1], "out.mp4")

    def test_dropping_setpts_would_be_the_422(self):
        """Сторож дефекта, а не строчки: без setpts Wan отвечал 422."""
        expr = fi.window_argv("in.mp4", "out.mp4", 0, 10)[
            fi.window_argv("in.mp4", "out.mp4", 0, 10).index("-vf") + 1]
        self.assertIn("setpts=", expr)

    def test_broken_bounds_raise_instead_of_guessing(self):
        for a, b in ((5, 4), (-1, 10)):
            with self.subTest(bounds=(a, b)):
                with self.assertRaises(ValueError):
                    fi.window_argv("in.mp4", "out.mp4", a, b)


# --------------------------------------------------------------------------
# Три числа рядом с вердиктом
# --------------------------------------------------------------------------

class ThreeOutcomesAndThreeNumbers(unittest.TestCase):

    def test_zero_violations_over_zero_checks_is_not_success(self):
        """Р2 дословно: ноль нарушений при нуле проверок — не «годно»."""
        self.assertEqual(fi.tally(0, 0, 0)["outcome"], "не смогли проверить")

    def test_a_partly_measured_run_does_not_round_up_to_good(self):
        self.assertEqual(fi.tally(10, 0, 3)["outcome"], "не смогли проверить")

    def test_a_violation_beats_an_unmeasured(self):
        self.assertEqual(fi.tally(10, 1, 3)["outcome"], "не годно")

    def test_a_clean_full_run_is_good(self):
        self.assertEqual(fi.tally(10, 0, 0)["outcome"], "годно")

    def test_the_three_outcomes_are_the_projects_three(self):
        self.assertEqual((PASS, FAIL, UNMEASURED),
                         ("годно", "не годно", "не смогли проверить"))


# --------------------------------------------------------------------------
# Приборы целиком, на заглушках: ни диска, ни сети
# --------------------------------------------------------------------------

class DrivingIntake(unittest.TestCase):

    # 95 кадров при 30 к/с — 3.167 с, то есть единственная сцена ПРОХОДИТ
    # планку 3.0 с. Число выбрано так намеренно: иначе на каждом стенде
    # краснела бы ось длины сцены, и ось, ради которой тест написан, была бы
    # не видна за ней.
    def _run(self, *, plain, fixed, poses, faces, n=95, cut_at=None,
             product=None):
        paths = [f"{i:05d}.png" for i in range(n)]
        cut_at = [] if cut_at is None else cut_at

        def gray(path):
            import numpy as np
            k = int(str(path).split(".")[0])
            # Ступенька яркости ровно на шве: типичный шаг 1, шов — 100.
            base = sum(100 for c in cut_at if k > c) + k
            return np.full((4, 4), float(base))

        def pose_reader(path):
            return {"points": poses.get(str(path), CLEAN_POSE), "why": "",
                    "people": None}

        def face_prober(path):
            return faces.get(str(path), {"face_px": 300})

        return fi.driving_intake(
            "clip.mp4", paths, product_seconds=product,
            prober=probe_stub(), decoder=decode_stub(plain, fixed),
            gray=gray, pose_reader=pose_reader, face_prober=face_prober)

    def test_a_clean_clip_passes_and_the_soft_axis_is_outside_the_verdict(self):
        r = self._run(plain=305, fixed=305,
                      poses={"00002.png": ORPHAN_POSE}, faces={})
        self.assertEqual(r["axes"]["timestamps"]["outcome"], "годно")
        self.assertEqual(r["axes"]["orphan_wrists"]["outcome"], "годно")
        self.assertGreater(r["axes"]["orphan_wrists"]["share"], 0.0)
        self.assertIn("orphan_wrists", r["soft"])
        self.assertEqual(r["outcome"], "годно")

    def test_the_timestamp_defect_alone_sinks_the_verdict(self):
        r = self._run(plain=307, fixed=305, poses={}, faces={})
        self.assertEqual(r["axes"]["timestamps"]["outcome"], "не годно")
        self.assertEqual(r["outcome"], "не годно")
        self.assertIn("-vsync 0", fi.render(r))

    def test_orphan_wrists_alone_never_sink_the_verdict(self):
        """Сторож решения владельца: 100% сирот — предупреждение, не отказ."""
        r = self._run(plain=305, fixed=305,
                      poses={f"{i:05d}.png": ORPHAN_POSE for i in range(95)},
                      faces={})
        self.assertEqual(r["axes"]["orphan_wrists"]["share"], 1.0)
        self.assertTrue(r["axes"]["orphan_wrists"]["warn"])
        self.assertEqual(r["outcome"], "годно")
        self.assertIn("orphan_wrists", r["warnings"])

    def test_small_faces_do_sink_the_verdict(self):
        r = self._run(plain=305, fixed=305, poses={},
                      faces={f"{i:05d}.png": {"face_px": 90} for i in range(95)})
        self.assertEqual(r["axes"]["face_size"]["outcome"], "не годно")
        self.assertEqual(r["outcome"], "не годно")

    def test_a_cut_is_marked_up_and_short_scenes_are_refused(self):
        r = self._run(plain=305, fixed=305, poses={}, faces={}, n=6, cut_at=[2])
        self.assertEqual(r["axes"]["cuts"]["cuts"], [2])
        self.assertEqual([s["frames"] for s in r["scenes"]], [3, 3])
        # 3 кадра при 30 к/с — 0.1 с, то есть заведомо короче трёх секунд.
        self.assertEqual(r["axes"]["scenes"]["outcome"], "не годно")
        self.assertEqual(r["axes"]["scenes"]["short"], [0, 1])

    def test_without_frames_the_frame_axes_are_unmeasured_not_clean(self):
        """Р1: нет кадров — «не смогли», а не «швов нет, сирот нет»."""
        r = fi.driving_intake("clip.mp4", [], prober=probe_stub(),
                              decoder=decode_stub(305, 305))
        for axis in ("cuts", "scenes", "orphan_wrists", "face_size"):
            with self.subTest(axis=axis):
                self.assertEqual(r["axes"][axis]["outcome"],
                                 "не смогли проверить")
        self.assertEqual(r["axes"]["timestamps"]["outcome"], "годно")
        self.assertEqual(r["outcome"], "не смогли проверить")

    def test_a_dead_prober_does_not_become_a_bad_file(self):
        def dead(path):
            return {"ran": False, "code": None, "out": "", "err": "",
                    "why": "ffprobe не найден"}
        r = fi.driving_intake("clip.mp4", [], prober=dead,
                              decoder=decode_stub(305, 305))
        self.assertEqual(r["axes"]["timestamps"]["outcome"],
                         "не смогли проверить")
        self.assertNotEqual(r["axes"]["timestamps"]["outcome"], "не годно")


class PhotoIntake(unittest.TestCase):

    def test_one_big_face_passes(self):
        r = fi.photo_intake("p.png", faces_prober=lambda p: {
            "faces": [{"face_px": 420, "det_score": 0.9}], "why": ""})
        self.assertEqual(r["outcome"], "годно")
        self.assertEqual(r["axes"]["face_size"]["face_px"], 420)

    def test_no_face_is_a_violation_and_not_the_third_outcome(self):
        r = fi.photo_intake("p.png",
                            faces_prober=lambda p: {"faces": [], "why": ""})
        self.assertEqual(r["axes"]["face_found"]["outcome"], "не годно")
        self.assertEqual(r["axes"]["face_size"]["outcome"],
                         "не смогли проверить")
        self.assertEqual(r["outcome"], "не годно")

    def test_two_people_are_refused(self):
        r = fi.photo_intake("p.png", faces_prober=lambda p: {
            "faces": [{"face_px": 420, "det_score": 0.9},
                      {"face_px": 200, "det_score": 0.8}], "why": ""})
        self.assertEqual(r["axes"]["one_person"]["outcome"], "не годно")
        self.assertEqual(r["axes"]["face_found"]["outcome"], "годно")
        self.assertEqual(r["outcome"], "не годно")

    def test_a_small_face_is_refused(self):
        r = fi.photo_intake("p.png", faces_prober=lambda p: {
            "faces": [{"face_px": 60, "det_score": 0.9}], "why": ""})
        self.assertEqual(r["axes"]["face_size"]["outcome"], "не годно")

    def test_a_dead_detector_is_the_third_outcome_on_every_axis(self):
        r = fi.photo_intake("p.png", faces_prober=lambda p: {
            "faces": None, "why": "ModuleNotFoundError: insightface"})
        for axis in ("face_found", "face_size", "one_person"):
            with self.subTest(axis=axis):
                self.assertEqual(r["axes"][axis]["outcome"],
                                 "не смогли проверить")
        self.assertEqual(r["outcome"], "не смогли проверить")

    def test_mutating_the_expected_head_count_both_ways(self):
        """Т1: PHOTO_PEOPLE_EXPECTED строже и слабее."""
        two = lambda p: {"faces": [{"face_px": 420}, {"face_px": 200}],  # noqa: E731
                         "why": ""}
        one = lambda p: {"faces": [{"face_px": 420}], "why": ""}  # noqa: E731
        was = fi.PHOTO_PEOPLE_EXPECTED
        try:
            fi.PHOTO_PEOPLE_EXPECTED = 2
            self.assertEqual(
                fi.photo_intake("p.png", faces_prober=two)["axes"]["one_person"]["outcome"],
                "годно")
            self.assertEqual(
                fi.photo_intake("p.png", faces_prober=one)["axes"]["one_person"]["outcome"],
                "не годно")
        finally:
            fi.PHOTO_PEOPLE_EXPECTED = was
        self.assertEqual(fi.PHOTO_PEOPLE_EXPECTED, 1)


class StyleIntake(unittest.TestCase):

    GOOD = {"colours": ["off white", "steel blue"], "value_key": "light",
            "saturation": "muted", "texture": "visible grain"}

    def test_a_full_card_passes(self):
        r = fi.style_intake("s.png",
                            card_reader=lambda p: {"card": dict(self.GOOD),
                                                   "why": ""})
        self.assertEqual(r["outcome"], "годно")
        self.assertEqual(r["axes"]["card_readable"]["missing"], [])
        self.assertEqual(r["checked"], 4)

    def test_an_empty_field_is_a_violation(self):
        card = {**self.GOOD, "texture": ""}
        r = fi.style_intake("s.png",
                            card_reader=lambda p: {"card": card, "why": ""})
        self.assertEqual(r["outcome"], "не годно")
        self.assertEqual(r["axes"]["card_readable"]["missing"], ["texture"])

    def test_a_missing_package_is_the_third_outcome(self):
        """Р1: пакета нет — это НЕ «стиль плохой»."""
        r = fi.style_intake("s.png", card_reader=lambda p: {
            "card": None, "why": "ModuleNotFoundError: creative_eval"})
        self.assertEqual(r["outcome"], "не смогли проверить")
        self.assertEqual(r["violations"], 0)

    def test_the_expected_fields_are_a_literal_here(self):
        """Т2: список полей написан в тесте руками и не импортируется."""
        r = fi.style_intake("s.png", card_reader=lambda p: {
            "card": {"colours": ["red"]}, "why": ""})
        self.assertEqual(r["axes"]["card_readable"]["missing"],
                         ["value_key", "saturation", "texture"])


class BarsAreImportedNotCopied(unittest.TestCase):
    """Е1: планка живёт в одном месте, и здесь только ссылка на неё."""

    def test_the_person_bar_is_the_one_from_fork_identity(self):
        from ball_reel import fork_identity
        self.assertIs(fi.SAME_PERSON_MAX, fork_identity.SAME_PERSON_MAX)
        self.assertEqual(fi.SAME_PERSON_MAX, 0.35)

    def test_the_cut_bar_is_the_one_from_fork_looper(self):
        from ball_reel import fork_looper
        self.assertIs(fi.CUT_JUMP, fork_looper.CUT_JUMP)
        self.assertEqual(fi.CUT_JUMP, 4.0)

    def test_the_visibility_bar_is_the_one_from_pose(self):
        from ball_reel import pose
        self.assertIs(fi.MIN_VISIBILITY, pose.MIN_VISIBILITY)
        self.assertEqual(fi.MIN_VISIBILITY, 0.5)

    def test_the_face_bar_is_the_one_the_identity_axis_uses(self):
        from ball_reel import identity_arcface
        self.assertIs(fi.MIN_FACE_PX, identity_arcface.MIN_FACE_PX)
        self.assertEqual(fi.MIN_FACE_PX, 100)

    def test_the_module_does_not_redefine_a_bar_it_borrowed(self):
        """Сторож дефекта: копия планки числом в тексте модуля.

        Ищется присваивание НАШЕГО имени чужой планке — то есть ровно та
        форма, которой Е1 нарушается: `SAME_PERSON_MAX = 0.35` в этом файле.
        """
        import ast
        from pathlib import Path

        src = Path(fi.__file__).read_text(encoding="utf-8")
        borrowed = {"SAME_PERSON_MAX", "CUT_JUMP", "MIN_VISIBILITY",
                    "MIN_FACE_PX"}
        offenders = []
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in borrowed:
                        offenders.append(t.id)
        self.assertEqual(offenders, [],
                         f"планки переопределены в модуле: {offenders}")


class EveryInjectionPointIsAParameter(unittest.TestCase):
    """Т4 обеспечивается конструкцией: тест ходит только через параметры."""

    def test_the_public_instruments_all_take_their_world_as_an_argument(self):
        import inspect

        expected = {
            "driving_intake": {"prober", "decoder", "gray", "pose_reader",
                               "face_prober"},
            "photo_intake": {"faces_prober"},
            "style_intake": {"card_reader"},
        }
        for name, points in expected.items():
            with self.subTest(fn=name):
                params = set(inspect.signature(getattr(fi, name)).parameters)
                self.assertTrue(points <= params,
                                f"{name}: нет точек внедрения "
                                f"{sorted(points - params)}")

    def test_the_default_style_reader_does_not_import_the_banned_name(self):
        """Гейт форка стоит на ИМЕНИ `style`, и обойти его надо честно."""
        import ast
        from pathlib import Path

        for node in ast.walk(ast.parse(
                Path(fi.__file__).read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "style")
                for a in node.names:
                    self.assertNotEqual(a.name, "style")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    self.assertFalse(a.name == "style"
                                     or a.name.endswith(".style"))


class TheRenderShowsTheNumbers(unittest.TestCase):

    def test_every_axis_prints_its_three_numbers(self):
        r = fi.photo_intake("p.png", faces_prober=lambda p: {
            "faces": [{"face_px": 420}], "why": ""})
        text = fi.render(r)
        self.assertIn("проверено", text)
        self.assertIn("нарушений", text)
        self.assertIn("не смогли", text)
        self.assertIn("ВЕРДИКТ: годно", text)


if __name__ == "__main__":
    unittest.main()
