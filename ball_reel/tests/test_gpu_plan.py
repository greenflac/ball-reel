"""Разрешение как условие работоспособности ГЕЙТА, а не как качество картинки.

Найдено живым прогоном и стоит того, чтобы стоять отдельным файлом: ArcFace
отказывается судить лицо мельче своего порога, а на полноростовом вертикальном
кадре доля лица зафиксирована геометрией. Значит порог идентичности
превращается в требование к разрешению — и это требование надо знать ДО аренды
карты, а не выяснять после.

Числа геометрические, а не модельные: они одинаковы для gateway-моделей и для
того опенсорса, который поедет в прод.
"""

from __future__ import annotations

import unittest


class ResolutionDecidesWhetherIdentityCanBeJudgedAtAll(unittest.TestCase):
    def setUp(self):
        from ball_reel import gpu_keyframes

        self.g = gpu_keyframes

    def test_the_measured_share_reproduces_the_live_numbers(self):
        # Замерено: 1080x1920 дал лицо 181 px, 464x832 дал 78 px.
        self.assertAlmostEqual(self.g.face_px_at(1920), 180, delta=3)
        self.assertAlmostEqual(self.g.face_px_at(832), 78, delta=3)

    def test_both_gpu_plans_fail_full_body_and_say_so(self):
        # 512x768 и 640x960 — обе наши конфигурации. Ни одна не даёт судимого
        # лица на полном росте, и план обязан об этом предупредить, а не
        # обнаружиться на арендованной карте.
        for height in (768, 960):
            ok, why = self.g.identity_verifiable(height)
            self.assertFalse(ok, height)
            self.assertIn("НЕ СМОЖЕТ", why)

    def test_720p_and_above_are_judgeable(self):
        for height in (1280, 1920):
            ok, why = self.g.identity_verifiable(height)
            self.assertTrue(ok, why)

    def test_the_refusal_names_the_three_ways_out(self):
        _, why = self.g.identity_verifiable(768)
        self.assertIn("по высоте", why)      # поднять разрешение
        self.assertIn("по пояс", why)        # сменить кадрирование
        self.assertIn("не проверяется", why)  # принять честно

    def test_a_tighter_framing_rescues_the_small_frame(self):
        # Выход для 4 ГБ: поясная композиция даёт вдвое большую долю лица.
        ok, _ = self.g.identity_verifiable(
            768, share=self.g.WAIST_UP_FACE_SHARE)
        self.assertTrue(ok)

    def test_the_plan_carries_the_warning_where_it_will_be_read(self):
        notes = " ".join(self.g.plan(vram_gb=4.0).notes)
        self.assertIn("ИДЕНТИЧНОСТЬ НА ПОЛНОМ РОСТЕ", notes)

    def test_the_shares_are_read_from_the_module_with_literal_heights(self):
        full, waist = (self.g.FULL_BODY_FACE_SHARE, self.g.WAIST_UP_FACE_SHARE)
        self.assertGreater(waist, full)
        self.assertLess(full, 0.15)
        self.assertFalse(self.g.identity_verifiable(768, share=full)[0])
        self.assertTrue(self.g.identity_verifiable(768, share=waist)[0])


class TheSmokeVERDICTUsesTheModulesOwnBars(unittest.TestCase):
    """Гейт целевого пути судил мягче собственных порогов, и это никто не ловил.

    Логика вердикта жила внутри `main()` литеральными числами: 0.35 для лица
    (случайно совпало с `SAME_PERSON_MAX`) и 0.25 для позы, при том что
    `pose.SAME_POSE_MAX` равен 0.15. Расхождение на две трети, на ЦЕЛЕВОМ
    пути, и увидеть его можно было только сличив два файла глазами: тест сюда
    не доставал, мутация тоже — константа, не участвующая в коде, неубиваема.
    """

    def setUp(self):
        from ball_reel import run_local

        self.r = run_local

    def _delta(self, mean, worst, joint="l_wrist"):
        return {"mean": mean, "worst": worst, "worst_joint": joint}

    def test_a_pose_between_the_two_bars_is_now_rejected(self):
        # Ровно та щель, что была открыта: 0.20 проходило старый бар 0.25 и не
        # проходит настоящий 0.15. Числа литеральные намеренно — тест, берущий
        # вход из константы, которую сторожит, едет вместе с ней.
        from ball_reel.pose import SAME_POSE_MAX

        self.assertLess(SAME_POSE_MAX, 0.20)
        got = self.r.smoke_verdict(0.1, self._delta(0.20, 0.30))
        self.assertFalse(got["pose_ok"], got["pose_note"])

    def test_a_faithful_start_frame_still_passes(self):
        # Обратная сторона: ужесточение, бракующее всё, — не фикс, а регресс.
        # 0.043 — верх диапазона, замеренного на живых старт-кадрах (0.018-0.043).
        got = self.r.smoke_verdict(0.1, self._delta(0.043, 0.12))
        self.assertTrue(got["pose_ok"], got["pose_note"])
        self.assertTrue(got["face_ok"], got["face_note"])

    def test_one_limb_gone_astray_is_caught_by_the_worst_joint(self):
        # Худший сустав ПЕЧАТАЛСЯ, но не проверялся, хотя сообщение обещало
        # пользователю бар 0.40. Замеренный случай: согнутая рука даёт среднее
        # 0.11 при запястье 0.97 — среднее её размывает полностью.
        got = self.r.smoke_verdict(0.1, self._delta(0.11, 0.97))
        self.assertFalse(got["pose_ok"], got["pose_note"])
        self.assertIn("0.97", got["pose_note"])

    def test_an_unmeasured_pose_is_a_failure_not_a_pass(self):
        got = self.r.smoke_verdict(0.1, None, driving="drive/0001.png")
        self.assertFalse(got["pose_ok"])
        self.assertIn("НЕ ИЗМЕРЕНА", got["pose_note"])
        self.assertIn("drive/0001.png", got["pose_note"])

    def test_a_missing_face_is_not_a_pass_either(self):
        got = self.r.smoke_verdict(None, self._delta(0.02, 0.05))
        self.assertFalse(got["face_ok"])
        self.assertIn("НЕ НАЙДЕНО", got["face_note"])

    def test_the_face_bar_bites_from_both_sides(self):
        from ball_reel.identity_arcface import SAME_PERSON_MAX

        d = self._delta(0.02, 0.05)
        self.assertTrue(self.r.smoke_verdict(SAME_PERSON_MAX - 0.01, d)["face_ok"])
        self.assertFalse(self.r.smoke_verdict(SAME_PERSON_MAX + 0.01, d)["face_ok"])


class TheDefaultEngineIsTheLocalOne(unittest.TestCase):
    """Умолчание — продуктовое решение, а не расположение строки в файле.

    «Внешний API уходит из пути генерации» — требование, а не пожелание.
    Проверяется оно здесь, потому что на карте это проверить некогда, а без
    проверки умолчание живёт ровно до первого рефакторинга аргументов.
    """

    def setUp(self):
        from ball_reel import run_local

        self.r = run_local
        self.base = ["--face", "f.jpg", "--prompt", "p"]

    def test_the_local_engine_is_what_you_get_without_asking(self):
        got = self.r.build_parser().parse_args(self.base)
        self.assertEqual(got.engine, "animatediff")

    def test_the_gateway_path_is_still_reachable(self):
        # Он проверен живьём; потерять его — значит поменять измеренное на
        # предполагаемое накануне демо.
        got = self.r.build_parser().parse_args(self.base + ["--engine", "chain"])
        self.assertEqual(got.engine, "chain")

    def test_an_unknown_engine_is_refused_by_the_parser(self):
        with self.assertRaises(SystemExit):
            self.r.build_parser().parse_args(self.base + ["--engine", "svd"])

    def test_the_two_lora_flags_cannot_be_given_together(self):
        # Замер «с LoRA / без неё» сам решает, что прицепить; заданная рядом
        # --motion-lora сделала бы «базовый» прогон прогоном С LoRA, то есть
        # разница вышла бы нулевой и читалась бы как «LoRA не влияет».
        with self.assertRaises(SystemExit):
            self.r.build_parser().parse_args(
                self.base + ["--motion-lora", "pan-left",
                             "--ab-motion-lora", "pan-left"])

    def test_the_scales_default_to_the_modules_numbers_not_to_copies(self):
        # None означает «не передавать» — число живёт в animate.animate.
        # Скопированное сюда, оно разошлось бы с модулем ровно так же молча,
        # как разошёлся бар по позе.
        got = self.r.build_parser().parse_args(self.base)
        self.assertIsNone(got.ip_adapter_scale)
        self.assertIsNone(got.controlnet_scale)
        self.assertIsNone(got.steps)


class TheWindowIsChosenBeforeAnyWeightsAreLoaded(unittest.TestCase):
    """Окно модуля движения набирается арифметикой, а не загрузкой весов.

    `animate.animate` отказывается, если условий не ровно `cfg.frames`, — но
    отказывается ПОСЛЕ сборки пайплайна, то есть после нескольких гигабайт.
    Тот же отказ здесь стоит миллисекунду. Это ровно тот дефект, что уже
    ловили в `animate.build`, где имя motion LoRA проверялось после загрузки.
    """

    def setUp(self):
        from ball_reel import run_local

        self.r = run_local
        # Вход литеральный: 24 условия, никакой связи с константами модуля.
        self.paths = [f"c/{i:04d}.png" for i in range(24)]

    def test_a_short_sequence_is_refused_with_both_numbers(self):
        got, fps, note = self.r.choose_conditions(self.paths[:9], 16)
        self.assertEqual(got, [])
        self.assertIn("9", note)
        self.assertIn("16", note)
        self.assertEqual(fps, 0.0)

    def test_the_refusal_names_what_to_change(self):
        _, _, note = self.r.choose_conditions(self.paths[:9], 16)
        self.assertIn("--stride", note)
        self.assertIn("render_sequence", note)

    def test_exactly_enough_conditions_are_taken_and_no_more(self):
        got, _, _ = self.r.choose_conditions(self.paths, 16)
        self.assertEqual(len(got), 16)
        self.assertEqual(got[0], "c/0000.png")
        self.assertEqual(got[-1], "c/0015.png")

    def test_a_stride_keeps_the_count_and_drops_the_playback_rate(self):
        # Классическая ошибка среза: paths[0:16:2] даёт ВОСЕМЬ кадров, а окно
        # требует шестнадцать — и узнать об этом можно было бы только после
        # сборки весов.
        got, fps, note = self.r.choose_conditions(
            [f"c/{i:04d}.png" for i in range(40)], 16, stride=2, source_fps=12.0)
        self.assertEqual(len(got), 16)
        self.assertEqual(got[1], "c/0002.png")
        self.assertEqual(fps, 6.0)
        self.assertIn("6.0 fps", note)

    def test_the_note_says_how_much_real_movement_is_covered(self):
        _, _, note = self.r.choose_conditions(self.paths, 16, source_fps=12.0)
        self.assertIn("1.3 c", note)

    def test_an_offset_window_is_measured_against_the_end_of_the_list(self):
        got, _, _ = self.r.choose_conditions(self.paths, 16, start=8)
        self.assertEqual(len(got), 16)
        self.assertEqual(got[0], "c/0008.png")
        refused, _, _ = self.r.choose_conditions(self.paths, 16, start=9)
        self.assertEqual(refused, [])

    def test_nonsense_arguments_are_refused_rather_than_normalised(self):
        for kwargs in ({"stride": 0}, {"stride": -3}, {"start": -1}):
            got, _, note = self.r.choose_conditions(self.paths, 16, **kwargs)
            self.assertEqual(got, [], kwargs)
            self.assertTrue(note)


class TheDrivingFrameIsFoundOrDeclaredMissing(unittest.TestCase):
    """Путь из манифеста относителен НЕ текущего каталога, и это ронял прогон.

    `skeleton.render_sequence` пишет путь в том виде, в каком получил, — то
    есть относительно каталога, откуда рендерили условия. Тест-кит собран
    изнутри `kit/`, и там записано `driving/0000.jpg`. Запущенный из корня
    репозитория прогон падал `FileNotFoundError` ВНУТРИ измерителя позы, то
    есть уже после генерации: на карте это минуты счёта, ушедшие в трейсбек
    мимо отчёта. Найдено сухим прогоном ветки, а не на железе.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        from ball_reel import run_local

        self.r = run_local
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "conditions").mkdir()
        (self.tmp / "driving").mkdir()
        (self.tmp / "driving" / "0000.jpg").write_bytes(b"x")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_path_relative_to_the_kit_root_is_found_next_to_the_conditions(self):
        got = self.r.resolve_driving("driving/0000.jpg",
                                     self.tmp / "conditions")
        self.assertTrue(got and got.endswith("driving/0000.jpg"))

    def test_a_path_that_already_works_is_left_alone(self):
        direct = str(self.tmp / "driving" / "0000.jpg")
        self.assertEqual(self.r.resolve_driving(direct, self.tmp / "conditions"),
                         direct)

    def test_a_frame_that_is_nowhere_is_None_not_a_broken_path(self):
        # None превращается в ПРОПУСК вердикта позы; выдуманный путь превратился
        # бы в исключение посреди гейта.
        self.assertIsNone(self.r.resolve_driving("driving/9999.jpg",
                                                 self.tmp / "conditions"))
        self.assertIsNone(self.r.resolve_driving("", self.tmp / "conditions"))


class WhatIsAttachedIsReadFromTheModel(unittest.TestCase):
    """«LoRA подключена» — заявление демо, и оно проверяется состоянием модели.

    Тот же принцип, по которому гейт не спрашивает генератор, получилось ли у
    него: спрашиваем не аргументы вызова, а `animate.active_loras`.
    """

    def setUp(self):
        from ball_reel import run_local

        self.r = run_local

    def test_a_pipeline_that_cannot_answer_is_a_SKIP_not_a_pass(self):
        ok, note = self.r.lora_verdict([])
        self.assertIsNone(ok)
        self.assertIn("не подтверждено", note)

    def test_a_missing_face_lora_is_a_failure_with_the_consequence_named(self):
        ok, note = self.r.lora_verdict(["motion_pan-left"],
                                       motion_lora="pan-left")
        self.assertFalse(ok)
        self.assertIn("личности", note)

    def test_a_requested_motion_lora_that_did_not_attach_is_caught(self):
        ok, note = self.r.lora_verdict(["faceid_0"], motion_lora="zoom-in")
        self.assertFalse(ok)
        self.assertIn("motion_zoom-in", note)

    def test_a_baseline_run_that_secretly_kept_the_lora_is_caught(self):
        # Половина замера «без LoRA», в которой LoRA осталась прицепленной,
        # даёт разницу ноль и читается как «LoRA ничего не меняет». Это худший
        # исход из возможных: неверный вывод, полученный уверенно.
        ok, note = self.r.lora_verdict(["faceid_0", "motion_tilt-up"])
        self.assertFalse(ok)
        self.assertIn("недействительно", note)

    def test_the_expected_pair_passes_and_is_printed(self):
        ok, note = self.r.lora_verdict(["faceid_0", "motion_pan-right"],
                                       motion_lora="pan-right")
        self.assertTrue(ok)
        self.assertIn("faceid_0", note)

    def test_the_gateway_pipeline_names_its_face_lora_differently(self):
        # На шлюзовом пути адаптер называется `faceid`, на локальном —
        # `faceid_0`: сверка идёт по префиксу, иначе один и тот же вердикт
        # ругался бы на исправный пайплайн.
        ok, _ = self.r.lora_verdict(["faceid", "realism"])
        self.assertTrue(ok)


class TheClipIsJudgedByTheModulesOwnBars(unittest.TestCase):
    """Строка идентичности в гейте была ПЕРЕВЁРНУТА, и это никто не ловил.

    Пройденной она считалась по условию `median is not None`, то есть судилось
    НАЛИЧИЕ ИЗМЕРЕНИЯ, а не его значение: клип с дрейфом 0.9 (другой человек)
    печатался PASS, а клип, где лицо мельче порога судимости, — FAIL. Гейт при
    этом выглядел работающим, потому что строчка печаталась.
    """

    def setUp(self):
        from ball_reel import run_local

        self.r = run_local

    def _good(self, **over):
        got = {
            "drift": {"median": 0.20, "note": "дрейф 0.20"},
            "seam": {"ratio": 0.2, "seamless": True, "note": "loop"},
            "quality": {"worst_jump": 1.4, "smooth": True, "note": "motion"},
            "limbs": {"wobble": {"a->b": 0.05}, "worst": ("a->b", 0.05),
                      "anatomical": True, "note": "limbs"},
            "garment": {"regions": {"torso": 0.01}, "worst": ("torso", 0.01),
                        "stable": True, "note": "одежда"},
        }
        got.update(over)
        return self.r.clip_verdict(**got)

    def _row(self, rows, label):
        return next(r for r in rows if r["label"] == label)

    def test_a_stranger_in_the_clip_no_longer_passes(self):
        from ball_reel.identity_arcface import SAME_PERSON_MAX

        # 0.90 — литерал; проверяем заодно, что бар модуля ниже него, иначе
        # тест сторожил бы сам себя.
        self.assertLess(SAME_PERSON_MAX, 0.90)
        rows = self._good(drift={"median": 0.90, "note": "дрейф 0.90"})
        self.assertFalse(self._row(rows, "идентичность")["ok"])

    def test_a_face_too_small_to_judge_is_a_SKIP_not_a_failure(self):
        rows = self._good(drift={"median": None, "note": "нечего судить"})
        row = self._row(rows, "идентичность")
        self.assertIsNone(row["ok"])
        self.assertIn("не «прошло»", row["note"])

    def test_a_faithful_clip_passes_every_row(self):
        for r in self._good():
            self.assertTrue(r["ok"], r)

    def test_the_unmeasurable_measures_report_themselves_as_such(self):
        rows = self._good(
            limbs={"wobble": {}, "worst": (None, None), "anatomical": False,
                   "note": "limb consistency NOT VERIFIABLE"},
            garment={"regions": {}, "worst": None, "stable": False,
                     "note": "одежда НЕ ПРОВЕРЕНА"},
            seam={"ratio": None, "seamless": False, "note": "need 3 frames"},
            quality={"worst_jump": None, "smooth": False, "note": "static"})
        for label in ("анатомия", "одежда", "луп", "движение"):
            self.assertIsNone(self._row(rows, label)["ok"], label)

    def test_a_torn_clip_still_fails(self):
        # Обратная сторона: вердикт, который никогда не краснеет, не вердикт.
        rows = self._good(
            quality={"worst_jump": 9.0, "smooth": False, "note": "jumps"})
        self.assertFalse(self._row(rows, "движение")["ok"])

    def test_the_named_worst_part_becomes_a_number_for_the_table(self):
        # `worst` приходит кортежем (имя, значение) — в колонку замера должно
        # попасть число, а не «('a->b', 0.05)».
        self.assertEqual(self._row(self._good(), "анатомия")["value"], 0.05)


class TheClipIsAssembledFromFramesLocally(unittest.TestCase):
    """Сборка mp4 — последний и самый хрупкий шаг, и он проверяется тут.

    В `chain._concat` ffmpeg падал ВСЕГДА (относительные пути вместе с cwd), и
    падал после того, как все платные сегменты уже сгенерированы. Здесь тот же
    шаг стоит минут на карте, поэтому он исполняется в тесте по-настоящему.
    """

    def setUp(self):
        import shutil
        import tempfile

        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg не в PATH")
        from ball_reel import run_local

        self.r = run_local
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _frames(self, n=16, size=(64, 96)):
        from PIL import Image

        return [Image.new("RGB", size, (i * 8, 40, 200 - i * 5))
                for i in range(n)]

    def test_frames_land_on_disk_before_anything_can_fail(self):
        from pathlib import Path

        made = self.r.save_frames(self._frames(3), Path(self.tmp) / "f")
        self.assertEqual(len(made), 3)
        self.assertTrue(all(Path(p).exists() for p in made))
        # Имена — четырёхзначные: при двух знаках сотый кадр встал бы между
        # десятым и одиннадцатым и при сборке, и при сортировке.
        self.assertTrue(made[0].endswith("f_0000.png"), made[0])

    def test_the_mp4_comes_back_with_every_frame_in_order(self):
        from pathlib import Path

        from ball_reel import pollinations

        d = Path(self.tmp) / "f"
        self.r.save_frames(self._frames(16), d)
        clip = self.r.frames_to_mp4(d, Path(self.tmp) / "out" / "clip.mp4",
                                    fps=8)
        self.assertTrue(Path(clip).exists())
        self.assertGreater(Path(clip).stat().st_size, 0)
        back = pollinations.extract_frames(clip, Path(self.tmp) / "back", fps=8)
        self.assertEqual(len(back), 16)

    def test_an_empty_directory_fails_loudly_instead_of_writing_nothing(self):
        import subprocess
        from pathlib import Path

        empty = Path(self.tmp) / "none"
        empty.mkdir()
        with self.assertRaises(subprocess.CalledProcessError):
            self.r.frames_to_mp4(empty, Path(self.tmp) / "x.mp4", fps=8)


class TheABTableIsReadableOnItsOwn(unittest.TestCase):
    """Две половины замера печатает ОДНА функция из одних и тех же полей.

    Два отчёта, разнесённые по времени и каталогам, сравнивают глазами, а глаз
    сравнивает то, что помнит: именно так бар по позе долго стоял мягче
    собственного порога.
    """

    def setUp(self):
        from ball_reel import run_local

        self.r = run_local
        self.runs = [
            {"label": "без motion LoRA", "loras": ["faceid_0"],
             "rows": [{"label": "движение", "ok": True, "value": 1.4,
                       "note": "n"},
                      {"label": "идентичность", "ok": None, "value": None,
                       "note": "n"}]},
            {"label": "с motion LoRA pan-left",
             "loras": ["faceid_0", "motion_pan-left"],
             "rows": [{"label": "движение", "ok": False, "value": 9.1,
                       "note": "n"}]},
        ]

    def test_both_columns_and_both_lora_states_are_printed(self):
        text = self.r.ab_report(self.runs)
        self.assertIn("без motion LoRA", text)
        self.assertIn("motion_pan-left", text)
        self.assertIn("1.400", text)
        self.assertIn("9.100", text)

    def test_an_unmeasured_number_is_a_word_not_an_empty_cell(self):
        # Пустая клетка читается как ноль, то есть как лучший возможный
        # результат — ровно та подмена, против которой написан `_say`.
        text = self.r.ab_report(self.runs)
        self.assertIn("не измерено", text)
        self.assertIn("ПРОПУСК", text)

    def test_a_row_present_in_one_half_only_is_not_silently_dropped(self):
        text = self.r.ab_report(self.runs)
        self.assertIn("нет строки", text)

    def test_the_verdicts_travel_with_the_numbers(self):
        text = self.r.ab_report(self.runs)
        self.assertIn("PASS", text)
        self.assertIn("FAIL", text)


if __name__ == "__main__":
    unittest.main()
