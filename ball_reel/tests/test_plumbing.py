"""Стыки, на которых прогон разваливается тихо.

Ни одна из этих проверок не про качество картинки. Все они про сантехнику:
пути, имена файлов, порядок кадров, отказ на неверном аргументе. Такие дефекты
не ловятся глазами по результату — они либо роняют прогон в конце (после того,
как всё уже оплачено), либо не роняют вовсе и портят вердикт.

Все проверенные здесь дефекты были настоящими и найдены адверсарным ревью кода,
который ни разу не исполнялся: сборка на GPU и склейка сегментов писались под
машину, которой в среде разработки нет. Отсюда и правило — то, что нельзя
исполнить, надо хотя бы обстрелять по стыкам.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


class FramesComeBackInPlaybackOrder(unittest.TestCase):
    """Кадры собираются сортировкой ИМЁН — ширины поля должно хватать."""

    def setUp(self):
        from ball_reel import pollinations

        self.p = pollinations

    def test_a_long_clip_still_sorts_right(self):
        # 192 кадра — это штатный прогон из GPU_RUNBOOK: 16 узлов по 2 с при
        # fps=6. Именно на нём двузначный шаблон и ломался.
        self.assertTrue(self.p.frame_names_sort_correctly(192))
        self.assertTrue(self.p.frame_names_sort_correctly(9999))

    def test_the_two_digit_pattern_is_what_broke(self):
        # Тест «на дефект»: он фиксирует, ПОЧЕМУ шаблон именно такой, и
        # покраснеет, если кто-нибудь решит вернуть короткий.
        self.assertTrue(self.p.frame_names_sort_correctly(99, "%02d.png"))
        self.assertFalse(self.p.frame_names_sort_correctly(100, "%02d.png"))

    def test_the_shipped_pattern_is_wide_enough_for_a_long_clip(self):
        self.assertTrue(
            self.p.frame_names_sort_correctly(192, self.p.FRAME_PATTERN))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not on PATH")
class SegmentsJoinFromARelativePath(unittest.TestCase):
    """Склейка обязана работать из относительного каталога.

    Это ровно тот путь, которым идут и рунбук, и прогон (`--out run_out`).
    Раньше _concat писал в segments.txt имена относительно cwd, а сам файл
    списка передавал ffmpeg относительным путём ВМЕСТЕ с cwd=каталог склейки —
    то есть ffmpeg искал run_out/chain/segments.txt внутри run_out/chain.
    Падало всегда и падало ПОСЛЕ того, как все платные сегменты уже оплачены.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = os.getcwd()
        self.addCleanup(os.chdir, self.cwd)
        os.chdir(self.tmp.name)

    def _segment(self, path: str, colour: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"color=c={colour}:s=64x64:d=0.3:r=10",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
            check=True, capture_output=True)
        return path

    def test_relative_out_dir_joins_instead_of_raising(self):
        from ball_reel.chain import _concat

        parts = [self._segment("run_out/chain/seg_0.mp4", "red"),
                 self._segment("run_out/chain/seg_1.mp4", "blue")]
        joined = _concat(parts, "run_out/chain/chain.mp4")
        self.assertTrue(Path(joined).exists())
        self.assertGreater(Path(joined).stat().st_size, 0)

    def test_the_listing_survives_a_change_of_working_directory(self):
        # Путь внутри списка должен быть абсолютным: список читается ffmpeg'ом,
        # у которого своё представление о текущем каталоге.
        from ball_reel.chain import _concat

        parts = [self._segment("run_out/chain/seg_0.mp4", "red"),
                 self._segment("run_out/chain/seg_1.mp4", "blue")]
        _concat(parts, "run_out/chain/chain.mp4")
        listing = Path("run_out/chain/segments.txt").read_text()
        for line in listing.strip().splitlines():
            self.assertTrue(line.startswith("file '/"), line)


class PoseDeltaRefusesAnEmptyPose(unittest.TestCase):
    """None от landmarks() должен объяснять себя, а не падать из глубины."""

    def test_a_missing_pose_names_the_usual_cause(self):
        from ball_reel.pose import pose_delta

        with self.assertRaises(ValueError) as ctx:
            pose_delta(None, {"nose": (0.5, 0.5, 1.0)})
        self.assertIn("УСЛОВИЕМ", str(ctx.exception))

    def test_both_sides_are_reported(self):
        from ball_reel.pose import pose_delta

        with self.assertRaises(ValueError) as ctx:
            pose_delta(None, None)
        self.assertIn("первой", str(ctx.exception))
        self.assertIn("второй", str(ctx.exception))


class TheModelTableIsOne(unittest.TestCase):
    """Список моделей с end_frame жил в трёх местах и разошёлся."""

    def test_end_frame_models_are_derived_from_the_price_table(self):
        from ball_reel.chain import END_FRAME_MODELS, END_FRAME_POLLEN

        self.assertEqual(set(END_FRAME_MODELS), set(END_FRAME_POLLEN))

    def test_the_table_matches_what_the_api_reported(self):
        # Снято с GET /video/models 2026-08-12. `wan` в этот список НЕ входит
        # (max_reference_images=1), хотя докстринг video_loop его называл.
        from ball_reel.chain import END_FRAME_POLLEN

        self.assertEqual(set(END_FRAME_POLLEN),
                         {"wan-fast", "veo", "wan-pro", "seedance-2.0"})
        self.assertNotIn("wan", END_FRAME_POLLEN)
        # wan-fast дешевле seedance-2.0 в 18 раз — на этом стоит совет
        # «сначала связка, потом качество», и цифра не должна уехать молча.
        self.assertAlmostEqual(END_FRAME_POLLEN["wan-fast"], 0.01)
        self.assertAlmostEqual(END_FRAME_POLLEN["seedance-2.0"], 0.18)


class TheRunRefusesBadArgumentsBeforeSpendingAnything(unittest.TestCase):
    """Отказ обязан случиться до предполёта, а не на пятом шаге."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _run(self, *extra) -> int:
        from ball_reel.run_local import main

        return main(["--face", "nope.jpg", "--prompt", "x",
                     "--out", str(Path(self.tmp.name) / "out"), *extra])

    def test_a_model_without_end_frame_is_refused_up_front(self):
        # `wan` существует и звучит правдоподобно — именно поэтому опечатка
        # такого рода и доживала до конца прогона.
        self.assertEqual(self._run("--video-model", "wan"), 1)

    def test_garment_ref_says_it_is_not_wired_instead_of_pretending(self):
        self.assertEqual(self._run("--garment-ref", "shirt.jpg"), 1)

    def test_a_duration_the_model_will_reject_is_caught_before_the_gpu(self):
        # Дефолт прогона — 2 с, и на wan-fast он верен. На seedance-2.0 шлюз
        # ответил бы 400 — на последнем, платном шаге, когда все кейфреймы уже
        # отрисованы. Это ровно тот провал, который обязан случиться раньше.
        self.assertEqual(self._run("--video-model", "seedance-2.0",
                                   "--seconds", "2"), 1)

    def test_veo_takes_three_values_not_a_range(self):
        self.assertEqual(self._run("--video-model", "veo", "--seconds", "5"), 1)


class SegmentLengthIsCheckedAgainstTheModel(unittest.TestCase):
    """Длительность валидируется шлюзом ПО МОДЕЛИ, и 400 приходит последним."""

    def setUp(self):
        from ball_reel.chain import segment_seconds_ok

        self.check = segment_seconds_ok

    def test_the_cheap_default_pair_is_allowed(self):
        ok, note = self.check("wan-fast", 2)
        self.assertTrue(ok, note)

    def test_seedance_refuses_two_seconds(self):
        ok, note = self.check("seedance-2.0", 2)
        self.assertFalse(ok)
        self.assertIn("4", note)

    def test_veo_refuses_five_and_names_the_nearest_legal_value(self):
        ok, note = self.check("veo", 5)
        self.assertFalse(ok)
        self.assertIn("4/6/8", note)

    def test_an_unknown_model_is_allowed_but_flagged(self):
        # Молча пропускать неизвестное нельзя, но и блокировать нечем: таблица
        # знает только то, что измерено.
        ok, note = self.check("something-new", 3)
        self.assertTrue(ok)
        self.assertIn("не проверялся", note)


class ConditionsExplainThemselves(unittest.TestCase):
    """Манифест должен лежать рядом с условиями, а не только возвращаться."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "conditions"

    @staticmethod
    def _fake_points(_path):
        pts = {n: (0.5, 0.5, 0.9) for n in
               ("nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder",
                "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
                "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle")}
        pts["neck"] = (0.5, 0.4, 0.9)
        pts["hip_c"] = (0.5, 0.6, 0.9)
        pts["__size__"] = (720.0, 1280.0, 1.0)
        return pts

    def test_the_manifest_maps_each_condition_to_its_driving_frame(self):
        import json

        from ball_reel.skeleton import render_sequence

        frames = ["drive/a.png", "drive/b.png", "drive/c.png"]
        render_sequence(frames, self.out,
                        source=self._fake_points)
        saved = json.loads((self.out / "manifest.json").read_text())
        # Ключ — основа имени условия, значение — исходный кадр. Без этой карты
        # сгенерированный кейфрейм не с чем сверять по позе: условие — рисунок,
        # на нём детектор поз ничего не находит.
        self.assertEqual(saved["driving_frames"],
                         {"0000": "drive/a.png", "0001": "drive/b.png",
                          "0002": "drive/c.png"})

    def test_frames_without_a_pose_leave_no_entry_and_no_condition(self):
        import json

        from ball_reel.skeleton import render_sequence

        def sometimes(path):
            return None if path.endswith("b.png") else self._fake_points(path)

        render_sequence(["drive/a.png", "drive/b.png", "drive/c.png"],
                        self.out, source=sometimes)
        saved = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(sorted(saved["driving_frames"]), ["0000", "0002"])
        self.assertEqual(saved["missing_frames"], [1])


if __name__ == "__main__":
    unittest.main()
