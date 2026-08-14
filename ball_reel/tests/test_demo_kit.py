"""Нарезка кита для git.

Модуль маленький, но ловушка в нём дорогая: нарезка, отрендеренная автономно,
выглядит как полный кит и переносит вдвое меньше тела. Тесты сторожат именно
это, а не факт копирования файлов.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np  # noqa: F401
    from PIL import Image
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _points(**over):
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
        "neck": (0.50, 0.25, 1.0), "hip_c": (0.50, 0.55, 1.0),
        "__size__": (1000.0, 1000.0, 1.0),
    }
    pts.update(over)
    return pts


@unittest.skipUnless(HAVE_DEPS, "numpy/Pillow not installed (live extra)")
class TheCutCarriesTheWHOLEBodyMeasurement(unittest.TestCase):
    """Телосложение донора мерится по всей записи, а не по нарезке.

    ИЗМЕРЕНО на живом ките: донор по 16 кадрам окна теряет ТРИ кости из шести
    (предплечья и одно плечо — они видны меньше чем на `MIN_DRIVING_FRAMES`
    кадрах окна) и уводит бедро с x0.701 на x0.630. Кит при этом выглядит
    полным: те же 16 png, тот же манифест, `retargeted: true`.
    """

    def setUp(self):
        from ball_reel import demo_kit

        self.d = demo_kit
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _fake_kit(self, n=40):
        src = self.dir / "kit"
        (src / "driving").mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (64, 96), (i, i, i)).save(
                src / "driving" / f"{i:04d}.jpg")
        Image.new("RGB", (64, 96), (9, 9, 9)).save(src / "face.jpg")
        return src

    def _patched(self, donor_seen):
        """Подменяет замеры так, чтобы считать, СКОЛЬКО кадров ушло в донора."""
        from ball_reel import pose, skeleton

        real_prop = pose.world_proportions
        real_drv = skeleton.driving_proportions
        real_pts = skeleton.pose_points
        pose.world_proportions = lambda _p: {"l_hip->l_knee": 0.6}

        def spy(frames, **kw):
            donor_seen.append(len(frames))
            return {k: 0.5 for k in skeleton.BONE_TO_PROPORTION.values()}, len(frames)

        skeleton.driving_proportions = spy
        skeleton.pose_points = lambda _p: _points()
        self.addCleanup(setattr, pose, "world_proportions", real_prop)
        self.addCleanup(setattr, skeleton, "driving_proportions", real_drv)
        self.addCleanup(setattr, skeleton, "pose_points", real_pts)
        # Экстрактор задаётся явно: с умолчанием тест зависел бы от того, лежат
        # ли на машине веса DWPose, и «нарезка пустая» подменялось бы «нет
        # весов».
        return lambda _p: _points()

    def test_the_donor_is_measured_on_every_frame_not_on_the_window(self):
        seen = []
        pts = self._patched(seen)
        src = self._fake_kit(40)
        self.d.build(src, self.dir / "cut", start=10, frames=8, source=pts)
        # Донору отдали ВСЕ 40 кадров, хотя условий рендерится 8.
        self.assertEqual(seen, [40])

    def test_the_manifest_records_how_many_frames_the_donor_came_from(self):
        seen = []
        pts = self._patched(seen)
        src = self._fake_kit(40)
        m = self.d.build(src, self.dir / "cut2", start=10, frames=8,
                         source=pts)
        self.assertEqual(m["donor_frames_measured"], 40)
        self.assertEqual(len(m["conditions"]), 8)

    def test_a_window_past_the_end_is_refused_before_anything_is_written(self):
        seen = []
        pts = self._patched(seen)
        src = self._fake_kit(20)
        with self.assertRaises(ValueError) as e:
            self.d.build(src, self.dir / "cut3", start=15, frames=8, source=pts)
        self.assertIn("не влезает", str(e.exception))
        self.assertFalse((self.dir / "cut3" / "conditions").exists())

    def test_a_missing_source_kit_is_named_not_silently_empty(self):
        with self.assertRaises(FileNotFoundError):
            self.d.build(self.dir / "нет-такого", self.dir / "cut4")

    def test_where_the_window_came_from_is_written_beside_it(self):
        seen = []
        pts = self._patched(seen)
        src = self._fake_kit(40)
        m = self.d.build(src, self.dir / "cut5", start=10, frames=8,
                         source=pts)
        self.assertEqual(m["cut_from"]["start"], 10)
        self.assertEqual(m["cut_from"]["of_total"], 40)
        self.assertTrue(m["cut_from"]["why"])
        on_disk = json.loads(
            (self.dir / "cut5" / "conditions" / "manifest.json").read_text())
        self.assertEqual(on_disk["cut_from"]["start"], 10)

    def test_the_window_length_is_taken_from_the_motion_module_not_copied(self):
        # Второй способ узнать длину окна — дефект: модуль движения обучен на
        # своём числе, и разойтись эти два числа могут только молча.
        from ball_reel.animate import CONTEXT_FRAMES

        self.assertEqual(self.d.window_frames(), CONTEXT_FRAMES)

    def test_the_chosen_start_is_the_clean_one_not_the_most_mobile(self):
        """Размен записан числом, а не вкусом.

        Старт 55 подвижнее (101.4 против 84.3), но все пять кадров кита с
        потерянной конечностью лежат в нём. Потерянный сустав — разрешение
        генератору сочинить руку, и 17% подвижности за это не жалко.
        """
        self.assertEqual(self.d.WINDOW_START, 50)
        src = __import__("inspect").getsource(self.d)
        self.assertIn("0.979", src, "покрытие суставов выбранного окна")
        self.assertIn("101.4", src, "подвижность отвергнутого окна")


class TheFramingIsAChoiceTheCutMustCarry(unittest.TestCase):
    """Кадрировка решает, БУДЕТ ЛИ у оси личности число вообще.

    ИЗМЕРЕНО на живом ките (512x768, окно 50..65):

        full_body   лицо  73.8 px   identity_judgeable: False
        waist_up    лицо 140.4 px   identity_judgeable: True

    Бар ArcFace для кадра видео — 100 px. Полный рост выбран владельцем
    продукта, и это остаётся так: движение тела важнее удобства измерения. Но
    тогда число по личности не добывается НИКАК, и второй кит — не замена
    первому, а единственный способ его получить.

    Пока параметр не доходил до рендера, `--framing waist_up` молча собирал бы
    полный рост: те же 16 png, тот же манифест, `framing: full_body` — и
    оператор, заказавший судимую кадрировку, получил бы несудимую.
    """

    def setUp(self):
        from ball_reel import demo_kit

        self.d = demo_kit
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_the_framing_reaches_the_renderer(self):
        from ball_reel import pose, skeleton

        seen = {}
        real_prop, real_drv, real_render = (pose.world_proportions,
                                            skeleton.driving_proportions,
                                            skeleton.render_sequence)
        pose.world_proportions = lambda _p: {"l_hip->l_knee": 0.6}
        skeleton.driving_proportions = lambda frames, **kw: ({}, len(frames))

        def spy(frames, out_dir, **kw):
            seen.update(kw)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return {"conditions": [], "framing": kw.get("framing"),
                    "warnings": [], "face_px": 0, "identity_judgeable": False,
                    "retarget_factors": {}, "retarget_origin": {},
                    "donor_frames_measured": 0}

        skeleton.render_sequence = spy
        self.addCleanup(setattr, pose, "world_proportions", real_prop)
        self.addCleanup(setattr, skeleton, "driving_proportions", real_drv)
        self.addCleanup(setattr, skeleton, "render_sequence", real_render)

        src = self.dir / "kit"
        (src / "driving").mkdir(parents=True)
        for i in range(40):
            Image.new("RGB", (64, 96), (i, i, i)).save(
                src / "driving" / f"{i:04d}.jpg")
        Image.new("RGB", (64, 96), (9, 9, 9)).save(src / "face.jpg")

        self.d.build(src, self.dir / "cut", start=0, frames=8,
                     framing="waist_up")
        self.assertEqual(seen.get("framing"), "waist_up",
                         "кадрировка не доехала до рендера условий")

    def test_default_stays_the_product_choice(self):
        import inspect

        sig = inspect.signature(self.d.build)
        self.assertEqual(sig.parameters["framing"].default, "full_body")

    def test_the_cli_offers_both_and_explains_why_there_are_two(self):
        import inspect

        src = inspect.getsource(self.d.main)
        self.assertIn("waist_up", src)
        self.assertIn("ВТОРОЙ прогон", src)
        self.assertIn("движение тела важнее", src)


class BothKitsAreWHOLEInGit(unittest.TestCase):
    """Полкита в репозитории выглядит как кит и не работает как кит.

    ИЗМЕРЕНО и допущено: `demo/kit_waist` попал в git ОДНИМИ driving-кадрами.
    Условия и фото отсеклись `.gitignore` — там с давних пор стоят `conditions/`
    и `face.jpg`, а первый кит когда-то добавили через `git add -f`, и про это
    забыли. Каталог при этом выглядит полным: имя на месте, файлы внутри есть.

    Проверять надо ИМЕННО индекс git, а не диск: на машине автора всё лежит, и
    отсутствие замечается только у того, кто склонировал. То есть у проверяющего.
    """

    def setUp(self):
        import subprocess

        self.tracked = set(subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2]).stdout.split())

    def _kit(self, name: str):
        return sorted(p for p in self.tracked if p.startswith(f"demo/{name}/"))

    def test_both_kits_carry_conditions_driving_face_and_manifest(self):
        for name in ("kit", "kit_waist"):
            with self.subTest(kit=name):
                files = self._kit(name)
                if not files:
                    self.skipTest(f"demo/{name} нет в этом дереве")
                for part in ("conditions/", "driving/", "face.jpg",
                             "manifest.json"):
                    self.assertTrue(any(part in f for f in files),
                                    f"demo/{name}: в git нет {part} — "
                                    f"каталог выглядит китом и им не является")

    def test_the_two_kits_have_the_same_shape(self):
        a, b = self._kit("kit"), self._kit("kit_waist")
        if not a or not b:
            self.skipTest("оба кита нужны для сравнения")
        self.assertEqual(len(a), len(b),
                         "киты разной полноты: один из них обрезан .gitignore")
