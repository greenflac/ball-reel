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
