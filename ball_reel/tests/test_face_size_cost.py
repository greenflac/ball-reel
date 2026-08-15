"""Сколько сходства съедает САМ РАЗМЕР лица. Ответ: почти ничего.

ЗАЧЕМ ЭТОТ ФАЙЛ. Весь вечер числа сходства читались с оговоркой «ну лицо же
мелкое». Оговорка звучала разумно, стояла в обосновании порога `MIN_FACE_PX`
(«мелкая детекция — растянутые пиксели, дистанции раздуваются») и НИ РАЗУ не
проверялась. Пока она в силе, любое плохое число можно списать на размер — то
есть прибор перестаёт что-либо доказывать.

Проверяется прямо: одна и та же фотография уменьшается так, чтобы лицо стало
N px, возвращается к исходному размеру и сравнивается САМА С СОБОЙ. Личность
одинакова по построению, значит всё, что покажет прибор, — цена потери деталей.

    лицо px   200    160    140    127    110     95     83     70
    ArcFace 0.002  0.002  0.003  0.004  0.005  0.005  0.007  0.007

Ноль целых семь тысячных при баре 0.35. Оговорка опровергнута: 0.55 на мелком
лице — НАСТОЯЩЕЕ расстояние.

Что при этом ОСТАЁТСЯ правдой и не проверено здесь: у генератора при 85 px
меньше места, куда положить черты. Это ограничение РИСОВАНИЯ, а не измерения, и
лечится разрешением кадра, а не доверием к числу.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "demo" / "kit"


class TheJudgeReadsSmallFacesFine(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ref = KIT / "face.jpg"
        if not self.ref.exists():
            self.skipTest("нет demo/kit/face.jpg")

    def _at(self, target_px: int) -> float | None:
        """Та же фотография с лицом в `target_px` -> дистанция до оригинала."""
        from PIL import Image

        from ball_reel.identity import arcface_drift
        from ball_reel.identity_arcface import face_detail

        d0 = face_detail(str(self.ref))
        if not d0:
            self.skipTest("детектор не нашёл лица на референсе")
        im = Image.open(self.ref).convert("RGB")
        w, h = im.size
        k = target_px / d0["face_px"]
        small = im.resize((max(8, int(w * k)), max(8, int(h * k))),
                          Image.BICUBIC)
        p = self.tmp / f"f{target_px}.png"
        small.resize((w, h), Image.BICUBIC).save(p)
        return arcface_drift([str(p)], str(self.ref),
                             min_face_px=1).get("median")

    def test_a_seventy_pixel_face_is_still_recognised_as_itself(self):
        """Самый мелкий судимый размер — и он почти бесплатен."""
        got = self._at(70)
        self.assertIsNotNone(got, "судья не нашёл лица — замер не состоялся")
        self.assertLess(got, 0.05,
                        f"70 px стоят {got:.3f}: если это перестало быть "
                        f"почти нулём, обоснование MIN_FACE_PX надо "
                        f"пересчитать, а не подправить тест")

    def test_the_cost_of_size_is_far_below_the_bar(self):
        from ball_reel.identity_arcface import SAME_PERSON_MAX

        got = self._at(83)
        self.assertIsNotNone(got)
        self.assertLess(got * 10, SAME_PERSON_MAX,
                        "цена размера подобралась к бару — тогда числа на "
                        "мелких лицах снова нельзя читать буквально")

    def test_shrinking_does_not_help_which_would_mean_the_test_is_broken(self):
        """Сторож самого замера.

        Если уменьшение вдруг УЛУЧШАЕТ сходство, значит сравнивается не то:
        например, обе стороны прошли через одно и то же преобразование.
        """
        big, small = self._at(200), self._at(70)
        self.assertIsNotNone(big)
        self.assertIsNotNone(small)
        self.assertLessEqual(big, small + 1e-6,
                             "мелкое лицо ближе к оригиналу, чем крупное — "
                             "замер сравнивает не то, что думает")

    def test_the_measurement_is_recorded_beside_the_threshold(self):
        # Число, живущее только в тесте, не найдёт тот, кто читает порог.
        import inspect

        from ball_reel import identity_arcface

        src = inspect.getsource(identity_arcface)
        head = src[:src.index("MIN_FACE_PX = 100")]
        self.assertIn("0.007", head[-1400:],
                      "замер не записан рядом с порогом — обоснование порога "
                      "снова станет догадкой при первом же чтении")
