"""Окно исходника выбирается ЗАМЕРОМ, а не умолчанием 0.

ЗАЧЕМ. Модуль движения берёт ровно 16 кадров, а исходник длиннее — 192. Какой
отрезок взять, до сих пор решало умолчание `--from-frame 0`, и цена этого
умолчания измерена на настоящих кадрах:

    текущее окно  8..23    стык 4.374
    лучшее окно  66..81    стык 1.245     (в 3.5 раза меньше)
    худшее окно             6.568

Разница бесплатна: та же генерация, другой отрезок. Бар 0.30 не берёт ни одно
окно — движение в этом видео нециклично, — но выбирать худшее, имея замер,
незачем.

ПОЧЕМУ ВЫБОР ОКНА, А НЕ ОБРЕЗКА. Обрезка (`best_loop_cut`) работает после
генерации и умеет только отбросить хвост, то есть ограничена содержимым уже
выбранного окна. Измерено, что этого не хватает: внутри окна 8..23 лучшая
обрезка даёт 3.742 против исходных 4.374 — движение за эти 16 кадров никуда не
возвращается, и резать нечего. Порядок поэтому: выбрать окно, сгенерировать,
при нужде подрезать.
"""

from __future__ import annotations

import glob
import inspect
import shutil
import tempfile
import unittest
from pathlib import Path

from ball_reel import motion, run_local


def _ramp(path, *, value: int, size: int = 64):
    """Однотонный кадр заданной яркости: расстояние между кадрами предсказуемо."""
    import numpy as np
    from PIL import Image

    a = np.full((size, size, 3), value, dtype="uint8")
    Image.fromarray(a).save(path)
    return str(path)


class Choosing(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_it_finds_the_window_that_returns_to_its_start(self):
        """Одно окно замкнуто, остальные нет — оно и обязано быть выбрано."""
        # Пила: 0..40 растёт, потом возврат. Окно, начинающееся на возврате,
        # заканчивается там же, откуда началось.
        vals = [0, 10, 20, 30, 40, 50, 60, 70, 60, 50, 40, 30, 20, 10, 0, 10,
                20, 30]
        frames = [_ramp(self.tmp / f"{i:03d}.png", value=v)
                  for i, v in enumerate(vals)]
        got = motion.best_loop_window(frames, size=15)
        self.assertIsNotNone(got["ratio"])
        # Кадр 0 и кадр 14 оба нулевые — это и есть замкнутое окно.
        self.assertEqual(got["start"], 0)
        self.assertTrue(got["seamless"], got["note"])

    def test_a_worse_window_is_reported_so_the_gain_is_visible(self):
        # Без «худшего» число «лучшего» не с чем сравнить, и выбор выглядит
        # бесплатным улучшением там, где выбирать было не из чего.
        vals = [0, 10, 20, 30, 40, 50, 60, 70, 60, 50, 40, 30, 20, 10, 0, 10]
        frames = [_ramp(self.tmp / f"{i:03d}.png", value=v)
                  for i, v in enumerate(vals)]
        got = motion.best_loop_window(frames, size=8)
        self.assertGreater(got["worst_ratio"], got["ratio"])
        self.assertGreater(got["candidates"], 1)

    def test_too_short_a_source_is_a_refusal_and_not_window_zero(self):
        frames = [_ramp(self.tmp / f"{i}.png", value=i * 10) for i in range(5)]
        got = motion.best_loop_window(frames, size=16)
        self.assertIsNone(got["ratio"])
        self.assertIn("выбирать не из чего", got["note"])

    def test_stride_widens_the_span_it_needs(self):
        # При шаге 2 окно из 8 кадров покрывает 15 кадров исходника, а не 8.
        frames = [_ramp(self.tmp / f"{i:03d}.png", value=i * 5)
                  for i in range(10)]
        self.assertIsNone(motion.best_loop_window(frames, size=8, stride=2)["ratio"])
        self.assertIsNotNone(motion.best_loop_window(frames, size=8)["ratio"])

    def test_the_note_admits_when_the_bar_is_not_reached(self):
        """Улучшение и починка — разные вещи, и путать их нельзя.

        На нашем драйвинге лучшее окно даёт 1.245 при баре 0.30: это в 3.5 раза
        лучше умолчания и всё равно провал оси. Примечание, умолчавшее об этом,
        подало бы выбор окна как решение проблемы.
        """
        vals = [i * 7 for i in range(20)]        # монотонный рост, возврата нет
        frames = [_ramp(self.tmp / f"{i:03d}.png", value=min(v, 255))
                  for i, v in enumerate(vals)]
        got = motion.best_loop_window(frames, size=8)
        self.assertFalse(got["seamless"])
        self.assertIn("бар не взят", got["note"])


class MeasuredOnTheRealDriving(unittest.TestCase):
    """Числа из шапки — воспроизводимые, а не запомненные.

    ДАННЫЕ ИЗ ИНДЕКСА, А НЕ С ДИСКА АВТОРА. Раньше здесь стоял `ref_frames` —
    каталог, которого в git НЕТ ни одного файла. У автора он лежал локально,
    тест был зелёным, а у склонировавшего молча пропускался: в клоне пропусков
    оказалось 22 против 7 в рабочем дереве, то есть пятнадцать сторожей спали.
    Пропущенный тест не сторожит ничего, и заметить это можно только по числу.

    Первая попытка починки — перевести тест на `demo/kit/driving` — БЫЛА
    НЕВЕРНОЙ, и это видно числом: там лежат 16 кадров, которые сами есть
    РЕЗУЛЬТАТ выбора окна. Выбирать окно внутри уже выбранного окна незачем, и
    тест краснел честно (2.817 против требуемых < 1.857). Мерить надо на том,
    на чём мера имеет смысл, — на длинном исходнике.

    Поэтому в индекс положен стенд `demo/bench/driving` — те же 96 кадров,
    уменьшенные до 240 px по ширине (1.46 МБ против 67 МБ полного размера).
    Пиксельная мера — отношение, и уменьшение его почти не трогает:

        полный размер 720x1278   окно 66  стык 1.245  худшее 6.568  выигрыш 5.28x
        стенд         240x426    окно 66  стык 1.227  худшее 6.394  выигрыш 5.21x

    Выбрано ТО ЖЕ окно, числа разошлись на 1.5%. Пересчитать:

        python3 -c "import glob;from ball_reel import motion;\
g=motion.best_loop_window(sorted(glob.glob('demo/bench/driving/*.jpg')),size=16);\
print(g['start'],g['ratio'],g['worst_ratio'])"
    """

    POOL = sorted(glob.glob(str(Path(__file__).resolve().parents[2]
                                / "demo" / "bench" / "driving" / "*.jpg")))

    def test_choosing_the_window_beats_the_default_by_a_lot(self):
        if len(self.POOL) < 96:
            self.skipTest("стенда demo/bench/driving нет в дереве")
        got = motion.best_loop_window(self.POOL, size=16)
        self.assertIsNotNone(got["ratio"])
        self.assertLess(got["ratio"], got["worst_ratio"] / 2,
                        "выбор окна перестал что-либо давать — проверить "
                        "числа в шапке, а не смягчать тест")

    def test_the_bench_picks_the_window_the_header_names(self):
        """Не «лучше худшего», а РОВНО то окно, о котором написано.

        Без этого предыдущий тест зеленел бы на любом окне, лишь бы выигрыш был
        двукратным, и шапка могла разойтись с кодом незаметно.
        """
        if len(self.POOL) < 96:
            self.skipTest("стенда demo/bench/driving нет в дереве")
        got = motion.best_loop_window(self.POOL, size=16)
        self.assertEqual(got["start"], 66, got["note"])
        self.assertAlmostEqual(got["ratio"], 1.227, places=2)
        self.assertAlmostEqual(got["worst_ratio"], 6.394, places=2)


class ItIsWiredIntoTheRun(unittest.TestCase):

    def test_the_flag_accepts_auto(self):
        args = run_local.build_parser().parse_args(
            ["--face", "f.jpg", "--prompt", "p", "--from-frame", "auto"])
        self.assertEqual(str(args.from_frame).lower(), "auto")

    def test_the_flag_still_accepts_a_number(self):
        args = run_local.build_parser().parse_args(
            ["--face", "f.jpg", "--prompt", "p", "--from-frame", "12"])
        self.assertEqual(str(args.from_frame), "12")

    def test_the_run_actually_calls_the_chooser(self):
        src = inspect.getsource(run_local._run_animatediff)
        self.assertIn("best_loop_window", src,
                      "`auto` объявлен и ничего не выбирает")

    def test_it_measures_on_driving_frames_and_not_on_the_skeletons(self):
        """Условия — это отрисовка скелета.

        Похожесть двух таких картинок меряется по линиям, а не по телу, и
        стык, снятый с них, говорит о рендере, а не о движении.
        """
        src = inspect.getsource(run_local._run_animatediff)
        chooser = src[src.index("best_loop_window"):]
        self.assertIn("driving_of", src[:src.index("best_loop_window")] + chooser[:400])
