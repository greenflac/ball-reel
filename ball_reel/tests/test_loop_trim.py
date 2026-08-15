"""Луп замыкается ОБРЕЗКОЙ, и обрезка обязана происходить, а не советоваться.

ЗАЧЕМ ЭТОТ ФАЙЛ. `motion.best_loop_cut` существовал, вызывался шлюзовым путём
(`produce`) — и на локальном пути только ПЕЧАТАЛСЯ СОВЕТОМ. Отчёт содержал
адрес починки, починка не происходила, и ось `луп` валилась на каждом прогоне
при готовом лечении ценой в секунды ffmpeg против 31 минуты второй генерации.

Это шестой случай формы «написано и не подключено» в этом репозитории, и
единственный, где неподключённым оказался не модуль, а ВЫЗОВ из одного из двух
путей. Мутационный аудит здесь бессилен по той же причине, что и в
`test_reachable`: константы `motion` убиваются его собственными тестами
независимо от того, зовёт ли их `run_local`.
"""

from __future__ import annotations

import inspect
import unittest

from ball_reel import motion, run_local


class TheLocalPathActuallyTrims(unittest.TestCase):

    def test_the_cut_is_computed_and_not_merely_recommended(self):
        src = inspect.getsource(run_local._animatediff_once)
        self.assertIn("best_loop_cut(paths)", src,
                      "локальный путь снова только советует резать: лечение "
                      "есть, а вызова нет")

    def test_the_advice_string_is_gone(self):
        """Совет и вызов вместе — это два источника правды об одном действии.

        Пока печатается «резать по лучшему стыку», читатель отчёта не может
        понять, порезали уже или ему предлагают порезать самому.
        """
        src = inspect.getsource(run_local._animatediff_once)
        self.assertNotIn("Резать по лучшему стыку", src)

    def test_the_judged_clip_is_not_overwritten(self):
        """Обрезанный ролик ложится РЯДОМ.

        Обрезка меняет длину, а гейт судил полный клип. Перезаписав его, мы
        отдали бы на приёмку не то, что измеряли, — и это тот же запрет, по
        которому апскейл и подгонка фактуры не трогают судимые кадры.
        """
        src = inspect.getsource(run_local._animatediff_once)
        self.assertIn('out / "clip_loop.mp4"', src)
        self.assertNotIn('trim_to_loop(clip, paths, clip', src)

    def test_a_failed_ffmpeg_does_not_kill_the_run(self):
        # Кадры уже на диске, число стыка посчитано; ролик пересобирается одной
        # строкой. Ронять из-за него прогон в 31 минуту нельзя.
        src = inspect.getsource(run_local._animatediff_once)
        cut = src[src.index("trim_to_loop"):]
        self.assertIn("except Exception", cut)

    def test_the_row_reaches_the_report_either_way(self):
        # Строка добавляется в `rows` ВСЕГДА, а не только при провале: «луп
        # сошёлся сам» — это тоже сведение, и его отсутствие читается как
        # «шаг не выполнялся».
        src = inspect.getsource(run_local._animatediff_once)
        self.assertIn("rows.append(loop_row)", src)


class TheCutItself(unittest.TestCase):
    """Свойства `best_loop_cut`, на которые опирается ступень."""

    def test_too_few_frames_is_a_refusal_and_not_a_cut_at_zero(self):
        got = motion.best_loop_cut(["a.png", "b.png"])
        self.assertIsNone(got["ratio"])

    def test_the_bar_the_stage_reads_exists_and_is_a_number(self):
        # Ступень печатает `при баре SEAMLESS_MAX`. Если константа переедет,
        # строка отчёта соврёт молча.
        self.assertIsInstance(motion.SEAMLESS_MAX, float)
        self.assertGreater(motion.SEAMLESS_MAX, 0.0)

    def test_the_report_carries_how_much_of_the_clip_survives(self):
        """Цена обрезки — длина, и она обязана быть В ОТЧЁТЕ.

        Замкнутый клип вдвое короче — это не бесплатная починка, а размен, и
        решать его человеку. Отчёт, умолчавший про `kept_fraction`, подаёт
        размен как чистый выигрыш.
        """
        src = inspect.getsource(run_local._animatediff_once)
        self.assertIn("kept_fraction", src)
