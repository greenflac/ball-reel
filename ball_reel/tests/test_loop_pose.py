"""Луп судится ПО СКЕЛЕТУ и БЕЗ ПОРОГА. Оба решения — следствия замеров.

ПОЧЕМУ СКЕЛЕТ. Пиксельная разность меряет свет, фон и шум компрессии заодно с
движением: реальное видео не возвращается в тот же КАДР, даже когда человек
вернулся в ту же ПОЗУ. Скелет от этого свободен, и он же — то, что потребляет
ControlNet: замкнув его, мы замыкаем ровно то, что доедет до генерации, а
пиксели финального рендера рисуются заново. Измерено: поза найдена на 96 кадрах
из 96; лучшее окно по пикселям 66..81, по скелету 52..67 — меры выбирают РАЗНОЕ.

ПОЧЕМУ БЕЗ ПОРОГА. Его пробовали вывести тем же способом, которым калибровался
отбор FaceNet, — два облака и граница в разрыве. Облака перекрылись:

    соседние кадры   медиана 0.0530  p95 0.3985
    далёкие кадры    min 0.0437      медиана 0.6080

Минимум «далёких» МЕНЬШЕ медианы «соседних»: упражнение циклично, и время не
есть мера различия поз. Это не сбой стенда — это доказательство, что луп
вообще существует. Глобальный порог не выводится в принципе: «далеко по позе» и
есть то, что мы определяем.

Критерий поэтому ЛОКАЛЬНЫЙ: стык незаметен, если не выбивается из ритма
соседних переходов. Разница между локальным и глобальным измерена:

    окно 35..50  стык 0.0357, локальная медиана 0.0305 -> мельче  6 из 15
    окно 52..67  стык 0.0695, локальная медиана 0.1048 -> мельче 11 из 15

Первое побеждает по глобальной мере и является МЕДЛЕННЫМ участком, где стык мал
просто потому, что там всё мелкое. Видно будет второе.
"""

from __future__ import annotations

import glob
import inspect
import unittest
from pathlib import Path

from ball_reel import motion, run_local

ROOT = Path(__file__).resolve().parents[2]


def _pose(x: float, y: float = 0.0) -> dict:
    """Поза-скелет с управляемым смещением рук. Торс неподвижен."""
    # Точка — тройка (x, y, видимость), ровно как её отдаёт `pose.landmarks`.
    return {
        "l_shoulder": (0.4, 0.3, 1.0), "r_shoulder": (0.6, 0.3, 1.0),
        "l_hip": (0.42, 0.7, 1.0), "r_hip": (0.58, 0.7, 1.0),
        "l_elbow": (0.3 + x, 0.45 + y, 1.0), "r_elbow": (0.7 - x, 0.45 + y, 1.0),
        "l_wrist": (0.25 + x, 0.6 + y, 1.0), "r_wrist": (0.75 - x, 0.6 + y, 1.0),
        "l_knee": (0.42, 0.85, 1.0), "r_knee": (0.58, 0.85, 1.0),
        "l_ankle": (0.42, 0.98, 1.0), "r_ankle": (0.58, 0.98, 1.0),
    }


class ItRefusesHonestly(unittest.TestCase):

    def test_too_few_poses_is_a_refusal_not_window_zero(self):
        got = motion.best_loop_window_pose([_pose(0)] * 5, size=16)
        self.assertIsNone(got["seam"])
        self.assertIn("выбирать не из чего", got["note"])

    def test_no_skeletons_at_all_is_a_refusal(self):
        got = motion.best_loop_window_pose([None] * 40, size=16)
        self.assertIsNone(got["seam"])
        self.assertIn("судить нечем", got["note"])

    def test_a_refusal_never_pretends_the_window_is_zero_by_choice(self):
        # `start` = 0 в отказе есть, но `seam` = None — вызывающий обязан
        # различать «выбрано ноль» и «выбирать не удалось».
        got = motion.best_loop_window_pose([None] * 40, size=16)
        self.assertEqual(got["start"], 0)
        self.assertIsNone(got["seam"])


class TheCriterionIsLocal(unittest.TestCase):

    def test_the_same_seam_wins_or_loses_depending_on_the_local_tempo(self):
        """Суть локального критерия, выделенная в чистом виде.

        У обоих участков стык ОДИНАКОВЫЙ по величине — 0.02. Различается только
        то, что происходит рядом:

            медленный: рядовые шаги 0.01 -> стык ВДВОЕ КРУПНЕЕ любого из них
            быстрый:   рядовые шаги 0.10 -> стык впятеро мельче любого

        Абсолютная величина стыка тут не различает ничего, а видно будет
        второй участок. Именно поэтому критерий локальный: зритель сравнивает
        шов не с константой и не со средним темпом ролика, а с движением,
        которое идёт вокруг шва.
        """
        # Медленный: подъём по 0.01 и возврат не в начало, а на 0.02 от него.
        slow_x = [0.0, .01, .02, .03, .04, .05, .06, .07,
                  .06, .05, .04, .03, .02, .01, .005, .02]
        # Быстрый: та же форма, шаги по 0.1, и тот же стык 0.02.
        fast_x = [1.0 + v for v in
                  [0.0, .1, .2, .3, .4, .5, .4, .3, .2, .1, .0, .1, .2, .1,
                   .0, .02]]
        got = motion.best_loop_window_pose(
            [_pose(v) for v in slow_x] + [_pose(v) for v in fast_x], size=16)
        self.assertIsNotNone(got["seam"])
        self.assertGreaterEqual(
            got["start"], len(slow_x),
            f"при одинаковом стыке выбран медленный участок: {got['note']}")

    def test_hidden_is_a_share_so_it_does_not_depend_on_window_length(self):
        # Доля, а не разность: иначе число нельзя перенести на другой темп и
        # другую длину окна без перекалибровки.
        poses = [_pose(0.05 * i) for i in range(40)]
        a = motion.best_loop_window_pose(poses, size=8)
        b = motion.best_loop_window_pose(poses, size=16)
        for got in (a, b):
            self.assertIsNotNone(got["hidden"])
            self.assertLessEqual(got["hidden"], 1.0)
            self.assertGreaterEqual(got["hidden"], 0.0)

    def test_the_note_says_there_is_no_bar_and_why(self):
        """Отсутствие порога обязано быть ЗАЯВЛЕНИЕМ, а не умолчанием.

        Читатель отчёта, не найдя бара, по умолчанию решит, что его забыли.
        """
        got = motion.best_loop_window_pose([_pose(0.03 * i) for i in range(40)],
                                           size=16)
        self.assertIn("Порога здесь НЕТ намеренно", got["note"])


class MeasuredOnTheRealDriving(unittest.TestCase):
    """Числа из шапки пересчитываются, а не запоминаются."""

    POOL = sorted(glob.glob(str(ROOT / "ref_frames" / "*.png")))

    def test_the_chosen_window_hides_the_seam_behind_most_steps(self):
        if len(self.POOL) < 32:
            self.skipTest("нет исходника драйвинга в дереве")
        from ball_reel.pose import landmarks

        got = motion.best_loop_window_pose([landmarks(p) for p in self.POOL],
                                           size=16)
        self.assertIsNotNone(got["seam"])
        self.assertGreater(got["hidden"], 0.5,
                           f"стык перестал прятаться за движением: {got['note']}")

    def test_the_skeleton_is_found_on_every_frame_unlike_pixels(self):
        if len(self.POOL) < 32:
            self.skipTest("нет исходника драйвинга в дереве")
        from ball_reel.pose import landmarks

        pts = [landmarks(p) for p in self.POOL[:24]]
        self.assertEqual(sum(1 for p in pts if p), len(pts),
                         "скелет найден не везде — тогда выбор окна опирается "
                         "на дырявые данные, и это надо сказать в отчёте")


class ItIsWiredIntoTheRun(unittest.TestCase):

    def test_the_run_chooses_by_pose_and_not_by_pixels(self):
        src = inspect.getsource(run_local._run_animatediff)
        self.assertIn("best_loop_window_pose", src)
        self.assertNotIn("best_loop_window(", src,
                         "прогон снова выбирает окно по пикселям")

    def test_it_uses_the_same_extractor_the_conditions_came_from(self):
        # Условия сняты DWPose; выбирать окно другим экстрактором значит
        # сравнивать одно с другим.
        src = inspect.getsource(run_local._run_animatediff)
        block = src[src.index("best_loop_window_pose") - 400:]
        self.assertIn("dwpose", block[:600])
