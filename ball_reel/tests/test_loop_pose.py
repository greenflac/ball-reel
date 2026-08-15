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


def _bench_poses() -> list:
    """96 скелетов драйвинга, снятых с ПОЛНОГО размера и положенных в индекс.

    ПОЧЕМУ ФИКСТУРА, А НЕ ЭКСТРАКЦИЯ НА ЛЕТУ. Раньше тест ходил в `ref_frames`
    — каталог, которого в git нет ни одним файлом. У автора он лежал локально,
    и тест был зелёным; в клоне он молча пропускался (22 пропуска против 7 в
    рабочем дереве). Полные кадры весят 67 МБ и в репозиторий не поедут, а
    уменьшенные дают ДРУГОЙ ВЫБОР ОКНА — см. `ResolutionChangesTheChoice`.
    Поэтому в индекс кладётся то, что тест на самом деле судит: выход
    экстрактора, 0.08 МБ JSON. Сам экстрактор сторожится отдельно, ниже.

    Побочно это снимает зависимость от весов MediaPipe: выбор окна теперь
    проверяется в любом клоне, а не только там, где веса скачались.
    """
    import json

    p = ROOT / "demo" / "bench" / "pose_96.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [{k: tuple(v) for k, v in f.items()} for f in data["frames"]]


class MeasuredOnTheRealDriving(unittest.TestCase):
    """Числа из шапки пересчитываются, а не запоминаются.

    Пересчитать:

        python3 -c "import json;from ball_reel import motion;\
d=json.load(open('demo/bench/pose_96.json',encoding='utf-8'));\
g=motion.best_loop_window_pose([{k:tuple(v) for k,v in f.items()} \
for f in d['frames']],size=16);print(g['note'])"
    """

    POSES = _bench_poses()

    def test_the_chosen_window_hides_the_seam_behind_most_steps(self):
        if len(self.POSES) < 96:
            self.skipTest("стенда demo/bench/pose_96.json нет в дереве")
        got = motion.best_loop_window_pose(self.POSES, size=16)
        self.assertIsNotNone(got["seam"])
        self.assertGreater(got["hidden"], 0.5,
                           f"стык перестал прятаться за движением: {got['note']}")

    def test_the_bench_reproduces_the_numbers_in_the_header(self):
        """Шапка называет окно 52..67 со стыком 0.0695 при медиане 0.1048.

        Проверяется именно это, а не «что-нибудь лучше нуля»: иначе шапка может
        разойтись с кодом, и разошедшийся документ вреднее отсутствующего.
        """
        if len(self.POSES) < 96:
            self.skipTest("стенда demo/bench/pose_96.json нет в дереве")
        got = motion.best_loop_window_pose(self.POSES, size=16)
        self.assertEqual(got["start"], 52, got["note"])
        self.assertAlmostEqual(got["seam"], 0.0695, places=3)
        self.assertAlmostEqual(got["local_median"], 0.1048, places=3)

    def test_the_slow_segment_the_header_warns_about_does_not_win(self):
        """Окно 35..50 выигрывает по глобальной мере и проигрывать обязано.

        Это тот самый медленный участок из шапки: стык там мельче абсолютно
        (0.0357 против 0.0695), но и всё вокруг мельче, так что виден он будет
        сильнее. Локальный критерий и заведён ради этого различия — тест
        краснеет, если критерий тихо станет глобальным.
        """
        if len(self.POSES) < 96:
            self.skipTest("стенда demo/bench/pose_96.json нет в дереве")
        got = motion.best_loop_window_pose(self.POSES, size=16)
        self.assertNotEqual(got["start"], 35, got["note"])


class ResolutionChangesTheChoice(unittest.TestCase):
    """НАХОДКА: выбор окна по позе зависит от размера кадра. Записана числом.

    Скелеты сняты одним и тем же экстрактором с одного и того же видео, разница
    только в ширине кадра:

        720 px  ->  окно 52   (то, что в шапке)
        360 px  ->  окно 64
        240 px  ->  окно 35   МЕДЛЕННЫЙ УЧАСТОК, против которого писан критерий
        180 px  ->  окно 66

    Согласия нет ни у одной пары. Причина не в критерии, а в дрожании
    MediaPipe: на мелком кадре шум сустава сравним с шагом движения, и
    локальная медиана — знаменатель критерия — плывёт вместе с ним. Следствие
    для продукта: позы для выбора окна снимаются с ИСХОДНОГО размера, а не с
    рабочего, иначе выбор достаётся участку, где просто мало движения.

    Тест сторожит ровно это: если кто-нибудь заменит фикстуру экстракцией с
    уменьшенных кадров стенда ради скорости, он тут же увидит цену.
    """

    POOL = sorted(glob.glob(str(ROOT / "demo" / "bench" / "driving" / "*.jpg")))

    def test_poses_from_the_shrunken_bench_pick_the_slow_segment(self):
        if len(self.POOL) < 96:
            self.skipTest("стенда demo/bench/driving нет в дереве")
        try:
            from ball_reel.pose import landmarks
        except Exception as exc:                     # весов MediaPipe нет
            self.skipTest(f"экстрактор недоступен: {exc}")

        pts = [landmarks(p) for p in self.POOL]
        if not all(pts):
            self.skipTest("веса MediaPipe недоступны — скелеты не сняты")
        got = motion.best_loop_window_pose(pts, size=16)
        self.assertEqual(
            got["start"], 35,
            "уменьшённый стенд перестал выбирать медленный участок — находка "
            f"про чувствительность к размеру устарела, перезамерить: {got['note']}")

    def test_the_skeleton_is_found_on_every_frame_unlike_pixels(self):
        if len(self.POOL) < 96:
            self.skipTest("стенда demo/bench/driving нет в дереве")
        try:
            from ball_reel.pose import landmarks
        except Exception as exc:
            self.skipTest(f"экстрактор недоступен: {exc}")

        pts = [landmarks(p) for p in self.POOL[:24]]
        if not any(pts):
            self.skipTest("веса MediaPipe недоступны")
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
