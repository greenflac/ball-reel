"""Отбор петель: арифметика стыка, подавление пересечений, три исхода.

ПОЧЕМУ ЗДЕСЬ НЕТ MEDIAPIPE (Т4). Снятие поз стоит секунды на кадр и требует
пакета с весами; тест, зависящий от него, краснел бы от чужого окружения и
зеленел бы от кэша. Позы приходят через точку внедрения `read_pose`, а все
последовательности порождаются здесь же формулой — маятник (петля есть),
монотонный дрейф (петли нет вовсе), два упражнения подряд (петель обязано быть
две). У каждого прибора негативный контроль С ОБЕИХ СТОРОН (И5): вход, где он
обязан найти, и вход, где он обязан сказать «не нашлось».
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from .. import fork_comfy, pose
from .. import fork_looper as fl
from ..fork_identity import FAIL, PASS, UNMEASURED

PERIOD = 44          # период фикстур: петля выходит 45 кадров, то есть 4k+1
NFRAMES = 96         # столько же, сколько в demo/bench/driving


# ---------------------------------------------------------------------------
# Синтетические скелеты. Бёдра и плечи неподвижны (торс = 0.2 в координатах
# кадра), поэтому приведение к торсу считается в уме, а не приборами.
# ---------------------------------------------------------------------------

def skeleton(phase, *, mode="arms", tired=0.0, amp=0.10, seen=()):
    ph = 2 * math.pi * phase
    ax, ay = amp * math.cos(ph), amp * math.sin(ph)
    lx = ly = 0.0
    if mode == "legs":
        lx, ly, ax, ay = ax, ay, 0.0, 0.0
    elif mode == "line":     # руки строго вверх-вниз: поза неоднозначна
        ax = 0.0
    elif mode == "legs_line":  # ноги строго вверх-вниз, руки стоят
        lx, ly, ax, ay = 0.0, ay, 0.0, 0.0
    pts = {
        "l_hip": (0.45, 0.60), "r_hip": (0.55, 0.60),
        "l_shoulder": (0.44, 0.40), "r_shoulder": (0.56, 0.40),
        "l_elbow": (0.40 + ax, 0.50 + ay + tired),
        "r_elbow": (0.60 - ax, 0.50 + ay + tired),
        "l_wrist": (0.38 + 2 * ax, 0.58 + 2 * ay + 2 * tired),
        "r_wrist": (0.62 - 2 * ax, 0.58 + 2 * ay + 2 * tired),
        "l_knee": (0.44 + lx, 0.75 + ly), "r_knee": (0.56 - lx, 0.75 + ly),
        "l_ankle": (0.44 + 2 * lx, 0.90 + 2 * ly),
        "r_ankle": (0.56 - 2 * lx, 0.90 + 2 * ly),
    }
    return {k: (x, y, 1.0 if (not seen or k in seen) else 0.0)
            for k, (x, y) in pts.items()}


def loop_sequence(n=NFRAMES, *, tire=0.0, mode="arms"):
    """Маятник: движение точно повторяется каждые PERIOD кадров."""
    return [skeleton(t / PERIOD, mode=mode, tired=tire * t) for t in range(n)]


def drift_sequence(n=NFRAMES, *, tire=0.002):
    """Монотонный дрейф: конфигурация уезжает в одну сторону, петли нет вовсе."""
    return [skeleton(0.25, tired=tire * t) for t in range(n)]


ANCHOR = ("l_hip", "r_hip", "l_shoulder", "r_shoulder")


def seated(points, tilt):
    """Та же механика движения, но ДРУГАЯ ПОСАДКА: конечности смещены целиком.

    Без этого «другое упражнение» фикстур отличалось на 14..17 типичных шагов,
    тогда как на боевом ролике разные упражнения расходятся на 55, а варианты
    одного — на 17. То есть синтетика описывала варианты, а не упражнения, и
    порог, откалиброванный по живому материалу, честно их схлопывал.
    """
    return {k: (x, y if k in ANCHOR else y + tilt, v)
            for k, (x, y, v) in points.items()}


def two_exercises(n=NFRAMES):
    """Первую половину работают руки, вторую — ноги в другой посадке.

    ИЗМЕРЕНО: 31.4 типичных шага между ними при пороге 24.
    """
    return [skeleton(t / PERIOD, mode="arms") if t < n // 2
            else seated(skeleton(t / PERIOD, mode="legs", amp=0.2), 0.6)
            for t in range(n)]


def two_exercises_second_is_perfect(n=NFRAMES, tire=0.0001):
    """Два упражнения, и второе замыкается ТОЧНЕЕ первого.

    A (руки) слегка уезжает — его мост стоит 1 кадр; B (ноги в другой посадке)
    замыкается точно — мост 0 кадров. Значит по цене моста первым обязан идти
    B, и это проверяемо в обе стороны: ослепив лицо на кадрах B, мы обязаны
    увидеть B ВНИЗУ списка, хотя его нижняя граница дешевле цены A.

    Снос `tire` выбран замером: 0.0001 даёт стык 0.15 типичного шага — петля A
    ещё проходит планку преимущества (иначе её просто не будет в выдаче), но
    уже дороже нулевого B.
    """
    return [skeleton(t / PERIOD, mode="arms", tired=tire * t) if t < n // 2
            else seated(skeleton(t / PERIOD, mode="legs", amp=0.2), 0.6)
            for t in range(n)]


def two_speeds(n=NFRAMES, tire=0.0001):
    """Два упражнения РАЗНОЙ СКОРОСТИ: медленное со сносом, потом быстрое.

    Нужна ровно затем, чтобы глобальный знаменатель разошёлся с локальным:
    медиана шага по клипу задаётся быстрым упражнением, а стык меряется у
    медленного. На боевом ролике это расхождение доходило до 5.15 раза и
    заказывало мост вчетверо длиннее нужного.
    """
    return [skeleton(t / PERIOD, mode="arms", amp=0.03, tired=tire * t)
            if t < n // 2
            else seated(skeleton(t / PERIOD, mode="legs", amp=0.25), 0.6)
            for t in range(n)]


def still_sequence(n=NFRAMES):
    """Человек не движется: ранжировать стыки нечем — исход «не смогли»."""
    return [skeleton(0.0) for _ in range(n)]


def norm(points):
    return fl.states([points])[0]


# ---------------------------------------------------------------------------
# Материал на диске плюс подменный детектор поз
# ---------------------------------------------------------------------------

class Material:
    """Кадры на диске плюс подменные детектор поз и читалка пикселей (Т4).

    `blank=True` кладёт ПУСТЫЕ файлы: когда обе точки внедрения подменены,
    картинки никто не открывает, а тысяча настоящих PNG стоит секунд.
    """

    def __init__(self, poses, *, size=(32, 32), missing=(), broken=False,
                 people=None, cuts=(), blank=False, head_mode="возвращается",
                 head_blind=(), head_broken=False):
        from PIL import Image

        self.dir = Path(tempfile.mkdtemp(prefix="looper_frames_"))
        self.calls = []
        self.gray_calls = []
        self.poses = poses
        self.missing = set(missing)
        self.broken = broken
        self.people = people or {}
        self.cuts = set(cuts)
        self.head_mode = head_mode
        self.head_blind = set(head_blind)
        self.head_broken = head_broken
        self.head_calls = []
        for k in range(len(poses)):
            f = self.dir / f"{k:04d}.png"
            if blank:
                f.touch()
            else:
                Image.new("RGB", size, (k * 2 % 256, 40, 200 - k % 200)).save(f)

    def reader(self, path):
        self.calls.append(path)
        idx = int(Path(path).stem)
        if self.broken:
            return {"points": None, "why": "mediapipe не установлен (фикстура)"}
        if idx in self.missing:
            return {"points": None, "why": "", "people": 0}
        return {"points": self.poses[idx], "why": "",
                "people": self.people.get(idx)}

    def gray(self, path):
        """Пиксели, ВЫВЕДЕННЫЕ ИЗ СКЕЛЕТА этого кадра, плюс скачок на резах.

        Так синтетическая камера ведёт себя как настоящая: когда движение
        повторяется, повторяется и картинка; когда поза уезжает — уезжает и
        она. Картинка, не связанная с позой (например, просто номер кадра),
        сделала бы пиксельную ось шумом и тихо выключила бы её из проверок.

        Берётся `self.poses[idx]`, а не ответ `reader`: человек, которого
        детектор не увидел, из кадра не исчезает — пропадает только поза.
        """
        import numpy as np

        self.gray_calls.append(path)
        idx = int(Path(path).stem)
        pts = self.poses[idx]
        # Множитель держит значения в восьмибитном диапазоне: сумма координат
        # порядка 14, скачок реза 100, потолок 255. Первая редакция брала 100 и
        # отдавала 1400 — склад заворачивал их по модулю 256, и пиксельная ось
        # молча считала мусор. Теперь такой кадр отвергается сторожем.
        body = (4.0 * sum(x + y for x, y, _ in pts.values())) if pts else 0.0
        # Ход 1.0 за кадр, на резе — скачок 20.0, то есть отношение 21 к
        # типичному. Настоящий монтажный рез даёт десятки; сотню брать нельзя —
        # такая фикстура пережила бы планку, поднятую до 100, и не заметила бы
        # этого (мутант выжил ровно на этом, пока скачок был 100).
        # Скачок на резе даёт отношение около 22 к типичному ходу этой
        # фикстуры. Настоящий монтажный рез даёт десятки; сотню брать
        # нельзя — такая фикстура пережила бы планку, поднятую до 100, и не
        # заметила бы этого (мутант выжил ровно на этом).
        base = body + sum(5.0 for c in self.cuts if idx > c)
        return np.full((8, 8), base, dtype="float64")

    def head(self, path):
        """Голова, ТРЕТЬЯ ТОЧКА ВНЕДРЕНИЯ (Т4). Настоящий детектор 133 точек
        стоит 0.436 с/кадр, и один его вызов из сьюта превратил четыре секунды
        прогона в пять минут — поймано прогоном, а не рассуждением.

        Три поведения головы, и все три нужны (И5):
            `возвращается` — голова выводится из позы, петля закрывается точно;
            `уезжает`      — голова монотонно сползает, петля не закрывается;
            `рывок`        — голова стоит, но один раз прыгает на середине.
        """
        import numpy as np

        self.head_calls.append(path)
        idx = int(Path(path).stem)
        if self.head_broken:
            return {"head": None, "why": "весов DWPose нет (фикстура)"}
        if idx in self.head_blind:
            return {"head": None, "why": ""}
        pts = self.poses[idx]
        if pts is None:
            return {"head": None, "why": ""}
        # Голова — над серединой плеч, в пикселях кадра 720x1278, и она
        # ПОКАЧИВАЕТСЯ вместе с движением: неподвижная голова дала бы нулевой
        # типичный шаг, ось выключилась бы, и проверки прошли бы вхолостую.
        # Поймано прогоном: первая редакция держала плечи неподвижно.
        bob = 6.0 * float(np.sin(2 * np.pi * (idx % PERIOD) / PERIOD))
        x = 720 * (pts["l_shoulder"][0] + pts["r_shoulder"][0]) / 2
        y = 1278 * (pts["l_shoulder"][1] + pts["r_shoulder"][1]) / 2 - 60 + bob
        if self.head_mode == "уезжает":
            y += 0.7 * idx
        elif self.head_mode == "рывок":
            y += 0.0 if idx < NFRAMES // 2 else 40.0
        return {"head": (float(x), float(y)), "why": ""}

    def paths(self):
        return fl.frame_paths(self.dir)


#: Частота фикстур. ПОДАЁТСЯ ЯВНО в каждом прогоне: у каталога кадров частоты
#: нет нигде, и прибор обязан это знать (см. класс `SourceFps`).
FIXTURE_FPS = 30


def analyse(material, **kw):
    """Прогон прибора на фикстуре: ВСЕ ТРИ точки внедрения подменены (Т4).

    Третью (голову) забыли подать в первой редакции, и настоящий детектор 133
    точек по 0.436 с на кадр растянул сьют с четырёх секунд до пяти с лишним
    минут. Умолчание здесь — не удобство, а сторож: прогон, зовущий сеть весов,
    краснеет от чужой аварии и зеленеет от кэша.
    """
    kw.setdefault("fps", FIXTURE_FPS)
    kw.setdefault("gif", False)
    kw.setdefault("head", material.head)
    return fl.find_loops(material.dir, reader=material.reader,
                         gray=material.gray, **kw)


# ---------------------------------------------------------------------------
# 1. Геометрия: поза
# ---------------------------------------------------------------------------

class PoseAxis(unittest.TestCase):
    def test_pose_gap_of_a_frame_against_itself_is_zero(self):
        """Негативный контроль «прибор обязан промолчать» (И5)."""
        a = norm(skeleton(0.3))
        self.assertEqual(fl.pose_gap(a, a), 0.0)

    def test_pose_gap_is_a_hand_computable_literal(self):
        """Ожидаемое — литерал, посчитанный на бумаге, а не импорт (Т2).

        Торс фикстуры равен 0.2 кадра. Сдвинув ОДНО запястье на 0.02, получаем
        0.1 длины торса у одного сустава из двенадцати: 0.1/12 = 0.008333.
        """
        a = skeleton(0.0)
        b = dict(a)
        b["l_wrist"] = (a["l_wrist"][0], a["l_wrist"][1] + 0.02, 1.0)
        self.assertAlmostEqual(fl.pose_gap(norm(a), norm(b)), 0.008333, places=6)

    def test_pose_gap_is_the_same_number_as_pose_delta(self):
        """Е1: своя реализация обязана давать то же, что приёмка поз проекта."""
        a, b = skeleton(0.0), skeleton(0.3)
        mine = fl.pose_gap(norm(a), norm(b))
        theirs = pose.pose_delta(a, b)["mean"]
        self.assertAlmostEqual(mine, theirs, places=4,
                               msg="расхождение с pose.pose_delta означает, что "
                                   "прибор судит не той величиной, которой "
                                   "судит вся остальная приёмка поз")

    def test_a_frame_without_hips_is_not_a_pose_at_all(self):
        """Приведение требует оба бедра и оба плеча — иначе позы нет.

        Отсюда же следует, что своего пола в 4 общих сустава прибору не нужно:
        меньше четырёх после приведения не бывает.
        """
        half = norm(skeleton(0.0, seen=("l_hip", "l_shoulder", "r_shoulder")))
        self.assertIsNone(half)
        self.assertIsNone(fl.pose_gap(half, norm(skeleton(0.0))))

    def test_pose_gap_says_nothing_when_a_frame_has_no_body(self):
        self.assertIsNone(fl.pose_gap(None, norm(skeleton(0.0))))
        self.assertIsNone(fl.pose_gap(norm(skeleton(0.0)), None))


# ---------------------------------------------------------------------------
# 2. Геометрия: поток. ГЛАВНАЯ ТОНКОСТЬ ЗАДАЧИ
# ---------------------------------------------------------------------------

class FlowAxis(unittest.TestCase):
    def _bounce(self):
        """Четыре кадра: поза в 0 и в 2 ОДНА И ТА ЖЕ, движение противоположно.

        Ровно тот случай, ради которого вторая ось и заведена: в кадре 0
        запястье идёт вниз, в кадре 2 — вверх. Склеив их по совпадению позы,
        получим отскок, видимый глазом и невидимый для первой оси.
        """
        a = skeleton(0.0)
        down = dict(a); up = dict(a)
        down["l_wrist"] = (a["l_wrist"][0], a["l_wrist"][1] + 0.02, 1.0)
        up["l_wrist"] = (a["l_wrist"][0], a["l_wrist"][1] - 0.02, 1.0)
        return fl.states([a, down, a, up])

    def test_the_pose_axis_alone_calls_a_bounce_a_perfect_seam(self):
        st = self._bounce()
        self.assertEqual(fl.pose_gap(st[0], st[2]), 0.0,
                         "если это перестанет быть нулём, фикстура больше не "
                         "показывает разбираемый дефект")

    def test_the_flow_axis_catches_it_with_a_hand_computable_literal(self):
        """0.02 кадра при торсе 0.2 — 0.1; направления противоположны, значит
        расхождение 0.2 у одного сустава из двенадцати: 0.2/12 = 0.016667."""
        st = self._bounce()
        self.assertAlmostEqual(fl.flow_gap(st, 0, 2), 0.016667, places=6)

    def test_the_flow_axis_is_silent_when_directions_agree(self):
        st = fl.states(loop_sequence(60))
        self.assertAlmostEqual(fl.flow_gap(st, 0, PERIOD), 0.0, places=9)

    def test_the_flow_axis_needs_the_frame_after_the_seam(self):
        st = fl.states(loop_sequence(50))
        self.assertIsNone(fl.flow_gap(st, 0, 49),
                          "производную в последнем кадре взять не из чего, и "
                          "это «не смогли», а не ноль")


# ---------------------------------------------------------------------------
# 3. Допустимые длины: 4k+1 и потолок
# ---------------------------------------------------------------------------

class Lengths(unittest.TestCase):
    def test_every_length_survives_the_wrapper_snap(self):
        """Прижатие обёртки на кадр разъезжает петлю (Е1: считает fork_comfy)."""
        for L in fl.admissible_lengths(NFRAMES):
            self.assertEqual(fork_comfy.snap_frames(L), L, f"длина {L} прижмётся")

    def test_the_floor_is_forty_one_frames(self):
        got = fl.admissible_lengths(NFRAMES)
        self.assertEqual(got[0], 41)
        self.assertNotIn(37, got, "37 кадров — 1.23 с, десяток повторов подряд")
        self.assertNotIn(5, got)

    def test_the_ceiling_comes_from_the_product_length_and_not_from_here(self):
        got = fl.admissible_lengths(100000, fps=30)
        self.assertLessEqual(got[-1], fork_comfy.SECONDS_MAX * 30)
        self.assertEqual(got[-1], 297)
        self.assertEqual(fl.admissible_lengths(100000, fps=24)[-1], 237,
                         "потолок обязан ехать за частотой источника: 10 с при "
                         "24 к/с — это 240 кадров, ближайшее 4k+1 снизу 237")

    def test_without_a_frame_rate_there_is_no_ceiling_at_all(self):
        """Потолок продуктовый и выражен в секундах; без частоты его нет."""
        got = fl.admissible_lengths(1000, fps=None)
        self.assertEqual(got[-1], 997)
        self.assertEqual(got[0], 41)

    def test_a_clip_shorter_than_the_floor_admits_nothing(self):
        self.assertEqual(fl.admissible_lengths(40), [])


# ---------------------------------------------------------------------------
# 4. Сведение двух осей в одну оценку
# ---------------------------------------------------------------------------

class SeamScore(unittest.TestCase):
    def _sim(self):
        # Пара A: поза разошлась на 0.02, направления совпали.
        # Пара B: поза сошлась идеально, направления противоположны (отскок).
        return {"pose": {(0, 44): 0.02, (10, 54): 0.0},
                "flow": {(0, 44): 0.0, (10, 54): 0.05},
                "lengths": [45], "pairs": 2, "measured": 2, "unmeasurable": 0}

    def test_the_bounce_ranks_worse_than_the_honest_seam(self):
        """При шаге клипа 0.05: A даёт max(0.4, 0) = 0.4, B — max(0, 1.0) = 1.0."""
        got = fl.score_pairs(self._sim(), 0.05)
        self.assertEqual([c["i"] for c in got], [0, 10])
        self.assertEqual([c["score"] for c in got], [0.4, 1.0])

    def test_a_pair_with_one_axis_unmeasured_is_not_a_candidate(self):
        sim = self._sim()
        sim["flow"][(0, 44)] = None
        self.assertEqual([c["i"] for c in fl.score_pairs(sim, 0.05)], [10])

    def test_a_motionless_clip_cannot_be_ranked(self):
        with self.assertRaises(ValueError):
            fl.score_pairs(self._sim(), 0.0)


# ---------------------------------------------------------------------------
# 5. Подавление пересечений
# ---------------------------------------------------------------------------

def cand(i, j, score):
    return {"i": i, "j": j, "frames": j - i + 1, "score": score}


class Suppression(unittest.TestCase):
    def test_overlap_is_a_share_of_the_shorter_loop(self):
        self.assertAlmostEqual(fl.overlap(cand(0, 44, 1), cand(30, 74, 1)),
                               15 / 45, places=6)
        self.assertEqual(fl.overlap(cand(0, 44, 1), cand(45, 89, 1)), 0.0)

    def test_a_copy_shifted_by_four_frames_is_not_a_second_loop(self):
        got = fl.select([cand(0, 44, 0.5), cand(4, 48, 0.6)])
        self.assertEqual([(c["i"], c["j"]) for c in got["kept"]], [(0, 44)])
        self.assertEqual(got["dropped_overlap"], 1)

    def test_a_third_of_a_loop_in_common_still_counts_as_a_different_loop(self):
        got = fl.select([cand(0, 44, 0.5), cand(30, 74, 0.6)])
        self.assertEqual([(c["i"], c["j"]) for c in got["kept"]],
                         [(0, 44), (30, 74)])

    def test_the_table_is_capped(self):
        many = [cand(k * 50, k * 50 + 44, 0.1 * k) for k in range(8)]
        got = fl.select(many)
        self.assertEqual(len(got["kept"]), 5)


# ---------------------------------------------------------------------------
# 6. Повторы до продуктовой длины
# ---------------------------------------------------------------------------

class Repeats(unittest.TestCase):
    def test_the_numbers_match_the_ones_measured_on_the_material(self):
        """Литералы из хэндофа: 45 кадров, склейка N*44+1."""
        got = fl.repeat_plan(45, fps=30)
        self.assertEqual([(r["repeats"], r["frames"], r["seconds"]) for r in got],
                         [(4, 177, 5.9), (5, 221, 7.37), (6, 265, 8.83)])

    def test_every_glued_length_survives_the_wrapper_snap(self):
        for L in fl.admissible_lengths(NFRAMES):
            for r in fl.repeat_plan(L):
                self.assertEqual(r["snapped"], r["frames"],
                                 f"склейка {r['repeats']}x{L} прижмётся")

    def test_every_admissible_loop_can_be_grown_to_product_length(self):
        """Свойство, а не совпадение: полоса 5-10 с шире вдвое, поэтому
        подходящее число повторов есть у любой допустимой длины."""
        for L in fl.admissible_lengths(NFRAMES, fps=FIXTURE_FPS):
            self.assertTrue(fl.repeat_plan(L, fps=FIXTURE_FPS),
                            f"длину {L} не растянуть в 5-10 с")

    def test_a_loop_longer_than_the_product_fits_nothing(self):
        self.assertEqual(fl.repeat_plan(1000, fps=30), [])


# ---------------------------------------------------------------------------
# 7. GIF
# ---------------------------------------------------------------------------

class Gif(unittest.TestCase):
    def test_the_seam_frame_is_not_in_the_gif(self):
        idx = fl.gif_indices(0, 44)
        self.assertIn(0, idx)
        self.assertNotIn(44, idx,
                         "кадр j — повтор кадра i; оставив его, мы показали бы "
                         "оператору два одинаковых кадра вместо стыка")

    def test_the_gif_is_thinned(self):
        self.assertEqual(len(fl.gif_indices(0, 44)), 22)
        self.assertEqual(len(fl.gif_indices(0, 300)), 24)
        self.assertLessEqual(len(fl.gif_indices(0, 300)), fl.GIF_MAX_FRAMES)

    def test_a_short_loop_is_not_thinned(self):
        self.assertEqual(fl.gif_indices(10, 22), list(range(10, 22)))

    def test_the_gif_is_written_and_scaled_down(self):
        m = Material(loop_sequence(46), size=(800, 600))
        out = Path(tempfile.mkdtemp(prefix="looper_gif_")) / "loop.gif"
        got = fl.make_gif(m.paths(), 0, 44, out)
        self.assertEqual(got["frames"], 22)
        self.assertEqual(got["size"], (320, 240))
        self.assertGreater(got["bytes"], 0)
        from PIL import Image
        with Image.open(out) as im:
            self.assertEqual(im.n_frames, 22)
            # Прорежённый вдвое GIF обязан идти с той же скоростью, что
            # материал: 2 кадра из 30 в секунду — 67 мс, а сам формат GIF меряет
            # время десятками миллисекунд, поэтому обратно читается 60.
            self.assertEqual(im.info["duration"], 60)


# ---------------------------------------------------------------------------
# 8. Разбор последовательностей: обе стороны негативного контроля (И5)
# ---------------------------------------------------------------------------

class Sequences(unittest.TestCase):
    def test_a_pendulum_has_a_loop_and_it_is_exactly_the_period(self):
        m = Material(loop_sequence())
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        best = got["loops"][0]
        self.assertEqual((best["i"], best["j"]), (0, PERIOD))
        self.assertEqual(best["frames"], PERIOD + 1)
        self.assertEqual(best["score"], 0.0)

    def test_a_drift_has_no_loop_and_the_margin_is_1_40(self):
        """ИЗМЕРЕНО: 1.40, и это то число, между которым и 4.2 стоит планка."""
        m = Material(drift_sequence())
        got = analyse(m)
        self.assertEqual(got["outcome"], FAIL, got["note"])
        self.assertEqual(got["loops"], [])
        self.assertAlmostEqual(got["advantage"], 1.4, places=1)
        self.assertIn("ПЕТЕЛЬ НЕ НАШЛОСЬ", got["note"])
        self.assertGreater(got["measured_pairs"], 300,
                           "Р2: «не нашлось» обязано стоять рядом с числом "
                           "разобранных пар, иначе оно неотличимо от «не искали»")

    def test_a_drift_stays_loopless_at_three_different_speeds(self):
        """Т3: фикстура берётся с обоих краёв диапазона и из середины."""
        for tire in (0.0005, 0.002, 0.01):
            with self.subTest(tire=tire):
                m = Material(drift_sequence(tire=tire))
                got = analyse(m)
                self.assertEqual(got["outcome"], FAIL)
                self.assertAlmostEqual(got["advantage"], 1.4, places=1)

    def test_a_tiring_pendulum_is_still_a_loop_but_a_worse_one(self):
        """Середина диапазона (Т3): повтор есть, но человек по ходу уезжает."""
        m = Material(loop_sequence(tire=0.0005))
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertGreater(got["loops"][0]["score"], 0.0,
                           "стык уже не идеален, и это обязано быть видно")

    def test_where_the_bar_bites_is_measured_from_both_sides(self):
        """ИЗМЕРЕНО на одной фикстуре с двумя скоростями сползания:

            сползание 0.004 на кадр -> преимущество 3.42 -> петля показана
            сползание 0.006 на кадр -> преимущество 1.66 -> петель не нашлось

        Планка 2.0 лежит между ними. Это и есть негативный контроль обеих
        сторон (И5) на одном и том же приборе.
        """
        for tire, outcome, lo, hi in ((0.004, PASS, 3.2, 3.6),
                                      (0.006, FAIL, 1.5, 1.8)):
            with self.subTest(tire=tire):
                m = Material([skeleton(t / PERIOD, tired=tire * t)
                              for t in range(NFRAMES)])
                got = analyse(m)
                self.assertEqual(got["outcome"], outcome, got["note"])
                self.assertGreater(got["advantage"], lo)
                self.assertLess(got["advantage"], hi)

    def test_two_exercises_give_two_loops_one_in_each(self):
        m = Material(two_exercises())
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(len(got["loops"]), 2,
                         f"упражнения два, петель {len(got['loops'])}: "
                         f"{[(l['i'], l['j']) for l in got['loops']]}")
        halves = sorted((lp["i"] < NFRAMES // 2) for lp in got["loops"])
        self.assertEqual(halves, [False, True],
                         "обе петли уехали в одну половину — второе упражнение "
                         "потеряно")

    def test_a_still_clip_is_not_measurable_rather_than_loopless(self):
        """Р1: «не смогли» не сворачивается ни в «годно», ни в «не годно»."""
        m = Material(still_sequence())
        got = analyse(m)
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertEqual(got["loops"], [])

    def test_no_detector_is_not_the_same_as_no_bodies(self):
        m = Material(loop_sequence(), broken=True)
        got = analyse(m)
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertIn("mediapipe", got["note"])

    def test_half_the_frames_without_a_body_is_not_measurable(self):
        m = Material(loop_sequence(), missing=range(0, NFRAMES, 2))
        got = analyse(m)
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertIn("48 из 96", got["note"])

    def test_a_tenth_of_the_frames_without_a_body_still_measures(self):
        m = Material(loop_sequence(), missing=range(0, NFRAMES, 10))
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["taken"], 86)
        self.assertGreater(got["unmeasurable_pairs"], 0,
                           "пары с пропавшей позой обязаны считаться отдельно")

    def test_material_shorter_than_a_loop_fails_with_the_number(self):
        m = Material(loop_sequence(20))
        got = analyse(m)
        self.assertEqual(got["outcome"], FAIL, got["note"])
        self.assertIn("20", got["note"])

    def test_a_missing_source_is_not_measurable(self):
        got = fl.find_loops("/no/such/place/at/all", gif=False)
        self.assertEqual(got["outcome"], UNMEASURED)


# ---------------------------------------------------------------------------
# 9. Числа рядом с вердиктом, коды возврата, кэш
# ---------------------------------------------------------------------------

class ReportAndCache(unittest.TestCase):
    def test_the_report_carries_its_numbers(self):
        """Р2: ноль нарушений при нуле проверок — не успех."""
        m = Material(loop_sequence())
        got = analyse(m)
        for key in ("frames", "taken", "pairs", "measured_pairs",
                    "unmeasurable_pairs", "candidates", "worthy",
                    "dropped_overlap", "advantage", "typical_step"):
            self.assertIn(key, got, f"в отчёте нет числа {key}")
        self.assertEqual(got["frames"], NFRAMES)
        self.assertEqual(got["pairs"],
                         got["measured_pairs"] + got["unmeasurable_pairs"])

    def test_the_verdict_never_claims_seamlessness(self):
        """Планки бесшовности у модуля нет, и заявлять её он не смеет."""
        m = Material(loop_sequence())
        got = analyse(m)
        self.assertIn("РАНГ, а не вердикт", got["note"])
        self.assertNotIn("бесшов", got["note"].replace("бесшовности", ""))

    def test_three_outcomes_get_three_exit_codes(self):
        self.assertEqual(sorted(fl.EXIT_BY_OUTCOME.values()), [0, 1, 2])
        self.assertEqual(fl.EXIT_BY_OUTCOME[UNMEASURED], 2,
                         "сведение «не смогли» в 0 читало бы отсутствие "
                         "детектора как успех")

    def test_the_table_prints_the_repeat_plan(self):
        m = Material(loop_sequence())
        got = analyse(m)
        txt = fl.table(got)
        self.assertIn("4x=177", txt)
        self.assertIn("5.9", txt)

    def test_the_cache_spares_the_expensive_step(self):
        m = Material(loop_sequence(50))
        cache = Path(tempfile.mkdtemp(prefix="looper_cache_")) / "poses.json"
        first = fl.read_all(m.paths(), reader=m.reader, cache=cache)
        self.assertEqual(len(m.calls), 50)
        second = fl.read_all(m.paths(), reader=m.reader, cache=cache)
        self.assertEqual(len(m.calls), 50, "кэш не сработал: детектор позвали снова")
        self.assertEqual(second["cached"], 50)
        self.assertEqual(first["poses"], second["poses"])

    def test_a_changed_frame_invalidates_its_cache_entry(self):
        """Кэш, переживающий подмену кадра, — второй источник истины (Е1)."""
        from PIL import Image

        m = Material(loop_sequence(50))
        cache = Path(tempfile.mkdtemp(prefix="looper_cache_")) / "poses.json"
        fl.read_all(m.paths(), reader=m.reader, cache=cache)
        Image.new("RGB", (64, 64), (1, 2, 3)).save(m.dir / "0007.png")
        got = fl.read_all(m.paths(), reader=m.reader, cache=cache)
        self.assertEqual(got["cached"], 49)
        self.assertEqual(len(m.calls), 51)

    def test_a_cache_of_the_current_version_is_honoured(self):
        """Версия закреплена ЛИТЕРАЛОМ (Т2): импортировав её из модуля, тест
        поехал бы вместе с ней и перестал бы что-либо сторожить."""
        m = Material(loop_sequence(10))
        cache = Path(tempfile.mkdtemp(prefix="looper_cache_")) / "poses.json"
        fl.read_all(m.paths(), reader=m.reader, cache=cache)
        raw = json.loads(cache.read_text(encoding="utf-8"))
        raw["version"] = 1
        cache.write_text(json.dumps(raw), encoding="utf-8")
        got = fl.read_all(m.paths(), reader=m.reader, cache=cache)
        self.assertEqual(got["cached"], 10)
        self.assertEqual(len(m.calls), 10)

    def test_a_cache_of_another_version_is_ignored(self):
        m = Material(loop_sequence(10))
        cache = Path(tempfile.mkdtemp(prefix="looper_cache_")) / "poses.json"
        cache.write_text(json.dumps({"version": 999,
                                     "frames": {"мусор": None}}),
                         encoding="utf-8")
        got = fl.read_all(m.paths(), reader=m.reader, cache=cache)
        self.assertEqual(got["cached"], 0)
        self.assertEqual(len(m.calls), 10)


# ---------------------------------------------------------------------------
# 10. МОНТАЖНЫЕ РЕЗЫ: дефект, который выглядит как удача
# ---------------------------------------------------------------------------

class Cuts(unittest.TestCase):
    def test_a_cut_is_found(self):
        m = Material(loop_sequence(), cuts=(47,), blank=True)
        got = fl.cuts(m.paths(), gray=m.gray)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["cuts"], [47])
        self.assertEqual(got["steps"], NFRAMES - 1)
        self.assertAlmostEqual(got["worst"], 21.9, places=1)

    def test_a_cut_is_not_invented_on_smooth_material(self):
        """Негативный контроль второй стороны (И5): ровный ход — не рез."""
        m = Material(loop_sequence(), blank=True)
        got = fl.cuts(m.paths(), gray=m.gray)
        self.assertEqual(got["cuts"], [])
        self.assertAlmostEqual(got["worst"], 1.41, places=2,
                               msg="самый резкий переход ровного маятника — "
                                   "полтора типичных, до планки 4.0 далеко")

    def test_a_shake_is_not_a_cut_either(self):
        """Скачок втрое против типичного — это ещё движение, а не монтаж."""
        import numpy as np

        m = Material(loop_sequence(), blank=True)
        m.gray = lambda path: np.full(
            (8, 8), float(sum(3 if k % 10 == 0 else 1
                              for k in range(int(Path(path).stem)))) % 200)
        got = fl.cuts(m.paths(), gray=m.gray)
        self.assertEqual(got["worst"], 3.0)
        self.assertEqual(got["cuts"], [],
                         "планка стоит выше тряски и ниже монтажа; сдвинув её "
                         "вниз, мы объявим резом каждый резкий мах")

    def test_the_default_pixel_reader_downscales_to_the_declared_side(self):
        """Точка внедрения по умолчанию читается настоящим кадром, а не макетом:
        иначе размер, на котором меряются резы, не сторожит никто."""
        m = Material(loop_sequence(2), size=(240, 426))
        arr = fl.read_gray(str(m.paths()[0]))
        self.assertEqual(arr.shape, (fl.CUT_SIDE, fl.CUT_SIDE))
        self.assertEqual(arr.shape, (96, 96))

    def test_a_frozen_clip_cannot_be_asked_about_cuts(self):
        """Р1: «резов нет» и «резы не искали» — разные ответы."""
        import numpy as np

        m = Material(loop_sequence(), blank=True)
        got = fl.cuts(m.paths(), gray=lambda p: np.zeros((8, 8)))
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("НЕ ИСКАЛИ", got["note"])

    def test_a_loop_across_a_cut_is_never_offered(self):
        """САМАЯ ОПАСНАЯ ИЗ ПРОВЕРОК.

        Позы по обе стороны реза здесь ОДНИ И ТЕ ЖЕ (последовательность —
        честный маятник), то есть по позам петля через рез выглядит идеально.
        Отличить её можно только пикселями.
        """
        m = Material(loop_sequence(), cuts=(47,), blank=True)
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["cuts"], [47])
        self.assertGreater(got["rejected"].get("рез внутри петли", 0), 0)
        for lp in got["loops"]:
            with self.subTest(loop=(lp["i"], lp["j"])):
                self.assertTrue(lp["j"] <= 47 or lp["i"] > 47,
                                "петля перешагнула монтажный рез")


# ---------------------------------------------------------------------------
# 11. ЧЕЛОВЕК В КАДРЕ: три разных исхода, а не одно падение
# ---------------------------------------------------------------------------

class Presence(unittest.TestCase):
    def test_several_people_are_not_this_module_to_decide(self):
        """Е1: выбор протагониста уже решён в `fork_props`, второго не заводим."""
        m = Material(loop_sequence(), people={k: 2 for k in range(NFRAMES)})
        got = analyse(m)
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertIn("fork_props", got["note"])
        self.assertIn("протагонист", got["note"])
        self.assertEqual(got["crowd"], [2])

    def test_nobody_at_all_is_not_the_same_as_no_loops(self):
        m = Material(loop_sequence(), missing=range(NFRAMES))
        got = analyse(m)
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertEqual(got["taken"], 0)
        self.assertIn("человека в кадре нет", got["note"])

    def test_a_person_leaving_mid_clip_blocks_loops_across_the_gap(self):
        m = Material(loop_sequence(200), missing=range(60, 80))
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertGreater(
            got["rejected"].get("человека нет в кадре внутри петли", 0), 0)
        for lp in got["loops"]:
            with self.subTest(loop=(lp["i"], lp["j"])):
                self.assertTrue(lp["j"] < 60 or lp["i"] > 79,
                                "петля идёт через участок, где человека нет")

    def test_a_single_blink_of_the_detector_does_not_kill_the_loop(self):
        """Другая сторона того же порога: одиночный промах — не уход из кадра."""
        m = Material(loop_sequence(), missing=range(0, NFRAMES, 10))
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["rejected"], {})

    def test_presence_separates_a_blink_from_a_departure(self):
        poses = loop_sequence(100)
        for k in list(range(20, 40)) + [70]:
            poses[k] = None
        who = fl.presence(poses)
        self.assertEqual(who["gaps"], [(20, 39), (70, 70)])
        self.assertEqual(who["long_gaps"], [(20, 39)])
        self.assertEqual(who["left_at"], 20)


# ---------------------------------------------------------------------------
# 12. ДЛИННОЕ ВИДЕО: коарс-ту-файн и потолок
# ---------------------------------------------------------------------------

class LongMaterial(unittest.TestCase):
    def test_thinning_lands_on_a_true_period_and_costs_less(self):
        """Что уточнение обещает — и чего оно НЕ обещает.

        Обещает: вернуть настоящий период движения, доведя его границы на
        полной частоте, и снять меньше поз. НЕ обещает совпасть с полным
        проходом кадр в кадр — ИЗМЕРЕНО: полный проход берёт 0..44 (стык
        0.833), прорежённый 90..134 (стык 1.00). Прежняя редакция теста
        требовала точного совпадения и проходила только потому, что на
        идеально периодической фикстуре все выравнивания были ничьей.
        """
        m = Material(loop_sequence(200, tire=0.0002), blank=True)
        full = analyse(m, stride=1)
        thin = analyse(m, stride=5)
        self.assertEqual(full["outcome"], PASS)
        self.assertEqual(thin["outcome"], PASS)
        for name, rep in (("полный", full), ("прорежённый", thin)):
            best = rep["loops"][0]
            with self.subTest(scan=name):
                self.assertEqual((best["j"] - best["i"]) % PERIOD, 0,
                                 f"{name} проход взял не период движения")
        self.assertLess(thin["pose_frames"], full["pose_frames"])
        # Оценки двух проходов сравнимы ПРИБЛИЗИТЕЛЬНО: тонкая нормирована по
        # медианам внутри окон, полная — по медианам всего клипа, и это разные
        # числа. Требовать точного порядка между ними значило бы требовать
        # однородности там, где её нет; требовать близости — можно.
        self.assertLess(
            abs(thin["loops"][0]["score"] - full["loops"][0]["score"])
            / max(full["loops"][0]["score"], 1e-9), 0.2,
            "тонкая оценка уехала от полной больше чем на пятую часть — "
            "значит единицы измерения разъехались всерьёз")
        self.assertEqual(thin["loops"][0]["coarse"]["i"] % 5, 0)

    def test_thinning_breaks_at_nyquist_and_here_is_where(self):
        """ИЗМЕРЕНО, отрицательный результат с числами (И6).

        Петля периода 44 кадра при шаге прореживания:
            5, 10, 20  — находится;
            21, 23, 25 — НЕ находится (21 даёт преимущество 1.00);
        то есть предел ровно найквистовский: шаг обязан быть меньше половины
        периода движения. При шаге 5 это движения быстрее трёх циклов в
        секунду (период короче 10 кадров при 30 к/с) — тряска и быстрые махи.
        """
        m = Material(loop_sequence(200), blank=True)
        for stride in (5, 10, 20):
            with self.subTest(stride=stride, expect=PASS):
                got = analyse(m, stride=stride)
                self.assertEqual(got["outcome"], PASS)
        for stride in (21, 23, 25):
            with self.subTest(stride=stride, expect=FAIL):
                got = analyse(m, stride=stride)
                self.assertEqual(got["outcome"], FAIL)

    def test_a_short_clip_is_scanned_at_full_rate(self):
        m = Material(loop_sequence(), blank=True)
        got = analyse(m)
        self.assertEqual(got["scan"], fl.SCAN_FULL)
        self.assertEqual(got["stride"], 1)

    def test_a_long_clip_thins_itself_and_says_so(self):
        m = Material(loop_sequence(950), blank=True)
        got = analyse(m)
        self.assertEqual(got["scan"], fl.SCAN_COARSE)
        self.assertEqual(got["stride"], 5)
        self.assertEqual(got["frames"], 950)
        # 190 опрошенных против 950: снятие поз подешевело ровно в пять раз,
        # плюс окна уточнения.
        self.assertLess(got["pose_frames"], 950 // 2)
        self.assertGreaterEqual(got["pose_frames"], 190)

    def test_material_past_the_ceiling_is_refused_rather_than_awaited(self):
        m = Material(loop_sequence(), blank=True)
        got = analyse(m,
                            max_frames=50)
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertEqual(got["scan"], fl.SCAN_TOO_LONG)
        self.assertIn("нарежьте", got["note"].lower())
        ok = analyse(m,
                           max_frames=200)
        self.assertEqual(ok["outcome"], PASS)

    def test_the_default_ceiling_is_twenty_minutes(self):
        """Умолчание закреплено литералом: 36000 кадров при 30 к/с."""
        self.assertEqual(fl.MAX_FRAMES, 36000)
        self.assertEqual(fl.MAX_FRAMES / fork_comfy.WRAP_FPS / 60, 20)


# ---------------------------------------------------------------------------
# 13. РОВНО ПЯТЬ НА ВЫХОДЕ, И ОНИ РАЗНЫЕ
# ---------------------------------------------------------------------------

def many_exercises(count, *, each=50):
    """РАЗНЫЕ упражнения: разная конфигурация тела, а не разная амплитуда.

    Первая редакция этой фикстуры меняла только амплитуду маха, и подавление
    дублей схлопнуло её — правильно схлопнуло: ИЗМЕРЕНО, соседние «упражнения»
    расходились на 8.3 типичных шага при пороге 12, то есть это было одно
    движение с разным размахом. Здесь у каждого своя конфигурация: что
    движется (руки, ноги, вертикаль), с каким размахом и в какой посадке —
    не ближе 28.8 типичных шагов друг от друга при пороге 24.

    Число пришлось поднимать дважды, и оба раза замером: сначала фикстура
    отличалась только амплитудой (8.3 шага — по мере прибора это одно движение
    с разным размахом, и он был прав), потом посадкой на 14..30 шагов, что по
    шкале боевого ролика (55 у разных упражнений) всё ещё «варианты одного».
    """
    plans = [("arms", 0.16, 0.0), ("legs", 0.20, 0.42), ("line", 0.20, -0.40),
             ("arms", 0.05, 0.85), ("legs", 0.06, -0.80), ("line", 0.09, 1.25)]
    anchor = ("l_hip", "r_hip", "l_shoulder", "r_shoulder")
    out = []
    for e in range(count):
        mode, amp, tilt = plans[e]
        for t in range(each):
            sk = skeleton((e * 7 + t) / PERIOD, mode=mode, amp=amp)
            out.append({k: (x, y if k in anchor else y + tilt, v)
                        for k, (x, y, v) in sk.items()})
    return out


class FiveOnTheOutput(unittest.TestCase):
    def test_five_exercises_give_five_different_loops(self):
        m = Material(many_exercises(5), blank=True)
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(len(got["loops"]), 5)
        starts = sorted(lp["i"] // 50 for lp in got["loops"])
        self.assertEqual(starts, [0, 1, 2, 3, 4],
                         "пять петель обязаны прийти из пяти разных упражнений, "
                         "а не пять раз из одного")

    def test_a_sixth_exercise_does_not_make_a_sixth_line(self):
        m = Material(many_exercises(6), blank=True)
        got = analyse(m)
        self.assertEqual(len(got["loops"]), 5)
        self.assertEqual(got["asked"], 5)

    def test_a_pendulum_is_one_movement_and_not_three_cards(self):
        """Маятник — ОДНО упражнение, сколько бы раз он ни повторился.

        Прежняя редакция теста принимала три петли из одного маятника: сито
        диапазонов их пропускало, потому что кадры 0..44, 23..67 и 46..90
        перекрываются меньше чем наполовину. Сито содержания видит, что
        движение одно, и оставляет одну карточку.
        """
        m = Material(loop_sequence(), blank=True)
        got = analyse(m)
        self.assertEqual(len(got["loops"]), 1, got["note"])
        self.assertGreater(got["dropped_duplicate"], 0)
        self.assertIn("РАЗНЫХ ДВИЖЕНИЙ МЕНЬШЕ ЗАКАЗАННЫХ 5", got["note"])
        self.assertIn(f"схлопнуто как повтор того же движения "
                      f"{got['dropped_duplicate']}", got["note"])


# ---------------------------------------------------------------------------
# 14. ТРЕТЬЯ ОСЬ: ПИКСЕЛИ. Заведена после того, как ошиблись первые две
# ---------------------------------------------------------------------------

class PixelAxis(unittest.TestCase):
    def _sim(self, pixel):
        return {"pose": {(0, 44): 0.0}, "flow": {(0, 44): 0.0},
                "pixel": {(0, 44): pixel}, "joints": {(0, 44): 12},
                "lengths": [45], "pairs": 1, "measured": 1, "unmeasurable": 0}

    def test_a_perfect_pose_with_a_jumped_picture_ranks_badly(self):
        """Поза и направление сошлись идеально, а картинка прыгнула вчетверо
        против обычного перехода: 0.4/0.1 = 4.0, и это и есть оценка."""
        got = fl.score_pairs(self._sim(0.4), 0.05, pix_step=0.1)
        self.assertEqual(got[0]["score"], 4.0)
        self.assertEqual(got[0]["seam_pixel"], 4.0)

    def test_the_pixel_axis_is_silent_when_the_picture_matches(self):
        got = fl.score_pairs(self._sim(0.0), 0.05, pix_step=0.1)
        self.assertEqual(got[0]["score"], 0.0)

    def test_a_pair_without_pixels_is_not_judged_by_two_axes_out_of_three(self):
        got = fl.score_pairs(self._sim(None), 0.05, pix_step=0.1)
        self.assertEqual(got, [], "стык, у которого не измерена одна из трёх "
                                  "осей, — это «не смогли», а не «идеально»")

    def test_without_a_pixel_store_the_instrument_works_on_two_axes(self):
        got = fl.score_pairs(self._sim(None), 0.05)
        self.assertEqual(got[0]["score"], 0.0)
        self.assertIsNone(got[0]["seam_pixel"])

    def test_a_drifting_picture_kills_a_loop_the_pose_calls_perfect(self):
        """СИНТЕТИЧЕСКИЙ ДВОЙНИК НАХОДКИ НА `chain_frames`.

        Позы повторяются точно — по двум позным осям стык нулевой. А картинка
        медленно уезжает (свет, фон, предмет в руках — что угодно, чего скелет
        из двенадцати точек не знает). Прибор обязан это увидеть.
        """
        import numpy as np

        m = Material(loop_sequence(), blank=True)
        m.gray = lambda path: np.full((8, 8), 2.5 * int(Path(path).stem))
        got = analyse(m)
        self.assertEqual(got["outcome"], FAIL, got["note"])
        self.assertEqual(got["loops"], [])
        # А без пиксельной оси тот же материал дал бы идеальную петлю — это и
        # есть замер того, что ось добавляет, а не украшает.
        blind = analyse(m, pixel_weight=0.0)
        self.assertEqual(blind["outcome"], PASS)
        self.assertEqual(blind["loops"][0]["score"], 0.0)


# ---------------------------------------------------------------------------
# 15. ЧАСТОТА ИСТОЧНИКА — ВХОД, А НЕ УМОЛЧАНИЕ
# ---------------------------------------------------------------------------

class SourceFps(unittest.TestCase):
    def test_the_same_frames_at_24_and_30_are_not_the_same_seconds(self):
        """И5, обе стороны: один и тот же материал при разной частоте обязан
        дать РАЗНЫЕ секунды. Литералы: 45 кадров — 1.5 с при 30 и 1.88 при 24."""
        m = Material(loop_sequence(), blank=True)
        at30 = analyse(m, fps=30)
        at24 = analyse(m, fps=24)
        self.assertEqual(at30["fps"], 30)
        self.assertEqual(at24["fps"], 24)
        self.assertEqual(at30["fps_source"], fl.FPS_GIVEN)
        self.assertEqual(at30["loops"][0]["frames"], 45)
        self.assertEqual(at24["loops"][0]["frames"], 45)
        self.assertEqual(at30["loops"][0]["seconds"], 1.5)
        self.assertEqual(at24["loops"][0]["seconds"], 1.88)

    def test_the_repeat_plan_follows_the_source_rate(self):
        """Числа владельца: петля 53 кадра при 24 к/с. Пятикратный повтор даёт
        10.88 с и ВЫЛЕТАЕТ за потолок, а по нашим 30 к/с он выглядел годным."""
        self.assertEqual(
            [(r["repeats"], r["frames"], r["seconds"])
             for r in fl.repeat_plan(53, fps=24)],
            [(3, 157, 6.54), (4, 209, 8.71)])
        self.assertEqual(
            [(r["repeats"], r["frames"], r["seconds"])
             for r in fl.repeat_plan(53, fps=30)],
            [(3, 157, 5.23), (4, 209, 6.97), (5, 261, 8.7)])

    def test_a_directory_alone_has_no_frame_rate_and_says_so(self):
        """Третий исход (Р1): не «30 по умолчанию», а «неизвестна»."""
        m = Material(loop_sequence(), blank=True)
        got = fl.find_loops(m.dir, reader=m.reader, gray=m.gray,
                            head=m.head, gif=False)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertIsNone(got["fps"])
        self.assertEqual(got["fps_source"], fl.FPS_UNKNOWN)
        self.assertTrue(got["loops"])
        self.assertIsNone(got["loops"][0]["seconds"])
        self.assertEqual(got["loops"][0]["repeats"], [])
        txt = fl.table(got)
        self.assertIn("частота неизвестна", txt)
        self.assertIn("45", txt, "кадры печатаются всегда: они измерены")
        self.assertIn("неизвестна", [s["note"] for s in got["steps"]
                                     if s["step"] == "частота"][0])

    def test_a_video_file_tells_its_own_frame_rate(self):
        m = Material(loop_sequence(), blank=True)
        movie = m.dir.parent / "driving.mp4"
        movie.write_text("не настоящее видео: раскодировщик подменён",
                         encoding="utf-8")
        seen = {}

        def decode(path, out_dir, **kw):
            seen["path"] = path
            return {"outcome": PASS, "paths": [str(p) for p in m.paths()],
                    "fps_in": 24, "fps_out": 24, "note": "фикстура"}

        got = fl.find_loops(movie, reader=m.reader, gray=m.gray,
                            head=m.head, gif=False, decode=decode)
        self.assertEqual(got["fps"], 24)
        self.assertEqual(got["fps_source"], fl.FPS_PROBED)
        self.assertEqual(got["loops"][0]["seconds"], 1.88)
        self.assertEqual(seen["path"], str(movie))

    def test_a_hand_given_rate_wins_over_the_file(self):
        """Частота, названная человеком, не перебивается файлом молча."""
        m = Material(loop_sequence(), blank=True)
        movie = m.dir.parent / "driving2.mp4"
        movie.write_text("фикстура", encoding="utf-8")

        def decode(path, out_dir, **kw):
            return {"outcome": PASS, "paths": [str(p) for p in m.paths()],
                    "fps_in": 24, "fps_out": 24, "note": "фикстура"}

        got = fl.find_loops(movie, reader=m.reader, gray=m.gray,
                            head=m.head, gif=False, decode=decode, fps=30)
        self.assertEqual(got["fps"], 30)
        self.assertEqual(got["fps_source"], fl.FPS_GIVEN)


# ---------------------------------------------------------------------------
# 16. СКОЛЬКО СУСТАВОВ УЧАСТВОВАЛО — ЧАСТЬ ОТВЕТА
# ---------------------------------------------------------------------------

class JointCoverage(unittest.TestCase):
    def test_the_loop_reports_how_many_joints_were_compared(self):
        """На драйвинге правое запястье видно на 46 кадрах из 96, и стык лучшей
        петли посчитан по 8 суставам из 12. Молчать об этом нельзя."""
        blind = ("l_hip", "r_hip", "l_shoulder", "r_shoulder", "l_knee",
                 "r_knee", "l_ankle", "r_ankle", "l_elbow", "r_elbow")
        m = Material([skeleton(t / PERIOD, seen=blind) for t in range(NFRAMES)],
                     blank=True)
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["loops"][0]["joints"], 10)
        txt = fl.table(got)
        self.assertIn("суст", txt)
        self.assertIn("сколько суставов из 12", txt)

    def test_a_fully_visible_body_reports_all_twelve(self):
        m = Material(loop_sequence(), blank=True)
        got = analyse(m)
        self.assertEqual(got["loops"][0]["joints"], 12)
        self.assertNotIn("сколько суставов из 12", fl.table(got),
                         "пояснение печатается только когда есть что пояснять")


# ---------------------------------------------------------------------------
# 17. ДВА СИТА, И ОНИ ЛОВЯТ РАЗНОЕ
# ---------------------------------------------------------------------------

def two_places_one_movement(each=120):
    """Упражнение A, потом B, потом СНОВА A — как в боевом ролике.

    Кадры первого и третьего блока не пересекаются вовсе, а движение одно и то
    же: ровно тот случай, который сито диапазонов не видит в принципе.
    """
    out = []
    for mode, amp, tilt in (("arms", 0.10, 0.0), ("legs", 0.20, 0.6),
                            ("arms", 0.10, 0.0)):
        for t in range(each):
            out.append(seated(skeleton(t / PERIOD, mode=mode, amp=amp), tilt))
    return out


def same_start_different_moves(each=120):
    """Два упражнения, начинающиеся из ОДНОЙ позы и расходящиеся дальше.

    Вертикальный мах: в фазе 0 смещение нулевое, поэтому стартовые скелеты
    совпадают до последней цифры, а размах отличается вшестеро (34.8 типичных
    шага друг от друга при пороге 24). Сверка «только по
    началу» назвала бы это одним движением.
    """
    return ([skeleton(t / PERIOD, mode="line", amp=0.05) for t in range(each)]
            + [skeleton(t / PERIOD, mode="line", amp=0.30) for t in range(each)])


class TwoSieves(unittest.TestCase):
    def test_the_signature_samples_the_loop_at_a_fixed_rate(self):
        """Литералы: петля 0..44 описывается каждым пятым кадром.

        Частота одна на все петли — иначе длинная петля описывается реже
        короткой, и одно движение раскладывается на два (поймано прогоном).
        """
        asked = []

        def state_at(frame):
            asked.append(frame)
            return norm(skeleton(frame / PERIOD))

        sig = fl.loop_signature(state_at, 0, 44)
        self.assertEqual(asked, [0, 5, 10, 15, 20, 25, 30, 35, 40])
        self.assertEqual(len(sig), 9)
        # Петля вдвое длиннее описывается вдвое подробнее, а не той же горстью.
        asked.clear()
        fl.loop_signature(state_at, 0, 88)
        self.assertEqual(len(asked), 18)

    def test_the_signature_refuses_when_a_pose_is_missing(self):
        """Р1: сверять движения по половине подписи нельзя."""
        self.assertIsNone(fl.loop_signature(lambda f: None, 0, 44))
        self.assertIsNone(
            fl.loop_signature(lambda f: None if f == 15 else norm(skeleton(0.0)),
                              0, 44))

    def test_the_signature_gap_ignores_where_the_cycle_starts(self):
        """То же упражнение с другой точки цикла — то же упражнение."""
        at = lambda f: norm(skeleton(f / PERIOD))
        a = fl.loop_signature(at, 0, 44)
        b = fl.loop_signature(at, 11, 55)      # тот же маятник, сдвиг на четверть
        # Не ровно ноль: 44 кадра на 8 точек делятся с остатком, и точки
        # ложатся на соседние кадры движения. 0.036 длины торса — это один
        # межкадровый шаг, тогда как другое упражнение даёт впятеро больше.
        # 0.071 длины торса — это два типичных межкадровых шага: точки двух
        # подписей ложатся между собой, ровнее не бывает. Другое упражнение
        # даёт впятеро больше, и порог 12 типичных шагов лежит далеко от обоих.
        self.assertLess(fl.signature_gap(a, b), 0.1)
        # А другое упражнение остаётся другим и после сдвига.
        other = fl.loop_signature(lambda f: norm(skeleton(f / PERIOD, mode="legs",
                                                          amp=0.14)), 0, 44)
        self.assertGreater(fl.signature_gap(a, other), 0.3)

    def test_the_vectorised_gap_equals_pose_gap(self):
        """Е1: быстрая арифметика обязана давать то же, чем судит вся приёмка."""
        at = lambda f: norm(skeleton(f / PERIOD))
        a = [at(0), at(7)]
        b = [at(3), at(19)]
        by_hand = max(
            max(min(fl.pose_gap(x, y) for y in b) for x in a),
            max(min(fl.pose_gap(x, y) for x in a) for y in b))
        # places=5, а не 9: `pose_gap` округляет до шести знаков, и разница
        # ровно в этом округлении — 2e-7.
        self.assertAlmostEqual(fl.signature_gap(a, b), by_hand, places=5)

    def test_the_gap_is_the_worst_phase_not_the_average(self):
        """Совпадения в одной точке цикла недостаточно: берётся худшая фаза."""
        at = lambda f: norm(skeleton(f / PERIOD))
        a = fl.loop_signature(at, 0, 44)
        b = list(a)
        b[3] = norm(skeleton(0.37, mode="legs", amp=0.25))
        gap = fl.signature_gap(a, b)
        self.assertGreater(gap, 0.2, "одна разошедшаяся фаза обязана решать")

    def test_one_movement_in_two_places_collapses(self):
        """ГЛАВНЫЙ СЛУЧАЙ. Кадры не пересекаются, движение одно."""
        m = Material(two_places_one_movement(), blank=True)
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(len(got["loops"]), 2,
                         f"движений два, петель {len(got['loops'])}: "
                         f"{[(l['i'], l['j']) for l in got['loops']]}")
        self.assertGreater(got["dropped_duplicate"], 0)

    def test_two_movements_that_start_alike_are_told_apart(self):
        """Сверка ТОЛЬКО ПО НАЧАЛУ схлопнула бы их: стартовые позы совпадают
        до последней цифры, а дальше движения расходятся вшестеро по размаху.

        Проверяется на самой мере, а не на конвейере: конвейеру для этого
        пришлось бы дать клип, у которого половины отличаются скоростью, и он
        честно нашёл бы там монтажный рез — то есть тест мерил бы не то.
        """
        seq = same_start_different_moves()
        self.assertEqual(seq[0], seq[120],
                         "фикстура сломана: стартовые позы обязаны совпадать")
        st = fl.states(seq)
        at = lambda f: st[f] if 0 <= f < len(st) else None
        typical = fl.typical_step(st)["step"]
        self.assertEqual(fl.pose_gap(st[0], st[120]), 0.0,
                         "по началу движения неразличимы — в этом и дело")
        gap = fl.signature_gap(fl.loop_signature(at, 0, 44),
                               fl.loop_signature(at, 120, 164))
        self.assertGreater(gap / typical, fl.DUPLICATE_MAX_STEPS,
                           f"по всей петле обязаны различаться: {gap/typical:.1f} "
                           f"типичных шага против порога {fl.DUPLICATE_MAX_STEPS}")

    def test_the_same_movement_with_a_smaller_swing_is_still_one_movement(self):
        """Один и тот же мах с размахом 0.05 и 0.12 — одно упражнение.

        ИЗМЕРЕНО: 10.4 типичных шага между ними при пороге 24. Это середина
        между «сдвиг на кадр» (2..3) и «другое упражнение» (55), и на боевом
        ролике ровно такая пара (17.1 шага) глазами читалась как одно движение
        с разной амплитудой. Опустив порог, мы отдадим клиенту две карточки
        одного упражнения — ради этого теста порог и мутируется вниз.
        """
        seq = ([skeleton(t / PERIOD, mode="line", amp=0.05) for t in range(120)]
               + [skeleton(t / PERIOD, mode="line", amp=0.12) for t in range(120)])
        st = fl.states(seq)
        at = lambda f: st[f] if 0 <= f < len(st) else None
        gap = (fl.signature_gap(fl.loop_signature(at, 0, 44),
                                fl.loop_signature(at, 120, 164))
               / fl.typical_step(st)["step"])
        self.assertGreater(gap, 6.0, "фикстура должна быть ВЫШЕ нижней мутации")
        self.assertLess(gap, fl.DUPLICATE_MAX_STEPS)
        got = analyse(Material(seq, blank=True))
        self.assertEqual(len(got["loops"]), 1,
                         f"это одно движение, петель {len(got['loops'])}: "
                         f"{[(l['i'], l['j']) for l in got['loops']]}")

    def test_the_two_sieves_catch_different_things(self):
        """Каждое сито обязано ловить своё, и это видно по счётчикам."""
        shifted = Material(loop_sequence(), blank=True)
        one_move = Material(two_places_one_movement(), blank=True)
        a = analyse(shifted)
        b = analyse(one_move)
        self.assertGreater(a["dropped_overlap"], 0,
                           "сдвиги на кадр внутри одного места ловит сито "
                           "диапазонов")
        self.assertGreater(b["dropped_duplicate"], 0,
                           "повтор упражнения в другом месте клипа ловит только "
                           "сито содержания")

    def test_comparing_movements_asks_the_detector_nothing(self):
        """Т4 и П2: сверка идёт по УЖЕ СНЯТЫМ позам, новых опросов нет."""
        m = Material(two_places_one_movement(), blank=True)
        analyse(m)
        self.assertEqual(len(m.calls), 360,
                         "детектор позы обязан быть вызван ровно по разу на "
                         "кадр: сверка движений своих опросов не делает")


# ---------------------------------------------------------------------------
# 18. ОСЬ ГОЛОВЫ
# ---------------------------------------------------------------------------

class HeadAxis(unittest.TestCase):
    def test_a_head_that_returns_keeps_the_loop(self):
        m = Material(loop_sequence(), blank=True)
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["loops"][0]["head_state"], PASS)
        self.assertEqual(got["dropped_head"], 0)
        self.assertLess(got["loops"][0]["seam_head"], 1.0)

    def test_a_head_that_never_returns_drops_the_loop(self):
        """И5, другая сторона: голова уезжает — петля не выпускается.

        И бюджет попыток не превращается в лазейку: исчерпав его на работающей
        оси, прибор ОСТАНАВЛИВАЕТСЯ, а не выпускает двадцать первого с пометкой
        «не проверена».
        """
        m = Material(loop_sequence(), blank=True, head_mode="уезжает")
        got = analyse(m)
        self.assertGreater(got["dropped_head"], 0, got["note"])
        self.assertEqual(got["loops"], [], got["note"])
        self.assertEqual(got["head_tried"], fl.HEAD_MAX_TRIES)
        self.assertIn("БЮДЖЕТ ГОЛОВЫ ИСЧЕРПАН",
                      [s["note"] for s in got["steps"]
                       if s["step"] == "финалисты"][0])

    def test_a_head_jump_is_caught_where_the_pose_is_perfect(self):
        """Поза и картинка идеальны, голова прыгает — этого не видит ничто,
        кроме своей оси."""
        m = Material(loop_sequence(), blank=True, head_mode="рывок")
        got = analyse(m)
        for lp in got["loops"]:
            with self.subTest(loop=(lp["i"], lp["j"])):
                self.assertFalse(lp["i"] < NFRAMES // 2 <= lp["j"]
                                 and lp["head_state"] == PASS,
                                 "петля перешагнула рывок головы")

    def test_no_head_detector_marks_the_loop_instead_of_failing_it(self):
        """Р1: «не смогли посмотреть» не значит «плохо» — но и не молчит."""
        m = Material(loop_sequence(), blank=True, head_broken=True)
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertTrue(got["loops"])
        self.assertEqual(got["loops"][0]["head_state"], UNMEASURED)
        self.assertIn("DWPose", got["loops"][0]["head_note"])
        self.assertIn("ГОЛОВА НЕ ПРОВЕРЕНА", fl.table(got))
        self.assertIsNone(got["head_step"])

    def test_a_face_not_seen_marks_the_loop_too_but_differently(self):
        m = Material(loop_sequence(), blank=True, head_blind=range(0, 60))
        got = analyse(m)
        self.assertTrue(got["loops"])
        self.assertEqual(got["loops"][0]["head_state"], UNMEASURED)
        self.assertIn("лица не видно", got["loops"][0]["head_note"])

    def test_a_head_that_almost_returns_is_kept_and_not_crushed(self):
        """Вес оси головы — 1.0, а не «побольше, чтобы наверняка».

        Голова возвращается почти точно (сползание 0.01 px за кадр, стык 0.78
        типичного смещения). Такую петлю прибор обязан ОТДАТЬ: при весе 100 её
        стык превратился бы в 78 шагов и петля отвалилась бы вместе с годным
        материалом.
        """
        import numpy as np

        m = Material(loop_sequence(), blank=True)
        m.head = lambda path: {
            "head": (360.0,
                     500.0 + 6.0 * float(np.sin(2 * np.pi *
                                                (int(Path(path).stem) % PERIOD)
                                                / PERIOD))
                     + 0.01 * int(Path(path).stem)),
            "why": ""}
        got = analyse(m)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["dropped_head"], 0, got["note"])
        self.assertGreater(got["loops"][0]["seam_head"], 0.4)
        self.assertLess(got["loops"][0]["seam_head"], 1.5)

    def test_the_head_is_asked_only_about_finalists(self):
        """П2: 0.436 с/кадр — голова спрашивается о единицах, а не о тысячах."""
        m = Material(two_places_one_movement(), blank=True)
        got = analyse(m)
        self.assertLessEqual(got["head_tried"], fl.HEAD_MAX_TRIES)
        # Кадров головы: масштаб плюс по два на попытку, и это в разы меньше,
        # чем 360 кадров материала.
        self.assertLess(got["head_frames"], 120)
        self.assertGreater(got["head_frames"], 0)

    def test_the_scale_of_the_head_axis_is_measured_on_the_clip(self):
        m = Material(loop_sequence(), blank=True)
        got = fl.head_scale(m.paths(), reader=m.head)
        self.assertEqual(got["outcome"], PASS)
        # Литералы, а не `fl.HEAD_SCALE_PAIRS`: ожидаемое, импортированное из
        # проверяемого модуля, поедет вместе с ним и промолчит (Т2). Поймано
        # мутацией: подмена 40 на 5 пережила этот тест.
        self.assertEqual(got["measured"], 40)
        self.assertEqual(got["frames"], 80)

    def test_the_scale_says_when_it_cannot_be_measured(self):
        m = Material(loop_sequence(), blank=True, head_broken=True)
        got = fl.head_scale(m.paths(), reader=m.head)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["step"])
        self.assertIn("спросить нечем", got["reason"])


class NoHeavyImports(unittest.TestCase):
    def test_importing_the_module_does_not_pull_mediapipe(self):
        """Т4 обеспечивается устройством модуля, а не договорённостью."""
        import subprocess
        import sys

        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; import ball_reel.fork_looper as m; "
             "print('mediapipe' in sys.modules)"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[2]))
        self.assertEqual(out.stdout.strip(), "False", out.stderr[-400:])


# ---------------------------------------------------------------------------
# 19. ЦЕНА МОСТА. Стык больше не отбраковывает — он назначает цену
# ---------------------------------------------------------------------------

def priced(i, j, frames, floor, seam, outcome):
    """Петля с уже посчитанной ценой моста — ВХОД для `rank_loops`.

    Собирается литералами, а не вызовом `bridge_cost`: иначе порядок
    проверялся бы вместе с арифметикой цены, и ошибка в одной пряталась бы за
    другой.
    """
    return {"i": i, "j": j, "frames": j - i + 1,
            "bridge": {"outcome": outcome, "frames": frames, "floor": floor,
                       "seam": seam, "worst_axis": "голова",
                       "unmeasured": [] if outcome == PASS else ["голова"],
                       "measured": ["тело"], "reason": ""}}


class BridgePrice(unittest.TestCase):
    """Стык, переведённый в кадры подгонки, и три исхода у этого перевода."""

    def test_the_seam_becomes_frames_by_rounding_up(self):
        """«Стык 3.48 типичных шага» — это «не хватает 3.5 кадров обычного
        движения», то есть мост в 4 кадра. Округление ВВЕРХ, а не к ближайшему:
        мост в 3 кадра провёл бы переход быстрее обычного движения клипа, то
        есть дал бы рывок — ровно то, ради чего петля и отбиралась.
        """
        self.assertEqual(fl.bridge_frames(3.48), 4)
        self.assertEqual(fl.bridge_frames(3.01), 4)
        self.assertEqual(fl.bridge_frames(0.01), 1)

    def test_a_seam_that_is_a_whole_number_does_not_get_a_spare_frame(self):
        """Негативный контроль округления с другой стороны (И5): вверх — это
        не «плюс один всегда»."""
        self.assertEqual(fl.bridge_frames(3.0), 3)
        self.assertEqual(fl.bridge_frames(0.0), 0)

    def test_an_unmeasured_seam_has_no_price_and_that_is_not_zero(self):
        """ФОРМА ГЛАВНОГО ДЕФЕКТА: «не смогли», ведущее себя как ноль.

        Цена неизмеренного стыка не выдумывается ни нулём (тогда он поднимает
        кандидата в рейтинге — это и случилось на боевом ролике), ни
        бесконечностью (тогда петля молча выбрасывается). Её просто нет.
        """
        self.assertIsNone(fl.bridge_frames(None))

    def test_four_measured_axes_give_a_price_and_name_the_worst(self):
        got = fl.bridge_cost({"поза": 0.99, "поток": 0.91, "пиксели": 1.45,
                              "голова": 3.28})
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["frames"], 4)
        self.assertEqual(got["worst_axis"], "голова")
        self.assertEqual(got["unmeasured"], [])

    def test_a_missing_axis_is_a_third_outcome_and_gives_a_bound(self):
        """Р1: «не смогли» не сворачивается ни в «годно», ни в «не годно».

        Числа — с боевого ролика: петля 309..357, у которой лица не видно на
        первом кадре. По трём осям её мост был бы 3 кадра, и ровно этим она
        обошла в рейтинге петлю, чей мост честно посчитан.
        """
        got = fl.bridge_cost({"поза": 1.479, "поток": 2.141, "пиксели": 1.935,
                              "голова": None})
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIsNone(got["frames"], "цены у неизмеренного стыка нет")
        self.assertEqual(got["floor"], 3, "но нижняя граница есть, и она в кадрах")
        self.assertEqual(got["unmeasured"], ["голова"])

    def test_the_same_loop_with_the_head_measured_gets_a_real_price(self):
        """Негативный контроль с другой стороны (И5): та же петля, у которой
        ось головы ответила, получает цену, а не границу. Настоящий стык
        головы 309..357, померенный по соседним измеримым краям, — 11.74 шага,
        и это мост в 12 кадров против кажущихся трёх."""
        got = fl.bridge_cost({"поза": 1.479, "поток": 2.141, "пиксели": 1.935,
                              "голова": 11.74}, max_frames=99)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["frames"], 12)

    def test_nothing_measured_at_all_has_no_bound_either(self):
        got = fl.bridge_cost({"поза": None, "голова": None})
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIsNone(got["frames"])
        self.assertIsNone(got["floor"], "границу тоже не из чего вывести")

    def test_a_long_bridge_is_a_no_and_not_a_dearer_yes(self):
        """Третий исход у моста: слишком длинный — это «эту петлю не берём»."""
        got = fl.bridge_cost({"тело": 3.0, "голова": 12.4}, max_frames=8)
        self.assertEqual(got["outcome"], "не годно")
        self.assertEqual(got["frames"], 13)
        self.assertIn("12.40", got["reason"])

    def test_a_lower_bound_over_the_ceiling_is_already_a_no(self):
        """«Не смогли» не выкупает длинный мост.

        Недостающая ось может мост только УДЛИНИТЬ, никогда не укоротить.
        Значит кандидат, у которого уже нижняя граница выше потолка, — «не
        годно», и доизмерять его незачем. Это единственное место, где третий
        исход законно сворачивается во второй, и законно оно ровно потому, что
        направление доизмерения известно заранее.
        """
        got = fl.bridge_cost({"тело": 9.5, "голова": None}, max_frames=8)
        self.assertEqual(got["outcome"], "не годно")
        self.assertEqual(got["floor"], 10)
        self.assertIsNone(got["frames"], "посчитанной ценой это не стало")

    def test_the_ceiling_shipped_is_eight_frames(self):
        """Б2: отгружаемое значение сторожит тест НА САМО ЗНАЧЕНИЕ, литералом.

        Взято по плато развёртки на боевом ролике: ответ не меняется на
        отрезке 5..12, ниже 5 порог стоит на склоне, на 13 в выдачу
        возвращаются соседи той петли, которую владелец забраковал глазами.
        """
        self.assertEqual(fl.BRIDGE_MAX_FRAMES, 8)

    def test_the_ceiling_decides_at_exactly_eight_frames(self):
        """Т1 с обеих сторон: мост ровно в потолок — годно, на кадр длиннее —
        нет. Сдвинь потолок в любую сторону, и один из этих двух покраснеет."""
        self.assertEqual(fl.bridge_cost({"тело": 8.0})["outcome"], "годно")
        self.assertEqual(fl.bridge_cost({"тело": 8.01})["outcome"], "не годно")


class TwoQueues(unittest.TestCase):
    """Неизмеренная ось не имеет права удешевлять кандидата."""

    def test_an_unmeasured_candidate_never_ranks_above_a_measured_one(self):
        """ГЛАВНАЯ ПРАВКА, в чистом виде.

        У непроверенного мост НЕ БОЛЬШЕ нуля, у проверенного — четыре кадра.
        По цене непроверенный обошёл бы проверенного; по очередям — нет,
        потому что ноль у него не измерен, а предположен.
        """
        got = fl.rank_loops([priced(200, 244, None, 0, 0.0, UNMEASURED),
                             priced(100, 144, 4, 4, 3.28, PASS)])
        self.assertEqual([(l["i"], l["j"]) for l in got],
                         [(100, 144), (200, 244)])

    def test_inside_the_measured_queue_the_cheaper_bridge_wins(self):
        """Негативный контроль с другой стороны (И5): когда измерены все,
        порядок обязан быть ровно по цене моста, а не по номеру кадра."""
        got = fl.rank_loops([priced(10, 54, 9, 9, 8.1, PASS),
                             priced(20, 64, 2, 2, 1.4, PASS),
                             priced(30, 74, 5, 5, 4.6, PASS)])
        self.assertEqual([l["bridge"]["frames"] for l in got], [2, 5, 9])

    def test_unmeasured_candidates_are_ordered_among_themselves_by_the_bound(self):
        """Вторая очередь — тоже очередь, а не свалка."""
        got = fl.rank_loops([priced(10, 54, None, 6, 5.2, UNMEASURED),
                             priced(20, 64, None, 2, 1.1, UNMEASURED)])
        self.assertEqual([l["i"] for l in got], [20, 10])

    def test_a_blind_face_no_longer_takes_the_first_place(self):
        """То же самое, но целиком через прибор (Т5: развилка вызывается).

        Упражнение B замыкается ТОЧНО (мост 0 кадров), упражнение A слегка
        уезжает (мост 1 кадр). Пока лицо видно, первым обязано идти B — это
        негативный контроль «прибор обязан шевельнуться». Как только лица на
        кадрах B не видно, B уходит в конец списка, хотя его нижняя граница (0)
        по-прежнему дешевле честной цены A (1): именно так «не смогли»
        переставало быть нулём.
        """
        seq = two_exercises_second_is_perfect()
        seen = analyse(Material(seq, blank=True))
        self.assertEqual([(l["i"], l["j"]) for l in seen["loops"]],
                         [(48, 92), (0, 44)], seen["note"])
        self.assertEqual([l["bridge"]["frames"] for l in seen["loops"]], [0, 1])

        blind = analyse(Material(seq, blank=True,
                                 head_blind=range(48, NFRAMES)))
        self.assertEqual([(l["i"], l["j"]) for l in blind["loops"]],
                         [(0, 44), (48, 92)], blind["note"])
        last = blind["loops"][1]
        self.assertEqual(last["head_state"], UNMEASURED)
        self.assertIsNone(last["bridge"]["frames"])
        self.assertEqual(last["bridge"]["floor"], 0,
                         "нижняя граница у него ДЕШЕВЛЕ, и всё равно он второй")
        self.assertEqual(blind["loops"][0]["bridge"]["frames"], 1)

    def test_the_deferred_candidate_suppresses_nobody(self):
        """Отложенный не занимает места и не подавляет соседей по кадрам.

        Ровно этим он и уводил измеримого соседа с боевого ролика: петля
        309..357 пересекалась с 296..344 на 73% и вытесняла её, оставаясь при
        этом непроверенной.
        """
        seq = two_exercises_second_is_perfect()
        blind = analyse(Material(seq, blank=True, head_blind={48}))
        first = blind["loops"][0]
        self.assertEqual(first["head_state"], PASS,
                         "первым обязан идти измеренный, а не отложенный")
        self.assertEqual((first["i"], first["j"]), (49, 93),
                         "сосед отложенного на кадр вправо — лицо на нём видно")

    def test_the_table_marks_a_bound_so_it_cannot_be_read_as_a_price(self):
        """(в) — пометка без изменения порядка — это то, что подвело.

        Поэтому пометок теперь две, и обе В САМОЙ СТРОКЕ таблицы: «≥» у моста
        и «≤» у выигрыша. Строка под таблицей осталась, но одна она не
        спасала: клиент читает таблицу.
        """
        # Ослепляется УПРАЖНЕНИЕ A: у него стык ненулевой, значит есть и
        # выигрыш, который можно напечатать границей. У точного B выигрыш не
        # печатается вовсе — делить на нулевой стык нечем.
        blind = analyse(Material(two_exercises_second_is_perfect(), blank=True,
                                 head_blind=range(0, 48)))
        txt = fl.table(blind)
        bound = [r for r in txt.splitlines() if r.strip().startswith("2 ")][0]
        self.assertIn("≥1к", bound, txt)
        self.assertIn("≤", bound, txt)
        self.assertIn("мост", txt.splitlines()[0])

    def test_the_counters_say_how_many_bridges_were_priced_and_how_many_not(self):
        """Р2: ноль отвергнутых при нуле посчитанных мостов — не успех."""
        blind = analyse(Material(two_exercises_second_is_perfect(), blank=True,
                                 head_blind=range(48, NFRAMES)))
        self.assertEqual(blind["bridge_measured"], 1)
        self.assertEqual(blind["head_unchecked"], 1)
        self.assertEqual(blind["dropped_bridge"], 0)
        note = [s["note"] for s in blind["steps"] if s["step"] == "финалисты"][0]
        self.assertIn("МОСТЫ: посчитано 1", note)
        self.assertIn("не смогли посчитать 1", note)

    def test_the_bridge_is_measured_by_the_local_step_not_the_clip_median(self):
        """Длина моста — величина АБСОЛЮТНАЯ, и знаменатель у неё локальный.

        РАСХОЖДЕНИЕ ПРИБОРОВ, из-за которого это появилось: смена по кадрам
        моста померила те же два стыка боевого ролика своим прибором (133
        точки, пиксели) и получила «мост 1 кадр» там, где у нас выходило 4.
        Причина — знаменатель: скорость движения по клипу неравномерна в 5.15
        раза, а клиповая медиана одна на всех. После перехода на локальный шаг
        приборы сошлись: 0.88 против 0.92 на стыке 344->296.

        Здесь то же самое на фикстуре, где скорость заведомо разная: движение
        МЕДЛЕННОЕ, а медиану шага задаёт быстрое соседнее. По клиповой медиане
        стык вышел бы 0.123 шага и мост в 1 кадр; по локальной — 1.028 шага и
        мост в 2 кадра, то есть клиповая медиана занижает его вдвое.

        Порядок петель при этом остаётся на ГЛОБАЛЬНОЙ мере: она одинакова для
        всех кандидатов и потому никого не искажает относительно других
        (`score`, «выигрыш»), а мост — то, что клиент получит кадрами.
        """
        got = analyse(Material(two_speeds(), blank=True))
        loop = got["loops"][0]
        self.assertEqual(loop["bridge"]["frames"], 2)
        self.assertEqual(loop["bridge_seams"]["поза"], 1.028)
        self.assertEqual(loop["seam_pose"], 0.123,
                         "тот же стык по клиповой медиане")
        self.assertEqual(math.ceil(max(loop["seam_pose"], loop["seam_flow"])), 1,
                         "по клиповой медиане мост вышел бы вдвое короче")

    def test_the_head_axis_too_is_divided_by_its_own_local_step(self):
        """У КАЖДОЙ ОСИ ЗНАМЕНАТЕЛЬ СВОЙ, и у головы он идёт в другую сторону.

        Голова здесь качается медленно в первом упражнении и быстро во втором,
        а петля выходит во втором. По клиповой медиане её стык — 11.381 шага,
        то есть мост в 12 кадров и отказ по потолку; по локальному шагу самой
        петли — 0.852 шага, то есть мост в 1 кадр. Разница в 13 раз, и она не
        придумана: на боевом ролике та же ось шла в ОБРАТНУЮ сторону от позы
        (внутри «свечи» голова медленнее клипа), поэтому общий знаменатель на
        две оси был бы выдуманным.
        """
        m = Material(two_exercises_second_is_perfect(), blank=True)
        m.head = lambda path: {
            "head": (360.0,
                     500.0
                     + (1.0 if int(Path(path).stem) < NFRAMES // 2 else 24.0)
                     * math.sin(2 * math.pi * (int(Path(path).stem) % PERIOD)
                                / PERIOD)
                     + 0.05 * int(Path(path).stem)),
            "why": ""}
        got = analyse(m)
        loop = got["loops"][0]
        self.assertEqual((loop["i"], loop["j"]), (48, 92), got["note"])
        self.assertEqual(loop["seam_head"], 0.852, "локальный шаг головы")
        self.assertEqual(loop["seam_head_clip"], 11.381, "клиповый шаг головы")
        self.assertEqual(loop["bridge"]["frames"], 1,
                         "по клиповому мосту вышло бы 12 кадров и отказ")

    def test_the_local_head_scale_is_asked_of_eight_pairs(self):
        """Б2: у отгружаемого значения свой тест, литералом и отдельно.

        Число выбрано по полке (см. константу): 8 и 16 пар дают одну и ту же
        медиану, 2 и 4 — другую. Поведенческого теста, различающего 8 и 16, не
        существует ПО ЗАМЕРУ, и это не дыра, а причина, по которой взято 8.
        """
        self.assertEqual(fl.HEAD_LOCAL_PAIRS, 8)
        m = Material(loop_sequence(), blank=True)
        got = fl.head_scale(m.paths()[0:45], reader=m.head, pairs=8)
        self.assertEqual(got["measured"], 8)
        self.assertEqual(got["frames"], 16)

    def test_a_bridge_priced_without_the_pixel_axis_says_so(self):
        """И7: та же форма дефекта, найденная грепом по модулю, — но здесь она
        закрывается пометкой, а не очередью, и это разные случаи.

        Пиксельная ось выключается У ВСЕГО КЛИПА сразу, а не у одной петли:
        занижены все кандидаты одинаково, порядок между ними не едет. Но цена
        моста при этом посчитана по трём осям из четырёх и может быть только
        нижней границей — и это обязано быть сказано.
        """
        got = analyse(Material(loop_sequence(), blank=True))
        self.assertTrue(got["loops"][0]["pixel_axis_off"])
        self.assertIn("пиксельная ось клипа НЕ ИЗМЕРЕНА", fl.table(got))

    def test_the_pixel_note_is_silent_on_a_clip_where_that_axis_worked(self):
        """Негативный контроль ЧЕРЕЗ ПРИБОР (И5): пометка обязана молчать.

        Фикстурам этого модуля пиксельная ось не светит: их серый кадр выведен
        из симметричного скелета, суммы гасятся, и типичный переход выходит
        нулевым. Поэтому картинка здесь подаётся ПЕРИОДИЧЕСКАЯ: масштаб оси
        измеряется (переход ненулевой), а на стыке петли она возвращается в ту
        же точку, что и поза. Без такого входа сторож проверялся бы только с
        одной стороны — и мутация «пометка кричит всегда» его пережила бы
        (поймано мутацией, а не рассуждением).
        """
        import numpy as np

        m = Material(loop_sequence(), blank=True)
        m.gray = lambda path: np.full(
            (8, 8), 120.0 + 30.0 * math.sin(2 * math.pi
                                            * (int(Path(path).stem) % PERIOD)
                                            / PERIOD), dtype="float64")
        got = analyse(m)
        self.assertGreater(got["pixel_step"], 0.0,
                           "фикстура обязана включить пиксельную ось")
        self.assertFalse(got["loops"][0]["pixel_axis_off"])
        self.assertNotIn("пиксельная ось клипа НЕ ИЗМЕРЕНА", fl.table(got))

    def test_the_pixel_note_is_silent_when_that_axis_worked(self):
        """Негативный контроль (И5): сторож обязан молчать на входе, где всё
        измерено. Отчёт собран здесь литералами — `table` чистая функция, и
        проверять её проще подставленным отчётом, чем прогоном."""
        quiet = {"fps": 30, "dropped_bridge": 0, "loops": [
            {"rank": 1, "i": 0, "j": 44, "frames": 45, "seconds": 1.5,
             "joints": 12, "score": 0.5, "seam_pose": 0.5, "seam_flow": 0.2,
             "seam_pixel": 0.4, "seam_head": 0.3, "advantage": 3.0,
             "repeats": [], "gif": None, "head_state": PASS,
             "pixel_axis_off": False,
             "bridge": {"outcome": PASS, "frames": 1, "floor": 1,
                        "seam": 0.5, "unmeasured": []}}]}
        txt = fl.table(quiet)
        self.assertNotIn("пиксельная ось", txt)
        self.assertNotIn("ЦЕНА МОСТА НЕ ПОСЧИТАНА", txt)
        self.assertIn("1к", txt)

    def test_a_head_that_never_returns_is_dropped_by_the_bridge_now(self):
        """Отбраковка по голове осталась, но её выносит ЦЕНА МОСТА, а не
        сравнение с типичной оценкой клипа, посчитанной без головы."""
        got = analyse(Material(loop_sequence(), blank=True, head_mode="уезжает"))
        self.assertEqual(got["loops"], [], got["note"])
        self.assertGreater(got["dropped_bridge"], 0)
        self.assertEqual(got["dropped_bridge"], got["dropped_head"],
                         "длинными эти мосты сделала именно голова")


if __name__ == "__main__":
    unittest.main()
