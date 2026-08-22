"""Финальная сборка: сторожа кропа, возврата звука и трёх исходов.

НИ ОДИН ТЕСТ ЗДЕСЬ НЕ ЗАПУСКАЕТ ffmpeg И НЕ ХОДИТ В СЕТЬ (Т4). Внешний мир
подменяется двумя параметрами — `prober` и `runner`; на диске появляются только
пустышки во временном каталоге, потому что `fork_video.probe` по устройству
сначала проверяет существование файла.

ОТВЕТЫ ffprobe В ЭТОМ ФАЙЛЕ — НАСТОЯЩИЕ, сняты прогоном на настоящих файлах
проекта (`assets/driving_arms.mp4`, `work/arm_out_control_plain.mp4`,
`work/finish_demo.mp4`) и вставлены ЛИТЕРАЛАМИ (Т2), обрезанные до полей,
которые читает разбор. Ожидаемые числа тоже литералы: `540`, `960`, `210`,
`43.75`, `1`, `100` — импортировать их из проверяемого модуля значило бы
проверять, что модуль согласен сам с собой.

КАРТА ДВИЖЕНИЯ `REAL_COLUMNS` — ТОЖЕ ЗАМЕР, а не выдумка: поколоночная
межкадровая разница настоящего выхода Kling, 20 кадров через каждые 5, сжатая
до 48 колонок. Она нужна именно как вход, на котором прибор смещения обязан
сказать «не смогли выбрать»: на живом материале человек стоит по центру.

ФИКСТУРЫ С ОБОИХ КРАЁВ И ИЗ СЕРЕДИНЫ (Т3): квадрат 960x960, уже готовые 9:16
(720x1280), кадр УЖЕ заказанного (540x1200), нечётный 961x961, вырожденный
2x2, частоты 24/30/60, расхождения -2/-1/0/+1/+3 кадра, карта движения ровная /
слева / справа / нулевая.

НЕГАТИВНЫЙ КОНТРОЛЬ У КАЖДОГО ПРИБОРА (И5): на каждый вход, где прибор обязан
сказать «не годно», рядом стоит соседний, где он обязан пропустить, и оба
проверяются равносильно; вход «не смогли» разведён с обоими.

СЛОВО «ГОДНО» — ПОДСТРОКА СЛОВА «НЕ ГОДНО», поэтому исход здесь сверяется
только `assertEqual`, и `assertIn("годно", ...)` в файле нет ни одного.
"""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from ball_reel import fork_finish as ff
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED

# ---------------------------------------------------------------------------
# Настоящие ответы ffprobe, снятые прогоном. Обрезаны до читаемых полей.
# ---------------------------------------------------------------------------

DRIVING_JSON = """{
 "streams": [
  {"index": 0, "codec_name": "h264", "codec_type": "video", "width": 720,
   "height": 1280, "r_frame_rate": "30/1", "avg_frame_rate": "30/1",
   "duration": "12.433333", "nb_frames": "373"},
  {"index": 1, "codec_name": "aac", "codec_type": "audio",
   "r_frame_rate": "0/0", "avg_frame_rate": "0/0", "duration": "12.492245",
   "nb_frames": "269"}
 ],
 "format": {"duration": "12.492245", "size": "2825859"}
}"""

KLING_JSON = """{
 "streams": [
  {"index": 0, "codec_name": "h264", "codec_type": "video", "width": 960,
   "height": 960, "r_frame_rate": "30/1", "avg_frame_rate": "30/1",
   "duration": "3.300000", "nb_frames": "99"}
 ],
 "format": {"duration": "3.300000", "size": "5207169"}
}"""

RESULT_JSON = """{
 "streams": [
  {"index": 0, "codec_name": "h264", "codec_type": "video", "width": 540,
   "height": 960, "r_frame_rate": "30/1", "avg_frame_rate": "30/1",
   "duration": "3.300000", "nb_frames": "99"},
  {"index": 1, "codec_name": "aac", "codec_type": "audio",
   "r_frame_rate": "0/0", "avg_frame_rate": "0/0", "duration": "3.300000",
   "nb_frames": "144"}
 ],
 "format": {"duration": "3.300000", "size": "1735695"}
}"""

#: Тот же результат, но БЕЗ звуковой дорожки — так выглядит файл, собранный
#: по ветке отказа от склейки.
RESULT_SILENT_JSON = RESULT_JSON.replace(
    """,
  {"index": 1, "codec_name": "aac", "codec_type": "audio",
   "r_frame_rate": "0/0", "avg_frame_rate": "0/0", "duration": "3.300000",
   "nb_frames": "144"}""", "")

#: Драйвинг без звука — не выдумка, а частый настоящий случай: в модель уезжает
#: именно такой файл, и однажды его подадут сюда вместо исходника.
DRIVING_SILENT_JSON = DRIVING_JSON.replace(
    """,
  {"index": 1, "codec_name": "aac", "codec_type": "audio",
   "r_frame_rate": "0/0", "avg_frame_rate": "0/0", "duration": "12.492245",
   "nb_frames": "269"}""", "")

#: ИЗМЕРЕНО на `work/arm_out_control_plain.mp4`: средняя межкадровая разница по
#: колонкам, 20 кадров, сжато до 48 колонок. Лучшее окно выигрывает у
#: центрального 1.0008x — то есть человек стоит по центру.
REAL_COLUMNS = [5.33, 5.1, 4.87, 5.08, 6.53, 9.06, 10.25, 18.58, 26.34, 27.9,
                32.57, 33.45, 33.33, 32.42, 38.37, 39.81, 35.84, 29.22, 34.86,
                38.12, 38.49, 36.32, 31.91, 30.63, 30.67, 29.46, 30.35, 30.77,
                34.37, 36.36, 35.47, 33.59, 32.75, 31.82, 27.61, 25.91, 27.57,
                28.73, 27.18, 22.66, 16.04, 12.96, 13.44, 13.14, 7.93, 4.6,
                2.88, 1.45]


def prober_of(mapping):
    """Подменный ffprobe: путь -> текст ответа. Ни одного процесса.

    Значение `None` означает «инструмента нет» — то есть ветку «не смогли»,
    которую нельзя перепутать с «файл плохой».
    """
    def _prober(path):
        answer = mapping.get(Path(path).name)
        if answer is None:
            return {"ran": False, "code": None, "out": "", "err": "",
                    "why": "ffprobe не найден: спросить нечем"}
        if isinstance(answer, int):
            return {"ran": True, "code": answer, "out": "{\n\n}\n",
                    "err": "moov atom not found", "why": ""}
        return {"ran": True, "code": 0, "out": answer, "err": "", "why": ""}
    return _prober


class Runner:
    """Подменный ffmpeg. Запоминает argv, отвечает заказанным исходом."""

    def __init__(self, *, ran=True, code=0, err=""):
        self.ran, self.code, self.err, self.calls = ran, code, err, []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if not self.ran:
            return {"ran": False, "code": None, "out": "", "err": "",
                    "why": "ffmpeg не найден: собрать нечем"}
        return {"ran": True, "code": self.code, "out": "", "err": self.err,
                "why": ""}


def _files(*names):
    """Пустышки на диске: `fork_video.probe` обязан видеть, что файл есть."""
    tmp = Path(tempfile.mkdtemp(prefix="fork_finish_"))
    out = []
    for n in names:
        p = tmp / n
        p.write_bytes(b"\x00" * 64)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# A. КРОП
# ---------------------------------------------------------------------------

class CropIsCountedAndNotGuessed(unittest.TestCase):
    def test_the_square_from_kling_becomes_nine_by_sixteen(self):
        # Настоящий случай: 960x960 -> 540x960. Все числа литералами.
        g = ff.crop_geometry(960, 960)
        self.assertEqual(g["outcome"], PASS)
        self.assertEqual((g["w"], g["h"]), (540, 960))
        self.assertEqual((g["x"], g["y"]), (210, 0))

    def test_the_lost_area_is_the_number_and_not_an_impression(self):
        # 540*960 = 518400 из 960*960 = 921600 -> остаётся 56.25%, теряется
        # 43.75%. Арифметика проверяется здесь ОТДЕЛЬНО от модуля.
        self.assertEqual(540 * 960, 518400)
        self.assertEqual(960 * 960, 921600)
        self.assertEqual(round(100 * 518400 / 921600, 2), 56.25)
        g = ff.crop_geometry(960, 960)
        self.assertEqual(g["lost_percent"], 43.75)
        self.assertEqual(g["kept_percent"], 56.25)
        self.assertEqual(round(g["lost_percent"] + g["kept_percent"], 2), 100.0)

    def test_a_frame_already_nine_by_sixteen_loses_nothing(self):
        # Негативный контроль к предыдущему: прибор обязан не только резать,
        # но и НЕ резать. 720x1280 — это ровно 9:16.
        g = ff.crop_geometry(720, 1280)
        self.assertEqual(g["outcome"], PASS)
        self.assertEqual((g["w"], g["h"], g["x"], g["y"]), (720, 1280, 0, 0))
        self.assertEqual(g["lost_percent"], 0.0)

    def test_a_frame_taller_than_asked_is_cut_along_the_other_axis(self):
        # Третья фикстура: исходник УЖЕ заказанного (540x1200 — это 9:20).
        # Резать по ширине тут нечего, режется высота.
        g = ff.crop_geometry(540, 1200)
        self.assertEqual(g["outcome"], PASS)
        self.assertEqual((g["w"], g["h"]), (540, 960))
        self.assertEqual((g["x"], g["y"]), (0, 120))

    def test_odd_sides_are_snapped_down_because_x264_refuses_them(self):
        # 961x961: и окно, и смещение обязаны выйти чётными, иначе кодировать
        # нечем, а не «чуть хуже».
        g = ff.crop_geometry(961, 961)
        self.assertEqual(g["outcome"], PASS)
        for name in ("w", "h", "x", "y"):
            with self.subTest(field=name):
                self.assertEqual(g[name] % 2, 0)
        self.assertEqual((g["w"], g["h"]), (540, 960))

    def test_the_window_never_leaves_the_frame_at_either_bias(self):
        for width, height, bias in ((960, 960, -1.0), (960, 960, 0.0),
                                    (960, 960, 1.0), (961, 961, 1.0),
                                    (540, 1200, 1.0), (540, 1200, -1.0)):
            with self.subTest(size=(width, height), bias=bias):
                g = ff.crop_geometry(width, height, bias=bias)
                self.assertEqual(g["outcome"], PASS)
                self.assertLessEqual(g["x"] + g["w"], width)
                self.assertLessEqual(g["y"] + g["h"], height)
                self.assertGreaterEqual(min(g["x"], g["y"]), 0)

    def test_bias_moves_the_window_to_the_edges_and_to_the_middle(self):
        self.assertEqual(ff.crop_geometry(960, 960, bias=-1.0)["x"], 0)
        self.assertEqual(ff.crop_geometry(960, 960, bias=1.0)["x"], 420)
        self.assertEqual(ff.crop_geometry(960, 960, bias=0.0)["x"], 210)
        # Середина между центром и правым краем: 0.75 * 420 = 315, вниз до
        # чётного — 314. Половина пикселя тут не украшение, см. yuv420p.
        self.assertEqual(ff.crop_geometry(960, 960, bias=0.5)["x"], 314)

    def test_nonsense_sizes_are_refused_and_missing_sizes_are_not_refused(self):
        # Три исхода не сворачиваются: «бессмыслица» и «не сняли» — разное.
        self.assertEqual(ff.crop_geometry(0, 960)["outcome"], FAIL)
        self.assertEqual(ff.crop_geometry(-960, 960)["outcome"], FAIL)
        self.assertEqual(ff.crop_geometry(960.0, 960)["outcome"], FAIL)
        self.assertEqual(ff.crop_geometry(None, 960)["outcome"], UNMEASURED)
        self.assertEqual(ff.crop_geometry(960, None)["outcome"], UNMEASURED)
        # И негативный контроль к обоим: годный вход обязан пройти.
        self.assertEqual(ff.crop_geometry(960, 960)["outcome"], PASS)

    def test_a_bias_outside_the_band_is_refused_and_the_edge_is_not(self):
        self.assertEqual(ff.crop_geometry(960, 960, bias=1.5)["outcome"], FAIL)
        self.assertEqual(ff.crop_geometry(960, 960, bias=-1.5)["outcome"], FAIL)
        self.assertEqual(ff.crop_geometry(960, 960, bias="лево")["outcome"], FAIL)
        self.assertEqual(ff.crop_geometry(960, 960, bias=1.0)["outcome"], PASS)
        self.assertEqual(ff.crop_geometry(960, 960, bias=-1.0)["outcome"], PASS)

    def test_a_degenerate_frame_cannot_produce_a_window(self):
        self.assertEqual(ff.crop_geometry(2, 2)["outcome"], FAIL)
        self.assertEqual(ff.crop_geometry(960, 960, ratio_w=0)["outcome"], FAIL)
        self.assertEqual(ff.crop_geometry(960, 960, ratio_h=-16)["outcome"], FAIL)

    def test_the_target_ratio_is_nine_by_sixteen(self):
        # Литералами: решение владельца от 2026-08-22. Разъедется — покраснеет.
        self.assertEqual((ff.TARGET_RATIO_W, ff.TARGET_RATIO_H), (9, 16))


class CropConstantsAreMutatedInBothDirections(unittest.TestCase):
    """Т1: константу подменяем строже и слабее, тест обязан покраснеть."""

    def test_the_even_multiple_guards_the_window(self):
        was = ff.DIM_MULTIPLE
        try:
            ff.DIM_MULTIPLE = 1          # СЛАБЕЕ: нечётное разрешено
            self.assertEqual(ff.crop_geometry(961, 961)["h"], 961)
            ff.DIM_MULTIPLE = 4          # СТРОЖЕ: кратно четырём
            self.assertEqual(ff.crop_geometry(961, 961)["w"], 540)
            self.assertEqual(ff.crop_geometry(961, 961)["h"], 960)
            self.assertEqual(ff.crop_geometry(962, 962)["h"], 960)
        finally:
            ff.DIM_MULTIPLE = was
        # И возврат: без него следующий тест поехал бы на чужой константе.
        self.assertEqual(ff.crop_geometry(961, 961)["h"], 960)

    def test_the_ratio_constants_change_the_window_both_ways(self):
        self.assertEqual(ff.crop_geometry(960, 960, ratio_w=16, ratio_h=9)["h"],
                         540)
        self.assertEqual(ff.crop_geometry(960, 960, ratio_w=1, ratio_h=1)["w"],
                         960)


# ---------------------------------------------------------------------------
# Прибор смещения и его негативный контроль
# ---------------------------------------------------------------------------

class BiasIsChosenOnlyWhenThereIsSomethingToChooseFrom(unittest.TestCase):
    def test_a_flat_map_gives_no_bias_and_says_so(self):
        # НЕГАТИВНЫЙ КОНТРОЛЬ (И5): равномерная карта — прибор обязан сказать
        # «не смогли выбрать», а не выдать бодрое число.
        r = ff.bias_from_columns([7.0] * 48)
        self.assertEqual(r["outcome"], UNMEASURED)
        self.assertEqual(r["bias"], 0.0)
        self.assertEqual(r["gain"], 1.0)

    def test_an_empty_map_gives_no_bias_either(self):
        r = ff.bias_from_columns([0.0] * 48)
        self.assertEqual(r["outcome"], UNMEASURED)
        self.assertEqual(r["bias"], 0.0)

    def test_the_real_material_puts_the_person_in_the_middle(self):
        # ИЗМЕРЕНО на выходе Kling: выигрыш лучшего окна 1.0008x — это шум.
        r = ff.bias_from_columns(REAL_COLUMNS)
        self.assertEqual(r["outcome"], UNMEASURED)
        self.assertEqual(r["bias"], 0.0)
        self.assertLess(r["gain"], 1.01)
        self.assertGreater(r["gain"], 1.0)

    def test_a_person_standing_aside_moves_the_window_there(self):
        # Вход, где прибор ОБЯЗАН шевельнуться (И5, вторая половина).
        left = [100.0] * 20 + [0.0] * 28
        right = [0.0] * 28 + [100.0] * 20
        rl, rr = ff.bias_from_columns(left), ff.bias_from_columns(right)
        self.assertEqual((rl["outcome"], rr["outcome"]), (PASS, PASS))
        self.assertLess(rl["bias"], 0.0)
        self.assertGreater(rr["bias"], 0.0)
        self.assertEqual(rl["bias"], -1.0)
        self.assertEqual(rr["bias"], 1.0)

    def test_a_person_a_little_off_centre_still_moves_the_window(self):
        # Фикстура ИЗ СЕРЕДИНЫ диапазона (Т3): не «весь свет слева», а просто
        # заметно больше движения в левой половине. Выигрыш 1.3125x — выше
        # порога 1.05 и ниже удвоения, то есть ровно та полоса, где прибор
        # обязан ответить, а не отмолчаться.
        r = ff.bias_from_columns([100.0] * 24 + [40.0] * 24)
        self.assertEqual(r["outcome"], PASS)
        self.assertEqual(r["gain"], 1.3125)
        self.assertLess(r["bias"], 0.0)

    def test_a_broken_map_is_refused_and_a_missing_one_is_not(self):
        self.assertEqual(ff.bias_from_columns([1.0, -1.0] * 24)["outcome"], FAIL)
        self.assertEqual(ff.bias_from_columns(["лево"] * 48)["outcome"], FAIL)
        self.assertEqual(ff.bias_from_columns(None)["outcome"], UNMEASURED)
        self.assertEqual(ff.bias_from_columns([])["outcome"], UNMEASURED)
        self.assertEqual(ff.bias_from_columns([1.0])["outcome"], UNMEASURED)

    def test_the_gain_threshold_is_mutated_in_both_directions(self):
        # Т1. Слабее порог — шумовой выигрыш 1.0008 начинает двигать кадр;
        # строже — даже человек сбоку перестаёт считаться доказанным.
        was = ff.BIAS_GAIN_MIN
        try:
            ff.BIAS_GAIN_MIN = 1.0001
            self.assertEqual(ff.bias_from_columns(REAL_COLUMNS)["outcome"], PASS)
            ff.BIAS_GAIN_MIN = 100.0
            self.assertEqual(
                ff.bias_from_columns([100.0] * 20 + [0.0] * 28)["outcome"],
                UNMEASURED)
        finally:
            ff.BIAS_GAIN_MIN = was
        self.assertEqual(ff.bias_from_columns(REAL_COLUMNS)["outcome"], UNMEASURED)


# ---------------------------------------------------------------------------
# B. ЗВУК
# ---------------------------------------------------------------------------

class TheWindowIsCountedInclusively(unittest.TestCase):
    def test_frames_one_hundred_to_one_hundred_ninety_nine_are_one_hundred(self):
        # Настоящее окно прогона. Файл окна на диске — 100 кадров.
        r = ff.window_frames(100, 199)
        self.assertEqual(r["outcome"], PASS)
        self.assertEqual(r["frames"], 100)

    def test_a_single_frame_window_is_one_frame(self):
        self.assertEqual(ff.window_frames(7, 7)["frames"], 1)

    def test_a_reversed_or_negative_window_is_refused(self):
        self.assertEqual(ff.window_frames(199, 100)["outcome"], FAIL)
        self.assertEqual(ff.window_frames(-1, 10)["outcome"], FAIL)
        self.assertEqual(ff.window_frames(1.5, 10)["outcome"], FAIL)
        self.assertEqual(ff.window_frames(None, 10)["outcome"], UNMEASURED)
        self.assertEqual(ff.window_frames(100, 199)["outcome"], PASS)


class TheToleranceIsTimeAndNotFrames(unittest.TestCase):
    def test_the_tolerance_at_our_rates(self):
        # 45 мс порога заметности: 30 к/с -> 1 кадр, 60 к/с -> 2, 24 -> 1.
        self.assertEqual(ff.drift_tolerance_frames(30), 1)
        self.assertEqual(ff.drift_tolerance_frames(24), 1)
        self.assertEqual(ff.drift_tolerance_frames(60), 2)
        self.assertEqual(ff.drift_tolerance_frames(120), 5)

    def test_an_unknown_rate_gives_no_tolerance_at_all(self):
        for bad in (None, 0, -30, "тридцать"):
            with self.subTest(fps=bad):
                self.assertIsNone(ff.drift_tolerance_frames(bad))
        self.assertEqual(ff.drift_tolerance_frames(30), 1)

    def test_the_perception_threshold_is_mutated_in_both_directions(self):
        # Т1. 30 мс — строже, допуска на 30 к/с не остаётся вовсе; 70 мс —
        # слабее, и внутрь допуска попадают два кадра, то есть 66.7 мс.
        was = ff.LIPSYNC_AUDIO_AHEAD_MS
        try:
            ff.LIPSYNC_AUDIO_AHEAD_MS = 30
            self.assertEqual(ff.drift_tolerance_frames(30), 0)
            self.assertEqual(ff.audio_drift(100, 99, fps=30)["outcome"], FAIL)
            ff.LIPSYNC_AUDIO_AHEAD_MS = 70
            self.assertEqual(ff.drift_tolerance_frames(30), 2)
            self.assertEqual(ff.audio_drift(90, 88, fps=30)["outcome"], PASS)
        finally:
            ff.LIPSYNC_AUDIO_AHEAD_MS = was
        self.assertEqual(ff.drift_tolerance_frames(30), 1)


class TheFourMeasuredKlingRunsAreJudgedCorrectly(unittest.TestCase):
    """ИЗМЕРЕНО 2026-08-22: четыре настоящих прогона, четыре вердикта."""

    def test_one_hundred_frames_came_back_as_ninety_nine(self):
        r = ff.audio_drift(100, 99, fps=30)
        self.assertEqual(r["outcome"], PASS)
        self.assertTrue(r["glue"])
        self.assertEqual(r["drift_frames"], -1)
        self.assertEqual(r["drift_ms"], -33.3)
        self.assertEqual(r["tolerance"], 1)

    def test_eighty_eight_frames_came_back_as_ninety_one(self):
        r = ff.audio_drift(88, 91, fps=30)
        self.assertEqual(r["outcome"], FAIL)
        self.assertFalse(r["glue"])
        self.assertEqual(r["drift_frames"], 3)
        self.assertEqual(r["drift_ms"], 100.0)

    def test_ninety_frames_came_back_as_eighty_eight(self):
        r = ff.audio_drift(90, 88, fps=30)
        self.assertEqual(r["outcome"], FAIL)
        self.assertFalse(r["glue"])
        self.assertEqual(r["drift_frames"], -2)

    def test_one_hundred_eighty_frames_came_back_exactly(self):
        r = ff.audio_drift(180, 180, fps=30)
        self.assertEqual(r["outcome"], PASS)
        self.assertTrue(r["glue"])
        self.assertEqual(r["drift_frames"], 0)
        self.assertEqual(r["drift_ms"], 0.0)


class TheAudioVerdictHasThreeOutcomes(unittest.TestCase):
    def test_the_boundary_is_symmetric_and_it_is_a_boundary(self):
        # Ровно на допуске — пропускаем, на кадр дальше — нет. В ОБЕ стороны,
        # потому что знак сдвига губ нам неизвестен по построению.
        self.assertEqual(ff.audio_drift(100, 101, fps=30)["outcome"], PASS)
        self.assertEqual(ff.audio_drift(100, 99, fps=30)["outcome"], PASS)
        self.assertEqual(ff.audio_drift(100, 102, fps=30)["outcome"], FAIL)
        self.assertEqual(ff.audio_drift(100, 98, fps=30)["outcome"], FAIL)

    def test_the_same_two_frames_pass_at_sixty_and_fail_at_thirty(self):
        # Допуск задан временем: 2 кадра на 60 к/с — те же 33 мс.
        self.assertEqual(ff.audio_drift(200, 202, fps=60)["outcome"], PASS)
        self.assertEqual(ff.audio_drift(200, 202, fps=30)["outcome"], FAIL)

    def test_an_unreadable_duration_is_neither_pass_nor_fail(self):
        for expected, actual, fps in ((100, None, 30), (None, 99, 30),
                                      (100, 99, None), (100, 99, "нет")):
            with self.subTest(expected=expected, actual=actual, fps=fps):
                r = ff.audio_drift(expected, actual, fps=fps)
                self.assertEqual(r["outcome"], UNMEASURED)
                self.assertFalse(r["glue"])
        self.assertEqual(ff.audio_drift(100, 99, fps=30)["outcome"], PASS)

    def test_nonsense_frame_counts_are_refused(self):
        self.assertEqual(ff.audio_drift(0, 99, fps=30)["outcome"], FAIL)
        self.assertEqual(ff.audio_drift(100, -1, fps=30)["outcome"], FAIL)

    def test_glue_never_travels_with_a_bad_verdict(self):
        # Инвариант Е2: «клеить» выводится из исхода, а не назначается рядом.
        for expected in range(90, 111):
            r = ff.audio_drift(expected, 100, fps=30)
            with self.subTest(expected=expected, outcome=r["outcome"]):
                self.assertEqual(r["glue"], r["outcome"] == PASS)
        self.assertTrue(ff.audio_drift(100, 100, fps=30)["glue"])
        self.assertFalse(ff.audio_drift(100, 90, fps=30)["glue"])


class TheAudioPlanReadsTheRealFilesShape(unittest.TestCase):
    def setUp(self):
        self.drv, self.kln = _files("driving_arms.mp4", "kling.mp4")

    def test_the_real_run_is_pass_with_a_named_warning(self):
        p = ff.audio_plan(self.drv, (100, 199), self.kln,
                          prober=prober_of({"driving_arms.mp4": DRIVING_JSON,
                                            "kling.mp4": KLING_JSON}))
        self.assertEqual(p["outcome"], PASS)
        self.assertTrue(p["glue"])
        self.assertEqual((p["expected"], p["actual"]), (100, 99))
        self.assertEqual(p["drift_frames"], -1)
        # Звук берётся С 100-го КАДРА, то есть с 3.333333 с, а не с нуля.
        self.assertEqual(p["start_seconds"], 3.333333)
        self.assertEqual(p["seconds"], 3.3)

    def test_a_driving_without_sound_is_refused_with_the_reason(self):
        p = ff.audio_plan(self.drv, (100, 199), self.kln,
                          prober=prober_of({"driving_arms.mp4": DRIVING_SILENT_JSON,
                                            "kling.mp4": KLING_JSON}))
        self.assertEqual(p["outcome"], FAIL)
        self.assertFalse(p["glue"])

    def test_a_window_outside_the_driving_is_refused(self):
        # В драйвинге 373 кадра, номера 0..372.
        p = ff.audio_plan(self.drv, (300, 399), self.kln,
                          prober=prober_of({"driving_arms.mp4": DRIVING_JSON,
                                            "kling.mp4": KLING_JSON}))
        self.assertEqual(p["outcome"], FAIL)
        # Негативный контроль: последний законный кадр обязан пройти.
        ok = ff.audio_plan(self.drv, (273, 372), self.kln,
                           prober=prober_of({"driving_arms.mp4": DRIVING_JSON,
                                             "kling.mp4": KLING_JSON}))
        self.assertEqual(ok["outcome"], PASS)

    def test_different_rates_cannot_be_compared_in_frames(self):
        p = ff.audio_plan(self.drv, (100, 199), self.kln,
                          prober=prober_of({
                              "driving_arms.mp4": DRIVING_JSON,
                              "kling.mp4": KLING_JSON.replace('"30/1"', '"24/1"')}))
        self.assertEqual(p["outcome"], FAIL)

    def test_no_ffprobe_is_not_a_bad_file(self):
        p = ff.audio_plan(self.drv, (100, 199), self.kln,
                          prober=prober_of({"driving_arms.mp4": None,
                                            "kling.mp4": None}))
        self.assertEqual(p["outcome"], UNMEASURED)
        self.assertFalse(p["glue"])

    def test_a_broken_file_is_not_an_absent_tool(self):
        p = ff.audio_plan(self.drv, (100, 199), self.kln,
                          prober=prober_of({"driving_arms.mp4": DRIVING_JSON,
                                            "kling.mp4": 1}))
        self.assertEqual(p["outcome"], FAIL)

    def test_every_step_is_named_and_none_is_silent(self):
        p = ff.audio_plan(self.drv, (100, 199), self.kln,
                          prober=prober_of({"driving_arms.mp4": DRIVING_JSON,
                                            "kling.mp4": KLING_JSON}))
        self.assertGreaterEqual(len(p["steps"]), 4)
        for name, outcome, note in p["steps"]:
            with self.subTest(step=name):
                self.assertIn(outcome, (PASS, FAIL, UNMEASURED))
                self.assertGreater(len(note), 10)


# ---------------------------------------------------------------------------
# Команда сборки
# ---------------------------------------------------------------------------

class TheCommandIsADecisionAndIsCheckedApartFromItsRun(unittest.TestCase):
    GEOM = {"w": 540, "h": 960, "x": 210, "y": 0}

    def test_the_crop_filter_carries_the_planned_window(self):
        argv = ff.mux_argv("k.mp4", "out.mp4", self.GEOM)
        self.assertIn("[0:v]crop=540:960:210:0[v]", argv)

    def test_the_sound_is_cut_from_the_input_and_not_from_the_output(self):
        argv = ff.mux_argv("k.mp4", "out.mp4", self.GEOM, driving_path="d.mp4",
                           start_seconds=3.333333, seconds=3.3)
        self.assertIn("-ss", argv)
        # -ss ПЕРЕД своим -i: иначе звук поедет от нулевой секунды исходника.
        self.assertLess(argv.index("-ss"), argv.index("d.mp4"))
        self.assertLess(argv.index("-t"), argv.index("d.mp4"))
        self.assertEqual(argv[argv.index("-ss") + 1], "3.333333")
        self.assertEqual(argv[argv.index("-t") + 1], "3.300000")
        self.assertIn("1:a", argv)
        self.assertNotIn("-an", argv)

    def test_without_a_driving_the_file_is_written_deliberately_mute(self):
        argv = ff.mux_argv("k.mp4", "out.mp4", self.GEOM)
        self.assertIn("-an", argv)
        self.assertNotIn("1:a", argv)
        self.assertNotIn("-shortest", argv)
        self.assertNotIn("-ss", argv)

    def test_the_streams_are_mapped_explicitly(self):
        argv = ff.mux_argv("k.mp4", "out.mp4", self.GEOM, driving_path="d.mp4",
                           start_seconds=0.0, seconds=1.0)
        self.assertIn("-map", argv)
        self.assertIn("[v]", argv)
        self.assertEqual(argv[-1], "out.mp4")
        self.assertIn("yuv420p", argv)

    def test_the_quality_keys_are_the_ones_we_chose(self):
        # Литералами: качество финального файла — тоже решение, и подмена
        # CRF 18 на 28 иначе уехала бы молча, а увидели бы её на показе.
        argv = ff.mux_argv("k.mp4", "out.mp4", self.GEOM, driving_path="d.mp4",
                           start_seconds=0.0, seconds=1.0)
        self.assertEqual(argv[argv.index("-crf") + 1], "18")
        self.assertEqual(argv[argv.index("-b:a") + 1], "128k")
        self.assertIn("libx264", argv)
        self.assertIn("aac", argv)

    def test_no_filter_ever_stretches_the_sound(self):
        # Растяжение убрало бы расхождение из чисел и оставило бы его в ушах.
        argv = ff.mux_argv("k.mp4", "out.mp4", self.GEOM, driving_path="d.mp4",
                           start_seconds=0.0, seconds=1.0)
        joined = " ".join(argv)
        for forbidden in ("atempo", "asetrate", "rubberband", "setpts"):
            with self.subTest(filter=forbidden):
                self.assertNotIn(forbidden, joined)


# ---------------------------------------------------------------------------
# C. СБОРКА
# ---------------------------------------------------------------------------

class TheAssemblyReportsWhatActuallyHappened(unittest.TestCase):
    def setUp(self):
        self.drv, self.kln, self.out = _files("driving_arms.mp4", "kling.mp4",
                                              "finish.mp4")
        self.answers = {"driving_arms.mp4": DRIVING_JSON,
                        "kling.mp4": KLING_JSON,
                        "finish.mp4": RESULT_JSON}

    def _finish(self, runner, **over):
        answers = dict(self.answers, **over.pop("answers", {}))
        return ff.finish(self.drv, self.kln, self.out, window=(100, 199),
                         prober=prober_of(answers), runner=runner, **over)

    def test_the_real_case_assembles_with_sound_and_says_the_numbers(self):
        run = Runner()
        rep = self._finish(run)
        self.assertEqual(rep["outcome"], PASS)
        self.assertTrue(rep["written"])
        self.assertTrue(rep["audio"])
        self.assertEqual((rep["crop"]["w"], rep["crop"]["h"]), (540, 960))
        self.assertEqual(rep["crop"]["lost_percent"], 43.75)
        self.assertEqual(rep["audio_plan"]["drift_frames"], -1)
        self.assertEqual(len(run.calls), 1)
        self.assertIn("[0:v]crop=540:960:210:0[v]", run.calls[0])

    def test_a_drift_beyond_tolerance_writes_a_mute_file_and_says_not_good(self):
        # Ровно тот исход, ради которого модуль пишется: файл есть, звука нет,
        # вердикт «не годно», и он не свёрнут в «годно, но без звука».
        run = Runner()
        rep = ff.finish(self.drv, self.kln, self.out, window=(100, 187),
                        prober=prober_of(dict(self.answers,
                                              **{"finish.mp4": RESULT_SILENT_JSON})),
                        runner=run)
        self.assertEqual(rep["outcome"], FAIL)
        self.assertTrue(rep["written"])
        self.assertFalse(rep["audio"])
        self.assertEqual(rep["audio_plan"]["expected"], 88)
        self.assertEqual(rep["audio_plan"]["actual"], 99)
        self.assertIn("-an", run.calls[0])
        self.assertNotIn("1:a", run.calls[0])

    def test_an_unreadable_duration_writes_nothing_at_all(self):
        # Третий исход: не клеим и не отказываем — и файл не пишем, потому что
        # немой ролик вместо ролика со звуком — это решение за человека.
        run = Runner()
        rep = self._finish(run, answers={"driving_arms.mp4": None})
        self.assertEqual(rep["outcome"], UNMEASURED)
        self.assertFalse(rep["written"])
        self.assertEqual(len(run.calls), 0)

    def test_an_absent_ffmpeg_is_not_a_bad_file(self):
        rep = self._finish(Runner(ran=False))
        self.assertEqual(rep["outcome"], UNMEASURED)
        self.assertFalse(rep["written"])

    def test_a_failing_ffmpeg_is_not_a_missing_ffmpeg(self):
        rep = self._finish(Runner(code=1, err="Invalid argument"))
        self.assertEqual(rep["outcome"], FAIL)
        self.assertFalse(rep["written"])

    def test_the_verdict_comes_from_the_file_and_not_from_the_intention(self):
        # Е2. ffmpeg «отработал», а на диске лежит квадрат — вердикт «не годно»,
        # хотя все планы были годные.
        run = Runner()
        rep = self._finish(run, answers={"finish.mp4": KLING_JSON})
        self.assertEqual(rep["outcome"], FAIL)
        self.assertTrue(rep["written"])

    def test_a_promised_sound_that_did_not_arrive_is_caught(self):
        # Вторая половина Е2: план обещал звук, в файле звука нет.
        run = Runner()
        rep = self._finish(run, answers={"finish.mp4": RESULT_SILENT_JSON})
        self.assertEqual(rep["outcome"], FAIL)
        self.assertFalse(rep["audio"])

    def test_a_broken_kling_output_stops_before_any_ffmpeg_runs(self):
        # П2: дешёвая проверка раньше дорогой.
        run = Runner()
        rep = self._finish(run, answers={"kling.mp4": 1})
        self.assertEqual(rep["outcome"], FAIL)
        self.assertEqual(len(run.calls), 0)

    def test_every_step_is_named_with_its_own_outcome(self):
        rep = self._finish(Runner())
        names = [n for n, _, _ in rep["steps"]]
        self.assertGreaterEqual(len(names), 6)
        for name, outcome, note in rep["steps"]:
            with self.subTest(step=name):
                self.assertIn(outcome, (PASS, FAIL, UNMEASURED))
                self.assertGreater(len(note), 10)
        self.assertIn("кроп", names)
        self.assertIn("звук", names)
        self.assertIn("сборка", names)

    def test_the_elapsed_time_of_the_run_is_printed(self):
        rep = self._finish(Runner())
        self.assertIsInstance(rep["elapsed"], float)
        self.assertGreaterEqual(rep["elapsed"], 0.0)


# ---------------------------------------------------------------------------
# Устройство модуля
# ---------------------------------------------------------------------------

class TheModuleDoesNotReinventWhatAlreadyExists(unittest.TestCase):
    SRC = Path(ff.__file__).read_text(encoding="utf-8")

    def test_the_outside_world_is_touched_only_through_the_neighbour(self):
        # Свой subprocess здесь означал бы второй способ звать ffmpeg и второй
        # разбор его отказа (Е1). Проверяется деревом, а не подстрокой.
        tree = ast.parse(self.SRC)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for forbidden in ("subprocess", "shutil", "os"):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, imported)

    def test_the_probe_answer_is_not_parsed_a_second_time(self):
        self.assertNotIn("json.loads", self.SRC)
        self.assertNotIn("avg_frame_rate", self.SRC)

    def test_the_verdict_words_are_not_reinvented(self):
        self.assertEqual((PASS, FAIL, UNMEASURED),
                         ("годно", "не годно", "не смогли проверить"))

    def test_the_three_outcomes_map_to_three_different_exit_codes(self):
        self.assertEqual(ff.EXIT_BY_OUTCOME[PASS], 0)
        self.assertEqual(ff.EXIT_BY_OUTCOME[FAIL], 1)
        self.assertEqual(ff.EXIT_BY_OUTCOME[UNMEASURED], 2)
        self.assertEqual(len(set(ff.EXIT_BY_OUTCOME.values())), 3)

    def test_the_injection_points_are_resolved_in_the_body(self):
        # Умолчание, связанное на импорте (`runner=fork_video.run_decode` в
        # сигнатуре), мутацией уже не достаётся — эту форму на проекте выгребали.
        self.assertIn("runner = fork_video.run_decode if runner is None", self.SRC)

    def test_every_decision_constant_declares_where_it_came_from(self):
        # И4: у каждой константы модуля назван источник. Проверяется наличие
        # пометки в комментарии НАД строкой, а не вера в аккуратность.
        lines = self.SRC.splitlines()
        names = ("TARGET_RATIO_W", "DIM_MULTIPLE", "LIPSYNC_AUDIO_AHEAD_MS",
                 "BIAS_GAIN_MIN", "BIAS_LIMIT", "VIDEO_CRF")
        for name in names:
            with self.subTest(constant=name):
                i = next(k for k, ln in enumerate(lines)
                         if ln.startswith(name))
                above = "\n".join(lines[max(0, i - 20):i])
                self.assertTrue(
                    any(mark in above for mark in
                        ("ИЗМЕРЕНО", "РАСЧЁТ", "ВЫБРАНО")),
                    f"{name}: происхождение не помечено")


if __name__ == "__main__":
    unittest.main()
