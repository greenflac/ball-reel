"""Метрика мимики судится по тому, различает ли она заведомо известные случаи.

Проверяется не «функция не падает», а четыре решения, на которых стоит вердикт:

* мёртвые коэффициенты не разбавляют ответ (иначе разные лица — «совпадение»);
* моргание и взгляд не влияют на вердикт вовсе;
* «не смогли измерить» отличимо и от «похоже», и от «не похоже»;
* замороженное лицо НЕ проходит, хотя по одному расстоянию проходило бы.

Синтетика собрана руками и числами из условия задачи, а не из порогов модуля:
тест, который берёт вход из константы, которую сторожит, всегда зелёный и ничего
не проверяет. Пороги здесь читаются из модуля только в правой части сравнений.

Это НЕ калибровка. Калибровка на настоящей генерации в проекте отсутствует —
ни одного сгенерированного клипа ещё не измерено, и эти тесты её не заменяют.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ball_reel import expression as ex

#: Все 52 имени, которые отдаёт MediaPipe FaceLandmarker. Литеральный список, а
#: не выборка из модуля: разбавление проверяется на настоящем составе выхода.
ALL_SHAPES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)


def face(**shapes) -> dict:
    """Полный набор из 52 коэффициентов: названные заданы, остальные в покое."""
    out = {name: 0.0 for name in ALL_SHAPES}
    for name, value in shapes.items():
        assert name in out, name
        out[name] = float(value)
    return out


def naive_mean(a: dict, b: dict) -> float:
    """Наивная метрика «среднее по всем 52» — та, от которой модуль отказался.

    Держится в тестах намеренно: без неё утверждение «мёртвые коэффициенты
    разбавляют» остаётся словами.
    """
    return sum(abs(a[k] - b[k]) for k in ALL_SHAPES) / len(ALL_SHAPES)


class DeadShapesMustNotDiluteTheAnswer(unittest.TestCase):
    """Два заведомо разных выражения обязаны читаться как разные."""

    def setUp(self):
        # Рот закрыт против рта открытого — размах взят с настоящего клипа
        # (jawOpen ходил 0.002 -> 0.389), но записан здесь числом, а не выведен
        # из модуля.
        self.closed = face(jawOpen=0.00)
        self.opened = face(jawOpen=0.39)

    def test_naive_all_52_would_call_them_the_same(self):
        """Обоснование отбора: по всем 52 разница неотличима от совпадения."""
        self.assertLess(naive_mean(self.closed, self.opened), ex.MATCH_MAX,
                        "среднее по 52 неожиданно велико — переписать "
                        "обоснование отбора, а не порог")

    def test_judged_set_calls_them_different(self):
        d = ex.frame_distance(self.closed, self.opened)
        self.assertIsNotNone(d)
        self.assertGreater(d, ex.MATCH_MAX)

    def test_identical_faces_are_zero(self):
        self.assertEqual(ex.frame_distance(self.opened, dict(self.opened)), 0.0)

    def test_dead_shapes_carry_no_weight(self):
        """Щёки и нос не судятся: изменение только в них не должно двигать цифру."""
        puffed = face(cheekPuff=1.0, noseSneerLeft=1.0, cheekSquintRight=1.0)
        self.assertEqual(ex.frame_distance(face(), puffed), 0.0)


class BlinkAndGazeAreNotExpression(unittest.TestCase):
    """Моргание меняется на 0.74 за 83 мс — по нему нельзя судить о выражении."""

    def test_blink_does_not_change_the_verdict(self):
        openeyed = face(jawOpen=0.20)
        blinking = face(jawOpen=0.20, eyeBlinkLeft=0.90, eyeBlinkRight=0.90)
        self.assertEqual(ex.frame_distance(openeyed, blinking), 0.0)

    def test_gaze_direction_does_not_change_the_verdict(self):
        ahead = face(browInnerUp=0.40)
        aside = face(browInnerUp=0.40, eyeLookOutLeft=0.70, eyeLookInRight=0.70)
        self.assertEqual(ex.frame_distance(ahead, aside), 0.0)

    def test_brow_and_jaw_do_change_it(self):
        """Обратная сторона: то, что судится, обязано двигать цифру."""
        self.assertGreater(ex.frame_distance(face(), face(browInnerUp=0.60)), 0.0)
        self.assertGreater(ex.frame_distance(face(), face(jawOpen=0.60)), 0.0)


class GroupsAreReportedSeparately(unittest.TestCase):
    """«Мимика не доехала» и «брови не доехали» чинятся в разных местах."""

    def test_only_the_moved_group_shows_error(self):
        errs = ex.group_errors(face(), face(jawOpen=0.40))
        self.assertGreater(errs["jaw"], 0.0)
        self.assertEqual(errs["brow"], 0.0)
        self.assertEqual(errs["smile"], 0.0)

    def test_incomplete_dict_is_not_judged(self):
        """Строка track из driving.py содержит не все коэффициенты.

        Недостающий коэффициент — это отсутствие наблюдения, и подставлять
        вместо него ноль нельзя: ноль означает «мышца расслаблена».
        """
        partial = {"jawOpen": 0.30, "mouthSmileLeft": 0.10}
        self.assertIsNone(ex.group_errors(face(), partial))
        self.assertIsNone(ex.frame_distance(face(), partial))


def ramp(n: int = 12, top: float = 0.60) -> list:
    """Клип, где рот плавно открывается: заведомо есть чему следовать."""
    return [face(jawOpen=round(top * i / (n - 1), 4)) for i in range(n)]


class TheClipVerdict(unittest.TestCase):
    """Три заведомо известных ответа: та же мимика, чужая, судить нельзя."""

    def test_exact_copy_passes(self):
        drv = ramp()
        got = ex.expression_match(drv, [dict(f) for f in drv])
        self.assertEqual(got["state"], "measured")
        self.assertIs(got["verdict"], True)
        self.assertEqual(got["distance_p50"], 0.0)
        self.assertLess(got["follow_ratio"], ex.FOLLOW_MAX)

    def test_small_error_still_passes(self):
        """Генерация не обязана совпадать бит в бит — 0.02 по jawOpen мала."""
        drv = ramp()
        gen = [face(jawOpen=round(f["jawOpen"] + 0.02, 4)) for f in drv]
        got = ex.expression_match(drv, gen)
        self.assertIs(got["verdict"], True, got["note"])

    def test_a_different_performance_fails(self):
        """Рот открывается там, где на референсе закрывается."""
        drv = ramp()
        gen = list(reversed([dict(f) for f in drv]))
        got = ex.expression_match(drv, gen)
        self.assertEqual(got["state"], "measured")
        self.assertIs(got["verdict"], False, got["note"])

    def test_frozen_face_fails_although_the_distance_alone_would_pass(self):
        """Главный тест модуля: замороженное лицо не должно проходить.

        Клип открывает и закрывает рот, генерация держит среднее выражение.
        Расстояние при этом МАЛЕНЬКОЕ — и если бы вердикт стоял на нём одном,
        подделка прошла бы. Ловит её сравнение со своим же нулевым
        распределением.
        """
        drv = [face(jawOpen=0.00 if i % 2 else 0.40) for i in range(12)]
        gen = [face(jawOpen=0.20) for _ in range(12)]
        got = ex.expression_match(drv, gen)
        self.assertEqual(got["state"], "measured")
        self.assertLessEqual(got["distance_p50"], ex.MATCH_MAX,
                             "синтетика перестала быть подделкой, которая "
                             "проходит по расстоянию — тест потерял смысл")
        self.assertGreater(got["follow_ratio"], ex.FOLLOW_MAX)
        self.assertIs(got["verdict"], False, got["note"])

    def test_following_but_damped_is_not_the_same_as_frozen(self):
        """Вдвое ослабленная мимика всё же СЛЕДИТ, и это должно быть видно.

        Ослабление берётся ВОКРУГ СРЕДНЕГО выражения клипа — иначе к ослаблению
        примешивается постоянный сдвиг («рот всё время приоткрыт меньше»), а это
        уже другой дефект, и метрика штрафует за него отдельно и справедливо.
        """
        drv = ramp()
        mid = sum(f["jawOpen"] for f in drv) / len(drv)
        gen = [face(jawOpen=round(mid + (f["jawOpen"] - mid) / 2, 4))
               for f in drv]
        got = ex.expression_match(drv, gen)
        self.assertLess(got["follow_ratio"], ex.FOLLOW_MAX, got["note"])

    def test_a_constant_offset_is_penalised_even_when_it_follows(self):
        """Мимика повторена по форме, но вся смещена: следование есть, вердикт
        отрицательный. Разные дефекты не должны склеиваться в один ответ."""
        drv = ramp()
        gen = [face(jawOpen=round(f["jawOpen"] / 2, 4)) for f in drv]
        got = ex.expression_match(drv, gen)
        self.assertIs(got["verdict"], False, got["note"])

    def test_worst_group_is_named(self):
        drv = [face(jawOpen=0.10 * i, browInnerUp=0.05 * i) for i in range(10)]
        gen = [face(jawOpen=0.10 * i, browInnerUp=0.00) for i in range(10)]
        got = ex.expression_match(drv, gen)
        self.assertGreater(got["per_group"]["brow"], got["per_group"]["jaw"])
        self.assertIn("brow", got["note"])


class CouldNotMeasureIsItsOwnAnswer(unittest.TestCase):
    """Отдельный исход: не «похоже» и не «не похоже», а «нечем судить»."""

    def test_too_few_frames(self):
        drv = ramp(3)
        got = ex.expression_match(drv, [dict(f) for f in drv])
        self.assertIsNone(got["verdict"])
        self.assertEqual(got["state"], "not_measurable")
        self.assertEqual(got["pairs"], 3)

    def test_most_frames_unreadable(self):
        """Кадров хватает, но лицо прочиталось меньше чем в половине."""
        drv = ramp(20)
        gen = [dict(f) if i < 7 else None for i, f in enumerate(drv)]
        got = ex.expression_match(drv, gen)
        self.assertIsNone(got["verdict"])
        self.assertEqual(got["state"], "not_measurable")
        self.assertLess(got["coverage"], ex.MIN_COVERAGE)
        self.assertEqual(got["unjudged"], 13)

    def test_unreadable_frames_are_not_counted_as_mismatch(self):
        """Дырка не должна ухудшать цифру: 7 совпавших кадров из 20 — это
        покрытие 35%, а не «расхождение выросло»."""
        drv = ramp(20)
        gen = [dict(f) if i < 7 else None for i, f in enumerate(drv)]
        got = ex.expression_match(drv, gen)
        self.assertIsNone(got["distance_p50"])

    def test_static_driving_cannot_be_judged_for_following(self):
        """Референс без мимики: совпасть с ним можно и не следя за ним."""
        drv = [face(jawOpen=0.20) for _ in range(12)]
        got = ex.expression_match(drv, [dict(f) for f in drv])
        self.assertIsNone(got["verdict"])
        self.assertEqual(got["state"], "driving_static")
        self.assertIsNotNone(got["distance_p50"])
        self.assertIsNone(got["follow_ratio"])

    def test_static_driving_is_distinguishable_from_no_measurement(self):
        drv = [face(jawOpen=0.20) for _ in range(12)]
        static = ex.expression_match(drv, [dict(f) for f in drv])
        thin = ex.expression_match(ramp(3), [dict(f) for f in ramp(3)])
        self.assertNotEqual(static["state"], thin["state"])

    def test_unaligned_lists_are_refused_not_guessed(self):
        got = ex.expression_match(ramp(12), [dict(f) for f in ramp(10)])
        self.assertIsNone(got["verdict"])
        self.assertEqual(got["state"], "not_measurable")

    def test_reading_states_are_honoured(self):
        """`read_expression`-строка с state != measured не идёт в расчёт."""
        drv = ramp(12)
        gen = [{"path": "x", "state": "too_small", "face_px": 40,
                "shapes": None, "note": ""} for _ in drv]
        got = ex.expression_match(drv, gen)
        self.assertIsNone(got["verdict"])
        self.assertEqual(got["pairs"], 0)


# `demo/kit`, а не `kit`: последний в `.gitignore`, и класс ниже молчал в любом
# клоне. Кадры те же, их 16 вместо 71.
KIT = Path(__file__).resolve().parents[2] / "demo" / "kit" / "driving"
MODEL = Path(os.environ.get("BALL_REEL_FACE_MODEL",
                            "~/.mediapipe/face_landmarker.task")).expanduser()
_CACHE: dict = {}


def _real_readings():
    """Каждый третий кадр настоящей съёмки, прочитанный один раз на весь файл."""
    if "rows" not in _CACHE:
        frames = sorted(KIT.glob("*.jpg"))[::3]
        _CACHE["rows"] = ex.read_clip(frames)
    return _CACHE["rows"]


@unittest.skipUnless(KIT.is_dir() and MODEL.exists(),
                     "нет kit/driving или весов face_landmarker.task")
class OnRealFootage(unittest.TestCase):
    """Настоящие кадры: 720x1278, один человек, съёмка в полный рост."""

    def test_faces_are_readable_but_too_small_to_identify(self):
        """Продуктовый факт, ради которого порог не одолжен у личности.

        На этом материале лицо мельче, чем нужно ArcFace, и крупнее, чем нужно
        сетке: мимику мерить МОЖНО, личность на тех же кадрах — НЕЛЬЗЯ.
        """
        from ball_reel.identity_arcface import MIN_FACE_PX

        rows = _real_readings()
        px = sorted(r["face_px"] for r in rows if r["face_px"])
        self.assertEqual(len(px), len(rows), "лицо найдено не во всех кадрах")
        self.assertGreaterEqual(min(px), ex.MIN_JUDGE_FACE_PX)
        self.assertLess(max(px), MIN_FACE_PX)

    def test_every_frame_is_measurable(self):
        rows = _real_readings()
        self.assertTrue(all(r["state"] == "measured" for r in rows),
                        [r["state"] for r in rows])

    def test_the_clip_matches_itself(self):
        rows = _real_readings()
        got = ex.expression_match(rows, list(rows))
        self.assertEqual(got["state"], "measured")
        self.assertIs(got["verdict"], True, got["note"])

    def test_the_clip_does_not_match_its_own_shuffled_frames(self):
        """Те же кадры, тот же человек, то же освещение — но не те моменты.

        Если метрика это не ловит, она меряет человека, а не мимику.
        """
        rows = _real_readings()
        shuffled = rows[len(rows) // 2:] + rows[:len(rows) // 2]
        got = ex.expression_match(rows, shuffled)
        self.assertEqual(got["state"], "measured")
        self.assertIs(got["verdict"], False, got["note"])
        self.assertGreater(got["follow_ratio"], ex.FOLLOW_MAX)

    def test_real_frames_moved_enough_to_judge_following(self):
        """Реальный клип обязан иметь мимику — иначе он не годится в референсы."""
        rows = _real_readings()
        got = ex.expression_match(rows, list(rows))
        self.assertGreater(got["null_p50"], ex.MIN_NULL_SPREAD)

    def test_too_small_is_reported_as_such_not_as_mismatch(self):
        """Тот же кадр, поднятая планка размера — исход обязан смениться на
        «не измерено», а не на «мимика не совпала»."""
        frame = sorted(KIT.glob("*.jpg"))[0]
        got = ex.read_expression(frame, min_face_px=400)
        self.assertEqual(got["state"], "too_small")
        self.assertIsNone(got["shapes"])
        self.assertIn("400", got["note"])


if __name__ == "__main__":
    unittest.main()
