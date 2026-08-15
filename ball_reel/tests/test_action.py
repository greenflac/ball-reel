"""Ось «выполняется ли заявленное движение».

Эта ось закрывает САМУЮ КРУПНУЮ дыру гейта, и дыра не гипотетическая: живой
прогон дал кадр с лучшим дрейфом лица за день (0.21), на котором человек в
бальном платье СИДЕЛ в позе с фотографии. Все двенадцать осей `CHECK_ORDER`
это пропустили, потому что ни одна не спрашивает, совершил ли человек то же
движение, что человек на driving-видео.

Отдельно сторожится вырожденный случай: клип из одинаковых кадров. Он не
проваливал НИ ОДНОЙ проверки — все отвечали «не смогли измерить», и ноль на
выходе читался как успех. Здесь он обязан быть ЗАБРАКОВАН, а не пропущен.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _pose(t=0.0, *, spread=0.0, visible=1.0):
    """Скелет, у которого запястья расходятся по мере роста `t`.

    Координаты нормированные; `pose._normalise` центрирует на бёдрах и делит на
    торс, поэтому важна не абсолютная позиция, а изменение конфигурации.
    """
    return {
        "nose": (0.50, 0.10, visible),
        "l_shoulder": (0.58, 0.25, visible), "r_shoulder": (0.42, 0.25, visible),
        "l_elbow": (0.62 + t, 0.40, visible), "r_elbow": (0.38 - t, 0.40, visible),
        "l_wrist": (0.64 + t + spread, 0.55, visible),
        "r_wrist": (0.36 - t - spread, 0.55, visible),
        "l_hip": (0.55, 0.55, visible), "r_hip": (0.45, 0.55, visible),
        "l_knee": (0.56, 0.75, visible), "r_knee": (0.44, 0.75, visible),
        "l_ankle": (0.57, 0.92, visible), "r_ankle": (0.43, 0.92, visible),
    }


def _moving(n=16, step=0.02):
    return [_pose(i * step) for i in range(n)]


def _frozen(n=16):
    return [_pose(0.0) for _ in range(n)]


@unittest.skipUnless(HAVE_DEPS, "numpy not installed (live extra)")
class TheAxisAnswersWhetherTHATMovementHappened(unittest.TestCase):
    def setUp(self):
        from ball_reel import action

        self.a = action

    def test_a_clip_following_the_driving_trajectory_passes(self):
        drv = _moving()
        got = self.a.compare_trajectories(drv, list(drv))
        self.assertEqual(got["verdict"], "same", got)

    def test_a_frozen_clip_is_REJECTED_not_excused_as_unmeasurable(self):
        """Исторический дефект: 16 одинаковых кадров проходили весь гейт.

        Каждая ось отвечала «не смогли измерить», код возврата выходил нулевым,
        и это читалось как успех. Здесь такой клип обязан получить именно
        «different»: донор двигался, клип нет — это измерено, а не неизвестно.
        """
        got = self.a.compare_trajectories(_moving(), _frozen())
        self.assertEqual(got["verdict"], "different", got)

    def test_a_shuffled_clip_is_caught_although_its_poses_are_all_correct(self):
        """Тот же набор поз в другом порядке — это ДРУГОЕ движение.

        Проверка того, что ось мерит траекторию, а не мешок поз: покадровое
        сравнение поз такой клип пропустило бы целиком.
        """
        drv = _moving()
        shuffled = [drv[i] for i in (0, 9, 3, 14, 7, 1, 12, 5, 15, 2, 10, 6,
                                     13, 4, 11, 8)]
        got = self.a.compare_trajectories(drv, shuffled)
        self.assertEqual(got["verdict"], "different", got)

    def test_a_time_reversed_clip_is_caught(self):
        # Приседание против вставания: те же позы, обратный порядок.
        drv = _moving()
        got = self.a.compare_trajectories(drv, list(reversed(drv)))
        self.assertEqual(got["verdict"], "different", got)

    def test_a_clip_moving_in_time_but_too_weakly_is_named_by_amplitude(self):
        # Двигается в такт, но вяло: направление сходится, размах нет. Это
        # ОТДЕЛЬНАЯ беда, и смешивать её с «не туда» нельзя — лечится она
        # по-разному.
        drv = _moving(step=0.02)
        weak = _moving(step=0.02 * 0.1)
        got = self.a.compare_trajectories(drv, weak)
        self.assertIsNotNone(got["direction"])
        self.assertLess(got["amplitude"], 1.0)


@unittest.skipUnless(HAVE_DEPS, "numpy not installed (live extra)")
class ThreeOutcomesNotTwo(unittest.TestCase):
    """«Не смогли измерить» отличимо и от «то движение», и от «не то»."""

    def setUp(self):
        from ball_reel import action

        self.a = action

    def test_too_few_frames_is_not_measurable_rather_than_a_verdict(self):
        got = self.a.compare_trajectories(_moving(4), _moving(4))
        self.assertEqual(got["verdict"], "not_measurable", got)
        self.assertTrue(got.get("note"), "исход обязан объяснять себя")

    def test_frames_without_a_body_are_not_measurable(self):
        n = self.a.MIN_TRAJECTORY_FRAMES + 4
        got = self.a.compare_trajectories([None] * n, [None] * n)
        self.assertEqual(got["verdict"], "not_measurable", got)

    def test_a_motionless_DONOR_cannot_judge_anything(self):
        """Донор стоит -> сравнивать не с чем, и это не «клип плохой».

        Если бы неподвижный донор давал «different», ось браковала бы клип за
        то, чего от него никто не требовал.
        """
        n = self.a.MIN_TRAJECTORY_FRAMES + 4
        got = self.a.compare_trajectories(_frozen(n), _moving(n))
        self.assertEqual(got["verdict"], "not_measurable", got)

    def test_the_boolean_beside_the_verdict_does_not_collapse_the_third(self):
        """`follows` при «не смогли измерить» обязан быть None, а не False.

        Булев флаг рядом с трёхзначным вердиктом — приглашение к той самой
        ошибке, которую проект ловил уже трижды: читатель берёт флаг, и
        «не знаем» превращается в «плохо». Это разные решения: первое требует
        разобраться, второе бракует клип.
        """
        got = self.a.compare_trajectories(_moving(4), _moving(4))
        self.assertIsNone(got["follows"], got)
        # А на настоящих исходах флаг обязан быть булевым, иначе он бесполезен.
        drv = _moving()
        self.assertIs(self.a.compare_trajectories(drv, list(drv))["follows"], True)
        self.assertIs(
            self.a.compare_trajectories(drv, _frozen())["follows"], False)

    def test_mismatched_lengths_are_refused_loudly(self):
        # Выравнивание один к одному — предпосылка всей арифметики. Молча
        # обрезать длинную последовательность значит сдвинуть траекторию и
        # получить «не то движение» на верном клипе.
        with self.assertRaises(ValueError):
            self.a.compare_trajectories(_moving(16), _moving(12))


@unittest.skipUnless(HAVE_DEPS, "numpy not installed (live extra)")
class TheBarsAreGuardedAndJustified(unittest.TestCase):
    def setUp(self):
        from ball_reel import action

        self.a = action

    def test_the_direction_bar_sits_inside_the_measured_gap(self):
        """Порог обязан лежать МЕЖДУ замеренными годными и бракованными.

        ИЗМЕРЕНО на 71 настоящем кадре (`action.controls`): годные случаи дают
        1.0 / 0.997 / 0.782, бракованные -0.158 / 0.039 / 0.149 / 0.176.
        Числа литеральные намеренно — тест, берущий вход из константы, которую
        сторожит, поедет вместе с ней и промолчит.
        """
        self.assertGreater(self.a.DIRECTION_MIN, 0.176)
        self.assertLess(self.a.DIRECTION_MIN, 0.782)

    def test_the_window_the_product_uses_clears_the_frame_floor(self):
        # Продуктовый клип — 16 кадров. Порог, который его же не пропускает,
        # выключил бы ось целиком и молча.
        from ball_reel.animate import CONTEXT_FRAMES

        self.assertLessEqual(self.a.MIN_TRAJECTORY_FRAMES, CONTEXT_FRAMES)

    def test_a_short_sequence_below_the_floor_is_refused(self):
        # Обратная сторона: порог обязан что-то запрещать, иначе он украшение.
        n = self.a.MIN_TRAJECTORY_FRAMES - 1
        got = self.a.compare_trajectories(_moving(n), _moving(n))
        self.assertEqual(got["verdict"], "not_measurable")

    def test_the_shared_joint_floor_matches_the_pose_module(self):
        # Второй способ узнать, сколько суставов достаточно, — дефект.
        self.assertGreaterEqual(self.a.MIN_SHARED_JOINTS, 4)

    def test_the_stride_is_at_least_the_measured_minimum(self):
        # ИЗМЕРЕНО: на шаге 1 годный клип с опозданием на кадр даёт direction
        # +0.02 — неотличимо от брака. На шаге 3 он же даёт +0.78.
        self.assertGreaterEqual(self.a.STEP_STRIDE, 3)


@unittest.skipUnless(HAVE_DEPS, "numpy not installed (live extra)")
class WhatThisAxisCannotSee(unittest.TestCase):
    """Границы оси записаны в модуле, а не подразумеваются.

    Ось геометрическая. Она НЕ видит, что в кадре не тот объект и не та одежда
    — а именно это было на кадре в бальном платье вместе с неверным движением.
    Молчать об этом нельзя: зелёная строка «движение то» читается как «кадр
    верный».
    """

    def setUp(self):
        from ball_reel import action

        self.a = action

    def test_the_module_says_it_does_not_see_objects_or_clothes(self):
        doc = " ".join((self.a.__doc__ or "").split())
        self.assertTrue(
            "одежд" in doc.lower() or "объект" in doc.lower(),
            "модуль обязан называть, чего он не видит")

    def test_the_direction_bar_is_marked_unverified_on_real_generation(self):
        # Порог выбран на производных от донора, а не на настоящем клипе с
        # карты. Пока живого прогона не было, это НЕПРОВЕРЕНО — и так и должно
        # быть написано, иначе число выдаётся за замер.
        import inspect

        src = inspect.getsource(self.a)
        self.assertIn("НЕПРОВЕРЕНО", src)


@unittest.skipUnless(HAVE_DEPS, "numpy not installed (live extra)")
class TheDrivingMapIsResolvedNotGuessed(unittest.TestCase):
    """Корень путей ИЩЕТСЯ, а не выводится из глубины манифеста.

    Дефект был тихим ровно там, где дороже всего. Резолвер брал «два уровня
    вверх от манифеста»: для `kit/conditions/manifest.json` это корень
    репозитория и всё работало, а для `demo/kit/conditions/manifest.json` —
    каталог `demo`, и все 16 путей переставали находиться. Ось действия при
    этом НЕ ПАДАЕТ: она честно отвечает «не смогли измерить». То есть на
    демо-ките — том самом, что лежит в git ради воспроизводимости, —
    единственная проверка «то ли движение совершено» замолчала бы, и выглядело
    бы это штатным третьим исходом.
    """

    def setUp(self):
        import tempfile

        from ball_reel import action

        self.a = action
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = __import__("pathlib").Path(self.tmp.name)

    def _kit(self, depth: int):
        """Кит на заданной глубине, с путями от корня, как их пишет skeleton."""
        import json

        root = self.dir
        rel = "/".join(["lvl"] * depth) if depth else ""
        kit = root / rel if rel else root
        (kit / "conditions").mkdir(parents=True, exist_ok=True)
        (kit / "driving").mkdir(parents=True, exist_ok=True)
        table = {}
        for i in range(3):
            (kit / "driving" / f"{i:04d}.jpg").write_bytes(b"x")
            (kit / "conditions" / f"{i:04d}.png").write_bytes(b"x")
            table[f"{i:04d}"] = f"{rel + '/' if rel else ''}driving/{i:04d}.jpg"
        mp = kit / "conditions" / "manifest.json"
        mp.write_text(json.dumps({"driving_frames": table}), encoding="utf-8")
        return mp, root

    def test_the_map_resolves_at_every_depth_not_just_the_lucky_one(self):
        import os

        for depth in (0, 1, 2):
            with self.subTest(depth=depth):
                mp, root = self._kit(depth)
                cwd = os.getcwd()
                os.chdir(root)
                self.addCleanup(os.chdir, cwd)
                got = self.a.driving_frames_for(mp)
                self.assertEqual(len(got), 3)
                for p in got:
                    self.assertTrue(os.path.exists(p), p)
                os.chdir(cwd)

    def test_an_explicit_root_still_wins(self):
        mp, root = self._kit(0)
        got = self.a.driving_frames_for(mp, root=root)
        self.assertTrue(all(str(root) in p for p in got))

    def test_frames_that_exist_nowhere_are_refused_loudly(self):
        # Молчаливое «не смогли измерить» тут недопустимо: разница между
        # «кадры не распакованы» и «движение неизмеримо» — это разные починки.
        import json

        mp = self.dir / "manifest.json"
        mp.write_text(json.dumps(
            {"driving_frames": {"0000": "нет-такого/0000.jpg"}}),
            encoding="utf-8")
        with self.assertRaises(FileNotFoundError) as e:
            self.a.driving_frames_for(mp)
        self.assertIn("не находятся", str(e.exception))
