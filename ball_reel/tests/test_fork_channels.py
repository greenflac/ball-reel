"""Поток D: два канала условий обязаны быть РАЗНЫМИ, и это проверяется.

Главная проверка здесь не «функция не упала», а НЕПЕРЕСЕЧЕНИЕ: в канале позы не
должно быть ни одной точки лица, в канале лица — ни одной точки тела. Дефект,
против которого написан модуль, тихий: смешанная картинка отрисуется, сохранится
и поедет в модель, у которой под неё нет входа.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ball_reel import fork_channels as fc


def _points(n: int = fc.WHOLEBODY_JOINTS, score: float = 0.9,
            spread: int = 400) -> list:
    """133 точки, разложенные так, чтобы каждая группа села в СВОЮ область.

    Раскладка по областям нужна, чтобы «нарисовано не то» было видно по
    пикселям, а не только по индексам: если канал лица нарисует точку тела,
    чернила окажутся в чужой полосе холста.
    """
    out = []
    for i in range(n):
        band = 0
        for k, group in enumerate(fc.WHOLEBODY_GROUPS):
            lo, hi = fc.WHOLEBODY_GROUPS[group]
            if lo <= i < hi:
                band = k
        out.append((20.0 + (i % 10) * 8, 20.0 + band * 60.0, score))
    return out


class TheLayoutSumsToWhatTheModelActuallyReturns(unittest.TestCase):
    """133 — ИЗМЕРЕНО прогоном, а не переписано из статьи."""

    def test_the_groups_cover_every_joint_exactly_once(self):
        seen: list[int] = []
        for group in fc.WHOLEBODY_GROUPS:
            seen.extend(fc.group_indices(group))
        self.assertEqual(sorted(seen), list(range(fc.WHOLEBODY_JOINTS)),
                         "раскладка COCO-WholeBody не покрывает 133 сустава "
                         "ровно по разу: где-то дыра или нахлёст, и точка "
                         "лица тихо уедет в кисть")
        self.assertEqual(len(seen), len(set(seen)),
                         "две группы делят один индекс")

    def test_the_face_group_is_the_sixty_eight_points_the_question_was_about(self):
        self.assertEqual(len(fc.group_indices("face")), 68)
        self.assertEqual(fc.FACE_POINTS, 68)

    def test_each_hand_is_twenty_one_points(self):
        for group in ("l_hand", "r_hand"):
            with self.subTest(group=group):
                self.assertEqual(len(fc.group_indices(group)), 21)

    def test_an_unknown_group_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            fc.group_indices("torso")
        self.assertIn("torso", str(caught.exception))


class TheTwoChannelsDoNotOverlap(unittest.TestCase):
    """Смысл всего модуля. Пересечение — брак, а не мелочь оформления."""

    def test_no_index_is_in_both_channels(self):
        face = set(fc.channel_indices(fc.FACE_CHANNEL))
        body = set(fc.channel_indices(fc.BODY_CHANNEL))
        self.assertEqual(face & body, set(),
                         "точка попала в оба канала: у модели два ВХОДА, и "
                         "смешанное условие не подходит ни к одному")

    def test_the_body_channel_carries_no_face_point(self):
        face = set(fc.group_indices("face"))
        self.assertEqual(set(fc.channel_indices(fc.BODY_CHANNEL)) & face, set())

    def test_the_face_channel_carries_no_body_or_hand_point(self):
        others: set[int] = set()
        for group in ("body", "feet", "l_hand", "r_hand"):
            others |= set(fc.group_indices(group))
        self.assertEqual(set(fc.channel_indices(fc.FACE_CHANNEL)) & others, set())

    def test_together_the_channels_cover_everything(self):
        both = set(fc.channel_indices(fc.FACE_CHANNEL))
        both |= set(fc.channel_indices(fc.BODY_CHANNEL))
        self.assertEqual(both, set(range(fc.WHOLEBODY_JOINTS)),
                         "часть суставов не попала ни в один канал — они "
                         "посчитаны моделью и выброшены, как 116 из 133 в "
                         "dwpose.py")

    def test_an_unknown_channel_is_refused(self):
        with self.assertRaises(ValueError):
            fc.channel_indices("pose+face")

    def test_the_body_channel_keeps_the_head_and_this_is_deliberate(self):
        """Найдено ГЛАЗАМИ на артефакте (П3), закреплено тестом.

        Нос, глаза и уши (0..4) — часть ТЕЛА в раскладке COCO, и вендорский
        openpose-скелет их несёт. Выкинуть их «ради чистоты канала» значило бы
        отдать модели обезглавленный скелет. Мимики в пяти точках нет, так что
        обещание «мимика отдельным каналом» они не нарушают.
        """
        body = set(fc.channel_indices(fc.BODY_CHANNEL))
        self.assertTrue({0, 1, 2, 3, 4} <= body,
                        "голова пропала из канала тела — модель поставит её "
                        "наугад")
        self.assertTrue({0, 1, 2, 3, 4}.isdisjoint(fc.group_indices("face")),
                        "точки головы COCO попали в лицевую группу — тогда "
                        "непересечение каналов доказано не тем")


class TheRenderingPutsInkOnlyWhereItsChannelAllows(unittest.TestCase):
    """Проверка ПО ПИКСЕЛЯМ, а не по индексам: индексы уже проверены выше."""

    def setUp(self):
        self.pts = _points()
        self.w, self.h = 200, 400

    def _rows_with_ink(self, img) -> set:
        px = img.load()
        return {y for y in range(img.height) for x in range(img.width)
                if px[x, y] != fc.GROUND}

    def test_the_face_render_leaves_the_body_bands_black(self):
        img = fc.render(self.pts, fc.FACE_CHANNEL, self.w, self.h)
        rows = self._rows_with_ink(img)
        self.assertTrue(rows, "канал лица не нарисовал ничего")
        band = list(fc.WHOLEBODY_GROUPS).index("face")
        self.assertTrue(all(abs(y - (20 + band * 60)) <= 6 for y in rows),
                        f"чернила канала лица легли вне полосы лица: {sorted(rows)}")

    def test_the_body_render_leaves_the_face_band_black(self):
        img = fc.render(self.pts, fc.BODY_CHANNEL, self.w, self.h)
        px = img.load()
        band = list(fc.WHOLEBODY_GROUPS).index("face")
        face_rows = range(max(0, 20 + band * 60 - fc.DOT_RADIUS),
                          min(self.h, 20 + band * 60 + fc.DOT_RADIUS + 1))
        # Рёбра тела соединяют далёкие полосы и ЗАКОННО проходят через полосу
        # лица. Поэтому проверяется не «полоса пуста», а «в ней нет точек там,
        # где их поставил бы канал лица»: по x точки лица лежат на сетке 20+8k.
        for y in face_rows:
            for i in fc.group_indices("face"):
                x = int(self.pts[i][0])
                self.assertEqual(
                    px[x, y], fc.GROUND,
                    f"канал тела поставил чернила в точке лица {i} "
                    f"({x},{y}) — каналы смешались")

    def test_a_point_below_the_score_bar_is_not_drawn_at_all(self):
        weak = [(x, y, fc.MIN_SCORE - 0.01) for x, y, _ in self.pts]
        img = fc.render(weak, fc.FACE_CHANNEL, self.w, self.h)
        self.assertEqual(self._rows_with_ink(img), set(),
                         "ненаблюдаемая точка нарисована: условие, поставленное "
                         "по шуму, модель прочтёт как факт")

    def test_the_bar_is_inclusive_and_the_test_can_tell_the_difference(self):
        """Негативный контроль к предыдущему тесту (И5): ровно на пороге — рисуем."""
        at_bar = [(x, y, fc.MIN_SCORE) for x, y, _ in self.pts]
        img = fc.render(at_bar, fc.FACE_CHANNEL, self.w, self.h)
        self.assertTrue(self._rows_with_ink(img),
                        "точка ровно на пороге не нарисована — порог поехал")

    def test_a_broken_canvas_is_refused_rather_than_silently_empty(self):
        for w, h in ((0, 10), (10, 0), (-1, 10)):
            with self.subTest(size=(w, h)):
                with self.assertRaises(ValueError):
                    fc.render(self.pts, fc.FACE_CHANNEL, w, h)

    def test_render_pair_returns_both_and_only_both(self):
        pair = fc.render_pair(self.pts, self.w, self.h)
        self.assertEqual(set(pair), set(fc.CHANNELS))


class TheThresholdIsActuallyGuarded(unittest.TestCase):
    """Т1: подмена константы-решения в ОБЕ стороны обязана ронять тест.

    Мутация делается здесь же, а не только в реестре codeaudit, потому что
    `MIN_SCORE` тут импортированный: подменять надо то значение, которым
    пользуется этот модуль, иначе мутация проверит чужой модуль.
    """

    def test_lifting_the_bar_to_impossible_hides_every_point(self):
        pts = _points(score=0.9)
        img = fc.render(pts, fc.FACE_CHANNEL, 200, 400, min_score=1.01)
        px = img.load()
        self.assertTrue(
            all(px[x, y] == fc.GROUND
                for y in range(img.height) for x in range(img.width)),
            "порог задран выше любого балла, а точки всё равно рисуются — "
            "значит порог не участвует в решении")

    def test_dropping_the_bar_to_nothing_shows_even_noise(self):
        pts = _points(score=0.0)
        img = fc.render(pts, fc.FACE_CHANNEL, 200, 400, min_score=-1.0)
        px = img.load()
        self.assertTrue(
            any(px[x, y] != fc.GROUND
                for y in range(img.height) for x in range(img.width)),
            "порог снят, а точки с нулевым баллом всё равно не рисуются")


class VisibilityIsANumberNotAFlag(unittest.TestCase):
    """Е3: `3 из 68` и `68 из 68` — разные состояния, флаг их сливает."""

    def test_it_counts_only_its_own_channel(self):
        pts = _points(score=0.0)
        for i in fc.group_indices("face"):
            pts[i] = (pts[i][0], pts[i][1], 0.9)
        self.assertEqual(fc.visible(pts, fc.FACE_CHANNEL), 68)
        self.assertEqual(fc.visible(pts, fc.BODY_CHANNEL), 0)

    def test_a_partial_face_reads_as_partial(self):
        pts = _points(score=0.0)
        for i in list(fc.group_indices("face"))[:3]:
            pts[i] = (pts[i][0], pts[i][1], 0.9)
        self.assertEqual(fc.visible(pts, fc.FACE_CHANNEL), 3)

    def test_a_short_point_list_does_not_crash_the_count(self):
        """Модель, отдавшая меньше суставов, не должна валить подсчёт индексом."""
        self.assertEqual(fc.visible(_points(n=20), fc.FACE_CHANNEL), 0)


class TheDecoderReadsEveryJointTheModelReturns(unittest.TestCase):
    """Т2: ожидаемое — литералы, а не пересчёт формулой самого модуля."""

    def _logits(self, joints: int, xbins: int = 576, ybins: int = 768):
        import numpy as np

        x = np.zeros((joints, xbins), dtype="float32")
        y = np.zeros((joints, ybins), dtype="float32")
        for i in range(joints):
            x[i, xbins // 2] = 1.0
            y[i, ybins // 4] = 1.0
        return x, y

    def test_it_returns_one_point_per_joint_not_seventeen(self):
        x, y = self._logits(fc.WHOLEBODY_JOINTS)
        got = fc.decode_wholebody(x, y, (0.0, 0.0, 288.0, 384.0))
        self.assertEqual(len(got), 133,
                         "декодер обрезал выход до своего представления о "
                         "модели — ровно дефект, ради которого поток заведён")

    def test_a_centre_peak_lands_in_the_middle_of_the_box(self):
        x, y = self._logits(1)
        (px, py, score), = fc.decode_wholebody(x, y, (0.0, 0.0, 288.0, 384.0))
        self.assertAlmostEqual(px, 144.0, places=3)
        self.assertAlmostEqual(py, 96.0, places=3)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_the_box_offset_is_carried_into_frame_coordinates(self):
        x, y = self._logits(1)
        (px, py, _), = fc.decode_wholebody(x, y, (100.0, 200.0, 388.0, 584.0))
        self.assertAlmostEqual(px, 244.0, places=3)
        self.assertAlmostEqual(py, 296.0, places=3)

    def test_mismatched_axes_are_refused_rather_than_zipped_short(self):
        x, _ = self._logits(133)
        _, y = self._logits(17)
        with self.assertRaises(ValueError) as caught:
            fc.decode_wholebody(x, y, (0.0, 0.0, 288.0, 384.0))
        self.assertIn("133", str(caught.exception))

    def test_a_wrong_shape_is_refused(self):
        import numpy as np

        with self.assertRaises(ValueError):
            fc.decode_wholebody(np.zeros((1, 133, 576)), np.zeros((1, 133, 768)),
                                (0.0, 0.0, 288.0, 384.0))


class TheAnswerToTheTenMinuteQuestionIsRecordedInCode(unittest.TestCase):
    """Число, ради которого поток стартовал, обязано жить в проверяемом месте.

    Иначе оно останется только в отчёте, а отчёт не прогоняется.
    """

    def test_dwpose_decodes_seventeen_of_the_hundred_thirty_three(self):
        from ball_reel import dwpose

        self.assertEqual(len(dwpose.WHOLEBODY_INDEX), 17)
        self.assertEqual(fc.WHOLEBODY_JOINTS, 133)

    def test_none_of_dwposes_names_is_a_face_landmark(self):
        """Уши и глаза COCO — это НЕ 68 точек лица, и путать их дорого."""
        from ball_reel import dwpose

        self.assertLess(max(dwpose.WHOLEBODY_INDEX.values()),
                        fc.WHOLEBODY_GROUPS["face"][0],
                        "имя из dwpose попало в диапазон лицевых точек — "
                        "раскладка разъехалась с чужим модулем")


class TheGeometryContractIsCheckedOnCpuBeforeTheCard(unittest.TestCase):
    """ХЭНДОФ §3.2: кратность 16 по сторонам, кратность 4 по длине.

    Нарушение тихое: Comfy либо откажет уже после загрузки весов, либо молча
    подгонит размер и сместит всё условие. Проверка стоит микросекунды на CPU.
    """

    def test_the_agreed_geometry_passes(self):
        """480x848 на 77 кадрах — то, что стоит в стеке (ХЭНДОФ §3).

        НАЙДЕННОЕ РАСХОЖДЕНИЕ, закреплённое тестом: §3.2 хэндофа говорит «длина
        кратна 4», а §3 задаёт 77 кадров, и 77 на 4 не делится. Первоисточник
        (`WanAnimateToVideo.doc.md:20`) говорит «default: 77, step: 4» при
        минимуме 1 — то есть годны 1, 5, …, 77. Верх за файлом; проверка,
        написанная по прозе, забраковала бы штатную геометрию.
        """
        got = fc.check_geometry(480, 848, 77)
        self.assertTrue(got["ok"], got["note"])

    def test_the_vendor_default_length_is_accepted(self):
        self.assertTrue(fc.check_geometry(832, 480, 77)["ok"])

    def test_a_length_that_is_a_plain_multiple_of_four_is_refused(self):
        """Прямая проверка, что спор разрешён в пользу файла, а не прозы."""
        got = fc.check_geometry(480, 848, 76)
        self.assertFalse(got["ok"], "76 принято — значит читали «кратна 4»")
        self.assertIn("шаге 4", got["note"])

    def test_a_side_off_the_multiple_is_caught_and_the_neighbours_named(self):
        got = fc.check_geometry(481, 848, 76)
        self.assertFalse(got["ok"])
        self.assertIn("не кратна 16", got["note"])
        self.assertIn("480", got["note"])
        self.assertIn("496", got["note"])

    def test_a_length_off_the_step_is_caught_and_neighbours_named(self):
        got = fc.check_geometry(480, 848, 78)
        self.assertFalse(got["ok"])
        self.assertIn("77", got["note"])
        self.assertIn("81", got["note"])

    def test_both_sides_are_checked_not_only_the_first(self):
        got = fc.check_geometry(481, 849, 77)
        self.assertEqual(len(got["problems"]), 2,
                         "проверена только одна сторона — вторая пройдёт молча")

    def test_nonsense_sizes_are_refused_rather_than_divided(self):
        for w, h, n in ((0, 848, 77), (480, -16, 77), (480, 848, 0)):
            with self.subTest(size=(w, h, n)):
                self.assertFalse(fc.check_geometry(w, h, n)["ok"])

    def test_the_multiples_are_the_ones_the_contract_names(self):
        """Т2: литералы из §3.2, а не импорт из проверяемого модуля."""
        self.assertEqual(fc.SIDE_MULTIPLE, 16)
        self.assertEqual(fc.LENGTH_STEP, 4)
        self.assertEqual(fc.LENGTH_MIN, 1)
        self.assertEqual(fc.FACE_SIDE, 512)

    def test_the_check_is_guarded_in_both_directions(self):
        """Т1: подмена кратности обязана менять исход."""
        original = fc.SIDE_MULTIPLE
        try:
            fc.SIDE_MULTIPLE = 1
            self.assertTrue(fc.check_geometry(481, 849, 77)["ok"],
                            "кратность снята, а размер всё равно отвергнут")
            fc.SIDE_MULTIPLE = 512
            self.assertFalse(fc.check_geometry(480, 848, 77)["ok"],
                             "кратность поднята, а размер всё равно принят")
        finally:
            fc.SIDE_MULTIPLE = original


class TheSequenceReportsThreeOutcomes(unittest.TestCase):
    """Р1/Р2: «не смогли» не сворачивается ни в успех, ни в провал."""

    def test_an_empty_run_says_zero_of_zero_and_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_sequence_empty(tmp)
        self.assertEqual((got["total"], got["rendered"]), (0, 0))
        self.assertIn("0 из 0", got["note"])

    def test_the_note_separates_no_person_from_unreadable_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fork_sequence_empty(tmp)
        self.assertIn("без человека", got["note"])
        self.assertIn("нечитаемым лицом", got["note"])
        self.assertIn("судить нечем", got["note"],
                      "«не смогли измерить» подано как провал — Р1 нарушено")

    def test_both_channel_directories_are_made(self):
        with tempfile.TemporaryDirectory() as tmp:
            fork_sequence_empty(tmp)
            for c in fc.CHANNELS:
                self.assertTrue((Path(tmp) / c).is_dir())


def fork_sequence_empty(tmp: str) -> dict:
    """Прогон на пустом списке кадров: весов не требует, ветку отчёта проверяет."""
    return fc.render_sequence([], tmp)


if __name__ == "__main__":
    unittest.main()
