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
from ball_reel import fork_identity as fi


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

    def test_the_module_level_bar_is_the_one_that_decides(self):
        """Т1 по КОНСТАНТЕ, а не по аргументу.

        Прежние два теста подменяют порог параметром — они проверяют развилку,
        но не то, что модуль слушается своей константы. Разница не
        умозрительная: пока умолчание стояло прямо в сигнатуре, подмена
        `fc.MIN_SCORE` не меняла ни одного пикселя, и сторож константы был
        украшением.
        """
        original = fc.MIN_SCORE
        pts = _points(score=0.5)
        try:
            fc.MIN_SCORE = 0.9
            img = fc.render(pts, fc.FACE_CHANNEL, 200, 400)
            self.assertEqual(self._ink(img), 0,
                             "порог поднят на модуле, а точки рисуются — "
                             "константа не участвует в решении")
            fc.MIN_SCORE = 0.1
            self.assertGreater(self._ink(fc.render(pts, fc.FACE_CHANNEL,
                                                   200, 400)), 0)
            self.assertEqual(fc.visible(pts, fc.FACE_CHANNEL), 68)
            fc.MIN_SCORE = 0.9
            self.assertEqual(fc.visible(pts, fc.FACE_CHANNEL), 0,
                             "visible() не слушается константы модуля")
        finally:
            fc.MIN_SCORE = original

    def _ink(self, img) -> int:
        px = img.load()
        return sum(1 for y in range(img.height) for x in range(img.width)
                   if px[x, y] != fc.GROUND)

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

    def test_the_face_side_is_checked_by_the_same_check_as_the_frame(self):
        """«512 кратно 16» — числом через проверку, а не глазами по константе."""
        got = fc.check_face_geometry()
        self.assertTrue(got["ok"], got["note"])
        self.assertEqual(512 % 16, 0)
        self.assertIn("лицевой канал", got["note"])

    def test_a_face_side_off_the_multiple_is_refused(self):
        self.assertFalse(fc.check_face_geometry(500)["ok"])
        self.assertFalse(fc.check_face_geometry(0)["ok"])

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


def _face_cluster(score: float = 0.9, seen: int = 68) -> list:
    """133 точки, где лицо — компактное облако 64×64, а тело далеко от него.

    Такая раскладка нужна именно лицевому каналу: он КРОПИТ по габариту лица, и
    проверять кроп на точках, размазанных по всему холсту, значило бы проверять
    не кроп. `seen` — сколько лицевых точек наблюдаемо, остальные под порогом.
    """
    out = [(600.0, 600.0, score)] * fc.WHOLEBODY_JOINTS
    face = list(fc.group_indices("face"))
    for k, i in enumerate(face):
        x = 100.0 + (k % 8) * 64.0 / 7.0
        y = 40.0 + (k // 8) * 64.0 / 8.0
        out[i] = (x, y, score if k < seen else fc.MIN_SCORE - 0.01)
    return out


class TheFaceChannelIsActuallyFiveHundredTwelve(unittest.TestCase):
    """Главное расхождение потока: `FACE_SIDE` была объявлена и не применялась.

    Нода приводит `face_video` к 512×512 сама, `crop="center"`
    (`nodes_wan.py:1208`), поэтому «нарисовали на холсте кадра» означает, что
    квадрат вырежет НЕ ЛИЦО, а середину кадра, и при кадрировке во весь рост
    голова в него не попадёт вовсе.
    """

    def setUp(self):
        self.pts = _face_cluster()

    def _ink_box(self, img):
        px = img.load()
        pts = [(x, y) for y in range(img.height) for x in range(img.width)
               if px[x, y] != fc.GROUND]
        self.assertTrue(pts, "лицевой канал пуст")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def test_the_rendered_face_is_five_hundred_twelve_square(self):
        """Т2: литерал 512, а не `fc.FACE_SIDE` — иначе тест поедет с кодом."""
        img = fc.render_face(self.pts)
        self.assertEqual(img.size, (512, 512))

    def test_the_side_comes_from_the_constant_and_is_not_hardcoded(self):
        """Т1, сторона вверх: константа поднята — картинка обязана вырасти."""
        original = fc.FACE_SIDE
        try:
            fc.FACE_SIDE = 1024
            self.assertEqual(fc.render_face(self.pts).size, (1024, 1024),
                             "512 зашито в отрисовку мимо константы")
            self.assertEqual(fc.blank_face().size, (1024, 1024))
        finally:
            fc.FACE_SIDE = original

    def test_a_side_off_the_multiple_of_sixteen_is_refused(self):
        """Т1, сторона вниз: негодная сторона обязана ронять отрисовку."""
        original = fc.FACE_SIDE
        try:
            fc.FACE_SIDE = 500
            with self.assertRaises(ValueError) as caught:
                fc.render_face(self.pts)
            self.assertIn("не кратна 16", str(caught.exception))
            with self.assertRaises(ValueError):
                fc.blank_face()
        finally:
            fc.FACE_SIDE = original

    def test_the_face_fills_the_frame_instead_of_sitting_in_a_corner(self):
        """Смысл кропа: 68 точек занимают кадр, а не десяток пикселей в нём.

        Ожидание — литерал 341: при запасе 0.25 сторона кропа равна габариту
        лица × 1.5, значит габарит занимает 512 / 1.5 = 341 px. Если бы лицо
        рисовалось на холсте кадра и ужималось нодой, здесь было бы 86 px
        (замер на demo/hero.png) — а при кадрировке во весь рост около 40.
        """
        x0, y0, x1, y1 = self._ink_box(fc.render_face(self.pts))
        self.assertAlmostEqual(x1 - x0, 341, delta=8)
        self.assertAlmostEqual(y1 - y0, 341, delta=8)

    def test_the_margin_is_guarded_in_both_directions(self):
        """Т1 по `FACE_MARGIN`: запас снят — лицо во весь кадр, задран — точка."""
        original = fc.FACE_MARGIN
        try:
            fc.FACE_MARGIN = 0.0
            x0, _, x1, _ = self._ink_box(fc.render_face(self.pts))
            self.assertGreater(x1 - x0, 500,
                               "запас снят, а поле вокруг лица осталось")
            fc.FACE_MARGIN = 2.0
            x0, _, x1, _ = self._ink_box(fc.render_face(self.pts))
            self.assertLess(x1 - x0, 120,
                            "запас впятеро, а лицо всё того же размера — "
                            "значит запас в кроп не участвует")
        finally:
            fc.FACE_MARGIN = original

    def test_the_face_is_centred_so_the_nodes_centre_crop_hits_it(self):
        x0, y0, x1, y1 = self._ink_box(fc.render_face(self.pts))
        self.assertAlmostEqual((x0 + x1) / 2, 256, delta=8)
        self.assertAlmostEqual((y0 + y1) / 2, 256, delta=8)

    def test_no_body_point_gets_into_the_face_five_twelve(self):
        """Негативный контроль (И5): точка тела ВНУТРИ лицевого бокса.

        Далёкая точка тела не попала бы в кадр и без всякого разделения
        каналов — такой тест доказывал бы кроп, а не непересечение.
        """
        pts = list(self.pts)
        pts[fc.WHOLEBODY_GROUPS["l_hand"][0]] = (132.0, 100.0, 0.9)
        img = fc.render_face(pts)
        box = fc.face_box(pts)
        k = 512 / (box[2] - box[0])
        x = int((132.0 - box[0]) * k)
        y = int((100.0 - box[1]) * k)
        self.assertEqual(img.load()[x, y], fc.GROUND,
                         "точка кисти нарисована в лицевом канале")


class TheFaceBoxSaysNoRatherThanGuessing(unittest.TestCase):
    """Р1: «точек мало» — третий исход, а не кроп наугад."""

    def test_a_full_face_gives_a_square_box(self):
        box = fc.face_box(_face_cluster())
        self.assertIsNotNone(box)
        self.assertAlmostEqual(
            box[2] - box[0], box[3] - box[1], places=6,
            msg="бокс не квадратный — нода дорежет его сама, crop=center, "
                "и решит за нас, что отрезать")

    def test_the_minimum_is_guarded_in_both_directions(self):
        """Т1 по `FACE_BOX_MIN_POINTS`: 17 видно — бокс, 16 — отказ."""
        self.assertEqual(fc.FACE_BOX_MIN_POINTS, 17)
        self.assertIsNotNone(fc.face_box(_face_cluster(seen=17)))
        self.assertIsNone(fc.face_box(_face_cluster(seen=16)),
                          "бокс построен по 16 точкам — кроп промахнётся мимо "
                          "лица, а модель прочтёт промах как лицо")

    def test_points_collapsed_into_one_spot_are_refused(self):
        pts = [(50.0, 50.0, 0.9)] * fc.WHOLEBODY_JOINTS
        self.assertIsNone(fc.face_box(pts))

    def test_an_unreadable_face_raises_instead_of_a_silent_black_frame(self):
        with self.assertRaises(ValueError) as caught:
            fc.render_face(_face_cluster(seen=0))
        self.assertIn("0", str(caught.exception))
        self.assertIn("судить нечем", str(caught.exception))

    def test_the_blank_frame_is_black_and_the_right_size(self):
        img = fc.blank_face()
        self.assertEqual(img.size, (512, 512))
        px = img.load()
        self.assertTrue(all(px[x, y] == fc.GROUND
                            for y in range(0, 512, 7) for x in range(0, 512, 7)))

    def test_the_pair_keeps_two_different_canvases(self):
        pair = fc.render_pair(_face_cluster(), 480, 848)
        self.assertEqual(pair[fc.FACE_CHANNEL].size, (512, 512))
        self.assertEqual(pair[fc.BODY_CHANNEL].size, (480, 848),
                         "тело ушло в 512 — нода ждёт его в геометрии кадра")

    def test_the_pair_survives_an_unreadable_face_with_a_black_frame(self):
        pair = fc.render_pair(_face_cluster(seen=0), 480, 848)
        self.assertEqual(pair[fc.FACE_CHANNEL].size, (512, 512))


class TheCropIsSharedAcrossTheSequence(unittest.TestCase):
    """Покадровый кроп дышит и читается как наезд камеры, которого не было."""

    def test_the_union_covers_every_box_and_stays_square(self):
        got = fc.union_box([(0.0, 0.0, 10.0, 10.0), (20.0, 4.0, 40.0, 24.0)])
        self.assertLessEqual(got[0], 0.0)
        self.assertLessEqual(got[1], 0.0)
        self.assertGreaterEqual(got[2], 40.0)
        self.assertGreaterEqual(got[3], 24.0)
        self.assertAlmostEqual(got[2] - got[0], got[3] - got[1], places=6)

    def test_no_box_at_all_gives_none_not_an_empty_square(self):
        self.assertIsNone(fc.union_box([None, None]))
        self.assertIsNone(fc.union_box([]))

    def test_a_shared_box_holds_the_face_still_while_it_moves(self):
        """Одно и то же лицо, сдвинутое на 30 px, обязано и сдвинуться в 512.

        Если бы бокс считался по кадру, лицо стояло бы в центре обоих кадров, и
        движение головы пропало бы из условия — а вместе с ним и наоборот:
        неподвижная голова начала бы дёргаться от дрожания детектора.
        """
        a = _face_cluster()
        b = [(x + 30.0, y, s) for x, y, s in a]
        box = fc.union_box([fc.face_box(a), fc.face_box(b)])
        ink = []
        for pts in (a, b):
            img = fc.render_face(pts, box=box)
            px = img.load()
            xs = [x for y in range(img.height) for x in range(img.width)
                  if px[x, y] != fc.GROUND]
            ink.append(min(xs))
        self.assertGreater(ink[1] - ink[0], 40,
                           "сдвиг лица не доехал до 512 — кроп едет вместе с "
                           "лицом и движение головы стёрто")


class TheSequenceKeepsFramesAligned(unittest.TestCase):
    """Кадр без человека всё равно записывается: нода сопоставляет по порядку.

    Веса DWPose не нужны — `wholebody_points` подменяется. Подменять честно:
    проверяется сборка последовательности, а не ONNX.
    """

    def _run(self, tmp, answers):
        from unittest import mock
        from PIL import Image

        paths = []
        for i in range(len(answers)):
            p = Path(tmp) / f"in{i}.png"
            Image.new("RGB", (480, 848), (10, 10, 10)).save(p)
            paths.append(p)
        out = Path(tmp) / "out"
        with mock.patch.object(fc, "wholebody_points",
                               side_effect=list(answers)):
            return fc.render_sequence(paths, out), out

    def test_a_frame_without_a_person_still_produces_both_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            got, out = self._run(tmp, [_face_cluster(), None, _face_cluster()])
            for c in fc.CHANNELS:
                names = sorted(p.name for p in (out / c).glob("*.png"))
                self.assertEqual(names, ["00000.png", "00001.png", "00002.png"],
                                 f"канал {c} потерял кадр — мимика и движение "
                                 f"разъедутся на всём остатке ролика")
        self.assertEqual((got["total"], got["rendered"]), (3, 2))
        self.assertEqual(got["no_person"], ["in1.png"])

    def test_the_face_frames_are_five_twelve_and_the_body_frames_are_not(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            _, out = self._run(tmp, [_face_cluster(), None])
            for name in ("00000.png", "00001.png"):
                with Image.open(out / fc.FACE_CHANNEL / name) as im:
                    self.assertEqual(im.size, (512, 512))
                with Image.open(out / fc.BODY_CHANNEL / name) as im:
                    self.assertEqual(im.size, (480, 848))

    def test_a_clean_run_is_pass_a_holed_one_is_fail_and_empty_is_unmeasured(self):
        """Р1: три исхода достижимы, и каждый достигается своим входом."""
        with tempfile.TemporaryDirectory() as tmp:
            clean, _ = self._run(tmp, [_face_cluster(), _face_cluster()])
        with tempfile.TemporaryDirectory() as tmp:
            holed, _ = self._run(tmp, [_face_cluster(), None])
        with tempfile.TemporaryDirectory() as tmp:
            blind, _ = self._run(tmp, [_face_cluster(seen=0)])
        self.assertEqual(clean["outcome"], fi.PASS)
        self.assertEqual(holed["outcome"], fi.FAIL)
        self.assertEqual(blind["outcome"], fi.UNMEASURED,
                         "лицо не прочиталось нигде, а вердикт не «не смогли»")

    def test_the_outcomes_are_the_shared_ones_and_not_local_strings(self):
        """Е1: три исхода — один словарь на проект, а не свой у каждого потока."""
        self.assertIs(fc.PASS, fi.PASS)
        self.assertIs(fc.FAIL, fi.FAIL)
        self.assertIs(fc.UNMEASURED, fi.UNMEASURED)


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
