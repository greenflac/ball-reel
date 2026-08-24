"""Кадры моста: три исхода, опора у обоих концов, кадры строго между ними.

ПОЧЕМУ ЗДЕСЬ СИНТЕТИЧЕСКИЕ КАДРЫ И ПОДСТАВНОЙ ПОТОК (Т4). Оптический поток на
кадре 720x1278 стоит десятки миллисекунд, а на сьюте из сотни прогонов —
секунды; но дело не в скорости. Настоящий поток — ВНЕШНИЙ ИНСТРУМЕНТ, и тест,
зовущий его, краснеет от смены версии `opencv`, а не от нашей ошибки. Поэтому
здесь «кадр» — это массив, у которого все пиксели равны номеру кадра, а поток
подставлен функцией с известным ответом. Пиксельная проверка на боевом
материале — дело прогона (П3), и её числа лежат в отчёте смены, а не здесь.

ОЖИДАЕМОЕ — ЛИТЕРАЛ (Т2). Ни одно число не берётся из проверяемого модуля:
потолок написан цифрой 8, «годно» написано словом. Импорт
`fork_bridge.SHIFT_MAX_STEPS` в ожидание поехал бы вместе с константой и
промолчал ровно тогда, когда её и надо сторожить.

НЕГАТИВНЫЙ КОНТРОЛЬ С ОБЕИХ СТОРОН (И5): вход, где модуль обязан сказать «не
годно» (стык шире потолка, кадры разного размера), и вход, где он обязан
пропустить (стык величиной с обычный межкадровый шаг).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .. import fork_bridge as fb

# Боевые числа (`assets/README.md` и отчёт смены), проставлены руками:
# стык петли 114..162 — 5.00 px при обычном шаге окрестности 2.61 px.
SLOW, FAST = 1.0, 10.0     # px на кадр в двух окрестностях синтетического клипа
SLOW_UNTIL = 15            # кадры до этого номера — медленная окрестность


def frames(n: int = 30):
    """`n` кадров-массивов; у кадра k все пиксели равны k."""
    return [np.full((8, 8, 3), k, dtype=np.uint8) for k in range(n)]


def fake_flow(a, b):
    """Поток с известным ответом: длина вектора = |Δномера| * скорость места.

    Смешанная пара получает БЫСТРУЮ скорость намеренно — так воспроизводится
    настоящий дефект, из-за которого мусор проходил как годный.
    """
    va, vb = int(a[0, 0, 0]), int(b[0, 0, 0])
    speed = FAST if (va >= SLOW_UNTIL or vb >= SLOW_UNTIL) else SLOW
    mag = abs(vb - va) * speed
    # Поле НЕ РАВНОМЕРНОЕ, и это не украшение фикстуры: в настоящем ролике
    # фон неподвижен и занимает большую часть кадра. Равномерное поле делает
    # все процентили одинаковыми, то есть фикстура не может отличить 95-й от
    # среднего — и мутация процентиля переживает сьют.
    out = np.zeros((*a.shape[:2], 2), dtype=np.float32)
    out[-2:, :, 0] = mag
    return out


def reader_of(imgs):
    return lambda path: imgs[int(Path(path).stem)]


def paths_of(n: int):
    return [f"{k:05d}.png" for k in range(n)]


class TheMomentsLieStrictlyBetweenTheEnds(unittest.TestCase):

    def test_a_bridge_of_two_splits_the_gap_in_three(self):
        self.assertEqual(fb.moments(2), [1 / 3, 2 / 3])

    def test_no_moment_ever_lands_on_a_donor(self):
        # Деление на k вместо k+1 положило бы первый кадр моста ровно на
        # донора — то есть удвоило бы кадр там, где мы убираем удвоение.
        for k in (1, 2, 5, 8, 13):
            with self.subTest(k=k):
                got = fb.moments(k)
                self.assertEqual(len(got), k)
                self.assertNotIn(0.0, got)
                self.assertNotIn(1.0, got)
                self.assertTrue(all(0.0 < t < 1.0 for t in got))

    def test_a_bridge_of_none_is_an_empty_list_not_an_error(self):
        self.assertEqual(fb.moments(0), [])

    def test_a_negative_bridge_is_refused(self):
        with self.assertRaises(ValueError):
            fb.moments(-1)

    def test_a_fractional_count_is_refused(self):
        with self.assertRaises(TypeError):
            fb.moments(2.5)


class TheShiftIsJudgedAgainstTheClipsOwnStep(unittest.TestCase):

    def test_a_shift_the_size_of_one_ordinary_step_passes(self):
        got = fb.shift_verdict(5.0, 2.61)
        self.assertEqual(got["outcome"], "годно")

    def test_a_shift_far_over_the_ceiling_is_refused(self):
        # Негативный контроль сверху: 96.71 px при опоре 2.81 — это 34.4 шага.
        got = fb.shift_verdict(96.71, 2.81)
        self.assertEqual(got["outcome"], "не годно")

    def test_the_ceiling_decides_at_exactly_eight_steps(self):
        # Литерал 8, не импорт (Т2).
        self.assertEqual(fb.shift_verdict(8.0, 1.0)["outcome"], "годно")
        self.assertEqual(fb.shift_verdict(8.001, 1.0)["outcome"], "не годно")

    def test_without_a_reference_step_the_answer_is_neither_yes_nor_no(self):
        got = fb.shift_verdict(5.0, None)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIsNone(got["steps"])

    def test_a_motionless_neighbourhood_is_unmeasured_not_instant(self):
        got = fb.shift_verdict(5.0, 0.0)
        self.assertEqual(got["outcome"], "не смогли проверить")

    def test_the_refusal_names_both_the_pixels_and_the_steps(self):
        note = fb.shift_verdict(96.71, 2.81)["note"]
        self.assertIn("96.71", note)
        self.assertIn("34.42", note)


class TheReferenceIsTakenFromBothEndsAndTheStricterWins(unittest.TestCase):
    """ДЕФЕКТ, НАЙДЕННЫЙ СВОИМ ЖЕ НЕГАТИВНЫМ КОНТРОЛЕМ (И5).

    Пока опора считалась только вокруг одного конца, пара кадров 114 и 300 —
    разъехавшихся на 96.71 px, то есть заведомый мусор — проходила как
    «годно», код 0. Кадр 300 лежит в быстром упражнении (16.47 px на шаг), и
    мусор делился на большое число. Вокруг кадра 114 шаг 2.81 px, и то же
    смещение даёт 34.4 шага — отказ.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.imgs = frames()
        self.kw = {"flow": fake_flow, "reader": reader_of(self.imgs),
                   "writer": lambda path, img: Path(path).write_bytes(b"x")}

    def test_a_gap_between_a_slow_and_a_fast_place_is_refused(self):
        # Концы 2 и 25: разница номеров 23, скорость смешанной пары быстрая
        # (10), то есть смещение 230 px. Опора вокруг 2 — медленная (1 px).
        got = fb.build(paths_of(30), 2, 25, 2, self.root / "out", **self.kw)
        self.assertEqual(got["outcome"], "не годно")
        self.assertEqual(got["written"], 0)

    def test_the_note_names_both_references(self):
        got = fb.build(paths_of(30), 2, 25, 2, self.root / "out", **self.kw)
        seen = [n for step, _, n in got["checks"] if step == "смещение"]
        self.assertTrue(seen)
        self.assertIn("вокруг 2", seen[0])
        self.assertIn("вокруг 25", seen[0])

    def test_the_larger_reference_would_have_let_this_gap_through(self):
        """Вход, различающий «меньшую опору» и «большую».

        Концы 13 и 17 лежат по разные стороны границы окрестностей. Смещение
        4 * 10 = 40 px. Опора вокруг 13 — медленная (1 px), вокруг 17 —
        быстрая (10 px). По меньшей это 40 шагов и отказ; по большей — 4 шага
        и «годно», то есть ровно тот мусор, который проходил до починки.
        """
        got = fb.build(paths_of(30), 13, 17, 2, self.root / "out", **self.kw)
        self.assertEqual(got["outcome"], "не годно")

    def test_the_reference_of_the_far_end_alone_would_have_let_it_through(self):
        # То же, но опора «только у конца i»: у 17 она быстрая и пропустила бы.
        got = fb.build(paths_of(30), 13, 17, 2, self.root / "out", **self.kw)
        seen = [n for step, _, n in got["checks"] if step == "смещение"][0]
        self.assertIn("вокруг 13 1.0 px", seen)
        self.assertIn("вокруг 17 10.0 px", seen)

    def test_the_same_gap_taken_the_other_way_round_is_also_refused(self):
        """Зеркало предыдущего: тот же стык, концы поменяны местами.

        Прибор обязан быть БЕЗРАЗЛИЧЕН к тому, какой конец быстрый. Без этого
        теста мутация «брать опору только у конца j» переживает сьют: на
        входе 13->17 она даёт тот же отказ, что и правило, и отличается
        только здесь.
        """
        got = fb.build(paths_of(30), 17, 13, 2, self.root / "out", **self.kw)
        self.assertEqual(got["outcome"], "не годно")

    def test_a_gap_inside_one_slow_place_passes(self):
        # Негативный контроль снизу (И5): концы 3 и 6, обе окрестности
        # медленные, смещение 3 px при опоре 1 px = 3 шага, потолок 8.
        got = fb.build(paths_of(30), 3, 6, 2, self.root / "out", **self.kw)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["written"], 2)

    def test_a_gap_inside_one_fast_place_passes(self):
        got = fb.build(paths_of(30), 20, 24, 2, self.root / "out", **self.kw)
        self.assertEqual(got["outcome"], "годно")


class TheBuildAnswersWithThreeOutcomes(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.imgs = frames()
        self.kw = {"flow": fake_flow, "reader": reader_of(self.imgs),
                   "writer": lambda path, img: Path(path).write_bytes(b"x")}

    def test_no_bridge_asked_is_good_and_writes_nothing(self):
        got = fb.build(paths_of(30), 6, 3, 0, self.root / "out", **self.kw)
        self.assertEqual(got["outcome"], "годно")
        self.assertEqual(got["written"], 0)
        self.assertFalse((self.root / "out").exists())

    def test_an_unreadable_end_is_refused(self):
        kw = dict(self.kw, reader=lambda path: None)
        got = fb.build(paths_of(30), 6, 3, 2, self.root / "out", **kw)
        self.assertEqual(got["outcome"], "не годно")

    def test_ends_of_different_size_are_refused(self):
        odd = list(self.imgs)
        odd[6] = np.full((4, 4, 3), 6, dtype=np.uint8)
        kw = dict(self.kw, reader=reader_of(odd))
        got = fb.build(paths_of(30), 6, 3, 2, self.root / "out", **kw)
        self.assertEqual(got["outcome"], "не годно")

    def test_an_end_outside_the_material_is_refused(self):
        got = fb.build(paths_of(30), 6, 99, 2, self.root / "out", **self.kw)
        self.assertEqual(got["outcome"], "не годно")

    def test_the_names_are_the_decoders_own(self):
        # Е1: дальше по пути кадры собираются sorted(glob('*.png')), и своя
        # схема имён сортировалась бы иначе ровно там, где никто не смотрит.
        got = fb.build(paths_of(30), 3, 6, 3, self.root / "out", **self.kw)
        self.assertEqual([p.name for p in got["paths"]],
                         ["00000.png", "00001.png", "00002.png"])

    def test_the_refusal_costs_not_a_single_written_frame(self):
        # П2: отказ по смещению стоит одного расчёта потока, а не count штук
        # плюс запись.
        wrote = []
        kw = dict(self.kw, writer=lambda path, img: wrote.append(path))
        fb.build(paths_of(30), 2, 25, 5, self.root / "out", **kw)
        self.assertEqual(wrote, [])

    def test_the_report_says_the_face_is_not_taken_from_these_frames(self):
        # Правило измерено (просадка 35.6% при разбросе 3.5%) и обязано быть
        # сказано вслух: молчание прочтут как «кадр моста обычный во всём».
        got = fb.build(paths_of(30), 3, 6, 2, self.root / "out", **self.kw)
        self.assertIn("Лицо с этих кадров НЕ БЕРЁТСЯ", got["note"])

    def test_the_three_exit_codes_are_distinct(self):
        self.assertEqual(sorted(fb.EXIT_BY_OUTCOME.values()), [0, 1, 2])


class TheTweenNeverLandsOnADonor(unittest.TestCase):

    def test_the_ends_themselves_are_refused_as_moments(self):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.full((4, 4, 3), 9, dtype=np.uint8)
        for t in (0.0, 1.0, -0.1, 1.5):
            with self.subTest(t=t):
                with self.assertRaises(ValueError):
                    fb.tween(a, b, t, flow=fake_flow)


class TheShiftIsTakenFromTheMovingPartNotTheAverage(unittest.TestCase):
    """95-й процентиль, а не среднее и не максимум.

    Фон в наших роликах неподвижен и занимает большую часть кадра. Среднее
    утопило бы в нём движение конечностей, ради которого всё и меряется;
    максимум поймал бы одиночный выброс потока. Поле построено так, что все
    три ответа РАЗНЫЕ: 90 нулей, девять единиц и одна десятка.
    """

    def field(self):
        f = np.zeros((10, 10, 2), dtype=np.float32)
        flat = f[..., 0].ravel()
        flat[90:99] = 1.0
        flat[99] = 10.0
        return flat.reshape(10, 10)[..., None].repeat(2, axis=2) * [1, 0]

    def test_the_moving_part_is_what_is_reported(self):
        # Литералы (Т2): у этого поля среднее место 0.0, 95-й процентиль 1.0,
        # максимум 10.0.
        self.assertEqual(fb.flow_shift(self.field()), 1.0)

    def test_it_is_not_the_median(self):
        self.assertNotEqual(fb.flow_shift(self.field()), 0.0)

    def test_it_is_not_the_maximum(self):
        self.assertNotEqual(fb.flow_shift(self.field()), 10.0)


class TheLocalStepIsAMedianOverNeighbours(unittest.TestCase):

    def test_a_slow_place_measures_slow(self):
        got = fb.local_step(paths_of(30), 5, flow=fake_flow,
                            reader=reader_of(frames()))
        self.assertEqual(got["outcome"], "годно")
        self.assertAlmostEqual(got["step"], 1.0, places=6)

    def test_a_fast_place_measures_fast(self):
        got = fb.local_step(paths_of(30), 25, flow=fake_flow,
                            reader=reader_of(frames()))
        self.assertAlmostEqual(got["step"], 10.0, places=6)

    def test_nothing_readable_is_unmeasured_not_zero(self):
        got = fb.local_step(paths_of(30), 5, flow=fake_flow,
                            reader=lambda path: None)
        self.assertEqual(got["outcome"], "не смогли проверить")
        self.assertIsNone(got["step"])
        self.assertEqual(got["pairs"], 0)


if __name__ == "__main__":
    unittest.main()
