"""Разметка предметов: что она обещает и чего обещать не может.

Главное, что здесь сторожится, — что «сохранить предмет» не превращается в
пустое слово. Предмет тоньше блока блокификации сохранить нельзя ничем, и
разметка такого предмета обязана читаться как «не смогли», а не как успех:
оператор потратил время, и соврать ему здесь дороже всего.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ball_reel import fork_mask, fork_props as fp
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED


def _person(h=256, w=256):
    m = np.zeros((h, w), dtype=bool)
    m[40:220, 80:170] = True
    return m


def _marking(objects, **kw):
    data = {"schema": fp.POINTS_SCHEMA, "template": "йога-01",
            "frames": {"0": objects}}
    data.update(kw)
    return data


class TheMarkingIsPerTemplateNeverPerClient(unittest.TestCase):
    """Разметка на клиента = оператор на каждый заказ, то есть отмена продукта."""

    def test_a_marking_tied_to_a_client_falls_over(self):
        with self.assertRaises(ValueError) as caught:
            fp.refuse_per_client({"template": "йога-01", "client": "ivan.png"})
        self.assertIn("ОДИН РАЗ на темплейт", str(caught.exception))

    def test_the_client_photo_field_is_refused_too(self):
        with self.assertRaises(ValueError):
            fp.refuse_per_client({"template": "a", "client_photo": "b.png"})

    def test_a_marking_without_a_template_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            fp.refuse_per_client({})
        self.assertIn("без имени темплейта", str(caught.exception))

    def test_a_proper_marking_passes(self):
        """Негативный контроль (И5): вход, где запрет обязан молчать."""
        fp.refuse_per_client({"template": "йога-01"})


class TheRoleIsAlwaysNamed(unittest.TestCase):

    def test_there_is_no_default_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            p.write_text(json.dumps(_marking([{"name": "мяч",
                                               "bbox": [0, 0, 10, 10]}]),
                                    ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                fp.load_marking(p)
        self.assertIn("Умолчания нет", str(caught.exception))

    def test_the_two_roles_are_opposite(self):
        """Одна вычитает из маски, другая добавляет. Перепутать нельзя."""
        person = _person()
        obj = np.zeros_like(person)
        obj[100:180, 100:160] = True
        kept = fp.apply(person, [{"name": "мяч", "role": fp.KEEP,
                                  "mask": obj}])
        given = fp.apply(person, [{"name": "мяч", "role": fp.REPLACE,
                                   "mask": obj}])
        self.assertLess(kept["share_after"], given["share_after"],
                        "сохранение и замена дали одинаковую маску — роли "
                        "перепутаны местами или не действуют")

    def test_an_unknown_role_falls_over(self):
        person = _person()
        with self.assertRaises(ValueError):
            fp.apply(person, [{"name": "x", "role": "может быть",
                               "mask": np.zeros_like(person)}])


class ASmallPropCannotBePromised(unittest.TestCase):
    """Предел тракта, а не осторожность прибора."""

    def _prop(self, side):
        m = np.zeros((256, 256), dtype=bool)
        m[100:100 + side, 100:100 + side] = True
        return m

    def test_a_prop_thinner_than_a_block_is_unmeasured_not_kept(self):
        got = fp.keepable(self._prop(fork_mask.BLOCK - 1))
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("блокификация затянет его обратно", got["note"])

    def test_a_prop_of_a_whole_block_can_be_promised(self):
        """Мутация в другую сторону: планка не должна браковать всё подряд."""
        got = fp.keepable(self._prop(fork_mask.BLOCK))
        self.assertEqual(got["outcome"], PASS)

    def test_the_note_carries_both_sides_in_pixels(self):
        got = fp.keepable(self._prop(20))
        self.assertEqual((got["width_px"], got["height_px"]), (20, 20))
        self.assertIn("20x20", got["note"])

    def test_the_thin_side_decides_not_the_area(self):
        """Длинная тонкая полоска — площадь большая, удержать нельзя."""
        m = np.zeros((256, 256), dtype=bool)
        m[100:104, 20:240] = True          # 220 x 4
        self.assertEqual(fp.keepable(m)["outcome"], UNMEASURED)

    def test_an_empty_marking_is_unmeasured_too(self):
        self.assertEqual(fp.keepable(np.zeros((32, 32), bool))["outcome"],
                         UNMEASURED)

    def test_the_bar_moves_with_the_block_size(self):
        """Т1: планка выведена из fork_mask.BLOCK, а не вписана числом."""
        saved = fp.MIN_KEEPABLE_PX
        try:
            fp.MIN_KEEPABLE_PX = 256
            self.assertEqual(fp.keepable(self._prop(64))["outcome"], UNMEASURED,
                             "планка подменена, а вердикт не изменился")
        finally:
            fp.MIN_KEEPABLE_PX = saved

    def test_an_unkeepable_prop_makes_the_whole_apply_unmeasured(self):
        person = _person()
        got = fp.apply(person, [{"name": "эклер", "role": fp.KEEP,
                                 "mask": self._prop(12)}])
        self.assertEqual(got["outcome"], UNMEASURED,
                         "«не смогли обещать» свёрнуто в успех — Р1 нарушено")
        self.assertIn("не смогли обещать 1", got["note"])


class TheReportIsNumbersNotAMask(unittest.TestCase):

    def test_the_area_taken_and_left_are_printed(self):
        person = _person()
        obj = np.zeros_like(person)
        obj[60:140, 170:230] = True
        got = fp.apply(person, [{"name": "мяч", "role": fp.REPLACE,
                                 "mask": obj}])
        self.assertGreater(got["share_after"], got["share_before"])
        self.assertIn("Маска была", got["note"])
        self.assertIn("под охраной осталось", got["note"])

    def test_giving_away_the_whole_frame_is_a_failure(self):
        """Маска на весь кадр — это не «всё получилось», это снятая охрана."""
        person = _person()
        whole = np.ones_like(person)
        got = fp.apply(person, [{"name": "всё", "role": fp.REPLACE,
                                 "mask": whole}])
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("ось протечки больше почти ничего не сторожит",
                      " ".join(p["note"] for p in got["objects"]
                               if p["outcome"] == FAIL) or got["note"])

    def test_the_guard_bar_can_go_red_and_green(self):
        """Т1 в обе стороны: планка охраняемой площади сторожится."""
        person = _person()
        obj = np.zeros_like(person)
        obj[0:200, 0:256] = True
        saved = fp.GUARDED_MIN_SHARE
        try:
            fp.GUARDED_MIN_SHARE = 0.0
            self.assertNotEqual(
                fp.apply(person, [{"name": "x", "role": fp.REPLACE,
                                   "mask": obj}])["outcome"], FAIL)
            fp.GUARDED_MIN_SHARE = 0.99
            self.assertEqual(
                fp.apply(person, [{"name": "x", "role": fp.REPLACE,
                                   "mask": obj}])["outcome"], FAIL)
        finally:
            fp.GUARDED_MIN_SHARE = saved

    def test_a_prop_mask_of_the_wrong_size_falls_over(self):
        with self.assertRaises(ValueError) as caught:
            fp.apply(_person(), [{"name": "x", "role": fp.KEEP,
                                  "mask": np.zeros((8, 8), bool)}])
        self.assertIn("против маски персонажа", str(caught.exception))


class TheFileContractIsTheEditorsOwn(unittest.TestCase):
    """Нода не нужна: её вычислительный вклад — разбор JSON.

    Формат держится совместимым нарочно, чтобы разметку можно было снять и
    браузерным редактором тоже, не таща экспериментальную ноду в путь клиента.
    """

    def test_the_bbox_order_is_xyxy_as_in_the_read_source(self):
        m = fp.bbox_mask([10, 20, 40, 60], 100, 100)
        ys, xs = np.where(m)
        self.assertEqual((xs.min(), xs.max() + 1), (10, 40))
        self.assertEqual((ys.min(), ys.max() + 1), (20, 60))

    def test_a_bbox_outside_the_frame_is_clipped_not_crashing(self):
        m = fp.bbox_mask([-50, -50, 500, 500], 64, 64)
        self.assertTrue(m.all())

    def test_a_wrong_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            p.write_text(json.dumps({"schema": "чужая", "template": "t"}),
                         encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                fp.load_marking(p)
        self.assertIn("не тот или устарел", str(caught.exception))

    def test_a_missing_file_says_which(self):
        with self.assertRaises(FileNotFoundError) as caught:
            fp.load_marking("/нет/такого.json")
        self.assertIn("такого.json", str(caught.exception))


class TheSequenceCountsFramesNotFlags(unittest.TestCase):

    def _write(self, tmp, objects):
        p = Path(tmp) / "m.json"
        p.write_text(json.dumps(_marking(objects), ensure_ascii=False),
                     encoding="utf-8")
        return p

    def test_frames_without_marking_are_counted_separately(self):
        """Один ключевой кадр — это ОДИН кадр, а не три.

        Прежде эти два кадра считались «разметки нет», и это читалось как
        «нечего делать». Теперь они «вне размеченного диапазона»: предмет
        размечен, но продолжать его рамку за последний ключевой кадр нельзя.
        """
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [{"name": "мяч", "role": fp.REPLACE,
                                   "bbox": [100, 100, 180, 180]}])
            got = fp.sequence(p, [_person(), _person(), _person()])
        self.assertEqual((got["applied"], got["outside_range"],
                          got["no_marking"]), (1, 2, 0))
        self.assertIn("вне размеченного диапазона 2", got["note"])
        self.assertEqual(got["outcome"], UNMEASURED,
                         "кадры без выводимой рамки свёрнуты в успех")

    def test_a_file_with_no_objects_at_all_is_no_marking_not_out_of_range(self):
        """Негативный контроль к предыдущему: два разных состояния — два счёта."""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [])
            got = fp.sequence(p, [_person(), _person()])
        self.assertEqual((got["outside_range"], got["no_marking"]), (0, 2))
        self.assertIn("разметки нет на 2", got["note"])

    def test_an_unkeepable_prop_keeps_the_sequence_unmeasured(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [{"name": "эклер", "role": fp.KEEP,
                                   "bbox": [100, 100, 110, 110]}])
            got = fp.sequence(p, [_person()])
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual(got["weak"], 1)

    def test_the_template_name_reaches_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [{"name": "мяч", "role": fp.REPLACE,
                                   "bbox": [90, 90, 200, 200]}])
            got = fp.sequence(p, [_person()])
        self.assertEqual(got["template"], "йога-01")
        self.assertIn("одна на всех клиентов", got["note"])

    def test_every_frame_gets_a_mask_even_without_marking(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [])
            got = fp.sequence(p, [_person(), _person()])
        self.assertEqual(len(got["masks"]), 2)
        for m in got["masks"]:
            self.assertEqual(np.asarray(m).shape, (256, 256))


class TheMasksCanArriveAsFilesBecauseThatIsHowTheyArrive(unittest.TestCase):
    """Стык, на котором первая версия падала в KeyError.

    `fork_mask.sequence` кладёт маски на диск и массивов не возвращает. Модуль,
    умеющий только массивы, печатал бы провал разметки там, где ей ничто не
    мешало отработать, — то есть врал бы в отчёте о чужой работе.
    """

    def test_a_mask_read_from_a_file_equals_the_same_mask_in_memory(self):
        from PIL import Image

        person = _person()
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "00000.png"
            Image.fromarray((person * 255).astype("uint8"), mode="L").save(f)
            from_file = fp._as_mask(f)
        self.assertTrue(np.array_equal(from_file, fp._as_mask(person)),
                        "маска из файла разошлась с той же маской в памяти — "
                        "порог бинаризации применяется не в одном месте")

    def test_the_sequence_accepts_paths(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "00000.png"
            Image.fromarray((_person() * 255).astype("uint8"),
                            mode="L").save(f)
            m = Path(tmp) / "m.json"
            m.write_text(json.dumps(_marking(
                [{"name": "мяч", "role": fp.REPLACE,
                  "bbox": [90, 90, 200, 200]}]), ensure_ascii=False),
                encoding="utf-8")
            got = fp.sequence(m, [f])
        self.assertEqual(got["applied"], 1, got["note"])


class TheSweepRunsTheMarkingStep(unittest.TestCase):
    """Модуль, не подключённый к пути, — мёртвый код в отчёте."""

    def test_the_step_exists_in_the_declared_order(self):
        from ball_reel import fork_run

        self.assertIn("предметы", fork_run.STEPS)

    def test_it_stands_after_masks_and_before_the_graph(self):
        from ball_reel import fork_run

        order = list(fork_run.STEPS)
        self.assertLess(order.index("маски"), order.index("предметы"),
                        "разметка правит маску — она не может идти раньше неё")
        self.assertLess(order.index("предметы"), order.index("граф"),
                        "граф получает маску — разметка обязана быть до него")

    def test_no_marking_is_unmeasured_and_says_what_that_means(self):
        """Отсутствие разметки — не успех: предмет в руке при этом остаётся
        внутри маски по решению чужого сегментатора, а не по нашему."""
        import numpy as np
        from PIL import Image
        from ball_reel import fork_run

        with tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp) / "p.png"
            Image.fromarray(
                (np.random.rand(64, 64, 3) * 255).astype("uint8")).save(photo)
            got = fork_run.run(photo, [], Path(tmp) / "out")
        step = next(s for s in got["steps"] if s["step"] == "предметы")
        self.assertEqual(step["outcome"], UNMEASURED)
        self.assertIn("others", step["note"])


def _keyframed(frames, **kw):
    """Файл разметки с произвольными ключевыми кадрами."""
    data = {"schema": fp.POINTS_SCHEMA, "template": "йога-01",
            "frames": {str(k): v for k, v in frames.items()}}
    data.update(kw)
    return data


def _write_json(tmp, data, name="m.json"):
    p = Path(tmp) / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


class TheBoxIsDerivedNotDrawnThreeHundredTimes(unittest.TestCase):
    """Ключевые кадры вместо рамки на каждый кадр.

    Прежняя версия читала `per_frame.get(i)` — на 300-кадровом шаблоне это 300
    рамок руками, то есть механизм, которым никто не воспользуется. Здесь
    сторожится ровно то, что рамка ВЫВОДИТСЯ между ключевыми и НЕ ВЫДУМЫВАЕТСЯ
    за их пределами.
    """

    def _track(self):
        return fp.tracks(_keyframed({
            0: [{"name": "мяч", "role": fp.KEEP, "bbox": [0, 0, 40, 40]}],
            10: [{"name": "мяч", "role": fp.KEEP, "bbox": [100, 200, 140, 240]}],
        }))["мяч"]

    def test_a_marked_frame_says_it_was_marked(self):
        got = fp.box_at(self._track(), 0)
        self.assertEqual(got["source"], fp.FROM_KEYFRAME)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["bbox"], [0.0, 0.0, 40.0, 40.0])

    def test_a_frame_between_two_keys_is_derived_with_the_right_numbers(self):
        """Т2: ожидаемое — литерал, посчитанный руками, а не тем же кодом."""
        got = fp.box_at(self._track(), 5)
        self.assertEqual(got["source"], fp.FROM_INTERPOLATION)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["bbox"], [50.0, 100.0, 90.0, 140.0])

    def test_both_edges_and_the_middle_of_the_range(self):
        """Т3: фикстуры с обоих краёв диапазона и из середины."""
        t = self._track()
        self.assertEqual(fp.box_at(t, 0)["bbox"], [0.0, 0.0, 40.0, 40.0])
        self.assertEqual(fp.box_at(t, 2)["bbox"], [20.0, 40.0, 60.0, 80.0])
        self.assertEqual(fp.box_at(t, 10)["bbox"], [100.0, 200.0, 140.0, 240.0])

    def test_outside_the_marked_range_it_is_unmeasured_not_invented(self):
        """ГЛАВНЫЙ тест дыры: за диапазоном — «не смогли», а не рамка."""
        t = self._track()
        for frame in (-1, 11, 300):
            got = fp.box_at(t, frame)
            self.assertEqual(got["outcome"], UNMEASURED, f"кадр {frame}")
            self.assertIsNone(got["bbox"],
                              f"кадр {frame}: рамка выдумана экстраполяцией")
            self.assertEqual(got["source"], fp.OUT_OF_RANGE)
            self.assertIn("экстраполировать НЕЛЬЗЯ", got["note"])

    def test_the_note_names_the_marked_range(self):
        self.assertIn("0..10", fp.box_at(self._track(), 42)["note"])

    def test_a_single_keyframe_covers_exactly_one_frame(self):
        t = fp.tracks(_keyframed(
            {7: [{"name": "мяч", "role": fp.KEEP, "bbox": [0, 0, 40, 40]}]}))["мяч"]
        self.assertEqual(fp.box_at(t, 7)["source"], fp.FROM_KEYFRAME)
        self.assertEqual(fp.box_at(t, 8)["outcome"], UNMEASURED)
        self.assertEqual(fp.box_at(t, 6)["outcome"], UNMEASURED)

    def test_a_track_without_keyframes_is_unmeasured(self):
        self.assertEqual(
            fp.box_at({"name": "мяч", "keys": {}}, 0)["outcome"], UNMEASURED)

    def test_the_derived_box_actually_moves_between_frames(self):
        """Негативный контроль (И5): интерполяция обязана ШЕВЕЛИТЬСЯ.

        Если бы она возвращала копию предыдущей ключевой рамки, все проверки
        выше прошли бы одинаково, а маска стояла бы на месте.
        """
        t = self._track()
        boxes = [tuple(fp.box_at(t, i)["bbox"]) for i in range(11)]
        self.assertEqual(len(set(boxes)), 11,
                         "выведенные рамки совпали — интерполяции нет, есть "
                         "копирование ключевой рамки")


class TheSequenceInterpolatesBetweenKeyframes(unittest.TestCase):

    def _file(self, tmp):
        return _write_json(tmp, _keyframed({
            0: [{"name": "мяч", "role": fp.REPLACE, "bbox": [60, 60, 120, 120]}],
            4: [{"name": "мяч", "role": fp.REPLACE, "bbox": [130, 60, 190, 120]}],
        }))

    def test_five_frames_from_two_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fp.sequence(self._file(tmp), [_person() for _ in range(5)])
        self.assertEqual((got["keyed"], got["interpolated"],
                          got["out_of_range"]), (2, 3, 0), got["note"])
        self.assertEqual(got["applied"], 5)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertIn("выведено между ключевыми 3", got["note"])

    def test_a_sixth_frame_is_out_of_range_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fp.sequence(self._file(tmp), [_person() for _ in range(6)])
        self.assertEqual(got["out_of_range"], 1)
        self.assertEqual(got["outside_range"], 1)
        self.assertEqual(got["outcome"], UNMEASURED,
                         "кадр за диапазоном свёрнут в успех — Р1 нарушено")

    def test_the_masks_differ_between_frames_because_the_prop_moves(self):
        """Негативный контроль: одинаковые маски = интерполяция не доехала."""
        with tempfile.TemporaryDirectory() as tmp:
            got = fp.sequence(self._file(tmp), [_person() for _ in range(5)])
        first, last = np.asarray(got["masks"][0]), np.asarray(got["masks"][4])
        self.assertFalse(np.array_equal(first, last),
                         "маска первого и последнего кадра совпала — рамка "
                         "не двигалась, хотя ключевые кадры разные")

    def test_the_block_limit_survives_interpolation(self):
        """Предел уже замерен и обходить его нельзя: предмет уже блока
        неудерживаем и на выведенных кадрах тоже."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_json(tmp, _keyframed({
                0: [{"name": "эклер", "role": fp.KEEP,
                     "bbox": [100, 100, 110, 110]}],
                3: [{"name": "эклер", "role": fp.KEEP,
                     "bbox": [106, 100, 116, 110]}]}))
            got = fp.sequence(p, [_person() for _ in range(4)])
        self.assertEqual(got["weak"], 4, got["note"])
        self.assertEqual(got["outcome"], UNMEASURED)


class TheBoxMustNotJumpBecauseThatMeansAnotherPerson(unittest.TestCase):
    """Сторож тождества, а не плавности.

    У вендора (`ComfyUI-WanAnimatePreprocess : nodes.py:120-124`) рамка на
    каждом кадре — первая детекция, какую вернул детектор. При двух-трёх людях
    порядок детекций меняется, и рамка перепрыгивает на другого человека.
    Здесь прыжок обязан становиться числом.
    """

    def _smooth(self, n=10, step=2, side=40):
        return [(i, [i * step, 0, i * step + side, side]) for i in range(n)]

    def test_a_smooth_track_passes_and_says_how_many_pairs(self):
        """Негативный контроль (И5): вход, где сторож обязан молчать."""
        got = fp.jump_verdict(self._smooth())
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual((got["checked"], got["violations"]), (9, 0))

    def test_a_teleport_is_caught(self):
        boxes = [(0, [0, 0, 40, 40]), (1, [200, 0, 240, 40])]
        got = fp.jump_verdict(boxes)
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual((got["checked"], got["violations"]), (1, 1))
        self.assertIn("ведём, судя по всему, разные объекты", got["note"])

    def test_a_doubling_area_is_caught_even_without_moving(self):
        boxes = [(0, [100, 100, 140, 140]), (1, [70, 70, 170, 170])]
        got = fp.jump_verdict(boxes)
        self.assertEqual(got["outcome"], FAIL)
        self.assertGreater(got["jumps"][0]["area_ratio"], 2.0)

    def test_zero_pairs_is_not_success(self):
        """Р2: ноль нарушений при нуле проверок — «не смогли»."""
        got = fp.jump_verdict([(0, [0, 0, 40, 40])])
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual((got["checked"], got["violations"]), (0, 0))
        self.assertIn("НЕ «прыжков нет»", got["note"])

    def test_a_hole_is_not_compared_across(self):
        """Через пропуск рамка могла смениться — сравнивать нечего."""
        got = fp.jump_verdict([(0, [0, 0, 40, 40]), (1, None),
                               (2, [200, 0, 240, 40])])
        self.assertEqual((got["checked"], got["unmeasured"]), (0, 1))
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_a_degenerate_box_is_unmeasured_not_a_jump(self):
        got = fp.jump_verdict([(0, [10, 10, 10, 10]), (1, [10, 10, 50, 50])])
        self.assertEqual(got["unmeasured"], 1)
        self.assertEqual(got["checked"], 0)

    def test_the_gap_between_keyframes_is_taken_into_account(self):
        """Смещение делится на число кадров: за 20 кадров то же расстояние
        прыжком не является."""
        far = [(0, [0, 0, 40, 40]), (20, [200, 0, 240, 40])]
        self.assertEqual(fp.jump_verdict(far)["outcome"], PASS)

    def test_the_center_bar_is_mutated_in_both_directions(self):
        """Т1: строже — краснеет исправный вход; слабее — зеленеет прыжок."""
        smooth, jump = self._smooth(), [(0, [0, 0, 40, 40]),
                                        (1, [200, 0, 240, 40])]
        saved = fp.JUMP_CENTER_SHARE
        try:
            fp.JUMP_CENTER_SHARE = 0.0
            self.assertEqual(fp.jump_verdict(smooth)["outcome"], FAIL,
                             "планка снята в ноль, а ровный трек всё ещё "
                             "проходит — планку никто не сторожит")
            fp.JUMP_CENTER_SHARE = 1e9
            self.assertEqual(fp.jump_verdict(jump)["outcome"], PASS,
                             "планка поднята до неба, а телепорт всё ещё "
                             "ловится — ловит его что-то другое")
        finally:
            fp.JUMP_CENTER_SHARE = saved

    def test_the_area_bar_is_mutated_in_both_directions(self):
        grew = [(0, [100, 100, 140, 140]), (1, [70, 70, 170, 170])]
        creeping = [(0, [100, 100, 140, 140]), (1, [100, 100, 141, 141])]
        saved = fp.JUMP_AREA_RATIO
        try:
            fp.JUMP_AREA_RATIO = 1.0
            self.assertEqual(fp.jump_verdict(creeping)["outcome"], FAIL,
                             "планка площади снята, а рост рамки не ловится")
            fp.JUMP_AREA_RATIO = 1e9
            self.assertEqual(fp.jump_verdict(grew)["outcome"], PASS,
                             "планка площади поднята, а рамка всё ещё "
                             "объявлена прыгнувшей")
        finally:
            fp.JUMP_AREA_RATIO = saved

    def test_the_bars_are_arguments_not_only_globals(self):
        jump = [(0, [0, 0, 40, 40]), (1, [200, 0, 240, 40])]
        self.assertEqual(
            fp.jump_verdict(jump, center_share=100.0)["outcome"], PASS)


class TheProtagonistIsChosenOnceAndCarried(unittest.TestCase):
    """Слой, которого нет ни у кого: детерминированный выбор человека.

    Вендор пишет «single-person videos ONLY» именно потому, что выбора у него
    нет. Здесь протагонист называется один раз на сборке темплейта и ведётся
    сквозь кадры теми же ключевыми кадрами, что и предметы.
    """

    def _lead_file(self, tmp, second=None):
        frames = {
            0: [{"name": "актриса", "role": fp.PROTAGONIST,
                 "bbox": [80, 40, 170, 220]}],
            9: [{"name": "актриса", "role": fp.PROTAGONIST,
                 "bbox": [90, 40, 180, 220]}],
        }
        if second is not None:
            for k, box in second.items():
                frames.setdefault(k, []).append(
                    {"name": "прохожий", "role": fp.BYSTANDER, "bbox": box})
        return _write_json(tmp, _keyframed(frames))

    def test_the_role_is_not_keep_and_not_replace(self):
        self.assertNotIn(fp.PROTAGONIST, fp.MASK_ROLES)
        self.assertNotIn(fp.BYSTANDER, fp.MASK_ROLES)
        self.assertIn(fp.PROTAGONIST, fp.ROLES)
        self.assertIn(fp.BYSTANDER, fp.ROLES)

    def test_the_protagonist_role_never_touches_the_mask(self):
        person = _person()
        with self.assertRaises(ValueError) as caught:
            fp.apply(person, [{"name": "актриса", "role": fp.PROTAGONIST,
                               "mask": np.zeros_like(person)}])
        self.assertIn("маску правят только эти", str(caught.exception))

    def test_two_protagonists_are_refused_at_load(self):
        """Выбор обязан быть детерминированным — иначе это дефект вендора."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_json(tmp, _keyframed({0: [
                {"name": "а", "role": fp.PROTAGONIST, "bbox": [0, 0, 9, 9]},
                {"name": "б", "role": fp.PROTAGONIST, "bbox": [9, 9, 20, 20]}]}))
            with self.assertRaises(ValueError) as caught:
                fp.load_marking(p)
        self.assertIn("вести можно ровно одного", str(caught.exception))

    def test_one_protagonist_is_carried_through_the_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fp.protagonist_track(fp.load_marking(self._lead_file(tmp)), 10)
        self.assertEqual(got["name"], "актриса")
        self.assertEqual((got["keyed"], got["interpolated"],
                          got["out_of_range"]), (2, 8, 0))
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["checked"], 9)

    def test_an_unmarked_protagonist_is_unmeasured_not_fine(self):
        """Третий исход: «не размечен» — это не «человек в кадре один»."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_json(tmp, _keyframed({0: [
                {"name": "мяч", "role": fp.KEEP, "bbox": [0, 0, 40, 40]}]}))
            got = fp.protagonist_track(fp.load_marking(p), 3)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["name"])
        self.assertIn("перепрыгнет с человека на человека", got["note"])

    def test_a_protagonist_that_jumps_to_another_person_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_json(tmp, _keyframed({
                0: [{"name": "актриса", "role": fp.PROTAGONIST,
                     "bbox": [10, 40, 70, 220]}],
                1: [{"name": "актриса", "role": fp.PROTAGONIST,
                     "bbox": [190, 40, 250, 220]}]}))
            got = fp.protagonist_track(fp.load_marking(p), 2)
            seq = fp.sequence(p, [_person(), _person()])
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["violations"], 1)
        self.assertEqual(seq["outcome"], FAIL,
                         "прыжок протагониста не доехал до вердикта прогона")
        self.assertIn("РАМКА ПРЫГНУЛА", seq["note"])

    def test_marking_a_protagonist_does_not_change_the_person_mask(self):
        """Негативный контроль: роль выбирает, а не правит маску."""
        with tempfile.TemporaryDirectory() as tmp:
            bare = _write_json(tmp, _keyframed({0: [], 9: []}), "bare.json")
            lead = self._lead_file(tmp)
            a = fp.sequence(bare, [_person() for _ in range(10)])
            b = fp.sequence(lead, [_person() for _ in range(10)])
        for i, (x, y) in enumerate(zip(a["masks"], b["masks"])):
            self.assertTrue(np.array_equal(np.asarray(x), np.asarray(y)),
                            f"кадр {i}: разметка протагониста сдвинула маску")


class TheDirectingRuleIsReflectedInACheck(unittest.TestCase):
    """Правило съёмки кодом не решается, но наблюдается.

    Второстепенные — на фоне, без контакта и без прохода перед протагонистом.
    Если рамка второстепенного накрыла рамку протагониста, он прошёл перед ним,
    и дальше уже неважно, что мы решили: детектор получит двух слипшихся людей.
    """

    def _file(self, tmp, bystander_boxes):
        frames = {
            0: [{"name": "актриса", "role": fp.PROTAGONIST,
                 "bbox": [80, 40, 170, 220]}],
            3: [{"name": "актриса", "role": fp.PROTAGONIST,
                 "bbox": [80, 40, 170, 220]}],
        }
        for k, box in bystander_boxes.items():
            frames.setdefault(k, []).append(
                {"name": "прохожий", "role": fp.BYSTANDER, "bbox": box})
        return _write_json(tmp, _keyframed(frames))

    def test_a_bystander_in_the_background_passes(self):
        """Негативный контроль: вход, где правило обязано молчать."""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._file(tmp, {0: [10, 40, 60, 200], 3: [10, 40, 60, 200]})
            got = fp.directing_rule(fp.load_marking(p), 4)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual((got["checked"], got["violations"]), (4, 0))

    def test_a_bystander_walking_in_front_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._file(tmp, {0: [10, 40, 60, 200], 3: [90, 40, 160, 200]})
            got = fp.directing_rule(fp.load_marking(p), 4)
        self.assertEqual(got["outcome"], FAIL)
        self.assertGreaterEqual(got["violations"], 1)
        self.assertIn("проход перед протагонистом", got["note"])

    def test_without_bystanders_it_is_unmeasured_not_clean(self):
        """Р2: ноль нарушений при нуле проверок — не успех."""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._file(tmp, {})
            got = fp.directing_rule(fp.load_marking(p), 4)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual((got["checked"], got["violations"]), (0, 0))
        self.assertIn("не успех", got["note"])

    def test_the_overlap_bar_is_mutated_in_both_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            far = self._file(tmp, {0: [10, 40, 60, 200], 3: [10, 40, 60, 200]})
            near = self._file(tmp, {0: [10, 40, 60, 200],
                                    3: [90, 40, 160, 200]}, )
            saved = fp.BYSTANDER_OVERLAP_MAX
            try:
                fp.BYSTANDER_OVERLAP_MAX = -1.0
                self.assertEqual(
                    fp.directing_rule(fp.load_marking(far), 4)["outcome"],
                    FAIL, "планка снята, а чистая сцена всё ещё проходит")
                fp.BYSTANDER_OVERLAP_MAX = 1.0
                self.assertEqual(
                    fp.directing_rule(fp.load_marking(near), 4)["outcome"],
                    PASS, "планка поднята, а проход всё ещё ловится")
            finally:
                fp.BYSTANDER_OVERLAP_MAX = saved


class TheLeakCanBeAddressedByName(unittest.TestCase):
    """Именованные области: чтобы протечка была адресом, а не одним числом."""

    def _file(self, tmp):
        return _write_json(tmp, _keyframed({
            0: [{"name": "актриса", "role": fp.PROTAGONIST,
                 "bbox": [80, 40, 170, 220]},
                {"name": "прохожий слева", "role": fp.BYSTANDER,
                 "bbox": [0, 0, 64, 64]},
                {"name": "прохожий справа", "role": fp.BYSTANDER,
                 "bbox": [192, 0, 256, 64]}],
            3: [{"name": "актриса", "role": fp.PROTAGONIST,
                 "bbox": [80, 40, 170, 220]},
                {"name": "прохожий слева", "role": fp.BYSTANDER,
                 "bbox": [0, 0, 64, 64]},
                {"name": "прохожий справа", "role": fp.BYSTANDER,
                 "bbox": [192, 0, 256, 64]}]}))

    def test_each_bystander_gets_its_own_named_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fp.leak_areas(fp.load_marking(self._file(tmp)), 1, 256, 256)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(sorted(a["name"] for a in got["areas"]),
                         ["прохожий слева", "прохожий справа"])
        self.assertEqual(got["named"], 2)
        for area in got["areas"]:
            self.assertEqual(np.asarray(area["mask"]).shape, (256, 256))

    def test_the_protagonist_is_not_a_leak_area(self):
        """Негативный контроль: адресуются ТОЛЬКО второстепенные."""
        with tempfile.TemporaryDirectory() as tmp:
            got = fp.leak_areas(fp.load_marking(self._file(tmp)), 1, 256, 256)
        self.assertNotIn("актриса", [a["name"] for a in got["areas"]])

    def test_out_of_range_areas_are_unmeasured_not_missing_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fp.leak_areas(fp.load_marking(self._file(tmp)), 9, 256, 256)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual((got["named"], got["unmeasured"]), (0, 2))

    def test_the_numbers_land_on_the_right_name(self):
        """Без имени «модель тронула прохожего» неотличимо от «перекрасила
        стену». Здесь тронута ровно одна область, и число обязано быть у неё."""
        with tempfile.TemporaryDirectory() as tmp:
            areas = fp.leak_areas(fp.load_marking(self._file(tmp)),
                                  0, 256, 256)["areas"]
        driving = np.zeros((256, 256, 3), dtype="uint8")
        output = driving.copy()
        output[0:64, 0:64] = 255            # тронут только левый прохожий
        got = fp.address_leak(output, driving, areas)
        by_name = {r["name"]: r for r in got["areas"]}
        self.assertEqual(got["measured"], 2, got["note"])
        self.assertAlmostEqual(by_name["прохожий слева"]["mean"], 1.0, places=3)
        self.assertAlmostEqual(by_name["прохожий справа"]["mean"], 0.0,
                               places=6)
        self.assertIn("прохожий слева", got["note"])

    def test_an_area_too_small_to_measure_says_so(self):
        """Третий исход приезжает из fork_leak и не сворачивается в ноль."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_json(tmp, _keyframed({0: [
                {"name": "далёкий", "role": fp.BYSTANDER,
                 "bbox": [0, 0, 16, 16]}]}))
            areas = fp.leak_areas(fp.load_marking(p), 0, 256, 256)["areas"]
        frame = np.zeros((256, 256, 3), dtype="uint8")
        got = fp.address_leak(frame, frame, areas)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual((got["measured"], got["unmeasured"]), (0, 1))

    def test_the_metric_is_borrowed_not_reimplemented(self):
        """Е1: своё расхождение здесь было бы вторым способом узнать то же."""
        import inspect

        src = inspect.getsource(fp.address_leak)
        self.assertIn("fork_leak.outside_divergence", src)


class TheNameIsIdentityNotALabel(unittest.TestCase):

    def test_an_object_without_a_name_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_json(tmp, _keyframed({0: [
                {"role": fp.KEEP, "bbox": [0, 0, 40, 40]}]}))
            with self.assertRaises(ValueError) as caught:
                fp.load_marking(p)
        self.assertIn("тождество между ключевыми кадрами",
                      str(caught.exception))

    def test_one_name_cannot_carry_two_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_json(tmp, _keyframed({
                0: [{"name": "мяч", "role": fp.KEEP, "bbox": [0, 0, 40, 40]}],
                5: [{"name": "мяч", "role": fp.REPLACE,
                     "bbox": [0, 0, 40, 40]}]}))
            with self.assertRaises(ValueError) as caught:
                fp.load_marking(p)
        self.assertIn("одно имя — один объект", str(caught.exception))

    def test_a_bbox_of_three_numbers_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_json(tmp, _keyframed({0: [
                {"name": "мяч", "role": fp.KEEP, "bbox": [0, 0, 40]}]}))
            with self.assertRaises(ValueError) as caught:
                fp.load_marking(p)
        self.assertIn("вместо четырёх", str(caught.exception))

    def test_a_frame_key_that_is_not_a_number_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_json(tmp, _keyframed({"первый": [
                {"name": "мяч", "role": fp.KEEP, "bbox": [0, 0, 40, 40]}]}))
            with self.assertRaises(ValueError) as caught:
                fp.load_marking(p)
        self.assertIn("не номер кадра", str(caught.exception))

    def test_a_proper_keyframed_file_loads(self):
        """Негативный контроль ко всем четырём запретам сразу."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_json(tmp, _keyframed({
                0: [{"name": "мяч", "role": fp.KEEP, "bbox": [0, 0, 40, 40]},
                    {"name": "актриса", "role": fp.PROTAGONIST,
                     "bbox": [80, 40, 170, 220]},
                    {"name": "прохожий", "role": fp.BYSTANDER,
                     "bbox": [0, 0, 64, 64]}],
                9: [{"name": "мяч", "role": fp.KEEP,
                     "bbox": [10, 10, 50, 50]}]}))
            data = fp.load_marking(p)
        self.assertEqual(data["_objects_checked"], 4)
        self.assertEqual(fp.protagonist_name(data), "актриса")



if __name__ == "__main__":
    unittest.main()
