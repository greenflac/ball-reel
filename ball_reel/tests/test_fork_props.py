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
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [{"name": "мяч", "role": fp.REPLACE,
                                   "bbox": [100, 100, 180, 180]}])
            got = fp.sequence(p, [_person(), _person(), _person()])
        self.assertEqual((got["applied"], got["no_marking"]), (1, 2))
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


if __name__ == "__main__":
    unittest.main()
