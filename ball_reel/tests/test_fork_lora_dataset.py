"""Набор LoRA темплейта: «лиц нет» обязано быть ИЗМЕРЕНО, а не обещано.

Главный тест здесь — негативный контроль (И5): проверка на лица обязана
СРАБАТЫВАТЬ на кропе, где лицо есть. Проверка, всегда говорящая «лиц нет»,
говорит это и про набор, полный лиц, — и такой набор потянет лицо клиента к
среднему по категории, то есть сделает ровно то, ради предотвращения чего
модуль написан.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ball_reel import fork_channels
from ball_reel import fork_lora_dataset as fld
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED

ROOT = Path(__file__).resolve().parents[2]
HERO = ROOT / "demo" / "hero.png"


def _weights_ready() -> bool:
    try:
        from ball_reel.identity_arcface import face_detail

        return face_detail(HERO) is not None
    except Exception:  # noqa: BLE001
        return False


def _points(head_y=100.0, body_top=140.0, body_bottom=600.0, score=0.9,
            hip_y=None):
    """133 точки: голова сверху, тело ниже. Числа литеральные (Т2).

    Таз задаётся ОТДЕЛЬНО и по умолчанию у самого низа: при кадрировке по пояс
    нижняя граница берётся именно по нему, и таз, случайно оказавшийся выше
    подбородка, даёт вырожденный прямоугольник. Поймано прогоном, когда
    умолчанием кадрировки стало `waist_up`.
    """
    pts = [(300.0, body_bottom, score)] * fork_channels.WHOLEBODY_JOINTS
    for i in list(fork_channels.group_indices("face")) + list(range(0, 5)):
        pts[i] = (300.0, head_y, score)
    for i in fork_channels.channel_indices("body"):
        if i >= 5:
            pts[i] = (300.0 + (i % 7) * 20, body_top + (i % 11) * 30, score)
    for i in (11, 12):
        pts[i] = (300.0, body_bottom if hip_y is None else hip_y, score)
    return pts


class TheCropStartsBelowTheWholeHead(unittest.TestCase):
    """Не «по плечам»: при наклоне плечо оказывается выше подбородка."""

    def test_the_top_edge_is_below_the_lowest_head_point(self):
        pts = _points(head_y=100.0, body_top=140.0)
        box = fld.body_box(pts, 800, 800, framing="full_body")
        self.assertIsNotNone(box)
        self.assertGreater(box[1], 100.0,
                           "верх кропа выше точки головы — лицо останется")

    def test_a_shoulder_above_the_chin_does_not_lift_the_cut(self):
        """Ровно случай наклона: плечо выше головы, а резать всё равно ниже."""
        pts = _points(head_y=300.0, body_top=140.0)
        lifted = list(pts)
        lifted[5] = (280.0, 100.0, 0.9)          # плечо ВЫШЕ подбородка
        box = fld.body_box(lifted, 800, 800, framing="full_body")
        self.assertGreater(box[1], 300.0,
                           "рез поехал за плечом — в кадре осталось лицо")

    def test_no_body_means_no_box_rather_than_a_guess(self):
        blank = [(0.0, 0.0, 0.0)] * fork_channels.WHOLEBODY_JOINTS
        self.assertIsNone(fld.body_box(blank, 800, 800))

    def test_an_invisible_head_does_not_crash_and_keeps_the_body(self):
        """Голова не распознана — это НЕ «лица нет». Решает прибор, не геометрия."""
        pts = _points()
        for i in list(fork_channels.group_indices("face")) + list(range(0, 5)):
            pts[i] = (300.0, 100.0, 0.0)
        box = fld.body_box(pts, 800, 800, framing="full_body")
        self.assertIsNotNone(box, "без головы кроп не построился вовсе")

    def test_the_box_stays_inside_the_frame(self):
        pts = _points(body_bottom=5000.0)
        box = fld.body_box(pts, 800, 800, framing="full_body")
        self.assertLessEqual(box[2], 800)
        self.assertLessEqual(box[3], 800)
        self.assertGreaterEqual(box[0], 0)
        self.assertGreaterEqual(box[1], 0)

    def test_the_chin_margin_is_guarded_in_both_directions(self):
        """Т1: константа-решение подменяется строже и слабее."""
        pts = _points(head_y=300.0, body_top=140.0, body_bottom=600.0)
        original = fld.CHIN_MARGIN
        try:
            fld.CHIN_MARGIN = 0.0
            tight = fld.body_box(pts, 800, 800, framing="full_body")[1]
            fld.CHIN_MARGIN = 0.5
            loose = fld.body_box(pts, 800, 800, framing="full_body")[1]
        finally:
            fld.CHIN_MARGIN = original
        self.assertGreater(loose, tight,
                           "отступ от подбородка ни на что не влияет")


class TheSetIsBuiltForTheFramingTheTemplateActuallyGives(unittest.TestCase):
    """Иначе LoRA учится на одном масштабе тела, а применяется на другом.

    Не вкус, а арифметика прибора: полный рост при высоте 848 даёт лицо
    63–80 px против бара ArcFace 100 px для видео, то есть «судить нечем»;
    по пояс — 126–159 px. Источник числа — §4a хэндофа, здесь не пересчитывается.
    """

    def _pts(self):
        pts = _points(head_y=100.0, body_top=140.0, body_bottom=600.0)
        for i in (11, 12):                       # таз явно посередине
            pts[i] = (300.0, 400.0, 0.9)
        return pts

    def test_waist_up_cuts_at_the_hips_and_full_body_does_not(self):
        pts = self._pts()
        waist = fld.body_box(pts, 800, 800, framing="waist_up")
        full = fld.body_box(pts, 800, 800, framing="full_body")
        self.assertAlmostEqual(waist[3], 400.0, places=6)
        self.assertGreater(full[3], waist[3],
                           "полный рост не ниже пояса — кадрировка ни на что "
                           "не влияет")

    def test_the_default_is_full_body_because_the_owner_decided_the_framing(self):
        """ХЭНДОФ §2. Набор идёт за кадрировкой темплейта, а не наоборот.

        Здесь стояло `waist_up` — по предыдущей редакции брифинга, где
        кадрировка ещё не была решена. Цена полного роста (лицо 63–80 px против
        видео-бара 100) закрывается доводкой лица на выходе, а не пересборкой
        набора под другой масштаб.
        """
        self.assertEqual(fld.DEFAULT_FRAMING, "full_body")

    def test_an_unknown_framing_is_refused_before_any_work(self):
        with self.assertRaises(ValueError) as caught:
            fld.body_box(None, 800, 800, framing="close_up")
        self.assertIn("close_up", str(caught.exception))

    def test_the_framing_vocabulary_is_the_pipelines_own(self):
        """Е1: разъехавшись, список дал бы набор под несуществующую кадрировку."""
        from ball_reel.skeleton import FRAMINGS

        self.assertIs(fld.FRAMINGS, FRAMINGS)
        self.assertIn(fld.DEFAULT_FRAMING, FRAMINGS)

    def test_waist_up_without_visible_hips_is_refused_rather_than_guessed(self):
        pts = self._pts()
        for i in (11, 12):
            pts[i] = (300.0, 400.0, 0.0)
        self.assertIsNone(fld.body_box(pts, 800, 800, framing="waist_up"))

    def test_the_framing_reaches_the_passport(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build([], tmp, build_type="т", domain="photoreal",
                            framing="full_body")
        self.assertEqual(got["passport"]["framing"], "full_body")


class TheFaceCheckIsAnInstrumentNotAPromise(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, "restore"):
            self.restore()

    def _fake(self, detail):
        original = fld._instrument
        stub = type("S", (), {"face_detail": staticmethod(lambda p: detail)})
        fld._instrument = lambda: stub
        self.restore = lambda: setattr(fld, "_instrument", original)

    def test_a_found_face_fails_the_crop(self):
        self._fake({"face_px": 120})
        got = fld.face_free("любой.png")
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("120px", got["note"])

    def test_no_face_passes(self):
        self._fake(None)
        self.assertEqual(fld.face_free("любой.png")["outcome"], PASS)

    def test_an_instrument_that_could_not_run_is_not_read_as_no_face(self):
        """Именно так набор с лицами и прошёл бы."""
        original = fld._instrument

        def boom():
            raise RuntimeError("весов нет")

        fld._instrument = boom
        self.restore = lambda: setattr(fld, "_instrument", original)
        got = fld.face_free("любой.png")
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertNotEqual(got["outcome"], PASS)


class TheAcceptanceCriterionIsPerTemplate(unittest.TestCase):
    """Один критерий на оба домена молча поменял бы знак у половины каталога."""

    def test_photoreal_is_judged_by_arcface(self):
        self.assertIn("ArcFace", fld.acceptance_for("photoreal"))
        self.assertIn("0.35", fld.acceptance_for("photoreal"))

    def test_stylised_explicitly_refuses_arcface_as_the_judge(self):
        got = fld.acceptance_for("stylised")
        self.assertIn("НЕ судья", got)
        self.assertIn("стилизация не сработала", got)

    def test_an_unknown_domain_is_refused_rather_than_defaulted(self):
        with self.assertRaises(ValueError) as caught:
            fld.acceptance_for("anime-ish")
        self.assertIn("до сборки", str(caught.exception))

    def test_the_two_domains_do_not_share_a_criterion(self):
        self.assertNotEqual(fld.acceptance_for("photoreal"),
                            fld.acceptance_for("stylised"))


class TheRankIsBelowTheTrainersDefaultAndSaysWhy(unittest.TestCase):

    def test_our_rank_is_half_the_trainer_default(self):
        """Т2: литералы. 32 прочитано из train.py:15, команда в docs/FORK_LORA.md."""
        self.assertEqual(fld.TRAINER_DEFAULT_RANK, 32)
        self.assertEqual(fld.DEFAULT_RANK, 16)
        self.assertLess(fld.DEFAULT_RANK, fld.TRAINER_DEFAULT_RANK)

    def test_the_rank_reaches_the_passport(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build([], tmp, build_type="ж40-бёдра", domain="photoreal")
        self.assertEqual(got["passport"]["rank"], fld.DEFAULT_RANK)
        self.assertEqual(got["passport"]["trainer_default_rank"], 32)


class TheBuildReportsNumbersAndRefusesABadSet(unittest.TestCase):

    def _stub_crops(self, outcomes):
        """Подменяет резчик: набор исходов задаётся таблицей, весов не нужно."""
        original = fld.crop_sample
        seq = list(outcomes)

        def fake(frame, out, *, framing=fld.DEFAULT_FRAMING):
            o = seq.pop(0)
            if o == PASS:
                return {"outcome": PASS, "path": str(out), "note": "ok"}
            if o == UNMEASURED:
                return {"outcome": UNMEASURED, "path": None, "note": "не смогли"}
            return {"outcome": FAIL, "path": None, "note": "найдено лицо 90px"}

        fld.crop_sample = fake
        self.addCleanup(lambda: setattr(fld, "crop_sample", original))

    def test_an_empty_run_is_unmeasured_not_a_good_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build([], tmp, build_type="т", domain="photoreal")
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_a_set_that_lost_most_crops_to_faces_is_refused(self):
        self._stub_crops([FAIL] * 14 + [PASS] * 6)
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["f"] * 20, tmp, build_type="т", domain="photoreal")
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("промахивается систематически", got["note"])

    def test_a_small_set_is_refused_by_count(self):
        self._stub_crops([PASS] * 5)
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["f"] * 5, tmp, build_type="т", domain="photoreal")
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn(f"нужно хотя бы {fld.MIN_SAMPLES}", got["note"])

    def test_a_healthy_set_passes(self):
        """Негативный контроль: сборка умеет не только отказывать."""
        self._stub_crops([PASS] * 20)
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["f"] * 20, tmp, build_type="т", domain="photoreal")
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertIn("набор годен", got["note"])

    def test_the_reject_share_bar_is_guarded_in_both_directions(self):
        """Т1 на MAX_REJECT_SHARE."""
        original = fld.MAX_REJECT_SHARE
        try:
            fld.MAX_REJECT_SHARE = 0.9
            self._stub_crops([FAIL] * 14 + [PASS] * 6)
            with tempfile.TemporaryDirectory() as tmp:
                loose = fld.build(["f"] * 20, tmp, build_type="т",
                                  domain="photoreal")
            self.assertIn("кропов 6", loose["note"])
            fld.MAX_REJECT_SHARE = 0.01
            self._stub_crops([FAIL] + [PASS] * 19)
            with tempfile.TemporaryDirectory() as tmp:
                strict = fld.build(["f"] * 20, tmp, build_type="т",
                                   domain="photoreal")
            self.assertEqual(strict["outcome"], FAIL,
                             "порог ужесточён до предела, а набор всё равно "
                             "принят — порог ни на что не влияет")
        finally:
            fld.MAX_REJECT_SHARE = original

    def test_unmeasured_crops_are_counted_apart_from_face_rejects(self):
        self._stub_crops([UNMEASURED] * 3 + [PASS] * 17)
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["f"] * 20, tmp, build_type="т", domain="photoreal")
        self.assertEqual(len(got["unmeasured"]), 3)
        self.assertEqual(len(got["face_rejected"]), 0)
        self.assertIn("не смогли проверить 3", got["note"])

    def test_the_passport_is_written_and_says_what_was_not_run(self):
        self._stub_crops([PASS] * 20)
        with tempfile.TemporaryDirectory() as tmp:
            fld.build(["f"] * 20, tmp, build_type="ж40-бёдра",
                      domain="photoreal")
            p = json.loads((Path(tmp) / "passport.json").read_text(
                encoding="utf-8"))
        self.assertEqual(p["build_type"], "ж40-бёдра")
        self.assertIn("НЕПРОВЕРЕНО", p["unrun"])
        self.assertIn("d_raw", p["unrun"])
        self.assertIn("проверено прибором", p["faces"])


@unittest.skipUnless(HERO.exists() and _weights_ready(),
                     "нет весов buffalo_l или demo/hero.png")
class TheFaceCheckActuallyCatchesAFaceOnRealPixels(unittest.TestCase):
    """НЕГАТИВНЫЙ КОНТРОЛЬ НА ЖИВОМ ВХОДЕ. Главный тест модуля.

    Заглушки проверяют арифметику. Что прибор ВИДИТ лицо на настоящем кадре —
    проверяется только настоящим кадром, и без этого весь отбор мог бы
    оказаться проверкой, которая всегда молчит.
    """

    def test_the_uncropped_frame_is_reported_as_having_a_face(self):
        got = fld.face_free(HERO)
        self.assertEqual(got["outcome"], FAIL,
                         "на полном кадре прибор не нашёл лица — тогда «лиц "
                         "нет» на кропах не значит ничего")
        self.assertGreater(got["face_px"], 0)

    def test_the_body_crop_of_that_same_frame_has_no_face(self):
        from PIL import Image

        points = fork_channels.wholebody_points(HERO)
        self.assertIsNotNone(points, "на демо-кадре не нашли человека")
        with Image.open(HERO) as im:
            rgb = im.convert("RGB")
            box = fld.body_box(points, *rgb.size)
            self.assertIsNotNone(box, "тела в демо-кадре не нашли")
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "body.png"
                rgb.crop(tuple(int(v) for v in box)).save(p)
                got = fld.face_free(p)
        self.assertEqual(got["outcome"], PASS,
                         f"в кропе тела осталось лицо: {got['note']}")


if __name__ == "__main__":
    unittest.main()
