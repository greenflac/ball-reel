"""Набор LoRA темплейта: «лиц нет» обязано быть ИЗМЕРЕНО, а не обещано.

Главный тест здесь — негативный контроль (И5): проверка на лица обязана
СРАБАТЫВАТЬ на кропе, где лицо есть. Проверка, всегда говорящая «лиц нет»,
говорит это и про набор, полный лиц, — и такой набор потянет лицо клиента к
среднему по категории, то есть сделает ровно то, ради предотвращения чего
модуль написан.

Второй такой же тест — приёмка до рендера: на картинке с лицом актёра
драйвинга она обязана СРАБОТАТЬ, на чужом лице — ПРОМОЛЧАТЬ. Оба случая идут
и на заглушках, и живьём на пикселях; порождение картинок требует карты, сама
судейская логика — нет.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from ball_reel import fork_channels, fork_identity
from ball_reel import fork_lora_dataset as fld
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED

ROOT = Path(__file__).resolve().parents[2]
HERO = ROOT / "demo" / "hero.png"
FOREIGN = ROOT / "ball_reel" / "fixtures" / "foreign_face.png"


def _people(n: int, distinct: int = 5) -> list:
    """Метки источников по кругу: n кадров от `distinct` разных людей."""
    return [f"человек_{i % distinct}" for i in range(n)]


def _copies(src: Path, into: Path, n: int) -> list:
    """n копий файла под РАЗНЫМИ именами.

    Не украшение: `fork_identity.distances` складывает результаты в словарь по
    ИМЕНИ файла, и четыре пути с одним именем схлопываются в один судимый кадр,
    роняя покрытие ниже MIN_COVERAGE. Поймано прогоном.
    """
    out = []
    for i in range(n):
        p = into / f"{src.stem}_{i}{src.suffix}"
        shutil.copyfile(src, p)
        out.append(p)
    return out


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
        """Т2: литералы. 32 прочитано из train.py:15, команда в docs/internal/FORK_LORA.md."""
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
            got = fld.build(["f"] * 20, tmp, build_type="т", domain="photoreal",
                            sources=_people(20))
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("промахивается систематически", got["note"])

    def test_a_small_set_is_refused_by_count(self):
        self._stub_crops([PASS] * 5)
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["f"] * 5, tmp, build_type="т", domain="photoreal",
                            sources=_people(5))
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn(f"нужно хотя бы {fld.MIN_SAMPLES}", got["note"])

    def test_a_healthy_set_passes(self):
        """Негативный контроль: сборка умеет не только отказывать."""
        self._stub_crops([PASS] * 20)
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["f"] * 20, tmp, build_type="т", domain="photoreal",
                            sources=_people(20))
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
                                  domain="photoreal", sources=_people(20))
            self.assertIn("кропов 6", loose["note"])
            fld.MAX_REJECT_SHARE = 0.01
            self._stub_crops([FAIL] + [PASS] * 19)
            with tempfile.TemporaryDirectory() as tmp:
                strict = fld.build(["f"] * 20, tmp, build_type="т",
                                   domain="photoreal", sources=_people(20))
            self.assertEqual(strict["outcome"], FAIL,
                             "порог ужесточён до предела, а набор всё равно "
                             "принят — порог ни на что не влияет")
        finally:
            fld.MAX_REJECT_SHARE = original

    def test_unmeasured_crops_are_counted_apart_from_face_rejects(self):
        self._stub_crops([UNMEASURED] * 3 + [PASS] * 17)
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["f"] * 20, tmp, build_type="т", domain="photoreal",
                            sources=_people(20))
        self.assertEqual(len(got["unmeasured"]), 3)
        self.assertEqual(len(got["face_rejected"]), 0)
        self.assertIn("не смогли проверить 3", got["note"])

    def test_the_passport_is_written_and_says_what_was_not_run(self):
        self._stub_crops([PASS] * 20)
        with tempfile.TemporaryDirectory() as tmp:
            fld.build(["f"] * 20, tmp, build_type="ж40-бёдра",
                      domain="photoreal", sources=_people(20))
            p = json.loads((Path(tmp) / "passport.json").read_text(
                encoding="utf-8"))
        self.assertEqual(p["build_type"], "ж40-бёдра")
        self.assertIn("НЕПРОВЕРЕНО", p["unrun"])
        self.assertIn("d_raw", p["unrun"])
        self.assertIn("проверено прибором", p["faces"])


class OnlyOneLoraIsTrainedAndTheOtherArmHasNoDataset(unittest.TestCase):
    """ХЭНДОФ §2a: «как в драйвинге» — это ОТСУТСТВИЕ LoRA, а не вторая LoRA.

    Набор под второе плечо состоял бы из кадров одного актёра драйвинга, то
    есть был бы LoRA личности на не того человека. Поэтому отказ кодом.
    """

    def test_the_baseline_arm_is_refused_by_name(self):
        for name in ("как в драйвинге", "baseline", "как_в_драйвинге",
                     "фигура: без LoRA", "нулевое плечо"):
            with self.subTest(name=name):
                with self.assertRaises(fld.BaselineHasNoDataset):
                    fld.refuse_baseline_build(name)

    def test_the_only_figure_is_accepted(self):
        """Негативный контроль отказа: он умеет не только запрещать."""
        self.assertIsNone(fld.refuse_baseline_build(fld.THE_ONLY_FIGURE))
        self.assertIsNone(fld.refuse_baseline_build("полная 40+, фотореализм"))

    def test_the_refusal_says_what_to_do_instead(self):
        with self.assertRaises(fld.BaselineHasNoDataset) as caught:
            fld.refuse_baseline_build("baseline")
        self.assertIn("ОТСУТСТВИЕ LoRA", str(caught.exception))
        self.assertIn(fld.THE_ONLY_FIGURE, str(caught.exception))

    def test_build_refuses_the_baseline_before_touching_the_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "набор"
            with self.assertRaises(fld.BaselineHasNoDataset):
                fld.build(["f"] * 20, target, build_type="как в драйвинге",
                          domain="photoreal", sources=_people(20))
            self.assertFalse(target.exists(),
                             "каталог набора создан до отказа — работа "
                             "началась под плечо, которому набор не нужен")


class TheSetMustBeManyPeopleAndThatIsCounted(unittest.TestCase):
    """§2a: «много разных людей этого телосложения, а не один».

    Кропы без лиц не делают набор категориальным: телосложение конкретного
    человека опознаваемо и без лица.
    """

    def test_a_set_of_one_person_is_refused(self):
        got = fld.source_mix(["актёр"] * 20)
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["people"], 1)
        self.assertIn("сводится к одному человеку", got["note"])

    def test_five_evenly_split_people_pass(self):
        """Негативный контроль: учёт умеет и пропускать."""
        got = fld.source_mix(_people(20))
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["people"], 5)
        self.assertEqual(got["top_share"], 0.2)

    def test_five_people_where_one_owns_the_set_are_refused(self):
        """Счётчика людей мало: пятеро при 90% у одного — это один человек."""
        got = fld.source_mix(["главный"] * 18 + [f"свидетель_{i}"
                                                 for i in range(4)])
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["people"], 5)
        self.assertIn("даёт 82%", got["note"])

    def test_no_labels_is_unmeasured_and_never_a_pass(self):
        got = fld.source_mix(None)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertNotEqual(got["outcome"], PASS)
        self.assertIsNone(got["people"])

    def test_the_people_bar_is_guarded_in_both_directions(self):
        """Т1: подмена MIN_DISTINCT_PEOPLE строже и слабее."""
        four = _people(20, distinct=4)
        original = fld.MIN_DISTINCT_PEOPLE
        try:
            fld.MIN_DISTINCT_PEOPLE = 4
            loose = fld.source_mix(four)
            fld.MIN_DISTINCT_PEOPLE = 9
            strict = fld.source_mix(four)
        finally:
            fld.MIN_DISTINCT_PEOPLE = original
        self.assertEqual(loose["outcome"], PASS, loose["note"])
        self.assertEqual(strict["outcome"], FAIL,
                         "порог людей ужесточён, а набор всё равно принят — "
                         "порог ни на что не влияет")

    def test_the_one_person_share_bar_is_guarded_in_both_directions(self):
        """Т1: подмена MAX_ONE_PERSON_SHARE строже и слабее."""
        skewed = ["главный"] * 7 + [f"человек_{i}" for i in range(1, 6)]
        original = fld.MAX_ONE_PERSON_SHARE
        try:
            fld.MAX_ONE_PERSON_SHARE = 0.9
            loose = fld.source_mix(skewed)
            fld.MAX_ONE_PERSON_SHARE = 0.1
            strict = fld.source_mix(skewed)
        finally:
            fld.MAX_ONE_PERSON_SHARE = original
        self.assertEqual(loose["outcome"], PASS, loose["note"])
        self.assertEqual(strict["outcome"], FAIL)

    def test_labels_are_counted_per_frame_and_a_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                fld.build(["f"] * 20, tmp, build_type="т", domain="photoreal",
                          sources=_people(3))
        self.assertIn("покадрово", str(caught.exception))

    def test_only_the_sources_of_KEPT_crops_are_counted(self):
        """Человек, все кадры которого отбракованы, в наборе не участвует."""
        original = fld.crop_sample
        seq = [PASS] * 16 + [FAIL] * 4

        def fake(frame, out, *, framing=fld.DEFAULT_FRAMING):
            o = seq.pop(0)
            return ({"outcome": PASS, "path": str(out), "reason": "kept",
                     "note": "ok"} if o == PASS else
                    {"outcome": FAIL, "path": None, "reason": "face",
                     "note": "найдено лицо 90px"})

        fld.crop_sample = fake
        self.addCleanup(lambda: setattr(fld, "crop_sample", original))
        labels = _people(16, distinct=4) + ["шестой"] * 4
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["f"] * 20, tmp, build_type="т",
                            domain="photoreal", sources=labels)
        self.assertEqual(got["sources"]["people"], 4,
                         "посчитаны источники выброшенных кропов")
        self.assertEqual(got["outcome"], FAIL, got["note"])

    def test_a_build_without_labels_is_unmeasured_not_good(self):
        original = fld.crop_sample
        fld.crop_sample = lambda frame, out, *, framing=None: {
            "outcome": PASS, "path": str(out), "reason": "kept", "note": "ok"}
        self.addCleanup(lambda: setattr(fld, "crop_sample", original))
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["f"] * 20, tmp, build_type="т",
                            domain="photoreal")
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertIn("учёт источников не подан", got["note"])


class TheDrivingSourcedSetIsStandardButInstrumented(unittest.TestCase):
    """§2a: из драйвинга собирать можно и лучше — но прибор обязателен.

    На кадрах драйвинга лицо актёра есть ПО ПОСТРОЕНИЮ. Значит детектор, не
    нашедший его ни на одном необрезанном исходнике, молчит, и «лиц нет» на
    кропах не значит ничего.
    """

    def _instrument_says(self, detail):
        original = fld._instrument
        stub = type("S", (), {"face_detail": staticmethod(lambda p: detail)})
        fld._instrument = lambda: stub
        self.addCleanup(lambda: setattr(fld, "_instrument", original))

    def _crops_all_pass(self):
        original = fld.crop_sample
        fld.crop_sample = lambda frame, out, *, framing=None: {
            "outcome": PASS, "path": str(out), "reason": "kept", "note": "ok"}
        self.addCleanup(lambda: setattr(fld, "crop_sample", original))

    def test_a_silent_detector_on_driving_frames_blocks_the_set(self):
        self._instrument_says(None)               # прибор лиц не находит нигде
        self._crops_all_pass()
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["кадр%d.png" % i for i in range(20)], tmp,
                            build_type=fld.THE_ONLY_FIGURE,
                            domain="photoreal", sources=_people(20),
                            from_driving=True)
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertIn("прибор молчит", got["note"])

    def test_a_working_detector_lets_the_driving_set_through(self):
        """Негативный контроль контроля: он умеет не только блокировать."""
        self._instrument_says({"face_px": 140})   # лицо есть на исходниках
        self._crops_all_pass()
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["кадр%d.png" % i for i in range(20)], tmp,
                            build_type=fld.THE_ONLY_FIGURE,
                            domain="photoreal", sources=_people(20),
                            from_driving=True)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["passport"]["from_driving"], True)

    def test_a_non_driving_set_does_not_need_the_positive_control(self):
        self._instrument_says(None)
        self._crops_all_pass()
        with tempfile.TemporaryDirectory() as tmp:
            got = fld.build(["кадр%d.png" % i for i in range(20)], tmp,
                            build_type=fld.THE_ONLY_FIGURE,
                            domain="photoreal", sources=_people(20))
        self.assertEqual(got["outcome"], PASS, got["note"])

    def test_the_control_sample_size_is_guarded_in_both_directions(self):
        """Т1 на DRIVING_CONTROL_SAMPLE: 0 кадров — контроля нет."""
        calls = []
        original = fld.face_free
        fld.face_free = lambda p: (calls.append(p) or
                                   {"outcome": FAIL, "face_px": 140,
                                    "note": "лицо"})
        self.addCleanup(lambda: setattr(fld, "face_free", original))
        wide = fld.driving_face_control(["a", "b", "c", "d", "e"], sample=5)
        self.assertEqual(len(calls), 5)
        self.assertEqual(wide["outcome"], PASS)
        none = fld.driving_face_control(["a", "b"], sample=0)
        self.assertEqual(none["outcome"], UNMEASURED,
                         "контроль без единого кадра объявлен пройденным")

    def test_an_instrument_that_could_not_run_is_not_a_passed_control(self):
        original = fld.face_free
        fld.face_free = lambda p: {"outcome": UNMEASURED, "face_px": None,
                                   "note": "весов нет"}
        self.addCleanup(lambda: setattr(fld, "face_free", original))
        got = fld.driving_face_control(["a", "b", "c"])
        self.assertEqual(got["outcome"], UNMEASURED)


def _fake_arcface(table: dict):
    """Прибор БЕЗ ВЕСОВ: имя файла → расстояние до якоря. Арифметика настоящая.

    Эмбеддинги строятся как единичные векторы под заданным углом, а косинус и
    квантиль берутся у `identity_arcface` — то есть проверяется наш вердикт, а
    не наша арифметика (Т2). Якорь обязан стоять в таблице на 0.0.
    """
    from ball_reel import identity_arcface as real

    def emb(distance):
        angle = math.acos(max(-1.0, min(1.0, 1.0 - distance)))
        return [math.cos(angle), math.sin(angle)]

    def face_detail(path):
        key = Path(path).name
        if key not in table:
            return None
        return {"embedding": emb(table[key]), "face_px": 200,
                "det_score": 0.9}

    return type("Fake", (), {
        "face_detail": staticmethod(face_detail),
        "cosine_distance": staticmethod(real.cosine_distance),
        "_quantile": staticmethod(real._quantile)})


class TheProbeCatchesTheActorBeforeAnyRender(unittest.TestCase):
    """ХЭНДОФ §6 G: сработать на лице актёра, промолчать на чужом. Без карты.

    Порождение картинок требует GPU и здесь не делается — судится уже
    порождённое, и ровно эта половина проверяема сегодня.
    """

    def _instrument(self, table):
        original = fork_identity._instrument
        fake = _fake_arcface(table)
        fork_identity._instrument = lambda name: fake
        self.addCleanup(lambda: setattr(fork_identity, "_instrument", original))

    def test_the_actors_face_on_the_probes_fails_the_set(self):
        self._instrument({"actor.png": 0.0, "p0.png": 0.05, "p1.png": 0.08,
                          "p2.png": 0.5, "p3.png": 0.6})
        got = fld.actor_memorisation_probe(
            ["p0.png", "p1.png", "p2.png", "p3.png"], driving_actor="actor.png")
        self.assertEqual(got["outcome"], FAIL, got["note"])
        self.assertIn("ПРОСТУПИЛО", got["note"])

    def test_a_foreign_face_on_the_probes_stays_silent(self):
        """Негативный контроль: проверка, всегда кричащая, не проверка."""
        self._instrument({"actor.png": 0.0, "p0.png": 0.7, "p1.png": 0.75,
                          "p2.png": 0.68, "p3.png": 0.8})
        got = fld.actor_memorisation_probe(
            ["p0.png", "p1.png", "p2.png", "p3.png"], driving_actor="actor.png")
        self.assertEqual(got["outcome"], PASS, got["note"])

    def test_the_bar_decides_and_is_guarded_in_both_directions(self):
        """Т1: расстояния по обе стороны бара 0.35 и подмена самого бара.

        Свой бар здесь не заводится — он приходит из `identity_arcface`
        (SAME_PERSON_MAX), поэтому подменяется именно он.
        """
        just_inside = {"actor.png": 0.0, "p0.png": 0.34, "p1.png": 0.34,
                       "p2.png": 0.34, "p3.png": 0.34}
        just_outside = {k: (0.0 if k == "actor.png" else 0.36)
                        for k in just_inside}
        probes = ["p0.png", "p1.png", "p2.png", "p3.png"]

        self._instrument(just_inside)
        self.assertEqual(
            fld.actor_memorisation_probe(probes,
                                         driving_actor="actor.png")["outcome"],
            FAIL, "0.34 внутри бара 0.35, а набор принят")
        self._instrument(just_outside)
        self.assertEqual(
            fld.actor_memorisation_probe(probes,
                                         driving_actor="actor.png")["outcome"],
            PASS, "0.36 вне бара 0.35, а набор забракован")

        original = fork_identity.SAME_PERSON_MAX
        try:
            fork_identity.SAME_PERSON_MAX = 0.9
            self.assertEqual(
                fld.actor_memorisation_probe(
                    probes, driving_actor="actor.png")["outcome"],
                FAIL, "бар растянут до 0.9, а 0.36 всё равно прошло — бар ни "
                      "на что не влияет")
        finally:
            fork_identity.SAME_PERSON_MAX = original

    def test_too_few_probes_are_unmeasured_rather_than_clean(self):
        self._instrument({"actor.png": 0.0, "p0.png": 0.7})
        got = fld.actor_memorisation_probe(["p0.png"], driving_actor="actor.png")
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn(str(fld.MIN_PROBE_IMAGES), got["note"])

    def test_probes_without_any_face_are_unmeasured_rather_than_clean(self):
        """Лиц не нашлось — «LoRA лиц не рисует» и «прибор слеп» неразличимы."""
        self._instrument({"actor.png": 0.0})
        got = fld.actor_memorisation_probe(
            ["p0.png", "p1.png", "p2.png", "p3.png"], driving_actor="actor.png")
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertNotEqual(got["outcome"], PASS)

    def test_a_thin_coverage_is_unmeasured_rather_than_clean(self):
        self._instrument({"actor.png": 0.0, "p0.png": 0.7})
        got = fld.actor_memorisation_probe(
            ["p0.png", "p1.png", "p2.png", "p3.png"], driving_actor="actor.png")
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertIn("покрытие", got["note"])

    def test_a_control_frame_that_the_instrument_misses_blocks_the_verdict(self):
        """И5: прибор, не узнавший актёра там, где актёр точно есть, молчит."""
        self._instrument({"actor.png": 0.0, "control.png": 0.9,
                          "p0.png": 0.7, "p1.png": 0.7, "p2.png": 0.7,
                          "p3.png": 0.7})
        got = fld.actor_memorisation_probe(
            ["p0.png", "p1.png", "p2.png", "p3.png"],
            driving_actor="actor.png", control_frame="control.png")
        self.assertEqual(got["outcome"], UNMEASURED, got["note"])
        self.assertIn("не признал актёра", got["note"])

    def test_a_working_control_lets_the_clean_verdict_stand(self):
        self._instrument({"actor.png": 0.0, "control.png": 0.1,
                          "p0.png": 0.7, "p1.png": 0.7, "p2.png": 0.7,
                          "p3.png": 0.7})
        got = fld.actor_memorisation_probe(
            ["p0.png", "p1.png", "p2.png", "p3.png"],
            driving_actor="actor.png", control_frame="control.png")
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertIn(PASS, got["control"])

    def test_a_missing_control_is_named_in_the_report_not_hidden(self):
        self._instrument({"actor.png": 0.0, "p0.png": 0.7, "p1.png": 0.7,
                          "p2.png": 0.7, "p3.png": 0.7})
        got = fld.actor_memorisation_probe(
            ["p0.png", "p1.png", "p2.png", "p3.png"], driving_actor="actor.png")
        self.assertEqual(got["control"], "НЕ СТАВИЛСЯ")
        self.assertIn("НЕ СТАВИЛСЯ", got["note"])

    def test_the_probe_says_out_loud_that_it_generated_nothing(self):
        self._instrument({"actor.png": 0.0})
        got = fld.actor_memorisation_probe(["a", "b", "c", "d"],
                                           driving_actor="actor.png")
        self.assertIn("НЕПРОВЕРЕНО", got["unrun"])


@unittest.skipUnless(HERO.exists() and FOREIGN.exists() and _weights_ready(),
                     "нет весов buffalo_l или demo/hero.png")
class TheProbeWorksOnRealPixelsToo(unittest.TestCase):
    """Заглушки проверяют вердикт. Что прибор ВИДИТ — проверяется пикселями.

    Числа получены командой (17 августа, эта машина):

        python3 -c "from ball_reel.identity_arcface import face_detail, \\
        cosine_distance as c; h=face_detail('demo/hero.png'); \\
        f=face_detail('ball_reel/fixtures/foreign_face.png'); \\
        print(c(h['embedding'], f['embedding']))"
        → 0.7193

    То есть чужое лицо стоит на 0.7193 при баре 0.35 — приёмка обязана
    молчать. Кадр против самого себя даёт 0.0 — обязана срабатывать.

    ПОЧЕМУ ПОЗИТИВНЫЙ КОНТРОЛЬ — ТОТ ЖЕ ФАЙЛ. Второго снимка того же человека,
    который прибор признал бы своим, в дереве нет: измерено той же командой,
    `d(demo/hero.png, demo/lora_dataset/img/real_0000.png) = 0.4637` при баре
    0.35 — прибор считает их разными людьми. Кросс-файловый случай закрыт
    заглушкой выше, а не выдан за живой.
    """

    def test_the_actors_own_face_among_the_probes_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            probes = _copies(HERO, Path(tmp), 4)
            got = fld.actor_memorisation_probe(probes, driving_actor=HERO,
                                               control_frame=HERO)
        self.assertEqual(got["outcome"], FAIL, got["note"])
        self.assertEqual(got["d_drv"]["median"], 0.0)

    def test_a_foreign_face_among_the_probes_is_not_reported_as_the_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            probes = _copies(FOREIGN, Path(tmp), 4)
            got = fld.actor_memorisation_probe(probes, driving_actor=HERO,
                                               control_frame=HERO)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["d_drv"]["median"], 0.7193,
                         "живое расстояние разошлось с замером 0.7193 — "
                         "изменился прибор или фикстура")


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
