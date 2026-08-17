"""Поток A: якорь — сырая фотография, медоид запрещён КОДОМ.

Тесты делятся надвое, и деление намеренное.

ОФЛАЙН — арифметика вердикта, три исхода и запрет медоида. Весов не требует,
краснеет всегда, когда сломали.

ЖИВЬЁ — воспроизведение снятых чисел на `demo/lora_dataset`. Требует
`buffalo_l`; без весов ПРОПУСКАЕТСЯ, и пропуск здесь честнее зелени: он значит
«не смогли проверить», а не «прошло» (Р1). ПРОПУСК ВНУТРИ живого класса при
этом запрещён: если веса есть, а фикстура потерялась — это находка, и тест
падает, а не молчит (Т6).

Приёмка §6 A требует ТРИ строки, воспроизводится ОДНА. Оба разрыва записаны
числами в `fork_identity.ACCEPTANCE_ROWS` и сторожатся тестами с обеих сторон:
живые тесты падают, если дерево изменилось (появилась сырая фотография,
контроль дотянул до полосы), офлайновые — если запись объявит закрытым то, что
не закрыто.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ball_reel import fork_identity as fi

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "demo" / "lora_dataset"
MANIFEST = DATASET / "manifest.json"


def _weights_ready() -> bool:
    try:
        from ball_reel.identity_arcface import face_detail

        return face_detail(DATASET / "img" / "real_0000.png") is not None
    except Exception:  # noqa: BLE001
        return False


class _FakeInstrument:
    """Прибор-заглушка: расстояние задаётся таблицей, весов не нужно.

    Нужен, чтобы арифметику вердикта можно было проверить БЕЗ 300 МБ весов.
    Без него ветки «покрытия не хватило» и «контроль провалился» проверялись бы
    только там, где есть живой прибор, то есть почти никогда.
    """

    def __init__(self, table: dict, sizes: dict | None = None):
        self.table = table
        self.sizes = sizes or {}

    def face_detail(self, path):
        name = Path(path).name
        if name not in self.table:
            return None
        return {"embedding": (self.table[name],),
                "face_px": self.sizes.get(name, 200)}

    @staticmethod
    def cosine_distance(a, b):
        return round(abs(a[0] - b[0]), 4)

    @staticmethod
    def _quantile(vals, q):
        from ball_reel.identity_arcface import _quantile

        return _quantile(vals, q)


def _verdict_of(text: str) -> str:
    """Голова вердиктной строки, до двоеточия.

    ЗАЧЕМ ОТДЕЛЬНАЯ ФУНКЦИЯ, А НЕ `assertIn`. `assertIn(PASS, text)` здесь
    НЕ УМЕЕТ КРАСНЕТЬ: `PASS` — «годно», а `FAIL` — «не годно», то есть
    «годно» есть подстрока «не годно», и проверка на успех зеленеет ровно на
    провале. Найдено подменой вердикта в `control_verdict` на FAIL: тесты
    остались зелёными. Это украшение из §9, и оно тут было.
    """
    return text.split(":", 1)[0].strip()


def _with_instrument(inst):
    """Подменить прибор на время вызова. Возвращает восстановитель."""
    original = fi._instrument
    fi._instrument = lambda name: inst
    return lambda: setattr(fi, "_instrument", original)


class TheMedoidIsBannedByCodeNotByAgreement(unittest.TestCase):
    """Соглашение уже было и продержалось до первого удобного случая."""

    def test_an_anchor_from_the_judged_list_is_refused(self):
        frames = ["/x/a.png", "/x/b.png"]
        with self.assertRaises(fi.DerivedAnchor) as caught:
            fi.refuse_derived_anchor("/x/a.png", frames)
        self.assertIn("a.png", str(caught.exception))

    def test_an_anchor_listed_in_the_manifest_is_refused_even_if_not_judged(self):
        """Ровно живой случай: медоид в наборе, судятся 21 порождённый."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "img").mkdir()
            (root / "manifest.json").write_text(json.dumps(
                {"samples": [{"path": "img/real_0000.png"},
                             {"path": "img/gen_0000.png"}]}), encoding="utf-8")
            with self.assertRaises(fi.DerivedAnchor) as caught:
                fi.refuse_derived_anchor(root / "img" / "real_0000.png",
                                         [root / "img" / "gen_0000.png"],
                                         manifest=root / "manifest.json")
            self.assertIn("samples", str(caught.exception))

    def test_a_manifest_that_calls_the_anchor_a_medoid_is_enough(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps(
                {"samples": [{"path": "img/gen_0000.png"}],
                 "identity_reference": "outside.png — МЕДОИД порождённых"},
                ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(fi.DerivedAnchor):
                fi.refuse_derived_anchor(root / "outside.png",
                                         [root / "img" / "gen_0000.png"],
                                         manifest=root / "manifest.json")

    def test_an_honest_uploaded_photo_passes(self):
        """Негативный контроль к запрету (И5): сторож обязан кого-то пропускать."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps(
                {"samples": [{"path": "img/gen_0000.png"}]}), encoding="utf-8")
            fi.refuse_derived_anchor(root / "upload.jpg",
                                     [root / "img" / "gen_0000.png"],
                                     manifest=root / "manifest.json")

    def test_the_ban_reaches_axis_and_is_not_only_a_helper(self):
        """Т5: развилка, до которой не доходит вызов, деградирует молча."""
        with self.assertRaises(fi.DerivedAnchor):
            fi.axis(["/x/a.png"], raw_photo="/x/a.png")

    def test_the_reference_anchor_is_banned_too(self):
        with self.assertRaises(fi.DerivedAnchor):
            fi.axis(["/x/a.png"], raw_photo="/x/raw.png",
                    upscaled_reference="/x/a.png")


class TheVerdictRestsOnTheRawPhotoAndNothingElse(unittest.TestCase):

    def setUp(self):
        # Кадры далеко от сырого фото (0.5) и близко к референсу (0.05).
        self.table = {"raw.png": 0.0, "ref.png": 0.45, "alien.png": 5.0,
                      "f1.png": 0.5, "f2.png": 0.52, "f3.png": 0.48}
        self.restore = _with_instrument(_FakeInstrument(self.table))
        self.frames = ["/x/f1.png", "/x/f2.png", "/x/f3.png"]

    def tearDown(self):
        self.restore()

    def test_a_good_reference_does_not_rescue_a_failing_raw(self):
        got = fi.axis(self.frames, raw_photo="/x/raw.png",
                      upscaled_reference="/x/ref.png")
        self.assertEqual(got["d_ref"]["outcome"], fi.PASS,
                         "фикстура задумана так, что до референса близко")
        self.assertEqual(got["verdict"], fi.FAIL,
                         "вердикт поехал за референсом — вернулся дефект медоида")

    def test_the_note_marks_the_reference_as_not_the_verdict(self):
        got = fi.axis(self.frames, raw_photo="/x/raw.png",
                      upscaled_reference="/x/ref.png")
        self.assertIn("НЕ ВЕРДИКТ", got["note"])

    def test_a_close_raw_photo_passes(self):
        """Негативный контроль: ось умеет не только заваливать."""
        self.table.update({"f1.png": 0.05, "f2.png": 0.1, "f3.png": 0.2})
        got = fi.axis(self.frames, raw_photo="/x/raw.png")
        self.assertEqual(got["verdict"], fi.PASS)

    def test_the_bar_is_the_projects_own_and_not_a_local_copy(self):
        from ball_reel.identity_arcface import SAME_PERSON_MAX

        self.assertEqual(fi.axis(self.frames, raw_photo="/x/raw.png")["bar"],
                         SAME_PERSON_MAX)


class DRefAsksWhatTheUpscaleDidAndNotWhatAGeneratorDid(unittest.TestCase):
    """ХЭНДОФ §2 и §6 A: референс = сырая фотография + АПСКЕЙЛ ЛИЦА.

    Порождённого референса в продукте нет, значит и вопрос у d_ref другой:
    не «что дал генератор референса», а «что дал апскейл». Проверяется и смысл
    (арифметика вердикта), и имя (старый аргумент обязан падать), потому что
    имя `reference` несло снятую формулировку и молчаливый алиас оставил бы
    вызывающего в уверенности, что он подаёт порождённое.
    """

    def tearDown(self):
        if hasattr(self, "restore"):
            self.restore()

    def _axis(self, table, **kw):
        self.restore = _with_instrument(_FakeInstrument(table))
        return fi.axis(["/x/f1.png", "/x/f2.png"], raw_photo="/x/raw.png", **kw)

    def test_the_old_argument_name_is_refused_and_not_silently_aliased(self):
        with self.assertRaises(TypeError):
            fi.axis(["/x/f1.png"], raw_photo="/x/raw.png",
                    reference="/x/ref.png")

    def test_the_signature_names_the_upscale(self):
        import inspect

        params = inspect.signature(fi.axis).parameters
        self.assertIn("upscaled_reference", params)
        self.assertNotIn("reference", params)

    def test_an_upscale_that_left_identity_alone_passes(self):
        got = self._axis({"raw.png": 0.0, "ref.png": 0.02,
                          "f1.png": 0.30, "f2.png": 0.32},
                         upscaled_reference="/x/ref.png")
        self.assertEqual(_verdict_of(got["upscale"]), fi.PASS)
        self.assertIn("апскейл личность не сдвинул", got["upscale"])

    def test_a_reference_the_upscaler_repainted_is_a_finding_not_a_success(self):
        """Кадры БЛИЖЕ к референсу — это тревога, а не улучшение."""
        got = self._axis({"raw.png": 0.0, "ref.png": 0.30,
                          "f1.png": 0.30, "f2.png": 0.32},
                         upscaled_reference="/x/ref.png")
        self.assertEqual(_verdict_of(got["upscale"]), fi.FAIL)
        self.assertIn("ДОРИСОВАЛ", got["upscale"])

    def test_an_upscaler_that_spoiled_the_face_is_caught_too(self):
        got = self._axis({"raw.png": 0.0, "ref.png": 0.9,
                          "f1.png": 0.30, "f2.png": 0.32},
                         upscaled_reference="/x/ref.png")
        self.assertEqual(_verdict_of(got["upscale"]), fi.FAIL)
        self.assertIn("испортил", got["upscale"])

    def test_the_drift_bar_is_guarded_in_both_directions(self):
        """Т1: подмена константы-решения строже и слабее."""
        pair = ({"median": 0.30}, {"median": 0.22})
        self.assertEqual(
            _verdict_of(fi.upscale_drift_verdict(*pair, drift_max=0.01)),
            fi.FAIL)
        self.assertEqual(
            _verdict_of(fi.upscale_drift_verdict(*pair, drift_max=0.5)),
            fi.PASS, "порог поднят выше расхождения, а вердикт не изменился")

    def test_a_missing_median_is_unmeasured_not_harmless(self):
        got = fi.upscale_drift_verdict({"median": 0.3}, {"median": None})
        self.assertEqual(_verdict_of(got), fi.UNMEASURED)
        self.assertIn("НЕ «апскейл безвреден»", got)

    def test_a_run_without_a_reference_says_the_check_did_not_happen(self):
        got = self._axis({"raw.png": 0.0, "f1.png": 0.3, "f2.png": 0.32})
        self.assertEqual(got["upscale"], "НЕ ПРОВЕРЯЛСЯ")
        self.assertIsNone(got["d_ref"])

    def test_the_note_says_the_reference_is_the_upscaled_photo(self):
        got = self._axis({"raw.png": 0.0, "ref.png": 0.02,
                          "f1.png": 0.30, "f2.png": 0.32},
                         upscaled_reference="/x/ref.png")
        self.assertIn("АПСКЕЙЛА ЛИЦА", got["note"])
        self.assertIn("ЧТО ДАЛ АПСКЕЙЛ", got["note"])

    def test_the_upscale_verdict_never_becomes_the_identity_verdict(self):
        """Апскейл идеален, личность провалена — вердикт обязан быть FAIL."""
        got = self._axis({"raw.png": 0.0, "ref.png": 0.0,
                          "f1.png": 0.50, "f2.png": 0.52},
                         upscaled_reference="/x/ref.png")
        self.assertEqual(_verdict_of(got["upscale"]), fi.PASS)
        self.assertEqual(got["verdict"], fi.FAIL)


class ThereAreThreeOutcomesNotTwo(unittest.TestCase):
    """Р1/Р2: «не смогли проверить» не сворачивается ни в одну сторону."""

    def tearDown(self):
        if hasattr(self, "restore"):
            self.restore()

    def test_no_faces_at_all_is_unmeasured_not_a_different_person(self):
        self.restore = _with_instrument(_FakeInstrument({"raw.png": 0.0}))
        got = fi.axis(["/x/f1.png", "/x/f2.png"], raw_photo="/x/raw.png")
        self.assertEqual(got["verdict"], fi.UNMEASURED)
        self.assertNotEqual(got["verdict"], fi.FAIL)
        self.assertIn("НЕ «другой человек»", got["d_raw"]["note"])

    def test_a_missing_face_on_the_anchor_is_unmeasured(self):
        self.restore = _with_instrument(_FakeInstrument({"f1.png": 0.1}))
        got = fi.axis(["/x/f1.png"], raw_photo="/x/raw.png")
        self.assertEqual(got["verdict"], fi.UNMEASURED)

    def test_thin_coverage_is_unmeasured_even_when_the_judged_ones_pass(self):
        """Ноль нарушений при одной отработавшей проверке — не успех (Р2)."""
        self.restore = _with_instrument(
            _FakeInstrument({"raw.png": 0.0, "f1.png": 0.05}))
        got = fi.axis(["/x/f1.png", "/x/f2.png", "/x/f3.png", "/x/f4.png"],
                      raw_photo="/x/raw.png")
        self.assertEqual(got["d_raw"]["inside"], 1)
        self.assertEqual(got["verdict"], fi.UNMEASURED,
                         "покрытие 25% выдано за успех")

    def test_the_note_prints_checked_inside_and_unmeasured_as_numbers(self):
        self.restore = _with_instrument(
            _FakeInstrument({"raw.png": 0.0, "f1.png": 0.05, "f2.png": 0.9}))
        got = fi.axis(["/x/f1.png", "/x/f2.png"], raw_photo="/x/raw.png")
        for piece in ("медиана", "в баре", "не смогли"):
            self.assertIn(piece, got["note"])


class TheNegativeControlIsPartOfTheMeasurement(unittest.TestCase):
    """И5: без входа, где прибор обязан сказать «нет», число ничего не значит."""

    def tearDown(self):
        self.restore()

    def _axis(self, table, **kw):
        self.restore = _with_instrument(_FakeInstrument(table))
        return fi.axis(["/x/f1.png", "/x/f2.png"], raw_photo="/x/raw.png", **kw)

    def test_a_run_without_a_control_says_so_and_does_not_claim_success(self):
        got = self._axis({"raw.png": 0.0, "f1.png": 0.05, "f2.png": 0.06})
        self.assertEqual(got["control"], "НЕ СТАВИЛСЯ")
        self.assertIn("НЕ СТАВИЛСЯ", got["note"])

    def test_a_control_the_instrument_mistakes_for_the_subject_voids_the_run(self):
        got = self._axis({"raw.png": 0.0, "f1.png": 0.05, "f2.png": 0.06,
                          "alien.png": 0.1}, foreign="/x/alien.png")
        self.assertEqual(_verdict_of(got["control"]), fi.FAIL)
        self.assertIn("недействительны", got["control"])

    def test_a_weak_control_is_unmeasured_not_a_pass(self):
        """0.70 — «другой человек», но не полоса «заведомо чужой»."""
        got = self._axis({"raw.png": 0.0, "f1.png": 0.05, "f2.png": 0.06,
                          "alien.png": 0.5}, foreign="/x/alien.png")
        self.assertEqual(_verdict_of(got["control"]), fi.UNMEASURED)

    def test_a_proper_control_passes(self):
        got = self._axis({"raw.png": 0.0, "f1.png": 0.05, "f2.png": 0.06,
                          "alien.png": 1.0}, foreign="/x/alien.png")
        self.assertEqual(_verdict_of(got["control"]), fi.PASS)


class TheFourthNumberSeparatesTwoDifferentIllnesses(unittest.TestCase):
    """`d_drv`: «похоже на драйвинг» и «похоже на АКТЁРА драйвинга» — разное.

    Без этой оси обе болезни одинаково ухудшают `d_raw`, а лечение у них
    разное. Нужна она гипотезе LoRA темплейта: если такая LoRA потечёт, выход
    поедет к актёру, а не к клиенту.
    """

    def tearDown(self):
        if hasattr(self, "restore"):
            self.restore()

    def _axis(self, table, **kw):
        self.restore = _with_instrument(_FakeInstrument(table))
        return fi.axis(["/x/f1.png", "/x/f2.png"], raw_photo="/x/raw.png", **kw)

    def test_a_run_without_the_actor_says_the_check_did_not_happen(self):
        got = self._axis({"raw.png": 0.0, "f1.png": 0.05, "f2.png": 0.06})
        self.assertEqual(got["leak_to_actor"], "НЕ ПРОВЕРЯЛАСЬ")
        self.assertIsNone(got["d_drv"])

    def test_frames_closer_to_the_actor_than_to_the_client_is_a_leak(self):
        got = self._axis({"raw.png": 0.0, "actor.png": 0.6,
                          "f1.png": 0.55, "f2.png": 0.56},
                         driving_actor="/x/actor.png")
        self.assertEqual(_verdict_of(got["leak_to_actor"]), fi.FAIL)
        self.assertIn("утекло", got["leak_to_actor"])

    def test_frames_closer_to_the_client_are_clean(self):
        """Негативный контроль: ось умеет и не находить утечку."""
        got = self._axis({"raw.png": 0.0, "actor.png": 0.9,
                          "f1.png": 0.1, "f2.png": 0.12},
                         driving_actor="/x/actor.png")
        self.assertEqual(_verdict_of(got["leak_to_actor"]), fi.PASS)

    def test_it_compares_two_distances_rather_than_using_a_bar(self):
        """Обе далеко от бара, но актёр ближе — обязано ловиться."""
        got = self._axis({"raw.png": 0.0, "actor.png": 0.7,
                          "f1.png": 0.65, "f2.png": 0.66},
                         driving_actor="/x/actor.png")
        self.assertEqual(_verdict_of(got["leak_to_actor"]), fi.FAIL)

    def test_a_missing_distance_is_unmeasured(self):
        got = fi.actor_leak_verdict({"median": None}, {"median": 0.5})
        self.assertEqual(_verdict_of(got), fi.UNMEASURED)

    def test_the_leak_verdict_reaches_the_note(self):
        got = self._axis({"raw.png": 0.0, "actor.png": 0.6,
                          "f1.png": 0.55, "f2.png": 0.56},
                         driving_actor="/x/actor.png")
        self.assertIn("УТЕЧКА К АКТЁРУ ДРАЙВИНГА", got["note"])


class IdentityIsMeasuredAfterFaceRestoreAndReportedAsAPair(unittest.TestCase):
    """ХЭНДОФ §6 A. Один бар на обе половины — иначе числа несравнимы.

    Кадрировка во весь рост даёт лицо 63–80 px против видео-бара 100, то есть
    «судить нечем». Доводка лица переводит кадры в судимые, и пара «до → после»
    показывает, что дала именно она.
    """

    def tearDown(self):
        # Не все тесты класса ставят прибор: проверка СИГНАТУРЫ прибора не
        # трогает вовсе. Безусловный `self.restore()` уронил её ошибкой в
        # уборке — поймано прогоном, а не рассуждением.
        if hasattr(self, "restore"):
            self.restore()

    def _pair(self, table, sizes=None):
        self.restore = _with_instrument(_FakeInstrument(table, sizes))
        return fi.before_after_restore(
            ["/x/b1.png", "/x/b2.png"], ["/x/a1.png", "/x/a2.png"],
            raw_photo="/x/raw.png", min_face_px=100)

    def test_one_bar_serves_both_halves(self):
        from ball_reel.identity_arcface import SAME_PERSON_MAX

        got = self._pair({"raw.png": 0.0, "b1.png": 0.4, "b2.png": 0.42,
                          "a1.png": 0.2, "a2.png": 0.22})
        self.assertEqual(got["bar"], SAME_PERSON_MAX)
        self.assertNotIn("min_face_px", got)
        # Обе половины отчитались одинаковыми условиями — сравнимость их чисел
        # видна из результата, а не подразумевается по коду вызывающего.
        self.assertEqual(got["before"]["bar"], SAME_PERSON_MAX)
        self.assertEqual(got["after"]["bar"], SAME_PERSON_MAX)
        self.assertEqual(got["before"]["min_face_px"],
                         got["after"]["min_face_px"])

    def test_the_signature_offers_no_way_to_set_two_different_bars(self):
        """Не «не рекомендуется», а НЕВОЗМОЖНО: параметра бара нет вовсе.

        Тест краснеет от появления любого `bar`, `before_bar`, `after_bar` —
        то есть сторожит саму возможность дефекта, а не один его экземпляр.
        """
        import inspect

        names = list(inspect.signature(fi.before_after_restore).parameters)
        self.assertEqual([n for n in names if "bar" in n], [])
        self.assertEqual([n for n in names if n.startswith(("before_",
                                                            "after_"))],
                         ["before_frames", "after_frames"])

    def test_both_halves_are_measured_with_literally_the_same_arguments(self):
        """Т1-мутация условий: разъехавшиеся половины обязаны краснеть.

        Половины меряются одной распаковкой `**common`. Здесь вызов `distances`
        подменяется писцом: если однажды кто-то передаст одной половине свой
        отсев, записи разойдутся и тест покраснеет.
        """
        seen = []
        original = fi.distances

        def spy(frames, anchor, **kw):
            seen.append(kw)
            return original(frames, anchor, **kw)

        self.restore = _with_instrument(_FakeInstrument(
            {"raw.png": 0.0, "b1.png": 0.4, "b2.png": 0.42,
             "a1.png": 0.2, "a2.png": 0.22}))
        fi.distances = spy
        try:
            fi.before_after_restore(["/x/b1.png", "/x/b2.png"],
                                    ["/x/a1.png", "/x/a2.png"],
                                    raw_photo="/x/raw.png", min_face_px=100)
        finally:
            fi.distances = original
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], seen[1])

    def test_halves_measured_by_different_conditions_are_refused(self):
        """Сверка на выходе: несравнимая пара роняет прогон, а не «улучшает».

        Сегодня эта ветка недостижима — и именно поэтому проверяется прямой
        подменой одной половины. Без теста защита от будущей правки была бы
        строкой, которую никто не исполнял.
        """
        self.restore = _with_instrument(_FakeInstrument(
            {"raw.png": 0.0, "b1.png": 0.4, "a1.png": 0.2}))
        original = fi.distances
        calls = {"n": 0}

        def skewed(frames, anchor, **kw):
            calls["n"] += 1
            out = original(frames, anchor, **kw)
            if calls["n"] == 2:  # вторая половина «сужает» бар себе
                out = {**out, "bar": 0.5}
            return out

        fi.distances = skewed
        try:
            with self.assertRaises(RuntimeError) as caught:
                fi.before_after_restore(["/x/b1.png"], ["/x/a1.png"],
                                        raw_photo="/x/raw.png")
        finally:
            fi.distances = original
        self.assertIn("несравнимы", str(caught.exception))

    def test_the_pair_is_reported_not_just_the_better_number(self):
        got = self._pair({"raw.png": 0.0, "b1.png": 0.4, "b2.png": 0.42,
                          "a1.png": 0.2, "a2.png": 0.22})
        self.assertIsNotNone(got["before"]["median"])
        self.assertIsNotNone(got["after"]["median"])
        self.assertIn("ДО доводки", got["note"])
        self.assertIn("ПОСЛЕ", got["note"])
        self.assertLess(got["delta"], 0, "фикстура задумана как улучшение")

    def test_frames_that_were_unjudgeable_and_became_judgeable_are_not_a_loss(self):
        """Главное различие модуля: первое измерение — не ухудшение."""
        got = self._pair(
            {"raw.png": 0.0, "b1.png": 0.4, "b2.png": 0.42,
             "a1.png": 0.5, "a2.png": 0.52},
            sizes={"raw.png": 200, "b1.png": 60, "b2.png": 60,
                   "a1.png": 200, "a2.png": 200})
        self.assertEqual(got["before"]["judged"], 0)
        self.assertEqual(got["after"]["judged"], 2)
        self.assertEqual(got["judged_gain"], 2)
        self.assertIsNone(got["delta"])
        self.assertIn("НЕ ухудшение, а первое измерение", got["note"])

    def test_the_judged_gain_is_printed_even_when_it_is_zero(self):
        got = self._pair({"raw.png": 0.0, "b1.png": 0.4, "b2.png": 0.42,
                          "a1.png": 0.2, "a2.png": 0.22})
        self.assertEqual(got["judged_gain"], 0)
        self.assertIn("не прибавилось", got["note"])

    def test_nothing_judgeable_after_restore_is_unmeasured(self):
        got = self._pair(
            {"raw.png": 0.0, "b1.png": 0.4, "b2.png": 0.42},
            sizes={"raw.png": 200})
        self.assertEqual(got["outcome"], fi.UNMEASURED)

    def test_a_worse_median_after_restore_shows_as_positive_delta(self):
        """Негативный контроль: пара умеет показать и ухудшение."""
        got = self._pair({"raw.png": 0.0, "b1.png": 0.2, "b2.png": 0.22,
                          "a1.png": 0.5, "a2.png": 0.52})
        self.assertGreater(got["delta"], 0)
        self.assertEqual(got["outcome"], fi.FAIL)


class TheFaceRestorerItselfNeedsANegativeControl(unittest.TestCase):
    """И5 применённое к ДОВОДЧИКУ: генератор стоит ПЕРЕД прибором.

    §7 требует мерить личность после доводки лица, но не даёт входа, на
    котором доводчик обязан провалиться. Без такого входа «хороший d_raw»
    неотличим от «доводчик напечатал лицо с референса»: пара «до → после» на
    кадрах клиента одинаково красива в обоих случаях.

    Прогона доводчика не было — нужна карта, помечено НЕПРОВЕРЕНО. Здесь
    проверяется арифметика вердикта, и она обязана краснеть.
    """

    def tearDown(self):
        if hasattr(self, "restore"):
            self.restore()

    def _control(self, table, **kw):
        self.restore = _with_instrument(_FakeInstrument(table))
        return fi.restore_negative_control(
            ["/x/fx1.png", "/x/fx2.png"], raw_photo="/x/raw.png", **kw)

    def test_a_run_that_never_happened_is_unmeasured_and_says_so(self):
        got = fi.restore_negative_control(raw_photo="/x/raw.png")
        self.assertEqual(got["outcome"], fi.UNMEASURED)
        self.assertIn("НЕПРОВЕРЕНО", got["note"])
        self.assertIn("НЕ «доводчик честен»", got["note"])

    def test_a_stranger_pulled_inside_the_bar_kills_the_measurement(self):
        """Главный исход: доводчик печатает референс — ось после него мертва."""
        got = self._control({"raw.png": 0.0, "fx1.png": 0.20, "fx2.png": 0.22})
        self.assertEqual(got["outcome"], fi.FAIL)
        self.assertIn("ПЕЧАТАЕТ РЕФЕРЕНС", got["note"])
        self.assertIn("НЕДЕЙСТВИТЕЛЬНЫ", got["note"])

    def test_a_stranger_that_stayed_a_stranger_passes(self):
        """Негативный контроль к контролю: он обязан уметь и пропускать."""
        got = self._control({"raw.png": 0.0, "fx1.png": 0.68, "fx2.png": 0.70})
        self.assertEqual(got["outcome"], fi.PASS)
        self.assertIn("ПОДТЯЖКА НЕ МЕРЕНА", got["note"])

    def test_an_early_stage_pull_is_caught_before_it_reaches_the_bar(self):
        got = self._control(
            {"raw.png": 0.0, "fb1.png": 0.68, "fb2.png": 0.70,
             "fx1.png": 0.50, "fx2.png": 0.52},
            foreign_frames_before=["/x/fb1.png", "/x/fb2.png"])
        self.assertEqual(got["outcome"], fi.FAIL)
        self.assertEqual(got["pull"], 0.18)
        self.assertIn("ПОДТЯНУЛ", got["note"])

    def test_a_pull_within_instrument_noise_passes(self):
        got = self._control(
            {"raw.png": 0.0, "fb1.png": 0.68, "fb2.png": 0.70,
             "fx1.png": 0.67, "fx2.png": 0.69},
            foreign_frames_before=["/x/fb1.png", "/x/fb2.png"])
        self.assertEqual(got["outcome"], fi.PASS)
        self.assertEqual(got["pull"], 0.01)

    def test_the_pull_bar_is_guarded_in_both_directions(self):
        """Т1: подмена константы-решения строже и слабее."""
        table = {"raw.png": 0.0, "fb1.png": 0.68, "fb2.png": 0.70,
                 "fx1.png": 0.60, "fx2.png": 0.62}
        before = ["/x/fb1.png", "/x/fb2.png"]
        self.assertEqual(
            self._control(table, foreign_frames_before=before,
                          pull_max=0.02)["outcome"], fi.FAIL)
        self.restore()
        self.assertEqual(
            self._control(table, foreign_frames_before=before,
                          pull_max=0.5)["outcome"], fi.PASS,
            "порог поднят выше подтяжки, а вердикт не изменился")

    def test_no_judgeable_stranger_frames_is_unmeasured_not_a_pass(self):
        got = self._control({"raw.png": 0.0})
        self.assertEqual(got["outcome"], fi.UNMEASURED)
        self.assertIn("НЕ «доводчик честен»", got["note"])

    def test_an_unjudgeable_before_half_is_unmeasured_not_a_pass(self):
        got = self._control(
            {"raw.png": 0.0, "fx1.png": 0.68, "fx2.png": 0.70},
            foreign_frames_before=["/x/fb1.png", "/x/fb2.png"])
        self.assertEqual(got["outcome"], fi.UNMEASURED)
        self.assertIsNone(got["pull"])

    def test_the_stranger_fixture_is_one_place_for_the_whole_project(self):
        """Е1: и ось, и контроль доводки берут чужое лицо из одного места."""
        self.assertTrue(fi.FOREIGN_FACE_FIXTURE.exists(),
                        f"нет {fi.FOREIGN_FACE_FIXTURE}: негативный контроль "
                        f"нечем ставить")
        self.assertEqual(fi.FOREIGN_FACE_FIXTURE.name, "foreign_face.png")


class TheAcceptanceSaysHowManyRowsItActuallyReproduced(unittest.TestCase):
    """Е3/Р2: «1 из 3» числом, а не агрегатным флагом и не прозой."""

    def test_exactly_one_row_of_three_is_reproduced(self):
        got = fi.acceptance_report()
        self.assertEqual((got["reproduced"], got["of"]), (1, 3))

    def test_the_unclosed_acceptance_is_not_reported_as_a_pass(self):
        got = fi.acceptance_report()
        self.assertEqual(got["outcome"], fi.UNMEASURED)
        self.assertNotEqual(got["outcome"], fi.PASS)
        self.assertIn("НЕ ЗАКРЫТА", got["note"])

    def test_the_two_gaps_are_named_and_not_merged_into_one_excuse(self):
        got = fi.acceptance_report()
        self.assertEqual(sorted(got["unmeasured"]),
                         ["негативный контроль", "против сырой фотографии"])
        self.assertEqual(got["failed"], [])

    def test_each_gap_carries_its_number_and_its_reason(self):
        rows = fi.ACCEPTANCE_ROWS
        self.assertIsNone(rows["против сырой фотографии"]["reproduced"],
                          "строка объявлена воспроизведённой — сырой "
                          "фотографии в дереве нет, мерить не от чего")
        self.assertEqual(rows["против сырой фотографии"]["target"]["median"],
                         0.5067)
        self.assertEqual(
            rows["негативный контроль"]["reproduced"]["median"], 0.6809)
        self.assertEqual(
            rows["негативный контроль"]["target"]["band"], (0.96, 1.05))
        for name, row in rows.items():
            self.assertIn(row["outcome"], (fi.PASS, fi.FAIL, fi.UNMEASURED),
                          name)
            self.assertGreater(len(row["why"]), 40, name)

    def test_the_reproduced_row_is_the_instrument_not_the_product(self):
        """Строка медоида воспроизводится, но про продукт не говорит ничего."""
        row = fi.ACCEPTANCE_ROWS["против медоида"]
        self.assertEqual(row["outcome"], fi.PASS)
        self.assertIn("порождённое против порождённого", row["why"])

    def test_the_report_would_redden_if_someone_declared_it_done(self):
        """Т1: подмена в другую сторону — «всё воспроизведено» обязано ловиться."""
        row = fi.ACCEPTANCE_ROWS["против сырой фотографии"]
        original = row["outcome"]
        row["outcome"] = fi.PASS
        try:
            self.assertEqual(fi.acceptance_report()["outcome"], fi.UNMEASURED,
                             "две строки объявлены закрытыми, а отчёт всё ещё "
                             "не PASS — сторож не сторожит")
            row["reproduced"] = row["target"]
            fi.ACCEPTANCE_ROWS["негативный контроль"]["outcome"] = fi.PASS
            self.assertEqual(fi.acceptance_report()["outcome"], fi.PASS)
            self.assertEqual(fi.acceptance_report()["reproduced"], 3)
            # Е2: текст отчёта выводится из состояния, а не дописан навсегда.
            # Первая редакция печатала «НЕ ЗАКРЫТА» и при трёх закрытых
            # строках — поймано этой самой мутацией.
            self.assertNotIn("НЕ ЗАКРЫТА", fi.acceptance_report()["note"])
        finally:
            row["outcome"] = original
            row["reproduced"] = None
            fi.ACCEPTANCE_ROWS["негативный контроль"]["outcome"] = fi.UNMEASURED


class TheLoraIsAcceptedByWhetherItSpoilsDRaw(unittest.TestCase):
    """Приёмка гипотезы LoRA темплейта одним прямым вопросом.

    НЕПРОВЕРЕНО: парного прогона не было, LoRA не обучалась. Проверяется здесь
    только арифметика сравнения — и то, что она умеет краснеть.
    """

    def test_a_worse_median_with_lora_fails(self):
        got = fi.lora_regression({"median": 0.30}, {"median": 0.40})
        self.assertEqual(got["outcome"], fi.FAIL)
        self.assertIn("тянет лицо к среднему по категории", got["note"])

    def test_a_noise_sized_difference_passes(self):
        got = fi.lora_regression({"median": 0.30}, {"median": 0.31})
        self.assertEqual(got["outcome"], fi.PASS)

    def test_an_improvement_passes_too(self):
        got = fi.lora_regression({"median": 0.40}, {"median": 0.30})
        self.assertEqual(got["outcome"], fi.PASS)
        self.assertEqual(got["delta"], -0.1)

    def test_the_discriminability_bar_is_guarded_in_both_directions(self):
        """Т1: подмена в обе стороны."""
        pair = ({"median": 0.30}, {"median": 0.35})
        self.assertEqual(fi.lora_regression(*pair, worse_by=0.01)["outcome"],
                         fi.FAIL)
        self.assertEqual(fi.lora_regression(*pair, worse_by=0.5)["outcome"],
                         fi.PASS,
                         "порог поднят выше разницы, а вердикт не изменился")

    def test_a_missing_run_is_unmeasured_not_harmless(self):
        got = fi.lora_regression({"median": 0.3}, {"median": None})
        self.assertEqual(got["outcome"], fi.UNMEASURED)
        self.assertIn("НЕ «LoRA безвредна»", got["note"])


class TheInstrumentIsAParameterAndItsLicenceIsSpoken(unittest.TestCase):

    def test_an_unknown_instrument_is_refused_rather_than_stubbed(self):
        with self.assertRaises(ValueError) as caught:
            fi._instrument("auraface")
        self.assertIn("обнуляет", str(caught.exception))

    def test_the_non_commercial_licence_reaches_the_report_without_blocking(self):
        restore = _with_instrument(_FakeInstrument(
            {"raw.png": 0.0, "f1.png": 0.05}))
        try:
            got = fi.axis(["/x/f1.png"], raw_photo="/x/raw.png")
        finally:
            restore()
        self.assertIn("non-commercial", got["note"].lower())
        self.assertIn("пересчёт всех порогов", got["note"].lower())
        self.assertIn("работу не блокирует", got["note"],
                      "лицензия подана как блокер — ХЭНДОФ §10 говорит, что на "
                      "этапе разработки она не блокирует, это вопрос отгрузки")


class TheSizeFilterChangesTheNumberAndSaysSo(unittest.TestCase):
    """Оба числа верны, и это РАЗНЫЕ числа. Режим обязан стоять рядом."""

    def tearDown(self):
        self.restore()

    def test_filtering_drops_frames_and_the_note_names_the_mode(self):
        table = {"raw.png": 0.0, "f1.png": 0.05, "f2.png": 0.06}
        self.restore = _with_instrument(
            _FakeInstrument(table, sizes={"f1.png": 200, "f2.png": 40}))
        off = fi.distances(["/x/f1.png", "/x/f2.png"], "/x/raw.png")
        on = fi.distances(["/x/f1.png", "/x/f2.png"], "/x/raw.png",
                          min_face_px=100)
        self.assertEqual((off["judged"], on["judged"]), (2, 1))
        self.assertIn("выключен", off["note"])
        self.assertIn("100px", on["note"])

    def test_a_dropped_frame_is_counted_as_unmeasured_not_as_drift(self):
        table = {"raw.png": 0.0, "f1.png": 0.05, "f2.png": 0.06}
        self.restore = _with_instrument(
            _FakeInstrument(table, sizes={"f1.png": 200, "f2.png": 40}))
        on = fi.distances(["/x/f1.png", "/x/f2.png"], "/x/raw.png",
                          min_face_px=100)
        self.assertEqual(on["too_small"], ["f2.png"])
        self.assertEqual(on["inside"], 1)


@unittest.skipUnless(MANIFEST.exists() and _weights_ready(),
                     "нет весов buffalo_l или demo/lora_dataset — "
                     "числа воспроизвести нечем")
class TheMeasuredRowsAreReproduced(unittest.TestCase):
    """Приёмка потока. Воспроизводит то, что ВОСПРОИЗВОДИМО, и только это.

    Строка «против сырой фотографии» здесь ОТСУТСТВУЕТ, и это не пропуск: сырой
    фотографии нет в репозитории — владелец исключил её намеренно. Тест ниже
    сторожит именно это: если фотография появится, он покраснеет и потребует
    дописать строку.
    """

    @classmethod
    def setUpClass(cls):
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.generated = [DATASET / s["path"] for s in data["samples"]
                         if s["origin"] == "generated"]
        cls.medoid = DATASET / "img" / "real_0000.png"

    def test_there_are_twenty_one_generated_frames(self):
        self.assertEqual(len(self.generated), 21)

    def test_the_medoid_row_reproduces_to_the_fourth_decimal(self):
        got = fi.distances(self.generated, self.medoid)
        self.assertEqual(got["median"], 0.2579)
        self.assertEqual((got["inside"], got["judged"]), (19, 21))

    def test_the_recorded_medoid_row_equals_what_the_run_gives(self):
        """Запись в `ACCEPTANCE_ROWS` сверяется с прогоном, а не с памятью.

        Иначе таблица приёмки — пересказ, который разъедется с деревом молча.
        """
        got = fi.distances(self.generated, self.medoid)
        row = fi.ACCEPTANCE_ROWS["против медоида"]["reproduced"]
        self.assertEqual((row["median"], row["inside"], row["judged"]),
                         (got["median"], got["inside"], got["judged"]))

    def test_the_medoid_anchor_is_refused_by_the_ban_when_asked_for_a_verdict(self):
        """То же измерение через `axis` обязано УПАСТЬ, а не выдать успех."""
        with self.assertRaises(fi.DerivedAnchor):
            fi.axis(self.generated, raw_photo=self.medoid, manifest=MANIFEST)

    def test_the_raw_photo_is_genuinely_absent_and_this_test_will_notice(self):
        text = MANIFEST.read_text(encoding="utf-8")
        self.assertIn("МЕДОИД порождённых", text,
                      "манифест перестал называть якорь медоидом — проверить, "
                      "не появилась ли настоящая сырая фотография, и если да "
                      "— дописать строку d_raw в приёмку потока A")

    def test_no_sample_claims_to_be_the_uploaded_photo(self):
        """Второй сторож той же дыры, с другой стороны — по составу набора.

        Первый читает прозу манифеста и упадёт от переписанной формулировки;
        этот смотрит на происхождение кадров и упадёт от ПОЯВЛЕНИЯ кадра с
        новым происхождением. Один сторож на такую находку — мало: строку
        приёмки, которая не закрыта, легко закрыть словами.
        """
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(sorted(data["by_origin"]),
                         ["augmented", "generated", "real"],
                         "в наборе появилось новое происхождение — если это "
                         "загруженная фотография, приёмка d_raw наконец "
                         "мерима: снять UNMEASURED с ACCEPTANCE_ROWS")
        self.assertEqual(data["by_origin"]["real"], 1)
        self.assertEqual(
            [s["path"] for s in data["samples"] if s["origin"] == "real"],
            ["img/real_0000.png"],
            "единственный «real» кадр — тот самый медоид, и якорем он запрещён")

    def test_the_unmeasurable_row_is_recorded_as_unmeasured_not_as_success(self):
        got = fi.acceptance_report()
        row = got["rows"]["против сырой фотографии"]
        self.assertEqual(row["outcome"], fi.UNMEASURED)
        self.assertIsNone(row["reproduced"])
        self.assertIn("против сырой фотографии", got["unmeasured"])

    #: Контрольный кадр лежит в `ball_reel/fixtures/`, а НЕ в `veoprobe/`, и это
    #: не вкус. Полный аудит с мутациями поймал: `veoprobe` не входит в
    #: `codeaudit.DATA_LINKS`, то есть в копию для мутаций не попадает, и оба
    #: теста контроля там ПРОПУСКАЛИСЬ — «2 сторожа молчат во время КАЖДОЙ
    #: мутации». Пропущенный тест мутанта не убивает, и аудит тихо слабел.
    #: `fixtures` в копию линкуется, поэтому негативный контроль живёт там же,
    #: где остальной измерительный инвентарь.
    #: Берётся из модуля, а не собирается здесь второй строкой пути (Е1):
    #: разъехавшийся строковый литерал на этом проекте уже стоил 1.7 ГБ.
    #: ~~`ALIEN = ROOT / "ball_reel" / "fixtures" / "foreign_face.png"`~~ —
    #: снято, закрыто константой `fork_identity.FOREIGN_FACE_FIXTURE`.
    ALIEN = fi.FOREIGN_FACE_FIXTURE

    def test_the_control_fixture_is_present_and_absence_is_a_failure(self):
        """Т6: пропуск — не «прошло». Кадр лежит в репозитории, и если его нет,
        это находка, а не повод промолчать. Раньше здесь стоял `skipTest`, и
        два сторожа контроля молчали бы в любом дереве, где кадр потерялся.
        """
        self.assertTrue(self.ALIEN.exists(),
                        f"нет {self.ALIEN}: негативный контроль не поставлен, "
                        f"и это НЕ «контроль прошёл»")

    def test_the_negative_control_says_different_person(self):
        got = fi.distances(self.generated, self.ALIEN)
        self.assertEqual(got["inside"], 0,
                         "чужого человека приняли за своего — числа прогона "
                         "недействительны")
        self.assertGreater(got["median"], fi.HARD_DRIFT_MAX)

    def test_the_control_does_not_reach_the_band_it_was_recorded_at(self):
        """И6: отрицательный результат записывается ЧИСЛОМ, а не сглаживается.

        В задании контроль стоит на 0.96–1.05. Самое далёкое, что нашлось в
        дереве, — 0.6809. Тест закрепляет РАЗРЫВ с ОБЕИХ сторон: снизу, чтобы
        контроль не ослаб незаметно, и сверху, чтобы день, когда он дотянет до
        полосы, не прошёл молча.
        """
        got = fi.distances(self.generated, self.ALIEN)
        self.assertEqual(got["median"], 0.6809,
                         "число контроля изменилось — записанное в "
                         "ACCEPTANCE_ROWS больше не то, что даёт прогон")
        self.assertLess(got["median"], 0.96,
                        "контроль дотянул до записанной полосы — обновить "
                        "docs/FORK_NUMBERS.md и снять эту оговорку")

    def test_the_recorded_control_row_equals_what_the_run_gives(self):
        got = fi.distances(self.generated, self.ALIEN)
        row = fi.ACCEPTANCE_ROWS["негативный контроль"]["reproduced"]
        self.assertEqual(
            (row["median"], row["min"], row["max"], row["inside"],
             row["judged"]),
            (got["median"], got["min"], got["max"], got["inside"],
             got["judged"]))
        lo, hi = fi.ACCEPTANCE_ROWS["негативный контроль"]["target"]["band"]
        self.assertFalse(lo <= got["median"] <= hi,
                         "контроль внутри целевой полосы — строку приёмки "
                         "можно закрывать, снять UNMEASURED")

    def test_two_different_statements_about_the_control_stay_separate(self):
        """Р1/Ц4: «контроль сработал» и «строка приёмки закрыта» — РАЗНОЕ.

        По собственному бару проекта 0.6809 выше HARD_DRIFT_MAX 0.6, поэтому
        `control_verdict` даёт годно: прибор действительно говорит «другой
        человек». Но записанная полоса 0.96–1.05 НЕ ВОСПРОИЗВЕДЕНА, и строка
        приёмки остаётся «не смогли». Именно так на этом проекте уже был
        отчёт, где две строки об одних пикселях говорили PASS и «судить
        нечем» одновременно, — здесь они разведены и обе названы.
        """
        d = fi.distances(self.generated, self.ALIEN)
        self.assertEqual(_verdict_of(fi.control_verdict(d)), fi.PASS)
        self.assertEqual(
            fi.ACCEPTANCE_ROWS["негативный контроль"]["outcome"],
            fi.UNMEASURED,
            "годный контроль выдан за воспроизведённую полосу 0.96–1.05")


if __name__ == "__main__":
    unittest.main()
