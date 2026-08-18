"""Слой оператора: у каждой проверки есть вход, на котором она обязана молчать.

ЗАЧЕМ ИМЕННО ТАК. Проверка, которая только краснеет, ловит не дефект, а всё
подряд, и отличить её от сломанной нельзя. Поэтому здесь у КАЖДОЙ оси две
стороны: вход, на котором ось обязана покраснеть, и вход, на котором она
обязана промолчать (негативный контроль, И5). Один раз это уже спасло: метрика
дала 0.3106 и 0.3072 на кадрах, отличавшихся на 37% пикселей, и без второго
входа это читалось бы как рабочий прибор.

ОЖИДАЕМЫЕ ЧИСЛА ЗДЕСЬ ЛИТЕРАЛЫ (Т2). `480`, `832`, `16`, `30`, `5`, `10`
написаны цифрами, а не импортированы из проверяемого модуля: импортированное
поедет вместе с кодом и промолчит.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ball_reel import fork_build_route, fork_comfy, fork_mask, fork_props
from ball_reel import fork_template as ft
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED


def _blind(_path):
    """Прибор, который ничего не измерил. Не «плохо», а «не смогли»."""
    return None


def _fps(value):
    return lambda _path: value


def _explodes(*_a, **_k):
    raise AssertionError("роутер не должен был вызываться")


class Fixture:
    """Годное описание и его ассеты на диске. Одно место на весь файл (Е1)."""

    def __init__(self, tmp: Path, **over):
        self.dir = tmp
        (tmp / "driving.mp4").write_bytes(b"not-a-video")
        (tmp / "photo.png").write_bytes(b"not-a-photo")
        self.desc = {
            "schema": "fork-template/1", "name": "проба",
            "driving": "driving.mp4", "photo": "photo.png",
            "prompt": "женщина идёт по улице", "negative": "размыто",
            "fps": 30, "width": 480, "height": 832,
            "seconds": 5, "frames": 149,
            "step": "Q4_K_M", "arm": "narrow", "face_refine": True,
        }
        self.desc.update(over)

    def write(self, name="template.json") -> Path:
        p = self.dir / name
        p.write_text(json.dumps(self.desc, ensure_ascii=False), encoding="utf-8")
        return p


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def fixture(self, **over) -> Fixture:
        return Fixture(self.tmp, **over)

    def check(self, desc, **kw):
        kw.setdefault("prober", _blind)
        return ft.check(desc, root=self.tmp, **kw)

    def axis(self, rep, name) -> dict:
        for f in rep["findings"]:
            if f["axis"] == name:
                return f
        raise AssertionError(
            f"оси {name!r} в отчёте нет; есть {[f['axis'] for f in rep['findings']]}")

    def assertOnlyBroken(self, rep, name):
        """Покраснела ровно эта ось и ничего кроме неё.

        Это и есть негативный контроль в собранном виде: проверка, краснеющая
        заодно с соседями, не отличает свой дефект от чужого.
        """
        red = [f["axis"] for f in rep["findings"] if f["outcome"] == FAIL]
        self.assertEqual(red, [name], f"покраснели {red}, ожидалась {name!r}")


class GoodDescriptionIsSilent(Base):
    """Общий негативный контроль: на годном входе не краснеет НИЧЕГО."""

    def test_every_axis_is_quiet_on_a_good_description(self):
        rep = self.check(self.fixture().desc)
        self.assertEqual(rep["violations"], 0, rep["note"])
        self.assertGreaterEqual(rep["checked"], 15, rep["note"])

    def test_zero_violations_over_zero_checks_is_not_success(self):
        # Р2 дословно: ноль нарушений при нуле отработавших проверок — не успех.
        rep = ft._report([], ["всё пропущено"])
        self.assertEqual(rep["outcome"], UNMEASURED)
        self.assertEqual(rep["checked"], 0)
        self.assertIn("не успех", rep["note"])

    def test_numbers_ride_next_to_the_verdict(self):
        rep = self.check(self.fixture(width=479).desc)
        for word in ("проверено", "нарушений", "не смогли"):
            self.assertIn(word, rep["note"])
        self.assertEqual(rep["passed"] + rep["violations"] + rep["unmeasured"],
                         rep["checked"])

    def test_the_third_outcome_does_not_fold_into_either_of_the_two(self):
        good = self.fixture().desc
        measured = self.check(good, prober=_fps(60))
        blind = self.check(good, prober=_blind)
        self.assertEqual(measured["outcome"], PASS, measured["note"])
        self.assertEqual(blind["outcome"], UNMEASURED, blind["note"])
        self.assertEqual(blind["violations"], 0,
                         "«не смогли» свернулось в «не годно»")


class RetargetingIsNotAKnob(Base):
    def test_a_retargeting_field_drops_the_load(self):
        p = self.fixture(pose_retarget=True).write()
        with self.assertRaises(ft.RetargetForbidden) as e:
            ft.load(p, prober=_blind)
        self.assertIn("ВЫКЛЮЧЕН ВСЕГДА", str(e.exception))

    def test_it_is_found_inside_a_nested_object_too(self):
        # Оператор напишет его там, где ему удобно, а не там, где мы ждём.
        with self.assertRaises(ft.RetargetForbidden):
            ft.refuse_retarget({"motion": {"ретаргетинг_позы": 0.5}})

    def test_a_clean_description_passes_and_says_how_much_it_looked_at(self):
        seen = ft.refuse_retarget(self.fixture().desc)
        self.assertGreater(seen, 10, "запрет обошёл слишком мало полей")

    def test_nothing_to_forbid_is_not_the_same_as_clean(self):
        # Пустой словарь проходит запрет тем, что запрещать в нём нечего.
        self.assertEqual(ft.refuse_retarget({}), 0)
        self.assertEqual(self.axis(self.check({}), "ретаргетинг")["outcome"],
                         UNMEASURED)

    def test_a_normal_field_is_not_mistaken_for_retargeting(self):
        rep = self.check(self.fixture(pose_strength=0.8).desc)
        self.assertEqual(self.axis(rep, "ретаргетинг")["outcome"], PASS)


class Paths(Base):
    def test_a_missing_driving_is_caught(self):
        rep = self.check(self.fixture(driving="нет-такого.mp4").desc)
        self.assertOnlyBroken(rep, "драйвинг")

    def test_a_missing_photo_is_caught(self):
        rep = self.check(self.fixture(photo="нет-такой.png").desc)
        self.assertOnlyBroken(rep, "фотореференс")

    def test_paths_are_resolved_against_the_description_not_the_cwd(self):
        # Негативный контроль к предыдущим двум: те же имена файлов существуют,
        # и ось молчит ровно потому, что корень взят у файла-описания.
        rep = self.check(self.fixture().desc)
        self.assertEqual(self.axis(rep, "драйвинг")["outcome"], PASS)


class Geometry(Base):
    def test_a_side_not_multiple_of_sixteen_is_caught(self):
        rep = self.check(self.fixture(width=481).desc)
        self.assertOnlyBroken(rep, "геометрия")
        self.assertIn("16", self.axis(rep, "геометрия")["note"])

    def test_the_multiple_is_read_from_fork_comfy_and_not_copied_here(self):
        # Е1 наблюдаемо: подменяем кратность у первоисточника — краснеет
        # штатная геометрия, которая при 16 годна.
        old = fork_comfy.SIDE_MULTIPLE
        fork_comfy.SIDE_MULTIPLE = 17
        try:
            rep = self.check(self.fixture().desc)
        finally:
            fork_comfy.SIDE_MULTIPLE = old
        self.assertEqual(self.axis(rep, "геометрия")["outcome"], FAIL)

    def test_the_standard_geometry_is_vertical_480x832(self):
        rep = self.check(self.fixture(width=480, height=832).desc)
        self.assertEqual(self.axis(rep, "геометрия")["outcome"], PASS)

    def test_a_horizontal_frame_is_caught_even_though_both_sides_are_legal(self):
        rep = self.check(self.fixture(width=832, height=480).desc)
        self.assertOnlyBroken(rep, "геометрия")
        self.assertIn("вертикаль", self.axis(rep, "геометрия")["note"])

    def test_a_side_that_is_not_an_integer_is_caught(self):
        rep = self.check(self.fixture(height=832.0).desc)
        self.assertEqual(self.axis(rep, "геометрия")["outcome"], FAIL)


class Length(Base):
    def test_too_short_is_caught(self):
        rep = self.check(self.fixture(seconds=4, frames=None).desc)
        self.assertEqual(self.axis(rep, "длина")["outcome"], FAIL)

    def test_too_long_is_caught(self):
        rep = self.check(self.fixture(seconds=11, frames=None).desc)
        self.assertEqual(self.axis(rep, "длина")["outcome"], FAIL)

    def test_both_ends_of_the_allowed_range_are_inside(self):
        # Негативный контроль к обоим краям сразу (Т3): границы включительно.
        for sec, frames in ((5, 149), (10, 297)):
            with self.subTest(seconds=sec):
                rep = self.check(self.fixture(seconds=sec, frames=frames).desc)
                self.assertEqual(self.axis(rep, "длина")["outcome"], PASS)
                self.assertEqual(self.axis(rep, "кадры")["outcome"], PASS)

    def test_the_middle_of_the_range_is_inside_too(self):
        rep = self.check(self.fixture(seconds=7, frames=209).desc)
        self.assertEqual(self.axis(rep, "длина")["outcome"], PASS)


class Frames(Base):
    def test_a_frame_count_off_the_sampler_step_is_caught(self):
        rep = self.check(self.fixture(frames=150).desc)
        self.assertOnlyBroken(rep, "кадры")

    def test_the_step_is_read_from_fork_comfy(self):
        old = fork_comfy.LENGTH_STEP
        fork_comfy.LENGTH_STEP = 5
        try:
            rep = self.check(self.fixture().desc)
        finally:
            fork_comfy.LENGTH_STEP = old
        self.assertEqual(self.axis(rep, "кадры")["outcome"], FAIL)

    def test_five_seconds_at_thirty_is_149_and_not_150(self):
        """Наивные 150 сэмплер не возьмёт — расхождение прозы с исходником."""
        got = ft.frames_for(5, fps=30)
        self.assertEqual(got["frames"], 149)
        self.assertEqual(got["wanted"], 150)
        self.assertEqual(got["dropped"], 1)

    def test_a_length_that_needs_no_trimming_is_not_trimmed(self):
        # Негативный контроль подрезки: она обязана НЕ срабатывать там, где
        # число и так годно, иначе «подрезано 0» неотличимо от «подрезано».
        got = ft.frames_for(5, fps=25)
        self.assertEqual(got["frames"], 125)
        self.assertEqual(got["dropped"], 0)

    def test_trimming_goes_down_never_up(self):
        got = ft.frames_for(10, fps=30)
        self.assertLess(got["frames"], got["wanted"])

    def test_a_frame_count_that_contradicts_the_length_is_caught(self):
        # 297 годно по шагу, но это десять секунд, а в описании пять.
        rep = self.check(self.fixture(seconds=5, frames=297).desc)
        self.assertOnlyBroken(rep, "кадры")

    def test_frames_are_derived_when_the_operator_did_not_write_them(self):
        desc = self.fixture()
        desc.desc.pop("frames")
        loaded = ft.load(desc.write(), prober=_blind)
        self.assertEqual(loaded["frames"], 149)
        self.assertEqual(loaded["_frames"]["dropped"], 1)


class Fps(Base):
    def test_a_frequency_other_than_thirty_is_caught(self):
        rep = self.check(self.fixture(fps=24).desc)
        self.assertEqual(self.axis(rep, "частота")["outcome"], FAIL)

    def test_thirty_is_silent(self):
        rep = self.check(self.fixture(fps=30).desc)
        self.assertEqual(self.axis(rep, "частота")["outcome"], PASS)

    def test_a_source_below_thirty_is_caught(self):
        rep = self.check(self.fixture().desc, prober=_fps(25))
        self.assertOnlyBroken(rep, "частота драйвинга")

    def test_a_source_at_or_above_thirty_is_silent(self):
        for value in (30, 60):
            with self.subTest(fps=value):
                rep = self.check(self.fixture().desc, prober=_fps(value))
                self.assertEqual(
                    self.axis(rep, "частота драйвинга")["outcome"], PASS)

    def test_no_prober_is_unmeasured_and_not_a_pass(self):
        rep = self.check(self.fixture().desc, prober=_blind)
        self.assertEqual(self.axis(rep, "частота драйвинга")["outcome"],
                         UNMEASURED)


class Step(Base):
    def test_an_unknown_weight_step_is_caught(self):
        rep = self.check(self.fixture(step="Q8_0").desc)
        self.assertOnlyBroken(rep, "ступень весов")

    def test_the_step_names_come_from_the_preflight_budget(self):
        rep = self.check(self.fixture(step="Q4_K_M").desc)
        self.assertEqual(self.axis(rep, "ступень весов")["outcome"], PASS)
        self.assertIn("Q4_K_M", ft.STEP_CHOICES)


class Arm(Base):
    def test_an_unknown_arm_is_caught(self):
        rep = self.check(self.fixture(arm="широкое").desc)
        self.assertOnlyBroken(rep, "плечо маски")

    def test_all_three_real_arms_are_silent(self):
        for name in ("narrow", "wide", "wider"):
            with self.subTest(arm=name):
                rep = self.check(self.fixture(arm=name).desc)
                self.assertEqual(self.axis(rep, "плечо маски")["outcome"], PASS)

    def test_auto_is_unmeasured_until_the_router_ran(self):
        rep = self.check(self.fixture(arm="auto").desc)
        self.assertEqual(self.axis(rep, "плечо маски")["outcome"], UNMEASURED)
        self.assertEqual(rep["violations"], 0, "«не смогли» стало «не годно»")

    def test_a_named_arm_does_not_ask_the_router(self):
        got = ft.resolve_arm(self.fixture(arm="wide").desc, router=_explodes,
                             root=self.tmp)
        self.assertEqual(got["arm"], "wide")
        self.assertEqual(got["grow_px"], fork_mask.arm("wide"))

    def test_auto_takes_the_arm_from_the_router(self):
        cases = ((fork_build_route.BUCKET_NO_LORA, "narrow"),
                 (fork_build_route.BUCKET_LORA_FULL_W40, "wider"))
        for bucket, expected in cases:
            with self.subTest(bucket=bucket):
                got = ft.resolve_arm(
                    self.fixture(arm="auto").desc,
                    router=lambda _p, b=bucket: {"bucket": b, "note": "проба"},
                    root=self.tmp)
                self.assertEqual(got["outcome"], PASS)
                self.assertEqual(got["arm"], expected)

    def test_an_unsure_router_leaves_the_arm_unchosen(self):
        got = ft.resolve_arm(
            self.fixture(arm="auto").desc,
            router=lambda _p: {"bucket": fork_build_route.UNSURE,
                               "note": "полоса"},
            root=self.tmp)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIsNone(got["arm"])
        self.assertEqual(got["unmeasured"], 1)

    def test_a_junk_arm_raises_rather_than_defaulting(self):
        with self.assertRaises(KeyError):
            ft.resolve_arm(self.fixture(arm="нет").desc, router=_explodes,
                           root=self.tmp)


class Props(Base):
    def _marking(self, schema="fork-props/1"):
        p = self.tmp / "props.json"
        p.write_text(json.dumps({
            "schema": schema, "template": "проба",
            "frames": {"0": [{"name": "сумка", "role": fork_props.KEEP,
                              "bbox": [10, 10, 90, 90]}]},
        }, ensure_ascii=False), encoding="utf-8")
        return p

    def test_a_wrong_marking_schema_is_caught(self):
        self._marking(schema="points/2")
        rep = self.check(self.fixture(props="props.json").desc)
        self.assertOnlyBroken(rep, "разметка предметов")

    def test_a_missing_marking_file_is_caught(self):
        rep = self.check(self.fixture(props="нет.json").desc)
        self.assertOnlyBroken(rep, "разметка предметов")

    def test_the_right_schema_is_silent_and_counts_the_objects(self):
        self._marking()
        rep = self.check(self.fixture(props="props.json").desc)
        found = self.axis(rep, "разметка предметов")
        self.assertEqual(found["outcome"], PASS)
        self.assertIn("предметов проверено 1", found["note"])

    def test_no_marking_at_all_is_skipped_not_passed(self):
        # Разметка необязательна, но «нечего проверять» не «проверено и годно».
        rep = self.check(self.fixture().desc)
        axes = [f["axis"] for f in rep["findings"]]
        self.assertNotIn("разметка предметов", axes)
        self.assertTrue(any("разметка" in s for s in rep["skipped"]))


class Protagonist(Base):
    def test_a_box_outside_the_frame_is_caught(self):
        rep = self.check(self.fixture(protagonist=[0, 0, 500, 832]).desc)
        self.assertOnlyBroken(rep, "рамка протагониста")

    def test_a_degenerate_box_is_caught(self):
        rep = self.check(self.fixture(protagonist=[100, 100, 100, 200]).desc)
        self.assertOnlyBroken(rep, "рамка протагониста")

    def test_a_box_inside_the_frame_is_silent(self):
        rep = self.check(self.fixture(protagonist=[10, 20, 470, 800]).desc)
        self.assertEqual(self.axis(rep, "рамка протагониста")["outcome"], PASS)

    def test_the_box_does_not_promise_it_is_the_right_person(self):
        rep = self.check(self.fixture(protagonist=[10, 20, 470, 800]).desc)
        self.assertEqual(self.axis(rep, "тождество протагониста")["outcome"],
                         UNMEASURED)

    def test_no_box_means_no_axis_at_all(self):
        rep = self.check(self.fixture().desc)
        self.assertNotIn("рамка протагониста",
                         [f["axis"] for f in rep["findings"]])


class RequiredFields(Base):
    def test_a_missing_required_field_is_caught(self):
        desc = self.fixture().desc
        desc.pop("face_refine")
        rep = self.check(desc)
        self.assertEqual(self.axis(rep, "обязательные поля")["outcome"], FAIL)

    def test_face_refine_has_no_default(self):
        rep = self.check(self.fixture(face_refine="да").desc)
        self.assertEqual(self.axis(rep, "доводка лица")["outcome"], FAIL)

    def test_both_states_of_face_refine_are_legal(self):
        for value in (True, False):
            with self.subTest(face_refine=value):
                rep = self.check(self.fixture(face_refine=value).desc)
                self.assertEqual(self.axis(rep, "доводка лица")["outcome"], PASS)

    def test_an_empty_prompt_is_caught_but_an_empty_negative_is_not(self):
        rep = self.check(self.fixture(prompt="   ").desc)
        self.assertOnlyBroken(rep, "промпт")
        rep = self.check(self.fixture(negative="").desc)
        self.assertEqual(self.axis(rep, "негатив")["outcome"], PASS)

    def test_a_wrong_schema_is_caught(self):
        rep = self.check(self.fixture(schema="fork-template/0").desc)
        self.assertOnlyBroken(rep, "схема")


class BenchNeverReachesProduction(Base):
    def test_a_bench_description_is_refused_for_production(self):
        with self.assertRaises(ValueError) as e:
            ft.assert_production(self.fixture(bench=True).desc)
        self.assertIn("синтетические", str(e.exception))

    def test_a_production_description_passes_with_numbers(self):
        got = ft.assert_production(self.fixture().desc)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["checked"], 1)

    def test_bench_false_is_also_production(self):
        self.assertEqual(
            ft.assert_production(self.fixture(bench=False).desc)["outcome"],
            PASS)

    def test_the_bench_is_the_same_file_and_it_validates(self):
        p = ft.make_bench(self.tmp / "стенд")
        loaded = ft.load(p, prober=_blind)
        self.assertEqual(loaded["_check"]["violations"], 0,
                         loaded["_check"]["note"])
        self.assertTrue(loaded["bench"])
        with self.assertRaises(ValueError):
            ft.assert_production(loaded)


class Loading(Base):
    def test_a_violation_drops_the_load(self):
        p = self.fixture(width=481).write()
        with self.assertRaises(ValueError) as e:
            ft.load(p, prober=_blind)
        self.assertIn("геометрия", str(e.exception))

    def test_unmeasured_alone_does_not_drop_the_load(self):
        # Р1: третий исход не сворачивается во второй. Отсутствие ffprobe не
        # имеет права запрещать работу.
        loaded = ft.load(self.fixture().write(), prober=_blind)
        self.assertEqual(loaded["_check"]["outcome"], UNMEASURED)
        self.assertEqual(loaded["_check"]["violations"], 0)

    def test_a_missing_description_file_says_so(self):
        with self.assertRaises(FileNotFoundError):
            ft.load(self.tmp / "нет.json", prober=_blind)

    def test_a_non_object_description_is_refused(self):
        p = self.tmp / "список.json"
        p.write_text("[1, 2]", encoding="utf-8")
        with self.assertRaises(ValueError):
            ft.load(p, prober=_blind)


class EntryPoint(Base):
    def test_three_exit_codes_and_not_two(self):
        good = self.fixture().write()
        self.assertEqual(ft.report_text(good, prober=_fps(30))["code"], 0)
        self.assertEqual(ft.report_text(good, prober=_blind)["code"], 2)
        bad = self.fixture(width=481).write("плохой.json")
        self.assertEqual(ft.report_text(bad, prober=_fps(30))["code"], 1)

    def test_production_flag_turns_a_bench_into_a_failure(self):
        p = ft.make_bench(self.tmp / "стенд2")
        loose = ft.report_text(p, prober=_fps(30))
        strict = ft.report_text(p, production=True, prober=_fps(30))
        self.assertEqual(loose["code"], 0, loose["text"])
        self.assertEqual(strict["code"], 1, strict["text"])

    def test_the_report_prints_every_axis_with_its_outcome(self):
        text = ft.report_text(self.fixture().write(), prober=_fps(30))["text"]
        for axis in ("геометрия", "длина", "кадры", "плечо маски",
                     "ступень весов", "доводка лица"):
            self.assertIn(axis, text)
        self.assertIn("проверено", text)

    def test_the_fork_of_the_entry_point_lives_outside_main(self):
        # Т5: развилка внутри main() недостижима для теста. Здесь она вызвана
        # напрямую, а main остаётся тонким.
        import contextlib
        import io

        self.assertTrue(callable(ft.report_text))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = ft.main([str(self.fixture().write())])
        self.assertEqual(code, 2)
        self.assertIn("проверено", buf.getvalue())


class ConstantsAreDeclared(unittest.TestCase):
    """Числа-решения названы литералами здесь, а не импортом из модуля (Т2)."""

    def test_the_output_geometry_is_480x832(self):
        self.assertEqual((ft.WIDTH_OUT, ft.HEIGHT_OUT), (480, 832))

    def test_the_output_frequency_is_thirty(self):
        self.assertEqual(ft.FPS_OUT, 30)

    def test_the_length_window_is_five_to_ten_seconds(self):
        self.assertEqual((ft.SECONDS_MIN, ft.SECONDS_MAX), (5, 10))

    def test_the_arm_choices_are_the_three_arms_plus_auto(self):
        self.assertEqual(sorted(ft.ARM_CHOICES),
                         sorted(list(fork_mask.ARMS) + ["auto"]))

    def test_every_decision_constant_carries_its_origin(self):
        src = (Path(ft.__file__)).read_text(encoding="utf-8")
        for name in ("FPS_OUT", "WIDTH_OUT", "SECONDS_MIN", "AUTO_ARM"):
            with self.subTest(constant=name):
                head = src.split(f"{name}")[0]
                block = head.rsplit("\n\n", 1)[-1]
                self.assertTrue(
                    any(w in block for w in ("ИЗМЕРЕНО", "РАСЧЁТ", "ВЫБРАНО")),
                    f"{name} без пометки происхождения (И4)")


class NoDefaultsInSignatures(unittest.TestCase):
    """Умолчание в сигнатуре связывается на импорте, и мутация до него не дойдёт."""

    def test_no_public_function_binds_a_module_constant_as_a_default(self):
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(ft))
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None]
            for d in defaults:
                if isinstance(d, ast.Constant) and d.value is None:
                    continue
                bad.append(f"{node.name}: {ast.dump(d)[:60]}")
        self.assertEqual(bad, [], f"умолчания связаны на импорте: {bad}")


if __name__ == "__main__":
    unittest.main()
