"""Порождение набора: круг не должен замкнуться, а порог — съехать в облако.

Сеть здесь не трогается и pollen не тратится: шлюз и распознаватель заглушены.
Проверяется то, что от них не зависит и при этом ломается тише всего:

* судья не участвует ни в порождении, ни в отборе — иначе вся затея с LoRA
  теряет смысл, и потерю нечем заметить постфактум;
* порог отбора стоит В РАЗРЫВЕ между измеренными облаками, а не внутри одного
  из них;
* калибровка ОТКАЗЫВАЕТ на перекрывшихся облаках, а не выдаёт середину
  перекрытия за порог;
* кадр без лица — это «не измерено», а не «далеко»;
* без явного согласия не тратится ничего.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ball_reel import synth


class Bar(unittest.TestCase):

    def test_working_bar_sits_in_the_measured_gap(self):
        """Порог внутри облака своих отсеивает настоящие кадры пачками."""
        self.assertGreater(synth.FACENET_BAR, synth.MEASURED_SAME_P95)
        self.assertLess(synth.FACENET_BAR, synth.MEASURED_OTHER_MIN)

    def test_computed_bar_also_sits_in_the_gap(self):
        self.assertGreater(synth.MEASURED_BAR, synth.MEASURED_SAME_P95)
        self.assertLess(synth.MEASURED_BAR, synth.MEASURED_OTHER_MIN)

    def test_the_two_bars_are_kept_apart_not_silently_merged(self):
        """Рабочий и посчитанный — разные числа, и оба записаны."""
        self.assertNotEqual(synth.FACENET_BAR, synth.MEASURED_BAR)

    def test_measured_clouds_do_not_overlap(self):
        self.assertLess(synth.MEASURED_SAME_MAX, synth.MEASURED_OTHER_MIN)


class BarFromPairs(unittest.TestCase):

    def test_separated_clouds_give_a_bar_between_them(self):
        got = synth.bar_from_pairs([0.10, 0.15, 0.20], [0.70, 0.80])
        self.assertTrue(got["ok"], got["note"])
        self.assertGreater(got["bar"], 0.20)
        self.assertLess(got["bar"], 0.70)

    def test_bar_leans_towards_our_own(self):
        """Пропущенный чужой учит LoRA чужому лицу; выброшенный свой стоит кадра."""
        got = synth.bar_from_pairs([0.1, 0.2], [0.8])
        self.assertLess(got["bar"], (0.2 + 0.8) / 2)

    def test_overlapping_clouds_are_a_refusal(self):
        """Середина перекрытия — не порог, а вид на его отсутствие."""
        got = synth.bar_from_pairs([0.1, 0.5, 0.9], [0.4, 0.6])
        self.assertFalse(got["ok"])
        self.assertIsNone(got["bar"])
        self.assertIn("перекрыл", got["note"])

    def test_empty_cloud_is_a_refusal(self):
        self.assertFalse(synth.bar_from_pairs([], [0.8])["ok"])
        self.assertFalse(synth.bar_from_pairs([0.1], [])["ok"])


class Independence(unittest.TestCase):
    """Главная гарантия: судья не участвует ни в порождении, ни в отборе."""

    def test_selector_is_not_of_the_judges_family(self):
        from ball_reel.dataset import independence_report

        got = independence_report(generator=f"pollinations/{synth.MODEL}",
                                  selector="facenet/InceptionResnetV1-vggface2",
                                  judge="arcface/buffalo_l")
        self.assertTrue(got["independent"], got["note"])

    def test_selecting_by_the_judge_is_caught(self):
        """Тест обязан уметь краснеть: подставляем ArcFace в отбор."""
        from ball_reel.dataset import independence_report

        got = independence_report(generator="pollinations/kontext",
                                  selector="arcface/buffalo_l",
                                  judge="arcface/buffalo_l")
        self.assertFalse(got["independent"])

    def test_the_two_bars_belong_to_different_recognisers(self):
        """Совпадение порогов было бы первым признаком, что отбор съехал к судье."""
        from ball_reel.identity_arcface import SAME_PERSON_MAX

        self.assertNotEqual(synth.FACENET_BAR, SAME_PERSON_MAX)


class Prompts(unittest.TestCase):

    def test_one_prompt_per_frame(self):
        self.assertEqual(len(synth.prompts_for(7)), 7)

    def test_more_frames_than_axes_cycles_instead_of_inventing(self):
        from ball_reel.dataset import variations

        n = len(variations())
        got = synth.prompts_for(n + 3)
        self.assertEqual(len(got), n + 3)
        self.assertEqual(got[n], got[0])

    def test_any_prefix_touches_more_than_one_axis(self):
        """Замерено: три кадра подряд различались ТОЛЬКО кадрировкой."""
        from ball_reel.dataset import variations

        rows = synth._interleaved(variations())
        base = rows[0]
        changed = {k for r in rows[1:4] for k in r if r[k] != base[k]}
        self.assertGreaterEqual(len(changed), 3,
                                f"короткий заказ трогает только {changed}")

    def test_interleaving_loses_no_row(self):
        from ball_reel.dataset import variations

        rows = variations()
        self.assertEqual(len(synth._interleaved(rows)), len(rows))

    def test_prompt_says_what_to_keep_and_what_to_change(self):
        """Одежда в подпись не идёт — значит одинаковая прилипнет к триггеру."""
        p = synth.prompts_for(1)[0]
        self.assertIn("same face", p)
        self.assertIn("change the clothing", p)

    def test_default_model_was_chosen_by_measurement(self):
        """`kontext` — умолчание клиента; замер показал, что оно худшее."""
        self.assertNotEqual(synth.MODEL, "kontext")

    def test_identity_is_not_described_in_words(self):
        """Личность несёт референс. Слова о ней увели бы её к описанию."""
        for p in synth.prompts_for(5, "a person"):
            low = p.lower()
            for word in ("ohwx", "face of", "resembling", "looks like"):
                self.assertNotIn(word, low)


class _Stub:
    """Заглушка шлюза: пишет файл, считает вызовы, не ходит в сеть."""

    def __init__(self, fail_at=()):
        self.calls, self.fail_at = [], set(fail_at)

    def images_edit(self, prompt, ref, out, **kw):
        self.calls.append(prompt)
        if len(self.calls) - 1 in self.fail_at:
            raise RuntimeError("шлюз отказал")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"stub")
        return str(out)


class Generate(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.face = self.tmp / "face.jpg"
        self.face.write_bytes(b"stub")

    def _run(self, distances, *, fail_at=(), count=None, bar=None):
        gw = _Stub(fail_at=fail_at)
        seq = iter(distances)
        with mock.patch.object(synth, "pollinations", gw), \
             mock.patch.object(synth, "embed", lambda p: "REF"), \
             mock.patch.object(synth, "face_distance",
                               lambda ref, p: next(seq)):
            return synth.generate(
                str(self.face), self.tmp / "out",
                count=count if count is not None else len(distances),
                bar=synth.FACENET_BAR if bar is None else bar, verbose=False)

    def test_close_frames_are_kept_and_far_ones_dropped(self):
        got = self._run([0.10, 0.90, 0.20, 0.80])
        self.assertEqual(len(got["kept"]), 2)
        self.assertTrue(got["ok"])

    def test_dropped_frames_stay_on_disk_and_in_the_report(self):
        """Отчёт без отброшенных выглядит одинаково при отсеве 5% и 80%."""
        got = self._run([0.10, 0.90])
        self.assertEqual(len(got["rows"]), 2)
        far = [r for r in got["rows"] if not r["kept"]]
        self.assertEqual(len(far), 1)
        self.assertTrue(Path(far[0]["path"]).exists())
        self.assertIn("dropped", far[0]["path"])

    def test_keep_share_is_reported_and_low_share_is_flagged(self):
        got = self._run([0.10] + [0.90] * 9)
        self.assertEqual(got["keep_share"], 0.1)
        self.assertFalse(got["keep_share_ok"])
        self.assertIn("смотреть кадры", got["note"])

    def test_low_share_note_does_not_advise_raising_the_bar(self):
        """Поднять бар — это починить измеритель вместо измеряемого."""
        got = self._run([0.90] * 5)
        self.assertIn("не поднимать бар", got["note"])

    def test_a_frame_without_a_face_is_not_measured_not_far(self):
        got = self._run([None, 0.10])
        row = got["rows"][0]
        self.assertIsNone(row["distance"])
        self.assertFalse(row["kept"])
        self.assertIn("лицо не найдено", row["note"])

    def test_gateway_failure_on_one_frame_does_not_stop_the_step(self):
        got = self._run([0.10, 0.10], fail_at=(0,), count=2)
        self.assertEqual(len(got["rows"]), 2)
        self.assertTrue(any("шлюз отказал" in r["note"] for r in got["rows"]))

    def test_reference_without_a_face_refuses_before_spending(self):
        """Отбирать нечем — значит примем всё подряд; это отказ, а не запуск."""
        gw = _Stub()
        with mock.patch.object(synth, "pollinations", gw), \
             mock.patch.object(synth, "embed", lambda p: None):
            got = synth.generate(str(self.face), self.tmp / "out", count=5,
                                 verbose=False)
        self.assertFalse(got["ok"])
        self.assertEqual(gw.calls, [], "потрачено при нечитаемом референсе")

    def test_absent_recogniser_refuses_before_spending(self):
        """Узнать, что отбирать нечем, после двадцати оплаченных кадров — дорого."""
        gw = _Stub()

        def boom(_):
            raise ImportError("No module named 'facenet_pytorch'")

        with mock.patch.object(synth, "pollinations", gw), \
             mock.patch.object(synth, "embed", boom):
            got = synth.generate(str(self.face), self.tmp / "out", count=20,
                                 verbose=False)
        self.assertFalse(got["ok"])
        self.assertIn("pip install", got["note"])
        self.assertEqual(gw.calls, [], "потрачено без отбирающего распознавателя")

    def test_report_names_who_generated_and_who_selected(self):
        got = self._run([0.10])
        self.assertIn("pollinations", got["generator"])
        self.assertIn("facenet", got["selector"])


class Money(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_without_yes_nothing_is_generated(self):
        called = []
        with mock.patch.object(synth, "generate",
                               lambda *a, **k: called.append(1)):
            code = synth.main(["--face", "f.jpg", "--count", "20",
                               "--out", str(self.tmp)])
        self.assertEqual(code, 0)
        self.assertEqual(called, [], "потрачено без --yes")

    def test_dataset_cli_also_refuses_to_spend_without_yes(self):
        from ball_reel import dataset

        called = []
        with mock.patch.object(synth, "generate",
                               lambda *a, **k: called.append(1)):
            code = dataset.main(["--face", "f.jpg", "--out", str(self.tmp),
                                 "--generate", "20"])
        self.assertEqual(code, 0)
        self.assertEqual(called, [])


class Wiring(unittest.TestCase):

    def test_generation_is_a_step_of_the_dataset_command(self):
        """«Досыпать порождённых» было советом, исполняемым руками."""
        from ball_reel import dataset

        args = dataset.main.__doc__  # noqa: F841 — читаем разборщик ниже
        import inspect

        src = inspect.getsource(dataset.main)
        self.assertIn("--generate", src)
        self.assertIn("synth", src)

    def test_licence_flag_is_stated_in_the_module_itself(self):
        """Веса vggface2 — research-only; вопрос открыт, а не решён молча."""
        self.assertIn("ЛИЦЕНЗИЯ", synth.__doc__)
        self.assertIn("VGGFace2", synth.__doc__)


if __name__ == "__main__":
    unittest.main()
