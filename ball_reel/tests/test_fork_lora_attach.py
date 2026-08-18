"""Приложилась ли LoRA: три канала, и каждый обязан уметь краснеть.

Прибор написан против конкретного дефекта: ComfyUI при несовпадении ключей не
падает, а печатает `NOT LOADED` и продолжает. Ноль совпавших ключей = чистый
прогон без ошибки и без эффекта, а `lora.accept` дальше молча предполагает, что
адаптер приложился, и меряет качество несуществующей работы.

Поэтому здесь у КАЖДОЙ проверки есть негативный контроль — вход, на котором она
обязана промолчать. Проверка, которая только краснеет, ловит не дефект, а всё
подряд.
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ball_reel import fork_lora_attach as fa
from ball_reel.fork_identity import FAIL, PASS, UNMEASURED


def _adapter(keys, tmp, name="a.safetensors"):
    """Файл safetensors с настоящим заголовком и нулевыми весами."""
    head = {k: {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 0]}
            for k in keys}
    blob = json.dumps(head).encode()
    p = Path(tmp) / name
    p.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return p


TARGET = {"blocks.0.self_attn.q.weight", "blocks.0.self_attn.q.bias",
          "blocks.0.ffn.0.weight", "blocks.1.self_attn.k.weight"}


class TheHeaderIsReadWithoutTheWeights(unittest.TestCase):
    """Адаптер бывает на гигабайт, а нужны имена. Грузить ради них веса значит
    не уметь проверить адаптер там, где карты нет."""

    def test_the_names_come_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _adapter(["a.lora_up.weight", "b.alpha"], tmp)
            self.assertEqual(fa.read_tensor_names(p),
                             {"a.lora_up.weight", "b.alpha"})

    def test_a_missing_file_says_which(self):
        with self.assertRaises(FileNotFoundError) as caught:
            fa.read_tensor_names("/нет/адаптера.safetensors")
        self.assertIn("адаптера.safetensors", str(caught.exception))

    def test_a_truncated_header_is_an_error_not_an_empty_set(self):
        """Пустое множество имён читалось бы как «адаптер без ключей»."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.safetensors"
            p.write_bytes(struct.pack("<Q", 9999) + b"{}")
            with self.assertRaises(ValueError) as caught:
                fa.read_tensor_names(p)
        self.assertIn("обрезан", str(caught.exception))

    def test_a_file_shorter_than_the_header_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tiny.safetensors"
            p.write_bytes(b"\x00\x01")
            with self.assertRaises(ValueError):
                fa.read_tensor_names(p)


class TheModuleNameIsDerivedNotGuessed(unittest.TestCase):

    def test_the_common_conventions_are_understood(self):
        for key, want in (
                ("diffusion_model.blocks.0.self_attn.q.lora_down.weight",
                 "blocks.0.self_attn.q"),
                ("base_model.model.blocks.3.ffn.0.lora_A.weight",
                 "blocks.3.ffn.0"),
                ("blocks.2.ffn.2.alpha", "blocks.2.ffn.2"),
        ):
            with self.subTest(key=key):
                self.assertEqual(fa.module_of(key), want)

    def test_an_unknown_key_returns_none_rather_than_a_guess(self):
        self.assertIsNone(fa.module_of("совершенно.посторонний.тензор"))

    def test_underscores_are_not_turned_into_dots(self):
        """Первая версия выдавала `blocks.0.self.attn.q` — имени, которого в
        модели нет. Адаптер браковался бы из-за нашей догадки, не из-за модели."""
        got = fa.module_of("lora_unet_blocks_0_self_attn_q.alpha")
        self.assertEqual(got, "blocks_0_self_attn_q")
        self.assertNotIn("self.attn", got)

    def test_the_kohya_style_still_matches_the_model(self):
        """Разделители не угадываются, но сопоставление обязано работать."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _adapter(["lora_unet_blocks_0_self_attn_q.lora_up.weight"], tmp)
            got = fa.fit(p, TARGET)
        self.assertEqual(got["lands"], 1, got["note"])

    def test_the_diff_convention_is_parsed_because_comfy_applies_it(self):
        """comfy/lora.py:78 берёт `{}.diff_b` и кладёт патчем. Не разобрав их,
        прибор занижал приложившееся на 773 ключа из 1749 на настоящем файле.

        Обе формы проверяются поимённо: мутация, снявшая ТОЛЬКО `.diff`,
        пережила первый прогон — тест сторожил один суффикс из двух.
        """
        for key, want in (
                ("diffusion_model.blocks.0.ffn.0.diff_b", "blocks.0.ffn.0"),
                ("diffusion_model.blocks.0.cross_attn.norm_k.diff",
                 "blocks.0.cross_attn.norm_k"),
        ):
            with self.subTest(key=key):
                self.assertEqual(fa.module_of(key), want)

    def test_both_diff_suffixes_reach_the_fit(self):
        """Не только разбор имени, но и попадание в счёт приложившегося."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _adapter(["diffusion_model.blocks.0.ffn.0.diff",
                          "diffusion_model.blocks.0.ffn.0.diff_b"], tmp)
            got = fa.fit(p, TARGET)
        self.assertEqual(got["unparsed"], 0, got["note"])
        self.assertEqual(got["lands"], 1)


class TheStaticFitHasThreeOutcomes(unittest.TestCase):

    def _fit(self, keys):
        with tempfile.TemporaryDirectory() as tmp:
            return fa.fit(_adapter(keys, tmp), TARGET)

    def test_a_matching_adapter_passes(self):
        """Негативный контроль (И5): вход, где прибор обязан молчать."""
        got = self._fit(["blocks.0.self_attn.q.lora_up.weight",
                         "blocks.0.ffn.0.lora_up.weight"])
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertEqual(got["absent"], 0)

    def test_a_foreign_adapter_fails(self):
        got = self._fit(["чужое.имя.lora_up.weight",
                         "второе.чужое.lora_up.weight"])
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("не от этой модели", got["note"])

    def test_a_half_landing_adapter_is_a_failure_not_a_partial_success(self):
        got = self._fit(["blocks.0.self_attn.q.lora_up.weight",
                         "чужое.a.lora_up.weight", "чужое.b.lora_up.weight"])
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("это чужой адаптер", got["note"])

    def test_a_mostly_landing_adapter_is_unmeasured_not_pass(self):
        """Р1: «приложится не целиком» — не успех и не провал."""
        got = self._fit(["blocks.0.self_attn.q.lora_up.weight",
                         "blocks.0.ffn.0.lora_up.weight",
                         "blocks.1.self_attn.k.lora_up.weight",
                         "чужое.одно.lora_up.weight"])
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("НЕ ЦЕЛИКОМ", got["note"])

    def test_an_empty_adapter_is_unmeasured(self):
        self.assertEqual(self._fit([])["outcome"], UNMEASURED)

    def test_an_all_unparsed_adapter_is_unmeasured_not_failed(self):
        """Незнакомое семейство адаптеров — «судить не о чем», а не «плохой»."""
        got = self._fit(["что.то.hada_w1_a", "что.то.hada_w2_a"])
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual(got["unparsed"], 2)
        self.assertIn("соглашение", got["note"])

    def test_the_counts_stand_next_to_the_verdict(self):
        got = self._fit(["blocks.0.self_attn.q.lora_up.weight"])
        self.assertIn("ключей в адаптере 1", got["note"])
        self.assertIn("сядет 1 модулей", got["note"])

    def test_the_partial_bar_can_move_in_both_directions(self):
        """Т1: подмена планки обязана менять вердикт в обе стороны."""
        keys = ["blocks.0.self_attn.q.lora_up.weight",
                "чужое.a.lora_up.weight"]          # ровно половина
        saved = fa.PARTIAL_ALARM
        try:
            fa.PARTIAL_ALARM = 0.9
            self.assertEqual(self._fit(keys)["outcome"], FAIL)
            fa.PARTIAL_ALARM = 0.1
            self.assertEqual(self._fit(keys)["outcome"], UNMEASURED)
        finally:
            fa.PARTIAL_ALARM = saved


class TheDoraMagnitudeIsCheckedByName(unittest.TestCase):
    """Дефект, проходящий молча: PEFT пишет `lora_magnitude_vector`, ComfyUI
    ищет `dora_scale`. Загрузится без ошибки, магнитуда потеряется, DoRA станет
    обычной LoRA — то есть тем, ради отказа от чего её и брали."""

    def test_peft_naming_is_caught(self):
        got = fa.dora_readiness({"a.lora_up.weight",
                                 "a.lora_magnitude_vector.weight"})
        self.assertEqual(got["outcome"], FAIL)
        self.assertEqual(got["kind"], "DoRA")
        self.assertIn("dora_scale", got["note"])

    def test_comfy_naming_passes(self):
        got = fa.dora_readiness({"a.lora_up.weight", "a.dora_scale"})
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["kind"], "DoRA")

    def test_plain_lora_is_not_flagged(self):
        """Негативный контроль: обычная LoRA не должна ловиться этой проверкой."""
        got = fa.dora_readiness({"a.lora_up.weight", "a.lora_down.weight"})
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["kind"], "LoRA")

    def test_both_namings_at_once_is_a_failure(self):
        got = fa.dora_readiness({"a.dora_scale", "a.lora_magnitude_vector"})
        self.assertEqual(got["outcome"], FAIL)
        self.assertIn("ОБОИМИ", got["note"])


class TheLogChannelNeedsADenominator(unittest.TestCase):

    LOG = ("loaded straight to GPU\n"
           "NOT LOADED diffusion_model.blocks.0.self_attn.q.weight\n"
           "NOT LOADED diffusion_model.blocks.1.ffn.0.weight\n"
           "Requested to load WAN21\n")

    def test_without_the_total_it_is_unmeasured(self):
        """Три «не приложилось» из тысячи и из трёх — разные вещи."""
        got = fa.from_log(self.LOG)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertEqual(got["not_loaded"], 2)
        self.assertIn("без знаменателя", got["note"])

    def test_with_the_total_it_counts(self):
        got = fa.from_log(self.LOG, keys_in_adapter=100)
        self.assertEqual((got["landed"], got["not_loaded"]), (98, 2))
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_nothing_landing_is_a_failure(self):
        got = fa.from_log(self.LOG, keys_in_adapter=2)
        self.assertEqual(got["outcome"], FAIL)

    def test_a_clean_log_passes(self):
        """Негативный контроль: журнал без единого NOT LOADED."""
        got = fa.from_log("всё хорошо\nloaded\n", keys_in_adapter=10)
        self.assertEqual(got["outcome"], PASS)
        self.assertEqual(got["not_loaded"], 0)

    def test_an_empty_log_is_not_treated_as_success_without_a_total(self):
        self.assertEqual(fa.from_log("")["outcome"], UNMEASURED)

    def test_the_colon_form_is_matched_too(self):
        """comfy печатает обе формы: `NOT LOADED x` и `NOT LOADED: x (type=…)`."""
        got = fa.from_log("NOT LOADED: blocks.0.q (type=lora)\n",
                          keys_in_adapter=1)
        self.assertEqual(got["not_loaded"], 1)


class TheNumericChannelJudgesAgainstTheGeneratorsOwnNoise(unittest.TestCase):
    """Побитовое совпадение доказывает ноль. Расхождение не доказывает ничего:
    генерация недетерминирована, и объявлять её шум эффектом — та самая ошибка,
    против которой на проекте заведён негативный контроль."""

    def _frames(self, noise, effect, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.random((32, 32, 3))
        b = a + rng.normal(0, noise, a.shape)
        c = a + rng.normal(0, noise, a.shape) + effect
        return a, b, c

    def test_bit_identical_output_is_a_proven_zero(self):
        a, b, _ = self._frames(0.01, 0.0)
        got = fa.effect_over_noise(a, b, a)
        self.assertEqual(got["outcome"], FAIL)
        self.assertTrue(got["identical"])
        self.assertIn("ПОБИТОВО", got["note"])

    def test_a_shift_inside_the_noise_is_not_an_effect(self):
        a, b, c = self._frames(0.01, 0.0)
        got = fa.effect_over_noise(a, b, c)
        self.assertEqual(got["outcome"], FAIL, got["note"])
        self.assertLess(got["ratio"], fa.EFFECT_OVER_NOISE)

    def test_a_shift_well_over_the_noise_is_an_effect(self):
        a, b, c = self._frames(0.01, 0.2)
        got = fa.effect_over_noise(a, b, c)
        self.assertEqual(got["outcome"], PASS, got["note"])
        self.assertGreater(got["ratio"], fa.EFFECT_OVER_NOISE)

    def test_a_deterministic_generator_leaves_the_channel_unmeasured(self):
        """Пол ноль — делить не на что. Это «не смогли», а не «эффект бесконечен»."""
        a, _, c = self._frames(0.01, 0.2)
        got = fa.effect_over_noise(a, a, c)
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("детерминирован", got["note"])

    def test_frames_of_different_size_are_unmeasured(self):
        a = np.zeros((8, 8, 3))
        got = fa.effect_over_noise(a, a, np.zeros((4, 4, 3)))
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("разного размера", got["note"])

    def test_the_bar_can_move_in_both_directions(self):
        a, b, c = self._frames(0.01, 0.03)
        saved = fa.EFFECT_OVER_NOISE
        try:
            fa.EFFECT_OVER_NOISE = 100.0
            self.assertEqual(fa.effect_over_noise(a, b, c)["outcome"], FAIL)
            fa.EFFECT_OVER_NOISE = 1.01
            self.assertEqual(fa.effect_over_noise(a, b, c)["outcome"], PASS)
        finally:
            fa.EFFECT_OVER_NOISE = saved


    def test_one_frame_slightly_over_one_does_not_get_rescaled_alone(self):
        """Дефект, пойманный собственной фикстурой и стоивший правки кода.

        `fork_leak._as_array` решает про масштаб ПОКАДРОВО: max > 1 значит
        0..255. Для картинок с диска верно, для трёх кадров одного опыта — нет:
        кадр без адаптера с максимумом 0.9999 остаётся как есть, а кадр с
        адаптером, перевалив за единицу, делится на 255. Замер тогда дал пол
        0.4955 при заложенном шуме 0.01 и отношение ровно 1.0 — прибор
        сравнивал кадр с ним же, уменьшенным в 255 раз, и печатал «эффекта нет».
        """
        a = np.full((8, 8, 3), 0.99)
        b = a.copy()
        b[0, 0] = 0.98
        c = a + 0.02                      # максимум 1.01 — вот и вся ловушка
        got = fa.effect_over_noise(a, b, c)
        self.assertIsNotNone(got["ratio"], got["note"])
        self.assertGreater(got["ratio"], 10,
                           f"сдвиг 0.02 против шума в один пиксель дал "
                           f"отношение {got['ratio']} — кадры приведены к "
                           f"разным масштабам")

    def test_the_scale_decision_is_taken_once_for_all_three(self):
        """Негативный контроль к предыдущему: настоящие 0..255 обязаны
        приводиться, иначе «одно решение на всех» выродится в «не приводить»."""
        a = np.full((8, 8, 3), 200.0)
        b = a.copy()
        c = a + 5.0
        x, y, z = fa._three_on_one_scale(a, b, c)
        self.assertLess(x.max(), 1.01, "кадр 0..255 не приведён")
        self.assertAlmostEqual(float(z.max() - x.max()), 5 / 255, places=6)

    def test_both_numbers_are_printed_not_just_the_verdict(self):
        a, b, c = self._frames(0.01, 0.2)
        note = fa.effect_over_noise(a, b, c)["note"]
        self.assertIn("разброс генератора", note)
        self.assertIn("сдвиг с адаптером", note)


class TheChannelsAreCombinedByTheWorstNotTheAverage(unittest.TestCase):

    def test_the_worst_wins(self):
        got = fa.report(static={"outcome": PASS, "note": "имена сошлись"},
                        numeric={"outcome": FAIL, "note": "эффекта нет"})
        self.assertEqual(got["outcome"], FAIL,
                         "зелёные имена перевесили красное число — а это и "
                         "есть находка, ради которой прибор написан")

    def test_unmeasured_is_not_swallowed_by_a_pass(self):
        got = fa.report(static={"outcome": PASS, "note": ""},
                        log={"outcome": UNMEASURED, "note": ""})
        self.assertEqual(got["outcome"], UNMEASURED)

    def test_all_green_is_the_only_way_to_green(self):
        got = fa.report(static={"outcome": PASS, "note": ""},
                        log={"outcome": PASS, "note": ""},
                        numeric={"outcome": PASS, "note": ""})
        self.assertEqual(got["outcome"], PASS)

    def test_no_channels_at_all_is_unmeasured_not_clean(self):
        got = fa.report()
        self.assertEqual(got["outcome"], UNMEASURED)
        self.assertIn("не успех", got["note"])

    def test_how_many_channels_ran_is_printed(self):
        got = fa.report(static={"outcome": PASS, "note": ""})
        self.assertIn("Каналов отработало 1 из 3", got["note"])


class TheSweepRunsTheAdapterStep(unittest.TestCase):
    """Прибор, не подключённый к пути, — мёртвый код в отчёте.

    Он и написан против того, что между «загрузили файл» и «оценили эффект»
    (`lora.accept`) шага «проверили, что приложилось» не было.
    """

    def _run(self, **kw):
        from PIL import Image
        from ball_reel import fork_run

        tmp = tempfile.mkdtemp()
        photo = Path(tmp) / "p.png"
        Image.fromarray(
            (np.random.rand(64, 64, 3) * 255).astype("uint8")).save(photo)
        got = fork_run.run(photo, [], Path(tmp) / "out", **kw)
        return tmp, next(s for s in got["steps"] if s["step"] == "адаптер")

    def test_the_step_stands_after_the_graph_and_before_generation_axes(self):
        from ball_reel import fork_run

        order = list(fork_run.STEPS)
        self.assertLess(order.index("граф"), order.index("адаптер"),
                        "имена тензоров берутся из того, что грузит граф")
        for axis in ("протечка", "шов"):
            self.assertLess(order.index("адаптер"), order.index(axis),
                            "адаптер, который не приложится, обесценивает "
                            "весь дорогой прогон — проверять его после осей "
                            "поздно")

    def test_no_adapter_is_unmeasured_and_not_read_as_lora_not_needed(self):
        _, step = self._run()
        self.assertEqual(step["outcome"], UNMEASURED)
        self.assertIn("базовая линия без адаптера штатна", step["note"])

    def test_a_peft_dora_adapter_makes_the_step_red(self):
        """Тот самый молчаливый дефект: магнитуда под чужим именем."""
        tmp = tempfile.mkdtemp()
        p = _adapter(["blocks.0.ffn.0.lora_up.weight",
                      "blocks.0.ffn.0.lora_magnitude_vector.weight"], tmp)
        _, step = self._run(adapter=p)
        self.assertEqual(step["outcome"], FAIL, step["note"])
        self.assertIn("dora_scale", step["note"])

    def test_a_plain_adapter_is_unmeasured_because_model_names_are_absent(self):
        """На моке имён тензоров модели взять неоткуда — и это «не смогли»,
        а не «годно»: зелёный вердикт без проверки хуже отсутствия вердикта."""
        tmp = tempfile.mkdtemp()
        p = _adapter(["blocks.0.ffn.0.lora_up.weight"], tmp)
        _, step = self._run(adapter=p)
        self.assertEqual(step["outcome"], UNMEASURED)
        self.assertIn("ключей в адаптере 1", step["note"])

    def test_a_broken_adapter_file_fails_and_names_the_file(self):
        tmp = tempfile.mkdtemp()
        bad = Path(tmp) / "bad.safetensors"
        bad.write_bytes(b"\x00")
        _, step = self._run(adapter=bad)
        self.assertEqual(step["outcome"], FAIL)


if __name__ == "__main__":
    unittest.main()
