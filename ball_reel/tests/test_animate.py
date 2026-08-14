"""Локальная генерация: проверяется то, что проверяемо БЕЗ карты.

Весов здесь нет и быть не может — карты в среде разработки нет. Поэтому
тестируется ровно то, что решается до загрузки: план по памяти, требование к
кадрировке, и отказы, которые обязаны случиться ДО дорогого шага.

Это не подмена живого прогона и не притворяется ею. Сегодняшний день показал,
что первое исполнение находит дефекты в каждом модуле; задача этих тестов —
чтобы к моменту первого исполнения уже не осталось ошибок, видных на бумаге.
"""

from __future__ import annotations

import unittest


class ThePlanIsMadeBeforeAnyWeightsAreLoaded(unittest.TestCase):
    """Ошибка режима обязана быть видна за миллисекунду, а не после падения."""

    def setUp(self):
        from ball_reel import animate

        self.a = animate

    def test_more_memory_buys_resolution_not_frames(self):
        # Кадры задаёт окно модуля движения, а не память: он обучен на 16, и
        # растягивать окно памятью нельзя. Разрешение — можно.
        small = self.a.plan(6.0)
        big = self.a.plan(12.0)
        self.assertGreater(big.height, small.height)
        self.assertEqual(small.frames, big.frames)

    def test_a_small_card_falls_back_to_sequential_offload(self):
        tiny = self.a.plan(4.0)
        self.assertEqual(tiny.offload, "sequential")
        self.assertTrue(any("медленнее" in n for n in tiny.notes))

    def test_six_gigabytes_is_flagged_as_tight_with_what_to_drop(self):
        # Предупреждение бесполезно, если не говорит, ЧТО снимать первым.
        got = self.a.plan(6.0)
        self.assertEqual(got.offload, "model")
        self.assertTrue(any("ControlNet" in n for n in got.notes), got.notes)


class ResolutionIsAREQUIREMENTOnFraming(unittest.TestCase):
    """Потолок памяти превращается в вопрос «увидит ли гейт лицо».

    Это главное открытие прошлых прогонов, и оно должно быть сторожено кодом,
    а не памятью автора: на полном росте лицо занимает 9.4% высоты кадра, и на
    512x768 это ~72 px при пороге судимости 70.
    """

    def setUp(self):
        from ball_reel import animate

        self.a = animate

    def test_waist_up_doubles_the_face_and_that_is_the_whole_point(self):
        full = self.a.plan(6.0, waist_up=False)
        waist = self.a.plan(6.0, waist_up=True)
        self.assertAlmostEqual(waist.face_px / full.face_px, 2.0, delta=0.15)

    def test_a_face_too_small_to_judge_is_declared_not_silently_passed(self):
        # Литеральные числа: 300 px высоты на полном росте дают ~28 px лица.
        # Тест, берущий вход из константы, которую сторожит, не падает никогда.
        got = self.a.plan(6.0, waist_up=False)
        got.face_px = 300 * self.a.FULL_BODY_FACE_SHARE
        self.assertLess(got.face_px, self.a.MIN_JUDGED_FACE_PX)
        low = self.a.plan(3.0, waist_up=False)
        self.assertFalse(low.identity_verifiable)
        self.assertTrue(any("не проверялась" in n for n in low.notes), low.notes)

    def test_the_verdict_is_spelled_out_in_the_rendered_plan(self):
        text = self.a.plan(3.0, waist_up=False).render()
        self.assertIn("НЕ ПРОВЕРЯЕТСЯ", text)
        self.assertIn("PX".lower(), text.lower())


class TheExpensiveStepIsRefusedEarly(unittest.TestCase):
    """Отказ обязан случиться до загрузки весов, а не в середине генерации."""

    def setUp(self):
        from ball_reel import animate

        self.a = animate

    def test_a_wrong_number_of_conditions_is_refused_with_the_reason(self):
        cfg = self.a.plan(6.0)
        with self.assertRaises(ValueError) as cm:
            self.a.animate(object(), cfg, face_embeds=None,
                           conditions=[None] * (cfg.frames + 3), prompt="x")
        self.assertIn(str(self.a.CONTEXT_FRAMES), str(cm.exception))

    def test_an_unknown_motion_lora_is_named_along_with_the_real_ones(self):
        cfg = self.a.plan(6.0)
        with self.assertRaises(ValueError) as cm:
            self.a.build(cfg, motion_lora="corkscrew")
        self.assertIn("zoom-in", str(cm.exception))


class WhatIsLOADEDIsCheckedNotClaimed(unittest.TestCase):
    """«Подключили LoRA» — утверждение, и оно обязано быть проверяемым.

    То же правило, по которому гейт не спрашивает генератор, получилось ли у
    него: на демо будет сказано, что LoRA несёт личность, и это должно
    подтверждаться состоянием модели, а не словами в слайде.
    """

    def setUp(self):
        from ball_reel import animate

        self.a = animate

    def test_adapters_are_read_from_the_pipeline(self):
        class Pipe:
            def get_active_adapters(self):
                return ["faceid_0", "motion_zoom-in"]

        self.assertEqual(self.a.active_loras(Pipe()),
                         ["faceid_0", "motion_zoom-in"])

    def test_a_dict_shaped_answer_is_flattened(self):
        class Pipe:
            def get_list_adapters(self):
                return {"unet": ["faceid_0"], "text_encoder": ["faceid_0"]}

        self.assertEqual(self.a.active_loras(Pipe()), ["faceid_0"])

    def test_a_pipeline_that_cannot_answer_reports_nothing_not_success(self):
        # Молчаливый пустой список честнее выдуманного имени: «не знаем, что
        # прицеплено» и «прицеплено то, что мы просили» — разные утверждения.
        class Pipe:
            def get_active_adapters(self):
                raise RuntimeError("peft не установлен")

        self.assertEqual(self.a.active_loras(Pipe()), [])


class PreflightSaysWhatToDoNotJustWhatIsWrong(unittest.TestCase):
    def setUp(self):
        from ball_reel import animate

        self.a = animate

    def test_a_cpu_build_of_torch_is_named_with_the_fix(self):
        # Ровно та ловушка, в которую мы уже попали: pip поставил torch без
        # индекса CUDA, получилась сборка +cpu, и вся GPU-ветка молча мертва.
        try:
            import torch
        except ImportError:
            self.skipTest("torch не установлен")
        if getattr(torch.version, "cuda", None):
            # Здесь колесо с PyPI собрано С CUDA (просто карты в контейнере
            # нет), значит ловушку «CPU-сборка» на этой машине не поставить.
            # Пропуск честнее, чем подмена условия заглушкой.
            self.skipTest(f"torch собран с CUDA {torch.version.cuda} — "
                          f"CPU-сборку не воспроизвести")
        rep = self.a.preflight(vram_gb=6.0)
        self.assertTrue(any("cu124" in n for n in rep["notes"]), rep["notes"])

    def test_the_plan_is_part_of_the_preflight_report(self):
        rep = self.a.preflight(vram_gb=6.0)
        self.assertIn("512x768", rep["plan"])


if __name__ == "__main__":
    unittest.main()


class MemoryIsPLANNEDBeforeAndMEASUREDAfter(unittest.TestCase):
    """Расчёт и замер — разные вещи, и путать их нельзя.

    До прогона модуль умеет только арифметику по размерам файлов весов. Это
    полезно (плохой режим виден за миллисекунду), но подать её как замер
    значило бы то же самое, за что в этом проекте уже правились метрики: число,
    полученное не тем способом, каким заявлено.
    """

    def setUp(self):
        from ball_reel import animate

        self.a = animate

    def test_the_weights_are_summed_from_the_real_file_sizes(self):
        # Числа взяты из HF API делением fp32 пополам. Тест сторожит не сами
        # значения, а то, что учтены ВСЕ потребители: забыть ControlNet в этой
        # сумме — значит обещать полтора лишних гигабайта запаса.
        cfg = self.a.plan(6.0)
        for part in ("unet", "motion_adapter", "controlnet", "vae",
                     "text_encoder", "ip_adapter"):
            self.assertIn(part, self.a.WEIGHTS_GB)
        self.assertAlmostEqual(cfg.weights_gb, sum(self.a.WEIGHTS_GB.values()),
                               places=2)

    def test_headroom_is_what_is_left_for_ACTIVATIONS(self):
        # Разделение потребителей: веса постоянны, активации растут как
        # разрешение x кадры. Смешать их в одно число — потерять возможность
        # рассуждать о том, что вообще экономить.
        cfg = self.a.plan(6.0)
        self.assertAlmostEqual(cfg.headroom_gb(6.0), 6.0 - cfg.weights_gb,
                               places=2)
        self.assertGreater(cfg.headroom_gb(12.0), cfg.headroom_gb(6.0))

    def test_the_plan_says_out_loud_that_it_is_a_CALCULATION(self):
        text = "\n".join(self.a.plan(6.0).notes)
        self.assertIn("РАСЧЁТ", text)
        self.assertIn("не замер", text)

    def test_a_tight_card_turns_on_vae_tiling_a_roomy_one_does_not(self):
        self.assertTrue(self.a.plan(6.0).vae_tiling)
        self.assertFalse(self.a.plan(12.0).vae_tiling)

    def test_the_allocator_is_set_before_cuda_and_reports_what_it_did(self):
        import os

        was = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        self.addCleanup(lambda: os.environ.__setitem__(
            "PYTORCH_CUDA_ALLOC_CONF", was) if was else None)
        got = self.a.prepare_allocator()
        self.assertIn("expandable_segments", got)
        self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"],
                         self.a.ALLOC_CONF)
        # Заданное человеком не перетирается: он мог знать больше нас.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.assertIn("оставлено", self.a.prepare_allocator())
        self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"],
                         "max_split_size_mb:128")

    def test_without_a_card_the_peak_is_NOT_MEASURED_rather_than_zero(self):
        # Ноль здесь читался бы как «памяти не понадобилось». Отсутствие
        # измерения — третий исход, и он обязан быть отличим.
        try:
            import torch
        except ImportError:
            self.skipTest("torch не установлен")
        if torch.cuda.is_available():
            self.skipTest("карта есть — случай «мерить нечем» не воспроизвести")
        self.assertIsNone(self.a.peak_memory())


class FusingQKVWouldDestroyTheIdentityChannel(unittest.TestCase):
    """Оптимизация памяти, стирающая то, ради чего собран пайплайн.

    ИЗМЕРЕНО НА ЖИВОМ ПРОГОНЕ и воспроизведено на CPU (diffusers 0.39):

        процессоров IP-Adapter до `fuse_qkv_projections()` : 16 из 72
        после                                              : 0
        первый шаг генерации  : AttributeError: 'tuple' object has no
                                attribute 'shape'

    Слияние заменяет ВСЕ процессоры внимания на `FusedAttnProcessor2_0` —
    включая те шестнадцать, которые и есть канал личности. Падение здесь
    ПОВЕЗЛО: при активном IP-Adapter diffusers пакует `encoder_hidden_states`
    в кортеж (текст, эмбеддинг лица), а слитый процессор кортежа не понимает.
    Не упади оно — вышел бы клип без личности вовсе, и списали бы это на дрейф.

    Проверяется УСЛОВИЕ, а не сам вызов: поднимать веса ради теста нельзя, а
    решение «сливать или нет» — чистая функция от состава процессоров.
    """

    def _decide(self, processors, *, fuse_qkv=True):
        """Тот же предикат, что в `animate.build`, на списке имён классов."""
        ip = sum(1 for name in processors if "IPAdapter" in name)
        return bool(fuse_qkv and not ip), ip

    def test_fusing_is_skipped_when_the_adapter_is_attached(self):
        fuse, ip = self._decide(["IPAdapterAttnProcessor2_0"] * 16
                                + ["AttnProcessor2_0"] * 56)
        self.assertEqual(ip, 16)
        self.assertFalse(fuse, "слияние затёрло бы канал личности")

    def test_fusing_still_happens_without_the_adapter(self):
        """Без адаптера экономия памяти бесплатна и остаётся в силе."""
        fuse, ip = self._decide(["AttnProcessor2_0"] * 72)
        self.assertEqual(ip, 0)
        self.assertTrue(fuse)

    def test_the_flag_still_wins_when_switched_off(self):
        fuse, _ = self._decide(["AttnProcessor2_0"] * 72, fuse_qkv=False)
        self.assertFalse(fuse)

    def test_build_decides_by_processors_not_by_a_try_except(self):
        """Ловить исключение было бы поздно: процессоры уже затёрты."""
        import inspect

        from ball_reel import animate

        src = inspect.getsource(animate.build)
        self.assertIn("IPAdapter", src,
                      "решение принимается не по составу процессоров")
        self.assertIn("ПРОПУЩЕНО", src)


class TheSchedulerIsNotLeftToTheBaseModel(unittest.TestCase):
    """Умолчание SD1.5 не годится модулю движения, и молчит об этом.

    ИЗМЕРЕНО НА ПЕРВОМ ЖИВОМ ПРОГОНЕ (31 минута на карте): промт доехал
    целиком — красная майка, зал, гантели, — а картинка вышла «живописной»,
    мазками вместо фактуры кожи, и ArcFace не нашёл лица НИ НА ОДНОМ из
    16 кадров. Мы списывали это на мелкое лицо (74 px при баре 100), но при
    74 px детектор обычно что-то находит: «не найдено» означает мусор.

    Причина: `animate.build` не задавал планировщик вовсе. Бралось умолчание
    SD1.5 — `PNDMScheduler` с `beta_schedule="scaled_linear"`, — а модуль
    движения AnimateDiff обучался с `linear`. Рассогласование не падает и
    ничего не печатает.

    Худший вид отказа: тот, который похож на работу. Гейт при этом честно
    краснел, но объяснял невнятно — «поза не та», «лицо не найдено», — то есть
    называл следствия, а не причину.
    """

    def setUp(self):
        from ball_reel import animate

        self.a = animate

    def test_beta_schedule_is_linear_not_the_sd15_default(self):
        """Главный из четырёх параметров: с ним и живёт модуль движения."""
        self.assertEqual(self.a.MOTION_SCHEDULER["beta_schedule"], "linear")
        self.assertNotEqual(self.a.MOTION_SCHEDULER["beta_schedule"],
                            "scaled_linear")

    def test_the_other_three_come_with_it(self):
        s = self.a.MOTION_SCHEDULER
        self.assertIs(s["clip_sample"], False)
        self.assertEqual(s["timestep_spacing"], "linspace")
        self.assertEqual(s["steps_offset"], 1)

    def test_build_uses_the_constant_not_a_literal(self):
        """Литерал внутри функции не мутируется аудитом и молча разъедется."""
        import inspect

        src = inspect.getsource(self.a.build)
        self.assertIn("MOTION_SCHEDULER", src)
        self.assertIn("DDIMScheduler", src)

    def test_the_base_default_is_what_we_are_overriding(self):
        """Замер, ради которого правка существует: у базы стоит scaled_linear.

        Читается из конфига скачанных весов, а не утверждается по памяти. Нет
        весов — нечего и сверять, тест пропускается.
        """
        import glob
        import json
        import os

        hub = os.path.join(os.path.expanduser(
            os.environ.get("HF_HOME", "~/.cache/huggingface")), "hub")
        found = glob.glob(os.path.join(
            hub, "models--stable-diffusion-v1-5--stable-diffusion-v1-5",
            "snapshots", "*", "scheduler", "scheduler_config.json"))
        if not found:
            self.skipTest("веса SD1.5 не скачаны — сверять не с чем")
        cfg = json.loads(open(found[0], encoding="utf-8").read())
        self.assertEqual(cfg.get("beta_schedule"), "scaled_linear",
                         "у базы уже не то умолчание — правку перепроверить")


class AttentionSlicingMustNotCostTheIdentityChannel(unittest.TestCase):
    """Вторая экономия, стирающая то, ради чего собран пайплайн.

    ИЗМЕРЕНО, diffusers 0.39, на `UNet2DConditionModel`:

        после load_ip_adapter    16 процессоров IP-Adapter
        после attention_slicing   0   класс SlicedAttnProcessor

    А на `UNetMotionModel` те же 16 ОСТАЮТСЯ — проверено тем же замером.
    Поэтому запрет был бы неверен: он отнял бы экономию там, где она
    безвредна. Правильный ответ — включить и посмотреть на СОСТОЯНИЕ МОДЕЛИ,
    откатив только при потере.

    Это уже второй случай той же формы за вечер (первый — fuse_qkv), и общее у
    них одно: оптимизация памяти молча снимает процессоры внимания, а замечаем
    мы это по падению `'tuple' object has no attribute 'shape'` через минуту
    генерации.
    """

    class _Proc:
        def __init__(self, name):
            self.__class__ = type(name, (object,), {})

    def _pipe(self, *, wipes: bool, ip: int = 16, total: int = 72):
        """Двойник пайплайна: считает процессоры и умеет их стирать."""
        import types

        class IPAdapterAttnProcessor2_0:
            pass

        class AttnProcessor2_0:
            pass

        class SlicedAttnProcessor:
            pass

        procs = {f"p{i}": (IPAdapterAttnProcessor2_0() if i < ip
                           else AttnProcessor2_0()) for i in range(total)}
        unet = types.SimpleNamespace(attn_processors=dict(procs))

        def set_attn_processor(d):
            unet.attn_processors = dict(d)

        unet.set_attn_processor = set_attn_processor
        pipe = types.SimpleNamespace(unet=unet, sliced=False)

        def enable():
            pipe.sliced = True
            if wipes:
                unet.attn_processors = {k: SlicedAttnProcessor()
                                        for k in unet.attn_processors}

        pipe.enable_attention_slicing = enable
        return pipe

    def _ip(self, pipe):
        return sum(1 for v in pipe.unet.attn_processors.values()
                   if "IPAdapter" in type(v).__name__)

    def test_a_wiping_slicer_is_rolled_back(self):
        from ball_reel.animate import enable_slicing_without_losing_identity

        pipe = self._pipe(wipes=True)
        note = enable_slicing_without_losing_identity(pipe, verbose=False)
        self.assertEqual(self._ip(pipe), 16, "канал личности не восстановлен")
        self.assertIn("ОТКАЧЕНА", note)

    def test_a_harmless_slicer_is_kept(self):
        """На UNetMotionModel экономия безвредна — отнимать её нельзя."""
        from ball_reel.animate import enable_slicing_without_losing_identity

        pipe = self._pipe(wipes=False)
        note = enable_slicing_without_losing_identity(pipe, verbose=False)
        self.assertTrue(pipe.sliced, "экономия отменена там, где безвредна")
        self.assertEqual(self._ip(pipe), 16)
        self.assertIn("цел", note)

    def test_without_an_adapter_there_is_nothing_to_protect(self):
        from ball_reel.animate import enable_slicing_without_losing_identity

        pipe = self._pipe(wipes=True, ip=0)
        enable_slicing_without_losing_identity(pipe, verbose=False)
        self.assertTrue(pipe.sliced)

    def test_a_pipeline_that_cannot_be_asked_still_gets_slicing(self):
        """Нельзя терять экономию из-за того, что модель не опрашивается."""
        import types

        from ball_reel.animate import enable_slicing_without_losing_identity

        pipe = types.SimpleNamespace(sliced=False)
        pipe.enable_attention_slicing = lambda: setattr(pipe, "sliced", True)
        enable_slicing_without_losing_identity(pipe, verbose=False)
        self.assertTrue(pipe.sliced)
