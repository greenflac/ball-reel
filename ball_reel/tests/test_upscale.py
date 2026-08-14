"""Апскейл: проверяется всё, что проверяемо БЕЗ карты и БЕЗ весов.

Карты в среде разработки нет, весов на чужой машине может не быть — поэтому
граница проведена так: механизм защиты порядка, геометрия тайлов, арифметика
памяти и спектральная метрика проверяются ЦЕЛИКОМ и никогда не пропускаются, а
всё, что требует Swin2SR, живёт в отдельном классе с явным `skipTest` и
причиной из `why_unavailable()`.

ОТ ЧЕГО ЭТИ ТЕСТЫ НАМЕРЕННО НЕ ЗАВИСЯТ. На этом проекте уже был тест, который
зеленел, пока пакета не было, и покраснел после установки. Здесь ни один
вердикт не выводится из наличия весов: запасной путь проверяется явным
`method="lanczos"`, а не «выключим transformers и посмотрим». Единственное
место, где наличие пакета вообще влияет на исход, — `why_unavailable`, и там
проверяется ФОРМА ответа (причина или None), а не то, какая из двух ветвей
сработала на этой машине.

Синтетика покрывает ДИАПАЗОН, а не одно удобное значение: кадры от 8x8 до
1024x1536, тайлы от «меньше кадра» до «больше кадра», спектры от плоской
заливки до шахматки на Найквисте. Причина измеренная — проект уже получил
слепую метрику из-за фикстур, заливавших окно на 100%.

Числа во входах записаны литералами. Тест, берущий вход из константы, которую
сторожит, не падает никогда: сдвинь константу — и «ожидаемое» уедет вместе с ней.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
    from PIL import Image

    HAVE_DEPS = True
except ImportError:                                 # pragma: no cover
    HAVE_DEPS = False


def _seal(mod, frames, gate="test.gate"):
    """Печать на эти кадры — короткая обёртка, чтобы тесты читались."""
    return mod.seal_verdict(frames, {"outcome": "ok"}, gate=gate)


def _img(w, h, kind="ramp", seed=0):
    """Синтетический кадр. `kind` задаёт СПЕКТР, а не красоту.

    flat — заливка (нулевая высокочастотная энергия), ramp — градиент (вся
    энергия внизу), checker — шахматка 1x1 (энергия на самом Найквисте),
    noise — равномерный шум (энергия размазана по всему спектру).
    """
    a = np.zeros((h, w, 3), dtype=np.uint8)
    if kind == "flat":
        a[:] = 128
    elif kind == "ramp":
        a[:] = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None]
    elif kind == "checker":
        yy, xx = np.mgrid[0:h, 0:w]
        a[:] = np.where(((yy + xx) % 2) == 0, 255, 0)[:, :, None]
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        a[:] = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    else:                                            # pragma: no cover
        raise ValueError(kind)
    return Image.fromarray(a)


# --------------------------------------------------------------- порядок


class TheOrderIsMadeImpossibleToGetWrong(unittest.TestCase):
    """Главное требование ступени: гейт судит ДО апскейла, и не по просьбе.

    Здесь проверяется не то, что в докстринге написано «сначала гейт», а то,
    что противоположный порядок НЕ СОБИРАЕТСЯ. Каждый тест — отдельный способ
    обойти правило, и каждый обязан упереться.
    """

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale
        self.u._PRODUCED.clear()

    def test_upscaling_without_any_seal_is_refused(self):
        frames = [_img(16, 24)]
        with self.assertRaises(ValueError) as ctx:
            self.u.upscale_frames(frames, None)
        self.assertIn("печат", str(ctx.exception).lower())

    def test_a_bare_truthy_value_is_not_a_seal(self):
        # Булев «да, гейт был» — ровно та ошибка, ради которой печать и
        # заведена: он сообщает не то, что было, а то, что помнит вызывающий.
        frames = [_img(16, 24)]
        for fake in (True, 1, "прошли", {"ok": True}, object()):
            with self.subTest(fake=type(fake).__name__):
                with self.assertRaises(ValueError):
                    self.u.upscale_frames(frames, fake)

    def test_a_seal_from_other_pixels_does_not_cover_these(self):
        # Печать привязана к СОДЕРЖИМОМУ: подменённый кадр обязан упереться,
        # хотя печать настоящая и гейт настоящий.
        judged = [_img(16, 24, "ramp")]
        other = [_img(16, 24, "noise")]
        seal = _seal(self.u, judged)
        with self.assertRaises(ValueError) as ctx:
            self.u.upscale_frames(other, seal)
        self.assertIn("не накрывает", str(ctx.exception))

    def test_one_smuggled_frame_among_judged_ones_is_enough_to_refuse(self):
        judged = [_img(16, 24, "ramp"), _img(16, 24, "noise", seed=1)]
        seal = _seal(self.u, judged)
        smuggled = judged + [_img(16, 24, "noise", seed=2)]
        with self.assertRaises(ValueError):
            self.u.upscale_frames(smuggled, seal)

    def test_the_seal_covers_a_subset_because_order_of_judging_is_free(self):
        judged = [_img(16, 24, "ramp"), _img(16, 24, "noise", seed=3)]
        seal = _seal(self.u, judged)
        got, rep = self.u.upscale_frames(judged[:1], seal, method="lanczos")
        self.assertEqual(rep["outcome"], "fallback")
        self.assertEqual(len(got), 1)

    def test_what_the_upscaler_produced_cannot_be_sealed_afterwards(self):
        # Обход «апскейлить -> посудить апскейленное -> запечатать задним
        # числом». Это и есть запрещённый порядок, просто записанный иначе.
        judged = [_img(16, 24, "noise", seed=4)]
        out, _ = self.u.upscale_frames(judged, _seal(self.u, judged),
                                       method="lanczos")
        with self.assertRaises(ValueError) as ctx:
            self.u.seal_verdict(out, {"outcome": "ok"}, gate="test.gate")
        self.assertIn("произведены апскейлером", str(ctx.exception))

    def test_a_seal_without_a_gate_name_is_refused(self):
        for name in ("", "   "):
            with self.subTest(name=repr(name)):
                with self.assertRaises(ValueError):
                    self.u.seal_verdict([_img(16, 24)], {"ok": True}, gate=name)

    def test_a_missing_verdict_is_not_the_same_as_a_verdict(self):
        with self.assertRaises(ValueError) as ctx:
            self.u.seal_verdict([_img(16, 24)], None, gate="test.gate")
        self.assertIn("None", str(ctx.exception))

    def test_the_seal_carries_the_gate_name_into_the_report(self):
        # Отчёт демо обязан показывать, ЧЕЙ вердикт относится к этим пикселям.
        frames = [_img(16, 24)]
        seal = self.u.seal_verdict(frames, {"outcome": "не проверено"},
                                   gate="identity_arcface.arcface_drift")
        _, rep = self.u.upscale_frames(frames, seal, method="lanczos")
        self.assertEqual(rep["gate"], "identity_arcface.arcface_drift")
        self.assertEqual(rep["verdict_outcome"], "не проверено")

    def test_a_refusing_verdict_still_seals_because_order_is_not_quality(self):
        # Печать — про ПОРЯДОК, а не про то, понравился ли вердикт. Клип,
        # который гейт забраковал, показать крупнее можно; соврать про то,
        # когда его судили, — нельзя.
        frames = [_img(16, 24)]
        seal = self.u.seal_verdict(frames, {"ok": False}, gate="test.gate")
        _, rep = self.u.upscale_frames(frames, seal, method="lanczos")
        self.assertIs(rep["verdict_outcome"], False)

    def test_an_empty_sequence_is_refused_on_both_sides(self):
        with self.assertRaises(ValueError):
            self.u.seal_verdict([], {"ok": True}, gate="test.gate")
        with self.assertRaises(ValueError):
            self.u.upscale_frames([], _seal(self.u, [_img(8, 8)]))

    def test_the_seal_cannot_be_extended_after_the_fact(self):
        seal = _seal(self.u, [_img(16, 24)])
        with self.assertRaises(Exception):
            seal.digests = frozenset()


class TheDigestIsTakenFromPixelsNotFromNames(unittest.TestCase):
    """Печать по имени файла разрешила бы судить одно, а апскейлить другое."""

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale

    def test_same_pixels_give_the_same_digest_through_different_objects(self):
        a = _img(16, 24, "ramp")
        b = _img(16, 24, "ramp")
        self.assertEqual(self.u.frame_digest(a), self.u.frame_digest(b))

    def test_one_changed_pixel_changes_the_digest(self):
        a = np.zeros((24, 16, 3), dtype=np.uint8)
        b = a.copy()
        b[12, 8, 0] = 1
        self.assertNotEqual(self.u.frame_digest(a), self.u.frame_digest(b))

    def test_the_same_bytes_in_a_different_shape_are_not_the_same_frame(self):
        # 512x768 и его транспонированный близнец состоят из одних байтов.
        a = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(24, 16, 3)
        b = a.reshape(16, 24, 3)
        self.assertNotEqual(self.u.frame_digest(a), self.u.frame_digest(b))

    def test_a_file_is_digested_by_content(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "0000.png"
            _img(16, 24, "ramp").save(p)
            first = self.u.frame_digest(str(p))
            _img(16, 24, "noise").save(p)          # то же имя, другие пиксели
            self.assertNotEqual(first, self.u.frame_digest(str(p)))

    def test_something_that_is_not_pixels_is_refused_loudly(self):
        for junk in (42, 3.5, object()):
            with self.subTest(junk=type(junk).__name__):
                with self.assertRaises(TypeError):
                    self.u.frame_digest(junk)


# --------------------------------------------------------------- три исхода


class ThreeOutcomesNotTwo(unittest.TestCase):
    """«Весами», «запасным путём» и «не выполнен» — три разных состояния.

    Схлопывание любых двух означает отчёт, в котором написано не то, что
    произошло: демо с Lanczos'ом вместо Swin2SR выглядело бы как демо с
    весами, а пропущенный апскейл — как неудачный.
    """

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale
        self.u._PRODUCED.clear()

    def test_the_fallback_says_it_is_the_fallback(self):
        frames = [_img(16, 24)]
        out, rep = self.u.upscale_frames(frames, _seal(self.u, frames),
                                         method="lanczos")
        self.assertEqual(rep["outcome"], "fallback")
        self.assertEqual(rep["method"], "lanczos")
        self.assertIs(rep["invents"], False)
        self.assertEqual(out[0].size, (32, 48))

    def test_skipping_is_an_outcome_and_not_a_failed_upscale(self):
        frames = [_img(16, 24)]
        out, rep = self.u.upscale_frames(frames, _seal(self.u, frames),
                                         method="none")
        self.assertEqual(rep["outcome"], "skipped")
        self.assertIsNone(rep["method"])
        self.assertIsNone(rep["invents"])
        self.assertEqual(out[0].size, (16, 24))

    def test_factor_one_is_skipped_rather_than_a_pointless_resize(self):
        frames = [_img(16, 24)]
        out, rep = self.u.upscale_frames(frames, _seal(self.u, frames), factor=1)
        self.assertEqual(rep["outcome"], "skipped")
        self.assertEqual(out[0].size, (16, 24))

    def test_the_asked_factor_and_the_applied_one_are_both_visible(self):
        # Попросили x2, исход «не выполнен» — применённый множитель 1. Если в
        # отчёте одно поле, отчёт врёт про одну из двух величин.
        frames = [_img(16, 24)]
        _, rep = self.u.upscale_frames(frames, _seal(self.u, frames),
                                       factor=2, method="none")
        self.assertEqual((rep["factor_asked"], rep["factor"]), (2, 1))
        self.assertEqual(rep["size_out"], (16, 24))

    def test_invents_is_three_valued_and_never_a_bare_flag(self):
        # Булев флаг рядом с трёхзначным исходом — приглашение схлопнуть
        # состояния. Здесь их именно три, и None означает «выдумывать было
        # нечему», а не «не выдумывает».
        frames = [_img(16, 24)]
        seen = {}
        for method in ("lanczos", "none"):
            _, rep = self.u.upscale_frames(frames, _seal(self.u, frames),
                                           method=method)
            seen[rep["outcome"]] = rep["invents"]
        self.assertEqual(seen, {"fallback": False, "skipped": None})

    def test_the_report_has_no_boolean_ok_to_collapse_the_outcomes_into(self):
        frames = [_img(16, 24)]
        _, rep = self.u.upscale_frames(frames, _seal(self.u, frames),
                                       method="lanczos")
        self.assertNotIn("ok", rep)
        self.assertNotIn("success", rep)

    def test_demanding_weights_never_silently_falls_back(self):
        # Демо, обещавшее Swin2SR и показавшее интерполяцию, врёт о себе.
        # Поэтому при method="weights" отсутствие весов — ошибка, а не откат.
        frames = [_img(16, 24)]
        seal = _seal(self.u, frames)
        if self.u.why_unavailable() is None:
            _, rep = self.u.upscale_frames(frames, seal, method="weights",
                                           tile=64)
            self.assertEqual(rep["outcome"], "weights")
        else:
            with self.assertRaises(ValueError) as ctx:
                self.u.upscale_frames(frames, seal, method="weights")
            self.assertIn("весов нет", str(ctx.exception))

    def test_an_unknown_method_is_refused_rather_than_guessed(self):
        frames = [_img(16, 24)]
        with self.assertRaises(ValueError):
            self.u.upscale_frames(frames, _seal(self.u, frames), method="swin")

    def test_the_licence_and_the_repo_are_named_in_every_report(self):
        # Лицензия сторонних весов — вопрос до сборки, а не после, и в отчёте
        # прогона она обязана быть видна без похода в докстринг.
        frames = [_img(16, 24)]
        _, rep = self.u.upscale_frames(frames, _seal(self.u, frames),
                                       method="lanczos")
        self.assertEqual(rep["license"], "apache-2.0")
        self.assertEqual(rep["repo"], "caidas/swin2SR-classical-sr-x2-64")


class TheReasonWeightsAreMissingIsNamed(unittest.TestCase):
    """«Апскейл не поехал» без причины стоит часа. Причины лечатся по-разному."""

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale

    def test_the_answer_is_either_none_or_an_actionable_reason(self):
        # Форма, а не ветвь: на машине с весами это None, без весов — строка
        # с командой. Тест обязан быть зелёным в обоих случаях, иначе он
        # меряет, что установлено на машине, а не поведение модуля.
        got = self.u.why_unavailable()
        if got is not None:
            self.assertIsInstance(got, str)
            self.assertTrue(got.strip())

    def test_the_upscaler_is_released_on_demand_because_the_card_is_6gb(self):
        # Ступень последняя, но веса с карты сами не уходят — на 6 ГБ это уже
        # не гигиена. Проверяется без весов: снимается сам кэш.
        self.u._LOADED["cpu"] = ("процессор", "модель")
        self.u.free_upscaler()
        self.assertEqual(self.u._LOADED, {})

    def test_a_repo_that_cannot_be_there_produces_a_reason_with_the_licence(self):
        got = self.u.why_unavailable("caidas/этого-репозитория-нет-и-не-было")
        self.assertIsNotNone(got)
        self.assertIn("apache-2.0", got)


# ------------------------------------------------------------------- план


class ThePlanIsArithmeticAndBorrowsItsConstants(unittest.TestCase):
    """План считается ДО GPU-минут, а доля лица и бар не копируются сюда."""

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale

    def test_the_generation_canvas_doubles_into_the_demo_canvas(self):
        got = self.u.upscale_plan(512, 768)
        self.assertEqual(got["size_out"], (1024, 1536))
        self.assertEqual(got["pixels_out"], 4 * got["pixels_in"])

    def test_the_face_bar_comes_from_the_gate_and_not_from_a_copy(self):
        from ball_reel.identity_arcface import MIN_FACE_PX

        self.assertEqual(self.u.upscale_plan(512, 768)["bar"], MIN_FACE_PX)

    def test_the_typical_share_comes_from_animate_and_not_from_a_copy(self):
        from ball_reel.animate import FULL_BODY_FACE_SHARE

        got = self.u.upscale_plan(512, 1000)
        self.assertAlmostEqual(got["face_px_in"],
                               round(FULL_BODY_FACE_SHARE * 1000, 1), places=1)

    def test_an_unmeasured_face_gives_none_and_not_false(self):
        # «Не мерили» и «не проходит» — разные утверждения. Типовая доля уже
        # обещала 146 px там, где по факту было 59.
        got = self.u.upscale_plan(512, 768)
        self.assertIsNone(got["judgeable"])
        self.assertFalse(got["measured"])
        self.assertIn("НЕ ИЗМЕРЕНО", got["note"])

    def test_a_measured_face_gets_a_yes_or_a_no(self):
        # Литералы, а не константы: 78 px x2 = 156 при баре 100 — проходит,
        # 40 px x2 = 80 — нет. Возьми числа из констант, и тест перестанет
        # падать при их сдвиге.
        self.assertIs(self.u.upscale_plan(512, 768, face_px=78)["judgeable"], True)
        self.assertIs(self.u.upscale_plan(512, 768, face_px=40)["judgeable"], False)

    def test_the_face_forecast_agrees_with_refine_because_neither_copies(self):
        # ЕДИНСТВЕННАЯ защита от того, чтобы два прогноза разошлись молча.
        # `refine.refine_plan` владеет вопросом «стоит ли доводка», этот
        # модуль — вопросом «во что превращается кадр», но арифметика лица у
        # них общая, и разойтись ей нельзя.
        from ball_reel.refine import refine_plan

        for face in (38, 57, 72, 78, 150, 400):
            with self.subTest(face=face):
                mine = self.u.upscale_plan(512, 768, face_px=face)
                theirs = refine_plan(face, 768, upscale=mine["factor"])
                self.assertEqual(mine["face_px_out"], theirs["face_px_out"])
                self.assertEqual(mine["bar"], theirs["bar"])

    def test_the_canonical_factor_is_the_one_the_checkpoint_was_trained_on(self):
        # Множитель взят из config.json чекпойнта (`upscale: 2`), а не из
        # имени репозитория: имя может соврать, конфиг — нет.
        self.assertEqual(self.u.UPSCALE_FACTOR, 2)

    def test_nonsense_sizes_are_refused_rather_than_forecast(self):
        for w, h in ((0, 768), (512, 0), (-1, 768)):
            with self.subTest(size=(w, h)):
                with self.assertRaises(ValueError):
                    self.u.upscale_plan(w, h)
        with self.assertRaises(ValueError):
            self.u.upscale_plan(512, 768, factor=0)

    def test_the_plan_says_out_loud_that_memory_is_computed_not_measured(self):
        got = self.u.upscale_plan(512, 768)
        self.assertIn("РАСЧЁТ, не замер", got["note"])
        self.assertIs(got["vram"]["measured"], False)
        self.assertIn("РАСЧЁТ", got["vram"]["note"])


class TheMemoryIsBoundedByTheTileNotByTheFrame(unittest.TestCase):
    """Главный вывод расчёта памяти: площадь кадра уходит во время, не в память."""

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale

    def test_a_frame_four_times_larger_costs_the_same_peak(self):
        small = self.u.vram_delta(512, 768)
        big = self.u.vram_delta(1024, 1536)
        self.assertEqual(small["peak_gb_lower_bound"], big["peak_gb_lower_bound"])
        self.assertGreater(big["tiles"], small["tiles"])

    def test_a_bigger_tile_costs_more_peak(self):
        self.assertGreater(self.u.vram_delta(512, 768, tile=512)["peak_gb_lower_bound"],
                           self.u.vram_delta(512, 768, tile=128)["peak_gb_lower_bound"])

    def test_a_frame_smaller_than_the_tile_costs_less_than_a_full_tile(self):
        self.assertLess(self.u.vram_delta(64, 64, tile=256)["peak_gb_lower_bound"],
                        self.u.vram_delta(256, 256, tile=256)["peak_gb_lower_bound"])

    def test_the_activation_estimate_counts_the_padding_the_processor_adds(self):
        # Арифметика записана литералами целиком: тайл 256 доходит до модели
        # как 264 (процессор добавляет 8 ВСЕГДА), на каждый пиксель — вектор
        # из 180 чисел по 4 байта. Возьми числа из констант — и тест
        # перестанет их сторожить, а он тут единственный сторож.
        got = self.u.vram_delta(512, 768, tile=256)
        self.assertAlmostEqual(got["feature_map_gb"],
                               round((256 + 8) * (256 + 8) * 180 * 4 / 2 ** 30, 4),
                               places=4)

    def test_the_attention_estimate_counts_windows_and_heads(self):
        # Окон в дополненном тайле (264*264)//64, на каждое 6 голов по
        # матрице 64x64 из четырёхбайтовых чисел.
        got = self.u.vram_delta(512, 768, tile=256)
        want = ((264 * 264) // 64) * 6 * 64 * 64 * 4 / 2 ** 30
        self.assertAlmostEqual(got["attention_gb"], round(want, 4), places=4)

    def test_the_weights_number_is_the_only_measured_one(self):
        got = self.u.vram_delta(512, 768)
        self.assertAlmostEqual(got["weights_gb"],
                               round(48_460_660 / 2 ** 30, 4), places=4)
        self.assertIs(got["measured"], False)

    def test_the_peak_is_called_a_lower_bound_and_not_an_answer(self):
        self.assertIn("НИЖНЯЯ ГРАНИЦА", self.u.vram_delta(512, 768)["note"])

    def test_nonsense_sizes_are_refused(self):
        for args in ((0, 768), (512, -3)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    self.u.vram_delta(*args)
        with self.assertRaises(ValueError):
            self.u.vram_delta(512, 768, tile=0)


# ------------------------------------------------------------------ тайлы


class TheTilingCoversTheFrameAndLeavesNoStubs(unittest.TestCase):
    """Сетка тайлов: покрытие, перекрытие и отсутствие огрызков у края."""

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale

    def test_every_pixel_of_the_frame_lands_in_some_tile(self):
        # Диапазон намеренно широкий: кадр мельче тайла, кратный тайлу и
        # некратный — это три разных места, где сетка ломается по-разному.
        for w, h in ((8, 8), (100, 100), (256, 256), (512, 768), (1024, 1536)):
            with self.subTest(size=(w, h)):
                seen = np.zeros((h, w), dtype=bool)
                for x0, y0, x1, y1 in self.u.tile_grid(w, h, tile=256, overlap=16):
                    seen[y0:y1, x0:x1] = True
                self.assertTrue(seen.all())

    def test_neighbours_actually_overlap_by_the_asked_band(self):
        boxes = self.u.tile_grid(600, 100, tile=256, overlap=16)
        xs = sorted({(b[0], b[2]) for b in boxes})
        self.assertGreater(len(xs), 1)
        for (a0, a1), (b0, _) in zip(xs, xs[1:]):
            self.assertGreaterEqual(a1 - b0, 16)

    def test_the_last_tile_hugs_the_edge_instead_of_being_a_stub(self):
        # Огрызок в несколько пикселей — это тайл без контекста, и шов на нём
        # виден. Последний тайл обязан быть полной ширины (кроме кадров,
        # которые сами мельче тайла).
        for total in (257, 300, 511, 700):
            with self.subTest(total=total):
                boxes = self.u.tile_grid(total, 64, tile=256, overlap=16)
                widths = {b[2] - b[0] for b in boxes}
                self.assertEqual(widths, {256})
                self.assertEqual(max(b[2] for b in boxes), total)

    def test_a_frame_smaller_than_the_tile_is_one_tile(self):
        self.assertEqual(self.u.tile_grid(100, 100, tile=256, overlap=16),
                         [(0, 0, 100, 100)])

    def test_zero_overlap_is_allowed_but_overlap_of_a_whole_tile_is_not(self):
        self.assertTrue(self.u.tile_grid(600, 64, tile=256, overlap=0))
        for bad in (256, 300, -1):
            with self.subTest(overlap=bad):
                with self.assertRaises(ValueError):
                    self.u.tile_grid(600, 64, tile=256, overlap=bad)

    def test_the_overlap_is_two_attention_windows_wide(self):
        # ВЫБРАНО, опорный факт — окно внимания 8 из config.json: пиксель шва
        # не должен оказаться на границе окна ни в одном из двух тайлов.
        self.assertEqual(self.u.SR_TILE_OVERLAP, 2 * self.u.SR_WINDOW)
        self.assertEqual(self.u.SR_TILE % self.u.SR_WINDOW, 0)


class TheTilesAreStitchedWithoutASeam(unittest.TestCase):
    """Шов проверяется ПОБИТОВО, а не «на глаз».

    Подставляется точный увеличитель — повтор пикселя. У него соседние тайлы
    обязаны совпадать в перекрытии бит в бит, значит и собранный из тайлов
    кадр обязан совпасть с несобранным. Шов, который «почти не виден», такой
    проверкой не проскочит.
    """

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale

    def test_tiled_nearest_equals_untiled_nearest_exactly(self):
        for w, h, tile in ((300, 200, 128), (512, 768, 256), (64, 64, 256)):
            with self.subTest(size=(w, h), tile=tile):
                src = _img(w, h, "noise", seed=w + h)
                got, rep = self.u._tiled(
                    src, factor=2, tile=tile, overlap=16,
                    fn=lambda p: self.u._nearest(p, 2))
                want = self.u._nearest(src, 2)
                self.assertEqual(got.size, want.size)
                self.assertTrue(np.array_equal(np.asarray(got),
                                               np.asarray(want)),
                                f"шов на {rep}")

    def test_the_stitched_frame_has_the_promised_output_size(self):
        got, rep = self.u._tiled(_img(300, 200), factor=2, tile=128, overlap=16,
                                 fn=lambda p: self.u._nearest(p, 2))
        self.assertEqual(got.size, (600, 400))
        self.assertEqual(rep["size_out"], (600, 400))
        self.assertGreater(rep["tiles"], 1)


# --------------------------------------------------------------- резкость


class TheSharpnessMetricSeesTheWholeRange(unittest.TestCase):
    """Спектральная метрика: на ней держится всё про «выдумывает / нет».

    Синтетика подобрана по СПЕКТРУ: заливка (энергии наверху нет вовсе),
    градиент (вся энергия внизу), шум (размазана), шахматка (сидит на самом
    Найквисте). Метрика, слепая к любому из этих случаев, не годится в
    свидетели, а именно свидетелем она здесь и работает.
    """

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale

    def _g(self, kind, w=128, h=128, seed=0):
        return self.u._gray(_img(w, h, kind, seed))

    def test_a_flat_fill_has_no_high_frequency_energy(self):
        self.assertLess(self.u.high_frequency_share(self._g("flat")), 1e-9)

    def test_a_checkerboard_sits_almost_entirely_above_the_band(self):
        self.assertGreater(self.u.high_frequency_share(self._g("checker")), 0.5)

    def test_white_noise_matches_the_area_of_the_band_it_is_spread_over(self):
        # ЕДИНСТВЕННЫЙ тест здесь с ответом, известным ЗАРАНЕЕ и не из кода:
        # у белого шума энергия равномерна по спектру, значит доля выше порога
        # равна доле ПЛОЩАДИ — для 0.5 это 1 - pi/8 = 0.6073. Именно он поймал
        # первую редакцию метрики: она показывала 0.078, потому что в
        # знаменателе сидела постоянная составляющая, и мерила яркость кадра,
        # а не спектр.
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                got = self.u.high_frequency_share(self._g("noise", seed=seed))
                self.assertAlmostEqual(got, 0.6073, delta=0.05)

    def test_the_range_is_ordered_flat_ramp_noise_checker(self):
        got = [self.u.high_frequency_share(self._g(k))
               for k in ("flat", "ramp", "noise", "checker")]
        self.assertEqual(got, sorted(got), got)
        self.assertLess(got[1], got[2])

    def test_interpolation_leaves_the_invention_band_empty(self):
        # То самое утверждение, ради которого метрика написана: Lanczos не
        # кладёт энергии выше Найквиста входа. Шум — самый жёсткий вход:
        # у него энергии наверху максимум, и терять её есть откуда.
        src = _img(128, 128, "noise", seed=7)
        before = self.u.high_frequency_share(self.u._gray(src))
        after = self.u.high_frequency_share(
            self.u._gray(self.u.lanczos(src, factor=2)))
        self.assertGreater(before, 0.3)
        self.assertLess(after, 0.01)

    def test_nearest_neighbour_is_not_interpolation_and_shows_it(self):
        # Контрольный опыт: повтор пикселя ДАЁТ энергию выше Найквиста входа
        # (ступеньки), и метрика обязана их видеть. Иначе она мерила бы не
        # спектр, а размер кадра.
        src = _img(128, 128, "noise", seed=8)
        got = self.u.high_frequency_share(self.u._gray(self.u._nearest(src, 2)))
        self.assertGreater(got, 0.05)

    def test_the_metric_is_comparable_across_sizes(self):
        # Доли считаются на своей сетке, поэтому одна и та же картинка в
        # разном размере даёт близкие числа — иначе сравнивать вход 512x768
        # с выходом 1024x1536 было бы нельзя.
        a = self.u.high_frequency_share(self._g("checker", 64, 64))
        b = self.u.high_frequency_share(self._g("checker", 256, 256))
        self.assertAlmostEqual(a, b, delta=0.05)

    def test_a_frame_too_small_for_a_spectrum_is_refused(self):
        with self.assertRaises(ValueError):
            self.u.high_frequency_share(np.zeros((2, 2)))

    def test_a_colour_frame_is_refused_because_the_metric_is_on_luma(self):
        with self.assertRaises(ValueError):
            self.u.high_frequency_share(np.zeros((16, 16, 3)))


class TheMeasuredNumbersAreCarriedInTheModuleNotInProse(unittest.TestCase):
    """Числа, на которых стоят утверждения, обязаны быть пересчитываемы."""

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale

    def test_the_three_rows_carry_the_same_three_measures(self):
        for row in ("source", "lanczos", "swin2sr"):
            with self.subTest(row=row):
                got = self.u.MEASURED_HF_SHARE[row]
                self.assertEqual(set(got), {"above_half", "above_quarter",
                                            "lapvar"})

    def test_the_measured_rows_say_what_the_module_claims(self):
        m = self.u.MEASURED_HF_SHARE
        # Lanczos не добавляет: в верхней октаве у него меньше, чем у Swin2SR.
        self.assertLess(m["lanczos"]["above_quarter"],
                        m["swin2sr"]["above_quarter"])
        self.assertLess(m["lanczos"]["lapvar"], m["swin2sr"]["lapvar"])
        # Выше Найквиста исходника пусто у обоих — это и есть находка замера,
        # опровергшая ожидание «апскейлер сочиняет из ничего».
        for who in ("lanczos", "swin2sr"):
            self.assertLess(m[who]["above_half"], 1e-4, who)

    def test_the_ratios_match_the_rows_they_are_derived_from(self):
        m = self.u.MEASURED_HF_SHARE
        self.assertAlmostEqual(
            self.u.MEASURED_INVENTION_RATIO,
            m["swin2sr"]["above_quarter"] / m["lanczos"]["above_quarter"],
            delta=0.05)
        self.assertAlmostEqual(
            self.u.MEASURED_LAPVAR_RATIO,
            m["swin2sr"]["lapvar"] / m["lanczos"]["lapvar"], delta=0.05)

    def test_the_sharpness_row_returns_the_same_shape_as_the_stored_rows(self):
        # Форма ответа команды из докстринга обязана совпадать с формой
        # хранимых замеров — иначе «пересчитать» превращается в «сверить на глаз».
        got = self.u._sharpness(_img(64, 64, "noise", seed=9))
        self.assertEqual(set(got), set(self.u.MEASURED_HF_SHARE["source"]))


class TheFallbackDoesNotInvent(unittest.TestCase):
    """Запасной путь обязан быть именно интерполяцией, а не «чем-нибудь»."""

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale

    def test_lanczos_doubles_the_side(self):
        self.assertEqual(self.u.lanczos(_img(16, 24)).size, (32, 48))

    def test_the_fallback_keeps_the_frame_count(self):
        self.u._PRODUCED.clear()
        frames = [_img(16, 24, "ramp"), _img(16, 24, "noise", seed=11)]
        out, _ = self.u.upscale_frames(frames, _seal(self.u, frames),
                                       method="lanczos")
        self.assertEqual(len(out), 2)

    def test_the_fallback_output_is_not_the_input_object(self):
        self.u._PRODUCED.clear()
        frames = [_img(16, 24)]
        out, _ = self.u.upscale_frames(frames, _seal(self.u, frames),
                                       method="lanczos")
        self.assertIsNot(out[0], frames[0])


# ------------------------------------------------------------------- веса


class WhatOnlyTheWeightsCanAnswer(unittest.TestCase):
    """Всё, что требует Swin2SR. Пропускается с причиной, а не молча."""

    def setUp(self):
        from ball_reel import upscale

        self.u = upscale
        reason = upscale.why_unavailable()
        if reason:
            self.skipTest(f"нет весов Swin2SR: {reason}")

    def test_the_checkpoint_config_says_the_factor_we_claim(self):
        # Множитель берётся из конфига ВЕСОВ, а не из имени репозитория: на
        # этом проекте имя весов уже оказывалось выдуманным.
        from transformers import Swin2SRForImageSuperResolution

        model = Swin2SRForImageSuperResolution.from_pretrained(self.u.SR_REPO)
        self.assertEqual(model.config.upscale, self.u.UPSCALE_FACTOR)
        self.assertEqual(model.config.embed_dim, self.u.SR_EMBED_DIM)
        self.assertEqual(model.config.window_size, self.u.SR_WINDOW)
        self.assertEqual(model.config.num_layers, self.u.SR_DEPTH)

    def test_the_processor_pads_even_an_already_divisible_side(self):
        # ЛОВУШКА API, из-за которой выход был бы 272x272 вместо 256x256.
        # Проверяется на настоящем процессоре, а не по памяти о докам.
        from transformers import Swin2SRImageProcessor

        proc = Swin2SRImageProcessor.from_pretrained(self.u.SR_REPO)
        got = proc(_img(128, 128), return_tensors="pt")["pixel_values"]
        self.assertEqual(tuple(got.shape[-2:]),
                         (128 + self.u.SR_PAD_DIVISOR, 128 + self.u.SR_PAD_DIVISOR))

    def test_the_upscaler_returns_exactly_the_promised_size(self):
        # То есть паддинг обрезан. Без обрезки по краю осталась бы полоса
        # зеркального отражения шириной SR_PAD_DIVISOR * factor.
        self.u._PRODUCED.clear()
        frames = [_img(48, 64, "noise", seed=12)]
        out, rep = self.u.upscale_frames(frames, _seal(self.u, frames),
                                         method="weights", tile=64)
        self.assertEqual(out[0].size, (96, 128))
        self.assertEqual(rep["outcome"], "weights")
        self.assertIs(rep["invents"], True)

    def test_asking_the_checkpoint_for_a_factor_it_was_not_trained_on_fails(self):
        with self.assertRaises(ValueError) as ctx:
            self.u._swin2sr(_img(32, 32), factor=3, tile=64)
        self.assertIn("обучен", str(ctx.exception))

    def test_the_weights_output_is_sharper_than_the_fallback(self):
        # Утверждение «весами резче» проверяется числом на настоящем кадре, а
        # не словами. Кадр берётся из кита — синтетика здесь не годится:
        # у шума нет структуры, которую апскейлеру есть смысл восстанавливать.
        from pathlib import Path

        frame = Path(__file__).resolve().parents[2] / "kit/driving/0000.jpg"
        if not frame.exists():
            self.skipTest(f"нет кадра {frame}")
        from PIL import Image

        with Image.open(frame) as raw:
            src = raw.convert("RGB").crop((300, 500, 428, 628))
        got, _ = self.u._swin2sr(src, factor=2, tile=128)
        lanc = self.u.lanczos(src, factor=2)
        self.assertGreater(self.u._sharpness(got)["lapvar"],
                           self.u._sharpness(lanc)["lapvar"])


if __name__ == "__main__":                          # pragma: no cover
    unittest.main()
