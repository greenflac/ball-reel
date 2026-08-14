"""Решения предполёта — без карты, без сети и без дорогих импортов.

Предполёт целиком состоит из проверок железа, и это его беда как кода: он
почти весь неисполним там, где мы разрабатываем. Мутационный аудит показал,
чем это кончается — снятый MIN_VRAM_GB не ронял ни одного теста, то есть
порог, ради которого предполёт и написан, не сторожил никто.

Поэтому каждое РЕШЕНИЕ отделено от чтения железа: `vram_verdict` берёт число,
`driver_verdict` — две строки версий, `size_verdict` — два размера,
`packages_verdict` — словарь. Судится здесь именно решение.

Три вещи проверяются в каждом отказе, и это не стиль, а требование:

* отказ называет ЛЕЧЕНИЕ, а не только проблему;
* «не смогли проверить» отличимо и от «хорошо», и от «плохо»;
* вход теста — литерал, а не та константа, которую тест сторожит.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    from PIL import Image  # noqa: F401
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

#: Слова, по которым видно, что в строке есть что делать, а не только что
#: сломалось. Одного достаточно: лечения бывают командой, флагом и заменой
#: файла.
CURE_WORDS = ("pip", "curl", "скачать", "лечение", "переставить", "обновить",
              "перерендерить", "export", "python3", "взять", "снять",
              "поставить", "проверить", "запускать", "смотреть", "уменьшить",
              "освободить", "нужна сборка")


def _has_cure(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in CURE_WORDS)


def _install_module(case: unittest.TestCase, name: str, module) -> None:
    """Подставить модуль на время теста. `None` заставляет import бросить."""
    had = name in sys.modules
    saved = sys.modules.get(name)
    sys.modules[name] = module

    def restore():
        if had:
            sys.modules[name] = saved
        else:
            sys.modules.pop(name, None)

    case.addCleanup(restore)


class TheCardIsJudgedBeforeAnythingIsRented(unittest.TestCase):
    """Порог VRAM зажат с двух сторон литералами, а не выведен из константы."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu

    def test_the_target_laptop_card_passes(self):
        # 4 ГБ — это RTX 3050 из ноутбука и A16 у хостера, то есть ровно та
        # карта, под которую собран gpu_keyframes.plan(). Если предполёт
        # разворачивает её, вся экономия на 4 ГБ бессмысленна.
        ok, detail = self.p.vram_verdict(4.0)
        self.assertTrue(ok, detail)

    def test_a_card_too_small_is_refused_and_told_what_to_do(self):
        # 2 ГБ не поедет ни с какими экономиями. Отказ обязан назвать выход,
        # иначе арендатор просто перезапустит то же самое.
        ok, detail = self.p.vram_verdict(2.0)
        self.assertFalse(ok)
        self.assertIn("448x640", detail)

    def test_the_floor_is_read_from_the_module_not_from_the_test(self):
        # Порог берётся из модуля, входы — литеральные: так тест не поедет
        # вместе с константой, но покраснеет, если её сдвинуть или снять.
        floor = self.p.MIN_VRAM_GB
        self.assertFalse(self.p.vram_verdict(floor - 0.1)[0])
        self.assertGreater(floor, 2.0)
        self.assertLess(floor, 4.0)

    def test_below_the_floor_and_weights_too_big_are_different_refusals(self):
        # Обе беды — «мало памяти», но лечатся разным: под порогом карта не
        # годится ни для чего, а между порогом и весами локального пути ещё
        # едет путь кейфреймов. Одинаковый текст отправил бы менять карту там,
        # где достаточно сменить путь.
        tiny = self.p.vram_verdict(2.0)[1]
        weights_dont_fit = self.p.vram_verdict(3.6)[1]
        self.assertNotEqual(tiny, weights_dont_fit)
        self.assertIn("кейфрейм", weights_dont_fit)

    def test_weights_and_activations_are_named_apart(self):
        # Главное требование к отказу по памяти: он обязан сказать, ЧТО именно
        # не влезло. «Не влезли веса» лечится выгрузкой и снятием ControlNet,
        # «не влезли активации» — тайлингом и числом кадров. Совет не тот —
        # это потерянный час на карте.
        ok_small, small = self.p.vram_verdict(3.6)
        self.assertFalse(ok_small)
        self.assertIn("ВЕСА", small)
        self.assertIn("Разрешение здесь НИ ПРИ ЧЁМ", small)
        ok_tight, tight = self.p.vram_verdict(4.5)
        self.assertTrue(ok_tight)
        self.assertIn("АКТИВАЦИИ", tight)
        self.assertTrue(_has_cure(small) and _has_cure(tight))

    def test_the_weight_size_comes_from_animate_not_from_a_second_copy(self):
        # Второй список тех же гигабайтов разошёлся бы с первым молча.
        from ball_reel.animate import WEIGHTS_GB

        weights, left = self.p.weights_headroom(6.0)
        self.assertEqual(weights, round(sum(WEIGHTS_GB.values()), 2))
        self.assertEqual(left, round(6.0 - weights, 2))

    def test_a_roomy_card_is_not_warned_about_anything(self):
        ok, detail = self.p.vram_verdict(12.0)
        self.assertTrue(ok)
        self.assertNotIn("впритык", detail)


class TheTorchBuildIsJudgedByWhatItWasBuiltWith(unittest.TestCase):
    """CPU-сборка, мёртвый драйвер и не-NVIDIA — три беды с разным лечением."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu

    def test_a_working_build_reports_the_card(self):
        ok, detail = self.p.torch_verdict("2.13.0+cu126", True, "RTX 3050",
                                          built_cuda="12.6")
        self.assertTrue(ok)
        self.assertIn("RTX 3050", detail)

    def test_a_wheel_without_a_suffix_is_still_caught_as_a_cpu_build(self):
        # ЭТОТ СЛУЧАЙ И ЕСТЬ ПРИЧИНА ПЕРЕПИСАТЬ ПРОВЕРКУ. Колесо с PyPI не
        # несёт суффикса `+cpu` вовсе, и старая проверка по строке молча
        # пропускала ровно то, ради чего написана: сборку без CUDA. Здесь
        # версия нарочно чистая, а признак — torch.version.cuda = None.
        ok, detail = self.p.torch_verdict("2.13.0", False, "", built_cuda=None)
        self.assertFalse(ok)
        self.assertIn("БЕЗ CUDA", detail)
        self.assertIn("uninstall", detail)
        self.assertNotIn("драйвер NVIDIA до", detail)

    def test_the_suffix_is_only_a_fallback_and_says_so(self):
        # Когда torch.version.cuda спросить не удалось, судить по суффиксу
        # можно — но тогда об этом надо сказать, иначе догадка выглядит
        # фактом.
        ok, detail = self.p.torch_verdict("2.13.0+cpu", False, "", built_cuda="")
        self.assertFalse(ok)
        self.assertIn("суффикс", detail)

    def test_a_cuda_build_without_a_card_points_at_the_driver_not_at_pip(self):
        # Живой случай этой среды: сборка с CUDA 13.0, карты нет. Совет
        # «переставить torch» здесь уводит в сторону на полчаса.
        ok, detail = self.p.torch_verdict("2.13.0+cu130", False, "",
                                          built_cuda="13.0")
        self.assertFalse(ok)
        self.assertIn("nvidia-smi", detail)
        self.assertIn("НЕ про переустановку", detail)
        self.assertNotIn("uninstall", detail)

    def test_an_intel_card_is_not_sent_to_nvidia_smi(self):
        # Совет «проверь nvidia-smi» владельцу Arc вреден: он уводит от
        # настоящей причины (сборка без XPU или драйверы Level Zero).
        _, detail = self.p.torch_verdict("2.13.0", False, "", built_cuda="")
        self.assertIn("Arc", detail)
        self.assertIn("XPU", detail)

    def test_an_accelerator_of_any_kind_passes(self):
        for kind, name in (("cuda", "RTX 3050"), ("xpu", "Arc A580")):
            ok, detail = self.p.torch_verdict("2.13.0", True, name,
                                              device_kind=kind)
            self.assertTrue(ok, detail)
            self.assertIn(name, detail)

    def test_every_refusal_says_what_to_do(self):
        for args, kw in (((("2.13.0"), False, ""), {"built_cuda": None}),
                         ((("2.13.0+cu130"), False, ""), {"built_cuda": "13.0"}),
                         ((("2.13.0"), False, ""), {"built_cuda": ""})):
            ok, detail = self.p.torch_verdict(*args, **kw)
            self.assertFalse(ok)
            self.assertTrue(_has_cure(detail), detail)

    def test_the_check_itself_calls_the_verdict_correctly(self):
        # РЕГРЕССИЯ НА ЖИВОЙ ДЕФЕКТ: check_torch звал torch_verdict, передавая
        # `device` и позиционно, и по имени, — то есть на карте эта строка
        # печатала бы `TypeError`, а не диагноз. Отлавливается только вызовом
        # целиком: сигнатуру и вызов иначе ничто не связывает.
        ok, name, detail = self.p.check_torch()
        self.assertIn(ok, (True, False))
        self.assertNotIn("TypeError", detail)
        self.assertTrue(name.startswith("torch"))


class TheDriverIsComparedToTheBuild(unittest.TestCase):
    """`nvidia-smi` печатает CUDA драйвера; ниже сборки — всё падает непонятно."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu

    def test_an_older_driver_major_is_a_refusal_with_two_ways_out(self):
        ok, detail = self.p.driver_verdict("12.4", "13.0")
        self.assertFalse(ok)
        self.assertIn("12.4", detail)
        self.assertIn("13.0", detail)
        self.assertIn("обновить драйвер", detail)
        self.assertIn("index-url", detail)

    def test_a_newer_driver_is_fine(self):
        ok, _ = self.p.driver_verdict("13.0", "12.6")
        self.assertTrue(ok)

    def test_an_older_minor_is_not_verified_rather_than_approved(self):
        # Минорная совместимость CUDA 11+ обычно покрывает это, но проверить
        # без карты нечем. Выдать «в порядке» — это ровно тот дефект, ради
        # которого третий исход и заведён.
        ok, detail = self.p.driver_verdict("12.4", "12.6")
        self.assertIsNone(ok)
        self.assertIn("ПРОВЕРИТЬ ЭТО ЗДЕСЬ НЕЧЕМ", detail)

    def test_a_missing_smi_is_not_reported_as_a_match(self):
        ok, detail = self.p.driver_verdict(None, "13.0")
        self.assertIsNone(ok)
        self.assertIn("НЕ «совпадает»", detail)

    def test_a_cpu_build_is_sent_to_the_torch_line_not_to_the_driver(self):
        ok, detail = self.p.driver_verdict("12.4", None)
        self.assertIsNone(ok)
        self.assertIn("переустановкой torch", detail)

    def test_the_four_outcomes_read_differently(self):
        lines = {self.p.driver_verdict(*a)[1] for a in
                 (("12.4", "13.0"), ("13.0", "12.6"), ("12.4", "12.6"),
                  (None, "13.0"))}
        self.assertEqual(len(lines), 4, lines)


class TheCardIsSeenByTheDriverFirst(unittest.TestCase):
    """nvidia-smi отвечает за 200 мс и решает то же, что torch за 2.4 с."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu

    def test_a_visible_card_is_named_with_its_memory(self):
        probe = {"cards": [{"name": "NVIDIA GeForce RTX 3050 Laptop GPU",
                            "vram_gb": 6.0, "driver": "550.54.14"}],
                 "cuda": "12.4", "reason": ""}
        ok, _, detail = self.p.check_smi(probe)
        self.assertTrue(ok, detail)
        self.assertIn("RTX 3050", detail)
        self.assertIn("12.4", detail)

    def test_a_card_too_small_is_refused_before_torch_is_even_imported(self):
        probe = {"cards": [{"name": "Quadro K620", "vram_gb": 2.0,
                            "driver": "470.0"}], "cuda": "11.4", "reason": ""}
        ok, _, detail = self.p.check_smi(probe)
        self.assertFalse(ok)
        self.assertIn("Quadro K620", detail)
        self.assertTrue(_has_cure(detail))

    def test_no_smi_is_unverified_not_broken(self):
        # На Intel Arc и на Apple nvidia-smi нет и быть не должно; объявлять
        # это отказом — ложная тревога, а она в предполёте стоит доверия ко
        # всем остальным строкам.
        ok, _, detail = self.p.check_smi(
            {"cards": [], "cuda": None, "reason": "nvidia-smi не найден в PATH"})
        self.assertIsNone(ok)
        self.assertIn("Intel Arc", detail)


class PackagesAreCheckedBeforeAnythingIsImported(unittest.TestCase):
    """peft — не опциональный пакет, и это самая дорогая ошибка окружения."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu

    def _all_present(self, **overrides):
        got = {mod: True for mod, _, _, _ in self.p.PACKAGES}
        got.update(overrides)
        return got

    def test_a_missing_peft_is_a_hard_refusal_with_the_reason_spelled_out(self):
        # Без peft diffusers НЕ ПАДАЕТ: пишет предупреждение и едет дальше, а
        # личность остаётся необусловленной. Отказ обязан объяснить именно
        # это, иначе peft выглядит необязательным — он ведь «просто варнинг».
        ok, detail = self.p.packages_verdict(self._all_present(peft=False))
        self.assertFalse(ok)
        self.assertIn("peft", detail)
        self.assertIn("НЕ ПАДАЕТ", detail)
        self.assertIn("pip install peft", detail)

    def test_peft_is_declared_mandatory_in_the_table(self):
        hard = {mod for mod, _, _, is_hard in self.p.PACKAGES if is_hard}
        self.assertIn("peft", hard)
        self.assertIn("insightface", hard)
        self.assertIn("mediapipe", hard)

    def test_an_optional_package_does_not_stop_the_run(self):
        ok, detail = self.p.packages_verdict(self._all_present(cv2=False))
        self.assertTrue(ok)
        self.assertIn("cv2", detail)

    def test_a_package_nobody_asked_about_is_unverified_not_present(self):
        # Словарь без ключа — это «не спрашивали», и выдавать его за «есть»
        # значит зеленеть на машине, где проверка сломалась.
        partial = self._all_present()
        partial.pop("peft")
        ok, detail = self.p.packages_verdict(partial)
        self.assertIsNone(ok)
        self.assertIn("НЕ ОПРОШЕНЫ", detail)

    def test_probing_costs_no_import(self):
        # find_spec против import — три порядка цены. Если кто-то заменит его
        # импортом, `peft` начнёт находиться там, где его подменили заглушкой
        # sys.modules, и проверка станет дороже того, что сторожит.
        import time

        t = time.time()
        got = self.p.find_specs()
        self.assertLess(time.time() - t, 0.5)
        self.assertEqual(set(got), {m for m, _, _, _ in self.p.PACKAGES})

    def test_the_generator_module_refuses_without_peft_too(self):
        # Сквозная проверка через границу модулей: `animate.preflight` — второе
        # место, где peft обязан быть жёстким отказом, и разойтись эти два
        # места могут молча. Здесь peft отбирается принудительно, поэтому тест
        # честен и на машине, где peft установлен.
        from ball_reel import animate

        _install_module(self, "peft", None)
        rep = animate.preflight(6.0)
        row = [c for c in rep["checks"] if c["name"] == "peft"]
        self.assertEqual(len(row), 1, rep["checks"])
        self.assertFalse(row[0]["ok"])
        self.assertFalse(rep["ok"])


class DiskIsMeasuredWhereTheWeightsLand(unittest.TestCase):
    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu

    def test_not_enough_space_names_the_way_out(self):
        ok, detail = self.p.disk_verdict(7.0, 15.0, "веса и выдача")
        self.assertFalse(ok)
        self.assertIn("HF_HOME", detail)

    def test_enough_space_says_how_much_was_needed_and_why(self):
        ok, detail = self.p.disk_verdict(40.0, 9.8, "недостающие веса")
        self.assertTrue(ok)
        self.assertIn("9.8", detail)
        self.assertIn("недостающие веса", detail)

    def test_the_default_floor_is_a_module_constant_but_the_inputs_are_not(self):
        floor = self.p.MIN_DISK_GB
        self.assertGreater(floor, 8.0)
        self.assertFalse(self.p.disk_verdict(1.0, floor, "по умолчанию")[0])

    def test_free_space_is_read_for_a_path_that_does_not_exist_yet(self):
        # HF_HOME часто указывает на ещё не созданный каталог; мерить в этом
        # случае нечего — надо подняться до существующего предка, а не упасть.
        got = self.p._free_gb(Path("/nonexistent-dir-xyz/deeper/still"))
        self.assertGreater(got, 0.0)


class MissingWeightsAreNamedFileByFile(unittest.TestCase):
    """«Нет весов» — плохо. «Нет ЭТОГО файла, качать ЭТОЙ командой» — годно."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu

    def _pretend_nothing_cached(self):
        saved = self.p._cached
        self.p._cached = lambda repo, filename: None
        self.addCleanup(lambda: setattr(self.p, "_cached", saved))

    def test_every_needed_repo_is_named_with_a_download_command(self):
        self._pretend_nothing_cached()
        ok, _, detail = self.p.check_weights()
        self.assertFalse(ok)
        for t in self.p.weight_targets():
            self.assertIn(t["repo"], detail)
        self.assertIn("snapshot_download", detail)
        self.assertIn("allow_patterns", detail)

    def test_the_repo_names_come_from_animate_not_from_a_second_list(self):
        # Имя весов, написанное по памяти рядом, уже было выдумано однажды.
        from ball_reel import animate

        repos = {t["repo"] for t in self.p.weight_targets()}
        self.assertEqual(repos, {animate.BASE_MODEL, animate.MOTION_ADAPTER,
                                 animate.CONTROLNET, animate.IP_ADAPTER_REPO})
        faceid = [t for t in self.p.weight_targets()
                  if t["repo"] == animate.IP_ADAPTER_REPO][0]
        self.assertIn(animate.IP_ADAPTER_WEIGHT, faceid["files"])

    def test_the_motion_module_is_not_hidden_behind_a_cache_size(self):
        # Старая проверка мерила размер кэша HF целиком порогом 3 ГБ и зеленела
        # на машине, где скачана только база: модуля движения нет, а строка
        # «веса» бодро сообщает про гигабайты.
        saved = self.p._cached
        from ball_reel import animate

        self.p._cached = lambda repo, filename: (
            None if repo == animate.MOTION_ADAPTER else "/cache/fake")
        self.addCleanup(lambda: setattr(self.p, "_cached", saved))
        ok, _, detail = self.p.check_weights()
        self.assertFalse(ok)
        self.assertIn(animate.MOTION_ADAPTER, detail)
        self.assertIn("1671", detail)

    def test_a_partially_downloaded_repo_reads_differently_from_an_absent_one(self):
        saved_cached, saved_started = self.p._cached, self.p._repo_started
        self.p._cached = lambda repo, filename: None
        self.p._repo_started = lambda repo: True
        self.addCleanup(lambda: setattr(self.p, "_cached", saved_cached))
        self.addCleanup(lambda: setattr(self.p, "_repo_started", saved_started))
        started = self.p.check_weights()[2]
        self.p._repo_started = lambda repo: False
        fresh = self.p.check_weights()[2]
        self.assertIn("недокачано", started)
        self.assertIn("не скачивалось", fresh)

    def test_an_unreadable_cache_is_unverified_not_green(self):
        saved = self.p.weight_targets
        self.p.weight_targets = lambda: (_ for _ in ()).throw(
            RuntimeError("кэш недоступен"))
        self.addCleanup(lambda: setattr(self.p, "weight_targets", saved))
        ok, _, detail = self.p.check_weights()
        self.assertIsNone(ok)
        self.assertIn("кэш недоступен", detail)


class TheJudgesWeightsAreNamedWithTheirCommands(unittest.TestCase):
    """DWPose, MediaPipe, сегментация, buffalo_l: у каждого файл и команда."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu

    def test_the_pose_model_refusal_keeps_the_curl_line(self):
        # Раньше сообщение резалось по первой строке — а команда curl лежит во
        # второй. Отказ терял ровно ту часть, ради которой он написан.
        import os

        from ball_reel import pose

        saved = os.environ.get(pose.MODEL_ENV)
        os.environ[pose.MODEL_ENV] = "/nonexistent/pose_landmarker_lite.task"
        self.addCleanup(lambda: os.environ.pop(pose.MODEL_ENV)
                        if saved is None else
                        os.environ.__setitem__(pose.MODEL_ENV, saved))
        ok, _, detail = self.p.check_pose_model()
        self.assertFalse(ok)
        self.assertIn("curl", detail)
        self.assertIn("storage.googleapis.com", detail)

    def test_the_face_pack_refusal_names_the_archive_and_where_it_goes(self):
        import os

        saved = os.environ.get("INSIGHTFACE_HOME")
        os.environ["INSIGHTFACE_HOME"] = "/nonexistent-insightface"
        self.addCleanup(lambda: os.environ.pop("INSIGHTFACE_HOME")
                        if saved is None else
                        os.environ.__setitem__("INSIGHTFACE_HOME", saved))
        ok, _, detail = self.p.check_face_model()
        self.assertFalse(ok)
        self.assertIn("buffalo_l.zip", detail)
        self.assertIn("det_10g.onnx", detail)
        self.assertIn("curl", detail)

    def test_dwpose_absence_is_not_a_refusal_when_conditions_are_ready(self):
        from ball_reel import dwpose

        saved = dwpose.why_unavailable
        dwpose.why_unavailable = lambda: "нет весов DWPose: curl ..."
        dwpose.available = lambda: False
        self.addCleanup(lambda: setattr(dwpose, "why_unavailable", saved))
        self.addCleanup(lambda: setattr(dwpose, "available",
                                        lambda: not dwpose.why_unavailable()))
        ok, _, detail = self.p.check_dwpose(needed=False)
        self.assertIsNone(ok)
        self.assertIn("НЕ НУЖНЫ", detail)
        self.assertFalse(self.p.check_dwpose(needed=True)[0])


@unittest.skipUnless(HAVE_PIL, "Pillow not installed (live extra)")
class BrokenConditionsAreCaughtAtHome(unittest.TestCase):
    """Условия рендерятся на CPU дома — значит и проверяются дома."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def _png(self, name: str, size=(512, 768), skeleton=True):
        from PIL import Image, ImageDraw

        im = Image.new("RGB", size, (0, 0, 0))
        if skeleton:
            ImageDraw.Draw(im).line((10, 10, size[0] - 10, size[1] - 10),
                                    fill=(255, 0, 0), width=9)
        im.save(self.root / name)
        return self.root / name

    def test_a_missing_directory_is_not_the_same_as_an_empty_one(self):
        ok, _, detail = self.p.check_conditions(str(self.root / "нет-такого"))
        self.assertFalse(ok)
        self.assertIn("kit", detail)

    def test_an_empty_directory_names_the_step_that_was_skipped(self):
        ok, _, detail = self.p.check_conditions(str(self.root))
        self.assertFalse(ok)
        self.assertIn("render_sequence", detail)

    def test_mixed_sizes_are_refused_because_controlnet_needs_one(self):
        self._png("0000.png", (512, 768))
        self._png("0001.png", (448, 640))
        ok, _, detail = self.p.check_conditions(str(self.root))
        self.assertFalse(ok)
        self.assertIn("размер", detail)

    def test_a_frame_with_no_skeleton_is_refused_not_averaged_over(self):
        # Пустое условие — это кадр, в котором генератор ничем не ограничен.
        # Именно этот случай проходил старую проверку насквозь: чёрный png
        # 512x768 весит 1224 байта при пороге в 500. Поэтому кадр здесь
        # рисуется В ПРОДАКШЕН-РАЗМЕРЕ — на маленьком тест был бы зелёным и
        # на сломанном коде тоже.
        self._png("0000.png")
        self._png("0001.png", skeleton=False)
        ok, _, detail = self.p.check_conditions(str(self.root))
        self.assertFalse(ok)
        self.assertIn("пуст", detail)

    def test_an_unreadable_file_is_named_not_thrown(self):
        # Недокопированное условие: раньше PIL кидал отсюда наружу, и вместо
        # «докопировать» арендатор получал трейсбек.
        self._png("0000.png")
        (self.root / "0001.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 40)
        ok, _, detail = self.p.check_conditions(str(self.root))
        self.assertFalse(ok)
        self.assertIn("не читаются", detail)

    def test_a_gap_in_the_numbering_is_refused(self):
        # render_sequence пропускает кадр, в котором позы не нашли, но нумерует
        # по исходному индексу. Прогон берёт каждый N-й файл и считает шаг по
        # времени постоянным — после дырки это неправда.
        for name in ("0000.png", "0001.png", "0004.png"):
            self._png(name)
        ok, _, detail = self.p.check_conditions(str(self.root))
        self.assertFalse(ok)
        self.assertIn("0001", detail)

    def test_too_few_conditions_for_the_window_is_caught_before_the_weights(self):
        # Модуль движения обучен на окне 16 кадров и откажется при другом
        # числе — но узнать это после загрузки 3.8 ГБ весов значит потерять
        # минуту на карте вместо миллисекунды дома.
        for i in range(3):
            self._png(f"{i:04d}.png")
        ok, _, detail = self.p.check_conditions(str(self.root), need_frames=16)
        self.assertFalse(ok)
        self.assertIn("16", detail)

    def test_a_clean_sequence_passes(self):
        for i in range(3):
            self._png(f"{i:04d}.png")
        ok, _, detail = self.p.check_conditions(str(self.root))
        self.assertTrue(ok, detail)
        self.assertIn("3", detail)


class ConditionSizeIsJudgedAgainstThePlan(unittest.TestCase):
    """diffusers масштабирует условие МОЛЧА — проверено по исходнику 0.39."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu

    def test_the_same_size_is_simply_confirmed(self):
        ok, note = self.p.size_verdict((512, 768), (512, 768))
        self.assertTrue(ok)
        self.assertIn("совпадает", note)

    def test_a_different_aspect_is_refused_because_the_skeleton_stretches(self):
        ok, note = self.p.size_verdict((512, 512), (512, 768))
        self.assertFalse(ok)
        self.assertIn("пропорция", note)
        self.assertTrue(_has_cure(note))

    def test_the_same_aspect_but_another_size_is_unverified_not_approved(self):
        # Масштабирование той же пропорции не ломает позу, но и не бесплатно:
        # мелкие суставы теряют точность. Это ровно «не смогли поручиться».
        ok, note = self.p.size_verdict((256, 384), (512, 768))
        self.assertIsNone(ok)
        self.assertIn("отмасштабирует молча", note)

    def test_no_plan_means_no_opinion(self):
        ok, note = self.p.size_verdict((512, 768), None)
        self.assertTrue(ok)
        self.assertEqual(note, "")

    def test_the_plan_size_is_taken_from_animate(self):
        from ball_reel.animate import plan

        cfg = plan(6.0, waist_up=False)
        self.assertEqual(self.p.plan_size(6.0), ((cfg.width, cfg.height),
                                                 cfg.frames))
        self.assertEqual(self.p.plan_size(None), (None, None))


class ThereIsSomethingToCheckThePoseAgainst(unittest.TestCase):
    """Без driving-кадров гейт не измеряет главное — и молчит об этом."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.cond = self.root / "conditions"
        self.cond.mkdir()

    def _manifest(self, mapping):
        (self.cond / "manifest.json").write_text(
            json.dumps({"driving_frames": mapping}))

    def _driving(self, names):
        d = self.root / "driving"
        d.mkdir(exist_ok=True)
        for n in names:
            (d / n).write_bytes(b"jpeg")

    def test_a_missing_manifest_names_what_writes_it(self):
        ok, _, detail = self.p.check_driving(str(self.cond))
        self.assertFalse(ok)
        self.assertIn("render_sequence", detail)

    def test_an_old_manifest_without_the_map_is_refused(self):
        (self.cond / "manifest.json").write_text(json.dumps({"size": [512, 768]}))
        ok, _, detail = self.p.check_driving(str(self.cond))
        self.assertFalse(ok)
        self.assertIn("driving_frames", detail)

    def test_paths_that_do_not_open_from_here_are_refused_with_the_reason(self):
        # Пути в манифесте записаны относительно каталога, откуда рендерили.
        # Это ловится дома и стоит миллисекунды; на карте это исключение
        # ВНУТРИ измерителя позы, уже после генерации.
        self._manifest({"0000": "driving/0000.jpg"})
        ok, _, detail = self.p.check_driving(str(self.cond))
        self.assertFalse(ok)
        self.assertIn("относительно", detail)
        self.assertTrue(_has_cure(detail))

    def test_a_full_map_passes(self):
        self._driving(["0000.jpg", "0001.jpg"])
        self._manifest({"0000": "driving/0000.jpg", "0001": "driving/0001.jpg"})
        ok, _, detail = self.p.check_driving(str(self.cond))
        self.assertTrue(ok, detail)
        self.assertIn("2/2", detail)

    def test_a_partial_map_is_unverified_not_a_pass(self):
        self._driving(["0000.jpg"])
        self._manifest({"0000": "driving/0000.jpg", "0001": "driving/0001.jpg"})
        ok, _, detail = self.p.check_driving(str(self.cond))
        self.assertIsNone(ok)
        self.assertIn("1/2", detail)


class TheGatewayDoctorAlsoHasThreeOutcomes(unittest.TestCase):
    """«Сеть не дошла» и «сервис отказал» лечатся противоположным."""

    def setUp(self):
        from ball_reel import doctor

        self.d = doctor

    def test_the_three_marks_are_distinct_words(self):
        marks = {self.d.outcome(v, "x")[0] for v in (True, False, None)}
        self.assertEqual(len(marks), 3)

    def test_a_network_failure_is_unverified_not_a_dead_service(self):
        # Закрытый прокси-политикой хост — это «проверить не вышло», и лечение
        # у него не «чинить ключ», а «попросить оператора открыть домен».
        def boom():
            raise __import__("requests").exceptions.ConnectionError("proxy 403")

        ok, detail, _ = self.d._t(boom)
        self.assertIsNone(ok)
        self.assertIn("НЕ ПРОВЕРЕН", detail)
        self.assertIn("оператора", detail)

    def test_a_service_failure_stays_a_failure(self):
        def boom():
            raise ValueError("модель сняли")

        ok, detail, _ = self.d._t(boom)
        self.assertFalse(ok)
        self.assertIn("модель сняли", detail)

    def test_a_check_that_did_not_run_is_not_silently_dropped(self):
        # Раньше зависимая проверка просто не вызывалась и не печаталась: в
        # счёте её не было ни с одной стороны, и «3/3» получалось на прогоне,
        # где половина не проверялась.
        import inspect

        src = inspect.getsource(self.d.main)
        self.assertIn("skip_if", src)
        self.assertIn("marks.append(UNKNOWN)", src)


class ChecksAreOrderedByPriceNotByImportance(unittest.TestCase):
    """Отказ после загрузки полутора гигабайт — дефект порядка, а не невезение."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu
        self.args = types.SimpleNamespace(conditions="kit/conditions",
                                          face="kit/face.jpg", out=".",
                                          vram=6.0, dwpose=False,
                                          skip_gateway=False)

    def _order(self):
        return [label for _, checks in self.p.build_checks(self.args)
                for label, _ in checks]

    def test_the_millisecond_checks_come_before_the_second_long_ones(self):
        order = self._order()
        for cheap in ("пакеты", "диск", "веса", "модель позы"):
            self.assertLess(order.index(cheap), order.index("torch"),
                            f"{cheap} стоит миллисекунды, torch — секунды")

    def test_peft_is_found_before_torch_is_imported(self):
        # Самая дорогая ошибка окружения обязана находиться самой дешёвой
        # проверкой: find_spec отвечает за доли миллисекунды, import torch —
        # за 2.4 секунды.
        order = self._order()
        self.assertLess(order.index("пакеты"), order.index("torch"))

    def test_the_face_and_the_gateway_are_last(self):
        # Лицо — 5.2 с (insightface + mediapipe), шлюз — сеть до 30 с.
        order = self._order()
        self.assertGreater(order.index("лицо"), order.index("torch"))
        self.assertEqual(order[-1], "шлюз")

    def test_the_gateway_is_absent_on_the_local_path(self):
        self.args.skip_gateway = True
        self.assertNotIn("шлюз", self._order())

    def test_every_check_in_the_list_has_a_measured_price(self):
        # Порядок обязан следовать из чисел, а не из ощущения; число, которого
        # нет, — это порядок, выбранный на глаз.
        for label in self._order():
            self.assertIn(label, self.p.COSTS, f"нет замера цены для {label}")

    def test_each_check_answers_with_one_of_exactly_three_outcomes(self):
        # Пропущенный третий исход — тот самый дефект, из-за которого
        # непроверенное показывается галочкой.
        for _, checks in self.p.build_checks(self.args):
            for label, fn in checks:
                if label in ("лицо", "шлюз", "torch", "onnxruntime", "vram"):
                    continue  # дорогие и/или требующие железа
                ok, _, _ = fn()
                self.assertIn(ok, (True, False, None), label)


if __name__ == "__main__":
    unittest.main()
