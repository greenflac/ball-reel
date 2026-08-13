"""Выбор устройства и — главное — молчаливый откат на CPU.

Второе важнее первого. onnxruntime принимает запрос на недоступный провайдер
БЕЗ ошибки и считает на процессоре; мы это наблюдали живьём, когда DWPose
просил CUDA и выдавал 438 мс. Деградация в десять раз, о которой система не
сообщает, — это худший вид дефекта: числа есть, они правдоподобны, и они не о
том железе, о котором думает читатель.

Устройств в среде разработки нет вообще, поэтому здесь судится логика выбора
на подставленных списках, а не факт ускорения.
"""

from __future__ import annotations

import sys
import types
import unittest


def _install_module(case: unittest.TestCase, name: str, module) -> None:
    """Подставить модуль на время теста и вернуть всё как было.

    Импорт в `device` делается ВНУТРИ функций, поэтому подмена в `sys.modules`
    доезжает до кода. `None` в `sys.modules` — это не «нет ключа», а прямой
    способ заставить `import` бросить ImportError: так изображается машина, где
    torch не поставлен вовсе.
    """
    had = name in sys.modules
    saved = sys.modules.get(name)
    sys.modules[name] = module

    def restore():
        if had:
            sys.modules[name] = saved
        else:
            sys.modules.pop(name, None)

    case.addCleanup(restore)


def _fake_torch(version: str, device: str, name=None, boom=None):
    """torch-заглушка: у бэкенда либо есть имя карты, либо он бросает.

    Второе — не выдумка ради теста: на xpu-сборках `get_device_name` падает при
    неподнятом драйвере, и именно этот случай модуль раньше выдавал за
    «torch не установлен».
    """
    torch = types.ModuleType("torch")
    torch.__version__ = version
    backend = types.SimpleNamespace()
    if boom is not None:
        def get_device_name(_idx):
            raise boom
    else:
        def get_device_name(_idx):
            return name

    backend.get_device_name = get_device_name
    setattr(torch, device, backend)
    return torch


def _fake_onnxruntime(providers):
    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: list(providers)
    return ort


class TheSilentCpuFallbackIsNamedOutLoud(unittest.TestCase):
    def setUp(self):
        from ball_reel import device

        self.d = device

    def test_a_missing_accelerator_provider_is_reported(self):
        want, missing = self.d.onnx_providers(
            "cuda", available=["CPUExecutionProvider"])
        self.assertIn("CUDAExecutionProvider", want)
        self.assertEqual(missing, ("CUDAExecutionProvider",))

    def test_everything_present_reports_nothing_missing(self):
        _, missing = self.d.onnx_providers(
            "cuda", available=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(missing, ())

    def test_cpu_is_never_reported_as_missing(self):
        # Он есть всегда; попасть в «не хватает» он может только по ошибке
        # в самой проверке.
        _, missing = self.d.onnx_providers("cpu", available=[])
        self.assertEqual(missing, ())

    def test_every_list_ends_in_cpu_so_the_fallback_is_explicit(self):
        # Без явного запасного получилось бы «выбрали ускоритель, считаем на
        # процессоре и не знаем об этом».
        for dev, chain in self.d.ONNX_PROVIDERS.items():
            self.assertEqual(chain[-1], "CPUExecutionProvider", dev)

    def test_intel_asks_for_intel_runtimes_not_cuda(self):
        want, _ = self.d.onnx_providers("xpu", available=[])
        self.assertNotIn("CUDAExecutionProvider", want)
        self.assertIn("OpenVINOExecutionProvider", want)


class DeviceChoiceHasConsequencesBeyondTheName(unittest.TestCase):
    def setUp(self):
        from ball_reel import device

        self.d = device

    def test_cpu_gets_full_precision_because_half_is_slower_there(self):
        # fp16 на процессоре эмулируется: это плата временем за ничего.
        self.assertEqual(self.d.dtype_for("cpu"), "float32")

    def test_accelerators_get_half_precision(self):
        for dev in ("cuda", "xpu"):
            self.assertEqual(self.d.dtype_for(dev), "float16")

    def test_insightface_is_told_cpu_rather_than_a_device_it_cannot_use(self):
        # У insightface нет понятия «xpu». Указать ему номер несуществующего
        # ускорителя — значит получить молчаливый откат внутри чужой
        # библиотеки, где мы его уже не увидим.
        self.assertEqual(self.d.insightface_ctx("cuda"), 0)
        self.assertEqual(self.d.insightface_ctx("xpu"), -1)
        self.assertEqual(self.d.insightface_ctx("cpu"), -1)

    def test_cuda_is_preferred_because_the_ecosystem_is_proven_there(self):
        self.assertEqual(self.d.DEVICE_ORDER[0], "cuda")
        self.assertEqual(self.d.DEVICE_ORDER[-1], "cpu")

    def test_detection_never_raises_and_always_names_something(self):
        # В среде разработки карты нет — обязан вернуть cpu, а не упасть.
        self.assertIn(self.d.detect(), self.d.DEVICE_ORDER)

    def test_the_description_starts_with_the_hardware(self):
        # Число без железа бессмысленно, поэтому шапка отчёта начинается с него.
        self.assertTrue(self.d.describe("cpu").startswith("устройство cpu"))


class AlternativesAreNotRequirements(unittest.TestCase):
    """Дефект: список провайдеров читался как «нужны все», а он «или-или».

    Для Intel годится OpenVINO ИЛИ DirectML; DirectML вдобавок бывает только на
    Windows, поэтому на линуксовой машине с работающим OpenVINO старый код
    вечно докладывал «упадём на CPU». Постоянная ложная тревога в
    диагностическом модуле хуже молчания: по ней принимают решение о железе, а
    она не отличает «ускорения нет» от «второго варианта ускорения нет».
    """

    def setUp(self):
        from ball_reel import device

        self.d = device

    def test_intel_with_only_openvino_is_not_an_alarm(self):
        # Ловит ложную тревогу: один из двух путей Intel есть — значит
        # ускорение есть, и жаловаться не на что.
        _, missing = self.d.onnx_providers(
            "xpu", available=["OpenVINOExecutionProvider",
                              "CPUExecutionProvider"])
        self.assertEqual(missing, ())

    def test_intel_with_only_directml_is_not_an_alarm_either(self):
        # Та же альтернатива с другой стороны: Windows-машина без OpenVINO,
        # но с DirectML тоже ускоряется.
        _, missing = self.d.onnx_providers(
            "xpu", available=["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(missing, ())

    def test_intel_without_any_accelerator_is_still_caught(self):
        # Обратная сторона: починка, которая просто гасит предупреждение,
        # ломает то, ради чего модуль написан. Нет НИ ОДНОГО пути — обязан
        # сказать, и назвать оба, чтобы было понятно, что ставить.
        _, missing = self.d.onnx_providers(
            "xpu", available=["CPUExecutionProvider"])
        self.assertIn("OpenVINOExecutionProvider", missing)
        self.assertIn("DmlExecutionProvider", missing)
        self.assertNotIn("CPUExecutionProvider", missing)

    def test_a_foreign_provider_does_not_count_as_acceleration(self):
        # Сторожит починку «непусто, значит ускоряемся»: в реальном
        # onnxruntime рядом с CPU лежит AzureExecutionProvider, к карте он
        # отношения не имеет. Проверено на этой машине: список рантайма —
        # ['AzureExecutionProvider', 'CPUExecutionProvider'].
        _, missing = self.d.onnx_providers(
            "xpu", available=["AzureExecutionProvider", "CPUExecutionProvider"])
        self.assertTrue(missing)

    def test_the_request_still_carries_every_alternative(self):
        # Отсутствующий провайдер onnxruntime переживает молча, поэтому просить
        # надо всё сразу: не угадали с OpenVINO — подхватит DirectML.
        want, _ = self.d.onnx_providers(
            "xpu", available=["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertIn("OpenVINOExecutionProvider", want)
        self.assertIn("DmlExecutionProvider", want)
        self.assertEqual(want[-1], "CPUExecutionProvider")

    def test_cuda_alone_in_its_set_still_reports_its_absence(self):
        # У NVIDIA альтернатива ровно одна, и «хотя бы один из одного» обязан
        # означать то же, что раньше, — иначе фиксом Intel мы бы ослепили CUDA.
        _, missing = self.d.onnx_providers(
            "cuda", available=["CPUExecutionProvider"])
        self.assertEqual(missing, ("CUDAExecutionProvider",))

    def test_cpu_has_no_alternatives_to_miss(self):
        # На процессоре ускорять нечего, поэтому пустая тревога тут — не
        # «всё хорошо», а отсутствие самого повода.
        self.assertEqual(self.d.ONNX_ACCELERATORS["cpu"], ())
        self.assertEqual(set(self.d.ONNX_ACCELERATORS["xpu"]),
                         {"OpenVINOExecutionProvider", "DmlExecutionProvider"})

    def test_the_warning_in_the_report_line_follows_the_same_rule(self):
        # Строка отчёта — то, что человек прочитает; проверяются обе стороны:
        # с рабочим OpenVINO про CPU речи быть не должно, без него — должна.
        _install_module(self, "torch", None)
        _install_module(self, "onnxruntime",
                        _fake_onnxruntime(["OpenVINOExecutionProvider",
                                           "CPUExecutionProvider"]))
        self.assertNotIn("CPU", self.d.describe("xpu"))

        sys.modules["onnxruntime"] = _fake_onnxruntime(["CPUExecutionProvider"])
        self.assertIn("CPU", self.d.describe("xpu"))


class TorchHasThreeStatesNotTwo(unittest.TestCase):
    """Дефект: «torch не установлен» говорилось и тогда, когда он установлен.

    Опрос устройства падает на xpu-сборках и при неподнятом драйвере; старый
    `describe` глотал любое исключение и объявлял пакет отсутствующим. Это
    диагностика, которая посылает чинить не то: ставить уже стоящее вместо
    того, чтобы смотреть на драйвер.
    """

    def setUp(self):
        from ball_reel import device

        self.d = device
        # onnxruntime тут не судится — фиксируем его, чтобы хвост строки не
        # мешал читать голову.
        _install_module(self, "onnxruntime",
                        _fake_onnxruntime(["OpenVINOExecutionProvider",
                                           "CPUExecutionProvider"]))

    def test_absent_torch_is_named_absent(self):
        # Первое состояние: пакета нет. Единственное, где уместно «не
        # установлен».
        _install_module(self, "torch", None)
        state, _, _, _ = self.d.torch_state("xpu")
        self.assertEqual(state, self.d.TORCH_ABSENT)
        self.assertIn("не установлен", self.d.describe("xpu"))

    def test_a_broken_device_query_is_not_called_a_missing_package(self):
        # Второе состояние, ради которого всё затевалось: пакет есть, версия
        # известна, а карта не опрашивается. Ловит подмену диагноза.
        _install_module(self, "torch",
                        _fake_torch("2.5.1+xpu", "xpu",
                                    boom=RuntimeError("XPU driver not found")))
        state, version, _, reason = self.d.torch_state("xpu")
        self.assertEqual(state, self.d.TORCH_SILENT)
        self.assertEqual(version, "2.5.1+xpu")
        self.assertIn("XPU driver not found", reason)
        line = self.d.describe("xpu")
        self.assertNotIn("не установлен", line)
        self.assertIn("2.5.1+xpu", line)
        # Текст исключения обязан доехать до строки: без него читатель знает,
        # что сломалось, но не знает — драйвер это или сборка.
        self.assertIn("XPU driver not found", line)

    def test_a_working_card_is_named(self):
        # Третье состояние: всё в порядке — тогда в строке имя карты и никаких
        # следов первых двух.
        _install_module(self, "torch",
                        _fake_torch("2.5.1+xpu", "xpu", name="Intel Arc A580"))
        state, _, name, reason = self.d.torch_state("xpu")
        self.assertEqual(state, self.d.TORCH_OK)
        self.assertEqual(name, "Intel Arc A580")
        self.assertEqual(reason, "")
        line = self.d.describe("xpu")
        self.assertIn("Intel Arc A580", line)
        self.assertNotIn("не установлен", line)

    def test_the_three_states_read_differently(self):
        # Различимость — это и есть требование: три разных беды не имеют права
        # выглядеть одинаково в шапке отчёта, по которой их чинят.
        lines = []
        _install_module(self, "torch", None)
        lines.append(self.d.describe("xpu"))
        sys.modules["torch"] = _fake_torch("2.5.1+xpu", "xpu",
                                           boom=OSError("Level Zero missing"))
        lines.append(self.d.describe("xpu"))
        sys.modules["torch"] = _fake_torch("2.5.1+xpu", "xpu",
                                           name="Intel Arc A580")
        lines.append(self.d.describe("xpu"))
        self.assertEqual(len(set(lines)), 3, lines)

    def test_a_torch_that_fails_to_load_is_not_a_torch_that_is_absent(self):
        # Крайний случай той же породы: пакет на месте, но импорт падает не
        # ImportError'ом (битая сборка, отсутствующая libze_loader). Свалить
        # это в «не установлен» — снова послать не туда.
        class Exploding(types.ModuleType):
            def __getattr__(self, item):
                raise OSError("libze_loader.so.1: cannot open shared object")

        _install_module(self, "torch", Exploding("torch"))
        state, _, _, reason = self.d.torch_state("xpu")
        self.assertNotEqual(state, self.d.TORCH_ABSENT)
        self.assertIn("libze_loader", reason)

    def test_cpu_is_a_healthy_state_not_a_broken_query(self):
        # У torch.cpu нет `get_device_name`, и это не поломка: отсутствие
        # имени не имеет права выглядеть как отказ опроса.
        torch = types.ModuleType("torch")
        torch.__version__ = "2.5.1"
        torch.cpu = types.SimpleNamespace()
        _install_module(self, "torch", torch)
        state, _, name, reason = self.d.torch_state("cpu")
        self.assertEqual(state, self.d.TORCH_OK)
        self.assertEqual((name, reason), ("", ""))


if __name__ == "__main__":
    unittest.main()
