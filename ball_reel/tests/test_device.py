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

import unittest


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


if __name__ == "__main__":
    unittest.main()
