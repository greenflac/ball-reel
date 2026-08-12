"""Решения предполёта — без арендованной машины и без torch.

Предполёт целиком состоит из проверок железа, и это его беда как кода: он
почти весь неисполним там, где мы разрабатываем. Мутационный аудит показал,
чем это кончается — снятый MIN_VRAM_GB не ронял ни одного теста, то есть
порог, ради которого предполёт и написан, не сторожил никто.

Поэтому решение отделено от чтения карты (`vram_verdict` берёт число, а не
устройство), и здесь судится оно. Тому же принципу подчинена проверка
условий: она про файлы, а не про GPU, и потому проверяема дома — а сломанная
последовательность условий стоит арендной минуты ровно так же.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image  # noqa: F401
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


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
        self.assertTrue(self.p.vram_verdict(floor)[0])
        self.assertFalse(self.p.vram_verdict(floor - 0.1)[0])
        self.assertGreater(floor, 2.0)
        self.assertLess(floor, 4.0)


class TheTorchBuildIsJudgedByItsName(unittest.TestCase):
    """CPU-сборка и мёртвый драйвер — разные беды, чинятся по-разному."""

    def setUp(self):
        from ball_reel import preflight_gpu

        self.p = preflight_gpu

    def test_a_working_build_reports_the_card(self):
        ok, detail = self.p.torch_verdict("2.13.0+cu126", True, "RTX 3050")
        self.assertTrue(ok)
        self.assertIn("RTX 3050", detail)

    def test_a_cpu_build_is_named_as_such_not_blamed_on_the_driver(self):
        # Живой случай: `torch>=2.6` в requirements без индекса даёт на Windows
        # именно это, молча перекрывая правильную установку. Совет «проверить
        # драйвер» здесь уводит в сторону — никакой драйвер CPU-сборку не
        # оживит, её надо переставить.
        ok, detail = self.p.torch_verdict("2.13.0+cpu", False, "")
        self.assertFalse(ok)
        self.assertIn("CPU-сборка", detail)
        self.assertIn("uninstall", detail)

    def test_a_cuda_build_without_a_card_points_at_the_driver(self):
        ok, detail = self.p.torch_verdict("2.13.0+cu126", False, "")
        self.assertFalse(ok)
        self.assertIn("драйвер", detail)
        self.assertNotIn("CPU-сборка", detail)


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

    def test_a_clean_sequence_passes(self):
        for i in range(3):
            self._png(f"{i:04d}.png")
        ok, _, detail = self.p.check_conditions(str(self.root))
        self.assertTrue(ok, detail)
        self.assertIn("3", detail)


if __name__ == "__main__":
    unittest.main()
