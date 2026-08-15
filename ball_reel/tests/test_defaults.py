"""Умолчания точек входа: рабочие, согласованные и объяснённые.

ЗАЧЕМ ЭТОТ ФАЙЛ. За одну смену умолчания стоили двух разборов не по адресу:

* `--vram 4.0` уводил план на 384x576, ниже родного разрешения SD1.5, и
  выдавал сломанный клип — а разбор пошёл на число шагов и на набор;
* `--faceid-lora-scale 1.0` был ровно той конфигурацией, про которую измерено,
  что она даёт радужные потёки; каждый годный кадр требовал руками писать 0;
* `--lora-scale` стоял 0.7 в прогоне и 0.8 в пробе — настройка, найденная за
  тридцать секунд, приезжала в клип изменённой.

Форма у всех трёх одна: УМОЛЧАНИЕ, ВЫБРАННОЕ ОСТОРОЖНО ИЛИ ПО ИНЕРЦИИ, ТИХО
СТАНОВИТСЯ ОБЫЧНЫМ СЛУЧАЕМ. Забыть флаг проще, чем указать лишний.
"""

from __future__ import annotations

import unittest


def _defaults(parser) -> dict:
    return {a.dest: a.default for a in parser._actions}


def _argparse_default(module: str, flag: str):
    """Умолчание флага, прочитанное по дереву разбора модуля.

    Для точек входа, у которых парсер собирается внутри `main` и достать его
    можно только запуском CLI.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / module).read_text(
        encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == flag):
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                return kw.value.value
        return None
    raise AssertionError(f"в {module} нет флага {flag}")


class OneKnobOneDefault(unittest.TestCase):
    """Ручка с двумя умолчаниями — два прибора под одним именем."""

    def test_lora_scale_is_the_same_in_probe_and_in_the_run(self):
        import argparse

        from ball_reel import probe
        from ball_reel.run_local import build_parser

        run = _defaults(build_parser())["lora_scale"]
        ap = argparse.ArgumentParser()
        ap.add_argument("--lora-scale", type=float,
                        default=probe.DEFAULT_LORA_SCALE)
        self.assertEqual(run, probe.DEFAULT_LORA_SCALE,
                         "прогон и проба снова расходятся по силе LoRA: "
                         "настройка, найденная пробой, приедет в клип другой")

    def test_the_shared_default_lives_in_one_place(self):
        # Если константу скопируют обратно в оба входа, копии разойдутся —
        # этот проект уже терял так целевой путь.
        import inspect

        from ball_reel import run_local

        src = inspect.getsource(run_local.build_parser)
        self.assertIn("DEFAULT_LORA_SCALE", src,
                      "умолчание силы LoRA снова вписано числом, а не взято "
                      "из общего места")


class DefaultsAreWorkingNotCautious(unittest.TestCase):

    def test_faceid_lora_is_off_by_default(self):
        """Измерено: 1.0 вместе с проекцией даёт потёки, 0 даёт фотографию.

        Читается РАЗБОРОМ ИСХОДНИКА, а не вызовом: у пробы парсер собирается
        внутри `main`, и вытащить его иначе нельзя, не запуская CLI.
        """
        self.assertEqual(_argparse_default("probe.py", "--faceid-lora-scale"),
                         0.0,
                         "умолчание FaceID-LoRA снова не ноль: это ровно та "
                         "конфигурация, про которую измерено, что она даёт "
                         "радужные потёки вместо лица")

    def test_the_run_default_vram_gives_native_resolution(self):
        from ball_reel import animate
        from ball_reel.run_local import build_parser

        vram = _defaults(build_parser())["vram"]
        plan = animate.plan(vram)
        self.assertGreaterEqual(min(plan.width, plan.height), 512,
                                f"умолчание --vram {vram} снова даёт "
                                f"{plan.width}x{plan.height}")

    def test_probe_and_run_agree_on_vram(self):
        # Проба настраивает то, что потом рисует прогон; разные умолчания по
        # памяти означают разные разрешения и несравнимые результаты.
        import argparse

        from ball_reel.run_local import build_parser

        run = _defaults(build_parser())["vram"]
        ap = argparse.ArgumentParser()
        ap.add_argument("--vram", type=float, default=6.0)
        self.assertEqual(run, _defaults(ap)["vram"])
