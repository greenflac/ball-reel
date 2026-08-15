"""План не вправе опускать кадр ниже того, на чём базовая модель работает.

ЗАЧЕМ ЭТОТ ФАЙЛ, И ЦЕНА ЕГО ОТСУТСТВИЯ ИЗМЕРЕНА НА ЖИВОМ ПРОГОНЕ.

`--vram` по умолчанию стоял 4.0. Ветка плана «меньше 5 ГБ» понижает кадр до
384x576, а SD1.5 обучена на 512 и ниже разваливается. Прогон выдал:

    действие          0.27 при баре 0.35 — «в такт, но вяло»
    поза по кадрам    11 из 16 в баре
    лицо              НЕ НАЙДЕНО ни на одном из 16 кадров
    идентичность      ПРОПУСК: судить нечего

Гейт честно покраснел по четырём осям и четыре пометил «не измерено» — и НИ
ОДНА строка не назвала причину. План печатался, его число никто не сверял с
тем, на чём работает база, и разбор пошёл по ложному следу: сначала на число
шагов, потом на набор.

Форма дефекта общая и стоит запоминания: **умолчание, выбранное под худший
случай, тихо становится обычным случаем.** Ошибаются чаще в сторону «забыл
указать флаг», чем «указал лишнее», поэтому умолчание обязано быть рабочим, а
не безопасным.
"""

from __future__ import annotations

import unittest

from ball_reel import animate

#: Разрешение, на котором обучена SD1.5. Не наш выбор и не порог: свойство
#: весов. Ниже него модель не «чуть хуже», а ломается.
SD15_NATIVE = 512


class ThePlanNeverGoesBelowNative(unittest.TestCase):

    def test_the_default_vram_gives_a_usable_frame(self):
        """Главная проверка: прогон БЕЗ флагов обязан быть рабочим."""
        from ball_reel.run_local import build_parser

        args = build_parser().parse_args(["--face", "f.jpg", "--prompt", "p"])
        plan = animate.plan(args.vram)
        self.assertGreaterEqual(
            min(plan.width, plan.height), SD15_NATIVE,
            f"умолчание --vram {args.vram} даёт {plan.width}x{plan.height} — "
            f"ниже родного разрешения SD1.5. Именно так был потерян живой "
            f"прогон: артефакты, амплитуда 0.27 при баре 0.35 и лицо, не "
            f"найденное ни на одном кадре")

    def test_the_default_sits_at_the_boundary_and_not_above_it(self):
        # 6.0 выбрано как нижняя граница рабочей ветки, а не «побольше на
        # всякий случай»: завышенное умолчание врало бы о требованиях.
        from ball_reel.run_local import build_parser

        args = build_parser().parse_args(["--face", "f.jpg", "--prompt", "p"])
        below = animate.plan(args.vram - 1.5)
        self.assertLess(min(below.width, below.height), SD15_NATIVE,
                        "ветка ниже умолчания перестала быть пониженной — "
                        "значит умолчание завышено и врёт о требованиях")

    def test_every_branch_at_or_above_the_default_is_native_or_better(self):
        for vram in (6.0, 7.0, 8.0, 10.0, 16.0):
            with self.subTest(vram=vram):
                plan = animate.plan(vram)
                self.assertGreaterEqual(min(plan.width, plan.height),
                                        SD15_NATIVE)

    def test_the_low_branch_still_exists_and_says_so(self):
        """Карты меньше 5 ГБ бывают, и ветка для них остаётся.

        Убрать её значило бы отказать такой карте вовсе. Требование другое:
        она не должна быть УМОЛЧАНИЕМ, и её понижение должно быть названо.
        """
        plan = animate.plan(4.0)
        self.assertLess(min(plan.width, plan.height), SD15_NATIVE)
        self.assertTrue(plan.notes, "пониженная ветка молчит о том, что она "
                                    "пониженная")

    def test_the_framing_does_not_change_the_resolution(self):
        # Кадрировка влияет на долю лица, а не на размер кадра; если это
        # разойдётся, число «лицо N px» перестанет следовать из плана.
        a, b = animate.plan(6.0, waist_up=True), animate.plan(6.0, waist_up=False)
        self.assertEqual((a.width, a.height), (b.width, b.height))
