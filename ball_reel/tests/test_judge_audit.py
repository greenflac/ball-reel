"""Набор читается СУДЬЁЙ при сборке. Диагностика, а не отбор.

ЗАЧЕМ ЭТОТ ФАЙЛ, И ЦЕНА ЕГО ОТСУТСТВИЯ ИЗМЕРЕНА. Набор `demo/lora_dataset`
собрался полностью «зелёным»: отбор 92%, подписи у всех тридцати одного кадра,
`independence_report` без замечаний, план обучения 350 шагов. На нём обучили
LoRA, получили 0.655 при баре 0.35 — чужой человек, — и два часа крутили силы
адаптеров, потому что причина не печаталась НИГДЕ.

Вскрытие судьёй, сделанное вручную, ответило за две минуты:

    real + augmented (9)    ArcFace 0.000 .. 0.024
    generated       (22)    ArcFace 0.418 .. 0.544
    в баре 0.35: 9 из 31

71% набора — для судьи другой человек. Всё, что печаталось при сборке, было
правдой, и ни одна строка не была ПРО ЭТО.

ПОЧЕМУ ЭТО НЕ ЛОМАЕТ НЕЗАВИСИМОСТЬ. Судья запрещён в порождении и в отборе: там
он оптимизировал бы или выбирал под себя. Здесь он ЧИТАЕТ готовый набор и не
выбрасывает ни одного кадра. Тест ниже это и сторожит: как только вскрытие
начнёт отбирать, независимость исчезнет молча.
"""

from __future__ import annotations

import inspect
import unittest

from ball_reel import dataset


class _S:
    """Двойник `Sample`: аудиту нужны только путь и происхождение."""

    def __init__(self, path, origin="generated"):
        self.path, self.origin = path, origin


class ItMeasuresAndDoesNotSelect(unittest.TestCase):

    def test_no_frame_is_dropped_by_the_judge(self):
        """Главный запрет. Отбор судьёй = судья грейдит свою работу."""
        src = inspect.getsource(dataset.judge_audit)
        for word in ("kept", "remove", "del ", "pop("):
            self.assertNotIn(word, src,
                             f"{word!r} в аудите: судья начал отбирать")

    def test_build_dataset_does_not_filter_on_the_audit(self):
        src = inspect.getsource(dataset.build_dataset)
        # Аудит вызывается ПОСЛЕ сборки манифеста и только кладётся в него.
        self.assertIn("judge_audit(samples, face)", src)
        self.assertLess(src.index("man = manifest("),
                        src.index("judge_audit(samples, face)"),
                        "аудит оказался до сборки — значит может на неё влиять")

    def test_every_sample_appears_in_the_rows(self):
        got = dataset.judge_audit([_S("нет-такого-1.png"),
                                   _S("нет-такого-2.png")], "нет-лица.jpg")
        self.assertEqual(len(got["rows"]), 2,
                         "кадр пропал из вскрытия — значит про него нечего "
                         "сказать, и это выглядит как «претензий нет»")


class ItRefusesHonestly(unittest.TestCase):

    def test_nothing_measurable_is_not_approval(self):
        got = dataset.judge_audit([_S("нет-такого.png")], "нет-лица.jpg")
        self.assertFalse(got["ok"])
        self.assertIn("не одобрение", got["note"])

    def test_an_empty_set_does_not_pass_by_vacuum(self):
        got = dataset.judge_audit([], "нет-лица.jpg")
        self.assertFalse(got["ok"])


class TheThresholdIsHonestAboutItself(unittest.TestCase):

    def test_it_is_a_majority_and_declared_CHOSEN(self):
        # Порог ВЫБРАН, а не измерен: точка данных одна и она в провале.
        # Строка про это обязана стоять рядом с числом.
        self.assertEqual(dataset.JUDGE_AGREEMENT_MIN, 0.5)
        src = inspect.getsource(dataset)
        head = src[:src.index("JUDGE_AGREEMENT_MIN = 0.5")]
        self.assertIn("ВЫБРАН", head[-1200:],
                      "порог не помечен как выбранный — на этом проекте "
                      "ИЗМЕРЕН и ВЫБРАН не смешиваются")

    def test_the_measured_failure_is_recorded_beside_it(self):
        # Число без истории через неделю выглядит произвольным, и его начинают
        # двигать. Здесь рядом стоит случай, ради которого оно появилось.
        src = inspect.getsource(dataset)
        head = src[:src.index("JUDGE_AGREEMENT_MIN = 0.5")]
        self.assertIn("0.655", head[-1500:])


class TheOperatorIsWarnedButNotBlocked(unittest.TestCase):

    def test_the_cli_prints_the_verdict(self):
        src = inspect.getsource(dataset.main)
        self.assertIn("судья о наборе", src)

    def test_a_bad_audit_does_not_stop_the_build(self):
        """Отказ здесь означал бы, что судья управляет составом набора.

        Владелец вправе обучать на чём хочет; наша обязанность — сказать цену,
        а не запретить. Но молчание недопустимо: оно стоило обучения впустую.
        """
        src = inspect.getsource(dataset.main)
        after = src[src.index("судья о наборе"):]
        block = after[:after.index("if man[\"size\"] < MIN_DATASET")]
        self.assertNotIn("return 1", block,
                         "плохой аудит останавливает сборку — судья начал "
                         "управлять составом набора")
        self.assertIn("ВНИМАНИЕ", block)

    def test_the_warning_names_what_does_not_help(self):
        # Без этого оператор пойдёт крутить силу адаптера — измерено, что это
        # даёт 0.019.
        src = inspect.getsource(dataset.main)
        self.assertIn("0.019", src)
