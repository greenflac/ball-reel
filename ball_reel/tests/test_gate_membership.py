"""СОСТАВ ГЕЙТА. Что останавливает выдачу, а что только считается.

ЗАЧЕМ ЭТОТ ФАЙЛ. Гейтов в проекте ДВА — локальный (`run_local`) и шлюзовой
(`produce`), — и списки осей у них разные. Пока это нигде не было закреплено,
документы описывали «тринадцать осей» одним списком, в котором стояли и оси,
не подключённые ни к одному вердикту:

    заявлено в схеме README     на деле
    -----------------------     -------
    жидкость                    `fluid.py` не вызывается НИ ОДНИМ модулем
    приметы                     считаются, в вердикт не входят
    объект                      порога нет, ось не заведена
    сцена (semantic)            только в `produce`, в `run_local` НЕТ
    действие                    только в `run_local`, в `produce` НЕТ

Читатель отчёта, увидев ось в списке, вправе считать, что её провал остановит
выдачу. Для пяти позиций из списка это было неверно, и проверить это можно было
только чтением кода — то есть никак, если ты ревьюер с ограниченным временем.

Тест закрепляет состав ОБОИХ гейтов и требование, чтобы README называл разницу.
Если ось подключат или отключат, красным станет здесь, а не на демо.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Строки локального гейта, которые считает `clip_verdict`. Порядок значим: по
#: нему строится таблица в выводе прогона.
LOCAL_CLIP_ROWS = ("идентичность", "луп", "движение", "анатомия", "одежда",
                   "действие")

#: Остальные останавливающие строки локального пути. Собираются не в
#: `clip_verdict`, а рядом, и попадают в тот же список `rows`, который читает
#: `run_verdict`, — значит их `False` тоже роняет код возврата.
LOCAL_EXTRA_BLOCKING = ("лицо по кадрам", "поза по кадрам", "прилегание",
                        "мимика", "обрезка лупа")

#: Строки, которые печатаются и кладутся в отчёт, но в `rows` НЕ попадают,
#: то есть код возврата не трогают. Список закрытый: попасть сюда — заявление
#: «эта ось справочная», и оно должно быть видно.
LOCAL_INFORMATIONAL = ("кадрировка", "идентичность заранее", "доводка лица",
                       "апскейл")


class TheLocalGateHasExactlyTheseAxes(unittest.TestCase):

    def test_clip_verdict_returns_the_six_axes_and_nothing_else(self):
        from ball_reel.run_local import clip_verdict

        rows = clip_verdict({}, {}, {}, {}, {}, {})
        self.assertEqual(tuple(r["label"] for r in rows), LOCAL_CLIP_ROWS)

    def test_with_nothing_measured_every_axis_says_so(self):
        """Пустой вход — это «не смогли измерить», а НЕ «прошло».

        Клип из 16 одинаковых кадров даёт ровно такую картину, и двузначный
        гейт объявил бы его годным.
        """
        from ball_reel.run_local import clip_verdict

        rows = clip_verdict({}, {}, {}, {}, {}, {})
        self.assertTrue(all(r["ok"] is None for r in rows),
                        [f'{r["label"]}={r["ok"]}' for r in rows])

    def test_nothing_measured_is_a_refusal_and_not_a_pass(self):
        from ball_reel.run_local import clip_verdict, run_verdict

        code, why = run_verdict(clip_verdict({}, {}, {}, {}, {}, {}))
        self.assertEqual(code, 1)
        self.assertIn("НИ ОДНА ось не измерена", why)

    def test_the_informational_rows_do_not_reach_the_verdict(self):
        """`доводка лица` и `апскейл` возвращаются ОТДЕЛЬНЫМИ ключами.

        Если их однажды добавят в `rows`, они начнут ронять прогон, и таблица
        «в вердикте» в `docs/REPORT.md` разойдётся с кодом молча.
        """
        import inspect

        from ball_reel import run_local

        # Строки собираются в `_animatediff_once` — это она возвращает `rows`,
        # которые потом читает `run_verdict`.
        src = inspect.getsource(run_local._animatediff_once)
        for name in ("refine_row", "up_row"):
            with self.subTest(row=name):
                self.assertNotIn(f"rows.append({name})", src,
                                 f"{name} попал в вердикт — тогда таблица "
                                 f"«в вердикте» в docs/REPORT.md устарела")
        self.assertIn('"refine": refine_row', src)
        self.assertIn('"upscale": up_row', src)

    def test_the_loop_trim_and_style_rows_DO_reach_the_verdict(self):
        """Зеркало предыдущего: эти две в `rows` попадают и роняют прогон.

        Пара нужна целиком. Проверка «эти не попадают» зеленела бы и в том
        случае, если бы в вердикт не попадало ВООБЩЕ ничего.
        """
        import inspect

        from ball_reel import run_local

        src = inspect.getsource(run_local._animatediff_once)
        self.assertIn("rows.append(loop_row)", src)
        self.assertIn("rows.append(style_row)", src)


class TwoUncalibratedInstrumentsDoFailTheRun(unittest.TestCase):
    """РАСХОЖДЕНИЕ ДОК/КОД, зафиксированное как есть, а не заглаженное.

    `docs/GUIDE.md` утверждал про `прилегание` и `мимику`: «печатают число, но
    **не останавливают** прогон», с обоснованием — бар на непроверенном приборе
    не должен браковать оплаченный результат. **Код делает обратное.** Обе
    строки кладутся в общий список `rows` (`wardrobe_rows`), а `run_verdict`
    возвращает 1 по любому `ok is False`.

    Прав по существу документ, и по правилу самого проекта: у мимики полосы
    СОМКНУЛИСЬ на калибровке — свои p90 0.0355 против чужих p10 0.0352
    (`docs/REPORT.md` §2), — а `bar_from_pairs` при перекрытии облаков
    отказывается выводить порог. Прилегание мерено только на живой съёмке и ни
    разу на генерации.

    Но поведение гейта — это не документация, и менять его правкой «под текст»
    нельзя: цена ошибки здесь — забракованный или пропущенный клип, а проверить
    последствия можно только живым прогоном на карте, которой в среде разработки
    нет. Поэтому расхождение ЗАКРЫТО В ДОКУМЕНТАХ: они теперь говорят то, что
    код делает, и отдельно называют, почему это спорно. Тест фиксирует
    фактическое поведение, чтобы оно не изменилось молча.
    """

    def test_a_failing_expression_row_DOES_fail_the_run(self):
        from ball_reel.run_local import run_verdict

        code, why = run_verdict([{"label": "идентичность", "ok": True},
                                 {"label": "мимика", "ok": False}])
        self.assertEqual(code, 1, why)
        self.assertIn("мимика", why)

    def test_a_failing_fit_row_DOES_fail_the_run(self):
        from ball_reel.run_local import run_verdict

        code, why = run_verdict([{"label": "идентичность", "ok": True},
                                 {"label": "прилегание", "ok": False}])
        self.assertEqual(code, 1, why)

    def test_both_rows_are_built_without_any_opt_out_marker(self):
        """Если появится способ пометить строку справочной — сюда же.

        Тест смотрит на исходник: пока ключа нет, оба измерителя блокируют, и
        документы обязаны говорить именно это.
        """
        import inspect

        from ball_reel import run_local

        src = inspect.getsource(run_local.wardrobe_rows)
        self.assertNotIn("blocking", src,
                         "у строк появился признак «справочная» — тогда правь "
                         "docs/REPORT.md §2 и docs/GUIDE.md вместе с кодом")

    def test_the_docs_admit_the_disagreement_instead_of_hiding_it(self):
        guide = (ROOT / "docs" / "GUIDE.md").read_text(encoding="utf-8")
        self.assertIn("НЕ СООТВЕТСТВУЕТ КОДУ", guide,
                      "GUIDE снова заявляет, что эти оси не останавливают "
                      "прогон, не отметив, что код делает иначе")


class TheTwoGatesAreNotTheSameGate(unittest.TestCase):
    """Ровно то место, где документы врали: списки РАЗНЫЕ."""

    def setUp(self):
        from ball_reel import produce

        self.order = produce.CHECK_ORDER

    def test_the_gateway_gate_has_thirteen_checks(self):
        self.assertEqual(len(self.order), 13, self.order)

    def test_semantic_is_in_the_gateway_gate(self):
        self.assertIn("semantic", self.order)

    def test_semantic_is_NOT_wired_into_the_local_path(self):
        """НЕ ПОДКЛЮЧЕНО, и это закреплено, а не забыто.

        `run_local` не вызывает `semantic`: ось «та ли одежда заказана» на
        локальном пути не считается вовсе. Если её подключат — тест покраснеет,
        и вместе с кодом придётся поправить README и `docs/REPORT.md`.
        """
        src = (ROOT / "ball_reel" / "run_local.py").read_text(encoding="utf-8")
        self.assertNotIn("from .semantic import", src)
        self.assertNotIn("semantic_match", src)

    def test_action_is_NOT_wired_into_the_gateway_path(self):
        """Зеркальный случай: ось действия есть только локально."""
        src = (ROOT / "ball_reel" / "produce.py").read_text(encoding="utf-8")
        self.assertNotIn("from .action import", src)
        self.assertNotIn("action_match", src)
        self.assertNotIn("action.", src.replace('action="store_true"', ""))


class WhatIsMeasuredButNotWired(unittest.TestCase):
    """Три состояния из правил проекта: работает / не подключено / частично.

    Здесь закрепляется среднее. Модуль, который никто не зовёт, — законная
    вещь: измеритель можно написать до канала. Незаконно другое — заявлять его
    осью гейта. Тест ловит момент, когда состояние поменяется в любую сторону.
    """

    #: Модули-измерители, которые НЕ вызываются ни одним модулем пакета.
    #: Проверяется буквально, импортом.
    NOT_WIRED = ("fluid",)

    def _importers(self, name: str) -> list:
        out = []
        for p in sorted((ROOT / "ball_reel").glob("*.py")):
            if p.name in (f"{name}.py", "codeaudit.py"):
                continue            # сам себя и таблица мутаций не считаются
            text = p.read_text(encoding="utf-8")
            if f"from .{name} import" in text or f"from . import {name}" in text:
                out.append(p.name)
        return out

    def test_the_unwired_measurers_are_still_unwired(self):
        for name in self.NOT_WIRED:
            with self.subTest(module=name):
                self.assertEqual(
                    self._importers(name), [],
                    f"`{name}` кто-то подключил — это хорошо, но тогда его "
                    f"надо перенести из «считается» в «останавливает» в "
                    f"README и docs/REPORT.md, иначе документ разойдётся с "
                    f"кодом")

    def test_marks_are_transferred_but_judge_nothing(self):
        """Приметы переносятся `refine`, но в вердикт не входят — так и в доках."""
        self.assertIn("refine.py", self._importers("marks"))
        from ball_reel.run_local import clip_verdict

        labels = [r["label"] for r in clip_verdict({}, {}, {}, {}, {}, {})]
        self.assertNotIn("приметы", labels)


class TheReadmeSaysTheSameThing(unittest.TestCase):
    """Документ, разошедшийся с кодом, — дефект. Сверяется автоматически."""

    def setUp(self):
        self.text = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_it_admits_there_are_two_gates(self):
        self.assertIn("Гейтов два", self.text,
                      "README снова описывает гейт как один список")

    def test_it_marks_semantic_as_not_wired_locally(self):
        self.assertIn("НЕ ПОДКЛЮЧЕНА", self.text,
                      "README перестал говорить, что семантики нет в "
                      "локальном пути — а её там по-прежнему нет")

    def test_it_no_longer_lists_fluid_as_a_gate_axis(self):
        """`жидкость` стояла в рамке гейта, не будучи подключена никуда."""
        frame = self.text[self.text.index("ГЕЙТ"):]
        frame = frame[:frame.index("```")]
        self.assertNotIn("жидкость", frame)
        self.assertNotIn("объект", frame)


if __name__ == "__main__":
    unittest.main()
