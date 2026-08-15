"""Ось «тот ли объект, та ли одежда, та ли сцена».

Вторая половина дыры, через которую прошёл кадр с лучшим за день дрейфом лица
(0.21): человек в БАЛЬНОМ ПЛАТЬЕ в позе с фотографии. `action` закрыл неверное
движение и сам записал, чего не видит: «человек в бальном платье и человек в
спортивной форме, совершающие одно движение, для неё неразличимы». Здесь
сторожится то, что видно только семантике.

Арифметика отделена от весов НАМЕРЕННО, и это не стиль, а исправленный дефект
проекта: тест, который зеленел, пока пакета не было, и краснел после установки,
уже случался. Поэтому всё, что можно проверить без 600 МБ весов, проверяется
без них, а прогон с весами живёт в отдельном классе под `skipUnless` и не
меняет исход набора ни в одну сторону.
"""

from __future__ import annotations

import unittest
from pathlib import Path

#: Кадры кита адресуются ОТ КОРНЯ ДЕРЕВА, а не от текущего каталога. Раньше
#: здесь стоял `glob("kit/driving/*.jpg")`: он зависел от того, откуда запущен
#: прогон, и вдобавок смотрел в `kit`, которого нет в git. Тест поэтому молчал
#: дважды — и в клоне, и при запуске из любого каталога, кроме одного.
KIT = Path(__file__).resolve().parents[2] / "demo" / "kit"


def _flat(value: float, n: int = 16) -> list:
    return [value] * n


class TheArithmeticOfTheAxis(unittest.TestCase):
    """Вердикт из покадровых марж. Ни весов, ни картинок."""

    def setUp(self):
        from ball_reel import semantic

        self.s = semantic

    def test_a_clip_that_matches_the_order_passes(self):
        got = self.s.judge_margins({"одежда": _flat(+0.11)})
        self.assertEqual(got["verdict"], "matches", got)
        self.assertIs(got["on_brief"], True)
        self.assertIsNone(got["check"])

    def test_the_ball_gown_case_is_REJECTED_and_names_itself(self):
        """Ради этого случая написан модуль: маржа кадров в платье.

        Числа литеральные: -0.09..-0.12 — это ИЗМЕРЕННЫЕ маржи четырёх кадров
        `startopt/face_first_*` и `plain_0`, размеченных глазами. Брать их из
        константы, которую тест сторожит, значит написать тест, который поедет
        вместе с ней и промолчит.
        """
        got = self.s.judge_margins(
            {"одежда": [-0.1201, -0.1155, -0.1114, -0.0897]})
        self.assertEqual(got["verdict"], "mismatched", got)
        self.assertIs(got["on_brief"], False)
        self.assertEqual(got["check"], "semantic_clothing")

    def test_a_margin_inside_the_measured_noise_is_not_a_verdict(self):
        """Разница меньше дрожания прибора — это «не смогли», а не «не то».

        Покадровый шум измерен: 71 кадр ОДНОГО видео с ОДНИМ набором текстов
        дают sd 0.0123. Объявлять вердикт на разнице 0.002 значит выдавать шум
        за измерение.
        """
        got = self.s.judge_margins({"одежда": _flat(+0.002)})
        self.assertEqual(got["verdict"], "not_measurable", got)
        self.assertIsNone(got["on_brief"])
        got = self.s.judge_margins({"одежда": _flat(-0.002)})
        self.assertEqual(got["verdict"], "not_measurable", got)

    def test_the_verdict_follows_the_median_not_a_single_frame(self):
        # Один провалившийся кадр из шестнадцати не бракует клип, но и не
        # тонет: доля выигрышей уходит в отчёт отдельным числом.
        vals = _flat(+0.10, 15) + [-0.12]
        got = self.s.judge_margins({"одежда": vals})
        self.assertEqual(got["verdict"], "matches", got)
        self.assertAlmostEqual(got["claims"]["одежда"]["share"], 15 / 16, 3)
        self.assertEqual(got["claims"]["одежда"]["worst"], -0.12)

    def test_a_mismatch_wins_over_an_unmeasurable_claim(self):
        """Находку нельзя терять из-за соседнего неизмеримого утверждения.

        Обратный порядок закрывал бы кадр в платье исходом «не смогли», если
        рядом оказалось утверждение без кадров, — то есть прятал бы брак за
        честным отказом.
        """
        got = self.s.judge_margins({"сцена": [], "одежда": _flat(-0.11)})
        self.assertEqual(got["verdict"], "mismatched", got)

    def test_the_sweep_from_wrong_to_right_crosses_the_bar_once(self):
        # Синтетика покрывает ДИАПАЗОН, а не одну точку: вердикт обязан быть
        # монотонным по марже, иначе порог означает не то, что написано.
        seen = []
        x = -0.20
        while x <= 0.2001:
            seen.append(self.s.judge_margins({"одежда": _flat(x)})["verdict"])
            x += 0.002
        self.assertEqual(seen[0], "mismatched")
        self.assertEqual(seen[-1], "matches")
        # ровно два перехода: mismatched -> not_measurable -> matches
        switches = sum(1 for a, b in zip(seen, seen[1:]) if a != b)
        self.assertEqual(switches, 2, seen)

    def test_a_claim_without_alternatives_has_no_scale(self):
        """Абсолютный скор CLIP шкалы не имеет — это измерено.

        Один текст про спортивную одежду даёт на кадрах с ВЕРНОЙ одеждой
        медианы 0.308 (съёмка), 0.266 и 0.182 (две генерации), на кадрах в
        ПЛАТЬЕ 0.105. Разброс внутри верного класса 0.20, зазор между классами
        0.013, покадровый шум 0.020 — зазор УЖЕ шума, порога нет. Утверждение
        без альтернатив обязано закрываться исходом «не смогли», а не
        превращаться в порог на абсолютном числе.
        """
        c = self.s.Claim(name="x", check="semantic_x",
                         expected=("a photo of a person",), decoys=())
        self.assertFalse(c.usable())
        got = self.s.semantic_match(["kit/face.jpg"], (c,))
        self.assertEqual(got["verdict"], "not_measurable", got)
        self.assertIsNone(got["on_brief"])

    def test_no_claims_at_all_is_not_a_pass(self):
        got = self.s.judge_margins({})
        self.assertEqual(got["verdict"], "not_measurable", got)
        self.assertIsNone(got["on_brief"])
        self.assertTrue(got["note"])

    def test_the_margin_is_the_best_of_each_side_not_the_first(self):
        """Набор формулировок меряет слот, одна фраза — удачность фразы.

        ИЗМЕРЕНО на 71 настоящем driving-кадре: с единственным «in athletic
        sportswear» (0.233) их обходило «in jeans and a shirt» (0.258) — все 71
        кадр объявлялись браком. «in a sports bra and leggings» даёт 0.297, то
        есть правило «максимум по набору» и есть то, что делает ось верной.
        """
        got = self.s.margins_from_scores(
            expected=[[0.233, 0.297, 0.242]], decoys=[[0.225, 0.258, 0.217]])
        self.assertAlmostEqual(got[0], 0.297 - 0.258, 6)

    def test_mismatched_score_tables_are_refused_loudly(self):
        with self.assertRaises(ValueError):
            self.s.margins_from_scores(expected=[[0.1], [0.2]], decoys=[[0.1]])
        with self.assertRaises(ValueError):
            self.s.margins_from_scores(expected=[[0.1]], decoys=[[]])


class ThreeOutcomesNotTwo(unittest.TestCase):
    """«Не смогли измерить» отличимо и от «по заказу», и от «не по заказу»."""

    def setUp(self):
        from ball_reel import semantic

        self.s = semantic

    def test_the_boolean_beside_the_verdict_does_not_collapse_the_third(self):
        """`on_brief` при «не смогли» обязан быть None, а не False.

        Булев флаг рядом с трёхзначным вердиктом — приглашение к ошибке,
        которую проект ловил в позе, в жидкости, в предполёте и в оси
        движения: читатель берёт флаг, и «не знаем» превращается в «плохо».
        """
        self.assertIsNone(self.s.judge_margins({"одежда": []})["on_brief"])
        self.assertIs(
            self.s.judge_margins({"одежда": _flat(+0.11)})["on_brief"], True)
        self.assertIs(
            self.s.judge_margins({"одежда": _flat(-0.11)})["on_brief"], False)

    def test_missing_weights_give_not_measurable_rather_than_a_crash(self):
        """600 МБ весов на машине демо может не быть — это исход, а не отказ.

        Модель называется заведомо отсутствующей, поэтому тест даёт один и тот
        же ответ и на машине с весами, и без них.
        """
        got = self.s.semantic_match(
            ["kit/face.jpg"], model_id="ball-reel/no-such-clip-model")
        self.assertEqual(got["verdict"], "not_measurable", got)
        self.assertIsNone(got["on_brief"])
        self.assertIn("snapshot_download", got["note"])

    def test_the_tokenizer_is_named_among_the_required_files(self):
        """Одних весов НЕ хватает, и это случившийся дефект, а не гипотеза.

        С весами в кэше, но без файлов токенизатора `CLIPProcessor`
        собирается молча и кодирует ЛЮБОЙ текст одинаково: в прогоне, где это
        случилось, все 19 текстов получили один и тот же скор 0.0519, все маржи
        вышли нулевыми, и ось печатала бы «соответствует» на чём угодно.
        """
        self.assertIn("tokenizer.json", self.s.REQUIRED_FILES)
        why = self.s.why_unavailable("ball-reel/no-such-clip-model")
        self.assertIn("tokenizer", why.lower())

    def test_a_collapsed_text_tower_is_caught_by_arithmetic(self):
        # Проверка ловит именно то, что случилось: одинаковые векторы.
        same = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        self.assertTrue(self.s.texts_collapsed(same))
        # А близкие, но разные тексты — не ловит. 0.9757 это ИЗМЕРЕННАЯ
        # ближайшая пара из пятнадцати текстов про одежду; если бы порог стоял
        # ниже неё, ось объявляла бы прибор сломанным на исправном приборе.
        near = [[1.0, 0.0], [0.9757, (1 - 0.9757 ** 2) ** 0.5]]
        self.assertFalse(self.s.texts_collapsed(near))

    def test_frames_that_do_not_open_are_not_a_pass(self):
        got = self.s.semantic_match(["kit/there-is-no-such-frame.png"])
        self.assertEqual(got["verdict"], "not_measurable", got)
        self.assertIsNone(got["on_brief"])

    def test_no_frames_at_all_is_not_a_pass(self):
        got = self.s.semantic_match([])
        self.assertEqual(got["verdict"], "not_measurable", got)
        self.assertIsNone(got["on_brief"])

    def test_render_prints_all_three_outcomes(self):
        for margins, word in ((_flat(+0.11), "ПО ЗАКАЗУ"),
                              (_flat(-0.11), "НЕ ПО ЗАКАЗУ"),
                              ([], "НЕ СМОГЛИ")):
            line = self.s.render(self.s.judge_margins({"одежда": margins}))
            self.assertIn(word, line)


class TheBarsAreGuardedAndJustified(unittest.TestCase):
    def setUp(self):
        from ball_reel import semantic

        self.s = semantic

    def test_the_bar_sits_inside_the_measured_gap(self):
        """Порог обязан лежать МЕЖДУ измеренными верными и неверными случаями.

        ИЗМЕРЕНО (`python3 -m ball_reel.semantic --controls`, разметка глазами):
            худшее ВЕРНОЕ окно из 56 на 71 driving-кадре      +0.0181
            генерация в спортивном топе, медиана по 5 кадрам  +0.0195
            ближайший НЕВЕРНЫЙ случай (скелет ControlNet)     -0.0216
            кадры в бальном платье                            -0.0897..-0.1201
        Числа литеральные намеренно.
        """
        self.assertLess(self.s.MARGIN_MIN, 0.0181)
        self.assertGreater(self.s.MARGIN_MIN, 0.0)
        # И с другой стороны разрыва: порог обязан браковать ближайший
        # неверный случай, а не только далёкое платье.
        self.assertLess(self.s.MARGIN_MIN, 0.0216)

    def test_the_bar_is_at_least_the_measured_frame_noise(self):
        # Шум измерен: sd покадровой маржи на 71 кадре ОДНОГО видео с ОДНИМ
        # набором текстов равен 0.0123. Порог ниже шума означал бы вердикты по
        # дрожанию прибора.
        self.assertGreaterEqual(self.s.MARGIN_MIN, 0.010)

    def test_the_bar_rejects_the_frame_the_whole_gate_let_through(self):
        # Обратная сторона: порог обязан что-то запрещать, иначе он украшение.
        got = self.s.judge_margins({"одежда": [-0.1201]})
        self.assertEqual(got["verdict"], "mismatched", got)

    def test_the_bar_accepts_the_worst_real_footage_window(self):
        # И обязан что-то ПРОПУСКАТЬ: порог, который бракует настоящую съёмку
        # заказанного, выключил бы ось молча — все клипы шли бы в брак.
        got = self.s.judge_margins({"одежда": [0.0181]})
        self.assertEqual(got["verdict"], "matches", got)

    def test_every_measured_single_frame_lands_where_it_was_measured(self):
        """Девять НАСТОЯЩИХ кадров `startopt`, судимых поштучно.

        Разметка глазами; маржи получены `python3 -m ball_reel.semantic
        --frames startopt/<файл>`. Ни одного неверного вердикта в обе стороны,
        но на двух верных кадрах из пяти одиночная маржа попадает в полосу
        шума — и ось обязана отвечать «не смогла», а не «не то».
        """
        for m in (-0.1201, -0.1155, -0.1114, -0.0897):   # платье
            self.assertEqual(
                self.s.judge_margins({"одежда": [m]})["verdict"],
                "mismatched", m)
        for m in (+0.0304, +0.0220, +0.0195):            # спортивный топ
            self.assertEqual(
                self.s.judge_margins({"одежда": [m]})["verdict"], "matches", m)
        for m in (+0.0107, +0.0046):                     # тоже топ, но в шуме
            self.assertEqual(
                self.s.judge_margins({"одежда": [m]})["verdict"],
                "not_measurable", m)

    def test_a_claim_needs_at_least_one_alternative(self):
        self.assertGreaterEqual(self.s.MIN_DECOYS, 1)

    def test_every_claim_carries_a_check_name_for_the_gate(self):
        # По именам строится статистика «что ломается первым»; текст причины
        # меняется при первой правке формулировки, и статистика по нему
        # разъезжается молча.
        for c in (self.s.CLOTHING, self.s.OBJECT_UNCALIBRATED,
                  self.s.SCENE_UNCALIBRATED):
            self.assertTrue(c.check.startswith("semantic_"), c.check)
            self.assertEqual(self.s.check_of(c.name), c.check)

    def test_the_collapse_bar_sits_between_measured_and_broken(self):
        # 0.9757 — ближайшая измеренная пара исправных текстов, 1.0000 —
        # сломанный токенизатор. Порог обязан лежать строго между.
        self.assertGreater(self.s.TEXT_COLLAPSE_MAX, 0.9757)
        self.assertLess(self.s.TEXT_COLLAPSE_MAX, 1.0)


class TheLicenceIsCheckedBeforeTheBuild(unittest.TestCase):
    """Продукт коммерческий, поэтому веса без объявленной лицензии не берём."""

    def setUp(self):
        from ball_reel import semantic

        self.s = semantic

    def test_the_weights_are_the_MIT_licensed_laion_ones(self):
        # У `openai/clip-vit-*` поля `license` в карточке НЕТ ВОВСЕ (проверено
        # `HfApi().model_info(...)`: `card_data['license']` -> None, тега
        # `license:*` нет). Отсутствие лицензии — не «разрешено по умолчанию».
        self.assertTrue(self.s.MODEL_ID.startswith("laion/"), self.s.MODEL_ID)
        self.assertNotIn("openai/", self.s.MODEL_ID)

    def test_the_module_records_why_openai_weights_are_refused(self):
        doc = self.s.__doc__ or ""
        self.assertIn("openai/clip-vit", doc)
        self.assertIn("mit", doc.lower())


class WhatThisAxisCannotSee(unittest.TestCase):
    """Границы оси записаны в модуле, а не подразумеваются.

    Зелёная строка «кадр по заказу» читается как «кадр верный», и молчать о
    том, чего ось не видит, нельзя.
    """

    def setUp(self):
        from ball_reel import semantic

        self.s = semantic

    def test_the_module_forbids_building_identity_on_CLIP(self):
        """Самое опасное применение этого модуля — и оно запрещено текстом.

        На наших кропах CLIP «разделяет» тех же и разных людей (0.068 против
        0.540), но два наших человека отличаются волосами, возрастом и тоном
        кожи, а кропы — резкостью: разведена могла быть внешность и качество, а
        не личность. Пар «похожие, но разные» в данных нет, значит это НЕ
        ИЗМЕРЕНО, а генератор делает похожего (дрейф 0.345 при баре 0.35).
        """
        doc = " ".join((self.s.__doc__ or "").split()).lower()
        self.assertIn("arcface", doc)
        self.assertIn("не измерено", doc)

    def test_the_module_says_it_does_not_see_motion(self):
        doc = " ".join((self.s.__doc__ or "").split()).lower()
        self.assertIn("движение", doc)
        self.assertIn("action", doc)

    def test_the_uncalibrated_claims_are_kept_out_of_the_gate(self):
        """Объект и сцена НЕ откалиброваны, и это записанный результат.

        Объект: на кадрах, где мяч ЕСТЬ, медиана -0.0087, на кадрах без мяча
        -0.0532 и -0.0622 — распределения перекрываются. Сцена: отрицательного
        контроля нет вовсе, все наборы лежат положительно. Пустить их в гейт
        значило бы браковать верные кадры по неизмеренному числу.
        """
        names = [c.name for c in self.s.DEFAULT_CLAIMS]
        self.assertEqual(names, [self.s.CLOTHING.name])
        self.assertNotIn(self.s.OBJECT_UNCALIBRATED, self.s.DEFAULT_CLAIMS)
        self.assertNotIn(self.s.SCENE_UNCALIBRATED, self.s.DEFAULT_CLAIMS)
        for c in (self.s.OBJECT_UNCALIBRATED, self.s.SCENE_UNCALIBRATED):
            self.assertIn("НЕ ОТКАЛИБРОВАНО", c.name)

    def test_the_axis_is_frame_wise_and_says_so(self):
        # Перетасованные кадры дают тот же вердикт: порядок ось не видит.
        vals = [+0.11, +0.09, -0.02, +0.13]
        a = self.s.judge_margins({"одежда": vals})
        b = self.s.judge_margins({"одежда": list(reversed(vals))})
        self.assertEqual(a["verdict"], b["verdict"])
        self.assertEqual(a["margin"], b["margin"])

    def test_the_cost_is_reported_rather_than_assumed(self):
        # Ось дорогая и обязана уметь назвать свою цену — на этом строится её
        # место в CHECK_ORDER (в самом конце: дешёвое раньше дорогого).
        got = self.s.cost(["kit/face.jpg"],
                          model_id="ball-reel/no-such-clip-model")
        self.assertFalse(got["measured"])
        self.assertTrue(got["note"])


#: Живой прогон включается ЯВНО, а не «если данные случайно лежат рядом».
#: Причина не в удобстве, а в двух измеренных вещах.
#: Во-первых, `codeaudit` гоняет мутации в КОПИИ пакета, куда линкуются только
#: `kit`, `demo`, `evidence`, `marktest` (`codeaudit.DATA_LINKS`). Тест,
#: который в дереве идёт, а в копии пропускается, ломает самопроверку
#: `copy_is_as_strong_as_the_tree`: «в копии пропущено больше, чем в дереве»
#: означает сторожа, молчащего во время КАЖДОЙ мутации.
#: Во-вторых, цена: прогон на настоящих кадрах стоит секунды на кадр, а мутаций
#: под сотню. Аудит, который перестают запускать, не сторожит ничего.
#: Числа живого прогона воспроизводятся командой из докстринга модуля:
#:     python3 -m ball_reel.semantic --controls
LIVE = __import__("os").environ.get("BALL_REEL_SEMANTIC_LIVE") == "1"


@unittest.skipUnless(
    LIVE and __import__("importlib").import_module(
        "ball_reel.semantic").available(),
    "живой прогон выключен: BALL_REEL_SEMANTIC_LIVE=1 и веса CLIP в кэше")
class WithTheRealWeights(unittest.TestCase):
    """Прогон на настоящих кадрах:

        BALL_REEL_SEMANTIC_LIVE=1 python3 -m unittest ball_reel.tests.test_semantic

    Класс намеренно не влияет на исход набора ни с весами, ни без них: тест,
    который зеленел, пока пакета не было, и краснел после установки, на этом
    проекте уже случался.
    """

    def setUp(self):
        from ball_reel import semantic

        self.s = semantic

    def test_the_ball_gown_frame_is_rejected_on_the_real_pixels(self):
        """Тот самый кадр: `startopt/face_first_0.png`, дрейф лица 0.2124."""
        import os

        if not os.path.exists("startopt/face_first_0.png"):
            self.skipTest("кадра нет в этом рабочем каталоге")
        got = self.s.semantic_match(["startopt/face_first_0.png"])
        self.assertEqual(got["verdict"], "mismatched", got)
        self.assertLess(got["margin"], -0.05, got)

    def test_real_footage_of_the_ordered_clothing_passes(self):
        frames = [str(p) for p in sorted((KIT / "driving").glob("*.jpg"))][:16]
        if not frames:
            self.skipTest("нет demo/kit/driving в дереве")
        got = self.s.semantic_match(frames)
        self.assertEqual(got["verdict"], "matches", got)


if __name__ == "__main__":
    unittest.main()


class WhoseClothesAreTheseAnyway(unittest.TestCase):
    """Определитель источника: заказ, фото личности или driving-видео.

    ЗАЧЕМ ОН ЕСТЬ. Пока заказ описывал то же, что видно на driving-кадре
    («спортивный топ в светлой комнате»), опыт был НЕИНТЕРПРЕТИРУЕМ: спортивная
    одежда на выходе могла быть и исполнением заказа, и утечкой из видео, и
    различить их нечем. Дефект не в коде, а в постановке опыта — и он важнее
    кодовых, потому что делает бессмысленным результат, а не строку.

    Стоит заказать ТРЕТЬЕ (красная майка вместо тёмно-синего топа и вместо
    розового платья) — и три исхода становятся различимы.
    """

    def setUp(self):
        from ball_reel import semantic

        self.s = semantic

    def test_no_frames_is_not_measurable_rather_than_a_source(self):
        got = self.s.attribution([], ordered=("a",), identity=("b",),
                                 driving=("c",))
        self.assertIsNone(got["source"])
        self.assertTrue(got["note"])

    def test_a_missing_source_description_is_refused_by_name(self):
        """Два источника вместо трёх не отличают утечку от исполнения заказа."""
        got = self.s.attribution(["x.png"], ordered=("a",), identity=(),
                                 driving=("c",))
        self.assertIsNone(got["source"])
        self.assertIn("фото личности", got["note"])

    def test_the_tie_threshold_is_borrowed_and_says_so(self):
        # Он измерен для ДРУГОЙ величины. Заимствование разумно, но выдавать
        # его за замер для этого применения нельзя — и модуль это признаёт.
        doc = " ".join((self.s.__doc__ or "").split())
        src = __import__("inspect").getsource(self.s)
        self.assertIn("ЗАИМСТВОВАН", src)
        self.assertIn("НЕПРОВЕРЕНО", src)

    def test_the_three_sources_are_named_constants_not_literals(self):
        # Имена источников попадают в отчёт и в разговор; разъехавшись между
        # модулем и вызывающим, они молча превратят «утечку» в «заказ».
        for name in (self.s.SOURCE_ORDER, self.s.SOURCE_IDENTITY,
                     self.s.SOURCE_DRIVING):
            self.assertTrue(name and isinstance(name, str))
        self.assertEqual(
            len({self.s.SOURCE_ORDER, self.s.SOURCE_IDENTITY,
                 self.s.SOURCE_DRIVING}), 3, "источники обязаны различаться")

    @unittest.skipUnless(__import__("os").environ.get("BALL_REEL_SEMANTIC_LIVE"),
                         "живой прогон под BALL_REEL_SEMANTIC_LIVE=1")
    def test_it_names_the_right_source_on_frames_with_a_known_answer(self):
        """ИЗМЕРЕНО: driving-кадры -> driving, фото -> фото. Отрывы 0.09 и 0.17."""
        ordered = ("in a bright red tank top and black shorts",)
        identity = ("in a frilly pink tulle dress", "in a tutu",
                    "in a party dress")
        driving = ("in a navy sports bra and leggings",
                   "in dark athletic sportswear")
        got = self.s.attribution(
            [str(p) for p in sorted((KIT / "driving").glob("*.jpg"))][::6],
            ordered=ordered, identity=identity, driving=driving)
        self.assertEqual(got["source"], self.s.SOURCE_DRIVING, got["note"])
        got = self.s.attribution([str(KIT / "face.jpg")], ordered=ordered,
                                 identity=identity, driving=driving)
        self.assertEqual(got["source"], self.s.SOURCE_IDENTITY, got["note"])
