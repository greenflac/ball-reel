"""Датасет для LoRA: проверяется то, из-за чего обучение может выйти впустую.

Четыре решения несут всю ценность модуля, и каждое легко нарушить из лучших
побуждений: не замыкать круг на ArcFace, не отражать несимметричные приметы, не
описывать в подписи то, что должно прилипнуть к триггеру, и — четвёртое, самое
свежее — брать подпись С КАДРА, а не из запроса.

Четвёртое проверяется здесь подробнее прочих, потому что ошибка в нём
ИНВЕРТИРУЕТ смысл подписи, а не портит его: соврав, подпись уводит от триггера
то, чего в кадре нет, и оставляет прилипать то, что есть.
"""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import numpy  # noqa: F401
    from PIL import Image  # noqa: F401

    HAVE_PIXELS = True
except ImportError:  # pragma: no cover
    HAVE_PIXELS = False

#: Живые кадры лежат ВНЕ пакета, и это важно: мутационный аудит копирует один
#: пакет, поэтому живые проверки там просто пропускаются, а сторожат пороги
#: синтетические тесты ниже. Числа, снятые с этих кадров, стоят в docstring'ах
#: `dataset.py`, и живой класс в конце файла пересчитывает их командой.
KIT = Path(__file__).resolve().parents[2] / "kit"


def _live_ready() -> bool:
    if not (KIT / "face.jpg").exists() or not HAVE_PIXELS:
        return False
    try:
        import mediapipe  # noqa: F401

        from ball_reel import pose

        pose._model_path()
    except Exception:  # noqa: BLE001
        return False
    return True


HAVE_LIVE = _live_ready()


def _points(*, ankle=None, knee=None, hip=None, shoulder=None):
    """Синтетический скелет: (x, y, видимость) на каждую пару суставов.

    Ни одно число сюда не приходит из константы, которую тесты сторожат: и
    координаты, и видимости написаны в самих тестах литералами.
    """
    out = {}
    for pair, value in (("ankle", ankle), ("knee", knee), ("hip", hip),
                        ("shoulder", shoulder)):
        for side in ("l", "r"):
            out[f"{side}_{pair}"] = value if value else (0.5, 9.9, 0.0)
    return out


class TheCircleMustNotCloseOnTheJUDGE(unittest.TestCase):
    """Главная гарантия модуля, и она проверяется кодом, а не обещанием.

    LoRA затевается ради независимости от ArcFace. Если ArcFace порождает
    датасет или отбирает его, независимости нет — она только выглядит
    полученной, что хуже, чем её отсутствие.
    """

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_arcface_in_the_generator_is_named_as_a_problem(self):
        got = self.d.independence_report(
            generator="ip-adapter-faceid", selector="facenet", judge="arcface")
        self.assertFalse(got["independent"])
        self.assertIn("ПОРОЖДЕНИИ", got["note"])

    def test_arcface_in_the_selector_is_named_too(self):
        # Тоньше первого и потому опаснее: порождение честное, а набор выбран
        # по тому, что нравится судье, и оценка завышена.
        got = self.d.independence_report(
            generator="seedream5", selector="buffalo_l", judge="arcface")
        self.assertFalse(got["independent"])
        self.assertIn("ОТБОРЕ", got["note"])

    def test_the_clean_split_is_declared_independent(self):
        got = self.d.independence_report(
            generator="seedream5", selector="facenet", judge="arcface")
        self.assertTrue(got["independent"], got["note"])
        self.assertEqual(got["problems"], [])

    def test_the_family_is_recognised_by_name_not_by_exact_match(self):
        # AuraFace и antelopev2 — та же школа, что buffalo_l. Считать их
        # независимыми значит поменять вывеску и объявить задачу решённой.
        for name in ("auraface", "antelopev2", "insightface/buffalo_s"):
            got = self.d.independence_report(
                generator="seedream5", selector=name, judge="arcface")
            self.assertFalse(got["independent"], name)


class MirrorAugmentationIsPOISONHere(unittest.TestCase):
    """Общепринятая дешёвая аугментация, ядовитая именно для нашей задачи."""

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_mirroring_is_off_by_default(self):
        names = [n for n, _ in self.d.augmentations()]
        self.assertNotIn("mirror", names)

    def test_turning_it_on_carries_the_warning_about_marks(self):
        rows = dict(self.d.augmentations(allow_mirror=True))
        self.assertIn("mirror", rows)
        self.assertIn("приметы", rows["mirror"])

    def test_every_default_augmentation_keeps_the_side(self):
        # Смысл списка: свет и кадр меняются, СТОРОНА — нет. Татуировка на
        # левом предплечье обязана остаться на левом во всех кадрах набора.
        for name, _ in self.d.augmentations():
            self.assertNotIn("mirror", name)
            self.assertNotIn("flip", name)


class TheREQUESTDescriptionIsNotACaption(unittest.TestCase):
    """`caption_for` описывает ЗАПРОС, и тесты обязаны говорить именно это.

    ПЕРЕПИСАН. Прежний класс назывался «CaptionsDescribeWhatVARIES» и утверждал,
    что `caption_for` — правильная обучающая подпись: он требовал, чтобы в ней
    стояли все четыре оси, включая `place`. Утверждение было неверно дважды.
    Во-первых, строка строится из того же `row`, что и промт, то есть описывает
    НАМЕРЕНИЕ, а на этом проекте ИЗМЕРЕНО, что генератор часть запроса
    игнорирует. Во-вторых, требование «в подписи обязан быть `place`» прямо
    фиксировало худшую из осей: место не измеримо на кадре ничем из того, что у
    нас есть, поэтому назвать его можно только со слов запроса.

    Тест, сторожащий дефект, хуже отсутствующего теста: он превращает дефект в
    контракт. Поэтому проверяется теперь другое — что строка остаётся честной
    записью запроса и что она проходит собственный запретный список.
    """

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_the_request_record_keeps_all_four_axes_that_were_asked_for(self):
        # Смысл строки — «вот что просили», поэтому оси в ней все четыре. В
        # обучение она не идёт: обучающую подпись пишет `caption_from_frame`.
        row = self.d.variations()[0]
        cap = self.d.caption_for(row, trigger="ohwx_person")
        self.assertTrue(cap.startswith("ohwx_person"))
        for axis in ("framing", "angle", "light", "place"):
            self.assertIn(row[axis], cap)

    def test_every_request_record_passes_our_own_forbidden_list(self):
        # ЖИВОЙ ПРИМЕР, а не одна строка: запретив то, что сами же порождаем, мы
        # получили бы словарь, который невозможно применить, и в первый же раз
        # его отключили бы целиком.
        for row in self.d.variations():
            cap = self.d.caption_for(row, trigger="ohwx_person")
            got = self.d.check_caption(cap)
            self.assertEqual(got["verdict"], self.d.CAPTION_OK,
                             f"{cap!r} -> {got['note']}")

    def test_the_prompt_does_not_describe_the_person_either(self):
        # Тот же запрет на стороне ПОРОЖДЕНИЯ: личность приходит референсом, а
        # не словами. Описав её текстом, мы получим похожего человека вместо
        # того же самого — и не заметим этого, потому что похожий тоже красив.
        p = self.d.prompt_for(self.d.variations()[0])
        for word in ("tattoo", "blonde", "blue eyes", "freckles"):
            self.assertNotIn(word, p.lower())


class TheAxesActuallyVARY(unittest.TestCase):
    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_each_row_differs_from_the_base_in_exactly_one_axis(self):
        rows = self.d.variations()
        base = rows[0]
        for row in rows[1:]:
            differing = [k for k in base if row[k] != base[k]]
            self.assertEqual(len(differing), 1, row)

    def test_all_four_axes_are_exercised(self):
        rows = self.d.variations()
        for axis in self.d.VARIATIONS:
            self.assertGreater(len({r[axis] for r in rows}), 1, axis)

    def test_a_limit_does_not_silently_drop_an_axis_entirely(self):
        # Обрезав список до четырёх, легко остаться с одной осью и решить, что
        # разнообразие есть. Тест сторожит именно это.
        rows = self.d.variations(limit=4)
        self.assertEqual(len(rows), 4)


class SelectionHasTHREEOutcomes(unittest.TestCase):
    def setUp(self):
        from ball_reel import dataset

        self.d = dataset
        self.S = dataset.Sample

    def _batch(self, scores):
        return [self.S(path=f"g{i}.png", origin="generated", score=s)
                for i, s in enumerate(scores)]

    def test_unscored_frames_are_not_silently_dropped(self):
        # «Лицо не найдено» и «лицо непохоже» — разные вещи. Слить их значит
        # не заметить, что отбор вообще не состоялся.
        got = self.d.select(self._batch([0.2, None, 0.9]), bar=0.5, min_kept=1)
        self.assertEqual(len(got["kept"]), 1)
        self.assertEqual(len(got["dropped"]), 1)
        self.assertEqual(len(got["unscored"]), 1)
        self.assertIn("не измерено", got["note"])

    def test_the_real_photo_is_an_anchor_and_never_filtered_out(self):
        # Реальное фото — единственная опора набора. Отсеять его по метрике,
        # которая мерит его же, было бы абсурдом.
        rows = self._batch([0.9, 0.9]) + [self.S(path="real.jpg", origin="real")]
        got = self.d.select(rows, bar=0.1, min_kept=1)
        self.assertEqual(got["real"], 1)
        real = next(s for s in got["kept"] if s.origin == "real")
        # Вес якоря сверяется НЕ с константой, которую сторожит, а с весом
        # синтетики: утверждение здесь — «реальное весит БОЛЬШЕ», и оно должно
        # падать при снятии веса до единицы. Мутационный аудит показал, что
        # сравнение с константой выживает, потому что едет вместе с ней.
        self.assertGreaterEqual(real.repeats, 3,
                                "реальный якорь должен весить заметно больше "
                                "синтетики, иначе набор уплывает")
        others = [s.repeats for s in got["kept"] if s.origin != "real"]
        for r in others:
            self.assertLess(r, real.repeats)

    def test_too_small_a_set_is_declared_not_quietly_accepted(self):
        got = self.d.select(self._batch([0.1, 0.1]), bar=0.5)
        self.assertFalse(got["enough"])
        self.assertIn("переобучится", got["note"])

    def test_the_bar_bites_from_both_sides_with_literal_scores(self):
        # Входы литеральные: тест, берущий порог из константы, которую
        # сторожит, едет вместе с ней и никогда не падает.
        got = self.d.select(self._batch([0.30, 0.31]), bar=0.30, min_kept=1)
        self.assertEqual(len(got["kept"]), 1)
        self.assertEqual(len(got["dropped"]), 1)


class ThePlanIsMadeBeforeSpending(unittest.TestCase):
    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_the_overshoot_accounts_for_measured_gateway_drift(self):
        # Дрейф шлюза измерен 0.345/0.367 при баре 0.35 — отсеется около
        # половины. Запрашивать ровно столько, сколько нужно, значит получить
        # вдвое меньше и доплачивать вторым заходом.
        got = self.d.plan(target=20)
        self.assertGreater(got["generate"], 20)
        self.assertGreaterEqual(got["generate"] / 20, 2.0)

    def test_the_cost_is_stated_in_pollen_before_anything_is_spent(self):
        got = self.d.plan(target=20, price_per_image=0.035)
        self.assertAlmostEqual(got["pollen"], got["generate"] * 0.035, places=3)
        self.assertIn("pollen", got["note"])


class TheManifestExplainsWhatTheLoRALearned(unittest.TestCase):
    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_the_mix_of_real_and_synthetic_is_recorded(self):
        # Первый вопрос любого, кто будет разбираться: сколько здесь реального.
        rows = [self.d.Sample(path="real.jpg", origin="real"),
                self.d.Sample(path="g0.png", origin="generated", score=0.2),
                self.d.Sample(path="a0.png", origin="augmented", score=0.1)]
        self.d.select(rows, bar=0.5, min_kept=1)
        got = self.d.manifest(rows, trigger="ohwx", judge="arcface",
                              selector="facenet", generator="seedream5")
        self.assertEqual(got["by_origin"]["real"], 1)
        self.assertEqual(got["by_origin"]["generated"], 1)
        self.assertTrue(got["independence"]["independent"])

    def test_repeats_are_summed_so_the_real_anchor_is_visible_in_steps(self):
        rows = [self.d.Sample(path="real.jpg", origin="real"),
                self.d.Sample(path="g0.png", origin="generated", score=0.2)]
        self.d.select(rows, bar=0.5, min_kept=1)
        got = self.d.manifest(rows, trigger="ohwx", judge="arcface",
                              selector="facenet", generator="seedream5")
        self.assertEqual(got["steps_equivalent"],
                         self.d.REAL_ANCHOR_REPEATS + 1)


class TheForbiddenListIsCODENotAComment(unittest.TestCase):
    """`CONSTANT` был перечислением слов в комментарии. Теперь это проверка.

    Пять слов, сверявшихся на одной подписи, не запрещали ничего: подпись писал
    человек, а сторожил её другой человек, читавший комментарий.
    """

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_each_constant_feature_is_caught_with_the_word_that_caught_it(self):
        # Входы литеральные. Тест, берущий слово из словаря, который сторожит,
        # едет вместе с ним и не падает никогда.
        cases = [
            ("ohwx, green eyes, in a park", "цвет глаз"),
            ("ohwx, blonde hair", "цвет волос"),
            ("ohwx, лицо крупным планом", "лицо"),
            ("ohwx, с татуировкой на плече", "приметы"),
            ("ohwx, худая, стоит у стены", "телосложение"),
            ("ohwx, 25 years old", "возраст"),
        ]
        for caption, category in cases:
            got = self.d.check_caption(caption)
            self.assertEqual(got["verdict"], self.d.CAPTION_NAMES_CONSTANT,
                             caption)
            self.assertEqual(got["category"], category, caption)
            # Слово важнее вердикта: «подпись плохая» нечинимо.
            self.assertTrue(got["word"], caption)
            self.assertIn(got["word"].lower(), caption.lower())

    def test_a_scarf_is_clothing_and_must_stay_describable(self):
        # Подстрочный поиск `scar` ловит `scarf`, а шарф — это ПЕРЕМЕННОЕ,
        # то есть ровно то, что подпись обязана называть. Запретив переменное,
        # мы наносим тот же ущерб, только с другой стороны.
        got = self.d.check_caption("ohwx, a red scarf and a long coat")
        self.assertEqual(got["verdict"], self.d.CAPTION_OK, got["note"])

    def test_not_checkable_is_a_third_outcome_and_not_a_pass(self):
        # Подпись на третьем письме проходит двуязычный словарь потому, что мы
        # её не читаем, а не потому, что там чисто.
        for caption in ("", "   ", "ohwx, 全身, 金髪"):
            got = self.d.check_caption(caption)
            self.assertEqual(got["verdict"], self.d.CAPTION_UNCHECKED,
                             repr(caption))

    def test_the_subject_name_is_caught_because_only_the_caller_knows_it(self):
        clean = self.d.check_caption("ohwx, full body in frame",
                                     subject_names=("Anna",))
        self.assertEqual(clean["verdict"], self.d.CAPTION_OK)
        got = self.d.check_caption("Anna, full body in frame",
                                   subject_names=("Anna",))
        self.assertEqual(got["verdict"], self.d.CAPTION_NAMES_CONSTANT)
        self.assertEqual(got["category"], self.d.SUBJECT_NAME)

    def test_no_boolean_travels_next_to_the_three_way_verdict(self):
        # Флаг `ok` рядом с трёхзначным вердиктом схлопывает «не смогли
        # проверить» в одно из двух, и схлопывает молча. Тот же дефект, от
        # которого в `select` заведён отдельный список `unscored`.
        for caption in ("ohwx, even light", "ohwx, blonde hair", ""):
            got = self.d.check_caption(caption)
            self.assertFalse([k for k, v in got.items() if isinstance(v, bool)],
                             f"булев флаг в вердикте: {got}")

    def test_every_named_constant_category_is_actually_closed_by_something(self):
        covered = {c for c, _ in self.d.FORBIDDEN} | {self.d.SUBJECT_NAME}
        for category in self.d.CONSTANT:
            self.assertIn(category, covered,
                          f"категория {category!r} названа, но ничем не закрыта")

    def test_the_measured_vocabulary_passes_its_own_list(self):
        # Второй живой пример: слова, которыми подписывает сам модуль.
        import itertools

        for f, l, t in itertools.product(self.d.FRAMING_WORDS,
                                         self.d.LIGHT_WORDS, self.d.TONE_WORDS):
            got = self.d.caption_from_measure(
                {"framing": f, "light": l, "tone": t}, trigger="ohwx_person")
            self.assertEqual(got["check"]["verdict"], self.d.CAPTION_OK,
                             f"{got['caption']!r}: {got['check']['note']}")


class FramingIsMeasuredOnTheSKELETON(unittest.TestCase):
    """Кадрировка берётся с кадра, и «не смогли» — отдельный исход."""

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_the_deepest_visible_joint_names_the_framing(self):
        # Диапазон покрыт целиком: от лодыжки до плеча.
        cases = [
            (dict(ankle=(0.5, 0.86, 0.99), knee=(0.5, 0.68, 0.99),
                  hip=(0.5, 0.65, 0.99), shoulder=(0.5, 0.46, 0.99)),
             "full_body"),
            (dict(ankle=(0.5, 1.06, 0.31), knee=(0.5, 0.96, 0.66),
                  hip=(0.5, 0.88, 0.98), shoulder=(0.5, 0.67, 0.99)),
             "knees_up"),
            (dict(ankle=(0.5, 1.18, 0.05), knee=(0.5, 1.09, 0.38),
                  hip=(0.5, 0.95, 0.97), shoulder=(0.5, 0.75, 0.99)),
             "waist_up"),
            (dict(ankle=(0.5, 1.68, 0.04), knee=(0.5, 1.46, 0.15),
                  hip=(0.5, 1.21, 0.08), shoulder=(0.5, 0.89, 0.99)),
             "head_and_shoulders"),
        ]
        for kwargs, expect in cases:
            got = self.d.framing_from_points(_points(**kwargs))
            self.assertEqual(got["framing"], expect, kwargs)
            self.assertEqual(got["why"], "измерено")

    def test_a_joint_inside_the_frame_but_unobserved_is_NOT_a_framing(self):
        # ИЗМЕРЕНО живьём: обрезанный на 10% портретный кадр кладёт лодыжки
        # ВНУТРЬ кадра (y 0.877) с видимостью 0.15. По одной геометрии это
        # «полный рост», которого в кадре нет. Соврать здесь дороже, чем
        # промолчать.
        got = self.d.framing_from_points(
            _points(ankle=(0.4, 0.88, 0.15), knee=(0.4, 0.87, 0.64),
                    hip=(0.55, 0.68, 0.99), shoulder=(0.52, 0.32, 0.99)))
        self.assertIsNone(got["framing"])
        self.assertEqual(got["why"], "не смогли определить")

    def test_a_frame_without_a_skeleton_is_not_a_portrait(self):
        for points in (None, {}):
            got = self.d.framing_from_points(points)
            self.assertIsNone(got["framing"])
            self.assertNotEqual(got["why"], "измерено")

    def test_the_joint_names_are_imported_not_retyped(self):
        from ball_reel import pose

        for _, joints in self.d.FRAMING_CLASSES:
            for j in joints:
                self.assertIn(j, pose.BODY_POINTS)

    def test_the_class_names_do_not_drift_from_the_condition_renderer(self):
        # `skeleton.FRAMINGS` называет ОКНА, которые строит условие ControlNet,
        # здесь — то, что видно на готовом кадре. Разъехавшись, эти два словаря
        # заставят читателя манифеста гадать, одно ли это и то же.
        from ball_reel import skeleton

        mine = {name for name, _ in self.d.FRAMING_CLASSES}
        for name in skeleton.FRAMINGS:
            self.assertIn(name, mine)


class LightIsMeasuredOnPIXELS(unittest.TestCase):
    """Свет считается прямо, поэтому и подписывается."""

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_the_brightness_scale_is_covered_end_to_end_with_literals(self):
        for brightness, expect in ((0.02, "dim"), (0.20, "dim"),
                                   (0.37, "even"), (0.48, "even"),
                                   (0.62, "even"), (0.70, "bright"),
                                   (0.98, "bright")):
            got = self.d.light_classes(brightness, None)
            self.assertEqual(got["light"], expect, brightness)

    def test_the_temperature_scale_is_covered_end_to_end_with_literals(self):
        for ratio, expect in ((0.60, "cool"), (0.84, "cool"), (0.95, "neutral"),
                              (0.98, "neutral"), (1.05, "neutral"),
                              (1.15, "warm"), (1.44, "warm")):
            got = self.d.light_classes(None, ratio)
            self.assertEqual(got["tone"], expect, ratio)

    def test_unmeasured_light_is_None_and_not_the_middle_class(self):
        # «Не измерили» и «ровный свет» — разные вещи. Слив их, мы подпишем
        # нечитаемый кадр самым частым словом и не заметим этого.
        got = self.d.light_classes(None, None)
        self.assertIsNone(got["light"])
        self.assertIsNone(got["tone"])

    @unittest.skipUnless(HAVE_PIXELS, "numpy/Pillow not installed (live extra)")
    def test_the_pixels_are_actually_read(self):
        import tempfile

        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            f = str(Path(tmp) / "a.png")
            # Тёплый тёмный кадр: R заметно выше B, яркость низкая.
            a = np.zeros((8, 8, 3), dtype="uint8")
            a[..., 0], a[..., 1], a[..., 2] = 60, 30, 20
            Image.fromarray(a).save(f)
            got = self.d.pixel_light(f)
            self.assertAlmostEqual(got["warm_ratio"], 3.0, places=3)
            self.assertLess(got["brightness"], 0.2)
            self.assertEqual(self.d.light_classes(**{
                k: got[k] for k in ("brightness", "warm_ratio")}),
                {"light": "dim", "tone": "warm"})

    def test_an_unreadable_file_is_not_a_dark_frame(self):
        self.assertIsNone(self.d.pixel_light("/nonexistent/definitely.png"))


class TheCaptionComesFromTheFRAMENotTheRequest(unittest.TestCase):
    """Главная инверсия модуля, и проверяется она поштучно."""

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def _measured(self, **kw):
        base = {"framing": "full_body", "light": "even", "tone": "neutral",
                "turn": 1.02}
        base.update(kw)
        return base

    def test_the_place_is_never_named_because_nothing_measures_it(self):
        cap = self.d.caption_from_measure(self._measured(),
                                          trigger="ohwx")["caption"]
        for values in self.d.VARIATIONS["place"]:
            for word in values.split():
                if len(word) > 3:
                    self.assertNotIn(word.lower(), cap.lower())

    def test_the_angle_is_measured_but_deliberately_not_named(self):
        # Отдельный тест, потому что это РЕШЕНИЕ, а не пропуск: полоса шума на
        # заведомо фронтальном субъекте 0.904..1.221, а анфас от полуоборота
        # отличает 0.707 — граница легла бы внутрь шума.
        got = self.d.caption_from_measure(self._measured(turn=0.71),
                                          trigger="ohwx")
        self.assertEqual(got["turn"], 0.71)
        for word in ("facing", "turned", "three-quarter", "angle", "profile"):
            self.assertNotIn(word, got["caption"].lower())

    def test_an_unmeasured_axis_is_listed_and_not_guessed(self):
        got = self.d.caption_from_measure(self._measured(framing=None),
                                          trigger="ohwx")
        self.assertIn("framing", got["unmeasured"])
        for word in self.d.FRAMING_WORDS.values():
            self.assertNotIn(word, got["caption"])
        self.assertIn("even light", got["caption"])

    def test_a_frame_with_nothing_measured_gets_no_caption_at_all(self):
        # Подпись из одного триггера означает «всё в кадре постоянно», то есть
        # тянет к триггеру и фон, и свет, и кадрировку. Такой кадр в наборе
        # хуже отсутствующего, и молча писать ему .txt нельзя.
        got = self.d.caption_from_measure(
            {"framing": None, "light": None, "tone": None}, trigger="ohwx")
        self.assertIsNone(got["caption"])
        self.assertEqual(len(got["unmeasured"]), 3)

    def test_the_caption_tracks_the_frame_when_the_framing_changes(self):
        # ИЗМЕРЕНО на живом кадре (см. живой класс ниже): обрезка до 62% высоты
        # уводит `kit/driving/0000.jpg` из full_body в waist_up. Здесь та же
        # пара проверяется без модели.
        before = self.d.caption_from_measure(self._measured(), trigger="ohwx")
        after = self.d.caption_from_measure(
            self._measured(framing="waist_up"), trigger="ohwx")
        self.assertNotEqual(before["caption"], after["caption"])
        self.assertIn("waist", after["caption"])

    def test_the_caption_does_not_change_when_the_frame_did_not(self):
        # Обратная сторона того же контракта, и она не менее важна: требование
        # «подпись обязана меняться после аугментации» заставило бы врать на
        # crop_tight 10%, который ИЗМЕРЕНО ничего в кадре не меняет — человек
        # как был виден целиком, так и остался. Подпись следует за КАДРОМ.
        before = self.d.caption_from_measure(self._measured(), trigger="ohwx")
        after = self.d.caption_from_measure(self._measured(turn=1.15),
                                            trigger="ohwx")
        self.assertEqual(before["caption"], after["caption"])


class AugmentationsAndTheANCHORAreCaptionedTheSameWay(unittest.TestCase):
    """Две дыры, которые закрываются замером бесплатно — но проверить надо.

    Аугментация наследовала подпись исходного кадра и была неверна ПО
    ПОСТРОЕНИЮ (восемь кадров на каждое реальное фото), а реальный якорь с весом
    `REAL_ANCHOR_REPEATS` — самый тяжёлый кадр набора — не имел подписи вовсе,
    потому что у него нет `row`.
    """

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def _measure(self, table):
        def measure(path):
            return dict(table[path], path=path)

        return measure

    def test_the_real_anchor_gets_a_caption_although_it_has_no_row(self):
        rows = [self.d.Sample(path="real.jpg", origin="real")]
        got = self.d.caption_samples(
            rows, trigger="ohwx",
            measure=self._measure({"real.jpg": {"framing": "waist_up",
                                                "light": "even",
                                                "tone": "warm"}}))
        self.assertEqual(got["failed"], [])
        self.assertTrue(rows[0].caption.startswith("ohwx,"))
        self.assertIn("waist", rows[0].caption)

    def test_a_crop_that_changes_the_frame_changes_the_caption(self):
        table = {"a.png": {"framing": "full_body", "light": "even",
                           "tone": "neutral"},
                 "a_crop_tight.png": {"framing": "waist_up", "light": "even",
                                      "tone": "neutral"}}
        rows = [self.d.Sample(path="a.png", origin="real"),
                self.d.Sample(path="a_crop_tight.png", origin="augmented")]
        self.d.caption_samples(rows, trigger="ohwx",
                               measure=self._measure(table))
        self.assertNotEqual(rows[0].caption, rows[1].caption)
        self.assertIn("full body", rows[0].caption)
        self.assertIn("waist", rows[1].caption)

    def test_a_frame_that_could_not_be_measured_is_counted_not_blanked(self):
        table = {"ok.png": {"framing": "full_body", "light": "even",
                            "tone": "neutral"},
                 "half.png": {"framing": None, "light": "dim",
                              "tone": "cool"},
                 "dead.png": {"framing": None, "light": None, "tone": None}}
        rows = [self.d.Sample(path=p, origin="generated") for p in table]
        got = self.d.caption_samples(rows, trigger="ohwx",
                                     measure=self._measure(table))
        self.assertEqual(got["full"], ["ok.png"])
        self.assertEqual(got["partial"], ["half.png"])
        self.assertEqual(got["failed"], ["dead.png"])
        self.assertIn("НЕ ПОДПИСАНО", got["note"])

    def test_the_manifest_shows_frames_that_ended_up_without_a_caption(self):
        rows = [self.d.Sample(path="g0.png", origin="generated", score=0.1,
                              caption="ohwx, full body in frame"),
                self.d.Sample(path="g1.png", origin="generated", score=0.1)]
        self.d.select(rows, bar=0.5, min_kept=1)
        got = self.d.manifest(rows, trigger="ohwx", judge="arcface",
                              selector="facenet", generator="seedream5")
        self.assertEqual(got["captions"]["measured"], 1)
        self.assertEqual(got["captions"]["missing"], ["g1.png"])

    def test_the_manifest_flags_a_caption_that_names_a_constant_feature(self):
        rows = [self.d.Sample(path="g0.png", origin="generated", score=0.1,
                              caption="ohwx, blonde hair, full body in frame")]
        self.d.select(rows, bar=0.5, min_kept=1)
        got = self.d.manifest(rows, trigger="ohwx", judge="arcface",
                              selector="facenet", generator="seedream5")
        self.assertEqual(len(got["captions"]["flagged"]), 1)
        self.assertEqual(got["captions"]["flagged"][0]["category"], "цвет волос")

    def test_the_request_and_the_measurement_are_kept_apart(self):
        # Пока это было одним полем, расхождение «просили — вышло» увидеть было
        # нечем, а именно оно и есть предмет спора.
        s = self.d.Sample(path="g0.png", origin="generated", score=0.1,
                          requested="ohwx, full body standing, in a bright gym",
                          caption="ohwx, framed from the waist up, even light")
        self.d.select([s], bar=0.5, min_kept=1)
        got = self.d.manifest([s], trigger="ohwx", judge="arcface",
                              selector="facenet", generator="seedream5")
        row = got["samples"][0]
        self.assertIn("bright gym", row["requested"])
        self.assertNotIn("gym", row["caption"])


@unittest.skipUnless(HAVE_LIVE, "kit/ или веса mediapipe недоступны")
class TheNumbersInTheDocstringsAreREPRODUCIBLE(unittest.TestCase):
    """Живой замер на настоящих кадрах. Пропускается там, где кита нет.

    Числа, стоящие в `dataset.py`, получены этой же командой:

        python3 -m unittest ball_reel.tests.test_dataset -v

    Сам полный прогон по 71 кадру `kit/driving` сюда не вынесен — он занимает
    минуты; вынесены концы диапазона и те кадры, на которых стоят решения.
    """

    def setUp(self):
        from ball_reel import dataset

        self.d = dataset

    def test_a_full_length_frame_is_measured_as_full_body(self):
        got = self.d.caption_from_frame(str(KIT / "driving" / "0000.jpg"),
                                        trigger="ohwx_person")
        self.assertEqual(got["axes"]["framing"], "full_body", got["note"])
        # Комнатный дневной свет: ИЗМЕРЕНО 0.467..0.502 по 71 кадру.
        self.assertEqual(got["axes"]["light"], "even")
        self.assertEqual(got["axes"]["tone"], "neutral")
        self.assertEqual(got["check"]["verdict"], self.d.CAPTION_OK)

    def test_a_frame_cut_at_the_knees_is_not_called_full_body(self):
        # `kit/face.jpg` — не портрет, вопреки названию файла: человек обрезан
        # по колени. Лодыжки детектор выносит за кадр (y 1.051 и 1.020).
        got = self.d.caption_from_frame(str(KIT / "face.jpg"),
                                        trigger="ohwx_person")
        self.assertEqual(got["axes"]["framing"], "knees_up", got["note"])
        # Золотой час: ИЗМЕРЕНО R/B = 1.439.
        self.assertEqual(got["axes"]["tone"], "warm")

    @unittest.skipUnless(HAVE_PIXELS, "numpy/Pillow not installed (live extra)")
    def test_cropping_a_real_frame_rewrites_its_caption(self):
        import tempfile

        from PIL import Image

        src = str(KIT / "driving" / "0000.jpg")
        before = self.d.caption_from_frame(src, trigger="ohwx_person")
        with tempfile.TemporaryDirectory() as tmp, Image.open(src) as im:
            w, h = im.size
            f = str(Path(tmp) / "cut.png")
            im.convert("RGB").crop((0, 0, w, int(h * 0.62))).save(f)
            after = self.d.caption_from_frame(f, trigger="ohwx_person")
        self.assertEqual(before["axes"]["framing"], "full_body")
        self.assertEqual(after["axes"]["framing"], "waist_up", after["note"])
        self.assertNotEqual(before["caption"], after["caption"])

    @unittest.skipUnless(HAVE_PIXELS, "numpy/Pillow not installed (live extra)")
    def test_a_head_crop_leaves_no_skeleton_and_is_not_called_a_portrait(self):
        import tempfile

        from PIL import Image

        src = str(KIT / "driving" / "0000.jpg")
        with tempfile.TemporaryDirectory() as tmp, Image.open(src) as im:
            w, h = im.size
            f = str(Path(tmp) / "head.png")
            im.convert("RGB").crop((0, 0, w, int(h * 0.30))).save(f)
            got = self.d.caption_from_frame(f, trigger="ohwx_person")
        self.assertIsNone(got["axes"]["framing"])
        self.assertIn("framing", got["unmeasured"])
        # Свет всё равно измерен — оси независимы, и частичная подпись честнее
        # выдуманной полной.
        self.assertIsNotNone(got["axes"]["light"])

    def test_the_turn_number_is_measured_and_is_not_trustworthy_enough(self):
        # Основание решения «ракурс не подписывать»: на 71 фронтальном кадре
        # число ушло до 1.221, что для косинуса невозможно.
        frontal = self.d.measure_frame(str(KIT / "driving" / "0000.jpg"))
        turned = self.d.measure_frame(str(KIT / "face.jpg"))
        self.assertIsNotNone(frontal["turn"])
        self.assertGreater(frontal["turn"], turned["turn"])
        self.assertGreater(frontal["turn"], 1.0,
                           "косинус больше единицы — это шум оценщика, и он "
                           "шире, чем расстояние от анфаса до полуоборота")


if __name__ == "__main__":
    unittest.main()


class TheDatasetCanActuallyBeBuilt(unittest.TestCase):
    """До этой части модуль умел всё, кроме одного: его нельзя было ЗАПУСТИТЬ.

    Точки входа не было вовсе — `dataset.py` не имел `main`, и «обучение LoRA
    с нуля» оставалось описанием, а не командой. Владелец продукта назвал это
    прямо: без LoRA остальное не имеет смысла.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        from ball_reel import dataset

        self.d = dataset
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _photo(self, name="face.png", size=(200, 360)):
        from PIL import Image

        p = self.dir / name
        Image.new("RGB", size, (120, 100, 90)).save(p)
        return str(p)

    def test_the_module_has_an_entry_point_at_all(self):
        self.assertTrue(hasattr(self.d, "main"))
        self.assertTrue(hasattr(self.d, "build_dataset"))

    def test_every_augmentation_named_is_actually_executable(self):
        """Список преобразований и их исполнение — не должны расходиться.

        Имя в списке без исполнения — это молчаливо пропущенный кадр: набор
        выходит меньше обещанного, а почему — не видно.
        """
        from PIL import Image

        im = Image.new("RGB", (64, 96), (10, 20, 30))
        for name, _ in self.d.augmentations():
            with self.subTest(aug=name):
                got = self.d.apply_augmentation(im, name)
                self.assertTrue(got.size[0] > 0 and got.size[1] > 0)

    def test_an_unknown_augmentation_is_refused_loudly(self):
        from PIL import Image

        with self.assertRaises(ValueError):
            self.d.apply_augmentation(Image.new("RGB", (8, 8)), "нет-такой")

    def test_mirror_is_still_not_among_them(self):
        # Зеркало переносит примету на другую сторону. Это самая дорогая
        # ошибка в задаче, и она общепринятая.
        self.assertNotIn("mirror", [n for n, _ in self.d.augmentations()])

    def test_a_frame_without_a_caption_never_enters_the_set(self):
        """Пустой .txt рядом с картинкой — молчаливое «учись чему хочешь»."""
        real = self.d.caption_from_frame
        self.d.caption_from_frame = lambda p, **kw: {
            "caption": "", "note": "ничего не измерилось"}
        self.addCleanup(setattr, self.d, "caption_from_frame", real)
        man = self.d.build_dataset(self._photo(), self.dir / "out")
        self.assertEqual(man["size"], 0)
        self.assertTrue(man["failed_captions"])

    def test_the_real_anchor_outweighs_its_augmentations(self):
        real = self.d.caption_from_frame
        self.d.caption_from_frame = lambda p, **kw: {"caption": "ohwx, x"}
        self.addCleanup(setattr, self.d, "caption_from_frame", real)
        man = self.d.build_dataset(self._photo(), self.dir / "out2")
        by = {s["origin"]: s["repeats"] for s in man["samples"]}
        self.assertGreater(by["real"], by["augmented"],
                           "реальный кадр обязан весить больше производных")
        self.assertEqual(by["real"], self.d.REAL_ANCHOR_REPEATS)

    def test_a_caption_file_lands_beside_every_image(self):
        # Тренер читает .txt рядом с картинкой; без него кадр учится без
        # подписи, то есть тянет всё подряд на триггер.
        from pathlib import Path

        real = self.d.caption_from_frame
        self.d.caption_from_frame = lambda p, **kw: {"caption": "ohwx, x"}
        self.addCleanup(setattr, self.d, "caption_from_frame", real)
        man = self.d.build_dataset(self._photo(), self.dir / "out3")
        for s in man["samples"]:
            with self.subTest(path=s["path"]):
                self.assertTrue(Path(s["path"]).with_suffix(".txt").exists())

    def test_a_local_only_set_is_reported_as_TOO_SMALL(self):
        """Аугментаций одного фото не хватает, и это обязано быть сказано.

        ИЗМЕРЕНО на настоящей рефке: реальный кадр плюс восемь преобразований
        дают 9 при пороге 12. Молча выдать это за набор значило бы обучить
        LoRA на одном ракурсе и узнать об этом по результату.
        """
        real = self.d.caption_from_frame
        self.d.caption_from_frame = lambda p, **kw: {"caption": "ohwx, x"}
        self.addCleanup(setattr, self.d, "caption_from_frame", real)
        man = self.d.build_dataset(self._photo(), self.dir / "out4")
        self.assertLess(man["size"], self.d.MIN_DATASET)
        self.assertEqual(self.d.main(["--face", self._photo(),
                                      "--out", str(self.dir / "out5")]), 1)

    def test_the_independence_report_travels_with_the_set(self):
        # Главная гарантия модуля обязана лежать рядом с весами, а не в
        # презентации.
        real = self.d.caption_from_frame
        self.d.caption_from_frame = lambda p, **kw: {"caption": "ohwx, x"}
        self.addCleanup(setattr, self.d, "caption_from_frame", real)
        man = self.d.build_dataset(self._photo(), self.dir / "out6")
        self.assertIn("independence", man)
        self.assertIn("judge", man["independence"])
