"""Подражание драйвингу: замер и подгонка.

Всё считается по пикселям — значит проверяется без карты, без сети и без
весов, на кадрах, нарисованных прямо здесь.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ball_reel import style


def _frame(path, *, size=256, noise=0.0, contrast=1.0, base=128, seed=0):
    """Кадр с ЗАДАННЫМИ свойствами: иначе проверять нечего, кроме тавтологии."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    # Крупные полосы дают контраст, не давая высоких частот: так контраст и
    # детальность можно двигать НЕЗАВИСИМО, а без этого тест не различит,
    # какая из двух правок сработала.
    a = base + contrast * 60.0 * np.sin(x / 24.0) * np.cos(y / 31.0)
    a = np.repeat(a[:, :, None], 3, axis=2)
    if noise:
        a = a + rng.normal(0, noise, a.shape)
    Image.fromarray(np.clip(a, 0, 255).astype("uint8")).save(path)
    return str(path)


class Measure(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_an_unreadable_file_is_none_and_not_a_crash(self):
        (self.tmp / "нет.png").write_text("не картинка", encoding="utf-8")
        self.assertIsNone(style.measure(self.tmp / "нет.png"))
        self.assertIsNone(style.measure(self.tmp / "нету-совсем.png"))

    def test_more_noise_reads_as_more_noise(self):
        a = style.measure(_frame(self.tmp / "a.png", noise=0.0))
        b = style.measure(_frame(self.tmp / "b.png", noise=12.0))
        self.assertLess(a["noise"], b["noise"])

    def test_more_contrast_reads_as_more_contrast(self):
        a = style.measure(_frame(self.tmp / "a.png", contrast=0.3))
        b = style.measure(_frame(self.tmp / "b.png", contrast=1.0))
        self.assertLess(a["contrast"], b["contrast"])

    def test_the_measure_is_taken_at_one_side_so_sizes_compare(self):
        """Детальность и зерно — величины НА ПИКСЕЛЬ.

        Сравнив 1024 с 736 без приведения, мы сравнили бы разрешения. Кадр
        драйвинга и кадр генератора именно такой парой и приходят.
        """
        a = style.measure(_frame(self.tmp / "a.png", size=256, seed=1))
        b = style.measure(_frame(self.tmp / "b.png", size=1024, seed=1))
        # Одна и та же картина в двух размерах обязана дать близкий контраст.
        self.assertAlmostEqual(a["contrast"], b["contrast"], delta=0.03)


class Profile(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_no_readable_frames_is_a_refusal_with_a_reason(self):
        got = style.profile([str(self.tmp / "нет.png")])
        self.assertFalse(got["ok"])
        self.assertIn("подражать нечему", got["note"])

    def test_one_wild_frame_does_not_drag_the_cloud(self):
        """Медиана, а не среднее: пересвеченный кадр видео — обычное дело."""
        frames = [_frame(self.tmp / f"{i}.png", base=120, seed=i)
                  for i in range(5)]
        frames.append(_frame(self.tmp / "wild.png", base=250, seed=9))
        prof = style.profile(frames)
        self.assertTrue(prof["ok"])
        self.assertLess(prof["axes"]["brightness"]["p50"], 0.6,
                        "один пересвеченный кадр утащил облако — это среднее, "
                        "а не медиана")


class Distance(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_a_frame_from_the_cloud_is_close_to_it(self):
        frames = [_frame(self.tmp / f"{i}.png", seed=i) for i in range(4)]
        prof = style.profile(frames)
        d = style.distance(style.measure(frames[0]), prof)
        self.assertTrue(d["ok"], d["axes"])

    def test_the_ratio_is_symmetric_so_too_soft_is_a_miss_too(self):
        """Промах ВНИЗ — такой же промах.

        Первая редакция подгонки уводила детальность втрое НИЖЕ цели и
        считала это попаданием, потому что мерила разницу, а не отношение.
        """
        prof = {"ok": True, "axes": {"detail": {"p50": 1.0}}}
        low = style.distance({"detail": 0.2}, prof)["axes"]["detail"]
        high = style.distance({"detail": 5.0}, prof)["axes"]["detail"]
        self.assertAlmostEqual(low["ratio"], 5.0, places=3)
        self.assertAlmostEqual(high["ratio"], 5.0, places=3)
        self.assertFalse(low["ok"])
        self.assertFalse(high["ok"])

    def test_the_tolerance_sits_inside_the_measured_gap(self):
        """Допуск обязан ЛЕЖАТЬ В РАЗРЫВЕ между облаками, а не рядом с ним.

        Замерено: детальность драйвинга не выше 1.387, генератора не ниже
        5.870 — разрыв в 4.2 раза; зерно 8.628 против 20.884 — в 2.4 раза.
        Допуск шире любого из них перестал бы различать облака, то есть стал
        бы украшением.
        """
        self.assertLess(style.TOLERANCE, 5.870 / 1.387)
        self.assertLess(style.TOLERANCE, 20.884 / 8.628)
        self.assertGreater(style.TOLERANCE, 1.0)


class Match(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _cloud(self):
        return style.profile([_frame(self.tmp / f"d{i}.png", contrast=0.35,
                                     seed=i) for i in range(4)])

    def test_no_cloud_is_a_refusal_and_not_a_silent_copy(self):
        got = style.match(_frame(self.tmp / "f.png"),
                          {"ok": False, "note": "нечего"}, self.tmp / "o.png")
        self.assertFalse(got["ok"])
        self.assertFalse((self.tmp / "o.png").exists(),
                         "кадр записан вопреки отказу — отчёт скажет «готово»")

    def test_contrast_lands_on_the_cloud(self):
        prof = self._cloud()
        got = style.match(_frame(self.tmp / "f.png", contrast=1.0),
                          prof, self.tmp / "o.png")
        self.assertTrue(got["ok"])
        want = prof["axes"]["contrast"]["p50"]
        self.assertAlmostEqual(got["after"]["contrast"], want, delta=0.02,
                               msg=f"было {got['before']['contrast']:.3f}")

    def test_the_linear_pass_runs_before_the_blur_not_after(self):
        """Порядок операций, обоснованный арифметикой, а не удобством.

        Растяжение контраста с коэффициентом g умножает отклик лапласиана на
        g, то есть его ДИСПЕРСИЮ на g². Сгладив первым, мы попадали в цель по
        детальности, а следующая линейная правка тут же давила её ещё в g²
        раз: измерено 1.179 -> 0.433 при цели 1.153.
        """
        import inspect

        src = inspect.getsource(style.match)
        first_linear = src.index("_linear")
        first_blur = src.index("GaussianBlur")
        self.assertLess(first_linear, first_blur,
                        "сглаживание идёт раньше линейной правки — детальность "
                        "уедет в g² раз ПОСЛЕ попадания в цель")

    def test_the_blur_search_takes_the_nearest_radius_not_the_first_under(self):
        prof = self._cloud()
        got = style.match(_frame(self.tmp / "f.png", noise=25.0, seed=3),
                          prof, self.tmp / "o.png")
        want = prof["axes"]["detail"]["p50"]
        have = got["after"]["detail"]
        ratio = have / want if have >= want else want / max(have, 1e-9)
        self.assertLess(ratio, 3.0,
                        f"детальность {have:.3f} против цели {want:.3f}: "
                        f"радиус взят не ближайший")

    def test_the_report_carries_numbers_before_and_after_not_a_verdict(self):
        got = style.match(_frame(self.tmp / "f.png"), self._cloud(),
                          self.tmp / "o.png")
        for key in ("before", "after", "distance_before", "distance_after"):
            self.assertIn(key, got)
        for axis in ("contrast", "detail", "noise"):
            self.assertIn(axis, got["before"])
            self.assertIn(axis, got["after"])


class ItIsWiredIntoTheRun(unittest.TestCase):
    """Модуль, написанный и не подключённый, ОПАСНЕЕ ненаписанного."""

    def test_the_run_has_a_flag_for_it(self):
        from ball_reel.run_local import build_parser

        args = build_parser().parse_args(
            ["--face", "f.jpg", "--prompt", "p", "--style"])
        self.assertTrue(args.style)

    def test_it_is_off_unless_asked(self):
        # Гаусс — это расфокус, и на глаз он читается именно так. Пока полосы
        # не разделены (шумодав + зерно), включать это по умолчанию нельзя.
        from ball_reel.run_local import build_parser

        args = build_parser().parse_args(["--face", "f.jpg", "--prompt", "p"])
        self.assertFalse(args.style)

    def test_it_runs_after_the_gate_and_leaves_judged_frames_alone(self):
        """Подгонка переписывает ту полосу частот, которую меряет ArcFace.

        Посудив подогнанный кадр, мы измерили бы подгонку, а не генератор, —
        то же правило, по которому после гейта стоит апскейл.
        """
        import inspect

        from ball_reel import run_local

        src = inspect.getsource(run_local._animatediff_once)
        self.assertLess(src.index("clip_gate"), src.index("ступень 3.5"),
                        "подгонка оказалась ДО гейта")
        self.assertIn('out / "styled"', src,
                      "подогнанные кадры обязаны ложиться рядом, а не поверх "
                      "судимых: иначе печать апскейла не сойдётся")
