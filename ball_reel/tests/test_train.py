"""Обучение, в котором ничего не обучается, идёт неотличимо от настоящего.

Цикл на карте здесь не проверяется: GPU в среде разработки нет, и всё, что
касается весов, помечено НЕПРОВЕРЕНО в шапке `train`. Проверяется то, что
можно проверить без карты, — и это ровно те места, где отказ МОЛЧАЛИВ:

* набор без подписи не должен доехать до цикла (тренер с пустой подписью
  учится успешно и тянет на триггер фон);
* число шагов должно учитывать повторы, иначе расписание короче заявленного;
* план и набор дают два разных числа шагов, и в отчёт обязано попасть, какое
  из них ограничило;
* повторы должны быть частотой по проходу, а не серией подряд;
* нулевая доля обучаемых параметров обязана быть ОТКАЗОМ, а не примечанием.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ball_reel import train


def _dataset(tmp: Path, *, n: int = 3, caption: str = "ohwx_person, portrait",
             repeats: int = 1, drop_caption: bool = False) -> Path:
    root = tmp / "ds"
    root.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(n):
        img = root / f"{i:04d}.png"
        img.write_bytes(b"not-a-real-png")  # содержимое не читается этими тестами
        samples.append({"path": str(img),
                        "caption": "" if drop_caption and i == 0 else caption,
                        "repeats": repeats, "origin": "real"})
    (root / "manifest.json").write_text(
        json.dumps({"samples": samples}), encoding="utf-8")
    return root


class Cfg:
    """Минимальный двойник `lora.TrainConfig` — только читаемые здесь поля."""

    def __init__(self, **kw):
        self.rank = 8
        self.alpha = 8
        self.resolution = 512
        self.batch_size = 1
        self.optimizer = "AdamW8bit"
        self.learning_rate = 1e-4
        self.max_train_steps = 1200
        self.gradient_checkpointing = True
        self.seed = 0
        self.__dict__.update(kw)


class LoadPairs(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_reads_path_caption_repeats(self):
        root = _dataset(self.tmp, n=3, repeats=5)
        pairs = train.load_pairs(root)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(r == 5 for _, _, r in pairs))
        self.assertTrue(all(c for _, c, _ in pairs))

    def test_missing_caption_is_refusal_not_empty_string(self):
        """Самая тихая ловушка: пустая подпись тянет на триггер фон и свет."""
        root = _dataset(self.tmp, n=3, drop_caption=True)
        with self.assertRaises(ValueError) as ctx:
            train.load_pairs(root)
        self.assertIn("подпис", str(ctx.exception))

    def test_missing_image_is_refusal(self):
        root = _dataset(self.tmp, n=2)
        (root / "0000.png").unlink()
        with self.assertRaises(ValueError):
            train.load_pairs(root)

    def test_missing_manifest_names_the_command(self):
        """Отказ обязан называть лечение: собрать набор нечем угадать."""
        with self.assertRaises(FileNotFoundError) as ctx:
            train.load_pairs(self.tmp / "нет-такого")
        self.assertIn("ball_reel.dataset", str(ctx.exception))

    def test_a_dataset_built_elsewhere_still_loads(self):
        """Манифест пишет абсолютные пути машины-сборщика.

        Набор собирается через шлюз на одной машине, а учат его на карте — на
        другой. Записанного `/tmp/.../ds/img/real.png` там нет, и обучение
        отказывало на ПОЛНОМ наборе со словами «набор неполон»: диагноз
        указывал не туда, а лечения у него не было вовсе.
        """
        root = _dataset(self.tmp, n=3)
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for s in man["samples"]:
            s["path"] = "/машина/которой/нет/ds/img/" + Path(s["path"]).name
        (root / "manifest.json").write_text(
            json.dumps(man, ensure_ascii=False), encoding="utf-8")
        got = train.load_pairs(root)
        self.assertEqual(len(got), 3)
        for path, _, _ in got:
            self.assertTrue(Path(path).exists(), path)

    def test_a_windows_separator_in_the_manifest_is_understood(self):
        # Сборщик мог быть на Windows, а карта — под Linux: `PurePath` разделитель
        # чужой платформы не разбирает, и имя файла не отделилось бы вовсе.
        root = _dataset(self.tmp, n=2)
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for s in man["samples"]:
            s["path"] = "C:\\build\\ds\\img\\" + Path(s["path"]).name
        (root / "manifest.json").write_text(
            json.dumps(man, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(len(train.load_pairs(root)), 2)

    def test_a_genuinely_missing_frame_is_still_a_refusal(self):
        # Починка обязана чинить перенос, а не глушить отказ: кадра нет нигде.
        root = _dataset(self.tmp, n=3)
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        Path(man["samples"][0]["path"]).unlink()
        with self.assertRaises(ValueError):
            train.load_pairs(root)

    def test_caption_falls_back_to_txt_beside_image(self):
        root = _dataset(self.tmp, n=1, caption="")
        (root / "0000.txt").write_text("ohwx_person, side view", encoding="utf-8")
        pairs = train.load_pairs(root)
        self.assertEqual(pairs[0][1], "ohwx_person, side view")


class Steps(unittest.TestCase):

    def test_repeats_are_counted_not_ignored(self):
        """Считать по числу ФАЙЛОВ значит оборвать обучение впятеро раньше."""
        pairs = [("a", "c", 5), ("b", "c", 1)]
        self.assertEqual(train.steps_for(pairs, epochs=1), 6)
        self.assertEqual(train.steps_for(pairs, epochs=10), 60)

    def test_never_zero(self):
        self.assertGreaterEqual(train.steps_for([("a", "c", 1)], epochs=0), 1)

    def test_batch_divides(self):
        pairs = [("a", "c", 4)]
        self.assertEqual(train.steps_for(pairs, epochs=1, batch_size=2), 2)


class Budget(unittest.TestCase):
    """Два числа шагов приходят с разных сторон; отчёт обязан назвать одно."""

    def test_dataset_bounds_when_small(self):
        pairs = [("a", "c", 5)] * 2          # 10 кадров с повторами
        got = train.budget(pairs, Cfg(max_train_steps=1200), epochs=2)
        self.assertEqual(got["steps"], 20)
        self.assertEqual(got["bound"], "набор")

    def test_plan_caps_when_dataset_would_overrun(self):
        pairs = [("a", "c", 5)] * 40         # 200 кадров с повторами
        got = train.budget(pairs, Cfg(max_train_steps=100), epochs=10)
        self.assertEqual(got["steps"], 100)
        self.assertEqual(got["bound"], "план")
        self.assertIn("2000", got["note"])   # сколько дал бы набор — названо

    def test_absent_plan_does_not_cap(self):
        pairs = [("a", "c", 1)] * 7
        got = train.budget(pairs, Cfg(max_train_steps=0), epochs=3)
        self.assertEqual(got["steps"], 21)
        self.assertEqual(got["bound"], "набор")


class Order(unittest.TestCase):

    def test_repeats_become_frequency_not_a_run(self):
        """Пять шагов Adam подряд по одной картинке — всплеск, а не вес."""
        pairs = [("real.png", "c", 5)] + [(f"g{i}.png", "c", 1)
                                          for i in range(15)]
        flat = train.order(pairs, seed=0)
        self.assertEqual(len(flat), 20)
        runs = [p for p, _ in flat]
        longest, cur = 1, 1
        for a, b in zip(runs, runs[1:]):
            cur = cur + 1 if a == b else 1
            longest = max(longest, cur)
        self.assertLess(longest, 5, "якорь идёт серией подряд, а не вразбивку")

    def test_deterministic_for_a_seed(self):
        pairs = [(f"{i}.png", "c", 2) for i in range(9)]
        self.assertEqual(train.order(pairs, seed=7),
                         train.order(pairs, seed=7))

    def test_seed_changes_the_order(self):
        pairs = [(f"{i}.png", "c", 2) for i in range(9)]
        self.assertNotEqual(train.order(pairs, seed=1),
                            train.order(pairs, seed=2))

    def test_every_repeat_survives(self):
        pairs = [("a.png", "c", 5), ("b.png", "c", 1)]
        flat = train.order(pairs, seed=0)
        self.assertEqual(sum(1 for p, _ in flat if p == "a.png"), 5)
        self.assertEqual(sum(1 for p, _ in flat if p == "b.png"), 1)


class _P:
    def __init__(self, n, grad):
        self.n, self.requires_grad = n, grad

    def numel(self):
        return self.n


class _M:
    def __init__(self, ps):
        self._ps = ps

    def parameters(self):
        return iter(self._ps)


class TrainableReport(unittest.TestCase):

    def test_zero_trainable_is_refusal(self):
        """Цикл отработает, лосс пошумит вниз, веса не изменятся."""
        got = train.trainable_report(_M([_P(860_000_000, False)]))
        self.assertFalse(got["ok"])
        self.assertEqual(got["trainable"], 0)

    def test_wrong_modules_are_refusal(self):
        """Адаптер на 100 параметрах при 860 млн — сел не на те модули."""
        got = train.trainable_report(
            _M([_P(860_000_000, False), _P(100, True)]))
        self.assertFalse(got["ok"])
        self.assertIn("to_q", got["note"])

    def test_rank8_attention_lora_passes(self):
        """Опорный факт порога: около 0.1% при LoRA на проекциях внимания."""
        got = train.trainable_report(
            _M([_P(860_000_000, False), _P(1_000_000, True)]))
        self.assertTrue(got["ok"], got["note"])

    def test_threshold_is_below_the_reference_share_and_above_zero(self):
        """Пол ловит НОЛЬ и «почти ноль», а не отличает 0.1% от 0.2%."""
        self.assertGreater(train.MIN_TRAINABLE_SHARE, 0.0)
        self.assertLess(train.MIN_TRAINABLE_SHARE, 1_000_000 / 860_000_000)

    def test_no_parameters_at_all_is_refusal(self):
        self.assertFalse(train.trainable_report(_M([]))["ok"])


class Preflight(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_bad_dataset_stops_before_anything_else(self):
        """Дешёвое раньше дорогого: набор читается за миллисекунды."""
        got = train.preflight(self.tmp / "нет", Cfg())
        self.assertFalse(got["ok"])
        self.assertEqual([c["name"] for c in got["checks"]], ["набор"])

    def test_small_dataset_fails_and_names_the_cure(self):
        from ball_reel.dataset import MIN_DATASET

        root = _dataset(self.tmp, n=max(1, MIN_DATASET - 1))
        got = train.preflight(root, Cfg())
        self.assertFalse(got["ok"])
        size = next(c for c in got["checks"] if c["name"] == "размер")
        self.assertFalse(size["ok"])
        self.assertTrue(any("--generated" in n for n in got["notes"]))

    def test_low_resolution_fails(self):
        from ball_reel.dataset import MIN_DATASET

        root = _dataset(self.tmp, n=MIN_DATASET)
        got = train.preflight(root, Cfg(resolution=64))
        self.assertFalse(got["ok"])
        res = next(c for c in got["checks"] if c["name"] == "разрешение")
        self.assertFalse(res["ok"])

    def test_full_dataset_passes_and_returns_pairs(self):
        from ball_reel.dataset import MIN_DATASET

        root = _dataset(self.tmp, n=MIN_DATASET)
        got = train.preflight(root, Cfg())
        self.assertEqual(len(got["pairs"]), MIN_DATASET)
        for name in ("peft", "diffusers", "torch"):
            c = next(x for x in got["checks"] if x["name"] == name)
            self.assertTrue(c["ok"], f"{name}: {c['detail']}")
        self.assertTrue(got["ok"], got["checks"])

    def test_resolution_floor_leaves_room_for_the_planned_512(self):
        from ball_reel.lora import config

        self.assertLessEqual(train.MIN_RESOLUTION, config(6.0).resolution)


class ShippedDataset(unittest.TestCase):
    """Набор для LoRA лежит В РЕПОЗИТОРИИ, а не только на машине сборщика.

    ЗАЧЕМ. Набор собирается через шлюз, шлюз требует ключа, и ключ есть не у
    всех и не всегда. Пока набор жил только в каталоге сборщика, «обучить
    LoRA» означало «сначала добудь ключ» — то есть на демо-дне шаг мог не
    состояться по причине, не имеющей отношения ни к коду, ни к карте.

    Собран живьём 2026-08-14 на `nanobanana-2` с промтом, просящим ЖАНР
    любительской съёмки: 24 порождённых -> взято 22 (92%) при баре FaceNet
    0.30, дистанции 0.14..0.24, плюс 1 реальный кадр и 8 аугментаций = 31,
    35 с повторами.

    Прежняя редакция набора (`nanobanana`, промт про «качественное фото») дала
    70% отбора при 23 кадрах. Заменена не по вкусу: выше и доля отбора, и
    сходство лица.
    """

    ROOT = Path(__file__).resolve().parents[2] / "demo" / "lora_dataset"

    def test_it_is_on_disk_and_loads(self):
        self.assertTrue(self.ROOT.is_dir(), f"нет {self.ROOT}")
        pairs = train.load_pairs(self.ROOT)
        self.assertEqual(len(pairs), 31)
        for path, caption, _ in pairs:
            self.assertTrue(Path(path).exists(), path)
            self.assertIn("ohwx_person", caption)

    def test_the_step_count_matches_the_number_in_the_runbook(self):
        # Число из LORA_RUNBOOK обязано пересчитываться командой, а не
        # запоминаться: разошедшееся с кодом число хуже отсутствующего.
        pairs = train.load_pairs(self.ROOT)
        self.assertEqual(train.steps_for(pairs, epochs=10), 350)

    def test_the_manifest_paths_are_relative_and_survive_a_move(self):
        man = json.loads((self.ROOT / "manifest.json").read_text(
            encoding="utf-8"))
        for s in man["samples"]:
            with self.subTest(path=s["path"]):
                self.assertFalse(Path(s["path"]).is_absolute(),
                                 "абсолютный путь в отгружаемом наборе: у "
                                 "склонировавшего такого каталога нет")

    def test_it_is_in_the_git_index_and_not_only_on_the_authors_disk(self):
        """Тот же дефект, что однажды съел половину `demo/kit_waist`.

        На диске автора набор полон и обучение идёт; у склонировавшего его нет,
        и проверяющий видит ссылку в рунбуке на пустоту.
        """
        import subprocess

        root = Path(__file__).resolve().parents[2]
        try:
            r = subprocess.run(["git", "ls-files", "demo/lora_dataset"],
                               capture_output=True, text=True, cwd=root,
                               timeout=30)
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git недоступен — проверка на диске выше")
        if r.returncode != 0:
            self.skipTest("не рабочее дерево git — проверка на диске выше")
        files = set(r.stdout.split())
        self.assertIn("demo/lora_dataset/manifest.json", files)
        self.assertEqual(
            sum(1 for f in files if f.endswith(".png")), 31,
            "в индексе не все 31 кадр набора")
        self.assertEqual(
            sum(1 for f in files if f.endswith(".txt")), 31,
            "в индексе не все 31 подпись — кадр без подписи тянет на триггер "
            "фон, свет и одежду")


class Cli(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_dry_run_never_loads_weights(self):
        from ball_reel.dataset import MIN_DATASET

        root = _dataset(self.tmp, n=MIN_DATASET)
        self.assertEqual(train.main(["--dataset", str(root), "--dry-run"]), 0)

    def test_bad_dataset_exits_nonzero(self):
        self.assertEqual(
            train.main(["--dataset", str(self.tmp / "нет"), "--dry-run"]), 1)

    def test_the_base_reaches_the_trainer_and_is_not_lost_in_the_parser(self):
        """`train()` умел принимать base, а CLI его НЕ ПЕРЕДАВАЛ.

        Цена молчания измерена в тот же вечер на чужой LoRA: FaceID-LoRA
        обучена под ванильную SD1.5, и на epiCRealism кадр разваливался в
        радужные потёки, а с выключенным каналом личности становился резким и
        фактурным. Своя LoRA, обученная не на той базе, — тот же дефект, только
        сделанный своими руками и после часа обучения.
        """
        from ball_reel.dataset import MIN_DATASET

        root = _dataset(self.tmp, n=MIN_DATASET)
        seen = {}

        def fake(dataset_dir, out_dir, cfg, *, base="", epochs=10, **kw):
            seen["base"] = base
            return {"ok": True, "note": "заглушка"}

        real, train.train = train.train, fake
        try:
            train.main(["--dataset", str(root), "--base", "чужая/база"])
        finally:
            train.train = real
        self.assertEqual(seen.get("base"), "чужая/база",
                         "--base не доехал до обучения: LoRA сядет на "
                         "умолчание, а рисовать будем на другой базе")

    def test_the_plan_names_the_base_out_loud(self):
        """Несовпадение баз обязано ловиться глазом ДО обучения, а не после.

        Печать в --dry-run стоит секунду; обнаружение того же по испорченному
        кадру — час обучения плюс прогон.
        """
        import contextlib
        import io

        from ball_reel.dataset import MIN_DATASET

        root = _dataset(self.tmp, n=MIN_DATASET)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train.main(["--dataset", str(root), "--dry-run",
                        "--base", "чужая/база"])
        self.assertIn("чужая/база", buf.getvalue())

    def test_the_default_base_is_named_too_and_not_left_blank(self):
        # Пустая строка в отчёте читается как «база не выбрана», хотя она
        # выбрана — умолчанием. Молчащее умолчание и есть то, на чём горят.
        import contextlib
        import io

        from ball_reel.animate import BASE_MODEL
        from ball_reel.dataset import MIN_DATASET

        root = _dataset(self.tmp, n=MIN_DATASET)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train.main(["--dataset", str(root), "--dry-run"])
        self.assertIn(BASE_MODEL, buf.getvalue())


class Stage(unittest.TestCase):
    """Ступень обучения внутри прогона: всё, что можно проверить без карты."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _args(self, **kw):
        from types import SimpleNamespace

        base = {"train_lora": "", "train_epochs": 10, "vram": 6.0, "lora": ""}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_bad_dataset_stops_the_run_before_the_card_is_touched(self):
        """Отказ обязан прийти до весов: обучение стоит десятки минут."""
        from ball_reel.run_local import train_subject_lora

        got = train_subject_lora(
            self._args(train_lora=str(self.tmp / "нет")), self.tmp)
        self.assertFalse(got["ok"])
        self.assertIn("предполёт обучения", got["note"])

    def test_the_run_trains_on_the_same_base_it_renders_on(self):
        """Внутри ОДНОГО прогона базы разойтись не имеют права.

        `run_local` звал обучение без `base`, то есть учил на ванильной SD1.5 и
        рисовал на `--base`. Адаптер выходил чужим для той базы, на которой его
        применяют, — тот самый механизм, которым FaceID-LoRA разваливала кадр
        на epiCRealism. Заметно это только через час обучения и по кадру.
        """
        from ball_reel import run_local, train
        from ball_reel.dataset import MIN_DATASET

        root = _dataset(self.tmp, n=MIN_DATASET)
        seen = {}

        def fake(dataset_dir, out_dir, cfg, *, base="", epochs=10, **kw):
            seen["base"] = base
            return {"ok": True, "steps": 1, "bound": "план",
                    "loss_first": 1.0, "loss_last": 0.9}

        real, train.train = train.train, fake
        try:
            got = run_local.train_subject_lora(
                self._args(train_lora=str(root), base="чужая/база"), self.tmp)
        finally:
            train.train = real
        self.assertTrue(got["ok"], got.get("note"))
        self.assertEqual(seen.get("base"), "чужая/база",
                         "прогон учит не на той базе, на которой рисует")

    def test_a_run_without_an_explicit_base_still_trains(self):
        # Отсутствие `--base` — это «умолчание», а не «сломать ступень»: у
        # старых вызовов поля может не быть вовсе.
        from ball_reel import run_local, train
        from ball_reel.dataset import MIN_DATASET

        root = _dataset(self.tmp, n=MIN_DATASET)
        seen = {}

        def fake(dataset_dir, out_dir, cfg, *, base="", epochs=10, **kw):
            seen["base"] = base
            return {"ok": True, "steps": 1, "bound": "план",
                    "loss_first": 1.0, "loss_last": 0.9}

        args = self._args(train_lora=str(root))
        if hasattr(args, "base"):
            del args.base
        real, train.train = train.train, fake
        try:
            got = run_local.train_subject_lora(args, self.tmp)
        finally:
            train.train = real
        self.assertTrue(got["ok"], got.get("note"))
        self.assertEqual(seen.get("base"), "")

    def test_small_dataset_stops_and_carries_the_cure(self):
        from ball_reel.dataset import MIN_DATASET
        from ball_reel.run_local import train_subject_lora

        root = _dataset(self.tmp, n=max(1, MIN_DATASET - 1))
        got = train_subject_lora(self._args(train_lora=str(root)), self.tmp)
        self.assertFalse(got["ok"])
        self.assertIn("ЛЕЧЕНИЕ", got["note"])

    def test_two_sources_of_an_adapter_are_refused_by_the_cli(self):
        """--lora и --train-lora вместе: отчёт потом не различит, кто был в кадре."""
        from ball_reel.run_local import build_parser

        args = build_parser().parse_args(
            ["--face", "f.jpg", "--prompt", "p", "--lora", "repo/id",
             "--train-lora", str(self.tmp)])
        self.assertTrue(args.lora and args.train_lora)  # парсер их пропускает
        # ...а прогон обязан остановиться на этом сочетании: см. main, шаг 1.5.
        import inspect

        from ball_reel import run_local

        src = inspect.getsource(run_local.main)
        self.assertIn("--lora, и --train-lora", src)


def _tiny_unet():
    """Крошечный UNet той же АРХИТЕКТУРЫ, что SD1.5, но на миллион параметров.

    Настоящие веса здесь не нужны и вредны: проверяется не качество обучения, а
    механика — на что сел адаптер, движутся ли его тензоры, читается ли то, что
    записано. Всё это одинаково на большой и на маленькой модели, и на
    маленькой считается за секунды без карты и без сети.
    """
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), cross_attention_dim=16, norm_num_groups=32,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8)


def _with_adapter(unet, rank: int = 8):
    from peft import LoraConfig

    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank,
                                init_lora_weights="gaussian",
                                target_modules=list(train.TARGET_MODULES)))
    return unet


class Machinery(unittest.TestCase):
    """Цикл на карте не исполнялся; его МЕХАНИКА исполняется здесь, на CPU.

    Именно тут живут все три молчаливые ловушки из шапки `train`, и ни одну из
    них нельзя поймать чтением кода: адаптер, севший не туда, выглядит в
    исходнике так же, как севший туда.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_frozen_model_reads_as_zero_trainable(self):
        unet = _tiny_unet()
        unet.requires_grad_(False)
        self.assertFalse(train.trainable_report(unet)["ok"])

    def test_adapter_on_the_named_modules_unfreezes_a_sane_share(self):
        got = train.trainable_report(_with_adapter(_tiny_unet()))
        self.assertTrue(got["ok"], got["note"])
        self.assertGreater(got["trainable"], 0)

    def test_gradient_checkpointing_is_available_after_the_adapter(self):
        """Без него план памяти врёт на ~1.5 ГБ — то есть отказ на 6 ГБ."""
        self.assertTrue(hasattr(_with_adapter(_tiny_unet()),
                                "enable_gradient_checkpointing"))

    def test_a_step_actually_moves_every_trainable_tensor(self):
        """Лосс может падать и при замороженном адаптере. Двигаются — веса."""
        import torch

        unet = _with_adapter(_tiny_unet())
        before = [p.detach().clone() for p in unet.parameters()
                  if p.requires_grad]
        opt = torch.optim.AdamW(
            [p for p in unet.parameters() if p.requires_grad], lr=1e-3)
        lat, emb = torch.randn(1, 4, 8, 8), torch.randn(1, 77, 16)
        noise = torch.randn_like(lat)
        pred = unet(lat + noise, torch.tensor([10]),
                    encoder_hidden_states=emb).sample
        torch.nn.functional.mse_loss(pred, noise).backward()
        opt.step()
        after = [p.detach().clone() for p in unet.parameters()
                 if p.requires_grad]
        moved = sum(1 for a, b in zip(before, after) if not torch.equal(a, b))
        self.assertEqual(moved, len(before),
                         f"из {len(before)} обучаемых тензоров сдвинулось "
                         f"{moved}: шаг оптимизатора не доходит до адаптера")

    def test_saved_adapter_is_named_as_the_loader_looks_for_it(self):
        """Каталог с `adapter_model.safetensors` загрузка НЕ найдёт."""
        from diffusers.loaders.lora_base import LORA_WEIGHT_NAME_SAFE

        self.assertEqual(train.ADAPTER_FILE, LORA_WEIGHT_NAME_SAFE)
        train.save_adapter(_with_adapter(_tiny_unet()), self.tmp / "out")
        self.assertTrue((self.tmp / "out" / train.ADAPTER_FILE).exists())

    def test_saved_keys_are_not_wrapped_by_peft(self):
        """`get_peft_model` уводит имена под `base_model.model.` — и адаптер
        загружается, ни на что не ложась, молча."""
        from diffusers import StableDiffusionPipeline

        train.save_adapter(_with_adapter(_tiny_unet()), self.tmp / "out")
        sd = StableDiffusionPipeline.lora_state_dict(str(self.tmp / "out"))
        sd = sd[0] if isinstance(sd, tuple) else sd
        self.assertTrue(sd, "ничего не прочиталось")
        for k in sd:
            self.assertTrue(k.startswith("unet."), k)
            self.assertNotIn("base_model.model.", k)

    def test_round_trip_changes_the_output_it_is_loaded_into(self):
        """Главный стык: выход обучения обязан быть входом генерации.

        Проверяется не «файл прочитался», а «выход модели изменился». Адаптер с
        несовпавшими именами читается без ошибки и не меняет НИЧЕГО — ровно тот
        отказ, который выясняется на демо в первую минуту.
        """
        import torch
        from diffusers import StableDiffusionPipeline

        torch.manual_seed(0)
        src = _with_adapter(_tiny_unet())
        for p in src.parameters():
            if p.requires_grad:
                p.data.add_(0.05)          # обученный адаптер ≠ нулевой
        train.save_adapter(src, self.tmp / "out")

        dst = _tiny_unet()
        dst.load_state_dict(src.state_dict(), strict=False)   # та же база
        lat, emb, t = torch.randn(1, 4, 8, 8), torch.randn(1, 77, 16), \
            torch.tensor([10])
        with torch.no_grad():
            before = dst(lat, t, encoder_hidden_states=emb).sample.clone()
        sd = StableDiffusionPipeline.lora_state_dict(str(self.tmp / "out"))
        sd = sd[0] if isinstance(sd, tuple) else sd
        dst.load_lora_adapter(sd, prefix="unet")
        with torch.no_grad():
            after = dst(lat, t, encoder_hidden_states=emb).sample
        self.assertGreater(float((after - before).abs().max()), 1e-6,
                           "адаптер загрузился и не изменил ничего")


class Wiring(unittest.TestCase):
    """Обучение — часть пайплайна, а не соседний репозиторий."""

    def test_targets_are_the_attention_projections(self):
        self.assertEqual(set(train.TARGET_MODULES),
                         {"to_q", "to_k", "to_v", "to_out.0"})

    def test_device_helper_exists_under_the_name_used(self):
        """`train` зовёт `device.detect`; опечатка здесь падает только на карте."""
        from ball_reel import device

        self.assertTrue(callable(device.detect))

    def test_adapter_lands_where_animate_can_load_it(self):
        """Выход обучения обязан быть входом генерации, иначе цепи нет."""
        import inspect

        from ball_reel import animate

        self.assertIn("subject_lora", inspect.signature(animate.build).parameters)


if __name__ == "__main__":
    unittest.main()
