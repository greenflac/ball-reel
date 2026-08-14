"""Обучение LoRA личности СВОИМ циклом, без внешнего тренера.

ПОЧЕМУ СВОЙ, А НЕ kohya. `lora.py` порождает конфиг и командную строку для
готового тренера — это было осознанное решение «не писать своего». Оно
разошлось с продуктовым требованием: обучение обязано быть ЧАСТЬЮ пайплайна, а
не инструкцией к постороннему репозиторию.

Цена внешнего тренера видна, если представить демо-день: склонировать чужой
репозиторий, поставить его зависимости рядом с нашими, попасть в его версию
torch, разобрать его конфиг. Каждый шаг может не выйти на машине, которой мы не
видели. Здесь же всё уже стоит: `diffusers`, `peft`, `torch`, `transformers` —
это те же веса и та же библиотека, которыми потом идёт генерация.

ЧТО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ. Не решает, КАКИМ должно быть обучение: ранг,
скорость, расписание, режим памяти считает `lora.config` и `lora.schedule` — там
это посчитано под 6 ГБ и объяснено. Здесь только исполнение.

ЧТО ИЗМЕРЕНО, А ЧТО НЕТ. Ни одна строка цикла не исполнялась на карте: GPU в
среде разработки нет. Арифметика (сколько шагов, сколько памяти, что во что
складывается) проверяется тестами без весов; сам проход — нет, и это помечено
НЕПРОВЕРЕНО, пока не пройдёт живьём.

ТРИ ЛОВУШКИ, КОТОРЫЕ МОЛЧАТ. Каждая даёт обучение, которое идёт и не работает:

1. **LoRA прицеплена не туда.** Адаптер на слоях, которые не участвуют в
   cross-attention, обучится и ничего не изменит. Поэтому целевые модули
   заданы явно и проверяются числом обучаемых параметров: ноль или подозрительно
   малое — отказ, а не «странно, но поедем».
2. **Подписи не читаются.** Тренер, не нашедший .txt рядом с картинкой, во
   многих реализациях учит на пустой подписи — то есть тянет на триггер всё
   подряд. Здесь отсутствие подписи это ОТКАЗ на этапе загрузки набора.
3. **Ничего не обучается, потому что всё заморожено.** Если забыть разморозить
   параметры адаптера, цикл честно отработает, лосс будет падать от шума, а
   веса не изменятся. Проверяется тем же счётчиком обучаемых параметров ДО
   первого шага.
"""

from __future__ import annotations

from pathlib import Path

#: Куда вешается LoRA. Имена — из архитектуры UNet SD1.5, и это ровно те
#: проекции внимания, через которые проходит обусловливание: заменить их
#: списком «на глаз» значит обучить адаптер, который ничего не решает.
TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0")

#: Ниже этой доли обучаемых параметров считаем, что адаптер прицепился не туда.
#: ВЫБРАН, опорный факт: LoRA ранга 16 на проекциях внимания SD1.5 даёт порядка
#: миллиона обучаемых параметров при 860 млн в UNet, то есть около 0.1%. Порог
#: на порядок ниже — он ловит НОЛЬ и «почти ноль», а не отличает 0.1 от 0.2.
MIN_TRAINABLE_SHARE = 1e-4

#: Разрешение обучения берётся из `lora.config`; здесь только пол, ниже
#: которого латент вырождается. ВЫБРАН по устройству VAE: 8 пикселей на ячейку.
MIN_RESOLUTION = 256

#: Множитель латента SD1.5. НЕ выбран нами: это `vae.config.scaling_factor` из
#: конфига весов, и разойтись с ним значит учить на латентах не той дисперсии,
#: что видит генерация. Число продублировано здесь только как значение по
#: умолчанию — фактическое берётся из загруженного VAE.
VAE_SCALING = 0.18215


def _resolve(root: Path, recorded: str) -> Path:
    """Путь из манифеста -> путь на ЭТОЙ машине.

    ЗАЧЕМ. Манифест пишет АБСОЛЮТНЫЕ пути — те, что были у машины-сборщика. Это
    незаметно, пока набор учат там же, где собрали, и ломается ровно в тот
    момент, ради которого набор и собирают: набор собран через шлюз на одной
    машине, а карта, на которой учат, — другая. Там `/tmp/.../ds/img/real.png`
    не существует, и обучение отказывает на ПОЛНОМ, ничем не повреждённом
    наборе, сообщая «набор неполон» — то есть указывает не туда.

    Правило: сначала записанный путь, потом тот же файл ПО ИМЕНИ внутри
    каталога набора. Молчаливой подмены не происходит: если файла нет и там,
    отказ приходит прежний, со списком недостающих.
    """
    p = Path(recorded)
    if p.exists():
        return p
    # `PurePosixPath`/`PureWindowsPath` не угадать по строке, а разделитель у
    # сборщика мог быть чужой: берём имя по обоим разделителям сразу.
    name = recorded.replace("\\", "/").rsplit("/", 1)[-1]
    for cand in (root / "img" / name, root / name):
        if cand.exists():
            return cand
    return p


def load_pairs(dataset_dir) -> list:
    """Набор с диска -> [(путь к картинке, подпись, повторов)]. Без torch.

    ОТСУТСТВИЕ ПОДПИСИ — ОТКАЗ, а не пустая строка. Многие тренеры в этом месте
    молча подставляют пустую подпись, и кадр начинает тянуть на триггер всё, что
    в нём есть: фон, одежду, свет. Обучение при этом идёт и выглядит успешным.
    """
    import json

    root = Path(dataset_dir)
    man_path = root / "manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(
            f"нет {man_path}: набор собирается командой "
            f"`python3 -m ball_reel.dataset --face ... --out {root}`, и без "
            f"манифеста неизвестно ни происхождение кадров, ни их веса")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    pairs, missing = [], []
    for s in man.get("samples", []):
        img = _resolve(root, s["path"])
        txt = img.with_suffix(".txt")
        if not img.exists():
            missing.append(f"{img} (картинка)")
            continue
        cap = s.get("caption") or (txt.read_text(encoding="utf-8").strip()
                                   if txt.exists() else "")
        if not cap:
            missing.append(f"{img} (подпись)")
            continue
        pairs.append((str(img), cap, int(s.get("repeats", 1))))
    if missing:
        raise ValueError(
            f"набор неполон, {len(missing)} записей без файла или подписи: "
            f"{missing[:3]}. Учить на пустой подписи нельзя — кадр потянет на "
            f"триггер фон, одежду и свет")
    if not pairs:
        raise ValueError(f"в {root} нет ни одного годного кадра")
    return pairs


def steps_for(pairs: list, *, epochs: int, batch_size: int = 1) -> int:
    """Сколько шагов даст этот набор. Повторы учитываются, а не игнорируются.

    Реальный кадр весит `dataset.REAL_ANCHOR_REPEATS`, и если считать шаги по
    числу ФАЙЛОВ, расписание окажется впятеро короче заявленного — обучение
    оборвётся раньше, чем адаптер что-то выучит.
    """
    per_epoch = sum(r for _, _, r in pairs)
    return max(1, (per_epoch * epochs) // max(1, batch_size))


def budget(pairs: list, cfg, *, epochs: int) -> dict:
    """Сколько шагов пройдёт на самом деле — и КТО из двух чисел ограничил.

    Чисел здесь два, и они приходят с разных сторон. `steps_for` считает, что
    даст НАБОР при заданном числе проходов. `cfg.max_train_steps` — это план из
    `lora.config`, посчитанный под память и под переобучение. Взять молча одно
    из них значит соврать в отчёте: при девяти кадрах набор даст 65 шагов, а
    план говорит 1200, и «обучено 1200 шагов» в этом случае — неправда.

    Поэтому план работает ПОТОЛКОМ, а не заданием, и в отчёт идёт имя того,
    кто ограничил.
    """
    want = steps_for(pairs, epochs=epochs, batch_size=cfg.batch_size)
    cap = int(getattr(cfg, "max_train_steps", 0) or 0)
    if cap and want > cap:
        return {"steps": cap, "bound": "план",
                "note": (f"набор дал бы {want} шагов за {epochs} проходов, но "
                         f"план `lora.config` ограничивает {cap} — идём по "
                         f"плану")}
    return {"steps": want, "bound": "набор",
            "note": (f"{want} шагов = {sum(r for _, _, r in pairs)} кадров с "
                     f"повторами x {epochs} проходов"
                     + (f", потолок плана {cap} не достигнут" if cap else ""))}


def order(pairs: list, *, seed: int) -> list:
    """Развернуть повторы в перемешанный список кадров. Детерминированно.

    ПОЧЕМУ НЕ ПОДРЯД. Кадр с весом `REAL_ANCHOR_REPEATS` в наивном цикле идёт
    пять раз ПОДРЯД: пять последовательных шагов Adam по одной и той же
    картинке — это не «весит больше», это всплеск в одну сторону, за которым
    адаптер уезжает и возвращается. Вес должен быть частотой по всему проходу,
    а не серией.

    Порядок задаётся `seed` из плана: два запуска одного набора обязаны дать
    одно и то же, иначе расхождение результата нечем объяснить.
    """
    import random

    flat = [(p, c) for p, c, r in pairs for _ in range(max(1, r))]
    random.Random(seed).shuffle(flat)
    return flat


def trainable_report(model) -> dict:
    """Сколько параметров реально обучается. -> отчёт с тремя исходами.

    Считается ДО первого шага. Ноль обучаемых — самая тихая из ловушек:
    цикл отработает, лосс будет шуметь вниз, веса не изменятся, а на выходе
    получится файл адаптера, не меняющий ничего.
    """
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    share = (train / total) if total else 0.0
    if train == 0:
        return {"ok": False, "trainable": 0, "total": total, "share": 0.0,
                "note": ("НИ ОДИН параметр не обучается: адаптер не прицеплен "
                         "либо всё заморожено. Цикл отработает и не изменит "
                         "ничего — самая тихая из ловушек обучения")}
    if share < MIN_TRAINABLE_SHARE:
        return {"ok": False, "trainable": train, "total": total,
                "share": share,
                "note": (f"обучаемых {train} из {total} ({share:.5%}) — ниже "
                         f"пола {MIN_TRAINABLE_SHARE:.4%}. Похоже, адаптер сел "
                         f"не на те модули: цель — проекции внимания "
                         f"{TARGET_MODULES}")}
    return {"ok": True, "trainable": train, "total": total, "share": share,
            "note": (f"обучаемых {train} из {total} ({share:.3%}) — адаптер на "
                     f"месте")}


def preflight(dataset_dir, cfg) -> dict:
    """Всё, что можно проверить ДО загрузки весов. Ни torch, ни сети.

    Порядок тот же, что во всём проекте: дешёвое раньше дорогого. Набор
    читается за миллисекунды, веса грузятся минуту.
    """
    checks, notes = [], []

    try:
        pairs = load_pairs(dataset_dir)
        checks.append({"name": "набор", "ok": True,
                       "detail": (f"{len(pairs)} кадров, "
                                  f"{sum(r for _, _, r in pairs)} с повторами")})
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "pairs": [],
                "checks": [{"name": "набор", "ok": False, "detail": str(e)}],
                "notes": []}

    from .dataset import MIN_DATASET

    enough = len(pairs) >= MIN_DATASET
    checks.append({
        "name": "размер", "ok": enough,
        "detail": (f"{len(pairs)} кадров при пороге {MIN_DATASET}"
                   + ("" if enough else " — LoRA переобучится на один ракурс"))})
    if not enough:
        notes.append(f"досыпать порождённых кадров: python3 -m ball_reel."
                     f"dataset --face ... --generated КАТАЛОГ")

    res = getattr(cfg, "resolution", 0)
    ok_res = res >= MIN_RESOLUTION
    checks.append({"name": "разрешение", "ok": ok_res,
                   "detail": (f"{res} px при поле {MIN_RESOLUTION}"
                              + ("" if ok_res else " — латент вырождается"))})

    for name in ("peft", "diffusers", "torch"):
        import importlib.util

        got = importlib.util.find_spec(name) is not None
        checks.append({"name": name, "ok": got,
                       "detail": "на месте" if got else
                                 f"НЕТ: python3 -m pip install {name}"})
        if not got:
            notes.append(f"без {name} обучение не начнётся")

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "pairs": pairs, "checks": checks, "notes": notes}


def train(dataset_dir, out_dir, cfg, *, base: str = "", epochs: int = 10,
          device: str = "", verbose: bool = True) -> dict:
    """Обучить LoRA. НЕ ИСПОЛНЯЛОСЬ НА КАРТЕ — см. шапку модуля.

    Возвращает отчёт: сколько шагов прошло, каким был лосс, куда лёг адаптер.
    Веса и оптимизатор берутся по плану из `lora.config`, а не выбираются здесь.
    """
    import torch
    from diffusers import (AutoencoderKL, DDPMScheduler,
                           UNet2DConditionModel)
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    from .animate import BASE_MODEL

    base = base or BASE_MODEL
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rep = preflight(dataset_dir, cfg)
    if not rep["ok"]:
        return {"ok": False, "checks": rep["checks"], "notes": rep["notes"],
                "note": "предполёт обучения не пройден"}
    pairs = rep["pairs"]

    from .device import detect

    dev = device or detect()
    dtype = torch.float16 if dev in ("cuda", "xpu") else torch.float32

    tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text = CLIPTextModel.from_pretrained(base, subfolder="text_encoder",
                                         torch_dtype=dtype).to(dev)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae",
                                        torch_dtype=dtype).to(dev)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet",
                                                torch_dtype=torch.float32).to(dev)
    sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    # Из КОНФИГА весов, а не из нашей константы: разойдясь, мы будем учить на
    # латентах не той дисперсии, которую потом увидит генерация.
    scaling = float(getattr(vae.config, "scaling_factor", VAE_SCALING))

    text.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # `add_adapter`, а НЕ `get_peft_model`. Оба вешают тот же адаптер на те же
    # модули и дают то же число обучаемых параметров — расходятся они на
    # сохранении. `get_peft_model` подменяет модель обёрткой `PeftModel`, и её
    # имена уезжают под префикс `base_model.model.`; сохранённый оттуда
    # адаптер diffusers прочитает, ни на что не наложит и промолчит. Здесь
    # модель остаётся `UNet2DConditionModel`, и ключи выходят такими, какими их
    # ждёт загрузка. ПРОВЕРЕНО round-trip'ом в `test_train`.
    unet.add_adapter(LoraConfig(
        r=cfg.rank, lora_alpha=cfg.alpha, init_lora_weights="gaussian",
        target_modules=list(TARGET_MODULES)))

    # СЧЁТЧИК ДО ПЕРВОГО ШАГА. Обучение, в котором ничего не обучается, идёт
    # ровно так же, как настоящее, и отличается только результатом.
    tr = trainable_report(unet)
    if verbose:
        print(f"  {tr['note']}")
    if not tr["ok"]:
        return {"ok": False, "trainable": tr, "note": tr["note"]}

    if cfg.gradient_checkpointing:
        # PeftModel проксирует неизвестные атрибуты в обёрнутую модель, но это
        # его частность, а не контракт. Молчаливое отсутствие метода здесь
        # стоит 1.5 ГБ памяти на 6-гигабайтной карте — то есть отказа посреди
        # обучения, а не замедления.
        fn = getattr(unet, "enable_gradient_checkpointing", None)
        if fn is None:
            return {"ok": False, "note": (
                "чекпоинты градиентов заказаны планом, но модель их не умеет: "
                "без них план памяти врёт примерно на 1.5 ГБ, и на 6 ГБ "
                "обучение упадёт по памяти посреди прогона")}
        fn()

    # Цикл идёт по одному кадру на шаг. Если план просит батч больше единицы,
    # об этом надо СКАЗАТЬ, а не тихо посчитать шаги по плану и пройти по
    # одному: расписание тогда врёт ровно во столько раз.
    if cfg.batch_size != 1 and verbose:
        print(f"  ВНИМАНИЕ: план просит батч {cfg.batch_size}, цикл идёт по "
              f"одному кадру на шаг — эффективный батч 1")

    opt = _optimizer([p for p in unet.parameters() if p.requires_grad], cfg)
    plan = budget(pairs, cfg, epochs=epochs)
    total = plan["steps"]
    if verbose:
        print(f"  {plan['note']}")
    flat = order(pairs, seed=getattr(cfg, "seed", 0))
    losses = []

    from PIL import Image

    for step in range(1, total + 1):
        path, caption = flat[(step - 1) % len(flat)]
        im = Image.open(path).convert("RGB").resize(
            (cfg.resolution, cfg.resolution), Image.LANCZOS)
        x = torch.from_numpy(
            (_as_array(im) / 127.5 - 1.0)).permute(2, 0, 1)[None]
        x = x.to(dev, dtype=dtype)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample() * scaling
            ids = tok(caption, padding="max_length", truncation=True,
                      max_length=tok.model_max_length,
                      return_tensors="pt").input_ids.to(dev)
            emb = text(ids)[0]
        lat = lat.float()
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps,
                          (1,), device=dev).long()
        noisy = sched.add_noise(lat, noise, t)
        pred = unet(noisy, t, encoder_hidden_states=emb.float()).sample
        loss = torch.nn.functional.mse_loss(pred, noise)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if verbose and step % 25 == 0:
            print(f"  шаг {step}/{total}  лосс "
                  f"{sum(losses[-25:]) / 25:.4f}", flush=True)

    saved = save_adapter(unet, out)
    return {"ok": True, "steps": total, "out": str(out), "file": saved,
            "bound": plan["bound"],
            "loss_first": round(sum(losses[:10]) / max(1, len(losses[:10])), 4),
            "loss_last": round(sum(losses[-10:]) / max(1, len(losses[-10:])), 4),
            "trainable": tr,
            "note": (f"обучено {total} шагов ({plan['bound']} ограничил), "
                     f"адаптер в {out}. Лосс сам по себе НЕ доказательство: "
                     f"он падает и у адаптера, выучившего фон. Приёмка — "
                     f"`lora.accept` на пробных промтах")}


#: Имя, под которым `load_lora_weights` ищет веса в КАТАЛОГЕ. Не наше
#: соглашение: это `diffusers.loaders.lora_base.LORA_WEIGHT_NAME_SAFE`, и
#: сверка с ним живёт в тестах, чтобы смена имени в апстриме краснела здесь, а
#: не на демо.
ADAPTER_FILE = "pytorch_lora_weights.safetensors"


def save_adapter(unet, out_dir) -> str:
    """Сохранить адаптер В ТОМ ФОРМАТЕ, который читает генерация.

    ЗДЕСЬ БЫЛ ДЕФЕКТ, и он из самых дорогих: обучение писало
    `peft.save_pretrained`, то есть `adapter_config.json` +
    `adapter_model.safetensors`. Это законный формат peft — и не тот, который
    `load_lora_weights`, получив КАТАЛОГ, идёт искать: он ищет
    `pytorch_lora_weights.safetensors`. Обучение при этом отрабатывает
    полностью, файл на диске есть, отчёт зелёный, а прицепить адаптер к
    прогону нельзя — и выясняется это в первую же минуту на демо, когда
    переучивать некогда.

    Разница ещё и в ключах: peft пишет `base_model.model.down_blocks...`, а
    diffusers ждёт `unet.down_blocks...` с суффиксами `lora.down/lora.up`.
    Перевод делает `convert_state_dict_to_diffusers` — не вручную, потому что
    имена меняются от версии к версии, и подобранные на глаз молча дадут
    адаптер, который загрузится и ничего не поменяет.
    """
    from diffusers import StableDiffusionPipeline
    from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(out), unet_lora_layers=layers,
        weight_name=ADAPTER_FILE, safe_serialization=True)
    return str(out / ADAPTER_FILE)


def _as_array(im):
    import numpy as np

    return np.asarray(im, dtype="float32")


def _optimizer(params, cfg):
    """AdamW, по возможности 8-битный. Выбор объяснён в `lora.config`."""
    import torch

    if "8bit" in (cfg.optimizer or "").lower():
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=cfg.learning_rate)
        except Exception:  # noqa: BLE001
            # НЕ молча: 8-битный оптимизатор был в плане памяти, и без него
            # план врёт примерно на 0.4 ГБ.
            print("  ВНИМАНИЕ: bitsandbytes нет, оптимизатор 32-битный — "
                  "плану памяти добавить ~0.4 ГБ")
    return torch.optim.AdamW(params, lr=cfg.learning_rate)


def main(argv: list) -> int:
    import argparse

    from .lora import config

    ap = argparse.ArgumentParser(
        prog="ball_reel.train",
        description="обучить LoRA личности своим циклом, без внешнего тренера")
    ap.add_argument("--dataset", required=True, help="каталог набора")
    ap.add_argument("--out", default="lora_out")
    ap.add_argument("--vram", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true",
                    help="только предполёт и план, без загрузки весов")
    # БАЗА ОБУЧЕНИЯ ОБЯЗАНА СОВПАДАТЬ С БАЗОЙ РИСОВАНИЯ, и это не гигиена.
    # FaceID-LoRA обучена под ванильную SD1.5; на epiCRealism она бьёт по тем
    # самым весам, которые дают этой базе фотореализм, — измерено вечером:
    # с `--ip-adapter-scale 0 --faceid-lora-scale 0` кадр стал резким и
    # фактурным, с ними — радужные потёки. Обучив СВОЮ LoRA на другой базе,
    # мы воспроизвели бы ровно тот же дефект своими руками.
    ap.add_argument("--base", default="",
                    help="база обучения; ДОЛЖНА совпадать с --base у probe и "
                         "run_local, иначе LoRA окажется чужой для той базы, "
                         "на которой рисуем")
    args = ap.parse_args(argv)

    cfg = config(args.vram)
    rep = preflight(args.dataset, cfg)
    for c in rep["checks"]:
        print(f"{'PASS' if c['ok'] else 'FAIL':<5} {c['name']:<12} {c['detail']}")
    for n in rep["notes"]:
        print(f"\nЛЕЧЕНИЕ: {n}")
    if rep["ok"]:
        plan = budget(rep["pairs"], cfg, epochs=args.epochs)
        from .animate import BASE_MODEL

        print(f"\nплан: {plan['steps']} шагов, ранг {cfg.rank}, разрешение "
              f"{cfg.resolution}, {cfg.optimizer}")
        print(f"      {plan['note']}")
        # База печатается ВСЕГДА, в том числе при --dry-run: несовпадение с
        # базой рисования обнаруживается глазом здесь за секунду, а иначе —
        # через час обучения и испорченный кадр.
        print(f"      база: {args.base or BASE_MODEL}"
              + ("" if args.base else "  (умолчание; на другой базе рисуете "
                                      "— передайте --base)"))
    if args.dry_run or not rep["ok"]:
        return 0 if rep["ok"] else 1

    got = train(args.dataset, args.out, cfg, base=args.base,
                epochs=args.epochs)
    print("\n" + got["note"])
    return 0 if got["ok"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
