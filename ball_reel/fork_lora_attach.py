"""ПРИЛОЖИЛАСЬ ЛИ LoRA ВООБЩЕ. Шаг, которого не было между загрузкой и оценкой.

ДЫРА, РАДИ КОТОРОЙ НАПИСАНО. `lora.accept` сравнивает плечо с адаптером и плечо
без него и отвечает, стало ли лучше. Он МОЛЧА ПРЕДПОЛАГАЕТ, что адаптер
приложился. А ComfyUI при несовпадении ключей не падает:

    comfy/sd.py, load_lora_for_models:
        for x in loaded:
            if (x not in k) and (x not in k1):
                logging.warning("NOT LOADED {}".format(x))

Предупреждение в журнал — и дальше. Ноль совпавших ключей означает чистый
прогон без ошибки и без эффекта. Кадры выйдут, вердикт будет «эффекта нет», и
следующие часы уйдут на набор данных, ранг и число шагов — то есть на объяснение
того, чего не происходило. В старом конвейере такой сторож есть
(`animate.active_loras`: «не „мы её загрузили“, а „она в модели“»), на пути форка
через ComfyUI его не было.

ТРИ НЕЗАВИСИМЫХ КАНАЛА, И ОНИ НУЖНЫ ВСЕ ТРИ, потому что каждый слеп к тому, что
видит следующий:

1. `fit` — ДО прогона, без карты и без Comfy: имена модулей адаптера против имён
   тензоров модели. Ловит несовместимый адаптер за секунды и стоит ноль.
2. `from_log` — ПОСЛЕ прогона: сколько ключей Comfy не приложил. Ловит случай,
   когда имена совпали статически, а карта ключей внутри Comfy устроена иначе.
3. `effect_over_noise` — ЧИСЛОМ по кадрам: два прогона без адаптера дают пол
   недетерминизма генератора, третий с адаптером — сигнал. Ловит случай, когда
   ключи приложились, а сила оказалась нулевой, и не зависит ни от журнала, ни
   от имён.

ПОЧЕМУ ТРЕТИЙ КАНАЛ УСТРОЕН ПОЛОМ, А НЕ СРАВНЕНИЕМ «ОТЛИЧАЮТСЯ ЛИ КАДРЫ».
Побитовое совпадение доказывает ноль, но НЕ совпадение ничего не доказывает:
генерация на карте недетерминирована сама по себе, и два прогона с одним сидом
разойдутся без всякого адаптера. Сравнивать надо не с нулём, а с разбросом
самого генератора — тот же приём, которым `fork_leak` судит протечку против пола
VAE, и по той же причине: планка ниже неустранимого пола объявляет нормой то,
что тракт тратит сам.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

from .fork_identity import FAIL, PASS, UNMEASURED

#: Во сколько раз эффект адаптера должен превышать разброс генератора, чтобы
#: считаться приложившимся.
#:
#: ВЫБРАНО по аналогии с `fork_leak.LEAK_OVER_FLOOR = 2.0`, а не измерено:
#: разброса генератора без карты не снять. Двойка — минимум, при котором сигнал
#: не спутать с полом; после первого прогона на карте это число ОБЯЗАНО быть
#: пересчитано из наблюдённого разброса, и до тех пор помечено здесь как
#: невыверенное.
EFFECT_OVER_NOISE = 2.0

#: Сколько ключей адаптера должно приложиться, чтобы считать его приложившимся.
#: ВЫБРАНО: единица, то есть «хоть один». Планка стоит низко НАРОЧНО — она
#: отвечает на вопрос «приложилось ли вообще», а не «хорошо ли»; частичное
#: приложение отдельным числом в том же отчёте, и оно опаснее полного нуля.
MIN_LANDED = 1

#: Доля приложившихся ключей, ниже которой это уже не «приложилось частично», а
#: «адаптер не от этой модели». ВЫБРАНО: половина.
PARTIAL_ALARM = 0.5

#: Имя, под которым ComfyUI ищет магнитуду DoRA. ПРОВЕРЕНО исходником:
#: `comfy/lora.py` -> `dora_scale_name = "{}.dora_scale".format(x)`.
COMFY_DORA_KEY = "dora_scale"

#: Имя, под которым магнитуду DoRA сохраняет PEFT. Проверено исходником
#: `peft/tuners/lora/dora.py` и конфигом `use_dora`.
PEFT_DORA_KEY = "lora_magnitude_vector"

#: Хвосты имён тензоров LoRA, которые надо снять, чтобы получить имя МОДУЛЯ.
#: Собраны из двух семейств сразу — kohya/ComfyUI и peft, — потому что адаптер
#: приезжает из тренера, а читает его Comfy, и это разные соглашения.
LORA_SUFFIXES = (
    ".lora_up.weight", ".lora_down.weight", ".lora_A.weight", ".lora_B.weight",
    ".lora_A.default.weight", ".lora_B.default.weight",
    ".lora_magnitude_vector.weight", ".lora_magnitude_vector",
    ".dora_scale", ".alpha", ".lora_up", ".lora_down",
    # `.diff` и `.diff_b` — прямые поправки веса и смещения, не низкоранговые.
    # ПРОВЕРЕНО исходником, а не догадкой: `comfy/lora.py:78-81` берёт
    # `"{}.diff_b".format(x)` и кладёт его патчем типа `diff` на `.bias`.
    # Сначала их тут не было, и на настоящем адаптере из нашего же лока прибор
    # объявлял НЕРАЗОБРАННЫМИ 773 ключа из 1749 — сорок четыре процента, —
    # то есть занижал приложившееся почти вдвое. Прибор, занижающий охват,
    # хуже отсутствующего: он выдаёт неполноту за свойство адаптера.
    ".diff_b", ".diff",
)

#: Семейства адаптеров, которые ComfyUI знает помимо LoRA (`comfy/weight_adapter
#: /__init__.py`): LoHa, LoKr, OFT; GLoRA и BOFT там же, но отключены в
#: `adapter_maps`. Их суффиксы этот прибор НЕ разбирает — такой адаптер попадёт
#: в «не разобрано» и будет виден числом, а не пропущен молча.
OTHER_FAMILIES = ("LoHa", "LoKr", "OFT", "GLoRA (отключён)", "BOFT (отключён)")

#: Приставки, которыми тренеры и загрузчики предваряют имя модуля.
LORA_PREFIXES = ("diffusion_model.", "base_model.model.", "lora_unet_",
                 "transformer.", "model.diffusion_model.")


def read_tensor_names(path: str | Path) -> set:
    """Имена тензоров из safetensors БЕЗ загрузки весов.

    Читается только заголовок: первые 8 байт — его длина, дальше JSON. Файл
    адаптера может быть на гигабайт, а нам нужны имена, и грузить ради них веса
    значит не уметь проверить адаптер там, где карты нет.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"нет файла адаптера {p}")
    with p.open("rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError(f"{p.name}: файл короче заголовка safetensors")
        n = struct.unpack("<Q", raw)[0]
        head = f.read(n)
    if len(head) < n:
        raise ValueError(f"{p.name}: заголовок обрезан, объявлено {n} байт")
    data = json.loads(head.decode("utf-8"))
    data.pop("__metadata__", None)
    return set(data)


def module_of(key: str) -> str | None:
    """Имя МОДУЛЯ, на который садится этот тензор адаптера.

    Возвращает `None`, если ключ не похож ни на одно известное соглашение, — и
    это не то же самое, что «модуля нет». Неразобранный ключ обязан попасть в
    отчёт отдельным числом: тихо пропустить его значит объявить адаптер меньше,
    чем он есть, и тем завысить долю приложившегося.
    """
    name = key
    for suffix in LORA_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    else:
        return None
    for prefix in LORA_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # ПОДЧЁРКИВАНИЯ KOHYA НЕ РАЗВОРАЧИВАЮТСЯ ЗДЕСЬ, И ЭТО НАРОЧНО. Тренер
    # пишет путь как `blocks_0_self_attn_q`, и восстановить из него точки
    # однозначно НЕЛЬЗЯ: `self_attn` — одно имя модуля, а `blocks_0` — два.
    # Первая версия меняла все подчёркивания на точки и выдавала
    # `blocks.0.self.attn.q` — имени, которого в модели нет; адаптер
    # объявлялся бы неподходящим из-за нашей догадки, а не из-за модели.
    # Имя возвращается как есть, а сопоставление идёт в `fit` по нормализации
    # ОБЕИХ сторон — то есть точным сравнением, а не угадыванием разделителей.
    return name or None


def _flat(name: str) -> str:
    """Имя без разделителей. Одно место, где решается, что `a.b` и `a_b` — одно."""
    return name.replace("_", "").replace(".", "").lower()


def fit(adapter: str | Path, target_names) -> dict:
    """Сядет ли адаптер на модель. ДО прогона, без карты и без Comfy.

    `target_names` — имена тензоров целевой модели (см.
    `tools/fork_probe_lora_fit.py`, который снимает их Range-запросами, не качая
    веса).
    """
    keys = read_tensor_names(adapter)
    target = {str(t) for t in target_names}
    bare = {re.sub(r"\.(weight|bias)$", "", t) for t in target}
    # Индекс без разделителей: `blocks.0.self_attn.q` и `blocks_0_self_attn_q`
    # дают один ключ. Так адаптер kohya сопоставляется с именами модели точно,
    # без восстановления точек по догадке (см. `module_of`).
    flat = {_flat(t): t for t in bare}

    lands, absent, unparsed = [], [], []
    for key in sorted(keys):
        mod = module_of(key)
        if mod is None:
            unparsed.append(key)
        elif mod in bare or f"{mod}.weight" in target:
            lands.append(mod)
        elif _flat(mod) in flat:
            lands.append(flat[_flat(mod)])
        else:
            absent.append(mod)

    lands, absent = sorted(set(lands)), sorted(set(absent))
    known = len(lands) + len(absent)
    share = (len(lands) / known) if known else 0.0

    if not keys:
        outcome, why = UNMEASURED, "в адаптере ноль тензоров"
    elif not known:
        outcome, why = UNMEASURED, ("ни один ключ не разобран: соглашение "
                                    "имён незнакомо, судить не о чем")
    elif len(lands) < MIN_LANDED:
        outcome, why = FAIL, "не сядет ни один модуль — адаптер не от этой модели"
    elif share < PARTIAL_ALARM:
        outcome, why = FAIL, (f"сядет меньше половины ({share:.0%}) — это не "
                              f"«частично», это чужой адаптер")
    elif absent:
        outcome, why = UNMEASURED, (f"{len(absent)} модулей не найдено в модели: "
                                    f"адаптер приложится НЕ ЦЕЛИКОМ, и что "
                                    f"именно потеряется — вопрос к обучению")
    else:
        outcome, why = PASS, "все разобранные модули есть в модели"

    return {
        "outcome": outcome, "keys": len(keys),
        "lands": len(lands), "absent": len(absent), "unparsed": len(unparsed),
        "share": round(share, 4),
        "absent_names": absent[:20], "unparsed_names": unparsed[:20],
        "dora": dora_readiness(keys),
        "note": (f"ключей в адаптере {len(keys)}, разобрано {known}, "
                 f"не разобрано {len(unparsed)}; сядет {len(lands)} модулей, "
                 f"не найдено {len(absent)} ({share:.0%} сядет). {why}"),
    }


def dora_readiness(keys) -> dict:
    """Найдёт ли ComfyUI магнитуду DoRA, если она там есть.

    Дефект, который иначе проходит молча: PEFT сохраняет магнитуду под именем
    `lora_magnitude_vector`, а ComfyUI ищет `dora_scale`. Адаптер загрузится,
    ошибки не будет, магнитуда потеряется — и DoRA тихо превратится в обычную
    LoRA, то есть в то, ради отказа от чего её и брали. На этом проекте уже был
    близнец этого дефекта: «адаптер пишется в формате peft, которого загрузка не
    ищет: обучение зелёное, прицепить нечего».
    """
    keys = set(keys)
    comfy = {k for k in keys if COMFY_DORA_KEY in k}
    peft = {k for k in keys if PEFT_DORA_KEY in k}
    if not comfy and not peft:
        return {"outcome": PASS, "kind": "LoRA", "comfy_keys": 0, "peft_keys": 0,
                "note": "магнитуды нет — это обычная LoRA, терять нечего"}
    if peft and not comfy:
        return {"outcome": FAIL, "kind": "DoRA", "comfy_keys": 0,
                "peft_keys": len(peft),
                "note": (f"магнитуда лежит под именем {PEFT_DORA_KEY!r} "
                         f"({len(peft)} ключей), а ComfyUI ищет "
                         f"{COMFY_DORA_KEY!r}. Загрузится без ошибки и без "
                         f"магнитуды: DoRA молча станет обычной LoRA")}
    if comfy and peft:
        return {"outcome": FAIL, "kind": "DoRA", "comfy_keys": len(comfy),
                "peft_keys": len(peft),
                "note": "магнитуда лежит под ОБОИМИ именами — какая из них "
                        "поедет, решит порядок ключей, а это не решение"}
    return {"outcome": PASS, "kind": "DoRA", "comfy_keys": len(comfy),
            "peft_keys": 0,
            "note": f"магнитуда под именем {COMFY_DORA_KEY!r} — ComfyUI найдёт"}


NOT_LOADED = re.compile(r"NOT LOADED:?\s+(\S+)")


def from_log(text: str, *, keys_in_adapter: int | None = None) -> dict:
    """Сколько ключей Comfy НЕ приложил. Разбор журнала прогона.

    `keys_in_adapter` — из `read_tensor_names`, и без него отчёт неполон: одни
    только «не приложенные» не отличают «не приложилось три из тысячи» от «не
    приложилось три из трёх». Поэтому без этого числа исход `не смогли`, а не
    вычисленная из воздуха доля.
    """
    missed = NOT_LOADED.findall(text or "")
    if keys_in_adapter is None:
        return {"outcome": UNMEASURED, "not_loaded": len(missed),
                "keys": None, "share": None,
                "not_loaded_names": missed[:20],
                "note": (f"в журнале {len(missed)} строк NOT LOADED, но сколько "
                         f"ключей было всего — не подано. Доля без знаменателя "
                         f"не считается")}
    landed = keys_in_adapter - len(missed)
    share = landed / keys_in_adapter if keys_in_adapter else 0.0
    outcome = (FAIL if landed < MIN_LANDED or share < PARTIAL_ALARM
               else UNMEASURED if missed else PASS)
    return {
        "outcome": outcome, "not_loaded": len(missed),
        "keys": keys_in_adapter, "landed": landed, "share": round(share, 4),
        "not_loaded_names": missed[:20],
        "note": (f"ключей {keys_in_adapter}, приложилось {landed}, не "
                 f"приложилось {len(missed)} ({share:.0%} приложилось)"),
    }


def _three_on_one_scale(*images):
    """Три кадра в массивы ОДНИМ решением о масштабе на всех.

    ДЕФЕКТ, ИЗ-ЗА КОТОРОГО ЭТА ФУНКЦИЯ ЕСТЬ, а не прямой вызов
    `fork_leak._as_array`. Тот решает про масштаб ПОКАДРОВО: «max > 1 — значит
    это 0..255, делю на 255». Для картинок с диска это верно, для трёх кадров
    одного опыта — нет. Кадр без адаптера с максимумом 0.9999 остаётся как есть,
    а кадр с адаптером, где максимум чуть перевалил за единицу, делится на 255.
    Замерено на фикстуре: пол вышел 0.4955 при заложенном шуме 0.01, отношение
    ровно 1.0 — то есть прибор сравнивал кадр с тем же кадром, уменьшенным в
    двести пятьдесят пять раз, и уверенно печатал «эффекта нет».

    Решение принимается по максимуму ПО ВСЕМ ТРЁМ и применяется ко всем трём.
    Разный масштаб у кадров одного опыта — это не данные, это ошибка приведения.
    """
    import numpy as np
    from PIL import Image

    raw = []
    for image in images:
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                raw.append(np.asarray(im.convert("RGB"), dtype="float64"))
        elif hasattr(image, "convert"):
            raw.append(np.asarray(image.convert("RGB"), dtype="float64"))
        else:
            arr = np.asarray(image, dtype="float64")
            if arr.ndim == 2:
                arr = arr[:, :, None].repeat(3, axis=2)
            raw.append(arr)
    top = max(float(r.max()) for r in raw) if raw else 0.0
    scale = 255.0 if top > 1.5 else 1.0
    return tuple(r / scale for r in raw)


def effect_over_noise(without_a, without_b, with_lora) -> dict:
    """Сдвинул ли адаптер картинку СИЛЬНЕЕ, чем генератор шумит сам.

    `without_a` и `without_b` — два прогона БЕЗ адаптера на одном сиде. Их
    расхождение и есть пол: столько генератор даёт сам по себе. `with_lora` —
    прогон с адаптером на том же сиде.

    Побитовое совпадение `with_lora` с `without_a` доказывает ноль сразу и без
    пола. Обратное неверно: любое расхождение может быть шумом карты, и
    объявлять его эффектом — это ровно та ошибка, из-за которой на проекте
    завели правило про негативный контроль.
    """
    import numpy as np

    a, b, c = _three_on_one_scale(without_a, without_b, with_lora)
    if not (a.shape == b.shape == c.shape):
        return {"outcome": UNMEASURED, "floor": None, "signal": None,
                "ratio": None,
                "note": f"кадры разного размера: {a.shape}, {b.shape}, {c.shape}"}

    floor = float(np.abs(a - b).mean())
    signal = float(np.abs(a - c).mean())
    identical = bool(np.array_equal(a, c))

    if identical:
        return {"outcome": FAIL, "floor": round(floor, 8), "signal": 0.0,
                "ratio": 0.0, "identical": True,
                "note": ("кадр с адаптером ПОБИТОВО совпал с кадром без него — "
                         "адаптер не сделал ничего. Это доказательство, а не "
                         "подозрение")}
    if floor <= 0.0:
        return {"outcome": UNMEASURED, "floor": 0.0, "signal": round(signal, 8),
                "ratio": None, "identical": False,
                "note": ("два прогона БЕЗ адаптера совпали побитово: генератор "
                         "детерминирован, и пола у него нет. Тогда судить "
                         "нечем этим прибором — любой сдвиг значим, но "
                         "насколько, скажет `lora.accept`, а не он")}

    ratio = signal / floor
    outcome = PASS if ratio >= EFFECT_OVER_NOISE else FAIL
    return {
        "outcome": outcome, "floor": round(floor, 8),
        "signal": round(signal, 8), "ratio": round(ratio, 3),
        "identical": False,
        "note": (f"разброс генератора {floor:.6f}, сдвиг с адаптером "
                 f"{signal:.6f} — в {ratio:.2f} раза при планке "
                 f"{EFFECT_OVER_NOISE}. "
                 + ("адаптер приложился" if outcome == PASS else
                    "сдвиг неотличим от собственного шума генератора — "
                    "считать это эффектом нельзя")),
    }


def report(*, static: dict | None = None, log: dict | None = None,
           numeric: dict | None = None) -> dict:
    """Свести три канала. Худший исход побеждает, и каждый назван поимённо.

    Не среднее и не «два из трёх»: каналы ловят РАЗНОЕ, и зелёный статический
    разбор при красном числовом означает «имена совпали, эффекта нет» — то есть
    именно ту находку, ради которой прибор написан.
    """
    channels = {"статически": static, "по журналу": log, "числом": numeric}
    given = {k: v for k, v in channels.items() if v is not None}
    if not given:
        return {"outcome": UNMEASURED, "channels": {},
                "note": ("ни один канал не отработал. Ноль нарушений при нуле "
                         "проверок — не успех (Р2)")}
    outcomes = {k: v.get("outcome") for k, v in given.items()}
    worst = (FAIL if FAIL in outcomes.values() else
             UNMEASURED if UNMEASURED in outcomes.values() else PASS)
    return {
        "outcome": worst,
        "channels": {k: {"outcome": v.get("outcome"), "note": v.get("note")}
                     for k, v in given.items()},
        "checked": len(given), "of": len(channels),
        "note": ("; ".join(f"{k}: {o}" for k, o in outcomes.items())
                 + f". Каналов отработало {len(given)} из {len(channels)}"),
    }
