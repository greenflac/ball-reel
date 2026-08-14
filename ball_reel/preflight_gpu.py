"""Предполёт: найти всё, что сломается, ДО дорогого шага — и назвать лечение.

    python3 -m ball_reel.preflight_gpu --conditions kit/conditions --face kit/face.jpg --skip-gateway

Живое исполнение находит дефекты в каждом модуле без исключения, и времени на
карте мало. Поэтому здесь не «проверка окружения вообще», а список конкретных
способов провалить прогон, каждый со своим лечением и со своей ценой.

ТРИ ПРАВИЛА, КОТОРЫМ ПОДЧИНЁН ЭТОТ ФАЙЛ.

**Сообщение об отказе называет лечение.** «Нет весов» — плохо; «нет
`ip-adapter-faceid_sd15.bin` (97 МБ), скачать: `python3 -c "..."`» — годно.
Отказ без лечения превращается в поиск по документации ровно в тот момент,
когда его нет.

**«Не смогли проверить» — отдельный исход.** Не «хорошо» и не «плохо».
Печатается словом НЕПРОВЕРЕНО, считается отдельно и попадает в итоговую
строку. Этот дефект встречался на проекте четырежды в разных модулях, и он
самый коварный: непроверенное показывается галочкой.

**Проверки упорядочены по цене, а не по важности.** Отказ, найденный после
загрузки полутора гигабайт, — это дефект порядка, а не невезение. Цены
замерены в этой среде и записаны в `COSTS` ниже; `main` печатает время каждой
проверки, чтобы порядок можно было пересмотреть по фактам, а не по памяти.

Ничего не генерирует, ничего не качает и не тратит баланс.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Ниже этого генерация не поедет даже со всеми экономиями (см. gpu_keyframes).
MIN_VRAM_GB = 3.5
#: Сколько диска нужно, когда пересчитать точнее не вышло: ~8 ГБ весов плюс
#: кэш и выдача. Точный расчёт — в `check_disk`: он вычитает уже скачанное.
MIN_DISK_GB = 15
#: Кадры, mp4 и отчёты одного прогона. ВЫБРАНО с запасом: 16 png 512x768 — это
#: десятки мегабайт, но прогонов за вечер бывает много, и место кончается не на
#: первом.
DISK_OUTPUT_GB = 2.0

#: Замерено в среде разработки (без карты), медиана нескольких запусков. Здесь
#: не для красоты: порядок проверок обязан следовать из этих чисел, а не из
#: ощущения, и при расхождении правится порядок, а не таблица.
COSTS = {
    "пакеты": "~1 мс",
    "диск": "~3 мс",
    "ffmpeg": "~1 мс",
    "веса": "~10 мс",
    "модель позы": "~1 мс",
    "веса лица": "~1 мс",
    "сегментация": "~1 мс",
    "DWPose": "~1 мс",
    "драйвер": "~200 мс (НЕПРОВЕРЕНО: nvidia-smi в этой среде нет)",
    "условия": "~290 мс на 71 файл",
    "driving-кадры": "~20 мс",
    "torch": "~2.4 с (импорт)",
    "драйвер/сборка": "~0 мс (переиспользует две предыдущие)",
    "vram": "~0.1 мс после torch",
    "onnxruntime": "~200 мс (импорт)",
    "лицо": "~5.2 с (insightface + mediapipe на одном кадре)",
    "шлюз": "сеть, до 30 с",
}


def _ok(name: str, detail: str = "") -> tuple:
    return True, name, detail


def _fail(name: str, detail: str) -> tuple:
    return False, name, detail


def _unknown(name: str, detail: str) -> tuple:
    """Третий исход: проверить не вышло.

    Возвращается `None`, а не False, намеренно. `run_local._say` печатает
    `None` словом ПРОПУСК и не выдаёт непроверенное за проверенное; здесь
    `main` печатает НЕПРОВЕРЕНО. Ни один из двух не имеет права показать это
    галочкой.
    """
    return None, name, detail


# --------------------------------------------------------------- пакеты (~1 мс)

#: Почему peft обязателен, хотя выглядит как необязательный. ЭТО НЕ ТЕОРИЯ:
#: проверено по исходнику diffusers (`IPAdapterMixin.load_ip_adapter` ->
#: `unet._load_ip_adapter_loras`) — без peft загрузчик НЕ ПАДАЕТ, он пишет в
#: лог «PEFT backend is required to load these weights.» и едет дальше. LoRA
#: лица при этом не применяется вовсе: канал личности обусловлен наполовину,
#: кадры выходят похожими на кого угодно, а гейт списывает это на «дрейф».
#: Худший вид отказа — тот, который похож на работу.
PEFT_WHY = ("LoRA лица приезжает внутри IP-Adapter FaceID и грузится через "
            "peft. БЕЗ НЕГО diffusers НЕ ПАДАЕТ: пишет в лог 'PEFT backend is "
            "required to load these weights.' и продолжает — личность "
            "остаётся необусловленной, а прогон выглядит успешным")

#: (модуль для импорта, что писать в pip, зачем нужен, обязателен ли).
#: Обязательный отсутствующий пакет — жёсткий отказ: всё, что ниже, будет
#: считать не то или не считать вовсе.
PACKAGES = (
    ("torch", "torch", "вся генерация", True),
    ("diffusers", "'diffusers>=0.30'", "AnimateDiff + ControlNet + FaceID", True),
    ("transformers", "transformers", "текстовый энкодер SD1.5", True),
    ("accelerate", "accelerate", "выгрузка весов на CPU (offload)", True),
    ("safetensors", "safetensors", "чтение весов", True),
    ("peft", "peft", PEFT_WHY, True),
    ("huggingface_hub", "huggingface_hub", "скачивание и кэш весов", True),
    ("insightface", "insightface", "эмбеддинг лица: и канал личности, и гейт",
     True),
    ("onnxruntime", "onnxruntime", "исполняет insightface и DWPose", True),
    ("mediapipe", "mediapipe", "судья позы и мимики — независимый от DWPose",
     True),
    ("PIL", "pillow", "чтение условий и запись кадров", True),
    ("numpy", "numpy", "вся арифметика метрик", True),
    ("cv2", "opencv-python", "чтение driving-кадров в некоторых измерителях",
     False),
    ("requests", "requests", "только шлюзовой путь (--engine chain)", False),
)


def packages_verdict(present: dict) -> tuple:
    """Хватает ли пакетов. Отдельно от щупания sys.path, чтобы проверялось.

    `present` — {имя модуля: есть ли}. Про неупомянутый модуль вердикт не
    выдумывает ничего: он попадает в «не смогли проверить», а не в «есть».
    """
    missing_hard, missing_soft, unseen = [], [], []
    for mod, pip_name, why, hard in PACKAGES:
        got = present.get(mod)
        if got is None:
            unseen.append(mod)
        elif not got:
            (missing_hard if hard else missing_soft).append((mod, pip_name, why))
    if missing_hard:
        lines = [f"НЕТ {m}: {why}" for m, _, why in missing_hard]
        pips = " ".join(p for _, p, _ in missing_hard)
        return False, ("; ".join(lines)
                       + f". Лечение: python3 -m pip install {pips}")
    detail = f"{sum(1 for v in present.values() if v)} из {len(PACKAGES)} на месте"
    if missing_soft:
        detail += (f"; необязательных нет: "
                   + ", ".join(f"{m} ({why})" for m, _, why in missing_soft)
                   + f" — pip install "
                   + " ".join(p for _, p, _ in missing_soft))
    if unseen:
        return None, detail + f"; НЕ ОПРОШЕНЫ: {', '.join(unseen)}"
    return True, detail


def find_specs() -> dict:
    """Какие пакеты видны импортёру. `find_spec`, а НЕ import.

    Разница в цене — три порядка: `import torch` стоит 2.4 с, `find_spec` —
    доли миллисекунды. Проверка, которая ищет отсутствующий peft, обязана
    отвечать раньше, чем начнёт грузиться torch, иначе она стоит дороже того,
    что сторожит.
    """
    import importlib.util

    got = {}
    for mod, _, _, _ in PACKAGES:
        try:
            got[mod] = importlib.util.find_spec(mod) is not None
        except Exception:  # noqa: BLE001 — битый пакет считаем отсутствующим
            got[mod] = False
    return got


def check_packages() -> tuple:
    ok, detail = packages_verdict(find_specs())
    return (_unknown if ok is None else (_ok if ok else _fail))("пакеты", detail)


# ------------------------------------------------------------- torch (~2.4 с)

def torch_verdict(version: str, accelerator_available: bool, card: str,
                  *, device_kind: str = "cuda", built_cuda: str = "") -> tuple:
    """Годится ли эта сборка torch. Отдельно от импорта, чтобы проверялось.

    Различает ТРИ провала, потому что чинятся они по-разному:

    * сборка без CUDA вовсе — переставить пакет, никакой драйвер её не оживит;
    * сборка с CUDA, карты не видно — драйвер, проброс, скрытая карта;
    * ускоритель не NVIDIA — совет «проверь nvidia-smi» там вреден.

    ПРИЗНАК CPU-СБОРКИ — `built_cuda` (`torch.version.cuda`), А НЕ СУФФИКС
    `+cpu` в номере версии. Колесо с PyPI суффикса не несёт вовсе, и проверка
    по строке молча пропускает ровно тот случай, ради которого написана: на
    этом проекте так уже терялась вся ветка генерации. Суффикс остаётся
    запасным признаком на случай, когда `built_cuda` спросить не удалось, и
    тогда об этом так и говорится.
    """
    if accelerator_available:
        line = f"{card or device_kind}, torch {version}"
        return True, line + (f", собран с CUDA {built_cuda}" if built_cuda
                             else "")
    cpu_build = (built_cuda is None) or (not built_cuda and "+cpu" in version)
    if cpu_build:
        how = ("torch.version.cuda пуст" if built_cuda is None
               else "по суффиксу +cpu в версии; torch.version.cuda спросить не "
                    "вышло")
        return False, (
            f"torch {version} собран БЕЗ CUDA ({how}) — карты он не увидит "
            f"никогда, и никакой драйвер этого не изменит. Лечение:\n"
            f"    python3 -m pip uninstall -y torch torchvision\n"
            f"    python3 -m pip cache purge\n"
            f"    python3 -m pip install torch    # Linux: колесо с PyPI и "
            f"есть сборка под CUDA\n"
            f"  Windows: колесо с PyPI CPU-only, нужен индекс "
            f"https://download.pytorch.org/whl/cuXXX (тег подобрать на "
            f"pytorch.org под свою CUDA из шапки nvidia-smi и свой Python).\n"
            f"  До установки посмотреть, откуда взялась эта сборка: "
            f"python3 -m pip config list; env | grep -i pip_ — прописанный "
            f"индекс .../whl/cpu вернёт CPU-колесо обратно.")
    if built_cuda:
        return False, (
            f"torch {version} собран с CUDA {built_cuda}, но карты не видит. "
            f"Это НЕ про переустановку torch. Смотреть по порядку: "
            f"1) nvidia-smi — видит ли карту сам драйвер; "
            f"2) CUDA Version в шапке nvidia-smi — она должна быть не ниже "
            f"{built_cuda}, иначе нужна сборка под старую CUDA с индекса "
            f"pytorch.org; "
            f"3) CUDA_VISIBLE_DEVICES — не спрятана ли карта переменной "
            f"окружения; 4) в контейнере — проброшена ли она вообще.")
    return False, (
        f"torch {version} карту не видит (ни cuda, ни xpu), и с какой CUDA он "
        f"собран — спросить не вышло. Для NVIDIA — смотреть драйвер "
        f"(nvidia-smi) и совпадает ли он с CUDA сборки; для Intel Arc нужна "
        f"сборка с поддержкой XPU (PyTorch 2.5+) и драйверы Level Zero. Пока "
        f"карты нет, всё поедет на процессоре.")


def check_torch() -> tuple:
    try:
        import torch  # type: ignore
    except ImportError:
        # Версию CUDA здесь не называем. Индексы pytorch.org живут и умирают:
        # cu121 собран под cp39-cp312, и на Python 3.13+ он отдаёт пустоту с
        # сообщением «No matching distribution», по которому не догадаться, что
        # дело в версии интерпретатора. Поэтому — своя версия и селектор.
        return _fail("torch",
                     f"не установлен. Python {sys.version_info.major}."
                     f"{sys.version_info.minor}: взять индекс под него на "
                     f"https://pytorch.org/get-started/locally/ "
                     f"(pip install torch --index-url "
                     f"https://download.pytorch.org/whl/cuXXX). Если pip "
                     f"пишет 'No matching distribution' — колёс под ЭТУ "
                     f"версию Python в том индексе нет, брать новее.")
    except Exception as e:  # noqa: BLE001 — пакет есть, но не грузится
        return _fail("torch", f"установлен, но не импортируется: "
                              f"{type(e).__name__}: {e}. Это битая сборка или "
                              f"несовпадение с драйвером, а не отсутствие "
                              f"пакета: переставлять его вслепую бессмысленно.")
    from .device import detect, torch_build_cuda

    dev = detect()
    backend = getattr(torch, dev, None) if dev != "cpu" else None
    name = ""
    if backend is not None:
        try:
            name = backend.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = dev
    ok, detail = torch_verdict(torch.__version__, dev != "cpu", name,
                               device_kind=dev, built_cuda=torch_build_cuda())
    return (_ok if ok else _fail)(f"torch.{dev}", detail)


# ------------------------------------------------- драйвер против сборки (~0 мс)

def driver_verdict(driver_cuda, build_cuda, *, card: str = "") -> tuple:
    """Потянет ли драйвер эту сборку torch. Правило, а не абзац в документе.

    Драйвер печатает свою CUDA в шапке `nvidia-smi`. Если она ниже, чем у
    сборки, всё падает непонятно: обычно `cuda=False` или «no kernel image is
    available for execution on the device» — сообщения, по которым про драйвер
    не догадаться.
    """
    from .device import (DRIVER_OK, DRIVER_OLD_MAJOR, DRIVER_OLD_MINOR,
                         driver_covers)

    if build_cuda is None:
        return None, ("сборка torch без CUDA — сравнивать не с чем "
                      "(лечится не драйвером, а переустановкой torch, "
                      "см. строку torch)")
    if not build_cuda or not driver_cuda:
        missing = "версию CUDA драйвера" if not driver_cuda else "CUDA сборки"
        return None, (f"не удалось узнать {missing}: сравнить драйвер со "
                      f"сборкой нечем. Это НЕ «совпадает» — если генерация "
                      f"упадёт с 'no kernel image is available', смотреть "
                      f"сюда первым делом: nvidia-smi | head -3 против "
                      f"python3 -c 'import torch; print(torch.version.cuda)'")
    state = driver_covers(driver_cuda, build_cuda)
    who = f" на {card}" if card else ""
    if state == DRIVER_OLD_MAJOR:
        return False, (
            f"драйвер{who} поддерживает CUDA {driver_cuda}, а torch собран под "
            f"{build_cuda} — старший номер ниже, это падение. Лечение, любое "
            f"из двух: обновить драйвер NVIDIA до поддерживающего CUDA "
            f"{build_cuda}, ЛИБО поставить сборку torch под CUDA "
            f"{driver_cuda}: pip uninstall -y torch torchvision && pip install "
            f"torch --index-url https://download.pytorch.org/whl/cu"
            f"{driver_cuda.replace('.', '')} (тег сверить на pytorch.org).")
    if state == DRIVER_OLD_MINOR:
        return None, (
            f"драйвер{who} даёт CUDA {driver_cuda}, сборка torch — "
            f"{build_cuda}: младший номер ниже. У CUDA 11+ есть минорная "
            f"совместимость, и обычно это едет, но ПРОВЕРИТЬ ЭТО ЗДЕСЬ НЕЧЕМ "
            f"(карты нет). Если упадёт по 'no kernel image' — обновить "
            f"драйвер, это первый подозреваемый.")
    if state == DRIVER_OK:
        return True, (f"драйвер{who} даёт CUDA {driver_cuda} >= "
                      f"{build_cuda} у сборки torch")
    return None, (f"версии не разобрались: драйвер {driver_cuda!r}, сборка "
                  f"{build_cuda!r}")


def check_driver_build(probe: dict | None = None) -> tuple:
    from .device import smi_probe, torch_build_cuda

    p = probe if probe is not None else smi_probe()
    card = p["cards"][0]["name"] if p.get("cards") else ""
    ok, detail = driver_verdict(p.get("cuda"), torch_build_cuda(), card=card)
    fn = _unknown if ok is None else (_ok if ok else _fail)
    return fn("драйвер/сборка", detail)


# ------------------------------------------------------ карта глазами драйвера

def check_smi(probe: dict | None = None) -> tuple:
    """Что о карте знает драйвер — до всякого torch и на два порядка дешевле.

    Стоит здесь, а не после torch, по цене: `nvidia-smi` отвечает за десятые
    доли секунды и уже даёт имя карты, объём памяти и версию CUDA драйвера.
    Карта, которой не хватит на веса, обязана отсеяться до того, как две с
    половиной секунды уйдут на импорт torch.

    Отсутствие `nvidia-smi` — НЕПРОВЕРЕНО, а не отказ: на Intel Arc и на Apple
    его нет и быть не должно, а решение о годности примет строка torch.
    """
    from .device import smi_probe

    p = probe if probe is not None else smi_probe()
    if not p.get("cards"):
        return _unknown("драйвер",
                        f"{p.get('reason') or 'карт не видно'}. Если это "
                        f"машина с NVIDIA — драйвер не поднят, и генерации не "
                        f"будет: ставить драйвер (nvidia-smi обязан печатать "
                        f"имя карты). Если это Intel Arc или Apple — так и "
                        f"должно быть, годность скажет строка torch.")
    c = p["cards"][0]
    extra = (f", CUDA драйвера {p['cuda']}" if p.get("cuda")
             else ", версию CUDA драйвера прочитать не вышло")
    if c.get("vram_gb"):
        ok, detail = vram_verdict(c["vram_gb"])
        if not ok:
            return _fail("драйвер", f"{c['name']}: {detail}")
    return _ok("драйвер",
               f"{c['name']}, {c.get('vram_gb') or '?'} ГБ, драйвер "
               f"{c.get('driver')}{extra}")


# -------------------------------------------------------------- VRAM (~0.1 мс)

def vram_verdict(total_gb: float) -> tuple:
    """Хватает ли карты — и ЧЕГО именно не хватает.

    ДВА РАЗНЫХ ОТКАЗА, и путать их дорого, потому что лечатся они разным:

    * **не влезли веса** — постоянный расход, от разрешения не зависит вовсе.
      Резать разрешение бессмысленно; помогает выгрузка на CPU (веса живут в
      RAM и приезжают на карту по слою) или снятие ControlNet;
    * **не влезли активации** — растущий расход, зависит от разрешения и числа
      кадров. Здесь помогают тайлинг VAE, меньше кадров, меньше разрешение.

    Размеры весов берутся из `animate.WEIGHTS_GB` — там они посчитаны по
    размерам файлов с HF. Второй способ узнать то же самое разошёлся бы с
    первым молча.

    Пока порог сравнивался прямо внутри check_vram, тронуть его было нечем:
    функция требует torch, а его в среде разработки нет, и мутационный аудит
    показал это — снятый MIN_VRAM_GB не ронял ни одного теста.
    """
    if total_gb < MIN_VRAM_GB:
        return False, (f"{total_gb:.1f} ГБ — меньше {MIN_VRAM_GB} ГБ, нужных "
                       f"даже на 512x768 со всеми экономиями. Взять карту "
                       f"больше или уйти на 448x640.")
    weights, left = weights_headroom(total_gb)
    if weights is None:
        return True, f"{total_gb:.1f} ГБ (сколько займут веса — не спросить)"
    if left < 0:
        return False, (
            f"{total_gb:.1f} ГБ, а ВЕСА локального пути занимают ~{weights} ГБ "
            f"— не влезают сами по себе, до всяких активаций. Разрешение здесь "
            f"НИ ПРИ ЧЁМ. Лечение по порядку: 1) выгрузка на CPU "
            f"(cfg.offload='sequential' — те же кадры, только медленнее); "
            f"2) снять ControlNet (-0.72 ГБ, но движение перестаёт приходить "
            f"из driving-видео); 3) путь кейфреймов gpu_keyframes без модуля "
            f"движения (-0.84 ГБ). См. DEMO_RUNBOOK.md, раздел 5.")
    if left < 1.0:
        return True, (
            f"{total_gb:.1f} ГБ: веса ~{weights} ГБ, на АКТИВАЦИИ остаётся "
            f"~{left} ГБ — впритык. Если упадёт по памяти, это будут активации, "
            f"и лечится это тайлингом VAE и числом кадров, а не выгрузкой. "
            f"РАСЧЁТ по размерам файлов, не замер.")
    return True, (f"{total_gb:.1f} ГБ: веса ~{weights} ГБ, на активации "
                  f"~{left} ГБ (расчёт, не замер)")


def weights_headroom(total_gb: float) -> tuple:
    """(веса, остаток на активации) по расчёту `animate`. (None, None) — не спросить."""
    try:
        from .animate import WEIGHTS_GB
    except Exception:  # noqa: BLE001 — animate правится параллельно
        return None, None
    weights = round(sum(WEIGHTS_GB.values()), 2)
    return weights, round(total_gb - weights, 2)


def check_vram() -> tuple:
    try:
        import torch  # type: ignore

        from .device import detect

        dev = detect()
        if dev == "cpu":
            return _fail("vram", "карта недоступна (см. torch)")
        props = getattr(torch, dev).get_device_properties(0)
        total = getattr(props, "total_memory") / 1024 ** 3
    except Exception as e:  # noqa: BLE001
        return _fail("vram", f"не прочитать: {e}")
    ok, detail = vram_verdict(total)
    return (_ok if ok else _fail)("vram", detail)


# ------------------------------------------------------ onnxruntime (~200 мс)

def check_onnx_providers() -> tuple:
    """Умеет ли onnxruntime ускорять то, что мы ему отдадим.

    Проверка стоит здесь, а не «где-нибудь потом», по одной причине: провайдер,
    которого нет, onnxruntime принимает МОЛЧА и считает на процессоре. Замерено
    живьём — DWPose просил CUDA, получил предупреждение в stderr и выдал 438 мс
    вместо десятков. Числа при этом выглядят правдоподобно, и понять, что они
    про другое железо, по ним нельзя.

    Это не блокирующая проверка: на CPU всё работает, просто медленно. Поэтому
    она предупреждает, а не останавливает — останавливать надо то, что не
    поедет вовсе.
    """
    from .device import detect, onnx_providers

    dev = detect()
    want, missing = onnx_providers(dev)
    if not missing:
        return _ok("onnxruntime", f"{want[0]} доступен ({dev})")
    return _ok("onnxruntime",
               f"НЕТ {', '.join(missing)} — insightface и DWPose пойдут на CPU "
               f"(замерено: 438 мс против десятков). Поставить "
               f"onnxruntime-gpu под свою CUDA, иначе прогон будет медленным, "
               f"но верным.")


# --------------------------------------------------------------- диск (~3 мс)

def disk_verdict(free_gb: float, need_gb: float, why: str) -> tuple:
    if free_gb < need_gb:
        return False, (f"свободно {free_gb:.1f} ГБ, нужно ~{need_gb:.1f} "
                       f"({why}): качка встанет на середине, и выяснится это "
                       f"через двадцать минут. Освободить место или увести "
                       f"кэш на другой диск: export HF_HOME=/путь/побольше")
    return True, f"{free_gb:.1f} ГБ свободно, нужно ~{need_gb:.1f} ({why})"


def _free_gb(path: Path) -> float:
    """Свободно на ФС, которой принадлежит путь. Несуществующий — вверх по дереву.

    Кэш весов часто лежит на другом разделе, чем каталог прогона (`HF_HOME` в
    домашнем каталоге против NVMe под проект). Мерить свободное место в cwd и
    качать в HF_HOME — значит проверить не тот диск.
    """
    import shutil

    p = path.expanduser().resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1024 ** 3


def check_disk(path: str = ".") -> tuple:
    """Хватит ли места — отдельно под веса и отдельно под выдачу.

    Нужное считается от того, чего ЕЩЁ НЕТ: на машине с уже скачанными весами
    требовать 15 ГБ — ложная тревога, а она в предполёте стоит доверия ко всем
    остальным строкам.
    """
    inv = weights_inventory()
    out_free = _free_gb(Path(path))
    cache = hf_cache_dir()
    cache_free = _free_gb(cache)
    if inv.get("unknown"):
        ok, detail = disk_verdict(min(out_free, cache_free), MIN_DISK_GB,
                                  "инвентаризацию весов провести не вышло, "
                                  "порог по умолчанию")
        return (_ok if ok else _fail)("disk", detail)
    need_weights = round(inv["missing_gb"], 1)
    if cache_free == out_free:
        ok, detail = disk_verdict(out_free, need_weights + DISK_OUTPUT_GB,
                                  f"{need_weights} ГБ недостающих весов + "
                                  f"{DISK_OUTPUT_GB} ГБ на кадры и отчёты")
        return (_ok if ok else _fail)("disk", detail)
    ok_w, det_w = disk_verdict(cache_free, need_weights,
                               f"недостающие веса в {cache}")
    ok_o, det_o = disk_verdict(out_free, DISK_OUTPUT_GB,
                               "кадры, mp4 и отчёты прогона")
    if not ok_w:
        return _fail("disk", det_w)
    if not ok_o:
        return _fail("disk", det_o)
    return _ok("disk", f"{det_w}; выдача: {det_o}")


# --------------------------------------------------------------- веса (~10 мс)

def hf_cache_dir() -> Path:
    """Куда huggingface_hub реально кладёт веса. Спрашиваем ЕГО, а не память.

    Переменных, которые на это влияют, три (`HF_HUB_CACHE`,
    `HUGGINGFACE_HUB_CACHE`, `HF_HOME`), и порядок между ними мы бы угадали
    неправильно.
    """
    try:
        from huggingface_hub import constants  # type: ignore

        return Path(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        import os

        return Path(os.environ.get("HF_HOME",
                                   "~/.cache/huggingface")).expanduser() / "hub"


def weight_targets() -> list:
    """Что обязано лежать локально до генерации. Имена — ИЗ `animate`.

    Не копия списка, а импорт: второй список тех же имён разойдётся с первым
    молча, и предполёт начнёт проверять веса, которых пайплайн не грузит. На
    этом проекте уже был случай, когда имя репозитория оказалось выдуманным.

    Размеры в МБ сверены живьём по кэшу и по HF API (см. DEMO_RUNBOOK.md §2).
    """
    from . import animate

    return [
        {"role": "база SD1.5", "repo": animate.BASE_MODEL,
         "files": ["model_index.json",
                   "unet/diffusion_pytorch_model.safetensors",
                   "vae/diffusion_pytorch_model.safetensors",
                   "text_encoder/model.safetensors"],
         "mb": 4267,
         "patterns": ['"*.json", "*.txt", "tokenizer/*",'
                      ' "unet/diffusion_pytorch_model.safetensors",'
                      ' "vae/diffusion_pytorch_model.safetensors",'
                      ' "text_encoder/model.safetensors"']},
        {"role": "модуль движения", "repo": animate.MOTION_ADAPTER,
         "files": ["diffusion_pytorch_model.safetensors"], "mb": 1671,
         "patterns": ['"*.json", "diffusion_pytorch_model.safetensors"']},
        {"role": "условия позы (ControlNet)", "repo": animate.CONTROLNET,
         "files": ["diffusion_pytorch_model.safetensors"], "mb": 1445,
         "patterns": ['"*.json", "diffusion_pytorch_model.safetensors"']},
        {"role": "личность (IP-Adapter FaceID)", "repo": animate.IP_ADAPTER_REPO,
         "files": [animate.IP_ADAPTER_WEIGHT], "mb": 97,
         "patterns": [f'"{animate.IP_ADAPTER_WEIGHT}"']},
    ]


def _cached(repo: str, filename: str):
    """Путь к файлу в кэше HF или None. Спрашиваем huggingface_hub."""
    from huggingface_hub import try_to_load_from_cache  # type: ignore

    got = try_to_load_from_cache(repo, filename)
    return got if isinstance(got, str) and Path(got).exists() else None


def _repo_started(repo: str) -> bool:
    """Начиналась ли качка этого репозитория — чтобы отличить «не качали» от
    «недокачали». Лечение разное: первое — запустить, второе — дождаться."""
    d = hf_cache_dir() / ("models--" + repo.replace("/", "--"))
    return d.exists()


def weights_inventory() -> dict:
    """Что есть, чего нет и сколько ГБ ещё качать. Основа и для диска, и для весов."""
    try:
        targets = weight_targets()
        _cached(targets[0]["repo"], targets[0]["files"][0])
    except Exception as e:  # noqa: BLE001
        return {"unknown": f"{type(e).__name__}: {e}", "missing_gb": 0.0,
                "missing": [], "present": []}
    missing, present, missing_mb = [], [], 0.0
    for t in targets:
        gone = [f for f in t["files"] if not _cached(t["repo"], f)]
        if gone:
            missing.append({**t, "gone": gone,
                            "started": _repo_started(t["repo"])})
            missing_mb += t["mb"]
        else:
            present.append(t)
    return {"unknown": "", "missing": missing, "present": present,
            "missing_gb": missing_mb / 1024}


def check_weights() -> tuple:
    """Веса на месте — иначе первая же генерация уйдёт их качать.

    Раньше здесь суммировался размер кэша HF целиком с порогом 3 ГБ. Такая
    строка зеленела на машине, где скачана только база, а модуля движения нет
    вовсе, — то есть проверка отвечала на другой вопрос. Теперь спрашивается
    каждый файл поимённо, и отказ называет ФАЙЛ и КОМАНДУ.
    """
    inv = weights_inventory()
    if inv["unknown"]:
        return _unknown("weights",
                        f"кэш HF не опросить ({inv['unknown']}): проверить "
                        f"вручную python3 -c \"from huggingface_hub import "
                        f"try_to_load_from_cache as t; "
                        f"print(t('h94/IP-Adapter-FaceID', "
                        f"'ip-adapter-faceid_sd15.bin'))\"")
    if not inv["missing"]:
        return _ok("weights", f"{len(inv['present'])} комплекта на месте в "
                              f"{hf_cache_dir()}")
    lines, cmds = [], []
    for m in inv["missing"]:
        state = "недокачано" if m["started"] else "не скачивалось"
        lines.append(f"{m['role']} [{state}]: нет {', '.join(m['gone'][:2])} "
                     f"({m['mb']} МБ) из {m['repo']}")
        cmds.append(f'd("{m["repo"]}", allow_patterns=[{m["patterns"][0]}])')
    body = "\n".join("    " + c for c in cmds)
    return _fail("weights",
                 "; ".join(lines)
                 + f". Не хватает ~{inv['missing_gb']:.1f} ГБ. Скачать:\n"
                   f"  python3 - <<'EOF'\n"
                   f"  from huggingface_hub import snapshot_download as d\n"
                   f"{body}\n"
                   f"  EOF\n"
                   f"  (allow_patterns не украшение: без них у базы прилетят "
                   f"ещё ~17 ГБ ckpt-ов, которые пайплайн не открывает)")


# ------------------------------------------------------- модели-судьи (~1 мс)

def check_pose_model() -> tuple:
    """Модель позы MediaPipe: ею проверяется движение, и без неё гейт слеп."""
    from .pose import _model_path

    try:
        return _ok("pose model", str(_model_path()))
    except RuntimeError as e:
        # Сообщение НЕ режется по первой строке: команда curl лежит во второй,
        # и обрезка превращала «вот лечение» в «не найдено». Ровно тот дефект,
        # ради которого этот модуль переписан.
        return _fail("pose model", " ".join(str(e).split()))


def check_face_model() -> tuple:
    """Пак buffalo_l для insightface: и канал личности, и весь гейт лица.

    Он скачивается сам при первом обращении — но «сам» означает «в середине
    прогона, из github, молча и минуту». На машине без интернета это отказ
    после загрузки диффузии, то есть в самом дорогом месте.
    """
    import os

    root = Path(os.environ.get("INSIGHTFACE_HOME", "~/.insightface")).expanduser()
    pack = root / "models" / "buffalo_l"
    need = ("det_10g.onnx", "w600k_r50.onnx")
    try:
        from insightface.utils import storage  # type: ignore

        base = getattr(storage, "BASE_REPO_URL", "")
    except Exception:  # noqa: BLE001
        base = ("https://github.com/deepinsight/insightface/releases/download"
                "/v0.7")
    gone = [f for f in need if not (pack / f).exists()]
    if not gone:
        return _ok("face model", f"buffalo_l в {pack}")
    return _fail("face model",
                 f"нет {', '.join(gone)} в {pack} (276 МБ). Скачать заранее:\n"
                 f"    mkdir -p {pack.parent} && cd {pack.parent}\n"
                 f"    curl -sSLO {base}/buffalo_l.zip && unzip -o "
                 f"buffalo_l.zip -d buffalo_l\n"
                 f"  или один раз вызвать python3 -c \"from "
                 f"ball_reel.identity_arcface import face_detail; "
                 f"print(face_detail('kit/face.jpg'))\" при живом интернете.")


def check_segmentation() -> tuple:
    """Сегментация одежды: нужна приметам и прилеганию, не нужна кадрам.

    Поэтому это НЕПРОВЕРЕНО, а не отказ: без неё прогон едет, но два
    измерителя молчат — и молчат они честно, строкой ПРОПУСК в гейте.
    """
    from .bodyparts import available, why_unavailable

    if available():
        from .bodyparts import model_path

        return _ok("segmentation", str(model_path()))
    return _unknown("segmentation", " ".join(why_unavailable().split()))


def check_dwpose(needed: bool = False) -> tuple:
    """DWPose: нужен, только если снимать условия с нового видео.

    Для демо условия уже сняты и лежат в ките, поэтому по умолчанию это не
    отказ, а сообщение — но именно сообщение, а не тишина: если условия
    придётся перерендерить, 350 МБ качаются не в тот момент, когда это удобно.
    """
    from .dwpose import available, why_unavailable

    if available():
        return _ok("dwpose", "веса на месте — условия можно снимать заново")
    why = " ".join(why_unavailable().split())
    if needed:
        return _fail("dwpose", why)
    return _unknown("dwpose",
                    f"весов нет, и для готовых условий они НЕ НУЖНЫ (условия "
                    f"в ките сняты заранее). Понадобятся, только если снимать "
                    f"условия с другого видео: {why}")


# ------------------------------------------------------------- ffmpeg (~1 мс)

def check_ffmpeg() -> tuple:
    """Сборщик mp4. Проверяется ДО генерации намеренно.

    Кадры при отказе ffmpeg уцелеют, и гейт считается по ним — но узнать, что
    ролика не будет, дешевле сейчас (миллисекунда), чем после генерации.
    """
    import shutil

    if shutil.which("ffmpeg"):
        return _ok("ffmpeg", "найден в PATH")
    return _fail("ffmpeg",
                 "не найден в PATH: mp4 не соберётся (кадры уцелеют, гейт "
                 "посчитается). Лечение: apt install ffmpeg | brew install "
                 "ffmpeg | winget install ffmpeg")


# --------------------------------------------------------- условия (~290 мс)

def check_conditions(conditions: str, *, expect_size=None,
                     need_frames: int | None = None) -> tuple:
    """Условия должны существовать, читаться, быть одного размера, непустыми
    и идти подряд — а ещё их должно хватить на окно и подходить их пропорция.

    Пустота меряется содержимым, а не размером файла. Раньше здесь стоял порог
    в 500 байт, и он не срабатывал никогда: чёрный png 512x768 весит 1224 байта,
    то есть ровно тот случай, ради которого проверка написана, проходил её
    насквозь. Скелет рисуется цветными линиями по чёрному, поэтому одноцветный
    кадр — это буквально «скелета нет», и это видно по экстремумам яркости.

    Пропуски в нумерации — отдельная беда. `render_sequence` не пишет кадр,
    в котором позу не нашли, но нумерует по индексу исходного кадра, так что
    дырка выглядит как 0002 -> 0004. Прогон берёт каждый N-й файл из
    отсортированного списка и считает, что между ними одинаковое время; после
    дырки это перестаёт быть правдой, и клип дёргается там, где никто не
    ошибался.

    Размер условия против размера плана — третья беда, и она молчаливая.
    Проверено по исходнику diffusers 0.39 (`prepare_video` ->
    `control_video_processor.preprocess_video(video, height, width)`): условие
    МАСШТАБИРУЕТСЯ под размер пайплайна без единого предупреждения. При той же
    пропорции это потеря точности, при другой — растянутый скелет: поза поедет,
    а гейт спишет это на генератор.
    """
    d = Path(conditions)
    if not d.is_dir():
        return _fail("conditions",
                     f"каталога {d} нет. Кит распаковывается в kit/: "
                     f"tar xzf ball_reel_kit.tar.gz, запускать из корня "
                     f"репозитория с --conditions kit/conditions")
    files = sorted(d.glob("*.png"))
    if not files:
        others = [p.suffix for p in d.iterdir() if p.is_file()][:4]
        return _fail("conditions", f"в {d} нет png (лежит {others or 'пусто'}). "
                                   f"Отрендерить ДОМА: "
                                   f"skeleton.render_sequence (это CPU).")
    from PIL import Image

    sizes = set()
    empty, broken = [], []
    for f in files:
        try:
            with Image.open(f) as im:
                sizes.add(im.size)
                lo, hi = im.convert("L").getextrema()
        except Exception as e:  # noqa: BLE001 — битый файл называем, а не падаем
            broken.append(f"{f.name} ({type(e).__name__})")
            continue
        if lo == hi:
            empty.append(f.name)
    if broken:
        return _fail("conditions", f"{len(broken)} файл(ов) не читаются "
                                   f"({broken[:3]}): докопировать условия "
                                   f"целиком — половина файла выглядит как "
                                   f"файл, но PIL на ней падает в середине "
                                   f"прогона.")
    if len(sizes) > 1:
        return _fail("conditions", f"разные размеры {sizes}: ControlNet ждёт "
                                   f"один размер на всю последовательность. "
                                   f"Перерендерить всю серию одним вызовом "
                                   f"skeleton.render_sequence.")
    if empty:
        return _fail("conditions",
                     f"{len(empty)} пустых условия ({empty[:3]}): в этих кадрах "
                     f"скелет не найден, генератор там не ограничен вовсе. "
                     f"Перерендерить сегмент или резать окно мимо этих кадров "
                     f"(--from-frame).")
    gaps = _numbering_gaps([f.stem for f in files])
    if gaps:
        return _fail("conditions",
                     f"пропуски в нумерации после {gaps[:3]}: в этих кадрах "
                     f"позы не нашли, и шаг по времени между условиями больше "
                     f"не одинаковый — цепочка поедет неровно. Перерендерить "
                     f"сегмент без дырок или резать по ним на куски "
                     f"(--from-frame после дырки).")
    size = sizes.pop()
    if need_frames and len(files) < need_frames:
        return _fail("conditions",
                     f"{len(files)} условий, а окно требует {need_frames}: "
                     f"модуль движения обучен ровно на {need_frames} кадрах и "
                     f"откажется при другом числе. Снять условия с более "
                     f"длинного отрезка видео или уменьшить окно "
                     f"(animate.plan(frames=...)).")
    ok, note = size_verdict(size, expect_size)
    detail = f"{len(files)} шт., {size}" + (f"; {note}" if note else "")
    fn = _unknown if ok is None else (_ok if ok else _fail)
    return fn("conditions", detail)


def size_verdict(size, expect) -> tuple:
    """Условие против плана. Отдельно от чтения файлов, чтобы проверялось.

    Молчаливое масштабирование внутри diffusers — не гипотеза: см. docstring
    `check_conditions`. Одинаковая пропорция — предупреждение, разная —
    отказ: растянутый скелет ломает именно то, ради чего взят ControlNet.
    """
    if not expect:
        return True, ""
    if tuple(size) == tuple(expect):
        return True, f"совпадает с планом {tuple(expect)}"
    ar_got = size[0] / size[1]
    ar_want = expect[0] / expect[1]
    if abs(ar_got - ar_want) > 0.01:
        return False, (
            f"пропорция условий {ar_got:.3f} против {ar_want:.3f} у плана "
            f"{tuple(expect)}: diffusers растянет условие молча "
            f"(prepare_video -> preprocess_video), скелет исказится, и поза "
            f"уедет по вине не генератора. Лечение: перерендерить условия под "
            f"{tuple(expect)} (skeleton.render_sequence) или считать планом с "
            f"этой пропорцией.")
    return None, (f"условия {tuple(size)}, план {tuple(expect)} — пропорция та "
                  f"же, diffusers отмасштабирует молча. Отказом это не "
                  f"считаем, но мелкие суставы теряют точность; ПРОВЕРИТЬ, что "
                  f"так и задумано.")


def _numbering_gaps(stems: list) -> list:
    """Имена, после которых номер прыгнул. Ненумерованные имена не трогаем."""
    nums = []
    for s in stems:
        if not s.isdigit():
            return []
        nums.append((int(s), s))
    return [name for (a, name), (b, _) in zip(nums, nums[1:]) if b != a + 1]


# ---------------------------------------------------- driving-кадры (~20 мс)

def check_driving(conditions: str) -> tuple:
    """Чем будет проверяться поза. Без этого гейт не измеряет главное.

    Условие — это цветные палки на чёрном фоне, и детектор поз на нём не
    находит ничего: сравнение молча превращается в None, а вердикт «в норме»
    получается при любой позе. Проверять надо driving-кадром, а его путь живёт
    в манифесте рядом с условиями и записан ОТНОСИТЕЛЬНО каталога, откуда
    рендерили, — то есть из другого каталога не открывается.
    """
    import json

    d = Path(conditions)
    mp = d / "manifest.json"
    if not mp.exists():
        return _fail("driving", f"нет {mp}: карту «условие -> driving-кадр» "
                                f"пишет skeleton.render_sequence. Без неё позу "
                                f"сверять не с чем, и прогон не отличит "
                                f"воспроизведённое движение от выдуманного. "
                                f"Перерендерить условия текущим "
                                f"skeleton.render_sequence.")
    try:
        manifest = json.loads(mp.read_text())
    except Exception as e:  # noqa: BLE001
        return _fail("driving", f"манифест {mp} не читается "
                                f"({type(e).__name__}): перерендерить условия.")
    raw = manifest.get("driving_frames") or {}
    if not raw:
        return _fail("driving", f"в {mp} нет driving_frames: манифест снят "
                                f"старой версией skeleton. Перерендерить "
                                f"условия — новая версия пишет карту сама.")
    try:
        from .run_local import resolve_driving
    except Exception as e:  # noqa: BLE001 — соседний модуль правится параллельно
        return _unknown("driving", f"пути манифеста не разрешить: "
                                   f"run_local.resolve_driving не "
                                   f"импортируется ({type(e).__name__}: {e}). "
                                   f"Проверить вручную: ls {d.parent}/driving")
    found = sum(1 for v in raw.values() if resolve_driving(v, conditions))
    if not found:
        first = next(iter(raw.values()))
        return _fail("driving",
                     f"ни один из {len(raw)} путей манифеста не открывается "
                     f"отсюда (первый: {first}). Пути записаны относительно "
                     f"каталога, откуда рендерили условия. Лечение: запускать "
                     f"оттуда же (для кита — из kit/), либо положить "
                     f"driving-кадры рядом с условиями: {d.parent}/driving/")
    if found < len(raw):
        return _unknown("driving",
                        f"{found}/{len(raw)} driving-кадров найдено: у "
                        f"остальных поза станет ПРОПУСКОМ, а не отказом. "
                        f"Проверить, что окно (--from-frame) попадает в "
                        f"найденные кадры.")
    return _ok("driving", f"{found}/{len(raw)} найдено рядом с условиями")


# ---------------------------------------------------------------- лицо (~5 с)

def check_face(face: str) -> tuple:
    """То же, что intake делает дома — на случай, если сюда приехало не то фото."""
    p = Path(face)
    if not p.exists():
        return _fail("face", f"нет файла {p}: взять фото лица крупным планом "
                             f"(для кита — kit/face.jpg, 148 px)")
    try:
        from .intake import inspect
    except Exception as e:  # noqa: BLE001
        return _unknown("face", f"не импортировать intake ({type(e).__name__}: "
                                f"{e}) — лицо НЕ ПРОВЕРЕНО, а не «в порядке». "
                                f"Обычно это отсутствующий insightface или "
                                f"mediapipe, см. строку пакетов.")
    try:
        got = inspect(str(p))
    except Exception as e:  # noqa: BLE001
        return _unknown("face", f"{type(e).__name__}: {e} — лицо НЕ ПРОВЕРЕНО. "
                                f"Чаще всего это недокачанный buffalo_l "
                                f"(см. строку face model).")
    if not got.usable:
        return _fail("face", got.blocked.get("identity", "лицо не найдено")
                     + ". Лечение: фото, где лицо крупнее и смотрит в камеру; "
                       "мельче порога оно не судится вовсе, и идентичность "
                       "перестанет проверяться.")
    px = got.measurements.get("face_px")
    detail = f"{px}px, поддерживает: {', '.join(got.supports)}"
    if got.warnings:
        detail += f" | {got.warnings[0][:90]}"
    return _ok("face", detail)


# ------------------------------------------------------------- шлюз (сеть)

def check_gateway() -> tuple:
    """Интерполяция идёт через шлюз — ключ должен быть жив ДО генерации.

    Нужен только пути `--engine chain`. Локальная генерация наружу не ходит
    вовсе, и там эта проверка пропускается намеренно: прогон, спотыкающийся о
    недоступный шлюз, противоречит продуктовому требованию.
    """
    try:
        import requests
    except ImportError:
        return _fail("gateway", "нет requests: python3 -m pip install requests "
                                "(либо гнать локальный путь --engine "
                                "animatediff и --skip-gateway)")
    from . import pollinations

    try:
        pollinations._key()
    except Exception as e:  # noqa: BLE001
        return _fail("gateway", f"{e}. Лечение: export "
                                f"POLLINATIONS_API_KEY=... — без ключа "
                                f"цепочку нечем сшить.")
    try:
        r = requests.get(pollinations._base() + "/account/key",
                         headers=pollinations._auth(), timeout=30)
    except Exception as e:  # noqa: BLE001
        return _unknown("gateway",
                        f"{type(e).__name__}: {e} — шлюз НЕ ПРОВЕРЕН, а не "
                        f"«мёртв». Сеть недоступна или закрыта прокси: "
                        f"проверить curl -sS {pollinations._base()}/v1/models. "
                        f"Локальный путь (--engine animatediff) шлюз не "
                        f"использует вовсе.")
    if r.status_code != 200:
        return _fail("gateway", f"/account/key -> {r.status_code}: ключ "
                                f"недействителен, цепочку будет нечем сшить. "
                                f"Проверить POLLINATIONS_API_KEY (401/403 — "
                                f"не тот ключ, 404 — не тот базовый адрес).")
    return _ok("gateway", f"ключ валиден ({r.json().get('name')})")


# ------------------------------------------------------------------- прогон

def plan_size(vram_gb: float | None) -> tuple:
    """(размер кадра, число кадров) по плану — чтобы судить условия по нему.

    Берётся у `animate.plan`, а не считается заново: два расчёта одного и того
    же расходятся, и расходятся молча.
    """
    if not vram_gb:
        return None, None
    try:
        from .animate import plan

        cfg = plan(vram_gb, waist_up=False)
        return (cfg.width, cfg.height), cfg.frames
    except Exception:  # noqa: BLE001 — animate правится параллельно
        return None, None


def build_checks(args) -> list:
    """Проверки, сгруппированные по цене. Внутри группы — все, потом решение.

    ПОЧЕМУ ГРУППАМИ, А НЕ ПО ОДНОЙ. Останавливаться на первом же отказе внутри
    миллисекундной группы — значит показать одну беду из пяти и заставить
    запускать предполёт заново пять раз. Всё, что стоит миллисекунды,
    выполняется целиком; дорогое не начинается, пока дешёвое красное.
    """
    size, frames = plan_size(args.vram)
    return [
        ("миллисекунды", [
            ("пакеты", check_packages),
            ("диск", lambda: check_disk(args.out)),
            ("ffmpeg", check_ffmpeg),
            ("веса", check_weights),
            ("модель позы", check_pose_model),
            ("веса лица", check_face_model),
            ("сегментация", check_segmentation),
            ("DWPose", lambda: check_dwpose(args.dwpose)),
        ]),
        ("доли секунды", [
            ("драйвер", check_smi),
            ("условия", lambda: check_conditions(args.conditions,
                                                 expect_size=size,
                                                 need_frames=frames)),
            ("driving-кадры", lambda: check_driving(args.conditions)),
        ]),
        ("секунды", [
            ("torch", check_torch),
            ("драйвер/сборка", check_driver_build),
            ("vram", check_vram),
            ("onnxruntime", check_onnx_providers),
        ]),
        ("секунды дорогие", [
            ("лицо", lambda: check_face(args.face)),
        ]),
        ("сеть", ([] if args.skip_gateway else [("шлюз", check_gateway)])),
    ]


def main(argv: list) -> int:
    import argparse
    import time

    ap = argparse.ArgumentParser(
        prog="ball_reel.preflight_gpu",
        description="проверить машину до первой генерации: от миллисекундных "
                    "проверок к секундным, каждая с лечением")
    ap.add_argument("--conditions", default="conditions")
    ap.add_argument("--face", default="face.jpg")
    ap.add_argument("--out", default=".",
                    help="куда лягут кадры и отчёты — на этом диске тоже "
                         "проверяется место")
    ap.add_argument("--vram", type=float, default=None,
                    help="ГБ видеопамяти: по нему считается план, а по плану "
                         "судится размер условий")
    ap.add_argument("--dwpose", action="store_true",
                    help="условия будут сниматься здесь же — тогда веса DWPose "
                         "обязательны, а не опциональны")
    ap.add_argument("--skip-gateway", action="store_true",
                    help="не проверять ключ Pollinations (локальный путь "
                         "наружу не ходит)")
    args = ap.parse_args(argv)

    failed, unknown, passed = [], [], 0
    for tier, checks in build_checks(args):
        for label, fn in checks:
            t0 = time.time()
            try:
                ok, _, detail = fn()
            except Exception as e:  # noqa: BLE001 — предполёт сообщает, а не падает
                ok, detail = False, (f"{type(e).__name__}: {e} — САМА ПРОВЕРКА "
                                     f"сломалась, это дефект предполёта, а не "
                                     f"машины")
            ms = (time.time() - t0) * 1000
            mark = "НЕПРОВЕРЕНО" if ok is None else ("PASS" if ok else "FAIL")
            print(f"{mark:<12}{label:<15}{ms:7.0f} мс  {detail}")
            if ok is None:
                unknown.append(label)
            elif ok:
                passed += 1
            else:
                failed.append(label)
        if failed:
            print(f"\nОСТАНОВЛЕНО на группе «{tier}»: {', '.join(failed)}. "
                  f"Дальше проверять дороже, чем починить это. Всё, что стоило "
                  f"меньше, проверено и напечатано выше.")
            if unknown:
                print(f"НЕПРОВЕРЕНО (это не «в порядке»): "
                      f"{', '.join(unknown)}")
            return 1

    if unknown:
        print(f"\nПРОШЛО {passed}, НЕПРОВЕРЕНО {len(unknown)}: "
              f"{', '.join(unknown)}.\nЭто НЕ готовность: непроверенное не "
              f"имеет права показываться галочкой. Решать по каждой строке "
              f"выше — годится ли прогон с этой дырой.")
        return 2
    print(f"\nВСЁ ГОТОВО ({passed}/{passed}), непроверенного нет. Следующий "
          f"шаг — дым (run_local --smoke), а не полный прогон: он собирает "
          f"веса и печатает, что реально прицеплено к модели.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
