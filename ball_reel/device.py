"""Какая карта под нами — и на что это влияет, кроме имени строки.

Модуль появился, когда выяснилось, что доступна Intel Arc A580. Весь стек до
этого был прибит к CUDA примерно в двадцати местах: `torch.cuda.is_available`,
`device="cuda"`, `CUDAExecutionProvider`, `ctx_id=0`. На карте Intel всё это
не просто не ускорится — оно тихо свалится на CPU или упадёт, а разбираться
придётся уже на машине, где считается время.

ТРИ ВЕЩИ, КОТОРЫЕ ЗДЕСЬ РЕШАЮТСЯ.

**Имя устройства для torch.** У Intel это `xpu` (нативная поддержка в
PyTorch 2.5+), у NVIDIA `cuda`, иначе `cpu`. Проверять надо именно наличие
атрибута И доступность: `torch.xpu` существует в сборках, где карты нет.

**Провайдер для onnxruntime.** insightface и DWPose ходят не через torch, а
через onnxruntime, и у него свой список провайдеров. Для Intel это OpenVINO
или DirectML, а НЕ CUDA. Запрошенный, но отсутствующий провайдер onnxruntime
принимает молча и уходит на CPU — мы это уже наблюдали на замерах латентности,
где `CUDAExecutionProvider` был запрошен и проигнорирован. Поэтому здесь
провайдер не просто выбирается, а сверяется с тем, что рантайм реально умеет.

**Тип данных.** fp16 на CPU медленнее fp32 и местами не поддержан, поэтому
дефолт зависит от устройства, а не берётся константой.

ЛОЖНАЯ ТРЕВОГА ЗДЕСЬ ХУЖЕ МОЛЧАНИЯ. Модуль диагностический: по его строке
решают, на какой карте считать, и «упадём на CPU», сказанное при рабочем
ускорении, стоит дороже, чем несказанное вовсе. Отсюда два правила, которые
пришлось чинить постфактум: провайдеры внутри устройства — альтернативы, и
жаловаться можно только когда нет НИ ОДНОГО; а неопрашиваемый torch — это не
отсутствующий torch, и в отчёте они обязаны выглядеть по-разному.

**Драйвер против сборки.** `nvidia-smi` печатает в шапке версию CUDA, которую
поддерживает драйвер. Если она НИЖЕ той, под которую собран torch, всё падает
непонятно — обычно на первом же вызове ядра, сообщением про «no kernel image is
available for execution on the device» или просто `cuda=False`. Это правило
живёт здесь исполняемым кодом (`smi_probe`, `driver_covers`), а не абзацем в
рунбуке: абзац читают после того, как отказ уже случился.

СОСТОЯНИЕ: логика выбора проверена тестами, но НА КАРТЕ INTEL НЕ ИСПОЛНЯЛАСЬ
НИ РАЗУ — в среде разработки нет ни одной карты. Первый запуск на A580 и есть
проверка; расхождения дописывать сюда. По той же причине `smi_probe` разобран
на «запустить» и «разобрать вывод»: разбор проверяется на записанных строках
`nvidia-smi` дома, а запуск НЕПРОВЕРЕН до первой машины с картой.
"""

from __future__ import annotations

import re

#: Порядок предпочтения. CUDA первой не из симпатии, а потому что вся
#: экосистема (diffusers, xformers, bitsandbytes) на ней проверена лучше всех.
DEVICE_ORDER = ("cuda", "xpu", "mps", "cpu")

#: Провайдеры onnxruntime под каждое устройство, от быстрого к запасному.
#: CPUExecutionProvider обязан замыкать каждый список: рантайм молча
#: игнорирует недоступный провайдер, и без явного запасного получится «выбрали
#: ускоритель, считаем на процессоре и не знаем об этом».
#:
#: Список — это ЗАПРОС целиком: onnxruntime сам возьмёт первый, который у него
#: есть. Просить надо всё сразу, а не угаданное — не подхватит OpenVINO,
#: подхватит DirectML.
ONNX_PROVIDERS = {
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    # У Intel два пути: OpenVINO кроссплатформенно, DirectML только Windows.
    # Оба ставятся отдельными пакетами onnxruntime-*, и если не поставлены —
    # останется CPU, о чём `onnx_providers` честно сообщит.
    "xpu": ["OpenVINOExecutionProvider", "DmlExecutionProvider",
            "CPUExecutionProvider"],
    "mps": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}

CPU_PROVIDER = "CPUExecutionProvider"

#: Кто из списка выше ускоряет — и это АЛЬТЕРНАТИВЫ, а не комплект: хватает
#: ЛЮБОГО ОДНОГО. Разница не косметическая. DirectML существует только на
#: Windows, поэтому на линуксовой Arc A580 с рабочим OpenVINO трактовка «нужны
#: все» давала вечное «упадём на CPU» при живом ускорении. Ложная тревога в
#: диагностическом модуле хуже молчания: по ней принимают решение о железе, а
#: отличить по ней «ускорения нет» от «второго варианта нет» нельзя.
#:
#: Выводится из ONNX_PROVIDERS, а не пишется рядом руками: два списка одного и
#: того же расходятся, и разойдутся они молча.
ONNX_ACCELERATORS = {
    dev: tuple(p for p in chain if p != CPU_PROVIDER)
    for dev, chain in ONNX_PROVIDERS.items()
}

#: Состояния torch. Их именно три, и «нет пакета» с «пакет есть, но карта не
#: отвечает» — разные беды с разным лечением: первая ставится pip'ом, вторая
#: только драйвером или другой сборкой. См. `torch_state`.
TORCH_ABSENT = "absent"
TORCH_SILENT = "silent"
TORCH_OK = "ok"

#: ctx_id для insightface: 0 и выше — номер ускорителя, -1 — CPU. У него нет
#: понятия «xpu», поэтому на Intel он всё равно пойдёт через onnxruntime, и
#: единственный честный вариант — не врать ему про номер устройства.
INSIGHTFACE_GPU_DEVICES = ("cuda",)


def detect() -> str:
    """Какое устройство доступно torch прямо сейчас.

    Проверяется и наличие атрибута, и доступность: `torch.xpu` существует в
    сборках без карты, а `torch.cuda` — в CPU-сборках, где `is_available()`
    возвращает False. Проверять только атрибут значит выбрать устройство,
    которого нет.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    for name in DEVICE_ORDER:
        if name == "cpu":
            continue
        backend = getattr(torch, name, None)
        try:
            if backend is not None and backend.is_available():
                return name
        except Exception:  # noqa: BLE001 — бэкенд есть, но сломан: не наш
            continue
    return "cpu"


def dtype_for(device: str) -> str:
    """Тип данных под устройство, именем — чтобы попадало в JSON-отчёт.

    fp16 на CPU не ускоряет, а замедляет: у процессора нет соответствующих
    блоков, и половинная точность эмулируется. Ставить его «на всякий случай»
    значит платить временем за ничего.
    """
    return "float32" if device == "cpu" else "float16"


def onnx_providers(device: str, available: list | None = None) -> tuple:
    """(что просим, чего не хватает). Пустой второй — жаловаться не на что.

    Второй элемент существует, потому что onnxruntime **молча** игнорирует
    недоступный провайдер и уходит на CPU. Мы это уже поймали на замерах
    латентности: DWPose просил CUDA, получал предупреждение в stderr и считал
    438 мс на процессоре. Молчаливая деградация в десять раз — это то, о чём
    предполёт обязан сказать вслух.

    Но сказать он обязан ровно тогда, когда деградация есть. Провайдеры внутри
    устройства — альтернативы (OpenVINO ИЛИ DirectML), поэтому «не хватает»
    заполняется только если не доступен НИ ОДИН из них; тогда в нём лежат все
    варианты сразу — не как список претензий, а как список того, что можно
    поставить. Хватило хотя бы одного — тревоги нет, и просим мы всё равно
    полный список: выбор из него делает сам рантайм.
    """
    want = ONNX_PROVIDERS.get(device, ONNX_PROVIDERS["cpu"])
    if available is None:
        try:
            import onnxruntime  # type: ignore

            available = list(onnxruntime.get_available_providers())
        except ImportError:
            available = []
    alternatives = ONNX_ACCELERATORS.get(device, ())
    if any(p in available for p in alternatives):
        return tuple(want), ()
    return tuple(want), tuple(alternatives)


def insightface_ctx(device: str) -> int:
    """ctx_id для insightface. -1 значит CPU, и это не всегда поражение.

    На Intel insightface не умеет ускоряться напрямую: он ходит через
    onnxruntime, и номер устройства ему передавать бессмысленно. Честнее
    отдать -1, чем указать несуществующий ускоритель и получить молчаливый
    откат внутри чужой библиотеки.
    """
    return 0 if device in INSIGHTFACE_GPU_DEVICES else -1


def empty_cache(device: str | None = None) -> None:
    """Освободить кэш ускорителя, если у него такой есть."""
    try:
        import torch  # type: ignore
    except ImportError:
        return
    backend = getattr(torch, device or detect(), None)
    fn = getattr(backend, "empty_cache", None)
    if callable(fn):
        fn()


def torch_state(device: str | None = None) -> tuple:
    """(состояние, версия, имя устройства, причина отказа опроса).

    Три исхода, и путать их нельзя, потому что чинят их по-разному:

    * ``TORCH_ABSENT`` — пакета нет. Лечится установкой.
    * ``TORCH_SILENT`` — пакет есть, а устройство не опрашивается. Типично для
      xpu-сборок и для неподнятого драйвера; в ``причине`` лежит текст
      исключения, потому что без него понятно только «что-то не так».
      Лечится драйвером или другой сборкой, но НЕ установкой.
    * ``TORCH_OK`` — опросили. Имя может быть пустым: у ``torch.cpu`` нет
      ``get_device_name``, и это не отказ, а отсутствие вопроса.

    Раньше все три сливались в «torch не установлен»: `describe` глотал любое
    исключение. Диагностика, которая уводит чинить не то, дороже отсутствия
    диагностики — за ней идут ставить уже стоящее.
    """
    device = device or detect()
    try:
        import torch  # type: ignore
    except ImportError:
        return TORCH_ABSENT, "", "", ""
    except Exception as e:  # noqa: BLE001 — пакет на месте, но не грузится
        return TORCH_SILENT, "", "", f"{type(e).__name__}: {e}"
    try:
        version = str(getattr(torch, "__version__", ""))
        backend = getattr(torch, device, None)
    except Exception as e:  # noqa: BLE001 — битая сборка отвечает на атрибуты
        return TORCH_SILENT, "", "", f"{type(e).__name__}: {e}"
    reason = ""
    for attr in ("get_device_name", "get_device_properties"):
        fn = getattr(backend, attr, None)
        if not callable(fn):
            continue
        try:
            got = fn(0)
        except Exception as e:  # noqa: BLE001 — второй способ ещё может выжить
            reason = reason or f"{type(e).__name__}: {e}"
            continue
        name = got if isinstance(got, str) else getattr(got, "name", "")
        return TORCH_OK, version, str(name or ""), ""
    if reason:
        return TORCH_SILENT, version, "", reason
    return TORCH_OK, version, "", ""


#: Сколько ждём nvidia-smi. Он отвечает за десятки миллисекунд, но на машине
#: с повисшим драйвером висит вечно, а предполёт обязан быть дешевле того, что
#: он сторожит. ВЫБРАНО.
SMI_TIMEOUT_S = 10

#: Исходы сравнения «драйвер против сборки». Их четыре, и «не смогли узнать»
#: здесь такой же полноправный, как остальные три: nvidia-smi нет на Intel, на
#: Apple и в контейнере без проброса, и объявлять это совпадением версий —
#: ровно тот дефект, из-за которого непроверенное показывается галочкой.
DRIVER_OK = "ok"
DRIVER_OLD_MAJOR = "old_major"
DRIVER_OLD_MINOR = "old_minor"
DRIVER_UNKNOWN = "unknown"


def version_pair(text) -> tuple | None:
    """'13.0' -> (13, 0). Не число — None, а не ноль.

    Ноль был бы худшим вариантом: он сравнивается, и любая нераспознанная
    строка молча превращалась бы в «драйвер древний».
    """
    if not isinstance(text, str):
        return None
    m = re.match(r"\s*(\d+)(?:\.(\d+))?", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


def smi_cuda(text: str) -> str | None:
    """Версия CUDA из шапки nvidia-smi. Именно её поддерживает драйвер.

    В шапке это выглядит так:

        | NVIDIA-SMI 550.54.14  Driver Version: 550.54.14  CUDA Version: 12.4 |

    Это НЕ версия, с которой собран torch, и путать их дорого: torch,
    собранный под 13.0, на драйвере 12.4 не поедет, а сообщение об этом
    приходит с той стороны, где его никто не связывает с драйвером.
    """
    m = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", text or "")
    return m.group(1) if m else None


def smi_cards(text: str) -> list:
    """Строки `--query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`.

    Разбор отделён от запуска, потому что запускать nvidia-smi в среде
    разработки не на чем, а ошибиться в разборе — легко: MiB против МБ,
    заголовок, который забыли выключить, локаль с запятой.
    """
    cards = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0] or parts[0].lower().startswith("name"):
            continue
        mib = version_pair(parts[1])
        cards.append({"name": parts[0],
                      "vram_gb": round(mib[0] * 1024 ** 2 / 1024 ** 3, 2)
                      if mib else None,
                      "driver": parts[2]})
    return cards


def smi_run(args: list, timeout: float = SMI_TIMEOUT_S) -> tuple:
    """(вывод, причина). Ровно одна из двух половин непустая.

    Причина возвращается текстом, а не выбрасывается: «nvidia-smi не найден» и
    «nvidia-smi повис» — это разные машины, и на предполёте их надо различать
    так же, как отсутствие пакета и мёртвый драйвер.
    """
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return "", "nvidia-smi не найден в PATH"
    try:
        r = subprocess.run([exe] + list(args), capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return "", (f"nvidia-smi не ответил за {timeout:.0f} c — обычно это "
                    f"повисший драйвер, а не медленная машина")
    except OSError as e:  # noqa: BLE001
        return "", f"nvidia-smi не запускается: {type(e).__name__}: {e}"
    if r.returncode != 0:
        return "", (f"nvidia-smi вернул {r.returncode}: "
                    f"{(r.stderr or r.stdout).strip().splitlines()[:1]}")
    return r.stdout, ""


def smi_probe(run=None) -> dict:
    """Что драйвер знает о карте: имя, память, версия драйвера и его CUDA.

    `run` подставляется тестом — иначе эта функция была бы неисполнима нигде,
    кроме машины с картой, а разбор её вывода тем самым непроверяем.

    Два вызова, а не один: версия CUDA есть ТОЛЬКО в шапке (её нельзя спросить
    через `--query-gpu`), а имя и память надёжно читаются только из csv —
    разбирать ASCII-таблицу шапки регулярками значит сломаться на первой же
    карте с длинным именем.
    """
    run = run or smi_run
    csv_out, csv_why = run(["--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader,nounits"])
    head_out, head_why = run([])
    cards = smi_cards(csv_out)
    cuda = smi_cuda(head_out)
    reason = ""
    if not cards and not cuda:
        reason = csv_why or head_why or "nvidia-smi ничего не сказал о картах"
    elif not cuda:
        reason = head_why or "в шапке nvidia-smi нет строки 'CUDA Version:'"
    return {"cards": cards, "cuda": cuda, "reason": reason}


def driver_covers(driver_cuda, build_cuda) -> str:
    """Потянет ли ДРАЙВЕР сборку torch. Четыре исхода, см. DRIVER_*.

    Правило ВЫБРАНО и вот из чего сложено:

    * старший номер драйвера ниже, чем у сборки (12.x против 13.0) — отказ.
      Это тот самый случай, где всё падает непонятно;
    * старший номер драйвера выше — всё в порядке, драйверы обратно
      совместимы со старыми сборками;
    * старший совпал, младший у драйвера ниже (12.4 против 12.6) — НЕ
      объявляем ни годным, ни негодным. У CUDA 11+ есть минорная
      совместимость, и обычно это едет, но проверить это здесь нечем:
      карты нет. Отдельный исход честнее выдуманного вердикта.
    """
    d, b = version_pair(driver_cuda), version_pair(build_cuda)
    if d is None or b is None:
        return DRIVER_UNKNOWN
    if d[0] < b[0]:
        return DRIVER_OLD_MAJOR
    if d[0] > b[0]:
        return DRIVER_OK
    return DRIVER_OK if d[1] >= b[1] else DRIVER_OLD_MINOR


def torch_build_cuda():
    """С какой CUDA собран torch, по `torch.version.cuda`. None — ни с какой.

    ЭТО ЕДИНСТВЕННЫЙ НАДЁЖНЫЙ ПРИЗНАК CPU-сборки. Суффикс `+cpu` в номере
    версии его не заменяет: колесо с PyPI суффикса не несёт вовсе, и проверка
    по строке молча пропускает ровно тот случай, ради которого написана. На
    этом проекте так уже терялась вся ветка генерации.

    Возвращает строку ('13.0'), None (собран без CUDA) или '' — «спросить не
    вышло», что не то же самое, что «собран без CUDA».
    """
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001 — нет пакета или битая сборка
        return ""
    try:
        return getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:  # noqa: BLE001
        return ""


def describe(device: str | None = None) -> str:
    """Одна строка про то, на чём считаем — для шапки отчёта.

    Число без железа бессмысленно, поэтому каждый отчёт этого проекта начинает
    с того, чем мерил. И по той же причине здесь нет обобщающих формулировок:
    три состояния torch выглядят по-разному, а «ускорения нет» пишется только
    тогда, когда его нет ни через один провайдер.
    """
    device = device or detect()
    state, version, name, reason = torch_state(device)
    line = f"устройство {device}" + (f" ({name})" if name else "")
    if state == TORCH_ABSENT:
        line += ", torch не установлен"
    elif state == TORCH_SILENT:
        line += (f", torch {version or 'неизвестной версии'} установлен, но "
                 f"устройство не опрашивается: {reason}")
    else:
        line += f", torch {version}"
    if device == "cuda" and state != TORCH_ABSENT:
        # Только для NVIDIA: у xpu-сборки `torch.version.cuda` пуст штатно, и
        # писать про CUDA владельцу Arc — уводить его чинить не то.
        built = torch_build_cuda()
        if built:
            line += f", собран с CUDA {built}"
        elif built is None:
            line += (", СОБРАН БЕЗ CUDA (torch.version.cuda пуст) — эта сборка "
                     "карту не увидит никогда, её надо переставить")
    line += f", dtype {dtype_for(device)}"
    _, missing = onnx_providers(device)
    if missing:
        line += (f" | ускорения нет ни через один из: {', '.join(missing)} — "
                 f"insightface и DWPose пойдут на CPU")
    return line
