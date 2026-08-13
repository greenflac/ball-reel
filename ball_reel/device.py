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

СОСТОЯНИЕ: логика выбора проверена тестами, но НА КАРТЕ INTEL НЕ ИСПОЛНЯЛАСЬ
НИ РАЗУ — в среде разработки нет ни одной карты. Первый запуск на A580 и есть
проверка; расхождения дописывать сюда.
"""

from __future__ import annotations

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
    line += f", dtype {dtype_for(device)}"
    _, missing = onnx_providers(device)
    if missing:
        line += (f" | ускорения нет ни через один из: {', '.join(missing)} — "
                 f"insightface и DWPose пойдут на CPU")
    return line
