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
    """(что просим, чего не хватает). Пустой второй элемент — всё на месте.

    Второй элемент существует, потому что onnxruntime **молча** игнорирует
    недоступный провайдер и уходит на CPU. Мы это уже поймали на замерах
    латентности: DWPose просил CUDA, получал предупреждение в stderr и считал
    438 мс на процессоре. Молчаливая деградация в десять раз — это то, о чём
    предполёт обязан сказать вслух.
    """
    want = ONNX_PROVIDERS.get(device, ONNX_PROVIDERS["cpu"])
    if available is None:
        try:
            import onnxruntime  # type: ignore

            available = list(onnxruntime.get_available_providers())
        except ImportError:
            available = []
    missing = [p for p in want
               if p != "CPUExecutionProvider" and p not in available]
    return tuple(want), tuple(missing)


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


def describe(device: str | None = None) -> str:
    """Одна строка про то, на чём считаем — для шапки отчёта.

    Число без железа бессмысленно, поэтому каждый отчёт этого проекта начинает
    с того, чем мерил.
    """
    device = device or detect()
    try:
        import torch  # type: ignore

        backend = getattr(torch, device, None)
        name = ""
        for attr in ("get_device_name", "get_device_properties"):
            fn = getattr(backend, attr, None)
            if callable(fn):
                got = fn(0)
                name = got if isinstance(got, str) else getattr(got, "name", "")
                break
        version = torch.__version__
    except Exception:  # noqa: BLE001
        name, version = "", "не установлен"
    _, missing = onnx_providers(device)
    line = f"устройство {device}" + (f" ({name})" if name else "")
    line += f", torch {version}, dtype {dtype_for(device)}"
    if missing:
        line += (f" | onnxruntime НЕ УМЕЕТ {', '.join(missing)} — "
                 f"insightface и DWPose пойдут на CPU")
    return line
