"""Предполёт на арендованной машине: упасть дёшево, до генерации.

    python3 -m ball_reel.preflight_gpu --conditions conditions --face face.jpg

Счётчик аренды тикает с первой секунды, поэтому самый дорогой ресурс — не VRAM,
а время до первой ошибки. Проверки идут от самых дешёвых и самых вероятных к
дорогим, каждая печатает результат и причину, и первая же неудача останавливает
всё: запускать генерацию на машине, где не хватает диска или не загрузился
адаптер лица, — значит платить за то, чтобы узнать это позже.

Ничего не генерирует и не тратит баланс.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Ниже этого генерация не поедет даже со всеми экономиями (см. gpu_keyframes).
MIN_VRAM_GB = 3.5
#: Веса ~7 ГБ плюс кэш и выдача.
MIN_DISK_GB = 15


def _ok(name: str, detail: str = "") -> tuple:
    return True, name, detail


def _fail(name: str, detail: str) -> tuple:
    return False, name, detail


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
    from .device import detect

    dev = detect()
    backend = getattr(torch, dev, None) if dev != "cpu" else None
    name = ""
    if backend is not None:
        try:
            name = backend.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = dev
    ok, detail = torch_verdict(torch.__version__, dev != "cpu", name,
                               device=dev)
    return (_ok if ok else _fail)(f"torch.{dev}", detail)


def torch_verdict(version: str, accelerator_available: bool, device: str,
                  *, device_kind: str = "cuda") -> tuple:
    """Годится ли эта сборка torch. Отдельно от импорта, чтобы проверялось.

    Различает два разных провала, потому что чинятся они по-разному. `+cpu`
    в версии — это CPU-сборка, и никакой драйвер её не оживит; отсутствие
    карты при CUDA-сборке — это уже про драйвер или про саму машину.

    Первый случай не гипотетический: `torch>=2.6` в requirements-файле без
    индекса даёт на Windows именно его, молча перекрывая правильную установку.

    `device_kind` — что нашлось (cuda/xpu/cpu). Ускоритель у Intel называется
    `xpu`, и совет «проверь nvidia-smi» там был бы вредным.
    """
    if accelerator_available:
        return True, f"{device or device_kind}, torch {version}"
    if "+cpu" in version:
        return False, (
            f"torch {version} — это CPU-сборка, карты она не увидит никогда. "
            f"Поставить из индекса под свою CUDA: pip uninstall -y torch, "
            f"затем pip install torch --index-url "
            f"https://download.pytorch.org/whl/cu126 (версию индекса подобрать "
            f"на pytorch.org). Внимание: pip install -r с обычной строкой "
            f"torch вернёт CPU-колесо обратно.")
    return False, (
        f"torch {version} карту не видит (ни cuda, ни xpu). Для NVIDIA — "
        f"смотреть драйвер (nvidia-smi) и совпадает ли он с CUDA сборки; для "
        f"Intel Arc нужна сборка с поддержкой XPU (PyTorch 2.5+) и драйверы "
        f"Level Zero. Пока карты нет, всё поедет на процессоре.")


def vram_verdict(total_gb: float) -> tuple:
    """Хватает ли карты. Отдельно от чтения карты, чтобы решение проверялось.

    Пока порог сравнивался прямо внутри check_vram, тронуть его было нечем:
    функция требует torch, а его в среде разработки нет, и мутационный аудит
    показал это — снятый MIN_VRAM_GB не ронял ни одного теста.
    """
    if total_gb < MIN_VRAM_GB:
        return False, (f"{total_gb:.1f} ГБ — меньше {MIN_VRAM_GB} ГБ, нужных "
                       f"даже на 512x768 со всеми экономиями. Взять карту "
                       f"больше или уйти на 448x640.")
    return True, f"{total_gb:.1f} ГБ"


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


def check_disk(path: str = ".") -> tuple:
    import shutil

    free = shutil.disk_usage(path).free / 1024 ** 3
    if free < MIN_DISK_GB:
        return _fail("disk", f"свободно {free:.1f} ГБ, нужно ~{MIN_DISK_GB}: "
                             f"веса не докачаются, и это выяснится в середине.")
    return _ok("disk", f"{free:.1f} ГБ свободно")


def check_weights() -> tuple:
    """Веса на месте — иначе первая же генерация уйдёт их качать."""
    import os

    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    if not home.exists():
        return _fail("weights", f"нет кэша {home}: скачать заранее "
                                f"(см. GPU_RUNBOOK.md, шаг 2)")
    size = sum(f.stat().st_size for f in home.rglob("*") if f.is_file())
    gb = size / 1024 ** 3
    if gb < 3:
        return _fail("weights", f"в кэше всего {gb:.1f} ГБ — скачивание не "
                                f"закончилось. Дождаться фоновой качки.")
    return _ok("weights", f"{gb:.1f} ГБ в {home}")


def check_pose_model() -> tuple:
    from .pose import _model_path

    try:
        return _ok("pose model", str(_model_path()))
    except RuntimeError as e:
        return _fail("pose model", str(e).split("\n")[0])


def check_conditions(conditions: str) -> tuple:
    """Условия должны существовать, читаться, быть одного размера, непустыми
    и идти подряд.

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
    """
    d = Path(conditions)
    files = sorted(d.glob("*.png")) if d.is_dir() else []
    if not files:
        return _fail("conditions", f"в {d} нет png. Отрендерить ДОМА: "
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
                                   f"({broken[:3]}): докопировать условия.")
    if len(sizes) > 1:
        return _fail("conditions", f"разные размеры {sizes}: ControlNet ждёт "
                                   f"один размер на всю последовательность.")
    if empty:
        return _fail("conditions",
                     f"{len(empty)} пустых условия ({empty[:3]}): в этих кадрах "
                     f"скелет не найден, генератор там не ограничен.")
    gaps = _numbering_gaps([f.stem for f in files])
    if gaps:
        return _fail("conditions",
                     f"пропуски в нумерации после {gaps[:3]}: в этих кадрах "
                     f"позы не нашли, и шаг по времени между условиями больше "
                     f"не одинаковый — цепочка поедет неровно. Перерендерить "
                     f"сегмент без дырок или резать по ним на куски.")
    return _ok("conditions", f"{len(files)} шт., {sizes.pop()}")


def _numbering_gaps(stems: list) -> list:
    """Имена, после которых номер прыгнул. Ненумерованные имена не трогаем."""
    nums = []
    for s in stems:
        if not s.isdigit():
            return []
        nums.append((int(s), s))
    return [name for (a, name), (b, _) in zip(nums, nums[1:]) if b != a + 1]


def check_face(face: str) -> tuple:
    """То же, что intake делает дома — на случай, если сюда приехало не то фото."""
    p = Path(face)
    if not p.exists():
        return _fail("face", f"нет файла {p}")
    try:
        from .intake import inspect
    except Exception as e:  # noqa: BLE001
        return _fail("face", f"не импортировать intake: {e}")
    got = inspect(str(p))
    if not got.usable:
        return _fail("face", got.blocked.get("identity", "лицо не найдено"))
    px = got.measurements.get("face_px")
    detail = f"{px}px, поддерживает: {', '.join(got.supports)}"
    if got.warnings:
        detail += f" | {got.warnings[0][:90]}"
    return _ok("face", detail)


def check_gateway() -> tuple:
    """Интерполяция идёт через шлюз — ключ должен быть жив ДО генерации."""
    try:
        import requests

        from . import pollinations

        r = requests.get(pollinations._base() + "/account/key",
                         headers=pollinations._auth(), timeout=30)
        if r.status_code != 200:
            return _fail("gateway", f"/account/key -> {r.status_code}: ключ "
                                    f"недействителен, цепочку будет нечем сшить")
        return _ok("gateway", f"ключ валиден ({r.json().get('name')})")
    except Exception as e:  # noqa: BLE001
        return _fail("gateway", f"{type(e).__name__}: {e}")


def main(argv: list) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="ball_reel.preflight_gpu",
        description="проверить арендованную машину до первой генерации")
    ap.add_argument("--conditions", default="conditions")
    ap.add_argument("--face", default="face.jpg")
    ap.add_argument("--skip-gateway", action="store_true",
                    help="не проверять ключ Pollinations")
    args = ap.parse_args(argv)

    checks = [
        ("диск", lambda: check_disk(".")),
        ("torch", check_torch),
        ("onnxruntime", check_onnx_providers),
        ("vram", check_vram),
        ("веса", check_weights),
        ("модель позы", check_pose_model),
        ("условия", lambda: check_conditions(args.conditions)),
        ("лицо", lambda: check_face(args.face)),
    ]
    if not args.skip_gateway:
        checks.append(("шлюз", check_gateway))

    failed = 0
    for label, fn in checks:
        try:
            ok, name, detail = fn()
        except Exception as e:  # noqa: BLE001 — предполёт сообщает, а не падает
            ok, name, detail = False, label, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {label:<12} {detail}")
        if not ok:
            failed += 1
            # Дальше проверять нет смысла: следующие всё равно упрутся в это.
            print("\nОСТАНОВЛЕНО. Чинить это и запускать предполёт заново — "
                  "генерация на непроверенной машине стоит дороже.")
            return 1
    print(f"\nВСЁ ГОТОВО ({len(checks)}/{len(checks)}). Следующий шаг — ОДИН "
          f"кейфрейм на дым, не полный прогон (GPU_RUNBOOK.md, шаг 4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
