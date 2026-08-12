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
        return _fail("torch", "не установлен: pip install torch --index-url "
                              "https://download.pytorch.org/whl/cu121")
    if not torch.cuda.is_available():
        return _fail("torch.cuda",
                     f"torch {torch.__version__} не видит карту. Обычно это "
                     f"сборка под другой CUDA: переустановить под драйвер "
                     f"(nvidia-smi покажет версию).")
    return _ok("torch.cuda", f"{torch.cuda.get_device_name(0)}, "
                             f"torch {torch.__version__}")


def check_vram() -> tuple:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return _fail("vram", "карта недоступна (см. torch.cuda)")
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception as e:  # noqa: BLE001
        return _fail("vram", f"не прочитать: {e}")
    if total < MIN_VRAM_GB:
        return _fail("vram", f"{total:.1f} ГБ — меньше {MIN_VRAM_GB} ГБ, "
                             f"нужных даже на 512x768 со всеми экономиями. "
                             f"Взять карту больше или уйти на 448x640.")
    return _ok("vram", f"{total:.1f} ГБ")


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
    """Условия должны существовать, быть непустыми и одного размера."""
    d = Path(conditions)
    files = sorted(d.glob("*.png")) if d.is_dir() else []
    if not files:
        return _fail("conditions", f"в {d} нет png. Отрендерить ДОМА: "
                                   f"skeleton.render_sequence (это CPU).")
    from PIL import Image

    sizes = set()
    empty = []
    for f in files:
        with Image.open(f) as im:
            sizes.add(im.size)
        if f.stat().st_size < 500:
            empty.append(f.name)
    if len(sizes) > 1:
        return _fail("conditions", f"разные размеры {sizes}: ControlNet ждёт "
                                   f"один размер на всю последовательность.")
    if empty:
        return _fail("conditions",
                     f"{len(empty)} пустых условия ({empty[:3]}): в этих кадрах "
                     f"скелет не найден, генератор там не ограничен.")
    return _ok("conditions", f"{len(files)} шт., {sizes.pop()}")


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
