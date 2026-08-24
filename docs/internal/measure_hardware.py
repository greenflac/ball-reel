"""Пересчёт всех чисел из `docs/internal/FORK_HARDWARE.md`. Не модуль пакета: скрипт.

    python3 docs/internal/measure_hardware.py            # всё
    python3 docs/internal/measure_hardware.py --offline   # только арифметика, без сети

Существует потому, что число без команды устаревает молча. В первой редакции
плана стояла «рекомендация 48 ГБ VRAM» — она была выведена из размера файла
весов и выдана за замер. Этот скрипт есть ответ на вопрос «чем проверить»,
чтобы такое возражение можно было закрыть за минуту, а не спором.

СЕТЕВАЯ ЧАСТЬ читает только HTTP-заголовки и заголовок safetensors через
Range-запрос: 195 КБ вместо 17 ГБ. Скачивания весов здесь нет.
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
from collections import defaultdict

GB = 2 ** 30

FP8_URL = ("https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/"
           "Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors")

#: Репозитории, чью лицензию надо знать до сборки. Проверять МОДЕЛЬНЫМ
#: эндпойнтом: поисковый (`?search=`) не отдаёт `cardData` и врёт `None` даже
#: для официальных репозиториев. На этом я один раз уже ошибся.
LICENCE_TARGETS = (
    "Wan-AI/Wan2.2-Animate-14B",
    "Kijai/WanVideo_comfy_fp8_scaled",
    "Kijai/WanVideo_comfy",
    "QuantStack/Wan2.2-Animate-14B-GGUF",
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
    "lightx2v/Wan2.2-Distill-Loras",
)

#: Из ЗАМЕРЕННОГО заголовка safetensors, а не из статьи.
DIM, FFN = 5120, 13824
PATCH = (1, 2, 2)

#: Архитектура Wan VAE. НЕ из заголовка — это допущение, и оно названо.
VAE_SPATIAL, VAE_TEMPORAL = 8, 4

#: Из исходников Comfy: `minimum_inference_memory()` = 0.8 ГБ +
#: `extra_reserved_memory()`, где последнее 0.40 ГБ на Linux.
COMFY_RESERVE_GB = 0.8 + 0.4


def _curl(*args: str) -> bytes:
    return subprocess.run(("curl", "-sL", "--max-time", "90") + args,
                          capture_output=True).stdout


def safetensors_header(url: str) -> dict:
    """Заголовок без скачивания файла: длина, потом сам JSON."""
    n = struct.unpack("<Q", _curl("-H", "Range: bytes=0-7", url)[:8])[0]
    raw = _curl("-H", f"Range: bytes=8-{8 + n - 1}", url)[:n]
    hdr = json.loads(raw.decode("utf-8"))
    hdr.pop("__metadata__", None)
    return hdr


def describe_weights(hdr: dict) -> None:
    dt_bytes, blocks, dtypes, total = defaultdict(int), defaultdict(int), defaultdict(int), 0
    for key, spec in hdr.items():
        lo, hi = spec["data_offsets"]
        size = hi - lo
        total += size
        dtypes[spec["dtype"]] += size
        m = re.match(r"(blocks\.\d+)\.", key)
        blocks[m.group(1) if m else "не блок: " + key.split(".")[0]] += size
    del dt_bytes

    print(f"тензоров: {len(hdr)}   сумма: {total / GB:.2f} GB")
    for name, size in sorted(dtypes.items(), key=lambda x: -x[1]):
        print(f"   {name:10s} {size / GB:7.3f} GB")

    blk = [v for k, v in blocks.items() if k.startswith("blocks.")]
    oth = {k: v for k, v in blocks.items() if not k.startswith("blocks.")}
    print(f"\nтрансформерных блоков: {len(blk)}")
    if blk:
        print(f"   каждый блок     {min(blk) / GB:7.4f} .. {max(blk) / GB:7.4f} GB")
        print(f"   все вместе      {sum(blk) / GB:7.3f} GB")
    print(f"   неблочная часть {sum(oth.values()) / GB:7.4f} GB")
    for k, v in sorted(oth.items(), key=lambda x: -x[1])[:4]:
        print(f"      {k:26s} {v / GB:7.4f} GB")
    return sum(oth.values()) / GB, (max(blk) / GB if blk else 0.0)


def tokens(width: int, height: int, frames: int) -> tuple:
    lw, lh = width // VAE_SPATIAL, height // VAE_SPATIAL
    lf = (frames - 1) // VAE_TEMPORAL + 1
    return lf * (lh // PATCH[1]) * (lw // PATCH[2]), lf, lw, lh


def activations_gb(n_tokens: int) -> float:
    """Пик на один блок при ЭФФЕКТИВНОМ внимании — O(N), не O(N**2).

    Допущение названо, потому что оно решающее: без SDPA/Sage память растёт
    квадратично по числу токенов, и вся таблица ниже теряет силу. Четыре
    слагаемых по `dim` — q, k, v и выход; одно по `ffn` — промежуточный тензор.
    """
    return (4 * n_tokens * DIM + n_tokens * FFN) * 2 / GB


def budget_table(nonblock_gb: float, block_gb: float) -> None:
    print(f"\n{'размер':>12} {'кадров':>7} {'токенов':>9} {'актив./блок':>12}")
    # Портрет, потому что драйвинг продукта портретный: demo/kit/driving —
    # 720x1278, отношение 0.563. Все размеры кратны 16 (требование ноды).
    for w, h, f in ((480, 848, 77), (480, 848, 49), (448, 800, 77),
                    (416, 736, 77), (640, 640, 77)):
        n, *_ = tokens(w, h, f)
        print(f"{f'{w}x{h}':>12} {f:>7} {n:>9,} {activations_gb(n):>11.2f} GB")

    print("\nпотолок весов при поблочной выгрузке:")
    for k in (1, 4, 8, 40):
        tail = "  <- всё целиком" if k == 40 else ""
        print(f"   неблочное + {k:2d} блок(ов)  {nonblock_gb + k * block_gb:6.2f} GB{tail}")

    quants = (("Q3_K_M", 8.04), ("Q4_K_M", 10.71), ("Q5_K_M", 12.11),
              ("Q6_K", 13.60), ("Q8_0", 17.43))
    n77, *_ = tokens(480, 848, 77)
    n49, *_ = tokens(480, 848, 49)
    print(f"\nбюджет 16 ГБ при 480x848 (резерв Comfy {COMFY_RESERVE_GB:.1f} GB):")
    print(f"{'квант':>8} {'веса':>8} {'77 кадров':>9} {'':>6}  {'49 кадров':>8} {'':>6}")
    for name, gb in quants:
        t77 = gb + activations_gb(n77) + COMFY_RESERVE_GB
        t49 = gb + activations_gb(n49) + COMFY_RESERVE_GB
        mark = lambda t: "OK " if t < 14.0 else ("тесно" if t < 16.0 else "НЕТ")
        print(f"{name:>8} {gb:>7.2f}  {t77:>8.2f} {mark(t77):>6}  {t49:>8.2f} {mark(t49):>6}")


def licences() -> None:
    print("\nлицензии (модельный эндпойнт, не поисковый):")
    for repo in LICENCE_TARGETS:
        out = _curl(f"https://huggingface.co/api/models/{repo}")
        try:
            lic = (json.loads(out).get("cardData") or {}).get("license")
        except Exception:
            lic = "не прочитано"
        flag = "OK " if lic else "!! "
        print(f"   {flag}{repo:42s} {lic}")
    print("   !! означает: лицензия НЕ объявлена. Для коммерческого продукта")
    print("      это блокер, а не мелочь — см. FORK_HARDWARE.md §6.")


def main() -> int:
    offline = "--offline" in sys.argv
    if offline:
        print("--offline: заголовок не читается, беру ЗАМЕРЕННЫЕ ранее величины")
        nonblock, block = 2.0885, 0.3762
    else:
        nonblock, block = describe_weights(safetensors_header(FP8_URL))
    budget_table(nonblock, block)
    if not offline:
        licences()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
