#!/usr/bin/env python3
"""Сядет ли LoRA, обученная на одной модели Wan, на другую. БЕЗ СКАЧИВАНИЯ ВЕСОВ.

ЗАЧЕМ. Вендор предостерегает: «If you're using Wan-Animate, we do not recommend
using LoRA models trained on Wan2.2, since weight changes during training may
lead to unexpected behavior». Предостережение не говорит, ЧТО именно сломается —
не прицепится вовсе, прицепится частично или прицепится и испортит качество. Три
разных исхода с разной ценой, и различить их можно только замером.

КАК. LoRA садится на двумерные веса по ИМЕНИ модуля и требует совпадения ФОРМЫ.
И то и другое лежит в заголовке safetensors, который читается Range-запросом:
первые 8 байт — длина заголовка, дальше JSON с формой и типом каждого тензора.
Ни один шард целиком не качается — на всю проверку уходят сотни килобайт против
34 ГБ весов.

ЧЕГО ЭТОТ ПРИБОР НЕ ГОВОРИТ. Совпадение имён и форм означает «прицепится», а НЕ
«сработает как надо». Веса за одинаковыми именами разные, и предостережение
вендора именно про это. Сюда же: у Animate есть слои, которых у I2V нет вовсе
(face_adapter, motion_encoder, k_img/v_img, pose_patch_embedding) — их чужая
LoRA не тронет никогда, и это видно в отчёте отдельной строкой.
"""

from __future__ import annotations

import json
import struct
import sys
import urllib.request

HF = "https://huggingface.co"


def _get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def header(url: str) -> dict:
    """Заголовок safetensors. Две пробы вместо скачивания файла."""
    n = struct.unpack("<Q", _get(url, 0, 7))[0]
    d = json.loads(_get(url, 8, 7 + n).decode("utf-8"))
    d.pop("__metadata__", None)
    return {k: (tuple(v["shape"]), v["dtype"]) for k, v in d.items()}


def model_tensors(repo: str, subfolder: str, shards: int, total: int) -> dict:
    out = {}
    for i in range(1, shards + 1):
        name = f"diffusion_pytorch_model-{i:05d}-of-{total:05d}.safetensors"
        path = f"{subfolder}/{name}" if subfolder else name
        out.update(header(f"{HF}/{repo}/resolve/main/{path}"))
    return out


def fit(donor: dict, target: dict) -> dict:
    """Три числа, а не флаг: сколько сядет, сколько не сядет, чего донор не знает."""
    two_d = {k: v for k, v in donor.items()
             if k.endswith(".weight") and len(v[0]) == 2}
    lands, shape_clash, absent = [], [], []
    for k, (shape, _) in two_d.items():
        if k not in target:
            absent.append(k)
        elif target[k][0] != shape:
            shape_clash.append((k, shape, target[k][0]))
        else:
            lands.append(k)
    untouched = sorted(k for k, v in target.items()
                       if k.endswith(".weight") and len(v[0]) == 2
                       and k not in donor)
    return {
        "donor_2d": len(two_d), "lands": len(lands),
        "shape_clash": shape_clash, "absent": absent,
        "target_untouched": untouched,
        "note": (f"двумерных весов у донора {len(two_d)}: сядет {len(lands)}, "
                 f"форма не сойдётся у {len(shape_clash)}, имени нет в цели у "
                 f"{len(absent)}. У цели останутся нетронутыми {len(untouched)} "
                 f"весов, которых у донора нет вовсе"),
    }


if __name__ == "__main__":
    print("донор: Wan2.2-I2V-A14B (high_noise)  цель: Wan2.2-Animate-14B")
    donor = model_tensors("Wan-AI/Wan2.2-I2V-A14B", "high_noise_model", 6, 6)
    target = model_tensors("Wan-AI/Wan2.2-Animate-14B", "", 4, 4)
    got = fit(donor, target)
    print(got["note"])
    for k, a, b in got["shape_clash"][:10]:
        print(f"  форма: {k} донор {a} цель {b}")
    for k in got["absent"][:10]:
        print(f"  нет имени: {k}")
    sys.exit(0 if got["lands"] and not got["shape_clash"] else 1)
