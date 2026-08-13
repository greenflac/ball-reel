"""Сделать из чистой рефки ту же рефку С ТАТУИРОВКОЙ. Управляемый эксперимент.

ЗАЧЕМ НЕ РЕДАКТИРОВАНИЕМ МОДЕЛЬЮ. Попросить генератор «добавь дракона на руку»
проще, но тогда он перерисует кадр целиком: сдвинутся поза, свет и лицо. Два
кадра будут отличаться не татуировкой, а всем сразу, и по такому сравнению
нельзя сказать, что именно поймал измеритель.

Здесь меняется РОВНО ОДНА переменная: два файла совпадают пиксель в пиксель
везде, кроме области татуировки. Это и есть условие, при котором ложные и
истинные срабатывания детектора сравнимы между собой.

ПОЧЕМУ УМНОЖЕНИЕ, А НЕ ВСТАВКА. Умножение сохраняет светотень: чернила темнеют
там, где темна кожа под ними, и в тени рука остаётся рукой. Вставка пикселей
дала бы ровное чёрное пятно, по которому любая метрика отчитается прекрасно, —
то есть образец, на котором нельзя провалиться.

Куда наносить, выбрано ЗАМЕРОМ, а не на глаз. По всем рефкам проекта бёдра
дают 0% кожи: на человеке леггинсы, и дракон на бедре лёг бы на ткань, то есть
проверял бы не перенос приметы, а отказ рисовать по одежде. Предплечье даёт
91%, плечо 74% — холст только там.

    python3 marktest/make_inked_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

#: Доля длины кости, которую занимает татуировка ВДОЛЬ неё.
ALONG_SPAN = 0.45

#: Где её середина на кости (0 — сустав-родитель, 1 — дочерний).
ALONG_CENTRE = 0.5

#: Насколько татуировка уже полуширины конечности. Чуть уже намеренно: рисунок,
#: доходящий до силуэта, проверял бы обрезку по краю, а не перенос.
ACROSS_FRACTION = 0.85


def ink(reference: str, design: str, out: str, bone: str = "l_forearm") -> dict:
    import numpy as np
    from PIL import Image

    from ball_reel import bodyparts, dwpose, marks

    with Image.open(reference) as im:
        base = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        size = im.size
    points = marks.attach_size(dwpose.pose_points(reference), size)
    frame = marks._bone_frame(points, bone)
    if frame is None:
        raise SystemExit(f"кость {bone} не видна на {reference}")
    angle, length = frame

    along_px = int(length * ALONG_SPAN)
    across_px = int(length * marks.half_width_for(bone) * 2 * ACROSS_FRACTION)

    with Image.open(design) as im:
        art = im.convert("L").resize((across_px, along_px), Image.LANCZOS)
    # Рисунок строится в системе кости (длинной стороной вдоль неё) и потом
    # поворачивается в кадр. Обратный порядок — повернуть, потом растянуть —
    # исказил бы пропорции дракона.
    art = art.rotate(-(angle - 90.0), resample=Image.BICUBIC, expand=True,
                     fillcolor=255)
    ink_layer = np.asarray(art, dtype=np.float64) / 255.0

    pair = marks.BONES[bone]
    w, h = size
    a, b = points[pair[0]], points[pair[1]]
    cx = int((a[0] + b[0]) / 2 * w + (b[0] - a[0]) * w * (ALONG_CENTRE - 0.5))
    cy = int((a[1] + b[1]) / 2 * h + (b[1] - a[1]) * h * (ALONG_CENTRE - 0.5))
    ih, iw = ink_layer.shape
    x0, y0 = cx - iw // 2, cy - ih // 2

    # Только по коже: татуировка не заходит ни на фон, ни на одежду. Это не
    # украшение образца, а его корректность — иначе «примета» частично лежала
    # бы там, где переносить её нельзя, и вердикт стал бы про две вещи сразу.
    skin = bodyparts.skin_mask(reference)
    out_arr = base.copy()
    for yy in range(max(0, y0), min(h, y0 + ih)):
        for xx in range(max(0, x0), min(w, x0 + iw)):
            if not skin[yy, xx]:
                continue
            out_arr[yy, xx] *= ink_layer[yy - y0, xx - x0]

    Image.fromarray((np.clip(out_arr, 0, 1) * 255).astype(np.uint8)).save(out)
    changed = int((np.abs(out_arr - base).sum(axis=-1) > 0.01).sum())
    return {"out": out, "bone": bone, "bone_px": round(length, 1),
            "tattoo_px": (iw, ih), "changed_pixels": changed,
            "changed_share": round(changed / (w * h), 5)}


if __name__ == "__main__":
    print(ink(str(ROOT.parent / "marktest_ref.png"),
              str(ROOT / "dragon_raw.png"),
              str(ROOT / "inked_ref.png")))
