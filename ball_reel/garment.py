"""Одежда: держится ли она постоянной по клипу.

Кто её задаёт. Не видео-модель и не ControlNet: скелет описывает позу и молчит
про ткань, IP-Adapter отвечает за лицо. Одежду определяет ТЕКСТОВЫЙ ПРОМТ на
этапе кейфрейма (в публичном пути — на старт-кадре). Проверено на kontext:
«красное худи, чёрные шорты, жёлтые высокие кроссовки, шапка» появились ровно
как написано, тогда как телосложение тем же способом задать не удалось.

Почему это нужно мерить. Кейфреймы цепочки генерируются НЕЗАВИСИМО друг от
друга. Промт один, но сид, шум и интерполяция — разные, поэтому оттенок и крой
плывут между узлами: сверху то темнее, то светлее, лямка то есть, то нет. На
статичном кадре это незаметно, а в склеенном клипе читается как мигание, и ни
один существующий гейт этого не видит — идентичность про лицо, поза про
скелет, анатомия про длины костей.

Как меряется. Область одежды берётся из позы, а не из фиксированного
прямоугольника: торс — четырёхугольник плечи-бёдра, ноги — полосы вдоль
бедренных костей. Человек в кадре двигается, и неподвижная рамка мерила бы
попеременно то футболку, то стену. Дальше — средний цвет области в каждом
кадре и разброс по клипу. Цвет берётся как ХРОМАТИЧНОСТЬ (доли каналов), а не
как сырой RGB: человек подпрыгивает, освещение торса гуляет, и сырой RGB меряет
именно яркость — на эталоне с заведомо одной одеждой он дал разброс 0.265 при
баре 0.18, то есть метрика объявляла эталон плывущим.

Только numpy и Pillow. Ни сети, ни модели.
"""

from __future__ import annotations

from pathlib import Path

#: Допустимый разброс ХРОМАТИЧНОСТИ одежды по клипу. Обе опоры измерены:
#:
#:     эталон (одно видео, одежда заведомо одна)  torso 0.0245  legs 0.0107
#:     контроль (кадры двух РАЗНЫХ съёмок)        torso 0.1799  legs 0.0719
#:
#: Разделение семикратное, бар лежит между ними. Верхняя опора получена
#: подмешиванием кадров другой съёмки, а не сгенерированной цепочкой: она
#: показывает, что метрика ловит смену одежды, но насколько сильно плывут
#: именно кейфреймы — покажет первый полный прогон, и число надо перепроверить.
GARMENT_DRIFT_MAX = 0.03

#: Доля пикселей области, по которой берётся цвет. Медиана по центральной
#: части, а не среднее по всей: край области цепляет фон и руки.
CORE_FRACTION = 0.6


def _regions(points: dict) -> dict:
    """Области одежды в нормированных координатах, из позы.

    Возвращает многоугольники: `torso` (плечи-бёдра) и `legs` (полосы вдоль
    бёдер). Пустой словарь, если опорные суставы не видны — мерить цвет
    «примерно там» хуже, чем не мерить.
    """
    need = ("l_shoulder", "r_shoulder", "l_hip", "r_hip")
    if any(points.get(n) is None or points[n][2] < 0.5 for n in need):
        return {}
    ls, rs = points["l_shoulder"], points["r_shoulder"]
    lh, rh = points["l_hip"], points["r_hip"]
    out = {"torso": [(ls[0], ls[1]), (rs[0], rs[1]), (rh[0], rh[1]), (lh[0], lh[1])]}

    legs = []
    for hip, knee in (("l_hip", "l_knee"), ("r_hip", "r_knee")):
        h, k = points.get(hip), points.get(knee)
        if h and k and k[2] >= 0.5:
            # Полоса шириной в треть расстояния бедро-колено вдоль кости.
            dx, dy = k[0] - h[0], k[1] - h[1]
            w = (dx * dx + dy * dy) ** 0.5 / 6 or 0.02
            legs.append([(h[0] - w, h[1]), (h[0] + w, h[1]),
                         (k[0] + w, k[1]), (k[0] - w, k[1])])
    if legs:
        out["legs"] = legs
    return out


def _region_colour(path: str | Path, polygon, core: float = CORE_FRACTION):
    """Медианный цвет внутри многоугольника, по центральной его части."""
    import numpy as np
    from PIL import Image, ImageDraw

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype="float64")
    h, w = rgb.shape[:2]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon([(x * w, y * h) for x, y in polygon], fill=255)
    m = np.asarray(mask) > 0
    if m.sum() < 20:
        return None
    ys, xs = np.nonzero(m)
    # Сжать область к центру: край ловит фон и руки.
    cy, cx = ys.mean(), xs.mean()
    keep = ((np.abs(ys - cy) < (ys.max() - ys.min()) * core / 2) &
            (np.abs(xs - cx) < max(xs.max() - xs.min(), 1) * core / 2))
    if keep.sum() < 20:
        keep = np.ones_like(ys, dtype=bool)
    px = rgb[ys[keep], xs[keep]]
    # Хроматичность, а не сырой RGB. Человек подпрыгивает, освещение торса
    # гуляет, и средний RGB ловит именно яркость: на исходном видео с заведомо
    # ОДНОЙ одеждой сырой RGB дал разброс 0.265 при баре 0.18, то есть метрика
    # объявляла эталон плывущим. Деление на сумму каналов убирает яркость и
    # оставляет цвет. [проверено live]
    total = px.sum(axis=1, keepdims=True)
    total[total < 1e-6] = 1e-6
    return np.median(px / total, axis=0)


def garment_drift(frames: list, points_by_frame: list,
                  *, max_drift: float = GARMENT_DRIFT_MAX) -> dict:
    """Плывёт ли одежда по клипу.

    `points_by_frame` — позы тех же кадров (из `skeleton.pose_points` или
    `dwpose.pose_points`), потому что область одежды выводится из тела.

    Возвращает разброс по каждой области, нормированный на её собственную
    яркость, и вердикт. Кадры без видимого торса не судятся и уходят в
    `unmeasured`: «не видно» и «поплыло» — разные ответы, и склеивать их
    нельзя, иначе отвернувшийся человек будет читаться как переодевшийся.
    """
    import numpy as np

    if len(frames) != len(points_by_frame):
        raise ValueError("кадров и поз должно быть поровну")

    series: dict = {}
    unmeasured = 0
    for path, pts in zip(frames, points_by_frame):
        regions = _regions(pts) if pts else {}
        if not regions:
            unmeasured += 1
            continue
        for name, poly in regions.items():
            polys = poly if isinstance(poly[0], list) else [poly]
            colours = [c for c in (_region_colour(path, p) for p in polys)
                       if c is not None]
            if colours:
                series.setdefault(name, []).append(np.mean(colours, axis=0))

    measured = len(frames) - unmeasured
    if not series or measured < 3:
        return {"regions": {}, "worst": None, "stable": False,
                "measured": measured, "unmeasured": unmeasured,
                "note": (f"одежда НЕ ПРОВЕРЕНА: торс различим лишь в "
                         f"{measured} кадре(ах) из {len(frames)}.")}

    drift = {}
    for name, colours in series.items():
        # Значения уже в долях (хроматичность), поэтому разброс берётся как
        # есть — делить на яркость второй раз нечего.
        drift[name] = round(float(np.mean(np.std(np.asarray(colours), axis=0))), 4)
    worst = max(drift, key=lambda n: drift[n])
    stable = drift[worst] <= max_drift
    return {
        "regions": drift, "worst": (worst, drift[worst]), "stable": stable,
        "measured": measured, "unmeasured": unmeasured,
        "note": (f"разброс цвета одежды: "
                 + ", ".join(f"{k} {v:.3f}" for k, v in sorted(drift.items()))
                 + f" (бар {max_drift}); "
                 + ("держится" if stable else
                    f"ПЛЫВЁТ на «{worst}» — кейфреймы рисуют разную одежду")
                 + f"; не судилось {unmeasured} кадр(ов)."),
    }


#: Формулировка одежды для промта. Здесь она собрана из полей `Subject`, а не
#: сочиняется на месте, чтобы ВСЕ кейфреймы получили побуквенно одинаковый
#: текст: одежда задаётся только промтом, и любое расхождение формулировки
#: между узлами — это расхождение одежды в кадре.
def wardrobe_clause(subject) -> str:
    """Одежда субъекта одной фразой, стабильной между вызовами."""
    worn = ", ".join(p for p in (getattr(subject, "outfit", ""),
                                 getattr(subject, "footwear", "")) if p)
    return f"Wearing {worn}." if worn else ""
