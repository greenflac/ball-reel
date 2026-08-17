"""Поток B форка: ОСЬ ПРОТЕЧКИ — что изменилось ВНЕ маски.

ОТКУДА БЕРЁТСЯ ЭТА ОСЬ. В режиме Mix всё внутри маски модель перерисовывает, а
всё снаружи кадры драйвинга проходят нетронутыми. Значит маска — единственная
поверхность риска, и мерить надо ГРАНИЦУ, а не общее впечатление от ролика.
Метрика «похоже ли на драйвинг» по всему кадру ответит «похоже» и тогда, когда
модель стёрла мяч: мяч занимает проценты площади.

Вне маски ожидается ТОЖДЕСТВО, а не сходство. Это редкая удача для измерения:
порог не надо калибровать на когорте, эталон известен точно — исходный кадр
драйвинга.

---

НО НОЛЬ НЕДОСТИЖИМ, И ЭТО ПЕРВОЕ, ЧТО НАДО ЗАМЕРИТЬ.

Кадры проходят кодек: драйвинг декодируется, результат кодируется обратно.
Перекодирование меняет пиксели там, где никто ничего не рисовал. Поэтому
планка берётся НЕ из головы и не из вкуса, а измеряется: `noise_floor`
прогоняет драйвинг САМ ПРОТИВ СЕБЯ через то же сжатие и говорит, сколько стоит
тождество. Всё, что выше этой планки, — протечка; всё, что ниже, — кодек.

Без этого замера любой порог был бы выдуманным, а выдуманный порог на такой
оси особенно вреден: он выглядит строгим (число маленькое) и пропускает
настоящую протечку, потому что шум кодека её накрывает.

---

ТРИ ИСХОДА (Р1), И ТРЕТИЙ ЗДЕСЬ НЕ ФОРМАЛЬНОСТЬ. «Планка не замерена» — обычное
состояние на первом прогоне, и сворачивать его в «годно» значит объявить
герметичность на основании отсутствия проверки.

НЕПРОВЕРЕНО (Ц4): ось откалибрована на синтетике и на прогоне кадров через
кодек. Живого выхода Wan-Animate в этой смене не было, поэтому НАСТОЯЩЕЙ
протечки прибор ещё не видел — он видел нарисованное пятно, и на нём краснеет.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: Три исхода. Берутся у потока A (Е1): одни и те же три слова в двух копиях
#: разъедутся и дадут два несравнимых отчёта.
from .fork_identity import FAIL, PASS, UNMEASURED

#: Во сколько раз замеренная протечка должна превысить шум кодека, чтобы её
#: можно было назвать протечкой. ВЫБРАНО 2.0, и вот из чего.
#:
#: Планка шума — это ВЕРХНЯЯ оценка тождества на одном прогоне кодека, а
#: сравниваемая величина проходит кодек тоже. Множитель 1.0 объявлял бы
#: протечкой любое отличие двух прогонов одного и того же сжатия. Двойка —
#: наименьшее число, при котором «вдвое хуже, чем тождество» уже не объяснить
#: дрожанием кодировщика.
#:
#: Это ВЫБРАНО, не ИЗМЕРЕНО, и калибруется первым же живым выходом. Пометка
#: стоит здесь, а не в документе, чтобы ехать вместе с числом.
LEAK_OVER_FLOOR = 2.0

#: Минимальная доля кадра ВНЕ маски, при которой измерение вообще осмысленно.
#: ВЫБРАНО: если маска накрыла 98% кадра, «вне маски» — это рамка в пару
#: пикселей, и среднее по ней пляшет сильнее измеряемого. Тогда честный ответ
#: «не смогли», а не «протечки нет».
MIN_OUTSIDE_SHARE = 0.02

#: Сколько кадров минимум нужно, чтобы планке шума можно было верить.
#: ВЫБРАНО: одна пара кадров даёт одну точку, а кодек шумит неодинаково на
#: разных кадрах — на статике почти не шумит, на движении заметно.
MIN_FLOOR_FRAMES = 3


def _as_array(image):
    """Кадр -> массив float64 в 0..1, RGB. Одно место, где решается это."""
    import numpy as np
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            arr = np.asarray(im.convert("RGB"), dtype="float64")
    elif hasattr(image, "convert"):
        arr = np.asarray(image.convert("RGB"), dtype="float64")
    else:
        arr = np.asarray(image, dtype="float64")
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, axis=2)
    return arr / 255.0 if arr.max() > 1.0 else arr


def outside_divergence(output, driving, mask) -> dict:
    """Насколько выход разошёлся с драйвингом ТАМ, ГДЕ ЕГО НЕ ТРОГАЛИ.

    Возвращает и среднее, и максимум, и это не избыточность. Среднее ловит
    общий сдвиг (модель перекрасила фон), максимум — локальное пятно (модель
    стёрла мяч). Одно без другого слепо ровно к тому, что ловит второе: пятно
    в 0.3% площади не сдвинет среднее, а равномерный сдвиг не даст максимума.
    """
    import numpy as np

    a = _as_array(output)
    b = _as_array(driving)
    m = np.asarray(mask, dtype=bool)

    if a.shape != b.shape:
        return {"outcome": UNMEASURED, "mean": None, "max": None,
                "outside_share": 0.0,
                "note": f"кадры разного размера: {a.shape} и {b.shape}"}
    if m.shape != a.shape[:2]:
        return {"outcome": UNMEASURED, "mean": None, "max": None,
                "outside_share": 0.0,
                "note": f"маска {m.shape} не по кадру {a.shape[:2]}"}

    outside = ~m
    share = round(float(outside.mean()), 4)
    if share < MIN_OUTSIDE_SHARE:
        return {"outcome": UNMEASURED, "mean": None, "max": None,
                "outside_share": share,
                "note": (f"вне маски осталось {share:.1%} кадра (нужно хотя бы "
                         f"{MIN_OUTSIDE_SHARE:.0%}): мерить не на чем. Это НЕ "
                         f"«протечки нет».")}

    diff = np.abs(a - b).mean(axis=2)
    vals = diff[outside]
    return {"outcome": None, "outside_share": share,
            "mean": round(float(vals.mean()), 6),
            "max": round(float(vals.max()), 6),
            "pixels": int(vals.size),
            "note": (f"вне маски ({share:.1%} кадра, {vals.size} px): "
                     f"среднее {vals.mean():.6f}, максимум {vals.max():.6f}")}


def noise_floor(frame_paths, *, work_dir: str | Path | None = None,
                crf: int = 18, fps: int = 12) -> dict:
    """Сколько стоит ТОЖДЕСТВО после одного прогона через кодек.

    Первое действие потока, и оно обязано быть первым: без этой планки любой
    порог протечки выдуман. Кадры собираются в ролик тем же сжатием, каким
    поедет выход, разбираются обратно и сравниваются С САМИМИ СОБОЙ.

    Три исхода. Нет ffmpeg — `НЕ СМОГЛИ ИЗМЕРИТЬ`, и это НЕ повод взять
    планку по умолчанию: планка по умолчанию — то же выдуманное число, только
    с алиби.
    """
    import shutil
    import tempfile

    import numpy as np

    frames = [Path(p) for p in frame_paths]
    if len(frames) < MIN_FLOOR_FRAMES:
        return {"outcome": UNMEASURED, "floor": None, "frames": len(frames),
                "note": (f"кадров {len(frames)}, нужно хотя бы "
                         f"{MIN_FLOOR_FRAMES}: кодек шумит неодинаково, и по "
                         f"одной точке планку не ставят")}
    if shutil.which("ffmpeg") is None:
        return {"outcome": UNMEASURED, "floor": None, "frames": len(frames),
                "note": ("ffmpeg не в PATH: планку шума замерить нечем. "
                         "Взять её «по умолчанию» нельзя — это выдуманное "
                         "число с алиби.")}

    owned = work_dir is None
    root = Path(tempfile.mkdtemp()) if owned else Path(work_dir)
    try:
        stage = root / "in"
        stage.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(frames):
            shutil.copyfile(p, stage / f"{i:05d}{p.suffix}")
        suffix = frames[0].suffix
        clip = root / "roundtrip.mp4"
        back = root / "out"
        back.mkdir(exist_ok=True)

        enc = subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps),
             "-i", str(stage / f"%05d{suffix}"),
             "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
             str(clip)], capture_output=True, text=True)
        if enc.returncode != 0:
            return {"outcome": UNMEASURED, "floor": None,
                    "frames": len(frames),
                    "note": f"ffmpeg не собрал ролик: {enc.stderr[-200:]}"}
        dec = subprocess.run(
            ["ffmpeg", "-y", "-i", str(clip), str(back / "%05d.png")],
            capture_output=True, text=True)
        if dec.returncode != 0:
            return {"outcome": UNMEASURED, "floor": None,
                    "frames": len(frames),
                    "note": f"ffmpeg не разобрал ролик: {dec.stderr[-200:]}"}

        got = sorted(back.glob("*.png"))
        if len(got) != len(frames):
            return {"outcome": UNMEASURED, "floor": None,
                    "frames": len(frames),
                    "note": (f"кодек вернул {len(got)} кадров из "
                             f"{len(frames)}: сравнивать попарно нечего")}

        means, maxes = [], []
        for src, out in zip(frames, got):
            d = np.abs(_as_array(src) - _as_array(out)).mean(axis=2)
            means.append(float(d.mean()))
            maxes.append(float(d.max()))
        return {"outcome": PASS, "frames": len(frames), "crf": crf,
                "floor": round(max(means), 6),
                "floor_max": round(max(maxes), 6),
                "note": (f"планка шума ИЗМЕРЕНА на {len(frames)} кадрах, "
                         f"crf {crf}: среднее до {max(means):.6f}, локально до "
                         f"{max(maxes):.6f}. Всё ниже — кодек, не протечка.")}
    finally:
        if owned:
            shutil.rmtree(root, ignore_errors=True)


def verdict(divergence: dict, floor: dict, *,
            over: float = LEAK_OVER_FLOOR) -> dict:
    """Протечка или кодек. Три исхода, и «планки нет» — один из них.

    Сравнение с ЭТАЛОНОМ, а не с калиброванным порогом: эталон здесь известен
    точно (тот же кадр драйвинга), и всё, чего не хватало, — цена тождества.
    """
    if divergence.get("mean") is None:
        return {"outcome": UNMEASURED, "ratio": None,
                "note": f"расхождение не измерено: {divergence.get('note', '')}"}
    if floor.get("floor") is None:
        return {"outcome": UNMEASURED, "ratio": None,
                "note": (f"планка шума не измерена: {floor.get('note', '')}. "
                         f"Расхождение {divergence['mean']:.6f} само по себе "
                         f"ничего не значит — неизвестно, сколько из него "
                         f"стоит кодек.")}
    base = floor["floor"]
    if base <= 0:
        return {"outcome": UNMEASURED, "ratio": None,
                "note": ("планка шума нулевая: кодек не тронул ни пикселя, "
                         "чего не бывает — прогон планки, скорее всего, "
                         "сравнивал файл сам с собой")}
    ratio = round(divergence["mean"] / base, 3)
    leaking = ratio > over

    # ЛОКАЛЬНЫЙ КАНАЛ СУДИТСЯ ОТДЕЛЬНО И СВОЕЙ ПЛАНКОЙ. Сначала вердикт стоял
    # на одном среднем, и максимум печатался без порога — то есть канал, ради
    # которого он и заведён (стёртый мяч занимает проценты площади), не мог
    # ничего решить. Дыру видно из замера: на настоящих кадрах при crf 18
    # среднее кодека 0.004513, а ЛОКАЛЬНО 0.071895 — в шестнадцать раз выше.
    # Судить максимум по средней планке значило бы объявлять протечкой каждый
    # блок сжатия.
    base_max = floor.get("floor_max")
    spot_ratio = None
    if base_max and divergence.get("max") is not None:
        spot_ratio = round(divergence["max"] / base_max, 3)
        if spot_ratio > over:
            leaking = True

    return {
        "outcome": FAIL if leaking else PASS,
        "ratio": ratio, "spot_ratio": spot_ratio,
        "floor": base, "floor_max": base_max,
        "mean": divergence["mean"], "max": divergence.get("max"),
        "note": (f"вне маски среднее {divergence['mean']:.6f} при планке "
                 f"{base:.6f} — {ratio}x"
                 + (f"; локально {divergence['max']:.6f} при планке "
                    f"{base_max:.6f} — {spot_ratio}x"
                    if spot_ratio is not None else
                    "; локальная планка не замерена, точечная протечка не "
                    "судится")
                 + f". Порог {over}x: "
                   f"{'ПРОТЕЧКА' if leaking else 'в пределах кодека'}."),
    }
