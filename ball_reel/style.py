"""Подражание драйвингу: снять стиль с настоящего видео и надеть на выход.

ЗАЧЕМ ЭТОТ МОДУЛЬ. Личность приходит из фотографии, и вместе с личностью
приходит ВИД этой фотографии. Референс в наборе — рекламный кадр с ретушью, и
результат наследует именно его: высокий микроконтраст, вылизанная кожа,
насыщенный цвет. На глаз это читается как «похоже на фото, но стерильно», и
жалоба верна.

Промтом это не лечится по существу. Промт торгуется с личностью за одни и те же
токены: редакция, просившая `no makeup` и `blemishes`, дала нужный жанр и увела
лицо с 0.217 до 0.304 при баре 0.30 (замерено, три кадра, nanobanana-2). Стиль,
взятый ПОСЛЕ генерации, не стоит сходства вовсе: он не участвует в рисовании.

ЧИСЛА, НА КОТОРЫХ ЭТО СТОИТ. Три облака, снятые по пикселям, без моделей
(512x512, чтобы разрешение не решало за нас):

    ось            драйвинг (n=16)   генератор nb2 (n=4)   рефка (n=1)
    контраст       0.190 .. 0.201    0.295 .. 0.317        0.191
    детальность    0.98  .. 1.39     5.87  .. 8.65         4.61
    зерно          7.55  .. 8.63     20.9  .. 25.1         14.3

Облака НЕ ПЕРЕКРЫВАЮТСЯ ни по одной из трёх осей — между ними разрыв, а не
размытая граница, и потому здесь есть что мерить и чем судить. Направление
обратно интуиции: драйвинг не «лучше детализирован», он в семь раз МЯГЧЕ. То,
что читается как вылизанность, — это избыток микроконтраста, а не недостаток
деталей.

ГЛАВНОЕ ОГРАНИЧЕНИЕ, И ОНО ЗАМЕЧЕНО ГЛАЗОМ, А НЕ ЧИСЛОМ. Мягкость здесь
достигается гауссом, а гаусс — это РАСФОКУС. Он честно сводит детальность
(20.8 -> 1.18 при цели 1.153) и так же честно выглядит как несведённый фокус,
потому что видео отличается от вылизанной картинки не отсутствием резкости, а
НАЛИЧИЕМ ЗЕРНА. Это видно и в числах: зерно после подгонки 13.4 против 7.8 у
драйвинга — единственная ось, которая не сходится, и не сходится она потому,
что гаусс давит зерно вместе с деталью, а не вместо неё.

Правильный инструмент здесь — избирательный шумодав плюс добавленное зерно, то
есть разделение полос, а не одно сглаживание на всё. Пока этого нет, ступень
`--style` в прогоне ВЫКЛЮЧЕНА по умолчанию, и включать её ради вида не следует.
Модуль оставлен потому, что его ЗАМЕР (три облака, разрыв между ними) верен и
нужен независимо от того, чем потом чинить.

ЧЕГО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ, и это надо сказать вслух. Он сводит СТАТИСТИКИ, а
не стиль. Направление света, характер оптики, композиция, цветовая схема сцены
остаются какими вышли. Кадр, снятый в другом зале при другом свете, после
подгонки будет иметь тот же контраст и ту же мягкость — и по-прежнему будет
другим кадром. Это честная граница метода, а не недоделка: за композицию
отвечает ControlNet, за свет — промт, здесь только фактура.

Воспроизвести числа:

    python3 -m ball_reel.style --driving demo/kit/driving --frame КАДР
"""

from __future__ import annotations

from pathlib import Path

#: Сторона, к которой приводится кадр перед замером. Детальность и зерно —
#: величины НА ПИКСЕЛЬ, и сравнивать 1024 с 736 без приведения значит сравнивать
#: разрешения, а не стили. Квадрат берётся намеренно: пропорции здесь не
#: измеряются, а растяжение одинаково искажает оба сравниваемых кадра.
MEASURE_SIDE = 512

#: Оси, по которым идёт подгонка. Порядок важен: яркость и контраст правятся
#: линейно и не трогают частоты, а мягкость правится сглаживанием и меняет обе
#: предыдущие — значит она последняя.
AXES = ("brightness", "contrast", "saturation", "detail", "noise")

#: Во сколько раз позволено промахнуться по оси, чтобы считать «совпало». Взято
#: НЕ с потолка: разрыв между облаками по детальности — 1.39 против 5.87, то
#: есть в 4.2 раза, по зерну 8.63 против 20.9, то есть в 2.4. Допуск 1.5
#: помещается внутрь обоих разрывов с запасом и потому различает облака, а не
#: сглаживает их. Порог, не лежащий в разрыве, был бы украшением.
TOLERANCE = 1.5


def measure(path) -> dict | None:
    """Кадр -> пять чисел стиля. None — файл не прочитался.

    Ни одна величина не берётся из запроса и ни одна не требует модели: это
    арифметика над пикселями, и потому она одинакова на любой машине.
    """
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            small = im.convert("RGB").resize(
                (MEASURE_SIDE, MEASURE_SIDE), Image.BICUBIC)
            a = np.asarray(small, dtype="float64")
    except Exception:  # noqa: BLE001
        return None
    if a.size == 0:
        return None
    return _stats(a)


def _stats(a) -> dict:
    """Массив HxWx3 в 0..255 -> пять чисел. Отдельно, чтобы тестировать без диска."""
    import numpy as np

    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 1e-9, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    # Лапласиан 3x3 — отклик на высокие частоты. Дисперсия отклика есть
    # классическая мера резкости; медианное абсолютное отклонение того же
    # отклика — РОБАСТНАЯ оценка зерна. Робастная нужна потому, что обычная
    # дисперсия считает края объектов, а не шум: кадр с забором окажется
    # «шумным», ничего шумного не имея.
    k = lum
    lap = (4 * k[1:-1, 1:-1] - k[:-2, 1:-1] - k[2:, 1:-1]
           - k[1:-1, :-2] - k[1:-1, 2:])
    mad = float(np.median(np.abs(lap - np.median(lap))))
    return {"brightness": float(lum.mean()),
            "contrast": float(lum.std()),
            "saturation": float(sat.mean()),
            "detail": float(lap.var()) * 1e3,
            "noise": 1.4826 * mad * 1e3,
            "warm": (float(r.mean() / b.mean()) if b.mean() > 1e-9 else None)}


def profile(frames: list) -> dict:
    """Несколько кадров драйвинга -> облако стиля (медиана и границы).

    МЕДИАНА, А НЕ СРЕДНЕЕ. Один пересвеченный или смазанный кадр видео — обычное
    дело, и среднее он утащит за собой. Границы хранятся не для красоты: по ним
    видно, насколько облако узкое, а узость облака — единственное основание
    считать, что у видео вообще ЕСТЬ единый стиль.
    """
    rows = [m for m in (measure(f) for f in frames) if m]
    if not rows:
        return {"ok": False, "n": 0,
                "note": "ни одного читаемого кадра драйвинга: подражать нечему"}
    out: dict = {"ok": True, "n": len(rows), "axes": {}}
    for axis in AXES:
        v = sorted(x[axis] for x in rows if x.get(axis) is not None)
        if not v:
            continue
        out["axes"][axis] = {"p50": v[len(v) // 2], "min": v[0], "max": v[-1],
                             "spread": (v[-1] / v[0]) if v[0] > 1e-9 else None}
    return out


def distance(frame_stats: dict, prof: dict) -> dict:
    """Насколько кадр далёк от облака драйвинга, ПО ОСЯМ и в разах.

    В разах, а не в единицах: детальность 1.15 и 8.06 — это «в семь раз», и
    именно так их надо читать. Абсолютная разница 6.9 не говорит ни о чём,
    потому что у яркости и у детальности разные шкалы.
    """
    got: dict = {"axes": {}, "worst": None, "worst_ratio": None}
    if not prof.get("ok"):
        return got
    worst, worst_axis = 0.0, None
    for axis, band in prof["axes"].items():
        have, want = frame_stats.get(axis), band["p50"]
        if have is None or want is None or want <= 1e-9:
            continue
        ratio = have / want if have >= want else (want / have if have > 1e-9
                                                  else float("inf"))
        got["axes"][axis] = {"have": have, "want": want, "ratio": ratio,
                             "ok": ratio <= TOLERANCE}
        if ratio > worst:
            worst, worst_axis = ratio, axis
    got["worst"], got["worst_ratio"] = worst_axis, (worst or None)
    got["ok"] = bool(got["axes"]) and all(a["ok"] for a in got["axes"].values())
    return got


def _linear(a, want_c, want_b, want_s):
    """Контраст, яркость и насыщенность — одним проходом, без частот.

    Отдельной функцией потому, что вызывается ДВАЖДЫ: до сглаживания и как
    поправка после него. Пока это было двумя копиями внутри `match`, они
    успели разойтись.
    """
    import numpy as np

    cur = _stats(np.clip(a, 0, 255))
    if want_c and cur["contrast"] > 1e-9:
        mean = a.mean(axis=(0, 1), keepdims=True)
        a = mean + (a - mean) * (want_c / cur["contrast"])
    if want_b:
        lum = (0.299 * a[..., 0] + 0.587 * a[..., 1]
               + 0.114 * a[..., 2]).mean() / 255.0
        if lum > 1e-9:
            a = a * (want_b / lum)
    if want_s:
        grey = (0.299 * a[..., 0:1] + 0.587 * a[..., 1:2]
                + 0.114 * a[..., 2:3])
        now = _stats(np.clip(a, 0, 255))["saturation"]
        if now > 1e-9:
            a = grey + (a - grey) * (want_s / now)
    return a


def match(frame, prof: dict, out_path, *, verbose: bool = False) -> dict:
    """Кадр + облако драйвинга -> кадр той же фактуры на диске.

    ПОРЯДОК ОПЕРАЦИЙ НЕ ПРОИЗВОЛЕН, И ОБОСНОВАН ОН ЧИСЛОМ, А НЕ РАССУЖДЕНИЕМ.
    Первая редакция сглаживала первой, «потому что линейная правка иначе
    уедет», — и промахивалась втрое: сглаживание попадало в цель (детальность
    1.179 при 1.153), а следующая за ним правка контраста давила её ещё раз.
    Причина арифметическая: линейное растяжение контраста с коэффициентом g
    умножает ОТКЛИК ЛАПЛАСИАНА на g, то есть его дисперсию — на g². При
    g = 0.196/0.315 = 0.62 это 0.39, и 1.179 превращается в 0.43.

    Поэтому линейное идёт ПЕРВЫМ, сглаживание — вторым, и лишь малая поправка
    контраста после него. Обратное влияние существует (сглаживание немного
    снижает контраст), но оно не квадратично, и одной поправки хватает.

    Возвращает замер ДО и ПОСЛЕ. Не «получилось», а два набора чисел: судить по
    ним, а не по слову.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    if not prof.get("ok"):
        return {"ok": False, "note": prof.get("note", "нет облака драйвинга")}
    with Image.open(frame) as im:
        rgb = im.convert("RGB")
        before = _stats(np.asarray(
            rgb.resize((MEASURE_SIDE, MEASURE_SIDE), Image.BICUBIC),
            dtype="float64"))

        want_c = prof["axes"].get("contrast", {}).get("p50")
        want_b = prof["axes"].get("brightness", {}).get("p50")
        want_s = prof["axes"].get("saturation", {}).get("p50")

        arr = _linear(np.asarray(rgb, dtype="float64"), want_c, want_b, want_s)
        rgb = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
        before_soft = _stats(np.asarray(
            rgb.resize((MEASURE_SIDE, MEASURE_SIDE), Image.BICUBIC),
            dtype="float64"))

        # МЯГКОСТЬ. Радиус подбирается поиском, а не формулой: связь между
        # радиусом гаусса и дисперсией лапласиана зависит от содержимого кадра,
        # и аналитическое выражение для неё было бы догадкой.
        want_detail = prof["axes"].get("detail", {}).get("p50")
        soft, radius = rgb, 0.0
        if (want_detail and before_soft
                and before_soft["detail"] > want_detail * TOLERANCE):
            # БЛИЖАЙШИЙ радиус, а не первый подходящий. Первая редакция брала
            # первый, у которого детальность упала НИЖЕ цели, и промахивалась
            # вниз: измерено — 7.93 -> 0.347 при цели 1.153, то есть втрое
            # мягче драйвинга. Пересглаженный кадр так же не похож на видео,
            # как недосглаженный, только с другой стороны, и на глаз он хуже:
            # мыло заметнее избытка резкости.
            # СЕТКА ГУСТАЯ НАМЕРЕННО. Редкая (0.3/0.5/0.8/1.1/1.5/2.0/2.6)
            # перепрыгивала цель: между 1.5 и 2.0 детальность падала с 3.8 до
            # 0.35 при цели 1.153, и ближайший радиус просто отсутствовал в
            # списке. Двадцать семь замеров на кадр стоят долю секунды, а
            # промах втрое виден глазом как мыло.
            best = None
            for r in [round(0.2 + 0.1 * i, 1) for i in range(27)]:
                cand = rgb.filter(ImageFilter.GaussianBlur(radius=r))
                s = _stats(np.asarray(
                    cand.resize((MEASURE_SIDE, MEASURE_SIDE), Image.BICUBIC),
                    dtype="float64"))
                # В РАЗАХ, а не в разнице: цель 1.153, и промах 0.35 снизу
                # (в 3.3 раза) хуже промаха 1.5 сверху (в 1.3 раза), хотя по
                # разнице он меньше.
                got_d = max(s["detail"], 1e-9)
                miss = (got_d / want_detail if got_d >= want_detail
                        else want_detail / got_d)
                if best is None or miss < best[0]:
                    best = (miss, cand, r)
                if got_d <= want_detail:
                    break        # дальше только мягче, то есть только хуже
            if best:
                _, soft, radius = best

        # ПОПРАВКА ПОСЛЕ СГЛАЖИВАНИЯ. Гаусс снижает контраст и слегка гасит
        # цвет; поправка мала (коэффициент около единицы), и потому её
        # обратное влияние на детальность — тоже около единицы, в отличие от
        # первой правки, которая давила её в g² раз.
        a = _linear(np.asarray(soft, dtype="float64"), want_c, want_b, want_s)
        a = np.clip(a, 0, 255)
        out = Image.fromarray(a.astype("uint8"))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path)
        after = _stats(np.asarray(
            out.resize((MEASURE_SIDE, MEASURE_SIDE), Image.BICUBIC),
            dtype="float64"))

    got = {"ok": True, "path": str(out_path), "blur_radius": radius,
           "before": before, "after": after,
           "distance_before": distance(before, prof),
           "distance_after": distance(after, prof)}
    if verbose:
        print(f"радиус сглаживания {radius}")
        for axis in AXES:
            b = before.get(axis)
            aft = after.get(axis)
            want = prof["axes"].get(axis, {}).get("p50")
            if b is None or want is None:
                continue
            print(f"  {axis:<11} {b:8.3f} -> {aft:8.3f}   драйвинг {want:8.3f}")
    return got


def main(argv: list) -> int:
    import argparse
    import glob

    ap = argparse.ArgumentParser(
        prog="ball_reel.style",
        description="снять стиль с драйвинга и надеть его на кадр")
    ap.add_argument("--driving", required=True,
                    help="каталог кадров драйвинга — образец фактуры")
    ap.add_argument("--frame", default="", help="кадр, который подгонять")
    ap.add_argument("--out", default="", help="куда класть подогнанный кадр")
    args = ap.parse_args(argv)

    frames = sorted(glob.glob(f"{args.driving}/*.jpg")
                    + glob.glob(f"{args.driving}/*.png"))
    prof = profile(frames)
    if not prof["ok"]:
        print(prof["note"])
        return 1
    print(f"драйвинг: {prof['n']} кадров")
    for axis, band in prof["axes"].items():
        print(f"  {axis:<11} p50={band['p50']:.3f}  "
              f"({band['min']:.3f} .. {band['max']:.3f})")

    if not args.frame:
        return 0
    stats = measure(args.frame)
    if not stats:
        print(f"не прочитался кадр {args.frame}")
        return 1
    d = distance(stats, prof)
    print(f"\nкадр {args.frame}:")
    for axis, a in d["axes"].items():
        mark = "OK" if a["ok"] else f"мимо в {a['ratio']:.1f} раза"
        print(f"  {axis:<11} {a['have']:8.3f} против {a['want']:8.3f}  {mark}")

    if args.out:
        print()
        match(args.frame, prof, args.out, verbose=True)
        print(f"\nподогнано: {args.out} — смотреть ГЛАЗАМИ: сведены "
              f"СТАТИСТИКИ, а не свет и композиция")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
