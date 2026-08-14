"""Урезанный кит для git: ровно то окно, на котором собран показанный клип.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ, А НЕ ПАРА КОМАНД В РУНБУКЕ. Клон репозитория приезжает
без `kit/` — он в `.gitignore`, потому что это бинарь на 7.8 МБ. Значит
повторить показанный прогон нельзя, и «а можно воспроизвести» остаётся без
ответа ровно там, где ответ важнее всего. В git кладётся НАРЕЗКА: 16 условий,
16 driving-кадров и фото лица, вместе ~1.7 МБ.

ЛОВУШКА, РАДИ КОТОРОЙ ЭТО КОД, А НЕ `cp`. Нарезать условия нельзя, их надо
ПЕРЕРЕНДЕРИТЬ — но перерендеренные автономно, они выходят ХУЖЕ полных, и по
картинке этого не видно. ИЗМЕРЕНО на нашем отрезке:

    телосложение донора по 71 кадру   6 костей: бедро x0.701, плечо x1.196, ...
    оно же по 16 кадрам окна          3 кости: предплечья и одно плечо ПОТЕРЯНЫ,
                                      бедро x0.630 — уехало на 10%

Причина не в ошибке, а в честном пороге: предплечья видны меньше чем на
`skeleton.MIN_DRIVING_FRAMES` кадрах окна, и медиана по ним не строится. Кит при
этом выглядит полным. Поэтому телосложение донора измеряется по ВСЕЙ
последовательности и передаётся в рендер снаружи вместе с числом кадров —
иначе манифест не отличит замер по 71 кадру от замера по шести.

Побочное следствие, которое стоит знать: условия нарезки получаются
БИТ-В-БИТ теми же, что соответствующие кадры полного кита. Это и есть проверка,
что нарезка — нарезка, а не пересъёмка.
"""

from __future__ import annotations

from pathlib import Path

#: Начало окна. ИЗМЕРЕНО, и это размен двух величин, а не одна максимизация.
#:
#: Самое подвижное окно начинается с 55 (сумма межкадровых изменений 101.4
#: против 61.2 у окна 0-15). Но ВСЕ ПЯТЬ кадров с потерянной конечностью,
#: какие есть во всём ките, лежат внутри именно этого окна: 5 из 16, покрытие
#: суставов 0.868 против 0.97 по киту целиком. Потерянный сустав — это не
#: косметика: конечность в таком кадре ничем не ограничена, и генератор её
#: ВЫДУМАЕТ.
#:
#:     старт  подвижность  частичных из 16  покрытие суставов
#:        55        101.4                5             0.868
#:        52         92.8                2             0.934
#:        50         84.3                0             0.979
#:
#: Взято 50: минус 17% подвижности, и ни одного кадра, в котором генератору
#: разрешено сочинять руку. Окон без частичных кадров всего 51 из 56, и это —
#: самое подвижное из них.
WINDOW_START = 50

#: 16 — не вкус, а требование модуля движения AnimateDiff: он обучен на окне
#: этой длины, и `animate.animate` отказывается работать при другом числе.
#: Берётся из `animate`, а не пишется числом: второй способ узнать то, что уже
#: известно, — дефект, и в этом проекте он уже стоил 1.7 ГБ лишней качки.
def window_frames() -> int:
    from .animate import CONTEXT_FRAMES

    return CONTEXT_FRAMES


def build(src: str | Path = "kit", dst: str | Path = "demo/kit", *,
          start: int = WINDOW_START, frames: int | None = None,
          framing: str = "full_body", source=None) -> dict:
    """Полный кит -> нарезка в git. Возвращает манифест нарезки.

    Телосложение донора меряется по ВСЕМ driving-кадрам источника, а условия
    рендерятся только для окна. Почему именно так — в шапке модуля.

    `source` прокидывается в `skeleton.render_sequence` и нужен ровно тестам:
    с умолчанием экстрактор выбирается по наличию весов DWPose, и «нарезка
    пустая» становится неотличимо от «весов нет на этой машине» — два разных
    отказа под одним красным тестом.
    """
    import glob
    import json
    import shutil

    from . import pose, skeleton

    src, dst = Path(src), Path(dst)
    frames = frames or window_frames()
    driving = sorted(glob.glob(str(src / "driving" / "*.jpg")))
    if not driving:
        raise FileNotFoundError(
            f"нет driving-кадров в {src / 'driving'}: полный кит не распакован. "
            f"Нарезку не из чего делать, и подменять её пустой папкой нельзя.")
    if start + frames > len(driving):
        raise ValueError(
            f"окно {start}..{start + frames - 1} не влезает в {len(driving)} "
            f"кадров: последнее допустимое начало {len(driving) - frames}")

    proportions = pose.world_proportions(str(src / "face.jpg"))
    # По ВСЕЙ последовательности, а не по окну — см. шапку.
    donor, donor_frames = skeleton.driving_proportions(driving)

    (dst / "driving").mkdir(parents=True, exist_ok=True)
    window = driving[start:start + frames]
    cut = []
    for i, f in enumerate(window):
        out = dst / "driving" / f"{i:04d}.jpg"
        shutil.copyfile(f, out)
        cut.append(str(out))
    shutil.copyfile(src / "face.jpg", dst / "face.jpg")

    manifest = skeleton.render_sequence(
        cut, dst / "conditions", proportions=proportions, framing=framing,
        donor=donor, donor_frames=donor_frames, source=source)
    # Откуда взялось окно — записывается РЯДОМ С НИМ. Иначе через неделю никто
    # не восстановит, что 50 — не круглое число, а размен двух замеров.
    manifest["cut_from"] = {
        "source": str(src), "start": start, "frames": frames,
        "of_total": len(driving),
        "why": ("самое подвижное окно БЕЗ частичных кадров: подвижность 84.3 "
                "против 101.4 у самого подвижного (старт 55), но там 5 кадров "
                "из 16 с потерянной конечностью, а здесь ни одного. Замерено "
                "по условиям полного кита"),
    }
    (dst / "conditions" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main(argv: list) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="собрать урезанный кит для git из полного")
    ap.add_argument("--src", default="kit")
    ap.add_argument("--dst", default="demo/kit")
    ap.add_argument("--start", type=int, default=WINDOW_START)
    ap.add_argument("--framing", choices=("full_body", "waist_up"),
                    default="full_body",
                    help="ПОЛНЫЙ РОСТ — продуктовый выбор: движение тела важнее "
                         "удобства измерения. Но на нём лицо выходит 78 px при "
                         "баре ArcFace 100, и ось личности честно отвечает «не "
                         "смогли измерить». ПОЯСНАЯ даёт 143.5 px и судимую "
                         "личность — это не замена, а ВТОРОЙ прогон, которым "
                         "число по личности вообще добывается")
    args = ap.parse_args(argv)
    m = build(args.src, args.dst, start=args.start, framing=args.framing)
    print(f"условий: {len(m['conditions'])}, кадрировка {m['framing']}, "
          f"лицо {m['face_px']} px, судимо: {m['identity_judgeable']}")
    print(f"ретаргет: {len(m['retarget_factors'])} из "
          f"{len(m['retarget_origin'])} костей, телосложение донора по "
          f"{m['donor_frames_measured']} кадрам")
    for w in m["warnings"]:
        print(f"  - {w[:160]}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
