"""Порождение кадров набора из ОДНОЙ фотографии — недостающая половина обучения.

`dataset.build_dataset` умел собрать набор из реального кадра и восьми
аугментаций — девять кадров при пороге двенадцать. Порождённые кадры он
принимал ГОТОВЫМИ (`--generated КАТАЛОГ`), а рисовал их кто-то снаружи. То есть
«генерация датасета из фото» в пайплайне отсутствовала: девять кадров одной
позы обучают LoRA одному ракурсу, и никакая длительность обучения этого не
чинит.

РОЛИ РАЗВЕДЕНЫ, И ПОРЯДОК НАРУШАТЬ НЕЛЬЗЯ. Это главное требование ко всему
модулю, и оно проверяется кодом (`dataset.independence_report`), а не обещанием:

    порождение  ->  шлюз Pollinations (свой механизм референса)
    отбор       ->  FaceNet (InceptionResnetV1, vggface2)
    обучение    ->  LoRA
    суд         ->  ArcFace — независим и от порождения, и от отбора

Отобрав кадры ArcFace'ом, мы получили бы набор «того, что нравится ArcFace», и
потеряли бы независимость ровно там, где собирались её получить. Поэтому
отбирающий распознаватель здесь ДРУГОЙ, из другого семейства и с другими
весами, и подменить его нечем: `dataset.select` принимает уже посчитанную
дистанцию, а считает её эта функция.

ЛИЦЕНЗИЯ — ФЛАГ ДЛЯ ЮРИСТА, НЕ РЕШЁННЫЙ ВОПРОС. Код `facenet-pytorch` под MIT
(проверено в `LICENSE.md` установленного пакета). ВЕСА — другое: `vggface2`
обучены на датасете VGGFace2, чьи условия ограничивают использование
исследовательскими целями. Продукт коммерческий, поэтому:

* веса участвуют ТОЛЬКО в отборе обучающего набора и не входят ни в продукт,
  ни в его выход;
* заменяемы: `bar_from_pairs` калибрует порог под любой другой распознаватель,
  и переход стоит одной константы;
* до решения юриста это помечено так же, как в своё время Sapiens: **вопрос
  открыт**, и молча считать его закрытым нельзя.

ЧТО ИЗМЕРЕНО. Команда, которой получены все числа ниже (весь kit, 71 кадр
driving, лицо найдено на всех, 2485 пар своих и 71 пара чужих):

    python3 -m ball_reel.synth --calibrate kit/driving --other kit/face.jpg

    один человек (driving x driving)    p95 0.242   максимум 0.343
    другой человек (face.jpg x driving) минимум 0.713
    разрыв между облаками               +0.370
    порог, посчитанный из облаков       0.36

Число 0.30 пришло из таблицы порогов `deepface` для Facenet512 — то есть из
ДРУГОЙ реализации, и брать его на веру было нельзя. Замер показал, что оно
лежит там, где нужно: выше p95 своих и много ниже ближайшего чужого.

ПОЧЕМУ ОСТАВЛЕНО 0.30, А НЕ ПОСЧИТАННЫЕ 0.36. Оба стоят в разрыве, и это
осознанный выбор, а не расхождение: калибровка считает порог по ЖИВЫМ снимкам
одного человека, а отбирать им предстоит ПОРОЖДЁННЫЕ кадры, где риск ровно
один — лицо уехало. Цена ошибок несимметрична: пропущенный чужой кадр учит
LoRA чужому лицу, выброшенный свой стоит одного кадра из оплаченных, и запас
`dataset.OVERSHOOT` заложен как раз на это. `--bar` меняет порог из командной
строки; `bar_from_pairs` пересчитает его под другой распознаватель целиком.

ЧТО НЕ ИЗМЕРЕНО. Разделение проверено на ЖИВЫХ снимках одного человека.
Порождённые кадры труднее: генератор уводит лицо, и распределение своих у них
шире. Первый живой прогон это и покажет — доля отсева печатается и кладётся в
отчёт. Пока это НЕПРОВЕРЕНО.

ДЕНЬГИ. Каждый кадр стоит pollen. Смета печатается ДО первого вызова, и без
`--yes` модуль ничего не тратит.
"""

from __future__ import annotations

from pathlib import Path

# Шлюз импортируется НА УРОВНЕ МОДУЛЯ, вопреки принятому здесь ленивому стилю,
# и причина одна: иначе его нечем заменить в тестах. Шаг, который тратит
# деньги, обязан проверяться без денег, а `from . import pollinations` внутри
# функции создаёт имя, до которого подмена не дотягивается.
from . import pollinations

#: Порог отбора. ИЗМЕРЕН — см. шапку модуля. Он ОБЯЗАН отличаться от того,
#: которым потом судят обученную LoRA (`identity_arcface.SAME_PERSON_MAX`):
#: совпадение семейств означало бы, что судья участвует в отборе.
FACENET_BAR = 0.30

#: Записанные распределения замера. Хранятся числами, а не абзацем, потому что
#: по ним проверяется, что порог всё ещё стоит В РАЗРЫВЕ, а не съехал в одно из
#: облаков после правки.
MEASURED_SAME_P95 = 0.242
MEASURED_SAME_MAX = 0.343
MEASURED_OTHER_MIN = 0.713

#: Порог, посчитанный из тех же облаков `bar_from_pairs`. Хранится рядом с
#: рабочим, потому что они РАЗНЫЕ, и молча выбрать один из двух нельзя: см.
#: «почему оставлено 0.30» в шапке.
MEASURED_BAR = 0.36

#: Модель шлюза для image-to-image. `kontext` — умолчание `pollinations.
#: images_edit`, проверенное живьём на этом проекте.
MODEL = "kontext"

#: Размер порождаемого кадра. Квадрат, потому что обучение идёт в квадрате
#: (`lora.config().resolution`), и кадрировать потом значит терять края там,
#: где генератор их и рисовал.
SIZE = 1024

#: Ниже этой доли отобранных считаем, что порождение не состоялось: либо промт
#: уводит лицо, либо модель не держит референс. ВЫБРАН как «меньше половины
#: оплаченного ушло в мусор» — при `dataset.OVERSHOOT` = 2.5 запас расчитан
#: примерно на такой отсев.
MIN_KEEP_SHARE = 0.4

_NET = None
_DET = None


def _models():
    """Ленивая загрузка распознавателя. Веса тянутся один раз на процесс."""
    global _NET, _DET
    if _NET is None:
        from facenet_pytorch import InceptionResnetV1, MTCNN

        _DET = MTCNN(image_size=160, margin=20, post_process=True)
        _NET = InceptionResnetV1(pretrained="vggface2").eval()
    return _DET, _NET


def embed(path) -> "object | None":
    """Кадр -> нормированный вектор лица, или None, если лица нет.

    None — это НЕ ноль и не «далеко»: это «не смогли измерить», и `dataset.
    select` держит для него отдельный исход. Кадр без лица, посчитанный
    «непохожим», молча исказил бы долю отсева, по которой судят о порождении.
    """
    import numpy as np
    import torch
    from PIL import Image

    det, net = _models()
    face = det(Image.open(path).convert("RGB"))
    if face is None:
        return None
    with torch.no_grad():
        v = net(face[None])[0].numpy()
    n = float(np.linalg.norm(v))
    return (v / n) if n else None


def distance(a, b) -> float:
    """Косинусная дистанция двух векторов: 0 — тот же, 2 — противоположный."""
    import numpy as np

    return float(1.0 - np.dot(a, b))


def face_distance(reference, path) -> float | None:
    """Насколько кадр далёк от референса. None = лица нет."""
    v = embed(path)
    return None if v is None else distance(reference, v)


def bar_from_pairs(same: list, other: list) -> dict:
    """Дистанции своих и чужих -> порог, лежащий В РАЗРЫВЕ между ними.

    Порог не выбирается по вкусу и не переносится из чужой реализации: он
    считается из двух облаков. Если облака перекрылись, честный ответ — «этим
    распознавателем отбирать нельзя», а не порог посередине перекрытия.
    """
    import numpy as np

    if not same or not other:
        return {"ok": False, "bar": None,
                "note": "нечем калибровать: пустое облако своих или чужих"}
    s, o = np.array(same, dtype=float), np.array(other, dtype=float)
    p95, top, low = (float(np.percentile(s, 95)), float(s.max()),
                     float(o.min()))
    if low <= p95:
        return {"ok": False, "bar": None, "same_p95": p95, "same_max": top,
                "other_min": low,
                "note": (f"облака перекрылись: свои доходят до {p95:.3f} (p95), "
                         f"чужие начинаются с {low:.3f}. Порога, разделяющего "
                         f"их, не существует — этим распознавателем отбирать "
                         f"нельзя")}
    # Порог ближе к своим, а не посередине: в отборе набора цена ошибок
    # несимметрична. Пропустить чужой кадр значит учить LoRA чужому лицу;
    # выбросить свой значит потратить один кадр из оплаченных.
    bar = round(p95 + (low - p95) * 0.25, 3)
    return {"ok": True, "bar": bar, "same_p95": p95, "same_max": top,
            "other_min": low, "gap": round(low - top, 3),
            "note": (f"свои до {p95:.3f} (p95, максимум {top:.3f}), чужие от "
                     f"{low:.3f}; порог {bar} стоит в разрыве, ближе к своим")}


def calibrate(same_dir, other_paths: list) -> dict:
    """Каталог кадров одного человека + кадры другого -> порог. Без сети."""
    import glob
    import itertools

    paths = sorted(glob.glob(f"{same_dir}/*.jpg") + glob.glob(f"{same_dir}/*.png"))
    vecs = [(p, embed(p)) for p in paths]
    found = [(p, v) for p, v in vecs if v is not None]
    same = [distance(a[1], b[1]) for a, b in itertools.combinations(found, 2)]
    other = []
    for op in other_paths:
        ov = embed(op)
        if ov is None:
            continue
        other += [distance(ov, v) for _, v in found]
    got = bar_from_pairs(same, other)
    got.update({"same_frames": len(found), "same_pairs": len(same),
                "other_pairs": len(other),
                "no_face": [p for p, v in vecs if v is None]})
    return got


def prompts_for(count: int, subject: str = "a person") -> list:
    """Сколько заказали кадров -> столько запросов, по одной оси различия.

    Оси берутся у `dataset.variations`, а не сочиняются здесь: там они уже
    подобраны так, чтобы каждая следующая комбинация отличалась от базовой
    ОДНОЙ вещью — так на кадр покупается максимум различия. Если заказали
    больше, чем есть комбинаций, список идёт по кругу: повтор оси с другим
    сидом честнее, чем выдуманная девятая ось.
    """
    from .dataset import prompt_for, variations

    rows = variations()
    if not rows:
        return []
    return [prompt_for(rows[i % len(rows)], subject) for i in range(count)]


def generate(face: str, out_dir, *, count: int, subject: str = "a person",
             model: str = MODEL, seed: int = 0, bar: float = FACENET_BAR,
             size: int = SIZE, verbose: bool = True) -> dict:
    """Фото -> `count` порождённых кадров на диске, отобранных FaceNet.

    Возвращает отчёт целиком, включая ОТБРОШЕННЫЕ кадры и их дистанции: доля
    отсева — главное число этого шага, и по нему видно, состоялось ли
    порождение вообще. Отчёт, в котором остались только принятые кадры,
    выглядит одинаково при отсеве 5% и 80%.

    Файлы принятых кадров кладутся в `out_dir`, отброшенные — в
    `out_dir/dropped`, а не удаляются: на них смотрят глазами, когда доля
    отсева не сходится с ожиданием.
    """
    out = Path(out_dir)
    (out / "dropped").mkdir(parents=True, exist_ok=True)

    # ОТБИРАЮЩИЙ РАСПОЗНАВАТЕЛЬ ПРОВЕРЯЕТСЯ ДО ПЕРВОЙ ТРАТЫ. Порядок здесь
    # денежный, а не гигиенический: узнать, что отбирать нечем, ПОСЛЕ двадцати
    # оплаченных кадров значит заплатить за набор, который придётся выбросить
    # целиком — принять всё подряд нельзя, это и есть потеря независимости.
    try:
        ref = embed(face)
    except ImportError as e:
        return {"ok": False, "kept": [], "rows": [],
                "note": (f"отбирающий распознаватель не поставлен ({e}). "
                         f"Лечение: python3 -m pip install --no-deps "
                         f"facenet-pytorch. Без него отбирать нечем, а принять "
                         f"всё подряд — это тот самый замкнутый круг")}
    if ref is None:
        return {"ok": False, "kept": [], "rows": [],
                "note": (f"на референсе {face} лицо не найдено отбирающим "
                         f"распознавателем. Отбирать нечем — порождать "
                         f"бессмысленно: примем всё подряд")}

    rows, kept = [], []
    for i, prompt in enumerate(prompts_for(count, subject)):
        tmp = out / "dropped" / f"gen_{i:04d}.png"
        try:
            pollinations.images_edit(prompt, face, tmp, model=model,
                                     width=size, height=size)
        except Exception as e:  # noqa: BLE001 — один упавший кадр не роняет шаг
            rows.append({"i": i, "prompt": prompt, "distance": None,
                         "kept": False, "note": f"шлюз отказал: {e}"})
            if verbose:
                print(f"  {i:3d} ОТКАЗ  {str(e)[:70]}")
            continue
        d = face_distance(ref, tmp)
        ok = d is not None and d <= bar
        if ok:
            dst = out / f"gen_{i:04d}.png"
            tmp.replace(dst)
            kept.append(str(dst))
        rows.append({"i": i, "prompt": prompt, "distance": d, "kept": ok,
                     "path": str(out / f"gen_{i:04d}.png") if ok else str(tmp),
                     "note": ("лицо не найдено" if d is None
                              else f"{d:.3f} {'<=' if ok else '>'} {bar}")})
        if verbose:
            print(f"  {i:3d} {'ВЗЯТ ' if ok else 'ОТСЕВ'}  "
                  f"{'—' if d is None else f'{d:.3f}'}  {prompt[:58]}")

    asked = len(rows)
    share = (len(kept) / asked) if asked else 0.0
    enough = share >= MIN_KEEP_SHARE
    return {
        "ok": bool(kept), "kept": kept, "rows": rows, "bar": bar,
        "model": model, "asked": asked, "keep_share": round(share, 3),
        "keep_share_ok": enough,
        "generator": f"pollinations/{model}",
        "selector": "facenet/InceptionResnetV1-vggface2",
        "note": (f"взято {len(kept)} из {asked} ({share:.0%}) при баре {bar}"
                 + ("" if enough else
                    f"; НИЖЕ {MIN_KEEP_SHARE:.0%}: либо промт уводит лицо, "
                    f"либо модель не держит референс — смотреть кадры в "
                    f"{out / 'dropped'} глазами, а не поднимать бар")),
    }


def main(argv: list) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="ball_reel.synth",
        description="породить кадры набора из одной фотографии через шлюз")
    ap.add_argument("--face", default="", help="фотография личности")
    ap.add_argument("--out", default="generated", help="куда класть кадры")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--subject", default="a person",
                    help="как называть человека в запросе; личность словами НЕ "
                         "описывается — её несёт референс")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--bar", type=float, default=FACENET_BAR)
    ap.add_argument("--yes", action="store_true",
                    help="согласие потратить pollen; без него только смета")
    ap.add_argument("--calibrate", default="",
                    help="каталог кадров ОДНОГО человека: посчитать порог")
    ap.add_argument("--other", default="",
                    help="кадр ДРУГОГО человека для калибровки (можно "
                         "несколько через запятую)")
    args = ap.parse_args(argv)

    if args.calibrate:
        got = calibrate(args.calibrate,
                        [p for p in args.other.split(",") if p])
        print(f"кадров своих: {got['same_frames']}, пар своих "
              f"{got['same_pairs']}, пар чужих {got['other_pairs']}")
        if got.get("no_face"):
            print(f"без лица: {len(got['no_face'])}")
        print(got["note"])
        if got["ok"]:
            print(f"\nтекущая константа FACENET_BAR = {FACENET_BAR}; "
                  f"посчитано {got['bar']}")
        return 0 if got["ok"] else 1

    if not args.face:
        ap.error("нужен --face (или --calibrate)")

    from .dataset import plan

    est = plan(args.count)
    print(f"смета: {args.count} кадров, ~{est['pollen']} pollen "
          f"(по {est['pollen'] / max(1, est['generate']):.3f} за кадр)")
    print(f"порождение: pollinations/{args.model}; отбор: FaceNet, бар "
          f"{args.bar}; суд потом — ArcFace, и он в этом шаге не участвует")
    if not args.yes:
        print("\nничего не потрачено. Повторить с --yes, чтобы породить.")
        return 0

    got = generate(args.face, args.out, count=args.count, subject=args.subject,
                   model=args.model, bar=args.bar)
    print("\n" + got["note"])
    (Path(args.out) / "synth_report.json").write_text(
        json.dumps(got, indent=2, ensure_ascii=False), encoding="utf-8")
    if got["ok"]:
        print(f"\nдальше: python3 -m ball_reel.dataset --face {args.face} "
              f"--out ds --generated {args.out}")
    return 0 if got["ok"] and got["keep_share_ok"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
