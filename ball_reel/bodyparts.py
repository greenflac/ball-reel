"""Где на кадре кожа, а где ткань. Сегментация частей тела на CPU.

ЗАЧЕМ ЭТОТ МОДУЛЬ ПОЯВИЛСЯ. В `marks.transfer` вклейка приметы ограничена
КАПСУЛОЙ вокруг кости: геометрической фигурой, за пределами которой заведомо не
тело. Капсула решает «не рисовать мимо руки» и не решает «не рисовать поверх
рукава» — она не знает, где кончается кожа. Это было записано как долг, и вот
чем он закрывается.

Цветовой признак «похоже на кожу» здесь уже пробовался и был ОПРОВЕРГНУТ
замером: на живом кадре плечо (кожа) дало разброс p50 0.063 / p90 0.586, а бедро
в леггинсах (ткань) — p50 0.021 / p90 0.031. Одежда оказалась ОДНОРОДНЕЕ кожи,
потому что в окно на руке попадает силуэт и светотень. Признак назывался «кожа»,
а мерил однородность. Отсюда правило: разделение кожи и ткани — задача
сегментации, а не цветового порога.

ЧТО ИМЕННО ВЗЯТО И ПОЧЕМУ ИМЕННО ЭТО

MediaPipe Image Segmenter, модель `selfie_multiclass_256x256`. Причины ровно
три, и все проверяемые:

* **лицензия Apache 2.0** — без оговорок, ничего не надо согласовывать.

  ЗДЕСЬ СТОЯЛО, ЧТО Sapiens2 ОТВЕРГНУТ ПО ЛИЦЕНЗИИ. Это устарело в тот же
  день: решение пересмотрено владельцем продукта после юридической проверки,
  и Sapiens2 был скачан и ИСПОЛНЕН на наших кадрах (коммит 90f720b,
  замеры — `PRIOR_ART.md`, раздел «ПРОВЕРЕНО ЖИВЬЁМ»). Комментарий пережил
  решение на несколько часов и потом дважды вводил в заблуждение — второй раз
  меня же.

  Действующее разделение такое: **MediaPipe остаётся рабочей лошадкой**
  сегментации (16 МБ, CPU, зависимость уже стоит), а Sapiens2 берётся там, где
  он даёт то, чего у MediaPipe нет вовсе, — 29 классов и pointmaps. Это не
  запрет, а разница в цене: 1.63 ГБ и 12-17 с на кадр против 16 МБ и
  миллисекунд;
* **зависимость уже стоит** — MediaPipe в проекте с самого начала, новых
  колёс не завозим;
* **16 МБ и CPU** — сегментация не отнимает карту у генерации.

Список классов прочитан НЕ из статьи, а из метаданных самого файла модели
(`labels.txt` внутри .tflite), потому что на этом проекте уже был случай, когда
имя весов оказалось выдуманным:

    0 background   1 hair   2 body-skin   3 face-skin   4 clothes   5 others

ЧЕГО ЭТА МОДЕЛЬ НЕ УМЕЕТ, И ЭТО НАДО ЗНАТЬ

* **Она не различает ЧАСТИ тела.** Левое предплечье и правое бедро для неё
  одинаково `body-skin`. Куда именно ставить примету, по-прежнему решает
  скелет; сегментация лишь говорит, можно ли туда рисовать.
* **Нативное разрешение 256x256.** Маска растягивается до размера кадра, то
  есть граница кожи и ткани грубая: на кадре 720 px по ширине один пиксель
  маски — это почти три пикселя картинки. Для «не залезь на рукав» этого
  хватает, для аккуратного края — нет, и край надо смягчать.
* **Она ничего не знает о теле в 3D** — ни глубины, ни поверхности. Задачу
  «та же самая точка кожи в другом кадре» она не решает; это `marks.limb_uv`.

Работает на CPU, ничего не генерирует, в сеть не ходит.
"""

from __future__ import annotations

from . import cure

from pathlib import Path

#: Классы модели, В ТОМ ЖЕ ПОРЯДКЕ, что в её собственном `labels.txt`.
#: Порядок здесь — не украшение: индекс класса и есть значение в маске.
LABELS = ("background", "hair", "body-skin", "face-skin", "clothes", "others")

#: На что МОЖНО наносить примету. Волосы и «прочее» (очки, аксессуары) сюда не
#: входят намеренно: татуировка под дужкой очков — это не примета на коже.
SKIN_CLASSES = (2, 3)

#: Что заведомо НЕ кожа, и вклейка поверх этого — брак, найденный глазами:
#: крест лёг поверх лямки топа, а метрика отрапортовала честные 74%
#: сохранившегося контраста. Число было правдой, картинка — нет.
COVERED_CLASSES = (1, 4, 5)

#: Куда класть веса. Тот же приём, что в `dwpose`: путь по умолчанию плюс
#: внятная инструкция, а не молчаливое падение импорта.
DEFAULT_MODEL = "~/.mediapipe/selfie_multiclass_256x256.tflite"

#: Откуда их взять. Хост проверен из этой среды и отвечает 200.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_multiclass_256x256/float32/latest/"
             "selfie_multiclass_256x256.tflite")

#: Ниже этой доли кожи в окне вклейка НЕ ДЕЛАЕТСЯ. Это не «мало кожи» в смысле
#: качества, а «мы целимся не туда»: если в окне приметы кожи меньше пятой
#: части, значит либо рука в рукаве, либо скелет промахнулся. ВЫБРАНО 0.2 —
#: калибровать не на чем, пока нет корпуса кадров с приметами.
MIN_SKIN_SHARE = 0.2

#: Нативная сторона входа модели. Нужна не для вызова (MediaPipe ресайзит сам),
#: а для честного отчёта о том, насколько груба граница маски.
NATIVE_SIDE = 256

#: Разделитель пути ДЛЯ ОБОЛОЧКИ, а не для Python.
SEP = "\\" if cure.WINDOWS else "/"


def model_path(path: str | Path | None = None) -> Path:
    return Path(path or DEFAULT_MODEL).expanduser()


def available(path: str | Path | None = None) -> bool:
    """Есть ли чем сегментировать. Отсутствие весов — не ошибка, а состояние."""
    return model_path(path).exists()


def why_unavailable(path: str | Path | None = None) -> str:
    """Внятная причина вместо ImportError в середине прогона."""
    p = model_path(path)
    if p.exists():
        return ""
    return (f"нет модели сегментации {p}. Скачать:\n"
            f"  {cure.mkdir(cure.home('.mediapipe'))}\n"
            f"  {cure.download(MODEL_URL, cure.home('.mediapipe') + SEP + p.name)}\n"
            f"Без неё вклейка примет ограничена только капсулой вокруг кости, "
            f"то есть может лечь поверх одежды.")


_SEGMENTER = None


def _segmenter(path: str | Path | None = None):
    """Модель грузится один раз на процесс: 16 МБ и инициализация XNNPACK."""
    global _SEGMENTER
    if _SEGMENTER is None:
        from mediapipe.tasks.python import BaseOptions  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        p = model_path(path)
        if not p.exists():
            raise FileNotFoundError(why_unavailable(path))
        _SEGMENTER = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(p)),
                output_category_mask=True,
                running_mode=vision.RunningMode.IMAGE))
    return _SEGMENTER


def _as_uint8(image):
    """Путь, массив 0..1 или массив 0..255 -> uint8 RGB. Один вход на модуль."""
    import numpy as np
    from PIL import Image

    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        with Image.open(image) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def category_mask(image, *, model=None):
    """Кадр -> маска классов той же формы, что кадр. Значения — индексы LABELS."""
    import mediapipe as mp  # type: ignore
    import numpy as np

    rgb = _as_uint8(image)
    res = _segmenter(model).segment(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return np.asarray(res.category_mask.numpy_view()).reshape(rgb.shape[:2])


def skin_mask(image, *, model=None):
    """Булева маска «сюда можно рисовать»: кожа тела и кожа лица."""
    import numpy as np

    cat = category_mask(image, model=model)
    return np.isin(cat, SKIN_CLASSES)


def shares(image, *, model=None) -> dict:
    """Доли классов в кадре. Полезно как быстрая проверка вменяемости входа."""
    cat = category_mask(image, model=model)
    return {name: round(float((cat == i).mean()), 4)
            for i, name in enumerate(LABELS)}


def paintable(image, box, *, model=None, min_share: float = MIN_SKIN_SHARE,
              mask=None) -> dict:
    """Можно ли наносить примету в это окно. Три исхода, и путать их дорого.

    * **`state="no_model"`** — сегментации нет. Это НЕ «нельзя» и НЕ «можно»:
      это отсутствие сведений, и вызывающий обязан решить сам, рисовать ли по
      одной капсуле. Молча вернуть True здесь значит выдать незнание за
      разрешение;
    * **`ok=False`** — в окне почти нет кожи: рука в рукаве или скелет
      промахнулся. Рисовать нельзя, и причина называется;
    * **`ok=True`** — можно, и сказано, какая доля окна закрыта тканью.

    `mask` позволяет передать уже посчитанную маску: на клипе из 60 кадров
    сегментировать один и тот же кадр по разу на примету — чистая трата.
    """
    import numpy as np

    x0, y0, x1, y1 = box
    if mask is None:
        if not available(model):
            return {"ok": None, "state": "no_model", "skin_share": None,
                    "note": why_unavailable(model)}
        mask = skin_mask(image, model=model)
    h, w = mask.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return {"ok": None, "state": "outside_frame", "skin_share": None,
                "note": "окно приметы целиком за краем кадра"}
    window = mask[y0:y1, x0:x1]
    share = float(window.mean())
    ok = share >= min_share
    return {
        "ok": bool(ok),
        "state": "measured",
        "skin_share": round(share, 4),
        "note": (f"кожи в окне {share:.0%}"
                 + ("" if ok else
                    f" < {min_share:.0%}: рисовать нельзя — либо конечность "
                    f"одета, либо скелет указал не туда")),
    }


def edge_coarseness(image_size) -> dict:
    """Насколько груба граница маски на кадре такого размера. Честная оговорка.

    Модель работает на стороне 256, маска растягивается. Значит один пиксель
    решения покрывает `сторона / 256` пикселей картинки, и край кожи не может
    быть точнее этого. Число полезно не само по себе, а как ответ на вопрос
    «почему вклейка чуть заходит на рукав»: потому что маска физически не умеет
    тоньше, и край надо смягчать не меньше, чем на эту величину.
    """
    w, h = image_size
    step = max(w, h) / NATIVE_SIDE
    return {
        "native_side": NATIVE_SIDE,
        "pixels_per_decision": round(step, 2),
        "note": (f"граница кожи и ткани определена с точностью около "
                 f"{step:.1f} px: модель решает на стороне {NATIVE_SIDE}, "
                 f"а кадр {w}x{h}. Смягчение края меньше этой величины "
                 f"ничего не улучшит"),
    }
