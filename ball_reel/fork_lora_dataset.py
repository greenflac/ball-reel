"""LoRA темплейта: сборка набора под ТИП ФИГУРЫ, из которого убраны лица.

ЧТО ЭТА LoRA ДЕЛАЕТ И ЧЕГО НЕ ДЕЛАЕТ. Она привязана к темплейту и отвечает за
телосложение и домен отрисовки — «женщина за 40 с большими бёдрами», «аниме,
молодой мужчина». Учится один раз при сборке темплейта; обучения на клиента нет.
Личность приходит НЕ отсюда: под неё у модели выделен `face_adapter` на 1.5628
ГБ и вход `reference_image`.

---

ГЛАВНАЯ ОПАСНОСТЬ, РАДИ КОТОРОЙ НАПИСАН МОДУЛЬ, И ПОЧЕМУ ОНА НЕ ЛЕЧИТСЯ
ОБЕЩАНИЕМ.

LoRA, обученная на изображениях «женщин за 40», выучит и их ЛИЦА — и потянет
лицо клиента к среднему по категории. Это тот же риск утечки, что убил LoRA
личности, только источник другой: не актёр драйвинга, а сама категория.

Лечение направлением известно — «учить на кропах тела с исключёнными лицами».
Но НАПРАВЛЕНИЕ НЕ ЕСТЬ ГАРАНТИЯ. Кроп «ниже шеи» промахивается ровно там, где
это дороже всего: наклон, поворот, лежащая поза, второй человек в кадре, отражение.
Набор, про который СКАЗАНО «лиц нет», и набор, в котором лиц нет, — разные вещи,
и отличить их можно только прибором.

ПОЭТОМУ ЗДЕСЬ ЛИЦА НЕ «ИСКЛЮЧАЮТСЯ», А ИСКЛЮЧАЮТСЯ И ПРОВЕРЯЮТСЯ. Каждый
готовый кроп прогоняется ТЕМ ЖЕ детектором лиц, которым судится личность
(`identity_arcface`), и кроп, где детектор нашёл лицо, В НАБОР НЕ ИДЁТ. Не
«помечается», не «понижается в весе» — отбрасывается.

Негативный контроль обязателен и стоит рядом (И5): на кропе, СОДЕРЖАЩЕМ лицо,
проверка обязана сработать. Проверка, которая всегда говорит «лиц нет», говорит
это и про набор, полный лиц.

---

ПОЧЕМУ НИЗКИЙ РАНГ — ОБОСНОВАНИЕ, А НЕ ОБЪЯВЛЕНИЕ.

Ранг LoRA ограничивает размерность добавки. Телосложение и домен отрисовки —
низкочастотные, крупномасштабные свойства: пропорции силуэта, распределение
объёма, характер штриха. Личность — высокочастотное и мелкомасштабное: взаимное
положение черт в области в несколько сотен пикселей.

Низкий ранг поэтому работает не как «поменьше на всякий случай», а как
СТРУКТУРНОЕ ограничение ёмкости: адаптеру, которому дали мало направлений,
дешевле выучить общую форму, чем набор конкретных лиц. Это АРГУМЕНТ, а не
замер — на нашем материале он не проверялся, и проверяется он ровно одним
вопросом приёмки: портится ли `d_raw` при включённой LoRA.

ЧИСЛО. `DEFAULT_RANK = 16`, и оно ВЫБРАНО, а не измерено. Опорная точка,
прочитанная из файла тренера, а не по памяти: у `DiffSynth-Studio`
`examples/wanvideo/model_training/train.py:15` умолчание `lora_rank=32`
(команда — в `docs/FORK_LORA.md`). Берём вдвое ниже умолчания, потому что
умолчание рассчитано на задачу «выучить облик», а наша — «выучить сложение»,
и ёмкость здесь ограничивают нарочно. Двигать это число можно только замером
`d_raw` с LoRA и без.

---

НЕПРОВЕРЕНО (Ц4), наверх:

* **опыт «LoRA поверх Wan-Animate против референса без LoRA» не проводился.**
  Всё, что здесь сказано про то, кто победит в споре личности и телосложения, —
  аргументы, не замеры;
* **обучение не запускалось.** Модуль собирает НАБОР и проверяет его. Что на
  таком наборе LoRA выучит телосложение, здесь не показано;
* **`d_raw` с включённой LoRA не измерен** — измерить его без карты нельзя.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import fork_channels
from .fork_identity import FAIL, PASS, UNMEASURED

#: Кадрировки. НЕ СВОЙ список: берётся у `skeleton`, потому что это один и тот
#: же словарь понятий, и разъехавшись, он дал бы набор под кадрировку, которой
#: пайплайн не умеет строить (Е1).
from .skeleton import FRAMINGS  # noqa: E402

#: Под какую кадрировку собирается набор. **Во весь рост — решение владельца**
#: (ХЭНДОФ §2), и набор обязан идти за ним: если темплейт отдаёт полный рост, а
#: набор собран по пояс, LoRA учится на одном масштабе тела и применяется на
#: другом.
#:
#: ЦЕНА ЭТОГО РЕШЕНИЯ ИЗМЕРЕНА И ЗАКРЫВАЕТСЯ НЕ ЗДЕСЬ. Полный рост при высоте
#: 848 даёт лицо 63–80 px против бара ArcFace в 100 px для видео — то есть
#: покадровая ось личности будет чаще всего возвращать «СУДИТЬ НЕЧЕМ», и это
#: отсутствие измерения, а не провал. Лечится не кадрировкой набора, а
#: доводкой лица на выходе, которая по ХЭНДОФ §2 объявлена несущим шагом
#: стека; личность меряется ПОСЛЕ неё.
#:
#: Числа не наши и здесь не пересчитываются: `skeleton.face_share()` и
#: `identity_arcface`, сведены в ХЭНДОФ §4.
#:
#: DEBT(2026-08-17): здесь стояло `waist_up` — я поставил его по предыдущей
#: редакции брифинга, где кадрировка ещё не была решена владельцем. Финальный
#: хэндоф решает иначе; перечёркнуто, а не стёрто, потому что тесты на
#: `waist_up` остаются годными и нужны, если решение когда-нибудь вернётся.
DEFAULT_FRAMING = "full_body"

#: Ранг адаптера. ВЫБРАНО — обоснование в докстринге модуля. Опорная точка:
#: умолчание тренера 32 (прочитано из train.py:15), берём вдвое ниже.
DEFAULT_RANK = 16

#: Умолчание тренера, ИЗМЕРЕНО чтением файла. Держится рядом с нашим числом,
#: чтобы «вдвое ниже умолчания» можно было проверить, а не принять на слово.
TRAINER_DEFAULT_RANK = 32

#: Сколько отступить ВНИЗ от самой нижней точки головы, в долях высоты кропа
#: тела. Не ноль: шея и подбородок дают детектору достаточно, чтобы поймать
#: лицо, а нам они не нужны — телосложение начинается с плеч.
#: ВЫБРАНО 0.04; проверяется не этим числом, а прибором ниже.
CHIN_MARGIN = 0.04

#: Минимальная сторона кропа, px. Кроп мельче не несёт телосложения — он несёт
#: шум масштабирования. ВЫБРАНО.
MIN_CROP_PX = 256

#: Доля кадров, которую допустимо потерять на отбраковке, прежде чем набор
#: считается негодным. ВЫБРАНО: если больше половины кропов не прошли проверку
#: на лица, значит кадрирование промахивается систематически, и чинить надо
#: его, а не добирать примеры.
MAX_REJECT_SHARE = 0.5

#: Минимальный размер набора. ВЫБРАНО: ниже этого разговор о телосложении
#: ведётся по единичным примерам.
MIN_SAMPLES = 12


class FaceLeak(ValueError):
    """В готовом кропе найдено лицо. Набор с таким кропом не собирается."""


def _instrument():
    from . import identity_arcface

    return identity_arcface


def body_box(points, width: int, height: int,
             framing: str = DEFAULT_FRAMING) -> tuple | None:
    """Прямоугольник тела БЕЗ головы, в пикселях, или None.

    Верхняя граница — ниже самой нижней точки головы, а не «на уровне плеч»:
    при наклоне плечо оказывается ВЫШЕ подбородка, и рез по плечам оставил бы
    в кадре половину лица. Голова берётся целиком — 68 лицевых точек плюс
    нос, глаза и уши COCO, — потому что край челюсти в лицевые точки входит,
    а ухо нет.

    НИЖНЯЯ граница зависит от КАДРИРОВКИ, и это не украшение. Набор собирается
    под ту кадрировку, которую темплейт реально даёт: если темплейт по пояс, а
    набор в рост, LoRA учится на одном масштабе тела, а применяется на другом.
    `waist_up` режет по тазу — ровно там же, где `skeleton._window`, и по той
    же причине, что записана там: опущенные руки иначе вернут кадр к полному
    росту и вся затея развалится.
    """
    # Негодный аргумент отвергается ДО любой работы (П2): проверка стоит
    # микросекунды, а стояла она ниже разбора точек.
    if framing not in FRAMINGS:
        raise ValueError(
            f"неизвестная кадрировка {framing!r}; бывают только {FRAMINGS}")
    if not points:
        return None
    n = len(points)

    def seen(indices):
        return [points[i] for i in indices
                if i < n and points[i][2] >= fork_channels.MIN_SCORE]

    head = seen(list(fork_channels.group_indices("face")) + list(range(0, 5)))
    body = seen([i for i in fork_channels.channel_indices("body")
                 if i >= 5])
    if not body:
        return None

    xs = [p[0] for p in body]
    ys = [p[1] for p in body]
    x0, x1 = max(0.0, min(xs)), min(float(width), max(xs))
    if framing == "waist_up":
        hips = [points[i][1] for i in (11, 12)
                if i < n and points[i][2] >= fork_channels.MIN_SCORE]
        if not hips:
            return None
        y1 = min(float(height), max(hips))
    else:
        y1 = min(float(height), max(ys))

    if head:
        chin = max(p[1] for p in head)
        y0 = chin + CHIN_MARGIN * max(1.0, y1 - chin)
    else:
        # Головы не видно — резать не от чего. Это НЕ «лица нет»: голова могла
        # быть просто не распознана. Верх берётся по телу, а решает прибор.
        y0 = max(0.0, min(ys))

    y0 = max(0.0, min(y0, float(height)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (x0, y0, x1, y1)


def face_free(path: str | Path) -> dict:
    """Есть ли на изображении лицо, по мнению ТОГО ЖЕ прибора, что судит личность.

    Три исхода, и третий не декоративный: прибор, который не смог отработать,
    не должен читаться как «лиц нет». Именно так набор с лицами и прошёл бы.
    """
    try:
        detail = _instrument().face_detail(path)
    except Exception as exc:  # noqa: BLE001
        return {"outcome": UNMEASURED, "face_px": None,
                "note": f"прибор не отработал: {str(exc)[:120]}"}
    if detail is None:
        return {"outcome": PASS, "face_px": None, "note": "лица не найдено"}
    return {"outcome": FAIL, "face_px": detail["face_px"],
            "note": (f"найдено лицо {detail['face_px']}px — кроп в набор не "
                     f"идёт")}


def crop_sample(frame_path: str | Path, out_path: str | Path,
                *, framing: str = DEFAULT_FRAMING) -> dict:
    """Один кроп тела без головы, ПРОВЕРЕННЫЙ прибором. Три исхода.

    Проверка идёт ПОСЛЕ записи кропа и по самому файлу, а не по намерению:
    геометрия могла промахнуться, изображение могло быть перевёрнуто, поза
    могла быть лежачей. Судить надо то, что поедет в обучение.
    """
    from PIL import Image

    src = Path(frame_path)
    points = fork_channels.wholebody_points(src)
    if points is None:
        return {"outcome": UNMEASURED, "path": None,
                "note": f"{src.name}: человека не нашли, резать нечего"}

    with Image.open(src) as im:
        rgb = im.convert("RGB")
        box = body_box(points, *rgb.size, framing=framing)
        if box is None:
            return {"outcome": UNMEASURED, "path": None,
                    "note": f"{src.name}: тела в кадре нет"}
        x0, y0, x1, y1 = (int(v) for v in box)
        if min(x1 - x0, y1 - y0) < MIN_CROP_PX:
            return {"outcome": FAIL, "path": None,
                    "note": (f"{src.name}: кроп {x1 - x0}x{y1 - y0} мельче "
                             f"{MIN_CROP_PX}px — это шум масштабирования, а не "
                             f"телосложение")}
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rgb.crop((x0, y0, x1, y1)).save(out)

    checked = face_free(out)
    if checked["outcome"] != PASS:
        out.unlink(missing_ok=True)
        return {"outcome": checked["outcome"], "path": None,
                "note": f"{src.name}: {checked['note']}"}
    return {"outcome": PASS, "path": str(out), "box": (x0, y0, x1, y1),
            "note": f"{src.name}: кроп {x1 - x0}x{y1 - y0}, лица не найдено"}


def build(frame_paths, out_dir: str | Path, *, build_type: str,
          domain: str, rank: int = DEFAULT_RANK,
          framing: str = DEFAULT_FRAMING) -> dict:
    """Собрать набор темплейта и написать его паспорт. Отчёт числами (Р2).

    `build_type` и `domain` — ВХОД, а не вывод. Классификатор телосложения
    здесь не строится: для MVP тип фигуры выбирает человек, а неверный выбор
    виден глазом.
    """
    out = Path(out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)

    frames = [Path(p) for p in frame_paths]
    kept, face_rejected, unmeasured, too_small = [], [], [], []
    for i, p in enumerate(frames):
        got = crop_sample(p, out / "img" / f"body_{i:04d}.png",
                          framing=framing)
        if got["outcome"] == PASS:
            kept.append(got["path"])
        elif got["outcome"] == UNMEASURED:
            unmeasured.append(got["note"])
        elif "мельче" in got["note"]:
            too_small.append(got["note"])
        else:
            face_rejected.append(got["note"])

    total = len(frames)
    rejected = len(face_rejected)
    reject_share = round(rejected / total, 3) if total else 0.0

    problems = []
    if len(kept) < MIN_SAMPLES:
        problems.append(f"кропов {len(kept)}, нужно хотя бы {MIN_SAMPLES}")
    if reject_share > MAX_REJECT_SHARE:
        problems.append(
            f"на лицах отбраковано {reject_share:.0%} — кадрирование "
            f"промахивается систематически, чинить его, а не добирать примеры")

    passport = {
        "build_type": build_type, "domain": domain, "rank": rank,
        "framing": framing,
        "trainer_default_rank": TRAINER_DEFAULT_RANK,
        "samples": len(kept), "from_frames": total,
        "face_rejected": rejected, "too_small": len(too_small),
        "unmeasured": len(unmeasured),
        "acceptance": acceptance_for(domain),
        "faces": ("проверено прибором identity_arcface на КАЖДОМ кропе; "
                  "кроп с найденным лицом в набор не попал"),
        "unrun": ("НЕПРОВЕРЕНО: обучение не запускалось, d_raw с включённой "
                  "LoRA не измерен"),
    }
    (out / "passport.json").write_text(
        json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "outcome": (UNMEASURED if not total else
                    FAIL if problems else PASS),
        "kept": kept, "face_rejected": face_rejected,
        "too_small": too_small, "unmeasured": unmeasured,
        "reject_share": reject_share, "passport": passport,
        "dir": str(out), "problems": problems,
        "note": (f"из {total} кадров в набор пошло {len(kept)}; "
                 f"на лицах отбраковано {rejected} ({reject_share:.0%}), "
                 f"мелких {len(too_small)}, не смогли проверить "
                 f"{len(unmeasured)}. "
                 + ("; ".join(problems) if problems else "набор годен")),
    }


#: Критерий приёмки ОБЪЯВЛЯЕТСЯ У КАЖДОГО ТЕМПЛЕЙТА, а не берётся один на всех.
#: Причина не организационная: для стилизованного домена ArcFace перестаёт быть
#: верным судьёй — фотографического сходства там не ожидается, и НИЗКОЕ
#: расстояние до сырого фото означало бы, что стилизация НЕ СРАБОТАЛА. Один
#: критерий на оба домена молча поменял бы знак у половины каталога.
ACCEPTANCE = {
    "photoreal": ("ArcFace против сырой фотографии, планка 0.35; "
                  "d_raw не должен ухудшиться при включённой LoRA"),
    "stylised": ("ArcFace НЕ судья: низкое расстояние здесь означало бы, что "
                 "стилизация не сработала. Приёмка человеком, критерий "
                 "объявляется в паспорте темплейта до сборки"),
}


def acceptance_for(domain: str) -> str:
    """Критерий приёмки домена. Неизвестный домен — отказ, а не умолчание.

    Умолчание здесь означало бы «судим аниме ArcFace'ом», то есть ровно ту
    ошибку, ради предотвращения которой таблица и заведена.
    """
    if domain not in ACCEPTANCE:
        raise ValueError(
            f"неизвестный домен отрисовки: {domain!r}. Известны: "
            f"{', '.join(ACCEPTANCE)}. Критерий приёмки объявляется до сборки "
            f"темплейта, а не выясняется на демо.")
    return ACCEPTANCE[domain]
