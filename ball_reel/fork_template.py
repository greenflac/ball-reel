"""Слой оператора: шаблон описывается ФАЙЛОМ, а не правкой кода.

ЗАЧЕМ ЭТОТ МОДУЛЬ. Требование владельца (ХЭНДОФ, «НУЖНО В MVP: слой
оператора»): завести новый драйвинг, фотореференс и промпт можно без правки
кода и без разработчика. Интерфейса в MVP нет — есть декларативный файл и эта
точка входа, которая его читает и ПРОВЕРЯЕТ. Проверка здесь не украшение:
единственный, кто раньше ловил кривую геометрию и кривую длину, был Comfy на
арендованной машине через девять секунд загрузки весов и по одной ошибке за
перезапуск (П2). Тут они ловятся за миллисекунду и все сразу.

ТРИ ИСХОДА (Р1). Каждая проверка отвечает `годно` / `не годно` /
`не смогли проверить`, и третий не сворачивается ни в один из двух. Рядом с
вердиктом всегда числа (Р2): проверено N, нарушено M, не смогли K. Ноль
нарушений при нуле проверок — НЕ успех, и `check` печатает это буквально.

ЧЕГО В ФАЙЛЕ НЕТ И НЕ БУДЕТ: ретаргетинга позы. Он выключен ВСЕГДА и не
является настройкой — ручка, которой нет, не поворачивается по ошибке. Поле с
таким именем роняет загрузку, а не игнорируется молча: молча проигнорированная
настройка выглядит для оператора как учтённая.

СТЕНД — ЭТО ТОТ ЖЕ ФАЙЛ, только указывающий на синтетические ассеты (поле
`bench`). Отсюда обратная опасность: испытательное описание, уехавшее в прод,
произведёт клип из заглушек и будет выглядеть как неудачная генерация вместо
негодного входа. За это отвечает `assert_production`.

ОДНО ЗНАНИЕ — ОДНО МЕСТО (Е1). Кратности сторон и шаг длины берутся у
`fork_comfy`, плечи маски — у `fork_mask`, схема разметки — у `fork_props`,
ступени весов — у `fork_preflight`, слова вердикта — у `fork_identity`.
Переписать их сюда числами значило бы завести второй способ узнать известное.
"""

# DEBT(2026-08-18): `test_reachable` краснеет на `fork_template` — модуль
# написан и не импортируется ни одним `ball_reel/*.py`. Это ТОТ ЖЕ структурный
# случай, что уже разобран в хэндофе для `fork_run`: у слоя оператора нет и не
# может быть вызывающего внутри пакета, потому что он сам точка входа. Лечится
# одной строкой в ЧУЖОМ файле `ball_reel/tests/test_reachable.py`:
#     "fork_template": "слой оператора, точка входа, запускается человеком",
# Строка не внесена: брифинг разрешает трогать только два файла (Ц2). Обходить
# циклическим импортом ради зелени нельзя — докстринг самого сторожа называет
# это способом заглушить тест.

from __future__ import annotations

import json
from pathlib import Path

from . import fork_build_route, fork_comfy, fork_mask, fork_preflight, fork_props
from .fork_identity import FAIL, PASS, UNMEASURED

#: Схема файла-описания. Отдельная от `fork_props.POINTS_SCHEMA`: это разные
#: файлы с разной судьбой, и общая строка склеила бы их версии.
TEMPLATE_SCHEMA = "fork-template/1"

#: Частота выхода. ВЫБРАНО владельцем (из чего: 24 против 30 — 24 портит
#: реализм; замер ХЭНДОФа показал, что 30 к/с стоит ровно столько же окон
#: сэмплера, сколько 24, и выбрасывает 3-19 кадров вместо 33-65). Интерполяции
#: нет: исходники снимаются ВЫШЕ и пересэмплируются вниз, кадры не выдумываются.
#: Отсюда же требование к съёмке драйвинга — не ниже этого числа.
FPS_OUT = 30

#: Геометрия выхода. ВЫБРАНО владельцем (из чего: 720p дороже по вниманию
#: примерно в пять раз — 72 000 токенов против 31 200; вертикаль при этом
#: бесплатна, 480x832 стоит ровно столько же, сколько 832x480).
WIDTH_OUT, HEIGHT_OUT = 480, 832

#: Границы длины клипа, секунды. ВЫБРАНО владельцем (из чего: ниже пяти секунд
#: нечего показывать, выше десяти растёт число окон сэмплера — 5 с это 2 окна,
#: 10 с уже 4, то есть вдвое дороже).
SECONDS_MIN, SECONDS_MAX = 5, 10

#: Плечо «пусть решит роутер». Слово ВЫБРАНО (кем: этот поток; из чего: нужно
#: было имя, которое НЕ является четвёртым плечом — оно обязано быть РАЗРЕШЕНО
#: до прогона, и до разрешения ось плеча честно отвечает «не смогли»).
AUTO_ARM = "auto"

#: Что вообще можно написать в поле плеча. Имена плеч НЕ переписаны — взяты у
#: `fork_mask.ARMS`, потому что там же лежат их `grow_px` и обоснования.
ARM_CHOICES = tuple(fork_mask.ARMS) + (AUTO_ARM,)

#: Ступени весов. Имена берутся у `fork_preflight.STEPS_GB`, где к ним привязан
#: вес на диске и арифметика бюджета памяти. Своего списка здесь нет намеренно:
#: разъехавшийся список ступеней означал бы шаблон, который проходит проверку
#: и не проходит предполёт.
STEP_CHOICES = tuple(sorted(fork_preflight.STEPS_GB))

#: Обломки имён, по которым узнаётся попытка завести ретаргетинг позы. Проверка
#: идёт по ПОДСТРОКЕ в имени поля и рекурсивно по вложенным словарям: оператор
#: напишет не то слово, которое мы угадали, а соседнее.
RETARGET_MARKERS = ("retarget", "ретаргет", "pose_transfer", "перенос_позы")

#: Поля, без которых описание не описание. Умолчаний у них нет НАРОЧНО: у
#: каждого цена ошибки разная в обе стороны, и молчаливое умолчание здесь — это
#: ровно тот дефект, из-за которого `--faceid-lora-scale 1.0` полсмены рисовал
#: радужные потёки, а разбор шёл не по адресу.
REQUIRED_FIELDS = ("schema", "name", "driving", "photo", "prompt", "negative",
                   "width", "height", "seconds", "step", "arm", "face_refine")

#: Необязательные поля и что значит их отсутствие. Отсутствие НЕ проверяется и
#: НЕ считается пройденной проверкой — оно попадает в `skipped`, потому что
#: «проверять было нечего» и «проверили и годно» — разные вещи (Р2).
OPTIONAL_FIELDS = {
    "fps": "частота выхода; отсутствие означает штатные 30",
    "frames": "число кадров; отсутствие означает «посчитать по длине»",
    "props": "файл разметки предметов",
    "protagonist": "рамка протагониста для сцен с несколькими людьми",
    "bench": "описание испытательное, ассеты синтетические",
}


class RetargetForbidden(ValueError):
    """Отдельный тип, чтобы запрет нельзя было поймать вместе с опечаткой."""


def _finding(axis: str, outcome: str, note: str) -> dict:
    return {"axis": axis, "outcome": outcome, "note": note}


def refuse_retarget(desc: dict, *, path: str | None = None) -> int:
    """Уронить загрузку, если в описании есть ЛЮБОЕ поле про ретаргетинг позы.

    Возвращает число просмотренных полей — чтобы «нарушений нет» нельзя было
    получить обходом нуля полей (Р2). Тот же дефект уже был пойман в
    `fork_props.load_marking`: обход шёл по ключу, которого в формате нет, и
    проверка всегда возвращала «чисто».

    Не предупреждение и не фильтрация. Поле, которое мы вычистили бы молча,
    осталось бы в файле оператора и в его голове как работающая настройка.
    """
    seen = 0
    stack = [(desc, path or "")]
    while stack:
        node, prefix = stack.pop()
        if isinstance(node, dict):
            items = node.items()
        elif isinstance(node, (list, tuple)):
            items = ((str(i), v) for i, v in enumerate(node))
        else:
            continue
        for key, value in items:
            seen += 1
            low = str(key).lower()
            hit = next((m for m in RETARGET_MARKERS if m in low), None)
            if hit is not None:
                where = f"{prefix}.{key}" if prefix else str(key)
                raise RetargetForbidden(
                    f"поле {where!r} про ретаргетинг позы (совпало по {hit!r}). "
                    f"Ретаргетинг ВЫКЛЮЧЕН ВСЕГДА и настройкой не является: "
                    f"ручка, которой нет, не поворачивается по ошибке. Убрать "
                    f"поле из файла — молча игнорировать его нельзя, оператор "
                    f"будет считать его учтённым")
            stack.append((value, f"{prefix}.{key}" if prefix else str(key)))
    return seen


def frames_for(seconds, *, fps=None) -> dict:
    """Сколько кадров просить у сэмплера на такую длину.

    ЗДЕСЬ ЖИВЁТ РАСХОЖДЕНИЕ, КОТОРОЕ НЕЛЬЗЯ ЗАМАЗАТЬ. Проза задания говорит
    «число кадров кратно 4», первоисточник `WanAnimateToVideo` — «step: 4 от 1»
    (`fork_comfy.LENGTH_STEP`, `LENGTH_BASE`), и 77 из штатной геометрии стека
    на 4 не делится. Наивные 5 с x 30 к/с = 150 кадров сэмплер НЕ ВОЗЬМЁТ:
    (150-1) % 4 = 1. Поэтому число подрезается ВНИЗ до ближайшего годного —
    вниз, а не вверх, потому что вверх пришлось бы выдумать кадр, которого в
    драйвинге нет, а интерполяции у нас нет по решению владельца.

    Возвращает и подрезанное число, и потерю в кадрах: молчаливая подрезка
    длины — это как раз то, что оператор потом ищет в промпте.
    """
    fps = FPS_OUT if fps is None else fps
    want = int(round(seconds * fps))
    rest = (want - fork_comfy.LENGTH_BASE) % fork_comfy.LENGTH_STEP
    frames = want - rest
    return {"frames": frames, "wanted": want, "dropped": rest,
            "seconds_effective": round(frames / fps, 4),
            "note": (f"{seconds} с x {fps} к/с = {want} кадров; шаг "
                     f"{fork_comfy.LENGTH_STEP} от {fork_comfy.LENGTH_BASE} "
                     f"даёт {frames}" + ("" if not rest else
                     f", подрезано {rest} кадр(ов) вниз — вверх нельзя, "
                     f"выдуманных кадров в драйвинге нет"))}


def _ffprobe_fps(path):
    """Частота исходника через ffprobe. Нет ffprobe — `None`, а не догадка."""
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of",
             "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    raw = (out.stdout or "").strip()
    if not raw or out.returncode != 0:
        return None
    try:
        num, _, den = raw.partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None


def _resolve(root, value):
    p = Path(value)
    return p if p.is_absolute() or root is None else Path(root) / p


def check(desc: dict, *, root=None, prober=None) -> dict:
    """Проверить описание. Возвращает ЧИСЛА и список находок, а не флаг.

    `root` — относительно чего разрешаются пути (обычно каталог самого файла).
    `prober` — чем измеряется частота исходника; подменяется в тестах, потому
    что иначе ось «драйвинг снят не ниже 30» зеленела бы от отсутствия ffprobe.

    Умолчания разрешаются ЗДЕСЬ, а не в сигнатуре: связанное на импорте
    значение мутация константы не достаёт, и такая ось охраняется только на
    словах.
    """
    prober = _ffprobe_fps if prober is None else prober
    found: list[dict] = []
    skipped: list[str] = []
    add = found.append

    # Ретаргетинг первым: он единственный, кто роняет загрузку целиком, и
    # проверять остальное в файле, который всё равно не поедет, значит тратить
    # время оператора на второстепенное (П2).
    try:
        seen = refuse_retarget(desc)
        # Ноль просмотренных полей — НЕ «чисто» (Р2). Пустой словарь прошёл бы
        # запрет тем, что запрещать в нём нечего, и отчёт сказал бы «годно».
        add(_finding("ретаргетинг", PASS if seen else UNMEASURED,
                     f"полей просмотрено {seen}, про ретаргетинг ни одного"
                     if seen else
                     "полей просмотрено 0 — запрет не отработал ни на чём, "
                     "и это не «чисто»"))
    except RetargetForbidden as e:
        add(_finding("ретаргетинг", FAIL, str(e)))

    got = desc.get("schema")
    add(_finding("схема", PASS if got == TEMPLATE_SCHEMA else FAIL,
                 f"схема {got!r}"
                 + ("" if got == TEMPLATE_SCHEMA
                    else f" вместо {TEMPLATE_SCHEMA!r}: файл не тот или устарел")))

    missing = [f for f in REQUIRED_FIELDS if f not in desc]
    add(_finding("обязательные поля", FAIL if missing else PASS,
                 f"нет полей: {missing}. Умолчаний у них нет нарочно"
                 if missing else
                 f"все {len(REQUIRED_FIELDS)} обязательных поля на месте"))

    for axis, key in (("драйвинг", "driving"), ("фотореференс", "photo")):
        if key not in desc:
            skipped.append(f"{axis}: поля нет, уже посчитано выше")
            continue
        p = _resolve(root, desc[key])
        add(_finding(axis, PASS if p.exists() else FAIL,
                     f"{p}" + ("" if p.exists() else " — файла нет")))

    for axis, key in (("промпт", "prompt"), ("негатив", "negative")):
        if key not in desc:
            skipped.append(f"{axis}: поля нет, уже посчитано выше")
            continue
        val = desc[key]
        ok = isinstance(val, str)
        # Пустой негатив — законное решение оператора, пустой промпт — нет.
        if ok and key == "prompt":
            ok = bool(val.strip())
        add(_finding(axis, PASS if ok else FAIL,
                     f"{len(val)} симв." if ok else
                     f"{val!r}: промпт обязан быть непустой строкой"
                     if key == "prompt" else f"{val!r}: ожидалась строка"))

    fps = desc.get("fps")
    if fps is None:
        skipped.append("частота: поля нет, берутся штатные 30")
        fps = FPS_OUT
    else:
        add(_finding("частота", PASS if fps == FPS_OUT else FAIL,
                     f"{fps} к/с" + ("" if fps == FPS_OUT else
                     f" вместо {FPS_OUT}: частота выхода не настройка, "
                     f"пересэмплирование идёт ВНИЗ и интерполяции нет")))

    w, h = desc.get("width"), desc.get("height")
    bad = []
    for name, val in (("width", w), ("height", h)):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            bad.append(f"{name}={val!r} не положительное целое")
        elif val % fork_comfy.SIDE_MULTIPLE:
            bad.append(f"{name}={val} не кратно {fork_comfy.SIDE_MULTIPLE}")
        elif val < fork_comfy.MIN_SIDE:
            bad.append(f"{name}={val} меньше блока маски {fork_comfy.MIN_SIDE}")
    if not bad and h <= w:
        bad.append(f"{w}x{h} не вертикаль: выход вертикальный, а горизонт "
                   f"стоит ровно столько же — перепутать бесплатно и заметно")
    add(_finding("геометрия", FAIL if bad else PASS,
                 "; ".join(bad) if bad else f"{w}x{h}, стороны кратны "
                 f"{fork_comfy.SIDE_MULTIPLE}, вертикаль"))

    sec = desc.get("seconds")
    numeric = isinstance(sec, (int, float)) and not isinstance(sec, bool)
    ok_sec = numeric and SECONDS_MIN <= sec <= SECONDS_MAX
    # КОРОЧЕ ПОЛА — ТОЛЬКО ИСПЫТАТЕЛЬНОМУ ОПИСАНИЮ, и это не смягчение пола.
    # Первый прогон на арендованной карте идёт на материале из репозитория:
    # 96 кадров, то есть 3.2 с. Пол продукта — 5 с, и опустить его вслед за
    # материалом значило бы навсегда потерять способность заметить, что ролик
    # короче заказанного. Поэтому пол остаётся, а короткий прогон обязан
    # НАЗВАТЬ СЕБЯ стендовым: метка `bench` уже есть в формате и уже
    # запрещает такому описанию уехать в прод (`assert_production`).
    #
    # Исход при этом «не смогли», а не «годно»: длина вне продуктовой полосы
    # — это отсутствие продуктового заявления, а не выполненное требование.
    short_bench = (numeric and desc.get("bench") is True
                   and 0 < sec < SECONDS_MIN)
    add(_finding("длина",
                 PASS if ok_sec else UNMEASURED if short_bench else FAIL,
                 f"{sec} с" if ok_sec else
                 f"{sec} с — КОРОЧЕ ПОЛА {SECONDS_MIN} с, допущено только "
                 f"потому, что описание помечено испытательным (bench). "
                 f"Продуктовое заявление про длину этим прогоном НЕ "
                 f"проверяется, и в прод такое описание не пойдёт"
                 if short_bench else
                 f"{sec!r} вне {SECONDS_MIN}-{SECONDS_MAX} с"
                 if numeric else f"{sec!r}: длина не число"))

    if "frames" not in desc:
        skipped.append("кадры: поля нет, считаются по длине через frames_for")
    elif not numeric:
        add(_finding("кадры", UNMEASURED,
                     "длина не число — сверить кадры с длиной не с чем"))
    else:
        fr = desc["frames"]
        step_ok = (isinstance(fr, int) and not isinstance(fr, bool) and fr > 0
                   and (fr - fork_comfy.LENGTH_BASE) % fork_comfy.LENGTH_STEP == 0)
        want = frames_for(sec, fps=fps)
        add(_finding("кадры", PASS if step_ok and fr == want["frames"] else FAIL,
                     f"{fr} кадров" if step_ok and fr == want["frames"] else
                     f"{fr!r}: шаг {fork_comfy.LENGTH_STEP} от "
                     f"{fork_comfy.LENGTH_BASE} нарушен (77 годится, 76 — нет)"
                     if not step_ok else
                     f"{fr} кадров против {want['frames']} по длине: "
                     f"{want['note']}"))

    step = desc.get("step")
    add(_finding("ступень весов", PASS if step in STEP_CHOICES else FAIL,
                 f"{step}" if step in STEP_CHOICES else
                 f"{step!r} не из {list(STEP_CHOICES)}"))

    arm = desc.get("arm")
    if arm in fork_mask.ARMS:
        add(_finding("плечо маски", PASS,
                     f"{arm}, расширение {fork_mask.arm(arm)} px"))
    elif arm == AUTO_ARM:
        add(_finding("плечо маски", UNMEASURED,
                     f"{AUTO_ARM}: плечо выберет роутер по фотографии. До "
                     f"вызова resolve_arm оси плеча нет — это не «годно»"))
    else:
        add(_finding("плечо маски", FAIL,
                     f"{arm!r} не из {list(ARM_CHOICES)}"))

    fr_face = desc.get("face_refine")
    add(_finding("доводка лица",
                 PASS if isinstance(fr_face, bool) else FAIL,
                 f"{'вкл' if fr_face else 'выкл'}"
                 if isinstance(fr_face, bool) else
                 f"{fr_face!r}: ожидалось да/нет без умолчания — при полном "
                 f"росте на 480p лицо выходит 63-80 px, а планка личности "
                 f"требует от 100, и доводка решает, измерима ли личность"))

    if "props" not in desc:
        skipped.append("разметка предметов: не задана, это допустимо")
    else:
        p = _resolve(root, desc["props"])
        try:
            marking = fork_props.load_marking(p)
            add(_finding("разметка предметов", PASS,
                         f"{p}: схема {fork_props.POINTS_SCHEMA}, предметов "
                         f"проверено {marking.get('_objects_checked')}"))
        except (OSError, ValueError) as e:
            add(_finding("разметка предметов", FAIL, f"{p}: {e}"))

    if "protagonist" not in desc:
        skipped.append("рамка протагониста: не задана, сцена одиночная")
    else:
        box = desc["protagonist"]
        why = _box_trouble(box, w, h)
        add(_finding("рамка протагониста", FAIL if why else PASS,
                     why or f"{list(box)} внутри кадра {w}x{h}"))
        if not why:
            add(_finding("тождество протагониста", UNMEASURED,
                         "тот ли это человек и держится ли он в рамке весь "
                         "клип — рамкой не проверяется. Вендорский детектор "
                         "берёт ПЕРВУЮ детекцию на каждом кадре и при двух "
                         "людях перепрыгивает между ними посреди клипа"))

    bench = desc.get("bench")
    if "bench" not in desc:
        skipped.append("метка стенда: поля нет, описание боевое")
    else:
        add(_finding("метка стенда", PASS if isinstance(bench, bool) else FAIL,
                     ("ИСПЫТАТЕЛЬНОЕ ОПИСАНИЕ: ассеты синтетические, в прод "
                      "не отдавать" if bench else "боевое описание")
                     if isinstance(bench, bool) else
                     f"{bench!r}: ожидалось да/нет"))

    drv = desc.get("driving")
    if drv is None:
        skipped.append("частота драйвинга: драйвинга нет, мерить нечего")
    else:
        drv_path = _resolve(root, drv)
        if drv_path.is_dir():
            # КАТАЛОГ КАДРОВ — ЭТО НЕ «ffprobe не смог». У набора PNG частоты
            # НЕТ ВООБЩЕ: она задана тем, с какой снимали, и в файлах не
            # записана. Прежний текст говорил «нет ffprobe или файл не
            # читается» и посылал оператора чинить то, что не сломано.
            add(_finding("частота драйвинга", UNMEASURED,
                         f"{drv} — КАТАЛОГ КАДРОВ, частоты в нём нет вовсе: "
                         f"её не с чего снять, а не «прибор не справился». "
                         f"Требование к съёмке (не ниже {FPS_OUT} к/с) "
                         f"остаётся НЕПРОВЕРЕННЫМ и держится на слове того, "
                         f"кто раскладывал кадры. Подайте видеофайл, и "
                         f"частота будет снята и сверена"))
            measured = None
        elif (measured := prober(drv_path)) is None:
            add(_finding("частота драйвинга", UNMEASURED,
                         "частоту исходника снять не удалось (нет ffprobe или "
                         "файл не читается). Это НЕ «годно»: требование к "
                         f"съёмке — не ниже {FPS_OUT} к/с, и оно осталось "
                         f"непроверенным"))
        else:
            add(_finding("частота драйвинга",
                         PASS if measured >= FPS_OUT else FAIL,
                         f"исходник {measured:g} к/с"
                         + ("" if measured >= FPS_OUT else
                            f" ниже {FPS_OUT}: пересэмплирование идёт только "
                            f"ВНИЗ, недостающие кадры выдумывать нечем")))

    return _report(found, skipped)


def _box_trouble(box, w, h):
    """Почему рамка негодная, или `None`. Отдельно от `check` ради Т5."""
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(not isinstance(v, int) or isinstance(v, bool) for v in box)):
        return f"{box!r}: ожидалась четвёрка целых xyxy"
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return f"{list(box)}: вырожденная рамка, порядок xyxy нарушен"
    if not (isinstance(w, int) and isinstance(h, int)):
        return None
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return f"{list(box)} выходит за кадр {w}x{h}"
    return None


def _report(found: list, skipped: list) -> dict:
    checked = len(found)
    violations = sum(1 for f in found if f["outcome"] == FAIL)
    unmeasured = sum(1 for f in found if f["outcome"] == UNMEASURED)
    if not checked:
        outcome = UNMEASURED
    elif violations:
        outcome = FAIL
    elif unmeasured:
        outcome = UNMEASURED
    else:
        outcome = PASS
    return {
        "outcome": outcome, "checked": checked, "violations": violations,
        "unmeasured": unmeasured, "passed": checked - violations - unmeasured,
        "findings": found, "skipped": skipped,
        "note": (f"проверено {checked}, нарушений {violations}, не смогли "
                 f"{unmeasured}, пропущено за отсутствием предмета проверки "
                 f"{len(skipped)}"
                 + ("" if checked else
                    ". ПРОВЕРОК НЕ ОТРАБОТАЛО НИ ОДНОЙ — ноль нарушений здесь "
                    "не успех, а отсутствие измерения")),
    }


def load(path, *, strict=None, prober=None) -> dict:
    """Прочитать файл-описание, проверить и вернуть его с отчётом.

    `strict` разрешается в теле, а не в сигнатуре: связанное на импорте
    умолчание мутация не достаёт. При `strict` нарушения роняют загрузку, а
    «не смогли проверить» — НЕ роняют: третий исход не сворачивается во второй
    (Р1), иначе отсутствие ffprobe запрещало бы работу.
    """
    strict = True if strict is None else strict
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"нет файла-описания {p}")
    desc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(desc, dict):
        raise ValueError(f"{p}: описание — это объект, пришло {type(desc).__name__}")
    # Ретаргетинг роняет ДО всего остального и независимо от `strict`: это
    # запрет, а не строгость проверки.
    refuse_retarget(desc, path=p.name)

    rep = check(desc, root=p.parent, prober=prober)
    if strict and rep["violations"]:
        bad = [f"{f['axis']}: {f['note']}" for f in rep["findings"]
               if f["outcome"] == FAIL]
        raise ValueError(f"{p}: {rep['note']}\n  - " + "\n  - ".join(bad))
    out = dict(desc)
    out["_path"] = str(p)
    out["_root"] = str(p.parent)
    out["_check"] = rep
    if "fps" not in out:
        out["fps"] = FPS_OUT
    if "frames" not in out and isinstance(out.get("seconds"), (int, float)):
        out["_frames"] = frames_for(out["seconds"], fps=out["fps"])
        out["frames"] = out["_frames"]["frames"]
    return out


def assert_production(desc: dict) -> dict:
    """Уронить прогон, если испытательное описание пытаются пустить в прод.

    Стенд и боевой шаблон — ОДИН формат, и это выгода: слой оператора и стенд
    оказались одной работой. Цена выгоды здесь: описание с синтетическими
    ассетами внешне неотличимо от боевого, а произведённый из заглушек клип
    читается как неудачная генерация — то есть разбор пойдёт в модель вместо
    входа. Поэтому метка проверяется кодом, а не глазами.
    """
    if desc.get("bench"):
        raise ValueError(
            f"описание {desc.get('name')!r} помечено испытательным (bench): "
            f"его ассеты синтетические, в прод оно не идёт. Клип из заглушек "
            f"выглядит как неудачная генерация, и разбор уйдёт в модель "
            f"вместо входа")
    return {"outcome": PASS, "checked": 1, "violations": 0, "unmeasured": 0,
            "note": "описание боевое: проверено 1, нарушений 0, не смогли 0"}


def resolve_arm(desc: dict, *, router=None, root=None) -> dict:
    """Разрешить плечо маски. `auto` отдаётся роутеру, и он вправе не решить.

    ОТОБРАЖЕНИЕ КОРЗИНЫ РОУТЕРА В ПЛЕЧО ВЫБРАНО (кем: этот поток; из чего:
    роутер отвечает, нужно ли ОТКЛОНЕНИЕ от фигуры драйвинга). Корзина
    «LoRA не нужна» означает, что скелет с маской дадут нужную фигуру сами —
    расширять маску незачем, `narrow`. Корзина «LoRA полная 40+» означает, что
    фигуру надо увести за силуэт драйвинга, а за границей маски модель не
    рисует — значит нужен запас, `wider`. Это НЕ ИЗМЕРЕНО: парного прогона двух
    плеч на одной фотографии не было.

    «Не уверен» роутера остаётся «не смогли» и здесь. Свернуть его в безопасное
    плечо было бы удобно и неверно: оператор увидел бы выбранное плечо и решил,
    что оно измерено.
    """
    arm = desc.get("arm")
    if arm in fork_mask.ARMS:
        return {"outcome": PASS, "arm": arm, "grow_px": fork_mask.arm(arm),
                "checked": 1, "violations": 0, "unmeasured": 0,
                "note": f"плечо {arm} названо в описании, роутер не спрашивался"}
    if arm != AUTO_ARM:
        raise KeyError(f"плечо {arm!r} не из {list(ARM_CHOICES)}")

    router = fork_build_route.route if router is None else router
    photo = _resolve(root if root is not None else desc.get("_root"),
                     desc["photo"])
    verdict = router(photo)
    bucket = verdict.get("bucket")
    if bucket == fork_build_route.BUCKET_NO_LORA:
        chosen = "narrow"
    elif bucket == fork_build_route.BUCKET_LORA_FULL_W40:
        chosen = "wider"
    else:
        return {"outcome": UNMEASURED, "arm": None, "grow_px": None,
                "checked": 1, "violations": 0, "unmeasured": 1,
                "bucket": bucket, "router": verdict,
                "note": (f"роутер ответил {bucket!r} — плечо не выбрано. "
                         f"Выбирается руками: {verdict.get('note')}")}
    return {"outcome": PASS, "arm": chosen, "grow_px": fork_mask.arm(chosen),
            "checked": 1, "violations": 0, "unmeasured": 0, "bucket": bucket,
            "router": verdict,
            "note": (f"роутер: {bucket} -> плечо {chosen} "
                     f"({fork_mask.arm(chosen)} px). Отображение ВЫБРАНО, "
                     f"не измерено")}


def make_bench(out_dir, *, name=None) -> Path:
    """Написать испытательное описание с синтетическими ассетами.

    Стенд — тот же файл, и делается он тем же кодом, что его читает. Иначе у
    стенда завёлся бы свой формат, и проверять он стал бы себя.
    """
    name = "стенд" if name is None else name
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    drv, photo = out / "driving.mp4", out / "photo.png"
    # Заглушки, а не съёмка: проверка существования файла — единственное, что
    # они обязаны пройти, и метка `bench` говорит об этом вслух.
    drv.write_bytes(b"SYNTHETIC-NOT-A-VIDEO")
    photo.write_bytes(b"SYNTHETIC-NOT-A-PHOTO")
    desc = {
        "schema": TEMPLATE_SCHEMA, "name": name,
        "driving": drv.name, "photo": photo.name,
        "prompt": "синтетический промпт стенда",
        "negative": "",
        "fps": FPS_OUT, "width": WIDTH_OUT, "height": HEIGHT_OUT,
        "seconds": SECONDS_MIN, "step": STEP_CHOICES[-1], "arm": "narrow",
        "face_refine": True, "bench": True,
    }
    desc["frames"] = frames_for(desc["seconds"], fps=FPS_OUT)["frames"]
    path = out / "template.json"
    path.write_text(json.dumps(desc, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def report_text(path, *, production=None, prober=None) -> dict:
    """Отчёт по файлу словами и числами. Отдельно от `main` ради Т5.

    Развилка внутри `main()` недостижима для теста и потому деградирует молча;
    здесь она вызывается тестом напрямую.
    """
    production = False if production is None else production
    lines: list[str] = []
    try:
        desc = load(path, strict=False, prober=prober)
    except (OSError, ValueError) as e:
        return {"outcome": FAIL, "code": 1, "checked": 0, "violations": 1,
                "unmeasured": 0, "text": f"{type(e).__name__}: {e}"}
    rep = desc["_check"]
    lines.append(f"описание: {desc.get('name')!r} ({path})")
    for f in rep["findings"]:
        lines.append(f"  [{f['outcome']}] {f['axis']}: {f['note']}")
    for s in rep["skipped"]:
        lines.append(f"  [пропущено] {s}")
    lines.append(rep["note"])
    outcome, code = rep["outcome"], {PASS: 0, FAIL: 1, UNMEASURED: 2}[rep["outcome"]]
    if production:
        try:
            assert_production(desc)
            lines.append("прод: описание боевое")
        except ValueError as e:
            outcome, code = FAIL, 1
            lines.append(f"прод: {e}")
    return {"outcome": outcome, "code": code, "checked": rep["checked"],
            "violations": rep["violations"], "unmeasured": rep["unmeasured"],
            "text": "\n".join(lines)}


def main(argv=None) -> int:
    """Точка входа оператора. Код возврата различает ТРИ исхода, а не два.

    Ноль — годно, единица — не годно, двойка — не смогли проверить (Р2). Свести
    двойку в ноль означало бы, что отсутствие ffprobe читается как успех.
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="python3 -m ball_reel.fork_template",
        description="проверить файл-описание шаблона")
    ap.add_argument("path", help="путь к файлу-описанию")
    ap.add_argument("--production", action="store_true",
                    help="дополнительно требовать, что описание НЕ испытательное")
    args = ap.parse_args(argv)
    rep = report_text(args.path, production=args.production)
    print(rep["text"])
    return rep["code"]


if __name__ == "__main__":
    raise SystemExit(main())
