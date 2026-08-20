"""Отбор петель в драйвинге: где движение повторяется и как это показать глазам.

ЗАЧЕМ. Драйвинг — это полный ролик тренировки: позы и упражнения по ходу
меняются. Модель, ведомая позой, берёт кадры драйвинга ОДИН К ОДНОМУ, поэтому
петля на входе даёт петлю на выходе, а «взять всё подряд» даёт ролик с рывком
на повторе. Промерено на `demo/bench/driving` (хэндоф, раздел «ОТБОРА КАДРОВ
ДЛЯ ПРОДАКШЕНА НЕТ»): весь клип как петля — 0.4473, лучшая петля 0..44 — 0.1064,
худшая пара — 0.4896. То есть «всё подряд» почти ХУДШИЙ из возможных стыков.

ЧЕГО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ, И ЭТО ГЛАВНОЕ. Он НЕ говорит «петля бесшовна».
Планки бесшовности для расстояния поз у проекта нет, и выдумать её нельзя:
`motion.SEAMLESS_MAX = 0.30` — отношение ПИКСЕЛЬНОЕ и для другого замера.
Прибор РАНЖИРУЕТ кандидатов и кладёт рядом GIF, чтобы решение принял человек
глазами. Все числа ниже — относительные, посчитанные ПО САМОМУ КЛИПУ: стык
меряется в единицах обычного межкадрового шага этого же клипа, а отбор идёт по
преимуществу над типичной парой этого же клипа. Ни одно абсолютное число из
чужого замера сюда не заимствовано.

ТРИ ИСХОДА (Р1), и они не сворачиваются друг в друга:
    годно              кандидаты найдены и отранжированы; решает оператор
    не годно           поза снята, петель в материале нет (монотонный дрейф)
    не смогли          позу снять нечем/не с чего, или клип не движется

ДВЕ ВЕЛИЧИНЫ НА СТЫКЕ, И ОДНОЙ МАЛО. Совпадение поз необходимо, но недостаточно:
человек в кадре i приседает, а в кадре j встаёт — поза та же, направление
противоположное, склейка даёт отскок, видимый глазом. Поэтому сверяется ещё и
производная позы. Как две величины сводятся в одну оценку — см. `SEAM_SCORE`.

ОПЕРАТОРА В КОНТУРЕ НЕТ (решение владельца, финальный продукт). Пользователь
кидает ссылку на своё видео и получает пять GIF-петель на согласование ещё до
оплаты. Отсюда три следствия, которых не было бы у внутреннего прибора:

    1. Верхние пять видит ПЛАТЯЩИЙ ЧЕЛОВЕК, а не оператор. Прополки не будет.
    2. Видео пользователя — не наш снятый драйвинг: в нём бывают МОНТАЖНЫЕ
       РЕЗЫ. Петля, перешагнувшая через рез, по позам выглядит отлично (до реза
       человек стоял так же, как после) и является мусором. Резы ловятся
       отдельно, по пикселям, и кандидат, накрывающий рез, не выпускается.
    3. Видео бывает минутами. Снятие поз стоит ~0.03-0.06 с на кадр, то есть
       десять минут материала — это 17 минут CPU. Поэтому поиск идёт КОАРС-ТУ-
       ФАЙН: кандидаты по прорежённой последовательности, уточнение на полной
       частоте только вокруг найденного.

ЦЕНА, ИЗМЕРЕННАЯ ЗДЕСЬ (И4: ИЗМЕРЕНО, команда — `read_all` и `cuts` на
материале репозитория, CPU этой машины):

    снятие позы   0.0328 с/кадр на 240x426 (demo/bench/driving, 96 кадров)
                  0.0537 с/кадр на 1280x720 (chain_frames, 96 кадров)
                  0.058  с/кадр — замер владельца, сошлось с крупным кадром
    пиксели (резы) 0.0012 с/кадр — в 27 раз дешевле позы, потому и первыми (П2)
    матрица        0.0296 с на 420 пар
    кэш поз        второй проход в 402 раза дешевле (2.37 с -> 0.0059 с),
                   88 КБ на 96 кадров

Отсюда полная цена по длине материала при 30 к/с (РАСЧЁТ по замеру выше):

    длина   кадров   позы как есть   позы 1 из 5   резы
    10 с       300      10 с             2 с        0.4 с
     1 мин    1800      59 с            12 с        2.2 с
    10 мин   18000      9.8 мин          2 мин       22 с
    20 мин   36000      20 мин           3.9 мин     43 с   <- потолок MAX_FRAMES

ЗАВИСИМОСТИ. Снятие поз — через `pose.landmarks` (mediapipe, локально), и
только через ЯВНУЮ ТОЧКУ ВНЕДРЕНИЯ `read_pose`: вся арифметика проверяется на
синтетических скелетах без модели (Т4). Пиксели для поиска резов — через
`read_gray`, вторую точку внедрения. GIF — через Pillow.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from . import fork_comfy, motion, pose
from .fork_identity import FAIL, PASS, UNMEASURED

# ---------------------------------------------------------------------------
# КОНСТАНТЫ-РЕШЕНИЯ. У каждой помечено происхождение (И4).
# ---------------------------------------------------------------------------

#: ВЫБРАНО (кем: эта смена; из чего: из замера в хэндофе, где кандидаты брались
#: не короче 41 кадра). 41 кадр при 30 к/с — 1.37 с. Короче брать нельзя не по
#: вкусу, а по назначению: петлю повторяют до 5-10 с, и петля в полсекунды даёт
#: десять повторов подряд — это читается как заедание, а не как движение.
#: Ровно 41, а не 40: длина обязана быть вида 4k+1 (см. `admissible_lengths`).
LOOP_MIN_FRAMES = 41

#: ВЫБРАНО: во сколько раз лучшая пара обязана быть лучше ТИПИЧНОЙ пары того же
#: клипа, чтобы называться петлёй. Это НЕ планка бесшовности — это отсев
#: материала, в котором повтора нет вовсе. Число зажато между двумя замерами:
#:   4.2  — во столько лучшая петля драйвинга лучше клипа целиком (хэндоф:
#:          0.4473 против 0.1064), то есть настоящий повтор проходит с запасом;
#:   1.40 — столько даёт монотонный дрейф, где повтора нет вовсе и «лучшая»
#:          пара выигрывает только тем, что она короче (ИЗМЕРЕНО на фикстуре
#:          `drift_sequence`, см. `test_a_drift_has_no_loop_and_the_margin_is_1_40`;
#:          число не зависит от скорости дрейфа — проверено на трёх).
#: 2.0 лежит между ними, ближе к дрейфу: пропустить сомнительное и показать
#: оператору дешевле, чем молча съесть настоящую петлю.
ADVANTAGE_MIN = 2.0

#: ВЫБРАНО: вес рассогласования ПОТОКА против рассогласования ПОЗЫ. Обе величины
#: перед этим уже приведены к одной единице — «обычный межкадровый шаг этого
#: клипа», — поэтому 1.0 означает буквально: один шаг расхождения по направлению
#: движения так же плох, как один шаг расхождения по позе. Молчаливого сложения
#: разнородных величин здесь нет: см. `SEAM_SCORE`.
FLOW_WEIGHT = 1.0

#: Как две приведённые величины сводятся в одну оценку. МАКСИМУМ, а не сумма и
#: не среднее: это два НЕЗАВИСИМЫХ дефекта стыка, и хороший показатель по одной
#: оси не выкупает провал по другой. Сумма позволила бы отскоку (поза сошлась,
#: направление противоположно) обменять свой провал на идеальную позу и
#: подняться в рейтинге — то есть ровно на тот дефект, ради которого вторая ось
#: и заведена. Максимум такого обмена не допускает.
SEAM_SCORE = "max(поза, вес*поток), обе в единицах обычного шага клипа"

#: ВЫБРАНО: какую долю более короткой из двух петель разрешено делить с уже
#: принятой. Без подавления сортировка по качеству даёт десяток вариантов
#: ОДНОГО движения, сдвинутых на кадр, — это не выбор. 0.5 читается как «петли
#: считаются разными, если больше половины короткой из них не общая».
OVERLAP_MAX = 0.5

#: ВЫБРАНО: сколько петель показывать оператору. Пять — это уже выбор и ещё не
#: свалка; больше пяти GIF-ов подряд человек не сравнивает.
TOP_LOOPS = 5

#: ВЫБРАНО: доля кадров, на которых поза обязана сняться, чтобы разбор вообще
#: имел смысл. Ниже — исход «не смогли», а не «петель нет»: на драйвинге, где
#: человека не видно в четверти кадров, отсутствие петель ничего не значит.
MIN_POSE_COVERAGE = 0.8

#: ВЫБРАНО: сколько кадров кладётся в GIF. GIF здесь не иллюстрация, а прибор:
#: он зациклен по своей природе, поэтому оператор видит ИМЕННО СТЫК. 24 кадра
#: хватает, чтобы движение читалось, и держит файл в сотнях килобайт.
GIF_MAX_FRAMES = 24

#: ВЫБРАНО: наибольшая сторона кадра в GIF. Драйвинг у нас бывает и 1280x720 —
#: два десятка таких кадров дают файл в мегабайты, который оператор ждёт.
GIF_MAX_SIDE = 320

#: Планка монтажного реза: во сколько раз попиксельный скачок между соседними
#: кадрами должен превысить типичный скачок этого же клипа. ЗАИМСТВОВАНО из
#: `motion.JUMP_MAX` — значением, а не копией числа (Е1). Заимствование законно
#: ровно потому, что величина ОДНОРОДНА: там пиксельное отношение к пиксельному
#: замеру («скачок, а не движение»), и здесь пиксельное отношение к пиксельному
#: замеру. Так же заимствовать `motion.SEAMLESS_MAX` для расстояния ПОЗ было бы
#: нельзя, и именно поэтому планки бесшовности у этого модуля нет.
#: Негативный контроль обеих сторон — `test_a_cut_is_found`/`..._is_not_invented`.
CUT_JUMP = motion.JUMP_MAX

#: Сторона уменьшенного кадра для поиска резов. ЗАИМСТВОВАНО: умолчание
#: `motion._gray`. Меньше — дешевле, но рез начинает путаться с движением руки.
CUT_SIDE = 96

#: ВЫБРАНО: столько подряд кадров без человека считаются УХОДОМ ИЗ КАДРА, а не
#: пропущенной детекцией. 15 кадров при 30 к/с — полсекунды: одиночный промах
#: детектора столько не длится, а человек, вышедший из кадра, длится дольше.
#: Разница между двумя этими случаями попадает в текст исхода, а не только в
#: числа: «детектор моргнул» и «человека нет в кадре» чинятся по-разному.
PRESENCE_GAP_MIN = 15

#: ВЫБРАНО: с какой длины материал разбирается по ПРОРЕЖЁННОЙ последовательности.
#: 900 кадров — 30 с при 30 к/с, то есть примерно полминуты CPU на снятие поз
#: (0.031 с/кадр ИЗМЕРЕНО здесь, 0.058 с/кадр ИЗМЕРЕНО владельцем). Ждать
#: полминуты человек согласен, семнадцать минут — нет.
COARSE_ABOVE_FRAMES = 900

#: ВЫБРАНО: шаг прореживания. 1 из 5 — это в пять раз дешевле: десять минут
#: материала превращаются из 17.4 в 3.5 минуты CPU.
#:
#: ЧЕМ ПЛАТИМ, ЧЕСТНО. Прореживание — это дискретизация, и у неё есть Найквист:
#: движение с периодом короче 2*STRIDE кадров прорежённая последовательность
#: НЕ ВИДИТ, а видит вместо него ложное медленное (алиасинг). При шаге 5 это
#: период короче 10 кадров, то есть быстрее трёх циклов в секунду при 30 к/с —
#: тряска, дрожь, быстрые махи. Обычные упражнения на мяче (1-3 с на цикл,
#: 30-90 кадров) от этого далеко. ИЗМЕРЕНО на фикстуре: петля периода 44
#: находится при шаге 5, 10, 20 и теряется при 22 = период/2 —
#: `test_thinning_breaks_exactly_at_nyquist`.
COARSE_STRIDE = 5

#: ВЫБРАНО: потолок длины материала. 36000 кадров — 20 минут при 30 к/с; на
#: прорежённой последовательности это 7200 снятий позы, 3.7-7 минут CPU.
#: Выше — исход «не смогли, слишком длинное», а не молчаливое ожидание.
MAX_FRAMES = 36000

#: Три слова о том, ЧЕМ считали. Едут в отчёт: «петель не нашлось на полной
#: частоте» и «не нашлось на прорежённой» — разные утверждения.
SCAN_FULL = "полная частота"
SCAN_COARSE = "прорежённая"
SCAN_TOO_LONG = "слишком длинное"

#: Коды возврата точки входа. Те же, что у `fork_video` и `fork_stand` (Е1):
#: 0 годно, 1 не годно, 2 не смогли.
EXIT_BY_OUTCOME = {PASS: 0, FAIL: 1, UNMEASURED: 2}

#: Расширения кадров, которые считаем кадрами каталога.
FRAME_SUFFIXES = (".png", ".jpg", ".jpeg")

#: Версия формата кэша поз. Меняется вместе с формой записи: старый кэш обязан
#: быть отвергнут, а не прочитан как новый.
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# ТОЧКА ВНЕДРЕНИЯ: единственное место, где модуль трогает детектор поз (Т4).
# ---------------------------------------------------------------------------

def read_pose(path) -> dict:
    """Снять позу с одного кадра. ТОЧКА ВНЕДРЕНИЯ: тест подменяет её целиком.

    Возвращает `{"points": dict|None, "why": str}`, и различие между двумя
    видами `None` здесь принципиально, как у `fork_stand.read_smi`:

        points is None, why == ""    тело в кадре НЕ НАЙДЕНО — это измерение;
        points is None, why != ""    СПРОСИТЬ НЕЧЕМ (нет mediapipe, нет весов,
                                     не читается файл) — это НЕ измерение.

    Третье поле, `people` — сколько человек нашлось в кадре, или None, если
    столько не спрашивали. Один человек и «не считали» — разные ответы (Р1).

    Свернув второе в первое, мы получили бы «петель не нашлось» на клипе, где
    детектор просто не запускался.
    """
    try:
        # `people` умолчание вернуть НЕ МОЖЕТ (см. DEBT в `presence`): общий
        # детектор настроен на одну позу. None здесь означает «не спрашивали»,
        # и это честнее единицы, которую нечем подтвердить.
        return {"points": pose.landmarks(path), "why": "", "people": None}
    except Exception as exc:  # noqa: BLE001 — намеренно широко: причин «спросить
        # нечем» много (нет пакета, нет весов, битый файл), и все они означают
        # ровно одно — исход «не смогли», а не «тела нет».
        return {"points": None, "why": f"{type(exc).__name__}: {str(exc)[:200]}"}


def read_gray(path):
    """Уменьшенный серый кадр. ВТОРАЯ ТОЧКА ВНЕДРЕНИЯ (Т4).

    Берётся `motion._gray` (Е1): уменьшение до квадрата 96 и перевод в серое
    уже промерены на этом проекте, и вторая такая функция разошлась бы с
    первой молча.
    """
    return motion._gray(path, CUT_SIDE)


# ---------------------------------------------------------------------------
# Монтажные резы. САМАЯ ОПАСНАЯ ИЗ ОСЕЙ: дефект выглядит как удача
# ---------------------------------------------------------------------------

def cuts(paths, *, gray=None, jump=None) -> dict:
    """Где в клипе монтажный рез. Дёшево, по пикселям, ДО снятия поз (П2).

    ЗАЧЕМ. Петля, перешагнувшая через рез, по позам может выглядеть отлично:
    до реза человек стоял так же, как после. Склеенная, она даёт мгновенную
    смену плана — то есть ровно то, чего не должно быть в петле. Ни одна из
    двух позных осей этого не видит: обе меряют ТЕЛО, а рез меняет КАДР.

    Мера относительная, как и всё в этом модуле: скачок сравнивается с типичным
    скачком этого же клипа, а не с абсолютным числом яркости.

    Возвращает `{"cuts": [k, ...], ...}`, где k — номер кадра, ПОСЛЕ которого
    стоит рез (то есть шов между k и k+1).

    ТРИ ИСХОДА И ЗДЕСЬ. Клип, в котором пиксели вообще не меняются, не даёт
    типичного скачка — тогда резы НЕ ИСКАЛИ, и это не то же, что «резов нет».
    """
    import numpy as np

    gray = read_gray if gray is None else gray
    jump = CUT_JUMP if jump is None else jump
    t = time.perf_counter()
    steps, prev = [], None
    for p in paths:
        # Потоком, по одному кадру: `motion._steps` держит в памяти ВЕСЬ клип
        # сразу, а 18000 кадров 96x96 float64 — это 1.3 ГБ.
        cur = gray(str(p))
        if prev is not None:
            steps.append(float(np.abs(cur - prev).mean()))
        prev = cur
    elapsed = round(time.perf_counter() - t, 4)
    if not steps:
        return {"outcome": UNMEASURED, "cuts": [], "steps": 0, "median": None,
                "worst": None, "elapsed": elapsed,
                "note": "кадров меньше двух: резы искать не в чем"}
    med = float(np.median(steps))
    if med <= 0:
        return {"outcome": UNMEASURED, "cuts": [], "steps": len(steps),
                "median": 0.0, "worst": round(max(steps), 4), "elapsed": elapsed,
                "note": ("типичный межкадровый скачок равен нулю: сравнивать не "
                         "с чем, резы НЕ ИСКАЛИ. Это не «резов нет»")}
    found = [k for k, v in enumerate(steps) if v / med > jump]
    worst = max(steps) / med
    return {"outcome": PASS, "cuts": found, "steps": len(steps),
            "median": round(med, 4), "worst": round(worst, 2),
            "elapsed": elapsed,
            "note": (f"резов найдено {len(found)} по {len(steps)} переходам "
                     f"(планка {jump}x типичного скачка, самый резкий переход "
                     f"{worst:.2f}x)"
                     + (f"; кадры-швы: {found[:10]}" if found else ""))}


# ---------------------------------------------------------------------------
# Человек в кадре: сколько их и не ушёл ли
# ---------------------------------------------------------------------------

def presence(poses, *, people=None, index=None, gap_min=None) -> dict:
    """Есть ли человек, один ли он, и не выходил ли он из кадра.

    Три РАЗНЫХ случая, и сворачивать их в один нельзя: «человека нет вовсе»,
    «людей несколько» и «человек ушёл посреди клипа» чинятся по-разному, а
    сведённые в «поза снялась плохо» неотличимы.

    СКОЛЬКО ЛЮДЕЙ — не наш вопрос по существу: выбор протагониста при
    нескольких людях уже решён в `fork_props` (роль `PROTAGONIST`, разметка
    оператором), и второго способа здесь не заводится (Е1). Наше дело —
    заметить и сказать. Число людей приходит ОТДЕЛЬНЫМ полем `people` от точки
    внедрения, а не внутри позы: поза — это словарь суставов, и посторонний ключ
    в нём поехал бы через всю арифметику приведения.

    DEBT(2026-08-20): умолчание `read_pose` это поле НЕ ЗАПОЛНЯЕТ и заполнить
    не может: общий детектор `pose._pose_model` создан с одной позой на кадр
    (`num_poses` по умолчанию 1), то есть больше одного человека он вернуть
    не в состоянии. Исход «людей несколько» достижим только через подменённый
    счётчик (и тестами он покрыт). Чтобы он заработал вживую, нужна одна строка
    в `pose.py` — чужом файле.
    """
    gap_min = PRESENCE_GAP_MIN if gap_min is None else gap_min
    index = list(range(len(poses))) if index is None else list(index)
    seen = [i for i, p in enumerate(poses) if p is not None]
    crowd = sorted({c for c in (people or []) if c is not None and c > 1})
    gaps, run = [], []
    for i, p in enumerate(poses):
        if p is None:
            run.append(i)
        elif run:
            gaps.append((run[0], run[-1]))
            run = []
    if run:
        gaps.append((run[0], run[-1]))
    long_gaps = [(a, b) for a, b in gaps if b - a + 1 >= gap_min]
    return {"frames": len(poses), "seen": len(seen), "crowd": crowd,
            "gaps": gaps, "long_gaps": long_gaps,
            "left_at": (index[long_gaps[0][0]] if long_gaps else None),
            "missing": [index[i] for i, p in enumerate(poses) if p is None]}


# ---------------------------------------------------------------------------
# Кадры на входе
# ---------------------------------------------------------------------------

def frame_paths(directory) -> list:
    """Кадры каталога, отсортованные по имени."""
    d = Path(directory)
    return sorted((p for p in d.iterdir()
                   if p.is_file() and p.suffix.lower() in FRAME_SUFFIXES),
                  key=lambda p: p.name)


# ---------------------------------------------------------------------------
# Снятие поз: дорогой шаг, поэтому он считает своё время и умеет кэш
# ---------------------------------------------------------------------------

def _cache_key(path) -> str:
    st = Path(path).stat()
    return f"{Path(path).name}:{st.st_size}:{st.st_mtime_ns}"


def read_all(paths, *, reader=None, cache=None) -> dict:
    """Снять позы со всех кадров. Числа рядом с результатом (Р2).

    `cache` — путь к JSON. Кэш ключуется именем, размером и mtime кадра:
    подменённый кадр обязан пересчитаться, иначе кэш станет вторым источником
    истины о материале (Е1).
    """
    reader = read_pose if reader is None else reader
    paths = [Path(p) for p in paths]
    t = time.perf_counter()

    store = {}
    if cache is not None and Path(cache).exists():
        try:
            raw = json.loads(Path(cache).read_text(encoding="utf-8"))
            if raw.get("version") == CACHE_VERSION:
                store = raw.get("frames") or {}
        except (OSError, ValueError):
            store = {}

    poses, whys, crowd = [], [], []
    hits = 0
    for p in paths:
        key = None
        if cache is not None:
            try:
                key = _cache_key(p)
            except OSError:
                key = None
        if key is not None and key in store:
            hits += 1
            got = store[key]
            poses.append(None if got is None
                         else {n: tuple(v) for n, v in got.items()})
            whys.append("")
            # Сколько людей было в кадре, кэш НЕ ХРАНИТ: это ответ детектора о
            # сцене, а не о позе, и подсунуть его из старого файла значило бы
            # утверждать про кадр то, чего в этот раз не спрашивали.
            crowd.append(None)
            continue
        got = reader(str(p))
        poses.append(got.get("points"))
        whys.append(got.get("why") or "")
        crowd.append(got.get("people"))
        if key is not None and not whys[-1]:
            store[key] = (None if poses[-1] is None
                          else {n: list(v) for n, v in poses[-1].items()})

    if cache is not None:
        try:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            Path(cache).write_text(
                json.dumps({"version": CACHE_VERSION, "frames": store}),
                encoding="utf-8")
        except OSError:
            pass

    taken = sum(1 for p in poses if p is not None)
    broken = [w for w in whys if w]
    return {"poses": poses, "why": whys, "people": crowd,
            "frames": len(paths), "taken": taken,
            "no_body": sum(1 for p, w in zip(poses, whys)
                           if p is None and not w),
            "unreadable": len(broken),
            "first_why": broken[0] if broken else "",
            "cached": hits,
            "elapsed": round(time.perf_counter() - t, 4)}


# ---------------------------------------------------------------------------
# Геометрия: поза, поток, стык
# ---------------------------------------------------------------------------

def states(poses) -> list:
    """Позы, приведённые к торсу и центрированные на бёдрах.

    Приведение берётся у `pose._normalise` (Е1): вторая копия этой арифметики
    разошлась бы с той, по которой судится вся остальная приёмка проекта.
    """
    return [None if p is None else pose._normalise(p) for p in poses]


def pose_gap(a, b):
    """Расхождение двух ПРИВЕДЁННЫХ поз в длинах торса, или None.

    Та же величина, что `pose.pose_delta(...)["mean"]`, и это не заявление, а
    проверяемое равенство: тест `test_pose_gap_is_the_same_number_as_pose_delta`
    сверяет два числа на фикстуре. Своя реализация нужна ровно потому, что
    `pose_delta` принимает СЫРЫЕ точки и приводит их заново на каждый вызов, а
    здесь приведение сделано один раз на кадр и переиспользуется парами: на 96
    кадрах это 1300 пар, на длинном драйвинге — сотни тысяч.

    СВОЕГО ПОЛА В 4 ОБЩИХ СУСТАВА ЗДЕСЬ НЕТ, и это не упущение: `pose._normalise`
    не возвращает позу без обоих бёдер и обоих плеч, то есть после приведения
    общих суставов заведомо не меньше четырёх. Второе такое число рядом с полом
    `pose_delta` было бы копией знания, которую нечем нарушить (Е1).
    """
    import numpy as np

    if a is None or b is None:
        return None
    shared = [n for n in a
              if a[n][1] >= pose.MIN_VISIBILITY and b[n][1] >= pose.MIN_VISIBILITY]
    return round(float(np.mean([float(np.linalg.norm(a[n][0] - b[n][0]))
                                for n in shared])), 6)


def flow_gap(states_list, i, j):
    """Расхождение НАПРАВЛЕНИЙ движения в кадрах i и j, в длинах торса за кадр.

    (поза[i+1] − поза[i]) против (поза[j+1] − поза[j]), по суставам, видимым во
    всех четырёх кадрах. Возвращает None, если хотя бы одна из четырёх поз не
    снята или кадра j+1 не существует.

    ПОЧЕМУ ОДНОЙ ПОЗЫ МАЛО: в приседании кадр на пути вниз и кадр на пути вверх
    неотличимы по позе. Склеенные, они дают отскок — видимый глазом и невидимый
    для первой оси.
    """
    import numpy as np

    n = len(states_list)
    if i + 1 >= n or j + 1 >= n:
        return None
    quad = [states_list[k] for k in (i, i + 1, j, j + 1)]
    if any(s is None for s in quad):
        return None
    shared = [k for k in quad[0]
              if all(s[k][1] >= pose.MIN_VISIBILITY for s in quad)]
    vals = [float(np.linalg.norm((quad[1][k][0] - quad[0][k][0])
                                 - (quad[3][k][0] - quad[2][k][0])))
            for k in shared]
    return round(float(np.mean(vals)), 6)


def typical_step(states_list) -> dict:
    """Обычный межкадровый шаг клипа — единица, в которой меряется стык.

    Медиана, а не среднее: одна выпавшая поза посреди клипа даёт выброс,
    который среднее втащит в масштаб, а медиана — нет.
    """
    import numpy as np

    steps = [g for g in (pose_gap(states_list[k], states_list[k + 1])
                         for k in range(len(states_list) - 1)) if g is not None]
    if not steps:
        return {"step": None, "measured": 0,
                "note": "межкадровый шаг НЕ ИЗМЕРЕН: нет ни одной пары соседних "
                        "кадров с пригодной позой"}
    step = float(np.median(steps))
    return {"step": step, "measured": len(steps),
            "note": (f"обычный шаг клипа {step:.4f} длин торса за кадр, "
                     f"по {len(steps)} парам соседних кадров")}


def length_is_admissible(length, *, fps=None, min_frames=None) -> bool:
    """Пройдёт ли такая длина петли обёртку и продуктовую полосу.

    ОДНО ЗНАНИЕ — ОДНО МЕСТО (Е1): и перечисление длин, и перебор пар спрашивают
    здесь. Разъехавшись, они дали бы кандидата, которого обёртка прижмёт.
    """
    fps = fork_comfy.WRAP_FPS if fps is None else fps
    min_frames = LOOP_MIN_FRAMES if min_frames is None else min_frames
    return (min_frames <= length <= int(fork_comfy.SECONDS_MAX * fps)
            and (length - fork_comfy.LENGTH_BASE) % fork_comfy.LENGTH_STEP == 0)


def admissible_lengths(n_frames, *, fps=None, min_frames=None) -> list:
    """Длины петли, которые обёртка НЕ прижмёт и продукт примет.

    Кратность 4k+1 — не вкус: `fork_comfy.snap_frames` прижимает длину к этому
    шагу молча, и петля, прижатая на кадр, разъезжается на стыке. Потолок —
    из `SECONDS_MAX` того же модуля (Е1), а не своё число.
    """
    return [L for L in range(1, n_frames + 1)
            if length_is_admissible(L, fps=fps, min_frames=min_frames)]


def admissible_pairs(index, *, fps=None, min_frames=None) -> list:
    """Пары ПОЗИЦИЙ, чья длина в ИСХОДНЫХ кадрах допустима.

    `index` — какие исходные кадры реально сняты. При прореживании соседние
    позиции отстоят на шаг, поэтому длина считается по исходным номерам, а не
    по расстоянию в списке: прижатие обёртки к 4k+1 живёт в исходных кадрах.
    """
    fps = fork_comfy.WRAP_FPS if fps is None else fps
    top = int(fork_comfy.SECONDS_MAX * fps)
    out = []
    for a in range(len(index)):
        for b in range(a + 1, len(index)):
            L = index[b] - index[a] + 1
            if L > top:
                break
            if length_is_admissible(L, fps=fps, min_frames=min_frames):
                out.append((a, b))
    return out


def similarity(states_list, *, fps=None, min_frames=None, index=None,
               blocked=None) -> dict:
    """Матрица самоподобия поз — в разреженном виде, по допустимым парам.

    ПОЧЕМУ РАЗРЕЖЕННО. Плотная матрица n×n на десятиминутном драйвинге (18000
    кадров при 30 к/с) — это 2.6 ГБ float64 ради тех диагоналей, которые
    единственно и нужны: длина петли обязана лежать в допустимом диапазоне и
    быть кратна 4k+1. Таких пар не n², а примерно n·(диапазон/4): на 96 кадрах
    420 вместо 9216 — то же множество, на котором считаны числа хэндофа; на
    18000 кадров 1.2 млн вместо 324 млн, то есть секунды, а не часы.

    `blocked(i, j)` — почему пару брать нельзя (рез внутри, дыра присутствия)
    или None. Отвергнутые считаются ПО ПРИЧИНАМ: «кандидатов нет» и «все
    кандидаты перешагивали рез» — разные ответы.

    Ключи словарей — ИСХОДНЫЕ номера кадров, а не позиции в списке.
    """
    index = list(range(len(states_list))) if index is None else list(index)
    pairs = admissible_pairs(index, fps=fps, min_frames=min_frames)
    pose_m, flow_m = {}, {}
    unmeasurable = 0
    rejected: dict = {}
    for a, b in pairs:
        i, j = index[a], index[b]
        why = blocked(i, j) if blocked is not None else None
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        pg = pose_gap(states_list[a], states_list[b])
        fg = flow_gap(states_list, a, b)
        pose_m[(i, j)] = pg
        flow_m[(i, j)] = fg
        if pg is None or fg is None:
            unmeasurable += 1
    return {"pose": pose_m, "flow": flow_m, "index": index,
            "pairs": len(pairs), "rejected": rejected,
            "kept": len(pose_m), "unmeasurable": unmeasurable,
            "measured": len(pose_m) - unmeasurable}


def score_pairs(sim, step, *, flow_weight=None) -> list:
    """Оценка стыка по каждой паре, в единицах обычного шага клипа.

    Две величины приводятся к одной единице ДО сведения, и сводятся максимумом
    — см. `SEAM_SCORE`. Пара, у которой не измерена хотя бы одна из осей, в
    кандидаты не попадает: судить стык по половине свидетельства нельзя.
    """
    flow_weight = FLOW_WEIGHT if flow_weight is None else flow_weight
    if not step or step <= 0:
        raise ValueError(
            f"обычный шаг клипа {step!r}: делить на него нельзя. Клип, который "
            f"не движется, — исход «не смогли», и решается он выше (Р1)")
    out = []
    for key, pg in sim["pose"].items():
        fg = sim["flow"].get(key)
        if pg is None or fg is None:
            continue
        i, j = key
        seam_pose = pg / step
        seam_flow = fg / step
        out.append({
            "i": i, "j": j, "frames": j - i + 1,
            "pose_gap": round(pg, 4), "flow_gap": round(fg, 4),
            "seam_pose": round(seam_pose, 3), "seam_flow": round(seam_flow, 3),
            "score": round(max(seam_pose, flow_weight * seam_flow), 3),
        })
    out.sort(key=lambda c: (c["score"], c["i"]))
    return out


def overlap(a, b) -> float:
    """Какую долю более КОРОТКОЙ из двух петель они делят.

    Долю от короткой, а не объединения: петля 41 кадра, целиком лежащая внутри
    петли 300 кадров, — это то же самое движение, и IoU в 0.14 сказал бы, что
    они разные.
    """
    inter = min(a["j"], b["j"]) - max(a["i"], b["i"]) + 1
    if inter <= 0:
        return 0.0
    return inter / min(a["frames"], b["frames"])


def select(cands, *, overlap_max=None, top=None) -> dict:
    """Принять несколько РАЗНЫХ петель, а не десяток сдвинутых на кадр.

    Кандидаты приходят отсортованными по стыку; принимается лучший, затем
    следующий, который существенно не перекрывается с уже принятыми.
    """
    overlap_max = OVERLAP_MAX if overlap_max is None else overlap_max
    top = TOP_LOOPS if top is None else top
    kept, dropped = [], 0
    for c in cands:
        if len(kept) >= top:
            break
        if any(overlap(c, k) > overlap_max for k in kept):
            dropped += 1
            continue
        kept.append(c)
    return {"kept": kept, "dropped_overlap": dropped,
            "considered": len(cands)}


def repeat_plan(length, *, fps=None) -> list:
    """Сколько повторов петли даёт продуктовую длину 5-10 с.

    Склейка N повторов без дублирования стыкового кадра — N*(L-1)+1 кадров.
    Длина при этом остаётся вида 4k+1 автоматически, раз L-1 кратно 4; это
    проверяется `fork_comfy.snap_frames`, а не утверждается (Е1).
    """
    fps = fork_comfy.WRAP_FPS if fps is None else fps
    out = []
    for n in range(1, 200):
        total = n * (length - 1) + 1
        seconds = total / fps
        if seconds > fork_comfy.SECONDS_MAX:
            break
        if seconds < fork_comfy.SECONDS_MIN:
            continue
        out.append({"repeats": n, "frames": total,
                    "seconds": round(seconds, 2),
                    "snapped": fork_comfy.snap_frames(total)})
    return out


# ---------------------------------------------------------------------------
# GIF: единственный способ увидеть стык, а не прочитать про него
# ---------------------------------------------------------------------------

def gif_indices(i, j, *, max_frames=None) -> list:
    """Номера кадров для GIF петли [i..j].

    Последний кадр петли (j) НЕ КЛАДЁТСЯ: он и есть повтор кадра i, и в склейке
    повторов его выбрасывают. GIF зациклен сам, поэтому его переход
    «последний -> первый» — это ровно тот стык, который увидит оператор.
    """
    max_frames = GIF_MAX_FRAMES if max_frames is None else max_frames
    body = list(range(i, j))  # без j — см. докстринг
    if len(body) <= max_frames:
        return body
    stride = math.ceil(len(body) / max_frames)
    return body[::stride][:max_frames]


def make_gif(paths, i, j, out_path, *, fps=None, max_frames=None,
             max_side=None) -> dict:
    """Собрать GIF петли. Возвращает путь, число кадров и размер файла."""
    from PIL import Image

    fps = fork_comfy.WRAP_FPS if fps is None else fps
    max_side = GIF_MAX_SIDE if max_side is None else max_side
    idx = gif_indices(i, j, max_frames=max_frames)
    if not idx:
        return {"path": None, "frames": 0, "bytes": 0,
                "note": f"петля [{i}..{j}] пуста — GIF собирать не из чего"}
    stride = (idx[1] - idx[0]) if len(idx) > 1 else 1
    frames_img = []
    for k in idx:
        im = Image.open(paths[k]).convert("RGB")
        if max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * scale)),
                            max(1, round(im.height * scale))))
        frames_img.append(im)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Длительность кадра — РЕАЛЬНАЯ: прорежённый GIF обязан идти с той же
    # скоростью, что материал, иначе оператор судит не то движение.
    frames_img[0].save(out_path, save_all=True, append_images=frames_img[1:],
                       duration=int(round(1000 * stride / fps)), loop=0,
                       optimize=True)
    return {"path": str(out_path), "frames": len(idx),
            "bytes": out_path.stat().st_size, "stride": stride,
            "size": frames_img[0].size}


# ---------------------------------------------------------------------------
# Разбор целиком
# ---------------------------------------------------------------------------

def _report(outcome, note, t0, steps, **extra) -> dict:
    out = {"outcome": outcome, "loops": [], "note": note,
           "elapsed": round(time.perf_counter() - t0, 4),
           "steps": [{"step": s, "outcome": o, "note": n, "seconds": round(e, 4)}
                     for s, o, n, e in steps]}
    out.update(extra)
    out["note"] = f"{outcome}: {note}"
    return out


def refine_all(loops, paths, *, stride, reader=None, cache=None, fps=None,
               min_frames=None, flow_weight=None, blocked=None) -> dict:
    """Уточнить границы петель на ПОЛНОЙ частоте в окне ±шаг вокруг найденного.

    Коарс-ту-файн: прорежённый проход отвечает на вопрос «где в клипе повтор»,
    а точные i и j он ответить не может — между двумя снятыми кадрами лежит
    stride-1 неснятых, и любой из них может дать стык лучше.

    ЧТО ЗДЕСЬ НЕ ДЕЛАЕТСЯ. Уточнение НЕ ПЕРЕУПОРЯДОЧИВАЕТ петли: планка
    преимущества применена на прорежённом проходе, где все кандидаты сравнимы
    между собой, а тонкие оценки посчитаны в окнах и сравнимы только внутри
    своей петли. Смешать два масштаба в один рейтинг — это тот самый дефект
    «сложили разнородное», от которого модуль лечится в оценке стыка.
    """
    if stride <= 1 or not loops:
        return {"loops": loops, "poses": 0, "elapsed": 0.0, "windows": 0}
    t = time.perf_counter()
    n = len(paths)
    wanted, windows = set(), []
    for lp in loops:
        wi = range(max(0, lp["i"] - stride), min(n - 1, lp["i"] + stride) + 1)
        wj = range(max(0, lp["j"] - stride), min(n - 1, lp["j"] + stride) + 1)
        # Плюс один кадр за каждым концом: производную берём вперёд.
        wanted |= {k for k in wi} | {k + 1 for k in wi if k + 1 < n}
        wanted |= {k for k in wj} | {k + 1 for k in wj if k + 1 < n}
        windows.append((list(wi), list(wj)))
    order = sorted(wanted)
    got = read_all([paths[k] for k in order], reader=reader, cache=cache)
    st = {k: v for k, v in zip(order, states(got["poses"]))}

    # Шаг ПОЛНОЙ частоты — по соседним парам внутри окон, то есть по настоящим
    # соседям, а не по прорежённым. Одно число на клип, чтобы тонкие оценки
    # были сравнимы хотя бы между собой.
    import numpy as np

    fine = [g for g in (pose_gap(st.get(k), st.get(k + 1)) for k in order)
            if g is not None]
    if not fine:
        return {"loops": loops, "poses": len(order),
                "elapsed": round(time.perf_counter() - t, 4),
                "windows": len(windows),
                "note": "уточнение не состоялось: на полной частоте поз нет"}
    step = float(np.median(fine))
    out = []
    for lp, (wi, wj) in zip(loops, windows):
        best = None
        for i in wi:
            for j in wj:
                if not length_is_admissible(j - i + 1, fps=fps,
                                            min_frames=min_frames):
                    continue
                if blocked is not None and blocked(i, j):
                    continue
                pg = pose_gap(st.get(i), st.get(j))
                fg = (None if any(st.get(k) is None for k in (i, i + 1, j, j + 1))
                      else _flow_between(st, i, j))
                if pg is None or fg is None or step <= 0:
                    continue
                sc = round(max(pg / step, (FLOW_WEIGHT if flow_weight is None
                                           else flow_weight) * fg / step), 3)
                if best is None or sc < best["score"]:
                    best = {"i": i, "j": j, "frames": j - i + 1,
                            "pose_gap": round(pg, 4), "flow_gap": round(fg, 4),
                            "seam_pose": round(pg / step, 3),
                            "seam_flow": round(fg / step, 3), "score": sc}
        item = dict(lp)
        if best is not None:
            item["coarse"] = {"i": lp["i"], "j": lp["j"], "score": lp["score"]}
            item.update(best)
        out.append(item)
    return {"loops": out, "poses": len(order), "fine_step": round(step, 4),
            "elapsed": round(time.perf_counter() - t, 4), "windows": len(windows)}


def _flow_between(st, i, j):
    """Расхождение направлений по словарю состояний (для уточнения)."""
    import numpy as np

    quad = [st.get(i), st.get(i + 1), st.get(j), st.get(j + 1)]
    if any(q is None for q in quad):
        return None
    shared = [k for k in quad[0]
              if all(q[k][1] >= pose.MIN_VISIBILITY for q in quad)]
    if not shared:
        return None
    return round(float(np.mean(
        [float(np.linalg.norm((quad[1][k][0] - quad[0][k][0])
                              - (quad[3][k][0] - quad[2][k][0])))
         for k in shared])), 6)


def find_loops(source, *, out_dir=None, fps=None, reader=None, gray=None,
               cache=None, min_frames=None, overlap_max=None, top=None,
               advantage_min=None, flow_weight=None, gif=True, decode=None,
               stride=None, max_frames=None) -> dict:
    """Найти петли в драйвинге. Дешёвое раньше дорогого (П2), три исхода (Р1).

    `source` — каталог кадров или видеофайл. Видео раскодируется
    `fork_video.frames` (Е1: своего раскодировщика здесь нет).
    """
    fps = fork_comfy.WRAP_FPS if fps is None else fps
    min_frames = LOOP_MIN_FRAMES if min_frames is None else min_frames
    advantage_min = ADVANTAGE_MIN if advantage_min is None else advantage_min
    top = TOP_LOOPS if top is None else top
    max_frames = MAX_FRAMES if max_frames is None else max_frames
    t = time.perf_counter()
    steps = []
    src = Path(source)

    # 1. Кадры. Миллисекунды, и отвечают на вопрос, ради которого иначе
    # пришлось бы минутами снимать позы (П2).
    t0 = time.perf_counter()
    if src.is_dir():
        paths = frame_paths(src)
    elif src.is_file():
        from . import fork_video
        decode = fork_video.frames if decode is None else decode
        dest = Path(out_dir or ".") / "frames"
        got = decode(str(src), str(dest), overwrite=True)
        if got.get("outcome") != PASS:
            steps.append(("кадры", got["outcome"], got["note"],
                          time.perf_counter() - t0))
            return _report(got["outcome"], got["note"], t, steps, frames=0)
        paths = [Path(p) for p in got["paths"]]
    else:
        note = f"{src} — не каталог кадров и не файл"
        steps.append(("кадры", UNMEASURED, note, time.perf_counter() - t0))
        return _report(UNMEASURED, note, t, steps, frames=0)

    n = len(paths)
    if n < min_frames + 1:
        note = (f"кадров {n}, а самая короткая петля требует {min_frames} плюс "
                f"кадр на производную. Материал короче петли — это измерение, "
                f"а не сбой прибора")
        steps.append(("кадры", FAIL, note, time.perf_counter() - t0))
        return _report(FAIL, note, t, steps, frames=n, scan=SCAN_FULL)
    if n > max_frames:
        note = (f"кадров {n}, потолок {max_frames} ({max_frames / fps / 60:.0f} "
                f"минут при {fps} к/с). Даже по прорежённой это "
                f"{n / (COARSE_STRIDE or 1) * 0.031 / 60:.0f}+ минут одного "
                f"только снятия поз — разбирать не беремся. Нарежьте материал")
        steps.append(("кадры", UNMEASURED, note, time.perf_counter() - t0))
        return _report(UNMEASURED, note, t, steps, frames=n, scan=SCAN_TOO_LONG)

    stride = (COARSE_STRIDE if n > COARSE_ABOVE_FRAMES else 1) \
        if stride is None else stride
    scan = SCAN_FULL if stride == 1 else SCAN_COARSE
    steps.append(("кадры", PASS,
                  f"кадров {n}, разбор по {scan}"
                  + ("" if stride == 1 else f" 1 из {stride}"),
                  time.perf_counter() - t0))

    # 2. РЕЗЫ. По пикселям, до снятия поз: кадр стоит миллисекунды против
    # десятков миллисекунд позы, а отбрасывает целые области кандидатов (П2).
    cut = cuts(paths, gray=gray)
    steps.append(("резы", cut["outcome"], cut["note"], cut["elapsed"]))
    cut_set = sorted(cut["cuts"])

    # 3. ПОЗЫ. Самый дорогой шаг из всех.
    index = list(range(0, n, stride))
    got = read_all([paths[k] for k in index], reader=reader, cache=cache)
    if got["unreadable"]:
        note = (f"позу снять нечем: {got['unreadable']} кадр(ов) не опрошены "
                f"({got['first_why']}). Это НЕ «петель нет»")
        steps.append(("позы", UNMEASURED, note, got["elapsed"]))
        return _report(UNMEASURED, note, t, steps, frames=n, taken=got["taken"],
                       scan=scan, stride=stride)

    who = presence(got["poses"], people=got["people"], index=index)
    coverage = got["taken"] / len(index)
    steps.append(("позы", PASS if coverage >= MIN_POSE_COVERAGE else UNMEASURED,
                  f"поза снята на {got['taken']} из {len(index)} опрошенных "
                  f"кадров ({coverage:.0%}), из кэша {got['cached']}, "
                  f"{got['elapsed'] / max(1, len(index) - got['cached']):.4f} "
                  f"с/кадр", got["elapsed"]))

    if who["crowd"]:
        note = (f"людей в кадре несколько (до {max(who['crowd'])}). Кого вести "
                f"— решает разметка `fork_props` (роль "
                f"{'протагонист'!r}); второго способа выбирать протагониста "
                f"здесь нет и не будет (Е1). Пока он не назван, детектор берёт "
                f"на каждом кадре первую попавшуюся рамку, и скелет прыгает с "
                f"человека на человека посреди клипа")
        steps.append(("люди", UNMEASURED, note, 0.0))
        return _report(UNMEASURED, note, t, steps, frames=n, taken=got["taken"],
                       scan=scan, stride=stride, crowd=who["crowd"])
    if got["taken"] == 0:
        note = (f"человека в кадре нет ни на одном из {len(index)} опрошенных "
                f"кадров. Это не «петель нет»: петли ищутся по телу")
        steps.append(("люди", UNMEASURED, note, 0.0))
        return _report(UNMEASURED, note, t, steps, frames=n, taken=0, scan=scan,
                       stride=stride)
    if who["long_gaps"]:
        spans = ", ".join(f"{index[a]}..{index[b]}" for a, b in who["long_gaps"])
        steps.append(("люди", UNMEASURED,
                      f"человек уходит из кадра: подряд без тела кадры {spans}. "
                      f"Петли через эти участки не выпускаются", 0.0))
    if coverage < MIN_POSE_COVERAGE:
        note = (f"поза снята только на {got['taken']} из {len(index)} кадров "
                f"({coverage:.0%}, планка {MIN_POSE_COVERAGE:.0%}). Отсутствие "
                f"петель на таком материале ничего не значит"
                + (f". Человек уходит из кадра с кадра {who['left_at']}"
                   if who["long_gaps"] else ""))
        return _report(UNMEASURED, note, t, steps, frames=n, taken=got["taken"],
                       coverage=round(coverage, 3), scan=scan, stride=stride,
                       left_at=who["left_at"])

    st = states(got["poses"])
    step_info = typical_step(st)
    if step_info["step"] is None or step_info["step"] <= 0:
        note = (f"{step_info['note']}. Стык не в чем мерить: единица измерения "
                f"— обычный шаг этого же клипа" if step_info["step"] is None
                else "клип не движется: обычный шаг равен нулю, ранжировать "
                     "стыки нечем")
        steps.append(("масштаб", UNMEASURED, note, 0.0))
        return _report(UNMEASURED, note, t, steps, frames=n, taken=got["taken"],
                       scan=scan, stride=stride)
    steps.append(("масштаб", PASS, step_info["note"], 0.0))

    # 4. Самоподобие и кандидаты. Пары, накрывающие рез или дыру присутствия,
    # не выпускаются: такой кандидат по позам выглядит удачей.
    import bisect

    # Блокируют ДЛИННЫЕ дыры присутствия, а не любой пропуск. Одиночный промах
    # детектора посреди петли — не причина выбрасывать петлю: человек в кадре,
    # и кадры для GIF на месте. А вот участок, где его нет полсекунды и больше
    # (`PRESENCE_GAP_MIN`), — это уход из кадра, и петля через него мусорная.
    gone = sorted(index[k] for a, b in who["long_gaps"] for k in range(a, b + 1))

    def blocked(i, j):
        if bisect.bisect_left(cut_set, i) < bisect.bisect_left(cut_set, j):
            return "рез внутри петли"
        if bisect.bisect_left(gone, i) < bisect.bisect_right(gone, j):
            return "человека нет в кадре внутри петли"
        return None

    t0 = time.perf_counter()
    sim = similarity(st, fps=fps, min_frames=min_frames, index=index,
                     blocked=blocked)
    cands = score_pairs(sim, step_info["step"], flow_weight=flow_weight)
    sim_elapsed = time.perf_counter() - t0
    blocked_note = ("; ".join(f"{why}: {cnt}"
                              for why, cnt in sorted(sim["rejected"].items()))
                    or "нет")
    if not cands:
        note = (f"допустимых пар {sim['pairs']}, измерить не удалось ни одной. "
                f"Отвергнуто до измерения — {blocked_note}")
        steps.append(("кандидаты", UNMEASURED, note, sim_elapsed))
        return _report(UNMEASURED, note, t, steps, frames=n, taken=got["taken"],
                       scan=scan, stride=stride, pairs=sim["pairs"],
                       rejected=sim["rejected"])

    import numpy as np
    median_score = float(np.median([c["score"] for c in cands]))
    best = cands[0]
    advantage = median_score / best["score"] if best["score"] > 0 else math.inf
    steps.append(("кандидаты", PASS,
                  f"пар допустимых {sim['pairs']}, отвергнуто до измерения "
                  f"({blocked_note}), измерено {sim['measured']}, не смогли "
                  f"{sim['unmeasurable']}; лучший стык {best['score']} шага, "
                  f"типичный {median_score:.3f}, преимущество {advantage:.2f}x",
                  sim_elapsed))

    # Планка преимущества применяется К КАЖДОМУ кандидату, а не только к
    # лучшему: иначе на клипе с одной хорошей петлёй прибор доберёт до пяти
    # заведомым мусором — просто потому, что место в таблице осталось. С уходом
    # оператора из контура это перестало быть вопросом вкуса: недобранное место
    # видит платящий человек, и лучше четыре петли, чем пятая плохая.
    worthy = [c for c in cands
              if (median_score / c["score"] if c["score"] > 0 else math.inf)
              >= advantage_min]
    if not worthy:
        note = (f"ПЕТЕЛЬ НЕ НАШЛОСЬ ({scan}): ни одна пара не обошла типичную в "
                f"{advantage_min} раза, лучшая обошла в {advantage:.2f}. Так "
                f"выглядит материал без повтора — лучшая пара в нём выигрывает "
                f"только тем, что она короче. Поза снята на {got['taken']} из "
                f"{len(index)} опрошенных кадров, пар разобрано "
                f"{sim['measured']}, отвергнуто до измерения ({blocked_note})")
        return _report(FAIL, note, t, steps, frames=n, taken=got["taken"],
                       coverage=round(coverage, 3), pairs=sim["pairs"],
                       measured_pairs=sim["measured"],
                       unmeasurable_pairs=sim["unmeasurable"],
                       rejected=sim["rejected"], scan=scan, stride=stride,
                       advantage=round(advantage, 3),
                       typical_score=round(median_score, 3),
                       best_score=best["score"], candidates=len(cands))

    # 5. Подавление пересечений: несколько РАЗНЫХ петель, и ровно столько,
    # сколько набралось. Недобор печатается числом, а не добивается мусором.
    chosen = select(worthy, overlap_max=overlap_max, top=top)

    # 6. Уточнение на полной частоте — только вокруг найденного.
    fine = refine_all(chosen["kept"], paths, stride=stride, reader=reader,
                      cache=cache, fps=fps, min_frames=min_frames,
                      flow_weight=flow_weight, blocked=blocked)
    if stride > 1:
        steps.append(("уточнение", PASS,
                      f"окон {fine['windows']}, снято поз {fine['poses']} на "
                      f"полной частоте", fine["elapsed"]))

    loops = []
    for rank, c in enumerate(fine["loops"], 1):
        loop = dict(c)
        loop["rank"] = rank
        loop["seconds"] = round(c["frames"] / fps, 2)
        loop["advantage"] = (round(median_score / c["score"], 2)
                             if c["score"] > 0 else None)
        loop["repeats"] = repeat_plan(c["frames"], fps=fps)
        loop["gif"] = None
        if gif and out_dir is not None:
            loop["gif"] = make_gif(
                paths, c["i"], c["j"],
                Path(out_dir) / f"loop_{c['i']:04d}_{c['j']:04d}.gif", fps=fps)
        loops.append(loop)

    short = ("" if len(loops) >= top else
             f" ПЕТЕЛЬ МЕНЬШЕ ЗАКАЗАННЫХ {top}: набралось {len(loops)} "
             f"существенно разных, добивать список ухудшенными копиями того же "
             f"движения нельзя.")
    note = (f"петель принято {len(loops)} из {len(worthy)} прошедших планку "
            f"преимущества {advantage_min}x (всего пар с оценкой {len(cands)}, "
            f"отброшено по пересечению {chosen['dropped_overlap']}); кадров "
            f"{n}, опрошено {len(index)} ({scan}), поза снята на "
            f"{got['taken']}, резов {len(cut_set)}, пар разобрано "
            f"{sim['measured']}, не смогли {sim['unmeasurable']}.{short} Это "
            f"РАНГ, а не вердикт: планки бесшовности для расстояния поз у "
            f"проекта нет")
    return _report(PASS, note, t, steps, frames=n, taken=got["taken"],
                   coverage=round(coverage, 3), pairs=sim["pairs"],
                   measured_pairs=sim["measured"],
                   unmeasurable_pairs=sim["unmeasurable"],
                   rejected=sim["rejected"], cuts=cut_set, scan=scan,
                   stride=stride, asked=top,
                   candidates=len(cands), worthy=len(worthy),
                   dropped_overlap=chosen["dropped_overlap"],
                   advantage=round(advantage, 3),
                   typical_score=round(median_score, 3),
                   typical_step=round(step_info["step"], 4),
                   pose_seconds=got["elapsed"] + fine["elapsed"],
                   pose_frames=len(index) + fine["poses"],
                   cut_seconds=cut["elapsed"], cached=got["cached"],
                   loops=loops)


def table(report) -> str:
    """Таблица петель для человека."""
    rows = [f"{'#':>2} {'кадры':>11} {'длина':>6} {'сек':>5} {'стык':>6} "
            f"{'поза':>6} {'поток':>6} {'выигрыш':>8}  повторы -> с        GIF"]
    for lp in report.get("loops", []):
        rep = ", ".join(f"{r['repeats']}x={r['frames']}к/{r['seconds']}с"
                        for r in lp["repeats"]) or "нет в полосе 5-10 с"
        g = lp.get("gif") or {}
        gtxt = (f"{Path(g['path']).name} {g['frames']}к {g['bytes']}Б"
                if g.get("path") else "-")
        rows.append(f"{lp['rank']:>2} {lp['i']:>5}..{lp['j']:<5} "
                    f"{lp['frames']:>6} {lp['seconds']:>5} {lp['score']:>6} "
                    f"{lp['seam_pose']:>6} {lp['seam_flow']:>6} "
                    f"{str(lp['advantage']) + 'x':>8}  {rep}  {gtxt}")
    return "\n".join(rows)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python3 -m ball_reel.fork_looper",
        description="Отбор петель в драйвинге: ранжирует стыки и кладёт GIF.")
    ap.add_argument("source", help="каталог кадров или видеофайл")
    ap.add_argument("--out", default="looper_out", help="куда класть GIF-ы")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--min-frames", type=int, default=None)
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--overlap-max", type=float, default=None)
    ap.add_argument("--cache", default=None, help="JSON с уже снятыми позами")
    ap.add_argument("--stride", type=int, default=None,
                    help="шаг прореживания; по умолчанию 1, а на длинном "
                         "материале включается сам")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args(argv)

    rep = find_loops(a.source, out_dir=a.out, fps=a.fps,
                     min_frames=a.min_frames, top=a.top,
                     overlap_max=a.overlap_max, cache=a.cache,
                     stride=a.stride, max_frames=a.max_frames,
                     gif=not a.no_gif)
    print(f"ИСХОД: {rep['outcome']}")
    for s in rep["steps"]:
        print(f"  [{s['outcome']:>18}] {s['step']:<12} {s['seconds']:>7.3f} с  "
              f"{s['note']}")
    if rep.get("loops"):
        print()
        print(table(rep))
    print()
    print(rep["note"])
    return EXIT_BY_OUTCOME[rep["outcome"]]


if __name__ == "__main__":
    raise SystemExit(main())
