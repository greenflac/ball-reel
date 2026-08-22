"""Финальная сборка ролика после Kling: кроп в 9:16 плюс возврат звука.

ЗАЧЕМ. После липсинка на руках лежат ДВА файла и ни одного готового ролика:
выход Kling — КВАДРАТ 960x960 БЕЗ ЗВУКА, и исходный драйвинг — 9:16 СО ЗВУКОМ,
из которого вырезали окно. Оба решения владельца от 2026-08-22 обязательны:

  1. КРОП В 9:16. Поля соотношения сторон у эндпоинта НЕТ (проверено щупом с
     негативным контролем предыдущей сменой), поэтому форму кадра задаём мы,
     после генерации, резкой. Из 960x960 берём 540x960.
  2. ВОЗВРАТ ЗВУКА. Драйвинг уезжает в модель без дорожки, но зритель обязан
     услышать оригинальный звук: это ЛИПСИНК, губы обязаны совпасть со
     словами. Ролик без звука — не результат, а полуфабрикат.

ГЛАВНАЯ ТРУДНОСТЬ, И ОНА ИЗМЕРЕНА, А НЕ ПРЕДПОЛОЖЕНА. Kling возвращает НЕ
РОВНО столько кадров, сколько отправили. ИЗМЕРЕНО 2026-08-22 на четырёх
прогонах (хэндоф смены, файлы в `work/`):

    отправили 100 -> вернулось  99   (-1)
    отправили  88 -> вернулось  91   (+3)
    отправили  90 -> вернулось  88   (-2)
    отправили 180 -> вернулось 180   ( 0)

Расхождение идёт В ОБЕ СТОРОНЫ и не объясняется ни округлением длины, ни
шагом обёртки. Значит приклеивать звук ВСЛЕПУЮ нельзя: три кадра на 30 к/с —
это 0.1 с, и на липсинке это слышно. Модуль ОБЯЗАН это обнаружить и ответить
тремя исходами, а не молча подогнать.

ПОЧЕМУ РАСХОЖДЕНИЕ МЕРЯЕТСЯ ПО ХУДШЕМУ СЛУЧАЮ, А НЕ ПО КОНЦУ РОЛИКА. Мы знаем
СКОЛЬКО кадров разошлось и не знаем ГДЕ. Если Kling выбросил кадр в самом
конце — сдвига губ нет нигде, и звук ляжет идеально. Если в самом начале —
ВЕСЬ ролик уехал на этот кадр. Разведать это нечем: сравнивать выход с
драйвингом покадрово бессмысленно, у выхода другая внешность и другая
композиция кадра. Поэтому расхождение читается как ВЕРХНЯЯ ОЦЕНКА сдвига губ,
и допуск ставится по ней. Занижать оценку тут — это ровно «молча подогнать».

ТРИ ИСХОДА (Р1), и третий не сворачивается ни в первый, ни во второй:

    годно            0 кадров расхождения — звук клеится как есть;
                     либо в пределах допуска — звук клеится, и в отчёте
                     сказано НА СКОЛЬКО кадров и В КАКУЮ сторону
    не годно         расхождение больше допуска — звук НЕ клеится, ролик
                     пишется немым. Молча растянутый звук ХУЖЕ отсутствующего:
                     немой ролик виден сразу, а уехавшие губы уезжают в показ
    не смогли        длительность не читается (нет ffprobe, битый файл,
                     контейнер без числа кадров) — ни клеить, ни отказывать
                     оснований нет

ЗВУК НЕ РАСТЯГИВАЕТСЯ НИКОГДА. Растяжение на 1% (`atempo=0.99`) убрало бы
расхождение в числах и оставило бы его в ушах: сдвиг всё равно накопится, а
след — «мы что-то сделали со звуком» — потеряется. Модуль либо кладёт
оригинальную дорожку, либо не кладёт ничего.

ЕДИНСТВЕННЫЙ ИСТОЧНИК ИСТИНЫ (Е1). Свой разбор ffprobe здесь НЕ ПИШЕТСЯ:
метаданные снимает `fork_video.probe`, команды исполняет `fork_video.run_decode`
(это общий «запусти ffmpeg с этим argv», а не только раскодирование), слова
вердиктов и коды возврата приходят из `fork_identity`/`fork_video`. Второй
разбор JSON от ffprobe в репозитории означал бы, что поля читаются двумя
способами и однажды разъедутся.

ТОЧКИ ВНЕДРЕНИЯ (Т4). Наружу модуль ходит только через ДВА параметра —
`prober` и `runner`, — и оба разрешаются В ТЕЛЕ функций, а не в сигнатуре:
умолчание, связанное на импорте, мутацией уже не достаётся. Ни один тест этого
модуля не запускает ffmpeg и не ходит в сеть; на диске тесты трогают только
пустышки во временном каталоге, потому что `fork_video.probe` по устройству
сначала проверяет, что файл есть, и это его правильное свойство.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import fork_video
from .fork_identity import FAIL, PASS, UNMEASURED

# ---------------------------------------------------------------------------
# КОНСТАНТЫ-РЕШЕНИЯ. У каждой помечено происхождение (И4).
# ---------------------------------------------------------------------------

#: ВЫБРАНО (кем: владелец, решение от 2026-08-22; из чего: из вертикальных
#: форматов площадок). 9:16 — форма кадра ленты. Не РАСЧЁТ: никакой формулой
#: она не выводится, это требование площадки.
TARGET_RATIO_W, TARGET_RATIO_H = 9, 16

#: РАСЧЁТ (по устройству yuv420p): цветность в 4:2:0 прорежена вдвое по обеим
#: осям, поэтому и ширина, и высота обязаны быть ЧЁТНЫМИ — на нечётной x264
#: отказывается кодировать («width not divisible by 2»). Смещение окна тоже
#: прижимается к чётному: при нечётном x яркость и цветность разъезжаются на
#: полпикселя, и это видно на контрастной границе как цветная кайма.
DIM_MULTIPLE = 2

#: РАСЧЁТ, и источник назван честно: это НЕ наш замер. Порог заметности
#: рассинхрона губ — ITU-R BT.1359-1: звук ВПЕРЕДИ картинки становится заметен
#: с 45 мс, звук ПОЗАДИ — только со 125 мс (человек привык, что звук приходит
#: позже: гром после молнии). Публикацию эта смена не измеряла и не проверяла
#: прибором — НЕПРОВЕРЕНО (Ц4).
#:
#: БЕРЁТСЯ УЗКАЯ СТОРОНА, 45 мс, И ЭТО НЕ ПЕРЕСТРАХОВКА. Асимметрия допуска
#: имела бы смысл, если бы мы знали ЗНАК сдвига губ. Мы знаем только знак
#: разницы ДЛИН, а сдвиг зависит от того, где Kling выбросил или вставил кадр,
#: — то есть знак нам неизвестен по построению (см. докстринг). Допуск,
#: рассчитанный по широкой стороне, был бы верен ровно в половине случаев.
LIPSYNC_AUDIO_AHEAD_MS = 45

#: ВЫБРАНО (кем: эта смена; из чего: из ИЗМЕРЕННОГО на настоящем материале).
#: Насколько окно, выбранное по движению, обязано быть ЛУЧШЕ центрального,
#: чтобы уводить кадр от центра. ИЗМЕРЕНО на `work/arm_out_control_plain.mp4`
#: (20 кадров через каждые 5, поколоночная межкадровая разница): лучшее окно
#: набирает 1.0024 от центрального — то есть на настоящем материале выигрыш
#: 0.24%, и это шум, а не человек сбоку. Порог 1.05 отправляет такой случай в
#: «не смогли выбрать, берём центр», и это верный ответ. Ниже 1.0024 порог
#: опускать нельзя — начнём дёргать кадр по шуму.
BIAS_GAIN_MIN = 1.05

#: ВЫБРАНО: смещение окна задаётся числом от -1 (окно прижато к левому краю)
#: до +1 (к правому), 0 — центр. Именно долей, а не пикселями: пиксели зависят
#: от разрешения выхода, а оно у эндпоинта уже менялось.
BIAS_LIMIT = 1.0

#: ВЫБРАНО: качество и кодек финального файла. CRF 18 — визуально почти
#: неотличимо от исходника у x264; звук перекодируется в aac 128k, потому что
#: копирование дорожки (`-c:a copy`) при обрезке по времени кладёт кусок,
#: начинающийся с ближайшего пакета, а не с заказанной миллисекунды.
VIDEO_CRF = 18
VIDEO_PRESET = "veryfast"
AUDIO_BITRATE = "128k"

#: Коды возврата берутся у соседа, а не заводятся свои (Е1): 0 годно,
#: 1 не годно, 2 не смогли.
EXIT_BY_OUTCOME = fork_video.EXIT_BY_OUTCOME


# ---------------------------------------------------------------------------
# Чистые функции: развилки вынесены из точки входа, чтобы их красил тест (Т5).
# ---------------------------------------------------------------------------

def _even(value: int) -> int:
    """Вниз до кратного DIM_MULTIPLE. Вниз, а не вверх: вверх — выйти за кадр."""
    return int(value) - int(value) % DIM_MULTIPLE


def crop_geometry(width, height, *, ratio_w=TARGET_RATIO_W,
                  ratio_h=TARGET_RATIO_H, bias=0.0) -> dict:
    """План кропа: откуда и какое окно резать, и сколько площади теряем.

    ГДЕ БРАТЬ ОКНО — ГЛАВНОЕ РЕШЕНИЕ ЭТОЙ ФУНКЦИИ, И ОНО ПАРАМЕТР.
    Умолчание — ЦЕНТР, и вот почему, а не «потому что просто».

      Аргумент за смещение настоящий: человек в кадре не всегда по центру, и
      центральный крой у стоящего сбоку срежет ему руку или пол-лица. Но
      смещение требует ЗНАТЬ, где человек, а знание это здесь взять неоткуда:
      детектора субъекта в этом модуле нет и не будет (один писатель на
      модуль), а смещение, выбранное наугад, ХУЖЕ центра — оно уводит кадр
      туда, где никого нет, и делает это уверенно.

      ИЗМЕРЕНО на настоящем выходе Kling (`work/arm_out_control_plain.mp4`, 20
      кадров): центр тяжести межкадрового движения стоит на x=0.492 ширины при
      центре 0.500, а лучшее по движению окно выигрывает у центрального 0.24%.
      То есть на ЭТОМ материале центр — не компромисс, а измеренно верный
      выбор: модель сама ставит человека в середину квадрата.

      Параметр `bias` оставлен для случая, когда положение человека ИЗВЕСТНО:
      его считает `bias_from_columns` по поколоночной карте движения, и она же
      честно отвечает «не смогли выбрать», когда карта плоская.

    Возвращает `{"outcome", "x", "y", "w", "h", "lost_percent", "kept_percent",
    "axis", "note"}`. Три исхода: годно — окно посчитано; не годно — размеры
    или соотношение бессмысленны; не смогли — размеры не сняты.
    """
    if width is None or height is None:
        return {**_geom_blank(), "outcome": UNMEASURED,
                "note": (f"размеры кадра не сняты (ширина {width}, высота "
                         f"{height}): резать вслепую нечего")}
    for name, value in (("ширина", width), ("высота", height)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return {**_geom_blank(), "outcome": FAIL,
                    "note": f"{name} кадра бессмысленна: {value!r}"}
    if not (isinstance(ratio_w, int) and isinstance(ratio_h, int)
            and ratio_w > 0 and ratio_h > 0):
        return {**_geom_blank(), "outcome": FAIL,
                "note": f"соотношение сторон бессмысленно: {ratio_w}:{ratio_h}"}
    try:
        bias = float(bias)
    except (TypeError, ValueError):
        return {**_geom_blank(), "outcome": FAIL,
                "note": f"смещение не число: {bias!r}"}
    if not -BIAS_LIMIT <= bias <= BIAS_LIMIT:
        return {**_geom_blank(), "outcome": FAIL,
                "note": (f"смещение {bias:g} вне полосы "
                         f"[{-BIAS_LIMIT:g}; {BIAS_LIMIT:g}]")}

    src, want = width * ratio_h, height * ratio_w
    if src > want:            # исходник ШИРЕ заказанного — режем по ширине
        w, h, axis = _even(height * ratio_w // ratio_h), _even(height), "по ширине"
    elif src < want:          # исходник УЖЕ заказанного — режем по высоте
        w, h, axis = _even(width), _even(width * ratio_h // ratio_w), "по высоте"
    else:
        w, h, axis = _even(width), _even(height), "ничего не режем"
    if w < DIM_MULTIPLE or h < DIM_MULTIPLE:
        return {**_geom_blank(), "outcome": FAIL,
                "note": (f"окно {w}x{h} вырождено: из {width}x{height} "
                         f"соотношение {ratio_w}:{ratio_h} не набирается")}

    free_x, free_y = width - w, height - h
    x = _even(round((bias + 1) / 2 * free_x))
    y = _even(round((bias + 1) / 2 * free_y)) if free_y else 0
    x, y = min(x, _even(free_x)), min(y, _even(free_y))
    kept = 100.0 * (w * h) / (width * height)
    lost = 100.0 - kept
    return {
        "outcome": PASS, "x": x, "y": y, "w": w, "h": h,
        "lost_percent": round(lost, 2), "kept_percent": round(kept, 2),
        "axis": axis,
        "note": (f"из {width}x{height} режем {w}x{h} {axis} со смещением "
                 f"{bias:+.2f} (окно x={x}, y={y}); остаётся "
                 f"{round(kept, 2):g}% площади, теряется "
                 f"{round(lost, 2):g}%"),
    }


def _geom_blank() -> dict:
    return {"x": None, "y": None, "w": None, "h": None, "lost_percent": None,
            "kept_percent": None, "axis": None}


def bias_from_columns(columns, *, ratio_w=TARGET_RATIO_W,
                      ratio_h=TARGET_RATIO_H) -> dict:
    """Смещение окна по поколоночной карте движения. Прибор с негативным контролем.

    На вход — по одному числу на КОЛОНКУ кадра: сколько в этой колонке
    движения (например, средняя межкадровая разница). Карту снимает вызывающий,
    здесь только решение — ровно чтобы решение можно было покрасить тестом на
    литералах, не открывая ни одного видео (Т5).

    НЕГАТИВНЫЙ КОНТРОЛЬ ВСТРОЕН (И5): на РОВНОЙ карте — пустой кадр, статика,
    равномерный шум — лучшее окно не отличается от центрального, выигрыш
    падает ниже `BIAS_GAIN_MIN`, и прибор обязан сказать «не смогли выбрать,
    берём центр», а не выдать бодрое число. Прибор, у которого нет входа с
    ответом «не знаю», меряет не то, что написано на его шкале.
    """
    if columns is None:
        return {"outcome": UNMEASURED, "bias": 0.0, "gain": None,
                "note": "карты движения нет: смещать нечем, берём центр"}
    try:
        cols = [float(c) for c in columns]
    except (TypeError, ValueError):
        return {"outcome": FAIL, "bias": 0.0, "gain": None,
                "note": "карта движения не разбирается в числа"}
    if any(c < 0 for c in cols):
        return {"outcome": FAIL, "bias": 0.0, "gain": None,
                "note": "в карте движения отрицательные значения"}
    width = len(cols)
    win = round(width * ratio_w / ratio_h)
    if width < 2 or win < 1 or win >= width:
        return {"outcome": UNMEASURED, "bias": 0.0, "gain": None,
                "note": (f"выбирать не из чего: колонок {width}, окно {win} — "
                         f"смещения не существует, берём центр")}
    if sum(cols) <= 0:
        return {"outcome": UNMEASURED, "bias": 0.0, "gain": None,
                "note": (f"движения в кадре нет вовсе (сумма карты 0 по "
                         f"{width} колонкам): выбирать не по чему, берём центр")}
    sums = [sum(cols[i:i + win]) for i in range(width - win + 1)]
    best = max(range(len(sums)), key=lambda i: sums[i])
    center = (width - win) // 2
    gain = sums[best] / sums[center] if sums[center] > 0 else float("inf")
    if gain < BIAS_GAIN_MIN:
        return {"outcome": UNMEASURED, "bias": 0.0, "gain": round(gain, 4),
                "note": (f"карта движения ровная: лучшее окно (x={best}) "
                         f"выигрывает у центрального (x={center}) всего "
                         f"{round(gain, 4)}x при пороге {BIAS_GAIN_MIN} — "
                         f"это шум, а не человек сбоку. Берём центр")}
    bias = (best / (width - win)) * 2 - 1
    return {"outcome": PASS, "bias": round(bias, 3), "gain": round(gain, 4),
            "note": (f"движение стоит на окне x={best} из {width - win} "
                     f"возможных (смещение {bias:+.3f}), выигрыш у центра "
                     f"{round(gain, 4)}x при пороге {BIAS_GAIN_MIN}")}


def window_frames(first, last) -> dict:
    """Сколько кадров в окне [first..last]. ОБЕ ГРАНИЦЫ ВКЛЮЧИТЕЛЬНО.

    Включительно, потому что так окно называет человек и так его нарезал
    предыдущий шаг: «кадры 100..199» — это 100 кадров, и файл окна на диске
    (`work/arm_control.mp4`) содержит ровно 100. Полуинтервал дал бы 99 и
    разошёлся бы с материалом ровно на тот один кадр, который мы здесь ловим.
    """
    if first is None or last is None:
        return {"outcome": UNMEASURED, "frames": None,
                "note": f"границы окна не заданы: [{first}..{last}]"}
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in (first, last)):
        return {"outcome": FAIL, "frames": None,
                "note": f"границы окна не целые: [{first!r}..{last!r}]"}
    if first < 0:
        return {"outcome": FAIL, "frames": None,
                "note": f"начало окна отрицательное: {first}"}
    if last < first:
        return {"outcome": FAIL, "frames": None,
                "note": f"конец окна {last} раньше начала {first}"}
    return {"outcome": PASS, "frames": last - first + 1,
            "note": f"окно [{first}..{last}] включительно — "
                    f"{last - first + 1} кадров"}


def drift_tolerance_frames(fps):
    """Допуск рассинхрона в КАДРАХ на данной частоте. Физика — в миллисекундах.

    Допуск задан временем, а не кадрами, потому что заметность рассинхрона —
    свойство человека, а не контейнера: 2 кадра на 60 к/с и 2 кадра на 24 к/с
    различаются в 2.5 раза. Вниз, а не к ближайшему: допуск, округлённый
    вверх, разрешает то, что уже слышно.

        30 к/с -> 45*30/1000 = 1.35 -> 1 кадр (33.3 мс, ещё не слышно)
                                       2 кадра = 66.7 мс — уже за порогом
        24 к/с -> 1.08 -> 1 кадр (41.7 мс)
        60 к/с -> 2.7  -> 2 кадра (33.3 мс)
    """
    if fps is None:
        return None
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return None
    if fps <= 0:
        return None
    return int(LIPSYNC_AUDIO_AHEAD_MS * fps / 1000)


def audio_drift(expected_frames, actual_frames, *, fps) -> dict:
    """Сверка ожидаемой длины окна с фактической длиной выхода Kling.

    Возвращает `{"outcome", "glue", "drift_frames", "drift_ms", "tolerance",
    "expected", "actual", "note"}`. `glue` — клеить ли звук; он выводится из
    ИСХОДА, а не назначается отдельно (Е2), поэтому «не годно с приклеенным
    звуком» тут невыразимо.
    """
    tol = drift_tolerance_frames(fps)
    blank = {"glue": False, "drift_frames": None, "drift_ms": None,
             "tolerance": tol, "expected": expected_frames,
             "actual": actual_frames}
    if tol is None:
        return {**blank, "outcome": UNMEASURED,
                "note": (f"частота не снята ({fps!r}): перевести кадры в "
                         f"миллисекунды нечем, судить о губах не по чему")}
    if expected_frames is None or actual_frames is None:
        return {**blank, "outcome": UNMEASURED,
                "note": (f"длительность не читается: окно {expected_frames}, "
                         f"выход {actual_frames} кадров")}
    if expected_frames <= 0 or actual_frames <= 0:
        return {**blank, "outcome": FAIL,
                "note": (f"кадров не может быть {expected_frames} и "
                         f"{actual_frames}")}
    drift = int(actual_frames) - int(expected_frames)
    ms = round(drift / float(fps) * 1000, 1)
    side = ("ДЛИННЕЕ" if drift > 0 else "КОРОЧЕ")
    common = {**blank, "drift_frames": drift, "drift_ms": ms}
    if drift == 0:
        return {**common, "outcome": PASS, "glue": True,
                "note": (f"кадр в кадр: окно {expected_frames}, выход "
                         f"{actual_frames}, расхождение 0 — звук клеится "
                         f"как есть")}
    if abs(drift) <= tol:
        return {**common, "outcome": PASS, "glue": True,
                "note": (f"выход {side} окна на {abs(drift)} кадр(ов) "
                         f"({abs(ms):g} мс): окно {expected_frames}, выход "
                         f"{actual_frames}. Допуск {tol} кадр(ов) при "
                         f"{float(fps):g} к/с — звук клеится, но сдвиг губ до "
                         f"{abs(ms):g} мс возможен, потому что где именно "
                         f"Kling потерял кадр, неизвестно")}
    return {**common, "outcome": FAIL, "glue": False,
            "note": (f"выход {side} окна на {abs(drift)} кадр(ов) "
                     f"({abs(ms):g} мс) при допуске {tol}: окно "
                     f"{expected_frames}, выход {actual_frames}. Звук НЕ "
                     f"клеится — молча уехавшие губы хуже немого ролика")}


def mux_argv(kling_path, out_path, geom, *, driving_path=None,
             start_seconds=None, seconds=None) -> list:
    """Команда сборки. Собирается ОТДЕЛЬНО от запуска: состав команды — решение.

    Почему именно так:

        -nostdin              иначе ffmpeg съест ввод вызывающей программы;
        -v error              нужен stderr, а не километр баннера;
        -y                    путь выхода выбрали мы, спрашивать некого;
        -ss/-t ПЕРЕД -i       обрезка ВХОДА драйвинга: так вырезается ровно то
                              окно, кадры которого уезжали в модель. После -i
                              это была бы обрезка выхода, то есть звук поехал
                              бы от нулевой секунды исходника — на 100-м кадре
                              это 3.3 с мимо;
        crop=w:h:x:y          окно из `crop_geometry`, ни одного числа своего;
        -map 0:v / -map 1:a   картинка от Kling, звук от драйвинга. Явно, а не
                              по умолчанию: умолчание ffmpeg берёт «лучший»
                              поток каждого типа и однажды возьмёт не тот;
        -shortest             ЗДЕСЬ БЕЗОПАСЕН, и только потому, что расхождение
                              длин УЖЕ измерено и признано допустимым выше. Без
                              той проверки он и был бы тем самым «молча
                              подогнать»;
        -pix_fmt yuv420p      иначе плеер площадки не покажет вовсе.
    """
    argv = [fork_video.FFMPEG_BIN, "-nostdin", "-v", "error", "-y",
            "-i", str(kling_path)]
    with_audio = driving_path is not None
    if with_audio:
        if start_seconds is not None:
            argv += ["-ss", f"{float(start_seconds):.6f}"]
        if seconds is not None:
            argv += ["-t", f"{float(seconds):.6f}"]
        argv += ["-i", str(driving_path)]
    argv += ["-filter_complex",
             f"[0:v]crop={geom['w']}:{geom['h']}:{geom['x']}:{geom['y']}[v]",
             "-map", "[v]"]
    if with_audio:
        argv += ["-map", "1:a", "-c:a", "aac", "-b:a", AUDIO_BITRATE,
                 "-shortest"]
    else:
        argv += ["-an"]
    argv += ["-c:v", "libx264", "-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    return argv


# ---------------------------------------------------------------------------
# Приборы
# ---------------------------------------------------------------------------

def audio_plan(driving_path, window, kling_path, *, prober=None) -> dict:
    """Можно ли вернуть звук и с каким сдвигом. Ни один шаг не молчит.

    `window` — пара НОМЕРОВ КАДРОВ драйвинга, обе границы включительно.
    """
    t = time.perf_counter()
    steps = []
    drv = fork_video.probe(driving_path, prober=prober)
    steps.append(("опрос драйвинга", drv["outcome"], drv["note"]))
    kln = fork_video.probe(kling_path, prober=prober)
    steps.append(("опрос выхода Kling", kln["outcome"], kln["note"]))
    win = window_frames(*window)
    steps.append(("окно", win["outcome"], win["note"]))

    out = {"steps": steps, "glue": False, "drift_frames": None,
           "drift_ms": None, "tolerance": None, "expected": win["frames"],
           "actual": kln.get("frames"), "fps": drv.get("fps"),
           "start_seconds": None, "seconds": None,
           "elapsed": round(time.perf_counter() - t, 4)}

    if UNMEASURED in (drv["outcome"], kln["outcome"]):
        return {**out, "outcome": UNMEASURED,
                "note": "метаданные не сняты, судить о звуке не по чему"}
    if FAIL in (drv["outcome"], kln["outcome"], win["outcome"]):
        return {**out, "outcome": FAIL,
                "note": "материал не годится: см. шаги выше"}
    if win["outcome"] == UNMEASURED:
        return {**out, "outcome": UNMEASURED, "note": win["note"]}
    if not drv.get("audio"):
        return {**out, "outcome": FAIL,
                "note": (f"в драйвинге {driving_path} НЕТ звуковой дорожки — "
                         f"возвращать нечего. Это не сбой сборки, это не тот "
                         f"файл: окно резалось из ролика СО звуком")}
    fps = drv.get("fps")
    kfps = kln.get("fps")
    if kfps is not None and abs(float(kfps) - float(fps)) > fork_video.FPS_TOLERANCE:
        return {**out, "outcome": FAIL,
                "note": (f"частоты разошлись: драйвинг {float(fps):g} к/с, "
                         f"выход Kling {float(kfps):g} к/с. Сравнивать длины "
                         f"В КАДРАХ при разных частотах нельзя — кадр значит "
                         f"разное время")}
    if win["frames"] is not None and drv.get("frames") is not None:
        if window[1] >= drv["frames"]:
            return {**out, "outcome": FAIL,
                    "note": (f"окно [{window[0]}..{window[1]}] выходит за "
                             f"драйвинг: в нём {drv['frames']} кадров "
                             f"(последний номер {drv['frames'] - 1})")}
    drift = audio_drift(win["frames"], kln.get("frames"), fps=fps)
    steps.append(("сверка длин", drift["outcome"], drift["note"]))
    return {**out, **{k: drift[k] for k in
                      ("outcome", "glue", "drift_frames", "drift_ms",
                       "tolerance", "expected", "actual")},
            "note": drift["note"],
            "start_seconds": round(window[0] / float(fps), 6),
            "seconds": round(kln["frames"] / float(fps), 6),
            "elapsed": round(time.perf_counter() - t, 4)}


def finish(driving_path, kling_path, out_path, *, window, bias=0.0,
           ratio_w=TARGET_RATIO_W, ratio_h=TARGET_RATIO_H,
           prober=None, runner=None) -> dict:
    """Собрать финальный ролик: кроп плюс звук плюс отчёт. Ни один шаг не молчит.

    Итог — ХУДШИЙ из исходов шагов, и он не сворачивается: «файл записан» и
    «файл годен» — разные утверждения. Расхождение длин больше допуска даёт
    НЕМОЙ файл и исход «не годно»: файл есть, звука в нём нет, и в отчёте
    сказано почему.
    """
    runner = fork_video.run_decode if runner is None else runner
    t = time.perf_counter()
    steps = []

    def report(outcome, note, **extra):
        return {"outcome": outcome, "note": note, "steps": steps,
                "out": str(out_path), "written": False, "audio": False,
                "crop": None, "audio_plan": None, "argv": None,
                "elapsed": round(time.perf_counter() - t, 4), **extra}

    kln = fork_video.probe(kling_path, prober=prober)
    steps.append(("опрос выхода Kling", kln["outcome"], kln["note"]))
    if kln["outcome"] != PASS:
        return report(kln["outcome"], f"выход Kling не опрошен: {kln['note']}")

    geom = crop_geometry(kln["width"], kln["height"], ratio_w=ratio_w,
                         ratio_h=ratio_h, bias=bias)
    steps.append(("кроп", geom["outcome"], geom["note"]))
    if geom["outcome"] != PASS:
        return report(geom["outcome"], f"кроп не посчитан: {geom['note']}",
                      crop=geom)

    plan = audio_plan(driving_path, window, kling_path, prober=prober)
    steps.extend(plan["steps"])
    steps.append(("звук", plan["outcome"], plan["note"]))
    if plan["outcome"] == UNMEASURED:
        # Не «клеим на авось» и не «отказываем»: длительность не прочиталась,
        # и это третий исход. Файл не пишется — писать немой ролик вместо
        # ролика со звуком значит принять решение вместо человека.
        return report(UNMEASURED, f"звук не проверен: {plan['note']}",
                      crop=geom, audio_plan=plan)

    argv = mux_argv(kling_path, out_path, geom,
                    driving_path=driving_path if plan["glue"] else None,
                    start_seconds=plan["start_seconds"],
                    seconds=plan["seconds"])
    ran = runner(argv)
    steps.append(("сборка", PASS if ran.get("ran") and not ran.get("code")
                  else (UNMEASURED if not ran.get("ran") else FAIL),
                  (ran.get("why") or f"ffmpeg вернул {ran.get('code')}: "
                                     f"{(ran.get('err') or '').strip()[:200]}")
                  if (not ran.get("ran") or ran.get("code"))
                  else f"ffmpeg отработал, команда из {len(argv)} слов"))
    if not ran.get("ran"):
        return report(UNMEASURED, f"собрать нечем: {ran.get('why')}",
                      crop=geom, audio_plan=plan, argv=argv)
    if ran.get("code"):
        return report(FAIL, f"ffmpeg вернул {ran['code']}: "
                            f"{(ran.get('err') or '').strip()[:200]}",
                      crop=geom, audio_plan=plan, argv=argv)

    # Е2: вердикт по тому, что ИСПОЛНИЛОСЬ. Файл опрашивается заново, и
    # размеры со звуком берутся из него, а не из наших намерений.
    got = fork_video.probe(out_path, prober=prober)
    steps.append(("опрос результата", got["outcome"], got["note"]))
    if got["outcome"] != PASS:
        return report(got["outcome"] if got["outcome"] == UNMEASURED else FAIL,
                      f"файл записан, но не подтверждён: {got['note']}",
                      crop=geom, audio_plan=plan, argv=argv, written=True)
    mismatch = []
    if (got["width"], got["height"]) != (geom["w"], geom["h"]):
        mismatch.append(f"размер {got['width']}x{got['height']} против "
                        f"плановых {geom['w']}x{geom['h']}")
    if bool(got.get("audio")) != bool(plan["glue"]):
        mismatch.append(f"звук {'есть' if got.get('audio') else 'нет'} против "
                        f"планового {'есть' if plan['glue'] else 'нет'}")
    if mismatch:
        return report(FAIL, "файл записан, но не тот: " + "; ".join(mismatch),
                      crop=geom, audio_plan=plan, argv=argv, written=True,
                      audio=bool(got.get("audio")))
    outcome = PASS if plan["outcome"] == PASS else FAIL
    tail = ("звук возвращён" if plan["glue"] else
            "БЕЗ ЗВУКА: " + plan["note"])
    return report(outcome,
                  (f"{out_path}: {got['width']}x{got['height']}, "
                   f"{got['frames']} кадров, {got['seconds']:g} с; "
                   f"потеряно {geom['lost_percent']:g}% площади кадра; {tail}"),
                  crop=geom, audio_plan=plan, argv=argv, written=True,
                  audio=bool(got.get("audio")))


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Финальная сборка: кроп в 9:16 плюс возврат звука драйвинга")
    ap.add_argument("--driving", required=True, help="исходный драйвинг СО звуком")
    ap.add_argument("--kling", required=True, help="выход Kling, квадрат, без звука")
    ap.add_argument("--out", required=True, help="куда положить финальный ролик")
    ap.add_argument("--from-frame", type=int, required=True,
                    help="первый кадр окна драйвинга, включительно")
    ap.add_argument("--to-frame", type=int, required=True,
                    help="последний кадр окна драйвинга, включительно")
    ap.add_argument("--bias", type=float, default=0.0,
                    help="смещение окна кропа: -1 влево, 0 центр, +1 вправо")
    args = ap.parse_args(argv)
    rep = finish(args.driving, args.kling, args.out,
                 window=(args.from_frame, args.to_frame), bias=args.bias)
    for name, outcome, note in rep["steps"]:
        print(f"  [{outcome}] {name}: {note}")
    print(f"[{rep['outcome']}] {rep['note']}")
    print(f"шагов {len(rep['steps'])}, за {rep['elapsed']:g} с")
    return EXIT_BY_OUTCOME[rep["outcome"]]


if __name__ == "__main__":
    raise SystemExit(main())
