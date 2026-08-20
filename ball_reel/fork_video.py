"""Раскодировщик видео: mp4 -> кадры PNG. Недостающее звено форка.

ЗАЧЕМ. Дефект №6 аудита 18.08.2026: карточка темплейта называет драйвинг
ВИДЕОФАЙЛОМ (поле `driving`, значение `driving.mp4`), а `fork_run` собирает
кадры `glob`-ом ПО КАТАЛОГУ. Раскодировщика в форке не было вовсе. Оператор,
положивший mp4 — а поле так и называется, — получал три «годно» подряд и граф
на 150 кадров при нуле поданных. Вчерашняя правка научила путь честно
останавливаться на входах, но работать всё равно нечем: этот модуль и есть то,
чем.

ДВА НАПРАВЛЕНИЯ, И ОНИ РАЗНЫЕ ПО ЦЕНЕ ОШИБКИ:

    драйвинг оператора -> кадры   вход: с него снимаются поза, лицо и маски;
                                  ошибка здесь портит ролик молча;
    готовый ролик -> кадры        выход: `fork_leak`, `fork_seam`,
                                  `fork_identity` судят ПО КАДРАМ, а ComfyUI
                                  отдаёт mp4. Без этого e2e не замыкается:
                                  сгенерировали, а судить нечем.

Функция одна и та же (`frames`), потому что задача одна и та же; разница только
в том, что на выходе пересэмплирование запрещено ВСЕГДА (ролик уже 30 к/с, и
трогать его частоту значило бы судить не то, что отдали заказчику).

---

ЧЕМ РАСКОДИРОВЫВАЕМ И ПОЧЕМУ ИМЕННО ИМ (решение записано здесь, а не в чате).

Кандидатов было три, все три есть в этой среде и все три ЗАМЕРЕНЫ на семи
фикстурах (`tools`-прогон 18.08.2026, числа в отчёте смены):

    ffmpeg/ffprobe CLI 6.1.1   /usr/bin, отдельный ПРОЦЕСС
    cv2 5.0.0                  wheel `opencv-python`, тащит СВОЙ FFmpeg внутрь
    imageio 2.37.4             видео не читает без `imageio-ffmpeg`, а его НЕТ

Выбран **ffmpeg/ffprobe CLI**, и вот четыре основания по убыванию веса:

1. **Е1, одно знание — одно место.** `fork_template._ffprobe_fps` УЖЕ снимает
   частоту драйвинга через ffprobe, и ось «драйвинг снят не ниже 30» стоит на
   его ответе. Второй прибор, отвечающий на тот же вопрос другим кодом, — это
   второй способ узнать известное, то есть дефект по определению. Поэтому здесь
   не заведён свой измеритель частоты: `fps_prober` ниже — совместимая замена
   для `_ffprobe_fps`, а не его дубль (см. отчёт смены, предложение).
2. **Лицензия (Ц5), и это НЕ формальность.** Сборка ffmpeg на этой машине
   `--enable-gpl` (проверено `ffmpeg -version`), и `/usr/share/doc/ffmpeg/
   copyright` говорит прямо: «For building the default Debian packages some of
   the GPL licensed files are used, so the resulting binaries are licensed under
   GPL v2+». Мы зовём его ОТДЕЛЬНЫМ ПРОЦЕССОМ через argv — не линкуемся, не
   импортируем, не встраиваем. Наш код от этого производным произведением не
   становится, и раскодированные кадры тоже: данные не производны от
   раскодировщика. Что это значит практически: (а) линковать эти библиотеки
   нам нельзя; (б) если мы когда-нибудь начнём отгружать ОБРАЗ МАШИНЫ с этим
   ffmpeg внутри — мы распространяем GPL-бинарник и обязаны предложить
   исходники (для дистрибутивной сборки это закрывается ссылкой на дистрибутив).
   `non-commercial` и `research-only` здесь НЕТ ни у одного из трёх — в отличие
   от `identity_arcface` (buffalo_l), который помечен в `fork_identity.LICENSES`.
   ЗАМЕРЕНО: у `cv2` внутри пакета лежат `libavcodec`, `libavformat`,
   `libavfilter` (каталог `opencv_python*.libs/`), а `cv2/LICENSE-3RD-PARTY.txt`
   говорит «FFmpeg is redistributed within all opencv-python packages» под
   LGPL-2.1. То есть выбор cv2 не убирает FFmpeg из поставки, а ЗАТАСКИВАЕТ его
   ВНУТРЬ нашего venv — вместе с обязательствами LGPL по замене библиотеки.
   Граница процесса дешевле границы линковки.
3. **imageio отпадает замером, а не мнением:** `import imageio_ffmpeg` в этой
   среде падает `ModuleNotFoundError`. Читать видео он будет только после
   ДОКАЧКИ бинарника из сети — то есть после похода наружу, которого тест
   позволить не может (Т4), а на арендованной машине он ещё и упрётся в
   политику прокси (Ц3).
4. **Отказ инструмента читается однозначно.** `ffprobe` на битом файле выдаёт
   код возврата 1 и текст «moov atom not found» — это «файл не годен». Отсутствие
   бинарника — это «спросить нечем». Два РАЗНЫХ исхода, и они не сливаются.
   У cv2 оба случая выглядят одинаково: `isOpened() == False`, `CAP_PROP_FPS`
   равен -1. ЗАМЕРЕНО на фикстурах: свести их к одному значило бы вернуть
   «видео плохое» там, где на машине просто нет кодека.

**ЧТО НЕ ЯВЛЯЕТСЯ ОСНОВАНИЕМ, хотя напрашивалось.** «cv2 врёт про число
кадров» — НЕ подтвердилось: на всех шести годных фикстурах
`CAP_PROP_FRAME_COUNT` совпал с числом реально прочитанных кадров (1, 60, 60,
72, 90, 320). Довод снят как неизмеренный; выбор держится на четырёх выше.

**ВТОРОГО БЭКЕНДА ЗДЕСЬ НЕТ НАМЕРЕННО.** Запасной путь через cv2 добавил бы
второй ответ на вопрос «сколько в файле кадров», и расходиться они начали бы
молча — ровно та форма, из-за которой в проекте уже разъезжались числа.
Отсутствие ffmpeg на арендованной машине даёт исход «не смогли проверить», а не
падение и не тихий перескок на другой прибор.

---

ЧАСТОТА — ГЛАВНАЯ ЛОВУШКА, И ОНА ПРОДУКТОВАЯ, А НЕ ТЕХНИЧЕСКАЯ.

Наш выход 30 к/с (`fork_comfy.WRAP_FPS`, ВЫБРАНО владельцем: 24 забракованы за
реализм). Драйвинг оператора может быть 24, 25, 29.97 или 60. Пересэмплирование
меняет ЧИСЛО КАДРОВ, а число кадров — это ДЛИНА РОЛИКА и ЧИСЛО ОКОН сэмплера:
60 кадров при 30 к/с — две секунды, те же 60 кадров, приведённые к 24, — 48
кадров и 1.6 секунды. Оператор закажет 10 секунд и получит другую длину, и
искать он это будет в промпте.

ПРИНЯТОЕ ПРАВИЛО, три ветки, и молчаливой среди них нет:

    fps не задан        БЕРЁМ КАК ЕСТЬ. Кадры выходят все до единого, число
                        кадров равно числу кадров в файле. Приведения нет.
    fps ниже исходного  РАЗРЕШЕНО, но только ЯВНЫМ аргументом, и в отчёте
                        печатается, сколько кадров пропало и какой стала длина.
    fps выше исходного  НЕ ГОДНО. Вверх не приводим никогда: интерполяции у нас
                        нет по решению владельца, а «выдуманный» кадр в
                        драйвинге — это выдуманное движение.

Последняя ветка — не наше изобретение, а та же норма, что уже записана в
`fork_template.FPS_OUT`: «интерполяции нет: исходники снимаются ВЫШЕ и
пересэмплируются вниз, кадры не выдумываются», и та же, по которой
`fork_template.check` бракует драйвинг ниже 30. Здесь она просто исполняется, а
не только проверяется.

29.97 ЭТО НЕ 30, и допуск сравнения выбран так, чтобы это было видно:
30000/1001 = 29.97003, разница с тридцатью 0.03 — втрое больше допуска
`FPS_TOLERANCE`. Значит запрос «привести к 30» на материале 29.97 честно
отвечает «не годно», а не подсовывает молча повтор кадра раз в 33 секунды.

СКОЛЬКО ДЛИТЬСЯ И СКОЛЬКО ОКОН — СЧИТАЕТСЯ НЕ ЗДЕСЬ. Число кадров под длину
даёт `fork_comfy.frames_for_seconds`, прижатие к шагу 4 — `fork_comfy.snap_frames`.
Этот модуль их ЗОВЁТ (`plan_for_seconds`), а не повторяет: своя арифметика
длины разошлась бы с обёрткой на первом же изменении шага.

---

ТРИ ИСХОДА (Р1) И ЧИСЛА РЯДОМ С ВЕРДИКТОМ (Р2). Каждый ответ несёт `expected`
(сколько кадров обещали метаданные), `written` (сколько файлов легло на диск),
`bytes` и `seconds`. НОЛЬ РАСКОДИРОВАННЫХ КАДРОВ — НЕ УСПЕХ: это отдельная
проверка, а не следствие того, что ffmpeg вернул ноль. Ровно на этой форме
дефект №6 и жил: пустой список кадров ехал дальше как «годно».

ПОРЯДОК ДЕТЕРМИНИРОВАННЫЙ. Имена `00000.png`, ширина поля пять знаков, номера
с нуля, сортировка строк совпадает с сортировкой чисел до 99999 кадров — а наш
потолок 305 (10 с, `fork_comfy.window_plan`). Возврат отдаёт уже отсортованный
список: сегодня в проекте уже находился дефект недетерминированного обхода
(`rglob` в `fork_stand`), и повторять его не будем.

«С нуля» — про ЭТОТ модуль (`-start_number 0` в `decode_argv`), а не про
`frame_name`: у неё начала нет вовсе, она форматирует поданное число, и второй
её вызывающий (`fork_splice.write_sequence`) нумерует с ЕДИНИЦЫ. Порядок от
этого не расходится — замерено, числа и негативный контроль в докстринге
`frame_name`; расходятся только имена, и номер из имени сегодня не читает
никто.

ИДЕМПОТЕНТНОСТЬ. Непустой каталог назначения — исход «не смогли проверить» и
отказ работать, а не тихая перезапись. Кадры предыдущего прогона (или чужой
смены) внешне неотличимы от наших, и раскодировав поверх 320 кадров 60 новых,
мы получили бы каталог из 260 чужих и 60 своих, отсортованный и правдоподобный.
Перезапись возможна только явным `overwrite=True`.

ТОЧКИ ВНЕДРЕНИЯ (Т4). `read_probe` и `run_decode` — единственные места, где
модуль трогает внешний мир; тест подменяет их целиком, как это сделано с
`fork_stand.read_smi`. Ни один тест этого модуля не зависит от того, стоит ли
ffmpeg на машине, и ни один не ходит в сеть.

---

НЕПРОВЕРЕНО (Ц4), наверх:

* **на настоящем драйвинге оператора не гонялось.** Все прогоны сняты на
  видео, порождённых `ffmpeg -f lavfi -i testsrc` этой же сменой: 64x64,
  h264/yuv420p. Ни одного файла с телефона или камеры через модуль не прошло;
* **звук не извлекается и не проверяется по существу** — `probe` только
  сообщает, есть ли аудиодорожка. Что в ней, модуль не знает;
* **поворот (`rotate`/`displaymatrix`) не разбирается.** Вертикальное видео с
  телефона, записанное как горизонтальное с флагом поворота, ffmpeg повернёт
  сам при раскодировании, а `probe` вернёт ГЕОМЕТРИЮ ПОТОКА, то есть до
  поворота. Расхождение возможно, замера нет;
* **переменная частота кадров (VFR) не разбирается.** `avg_frame_rate` и
  `r_frame_rate` у VFR расходятся; здесь берётся `avg_frame_rate`, и на VFR
  число кадров по метаданным будет оценкой. Такой файл модуль не забракует —
  он покажет расхождение `expected` и `written`, то есть скажет «не смогли
  подтвердить полноту»;
* **на арендованной машине не исполнялось.**
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from . import fork_comfy
from .fork_identity import FAIL, PASS, UNMEASURED

#: Имена бинарников. Отдельными константами, а не литералами в argv: тест
#: подменяет их, чтобы проверить ветку «инструмента нет», не удаляя ffmpeg.
FFPROBE_BIN = "ffprobe"
FFMPEG_BIN = "ffmpeg"

#: ВЫБРАНО (кем: эта смена; из чего): 20 с на ОПРОС метаданных — ровно столько
#: же стоит в `fork_template._ffprobe_fps`, и разъезд двух таймаутов на один и
#: тот же вызов был бы вторым источником истины.
PROBE_TIMEOUT_S = 20

#: ВЫБРАНО: 600 с на РАСКОДИРОВАНИЕ. Порядок оценивается так: 305 кадров
#: 480x832 — это секунды на любой машине, но раскодировщик может упереться в
#: сетевой диск. Таймаут здесь — предохранитель от зависания, а не норматив;
#: превышение даёт «не смогли», а не «не годно».
DECODE_TIMEOUT_S = 600

#: ВЫБРАНО: пять знаков в имени кадра. 99999 кадров против нашего потолка 305
#: (`fork_comfy.window_plan` на 10 с) — запас в 300 раз. Меньше пяти опасно:
#: при переполнении ffmpeg НЕ падает, а печатает шестизначное имя, и сортировка
#: строк перестаёт совпадать с сортировкой номеров ровно на этом кадре.
NAME_DIGITS = 5

#: Расширение кадров. PNG, а не JPEG: кадры идут в приборы (`fork_leak`,
#: `fork_seam`), которые меряют РАЗНИЦУ ПИКСЕЛЕЙ, и шум сжатия у оси протечки
#: уже однажды оказался в шестнадцать раз выше среднего сигнала.
FRAME_SUFFIX = ".png"

#: ВЫБРАНО: расхождение с метаданными в 1 кадр считается округлением, в 2 —
#: потерей. Основание: число кадров у контейнера без `nb_frames` считается как
#: длительность на частоту, и округление до целого кадра даёт ровно единицу.
#: Больше единицы объяснить округлением уже нельзя, и такой прогон отвечает
#: «не смогли подтвердить полноту», а не «годно».
FRAME_COUNT_TOLERANCE = 1

#: ВЫБРАНО/РАСЧЁТ: допуск сравнения частот. 30000/1001 = 29.97003 отличается от
#: 30 на 0.02997 — то есть допуск обязан быть МЕНЬШЕ трёх сотых, иначе NTSC-ное
#: 29.97 сойдёт за наши 30 и приведение «вверх» пройдёт молча. Ноль тоже нельзя:
#: частота приходит из деления целых, и 30/1 в double точен, а 25/1 после
#: пересчёта из другого поля — не всегда.
FPS_TOLERANCE = 0.01

#: Слова режимов частоты. Строками, потому что они едут в отчёт оператору.
AS_IS, DROP, REFUSE = "как есть", "прорежаем", "отказ"

#: Коды возврата точки входа. Те же, что у `fork_stand`: 0 годно, 1 не годно,
#: 2 не смогли. Сведение двойки в ноль означало бы, что отсутствие ffmpeg
#: читается как успех.
EXIT_BY_OUTCOME = {PASS: 0, FAIL: 1, UNMEASURED: 2}


# --------------------------------------------------------------------------
# Точки внедрения: ЕДИНСТВЕННОЕ место, где модуль трогает внешний мир (Т4).
# --------------------------------------------------------------------------

def read_probe(path) -> dict:
    """Спросить у ffprobe метаданные. ТОЧКА ВНЕДРЕНИЯ: тест подменяет целиком.

    Возвращает `{"ran": bool, "code": int|None, "out": str, "err": str,
    "why": str}`.

    `ran is False` означает РОВНО ОДНО: спросить нечем — бинарника нет, или он
    не запустился. Что файл плохой, отсюда НЕ СЛЕДУЕТ НИКОГДА. Именно ради
    этого различия здесь не `text: str|None`, как у `fork_stand.read_smi`:
    там ненулевой код возврата тоже означал «спросить нечем», а тут он означает
    «спросили, и ответ — файл не читается», и это разные исходы (Р1).
    """
    if shutil.which(FFPROBE_BIN) is None:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": (f"{FFPROBE_BIN} не найден: спросить нечем. Это НЕ "
                        f"«файл плохой» — утилита ставится пакетом ffmpeg")}
    try:
        raw = subprocess.run(
            [FFPROBE_BIN, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": f"{FFPROBE_BIN} не отработал: {str(exc)[:120]}"}
    return {"ran": True, "code": raw.returncode, "out": raw.stdout or "",
            "err": raw.stderr or "", "why": ""}


def run_decode(argv) -> dict:
    """Раскодировать. ТОЧКА ВНЕДРЕНИЯ: тест подменяет целиком.

    Возвращает ту же форму, что `read_probe`. Аргументы приходят готовым
    списком из `decode_argv`, чтобы состав команды можно было проверить
    ОТДЕЛЬНО от её запуска: команда — тоже решение, и она обязана краснеть
    в тесте, а не только в прогоне (Т5).
    """
    if shutil.which(FFMPEG_BIN) is None:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": (f"{FFMPEG_BIN} не найден: раскодировать нечем. Это НЕ "
                        f"«видео плохое»")}
    try:
        raw = subprocess.run(argv, capture_output=True, text=True,
                             timeout=DECODE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": (f"{FFMPEG_BIN} не уложился в {DECODE_TIMEOUT_S} с и "
                        f"убит: раскодировано неизвестно сколько")}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": f"{FFMPEG_BIN} не отработал: {str(exc)[:120]}"}
    return {"ran": True, "code": raw.returncode, "out": raw.stdout or "",
            "err": raw.stderr or "", "why": ""}


# --------------------------------------------------------------------------
# Чистые функции: развилки вынесены из точек входа, чтобы их можно было
# покрасить тестом на литералах (Т5).
# --------------------------------------------------------------------------

def frame_name(index: int) -> str:
    """Имя кадра. Ширина поля — константа, а не литерал в двух местах.

    НАЧАЛА НУМЕРАЦИИ ЗДЕСЬ НЕТ, и это решение, а не упущение: функция
    форматирует ТОТ номер, который ей дали. Начало выбирает вызывающий, и
    вызывающих сейчас двое:

        `decode_argv` (этот модуль)  `-start_number 0` -> 00000.png ...
        `fork_splice.write_sequence` `frame_name(k + 1)` -> 00001.png ...

    ЗАМЕРЕНО (чем: `sorted(glob('*.png'))` на настоящих файлах; на чём: N =
    9, 10, 99, 100, 362, 999, 1000 — переходы разрядности и длина боевого
    ролика): ПОРЯДОК у обеих раскладок ОДИН И ТОТ ЖЕ, расхождений 0 из 7.
    Причина — дополнение нулями до `NAME_DIGITS`: при фиксированной ширине
    сортировка строк совпадает с сортировкой чисел при любом начале.
    Негативный контроль (И5): без дополнения нулями порядок ломается на 5
    раскладках из 6 — то есть замер умеет увидеть поломку, а не молчит всегда.
    Сторожа обоих утверждений — `FrameNames` в тестах этого модуля.

    Расходятся при этом ИМЕНА: первый кадр — `00000.png` у раскодировщика и
    `00001.png` у склейки, и потребитель, который однажды начнёт читать НОМЕР
    из имени (а не брать порядок), получит сдвиг на единицу. Сегодня таких
    потребителей нет: ни один модуль не разбирает номер из имени кадра.
    """
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"номер кадра {index!r}: ожидалось целое от нуля")
    return f"{index:0{NAME_DIGITS}d}{FRAME_SUFFIX}"


def _ratio(raw) -> float | None:
    """`30000/1001` -> 29.97003. Кривое или нулевое — `None`, а не догадка."""
    if raw is None:
        return None
    try:
        num, _, den = str(raw).partition("/")
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def parse_probe(text: str) -> dict:
    """Разбор JSON от ffprobe в наши поля. Чистая функция, тест — на литерале.

    Отдельно от `probe` намеренно: тут разбирается ФОРМАТ ОТВЕТА, и проверять
    его надо на записанном ответе настоящего ffprobe, а не на живом вызове.

    `frames_from` говорит, ОТКУДА взялось число кадров, и это не украшение:
    `nb_frames` контейнер иногда не пишет вовсе, и тогда число — оценка
    «длительность на частоту». Оценку и замер здесь нельзя перепутать.
    """
    try:
        data = json.loads(text or "")
    except (ValueError, TypeError):
        return {"ok": False, "why": f"ответ ffprobe не разобрался как JSON: "
                                    f"{(text or '')[:120]!r}"}
    if not isinstance(data, dict):
        return {"ok": False, "why": f"ждали объект, пришло {type(data).__name__}"}
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = any(s.get("codec_type") == "audio" for s in streams)
    if video is None:
        return {"ok": False, "audio": audio,
                "why": (f"видеопотока в файле нет (потоков всего "
                        f"{len(streams)}, звуковых {'есть' if audio else 'нет'})")}
    # `avg_frame_rate` — средняя по файлу, `r_frame_rate` — «базовая» решётка
    # контейнера. На CFR они совпадают; на VFR расходятся, и врёт тогда именно
    # вторая, потому что она описывает решётку тайминга, а не поток кадров.
    fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate"))
    try:
        seconds = float(video.get("duration")
                        or (data.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        seconds = None
    nb = video.get("nb_frames")
    frames = frames_from = None
    try:
        if nb is not None and int(nb) > 0:
            frames, frames_from = int(nb), "nb_frames"
    except (TypeError, ValueError):
        frames = None
    if frames is None and fps and seconds:
        frames, frames_from = int(round(seconds * fps)), "длительность x частота"
    return {
        "ok": True, "why": "", "fps": fps, "frames": frames,
        "frames_from": frames_from, "seconds": seconds,
        "width": video.get("width"), "height": video.get("height"),
        "audio": audio, "codec": video.get("codec_name"),
    }


def fps_plan(source_fps, *, want=None) -> dict:
    """Что делаем с частотой. Три ветки, и молчаливого приведения среди них нет.

    Возвращает `{"outcome", "mode", "fps", "note"}`. Развилка вынесена сюда из
    `frames` нарочно (Т5): продуктовое решение обязано проверяться отдельно от
    того, стоит ли на машине раскодировщик.
    """
    if source_fps is None:
        return {"outcome": UNMEASURED, "mode": REFUSE, "fps": None,
                "note": ("частота исходника не снята — решать про приведение "
                         "не из чего. Это НЕ «берём как есть»: как есть — это "
                         "тоже решение, и оно требует знать, что есть")}
    if want is None:
        return {"outcome": PASS, "mode": AS_IS, "fps": source_fps,
                "note": (f"частота исходника {source_fps:g} к/с, кадры берутся "
                         f"ВСЕ, приведения нет. Число кадров равно числу "
                         f"кадров в файле")}
    # `math.isfinite` здесь не украшение: NaN проваливается СКВОЗЬ все три
    # сравнения ниже (`nan <= 0`, `abs(nan-x) <= t`, `nan > x` — все False) и
    # доезжает до ветки прорежения, откуда уходит в команду как `-vf fps=nan`.
    # Найдено падающим тестом до правки (И2), а не рассуждением.
    if (not isinstance(want, (int, float)) or isinstance(want, bool)
            or not math.isfinite(want) or want <= 0):
        return {"outcome": FAIL, "mode": REFUSE, "fps": None,
                "note": f"частота {want!r}: ожидалось положительное конечное число"}
    if abs(want - source_fps) <= FPS_TOLERANCE:
        return {"outcome": PASS, "mode": AS_IS, "fps": source_fps,
                "note": (f"запрошено {want:g} к/с при исходных {source_fps:g} — "
                         f"это одно и то же в пределах допуска "
                         f"{FPS_TOLERANCE}, ничего не трогаем")}
    if want > source_fps:
        return {"outcome": FAIL, "mode": REFUSE, "fps": None,
                "note": (f"запрошено {want:g} к/с при исходных {source_fps:g}: "
                         f"ВВЕРХ НЕ ПРИВОДИМ. Интерполяции нет по решению "
                         f"владельца ({want:g} - {source_fps:g} = "
                         f"{want - source_fps:g} к/с пришлось бы выдумать), а "
                         f"выдуманный кадр в драйвинге — это выдуманное "
                         f"движение. Снимать драйвинг не ниже "
                         f"{fork_comfy.WRAP_FPS} к/с — требование к съёмке")}
    return {"outcome": PASS, "mode": DROP, "fps": float(want),
            "note": (f"прорежаем {source_fps:g} -> {want:g} к/с. ДЛИНА В "
                     f"КАДРАХ МЕНЯЕТСЯ: на секунду выйдет {want:g} кадров "
                     f"вместо {source_fps:g}, и число окон сэмплера считается "
                     f"уже по новому числу")}


def expected_frames(source_frames, *, source_fps=None, out_fps=None,
                    limit=None) -> int | None:
    """Сколько кадров ОБЯЗАНО лечь на диск. `None` — если считать не из чего.

    Прорежение считается отношением частот, а не заново по длительности:
    длительность у контейнера бывает округлена до миллисекунд, и второй способ
    посчитать то же самое разошёлся бы с первым на границе (Е1).
    """
    if source_frames is None:
        return None
    n = int(source_frames)
    if out_fps is not None and source_fps and abs(out_fps - source_fps) > FPS_TOLERANCE:
        n = int(round(n * out_fps / source_fps))
    if limit is not None:
        n = min(n, int(limit))
    return max(n, 0)


def count_outcome(expected, written: int) -> dict:
    """Вердикт по числам кадров. Чистая функция — тест кормит её литералами.

    Три исхода (Р1), и первый из них тот самый, ради которого модуль написан:
    НОЛЬ КАДРОВ — НЕ УСПЕХ. Дефект №6 жил ровно на этой форме: пустой список
    ехал дальше и получал «годно» на дальних шагах.
    """
    if written < 0:
        raise ValueError(f"записано {written}: отрицательных кадров не бывает")
    if written == 0:
        return {"outcome": FAIL,
                "note": ("кадров записано 0 — это НЕ успех, а отсутствие "
                         "результата: судить и анимировать нечем")}
    if expected is None:
        return {"outcome": UNMEASURED,
                "note": (f"записано {written} кадров, но метаданные не сказали, "
                         f"сколько их в файле — подтвердить полноту нечем")}
    diff = abs(written - int(expected))
    if diff <= FRAME_COUNT_TOLERANCE:
        return {"outcome": PASS,
                "note": (f"ожидалось {expected}, записано {written} "
                         f"(расхождение {diff}, допуск {FRAME_COUNT_TOLERANCE} "
                         f"— это округление, а не потеря)")}
    return {"outcome": UNMEASURED,
            "note": (f"ожидалось {expected}, записано {written}: расхождение "
                     f"{diff} больше допуска {FRAME_COUNT_TOLERANCE}. "
                     f"Раскодировано что-то, но что именно — метаданные не "
                     f"подтверждают. Это НЕ «годно»")}


def decode_argv(video_path, out_dir, *, out_fps=None, limit=None) -> list:
    """Команда раскодирования. Собирается отдельно — состав команды это решение.

    Почему именно эти ключи:

        -nostdin              ffmpeg иначе читает stdin и может съесть ввод
                              вызывающей программы;
        -v error              нужен stderr, а не километр баннера;
        -fps_mode passthrough один PNG на один раскодированный кадр. Без него
                              ffmpeg вправе дублировать и выбрасывать кадры,
                              подгоняя их под свою решётку, — то есть менять
                              ЧИСЛО КАДРОВ молча;
        -start_number 0       нумерация с нуля; умолчание ffmpeg — единица,
                              и тогда `00000.png` не существует, а первый кадр
                              называется `00001.png`;
        %0Nd.png              сортируемые имена фиксированной ширины.
    """
    argv = [FFMPEG_BIN, "-nostdin", "-v", "error", "-i", str(video_path)]
    if out_fps is not None:
        argv += ["-vf", f"fps={out_fps:g}"]
    argv += ["-fps_mode", "passthrough", "-start_number", "0"]
    if limit is not None:
        argv += ["-frames:v", str(int(limit))]
    argv.append(str(Path(out_dir) / f"%0{NAME_DIGITS}d{FRAME_SUFFIX}"))
    return argv


# --------------------------------------------------------------------------
# Приборы
# --------------------------------------------------------------------------

def probe(video_path, *, prober=None) -> dict:
    """Метаданные видео. Три исхода, числа рядом с вердиктом.

    `prober` разрешается В ТЕЛЕ, а не в сигнатуре: умолчание, связанное на
    импорте, мутация константы уже не достаёт — эту форму на проекте выгребали.
    """
    prober = read_probe if prober is None else prober
    t = time.perf_counter()
    p = Path(video_path)
    # Дешёвое раньше дорогого (П2): отсутствие файла ловится за микросекунды,
    # и платить за запуск процесса ради этого ответа незачем.
    if not p.exists():
        return _probe_report(FAIL, f"файла нет: {p}", t)
    if p.is_dir():
        return _probe_report(
            FAIL, f"{p} — это КАТАЛОГ, а не видеофайл. Кадры в каталоге "
                  f"раскодировать не надо: их надо подавать как есть", t)
    size = p.stat().st_size
    if size == 0:
        return _probe_report(FAIL, f"{p}: файл пустой, 0 байт", t)

    raw = prober(p)
    if not raw.get("ran"):
        return _probe_report(UNMEASURED, raw.get("why") or "спросить нечем", t)
    if raw.get("code"):
        return _probe_report(
            FAIL, f"{FFPROBE_BIN} вернул {raw['code']}: "
                  f"{(raw.get('err') or '').strip()[:200] or 'без объяснения'}", t)
    parsed = parse_probe(raw.get("out") or "")
    if not parsed.get("ok"):
        return _probe_report(FAIL, parsed.get("why", "ответ не разобран"), t,
                             **({"audio": parsed["audio"]}
                                if "audio" in parsed else {}))
    rep = _probe_report(
        PASS,
        (f"{parsed['width']}x{parsed['height']}, {parsed['fps']:g} к/с, "
         f"кадров {parsed['frames']} (по «{parsed['frames_from']}»), "
         f"{parsed['seconds']:g} с, звук "
         f"{'есть' if parsed['audio'] else 'нет'}, кодек {parsed['codec']}")
        if parsed.get("fps") and parsed.get("seconds") is not None else
        (f"метаданные разобрались не полностью: частота {parsed.get('fps')}, "
         f"кадров {parsed.get('frames')}, длительность {parsed.get('seconds')}"),
        t, **{k: parsed[k] for k in
              ("fps", "frames", "frames_from", "seconds", "width", "height",
               "audio", "codec")})
    if rep["fps"] is None or rep["frames"] is None:
        # Разобралось, но не всё: это «не смогли», а не «годно». Отдать PASS с
        # `fps=None` значило бы, что вызывающий примет отсутствие числа за
        # проверенное отсутствие проблемы.
        rep["outcome"] = UNMEASURED
    rep["bytes"] = size
    return rep


def _probe_report(outcome: str, note: str, t0: float, **extra) -> dict:
    rep = {"outcome": outcome, "note": note, "fps": None, "frames": None,
           "frames_from": None, "seconds": None, "width": None, "height": None,
           "audio": None, "codec": None, "bytes": None,
           "elapsed": round(time.perf_counter() - t0, 4)}
    # Обновление БЕЗ фильтра по `None`: `audio=False` — это ответ «звука нет»,
    # а не отсутствие ответа, и фильтр по ложности съел бы именно его.
    rep.update(extra)
    return rep


def fps_prober(path):
    """Частота исходника как одно число. Совместимая замена `_ffprobe_fps`.

    ЗАЧЕМ ИМЕННО ТАКАЯ ПОДПИСЬ. `fork_template.check` принимает `prober`
    параметром и зовёт его как `prober(path) -> float|None`. Эта функция —
    ровно такая, поэтому `fork_template` может перестать держать СВОЙ разбор
    ffprobe и начать звать этот (Е1: один вопрос — один ответчик), не меняя
    ничего, кроме одной строки умолчания. Правку в чужом файле эта смена не
    делает; предложение записано в отчёт.

    Отличие от `_ffprobe_fps` не в интерфейсе, а в честности внутри: там
    отсутствие ffprobe и битый файл дают одинаковый `None`, здесь оба случая
    сначала разведены `probe`, и только потом схлопнуты в `None` — потому что
    так требует ЧУЖОЙ интерфейс, а не потому, что они одинаковы.
    """
    rep = probe(path)
    return rep["fps"] if rep["outcome"] == PASS else None


def plan_for_seconds(seconds, *, fps=None) -> dict:
    """Сколько кадров драйвинга нужно под ролик такой длины.

    Арифметики здесь НЕТ: всё считает `fork_comfy.frames_for_seconds`, включая
    прижатие к шагу 4 (`snap_frames`). Функция существует ради одного —
    чтобы вызывающему не пришлось знать про два модуля сразу и чтобы соблазна
    посчитать «секунды на частоту» руками не возникло (Е1).
    """
    return fork_comfy.frames_for_seconds(seconds, fps=fps)


def frames(video_path, out_dir, *, fps=None, limit=None, overwrite=False,
           prober=None, decoder=None) -> dict:
    """Раскодировать видео в PNG. Три исхода, числа рядом с вердиктом.

    `fps=None` — берём как есть (см. правило в шапке модуля). `limit` —
    сколько кадров взять с начала; больше в файле — остальное не читается.

    Возвращает словарь с полями `outcome`, `expected`, `written`, `bytes`,
    `elapsed`, `paths` (отсортованный список) и `note`. Ноль кадров — исход
    `не годно`, а не пустой успех.
    """
    prober = read_probe if prober is None else prober
    decoder = run_decode if decoder is None else decoder
    t = time.perf_counter()
    steps = []

    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            return _frames_report(FAIL, f"limit={limit!r}: ожидалось целое от 1",
                                  t, steps)

    # 1. Метаданные первыми (П2): они стоят миллисекунды и отвечают на вопрос,
    # ради которого иначе пришлось бы раскодировать всё видео.
    meta = probe(video_path, prober=prober)
    steps.append(("метаданные", meta["outcome"], meta["note"], meta["elapsed"]))
    if meta["outcome"] != PASS:
        return _frames_report(meta["outcome"], meta["note"], t, steps, meta=meta)

    # 2. Частота — продуктовое решение, и оно принимается ДО раскодирования:
    # отказ по частоте не должен стоить прогона по всему файлу.
    plan = fps_plan(meta["fps"], want=fps)
    steps.append(("частота", plan["outcome"], plan["note"], 0.0))
    if plan["outcome"] != PASS:
        return _frames_report(plan["outcome"], plan["note"], t, steps, meta=meta,
                              plan=plan)

    # Ожидаемое число кадров считается ЗДЕСЬ, один раз, и едет во ВСЕ отчёты
    # ниже (Е1): раньше оно считалось только перед вердиктом, и отказы до
    # раскодирования печатали «ожидалось неизвестно» при полностью разобранных
    # метаданных — то есть отчёт был беднее того, что уже было известно.
    want = plan["fps"] if plan["mode"] == DROP else None
    expected = expected_frames(meta["frames"], source_fps=meta["fps"],
                               out_fps=want, limit=limit)

    # 3. Идемпотентность: чужие кадры не перетираются молча.
    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        return _frames_report(FAIL, f"{out} — не каталог", t, steps, meta=meta,
                              plan=plan, expected=expected)
    already = sorted(out.glob(f"*{FRAME_SUFFIX}")) if out.is_dir() else []
    # Смотрели — значит числа известны, и в отчёт едут они, а не умолчание
    # `не осматривали` (Е2: отчёт о том, что ИСПОЛНИЛОСЬ). Ноль здесь — это
    # ответ «каталог пуст», а не отсутствие ответа.
    present = len(already)
    present_bytes = sum(f.stat().st_size for f in already)
    if already and not overwrite:
        note = (f"в {out} уже лежит кадров: {present} (первый "
                f"{already[0].name}, последний {already[-1].name}, байт "
                f"{present_bytes}). Молча поверх не пишем: раскодировав 60 "
                f"кадров поверх 320, мы получили бы каталог из 260 чужих и 60 "
                f"своих — отсортованный и правдоподобный. Мы НЕ ПИСАЛИ ни "
                f"одного кадра: эти {present} — чужие. Задайте overwrite=True "
                f"или другой каталог")
        steps.append(("каталог", UNMEASURED, note, 0.0))
        return _frames_report(UNMEASURED, note, t, steps, meta=meta, plan=plan,
                              expected=expected, present=present,
                              present_bytes=present_bytes)
    if already and overwrite:
        for f in already:
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)

    # 4. Раскодирование.
    argv = decode_argv(video_path, out, out_fps=want, limit=limit)
    t_dec = time.perf_counter()
    got = decoder(argv)
    dec_elapsed = round(time.perf_counter() - t_dec, 4)
    written_paths = sorted(out.glob(f"*{FRAME_SUFFIX}"))
    written = len(written_paths)
    size = sum(p.stat().st_size for p in written_paths)

    if not got.get("ran"):
        note = (f"{got.get('why') or 'раскодировать нечем'}. Успело лечь "
                f"кадров: {written}, ожидалось {expected}")
        steps.append(("раскодирование", UNMEASURED, note, dec_elapsed))
        return _frames_report(UNMEASURED, note, t, steps, meta=meta, plan=plan,
                              expected=expected, written=written, nbytes=size,
                              paths=written_paths, present=present,
                              present_bytes=present_bytes)
    if got.get("code"):
        note = (f"{FFMPEG_BIN} вернул {got['code']}: "
                f"{(got.get('err') or '').strip()[:200] or 'без объяснения'}. "
                f"Кадров записано {written}, ожидалось {expected}")
        steps.append(("раскодирование", FAIL, note, dec_elapsed))
        return _frames_report(FAIL, note, t, steps, meta=meta, plan=plan,
                              expected=expected, written=written, nbytes=size,
                              paths=written_paths, present=present,
                              present_bytes=present_bytes)
    steps.append(("раскодирование", PASS,
                  f"{FFMPEG_BIN} отработал, код 0", dec_elapsed))

    # 5. Вердикт по числам, а не по коду возврата инструмента. Именно здесь
    # ноль кадров перестаёт быть успехом.
    verdict = count_outcome(expected, written)
    steps.append(("кадры", verdict["outcome"], verdict["note"], 0.0))
    return _frames_report(verdict["outcome"], verdict["note"], t, steps,
                          meta=meta, plan=plan, expected=expected,
                          written=written, nbytes=size, paths=written_paths,
                          present=present, present_bytes=present_bytes)


#: Как отчёт называет каталог назначения. Три состояния, и третье не
#: сворачивается ни в одно из первых двух (Р1): `present=None` означает «мы
#: туда НЕ СМОТРЕЛИ» (отказ случился раньше осмотра), `present=0` — «смотрели,
#: пусто», `present>0` — «смотрели, лежит столько-то». Раньше все три
#: печатались одинаково — «записано 0, байт 0», — и повторный прогон поверх 60
#: чужих кадров читался как пустой каталог.
DIR_UNSEEN = "каталог назначения не осматривали"
DIR_EMPTY = "каталог назначения был пуст"


def _dir_fact(present, present_bytes) -> str:
    """Фраза про каталог назначения. Выведена из того, что ИСПОЛНИЛОСЬ (Е2)."""
    if present is None:
        return DIR_UNSEEN
    if present == 0:
        return DIR_EMPTY
    return (f"до нас в каталоге лежало кадров {present}, "
            f"байт {0 if present_bytes is None else present_bytes}")


def _frames_report(outcome: str, note: str, t0: float, steps, *, meta=None,
                   plan=None, expected=None, written=0, nbytes=0,
                   paths=None, present=None, present_bytes=None) -> dict:
    """Один отчёт на все исходы.

    `written` — сколько кадров записали МЫ этим прогоном, и ноль здесь значит
    ровно это, а не «в каталоге пусто»: про каталог отвечает `present`
    (см. `_dir_fact`), и итоговая строка называет оба числа раздельно (Е3).
    """
    elapsed = round(time.perf_counter() - t0, 4)
    paths = list(paths or [])
    return {
        "outcome": outcome,
        "expected": expected, "written": written, "bytes": nbytes,
        "present": present, "present_bytes": present_bytes,
        "elapsed": elapsed,
        "fps_in": (meta or {}).get("fps"), "fps_out": (plan or {}).get("fps"),
        "mode": (plan or {}).get("mode"),
        "paths": paths,
        "steps": [{"step": s, "outcome": o, "note": n, "seconds": round(e, 4)}
                  for s, o, n, e in steps],
        "note": (f"{outcome}: {note}. Ожидалось кадров "
                 f"{'неизвестно' if expected is None else expected}, записано "
                 f"нами {written}, байт {nbytes}, "
                 f"{_dir_fact(present, present_bytes)}, за {elapsed} с"),
    }


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    """`python3 -m ball_reel.fork_video probe|frames ...`.

    Развилка тонкая нарочно (Т5): всё, что можно проверить, лежит в функциях
    выше, а здесь только печать и код возврата.
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="fork_video", description="раскодировщик видео форка")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("probe", help="метаданные видео")
    p1.add_argument("video")
    p2 = sub.add_parser("frames", help="раскодировать в PNG")
    p2.add_argument("video")
    p2.add_argument("out_dir")
    p2.add_argument("--fps", type=float, default=None,
                    help="привести частоту ВНИЗ; без него берётся как есть")
    p2.add_argument("--limit", type=int, default=None)
    p2.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "probe":
        rep = probe(args.video)
        print(f"{rep['outcome']:20s} {rep['note']}")
    else:
        rep = frames(args.video, args.out_dir, fps=args.fps, limit=args.limit,
                     overwrite=args.overwrite)
        for s in rep["steps"]:
            print(f"{s['outcome']:20s} {s['step']:15s} {s['seconds']:7.3f} с  "
                  f"{s['note']}")
        print(rep["note"])
    return EXIT_BY_OUTCOME[rep["outcome"]]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
