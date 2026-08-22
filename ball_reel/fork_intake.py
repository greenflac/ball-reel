"""Приём входов сквозного стенда: драйвинг, фотография клиента, стилевой референс.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. Прогон стенда стоит денег и часа времени, а все три
входа отказывают ТИХО: видео с дырками во временных метках распаковывается
«успешно», просто часть кадров в нём — подделанные дубли; сцена короче трёх
секунд уезжает в шлюз и возвращается ошибкой уже после оплаты; фотография с
двумя лицами даёт личность не того человека; нечитаемая карточка стиля даёт
промт по умолчанию, то есть измерение НЕ ТОГО стиля. Каждый из этих отказов
дешевле поймать до прогона, чем объяснить после (П2).

ЧЕГО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ. Он ничего не чинит и ничего не режет сам: он
выносит вердикт и отдаёт ГРАНИЦЫ и КОМАНДУ. Резать — работа `fork_video` и
`fork_splice`, и заводить здесь второй раскодировщик значило бы завести второй
способ узнать известное (Е1).

ТРИ ИСХОДА (Р1), и третий не сворачивается ни в первый, ни во второй:

    годно               прибор отработал, нарушений не нашёл
    не годно            прибор отработал, нарушение измерено
    не смогли проверить прибора нет, файла нет, ответ не разобрался

Рядом с каждым вердиктом — три числа (Р2): `checked` сколько проверено,
`violations` сколько нарушений, `unmeasured` сколько не смогли. Ноль нарушений
при нуле проверенных — не успех, и здесь это видно в самом отчёте.

МЯГКАЯ ОСЬ, И ОНА ЗДЕСЬ ОДНА. Сиротские кисти (см. `orphan_wrists`) вердикт НЕ
РОНЯЮТ: владелец посмотрел выход с 21% таких кадров и назвал кисти
правильными. Это ПОПРАВКА К ОЖИДАНИЮ по личности, а не критерий отказа, и
превращать её в отказ значит выбросить материал, который владелец принял.

ТОЧКИ ВНЕДРЕНИЯ (Т4). Внешнего мира модуль касается ровно в шести местах, и
каждое — параметр: `prober`/`decoder` (ffprobe/ffmpeg), `gray` (пиксели),
`pose_reader` (поза), `face_prober`/`faces_prober` (лицо), `card_reader`
(карточка стиля). Тесты идут без диска и без сети.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from . import fork_looper, fork_video
from .fork_identity import FAIL, PASS, UNMEASURED

# ---------------------------------------------------------------------------
# Планки. НИ ОДНА не скопирована: чужое значение импортируется (Е1)
# ---------------------------------------------------------------------------

#: Планка личности. ЧУЖАЯ, живёт в `fork_identity`, сюда приходит импортом.
#: Копия разъехалась бы молча: признак настоящего дубля — «изменил одно, второе
#: ОБЯЗАНО измениться», и это ровно тот случай.
from .fork_identity import SAME_PERSON_MAX  # noqa: E402

#: Планка монтажного реза. ЧУЖАЯ, живёт в `fork_looper` (а та берёт её у
#: `motion.JUMP_MAX`). Здесь она нужна только чтобы попасть в отчёт рядом с
#: числом найденных швов: разметку делает `fork_looper.cuts`, не мы.
from .fork_looper import CUT_JUMP  # noqa: E402

#: Планка видимости точки позы. ЧУЖАЯ, живёт в `pose`: ниже неё координата —
#: догадка детектора, а не наблюдение. Определение сиротской кисти стоит
#: ЦЕЛИКОМ на ней, поэтому копия здесь была бы дефектом.
from .pose import MIN_VISIBILITY  # noqa: E402

#: Планка размера лица. ЧУЖАЯ, живёт в `identity_arcface` — том самом приборе,
#: которым `fork_identity` меряет личность, и планка откалибрована на его
#: шкале. Импортируется оттуда напрямую, потому что `fork_identity` её не
#: переэкспортирует, а второе значение с тем же именем — это Е1 наизнанку.
from .identity_arcface import MIN_FACE_PX  # noqa: E402

#: КРИТЕРИЙ ПРИЁМА. ВЫБРАНО: решение владельца 22.08.2026 (кем: владелец; из
#: чего: отказ шлюза плюс три опровергнутых обхода). Сцена между монтажными
#: швами не короче трёх секунд.
#:
#: ОБОСНОВАНИЕ, И ОНО НЕ ТЕОРЕТИЧЕСКОЕ. Kling отвечает дословно
#: "Video duration can not less than 3s". Проверены и ОПРОВЕРГНУТЫ все три
#: обхода, каждый по одному разу и с наблюдаемым результатом:
#:   1. растяжка частоты — 15 кадров поданы как 5 к/с; вернулось 88 кадров
#:      ВЫДУМАННОГО движения, то есть модель дорисовала то, чего в драйвинге
#:      не было;
#:   2. добивка заморозкой — хвост доложен замороженным кадром; модель
#:      ОЖИВИЛА заморозку, и в выходе снова движение, которого не задавали;
#:   3. посценный рендер напрямую, мимо сборки — гейт шлюза не пускает.
#: Поэтому это планка ПРИЁМА ВХОДА, а не пожелание к монтажу: материал, не
#: проходящий её, дальше по конвейеру не чинится ничем.
MIN_SCENE_SECONDS = 3.0

#: Доля сиротских кистей, начиная с которой в отчёт идёт ПРЕДУПРЕЖДЕНИЕ (не
#: отказ). ВЫБРАНО 0.10 (кем: смена приёма 22.08; из чего: измерены ровно две
#: точки — 21% сирот дали ArcFace 0.2960 при 81/99 в баре, 0% дали 0.2430 при
#: 98/99, то есть вклад около 0.05 по личности на 21%. Линейно 10% — это
#: примерно 0.024, то есть ПОЛОВИНА собственного шума прибора (0.05 по
#: `fork_identity.UPSCALE_DRIFT_MAX`). Ниже 10% поправка тонет в шуме и
#: предупреждать не о чем).
#: НЕ ИЗМЕРЕНО в самой точке 10%: замеров между 0% и 21% не делали, и линейность
#: между двумя точками — ДОПУЩЕНИЕ, а не наблюдение.
ORPHAN_WRIST_WARN = 0.10

#: Частота, на которой ПРОВЕРЕНА команда вырезки окна (см. `window_argv`).
#: ИЗМЕРЕНО: `assets/driving_selfie.mp4`, `assets/driving_arms.mp4` — обе 30/1
#: по ffprobe; на этой частоте команда прогонялась и её выход принимался Wan.
WINDOW_FPS_PROVEN = 30.0

#: Сколько кадров ffmpeg вправе разойтись с ffprobe, прежде чем это дырки в
#: метках. ИЗМЕРЕНО: на `assets/driving_selfie.mp4` ffprobe -count_frames даёт
#: 305, ffmpeg без ключей — 307, ffmpeg с `-vsync 0` — ровно 305; на
#: `assets/driving_arms.mp4` и `assets/driving_yogaball.mp4` все три числа
#: совпадают (373 и 362). То есть у здорового файла расхождение РОВНО НОЛЬ, и
#: допуск здесь не нужен: единица допуска скрыла бы ровно тот дефект, ради
#: которого прибор написан.
FRAME_COUNT_EXACT = 0

#: Сколько людей должно быть на фотографии клиента. ВЫБРАНО 1 (кем: продукт; из
#: чего: личность меряется по САМОМУ КРУПНОМУ лицу — `identity_arcface.
#: face_detail` сортирует по площади. На фотографии с двумя людьми «самое
#: крупное» выбирает прибор, а не человек, и вся ось личности тогда меряет
#: неизвестно кого).
PHOTO_PEOPLE_EXPECTED = 1

#: Совет, который выдаётся при расхождении счётчиков. Одна строка на модуль
#: (Е1): она попадает и в вердикт, и в отчёт, и разъехаться им нельзя.
VSYNC_ADVICE = ("распаковывать только с `-vsync 0` (в новых ffmpeg — "
                "`-fps_mode passthrough`, ИЗМЕРЕНО: оба дают 305 на "
                "driving_selfie): без него ffmpeg ПОДДЕЛЫВАЕТ пропущенные "
                "кадры дублями, подгоняя поток под свою решётку")

#: Коды возврата по исходу. Взяты у `fork_looper` (Е1), а не назначены заново:
#: два разных отображения исхода в код в одном конвейере — это дефект, который
#: виден только на CI и только через полгода.
EXIT_BY_OUTCOME = fork_looper.EXIT_BY_OUTCOME


# ---------------------------------------------------------------------------
# Точки внедрения: ЕДИНСТВЕННЫЕ места, где модуль трогает внешний мир (Т4)
# ---------------------------------------------------------------------------

def read_count_frames(path) -> dict:
    """Спросить ffprobe ПОКАДРОВО. ТОЧКА ВНЕДРЕНИЯ: тест подменяет целиком.

    Именно `-count_frames`, а не `nb_frames` из заголовка: заголовок — это то,
    что контейнер О СЕБЕ ЗАЯВИЛ, а нам нужно, сколько кадров в потоке НА САМОМ
    ДЕЛЕ. `fork_video.parse_probe` читает заголовок и для своей задачи прав;
    здесь вопрос другой, поэтому и команда другая.

    Форма ответа — та же, что у `fork_video.read_probe`, и это не совпадение:
    вызывающий не должен помнить, чей ответ он держит.
    """
    if shutil.which(fork_video.FFPROBE_BIN) is None:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": (f"{fork_video.FFPROBE_BIN} не найден: спросить нечем. "
                        f"Это НЕ «файл плохой»")}
    try:
        raw = subprocess.run(
            [fork_video.FFPROBE_BIN, "-v", "error", "-count_frames",
             "-select_streams", "v:0", "-show_entries",
             "stream=nb_read_frames,avg_frame_rate,duration",
             "-of", "json", str(path)],
            capture_output=True, text=True,
            timeout=fork_video.DECODE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": f"{fork_video.FFPROBE_BIN} не отработал: {str(exc)[:120]}"}
    return {"ran": True, "code": raw.returncode, "out": raw.stdout or "",
            "err": raw.stderr or "", "why": ""}


def read_decoded_frames(path, *, vsync0: bool) -> dict:
    """Сколько кадров ВЫДАЛ БЫ распаковщик. ТОЧКА ВНЕДРЕНИЯ (Т4).

    Считается по-настоящему: поток гонится через мультиплексор `image2` — тот
    самый, которым распаковка и делается, — но кадры уменьшены до 16x16 и
    выброшены в `/dev/null`. Это НЕ упрощение замера, а его условие: с
    `-f null -` дублирования НЕ ПРОИСХОДИТ (ИЗМЕРЕНО: 305 и 305 на
    driving_selfie), потому что подделка кадров рождается у мультиплексора с
    фиксированной решёткой, а не у декодера. Мерить надо тот путь, который
    потом пойдёт в работу, иначе прибор зелёный там, где дефект есть.

    ПРОВЕРЕНО обоими способами: `ffmpeg -i selfie -an out/%05d.png` пишет на
    диск 307 файлов, этот замер на том же файле даёт 307; с `-vsync 0` оба
    дают 305.
    """
    if shutil.which(fork_video.FFMPEG_BIN) is None:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": (f"{fork_video.FFMPEG_BIN} не найден: посчитать "
                        f"распакованное нечем. Это НЕ «видео плохое»")}
    argv = [fork_video.FFMPEG_BIN, "-nostdin", "-v", "error", "-stats",
            "-i", str(path), "-an", "-vf", "scale=16:16"]
    if vsync0:
        argv += ["-vsync", "0"]
    argv += ["-f", "image2", "-update", "1", "-y", "/dev/null"]
    try:
        raw = subprocess.run(argv, capture_output=True, text=True,
                             timeout=fork_video.DECODE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": (f"{fork_video.FFMPEG_BIN} не уложился в "
                        f"{fork_video.DECODE_TIMEOUT_S} с: сколько кадров "
                        f"вышло — НЕИЗВЕСТНО")}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ran": False, "code": None, "out": "", "err": "",
                "why": f"{fork_video.FFMPEG_BIN} не отработал: {str(exc)[:120]}"}
    return {"ran": True, "code": raw.returncode, "out": raw.stdout or "",
            "err": raw.stderr or "", "why": ""}


def read_faces(path) -> dict:
    """Все лица кадра, а не только самое крупное. ТОЧКА ВНЕДРЕНИЯ (Т4).

    Возвращает `{"faces": [{"face_px", "det_score"}, ...] | None, "why": str}`,
    и три состояния здесь такие же, как у `fork_looper.read_pose`:

        faces == [],  why == ""     лиц НЕ НАЙДЕНО — это измерение;
        faces is None, why != ""    СПРОСИТЬ НЕЧЕМ — это НЕ измерение.

    ПОЧЕМУ НЕ `identity_arcface.face_detail`. Тот отдаёт ОДНО лицо, самое
    крупное, и на вопрос «сколько тут людей» ответить им нельзя: и один
    человек, и трое дают одинаковый ответ. Заводить ради счёта второй детектор
    было бы хуже (Е1, и вторая модель в памяти), поэтому берётся тот же
    анализатор того же модуля.
    """
    try:
        from . import identity_arcface

        faces = identity_arcface._analyzer().get(
            identity_arcface._read_bgr(path))
        out = []
        for f in faces:
            x0, y0, x1, y1 = (float(v) for v in f.bbox)
            out.append({"face_px": round(min(x1 - x0, y1 - y0)),
                        "det_score": round(float(f.det_score), 3)})
        out.sort(key=lambda d: d["face_px"], reverse=True)
        return {"faces": out, "why": ""}
    except Exception as exc:  # noqa: BLE001 — причин «спросить нечем» много
        # (нет insightface, нет весов, битый файл), и все они означают исход
        # «не смогли», а не «лиц нет».
        return {"faces": None, "why": f"{type(exc).__name__}: {str(exc)[:200]}"}


def read_style_card(path) -> dict:
    """Карточка стиля. ТОЧКА ВНЕДРЕНИЯ (Т4), и она же — единственный вход
    во ВНЕШНИЙ пакет `vertical-creative-eval`.

    Импортируется ФУНКЦИЯ `style_card`, а не модуль под именем `style`. Это не
    косметика: в форке действует решение владельца «`style.py` не используется
    ни исполнителем, ни прибором», и гейт
    `test_style_is_not_imported_anywhere_in_the_fork` ловит имя `style` в
    импортах. Здесь речь про ЧУЖОЙ `creative_eval.style` из другого пакета, но
    различить их сканер не может и не должен: запрет стоит на ИМЕНИ.

    Импорт отложен внутрь функции намеренно: без внешнего пакета модуль обязан
    импортироваться и тесты обязаны идти — иначе прибор нельзя проверить там,
    где его чинят.
    """
    try:
        from creative_eval.style import style_card  # noqa: PLC0415

        return {"card": style_card(str(path)), "why": ""}
    except Exception as exc:  # noqa: BLE001
        return {"card": None,
                "why": f"{type(exc).__name__}: {str(exc)[:200]}"}


# ---------------------------------------------------------------------------
# Чистые функции. Развилки вынесены из точек входа, чтобы их красил тест (Т5)
# ---------------------------------------------------------------------------

def tally(checked: int, violations: int, unmeasured: int) -> dict:
    """Три числа рядом с вердиктом (Р2), и вердикт, выведенный ИЗ НИХ.

    Порядок ветвей — решение, а не стиль: «не смогли» проверяется ПЕРВЫМ,
    потому что при нуле отработавших проверок ноль нарушений не значит ничего.
    """
    out = {"checked": int(checked), "violations": int(violations),
           "unmeasured": int(unmeasured)}
    if checked == 0:
        out["outcome"] = UNMEASURED
    elif violations > 0:
        out["outcome"] = FAIL
    elif unmeasured > 0:
        # Часть проверок отработала и нарушений не нашла, часть не отработала.
        # Это НЕ «годно»: непроверенное могло быть нарушением (Р1).
        out["outcome"] = UNMEASURED
    else:
        out["outcome"] = PASS
    return out


def parse_count_frames(text: str) -> dict:
    """Ответ `ffprobe -count_frames` -> число кадров. Тест — на литерале.

    Отдельно от вызова намеренно: разбор ФОРМАТА ответа надо проверять на
    записанном ответе настоящего ffprobe, а не на живом вызове (Т4).
    """
    import json

    try:
        data = json.loads(text or "")
    except (ValueError, TypeError):
        return {"ok": False, "frames": None, "fps": None, "seconds": None,
                "why": (f"ответ ffprobe не разобрался как JSON: "
                        f"{(text or '')[:120]!r}")}
    streams = (data or {}).get("streams") or []
    if not streams:
        return {"ok": False, "frames": None, "fps": None, "seconds": None,
                "why": "видеопотока в ответе нет: считать нечего"}
    s = streams[0]
    raw = s.get("nb_read_frames")
    try:
        frames = int(raw)
    except (TypeError, ValueError):
        return {"ok": False, "frames": None, "fps": None, "seconds": None,
                "why": (f"nb_read_frames = {raw!r}: ffprobe кадры НЕ СЧИТАЛ. "
                        f"Это не «кадров нет»")}
    if frames <= 0:
        return {"ok": False, "frames": None, "fps": None, "seconds": None,
                "why": f"ffprobe насчитал {frames} кадров: считать нечего"}
    # `_ratio` берётся у `fork_video` (Е1): разбор `30000/1001` уже написан и
    # промерен там, второй такой разошёлся бы молча.
    fps = fork_video._ratio(s.get("avg_frame_rate"))
    try:
        seconds = float(s.get("duration"))
    except (TypeError, ValueError):
        seconds = None
    return {"ok": True, "frames": frames, "fps": fps, "seconds": seconds,
            "why": ""}


def parse_decoded_frames(text: str) -> dict:
    """Строка `-stats` от ffmpeg -> сколько кадров он выдал. Тест — на литерале.

    ffmpeg печатает прогресс в stderr и ПЕРЕЗАПИСЫВАЕТ строку возвратом
    каретки, поэтому берётся ПОСЛЕДНЕЕ вхождение `frame=`, а не первое: первое
    почти всегда `frame= 0`, и прибор, читающий первое, всегда отвечает нулём.
    Проверено: на живом ответе первое вхождение — ровно `frame= 0`.
    """
    import re

    hits = re.findall(r"frame=\s*(\d+)", text or "")
    if not hits:
        return {"ok": False, "frames": None,
                "why": (f"в ответе ffmpeg нет ни одного `frame=`: сколько "
                        f"кадров вышло — НЕИЗВЕСТНО. Хвост: "
                        f"{(text or '')[-120:]!r}")}
    return {"ok": True, "frames": int(hits[-1]), "why": ""}


def timestamp_verdict(probed: int | None, plain: int | None,
                      fixed: int | None) -> dict:
    """Дырки во временных метках. Три исхода, и совет вместо догадки.

    `probed` — сколько кадров НАСЧИТАЛ ffprobe покадрово;
    `plain`  — сколько выдал ffmpeg БЕЗ ключей;
    `fixed`  — сколько выдал ffmpeg с `-vsync 0`.

    ЗАЧЕМ ТРЕТЬЕ ЧИСЛО. Без него расхождение читается как «файл битый», и
    материал выбрасывается. С ним видно, что файл цел, а подделывает кадры
    РАСПАКОВКА, и лечится это ключом, а не пересъёмкой. Это разные решения,
    и стоят они по-разному.
    """
    known = [v for v in (probed, plain, fixed) if v is not None]
    if len(known) < 3:
        missing = [n for n, v in (("ffprobe", probed), ("ffmpeg", plain),
                                  ("ffmpeg -vsync 0", fixed)) if v is None]
        return {**tally(0, 0, 1), "probed": probed, "plain": plain,
                "fixed": fixed, "gap": None, "advice": VSYNC_ADVICE,
                "note": (f"счётчики не сняты: {', '.join(missing)}. Это НЕ "
                         f"«кадры на месте» и НЕ «файл битый»")}
    gap = plain - probed
    if abs(gap) <= FRAME_COUNT_EXACT:
        return {**tally(1, 0, 0), "probed": probed, "plain": plain,
                "fixed": fixed, "gap": gap, "advice": "",
                "note": (f"ffprobe {probed}, ffmpeg {plain}, ffmpeg -vsync 0 "
                         f"{fixed}: расхождение {gap}, дырок во временных "
                         f"метках не видно")}
    healed = fixed == probed
    return {**tally(1, 1, 0), "probed": probed, "plain": plain, "fixed": fixed,
            "gap": gap, "advice": VSYNC_ADVICE,
            "note": (f"ffprobe {probed}, ffmpeg БЕЗ ключей {plain} "
                     f"(расхождение {gap:+d}), с -vsync 0 {fixed}. В файле "
                     f"пропущены кадры, и обычная распаковка их ПОДДЕЛЫВАЕТ "
                     f"дублями"
                     + (f"; {VSYNC_ADVICE}" if healed else
                        f"; и `-vsync 0` НЕ ЛЕЧИТ ({fixed} против {probed}) — "
                        f"материал в работу не брать"))}


def scenes(n_frames: int, cut_list) -> list:
    """Разбиение на сцены по швам. Чистая арифметика, тест — на литералах.

    `cut_list` — то, что отдаёт `fork_looper.cuts`: номера кадров, ПОСЛЕ
    которых стоит шов. Значит сцена — полуинтервал по номерам кадров, и
    границы здесь ВКЛЮЧИТЕЛЬНЫЕ с обеих сторон: дальше они уедут в `select`
    ffmpeg, где `between(n,A,B)` тоже включителен с обеих сторон. Одно
    соглашение на весь путь, потому что сдвиг на единицу здесь не виден
    глазом и стоит целого прогона.
    """
    if n_frames <= 0:
        return []
    marks = sorted({int(c) for c in (cut_list or []) if 0 <= int(c) < n_frames - 1})
    out, start = [], 0
    for k in marks:
        out.append({"start": start, "end": k, "frames": k - start + 1})
        start = k + 1
    out.append({"start": start, "end": n_frames - 1,
                "frames": n_frames - 1 - start + 1})
    return out


def scene_length_verdict(scene_list, fps: float | None,
                         *, min_seconds: float | None = None) -> dict:
    """Каждая ли сцена не короче планки. КРИТЕРИЙ ПРИЁМА, а не пожелание.

    `fps is None` — исход «не смогли»: без частоты кадры в секунды не
    переводятся, а подставить 30 «потому что обычно 30» значит выдать догадку
    за замер.
    """
    bar = MIN_SCENE_SECONDS if min_seconds is None else min_seconds
    if not scene_list:
        return {**tally(0, 0, 1), "bar_seconds": bar, "short": [],
                "seconds": [],
                "note": "сцен нет: разметка не снята, длину мерить не у чего"}
    if not fps or fps <= 0:
        return {**tally(0, 0, len(scene_list)), "bar_seconds": bar,
                "short": [], "seconds": [],
                "note": (f"частота не снята: {len(scene_list)} сцен есть, а "
                         f"перевести кадры в секунды нечем. Это НЕ «сцены "
                         f"короткие» и НЕ «сцены длинные»")}
    secs = [round(s["frames"] / fps, 3) for s in scene_list]
    short = [i for i, v in enumerate(secs) if v < bar]
    return {**tally(len(scene_list), len(short), 0), "bar_seconds": bar,
            "short": short, "seconds": secs,
            "note": (f"сцен {len(scene_list)}, планка {bar} с, короче планки "
                     f"{len(short)}"
                     + (f": номера {short[:10]}, длины {[secs[i] for i in short[:10]]}"
                        if short else
                        f"; самая короткая {min(secs)} с, самая длинная {max(secs)} с"))}


def is_orphan_wrist(points) -> bool | None:
    """Один кадр: есть ли на нём сиротская кисть. Определение — здесь и только.

    Сиротская кисть — запястье, которое ВИДНО (видимость не ниже
    `pose.MIN_VISIBILITY`, координаты внутри 0..1), при том что локоть ИЛИ
    плечо той же руки НЕ видно. Такой кадр даёт модели руку, растущую из
    ниоткуда.

    `None` значит «на этом кадре ответить нечем» (позы нет) — и это ТРЕТИЙ
    исход, не `False`.

    ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ, ЗАПИСАННЫЙ ЧИСЛОМ (И6). Ожидание смены было «на
    `assets/driving_arms.mp4` сирот 4%». Воспроизвести 4% не удалось НИ ОДНИМ
    прочтением определения; измерено на 373 кадрах, mediapipe pose_landmarker_lite:

        бар «видно» (видимость И координаты 0..1) на ВСЕХ ТРЁХ точках   6.4% (24)
        бар «видно» только на кисти, локоть и плечо — по видимости      0.5% (2)
        только локоть, плеча не спрашиваем, бар на всех трёх            6.4% (24)
        только плечо, локтя не спрашиваем, бар на всех трёх             0.0% (0)

    Разница между 6.4% и 0.5% — это 22 кадра, где локоть ВИДЕН детектором, но
    его координаты вынесены ЗА кадр. Здесь выбрано первое прочтение (бар на
    всех трёх точках): точка, спроецированная за границу кадра, не наблюдается
    так же, как и точка с низкой видимостью, и разделять эти два способа не
    увидеть точку было бы двумя определениями одного слова.
    ЧЕТЫРЁХ ПРОЦЕНТОВ НЕ ДАЁТ НИ ОДНО из четырёх, и это записано, чтобы
    следующая смена не переставляла те же ручки заново.
    """
    if not points:
        return None

    def seen(name):
        p = points.get(name)
        if p is None or len(p) < 3:
            return False
        x, y, vis = float(p[0]), float(p[1]), float(p[2])
        return vis >= MIN_VISIBILITY and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

    for side in ("l", "r"):
        if seen(f"{side}_wrist") and not (seen(f"{side}_elbow")
                                          and seen(f"{side}_shoulder")):
            return True
    return False


def orphan_verdict(share: float | None, checked: int, unmeasured: int) -> dict:
    """МЯГКАЯ ось: доля сирот и предупреждение. Вердикт НЕ РОНЯЕТ.

    Возвращаемый `outcome` относится к тому, УДАЛОСЬ ЛИ ПОСЧИТАТЬ долю, а не к
    тому, велика ли она: `не годно` здесь не бывает НИКОГДА, и это не
    упущение — см. заголовок модуля. Велика доля или нет, говорит поле `warn`.
    """
    if share is None or checked == 0:
        return {**tally(0, 0, max(1, unmeasured)), "share": None, "warn": False,
                "bar": ORPHAN_WRIST_WARN,
                "note": ("позу снять не удалось ни на одном кадре: доля сирот "
                         "НЕ ИЗМЕРЕНА. Это не «сирот нет»")}
    warn = share >= ORPHAN_WRIST_WARN
    base = tally(checked, 0, unmeasured)
    return {**base, "share": round(share, 4), "warn": warn,
            "bar": ORPHAN_WRIST_WARN,
            "note": (f"сиротских кистей {round(share * 100, 1)}% "
                     f"({checked} кадров с позой, {unmeasured} без)"
                     + (f". ПРЕДУПРЕЖДЕНИЕ: доля не ниже "
                        f"{round(ORPHAN_WRIST_WARN * 100)}%. Это ПОПРАВКА К "
                        f"ОЖИДАНИЮ по личности, а не отказ: ИЗМЕРЕНО, что 21% "
                        f"сирот дали ArcFace 0.2960 (81/99 в баре "
                        f"{SAME_PERSON_MAX}) против 0.2430 (98/99) при 0%, то "
                        f"есть около 0.05 по личности. Владелец посмотрел "
                        f"выход с 21% и назвал кисти правильными"
                        if warn else
                        f"; ниже планки предупреждения "
                        f"{round(ORPHAN_WRIST_WARN * 100)}%"))}


def face_size_verdict(sizes: list, no_face: int, unmeasured: int,
                      *, min_face_px: int | None = None) -> dict:
    """Хватает ли лицу пикселей, чтобы личность вообще было чем мерить.

    ИЗМЕРЕНО, зачем планка: на `driving_selfie` лицо 234..369 px и планка не
    мешает; на `driving_yogaball` лицо 87..96 px и планка выбрасывает 101 кадр
    из 101 — то есть по этому материалу личность НЕ ИЗМЕРИМА, и это надо знать
    ДО прогона, а не после.

    Кадр без лица — нарушение, а не «не смогли»: лицо в кадре либо есть, либо
    нет, и это ИЗМЕРЕНИЕ. «Не смогли» — только когда спросить было нечем.
    """
    bar = MIN_FACE_PX if min_face_px is None else min_face_px
    checked = len(sizes) + no_face
    if checked == 0:
        return {**tally(0, 0, max(1, unmeasured)), "bar_px": bar,
                "small": 0, "no_face": no_face, "min": None, "max": None,
                "note": ("лицо не спрашивали ни на одном кадре: размер НЕ "
                         "ИЗМЕРЕН. Это не «лица нет»")}
    small = [v for v in sizes if v < bar]
    # РЕШЕНИЕ ВЛАДЕЛЬЦА 22.08: эта ось — ПРЕДУПРЕЖДЕНИЕ, а не отказ.
    #
    # ПОЧЕМУ. Планка отвечает на вопрос «чем мы будем мерить личность», а не
    # «годится ли материал продукту». На трендовых танцевальных драйвингах
    # человек снят в полный рост, лицо мелкое, и жёсткий отказ выбрасывал
    # ЧЕТЫРЕ ГОДНЫХ ролика из четырёх (ИЗМЕРЕНО на b2..b5: сцена одна,
    # склеек ноль, длина 14.6..31.5 с — по всем остальным осям чисто).
    #
    # ЧТО ВЗАМЕН. Мелкое лицо означает, что ArcFace на выходе ответит «не
    # смогли измерить» — ровно как на `driving_yogaball` (87..96 px, планка
    # выбросила 101 кадр из 101). Значит личность на таком материале СУДИТ
    # ОПЕРАТОР ГЛАЗАМИ, как уже решено по фигуре, позе и стилю. Ось честно
    # печатает числа и предупреждает, но прогон не роняет.
    #
    # ТРЕТИЙ ИСХОД СОХРАНЁН: если лицо не спрашивали вовсе — по-прежнему «не
    # смогли» (ветка `checked == 0` выше), и это НЕ «лица нет».
    hurt = len(small) + no_face
    warn = (f"; ПРЕДУПРЕЖДЕНИЕ: {hurt} из {checked} кадров непригодны для "
            f"ArcFace — личность на выходе СУДИТ ОПЕРАТОР ГЛАЗАМИ, прибор "
            f"здесь не судья") if hurt else ""
    return {**tally(checked, 0, unmeasured), "bar_px": bar,
            "small": len(small), "no_face": no_face, "hurt": hurt,
            "min": min(sizes) if sizes else None,
            "max": max(sizes) if sizes else None,
            "note": (f"планка {bar}px: кадров {checked}, лицо найдено на "
                     f"{len(sizes)}, мельче планки {len(small)}, без лица "
                     f"{no_face}"
                     + (f"; размах {min(sizes)}..{max(sizes)} px" if sizes else "")
                     + (f", не спросили {unmeasured}" if unmeasured else "")
                     + warn)}


def window(scene_list, product_seconds: float, fps: float | None) -> dict:
    """Границы окна В НОМЕРАХ КАДРОВ. По времени резать НЕЛЬЗЯ.

    ПОЧЕМУ НЕ ПО ВРЕМЕНИ. Временные метки файла не совпадают с номерами кадров
    ровно на тех файлах, ради которых написан `timestamp_verdict`: там, где
    метки с дырками, `-ss 4.0` и «кадр 120» — разные кадры, и разъезд не виден
    глазом. Номер кадра однозначен всегда.

    ВЫБОР СЦЕНЫ: самая длинная (кем: смена приёма; из чего: у неё максимальный
    запас до планки, то есть меньше всего шансов, что окно упрётся в шов).
    ВЫБОР МЕСТА ВНУТРИ СЦЕНЫ: середина, равные поля с обеих сторон (из чего:
    кадры у самого шва — это кадры смены плана, и брать их в окно значит
    втащить в прогон именно тот стык, ради обхода которого сцены и размечали).

    Три исхода: `годно` — окно найдено; `не годно` — ни одна сцена не вмещает
    продуктовую длину; `не смогли` — нет частоты или нет разметки.
    """
    if not scene_list:
        return {**tally(0, 0, 1), "start": None, "end": None, "frames": None,
                "scene": None,
                "note": "разметки сцен нет: выбирать окно не из чего"}
    if not fps or fps <= 0:
        return {**tally(0, 0, len(scene_list)), "start": None, "end": None,
                "frames": None, "scene": None,
                "note": ("частота не снята: продуктовую длину в кадры "
                         "перевести нечем. Догадку 30 не подставляем")}
    need = int(round(product_seconds * fps))
    if need <= 0:
        return {**tally(0, 0, 1), "start": None, "end": None, "frames": None,
                "scene": None,
                "note": (f"продуктовая длина {product_seconds} с при {fps} к/с "
                         f"— это {need} кадров: резать нечего")}
    best = max(range(len(scene_list)), key=lambda i: scene_list[i]["frames"])
    have = scene_list[best]["frames"]
    if have < need:
        return {**tally(len(scene_list), 1, 0), "start": None, "end": None,
                "frames": None, "scene": None,
                "note": (f"нужно {need} кадров ({product_seconds} с при {fps} "
                         f"к/с), самая длинная сцена {have} кадров "
                         f"({round(have / fps, 3)} с): окно НЕ ВМЕЩАЕТСЯ")}
    pad = (have - need) // 2
    start = scene_list[best]["start"] + pad
    end = start + need - 1
    return {**tally(len(scene_list), 0, 0), "start": start, "end": end,
            "frames": need, "scene": best,
            "note": (f"окно {start}..{end} ({need} кадров, "
                     f"{round(need / fps, 3)} с) из сцены {best} "
                     f"({scene_list[best]['start']}..{scene_list[best]['end']}, "
                     f"{have} кадров), поля по {pad} кадров с каждой стороны")}


def window_argv(video_path, out_path, start: int, end: int,
                *, fps: float | None = None) -> list:
    """Команда вырезки окна. Собирается ОТДЕЛЬНО от запуска: состав команды —
    тоже решение, и он обязан краснеть в тесте, а не только в прогоне (Т5).

    `setpts=N/{fps}/TB` — не украшение. Без него у выхода остаётся `start_time`
    не равный нулю, и НА ЭТОМ Wan отвечал 422. Проверено: с `setpts` тот же
    файл принимается.

    Частота по умолчанию — `WINDOW_FPS_PROVEN`, и это ИЗМЕРЕННОЕ число обоих
    рабочих драйвингов, а не «обычно 30».
    """
    if not isinstance(start, int) or not isinstance(end, int) or start < 0:
        raise ValueError(f"границы окна {start!r}..{end!r}: ждали целые от нуля")
    if end < start:
        raise ValueError(f"границы окна {start}..{end}: конец раньше начала")
    rate = WINDOW_FPS_PROVEN if fps is None else float(fps)
    if rate <= 0:
        raise ValueError(f"частота {fps!r}: ждали положительное число")
    return [fork_video.FFMPEG_BIN, "-v", "error", "-y", "-i", str(video_path),
            "-vf", f"select='between(n\\,{start}\\,{end})',setpts=N/{rate:g}/TB",
            "-an", str(out_path)]


# ---------------------------------------------------------------------------
# Приборы: три входа
# ---------------------------------------------------------------------------

def driving_intake(video_path, frame_paths=None, *, product_seconds=None,
                   prober=None, decoder=None, gray=None, pose_reader=None,
                   face_prober=None) -> dict:
    """ПРИЁМ ДРАЙВИНГА: пять осей, из них четыре жёсткие и одна мягкая.

    `frame_paths` — уже распакованные кадры. Их распаковка НЕ ЗДЕСЬ: это
    работа `fork_video.frames`, и второй распаковщик в проекте был бы вторым
    способом узнать известное (Е1). Без кадров оси 2-5 честно отвечают «не
    смогли», а ось 1 (метки) работает: она смотрит на ФАЙЛ.

    Все точки внедрения разрешаются В ТЕЛЕ, а не в сигнатуре: умолчание,
    связанное на импорте, мутация уже не достаёт — эту форму на проекте
    выгребали.
    """
    t0 = time.perf_counter()
    prober = read_count_frames if prober is None else prober
    decoder = read_decoded_frames if decoder is None else decoder
    gray = fork_looper.read_gray if gray is None else gray
    pose_reader = fork_looper.read_pose if pose_reader is None else pose_reader
    if face_prober is None:
        def face_prober(p):
            from . import identity_arcface

            return identity_arcface.face_detail(p)

    axes, steps = {}, {}

    # --- ось 1: дырки во временных метках -------------------------------
    t = time.perf_counter()
    raw = prober(video_path)
    probed = parse_count_frames(raw.get("out", "")) if raw.get("ran") else {
        "ok": False, "frames": None, "fps": None, "seconds": None,
        "why": raw.get("why", "")}
    plain = decoder(video_path, vsync0=False)
    fixed = decoder(video_path, vsync0=True)
    p_n = parse_decoded_frames(plain.get("err", "")) if plain.get("ran") else {
        "ok": False, "frames": None, "why": plain.get("why", "")}
    f_n = parse_decoded_frames(fixed.get("err", "")) if fixed.get("ran") else {
        "ok": False, "frames": None, "why": fixed.get("why", "")}
    axes["timestamps"] = timestamp_verdict(probed.get("frames"),
                                           p_n.get("frames"),
                                           f_n.get("frames"))
    axes["timestamps"]["why"] = "; ".join(
        w for w in (probed.get("why"), p_n.get("why"), f_n.get("why")) if w)
    steps["timestamps"] = round(time.perf_counter() - t, 3)
    fps = probed.get("fps")
    seconds = probed.get("seconds")

    paths = list(frame_paths or [])
    n = len(paths)

    # --- ось 2: монтажные швы -------------------------------------------
    t = time.perf_counter()
    if not paths:
        axes["cuts"] = {**tally(0, 0, 1), "cuts": [], "bar": CUT_JUMP,
                        "note": ("кадров не подано: швы искать не в чем. Это "
                                 "НЕ «швов нет»")}
        marks = []
    else:
        c = fork_looper.cuts(paths, gray=gray)
        marks = c.get("cuts") or []
        # Разметка, а не фильтр: наличие швов вердикт НЕ роняет. Роняет его
        # ось 3 — и только через длину получившихся сцен.
        axes["cuts"] = {**(tally(len(paths) - 1, 0, 0)
                           if c.get("outcome") != UNMEASURED
                           else tally(0, 0, 1)),
                        "cuts": marks, "bar": CUT_JUMP,
                        "note": c.get("note", "")}
    steps["cuts"] = round(time.perf_counter() - t, 3)

    scene_list = scenes(n, marks) if n else []
    axes["scenes"] = scene_length_verdict(scene_list, fps)

    # --- ось 4: сиротские кисти (МЯГКАЯ) и ось 5: размер лица ------------
    t = time.perf_counter()
    orphans = seen_poses = pose_blind = 0
    sizes, no_face, face_blind = [], 0, 0
    for p in paths:
        r = pose_reader(str(p))
        if r.get("why"):
            pose_blind += 1
        else:
            verdict = is_orphan_wrist(r.get("points"))
            if verdict is None:
                pose_blind += 1
            else:
                seen_poses += 1
                orphans += 1 if verdict else 0
        try:
            d = face_prober(str(p))
        except Exception:  # noqa: BLE001 — «спросить нечем» на этом кадре
            face_blind += 1
        else:
            if d is None:
                no_face += 1
            else:
                sizes.append(int(d["face_px"]))
    axes["orphan_wrists"] = orphan_verdict(
        (orphans / seen_poses) if seen_poses else None, seen_poses, pose_blind)
    axes["face_size"] = face_size_verdict(sizes, no_face, face_blind)
    steps["pose_and_face"] = round(time.perf_counter() - t, 3)

    # --- окно -----------------------------------------------------------
    axes["window"] = (window(scene_list, product_seconds, fps)
                      if product_seconds is not None else
                      {**tally(0, 0, 1), "start": None, "end": None,
                       "frames": None, "scene": None,
                       "note": "продуктовая длина не задана: окно не выбирали"})

    return _report("драйвинг", video_path, axes, steps, t0,
                   soft=("orphan_wrists", "window"),
                   extra={"fps": fps, "seconds": seconds, "frames": n,
                          "scenes": scene_list})


def photo_intake(photo_path, *, faces_prober=None) -> dict:
    """ПРИЁМ ФОТОГРАФИИ КЛИЕНТА: лицо найдено, размер в px, один человек.

    Три оси, и все три жёсткие. Ноль лиц и два лица — РАЗНЫЕ нарушения, и
    сводить их в одно «фото не годится» нельзя: первое чинится другой
    фотографией, второе — кадрированием.
    """
    t0 = time.perf_counter()
    faces_prober = read_faces if faces_prober is None else faces_prober
    r = faces_prober(str(photo_path))
    axes = {}
    if r.get("why") or r.get("faces") is None:
        blind = {**tally(0, 0, 1),
                 "note": (f"спросить нечем: {r.get('why') or 'детектор молчит'}. "
                          f"Это НЕ «лица нет»")}
        axes = {"face_found": dict(blind), "face_size": dict(blind),
                "one_person": dict(blind)}
        return _report("фото клиента", photo_path, axes, {}, t0, soft=())

    faces = r["faces"]
    axes["face_found"] = {**tally(1, 0 if faces else 1, 0), "faces": len(faces),
                          "note": (f"лиц найдено {len(faces)}" if faces else
                                   "лица не найдено: якорь личности брать не с чего")}
    if faces:
        biggest = faces[0]["face_px"]
        axes["face_size"] = {**tally(1, 0 if biggest >= MIN_FACE_PX else 1, 0),
                             "face_px": biggest, "bar_px": MIN_FACE_PX,
                             "note": (f"самое крупное лицо {biggest} px при "
                                      f"планке {MIN_FACE_PX} px")}
    else:
        axes["face_size"] = {**tally(0, 0, 1), "face_px": None,
                             "bar_px": MIN_FACE_PX,
                             "note": "лица нет: размер мерить не у чего"}
    axes["one_person"] = {
        **tally(1, 0 if len(faces) == PHOTO_PEOPLE_EXPECTED else 1, 0),
        "faces": len(faces), "expected": PHOTO_PEOPLE_EXPECTED,
        "note": (f"людей на кадре {len(faces)}, ждали {PHOTO_PEOPLE_EXPECTED}"
                 + ("" if len(faces) == PHOTO_PEOPLE_EXPECTED else
                    ". Личность меряется по САМОМУ КРУПНОМУ лицу, то есть "
                    "выбирает его прибор, а не человек"))}
    return _report("фото клиента", photo_path, axes, {}, t0, soft=())


def style_intake(ref_path, *, card_reader=None) -> dict:
    """ПРИЁМ СТИЛЕВОГО РЕФЕРЕНСА: читается ли карточка стиля.

    Карточка НЕ ВЫЧИСЛЯЕТСЯ здесь и не проверяется на смысл — это работа
    `creative_eval.style.style_card` и `fork_style_prompt`. Здесь один вопрос:
    можно ли вообще получить с этой картинки все четыре поля. Пустое поле —
    `не годно`, недоступный пакет — `не смогли`, и это разные исходы: первое
    про картинку, второе про машину.
    """
    t0 = time.perf_counter()
    card_reader = read_style_card if card_reader is None else card_reader
    r = card_reader(str(ref_path))
    card = r.get("card")
    if r.get("why") or card is None:
        axes = {"card_readable": {
            **tally(0, 0, 1), "card": None,
            "note": (f"карточку прочитать нечем: {r.get('why') or 'ответа нет'}. "
                     f"Это НЕ «стиль плохой»")}}
        return _report("стилевой референс", ref_path, axes, {}, t0, soft=())

    # Состав полей — тот, что объявляет `style_card`. Список назван здесь
    # ЛИТЕРАЛОМ намеренно: если карточка однажды потеряет поле, приём обязан
    # покраснеть, а импортированный список поехал бы вместе с ней и промолчал.
    need = ("colours", "value_key", "saturation", "texture")
    if not isinstance(card, dict):
        missing = list(need)
    else:
        missing = [k for k in need if not card.get(k)]
    axes = {"card_readable": {
        **tally(len(need), len(missing), 0), "card": card, "missing": missing,
        "note": (f"полей в карточке {len(need) - len(missing)} из {len(need)}"
                 + (f", пусты: {missing}" if missing else
                    f"; палитра {list(card.get('colours') or [])}, "
                    f"тональность {card.get('value_key')!r}, насыщенность "
                    f"{card.get('saturation')!r}, фактура {card.get('texture')!r}"))}}
    return _report("стилевой референс", ref_path, axes, {}, t0, soft=())


# ---------------------------------------------------------------------------
# Свод
# ---------------------------------------------------------------------------

def _report(kind, source, axes: dict, steps: dict, t0: float, *,
            soft=(), extra: dict | None = None) -> dict:
    """Свести оси в один вердикт. Мягкие оси в него НЕ ВХОДЯТ.

    ВЕРДИКТ СВОДА ВЫВОДИТСЯ ИЗ ОСЕЙ, а не назначается рядом с ними (Е2): при
    расхождении флага и свидетельства верь свидетельству. Поэтому здесь нет ни
    одного места, где своду можно было бы проставить исход руками.
    """
    hard = {k: v for k, v in axes.items() if k not in soft}
    checked = sum(v.get("checked", 0) for v in hard.values())
    violations = sum(v.get("violations", 0) for v in hard.values())
    unmeasured = sum(v.get("unmeasured", 0) for v in hard.values())
    total = tally(checked, violations, unmeasured)
    # Свод не вправе быть зеленее худшей оси: сумма чисел может дать «годно»
    # там, где одна ось не измерилась на фоне тысячи измеренных.
    outcomes = [v.get("outcome") for v in hard.values()]
    if FAIL in outcomes:
        total["outcome"] = FAIL
    elif UNMEASURED in outcomes:
        total["outcome"] = UNMEASURED
    warns = [k for k, v in axes.items() if v.get("warn")]
    return {"kind": kind, "source": str(source), "axes": axes,
            "outcome": total["outcome"], "checked": total["checked"],
            "violations": total["violations"],
            "unmeasured": total["unmeasured"],
            "soft": list(soft), "warnings": warns, "steps": steps,
            "elapsed": round(time.perf_counter() - t0, 3),
            **(extra or {})}


def render(report: dict) -> str:
    """Отчёт глазами (П3). Числа рядом с вердиктом на каждой строке."""
    lines = [f"ПРИЁМ: {report['kind']} — {Path(report['source']).name}",
             f"  ВЕРДИКТ: {report['outcome']}  "
             f"(проверено {report['checked']}, нарушений "
             f"{report['violations']}, не смогли {report['unmeasured']})"]
    for name, ax in report["axes"].items():
        mark = " [мягкая]" if name in report.get("soft", []) else ""
        lines.append(f"  {name}{mark}: {ax.get('outcome')} "
                     f"(проверено {ax.get('checked')}, нарушений "
                     f"{ax.get('violations')}, не смогли {ax.get('unmeasured')})")
        if ax.get("note"):
            lines.append(f"      {ax['note']}")
    if report.get("warnings"):
        lines.append(f"  ПРЕДУПРЕЖДЕНИЯ: {report['warnings']}")
    if report.get("steps"):
        lines.append(f"  длительность шагов, с: {report['steps']}")
    return "\n".join(lines)
