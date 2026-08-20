"""Склейка петли в кадры драйвинга: звено между отбором петли и прогоном.

ЗАЧЕМ. `fork_looper` находит петлю и печатает план — «петля 114..162, 49
кадров; 3 повтора = 145 кадров = 6.04 с». **Эти 145 кадров никто не собирал.**
Сквозной путь `fork_run` брал каталог кадров подряд, то есть все 362 кадра
боевого ролика, в которых петли нет: на повторе получался жёсткий стык
(ИЗМЕРЕНО на `assets/driving_yogaball.mp4`: средняя попиксельная разница 17.82
у наивного «взять всё подряд» против 2.72 у петли 114..162, хэндоф и
`assets/README.md`). Этот модуль стоит МЕЖДУ выбором петли и рендером:
«выбрана петля [i..j], нужен ролик на S секунд» -> «вот кадры драйвинга на
диске, готовые для `fork_run`».

СТЫКОВОЙ КАДР НЕ ДУБЛИРУЕТСЯ, И ЭТО НЕ ОКРУГЛЕНИЕ. Петля [114..162] — это 49
кадров, но кадр 162 и кадр 114 — ОДИН И ТОТ ЖЕ момент движения (ровно поэтому
пара и выбрана петлёй: стык между ними меряется как обычный межкадровый шаг).
Положив оба, мы получим два одинаковых кадра подряд — заедание на КАЖДОМ стыке,
ровно то, от чего уходим. Поэтому тело повтора — [i..j-1], а кадр j кладётся
ОДИН РАЗ в самом конце, чтобы ролик замыкался сам на себя:

    N повторов = N*(L-1) + 1 кадров.   49 кадров, 3 повтора -> 145, не 147.

Та же арифметика уже живёт в `fork_looper.repeat_plan` и в `gif_indices`
(«последний кадр петли НЕ КЛАДЁТСЯ»); здесь она не переписана, а СВЕРЯЕТСЯ:
длина построенной последовательности обязана совпасть с `frames` выбранного
плана, иначе `_agree` роняет прогон (Е1 — расхождение двух мест ловится
машиной, а не глазами).

ДЛИНА ОБЯЗАНА ЛОЖИТЬСЯ НА ШАГ 4k+1. `fork_comfy.snap_frames` прижимает длину
ВНИЗ молча, и прижатая на кадр склейка обрывает последний повтор посередине
движения — петля разъезжается именно там, где мы её собирали. Проверяется не
рассуждением, а вызовом: план принимается, только если
`snap_frames(frames) == frames`.

ЧТО ДЕЛАТЬ С ПЕТЛЁЙ НЕКРАТНОЙ ДЛИНЫ — РЕШЕНИЕ И ЕГО ЦЕНА. У петли длины L шаг
склейки равен L-1. Для L=49 шаг 48 кратен 4, и любое число повторов даёт 4k+1
(97, 145, 193, 241) — но это свойство ЭТОЙ петли, а не закон. Для L=43 шаг 42,
и 4k+1 получается только при ЧЁТНОМ числе повторов (85, 169), нечётные (43,
127) обёртка прижмёт. Рассмотрены три выхода:

    подрезать петлю до кратной длины  ОТВЕРГНУТО. Подрезка снимает 1-3 кадра
                                      С КОНЦА, то есть ломает ровно тот стык,
                                      ради которого петля и отбиралась: кадр
                                      j-2 уже не сосед кадра i. Цена платится
                                      той единственной величиной, которую мы
                                      здесь бережём.
    отказать                          ОТВЕРГНУТО как ЕДИНСТВЕННЫЙ ответ: у
                                      L=43 годные склейки существуют, и
                                      отказ выбросил бы рабочий материал.
    выбрать число повторов            ПРИНЯТО. Перебираем N, оставляем те, чья
                                      длина уже лежит на шаге, и выбираем
                                      ближайшую к заказу. Цена НЕ в кадрах —
                                      ни один кадр не подрезан, — а в
                                      ЗЕРНИСТОСТИ ЗАКАЗА: у L=43 шаг длины
                                      становится 2*(L-1)=84 кадра (3.5 с при
                                      24 к/с) вместо 42 (1.75 с), и часть
                                      заказов из полосы 5-10 с не набирается
                                      вовсе. Это и есть третий исход, а не
                                      молчаливое приближение.

ТРИ ИСХОДА (Р1), и они не сворачиваются друг в друга:

    годно          кадры записаны; сказано сколько, из скольких повторов,
                   сколько это секунд по частоте ИСТОЧНИКА и куда легло
    не годно       петля выходит за границы материала, петля короче двух
                   кадров, кадры поданы на частоте, которой мы не отдаём
    не смогли      частота источника неизвестна (секунд назвать нечем) или
                   заказанная длина из ЭТОЙ петли не набирается — с числами,
                   какие длины набираются

СЕКУНДЫ СЧИТАЮТСЯ ПО ЧАСТОТЕ ИСТОЧНИКА, НЕ ПО НАШЕЙ. Модель берёт кадры позы
один к одному: 145 кадров ролика, снятого на 24 к/с, — это 6.04 с, а не 4.83.
Правило частоты живёт в `fork_comfy.output_fps` и вызывается отсюда, а не
повторяется (Е1); подставленная своя частота на боевом ролике дала бы ошибку
25% в каждой строке — этот дефект на проекте уже ловили.

КАДРЫ НЕ КОПИРУЮТСЯ БАЙТАМИ. ИЗМЕРЕНО на боевом материале: кадр 720x1278 PNG
весит 742 КБ, склейка на 145 кадров — 107 МБ на КАЖДЫЙ заказ, притом что все
145 ссылаются на 49 разных файлов. Умолчание — жёсткая ссылка (`os.link`):
данных на диске ноль, только записи в каталоге. Цена названа и не заметена:
жёсткая ссылка — ТОТ ЖЕ файл, и шаг конвейера, который запишет кадр НА МЕСТЕ,
испортит исходный кадр (у нас таких шагов нет: `fork_channels`, `fork_mask` и
`fork_video` пишут в свои каталоги). Разные файловые системы жёсткую ссылку не
дают — тогда символическая, а если и её нет — копия; чем именно легло,
печатается, а не предполагается (Е2: вердикт по тому, что исполнилось).
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from . import fork_comfy, fork_looper, fork_video
from .fork_identity import FAIL, PASS, UNMEASURED

# ---------------------------------------------------------------------------
# КОНСТАНТЫ-РЕШЕНИЯ. У каждой помечено происхождение (И4).
# ---------------------------------------------------------------------------

#: ДОПУСК АСИММЕТРИЧЕН, И ЭТО НЕ ВКУСОВЩИНА. Кадров МЕНЬШЕ заказанного —
#: недостача, кадров БОЛЬШЕ — ролик чуть длиннее просимого. Следующий по
#: конвейеру прибор `fork_run.length_fits_driving` устроен ровно так же:
#: `PASS если need - have <= 0`, то есть избыток пропускает, а недостачу —
#: никогда.
#:
#: ВЫБРАНО (кем: эта смена; из чего: из согласия двух приборов). Было
#: `LENGTH_SLACK = LENGTH_STEP` в обе стороны, и это ДВОЙНОЙ СЧЁТ: `want`
#: приходит из `frames_for_seconds` УЖЕ прижатым к шагу, поэтому скидка на
#: прижатие в допуске учтена второй раз. ИЗМЕРЕНО сторонним прогоном по 323
#: планам (частоты 24/25/30, длины петли 2..119, шесть заказов): по плановым
#: секундам расхождений с `length_fits_driving` — 0, по ЗАКАЗАННЫМ — 93 из
#: 323, и каждое это «склейка: годно» против «сводящий проход: не смогли».
LENGTH_SHORT_MAX = 0

#: РАСЧЁТ (по `fork_comfy.snap_frames`): наверх допуск остаётся шагом обёртки
#: — набрать РОВНО заказ удаётся не всегда, а лишние кадры ролик не ломают.
#: Без верхней границы вовсе заказ 5 с мог бы честно закрыться десятью.
LENGTH_OVER_MAX = fork_comfy.LENGTH_STEP

#: ВЫБРАНО (кем: эта смена; из чего: из устройства склейки). Петля из одного
#: кадра — не петля: шаг склейки L-1 равен нулю, и сколько её ни повторяй,
#: получается тот же один кадр. Это не «мало кадров», это отсутствие движения,
#: поэтому проверка стоит отдельно от полосы длин и говорит своими словами.
MIN_LOOP_FRAMES = 2

#: Как кладём кадр на диск, по убыванию предпочтения. Порядок — это П2 наоборот
#: (дешёвое раньше дорогого по РЕСУРСУ, а не по времени): жёсткая ссылка стоит
#: записи в каталоге, копия — 742 КБ на кадр.
LINK_HARD, LINK_SYM, LINK_COPY = "жёсткая ссылка", "символическая", "копия"
LINK_ORDER = (LINK_HARD, LINK_SYM, LINK_COPY)

EXIT_BY_OUTCOME = {PASS: 0, FAIL: 1, UNMEASURED: 2}


# ---------------------------------------------------------------------------
# Чистые функции. Развилки вынесены из точки входа, чтобы их красил тест (Т5).
# ---------------------------------------------------------------------------

def sequence_indices(i: int, j: int, repeats: int) -> list:
    """Номера ИСХОДНЫХ кадров склейки петли [i..j], повторённой `repeats` раз.

    Кадр `j` — тот же момент движения, что кадр `i`, поэтому тело повтора это
    [i..j-1], а `j` кладётся один раз в конце. Длина списка — N*(L-1)+1.
    """
    for name, v in (("i", i), ("j", j), ("repeats", repeats)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"{name}={v!r}: ожидалось целое")
    if i < 0:
        raise ValueError(f"i={i}: номер кадра не бывает отрицательным")
    if j <= i:
        raise ValueError(f"петля [{i}..{j}]: конец не позже начала")
    if repeats < 1:
        raise ValueError(f"повторов {repeats}: меньше одного не бывает")
    body = list(range(i, j))
    return body * repeats + [j]


def loop_bounds_ok(i: int, j: int, n_frames: int) -> dict:
    """Лежит ли петля внутри материала. Числа рядом с вердиктом (Р2)."""
    length = j - i + 1
    if i < 0 or j < 0:
        return {"outcome": FAIL, "frames": length,
                "note": f"петля [{i}..{j}]: отрицательный номер кадра"}
    if j <= i:
        return {"outcome": FAIL, "frames": length,
                "note": (f"петля [{i}..{j}]: конец не позже начала — "
                         f"это не петля, а точка")}
    if j > n_frames - 1:
        return {"outcome": FAIL, "frames": length,
                "note": (f"петля [{i}..{j}] выходит за материал: кадров всего "
                         f"{n_frames}, последний номер {n_frames - 1}. "
                         f"Не хватает {j - (n_frames - 1)}")}
    if length < MIN_LOOP_FRAMES:
        return {"outcome": FAIL, "frames": length,
                "note": (f"петля [{i}..{j}] — {length} кадр(ов), минимум "
                         f"{MIN_LOOP_FRAMES}: шаг склейки {length - 1}, "
                         f"повторять нечего")}
    return {"outcome": PASS, "frames": length,
            "note": (f"петля [{i}..{j}] — {length} кадров, лежит внутри "
                     f"материала из {n_frames}")}


def _agree(plan: dict, i: int, j: int) -> int:
    """Сверка двух мест, где живёт длина склейки (Е1). Возвращает длину.

    `repeat_plan` считает её формулой, `sequence_indices` — построением списка.
    Разъехавшись, они дали бы каталог, длина которого не та, что напечатана в
    отчёте. Здесь это падает, а не печатается.
    """
    built = len(sequence_indices(i, j, plan["repeats"]))
    if built != plan["frames"]:
        raise AssertionError(
            f"длина склейки разошлась: fork_looper.repeat_plan говорит "
            f"{plan['frames']}, построено {built} для петли [{i}..{j}] "
            f"x{plan['repeats']}")
    return built


def playback_fps(frames_fps) -> dict:
    """С какой частотой пойдут ПОДАННЫЕ кадры. Правило — из `fork_comfy` (Е1).

    Кадры драйвинга съедаются моделью один к одному, поэтому частота, на
    которой они лягут в ролик, — это выходная частота по правилу владельца.
    Кадры, поданные ВЫШЕ потолка, здесь не прореживаются: прореживание умеет
    `fork_video`, и делать вторую его копию значит завести второе место для
    одного знания.
    """
    # Негодная частота — ОТЧЁТ, а не трассировка: проверка существует, чтобы
    # собрать находки, а падение на первой посылает чинить окружение там, где
    # негоден вход. Тот же класс дефекта уже чинился в `fork_comfy.check`.
    try:
        rate = fork_comfy.output_fps(frames_fps)
    except (TypeError, ValueError) as exc:
        return {"outcome": FAIL, "fps": None,
                "note": f"частота {frames_fps!r} не принята: {exc}"}
    if rate["outcome"] != PASS:
        return {"outcome": UNMEASURED, "fps": None, "note": rate["note"]}
    if rate["fps"] != frames_fps:
        return {"outcome": FAIL, "fps": None,
                "note": (f"кадры поданы на {frames_fps} к/с, а выход по "
                         f"правилу владельца {rate['fps']} к/с — лишние кадры "
                         f"прореживает раскодировщик (`fork_video.frames` с "
                         f"fps={int(rate['fps'])}), а не склейка. Подайте "
                         f"кадры, уже раскодированные на {rate['fps']}")}
    return {"outcome": PASS, "fps": rate["fps"], "note": rate["note"]}


def source_fps(claimed, measured) -> dict:
    """Какая частота у источника: СНЯТАЯ с файла или названная ключом.

    Е2 — при расхождении флага и свидетельства верят свидетельству. Ключ
    `--fps` — это НАМЕРЕНИЕ оператора, а `measured` снято с самого файла
    `ffprobe`. Раньше ключ молча перебивал снятое, и ВОСПРОИЗВЕДЕНО на
    настоящем mp4 60 к/с с `--fps 30`: заказ 5.4 с отдавался как «годно,
    161 кадр = 5.37 с», а движения в этих кадрах 2.68 с — ошибка ровно вдвое,
    и отчёт при этом сходился сам с собой.

    Расхождение — «не годно» с ОБОИМИ числами, а не тихий выбор одного из
    них: оператор, попросивший 30 на шестидесяти, либо ошибся файлом, либо
    хотел прореживание, которое склейка не делает (его делает `fork_video`).
    """
    if measured is None:
        # Каталог кадров: снимать частоту не с чего, остаётся названное.
        return {"outcome": PASS, "fps": claimed, "measured": None,
                "note": (f"частота названа ключом: {claimed}"
                         if claimed is not None else
                         "частота источника не названа и снять её не с чего")}
    if claimed is None:
        return {"outcome": PASS, "fps": measured, "measured": measured,
                "note": f"частота снята с файла: {measured} к/с"}
    if claimed != measured:
        return {"outcome": FAIL, "fps": None, "measured": measured,
                "note": (f"ключ говорит {claimed} к/с, а с файла снято "
                         f"{measured} к/с. Верим снятому (Е2) и не гадаем: "
                         f"взяв {claimed}, склейка назвала бы длину ролика с "
                         f"ошибкой в {measured / claimed:.2f} раза. Уберите "
                         f"ключ либо прорядите файл заранее "
                         f"(`fork_video.frames` с fps={claimed})")}
    return {"outcome": PASS, "fps": measured, "measured": measured,
            "note": f"частота {measured} к/с — ключ и файл сошлись"}


def admissible_plans(length: int, *, fps) -> dict:
    """Склейки петли длины `length`, которые обёртка НЕ прижмёт.

    Перебор повторов и полоса 5-10 с — из `fork_looper.repeat_plan` (Е1),
    кратность — из `fork_comfy.snap_frames` через поле `snapped` того же плана.
    Здесь только отбор и счёт отброшенного.
    """
    plans = fork_looper.repeat_plan(length, fps=fps)
    kept = [p for p in plans if p["snapped"] == p["frames"]]
    dropped = [p for p in plans if p["snapped"] != p["frames"]]
    return {"kept": kept, "dropped_step": dropped, "considered": len(plans)}


def choose_repeats(length: int, seconds: float, *, fps) -> dict:
    """Сколько раз повторить петлю, чтобы получить заказанную длину.

    Три исхода. `не смогли` — когда из ЭТОЙ петли такой длины не набирается;
    рядом печатается, что набирается, а не тихо берётся ближайшее.
    """
    if fps is None:
        return {"outcome": UNMEASURED, "plan": None, "plans": [], "dropped": 0,
                "note": ("частота источника неизвестна — сколько это секунд, "
                         "сказать нечем. Это НЕ «берём свою»: 145 кадров это "
                         "6.04 с на 24 к/с и 4.83 с на 30")}
    if length < MIN_LOOP_FRAMES:
        return {"outcome": FAIL, "plan": None, "plans": [], "dropped": 0,
                "note": (f"петля {length} кадр(ов): шаг склейки "
                         f"{length - 1}, повторение не удлиняет ролик")}
    try:
        want = fork_comfy.frames_for_seconds(seconds, fps=fps)
    except (ValueError, TypeError) as exc:
        return {"outcome": FAIL, "plan": None, "plans": [], "dropped": 0,
                "note": f"заказ {seconds} с не принят: {exc}"}

    got = admissible_plans(length, fps=fps)
    kept, dropped = got["kept"], got["dropped_step"]
    step_note = ""
    if dropped:
        step_note = (f" Отброшено планов по шагу {fork_comfy.LENGTH_STEP} от "
                     f"{fork_comfy.LENGTH_BASE}: {len(dropped)} "
                     + ", ".join(f"{p['repeats']}x={p['frames']}к"
                                 f"(обёртка прижмёт к {p['snapped']})"
                                 for p in dropped) + ".")
    if not kept:
        return {"outcome": UNMEASURED, "plan": None, "plans": [],
                "dropped": len(dropped),
                "note": (f"из петли в {length} кадров при {fps} к/с не "
                         f"набирается НИ ОДНОЙ длины в полосе "
                         f"{fork_comfy.SECONDS_MIN}-{fork_comfy.SECONDS_MAX} с: "
                         f"шаг склейки {length - 1} кадров "
                         f"({(length - 1) / fps:.2f} с)." + step_note)}

    achievable = ", ".join(f"{p['repeats']}x={p['frames']}к/{p['seconds']}с"
                           for p in kept)
    # Ищем среди тех, что НЕ КОРОЧЕ заказа, и берём самый близкий сверху.
    # Просто «ближайший по модулю» выбирал бы недостачу там, где рядом лежит
    # годный избыток, и отказывал бы на ровном месте.
    enough = [p for p in kept
              if want["frames"] - p["frames"] <= LENGTH_SHORT_MAX]
    if not enough:
        best = max(kept, key=lambda p: p["frames"])
        return {"outcome": UNMEASURED, "plan": None, "plans": kept,
                "dropped": len(dropped),
                "note": (f"заказ {seconds} с = {want['frames']} кадров из "
                         f"петли в {length} кадров НЕ НАБИРАЕТСЯ: самое "
                         f"длинное {best['repeats']}x = {best['frames']} "
                         f"кадров ({best['seconds']} с), НЕ ХВАТАЕТ "
                         f"{want['frames'] - best['frames']} кадров. "
                         f"Набирается: {achievable}." + step_note)}
    best = min(enough, key=lambda p: p["frames"])
    miss = best["frames"] - want["frames"]
    if miss > LENGTH_OVER_MAX:
        return {"outcome": UNMEASURED, "plan": None, "plans": kept,
                "dropped": len(dropped),
                "note": (f"заказ {seconds} с = {want['frames']} кадров из "
                         f"петли в {length} кадров НЕ НАБИРАЕТСЯ: ближайшее "
                         f"сверху {best['repeats']}x = {best['frames']} "
                         f"кадров ({best['seconds']} с), промах {miss:+d} "
                         f"кадров при допуске сверху {LENGTH_OVER_MAX}. "
                         f"Набирается: {achievable}." + step_note)}
    return {"outcome": PASS, "plan": best, "plans": kept,
            "dropped": len(dropped),
            "note": (f"заказ {seconds} с = {want['frames']} кадров; берём "
                     f"{best['repeats']} повтора(ов) петли в {length} кадров = "
                     f"{best['frames']} кадров = {best['seconds']} с при "
                     f"{fps} к/с (промах {miss:+d} кадров, допуск сверху "
                     f"{LENGTH_OVER_MAX}, недостача не допускается). "
                     f"Набиралось: {achievable}." + step_note)}


# ---------------------------------------------------------------------------
# Запись на диск
# ---------------------------------------------------------------------------

def place(src: Path, dst: Path, *, prefer=None) -> str:
    """Положить кадр, не копируя байты, если это возможно. Возвращает — ЧЕМ.

    Возвращается то, что ИСПОЛНИЛОСЬ (Е2), а не то, что просили: на другой
    файловой системе жёсткая ссылка не встанет, и молчаливая копия в отчёте
    выглядела бы как ссылка.
    """
    order = LINK_ORDER if prefer is None else (
        (prefer,) + tuple(m for m in LINK_ORDER if m != prefer))
    last = None
    for mode in order:
        try:
            if mode == LINK_HARD:
                os.link(src, dst)
            elif mode == LINK_SYM:
                os.symlink(os.path.abspath(src), dst)
                # os.symlink молча удаётся на несуществующую цель: сообщать
                # «символическая» про битую ссылку — нарушение Е2 (отчёт о
                # том, что исполнилось). Проверяем, что ссылка читается.
                if not dst.exists():
                    dst.unlink()
                    raise OSError(f"ссылка на {src} не разыменовывается")
            else:
                shutil.copyfile(src, dst)
            return mode
        except (OSError, NotImplementedError, AttributeError) as exc:
            last = exc
            continue
    raise OSError(f"{src} -> {dst}: не легло ни ссылкой, ни копией: {last}")


def write_sequence(paths, indices, out_dir, *, prefer=None,
                   overwrite=False) -> dict:
    """Разложить кадры склейки в каталог. Имена — как у раскодировщика (Е1).

    Нумерация с единицы и ширина поля берутся у `fork_video.frame_name`:
    дальше по пути кадры собираются `sorted(glob('*.png'))`, и своя схема имён
    сортировалась бы иначе ровно там, где этого никто не проверяет.
    """
    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        return {"outcome": FAIL, "written": 0, "bytes": 0, "paths": [],
                "modes": {}, "note": f"{out} — не каталог"}
    # Кадры источника стираются ниже ДО того, как их прочитают. Если
    # назначение — тот же каталог, материал владельца уничтожается, а отчёт
    # при этом выходит «годно». Дешёвая проверка раньше дорогой записи (П2).
    if out.is_dir():
        try:
            here = out.resolve()
        except OSError:
            here = out.absolute()
        clash = 0
        for raw in paths:
            try:
                parent = Path(raw).resolve().parent
            except OSError:
                parent = Path(raw).absolute().parent
            if parent == here:
                clash += 1
        if clash:
            return {"outcome": FAIL, "written": 0, "bytes": 0, "paths": [],
                    "modes": {},
                    "note": (f"каталог назначения {out} — он же каталог "
                             f"источника (совпало кадров: {clash} из "
                             f"{len(paths)}). Запись начинается со стирания "
                             f"*{fork_video.FRAME_SUFFIX} и уничтожила бы "
                             f"исходный материал. Задайте другой каталог")}
    already = sorted(out.glob(f"*{fork_video.FRAME_SUFFIX}")) if out.is_dir() else []
    if already and not overwrite:
        return {"outcome": UNMEASURED, "written": 0, "bytes": 0, "paths": [],
                "modes": {},
                "note": (f"в {out} уже лежит кадров: {len(already)}. Молча "
                         f"поверх не пишем: чужие кадры вперемешку со своими "
                         f"дают отсортованный и правдоподобный каталог. "
                         f"Задайте overwrite=True или другой каталог")}
    for f in already:
        f.unlink()
    out.mkdir(parents=True, exist_ok=True)

    modes, written = {}, []
    for k, idx in enumerate(indices):
        dst = out / fork_video.frame_name(k + 1)
        mode = place(Path(paths[idx]), dst, prefer=prefer)
        modes[mode] = modes.get(mode, 0) + 1
        written.append(dst)
    # Размер СВОЕГО каталога, а не исходных кадров: у жёстких ссылок это
    # ровно то, чего склейка стоила диску сверх уже лежащего материала.
    own = sum(0 if p.is_symlink() or p.stat().st_nlink > 1
              else p.stat().st_size for p in written)
    return {"outcome": PASS, "written": len(written), "bytes": own,
            "paths": written, "modes": modes,
            "note": (f"кадров записано {len(written)}, своих байт на диске "
                     f"{own} ("
                     + ", ".join(f"{m}: {c}" for m, c in modes.items()) + ")")}


# ---------------------------------------------------------------------------
# Разбор целиком
# ---------------------------------------------------------------------------

def _report(outcome, note, t0, steps, **extra) -> dict:
    out = {"outcome": outcome, "frames": 0, "repeats": None, "seconds": None,
           "out_dir": None, "paths": [],
           "elapsed": round(time.perf_counter() - t0, 4),
           "steps": [{"step": s, "outcome": o, "note": n, "seconds": round(e, 4)}
                     for s, o, n, e in steps]}
    out.update(extra)
    out["note"] = f"{outcome}: {note}"
    return out


def splice(source, loop, seconds, out_dir, *, fps=None, decode=None,
           prober=None, prefer=None, overwrite=False) -> dict:
    """Собрать кадры драйвинга из петли. Дешёвое раньше дорогого (П2).

    `source` — каталог кадров или видеофайл (раскодирует `fork_video`, Е1).
    `loop` — пара (i, j) НОМЕРОВ В ОТСОРТИРОВАННОМ СПИСКЕ кадров: те же номера,
    что печатает `fork_looper`. `seconds` — заказанная длина ролика.

    ВНЕШНИХ ИНСТРУМЕНТА ДВА, И ТОЧЕК ВНЕДРЕНИЯ ТОЖЕ ДВЕ: `decode` (ffmpeg) и
    `prober` (ffprobe). Пока частота бралась из отчёта раскодировщика, хватало
    одной — но частота теперь снимается ДО раскодирования, и подставной
    раскодировщик без подставного щупа оставлял бы тест ходить в настоящий
    ffprobe (Т4).
    """
    t = time.perf_counter()
    steps = []
    i, j = loop
    src = Path(source)

    # 1. Частота — ПЕРВОЙ, и это не косметика (П2). Раскодирование боевого
    #    ролика — минуты и гигабайты в out/src; отказ по частоте, поставленный
    #    после него, эти гигабайты уже потратил и оставил лежать. ИЗМЕРЕНО на
    #    mp4 60 к/с: отказ печатался после 180 распакованных кадров, 728 КБ.
    #    `ffprobe` отвечает за миллисекунды и знает частоту целиком.
    t0 = time.perf_counter()
    measured = None
    if src.is_file():
        seen = fork_video.probe(str(src), prober=prober)
        if seen["outcome"] != PASS:
            steps.append(("частота", seen["outcome"], seen["note"],
                          time.perf_counter() - t0))
            return _report(seen["outcome"], seen["note"], t, steps)
        measured = seen["fps"]
    elif not src.is_dir():
        note = f"{src} — не каталог кадров и не файл"
        steps.append(("источник", UNMEASURED, note, time.perf_counter() - t0))
        return _report(UNMEASURED, note, t, steps)

    agreed = source_fps(fps, measured)
    if agreed["outcome"] != PASS:
        steps.append(("частота", agreed["outcome"], agreed["note"],
                      time.perf_counter() - t0))
        return _report(agreed["outcome"], agreed["note"], t, steps)
    rate = playback_fps(agreed["fps"])
    steps.append(("частота", rate["outcome"],
                  f"{agreed['note']}; {rate['note']}",
                  time.perf_counter() - t0))
    if rate["outcome"] == FAIL:
        return _report(FAIL, rate["note"], t, steps)
    fps_out = rate["fps"]

    # 2. Кадры. Миллисекунды на каталоге; на файле это самый дорогой шаг после
    #    записи, и до него мы уже знаем, что частота годна.
    t0 = time.perf_counter()
    if src.is_dir():
        paths = fork_looper.frame_paths(src)
    else:
        decode = fork_video.frames if decode is None else decode
        got = decode(str(src), str(Path(out_dir) / "src"), overwrite=True)
        if got.get("outcome") != PASS:
            steps.append(("кадры", got["outcome"], got["note"],
                          time.perf_counter() - t0))
            return _report(got["outcome"], got["note"], t, steps)
        paths = [Path(p) for p in got["paths"]]
    n = len(paths)
    if not n:
        note = f"в {src} нет кадров"
        steps.append(("кадры", FAIL, note, time.perf_counter() - t0))
        return _report(FAIL, note, t, steps)
    steps.append(("кадры", PASS, f"кадров подано {n}",
                  time.perf_counter() - t0))

    # 3. Петля внутри материала.
    t0 = time.perf_counter()
    bounds = loop_bounds_ok(i, j, n)
    steps.append(("петля", bounds["outcome"], bounds["note"],
                  time.perf_counter() - t0))
    if bounds["outcome"] != PASS:
        return _report(bounds["outcome"], bounds["note"], t, steps)
    length = bounds["frames"]

    # 4. План повторов.
    t0 = time.perf_counter()
    chosen = choose_repeats(length, seconds, fps=fps_out)
    steps.append(("план", chosen["outcome"], chosen["note"],
                  time.perf_counter() - t0))
    if chosen["outcome"] != PASS:
        return _report(chosen["outcome"], chosen["note"], t, steps,
                       loop=[i, j], loop_frames=length)
    plan = chosen["plan"]
    total = _agree(plan, i, j)
    indices = sequence_indices(i, j, plan["repeats"])

    # 5. Запись — единственный дорогой шаг, и он последний.
    t0 = time.perf_counter()
    wrote = write_sequence(paths, indices, out_dir, prefer=prefer,
                           overwrite=overwrite)
    steps.append(("запись", wrote["outcome"], wrote["note"],
                  time.perf_counter() - t0))
    if wrote["outcome"] != PASS:
        return _report(wrote["outcome"], wrote["note"], t, steps,
                       loop=[i, j], loop_frames=length)
    if wrote["written"] != total:
        note = (f"записано {wrote['written']} кадров, а план обещал {total} — "
                f"каталог отдавать нельзя")
        return _report(FAIL, note, t, steps, loop=[i, j], loop_frames=length)

    note = (f"петля [{i}..{j}] ({length} кадров, "
            f"{round(length / fps_out, 2)} с) x{plan['repeats']} = "
            f"{total} кадров = {plan['seconds']} с при {fps_out} к/с; "
            f"стыковой кадр не продублирован ({plan['repeats']}*"
            f"{length - 1}+1); легло в {out_dir}")
    return _report(PASS, note, t, steps, frames=total,
                   repeats=plan["repeats"], seconds=plan["seconds"],
                   fps=fps_out, loop=[i, j], loop_frames=length,
                   out_dir=str(out_dir), bytes=wrote["bytes"],
                   modes=wrote["modes"], paths=wrote["paths"],
                   indices=indices)


def parse_loop(text: str) -> tuple:
    """`114..162` -> (114, 162). Разбор вынесен из точки входа ради теста (Т5)."""
    part = str(text).split("..")
    if len(part) != 2:
        raise ValueError(f"петля {text!r}: ожидалось i..j, например 114..162")
    try:
        i, j = int(part[0]), int(part[1])
    except ValueError:
        raise ValueError(f"петля {text!r}: номера кадров — целые") from None
    return i, j


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python3 -m ball_reel.fork_splice",
        description="Склейка выбранной петли в кадры драйвинга под заказ.")
    ap.add_argument("source", help="каталог кадров или видеофайл")
    ap.add_argument("--loop", required=True, help="границы петли, i..j")
    ap.add_argument("--seconds", type=float, required=True,
                    help="заказанная длина ролика в секундах")
    ap.add_argument("--out", required=True, help="куда класть кадры драйвинга")
    ap.add_argument("--fps", type=float, default=None,
                    help="частота ИСТОЧНИКА; с видеофайла снимается сама")
    ap.add_argument("--copy", action="store_true",
                    help="копировать байтами вместо жёсткой ссылки")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    try:
        loop = parse_loop(a.loop)
    except ValueError as exc:
        print(f"ИСХОД: {FAIL}")
        print(f"  {exc}")
        return EXIT_BY_OUTCOME[FAIL]

    rep = splice(a.source, loop, a.seconds, a.out, fps=a.fps,
                 prefer=LINK_COPY if a.copy else None,
                 overwrite=a.overwrite)
    print(f"ИСХОД: {rep['outcome']}")
    for s in rep["steps"]:
        print(f"  [{s['outcome']:>18}] {s['step']:<9} {s['seconds']:>7.3f} с  "
              f"{s['note']}")
    print()
    print(rep["note"])
    return EXIT_BY_OUTCOME[rep["outcome"]]


if __name__ == "__main__":
    raise SystemExit(main())
