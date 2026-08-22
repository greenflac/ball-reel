"""Сквозной стенд продукта: фото клиента + драйвинг + стилевая рефка -> ролик.

ЦЕЛЬ ПРОДУКТА, дословно от владельца: «стенд, который на основе фоторефки,
драйвинга, фоторефки стиля соберёт консистентное видео в клинг, где личность
с фоторефки будет выполнять движения с драйвинга и всё будет стилизовано как
фотореференс стиля».

ЗАЧЕМ ЭТОТ МОДУЛЬ ОТДЕЛЬНО. Ступени по одиночке проверены и измерены, а вместе
не собирались ни разу. Сборка «вручную по мануалу» уже стоила денег дважды:
кусок драйвинга уехал на два кадра длиннее заказанного, а прогон в 25 минут
шёл МОЛЧА, упёрся в таймаут и унёс с собой всё, что успел измерить. Поэтому
здесь три свойства, и они важнее любой из ступеней:

  1. каждая ступень печатается В STDERR СРАЗУ, как только досчиталась;
  2. каждая ступень даёт ТРИ ИСХОДА и числа рядом с вердиктом
     (`проверено N`, `нарушений M`, `не смогли K`);
  3. прогон останавливается на первой `не годно` и говорит, на какой именно.

ВСЕ ВНЕШНИЕ ВЫЗОВЫ — ТОЧКИ ВНЕДРЕНИЯ (Т4). Стилизация, загрузка файлов, вызов
Kling, соседние модули приёма и сборки приходят параметрами. Умолчания ходят в
сеть и стоят денег; тесты обязаны прогонять ВЕСЬ путь на подставных функциях,
и они это делают — ни одного байта наружу.

## ЛЕСТНИЦА, ПО КОТОРОЙ ЧИТАЮТСЯ ЧИСЛА ЛИЧНОСТИ (ИЗМЕРЕНО, см. assets/README.md)

    0.0652   стилизованное фото против фото клиента — ТОТ ЖЕ человек
    0.35     планка проекта `fork_identity.SAME_PERSON_MAX`
    0.7137   отбракованный референс против боевого — РАЗНЫЕ люди
    1.0217   актёр драйвинга против фото клиента — заведомо разные

## ЧТО РЕШЕНО И БОЛЬШЕ НЕ ОБСУЖДАЕТСЯ

* `pro`-версии Kling ИСКЛЮЧЕНЫ НАВСЕГДА решением владельца: $2.6880 против
  $0.2100 (в 12.8 раза), лейбл всё равно не выжил, фон получил дорисованную
  анимацию. Сторож `refuse_pro` роняет вызов ДО денег.
* У эндпоинта motion-control ровно три поля. Не «мы знаем три» — щуп с
  негативным контролем показал, что известные поля с мусором отвергаются
  точной формулировкой, а лишние отвергаются как несуществующие.
* Стиль задаётся СТИЛИЗАЦИЕЙ ФОТОГРАФИИ до Kling. Промтом в Kling стиль не
  работает: с промтом и без него выход совпал (similarity 0.9618/0.9744/0.9445
  при негативном контроле 1.0000).
* Бренды, логотипы и надписи на одежде НЕ ГЕНЕРИРУЕМ (решение владельца).
  Запрет входит в промт стилизации строкой `NO_BRANDS_CLAUSE`, и ступень 2
  проверяет, что он там есть.

## ЧЕГО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ

Он не приём входов и не финальная сборка: обе ступени живут в соседних модулях
(`fork_intake`, `fork_finish`) и зовутся МЯГКИМ ИМПОРТОМ. Нет модуля — исход
`не смогли проверить` с именем модуля и подсказкой, чем его подменить, а НЕ
`не годно`: отсутствие соседа не есть брак продукта (Р1).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# Три исхода — ОДНИМ источником на весь проект (Е1). Копия строк здесь
# разъехалась бы с прибором молча, и вердикты перестали бы сравниваться.
from .fork_identity import FAIL, PASS, UNMEASURED, SAME_PERSON_MAX
# Код возврата тоже НЕ СВОЙ: `fork_video` уже отгрузил ровно это отображение,
# и второй способ узнать известное — дефект (Е1).
from .fork_video import EXIT_BY_OUTCOME

# ---------------------------------------------------------------------------
# ЧИСЛА. У каждого — происхождение (И4)
# ---------------------------------------------------------------------------

#: ИЗМЕРЕНО 22.08.2026 боевыми заказами (`work/bake_kling26_video.json`):
#: успешный ответ, латентность 107.4 с, выход 960x960 30 к/с.
KLING_ENDPOINT = "fal-ai/kling-video/v2.6/standard/motion-control"

#: ИЗМЕРЕНО щупом с негативным контролем: РОВНО эти три поля. Лишнее поле
#: отвергается как несуществующее, известное поле с мусором — точной
#: формулировкой (`character_orientation: 0` -> "Input should be 'image' or
#: 'video'", см. `work/bake_kling26_o0.json`). Это множество — константа-решение:
#: ступень 5 сверяет payload с ним и не пускает «ещё одно поле на всякий случай».
KLING_FIELDS = ("video_url", "image_url", "character_orientation")

#: ИЗМЕРЕНО там же: допустимые значения ровно два, сообщение об ошибке их
#: перечисляет само.
KLING_ORIENTATIONS = ("image", "video")

#: ВЫБРАНО (кем: владелец; из чего: движение берётся с драйвинга, а личность с
#: фотографии — значит ориентируемся по видео).
CHARACTER_ORIENTATION = "video"

#: ИЗМЕРЕНО ДВАЖДЫ ПО БАЛАНСУ fal, и второе измерение исправляет первое.
#:
#:   22.08, трёхсекундные заказы: $0.2100 за вызов standard.
#:   22.08, волна 1 продуктовой длины: баланс 10.8490375 -> 10.1490375 за ДВА
#:          пятисекундных вызова, то есть $0.3500 за вызов.
#:
#: Отсюда РАСЧЁТ (не с прогона, а делением измеренного): $0.0700 за секунду —
#: 0.21/3 и 0.35/5 дают одно и то же число. Значит Kling берёт посекундно, и
#: «цена вызова» без длины — величина без смысла. Именно на этом батч чуть не
#: посчитал 20 ячеек по $0.21 (=$4.20) вместо $7.00.
KLING_PRICE_PER_SECOND_USD = 0.07

#: ВЫБРАНО ВЛАДЕЛЬЦЕМ: «надо подогнать под 10 долларов, сохранив 5 секунд».
PRODUCT_SECONDS = 5.0

#: Цена ПРОДУКТОВОГО ролика. Имя оставлено прежним намеренно: соседи читают
#: цену через него (`E.KLING_PRICE_USD`), и правка доезжает до них сама (Е1).
KLING_PRICE_USD = round(KLING_PRICE_PER_SECOND_USD * PRODUCT_SECONDS, 4)

#: ИЗМЕРЕНО на ТРЁХСЕКУНДНОМ заказе: $2.6880 против $0.2100, в 12.8 раза.
#: Число историческое и с продуктовой длиной не сравнивается напрямую —
#: сравнивать надо посекундные, $0.8960 против $0.0700.
KLING_PRO_PRICE_USD = 2.6880
KLING_PRO_PRICE_3S_USD = 2.6880
KLING_PRICE_3S_USD = 0.21


def kling_price(seconds: float) -> float:
    """Цена заказа по длине. Мусор — исключение, а не догадка."""
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        raise TypeError(f"длина {seconds!r}: ждали число секунд")
    if seconds <= 0:
        raise ValueError(f"длина {seconds}: ждали больше нуля")
    return round(KLING_PRICE_PER_SECOND_USD * float(seconds), 4)

#: ИЗМЕРЕНО: 107.4 с и 184.1 с на двух боевых заказах. Полоса нужна затем,
#: чтобы таймаут ожидания не ставился «на глаз» — см. `KLING_WAIT_S`.
KLING_LATENCY_S = (107.4, 190.0)

#: РАСЧЁТ: верх измеренной полосы 190 с, умноженный на 8. Запас грубый
#: намеренно: очередь fal видели и на 15 минут, а лишнее ожидание стоит времени,
#: тогда как ранний обрыв стоит ДЕНЕГ — заказ уже оплачен.
KLING_WAIT_S = 1520

#: ИЗМЕРЕНО на восьми боевых заказах до 22.08 включительно: 960x960, 30 к/с.
#: Оставлено как ИСТОРИЯ, а не как требование — см. ниже, почему.
KLING_OUT_SIZE = (960, 960)
KLING_OUT_FPS = 30.0

#: ЧТО ПРОВЕРЯЕТСЯ НА САМОМ ДЕЛЕ: ВЕРТИКАЛЬНОСТЬ, а не конкретные числа.
#:
#: ПОЧЕМУ ПЕРЕПИСАНО. Первая редакция сверяла выход с `KLING_OUT_SIZE` и
#: забраковала боевой прогон, вернувший **816x1104** — то есть ВЕРТИКАЛЬ,
#: которой мы весь день добивались и считали недостижимой. Прибор был прав по
#: букве и неправ по существу: он сторожил старое знание и завернул самый
#: желанный исход. Это ровно тот дефект, из-за которого планка не должна
#: стоять на числе, если продукту важно СВОЙСТВО.
#:
#: ЧТО ИЗМЕНИЛОСЬ МЕЖДУ ЗАКАЗАМИ: на вход подавалось СТИЛИЗОВАННОЕ фото
#: 768x1024 (вертикальное) вместо квадратного. Похоже, Kling наследует
#: пропорции от ФОТОГРАФИИ, а не от драйвинга. НЕПРОВЕРЕНО: одно наблюдение,
#: одна пара. Проверяется одним заказом с квадратным фото.
#:
#: ВЫБРАНО 1.0: выше — горизонталь, и это брак для вертикального продукта.
#: Квадрат (ровно 1.0) допускается: восемь заказов его давали, и он режется
#: в 9:16 кропом, просто дороже по потерям.
FRAME_SUFFIXES = {".png", ".jpg", ".jpeg"}

OUT_RATIO_MAX = 1.0

#: ИЗМЕРЕНО: гейт Kling отвергает «Video duration can not less than 3s», и все
#: три обхода проверены и не работают (растяжка частоты вернула 88 кадров
#: вместо 15, добивка заморозкой была оживлена моделью, посценный рендер не
#: пускает гейт). Значит это КРИТЕРИЙ ПРИЁМА окна, а не пожелание.
MIN_SCENE_S = 3.0

#: ВЫБРАНО (кем: поток E2E; из чего: `pro` исключён владельцем навсегда, и
#: запрет обязан быть машинным — строка в правилах не роняет заказ).
FORBIDDEN_TIERS = ("pro",)

#: ВЫБРАНО ВЛАДЕЛЬЦЕМ ГЛАЗАМИ 22.08.2026, ВОПРЕКИ ЧИСЛУ: `nanobanana-2` через
#: `pollinations.compose` ДВУМЯ картинками (первая на личность, вторая на
#: стиль).
#:
#: ПОЧЕМУ НЕ ПОБЕДИТЕЛЬ ПО ЧИСЛУ. Мера попадания в стиль дала `gpt-image-2`
#: 0.8801 против 0.8156 у `nanobanana-2` — и он выиграл ИМЕННО ПОТОМУ, ЧТО
#: СКОПИРОВАЛ ЛИШНЕЕ: оливковое платье с поясом вместо серой майки клиента,
#: наклон корпуса с рукой на бедре вместо прямой стойки, чужое кадрирование.
#: От фотографии клиента осталось одно лицо. `nanobanana-2` удержал майку и
#: позу, взяв из референса только небо, свет и цветокор.
#:
#: ОТКУДА ДЕФЕКТ МЕРЫ. Она построена на цвете и фактуре, а одежда и поза —
#: тоже цвет и фактура. Значит мера НАГРАЖДАЕТ ПЕРЕРИСОВКУ: чем больше модель
#: заменила, тем выше балл. Нужного здесь — «сходство ПО ОДНИМ осям при
#: РАЗЛИЧИИ по другим» — односторонняя мера выразить не может.
#:
#: ЧТО ПРОБОВАЛИ И НЕ СМОГЛИ (Р1, И6). Встречную ось хотели снять `pose`:
#: расстояние позы выхода до фото клиента против расстояния до референса.
#: `pose.pose_distance` вернул `None` на всех вариантах И НА НЕГАТИВНОМ
#: КОНТРОЛЕ (фото само с собой) — значит прибор не измеряет, а не «позы
#: совпали». Ось остаётся за глазом оператора, как и телосложение.
STYLE_MODEL = "nanobanana-2"
STYLE_ROUTE = "pollinations.compose"
STYLE_IMAGES = 2

#: ИЗМЕРЕНО тем же замером — числа отвергнутого победителя, оставлены как
#: происхождение планки и как памятник тому, что число тут не решает.
STYLE_HIT_REFERENCE = 0.8156          # nanobanana-2, ВЫБРАННЫЙ
STYLE_HIT_REJECTED = 0.8801           # gpt-image-2, отвергнут глазами
STYLE_FLOOR_REFERENCE = 0.6409
STYLE_TEXT_ROUTE_REFERENCE = 0.6773

#: ВЫБРАНО 0.05 (кем: поток E2E; из чего: на замере стилизаторов ОТВЕРГНУТЫЙ
#: текстовый путь дал 0.6773 при поле 0.6409, то есть шум прибора здесь
#: +0.0364. Планка обязана быть ВЫШЕ шума, иначе шум пройдёт как стиль, и
#: заметно ниже победителя +0.2392, иначе не пройдёт ничего).
#: ЧЕГО ЭТО ЧИСЛО НЕ ЗНАЕТ: оно снято на приборе владельца замера. У другого
#: прибора другая шкала — поэтому ПОЛ СЧИТАЕТСЯ НА МЕСТЕ, тем же прибором, и
#: сравнивается только с ним. DEBT(2026-08-22): запас в долях, а не в сигмах.
STYLE_MARGIN_MIN = 0.05

#: ВЫБРАНО (кем: владелец; из чего: монтажный рез внутри одной сцены — это
#: дефект генерации, а не стиль). Планка самого прибора — `fork_looper.CUT_JUMP`
#: (4.0), и она берётся ОТТУДА, а не копируется сюда.
MAX_CUTS_OUT = 0

#: Запрет брендов. Решение владельца, и оно в ПРОМТЕ, а не в голове: ступень 2
#: проверяет наличие этой строки и краснеет, если её вынули.
NO_BRANDS_CLAUSE = ("no brand names, no logos, no lettering or text on "
                    "clothing or in the frame")

#: Роли картинок в `compose`. ИЗМЕРЕНО чужим замером и подтверждено нашим:
#: модель держит роли, если они названы ПОЗИЦИЕЙ в промте.
#:
#: ЧТО ИМЕННО БЕРЁТСЯ ИЗ ВТОРОЙ КАРТИНКИ, названо СПИСКОМ, а не словом
#: «стиль». Причина ИЗМЕРЕНА боевым прогоном 22.08: стилизация надела на
#: клиента ОЧКИ со стилевого референса, ArcFace дал 0.3928 при планке 0.35 и
#: стенд встал. Это не «личность потеряна» — это ЛИЦО ЗАКРЫТО: ArcFace
#: опирается на область глаз, и любая окклюзия раздувает расстояние даже на
#: том же человеке. Тот же дефект раньше проявился одеждой и позой
#: (`gpt-image-2`, забракован владельцем глазами).
ROLE_CLAUSE = ("keep the person from the FIRST image unchanged — same face, "
               "same identity, same clothing, same pose, same accessories; "
               "take ONLY the lighting, colour grade, background and "
               "photographic look from the SECOND image")

#: ЛЕСТНИЦА ArcFace. ИЗМЕРЕНО на нашем материале и живёт ОДНИМ местом (Е1):
#: 0.0652 — то же лицо после стилизации; 0.7137 — отбракованный референс,
#: то есть уже ДРУГОЙ человек; 1.0217 — заведомо чужой (актёр драйвинга
#: против фото клиента).
LADDER_SAME = 0.0652
LADDER_REJECTED = 0.7137
LADDER_STRANGER = 1.0217

#: РЕШЕНИЕ ВЛАДЕЛЬЦА 22.08, ПОСЛЕ боевого прогона: «очки сели — это не баг, а
#: фича; прибор сработал, это хорошо, но в целом не страшно, если со стиля
#: берётся одежда». Значит ось личности на СТИЛИЗОВАННОМ фото перестаёт быть
#: гейтом и становится трёхисходной по лестнице:
#:
#:   <= 0.35            годно — лицо на месте
#:   0.35 .. 0.7137     НЕ СМОГЛИ — лицо частично закрыто аксессуаром;
#:                      ArcFace здесь не судья, судит оператор глазами
#:   >= 0.7137          НЕ ГОДНО — это уже другой человек, а не аксессуар
#:
#: ПОЧЕМУ НЕ ПРОСТО ПОДНЯТЬ ПЛАНКУ. Поднятая планка перестала бы ловить и
#: настоящую подмену личности. Средняя полоса — это честный третий исход:
#: «прибор не может судить», а не «плохо» и не «хорошо».
#:
#: ЧЕМ ЭТО ВЫЗВАНО, ИЗМЕРЕНО: стилизация надела очки со стилевого референса,
#: ArcFace дал 0.3928. Окклюзия области глаз раздувает расстояние даже на том
#: же человеке — то есть 0.3928 означало «лицо закрыто», а не «личность иная».

#: Запрет протечки внешности из стилевого референса. Отдельной константой от
#: `NO_BRANDS_CLAUSE`, потому что это РАЗНЫЕ решения с разной историей: бренды
#: запрещены владельцем как продуктовая позиция, а аксессуары — как ответ на
#: измеренный дефект. Слипшись в одну строку, они потеряли бы обе истории.
NO_LOOK_TRANSFER_CLAUSE = ("do not copy any garment, accessory, eyewear, "
                           "headwear, hairstyle or pose from the second "
                           "image; the second image is a colour and lighting "
                           "reference only")

#: Ступени по порядку. Список — это и есть порядок прогона (Е1): печать,
#: остановка и отчёт берут имена отсюда, а не из своих строк.
STAGES = (
    "1 приём трёх входов",
    "2 стилизация фото клиента",
    "3 приёмка стилизованного фото",
    "4 окно драйвинга и нарезка",
    "5 загрузка входов и вызов Kling",
    "6 приёмка выхода",
    "7 финальная сборка",
    "8 отчёт",
)


# ---------------------------------------------------------------------------
# Служебное: печать по ходу, мягкий импорт, счёт исходов
# ---------------------------------------------------------------------------

def say(text: str, *, log=None) -> None:
    """Строка в stderr НЕМЕДЛЕННО. Молчание длинного прогона уже стоило прогона."""
    stream = sys.stderr if log is None else log
    stream.write(text + "\n")
    flush = getattr(stream, "flush", None)
    if flush:
        flush()


def verdict(checked: int, violations: int, unmeasured: int) -> str:
    """Три исхода из трёх чисел. Ноль проверок — НЕ успех (Р2).

    Порядок ветвей — сам по себе решение: `не годно` перебивает `не смогли`,
    потому что найденное нарушение не перестаёт быть нарушением от того, что
    рядом что-то не измерилось.
    """
    if checked <= 0:
        return UNMEASURED
    if violations > 0:
        return FAIL
    if unmeasured > 0:
        return UNMEASURED
    return PASS


def _result(stage: str, checks: list, *, note: str = "", **extra) -> dict:
    """Ступень из списка проверок. Каждая проверка — `(имя, исход, строка)`."""
    checked = sum(1 for c in checks if c[1] in (PASS, FAIL))
    violations = sum(1 for c in checks if c[1] == FAIL)
    unmeasured = sum(1 for c in checks if c[1] == UNMEASURED)
    return {"stage": stage, "outcome": verdict(checked, violations, unmeasured),
            "checked": checked, "violations": violations,
            "unmeasured": unmeasured,
            "checks": [{"name": n, "outcome": o, "note": t} for n, o, t in checks],
            "note": note, **extra}


def line(res: dict) -> str:
    """Одна строка ступени: вердикт и числа РЯДОМ с ним (Р2)."""
    return (f"[{res['outcome']:<18}] {res['stage']:<34} "
            f"проверено {res['checked']}, нарушений {res['violations']}, "
            f"не смогли {res['unmeasured']}"
            + (f" | {res['note']}" if res.get("note") else ""))


def soft_import(name: str):
    """Соседний модуль или ПОНЯТНЫЙ отказ. Никогда не исключение наружу.

    Возвращает `(модуль, None)` либо `(None, причина)`. Причина написана для
    человека и называет, чем модуль подменить: в тестах и на стенде соседа
    подменяют параметром, а не ожиданием.
    """
    try:
        mod = __import__(f"ball_reel.{name}", fromlist=["*"])
    except ImportError as exc:
        return None, (f"модуля ball_reel.{name} нет ({exc}). Это НЕ брак "
                      f"продукта: ступень не измерена. Подменить можно "
                      f"параметром прогона")
    return mod, None


def entry_point(mod, candidates):
    """Первая существующая функция из списка имён, либо отказ с перечнем.

    Соседние модули пишутся ПАРАЛЛЕЛЬНО, и их точное имя входа неизвестно.
    Догадка «наверное, `run`» дала бы `AttributeError` посреди прогона; здесь
    перебор назван вслух и попадает в отчёт.
    """
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, name, None
    return None, None, (f"в {mod.__name__} нет ни одной из точек входа "
                        f"{list(candidates)}: звать нечего")


def _call(fn, kwargs: dict, positional: tuple):
    """Вызов соседа: сначала по именам, при несовпадении — позиционно.

    Чужая сигнатура неизвестна, и `TypeError` от несовпадения имён НЕ
    отличается по типу от `TypeError` внутри чужой функции — поэтому вторая
    попытка делается только на ошибке про аргументы самого вызова.
    """
    try:
        return fn(**kwargs)
    except TypeError as exc:
        if "argument" not in str(exc) and "parameter" not in str(exc):
            raise
        return fn(*positional)


def outcome_of(reply, *, what: str) -> tuple:
    """Вердикт из ответа соседа. Ответ без вердикта — «не смогли», не «годно».

    Сосед, вернувший `None` или строку, НЕ проверен: считать это успехом —
    ровно тот способ, которым «не смогли измерить» превращается в «прошло».
    """
    if isinstance(reply, dict) and reply.get("outcome") in (PASS, FAIL, UNMEASURED):
        return reply["outcome"], str(reply.get("note") or "")[:400]
    return UNMEASURED, (f"{what} ответил {type(reply).__name__} без поля "
                        f"outcome: вердикта нет, судить нечем")


def refuse_pro(endpoint: str) -> None:
    """Сторож денег. `pro` исключён владельцем НАВСЕГДА, и запрет машинный.

    Ловится по сегменту пути, а не по подстроке: `.../standard/...` не должен
    краснеть из-за слова внутри другого слова.
    """
    parts = str(endpoint).split("/")
    hit = [p for p in parts if p in FORBIDDEN_TIERS]
    if hit:
        # Сравниваются ПОСЕКУНДНЫЕ цены, а не цены заказов. Обе цены pro и
        # standard измерены на ТРЁХСЕКУНДНЫХ заказах, а продуктовая длина
        # теперь 5 с: делить $2.688 на $0.35 значит сравнивать три секунды с
        # пятью и получить 7.7 раза вместо настоящих 12.8.
        pro_per_s = round(KLING_PRO_PRICE_3S_USD / 3.0, 4)
        raise ValueError(
            f"эндпоинт {endpoint} содержит {hit}: {FORBIDDEN_TIERS} исключены "
            f"владельцем НАВСЕГДА (${pro_per_s} против "
            f"${KLING_PRICE_PER_SECOND_USD} за секунду, в "
            f"{round(pro_per_s / KLING_PRICE_PER_SECOND_USD, 1)} раза; лейбл "
            f"всё равно не выжил, фон получил дорисованную анимацию)")


# ---------------------------------------------------------------------------
# Прибор стиля. Свой, дешёвый, БЕЗ СЕТИ — и с обоими контролями (И5)
# ---------------------------------------------------------------------------

#: ВЫБРАНО 8 (кем: поток E2E; из чего: 8^3 = 512 корзин на картинку — крупнее
#: не различает палитры, мельче считает шум съёмки). Мутация в обе стороны —
#: в тестах модуля.
PALETTE_BINS = 8
#: ВЫБРАНО 256: сторона, к которой приводятся ОБЕ картинки, чтобы сравнивалась
#: палитра, а не разрешение.
PALETTE_SIDE = 256


def shipped_similarity(left, right) -> float | None:
    """Прибор попадания в стиль, ОДИН на весь конвейер. `None` — не смогли.

    ПОЧЕМУ ОН ЗДЕСЬ ОДИН. 22.08 в дереве оказалось ДВА способа померить одно
    и то же: этот и `creative_eval.style.similarity` из внешнего пакета,
    которым сняты 0.8801/0.8156/0.6409. Их шкалы РАЗНЫЕ (пол 0.3170 против
    0.6409), и смешивать их нельзя — это ровно тот дефект, от которого
    защищает Е1. Пакет теперь виден среде, поэтому ОТГРУЖАЕТСЯ ВНЕШНИЙ, а
    этот остаётся запасным на случай, когда пакета нет: без него ступень
    ответила бы «не смогли» и стенд встал бы, как встал 22.08.

    Порядок: сперва внешний, при неудаче импорта — свой. Какой сработал,
    видно в `note` ступени, а не по догадке.
    """
    try:
        from creative_eval.style import similarity as _external  # noqa: PLC0415
    except Exception:                                            # noqa: BLE001
        return palette_similarity(left, right)
    try:
        return float(_external(str(left), str(right)))
    except Exception:                                            # noqa: BLE001
        return palette_similarity(left, right)


def similarity_source() -> str:
    """Каким прибором меряем СЕЙЧАС. Печатается в отчёт (Е2: верим свидетельству)."""
    try:
        from creative_eval.style import similarity  # noqa: F401,PLC0415
        return "creative_eval.style.similarity (внешний, отгружаемый)"
    except Exception:                               # noqa: BLE001
        return "palette_similarity (запасной: внешнего пакета нет)"


def palette_similarity(left, right) -> float | None:
    """ЗАПАСНОЙ прибор: косинус между палитрами. `None` — не смогли.

    Используется, только когда внешнего пакета нет в среде. Его числа
    ИЗМЕРЕНЫ 22.08.2026 на нашем же материале:

        styleref_bluesky против st_img_gpt-image-2 (стилизованное)   0.8547
        styleref_bluesky против fork_ref_gym       (НЕстилизованное) 0.3170
        styleref_bluesky сам с собой                                 1.0000

    Оба контроля на месте (И5): прибор умеет сказать «нет» (0.3170 на
    нестилизованном) и умеет шевельнуться (1.0 на самом себе). Именно поэтому
    ПОЛ СЧИТАЕТСЯ НА МЕСТЕ тем же прибором: числа двух приборов несравнимы, а
    отношение «стилизованное дальше от пола, чем нестилизованное» — сравнимо.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    try:
        vecs = []
        for p in (left, right):
            arr = np.asarray(Image.open(str(p)).convert("RGB")
                             .resize((PALETTE_SIDE, PALETTE_SIDE)),
                             dtype="float64").reshape(-1, 3)
            hist, _ = np.histogramdd(
                arr, bins=(PALETTE_BINS,) * 3, range=((0, 255),) * 3)
            v = hist.ravel()
            norm = float(np.linalg.norm(v))
            if norm <= 0:
                return None
            vecs.append(v / norm)
    except Exception:                                    # noqa: BLE001
        return None
    return round(float(vecs[0] @ vecs[1]), 4)


# ---------------------------------------------------------------------------
# Умолчания, которые ХОДЯТ В СЕТЬ И СТОЯТ ДЕНЕГ. В тестах не зовутся никогда
# ---------------------------------------------------------------------------

def live_upload(path) -> str:
    """Файл -> публичная ссылка на fal. НЕПРОВЕРЕНО в этой смене (денег не тратили).

    `fal_client.upload_file` — путь, которым 22.08 уехали боевые входы
    (ссылки вида `https://v3b.fal.media/files/...`, см. `work/bake_urls.json`).
    """
    import fal_client                                    # noqa: PLC0415

    return fal_client.upload_file(str(path))


def live_kling(*, video_url: str, image_url: str, character_orientation: str,
               out_path, endpoint: str = KLING_ENDPOINT, poll_s: int = 15,
               wait_s: int = KLING_WAIT_S) -> str:
    """Заказ у fal и скачивание выхода. ПЛАТНЫЙ путь: ровно $0.21 за вызов.

    НЕПРОВЕРЕНО в этой смене: смена работала без единого цента, и вызов не
    исполнялся ни разу (Ц4). Форма запроса и разбор ответа списаны с боевого
    прогона 22.08 (`work/bake_order.py`, `work/bake_kling26_video.json`), где
    ответ пришёл как `{"video": {"url": ...}}` за 107.4 с.
    """
    import os                                            # noqa: PLC0415
    import urllib.request                                # noqa: PLC0415

    refuse_pro(endpoint)
    key = os.environ.get("FAL_KEY")
    if not key:
        raise RuntimeError("FAL_KEY не задан: заказывать нечем")
    head = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    payload = {"video_url": video_url, "image_url": image_url,
               "character_orientation": character_orientation}

    def _req(url, data=None):
        req = urllib.request.Request(url, data=data, headers=head)
        with urllib.request.urlopen(req, timeout=60) as fh:
            return json.loads(fh.read().decode() or "{}")

    app = "/".join(endpoint.split("/")[:2])
    sub = _req(f"https://queue.fal.run/{endpoint}",
               data=json.dumps(payload).encode())
    rid = sub["request_id"]
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(poll_s)
        st = _req(f"https://queue.fal.run/{app}/requests/{rid}/status")
        if st.get("status") in ("COMPLETED", "FAILED", "ERROR"):
            break
    res = _req(f"https://queue.fal.run/{app}/requests/{rid}")
    url = (res.get("video") or {}).get("url")
    if not url:
        raise RuntimeError(f"в ответе нет ссылки на видео: {str(res)[:300]}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as fh:
        Path(out_path).write_bytes(fh.read())
    return str(out_path)


def live_stylize(*, person, style, prompt: str, out_path,
                 model: str = STYLE_MODEL) -> str:
    """Стилизация ДВУМЯ картинками через победителя замера. Ходит в сеть.

    Две ссылки, а не одна, и порядок значим: первая — личность, вторая — стиль
    (ИЗМЕРЕНО: роли держатся, когда названы позицией, см. `ROLE_CLAUSE`).
    """
    from . import pollinations                           # noqa: PLC0415

    urls = [pollinations.upload(person), pollinations.upload(style)]
    if len(urls) != STYLE_IMAGES:
        raise RuntimeError(f"нужно ровно {STYLE_IMAGES} ссылки, вышло {len(urls)}")
    return pollinations.compose(prompt, urls, out_path, model=model)


# ---------------------------------------------------------------------------
# Ступень 1. Приём трёх входов
# ---------------------------------------------------------------------------

def file_fact(path, what: str) -> tuple:
    """Дешёвая проверка раньше дорогой (П2): файл есть и он не пуст."""
    p = Path(path)
    if not p.exists():
        return (what, FAIL, f"{p} нет на диске")
    size = p.stat().st_size
    if size == 0:
        return (what, FAIL, f"{p} пуст (0 Б)")
    return (what, PASS, f"{p} — {size} Б")


#: Форма соседа-приёмщика: три входа — три функции. ИЗМЕРЕНО чтением
#: `fork_intake` 22.08.2026, а не угадано: у него нет одной общей точки входа,
#: и «наверное, `run`» дало бы `AttributeError` посреди прогона.
INTAKE_TRIO = ("photo_intake", "style_intake", "driving_intake")


def _numbers_of(reply) -> str:
    """Числа соседа рядом с его вердиктом (Р2). Нет чисел — так и сказано."""
    if not isinstance(reply, dict):
        return ""
    if any(k not in reply for k in ("checked", "violations", "unmeasured")):
        return ""
    return (f"проверено {reply['checked']}, нарушений {reply['violations']}, "
            f"не смогли {reply['unmeasured']}; ")


def stage_intake(*, client_photo, style_ref, driving, intake=None,
                 driving_frames=None, card_reader=None) -> dict:
    """Три входа на месте, и сосед `fork_intake` их принял.

    Порядок именно такой: сначала СВОЯ проверка существования (1 мс и не
    зависит ни от кого), потом делегирование. Иначе отсутствие соседа скрыло
    бы отсутствие файла.

    `driving_frames` — уже распакованные кадры драйвинга. Без них приёмщик
    честно отвечает «не смогли» по четырём осям из пяти: он НЕ распаковывает
    сам (второй распаковщик в проекте был бы вторым способом узнать известное).
    """
    checks = [file_fact(client_photo, "фото клиента"),
              file_fact(style_ref, "стилевой референс"),
              file_fact(driving, "драйвинг")]
    note = ""
    if intake is None:
        mod, why = soft_import("fork_intake")
        if mod is None:
            checks.append(("приём соседним модулем", UNMEASURED, why))
            return _result(STAGES[0], checks, note=why)
        trio = [getattr(mod, n, None) for n in INTAKE_TRIO]
        if all(callable(f) for f in trio):
            photo, style, drive = trio
            # `card_reader` пробрасывается ИМЕННО в приём стиля: без него у
            # соседа нет чем прочитать карточку (`creative_eval` в среде нет),
            # и ступень честно встаёт на «не смогли». ИЗМЕРЕНО 22.08 живым
            # прогоном соседа: photo годно (3 проверки), driving годно (747),
            # style — «не смогли», и весь стенд останавливается на ступени 1.
            calls = (("приём фото клиента", photo, (str(client_photo),), {}),
                     ("приём стилевого референса", style, (str(style_ref),),
                      {} if card_reader is None else {"card_reader": card_reader}),
                     ("приём драйвинга", drive, (str(driving), driving_frames), {}))
            for name, fn, args, extra in calls:
                try:
                    reply = fn(*args, **extra)
                except Exception as exc:             # noqa: BLE001
                    checks.append((name, UNMEASURED, f"{type(exc).__name__}: {exc}"))
                    continue
                out, why = outcome_of(reply, what=f"fork_intake.{fn.__name__}")
                checks.append((name, out, _numbers_of(reply) + why))
            return _result(STAGES[0], checks, note="fork_intake: " +
                           ", ".join(INTAKE_TRIO))
        intake, name, why = entry_point(
            mod, ("accept", "intake", "take", "check", "run"))
        if intake is None:
            checks.append(("приём соседним модулем", UNMEASURED, why))
            return _result(STAGES[0], checks, note=why)
        note = f"fork_intake.{name}"
    try:
        reply = _call(intake,
                      {"client_photo": str(client_photo),
                       "style_ref": str(style_ref), "driving": str(driving)},
                      (str(client_photo), str(style_ref), str(driving)))
    except Exception as exc:                             # noqa: BLE001
        checks.append(("приём соседним модулем", UNMEASURED,
                       f"{type(exc).__name__}: {exc}"))
        return _result(STAGES[0], checks, note="сосед упал: это НЕ «не годно»")
    out, why = outcome_of(reply, what="fork_intake")
    checks.append(("приём соседним модулем", out, why))
    return _result(STAGES[0], checks, note=note or why)


# ---------------------------------------------------------------------------
# Ступень 2. Стилизация фотографии клиента
# ---------------------------------------------------------------------------

def style_prompt(style_ref, *, card_reader=None) -> dict:
    """Промт стилизации: роли, стиль словами (если читается) и запрет брендов.

    Стиль едет КАРТИНКОЙ, а не текстом: текстовый путь измерен и отвергнут
    (0.6773 против пола 0.6409 — шум). Поэтому словесная карточка здесь
    НЕОБЯЗАТЕЛЬНА: не прочиталась — промт всё равно полон, и это не «не смогли»,
    а сознательное умолчание.
    """
    from . import fork_style_prompt                      # noqa: PLC0415

    card = fork_style_prompt.from_image(style_ref, reader=card_reader)
    words = card.get("prompt")
    parts = ([ROLE_CLAUSE] + ([words] if words else [])
             + [NO_LOOK_TRANSFER_CLAUSE, NO_BRANDS_CLAUSE])
    return {"prompt": ", ".join(parts), "card_outcome": card.get("outcome"),
            "card_note": card.get("note"), "words": words}


def stage_stylize(*, client_photo, style_ref, out_path, stylize=None,
                  card_reader=None, prompt=None) -> dict:
    """Фото клиента + стилевой референс -> стилизованное фото.

    `prompt` — точка внедрения и одновременно негативный контроль сторожа
    брендов: подать промт БЕЗ запрета и увидеть красное — единственный способ
    отличить работающую проверку от строки, которая всегда зелена.
    """
    built = ({"prompt": prompt, "card_note": "промт подан снаружи"}
             if prompt is not None else style_prompt(style_ref,
                                                     card_reader=card_reader))
    prompt = built["prompt"]
    checks = [("запрет брендов в промте",
               PASS if NO_BRANDS_CLAUSE in prompt else FAIL,
               NO_BRANDS_CLAUSE if NO_BRANDS_CLAUSE in prompt
               else "запрет вынули из промта: бренды поедут в кадр")]
    stylize = live_stylize if stylize is None else stylize
    t0 = time.perf_counter()
    try:
        got = stylize(person=str(client_photo), style=str(style_ref),
                      prompt=prompt, out_path=str(out_path))
    except Exception as exc:                             # noqa: BLE001
        checks.append(("стилизация", UNMEASURED, f"{type(exc).__name__}: {exc}"))
        return _result(STAGES[1], checks, prompt=prompt,
                       note="стилизатор не ответил: измерять нечего")
    checks.append(("стилизация", PASS,
                   f"{STYLE_ROUTE}/{STYLE_MODEL}, {STYLE_IMAGES} картинки, "
                   f"{round(time.perf_counter() - t0, 1)} с"))
    checks.append(file_fact(got or out_path, "стилизованное фото"))
    return _result(STAGES[1], checks, styled=str(got or out_path),
                   prompt=prompt, note=str(built["card_note"] or "")[:160])


# ---------------------------------------------------------------------------
# Ступень 3. Приёмка стилизованного фото: стиль И личность
# ---------------------------------------------------------------------------

def stage_style_acceptance(*, styled, style_ref, client_photo, operator_ok_identity=False,
                           similarity=None, distances=None) -> dict:
    """Попал ли в стиль (против ПОЛА) и уцелела ли личность (против планки).

    ПОЛ — негативный контроль ступени, и он обязателен: `similarity(стиль,
    НЕстилизованное фото)`. Всё, что не бьёт пол с запасом `STYLE_MARGIN_MIN`,
    стилем НЕ ЯВЛЯЕТСЯ — так отвергнут текстовый путь, давший +0.0364 к полу.
    """
    similarity = shipped_similarity if similarity is None else similarity
    checks, numbers = [], {}

    floor = similarity(style_ref, client_photo)
    hit = similarity(style_ref, styled)
    numbers["floor"] = floor
    numbers["hit"] = hit
    if floor is None or hit is None:
        checks.append(("попадание в стиль", UNMEASURED,
                       f"прибор стиля не дал числа: пол={floor}, попадание={hit}"))
    else:
        margin = round(hit - floor, 4)
        numbers["margin"] = margin
        ok = margin >= STYLE_MARGIN_MIN
        checks.append(("попадание в стиль", PASS if ok else FAIL,
                       f"попадание {hit} при поле {floor} (пол = стиль против "
                       f"НЕстилизованного фото), запас {margin} при планке "
                       f"{STYLE_MARGIN_MIN}"))

    distances = _default_distances() if distances is None else distances
    try:
        d = distances([str(styled)], str(client_photo))
    except Exception as exc:                             # noqa: BLE001
        checks.append(("личность на стилизованном", UNMEASURED,
                       f"{type(exc).__name__}: {exc}"))
        return _result(STAGES[2], checks, numbers=numbers)
    numbers["identity_median"] = d.get("median")
    numbers["identity_bar"] = SAME_PERSON_MAX
    if d.get("outcome") == UNMEASURED:
        checks.append(("личность на стилизованном", UNMEASURED,
                       str(d.get("note"))[:300]))
    else:
        med = d.get("median")
        if med is None:
            checks.append(("личность на стилизованном", UNMEASURED,
                           "медианы нет: судить нечем"))
        elif med <= SAME_PERSON_MAX:
            checks.append(("личность на стилизованном", PASS,
                           f"медиана {med} при планке {SAME_PERSON_MAX} "
                           f"(лестница: {LADDER_SAME} тот же, "
                           f"{LADDER_REJECTED} другой, {LADDER_STRANGER} чужой)"))
        elif med < LADDER_REJECTED:
            # Средняя полоса: прибор не судья. Пройти её можно ТОЛЬКО явным
            # допуском оператора, и допуск виден в отчёте (Е2: верим
            # свидетельству). Молча она не проходится никогда.
            band = (f"медиана {med} между планкой {SAME_PERSON_MAX} и ступенью "
                    f"«другой человек» {LADDER_REJECTED}: лицо ЧАСТИЧНО "
                    f"ЗАКРЫТО или изменено аксессуаром — ArcFace здесь НЕ СУДЬЯ")
            if operator_ok_identity:
                checks.append(("личность на стилизованном", PASS,
                               band + "; ДОПУЩЕНО ОПЕРАТОРОМ явным флагом"))
            else:
                checks.append(("личность на стилизованном", UNMEASURED,
                               band + ", судит оператор глазами"))
        else:
            checks.append(("личность на стилизованном", FAIL,
                           f"медиана {med} выше ступени «другой человек» "
                           f"{LADDER_REJECTED}: это подмена личности, а не "
                           f"аксессуар (лестница: {LADDER_SAME} тот же, "
                           f"{LADDER_STRANGER} чужой)"))
    return _result(STAGES[2], checks, numbers=numbers)


def _default_distances():
    from . import fork_identity                          # noqa: PLC0415

    return fork_identity.distances


# ---------------------------------------------------------------------------
# Ступень 4. Окно драйвинга по номерам кадров и нарезка
# ---------------------------------------------------------------------------

def cut_argv(src, dst, *, first: int, last: int, fps: float, exe: str) -> list:
    """Рез: старт по времени, длина ПО КАДРАМ.

    ИЗМЕРЕНО 22.08 (см. `fork_bench/fork_bench_input.py`): `-t` по секундам
    даёт 103 кадра вместо заказанного 101, а `-frames:v` — ровно 101. Здесь
    та же форма, но окно ПАРАМЕТР, а не константа файла-соседа.
    """
    return [exe, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{first / fps:.6f}", "-i", str(src),
            "-frames:v", str(last - first + 1),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-an", str(dst)]


def _decoded_frames(path, exe: str):
    out = subprocess.run([exe, "-hide_banner", "-i", str(path), "-map", "0:v:0",
                          "-f", "null", "-"], capture_output=True, text=True)
    for row in reversed(out.stderr.splitlines()):
        if "frame=" in row:
            try:
                return int(row.split("frame=")[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def stage_window(*, driving, first: int, last: int, out_path,
                 probe=None, cutter=None) -> dict:
    """Окно по номерам кадров, проверка длины и рез с ПЕРЕСЧЁТОМ кадров.

    Пересчёт после реза — не украшение: ИЗМЕРЕНО, что ffmpeg с точкой реза за
    концом ролика молча отдаёт файл ЦЕЛИКОМ, и без пересчёта это уехало бы в
    оплаченный заказ.
    """
    checks, numbers = [], {"first": first, "last": last}
    probe = _default_probe() if probe is None else probe
    info = probe(str(driving))
    fps = info.get("fps")
    total = info.get("frames")
    numbers["fps"] = fps
    numbers["source_frames"] = total
    if not fps or not total:
        checks.append(("опрос драйвинга", UNMEASURED, str(info.get("note"))[:200]))
        return _result(STAGES[3], checks, numbers=numbers)
    checks.append(("опрос драйвинга", PASS, str(info.get("note"))[:200]))

    want = last - first + 1
    numbers["want_frames"] = want
    inside = 0 <= first <= last < total
    checks.append(("окно внутри драйвинга", PASS if inside else FAIL,
                   f"кадры {first}..{last} при {total} кадрах в ролике"))
    seconds = round(want / fps, 3)
    numbers["seconds"] = seconds
    long_enough = seconds >= MIN_SCENE_S
    checks.append(("сцена не короче порога", PASS if long_enough else FAIL,
                   f"{seconds} с при пороге {MIN_SCENE_S} с (гейт Kling: "
                   f"«Video duration can not less than 3s»)"))
    if not (inside and long_enough):
        return _result(STAGES[3], checks, numbers=numbers)

    if cutter is None:
        import shutil                                    # noqa: PLC0415
        exe = shutil.which("ffmpeg")
        if not exe:
            checks.append(("рез", UNMEASURED, "ffmpeg не найден: резать нечем"))
            return _result(STAGES[3], checks, numbers=numbers)

        def cutter(src, dst, first=first, last=last, fps=fps, exe=exe):
            run = subprocess.run(cut_argv(src, dst, first=first, last=last,
                                          fps=fps, exe=exe),
                                 capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(f"ffmpeg вернул {run.returncode}: "
                                   f"{run.stderr[-300:]}")
            return {"path": str(dst), "frames": _decoded_frames(dst, exe)}

    try:
        got = cutter(str(driving), str(out_path))
    except Exception as exc:                             # noqa: BLE001
        checks.append(("рез", UNMEASURED, f"{type(exc).__name__}: {exc}"))
        return _result(STAGES[3], checks, numbers=numbers)
    got = got if isinstance(got, dict) else {"path": str(out_path), "frames": None}
    numbers["cut_frames"] = got.get("frames")
    if got.get("frames") is None:
        checks.append(("кадров в куске", UNMEASURED,
                       "пересчитать кадры не вышло: рез не подтверждён"))
    else:
        checks.append(("кадров в куске",
                       PASS if got["frames"] == want else FAIL,
                       f"{got['frames']} при заказанных {want}"))
    checks.append(file_fact(got.get("path") or out_path, "кусок драйвинга"))
    return _result(STAGES[3], checks, numbers=numbers,
                   window=str(got.get("path") or out_path))


def _default_probe():
    from . import fork_video                             # noqa: PLC0415

    return fork_video.probe


# ---------------------------------------------------------------------------
# Ступень 5. Загрузка входов и вызов Kling. ЕДИНСТВЕННОЕ место, где тратятся деньги
# ---------------------------------------------------------------------------

def kling_payload(*, video_url: str, image_url: str,
                  character_orientation: str = CHARACTER_ORIENTATION) -> dict:
    """Ровно три поля и ни одним больше. Значение ориентации — из измеренных."""
    if character_orientation not in KLING_ORIENTATIONS:
        raise ValueError(f"character_orientation={character_orientation!r}: у "
                         f"эндпоинта ровно {list(KLING_ORIENTATIONS)} "
                         f"(ИЗМЕРЕНО щупом)")
    return {"video_url": video_url, "image_url": image_url,
            "character_orientation": character_orientation}


def stage_kling(*, styled, window, out_path, upload=None, kling=None,
                endpoint: str = KLING_ENDPOINT,
                orientation: str = CHARACTER_ORIENTATION) -> dict:
    """Две загрузки и один платный вызов. Любой отказ — «не смогли», не «не годно».

    ПОЧЕМУ ИМЕННО ТАК. Упавшая сеть, пустой баланс и очередь fal ничего не
    говорят о качестве продукта. Свернуть их в `не годно` значит объявить
    брак там, где не было измерения, и снять это тем же способом, что и
    настоящий брак (Р1).
    """
    checks, numbers = [], {"endpoint": endpoint, "price_usd": KLING_PRICE_USD}
    try:
        refuse_pro(endpoint)
        checks.append(("сторож pro", PASS, f"{endpoint}: тарифов "
                                           f"{list(FORBIDDEN_TIERS)} нет"))
    except ValueError as exc:
        checks.append(("сторож pro", FAIL, str(exc)))
        return _result(STAGES[4], checks, numbers=numbers)

    upload = live_upload if upload is None else upload
    kling = live_kling if kling is None else kling
    try:
        video_url = upload(str(window))
        image_url = upload(str(styled))
    except Exception as exc:                             # noqa: BLE001
        checks.append(("загрузка входов", UNMEASURED,
                       f"{type(exc).__name__}: {exc}"))
        return _result(STAGES[4], checks, numbers=numbers,
                       note="входы не уехали: заказ не делался, денег не потрачено")
    checks.append(("загрузка входов", PASS, "video_url и image_url получены"))

    try:
        payload = kling_payload(video_url=video_url, image_url=image_url,
                                character_orientation=orientation)
    except ValueError as exc:
        checks.append(("состав запроса", FAIL, str(exc)))
        return _result(STAGES[4], checks, numbers=numbers)
    extra = sorted(set(payload) - set(KLING_FIELDS))
    missing = sorted(set(KLING_FIELDS) - set(payload))
    checks.append(("состав запроса", PASS if not (extra or missing) else FAIL,
                   f"поля {sorted(payload)} при измеренных {sorted(KLING_FIELDS)}"
                   + (f", лишние {extra}" if extra else "")
                   + (f", нет {missing}" if missing else "")))
    if extra or missing:
        return _result(STAGES[4], checks, numbers=numbers)

    t0 = time.perf_counter()
    try:
        got = kling(video_url=payload["video_url"],
                    image_url=payload["image_url"],
                    character_orientation=payload["character_orientation"],
                    out_path=str(out_path))
    except Exception as exc:                             # noqa: BLE001
        checks.append(("вызов Kling", UNMEASURED, f"{type(exc).__name__}: {exc}"))
        return _result(STAGES[4], checks, numbers=numbers,
                       note="заказ не состоялся: измерять нечего")
    spent = round(time.perf_counter() - t0, 1)
    numbers["latency_s"] = spent
    lo, hi = KLING_LATENCY_S
    checks.append(("вызов Kling", PASS,
                   f"{spent} с (измеренная полоса {lo}..{hi} с), "
                   f"${KLING_PRICE_USD}"))
    checks.append(file_fact(got or out_path, "выход Kling"))
    return _result(STAGES[4], checks, numbers=numbers,
                   produced=str(got or out_path))


# ---------------------------------------------------------------------------
# Ступень 6. Приёмка выхода: кадры, личность, склейки
# ---------------------------------------------------------------------------

def stage_output_acceptance(*, produced, client_photo, frames_dir,
                            probe=None, decode=None, distances=None,
                            cuts=None, operator_ok_identity=False) -> dict:
    """Геометрия, личность и монтажные резы на выходе Kling."""
    checks, numbers = [], {}
    probe = _default_probe() if probe is None else probe
    info = probe(str(produced))
    numbers["width"] = info.get("width")
    numbers["height"] = info.get("height")
    numbers["fps"] = info.get("fps")
    numbers["frames"] = info.get("frames")
    if not info.get("width"):
        checks.append(("геометрия выхода", UNMEASURED, str(info.get("note"))[:200]))
    else:
        w, h = info.get("width"), info.get("height")
        ratio = w / h
        numbers["ratio"] = round(ratio, 4)
        known = (w, h) == KLING_OUT_SIZE
        fps_ok = info.get("fps") == KLING_OUT_FPS
        if ratio > OUT_RATIO_MAX:
            checks.append(("геометрия выхода", FAIL,
                           f"{w}x{h}, соотношение {ratio:.4f} > "
                           f"{OUT_RATIO_MAX}: ГОРИЗОНТАЛЬ, для вертикального "
                           f"продукта это брак"))
        elif not fps_ok:
            checks.append(("геометрия выхода", UNMEASURED,
                           f"{w}x{h} при {info.get('fps')} к/с вместо "
                           f"{KLING_OUT_FPS}: частота не та, сборка звука "
                           f"считает кадры по 30 — судить нечем"))
        else:
            was = "как на прежних заказах" if known else (
                f"НОВАЯ геометрия, прежние восемь давали "
                f"{KLING_OUT_SIZE[0]}x{KLING_OUT_SIZE[1]}")
            checks.append(("геометрия выхода", PASS,
                           f"{w}x{h}, соотношение {ratio:.4f} при потолке "
                           f"{OUT_RATIO_MAX} — вертикаль или квадрат; {was}"))

    decode = _default_decode() if decode is None else decode
    try:
        got = decode(str(produced), str(frames_dir))
    except Exception as exc:                             # noqa: BLE001
        checks.append(("раскладка на кадры", UNMEASURED,
                       f"{type(exc).__name__}: {exc}"))
        return _result(STAGES[5], checks, numbers=numbers)
    paths = list(got.get("paths") or [])
    numbers["decoded"] = len(paths)
    if not paths:
        checks.append(("раскладка на кадры", UNMEASURED,
                       f"кадров не вышло: {str(got.get('note'))[:200]}"))
        return _result(STAGES[5], checks, numbers=numbers)
    checks.append(("раскладка на кадры", PASS, f"кадров {len(paths)}"))

    distances = _default_distances() if distances is None else distances
    try:
        d = distances(paths, str(client_photo))
    except Exception as exc:                             # noqa: BLE001
        d = {"outcome": UNMEASURED, "note": f"{type(exc).__name__}: {exc}"}
    numbers["identity_median"] = d.get("median")
    numbers["identity_inside"] = d.get("inside")
    numbers["identity_judged"] = d.get("judged")
    if d.get("outcome") == UNMEASURED:
        checks.append(("личность на выходе", UNMEASURED, str(d.get("note"))[:300]))
    else:
        # ТА ЖЕ ЛЕСТНИЦА, ЧТО НА СТИЛИЗОВАННОМ ФОТО (Е1: одно знание — одно
        # место). Причина та же и измерена: аксессуар со стилевого референса
        # доезжает до ролика и закрывает лицо на всех кадрах — боевой прогон
        # 22.08 дал 0.5109 при 0 из 99 в баре, при том что тот же Kling на
        # НЕстилизованном фото давал 0.2430 и 98 из 99. Это окклюзия, а не
        # подмена личности, и «не годно» тут было бы неправдой.
        med = d.get("median")
        tail = (f"в баре {d.get('inside')} из {d.get('judged')} судимых "
                f"(лестница: {LADDER_SAME} тот же, {LADDER_REJECTED} другой, "
                f"{LADDER_STRANGER} чужой)")
        if med is None:
            checks.append(("личность на выходе", UNMEASURED,
                           "медианы нет: судить нечем"))
        elif med <= SAME_PERSON_MAX:
            checks.append(("личность на выходе", PASS,
                           f"медиана {med} при планке {SAME_PERSON_MAX}, {tail}"))
        elif med < LADDER_REJECTED:
            band = (f"медиана {med} между планкой {SAME_PERSON_MAX} и ступенью "
                    f"«другой человек» {LADDER_REJECTED}: лицо ЧАСТИЧНО "
                    f"ЗАКРЫТО, ArcFace здесь НЕ СУДЬЯ; {tail}")
            if operator_ok_identity:
                checks.append(("личность на выходе", PASS,
                               band + "; ДОПУЩЕНО ОПЕРАТОРОМ явным флагом"))
            else:
                checks.append(("личность на выходе", UNMEASURED,
                               band + ", судит оператор глазами"))
        else:
            checks.append(("личность на выходе", FAIL,
                           f"медиана {med} выше ступени «другой человек» "
                           f"{LADDER_REJECTED}: подмена личности; {tail}"))

    cuts = _default_cuts() if cuts is None else cuts
    try:
        c = cuts(paths)
    except Exception as exc:                             # noqa: BLE001
        c = {"outcome": UNMEASURED, "note": f"{type(exc).__name__}: {exc}"}
    numbers["cuts"] = None if c.get("outcome") == UNMEASURED else len(c.get("cuts") or [])
    if c.get("outcome") == UNMEASURED:
        checks.append(("монтажные резы", UNMEASURED, str(c.get("note"))[:300]))
    else:
        found = len(c.get("cuts") or [])
        checks.append(("монтажные резы", PASS if found <= MAX_CUTS_OUT else FAIL,
                       f"резов {found} при допуске {MAX_CUTS_OUT}; "
                       f"{str(c.get('note'))[:160]}"))
    return _result(STAGES[5], checks, numbers=numbers)


def _default_decode():
    from . import fork_video                             # noqa: PLC0415

    def decode(video, out_dir):
        return fork_video.frames(video, out_dir, overwrite=True)

    return decode


def _default_cuts():
    from . import fork_looper                            # noqa: PLC0415

    return fork_looper.cuts


# ---------------------------------------------------------------------------
# Ступень 7. Финальная сборка — соседний модуль
# ---------------------------------------------------------------------------

def stage_finish(*, produced, driving, out_path, window=None,
                 finish=None) -> dict:
    """Кроп 9:16 и возврат звука. Живёт в `fork_finish`, зовётся мягко.

    `window` — пара номеров кадров драйвинга, ОБЕ границы включительно: сосед
    берёт по ним звук, и без них он честно откажется. Порядок аргументов у
    соседа `(драйвинг, выход Kling, куда)` — ИЗМЕРЕНО чтением его сигнатуры, а
    не угадано: перепутанный порядок дал бы кроп не того файла.
    """
    checks = []
    note = ""
    if finish is None:
        mod, why = soft_import("fork_finish")
        if mod is None:
            checks.append(("финальная сборка", UNMEASURED, why))
            return _result(STAGES[6], checks, note=why)
        finish, name, why = entry_point(
            mod, ("finish", "assemble", "build", "compose", "run"))
        if finish is None:
            checks.append(("финальная сборка", UNMEASURED, why))
            return _result(STAGES[6], checks, note=why)
        note = f"fork_finish.{name}"
    try:
        reply = _call(finish,
                      {"driving_path": str(driving), "kling_path": str(produced),
                       "out_path": str(out_path), "window": window},
                      (str(driving), str(produced), str(out_path)))
    except Exception as exc:                             # noqa: BLE001
        checks.append(("финальная сборка", UNMEASURED,
                       f"{type(exc).__name__}: {exc}"))
        return _result(STAGES[6], checks, note="сосед упал: это НЕ «не годно»")
    out, why = outcome_of(reply, what="fork_finish")
    checks.append(("финальная сборка", out, why))
    if out == PASS:
        target = (reply.get("path") if isinstance(reply, dict) else None) or out_path
        checks.append(file_fact(target, "финальный ролик"))
    return _result(STAGES[6], checks, note=note or why)


# ---------------------------------------------------------------------------
# Ступень 8. Отчёт
# ---------------------------------------------------------------------------

def stage_report(stages: list, *, out_path=None) -> dict:
    """Свод по ступеням. Частичный результат — ЧИСЛАМИ, а не флагом (Е3)."""
    checked = sum(s["checked"] for s in stages)
    violations = sum(s["violations"] for s in stages)
    unmeasured = sum(s["unmeasured"] for s in stages)
    done = sum(1 for s in stages if s["outcome"] == PASS)
    checks = [("свод по ступеням", PASS,
               f"ступеней пройдено {done} из {len(STAGES) - 1} до отчёта; "
               f"проверок {checked}, нарушений {violations}, "
               f"не смогли {unmeasured}")]
    if out_path is not None:
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(
                json.dumps({"stages": stages, "checked": checked,
                            "violations": violations, "unmeasured": unmeasured},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            checks.append(("отчёт на диск", PASS, str(out_path)))
        except OSError as exc:
            checks.append(("отчёт на диск", UNMEASURED, f"{type(exc).__name__}: {exc}"))
    return _result(STAGES[7], checks,
                   totals={"checked": checked, "violations": violations,
                           "unmeasured": unmeasured, "stages_passed": done})


# ---------------------------------------------------------------------------
# Оркестратор
# ---------------------------------------------------------------------------

def run(*, client_photo, style_ref, driving, first: int, last: int,
        out_dir="work/e2e", intake=None, stylize=None, similarity=None,
        distances=None, probe=None, cutter=None, decode=None, cuts=None,
        upload=None, kling=None, finish=None, card_reader=None,
        driving_frames=None, operator_ok_identity: bool = False,
        orientation: str = CHARACTER_ORIENTATION, endpoint: str = KLING_ENDPOINT,
        log=None) -> dict:
    """Весь путь по ступеням. Печатает КАЖДУЮ сразу и стоит на первой «не годно».

    Возвращает свод: исход, номер и имя ступени, на которой встали, все ступени
    целиком и код возврата. Останов — на любом исходе, кроме `годно`: и `не
    годно`, и `не смогли` означают, что дальше идти НЕЛЬЗЯ (следующая ступень
    получила бы на вход то, чего нет). Различие между ними едет в код возврата,
    а не в поведение.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say(f"стенд: {len(STAGES)} ступеней, останов на первой «{FAIL}»; "
        f"платный вызов ровно один (${KLING_PRICE_USD})", log=log)

    styled = out / "styled.png"
    window = out / "window.mp4"
    produced = out / "kling_out.mp4"
    final = out / "final_9x16.mp4"

    stages, stopped = [], None

    def step(fn):
        nonlocal stopped
        t0 = time.perf_counter()
        res = fn()
        res["elapsed"] = round(time.perf_counter() - t0, 2)
        stages.append(res)
        say(line(res) + f" | {res['elapsed']} с", log=log)
        for c in res["checks"]:
            say(f"      · {c['name']}: {c['outcome']} — {c['note']}", log=log)
        if res["outcome"] != PASS and stopped is None:
            stopped = res
        return res

    r1 = step(lambda: stage_intake(client_photo=client_photo, style_ref=style_ref,
                                   driving=driving, intake=intake,
                                   driving_frames=driving_frames,
                                   card_reader=card_reader))
    if r1["outcome"] == PASS:
        r2 = step(lambda: stage_stylize(client_photo=client_photo,
                                        style_ref=style_ref, out_path=styled,
                                        stylize=stylize, card_reader=card_reader))
        if r2["outcome"] == PASS:
            r3 = step(lambda: stage_style_acceptance(
                styled=r2.get("styled", styled), style_ref=style_ref,
                client_photo=client_photo, similarity=similarity,
                distances=distances, operator_ok_identity=operator_ok_identity))
            if r3["outcome"] == PASS:
                r4 = step(lambda: stage_window(driving=driving, first=first,
                                               last=last, out_path=window,
                                               probe=probe, cutter=cutter))
                if r4["outcome"] == PASS:
                    r5 = step(lambda: stage_kling(
                        styled=r2.get("styled", styled),
                        window=r4.get("window", window), out_path=produced,
                        upload=upload, kling=kling, endpoint=endpoint,
                        orientation=orientation))
                    if r5["outcome"] == PASS:
                        r6 = step(lambda: stage_output_acceptance(
                            produced=r5.get("produced", produced),
                            client_photo=client_photo,
                            frames_dir=out / "out_frames", probe=probe,
                            decode=decode, distances=distances, cuts=cuts,
                            operator_ok_identity=operator_ok_identity))
                        # ПОЧЕМУ ЗДЕСЬ НЕ `== PASS`, В ОТЛИЧИЕ ОТ ВСЕХ
                        # ОСТАЛЬНЫХ СТУПЕНЕЙ.
                        #
                        # Ступень 7 — механическая: обрезка в 9:16 и возврат
                        # звука. Она НЕ ЗАВИСИТ от того, доказана личность или
                        # нет; она делает тот самый файл, КОТОРЫМ ОПЕРАТОР И
                        # СУДИТ. Остановиться перед ней на исходе «не смогли»
                        # значит заплатить за Kling и не отдать оператору
                        # предмет суждения — это и был дефект.
                        #
                        # ИЗМЕРЕНО, почему «не смогли» здесь — норма, а не
                        # редкость: приём driving_b4 даёт лицо 31..98 px и 161
                        # кадр вообще без лица из 450. ArcFace на таком выходе
                        # физически не судья, и решением владельца от 22.08 это
                        # ПРЕДУПРЕЖДЕНИЕ, а не брак.
                        #
                        # ВЕРДИКТ ПРОГОНА ОТ ЭТОГО НЕ ОТБЕЛИВАЕТСЯ: `stopped`
                        # уже проставлен ступенью 6, итог остаётся «не смогли»
                        # и код возврата 2. Мы доделываем работу, а не
                        # переписываем оценку (Р1).
                        #
                        # На `не годно` ступень 7 НЕ идёт: там выход
                        # горизонтальный или подменена личность, и резать
                        # заведомый брак незачем.
                        if r6["outcome"] != FAIL:
                            step(lambda: stage_finish(
                                produced=r5.get("produced", produced),
                                driving=driving, out_path=final,
                                window=(first, last), finish=finish))

    # Отчёт печатается ВСЕГДА, в том числе после останова: прогон, молча
    # умерший на середине, уже уносил с собой всё измеренное.
    report = stage_report(stages, out_path=out / "e2e_report.json")
    stages_before_report = list(stages)
    stages.append(report)
    say(line(report), log=log)

    outcome = stopped["outcome"] if stopped is not None else report["outcome"]
    where = (f"{stopped['stage']}" if stopped is not None else "все ступени")
    totals = report["totals"]
    say(f"ИТОГ: {outcome} на ступени «{where}» | ступеней пройдено "
        f"{totals['stages_passed']} из {len(STAGES) - 1} | проверок "
        f"{totals['checked']}, нарушений {totals['violations']}, не смогли "
        f"{totals['unmeasured']}", log=log)
    return {"outcome": outcome, "stopped_at": where,
            "stopped_index": (stages_before_report.index(stopped) + 1
                              if stopped is not None else None),
            "stages": stages, "totals": totals,
            "exit_code": EXIT_BY_OUTCOME[outcome],
            "report": str(out / "e2e_report.json")}


def parse_window(text: str) -> tuple:
    """`первый:последний` -> пара чисел. Мусор — исключение, а не догадка."""
    parts = str(text).split(":")
    if len(parts) != 2 or not all(p.strip().lstrip("-").isdigit() for p in parts):
        raise ValueError(f"окно {text!r} не вида «первый:последний», например 100:199")
    first, last = int(parts[0]), int(parts[1])
    if first > last:
        raise ValueError(f"окно {text!r}: первый кадр за последним")
    return first, last


def frame_paths(directory) -> list | None:
    """Кадры каталога по порядку. Пустой каталог — исключение, а не тишина.

    Молчаливый пустой список неотличим от «кадров не просили», и разница
    стоит ступени приёма: с ним четыре оси из пяти отвечают «не смогли».
    """
    if directory is None:
        return None
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"каталог кадров {directory!r} не существует")
    got = sorted(str(p) for p in root.iterdir()
                 if p.suffix.lower() in FRAME_SUFFIXES)
    if not got:
        raise ValueError(f"каталог кадров {directory!r} пуст: "
                         f"ждали файлы {', '.join(sorted(FRAME_SUFFIXES))}")
    return got


def main(argv=None) -> int:
    """Тонкая точка входа: разбор аргументов и вызов `run` (Т5)."""
    import argparse                                      # noqa: PLC0415

    ap = argparse.ArgumentParser(description="сквозной стенд форка")
    ap.add_argument("--client", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--driving", required=True)
    ap.add_argument("--window", required=True, help="первый:последний, напр. 100:199")
    ap.add_argument("--out", default="work/e2e")
    # Кадры драйвинга РАСПАКОВЫВАЕТ `fork_video.frames`, а не мы: второй
    # распаковщик в проекте был бы вторым способом узнать известное (Е1).
    # Без этого канала приёмщик честно отвечает «не смогли» по четырём осям
    # из пяти, и стенд встаёт на ступени 1 — ИЗМЕРЕНО двумя прогонами 22.08
    # (b2 и b4: «приём драйвинга — не смогли, не смогли 3»). Деньги при этом
    # не тратятся, но и работа не делается.
    ap.add_argument("--frames", default=None,
                    help="каталог с уже распакованными кадрами драйвинга")
    # Канал ОПЕРАТОРСКОГО ВЕРДИКТА по личности. Он существует потому, что на
    # средней полосе лестницы (между планкой и ступенью «другой человек»)
    # ArcFace измеряет окклюзию, а не подмену, и решение там принимает глаз.
    # Флаг ЯВНЫЙ и виден в отчёте строкой «ДОПУЩЕНО ОПЕРАТОРОМ»: молча средняя
    # полоса не проходится никогда.
    ap.add_argument("--operator-ok-identity", action="store_true",
                    help="оператор посмотрел глазами и допустил личность")
    a = ap.parse_args(argv)
    first, last = parse_window(a.window)
    got = run(client_photo=a.client, style_ref=a.style, driving=a.driving,
              first=first, last=last, out_dir=a.out,
              driving_frames=frame_paths(a.frames),
              operator_ok_identity=a.operator_ok_identity)
    return got["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
