"""Установщик стека: превратить голую арендованную машину в готовую к прогону.

ЗАЧЕМ ОТДЕЛЬНО ОТ `fork_stand`. Приёмка умеет ровно СКАЗАТЬ, чего нет: реестр,
размеры, место, паки, карта. Она ничего не ставит и не качает — и это верно,
приёмка обязана быть дешёвой. Но между «сказала, чего нет» и «прогон пошёл»
лежит час-полтора: клонировать ComfyUI и два набора нод, скачать 18.007 ГиБ
весов. Этот час оплачивается ПО ТАРИФУ GPU, а идёт он на скорости сети, то есть
карта простаивает за деньги. Модуль закрывает ровно этот разрыв.

    fork_stand    что уже стоит         минуты, ничего не трогает
    fork_install  поставить недостающее часы, трогает диск и сеть

ГЛАВНОЕ ТРЕБОВАНИЕ КО ВСЕМУ МОДУЛЮ — ОН УМЕЕТ РАБОТАТЬ НА СУХУЮ. `--dry-run`
печатает полный план: что уже на месте, что качать, сколько это байт, куда
ляжет, какими командами клонируются паки. Ни байта в сеть, ни файла на диск.
Это не удобство, а ЕДИНСТВЕННЫЙ способ проверить установщик там, где его
пишут: сети здесь нет, ComfyUI нет, весов нет, карты нет. Поэтому сухой прогон
— УМОЛЧАНИЕ (`DEFAULT_DRY_RUN`), а ставить нужно попросить явно ключом
`--install`. Случайный запуск не имеет права начать качать 18 ГиБ.

ПОРЯДОК ШАГОВ — ЧАСТЬ ПРИБОРА, А НЕ ОФОРМЛЕНИЕ (П2). Длительность каждого
печатается.

    реестр       мс      прочитать лок-файл
    лицензии     мкс     Ц5 до встраивания, а не после часа загрузки
    каталоги     мс      есть ли куда класть и можно ли туда писать
    место        мс      влезет ли ОСТАТОК (не «40 ГБ вообще»)
    паки         секунды git clone трёх репозиториев
    зависимости  десятки секунд, pip внутри паков
    веса         ЧАСЫ    18.007 ГиБ

Оборваться на последнем шаге из-за отсутствующего каталога или права на запись
— это оплаченный час. Поэтому всё, что ловится за миллисекунду, ловится ДО
первого байта; и поэтому же лицензии стоят вторыми, а не в отчёте по итогам:
`non-commercial`, найденный после загрузки, стоит той же минуты чтения, но уже
переделки (Ц5).

ТРИ ИСХОДА, И ЗДЕСЬ ОНИ РАЗЪЕЗЖАЮТСЯ ДОРОГО (Р1):

    не смогли  сети нет, соединение оборвалось, сервер ответил 5xx. Файл
               ДОКАЧИВАЕТСЯ со следующего запуска, недокачанный хвост лежит
               во временном файле и не выброшен;
    не годно   sha256 не сошёлся, размер больше эталонного, адрес ответил 404.
               Файл НЕ УКЛАДЫВАЕТСЯ под правильным именем НИКОГДА;
    годно      всё на месте. Повторный запуск ничего не делает и говорит об
               этом («качать нечего») — идемпотентность здесь наблюдаемая, а
               не обещанная.

Свести «сети нет» в «не годно» значит послать человека искать дефект в наших
адресах; свести «хэш не сошёлся» в «не смогли» — оставить на диске 10.7 ГиБ
мусора под правильным именем, и приёмка объявит машину готовой.

АТОМАРНАЯ УКЛАДКА, И ОНА ЖЕ САМЫЙ ЧАСТЫЙ ОТКАЗ. Загрузка идёт во ВРЕМЕННЫЙ
файл `<имя>.part` РЯДОМ с целевым (тот же том, значит `os.replace` атомарен), и
переименование происходит ТОЛЬКО после того, как сошлись и размер, и sha256.
Оборванная загрузка оставляет `.part` — файл с неправильным именем и
неправильным размером, то есть ровно то, что докачивается и не может быть
принято за готовое. Приёмка ловит «правильное имя, неправильный размер» за
миллисекунду, но лучше такого файла вообще не создавать.

ДОКАЧКА — HTTP Range, и у неё есть негативный контроль (И5). Сервер вправе
проигнорировать заголовок и отдать файл целиком: тогда ответ будет `200`, а не
`206`, и дописывание в конец `.part` дало бы файл длиннее эталона со сдвинутым
содержимым — то есть провал, замаскированный под успех. Ответ `200` при
ненулевом смещении означает РОВНО ОДНО: начинаем сначала, `.part` усекается.

Е1: НИ ОДНОГО ИМЕНИ, РАЗМЕРА, ХЭША И АДРЕСА ВЕСОВ В ЭТОМ ФАЙЛЕ НЕТ. Всё это
живёт в `workflows/fork_stack.lock.json`, где у каждого числа стоит команда,
которой оно снято. Разбор размеров и хэшей берётся у `fork_stand.read_lock`, а
не пишется здесь второй раз; отсюда добавляется только то, чего приёмке не
нужно, — `url` и `license`. Второй способ узнать известное на этом проекте уже
стоил 1.7 ГБ, скачанных дважды.

Имена паков, адреса их репозиториев и sha256 тел исходников тоже НЕ ЗАПИСАНЫ
здесь: они выводятся из `fork_comfy.PROVEN_BY_SOURCE` — реестра, где каждое имя
узла доказано скачанным телом. Это и есть основание пришпиливания (см.
`REVISION_POLICY` ниже).

---

НЕПРОВЕРЕНО (Ц4), НАВЕРХ. Здесь этого много, и это честное состояние:

* **ни одного байта не скачано, ни один репозиторий не склонирован.** В сеть
  модуль в этой среде не ходил ни разу — запрещено заданием и нечем проверить.
  Все ветки загрузки прогнаны на ПОДСТАВЛЕННОМ транспорте, все ветки `git` — на
  подставленном раннере. Доказано, что модуль верно разбирает ПОДАННЫЕ ему
  ответы, а не что HuggingFace и GitHub отвечают именно так;
* **существование адресов не подтверждено этой сменой** (Ц10). Адреса весов
  замерены потоком E 17-18.08.2026 командами, записанными в самом лок-файле;
  адреса репозиториев выведены из URL, с которых скачаны тела исходников
  (`fork_comfy.PROVEN_BY_SOURCE`, sha256 каждого тела там же). Своей командой
  здесь не проверен НИ ОДИН: `github.com` и `api.github.com` закрыты прокси
  (403, названы и не обойдены — Ц3), `huggingface.co` открыт, но ходить в него
  запрещено заданием;
* **ревизия паков коммитом НЕ пришпилена** — нечем, см. `REVISION_POLICY`;
* **раскладка весов по подкаталогам `models/` — КОНВЕНЦИЯ ComfyUI, не замер.**
  Берётся из `fork_stand.ROLE_DIRS`, где она так и помечена. Проверить её можно
  только на установленном ComfyUI, которого здесь нет;
* **лицензия самого ComfyUI не читана** (`LICENSES`, запись `ComfyUI`). Для
  коммерческого продукта это не мелочь, и молчать об этом нельзя;
* **`pip` не запускался.** Зависимости паков ставятся тем же подставляемым
  раннером, что и `git`, и на этой машине шаг честно отвечает «не смогли».
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import fork_comfy as _fc
from . import fork_preflight as _fp
from . import fork_stand as _fs
from .fork_identity import FAIL, PASS, UNMEASURED

GIB = 1024 ** 3


def _gib(n) -> float | None:
    return None if n is None else round(n / GIB, 3)


# ---------------------------------------------------------------------------
# КОНСТАНТЫ-РЕШЕНИЯ. У каждой — происхождение (И4).
# ---------------------------------------------------------------------------

#: Сухой прогон — УМОЛЧАНИЕ. ВЫБРАНО (кем: эта смена; из чего: два поведения
#: умолчания — «печатать план» и «начать качать»). Решающее основание: цена
#: ошибки несимметрична. Лишний сухой прогон стоит миллисекунды; случайный
#: боевой — час аренды и 18 ГиБ трафика, и остановить его на середине уже
#: дорого. Ставить просят явно: `--install`.
DEFAULT_DRY_RUN = True

#: Расширение временного файла загрузки. РАСЧЁТ, а не вкус: файл обязан лежать
#: В ТОМ ЖЕ КАТАЛОГЕ, что и целевой (иначе `os.replace` перестаёт быть
#: атомарным — это переезд между томами), и обязан НЕ СОВПАДАТЬ с целевым
#: именем, иначе оборванная загрузка даёт правильное имя при неправильном
#: размере. Ровно этот отказ ловит приёмка, и ровно его мы не создаём.
PART_SUFFIX = ".part"

#: Размер куска чтения. ВЫБРАНО 1 МиБ: на 10.7 ГиБ это ~10 900 обращений, то
#: есть накладные расходы неразличимы, а память процесса не растёт.
CHUNK_BYTES = 1 << 20

#: Оставлять ли временный файл, содержимое которого ПРОВАЛИЛО sha256. ВЫБРАНО
#: `False`. Основание: содержимое доказано неверным — докачивать в него нечего,
#: а 10.7 ГиБ заведомого мусора на арендованном диске отнимают место у
#: повторной загрузки того же файла. Доказательством остаётся напечатанный
#: полученный хэш, а не сами байты.
KEEP_BAD_TEMP = False

#: С какого кода ответа отказ считается ЧУЖИМ (не смогли), а не НАШИМ (не
#: годно). РАСЧЁТ по RFC 9110: 4xx — «запрос неверен», то есть наш адрес; 5xx —
#: «серверу плохо», то есть повторяемое. Свести их вместе значило бы либо
#: посылать человека чинить адрес при аварии HuggingFace, либо тихо ждать
#: повторов на 404, которого не будет никогда.
SERVER_ERROR_FROM = 500

#: Секунды ожидания ответа. ВЫБРАНО 60: это ожидание ЗАГОЛОВКОВ, не тела.
#: НЕ ИЗМЕРЕНО — замер требует сети. СТОП-УСЛОВИЕ (Ц9): первый настоящий
#: прогон печатает `seconds` шага; если загрузка срывается по таймауту на
#: живой сети — число переписывается как ИЗМЕРЕННОЕ вместе с командой.
CONNECT_TIMEOUT_S = 60

#: Утилита клонирования и глубина клона. `CLONE_DEPTH = 1` ВЫБРАНО: истории
#: паков нам не нужно, а `--depth 1` на ComfyUI экономит сотни мегабайт и
#: минуты. Оговорка, из-за которой это именно РЕШЕНИЕ, а не оптимизация: по
#: неглубокому клону НЕЛЬЗЯ переключиться на произвольный коммит. Поэтому при
#: заданной ревизии глубина не применяется (см. `clone_commands`).
GIT_BIN = "git"
CLONE_DEPTH = 1

#: Пришпиливать ли ревизию пака СОДЕРЖИМЫМ, а не коммитом. ВЫБРАНО `True`, и
#: это вынужденное решение, а не предпочтение — см. `REVISION_POLICY`.
PIN_BY_BODY_SHA = True

#: Чем ставятся зависимости пака. ВЫБРАНО: тот же интерпретатор, что запустил
#: установщик (`-m pip`), а не `pip` из `PATH`: на арендованных образах в PATH
#: обычно лежит системный pip другого окружения, и пакеты уезжают мимо того
#: Python, которым потом стартует ComfyUI.
PIP_ARGS = ("-m", "pip", "install", "-r")
REQUIREMENTS_NAME = "requirements.txt"

#: Слова в лицензии, при которых установка ОСТАНАВЛИВАЕТСЯ (Ц5). ВЫБРАНО по
#: заданию: продукт коммерческий, и артефакт с такой пометкой встраивать
#: нельзя, сколько бы времени ни было потрачено на его загрузку. Сравнение
#: регистронезависимое.
#:
#: «Лицензия не объявлена» СЮДА НЕ ВХОДИТ, и это решение владельца (ХЭНДОФ,
#: §10): два файла LoRA у `Kijai/WanVideo_comfy` без `cardData.license` — это
#: строка в учёт к отгрузке, а не блокер разработки. Поэтому необъявленная
#: лицензия даёт «не смогли» и печатается поимённо, а не роняет установку.
BLOCKING_LICENSE_MARKS = ("non-commercial", "noncommercial", "research-only",
                          "research only", "cc-by-nc")

#: Лицензии того, что этот модуль СТАВИТ, — отдельно от того, что он КАЧАЕТ.
#: Лицензии весов приходят из лок-файла (Е1) и здесь не повторяются.
#:
#: Записи ниже — не «я помню»: у каждой стоит, чем именно она снята и кем. Две
#: сняты командой (не этой сменой), одна НЕ СНЯТА ВООБЩЕ, и последнее —
#: находка, а не пропуск: продукт коммерческий.
LICENSES = {
    _fc.WRAP_PACK: dict(_fc.WRAP_LICENSE),
    _fc.GGUF_PACK: {
        "pack": _fc.GGUF_PACK,
        "license": "apache-2.0",
        "version": "2.0.0",
        "checked": "2026-08-17",
        "checked_by": ("curl https://raw.githubusercontent.com/city96/"
                       "ComfyUI-GGUF/main/LICENSE -> «Apache License Version "
                       "2.0»; .../pyproject.toml -> name = \"comfyui-gguf\", "
                       "version = \"2.0.0\". Снято ПРЕДЫДУЩЕЙ сменой и записано "
                       "комментарием у fork_comfy.GGUF_PACK; здесь оно "
                       "переведено в данные, потому что установщик обязан "
                       "судить лицензию кодом, а не глазами читателя"),
    },
    "ComfyUI": {
        "pack": "ComfyUI",
        "license": None,
        "version": None,
        "checked": None,
        "checked_by": ("НЕ ПРОВЕРЕНА НИКЕМ И НИКОГДА в этом проекте. В сеть "
                       "этой смене ходить запрещено, github.com закрыт прокси "
                       "(403, Ц3, не обходился). Проверяется одной командой: "
                       "curl -sS https://raw.githubusercontent.com/"
                       "comfyanonymous/ComfyUI/master/LICENSE. Это САМ ДВИЖОК, "
                       "а не пак: продукт коммерческий, и решение здесь за "
                       "владельцем"),
    },
}

#: ПОЧЕМУ РЕВИЗИЯ ПРИШПИЛЕНА СОДЕРЖИМЫМ, А НЕ КОММИТОМ.
#:
#: Требование верное и не обсуждается: `main` меняется под ногами, и реестр
#: доказанных имён узлов (`fork_comfy.PROVEN_BY_SOURCE`) перестал бы
#: соответствовать МОЛЧА — граф собрался бы на именах, которых в установленном
#: паке уже нет, и первое, что об этом сказало бы, — ошибка исполнения на
#: арендованной карте.
#:
#: Коммита у нас нет и взять его неоткуда: `api.github.com/repos` отвечает 403
#: через прокси (ХЭНДОФ, дважды), а обходить закрытый хост запрещено (Ц3).
#: Выдумать хэш коммита нельзя — это ровно тот класс, против которого написано
#: Ц10 («у моделей ~20% предлагаемых имён не существует»).
#:
#: ЧТО ЕСТЬ ВЗАМЕН, и почему это не хуже для НАШЕЙ задачи. У каждого имени в
#: `PROVEN_BY_SOURCE` записан sha256 ТЕЛА ФАЙЛА, из которого имя прочитано.
#: После клонирования установщик считает sha256 тех же файлов в рабочем дереве
#: и сверяет. Сошлось — дерево БАЙТ В БАЙТ то, по которому доказаны имена, то
#: есть пришпилено ровно то, ради чего пришпиливают. Не сошлось — «не годно»
#: с печатью ожидаемого хэша и команды `git rev-parse HEAD`, которой владелец
#: закрепит ревизию навсегда.
#:
#: Чего это НЕ даёт: файлы вне реестра (веса пака, вспомогательные модули) не
#: сторожатся ничем. Пришпиливание по коммиту накрыло бы весь пак; наше —
#: только те три файла, по которым мы что-то утверждаем.
REVISION_POLICY = (
    "ревизия пришпилена СОДЕРЖИМЫМ (sha256 тел исходников из "
    "fork_comfy.PROVEN_BY_SOURCE), а не коммитом: api.github.com закрыт "
    "прокси 403 и не обходится (Ц3). Задать коммит явно: --rev ПАК=РЕВИЗИЯ"
)

#: Хост, с которого скачивались тела исходников. Разбирается, а не пишется:
#: адреса приходят из `PROVEN_BY_SOURCE`.
_RAW_HOST = "raw.githubusercontent.com"


# ---------------------------------------------------------------------------
# РЕЕСТР: то же, что читает приёмка, плюс `url` и `license`
# ---------------------------------------------------------------------------

def read_lock(path: str | Path) -> dict:
    """Записи лок-файла с адресом и лицензией.

    РАЗБОР РАЗМЕРОВ И ХЭШЕЙ НЕ ПИШЕТСЯ ЗДЕСЬ ЗАНОВО (Е1): его делает
    `fork_stand.read_lock`, и он же решает, что запись без размера — это
    «сверить нечем», а не годная запись. Отсюда добавляются ровно два поля,
    которых приёмке не нужно: `url` (ей нечего качать) и `license` (она ничего
    не встраивает). Третий разбор того же файла — это третий способ узнать
    известное, то есть дефект по определению Е1.
    """
    p = Path(path)
    base = _fs.read_lock(p)
    if base["outcome"] != PASS:
        return {**base, "entries": []}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover - уже прочтён выше
        return {"outcome": UNMEASURED, "path": str(p), "entries": [],
                "note": f"реестр {p.name} перестал читаться: {str(exc)[:80]}"}
    raw = doc.get("weights") if isinstance(doc, dict) else doc
    extra = {str(w.get("path")): w for w in raw
             if isinstance(w, dict) and w.get("path")}
    entries = []
    for e in base["entries"]:
        w = extra.get(e["path"], {})
        entries.append({
            **e,
            "url": (str(w["url"]).strip() if w.get("url") else None),
            "repo": w.get("repo"),
            "license": (str(w["license"]).strip() if w.get("license") else None),
            "license_note": w.get("license_note"),
        })
    без_адреса = [e["name"] for e in entries if not e["url"]]
    return {
        "outcome": PASS, "path": str(p), "entries": entries,
        "note": (base["note"]
                 + (f". БЕЗ АДРЕСА (качать нечем): {', '.join(без_адреса)}"
                    if без_адреса else "")),
    }


def find_lock(root: str | Path = ".") -> dict:
    """Найти реестр там же, где его ищет приёмка. Маска — из `fork_preflight` (Е1)."""
    d = Path(root) / _fp.LOCK_DIR
    found = sorted(d.glob(_fp.LOCK_GLOB)) if d.is_dir() else []
    if not found:
        return {"outcome": UNMEASURED, "path": None, "entries": [],
                "note": (f"реестра нет: {d}/{_fp.LOCK_GLOB} не найден. Без него "
                         f"НЕИЗВЕСТНО, ЧТО КАЧАТЬ — это «не смогли», а не "
                         f"«качать нечего»")}
    if len(found) > 1:
        return {"outcome": UNMEASURED, "path": None, "entries": [],
                "note": (f"реестров сразу {len(found)}: "
                         + ", ".join(p.name for p in found)
                         + ". Качать неизвестно по какому нельзя")}
    return read_lock(found[0])


# ---------------------------------------------------------------------------
# ШАГ «ЛИЦЕНЗИИ» (Ц5): до встраивания, а не после часа загрузки
# ---------------------------------------------------------------------------

def licenses(entries: list) -> dict:
    """Лицензии всего, что ставится и качается. Три исхода (Р1).

        не годно   в лицензии есть `non-commercial` / `research-only` —
                   продукт коммерческий, встраивать нельзя;
        не смогли  лицензия НЕ ОБЪЯВЛЕНА или не читана. Решение владельца
                   (§10): это учёт к отгрузке, а не блокер разработки;
        годно      все объявлены и ни одной запрещающей пометки.

    Отсутствие поля читается как «не объявлена», а НЕ как «свободна» — так же,
    как это записано в лок-файле потока E.
    """
    rows = []
    for e in entries:
        rows.append({"what": e["name"], "kind": "вес", "source": e.get("repo"),
                     "license": e.get("license"), "note": e.get("license_note")})
    for name, rec in LICENSES.items():
        rows.append({"what": name, "kind": "пак", "source": rec.get("checked_by"),
                     "license": rec.get("license"), "note": None})

    blocking, unknown = [], []
    for r in rows:
        lic = (r["license"] or "").lower()
        if not lic:
            unknown.append(r["what"])
            r["state"] = "не объявлена"
            continue
        if any(mark in lic for mark in BLOCKING_LICENSE_MARKS):
            blocking.append(f"{r['what']} ({r['license']})")
            r["state"] = "запрещающая"
            continue
        r["state"] = "объявлена"
    checked = sum(1 for r in rows if r["state"] == "объявлена")
    if blocking:
        outcome = FAIL
    elif not rows or not checked:
        outcome = UNMEASURED
    else:
        outcome = UNMEASURED if unknown else PASS
    return {
        "outcome": outcome, "rows": rows, "blocking": blocking,
        "unknown": sorted(set(unknown)), "checked": checked,
        "note": (f"проверено записей {len(rows)}: объявлено {checked}, не "
                 f"объявлено {len(unknown)}, запрещающих {len(blocking)}"
                 + (f". ЗАПРЕЩАЮЩАЯ ЛИЦЕНЗИЯ: {', '.join(blocking)} — продукт "
                    f"коммерческий, встраивать нельзя (Ц5)" if blocking else "")
                 + (f". НЕ ОБЪЯВЛЕНЫ: {', '.join(sorted(set(unknown)))} — "
                    f"отсутствие поля прочитано как «не объявлена», а не как "
                    f"«свободна»; решением владельца (§10) это учёт к отгрузке, "
                    f"а не блокер" if unknown else "")),
    }


# ---------------------------------------------------------------------------
# ГДЕ ЧТО ЛЕЖИТ
# ---------------------------------------------------------------------------

def comfy_dir_for(root: str | Path = ".", comfy_root=None) -> Path:
    """Где ComfyUI будет ЛЕЖАТЬ. Не то же, что `fork_stand.find_comfy`.

    Приёмка спрашивает «где он стоит» и на пустой машине честно отвечает «не
    смогли». Установщику этого мало: он ставит, и ему нужен адрес ДО того, как
    там что-то появилось. Поэтому не найденный ComfyUI — это `<root>/ComfyUI`,
    а не отказ.

    Одно место на весь модуль (Е1): каталог весов выводится отсюда же, и
    разъехавшись, они дали бы веса, скачанные мимо установленного движка.
    """
    if comfy_root:
        return Path(comfy_root).expanduser()
    found = _fs.find_comfy(root)
    return Path(found) if found is not None else Path(root) / "ComfyUI"


def models_root(root: str | Path = ".", comfy_root=None) -> Path:
    """Куда класть веса, если `--models` не назван."""
    return comfy_dir_for(root, comfy_root) / "models"


def target_for(entry: dict, models_dir: Path) -> Path:
    """Куда ЛЯЖЕТ вес, которого ещё нет.

    Подкаталог берётся из `fork_stand.ROLE_DIRS` — ПЕРВЫЙ из перечисленных для
    роли. Раскладка `models/` это КОНВЕНЦИЯ ComfyUI, а не замер (так она и
    помечена там, где живёт), проверить её можно только на установленном
    ComfyUI, которого здесь нет. Роль неизвестна — кладём по пути из реестра
    как он записан: выдумывать подкаталог не из чего.
    """
    subs = _fs.ROLE_DIRS.get(entry["role"])
    if subs:
        return Path(models_dir) / subs[0] / entry["name"]
    return Path(models_dir) / entry["path"]


def survey(entries: list, models_dir: str | Path) -> dict:
    """Что уже на месте, что докачивается, что качать с нуля. Пять состояний.

        ok          файл есть, размер сошёлся байт в байт — трогать нечего
        partial     лежит `.part`, докачиваем с его длины (HTTP Range)
        missing     ни целевого, ни временного — качать целиком
        wrong_size  ЦЕЛЕВОЕ ИМЯ при неверном размере. Докачкой не лечится:
                    под этим именем может лежать другая ступень квантования,
                    а дописать в конец чужого файла — способ получить мусор
                    правильной длины. Качаем заново во временный
        no_url      в реестре нет адреса — качать нечем, «не смогли»

    `partial` отдельно от `missing` потому, что это разные ДЕНЬГИ: у 10.7 ГиБ
    файла, оборвавшегося на девяти, разница между докачкой и перезакачкой —
    сорок минут аренды.
    """
    base = Path(models_dir)
    rows = []
    for e in entries:
        found = _fs.locate(e, base) if base.is_dir() else None
        dest = Path(found) if found is not None else target_for(e, base)
        part = dest.with_name(dest.name + PART_SUFFIX)
        have_part = part.stat().st_size if part.is_file() else 0
        if not e.get("url"):
            state, todo = "no_url", None
        elif found is not None and e["bytes"] is not None:
            got = dest.stat().st_size
            if got == e["bytes"]:
                state, todo = "ok", 0
            else:
                state, todo = "wrong_size", e["bytes"]
        elif found is not None:
            state, todo = "no_size", None
        elif have_part:
            state = "partial"
            todo = (max(e["bytes"] - have_part, 0)
                    if e["bytes"] is not None else None)
        else:
            state = "missing"
            todo = e["bytes"]
        rows.append({**e, "state": state, "target": str(dest), "part": str(part),
                     "have_bytes": (dest.stat().st_size if found is not None
                                    else have_part),
                     "todo_bytes": todo})
    counts = {s: sum(1 for r in rows if r["state"] == s)
              for s in ("ok", "partial", "missing", "wrong_size", "no_url",
                        "no_size")}
    todo = sum(r["todo_bytes"] or 0 for r in rows)
    # ЧТО ЗДЕСЬ «НЕ ГОДНО», А ЧТО ПРОСТО РАБОТА. Отсутствующий вес — НОРМАЛЬНОЕ
    # состояние голой машины, ради него установщик и написан; объявить его
    # провалом значило бы, что сухой прогон на пустой машине всегда красный, то
    # есть вердикт ничего не различает. Провал здесь ровно один: ПРАВИЛЬНОЕ ИМЯ
    # ПРИ НЕПРАВИЛЬНОМ РАЗМЕРЕ — файл, который уже лежит и уже не тот.
    #
    # Р2: ноль к загрузке ПРИ НУЛЕ разобранных записей — не «качать нечего».
    if not rows:
        outcome = UNMEASURED
    elif counts["wrong_size"]:
        outcome = FAIL
    elif counts["no_url"] or counts["no_size"]:
        outcome = UNMEASURED
    elif todo == 0 and counts["ok"] == len(rows):
        outcome = PASS
    else:
        outcome = UNMEASURED
    return {
        "outcome": outcome, "rows": rows, "models_dir": str(base),
        "todo_bytes": todo, **counts,
        "note": (f"весов в реестре {len(rows)}: на месте {counts['ok']}, "
                 f"докачать {counts['partial']}, качать с нуля "
                 f"{counts['missing']}, неверный размер {counts['wrong_size']}, "
                 f"без адреса {counts['no_url']}, без эталона размера "
                 f"{counts['no_size']}. К загрузке {_gib(todo)} ГиБ"
                 + (" — КАЧАТЬ НЕЧЕГО" if outcome == PASS else "")
                 + (". ПРАВИЛЬНОЕ ИМЯ ПРИ НЕПРАВИЛЬНОМ РАЗМЕРЕ: "
                    + ", ".join(f"{r['name']} ({r['have_bytes']} вместо "
                                f"{r['bytes']})"
                                for r in rows if r["state"] == "wrong_size")
                    if counts["wrong_size"] else "")),
    }


# ---------------------------------------------------------------------------
# ШАГ «КАТАЛОГИ»: миллисекунды, а ловят оплаченный час
# ---------------------------------------------------------------------------

def dirs(rows: list, *, dry_run: bool = True) -> dict:
    """Есть ли куда класть и можно ли туда писать.

    Стоит ДО клонирования и ДО загрузки (П2) по одной причине: отсутствующий
    каталог или примонтированный только на чтение том обнаруживаются `stat`-ом
    за микросекунду, а на 39-й минуте загрузки стоят этой минуты и всех
    предыдущих.

    Права проверяются НА БЛИЖАЙШЕМ СУЩЕСТВУЮЩЕМ предке, а не на самом каталоге:
    каталога может ещё не быть, и «нет каталога» — не то же самое, что «писать
    нельзя». В сухом прогоне не создаётся ничего.
    """
    want = sorted({str(Path(r["target"]).parent) for r in rows
                   if r["state"] not in ("no_url",)})
    out = []
    for d in want:
        p = Path(d)
        if p.is_dir():
            state = "есть" if os.access(p, os.W_OK) else "только чтение"
        else:
            anc = p
            while not anc.exists() and anc != anc.parent:
                anc = anc.parent
            if not anc.is_dir():
                state = "нет предка"
            elif not os.access(anc, os.W_OK):
                state = "только чтение"
            elif dry_run:
                state = "будет создан"
            else:
                try:
                    p.mkdir(parents=True, exist_ok=True)
                    state = "создан"
                except OSError as exc:
                    state = f"не создан ({str(exc)[:60]})"
        out.append({"dir": d, "state": state})
    bad = [o for o in out if o["state"].startswith(("только чтение", "нет предка",
                                                    "не создан"))]
    if not out:
        outcome = UNMEASURED
    elif bad:
        outcome = FAIL
    else:
        outcome = PASS
    return {
        "outcome": outcome, "dirs": out,
        "note": (f"каталогов под веса {len(out)}: "
                 + ", ".join(f"{o['dir']} — {o['state']}" for o in out)
                 if out else
                 "ни одного каталога назвать не удалось — качать некуда, и это "
                 "«не смогли», а не «всё на месте»"),
    }


def space(rows: list, path: str | Path) -> dict:
    """Влезет ли ОСТАТОК. Запас и формула — у приёмки (Е1), здесь только вход.

    `fork_stand.disk` считает по строкам вида `{bytes, state}`, где всё, кроме
    `ok`, идёт в остаток. Ей подаются ОСТАТКИ, а не полные размеры: у файла,
    оборвавшегося на девяти гигабайтах из десяти, докачать надо один, и мерить
    место против десяти значит забраковать годную машину.
    """
    synth = [{"bytes": r["todo_bytes"] or 0,
              "state": "ok" if (r["todo_bytes"] or 0) == 0 else "missing"}
             for r in rows]
    return _fs.disk(synth, path)


# ---------------------------------------------------------------------------
# ШАГ «ПАКИ»: что клонировать, куда и чем это пришпилено
# ---------------------------------------------------------------------------

def _split_raw(url: str) -> tuple | None:
    """`raw.githubusercontent.com/OWNER/REPO/REF/ПУТЬ` -> (owner, repo, ref, путь)."""
    if _RAW_HOST not in url:
        return None
    tail = url.split(_RAW_HOST + "/", 1)[1]
    parts = tail.split("/")
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2], "/".join(parts[3:])


def packs(*, revisions: dict | None = None) -> dict:
    """Что клонировать и чем это пришпилено. НИ ОДНО ИМЯ НЕ НАПИСАНО ЗДЕСЬ.

    Всё выводится из `fork_comfy.PROVEN_BY_SOURCE` — реестра, где каждое имя
    узла доказано скачанным телом исходника. Из адреса тела получается адрес
    репозитория, из `body_sha256` — пришпиливание содержимым
    (`REVISION_POLICY`). Написать список репозиториев здесь значило бы завести
    второй источник истины, который разъедется с реестром при первом же
    переезде пака (Е1).

    Куда клонировать, решается ПРИНАДЛЕЖНОСТЬЮ К `fork_stand.REQUIRED_PACKS`:
    что приёмка ищет в `custom_nodes`, то туда и ставится; остальное — сам
    ComfyUI, то есть корень. Сравнение регистронезависимое: `comfyui-gguf` в
    реестре паков и `ComfyUI-GGUF` на GitHub — одно и то же, и на регистре имён
    паков в этом проекте уже один раз разошлись.
    """
    revisions = {k.lower(): v for k, v in (revisions or {}).items()}
    want_packs = {p.lower() for p in _fs.REQUIRED_PACKS}
    repos: dict = {}
    for node_type, proof in _fc.PROVEN_BY_SOURCE.items():
        got = _split_raw(str(proof.get("url", "")))
        if got is None:
            continue
        owner, repo, ref, rel = got
        key = repo.lower()
        rec = repos.setdefault(key, {
            "repo": repo, "owner": owner, "ref": ref,
            "clone_url": f"https://github.com/{owner}/{repo}.git",
            "where": "custom_nodes" if key in want_packs else "base",
            "pins": {}, "proves": [],
            "revision": revisions.get(key),
            "license": None,
        })
        rec["proves"].append(node_type)
        sha = proof.get("body_sha256")
        if sha:
            rec["pins"][rel] = str(sha)
    for rec in repos.values():
        for name, lic in LICENSES.items():
            if name.lower() == rec["repo"].lower():
                rec["license"] = lic.get("license")
        rec["proves"] = sorted(rec["proves"])
    ordered = sorted(repos.values(), key=lambda r: (r["where"] != "base",
                                                    r["repo"].lower()))
    return {"repos": ordered, "policy": REVISION_POLICY,
            "note": (f"репозиториев {len(ordered)}: "
                     + "; ".join(f"{r['repo']} -> {r['where']}, пинов "
                                 f"{len(r['pins'])}, ревизия "
                                 + (r["revision"] or f"НЕ ЗАДАНА (ref {r['ref']})")
                                 for r in ordered))}


def clone_commands(rec: dict, dest: Path) -> list:
    """Точные команды клонирования. Печатаются в сухом прогоне ДОСЛОВНО.

    `--depth 1` ставится ТОЛЬКО когда ревизия не задана: по неглубокому клону
    нельзя переключиться на произвольный коммит, и «сэкономили минуту клона —
    потеряли пришпиливание» здесь был бы размен не в ту сторону.
    """
    if rec.get("revision"):
        return [[GIT_BIN, "clone", rec["clone_url"], str(dest)],
                [GIT_BIN, "-C", str(dest), "-c", "advice.detachedHead=false",
                 "checkout", str(rec["revision"])]]
    return [[GIT_BIN, "clone", "--depth", str(CLONE_DEPTH),
             "--branch", rec["ref"], rec["clone_url"], str(dest)]]


def run_git(argv: list, *, timeout: float = 600.0) -> dict:
    """Запустить `git`. ТОЧКА ВНЕДРЕНИЯ: тест подменяет её целиком (Т4).

    Возвращает `{"code": int|None, "out": str, "err": str}`. `code is None`
    означает РОВНО ОДНО: команда не запустилась (нет `git`, нет прав, таймаут).
    Что клонировать нечего, отсюда не следует НИКОГДА — сделано симметрично
    `fork_stand.read_smi` и `fork_backend.HttpTransport`.
    """
    if shutil.which(argv[0]) is None:
        return {"code": None, "out": "",
                "err": f"{argv[0]} не найден: запускать нечем"}
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"code": None, "out": "", "err": str(exc)[:200]}
    return {"code": done.returncode, "out": done.stdout, "err": done.stderr}


def verify_pack(rec: dict, dest: Path) -> dict:
    """Сошлось ли содержимое пака с тем, по которому доказаны имена узлов.

    ЭТО И ЕСТЬ ПРИШПИЛИВАНИЕ (`REVISION_POLICY`). Три исхода:

        годно      все тела сошлись байт в байт — дерево то самое
        не годно   тело разошлось: `main` уехал, и реестр `PROVEN_BY_SOURCE`
                   больше не описывает установленное. МОЛЧА этого допустить
                   нельзя — граф соберётся на именах, которых в паке нет
        не смогли  файла нет, каталог не прочитан, пинов нет вовсе

    `PIN_BY_BODY_SHA = False` выключает сверку целиком — и тогда пак принимается
    на любой ревизии. Константа существует, чтобы это выключение было видно в
    диффе, а не случалось само.
    """
    if not PIN_BY_BODY_SHA:
        return {"outcome": UNMEASURED, "files": [],
                "note": ("сверка тел ВЫКЛЮЧЕНА (PIN_BY_BODY_SHA=False): пак "
                         "принят на любой ревизии, и что в нём стоит — "
                         "неизвестно")}
    if not rec["pins"]:
        return {"outcome": UNMEASURED, "files": [],
                "note": (f"у {rec['repo']} нет ни одного пина: сверять "
                         f"содержимое не с чем")}
    files = []
    for rel, want in sorted(rec["pins"].items()):
        f = Path(dest) / rel
        if not f.is_file():
            files.append({"file": rel, "state": "нет файла", "want": want,
                          "got": None})
            continue
        try:
            got = _fp.sha256_of(f)
        except OSError as exc:
            files.append({"file": rel, "state": "не прочитан", "want": want,
                          "got": None, "why": str(exc)[:60]})
            continue
        files.append({"file": rel, "want": want, "got": got,
                      "state": "сошлось" if got == want else "РАЗОШЛОСЬ"})
    bad = [f for f in files if f["state"] == "РАЗОШЛОСЬ"]
    unk = [f for f in files if f["state"] in ("нет файла", "не прочитан")]
    ok = [f for f in files if f["state"] == "сошлось"]
    outcome = FAIL if bad else (UNMEASURED if unk or not ok else PASS)
    return {
        "outcome": outcome, "files": files,
        "note": (f"тел сверено {len(ok) + len(bad)} из {len(files)}: сошлось "
                 f"{len(ok)}, разошлось {len(bad)}, не смогли {len(unk)}"
                 + ("". join(f". {f['file']}: ждали {f['want'][:16]}…, "
                             f"получили {(f['got'] or '')[:16]}… — ревизия "
                             f"уехала, закрепить: git -C {dest} rev-parse HEAD"
                             for f in bad) if bad else "")),
    }


def install_packs(plan_packs: dict, *, comfy_dir: Path, dry_run: bool = True,
                  git=run_git) -> dict:
    """Склонировать недостающее и сверить пришпиленное.

    Уже склонированный репозиторий НЕ трогается (идемпотентность): он только
    сверяется. Клонировать поверх существующего каталога `git` всё равно
    откажется, и превращать его отказ в наш — значит терять разницу между
    «уже стоит» и «не смогли поставить».
    """
    out = []
    for rec in plan_packs["repos"]:
        dest = (Path(comfy_dir) if rec["where"] == "base"
                else Path(comfy_dir) / "custom_nodes" / rec["repo"])
        cmds = clone_commands(rec, dest)
        row = {"repo": rec["repo"], "dest": str(dest), "where": rec["where"],
               "commands": [" ".join(c) for c in cmds], "revision": rec["revision"],
               "license": rec["license"], "proves": rec["proves"]}
        if dest.is_dir() and any(dest.iterdir()):
            row["state"] = "уже на месте"
        elif dry_run:
            row["state"] = "клонировать (сухой прогон: не выполнено)"
            row["verify"] = {"outcome": UNMEASURED,
                             "note": "сухой прогон: сверять нечего"}
            out.append(row)
            continue
        else:
            failed = None
            for c in cmds:
                res = git(c)
                if res["code"] != 0:
                    failed = (c, res)
                    break
            if failed is not None:
                c, res = failed
                row["state"] = ("не смогли: " + " ".join(c) + " -> "
                                + (f"код {res['code']}" if res["code"] is not None
                                   else "не запустилось")
                                + f"; {(res['err'] or '')[:120]}")
                row["verify"] = {"outcome": UNMEASURED,
                                 "note": "клон не состоялся, сверять нечего"}
                out.append(row)
                continue
            row["state"] = "склонирован"
        row["verify"] = verify_pack(rec, dest)
        out.append(row)

    verdicts = [r["verify"]["outcome"] for r in out]
    if not out:
        outcome = UNMEASURED
    elif FAIL in verdicts:
        outcome = FAIL
    elif UNMEASURED in verdicts:
        outcome = UNMEASURED
    else:
        outcome = PASS
    return {
        "outcome": outcome, "repos": out, "policy": plan_packs["policy"],
        # Записка расхождения ПОДНИМАЕТСЯ НАВЕРХ, а не остаётся в деталях: без
        # неё верхняя строка говорила «сверка не годно» и не называла ни файла,
        # ни команды закрепления — то есть человек шёл искать причину вручную.
        "note": (f"репозиториев {len(out)}: "
                 + "; ".join(f"{r['repo']} — {r['state']}, сверка "
                             f"{r['verify']['outcome']}" for r in out)
                 + "".join(f". {r['repo']}: {r['verify']['note']}"
                           for r in out if r["verify"]["outcome"] == FAIL)
                 + f". {plan_packs['policy']}"),
    }


def deps(installed: dict, *, dry_run: bool = True, runner=run_git,
         python: str | None = None) -> dict:
    """Зависимости паков. Без них пак импортируется и не даёт НИ ОДНОЙ ноды.

    Тот же раннер, что и у `git` (Т4): «запустить внешнюю команду» — одна
    точка внедрения, а не две. Пака без `requirements.txt` не бывает ошибкой:
    у части паков зависимости объявлены только в `pyproject.toml`, и молча
    считать это провалом значило бы браковать годную установку.
    """
    import sys
    python = sys.executable if python is None else python
    rows = []
    for r in installed["repos"]:
        req = Path(r["dest"]) / REQUIREMENTS_NAME
        cmd = [python, *PIP_ARGS, str(req)]
        row = {"repo": r["repo"], "requirements": str(req),
               "command": " ".join(cmd)}
        if not req.is_file():
            row["state"] = "нет requirements.txt"
        elif dry_run:
            row["state"] = "поставить (сухой прогон: не выполнено)"
        else:
            res = runner(cmd)
            row["state"] = ("поставлено" if res["code"] == 0 else
                            "не смогли: "
                            + (f"код {res['code']}" if res["code"] is not None
                               else "не запустилось")
                            + f"; {(res['err'] or '')[:120]}")
        rows.append(row)
    done = [r for r in rows if r["state"] == "поставлено"]
    bad = [r for r in rows if r["state"].startswith("не смогли")]
    if bad or not done:
        outcome = UNMEASURED
    else:
        outcome = PASS
    return {
        "outcome": outcome, "rows": rows,
        "note": (("СУХОЙ ПРОГОН: pip не запускался. " if dry_run else "")
                 + f"паков {len(rows)}: поставлено {len(done)}, не смогли "
                 f"{len(bad)}, без файла зависимостей "
                 f"{sum(1 for r in rows if r['state'] == 'нет requirements.txt')}"
                 + (". " + "; ".join(f"{r['repo']}: {r['state']}" for r in bad)
                    if bad else "")),
    }


# ---------------------------------------------------------------------------
# ШАГ «ВЕСА»: транспорт, докачка, атомарная укладка
# ---------------------------------------------------------------------------

class HttpFetcher:
    """Штатный транспорт на `urllib`. ТОЧКА ВНЕДРЕНИЯ ЦЕЛИКОМ (Т4).

    Отдельным классом, а не тремя вызовами `urlopen` по коду, ровно по той же
    причине, что и `fork_backend.HttpTransport`: набор проверок подставляет
    сюда объект, который отдаёт байты из памяти, и НИ ОДНА ветка загрузки не
    зависит от того, есть ли на машине с тестами сеть.

    `open` возвращает словарь, а не объект ответа:

        status   int | None — `None` означает РОВНО «ответа не было»
        length   int | None — сколько байт обещано В ЭТОМ ответе
        chunks   итератор кусков
        error    текст, если ответа не было
    """

    def __init__(self, *, timeout: float = CONNECT_TIMEOUT_S,
                 chunk: int = CHUNK_BYTES):
        self.timeout = timeout
        self.chunk = chunk

    def open(self, url: str, *, offset: int = 0) -> dict:
        headers = {}
        if offset:
            # Открытый диапазон: «с этого байта и до конца». Закрытый требовал
            # бы знать длину, а длину мы как раз и проверяем ПОСЛЕ.
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(url, method="GET", headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            # Сервер ОТВЕТИЛ, просто отказом: код есть, и он различает наш
            # неверный адрес (4xx) от чужой аварии (5xx).
            return {"status": exc.code, "length": None, "chunks": iter(()),
                    "error": None}
        except (urllib.error.URLError, OSError) as exc:
            return {"status": None, "length": None, "chunks": iter(()),
                    "error": repr(exc)[:200]}
        raw = resp.headers.get("Content-Length")
        try:
            length = int(raw) if raw is not None else None
        except ValueError:
            length = None

        def _chunks():
            with resp:
                while True:
                    block = resp.read(self.chunk)
                    if not block:
                        return
                    yield block

        return {"status": resp.status, "length": length, "chunks": _chunks(),
                "error": None}


def download(row: dict, *, fetch, dry_run: bool = True) -> dict:
    """Скачать ОДИН вес: докачка, сверка, атомарная укладка. Три исхода (Р1).

    Порядок здесь — тоже прибор, и переставлять его нельзя:

        1. размер эталона неизвестен -> НЕ КАЧАЕМ ВОВСЕ. Принять «что дали»
           значит остаться без единственной дешёвой проверки;
        2. временный файл уже длиной с эталон -> запрос НЕ ОТПРАВЛЯЕТСЯ,
           сразу сверка. Иначе сервер ответил бы `416` на пустой диапазон, и
           готовый файл был бы объявлен ошибкой;
        3. ответ `200` при ненулевом смещении -> сервер ПРОИГНОРИРОВАЛ Range,
           тело полное. Дописывание в конец дало бы файл длиннее эталона со
           сдвинутым содержимым — провал, замаскированный под успех.
           Начинаем сначала (И5: у докачки должен быть отрицательный вход);
        4. размер сошёлся -> считаем sha256 и только ПОТОМ переименовываем.

    Оборвалось на середине — `.part` ОСТАЁТСЯ (это деньги: докачать против
    качать заново), исход «не смогли». Сошёлся размер и не сошёлся хэш —
    `.part` УДАЛЯЕТСЯ (`KEEP_BAD_TEMP`), исход «не годно», целевого имени не
    появляется.
    """
    dest = Path(row["target"])
    part = Path(row["part"])
    want = row.get("bytes")
    name = row["name"]

    if not row.get("url"):
        return {"outcome": UNMEASURED, "name": name, "got_bytes": None,
                "downloaded": 0,
                "note": f"{name}: адреса в реестре нет — качать нечем"}
    if want is None:
        return {"outcome": UNMEASURED, "name": name, "got_bytes": None,
                "downloaded": 0,
                "note": (f"{name}: эталонного размера в реестре нет. НЕ КАЧАЕМ: "
                         f"принять «сколько дали» значит потерять единственную "
                         f"дешёвую проверку")}
    if dest.is_file() and dest.stat().st_size == want:
        return {"outcome": PASS, "name": name, "got_bytes": want,
                "downloaded": 0,
                "note": f"{name}: уже на месте, {want} байт — качать нечего"}
    have = part.stat().st_size if part.is_file() else 0
    if have > want:
        # Временный файл ДЛИННЕЕ эталона — докачивать в него нечего, это
        # чужие байты. Начинаем сначала, иначе размер не сойдётся никогда.
        have = 0
    if dry_run:
        return {"outcome": UNMEASURED, "name": name, "got_bytes": None,
                "downloaded": 0,
                "note": (f"{name}: сухой прогон — {want - have} байт "
                         f"({_gib(want - have)} ГиБ) с {row['url']} в {part}, "
                         f"после сверки sha256 -> {dest}")}

    if have < want:
        try:
            part.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"outcome": UNMEASURED, "name": name, "got_bytes": None,
                    "downloaded": 0,
                    "note": f"{name}: {part.parent} не создан ({str(exc)[:60]})"}
        resp = fetch(row["url"], offset=have)
        status = resp.get("status")
        if status is None:
            return {"outcome": UNMEASURED, "name": name, "got_bytes": have,
                    "downloaded": 0,
                    "note": (f"{name}: ответа не было ({resp.get('error')}). "
                             f"Это «не смогли», а НЕ «файл негоден»: "
                             f"{have} байт лежат в {part.name} и докачаются")}
        if status >= SERVER_ERROR_FROM:
            return {"outcome": UNMEASURED, "name": name, "got_bytes": have,
                    "downloaded": 0,
                    "note": (f"{name}: сервер ответил {status} — авария на той "
                             f"стороне, повторяемо. {have} байт докачаются")}
        if status >= 400:
            return {"outcome": FAIL, "name": name, "got_bytes": have,
                    "downloaded": 0,
                    "note": (f"{name}: сервер ответил {status} на {row['url']} "
                             f"— это НАША находка, адрес не отдаёт этот файл")}
        mode = "ab"
        if have and status == 200:
            # Range проигнорирован: тело ПОЛНОЕ. Дописать его в конец — способ
            # получить мусор правильной длины.
            have, mode = 0, "wb"
        elif not have:
            mode = "wb"
        written = 0
        try:
            # РЕЖИМ ЛИТЕРАЛОМ НА МЕСТЕ ВЫЗОВА, а не переменной. Равносильно
            # `mode` выше (`ab` ровно тогда, когда `have` ненулевой), но
            # сторож кодировок разбирает исходник СТАТИЧЕСКИ и переменную
            # проследить не может — он считает такой вызов текстовым и
            # краснеет. Ослаблять сторожа ради этого нельзя: он ловит
            # настоящий класс (кодировка, отданная локали машины). Проще
            # написать так, чтобы читалось однозначно и человеком, и разбором.
            # Условное выражение ВНУТРИ вызова тоже не годится: разбор берёт
            # второй аргумент и требует от него быть константой, а тернарник
            # константой не является. Значит открываем двумя вызовами.
            fh = open(part, "ab") if have else open(part, "wb")
            with fh:
                for block in resp["chunks"]:
                    fh.write(block)
                    written += len(block)
        except OSError as exc:
            return {"outcome": UNMEASURED, "name": name,
                    "got_bytes": part.stat().st_size if part.is_file() else 0,
                    "downloaded": written,
                    "note": (f"{name}: запись оборвалась ({str(exc)[:80]}). "
                             f"Скачано {written} байт, они докачаются")}
        except Exception as exc:  # обрыв соединения посреди тела
            return {"outcome": UNMEASURED, "name": name,
                    "got_bytes": part.stat().st_size if part.is_file() else 0,
                    "downloaded": written,
                    "note": (f"{name}: соединение оборвалось ({repr(exc)[:80]}). "
                             f"Скачано {written} байт, они лежат в {part.name} "
                             f"и докачаются — это «не смогли», не «не годно»")}
    else:
        written = 0

    got = part.stat().st_size if part.is_file() else 0
    if got < want:
        return {"outcome": UNMEASURED, "name": name, "got_bytes": got,
                "downloaded": written,
                "note": (f"{name}: доехало {got} из {want} байт (разница "
                         f"{got - want:+d}). Хвост докачается со следующего "
                         f"запуска; целевого имени НЕ СОЗДАНО")}
    if got > want:
        if not KEEP_BAD_TEMP:
            part.unlink(missing_ok=True)
        return {"outcome": FAIL, "name": name, "got_bytes": got,
                "downloaded": written,
                "note": (f"{name}: приехало {got} байт вместо {want} (разница "
                         f"{got - want:+d}) — это НЕ ТОТ ФАЙЛ, докачкой не "
                         f"лечится")}
    want_sha = row.get("sha256")
    if not want_sha:
        return {"outcome": UNMEASURED, "name": name, "got_bytes": got,
                "downloaded": written,
                "note": (f"{name}: размер сошёлся, а эталонного sha256 в "
                         f"реестре нет — СОДЕРЖИМОЕ НЕ ПОДТВЕРЖДЕНО, и файл "
                         f"не укладывается")}
    got_sha = _fp.sha256_of(part)
    if got_sha != want_sha:
        if not KEEP_BAD_TEMP:
            part.unlink(missing_ok=True)
        return {"outcome": FAIL, "name": name, "got_bytes": got,
                "downloaded": written, "got_sha256": got_sha,
                "note": (f"{name}: размер сошёлся, а sha256 НЕТ. Ждали "
                         f"{want_sha}, получили {got_sha}. Файл НЕ УЛОЖЕН — "
                         f"под правильным именем неверного содержимого не "
                         f"появится")}
    os.replace(part, dest)
    return {"outcome": PASS, "name": name, "got_bytes": got,
            "downloaded": written, "got_sha256": got_sha,
            "note": (f"{name}: {got} байт, sha256 сошёлся, уложен в {dest} "
                     f"переименованием после сверки")}


def weights(rows: list, *, fetch, dry_run: bool = True) -> dict:
    """Все веса подряд. Числа рядом с вердиктом (Р2), частичный итог — числами (Е3).

    Порядок — ОТ МЕЛКИХ К КРУПНЫМ. Основание: 0.24 ГиБ VAE ловит неверный
    адрес, закрытый прокси и полный диск за секунды, а 10.7 ГиБ диффузии —
    за сорок минут. Тот же довод П2, только внутри одного шага.
    """
    todo = [r for r in rows if r["state"] != "ok"]
    done = [r for r in rows if r["state"] == "ok"]
    order = sorted(todo, key=lambda r: (r.get("todo_bytes") or 0, r["name"]))
    out = [{"outcome": PASS, "name": r["name"], "got_bytes": r["bytes"],
            "downloaded": 0,
            "note": f"{r['name']}: уже на месте, {r['bytes']} байт"}
           for r in done]
    for r in order:
        out.append(download(r, fetch=fetch, dry_run=dry_run))
    ok = [o for o in out if o["outcome"] == PASS]
    bad = [o for o in out if o["outcome"] == FAIL]
    unk = [o for o in out if o["outcome"] == UNMEASURED]
    if not out:
        outcome = UNMEASURED
    elif bad:
        outcome = FAIL
    elif unk:
        outcome = UNMEASURED
    else:
        outcome = PASS
    fetched = sum(o.get("downloaded") or 0 for o in out)
    return {
        "outcome": outcome, "files": out, "downloaded_bytes": fetched,
        "ok": len(ok), "failed": len(bad), "unmeasured": len(unk),
        # Скачанное печатается В БАЙТАХ И В ГиБ. Найдено глазами на прогоне
        # (П3): 200 скачанных байт округлялись в «0.0 ГиБ», то есть отчёт об
        # оборванной загрузке выглядел как отчёт о бездействии — ровно тот
        # дефект, который приёмка уже ловила у себя на разнице размеров.
        "note": (f"весов {len(out)}: на месте и сверено {len(ok)}, не годно "
                 f"{len(bad)}, не смогли {len(unk)}. Скачано за этот запуск "
                 f"{fetched} байт ({_gib(fetched)} ГиБ)"
                 # Поимённо, но БЕЗ полного текста каждой записки: она
                 # печатается отдельной строкой на файл. Первая редакция
                 # повторяла её дважды, и в выводе тонули числа (П3).
                 + (". НЕ ГОДНО: " + ", ".join(o["name"] for o in bad)
                    if bad else "")
                 + (". НЕ СМОГЛИ: " + ", ".join(o["name"] for o in unk)
                    if unk else "")
                 + (" — КАЧАТЬ НЕЧЕГО, повторный запуск ничего не делает"
                    if outcome == PASS and not fetched else "")),
    }


# ---------------------------------------------------------------------------
# ОТЧЁТ
# ---------------------------------------------------------------------------

def _step(name: str, res: dict, seconds: float) -> dict:
    return {"name": name, "outcome": res["outcome"], "seconds": round(seconds, 4),
            "note": res["note"], "detail": res}


def report(*, root: str = ".", models_dir=None, comfy_root=None,
           lock_path=None, disk_path=None, dry_run: bool | None = None,
           fetch=None, git=run_git, revisions: dict | None = None) -> dict:
    """Вся установка, от миллисекунд к часам. Длительность каждого шага — в отчёт.

    Порядок вызовов ниже И ЕСТЬ ПРИБОР (П2). Переставить загрузку 18 ГиБ выше
    проверки прав на каталог — значит вернуть тот самый отказ, ради которого
    модуль написан: сорок минут до вывода, доступного на первой миллисекунде.
    """
    dry_run = DEFAULT_DRY_RUN if dry_run is None else dry_run
    fetch = HttpFetcher().open if fetch is None else fetch
    steps = []

    t = time.perf_counter()
    lock = find_lock(root) if lock_path is None else read_lock(lock_path)
    steps.append(_step("реестр", lock, time.perf_counter() - t))
    entries = lock["entries"]

    # Лицензии — вторыми, до единого байта: `research-only`, найденный после
    # часа загрузки, стоит той же минуты чтения, но уже переделки (Ц5).
    t = time.perf_counter()
    steps.append(_step("лицензии", licenses(entries), time.perf_counter() - t))

    base_models = (Path(models_dir) if models_dir is not None
                   else models_root(root, comfy_root))
    t = time.perf_counter()
    sur = survey(entries, base_models)
    steps.append(_step("осмотр", sur, time.perf_counter() - t))
    rows = sur["rows"]

    t = time.perf_counter()
    steps.append(_step("каталоги", dirs(rows, dry_run=dry_run),
                       time.perf_counter() - t))

    where = disk_path if disk_path is not None else (
        base_models if Path(base_models).exists() else root)
    t = time.perf_counter()
    steps.append(_step("место", space(rows, where), time.perf_counter() - t))

    comfy_dir = comfy_dir_for(root, comfy_root)
    t = time.perf_counter()
    plan_packs = packs(revisions=revisions)
    inst = install_packs(plan_packs, comfy_dir=Path(comfy_dir), dry_run=dry_run,
                         git=git)
    steps.append(_step("паки", inst, time.perf_counter() - t))

    t = time.perf_counter()
    steps.append(_step("зависимости", deps(inst, dry_run=dry_run, runner=git),
                       time.perf_counter() - t))

    t = time.perf_counter()
    steps.append(_step("веса", weights(rows, fetch=fetch, dry_run=dry_run),
                       time.perf_counter() - t))

    passed = [s["name"] for s in steps if s["outcome"] == PASS]
    failed = [s["name"] for s in steps if s["outcome"] == FAIL]
    unmeasured = [s["name"] for s in steps if s["outcome"] == UNMEASURED]
    outcome = FAIL if failed else (UNMEASURED if unmeasured else PASS)
    return {
        "steps": steps, "outcome": outcome, "dry_run": dry_run,
        "comfy_dir": str(comfy_dir), "models_dir": str(base_models),
        "passed": len(passed), "failed": len(failed),
        "unmeasured": len(unmeasured),
        "passed_names": passed, "failed_names": failed,
        "unmeasured_names": unmeasured,
        "todo_gib": _gib(sur.get("todo_bytes") or 0),
        "seconds": round(sum(s["seconds"] for s in steps), 3),
        "note": (("СУХОЙ ПРОГОН (ничего не тронуто). " if dry_run else "")
                 + f"шагов {len(steps)}: пройдено {len(passed)}, провалено "
                 f"{len(failed)}, не смогли {len(unmeasured)}"
                 + (f". ПРОВАЛЕНО: {', '.join(failed)}" if failed else "")
                 + (f". НЕ СМОГЛИ: {', '.join(unmeasured)}" if unmeasured
                    else "")),
    }


#: Как исход ложится в код возврата. Три исхода — три кода (Р1), те же числа,
#: что у приёмки (`fork_stand.EXIT_CODES`): оператор гоняет оба модуля подряд
#: одной командой, и разная нумерация была бы ловушкой.
EXIT_CODES = {PASS: 0, FAIL: 1, UNMEASURED: 2}


def render(rep: dict) -> str:
    """Человекочитаемый план или отчёт. Числа и адреса — рядом с вердиктом."""
    width = max(len(s["name"]) for s in rep["steps"])
    head = ("ПЛАН УСТАНОВКИ СТЕКА (СУХОЙ ПРОГОН, ничего не тронуто)"
            if rep["dry_run"] else "УСТАНОВКА СТЕКА")
    lines = [head, "",
             f"  ComfyUI: {rep['comfy_dir']}",
             f"  веса:    {rep['models_dir']}",
             f"  к загрузке: {rep['todo_gib']} ГиБ", ""]
    for s in rep["steps"]:
        lines.append(f"  {s['name']:<{width}}  {s['seconds']:>8.3f} с  "
                     f"{s['outcome']}")
        lines.append(f"  {'':<{width}}  {'':>8}    {s['note']}")
        det = s["detail"]
        if s["name"] == "паки":
            for r in det.get("repos", []):
                lines.append(f"  {'':<{width}}  {'':>8}    · {r['repo']} "
                             f"[{r['license'] or 'ЛИЦЕНЗИЯ НЕ ЧИТАНА'}] -> "
                             f"{r['dest']}: {r['state']}")
                for c in r["commands"]:
                    lines.append(f"  {'':<{width}}  {'':>8}      $ {c}")
        if s["name"] == "веса":
            for f in det.get("files", []):
                lines.append(f"  {'':<{width}}  {'':>8}    · {f['note']}")
    lines += ["", f"ИТОГ: {rep['outcome']}", f"  {rep['note']}",
              f"  всего {rep['seconds']} с",
              f"  код возврата {EXIT_CODES[rep['outcome']]}"]
    return "\n".join(lines)


def parse_revisions(pairs) -> dict:
    """`--rev ПАК=РЕВИЗИЯ` в словарь. Вынесено из `main` ради проверки (Т5)."""
    out = {}
    for item in pairs or ():
        name, sep, rev = str(item).partition("=")
        if not sep or not name.strip() or not rev.strip():
            raise ValueError(f"--rev ждёт вид ПАК=РЕВИЗИЯ, получено {item!r}")
        out[name.strip()] = rev.strip()
    return out


def main(argv=None) -> int:
    """Установщик. Ноль — стек на месте, единица — не годно, двойка — не смогли.

    УМОЛЧАНИЕ — СУХОЙ ПРОГОН. Чтобы что-то поставить, нужен явный `--install`:
    случайный запуск не имеет права начать качать 18 ГиБ по тарифу GPU.
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="python3 -m ball_reel.fork_install",
        description="установщик стека: план по умолчанию, установка по --install")
    ap.add_argument("--install", action="store_true",
                    help="ДЕЙСТВОВАТЬ: клонировать и качать (умолчание — план)")
    ap.add_argument("--dry-run", action="store_true",
                    help="только план (умолчание и без этого ключа)")
    ap.add_argument("--root", default=".", help="корень репозитория с реестром")
    ap.add_argument("--comfy", default=None, help="путь к ComfyUI")
    ap.add_argument("--models", default=None, help="путь к каталогу весов")
    ap.add_argument("--lock", default=None, help="путь к лок-файлу")
    ap.add_argument("--disk", default=None, help="на каком пути мерить место")
    ap.add_argument("--rev", action="append", default=[],
                    help="пришпилить ревизию пака: --rev ПАК=КОММИТ")
    args = ap.parse_args(argv)

    if args.install and args.dry_run:
        print("--install и --dry-run вместе не имеют смысла: скажите одно")
        return EXIT_CODES[UNMEASURED]
    try:
        revs = parse_revisions(args.rev)
    except ValueError as exc:
        print(str(exc))
        return EXIT_CODES[UNMEASURED]

    rep = report(root=args.root, models_dir=args.models, comfy_root=args.comfy,
                 lock_path=args.lock, disk_path=args.disk,
                 dry_run=not args.install, revisions=revs)
    print(render(rep))
    return EXIT_CODES[rep["outcome"]]


if __name__ == "__main__":
    raise SystemExit(main())
