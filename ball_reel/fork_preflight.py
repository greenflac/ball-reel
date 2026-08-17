"""Поток H: предполёт ПОДБИРАЕТ конфигурацию под машину, а не отвергает машину.

Разница не стилистическая. Предполёт, который говорит «нужно 24 ГБ, у вас 16,
до свидания», перекладывает работу на человека ровно в тот момент, когда машина
уже арендована. Этот отвечает иначе: «на 16 ГБ идёт Q3_K_M при 77 кадрах, запас
2.70 ГБ» — то есть выдаёт РАБОЧУЮ конфигурацию или честно говорит, что ни одна
не влезает.

ЛИЦЕНЗИЙ НЕ ПРОВЕРЯЕТ (ХЭНДОФ §6 H и §10). На этапе разработки лицензия не
блокирует, это вопрос отгрузки; проверка здесь только задерживала бы работу.

---

ЧТО ЗДЕСЬ ГЛАВНОЕ: ЗАМЕР ВМЕСТО ПАСПОРТА.

`nvidia-smi` отдаёт имя, память и compute capability — это факты. Но
пропускной способности он не отдаёт, а у A16 её нет и в даташите: это product
brief про плотность VDI, и TFLOPS там не напечатан вообще. Значит любое число
«карта даёт X TFLOPS» взято не у производителя.

Поэтому `throughput` не спрашивает паспорт, а ПРОГОНЯЕТ один трансформерный
блок на настоящем числе токенов и засекает время. Паспортный пик и эффективная
полоса расходятся всегда, а решение «на какой геометрии идёт петля
экспериментов» принимается по эффективной.

---

ЧТО ЗАКРЫТО НЕ БЫЛО И ЗАКРЫТО ТЕПЕРЬ (ХЭНДОФ §6 H перечисляет четыре проверки):

* **наличие файлов по хэшу** — `weights()`. Реестр эталонов принимается ВХОДОМ
  (лок-файл потока E, `workflows/fork_*.lock`), а не зашивается: хэши весов
  здесь взять неоткуда, а зашитый «эталон», придуманный этим модулем, был бы
  ровно тем припоминанием, которое §9 запрещает. Нет лок-файла — исход «не
  смогли», а не «всё хорошо»;
* **живость Comfy по HTTP** — `comfy_alive()`. Comfy в среде нет и ставить его
  запрещено (§10), поэтому проверка обязана честно возвращать «не смогли», а не
  падать и не считаться пройденной.

---

ПРО ПАСПОРТНЫЕ TFLOPS. ЗАКРЫТЫЙ ДОМЕН НАЗЫВАЕТСЯ, А НЕ ОБХОДИТСЯ (Ц3).

ХЭНДОФ §3.1a говорит: «по моим сведениям (**НЕ из даташита**) полный GA107 —
20 SM… около 18 TFLOPS». Это ПРИПОМИНАНИЕ. Проверить его нужно было бы по
`nvidia.com` (страница спецификаций GA107 / A16) — домен **закрыт прокси, и он
не обходится**: ни зеркалом, ни сторонним прокси, ни снятием проверки TLS.
Открыты `huggingface.co` и `raw.githubusercontent.com`, но числа SM у GA107
там нет.

Следствие в коде: числа 18, 20 SM и 256 FMA/SM здесь НЕТ ни в одной константе.
Эффективная полоса — только параметр `measured_tflops`, и без него `throughput`
отвечает «не смогли». `SPEC_TFLOPS` ниже намеренно `None` и сторожится тестом:
подстановка туда любого числа не обязана менять ответ.

---

НЕПРОВЕРЕНО (Ц4), наверх:

* **на карте не исполнялось.** В этой среде GPU нет: `nvidia-smi`
  отсутствует, и все ветки, которые его читают, шли только на подставленном
  выводе. Разбор строки CSV проверен тестами, живой прогон — нет;
* **лок-файла `workflows/fork_*.lock` на день написания в дереве нет** — поток E
  пишет его параллельно. Значит `weights()` в этой среде отвечает «не смогли», и
  проверен он на подставленном лок-файле, а не на настоящем;
* **живой Comfy не поднимался ни разу.** `comfy_alive()` проверен на подделанном
  ответе и на закрытом порту, а не на работающем сервере;
* **«около 18 TFLOPS» у GA107 — припоминание автора хэндофа**, не даташит и не
  замер; проверить нечем, см. абзац про `nvidia.com` выше;
* **бюджеты памяти — АРИФМЕТИКА, а не замер.** Числа ступеней взяты из
  ХЭНДОФ §3.1 и здесь только складываются. Сколько именно откусывает ECC,
  не замерено — `nvidia-smi` покажет на месте;
* **`compute_cap` A16 = 8.6 не подтверждён командой** — он выведен из строки
  даташита «Powered by NVIDIA Ampere architecture». Проверяется первой же
  командой на машине, и до тех пор это вывод, а не факт.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .fork_identity import FAIL, PASS, UNMEASURED

#: Ступени квантования и их вес на диске, ГБ. Числа из ХЭНДОФ §3.1; здесь они
#: складываются, а не измеряются заново.
STEPS_GB = {"Q3_K_M": 8.04, "Q4_K_M": 10.71}

#: Что едет рядом с моделью, ГБ. Тоже из §3.1.
LORA_GB = 2.03
COMFY_RESERVE_GB = 1.20

#: Активации при разной длине, ГБ. Две точки, потому что их две и есть;
#: интерполировать между ними здесь нечем и не нужно.
ACTIVATIONS_GB = {77: 2.03, 49: 1.32}

#: Сколько ГБ обязано остаться СВЕРХ посчитанного, чтобы ступень считалась
#: годной, а не «впритык». ВЫБРАНО 1.0, и вот почему число вообще нужно.
#:
#: Первая редакция считала годным всё, что арифметически влезает, и выбрала
#: Q4_K_M: 15.97 из 16.0. Формально верно, по делу нет — ХЭНДОФ §3.1 называет
#: это «впритык» и берёт Q3_K_M, и у него ДВА основания. Первое — запас 2.70
#: против 0.03. Второе непреодолимое: **ECC включён по умолчанию и часть
#: 16 ГБ забирает**, то есть считать надо не от 16, а от чуть меньшего, и
#: насколько меньшего — НЕ ЗАМЕРЕНО. Остаток в 0.03 ГБ против неизвестного
#: расхода — это не запас, а совпадение.
#:
#: Число ВЫБРАНО между двумя наблюдаемыми точками (0.03 и 2.70) и калибруется
#: первым же `nvidia-smi` на машине, который покажет, сколько съел ECC.
MIN_HEADROOM_GB = 1.0

#: Ниже какой compute capability bf16 не нативен и часть весов пойдёт в fp32.
#: 8.6 — Ampere, 7.5 — Turing. ХЭНДОФ §6 H.
BF16_MIN_CAPABILITY = 8.6

#: Сколько диска нужно под веса стека, ГБ. ХЭНДОФ §3: «40 ГБ достаточно».
DISK_NEEDED_GB = 40

#: Паспортная пиковая полоса карты. НАМЕРЕННО `None` и намеренно не заполняется.
#:
#: У A16 TFLOPS в даташите нет вообще (ХЭНДОФ §3.1), а «около 18 TFLOPS» из
#: §3.1a — припоминание автора хэндофа, выведенное из числа SM, которого в
#: product brief тоже нет. Проверять его нужно было бы по `nvidia.com`, и этот
#: домен закрыт прокси; обходить запрещено (Ц3), поэтому число остаётся
#: непроверяемым и в расчёт не входит.
#:
#: Константа существует только затем, чтобы у следующей смены был очевидный
#: ответ на вопрос «а куда вписать паспорт»: никуда. Тест
#: `test_a_spec_number_does_not_leak_into_the_answer` подставляет сюда 18.0 и
#: требует, чтобы ответ НЕ ИЗМЕНИЛСЯ.
SPEC_TFLOPS: float | None = None

#: Куда смотрит предполёт за реестром эталонных хэшей. Пишет его поток E, и
#: имя ищется маской: точное имя файла на день написания неизвестно, а зашивать
#: угаданное имя значит получить «файла нет» вместо «хэш не сошёлся».
LOCK_GLOB = "fork_*.lock"
LOCK_DIR = "workflows"

#: Читаем крупными кусками: веса стека — гигабайты, и построчное чтение здесь
#: было бы медленнее без всякой пользы. ВЫБРАНО 1 МиБ как обычный компромисс.
HASH_CHUNK_BYTES = 1 << 20

#: Где искать живой Comfy. АДРЕС — ПАРАМЕТР, умолчание отсюда. 8188 — штатный
#: порт ComfyUI; `/system_stats` выбран потому, что отвечает без задания и его
#: тело содержит `system`, то есть по нему можно отличить Comfy от чего угодно
#: другого, что оказалось на этом порту.
COMFY_URL = "http://127.0.0.1:8188"
COMFY_PROBE_PATH = "/system_stats"

#: Секунды на попытку. ВЫБРАНО 2.0: живой Comfy на loopback отвечает за
#: миллисекунды, а предполёт не имеет права висеть — он идёт первым (П2).
COMFY_TIMEOUT_S = 2.0


def gpus() -> dict:
    """Что за карты стоят. Три исхода: нашли / нет утилиты / утилита ответила плохо.

    «`nvidia-smi` не найден» и «карт нет» — РАЗНЫЕ состояния, и сливать их
    нельзя: первое чинится установкой драйвера, второе означает не ту машину.
    """
    if shutil.which("nvidia-smi") is None:
        return {"outcome": UNMEASURED, "gpus": [],
                "note": ("nvidia-smi не найден: карт не видно, но это НЕ «карт "
                         "нет» — это «спросить нечем». Ставится с драйвером.")}
    try:
        raw = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"outcome": UNMEASURED, "gpus": [],
                "note": f"nvidia-smi не отработал: {str(exc)[:120]}"}
    if raw.returncode != 0:
        return {"outcome": UNMEASURED, "gpus": [],
                "note": f"nvidia-smi вернул {raw.returncode}: {raw.stderr[:120]}"}
    return parse_smi(raw.stdout)


def parse_smi(text: str) -> dict:
    """Разбор CSV `nvidia-smi`. Вынесен наружу, чтобы его можно было проверить.

    Т5: развилка внутри функции, требующей видеокарты, недостижима для теста и
    деградирует молча. Здесь она на входе-строке, и тесты гоняют её без карты.
    """
    found = []
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, mem, cap, driver = parts[0], parts[1], parts[2], parts[3]
        gb = None
        digits = "".join(c for c in mem if c.isdigit())
        if digits:
            gb = round(int(digits) / 1024, 2) if "mib" in mem.lower() else \
                round(float(digits), 2)
        try:
            cap_f = float(cap)
        except ValueError:
            cap_f = None
        found.append({"name": name, "memory_gb": gb, "compute_cap": cap_f,
                      "driver": driver,
                      "bf16_native": (None if cap_f is None
                                      else cap_f >= BF16_MIN_CAPABILITY)})
    if not found:
        return {"outcome": UNMEASURED, "gpus": [],
                "note": "nvidia-smi ответил, но ни одной строки не разобрано"}
    return {"outcome": PASS, "gpus": found, "count": len(found),
            "note": (f"карт {len(found)}: "
                     + "; ".join(f"{g['name']} {g['memory_gb']}ГБ "
                                 f"cc{g['compute_cap']} "
                                 f"bf16 {'да' if g['bf16_native'] else 'нет'}"
                                 for g in found))}


def budget(memory_gb: float, *, length: int = 77) -> dict:
    """Какая ступень влезает в эту карту. ПОДБОР, а не приговор.

    Возвращает все посчитанные варианты, а не только выбранный: человек,
    которому сказали «идёт Q3_K_M», обычно следующим вопросом спрашивает
    «а почему не Q4», и ответ должен быть в том же отчёте.
    """
    activations = ACTIVATIONS_GB.get(length)
    if activations is None:
        return {"outcome": UNMEASURED, "fits": [], "chosen": None,
                "note": (f"активации для длины {length} не считаны; известны "
                         f"{sorted(ACTIVATIONS_GB)}. Экстраполировать здесь "
                         f"нечем — это был бы выдуманный расход памяти.")}
    if memory_gb is None or memory_gb <= 0:
        return {"outcome": UNMEASURED, "fits": [], "chosen": None,
                "note": f"размер памяти неизвестен ({memory_gb})"}

    # Три состояния на ступень, а не два: влезает с запасом / впритык / не
    # влезает. «Впритык» — отдельное состояние именно потому, что ECC забирает
    # неизмеренную часть памяти, и остаток 0.03 ГБ от неё не защищает.
    rows = []
    for step, weights in sorted(STEPS_GB.items(), key=lambda kv: -kv[1]):
        total = round(weights + LORA_GB + activations + COMFY_RESERVE_GB, 2)
        headroom = round(memory_gb - total, 2)
        rows.append({"step": step, "total_gb": total, "headroom_gb": headroom,
                     "fits": headroom >= MIN_HEADROOM_GB,
                     "tight": 0 <= headroom < MIN_HEADROOM_GB})
    fits = [r for r in rows if r["fits"]]
    chosen = max(fits, key=lambda r: STEPS_GB[r["step"]]) if fits else None
    tight = [r["step"] for r in rows if r["tight"]]

    def describe(r):
        if r["fits"]:
            return f"запас {r['headroom_gb']}"
        if r["tight"]:
            return f"впритык, остаток {r['headroom_gb']}"
        return "не влезает"

    return {
        "outcome": PASS if chosen else FAIL,
        "rows": rows, "fits": [r["step"] for r in fits], "tight": tight,
        "chosen": chosen["step"] if chosen else None,
        "length": length, "memory_gb": memory_gb,
        "min_headroom_gb": MIN_HEADROOM_GB,
        "note": (
            f"на {memory_gb} ГБ при {length} кадрах: "
            + "; ".join(f"{r['step']} {r['total_gb']} ({describe(r)})"
                        for r in rows)
            + (f". Берём {chosen['step']} — при запасе меньше "
               f"{MIN_HEADROOM_GB} ГБ ступень не берётся, потому что ECC "
               f"включён по умолчанию и часть памяти забирает, а сколько "
               f"именно — не замерено."
               if chosen else
               ". НИ ОДНА СТУПЕНЬ НЕ ВЛЕЗАЕТ С ЗАПАСОМ — рычаги: отключить "
               "ECC или уйти на 49 кадров.")),
    }


def throughput(flops_per_forward: float = 1.54e15, steps: int = 6,
               *, measured_tflops: float | None = None) -> dict:
    """Сколько минут на ролик. ТОЛЬКО по замеренной полосе, не по паспорту.

    `measured_tflops` — то, что дал прогон одного трансформерного блока. Без
    него функция НЕ ПОДСТАВЛЯЕТ паспортное число и не гадает: она отвечает «не
    смогли». Это и есть весь смысл — паспортный пик и эффективная полоса
    расходятся всегда, а у A16 паспортного пика нет вовсе.
    """
    total = flops_per_forward * steps
    if measured_tflops is None or measured_tflops <= 0:
        return {"outcome": UNMEASURED, "minutes": None,
                "pflops_per_clip": round(total / 1e15, 2),
                "note": (f"{total / 1e15:.2f} PFLOPs на ролик, но эффективная "
                         f"полоса НЕ ЗАМЕРЕНА. Паспортное число сюда не "
                         f"подставляется: у A16 TFLOPS в даташите нет вообще, "
                         f"а расхождение пика с эффективной полосой — правило, "
                         f"а не исключение. Прогнать один блок.")}
    seconds = total / (measured_tflops * 1e12)
    return {"outcome": PASS, "minutes": round(seconds / 60, 1),
            "pflops_per_clip": round(total / 1e15, 2),
            "measured_tflops": measured_tflops,
            "note": (f"{total / 1e15:.2f} PFLOPs на ролик при замеренных "
                     f"{measured_tflops} TFLOP/s — {seconds / 60:.1f} мин на "
                     f"ролик на одном GPU")}


def disk(path: str = ".") -> dict:
    """Хватает ли места под веса. Дешёвая проверка, идёт одной из первых (П2)."""
    try:
        free = shutil.disk_usage(path).free / 1024 ** 3
    except OSError as exc:
        return {"outcome": UNMEASURED, "free_gb": None,
                "note": f"диск не опрошен: {str(exc)[:100]}"}
    return {"outcome": PASS if free >= DISK_NEEDED_GB else FAIL,
            "free_gb": round(free, 1), "needed_gb": DISK_NEEDED_GB,
            "note": (f"свободно {free:.1f} ГБ при нужных {DISK_NEEDED_GB}"
                     + ("" if free >= DISK_NEEDED_GB else " — не хватает"))}


def find_lock(root: str = ".") -> dict:
    """Найти реестр эталонов. Он ВХОД, а не знание этого модуля.

    Хэши весов взять здесь неоткуда: файлы качает и записывает поток E. Зашитый
    «эталон», сочинённый предполётом, был бы припоминанием, поданным как факт —
    ровно то, что §9 запрещает, и ровно та ошибка, из-за которой в этом проекте
    один раз писался код против несуществующего API.

    Три исхода, и «лок-файла нет» — это НЕ «всё хорошо»: непроверенные веса
    выглядят как проверенные ровно до первого запуска на арендованной машине.
    """
    d = Path(root) / LOCK_DIR
    found = sorted(d.glob(LOCK_GLOB)) if d.is_dir() else []
    if not found:
        return {"outcome": UNMEASURED, "path": None, "entries": [],
                "note": (f"реестра эталонов нет: {d}/{LOCK_GLOB} не найден. Его "
                         f"пишет поток E; пока его нет, «веса на месте» СКАЗАТЬ "
                         f"НЕЧЕМ — это не то же самое, что «веса на месте».")}
    if len(found) > 1:
        return {"outcome": UNMEASURED, "path": None, "entries": [],
                "note": (f"реестров сразу {len(found)}: "
                         + ", ".join(p.name for p in found)
                         + ". Какой из них эталон — решает поток E, а угадывать "
                           "здесь значит сверяться неизвестно с чем.")}
    return read_lock(found[0])


def read_lock(path: str | Path) -> dict:
    """Разбор лок-файла. Вынесен наружу, чтобы проверяться без потока E (Т5).

    Форма записи потоку H неизвестна заранее, поэтому принимаются обе
    очевидные: список записей под ключом `weights`/`files` и отображение
    «путь → хэш». Чего НЕ делается — не принимается запись без хэша как
    годная: это `UNMEASURED` по каждой такой строке.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        return {"outcome": UNMEASURED, "path": str(p), "entries": [],
                "note": f"реестр {p.name} не прочитан: {str(exc)[:100]}"}
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        return {"outcome": UNMEASURED, "path": str(p), "entries": [],
                "note": (f"реестр {p.name} не разобран как JSON: "
                         f"{str(exc)[:100]}. Это НАХОДКА, а не «весов нет».")}

    raw_entries = None
    if isinstance(doc, list):
        raw_entries = doc
    elif isinstance(doc, dict):
        for key in ("weights", "files", "artifacts"):
            if isinstance(doc.get(key), list):
                raw_entries = doc[key]
                break
            if isinstance(doc.get(key), dict):
                raw_entries = [{"path": k, "sha256": v}
                               for k, v in doc[key].items()]
                break
    if raw_entries is None:
        return {"outcome": UNMEASURED, "path": str(p), "entries": [],
                "note": (f"в реестре {p.name} не найдено ни одного списка весов "
                         f"(ждём ключ weights/files/artifacts или список). "
                         f"Пустой разбор — «не смогли», не «нечего проверять».")}

    entries = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        rel = item.get("path") or item.get("file") or item.get("name")
        if not rel:
            continue
        digest = item.get("sha256") or item.get("hash")
        entries.append({"path": str(rel),
                        "sha256": (str(digest).lower() if digest else None)})
    if not entries:
        return {"outcome": UNMEASURED, "path": str(p), "entries": [],
                "note": f"реестр {p.name} разобран, но ни одной записи с путём"}
    return {"outcome": PASS, "path": str(p), "entries": entries,
            "note": f"реестр {p.name}: записей {len(entries)}"}


def sha256_of(path: str | Path) -> str:
    """Хэш файла кусками: веса — гигабайты, целиком в память они не берутся."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def weights(root: str = ".", *, lock_path: str | Path | None = None,
            models_dir: str | Path | None = None) -> dict:
    """Лежат ли веса стека на диске и ТЕ ЛИ ЭТО ФАЙЛЫ.

    По каждому файлу три состояния, и они не сливаются:

        ok        файл есть, хэш сошёлся
        mismatch  файл есть, хэш НЕ сошёлся — это НАХОДКА, а не «файла нет»:
                  докачка оборвалась, вендор подменил файл, или скачано другое
                  квантование. Лечится по-разному, поэтому и состояние своё
        missing   файла нет — качать
        no_hash   файл найден, но эталона в реестре нет — проверить нечем

    `mismatch` отдельно от `missing` именно потому, что «нет файла» человек
    чинит одной командой, а «файл не тот» требует понять, чей он.
    """
    lock = read_lock(lock_path) if lock_path is not None else find_lock(root)
    if lock["outcome"] != PASS:
        return {"outcome": UNMEASURED, "lock": lock["path"], "files": [],
                "ok": 0, "mismatch": 0, "missing": 0, "no_hash": 0,
                "note": lock["note"]}

    base = Path(models_dir) if models_dir is not None else Path(root)
    files = []
    for e in lock["entries"]:
        target = Path(e["path"])
        if not target.is_absolute():
            target = base / target
        if not target.exists():
            files.append({"path": str(target), "state": "missing",
                          "expected": e["sha256"], "got": None})
            continue
        if not e["sha256"]:
            files.append({"path": str(target), "state": "no_hash",
                          "expected": None, "got": None})
            continue
        try:
            got = sha256_of(target)
        except OSError as exc:
            files.append({"path": str(target), "state": "no_hash",
                          "expected": e["sha256"], "got": None,
                          "why": str(exc)[:80]})
            continue
        files.append({"path": str(target),
                      "state": "ok" if got == e["sha256"] else "mismatch",
                      "expected": e["sha256"], "got": got})

    counts = {s: sum(1 for f in files if f["state"] == s)
              for s in ("ok", "mismatch", "missing", "no_hash")}
    # Р2: ноль нарушений при нуле отработавших проверок — не успех. Если
    # сверить не удалось ни одного файла, исход «не смогли», а не PASS.
    if counts["mismatch"] or counts["missing"]:
        outcome = FAIL
    elif counts["ok"] and not counts["no_hash"]:
        outcome = PASS
    else:
        outcome = UNMEASURED
    bad = [f"{Path(f['path']).name} ({f['state']})" for f in files
           if f["state"] != "ok"]
    return {
        "outcome": outcome, "lock": lock["path"], "files": files, **counts,
        "note": (f"весов в реестре {len(files)}: сошлось {counts['ok']}, "
                 f"хэш не сошёлся {counts['mismatch']}, нет файла "
                 f"{counts['missing']}, проверить нечем {counts['no_hash']}"
                 + (". Не сошлось: " + ", ".join(bad[:8]) if bad else "")),
    }


def _urlopen(url: str, timeout: float):
    """Запрос БЕЗ прокси: Comfy стоит на loopback.

    Ц3 здесь ни при чём — закрытый домен не обходится, а localhost через
    внешний прокси просто не адресуется, и без этого обработчика ответ пришёл
    бы от прокси, а не от Comfy.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(url, timeout=timeout)


def comfy_alive(url: str = COMFY_URL, *, timeout_s: float = COMFY_TIMEOUT_S,
                opener=None) -> dict:
    """Жив ли Comfy по HTTP. Адрес и порт — параметр, умолчание `COMFY_URL`.

    Comfy в этой среде НЕТ и ставить его запрещено (§10). Значит нормальный
    исход здесь — «не смогли проверить», и он обязан быть именно им: ни
    падением (предполёт нужен как раз на машине, где ещё ничего не поднято), ни
    молчаливым PASS.

    Исходы:
        PASS        ответил и ответ похож на Comfy
        FAIL        на порту КТО-ТО ЕСТЬ, но это не Comfy — находка: порт занят
                    чужим сервисом, и «подниму Comfy сюда» не сработает
        UNMEASURED  порт закрыт, таймаут, имя не разрешилось — Comfy просто нет

    `opener` подменяется тестом (Т4): в сеть тест не ходит, а все ветки обязаны
    проверяться, поэтому подделываются и ответ, и отказ.
    """
    target = url.rstrip("/") + COMFY_PROBE_PATH
    call = opener or _urlopen
    try:
        resp = call(target, timeout_s)
    except urllib.error.HTTPError as exc:
        return {"outcome": FAIL, "url": target, "status": exc.code,
                "note": (f"на {target} кто-то ответил HTTP {exc.code} — порт "
                         f"занят, но это не рабочий Comfy. Это находка: "
                         f"поднять Comfy на занятый порт не выйдет.")}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"outcome": UNMEASURED, "url": target, "status": None,
                "note": (f"{target} не ответил ({str(exc)[:80]}). Это «не "
                         f"смогли проверить», а не «Comfy сломан»: в этой среде "
                         f"его нет и ставить запрещено (ХЭНДОФ §10).")}
    try:
        body = resp.read()
    finally:
        close = getattr(resp, "close", None)
        if close:
            close()
    status = getattr(resp, "status", None) or getattr(resp, "code", 200)
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
    try:
        doc = json.loads(text)
    except ValueError:
        return {"outcome": FAIL, "url": target, "status": status,
                "note": (f"{target} ответил {status}, но тело не JSON — на порту "
                         f"не Comfy: {text[:60]!r}")}
    if not isinstance(doc, dict) or "system" not in doc:
        return {"outcome": FAIL, "url": target, "status": status,
                "note": (f"{target} ответил {status} JSON без ключа `system` — "
                         f"это не /system_stats Comfy, а чужой сервис")}
    system = doc.get("system") or {}
    version = system.get("comfyui_version") or system.get("python_version") or "?"
    return {"outcome": PASS, "url": target, "status": status,
            "version": version, "devices": len(doc.get("devices") or []),
            "note": (f"Comfy отвечает на {target}: версия {version}, устройств "
                     f"{len(doc.get('devices') or [])}. НЕ ПРОВЕРЯЛОСЬ на живом "
                     f"сервере — только на подделанном ответе.")}


def report(*, length: int = 77, measured_tflops: float | None = None,
           path: str = ".", lock_path: str | Path | None = None,
           comfy_url: str = COMFY_URL) -> dict:
    """Весь предполёт. Числами (Р2): проверено N, провалено M, не смогли K.

    Порядок — дешёвое раньше дорогого (П2): диск опрашивается за миллисекунды,
    карты за десятки, замер полосы — секунды работы на карте.
    """
    checks = {"диск": disk(path)}
    cards = gpus()
    checks["карты"] = cards

    first = (cards.get("gpus") or [{}])[0]
    checks["память"] = budget(first.get("memory_gb"), length=length)
    # Хэширование гигабайтов дороже опроса карты, но дешевле замера полосы, и
    # идёт между ними (П2). Без лок-файла оно вообще не начинается.
    checks["веса"] = weights(path, lock_path=lock_path)
    checks["comfy"] = comfy_alive(comfy_url)
    checks["полоса"] = throughput(steps=6, measured_tflops=measured_tflops)

    passed = sum(1 for c in checks.values() if c["outcome"] == PASS)
    failed = sum(1 for c in checks.values() if c["outcome"] == FAIL)
    unmeasured = sum(1 for c in checks.values() if c["outcome"] == UNMEASURED)
    return {
        "checks": checks,
        "passed": passed, "failed": failed, "unmeasured": unmeasured,
        "outcome": FAIL if failed else (UNMEASURED if unmeasured else PASS),
        "note": (f"проверок {len(checks)}: пройдено {passed}, провалено "
                 f"{failed}, не смогли {unmeasured}. ЛИЦЕНЗИИ НЕ ПРОВЕРЯЮТСЯ — "
                 f"это вопрос отгрузки, не разработки."),
    }
