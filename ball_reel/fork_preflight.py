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

НЕПРОВЕРЕНО (Ц4), наверх:

* **на карте не исполнялось.** В этой среде GPU нет: `nvidia-smi`
  отсутствует, и все ветки, которые его читают, шли только на подставленном
  выводе. Разбор строки CSV проверен тестами, живой прогон — нет;
* **бюджеты памяти — АРИФМЕТИКА, а не замер.** Числа ступеней взяты из
  ХЭНДОФ §3.1 и здесь только складываются. Сколько именно откусывает ECC,
  не замерено — `nvidia-smi` покажет на месте;
* **`compute_cap` A16 = 8.6 не подтверждён командой** — он выведен из строки
  даташита «Powered by NVIDIA Ampere architecture». Проверяется первой же
  командой на машине, и до тех пор это вывод, а не факт.
"""

from __future__ import annotations

import shutil
import subprocess

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


def report(*, length: int = 77, measured_tflops: float | None = None,
           path: str = ".") -> dict:
    """Весь предполёт. Числами (Р2): проверено N, провалено M, не смогли K.

    Порядок — дешёвое раньше дорогого (П2): диск опрашивается за миллисекунды,
    карты за десятки, замер полосы — секунды работы на карте.
    """
    checks = {"диск": disk(path)}
    cards = gpus()
    checks["карты"] = cards

    first = (cards.get("gpus") or [{}])[0]
    checks["память"] = budget(first.get("memory_gb"), length=length)
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
