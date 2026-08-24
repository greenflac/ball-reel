"""Батч поверх сквозного стенда: матрица драйвинг x стиль x личность.

ЧТО ЭТО. `fork_e2e.run` собирает ОДИН ролик за один платный вызов Kling
($0.21). Владельцу нужен не один ролик, а КРЕСТ: 5 драйвингов, 5 стилей,
2 личности. Этот модуль не переписывает стенд — он его ЗОВЁТ по одной ячейке,
печатает исход каждой сразу и считает деньги ДО того, как их потратит.

ГЛАВНОЕ СВОЙСТВО — ДЕНЬГИ СЧИТАЮТСЯ ДО ЗАПУСКА. Полный крест 5x5x2 = 50
роликов = $10.50. Баланс на счету на момент написания — $0.8490, то есть
хватает на 4 ролика. Батч, стартовавший «посмотрим, как пойдёт», потратил бы
всё и встал бы посередине: половина матрицы, ни одной сравнимой пары, и денег
на добор нет. Это ХУДШИЙ ИЗ ВОЗМОЖНЫХ ИСХОДОВ, потому что он невосстановим —
время вернуть можно, списанные деньги нельзя. Поэтому `afford` вызывается
ПЕРВЫМ, до единого заказа, и при нехватке батч не стартует ВОВСЕ и называет
недостачу числом.

ПОЧЕМУ БАЛАНС — ТОЧКА ВНЕДРЕНИЯ, А НЕ HTTP-ВЫЗОВ ВНУТРИ. Проверенного
эндпоинта баланса fal у нас нет: в эту смену наружу не ушло ни байта, и
выдумывать URL «наверное, такой» запрещено (Ц10 — у моделей около каждого
пятого предложенного имени не существует). Поэтому число приходит ВЫЗОВОМ
подставляемой функции: на стенде это переменная окружения `FAL_BALANCE_USD`,
которую оператор списывает с панели fal, в тестах — подставная функция, в
будущем — проверенный эндпоинт, и ни одна строка батча от этого не изменится.
Баланса нет — исход `не смогли проверить`, и батч ТОЖЕ не стартует: тратить
деньги вслепую и не знать остатка — то же самое, что не считать их вовсе.

РЕЖИМЫ ПОКРЫТИЯ. «Затронуть 5 драйвингов, 5 стилей и 2 личности» — это НЕ
обязательно 50 роликов, и разница здесь в деньгах в десять раз:

  `full`   полный крест, N = d*s*p. На 5x5x2 это 50 роликов и $10.50.
           Единственный режим, в котором оси СРАВНИМЫ: любые две ячейки
           отличаются ровно одной координатой, поэтому «этот стиль хуже» —
           утверждение, которое можно доказать.

  `cover`  покрывающий набор, N = max(d, s, p). На 5x5x2 это 5 роликов и
           $1.05. Каждый драйвинг, каждый стиль и каждая личность встречаются
           ХОТЯ БЫ РАЗ: ячейка i берёт `drivings[i % d]`, `styles[i % s]`,
           `persons[i % p]`, и при N = max все остатки по каждой оси
           пробегаются целиком. Это латинский квадрат по трём осям.

           ЧЕГО `cover` НЕ ДАЁТ, И ЭТО НАДО ЗНАТЬ ДО ЗАКАЗА: сравнивать оси
           между собой им НЕЛЬЗЯ. Каждая клетка уникальна по всем трём
           координатам, общей опоры у двух клеток нет, и плохой результат
           ячейки не отделить — драйвинг виноват, стиль или личность.
           `cover` отвечает на вопрос «работает ли вообще на всём материале»,
           а НЕ на вопрос «что лучше». Это осознанный размен цены на охват:
           охват полный, сравнение потеряно.

ЧТО ЕЩЁ ОБЯЗАТЕЛЬНО. Ячейки идут ПО ОДНОЙ, каждая печатается в stderr сразу
(молчащий длинный прогон уже уносил с собой всё измеренное), готовый ролик
кладётся отдельным файлом с говорящим именем `драйвинг__стиль__личность.mp4`,
а неудачная ячейка НЕ роняет батч: одна плохая пара — не повод бросать
оплаченную матрицу. Но подряд идущие неудачи — это уже не пара, а поломка,
и она жжёт деньги; сторож серии описан у `MAX_STREAK`.

ВСЕ ВНЕШНИЕ ВЫЗОВЫ — ТОЧКИ ВНЕДРЕНИЯ (Т4): баланс, прогон ячейки, забор
готового ролика. Тесты гоняют ВЕСЬ батч на подставном `fork_e2e.run` — без
сети и без единого цента.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Три исхода и код возврата — ОДНИМ источником на весь проект (Е1). Копия
# строк здесь разъехалась бы с прибором молча.
from .fork_identity import FAIL, PASS, UNMEASURED
from .fork_video import EXIT_BY_OUTCOME

# Модуль целиком, а не отдельные значения: цена, эндпоинт и печать читаются
# ЧЕРЕЗ него (`E.KLING_PRICE_USD`), чтобы правка в стенде доезжала сюда сама
# и чтобы мутация константы в тесте была наблюдаемой (Е1, Т1).
from . import fork_e2e as E

# ---------------------------------------------------------------------------
# ЧИСЛА. У каждого — происхождение (И4)
# ---------------------------------------------------------------------------

#: ВЫБРАНО ВЛАДЕЛЬЦЕМ: цель — «затронуть 5 драйвингов, 5 стилей, 2 личности».
#: Стоит здесь не как ограничение (батч принимает любую матрицу), а как
#: происхождение чисел в докстринге и в отчёте: 50 ячеек и $10.50 считаются
#: именно от неё.
OWNER_MATRIX = (5, 5, 2)

#: ВЫБРАНО (кем: поток батча; из чего: единственные два режима, у которых есть
#: доказуемое свойство. `full` даёт сравнимость, `cover` — охват. Третьего
#: «на глазок» здесь быть не должно: режим без сформулированного свойства
#: продаёт наугад выбранное подмножество как замер).
MODES = ("full", "cover")

#: ВЫБРАНО 3 (кем: поток батча; из чего: цена одной неудачи — до $0.21, то
#: есть три подряд стоят $0.63).
#:
#: ПОЧЕМУ НЕ 1. Одна неудачная ячейка — норма продукта, а не поломка: пара
#: «этот стиль на этой личности» может законно не пройти приёмку личности
#: (ArcFace на очках уже давал 0.3928 при планке 0.35). Останов на первой
#: неудаче выбросил бы оплаченную матрицу из-за одной плохой пары.
#:
#: ПОЧЕМУ НЕ 2. Две подряд — это ровно то, что даёт один плохой стилевой
#: референс, стоящий соседом в списке: совпадение вероятное и невинное.
#:
#: ПОЧЕМУ НЕ 5+. Систематическая поломка (упал ключ, сменился эндпоинт, пустой
#: баланс) выглядит как СПЛОШНАЯ серия неудач, и каждая её ячейка может
#: списать $0.21 до того, как отвалится. При 5 сгорит $1.05 — больше, чем
#: весь остаток счёта на момент написания ($0.8490).
#:
#: 3 — граница, на которой «невезение» уже неправдоподобно, а сожжено меньше
#: остатка счёта. Мутация в обе стороны (2 и 5) — в тестах модуля.
MAX_STREAK = 3

#: ВЫБРАНО: разделитель в имени ролика. Двойное подчёркивание, потому что
#: одинарное встречается в именах файлов материала (`fork_ref_gym`), и по
#: одинарному имя ячейки не разобрать обратно.
NAME_SEP = "__"

#: Переменная окружения запасного прибора баланса. Оператор списывает число с
#: панели fal — см. `live_balance`.
BALANCE_ENV = "FAL_BALANCE_USD"


# ---------------------------------------------------------------------------
# Точки внедрения. Умолчания трогают внешний мир, тесты их не зовут никогда
# ---------------------------------------------------------------------------

def live_balance() -> float | None:
    """Остаток на счету fal в долларах, либо `None` — «не смогли узнать».

    НЕПРОВЕРЕНО СЕТЬЮ (Ц4): проверенного эндпоинта баланса fal у нас нет, а
    выдуманный URL в коде хуже отсутствия — он выглядит проверкой и молча
    возвращает мусор (Ц10). Поэтому число берётся из окружения: оператор
    смотрит панель fal и запускает батч как

        FAL_BALANCE_USD=0.8490 python3 -m ball_reel.fork_batch ...

    Не задано или не число — `None`, и батч откажется стартовать с исходом
    `не смогли проверить`. Это НЕ придирка: батч без известного остатка не
    умеет ответить «хватит ли», а «посмотрим, как пойдёт» — тот самый худший
    исход, ради которого написан модуль.
    """
    import os                                            # noqa: PLC0415

    raw = os.environ.get(BALANCE_ENV)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def copy_clip(src, dst) -> tuple:
    """Готовый ролик -> отдельный файл с говорящим именем. `(путь, причина)`.

    Возвращает `(str, None)` при успехе и `(None, причина)`, если брать
    нечего. Отсутствие файла НЕ исключение наружу: ячейка, сказавшая «годно»
    и не оставившая ролика, — это `не смогли проверить`, а не обвал батча.
    """
    s = Path(src)
    if not s.is_file():
        return None, f"ролика {s} нет на диске: забирать нечего"
    if s.stat().st_size <= 0:
        return None, f"ролик {s} пустой (0 Б)"
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(s), str(dst))
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return str(dst), None


# ---------------------------------------------------------------------------
# Матрица
# ---------------------------------------------------------------------------

def cell_name(driving, style, person) -> str:
    """`драйвинг__стиль__личность` — имя ролика, читаемое без отчёта.

    Берутся ИМЕНА БЕЗ РАСШИРЕНИЯ: `driving_gym.mp4` -> `driving_gym`. Полный
    путь в имени файла нечитаем, а один индекс (`cell_07.mp4`) требует отчёта
    рядом, чтобы понять, что на видео.
    """
    return NAME_SEP.join(Path(str(p)).stem for p in (driving, style, person))


def cells(drivings, styles, persons, *, mode: str = "full") -> list:
    """Список ячеек матрицы по режиму покрытия. Неизвестный режим — отказ.

    `full`  — N = d*s*p, все сочетания; порядок обхода — драйвинг, стиль,
              личность (внешний цикл дороже переключать оператору глазами).
    `cover` — N = max(d, s, p), ячейка i = (d[i%d], s[i%s], p[i%p]).

    ПОЧЕМУ `cover` ДЕЙСТВИТЕЛЬНО ПОКРЫВАЕТ. При N = max(d,s,p) индекс i
    пробегает 0..N-1, а N >= d, значит `i % d` принимает все значения
    0..d-1 хотя бы раз; то же для s и p. Это доказательство, а не надежда, и
    тест проверяет его на матрице, где оси разной длины.
    """
    if mode not in MODES:
        raise ValueError(f"режим {mode!r} неизвестен, есть {list(MODES)}")
    axes = {"драйвингов": list(drivings), "стилей": list(styles),
            "личностей": list(persons)}
    empty = [k for k, v in axes.items() if not v]
    if empty:
        raise ValueError(f"пустые оси матрицы: {empty}; заказывать нечего")
    dr, st, pe = axes["драйвингов"], axes["стилей"], axes["личностей"]
    out = []
    if mode == "full":
        for d in dr:
            for s in st:
                for p in pe:
                    out.append((d, s, p))
    else:
        for i in range(max(len(dr), len(st), len(pe))):
            out.append((dr[i % len(dr)], st[i % len(st)], pe[i % len(pe)]))
    return [{"driving": d, "style": s, "person": p,
             "name": cell_name(d, s, p), "index": i + 1}
            for i, (d, s, p) in enumerate(out)]


def plan_cost(n: int) -> float:
    """Цена n ячеек. Цена одного вызова ИМПОРТИРУЕТСЯ из стенда (Е1)."""
    return round(int(n) * E.KLING_PRICE_USD, 4)


def afford(n: int, balance) -> dict:
    """Хватит ли денег на n ячеек. ТРИ исхода, и два из них НЕ пускают батч.

    годно                — `balance >= n * цена`;
    не годно             — денег меньше, `short` говорит, сколько НЕ ХВАТАЕТ;
    не смогли проверить  — баланс неизвестен (`None` или не число).

    Третий исход не сворачивается ни в первый («ну наверное хватит»), ни во
    второй («считаем, что пусто»): первое тратит деньги вслепую, второе
    блокирует работу при исправном счёте. Оба схлопывания у нас уже случались
    в других приборах и оба стоили прогонов (Р1).
    """
    need = plan_cost(n)
    try:
        have = None if balance is None else float(balance)
    except (TypeError, ValueError):
        have = None
    if have is None:
        return {"outcome": UNMEASURED, "need": need, "have": None, "short": None,
                "note": (f"остаток счёта неизвестен: заказ на ${need} "
                         f"({n} ячеек по ${E.KLING_PRICE_USD}) не начат. "
                         f"Задай {BALANCE_ENV} или подставь свой прибор баланса")}
    if have + 1e-9 < need:
        short = round(need - have, 4)
        return {"outcome": FAIL, "need": need, "have": round(have, 4),
                "short": short,
                "note": (f"НЕ ХВАТАЕТ ${short}: заказ {n} ячеек по "
                         f"${E.KLING_PRICE_USD} стоит ${need}, на счету "
                         f"${round(have, 4)}. Батч НЕ начат: потратить половину "
                         f"и встать посередине хуже, чем не начинать")}
    return {"outcome": PASS, "need": need, "have": round(have, 4), "short": 0.0,
            "note": (f"хватает: заказ ${need} ({n} ячеек по "
                     f"${E.KLING_PRICE_USD}), на счету ${round(have, 4)}, "
                     f"останется ${round(have - need, 4)}")}


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------

def _cell_line(cell: dict) -> str:
    """Строка ячейки: вердикт и числа РЯДОМ с ним (Р2)."""
    t = cell.get("totals") or {}
    return (f"[{cell['outcome']:<18}] ячейка {cell['index']:>2} "
            f"{cell['name']:<46} проверено {t.get('checked', 0)}, "
            f"нарушений {t.get('violations', 0)}, не смогли "
            f"{t.get('unmeasured', 0)} | {cell.get('note', '')}")


def run_batch(*, drivings, styles, persons, mode: str = "full",
              first: int = 0, last: int = 0, windows=None,
              out_dir="work/batch", balance=None, cell_runner=None,
              collect=None, max_streak: int = MAX_STREAK,
              endpoint: str = None, log=None, **cell_kwargs) -> dict:
    """Весь батч: деньги -> ячейки по одной -> сводка. Возвращает свод.

    ТОЧКИ ВНЕДРЕНИЯ (Т4): `balance` (остаток счёта), `cell_runner` (прогон
    одной ячейки, по умолчанию `fork_e2e.run`), `collect` (забор готового
    ролика). Всё остальное — местная арифметика и печать.

    ПОРЯДОК ЖЁСТКИЙ, И ЭТО РЕШЕНИЕ: сначала матрица (сколько заказов), потом
    сторож `pro` (эндпоинт), потом деньги, и только затем первый вызов. Любая
    из трёх проверок стоит миллисекунды, а первый заказ стоит $0.21 и
    невозвратен (П2: дешёвое раньше дорогого).
    """
    runner = E.run if cell_runner is None else cell_runner
    take = copy_clip if collect is None else collect
    get_balance = live_balance if balance is None else balance
    where = Path(out_dir)
    clips_dir = where / "clips"
    wins = dict(windows or {})

    grid = cells(drivings, styles, persons, mode=mode)
    n = len(grid)
    full_n = len(drivings) * len(styles) * len(persons)
    E.say(f"батч: матрица {len(drivings)}x{len(styles)}x{len(persons)}, режим "
          f"«{mode}» -> {n} ячеек (полный крест дал бы {full_n}); цена "
          f"${E.KLING_PRICE_USD} за ячейку, заказ ${plan_cost(n)}", log=log)

    # Сторож `pro` — ДО денег и до первой ячейки: он и написан ради этого.
    ep = E.KLING_ENDPOINT if endpoint is None else endpoint
    E.refuse_pro(ep)

    before = get_balance()
    money = afford(n, before)
    E.say(f"[{money['outcome']:<18}] деньги до старта{'':<28} "
          f"нужно ${money['need']}, есть "
          f"{'?' if money['have'] is None else '$' + str(money['have'])} | "
          f"{money['note']}", log=log)
    if money["outcome"] != PASS:
        E.say(f"ИТОГ: {money['outcome']} — батч НЕ НАЧАТ, заказов 0, "
              f"потрачено $0.0", log=log)
        return {"outcome": money["outcome"], "mode": mode, "planned": n,
                "attempted": 0, "passed": 0, "failed": 0, "unmeasured": 0,
                "money": money, "balance_before": money["have"],
                "balance_after": None, "spent_expected": 0.0,
                "spent_actual": None, "cells": [], "clips": [],
                "stopped_early": False,
                "exit_code": EXIT_BY_OUTCOME[money["outcome"]]}

    done, streak, stopped = [], 0, False
    for cell in grid:
        if stopped:
            cell = dict(cell, outcome=UNMEASURED, totals={}, clip=None,
                        launched=False,
                        note=(f"не запускалась: батч остановлен после "
                              f"{max_streak} неудач подряд"))
            done.append(cell)
            E.say(_cell_line(cell), log=log)
            continue
        cell = dict(cell, launched=True)
        cell_dir = where / "cells" / cell["name"]
        win = wins.get(str(cell["driving"]), (first, last))
        try:
            got = runner(client_photo=cell["person"], style_ref=cell["style"],
                         driving=cell["driving"], first=win[0], last=win[1],
                         out_dir=str(cell_dir), log=log, **cell_kwargs)
        except Exception as exc:                          # noqa: BLE001
            # Обвал ячейки — «не смогли», а НЕ «не годно»: упавшая сеть и
            # пустая очередь ничего не говорят о качестве продукта (Р1).
            cell.update(outcome=UNMEASURED, totals={}, clip=None,
                        note=f"прогон обвалился: {type(exc).__name__}: {exc}")
        else:
            # Ответ соседа разбирается по ФОРМЕ, а не по надежде: `None` и
            # строка «готово» — не вердикт, и `.get` по ним обвалил бы весь
            # батч посреди оплаченной матрицы (поймано прогоном теста).
            reply = got if isinstance(got, dict) else {}
            outcome = reply.get("outcome")
            if outcome not in (PASS, FAIL, UNMEASURED):
                cell.update(outcome=UNMEASURED, totals={}, clip=None,
                            note=(f"прогон ответил {type(got).__name__} без "
                                  f"вердикта: судить нечем"))
            else:
                note = f"встал на «{reply.get('stopped_at', '?')}»"
                clip, why = None, None
                if outcome == PASS:
                    clip, why = take(cell_dir / "final_9x16.mp4",
                                     clips_dir / f"{cell['name']}.mp4")
                    if clip is None:
                        # Е2: верим свидетельству, а не флагу. «Годно» без
                        # ролика на диске не проверено — значит не смогли.
                        outcome = UNMEASURED
                        note = f"вердикт «{PASS}», но ролика нет: {why}"
                    else:
                        note = f"ролик {clip}"
                cell.update(outcome=outcome, totals=reply.get("totals") or {},
                            clip=clip, note=note)
        done.append(cell)
        E.say(_cell_line(cell), log=log)
        streak = 0 if cell["outcome"] == PASS else streak + 1
        if streak >= max_streak:
            stopped = True
            E.say(f"ОСТАНОВ: {streak} неудач подряд при пороге {max_streak} — "
                  f"это уже не плохая пара, а поломка; дальше жгли бы по "
                  f"${E.KLING_PRICE_USD} за ячейку", log=log)

    attempted = sum(1 for c in done if c.get("launched"))
    passed = sum(1 for c in done if c["outcome"] == PASS)
    failed = sum(1 for c in done if c["outcome"] == FAIL)
    unmeasured = sum(1 for c in done if c["outcome"] == UNMEASURED)
    after = get_balance()
    spent_expected = plan_cost(attempted)
    spent_actual = (None if (before is None or after is None)
                    else round(float(before) - float(after), 4))
    outcome = E.verdict(passed + failed, failed, unmeasured)

    for c in done:
        E.say(f"      · {c['name']}: {c['outcome']} — {c.get('note', '')}",
              log=log)
    E.say(f"ИТОГ: {outcome} | годно {passed}, не годно {failed}, не смогли "
          f"{unmeasured} из {len(grid)} ячеек (запущено {attempted}) | "
          f"потрачено фактически "
          f"{'не смогли посчитать' if spent_actual is None else '$' + str(spent_actual)}"
          f" при ожидаемых ${spent_expected} | баланс "
          f"{before} -> {after} | роликов "
          f"{sum(1 for c in done if c.get('clip'))}", log=log)
    return {"outcome": outcome, "mode": mode, "planned": len(grid),
            "attempted": attempted, "passed": passed, "failed": failed,
            "unmeasured": unmeasured, "money": money,
            "balance_before": before, "balance_after": after,
            "spent_expected": spent_expected, "spent_actual": spent_actual,
            "cells": done, "clips": [c["clip"] for c in done if c.get("clip")],
            "stopped_early": stopped, "exit_code": EXIT_BY_OUTCOME[outcome]}


def main(argv=None) -> int:
    """Тонкая точка входа: разбор аргументов и вызов `run_batch` (Т5)."""
    import argparse                                      # noqa: PLC0415

    ap = argparse.ArgumentParser(description="батч сквозного стенда форка")
    ap.add_argument("--driving", action="append", required=True)
    ap.add_argument("--style", action="append", required=True)
    ap.add_argument("--person", action="append", required=True)
    ap.add_argument("--mode", default="full", choices=list(MODES))
    ap.add_argument("--window", required=True, help="первый:последний, напр. 100:199")
    ap.add_argument("--out", default="work/batch")
    a = ap.parse_args(argv)
    first, last = E.parse_window(a.window)
    got = run_batch(drivings=a.driving, styles=a.style, persons=a.person,
                    mode=a.mode, first=first, last=last, out_dir=a.out)
    return got["exit_code"]


if __name__ == "__main__":                               # pragma: no cover
    sys.exit(main())
