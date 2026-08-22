"""Приёмка ПЕРЕД прогоном: что именно блокирует, за секунды, а не за смену.

ЗАЧЕМ ОТДЕЛЬНЫЙ ПРИБОР. Две смены подряд выясняли блокеры вручную и обе
потратили на это основное время. Дешёвая проверка обязана идти раньше дорогой
(П2): отсутствующий баланс ловится за 300 мс, а не после нарезки, загрузки и
постановки в очередь.

ЧТО ПРОВЕРЯЕТСЯ, и почему именно это — каждый пункт стоил смены:

  1. домены        ЗАМЕРЕНО 22.08: политика песочницы менялась между сменами;
  2. ключи         есть/нет и ФОРМА (значение не печатается никогда);
  3. БАЛАНС        оба вендора отказали именно балансом, а не схемой;
  4. вход          боевого драйвинга нет в репозитории (`*.mp4` в .gitignore).

НЕГАТИВНЫЙ КОНТРОЛЬ ВСТРОЕН (И5). Проверка доменов спрашивает и заведомо
открытый `pypi.org`: если он тоже молчит, прибор меряет не политику вендора,
а собственную сломанную сеть, и обязан сказать «не смогли», а не «закрыто».

ТРИ ИСХОДА, а не два. «Баланс пуст» — это НЕ «сервис плохой» и НЕ «всё
хорошо»: это `не смогли проверить`, и оно не сворачивается ни в одну сторону.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PASS, FAIL, UNMEASURED = "годно", "не годно", "не смогли проверить"

#: ИЗМЕРЕНО 22.08.2026 curl-ом из этой песочницы: что открыто, что нет.
HOSTS = {
    "queue.fal.run": "fal.ai: постановка в очередь и опрос",
    "rest.fal.ai": "fal.ai: REST, загрузка входа, баланс",
    "v3.fal.media": "fal.ai: CDN, выдача результата",
    "api-singapore.klingai.com": "Kling: боевая база (ПОДТВЕРЖДЕНА ответом)",
    "api.klingai.com": "Kling: вторая база, отвечает так же",
}
#: Негативный контроль (И5): заведомо открытый хост.
CONTROL_HOST = "pypi.org"

DRIVING_SRC = "assets/driving_yogaball.mp4"
PHOTO = "assets/photoref_profile.jpg"
CUT = "work/bench_driving.mp4"

#: ПОДТВЕРЖДЕНО ответом сервиса 22.08 (см. fork_bench_kling): готовый Bearer.
KLING_BASE = "https://api-singapore.klingai.com"
KLING_PATH = "/v1/videos/motion-control"


def _get(url: str, headers=None, timeout=25):
    """Возвращает (код, тело) либо (None, причина). None — третий исход."""
    try:
        import httpx
    except ImportError:
        return None, "нет пакета httpx"
    try:
        r = httpx.get(url, headers=headers or {}, timeout=timeout)
        return r.status_code, r.text[:300]
    except Exception as exc:                        # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


def check_hosts() -> dict:
    """Открыты ли домены. Без негативного контроля вердикт недействителен."""
    ctl_code, ctl_note = _get(f"https://{CONTROL_HOST}/simple/")
    if ctl_code != 200:
        return {"outcome": UNMEASURED, "hosts": {},
                "note": (f"негативный контроль {CONTROL_HOST} дал {ctl_code} "
                         f"({ctl_note}) вместо 200. Сеть сломана у НАС — "
                         f"судить о политике вендоров нельзя")}
    seen, closed = {}, []
    for host, why in HOSTS.items():
        code, note = _get(f"https://{host}/")
        # Любой HTTP-код = соединение состоялось, хост открыт. 404 на корне
        # у fal.ai и есть норма: корень не обслуживается, а туннель работает.
        ok = code is not None
        seen[host] = {"code": code, "открыт": ok, "зачем": why, "ответ": note[:120]}
        if not ok:
            closed.append(host)
    if closed:
        return {"outcome": FAIL, "hosts": seen,
                "note": f"закрыты политикой: {', '.join(closed)}"}
    return {"outcome": PASS, "hosts": seen,
            "note": f"все {len(seen)} хостов открыты; контроль {CONTROL_HOST} 200"}


def check_keys() -> dict:
    """Форма ключей. ЗНАЧЕНИЕ НЕ ПЕЧАТАЕТСЯ И НЕ ВОЗВРАЩАЕТСЯ никогда."""
    out, missing = {}, []
    for name in ("FAL_KEY", "KLING_KEY"):
        val = os.environ.get(name)
        if not val:
            missing.append(name)
            out[name] = {"есть": False}
        else:
            out[name] = {"есть": True, "длина": len(val),
                         "двоеточий": val.count(":")}
    if missing:
        return {"outcome": FAIL, "keys": out,
                "note": f"нет в окружении: {', '.join(missing)}"}
    return {"outcome": PASS, "keys": out, "note": "оба ключа заданы"}


def check_fal_balance() -> dict:
    """Баланс fal.ai. ЗАМЕРЕНО 22.08: `/billing/user_balance` -> 200 и число.

    Негативный контроль того же хоста: `/users/me` отвечает 404, то есть
    эндпоинт действительно различает существующее и выдуманное, а 200 выше —
    не «сервис отвечает 200 на что угодно».
    """
    key = os.environ.get("FAL_KEY")
    if not key:
        return {"outcome": UNMEASURED, "note": "FAL_KEY не задан"}
    h = {"Authorization": f"Key {key}"}
    code, body = _get("https://rest.fal.ai/billing/user_balance", headers=h)
    if code is None:
        return {"outcome": UNMEASURED, "note": f"спросить не вышло: {body}"}
    if code != 200:
        return {"outcome": UNMEASURED, "http": code,
                "note": f"баланс не отдан, HTTP {code}: {body[:200]}"}
    try:
        amount = float(body.strip())
    except ValueError:
        return {"outcome": UNMEASURED, "http": code,
                "note": f"ответ 200, но это не число: {body[:120]}"}
    if amount <= 0:
        return {"outcome": FAIL, "balance": amount,
                "note": f"баланс fal.ai = {amount}. Прогон невозможен: сервис "
                        f"отвечает 403 «User is locked. Exhausted balance» "
                        f"ДО всякой валидации схемы"}
    return {"outcome": PASS, "balance": amount, "note": f"баланс fal.ai = {amount}"}


def check_kling_balance() -> dict:
    """Баланс Kling. Меряется единственным доступным способом — отказом.

    Отдельного эндпоинта баланса у Kling не нашлось: `/account/costs` отвечает
    200 с пустым телом. Зато боевой метод при пустом счёте отвечает ровно
    `429 / code 1102 / Account balance not enough` — ЗАМЕРЕНО 22.08. Запрос
    ниже намеренно НЕ создаёт задачу: `video_url` заведомо нескачиваемый.
    """
    key = os.environ.get("KLING_KEY")
    if not key:
        return {"outcome": UNMEASURED, "note": "KLING_KEY не задан"}
    try:
        import httpx
    except ImportError:
        return {"outcome": UNMEASURED, "note": "нет пакета httpx"}
    probe = {"model_name": "kling-v2-6", "mode": "std",
             "character_orientation": 0,
             "image_url": "https://example.invalid/probe.jpg",
             "video_url": "https://example.invalid/probe.mp4"}
    try:
        r = httpx.post(KLING_BASE + KLING_PATH,
                       headers={"Authorization": f"Bearer {key}",
                                "Content-Type": "application/json"},
                       json=probe, timeout=60)
    except Exception as exc:                        # noqa: BLE001
        return {"outcome": UNMEASURED,
                "note": f"{type(exc).__name__}: {str(exc)[:200]}"}
    try:
        body = r.json()
    except ValueError:
        body = {"_raw": r.text[:300]}
    code = body.get("code")
    if code == 1102:
        return {"outcome": FAIL, "http": r.status_code, "code": code,
                "note": "баланс Kling пуст: code 1102 «Account balance not "
                        "enough». Схема при этом ПРОШЛА валидацию целиком — "
                        "упёрлись именно в деньги, а не в поля"}
    if code == 1001:
        return {"outcome": FAIL, "http": r.status_code, "code": code,
                "note": f"KLING_KEY не принят: {body.get('message')}"}
    return {"outcome": UNMEASURED, "http": r.status_code, "code": code,
            "note": f"ответ, которого прибор не знает: {json.dumps(body, ensure_ascii=False)[:250]}"}


def check_input() -> dict:
    """Вход. Без боевого драйвинга «один вход на всех» недостижим."""
    seen = {}
    for path in (DRIVING_SRC, PHOTO):
        p = Path(path)
        seen[path] = p.stat().st_size if p.exists() else None
    if seen[DRIVING_SRC] is None:
        return {"outcome": FAIL, "files": seen,
                "note": f"{DRIVING_SRC} нет на диске. Резать нечего, а подменять "
                        f"драйвинг другим файлом НЕЛЬЗЯ: сравнение потеряет "
                        f"смысл, ради которого затевалось. Причина отсутствия — "
                        f"`*.mp4` в .gitignore, нужен `git add -f`"}
    if seen[PHOTO] is None:
        return {"outcome": FAIL, "files": seen, "note": f"{PHOTO} нет на диске"}
    return {"outcome": PASS, "files": seen,
            "note": f"оба входа на месте: драйвинг {seen[DRIVING_SRC]} Б, "
                    f"фото {seen[PHOTO]} Б"}


CHECKS = (("домены", check_hosts), ("ключи", check_keys),
          ("баланс fal.ai", check_fal_balance),
          ("баланс Kling", check_kling_balance), ("вход", check_input))


def preflight() -> dict:
    """Все проверки. Рядом с вердиктом ЧИСЛА: проверено / блокирует / не смогли."""
    reports = {}
    for name, fn in CHECKS:
        try:
            reports[name] = fn()
        except Exception as exc:                    # noqa: BLE001
            reports[name] = {"outcome": UNMEASURED,
                             "note": f"проверка упала: {type(exc).__name__}: {exc}"}
    blocking = [n for n, r in reports.items() if r["outcome"] == FAIL]
    unknown = [n for n, r in reports.items() if r["outcome"] == UNMEASURED]
    if blocking:
        outcome = FAIL
    elif unknown:
        # Ноль блокеров при непроверенных пунктах — НЕ успех (Р2).
        outcome = UNMEASURED
    else:
        outcome = PASS
    return {"outcome": outcome, "checks": reports,
            "проверено": len(reports), "блокирует": len(blocking),
            "не смогли": len(unknown),
            "блокеры": blocking, "неизвестно": unknown}


if __name__ == "__main__":
    rep = preflight()
    for name, r in rep["checks"].items():
        print(f"[{r['outcome']:<18}] {name:<14} {r['note']}")
    print(f"\nисход: {rep['outcome']}")
    print(f"проверено {rep['проверено']}, блокирует {rep['блокирует']}, "
          f"не смогли {rep['не смогли']}")
    if rep["блокеры"]:
        print(f"блокеры: {', '.join(rep['блокеры'])}")
    if rep["неизвестно"]:
        print(f"не смогли проверить: {', '.join(rep['неизвестно'])}")
    raise SystemExit(0 if rep["outcome"] == PASS else 1)
