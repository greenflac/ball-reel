"""Планка тир-1: Kling Motion Control. ДРУГАЯ модель, не «то же дешевле».

Kling здесь не для честного сравнения цены с нашей — он отвечает на вопрос
«как выглядит верх рынка». Сравнивать нашу цену прямо с ним нечестно, и в
таблице он стоит отдельной колонкой именно поэтому.

## ЧЕСТНОЕ ПРЕДУПРЕЖДЕНИЕ О СХЕМЕ (Ц10)

В отличие от fal.ai, где пакет `fal-client` установлен и его контракт вынут
командой, про HTTP-схему Kling у меня НЕТ НИ ОДНОГО ПРОВЕРЕННОГО ФАКТА:
домен закрыт политикой, официального пакета не ставилось, ответа сервиса не
видели. Поэтому здесь НЕ ВЫДУМЫВАЕТСЯ путь эндпоинта: он берётся из окружения,
и без него скрипт отказывается работать, вместо того чтобы угадать и молча
отправить запрос не туда.

    KLING_BASE      база API, напр. https://api.klingai.com   (из кабинета)
    KLING_PATH      путь метода Motion Control                (из документации)
    KLING_KEY       ключ                                      (уже в окружении)

## ЗАМЕР ФОРМЫ КЛЮЧА, снятый 22.08 (значение не печаталось)

    FAL_KEY    длина 69, ровно одно двоеточие, сегменты 36:32  -> пара id:secret
    KLING_KEY  длина 57, двоеточий 0                           -> ОДИН токен

У Kling в открытой документации фигурирует пара AccessKey/SecretKey, из которой
собирается JWT. Одиночный токен на пару не похож. ЧТО ЭТО ЗНАЧИТ — НЕПРОВЕРЕНО:
либо это готовый Bearer-токен, либо половина пары, либо ключ другого профиля.
Выяснится ПЕРВЫМ ЖЕ ОТВЕТОМ сервиса, и ради этого ответ печатается целиком.

Ограничения вендора (НЕПРОВЕРЕНО, из мануала): видео .mp4/.mov, 3-30 с, до
100 МБ; фото .jpg/.png, до 10 МБ, сторона > 300 px, соотношение 2:5..5:2.
Наш кусок 4.208 с и фото 736x1318 в эти рамки укладываются.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

DRIVING = "work/bench_driving.mp4"
PHOTO = "assets/photoref_profile.jpg"
JOURNAL = "work/bench_kling_journal.jsonl"

PASS, FAIL, UNMEASURED = "годно", "не годно", "не смогли проверить"


def journal(record: dict) -> None:
    Path(JOURNAL).parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run() -> dict:
    key = os.environ.get("KLING_KEY")
    base = os.environ.get("KLING_BASE")
    path = os.environ.get("KLING_PATH")
    if not key:
        return {"outcome": UNMEASURED, "note": "KLING_KEY не задан"}
    if not base or not path:
        return {"outcome": UNMEASURED,
                "note": ("KLING_BASE и/или KLING_PATH не заданы. Путь эндпоинта "
                         "НЕ УГАДЫВАЕТСЯ намеренно (Ц10): непроверенное внешнее "
                         "имя в код не попадает. Возьмите их из кабинета Kling "
                         "и передайте окружением")}
    for p in (DRIVING, PHOTO):
        if not Path(p).exists():
            return {"outcome": UNMEASURED,
                    "note": f"входа {p} нет на диске: отправлять нечего"}
    try:
        import requests
    except ImportError:
        return {"outcome": UNMEASURED, "note": "pip install requests"}

    payload = {
        "image": base64.b64encode(Path(PHOTO).read_bytes()).decode(),
        "video": base64.b64encode(Path(DRIVING).read_bytes()).decode(),
    }
    started = time.time()
    try:
        resp = requests.post(f"{base.rstrip('/')}/{path.lstrip('/')}",
                             headers={"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"},
                             json=payload, timeout=300)
        waited = time.time() - started
    except Exception as exc:                        # noqa: BLE001
        rec = {"outcome": FAIL, "waited_s": round(time.time() - started, 1),
               "error_type": type(exc).__name__, "error": str(exc)[:2000]}
        journal(rec)
        return {**rec, "note": f"{type(exc).__name__}: {str(exc)[:400]}"}

    try:
        body = resp.json()
    except ValueError:
        body = {"_raw_text": resp.text[:4000]}
    rec = {"outcome": PASS if resp.ok else FAIL,
           "http_status": resp.status_code, "waited_s": round(waited, 1),
           "response": body}
    journal(rec)
    # Ответ печатается ЦЕЛИКОМ и при неуспехе тоже: в ошибке приезжает
    # настоящая схема — имена полей, которых мы не знаем.
    return {**rec, "note": f"HTTP {resp.status_code} за {waited:.1f} с"}


if __name__ == "__main__":
    rep = run()
    print(f"исход: {rep['outcome']}")
    print(rep.get("note", ""))
    if "response" in rep:
        print("\nПОЛНЫЙ ОТВЕТ (это и есть фактическая схема):")
        print(json.dumps(rep["response"], ensure_ascii=False, indent=2)[:4000])
    raise SystemExit(0 if rep["outcome"] == PASS else 1)
