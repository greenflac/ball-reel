"""Эталон на ТОЙ ЖЕ модели: fal.ai, Wan 2.2 Animate, режим replace.

ЗАЧЕМ ЭТО ДОРОЖЕ ЦЕНЫ. На fal.ai крутится тот же движок в том же режиме, что
собран у нас. Значит их ролик — эталон НАШЕЙ реализации: если он выглядит
лучше нашего, дефект у нас, а не в модели. И это единственный способ узнать,
какие числа бывают у ГОДНОГО ролика, — своих у нас нет ни одного.

## ЧТО ЗДЕСЬ ДОКАЗАНО ПРОГОНОМ, А ЧТО НЕТ (Ц4/Ц10)

ДОКАЗАНО командой 22.08.2026 в этой песочнице:
  - пакет `fal-client` существует, версия 1.0.1;
  - `fal_client.subscribe(application, arguments, ...)` существует;
  - `fal_client.upload_file(path) -> str` СУЩЕСТВУЕТ. Этим закрыт пункт
    мануала «fal.ai принимает URL, а не байты, либо загрузка клиентом
    (НЕПРОВЕРЕНО)»: загрузка клиентом есть, и переменные `BENCH_VIDEO_URL` /
    `BENCH_PHOTO_URL` больше не нужны;
  - хосты, которые клиент реально дёргает, вынуты из констант пакета:
        queue.fal.run    очередь: постановка и опрос
        rest.fal.ai      REST
        v3.fal.media     CDN: загруженные файлы и выдача
        auth.fal.ai      только OAuth-вход, при FAL_KEY не нужен

НЕ ДОКАЗАНО, потому что домены закрыты сетевой политикой (CONNECT -> 403):
  - имя эндпоинта `fal-ai/wan/v2.2-14b/animate/replace`;
  - имена полей `video_url`, `image_url`, `resolution`;
  - что счёт и латентность именно такие, как в прайсе.
Всё это взято из мануала, то есть из поиска. ПЕРВЫЙ ЖЕ ВЫЗОВ их подтвердит
или опровергнет — ради этого скрипт печатает ответ и ошибку ЦЕЛИКОМ, включая
имена полей, которые вернёт сервис.

## КЛЮЧ
Только из окружения (`FAL_KEY`). В код, в лог и в карточку он не попадает
никогда: ключ, попавший в историю git, отзывается у вендора, а не убирается
коммитом.

## ЗАПУСК
    pip install fal-client
    export FAL_KEY=...
    python3 fork_bench/fork_bench_fal.py 480p
    python3 fork_bench/fork_bench_fal.py 720p
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

#: НЕПРОВЕРЕНО: имя из мануала (поиск), не из ответа сервиса.
ENDPOINT = "fal-ai/wan/v2.2-14b/animate/replace"
#: НЕПРОВЕРЕНО: имена полей из мануала, не из ответа сервиса.
FIELD_VIDEO, FIELD_IMAGE, FIELD_RESOLUTION = "video_url", "image_url", "resolution"

DRIVING = "work/bench_driving.mp4"
PHOTO = "assets/photoref_profile.jpg"
JOURNAL = "work/bench_fal_journal.jsonl"

PASS, FAIL, UNMEASURED = "годно", "не годно", "не смогли проверить"


def journal(record: dict) -> None:
    """Пишем ДО просмотра ролика и независимо от исхода.

    Отрицательный результат с числом и условиями — тоже запись (И6): серия
    неудач это измеренная граница, а без записи следующая сессия переставит
    те же ручки заново.
    """
    Path(JOURNAL).parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(resolution: str = "480p") -> dict:
    if not os.environ.get("FAL_KEY"):
        return {"outcome": UNMEASURED, "note": "FAL_KEY не задан в окружении"}
    for path in (DRIVING, PHOTO):
        if not Path(path).exists():
            return {"outcome": UNMEASURED,
                    "note": f"входа {path} нет на диске: отправлять нечего"}
    try:
        import fal_client
    except ImportError:
        return {"outcome": UNMEASURED, "note": "pip install fal-client"}

    started = time.time()
    try:
        video_url = fal_client.upload_file(DRIVING)
        image_url = fal_client.upload_file(PHOTO)
        uploaded = time.time() - started

        args = {FIELD_VIDEO: video_url, FIELD_IMAGE: image_url,
                FIELD_RESOLUTION: resolution}
        got = fal_client.subscribe(ENDPOINT, arguments=args,
                                   on_queue_update=lambda s: print(f"  … {s}"))
        waited = time.time() - started
    except Exception as exc:                       # noqa: BLE001 — печатаем ЦЕЛИКОМ
        # Ошибка здесь ЦЕННЕЕ успеха: в ней приезжают настоящие имена полей.
        rec = {"outcome": FAIL, "resolution": resolution, "endpoint": ENDPOINT,
               "sent_fields": [FIELD_VIDEO, FIELD_IMAGE, FIELD_RESOLUTION],
               "waited_s": round(time.time() - started, 1),
               "error_type": type(exc).__name__, "error": str(exc)[:4000]}
        journal(rec)
        return {**rec, "note": f"{type(exc).__name__}: {str(exc)[:600]}"}

    rec = {"outcome": PASS, "resolution": resolution, "endpoint": ENDPOINT,
           "upload_s": round(uploaded, 1), "waited_s": round(waited, 1),
           "response": got}
    journal(rec)
    return {**rec, "note": f"готово за {waited:.1f} с от отправки"}


if __name__ == "__main__":
    res = sys.argv[1] if len(sys.argv) > 1 else "480p"
    rep = run(res)
    print(f"\nисход: {rep['outcome']}")
    print(rep.get("note", ""))
    if "response" in rep:
        print("\nПОЛНЫЙ ОТВЕТ СЕРВИСА (это и есть фактическая схема):")
        print(json.dumps(rep["response"], ensure_ascii=False, indent=2))
    print(f"\nжурнал: {JOURNAL}")
    raise SystemExit(0 if rep["outcome"] == PASS else 1)
