"""Эталон на ТОЙ ЖЕ модели: fal.ai, Wan 2.2 Animate, режим replace.

ЗАЧЕМ ЭТО ДОРОЖЕ ЦЕНЫ. На fal.ai крутится тот же движок в том же режиме, что
собран у нас. Значит их ролик — эталон НАШЕЙ реализации: если он выглядит
лучше нашего, дефект у нас, а не в модели. И это единственный способ узнать,
какие числа бывают у ГОДНОГО ролика, — своих у нас нет ни одного.

## СОСТОЯНИЕ НА 22.08.2026, ВТОРАЯ СМЕНА: домены открылись, деньги кончились

Прошлая редакция этого файла числила блокером сетевую политику. ПЕРЕМЕРЕНО:

    queue.fal.run     404 на корне  -> туннель работает, хост ОТКРЫТ
    rest.fal.ai       404 на корне  -> ОТКРЫТ
    v3.fal.media      404 на корне  -> ОТКРЫТ
    fal.ai            соединения нет -> сайт и его OpenAPI ЗАКРЫТЫ политикой
    pypi.org          200            -> негативный контроль (И5): сеть цела

Новый блокер, и он ВЫШЕ прежнего:

    GET  rest.fal.ai/billing/user_balance   200, тело «0.0»
    POST queue.fal.run/<любой эндпоинт>     403 «User is locked.
                                            Reason: Exhausted balance.»

Баланс аккаунта — НОЛЬ, и сервис запирает пользователя ДО всякой валидации
схемы. Загрузка входа заперта тем же: `rest.fal.ai/storage/auth/token` тоже
отдаёт 403 lock. То есть без пополнения нельзя даже выложить файл.

## ИМЯ ЭНДПОИНТА: ПОДТВЕРДИТЬ НЕ УДАЛОСЬ, И ПОЧТИ ОШИБЛИСЬ

Соблазнительное наблюдение: `fal-ai/wan/v2.2-14b/animate/replace` отвечает
403 (lock), а `fal-ai/nosuchvendor/...` — 404 «Application not found». Похоже
на подтверждение имени. НЕГАТИВНЫЙ КОНТРОЛЬ ЭТО ОПРОВЕРГ (И5):

    fal-ai/wan/v2.2-14b/animate/replace      403 lock
    fal-ai/wan/v2.2-14b/animate/replaceXX    403 lock   <- ВЫДУМАННЫЙ, тоже 403
    fal-ai/wan/v9.9-99b/animate/replace      403 lock   <- ВЫДУМАННЫЙ, тоже 403
    fal-ai/nosuchvendor/nosuchmodel/...      404 «Application "nosuchvendor" not found»

Без ключа сервис говорит прямо: «Cannot access application "fal-ai/wan"».
Значит резолвится ТОЛЬКО первый сегмент. Подтверждено существование
приложения `fal-ai/wan` — и НИЧЕГО не подтверждено про хвост
`v2.2-14b/animate/replace` и про имена полей.

**Вывод, который надо держать: имя эндпоинта и поля `video_url`/`image_url`/
`resolution` ПО-ПРЕЖНЕМУ НЕПРОВЕРЕНЫ (Ц10).** Они взяты из поиска. Первый же
вызов после пополнения баланса их подтвердит или опровергнет, и ошибка тут
ЦЕННЕЕ успеха: в ней приезжают настоящие имена. Ради этого скрипт печатает
ответ и ошибку ЦЕЛИКОМ.

Достать схему в обход lock'а не вышло ничем: без ключа 401, с ключом 403,
OpenAPI живёт на закрытом `fal.ai`. Обходить политику нельзя (Ц3).

## ДОКАЗАНО КОМАНДОЙ (держится с прошлой смены)

  - пакет `fal-client` 1.0.1, `fal_client.subscribe(application, arguments)`;
  - `fal_client.upload_file(path) -> str` СУЩЕСТВУЕТ — выкладывать вход на
    сторонний HTTP-хост не нужно, `BENCH_VIDEO_URL`/`BENCH_PHOTO_URL` не нужны;
  - `FAL_KEY` ВАЛИДЕН: с ним 403 lock, с испорченным — 401 «No user found
    for Key ID and Secret». То есть отказ именно про деньги, а не про ключ.

## КЛЮЧ
Только из окружения (`FAL_KEY`). В код, в лог и в карточку он не попадает
никогда: ключ, попавший в историю git, отзывается у вендора, а не убирается
коммитом.

## ЗАПУСК
    pip install fal-client
    export FAL_KEY=...
    python3 -m fork_bench.fork_bench_fal 480p
    python3 -m fork_bench.fork_bench_fal 720p
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

#: НЕПРОВЕРЕНО: хвост пути из поиска. Подтверждён только сегмент `fal-ai/wan`.
ENDPOINT = "fal-ai/wan/v2.2-14b/animate/replace"
#: НЕПРОВЕРЕНО: имена полей из мануала, не из ответа сервиса.
FIELD_VIDEO, FIELD_IMAGE, FIELD_RESOLUTION = "video_url", "image_url", "resolution"

DRIVING = "work/bench_driving.mp4"
PHOTO = "assets/photoref_profile.jpg"
JOURNAL = "work/bench_fal_journal.jsonl"

PASS, FAIL, UNMEASURED = "годно", "не годно", "не смогли проверить"


def journal(record: dict) -> None:
    """Пишем ДО просмотра ролика и независимо от исхода (И6)."""
    Path(JOURNAL).parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def classify(exc: Exception) -> tuple[str, str]:
    """Три исхода, а не два. «Денег нет» — это НЕ «схема плохая»."""
    text = str(exc)
    if "Exhausted balance" in text or "User is locked" in text:
        return UNMEASURED, ("баланс fal.ai исчерпан: сервис запирает аккаунт "
                            "ДО валидации схемы. Это НЕ «схема неверна» и НЕ "
                            "«ролик плохой» — это «померить нечем»")
    if "No user found" in text or "401" in text:
        return UNMEASURED, "FAL_KEY не принят сервисом"
    return FAIL, f"{type(exc).__name__}: {text[:600]}"


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
        outcome, note = classify(exc)
        rec = {"outcome": outcome, "resolution": resolution, "endpoint": ENDPOINT,
               "sent_fields": [FIELD_VIDEO, FIELD_IMAGE, FIELD_RESOLUTION],
               "waited_s": round(time.time() - started, 1),
               "error_type": type(exc).__name__, "error": str(exc)[:4000]}
        journal(rec)
        return {**rec, "note": note}

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
