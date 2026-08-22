"""Планка тир-1: Kling Motion Control. ДРУГАЯ модель, не «то же дешевле».

Kling здесь не для честного сравнения цены с нашей — он отвечает на вопрос
«как выглядит верх рынка». Сравнивать нашу цену прямо с ним нечестно, и в
таблице он стоит отдельной колонкой именно поэтому.

## СХЕМА БОЛЬШЕ НЕ УГАДЫВАЕТСЯ: она ВЫНУТА ИЗ ОТВЕТОВ СЕРВИСА 22.08.2026

Прошлая редакция этого файла требовала `KLING_BASE`/`KLING_PATH` из окружения
и честно отказывалась угадывать. Теперь угадывать не нужно: домены открылись,
и каждое имя ниже подтверждено ответом сервиса, а не памятью (Ц10).

    база     https://api-singapore.klingai.com   200; api.klingai.com отвечает так же
    путь     /v1/videos/motion-control           GET списка задач -> code 0 SUCCEED

НЕГАТИВНЫЙ КОНТРОЛЬ ПУТИ (И5), тем же ключом и тем же методом:

    /v1/videos/motion-control      200   <- есть
    /v1/videos/motion_control      404
    /v1/videos/motioncontrol       404
    /v2/videos/motion-control      404
    /v1/videos/ZZZ_negative_ZZZ    404

То есть прибор различает «путь есть» и «пути нет», а не отвечает 200 на что
угодно.

## АВТОРИЗАЦИЯ: подозрение прошлой смены ОПРОВЕРГНУТО ПРОГОНОМ

Прошлая смена записала: `KLING_KEY` — 57 символов, двоеточий 0, «на пару
AccessKey/SecretKey не похож», и числила это первым подозреваемым.

ЗАМЕРЕНО: `Authorization: Bearer <KLING_KEY>` принимается сервисом —
`{"code":0,"message":"SUCCEED"}`. Никакой JWT собирать не надо. Контроли:

    без заголовка вовсе      401  code 1001 «Authorization is empty»
    мусорный ASCII-ключ      code 1002 «token was expected to have 3 parts»
    настоящий ключ           code 0 SUCCEED, дальше валидация полей

Оговорка, чтобы наблюдение не выдавалось за большее: сообщение про «3 parts»
показывает, что у Kling есть и JWT-ветка. Наш ключ (0 точек) в неё не попадает
и принимается напрямую. ПОЧЕМУ сервис принимает оба вида — НЕПРОВЕРЕНО;
ЧТО принимает наш — доказано ответом.

## ПОЛЯ: каждое имя и каждое допустимое значение названы САМИМ СЕРВИСОМ

Схема вытащена лестницей отказов: пустое тело -> сервис называет недостающее
поле -> добавляем -> называет следующее. Валидация дошла до конца и упёрлась
в деньги, а не в поля.

    model_name             обязателен; «must be 'kling-v2-6' or 'kling-v3'»
                           (сообщение сервиса дословно; kling-v1 отвергнут)
    image_url              обязателен; «imageUrl: must not be blank»
    video_url              обязателен; «videoUrl: must not be blank»
    mode                   обязателен; «video mode must be specified»; 'std' принят
    character_orientation  обязателен; принято 0 и 1 (и «0»/«1»),
                           2, 3 и все опробованные слова отвергнуты

АСИММЕТРИЯ ЗАГРУЗКИ, ЗАМЕРЕНА ОТДЕЛЬНО И ВАЖНА ДЛЯ ПЛАНА:

    image_url = base64 реального фото     -> прошло до проверки баланса
    image_url = мусор                     -> «File is not in a valid base64 format»
    video_url = base64 реального mp4      -> «Video URL is invalid»
    video_url = мусор                     -> «Video URL is invalid»

То есть **фото можно слать байтами, а видео — только ссылкой.** Внешний
HTTP-хост для драйвинга нужен, и это открытый вопрос: CDN fal.ai на эту роль
не годится, пока его аккаунт заперт балансом.

## ЧЕГО МЫ ПО-ПРЕЖНЕМУ НЕ ЗНАЕМ (Ц4)

  - что означают 0 и 1 в `character_orientation`. Оба приняты, смысл ни одного
    не подтверждён. Для нашего входа это НЕ мелочь: фотореференс снят СО СПИНЫ,
    а драйвинг начинается лицом в камеру — если поле про это, оно и есть
    развилка между «модель не тянет» и «референс не с той стороны»;
  - допустимые значения `mode`, кроме принятого 'std';
  - есть ли `duration` и что он делает: пока `mode` не задан, сервис
    отвечает про `mode`, а после — про `character_orientation`;
  - тарификация: счёта не было НИ ОДНОГО, баланс пуст.
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

#: ПОДТВЕРЖДЕНО ответом сервиса 22.08.2026, с негативным контролем пути.
BASE = os.environ.get("KLING_BASE", "https://api-singapore.klingai.com")
PATH = os.environ.get("KLING_PATH", "/v1/videos/motion-control")

#: ВЫБРАНО из двух названных сервисом: «must be 'kling-v2-6' or 'kling-v3'».
MODEL = "kling-v2-6"
#: ПОДТВЕРЖДЕНО: 'std' принято. Другие значения НЕ проверены.
MODE = "std"
#: ПОДТВЕРЖДЕНО: принимаются 0 и 1. СМЫСЛ НИ ОДНОГО НЕ ПОДТВЕРЖДЁН.
ORIENTATION = 0

PASS, FAIL, UNMEASURED = "годно", "не годно", "не смогли проверить"


def journal(record: dict) -> None:
    """Пишем ДО просмотра ролика и при неуспехе тоже (И6)."""
    Path(JOURNAL).parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def photo_field(path: str) -> str:
    """Фото уходит байтами: ЗАМЕРЕНО, что `image_url` принимает base64."""
    return base64.b64encode(Path(path).read_bytes()).decode()


def run(video_url: str | None = None) -> dict:
    """`video_url` — ссылка на драйвинг. Байтами он НЕ принимается (замерено)."""
    key = os.environ.get("KLING_KEY")
    if not key:
        return {"outcome": UNMEASURED, "note": "KLING_KEY не задан в окружении"}
    if not Path(PHOTO).exists():
        return {"outcome": UNMEASURED, "note": f"фото {PHOTO} нет на диске"}
    video_url = video_url or os.environ.get("BENCH_VIDEO_URL")
    if not video_url:
        return {"outcome": UNMEASURED,
                "note": (f"нужна HTTP-ссылка на драйвинг. ЗАМЕРЕНО 22.08: "
                         f"`video_url` принимает ТОЛЬКО ссылку, base64 "
                         f"отвергается («Video URL is invalid»). Локальный "
                         f"{DRIVING} отправить нечем — передайте ссылку "
                         f"аргументом или в BENCH_VIDEO_URL")}
    try:
        import httpx
    except ImportError:
        return {"outcome": UNMEASURED, "note": "pip install httpx"}

    payload = {"model_name": MODEL, "mode": MODE,
               "character_orientation": ORIENTATION,
               "image_url": photo_field(PHOTO), "video_url": video_url}
    sent = {k: (v if k != "image_url" else f"<base64 {len(v)} символов>")
            for k, v in payload.items()}
    started = time.time()
    try:
        resp = httpx.post(BASE + PATH,
                          headers={"Authorization": f"Bearer {key}",
                                   "Content-Type": "application/json"},
                          json=payload, timeout=300)
        waited = time.time() - started
    except Exception as exc:                        # noqa: BLE001
        rec = {"outcome": UNMEASURED, "sent": sent,
               "waited_s": round(time.time() - started, 1),
               "error_type": type(exc).__name__, "error": str(exc)[:2000]}
        journal(rec)
        return {**rec, "note": f"{type(exc).__name__}: {str(exc)[:400]}"}

    try:
        body = resp.json()
    except ValueError:
        body = {"_raw_text": resp.text[:4000]}
    code = body.get("code")
    # Три исхода, а не два: «денег нет» и «ключ не тот» — это НЕ «плохой ролик».
    if code == 1102:
        outcome, note = UNMEASURED, ("баланс Kling пуст (code 1102). Схема "
                                     "прошла валидацию целиком — упёрлись "
                                     "в деньги, а не в поля")
    elif code in (1001, 1002, 1003, 1004):
        outcome, note = UNMEASURED, f"авторизация: code {code}, {body.get('message')}"
    elif resp.status_code == 200 and code == 0:
        outcome, note = PASS, f"задача принята за {waited:.1f} с"
    else:
        outcome, note = FAIL, f"HTTP {resp.status_code}, code {code}: {body.get('message')}"

    rec = {"outcome": outcome, "http_status": resp.status_code, "code": code,
           "waited_s": round(waited, 1), "sent": sent, "response": body}
    journal(rec)
    return {**rec, "note": note}


if __name__ == "__main__":
    import sys
    rep = run(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"исход: {rep['outcome']}")
    print(rep.get("note", ""))
    if "response" in rep:
        print("\nПОЛНЫЙ ОТВЕТ (это и есть фактическая схема):")
        print(json.dumps(rep["response"], ensure_ascii=False, indent=2)[:4000])
    raise SystemExit(0 if rep["outcome"] == PASS else 1)
