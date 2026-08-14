"""Живой предполёт шлюза: доказать каждую точку Pollinations, дешёвые раньше.

    python3 -m ball_reel.doctor            # бесплатное и дешёвое, БЕЗ трат на видео
    python3 -m ball_reel.doctor --video    # плюс один короткий клип (тратит баланс)

Порядок — по цене, чтобы отказ случился до траты: список моделей (бесплатно) ->
картинка (дёшево) -> редактирование картинки (дёшево) -> загрузка (дёшево) ->
TTS (дёшево) -> видео (только с --video). Каждая строка печатает исход, время и
причину.

ЧЕГО ЭТОТ МОДУЛЬ НЕ КАСАЕТСЯ. Локальный путь генерации (`--engine animatediff`)
наружу не ходит вовсе, и шлюз ему не нужен: там предполёт — это
`ball_reel.preflight_gpu`. Здесь проверяется запасной путь `--engine chain`, у
которого сшивка кейфреймов идёт через API.

ТРИ ИСХОДА, А НЕ ДВА. «Не смогли проверить» печатается словом НЕПРОВЕРЕНО и
считается отдельно. Раньше проверка, зависящая от упавшей предыдущей, просто не
выполнялась и не печаталась вовсе: в итоговом счёте её не было ни с одной
стороны, и отчёт «3/3» получался на прогоне, где половина не проверялась. Это
тот же дефект, что ловили в гейте позы, — непроверенное выглядит успехом.

И каждый отказ называет ЛЕЧЕНИЕ: строка «FAIL image» без «что делать» стоит
ровно столько же времени, сколько поиск по документации в тот момент, когда его
нет.
"""

from __future__ import annotations

from . import cure

import os
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("BALL_REEL_DOCTOR_OUT", "doctor_out"))

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "НЕПРОВЕРЕНО"

#: Сетевые беды, которые НЕ означают «сервис сломан». Закрытый прокси-политикой
#: хост, оборванный DNS и таймаут — это «проверить не вышло», и путать их с
#: отказом сервиса нельзя: лечения противоположные (просить оператора открыть
#: домен против чинить ключ).
NETWORK_EXCEPTIONS = ("ConnectionError", "Timeout", "ConnectTimeout",
                      "ReadTimeout", "SSLError", "ProxyError",
                      "NewConnectionError", "MaxRetryError")


def outcome(ok, detail: str) -> tuple:
    """(исход, текст). `ok is None` — не смогли проверить."""
    return (UNKNOWN if ok is None else (PASS if ok else FAIL)), detail


def _t(fn):
    t = time.time()
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 — предполёт сообщает, а не падает
        name = type(e).__name__
        if any(n in name for n in NETWORK_EXCEPTIONS):
            ok, detail = None, (
                f"{name}: {e} — сеть до шлюза не дошла, СЕРВИС НЕ ПРОВЕРЕН. "
                f"Если хост закрыт политикой прокси — не обходить, а попросить "
                f"оператора открыть его: без *.pollinations.ai путь --engine "
                f"chain не работает вовсе. Локальный путь (--engine "
                f"animatediff) от этого не зависит.")
        else:
            ok, detail = False, f"{name}: {e}"
    return ok, detail, time.time() - t


def main(argv: list[str]) -> int:
    import requests

    from . import pollinations

    OUT.mkdir(parents=True, exist_ok=True)
    base = pollinations._base()
    try:
        pollinations._key()
    except RuntimeError as e:
        print(f"{FAIL:<12}ключ            {e}\n"
              f"Лечение: {cure.set_env('POLLINATIONS_API_KEY', 'sk_...')} Без ключа проверять "
              f"нечего — все точки ниже требуют авторизации. Локальному пути "
              f"(--engine animatediff) ключ не нужен вовсе.")
        return 2

    marks: list[str] = []

    def check(name, fn, *, skip_if=None):
        """Одна строка отчёта. `skip_if` — причина, по которой не проверяли."""
        if skip_if:
            print(f"{UNKNOWN:<12}{name:<16}    -    {skip_if}")
            marks.append(UNKNOWN)
            return None
        ok, detail, dt = _t(fn)
        mark, detail = outcome(ok, detail)
        print(f"{mark:<12}{name:<16}{dt:5.1f}s  {detail}")
        marks.append(mark)
        return ok

    # 1. список моделей (бесплатно, без авторизации) — стек вообще на месте?
    def _models():
        r = requests.get(f"{base}/v1/models", timeout=60)
        ids = {m.get("id") for m in r.json().get("data", [])}
        need = ["kontext", "seedance-2.0", "eleven-multilingual-v2"]
        missing = [m for m in need if m not in ids]
        if missing:
            return False, (f"{len(ids)} моделей, НЕТ: {missing}. Модель "
                           f"переименована или снята — сверить имена с "
                           f"POLLINATIONS_CONTRACT.md и с этим же списком "
                           f"({base}/v1/models), а не подставлять похожее.")
        return True, f"{len(ids)} моделей, все нужные на месте"
    check("models", _models)

    # 2. текст -> картинка (дёшево)
    def _image():
        p = pollinations.image("a plain photo portrait, neutral background",
                               OUT / "img.jpg", model="flux",
                               width=512, height=512, seed=7)
        size = Path(p).stat().st_size
        if size <= 1000:
            return False, (f"{p} весит {size} Б — это не картинка, а страница "
                           f"ошибки. Открыть файл глазами: там обычно текст "
                           f"про лимит или про модерацию.")
        return True, f"-> {p} ({size // 1024} КБ)"
    have_face = check("image", _image)

    # 3. локальное лицо -> старт-кадр (дёшево, без медиа-хоста)
    def _edit():
        p = pollinations.images_edit(
            "Keep this exact face; put him mid-jump on a fitness ball, vertical.",
            OUT / "img.jpg", OUT / "start.jpg", model="kontext",
            width=720, height=1280)
        size = Path(p).stat().st_size
        return size > 1000, (f"-> {p} ({size // 1024} КБ)" if size > 1000 else
                             f"{p} весит {size} Б — см. строку image")
    have_start = check("images_edit", _edit,
                       skip_if=None if have_face else
                       "нет входной картинки: строка image не прошла — чинить "
                       "её, эта проверка НЕ выполнялась")

    # 4. загрузка -> URL медиа (нужен вайтлист на *.pollinations.ai)
    media_url: dict = {}

    def _upload():
        u = pollinations.upload(OUT / "start.jpg")
        media_url["url"] = u
        if not u.startswith("http"):
            return False, (f"ответ не похож на URL: {u[:70]}. Видео-модель "
                           f"принимает старт-кадр только ссылкой, локальный "
                           f"файл ей не передать.")
        return True, f"-> {u[:70]}"
    check("upload", _upload,
          skip_if=None if have_start else
          "нет старт-кадра: строка images_edit не прошла или не проверялась")

    # 5. TTS по-русски (дёшево)
    def _tts():
        p = pollinations.tts("Привет! Прыгай на мяче каждое утро.",
                             OUT / "voice.mp3", voice="nova",
                             model="eleven-multilingual-v2")
        size = Path(p).stat().st_size
        return size > 1000, (f"-> {p} ({size // 1024} КБ)" if size > 1000 else
                             f"{p} весит {size} Б — голосовая модель отказала, "
                             f"сверить имя модели со списком из строки models")
    check("tts", _tts)

    # 6. картинка -> видео (ТОЛЬКО с --video: эта строка тратит баланс)
    if "--video" in argv:
        def _video():
            p = pollinations.video(
                "The person jumps on the fitness ball; it compresses and rebounds.",
                OUT / "clip.mp4",
                model=os.environ.get("BALL_REEL_VIDEO_MODEL", "seedance-2.0"),
                image_url=media_url["url"],
                duration=int(os.environ.get("BALL_REEL_VIDEO_DURATION", "4")),
                aspect_ratio="9:16")
            size = Path(p).stat().st_size
            return size > 10000, (f"-> {p} ({size // 1024} КБ)" if size > 10000
                                  else f"{p} весит {size} Б — клип не пришёл, "
                                       f"баланс при этом мог списаться: "
                                       f"проверить расход до повтора")
        check("video(seedance)", _video,
              skip_if=None if media_url.get("url") else
              "нет URL старт-кадра (upload не прошёл) — сослаться не на что, "
              "и тратить баланс вслепую незачем")
    else:
        print(f"{UNKNOWN:<12}{'video':<16}    -    "
              f"не запускалось: --video тратит баланс. Это НЕ «видео "
              f"работает» — самая дорогая точка остаётся непроверенной.")
        marks.append(UNKNOWN)

    bad = marks.count(FAIL)
    unknown = marks.count(UNKNOWN)
    good = marks.count(PASS)
    print(f"\nПРОШЛО {good}, ОТКАЗОВ {bad}, НЕПРОВЕРЕНО {unknown} "
          f"(из {len(marks)}). Непроверенное — не «в порядке»: по каждой такой "
          f"строке решать отдельно.")
    print(f"артефакты в {OUT}/")
    return 0 if (bad == 0 and unknown == 0) else (1 if bad else 2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
