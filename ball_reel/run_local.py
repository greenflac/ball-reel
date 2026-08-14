"""Один прогон от начала до конца на локальной карте.

    # локально, без единого сетевого вызова в пути генерации (умолчание):
    python3 -m ball_reel.run_local --face face.jpg --conditions kit/conditions \\
        --prompt "a woman exercising on a fitness ball in a bright studio, \\
                  black sports top and leggings, natural light, photographic"

    # запасной путь через шлюз, тот самый, что уже проверен живьём:
    python3 -m ball_reel.run_local --engine chain ...

    python3 -m ball_reel.run_local ... --smoke      # остановиться до генерации

Собрано в одну команду не для красоты. Разложенный по десяти сниппетам прогон
на своей машине разваливается ровно там, где что-то пошло не так: половина
шагов сделана, половина нет, и непонятно, какой артефакт от какого запуска.
Здесь каждый этап печатает измерение и останавливает всё при провале — тот же
принцип, что и в предполёте, только теперь дорога не арендная минута, а твоё
время на разбор.

ДВА ДВИЖКА, ОДИН ГЕЙТ
---------------------

`--engine animatediff` (умолчание) — целевой путь: кадры считаются СОВМЕСТНО
одним проходом AnimateDiff + ControlNet + FaceID, наружу не ходит ничего.
Связность между кадрами обеспечивает модуль движения внутри той же UNet, а не
чужой сервис.

`--engine chain` — запасной: кейфреймы рисуются поодиночке и сшиваются
видеомоделью через шлюз. Он ПРОВЕРЕН ЖИВЬЁМ и потому остаётся; терять
работающий путь ради нового — это менять измеренное на предполагаемое.

Судятся оба ОДНИМИ И ТЕМИ ЖЕ мерами: `smoke_verdict` на узлах, `garment_drift`,
`garment_fit`, `clip_expression` на кадрах и `clip_verdict` на клипе. Это не
экономия кода: числа двух путей сравнимы ровно постольку, поскольку получены
одним прибором. Отдельный «гейт для AnimateDiff» означал бы, что выбор движка
меняет и то, что считается годным, — а тогда сравнивать их нечем.

ЧТО ДЕЛАЕТ, по шагам (animatediff):
  1. дешёвые проверки: движок, имя motion LoRA, ffmpeg, длина промта, хватает
     ли условий на окно модуля движения — всё ДО единого байта весов;
  2. предполёт (карта, веса, условия, лицо) — не начинать на сломанной машине;
  3. КАДРИРОВКА ИЗ МАНИФЕСТА УСЛОВИЙ (а не из флага), план по VRAM, прогноз
     «увидит ли гейт лицо», окно условий, эмбеддинг лица;
  4. сборка пайплайна и ПЕЧАТЬ ТОГО, ЧТО РЕАЛЬНО ПРИЦЕПЛЕНО (`active_loras`);
  5. генерация, кадры на диск, mp4 через ffmpeg;
  6. гейт: узловые меры + клиповые, теми же порогами.

Для `chain` шаги те же, кроме 5: там сшивка через шлюз, и это единственный
платный шаг во всём файле — он считается заранее и печатается.

ЗАМЕР «С LoRA / БЕЗ НЕЁ» (`--ab-motion-lora pan-left`) прогоняет шаги 4-6 ДВАЖДЫ
с одним сидом и одним окном условий и печатает числа гейта рядом. Одной
командой — потому что ручная пересборка между двумя прогонами меняет больше
одной вещи, и разница потом не приписывается ничему конкретному.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Движки генерации. Порядок значим: первый — умолчание, и это целевой путь
#: продукта («внешний API из пути генерации уходит»).
ENGINES = ("animatediff", "chain")

#: Каждый N-й кадр условий становится узлом цепочки. При 12 fps шестой кадр —
#: два узла в секунду: плотнее держит траекторию, но и сегментов вдвое больше.
#: Относится ТОЛЬКО к `--engine chain`: у AnimateDiff кадры не узлы, а окно.
DEFAULT_EVERY = 6

#: Имена кадров при сборке mp4. Четыре знака, а не два, ровно по той же
#: причине, что и в `pollinations.FRAME_PATTERN`: кадры собираются обратно
#: лексикографической сортировкой, и при `%02d` сотый кадр называется `100.png`
#: и встаёт между `10.png` и `11.png`. Окно AnimateDiff сегодня 16 кадров, но
#: скользящее окно снимет этот предел, и тогда дефект тихо оживёт.
FRAME_PATTERN = "f_%04d.png"


def _say(step: str, ok, detail: str = "") -> None:
    """Три исхода, а не два. `ok is None` — «не смогли измерить».

    Печать булевым флагом склеивала «проверено и хорошо» с «проверить не
    вышло»: непроверенное показывалось галочкой. Это тот же дефект, что уже
    ловили в позе (`delta is None or ...` пропускало неизмеренную позу) и в
    жидкости (нулевое расстояние от невозможности измерить подавалось как
    идеальное совпадение). Одна и та же ошибка в трёх местах — значит она в
    способе печатать, а не в местах.
    """
    print(f"{'ПРОПУСК' if ok is None else ('PASS' if ok else 'FAIL'):<7} "
          f"{step:<22} {detail}")


def preflight_verdict(probes: list) -> tuple:
    """Прогнать проверки по порядку. -> (на чём остановились, непроверенное).

    ТРИ ИСХОДА, а не два. Проверка возвращает None, когда не смогла ответить:
    нет `nvidia-smi` (на Apple его и не должно быть), минорная совместимость
    CUDA, не спросить размер весов. Здесь стояло `if not ok`, и None
    останавливал прогон наравне с отказом — «не знаю» подавалось как «плохо».
    Обратная ошибка не лучше: покрасить None в зелёное значит спрятать
    непроверенное под галочкой.

    Поэтому непроверенное копится и возвращается отдельным списком, а
    останавливает прогон только настоящий отказ. Живёт функцией, а не телом
    `main()`, ровно по этой причине: внутри `main()` до этой развилки не
    добирался ни один тест, и она молча стояла в двух исходах.

    Порядок в `probes` — это цена: дешёвое раньше дорогого, чтобы отсутствующий
    пакет (1 мс) не находился после импорта torch (2.4 с) и загрузки лица (7 с).
    """
    unknown: list = []
    for label, fn in probes:
        try:
            ok, _, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        _say(label, ok, detail[:96])
        if ok is None:
            unknown.append(label)
        elif not ok:
            return label, unknown
    return None, unknown


def run_verdict(rows: list) -> tuple:
    """Строки гейта -> (код возврата, пояснение). Чистая функция.

    ЗАЧЕМ ОНА ПОЯВИЛАСЬ. Прогон возвращал 0 при любом исходе: строки гейта
    считались, печатались и на код возврата не влияли. Шапка модуля при этом
    обещает «останавливает всё при провале». То есть автоматическая проверка
    «сошлось ли» была невозможна — а завтра по этому коду будет решаться,
    показывать клип или нет.

    Склеено это с другим дефектом, найденным тем же разбором: клип из 16
    одинаковых кадров не проваливает НИ ОДНОЙ строки — все они дают «не смогли
    измерить», потому что `worst_jump is None` означает и «мало кадров», и
    «ничего не движется». Ноль на выходе означал бы «успех».

    Отсюда правило из трёх исходов, а не из двух:

    * хоть одна ИЗМЕРЕННАЯ строка провалилась -> 1;
    * не измерено НИЧЕГО -> 1, потому что успех надо доказать, а не
      унаследовать из тишины. «Нечего было судить» — это не «сошлось»;
    * измерено хоть что-то и провалов нет -> 0.

    Пропуски при этом не считаются провалами: клип, где половину осей измерить
    не вышло, а измеренные сошлись, — это годный клип с оговоркой, и оговорка
    остаётся в отчёте.
    """
    failed = [r["label"] for r in rows if r.get("ok") is False]
    measured = [r["label"] for r in rows if r.get("ok") is not None]
    skipped = [r["label"] for r in rows if r.get("ok") is None]
    if failed:
        return 1, (f"ПРОВАЛ по осям: {', '.join(failed)}"
                   + (f"; не измерено: {', '.join(skipped)}" if skipped else ""))
    if not measured:
        return 1, ("НИ ОДНА ось не измерена — судить не по чему. Это не успех: "
                   "клип из одинаковых кадров даёт ровно такую картину, и "
                   "нулевой код возврата объявил бы его годным")
    return 0, (f"измерено {len(measured)}, провалов нет"
               + (f"; не измерено: {', '.join(skipped)}" if skipped else ""))


def _row(r: dict, *, width: int = 110) -> None:
    """Строка вердикта. `width=0` — не резать, а переносить.

    Резать по 110 символов удобно для таблицы гейта, где число стоит в начале.
    Но у строк про кадрировку и прогноз идентичности главное — В КОНЦЕ: «гейт
    вернёт НЕ ПРОВЕРЕНО», «ЗАРАНЕЕ НЕИЗВЕСТНО», «верю манифесту». Проверено
    сухим прогоном: обрезка съедала ровно эти слова, и строка превращалась в
    невнятное начало фразы — то есть завтра на демо пропуск читался бы как
    успех именно там, где мы старались этого избежать.
    """
    note = r.get("note") or ""
    if width:
        _say(r["label"], r["ok"], note[:width])
        return
    import textwrap

    lines = textwrap.wrap(note, 96) or [""]
    _say(r["label"], r["ok"], lines[0])
    for extra in lines[1:]:
        print(f"{'':<7} {'':<22} {extra}")


def _value(v):
    """Число из поля `worst`.

    Клиповые меры кладут туда КОРТЕЖ (имя, значение) — `limb_consistency`,
    `garment_drift`, `garment_fit`, — а `pose_delta` кладёт голое число и имя
    отдельно. Одно имя, две формы; перепутать их легко, и в таблице замера это
    выглядело бы как «('l_thigh', 0.031)» в колонке, где ждут число.
    """
    return v[1] if isinstance(v, (tuple, list)) and len(v) == 2 else v


def smoke_verdict(ident, delta, *, driving=None) -> dict:
    """Сошёлся ли дым: числа -> вердикт. Чистая функция, без карты и без сети.

    ПОРОГИ БЕРУТСЯ ИЗ МОДУЛЕЙ, А НЕ ЛИТЕРАЛАМИ, и это не косметика. Здесь
    стояли числа 0.35 и 0.25 прямо в теле `main`. Первое случайно совпало с
    `identity_arcface.SAME_PERSON_MAX`; второе — НЕТ: `pose.SAME_POSE_MAX`
    равен 0.15, потому что старт-кадр судится строго (ещё ничто не двигалось, и
    всё сверх этого — уже переосмысление позы генератором, а не движение
    субъекта). То есть ЦЕЛЕВОЙ путь на своём железе пропускал расхождение,
    которое шлюзовой путь забраковал бы: бар был мягче собственного на две
    трети, и заметить это можно было только сличив два файла глазами.

    Второе исправление: худший сустав ПЕЧАТАЛСЯ, но не проверялся, хотя
    сообщение обещало пользователю бар 0.40. Между тем именно он и есть
    работающая мера — среднее размывает уехавшую в сторону руку (замер:
    согнутая рука даёт среднее 0.11 при запястье 0.97).

    Вынесено из `main` отдельной функцией по третьей причине: логика вердикта,
    живущая внутри CLI, не проверяется тестом и не ломается мутацией, то есть
    формально защищена и фактически нет.
    """
    from .identity_arcface import SAME_PERSON_MAX
    from .pose import SAME_POSE_MAX, WORST_JOINT_MAX

    face_ok = ident is not None and ident <= SAME_PERSON_MAX
    face_note = (f"дрифт {ident} (бар {SAME_PERSON_MAX}; выше — поднять "
                 f"ip_adapter_scale до 0.85)")
    if ident is None:
        face_note = ("лицо НЕ НАЙДЕНО на кейфрейме или на референсе — это не "
                     "«похоже», а отсутствие измерения. Считать провалом.")

    if not delta:
        # «Не измерено» — это не «в норме». Здесь стояло `delta is None or ...`,
        # и непроверенная поза шла как пройденная.
        return {"face_ok": face_ok, "pose_ok": False, "face_note": face_note,
                "pose_note": (f"НЕ ИЗМЕРЕНА: на driving-кадре {driving} или на "
                              f"кейфрейме тело не найдено. Считать это "
                              f"провалом, а не пропуском.")}

    # `pose_delta` кладёт в `worst` ЧИСЛО, а имя сустава отдельно в
    # `worst_joint`. Клиповая функция в том же модуле кладёт в `worst` КОРТЕЖ
    # (имя, значение) — одно имя, две формы, и перепутать их легко.
    pose_ok = (delta["mean"] <= SAME_POSE_MAX
               and delta["worst"] <= WORST_JOINT_MAX)
    return {
        "face_ok": face_ok, "pose_ok": pose_ok, "face_note": face_note,
        "pose_note": (f"среднее {delta['mean']}, худший сустав "
                      f"{delta.get('worst_joint')} {delta['worst']} "
                      f"(бары {SAME_POSE_MAX}/{WORST_JOINT_MAX}; выше — "
                      f"controlnet_scale до 1.2)"),
    }


def clip_verdict(drift, seam, quality, limbs, garment) -> list:
    """Измерения клипа -> строки гейта. Чистая функция, тот же прибор для обоих
    движков.

    ЗАЧЕМ ОТДЕЛЬНО. Раньше это была таблица прямо в `main`, и в ней жила
    перевёрнутая строка: идентичность считалась пройденной по условию
    `drift["median"] is not None`, то есть судилось НАЛИЧИЕ ИЗМЕРЕНИЯ, а не его
    значение. Следствия ровно два, и оба плохие: клип с дрейфом 0.9 (чужой
    человек) печатался как PASS, а клип, на котором лицо мельче
    `identity_arcface.MIN_FACE_PX` и потому не судится вовсе, — как FAIL. Гейт
    при этом выглядел работающим.

    Здесь исход трёхзначный, и «не смогли измерить» берётся из собственного
    сигнала каждого измерителя, а не из нового порога:

    * `median is None` — ArcFace не нашёл судимого лица;
    * `ratio is None` / `worst_jump is None` — кадров меньше трёх либо клип не
      движется вообще, и `motion` сам это говорит;
    * пустой `wobble` — ни одна конечность не прослежена (`limb_consistency`
      печатает «NOT VERIFIABLE»);
    * пустые `regions` — торс не различим, `garment_drift` не проверил одежду.

    Все три состояния раньше приходили как `False`, то есть «плохо», и вели
    чинить генерацию вместо того, чтобы чинить съёмку или разрешение.
    """
    from .identity_arcface import SAME_PERSON_MAX

    median = (drift or {}).get("median")
    rows = [{
        "label": "идентичность",
        "ok": None if median is None else bool(median <= SAME_PERSON_MAX),
        "value": median,
        "note": ((drift or {}).get("note") or "")
                if median is not None else
                ("лицо ни на одном кадре не судимо (мельче порога или не "
                 "найдено) — это не «прошло» и не «не прошло»"),
    }]

    ratio = (seam or {}).get("ratio")
    rows.append({"label": "луп",
                 "ok": None if ratio is None else bool(seam.get("seamless")),
                 "value": ratio, "note": (seam or {}).get("note")})

    jump = (quality or {}).get("worst_jump")
    rows.append({"label": "движение",
                 "ok": None if jump is None else bool(quality.get("smooth")),
                 "value": jump, "note": (quality or {}).get("note")})

    wobble = (limbs or {}).get("wobble") or {}
    rows.append({"label": "анатомия",
                 "ok": bool(limbs.get("anatomical")) if wobble else None,
                 "value": _value((limbs or {}).get("worst")),
                 "note": (limbs or {}).get("note")})

    regions = (garment or {}).get("regions") or {}
    rows.append({"label": "одежда",
                 "ok": bool(garment.get("stable")) if regions else None,
                 "value": _value((garment or {}).get("worst")),
                 "note": (garment or {}).get("note")})
    return rows


def framing_from_manifest(manifest: dict, *, full_body: bool) -> tuple:
    """Кадрировка берётся из МАНИФЕСТА, а не из флага. -> (waist_up, строка).

    ШОВ, КОТОРЫЙ ЭТО ЗАКРЫВАЕТ. `--full-body` управлял только планом
    (`animate.plan(waist_up=...)`), а условия приходят готовыми из
    `--conditions` и уже отрендерены с какой-то кадрировкой — `render_sequence`
    по умолчанию рисует полный рост. Типовой сценарий «отрендерить умолчанием и
    запустить без флага» давал план, обещающий лицо ~146 px, при фактических
    ~59 px на живом ките: план врал ровно про то, ради чего существует.

    Это тот же дефект, что чинился в `skeleton.py` (флаг `from_mediapipe`
    подписывал манифест независимо от того, кто отработал), и вывод тот же:
    ИМЯ ОБЯЗАНО ВЫВОДИТЬСЯ ИЗ ТОГО, ЧТО ДЕЙСТВИТЕЛЬНО ИСПОЛНИЛОСЬ. Манифест —
    свидетельство, флаг — намерение. При расхождении верим свидетельству:
    условия уже нарисованы, и флаг их не перерисует.

    Манифеста нет или он старый — это ТРЕТИЙ исход, «не смогли измерить», а не
    «значит полный рост». Тогда флаг остаётся единственным, что у нас есть, и
    об этом говорится прямо.
    """
    framing = (manifest or {}).get("framing")
    intent = "full_body" if full_body else "waist_up"
    if framing not in ("full_body", "waist_up"):
        return not full_body, {
            "label": "кадрировка", "ok": None, "value": None,
            "note": (f"манифест не называет кадрировку (условия отрендерены "
                     f"старым skeleton). Беру намерение флага: {intent} — это "
                     f"НАМЕРЕНИЕ, а не свидетельство: реальная доля лица в этих "
                     f"условиях неизвестна, и план ниже может обещать лицо, "
                     f"которого там нет")}
    waist_up = framing == "waist_up"
    if framing != intent:
        return waist_up, {
            "label": "кадрировка", "ok": False, "value": None,
            "note": (f"флаг говорит {intent}, а условия отрендерены как "
                     f"{framing}. ВЕРЮ МАНИФЕСТУ: он свидетельство, флаг — "
                     f"намерение, и перерисовать уже готовые условия флаг не "
                     f"может. План строится по {framing}")}
    return waist_up, {"label": "кадрировка", "ok": True, "value": None,
                      "note": f"{framing}, подтверждено манифестом условий"}


def identity_forecast(manifest: dict, height: int, *, waist_up: bool,
                      plan_px: float | None = None) -> dict:
    """Увидит ли гейт лицо на ЭТОМ разрешении — сказать ДО генерации.

    Доля лица берётся ИЗМЕРЕННОЙ по скелетам самих условий
    (`face_share_by_framing` в манифесте), а не из типовой таблицы: широкая
    стойка растягивает габарит и роняет долю (на живом kit-кадре 0.074 вместо
    типовых 0.094), и знать надо фактическое число.

    Умножается она на высоту ПЛАНА, а не на высоту условий: манифест считал px
    для своего холста 512x768, а рисовать мы можем 384x576 или 640x960, и в
    этих трёх случаях гейт видит разное.

    Вердикт трёхзначен НАМЕРЕННО. Мелкое лицо — это не «плохо», это «нечем
    судить»: гейт вернёт «НЕ ПРОВЕРЕНО», и завтра на демо пропуск нельзя
    принять за успех. Поэтому здесь `ok is None`, а не False.
    """
    from .identity_arcface import MIN_FACE_PX, START_MIN_FACE_PX

    framing = "waist_up" if waist_up else "full_body"
    shares = (manifest or {}).get("face_share_by_framing") or {}
    share = shares.get(framing)
    if share is None and (manifest or {}).get("framing") == framing:
        share = (manifest or {}).get("face_share")
    if not share:
        # Про число из плана сказано ОТДЕЛЬНО и прямо: оно посчитано по типовой
        # доле (9.4% / 19% высоты), а не по этим условиям, и именно это
        # расхождение — 146 обещанных px против 59 фактических — стоило нам
        # плана, врущего про то, ради чего он существует.
        plan_note = (f" Число в плане (~{plan_px:.0f} px) взято из ТИПОВОЙ "
                     f"доли, а не измерено по этим условиям — верить ему "
                     f"нельзя." if plan_px else "")
        return {"label": "идентичность заранее", "ok": None, "value": None,
                "note": ("манифест не измерил долю лица (нет шеи или таза на "
                         "условиях, либо условия старые) — сможет ли гейт "
                         "судить лицо, ЗАРАНЕЕ НЕИЗВЕСТНО. Это отдельный исход, "
                         "а не «сможет»." + plan_note)}
    px = round(share * height, 1)
    # Манифест мерил лицо на СВОЁМ холсте; если план рисует другой высоты,
    # оба числа печатаются рядом — иначе читатель сравнит наше с чужим.
    canvas = ((manifest or {}).get("size") or [None, None])[1]
    at_canvas = ((f"; манифест мерил на холсте {canvas} px, там "
                  f"~{round(share * canvas, 1)} px")
                 if canvas and canvas != height else "")
    if px >= MIN_FACE_PX:
        return {"label": "идентичность заранее", "ok": True, "value": px,
                "note": (f"лицо ~{px} px при {height} px высоты (доля {share} "
                         f"измерена по условиям) — бар видео {MIN_FACE_PX}, "
                         f"старт-кадра {START_MIN_FACE_PX}: гейту будет что "
                         f"судить{at_canvas}")}
    other = "waist_up" if not waist_up else "full_body"
    rescue = shares.get(other)
    way_out = (f"перерендерить условия с framing={other}: там доля {rescue} "
               f"даёт ~{round(rescue * height, 1)} px"
               if rescue else "перерендерить условия по пояс или поднять высоту")
    return {"label": "идентичность заранее", "ok": None, "value": px,
            "note": (f"лицо ~{px} px при {height} px высоты — это ниже бара "
                     f"{MIN_FACE_PX}: гейт вернёт «НЕ ПРОВЕРЕНО», и это НЕ то же "
                     f"самое, что «плохо». На демо пропуск нельзя принять за "
                     f"успех. Выход: {way_out}{at_canvas}")}


def lora_verdict(active: list, *, motion_lora: str | None = None,
                 face_prefix: str = "faceid") -> tuple:
    """Совпадает ли ЗАЯВЛЕННОЕ с тем, что реально прицеплено к модели.

    «Мы подключили LoRA» — это утверждение, и на демо оно будет произнесено
    вслух. Проверяется оно тем же правилом, по которому гейт не спрашивает
    генератор, получилось ли у него: спрашиваем не аргументы вызова, а
    состояние пайплайна (`animate.active_loras`).

    Три исхода, и средний важнее крайних:

    * пайплайн НЕ ОТВЕТИЛ (пустой список — нет peft, старый diffusers) —
      `None`, ПРОПУСК. «Не знаем, что прицеплено» честнее, чем зелёная
      галочка, выданная по списку аргументов;
    * нет LoRA лица — `False`. Она несущая: у FaceID своя LoRA грузится вместе
      с адаптером, и без неё канал личности отключён наполовину. Внешне это
      выглядит как «LoRA реализма испортила лицо», хотя испортила его потеря
      FaceID-LoRA;
    * попросили motion LoRA, а её нет (или НЕ просили, а она есть) — `False`.
      Второй случай выглядит безобидно и ломает ровно то, ради чего написан
      замер «с LoRA / без неё»: базовый прогон, в котором LoRA осталась
      прицепленной с прошлого раза, даёт разницу ноль и читается как «LoRA не
      влияет».
    """
    if not active:
        return None, ("пайплайн не ответил, какие LoRA прицеплены (нет peft "
                      "или старый diffusers). Это ПРОПУСК: заявление «LoRA "
                      "подключена» здесь не подтверждено и не опровергнуто")
    names = list(active)
    shown = ", ".join(names)
    if not any(n.startswith(face_prefix) for n in names):
        return False, (f"среди [{shown}] нет LoRA лица ({face_prefix}*): канал "
                       f"личности подключён наполовину, кадры будут похожи на "
                       f"кого угодно. Грузить адаптер лица ДО выгрузки на CPU")
    wanted = f"motion_{motion_lora}" if motion_lora else None
    got_motion = [n for n in names if n.startswith("motion_")]
    if wanted and wanted not in names:
        return False, (f"просили {wanted}, а прицеплено [{shown}] — движение "
                       f"камеры не подключено, и приписывать ему разницу "
                       f"нельзя")
    if not wanted and got_motion:
        return False, (f"motion LoRA {got_motion} прицеплена, хотя её не "
                       f"просили: замер «без LoRA» на самом деле С НЕЙ, "
                       f"сравнение недействительно")
    return True, f"прицеплено: {shown}"


def choose_conditions(paths: list, need: int, *, stride: int = 1,
                      start: int = 0, source_fps: float | None = None) -> tuple:
    """Какие условия попадут в окно модуля движения. (кадры, fps, пояснение).

    ДЕШЁВАЯ ПРОВЕРКА ВМЕСТО ДОРОГОГО ПАДЕНИЯ. `animate.animate` отказывается,
    если условий не ровно `cfg.frames`, — но отказывается ПОСЛЕ сборки
    пайплайна, то есть после загрузки UNet, модуля движения, ControlNet и
    адаптера лица. Здесь тот же отказ стоит миллисекунду и случается до первого
    байта весов. Ровно тот дефект, что уже ловили в `animate.build`, где имя
    motion LoRA проверялось после полутора гигабайт загрузки.

    ТЕМП ВОСПРОИЗВЕДЕНИЯ — АРИФМЕТИКА, А НЕ ВКУС. Условия сняты с driving-видео
    с частотой `source_fps` (`driving.SPEC_FPS` = 12). Если брать каждый
    `stride`-й кадр, то, чтобы движение шло в РЕАЛЬНОМ времени, mp4 обязан
    воспроизводиться на `source_fps / stride`. Шаг 1 — истинный темп и 16/12 =
    1.3 с движения; шаг 4 покрывает вчетверо больший кусок, но играется на 3 fps
    и выглядит рвано. Выбор за оператором, но обе цифры печатаются, чтобы он
    выбирал числами.

    `source_fps=None` — взять частоту у `driving.SPEC_FPS`, а не хранить её
    копию здесь: разошедшиеся копии одного числа на этом проекте уже стоили
    целевого пути, судившего мягче собственных порогов.
    """
    source_fps = _source_fps() if source_fps is None else source_fps
    if need <= 0:
        return [], 0.0, f"нужно положительное число кадров, а не {need}"
    if stride < 1 or start < 0:
        return [], 0.0, (f"шаг {stride} и начало {start} бессмысленны: шаг >= 1, "
                         f"начало >= 0")
    span = start + (need - 1) * stride + 1
    if len(paths) < span:
        return [], 0.0, (
            f"условий {len(paths)}, а окно требует {span} (кадры "
            f"{start}..{start + (need - 1) * stride} шагом {stride}). Либо "
            f"--stride {max(1, (len(paths) - 1 - start) // max(1, need - 1))} и "
            f"меньше, либо --from-frame ближе к началу, либо перерендерить "
            f"условия длиннее: skeleton.render_sequence, это CPU и бесплатно")
    chosen = [paths[start + i * stride] for i in range(need)]
    fps = source_fps / stride
    note = (f"взято {need} условий из {len(paths)}: кадры "
            f"{start}..{start + (need - 1) * stride} шагом {stride}; это "
            f"{need * stride / source_fps:.1f} c движения при съёмке "
            f"{source_fps:g} fps, значит mp4 играется на {fps:.1f} fps, чтобы "
            f"темп остался настоящим")
    return chosen, fps, note


def resolve_driving(raw: str, conditions_dir) -> str | None:
    """Путь к driving-кадру из манифеста, приведённый к ТЕКУЩЕМУ каталогу.

    ЗАЧЕМ. `skeleton.render_sequence` кладёт в манифест путь ровно в том виде,
    в каком его получил, — то есть относительно каталога, откуда рендерили
    условия. Тест-кит собран изнутри `kit/`, и там записано `driving/0000.jpg`;
    запущенный из корня репозитория прогон открывает `driving/0000.jpg`, не
    находит его и падает с `FileNotFoundError` ВНУТРИ измерителя позы. Поймано
    сухим прогоном ветки, а на карте это случилось бы после генерации: минуты
    счёта — в трейсбек, мимо отчёта.

    Поэтому путь ищется рядом с условиями (`kit/conditions` -> `kit/driving`),
    а не найденный возвращается как None — «нет driving-кадра» обязано стать
    ПРОПУСКОМ в вердикте позы, а не исключением на полпути.
    """
    if not raw:
        return None
    p = Path(raw)
    d = Path(conditions_dir)
    for cand in (p, d.parent / p, d / p, d.parent / p.name, d / p.name):
        if cand.exists():
            return str(cand)
    return None


def save_frames(images: list, out_dir) -> list:
    """PIL-кадры на диск под именами, которые ffmpeg соберёт шаблоном.

    Кадры сохраняются ВСЕГДА, даже когда mp4 потом не собрался: сборка ролика —
    это последний и самый хрупкий шаг (см. историю `chain._concat`, где ffmpeg
    падал уже после оплаченной генерации), и терять из-за него минуты счёта на
    карте нельзя. По кадрам гейт считается целиком, mp4 нужен только человеку.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for i, im in enumerate(images):
        path = out_dir / (FRAME_PATTERN % i)
        im.save(path)
        made.append(str(path))
    return made


def frames_to_mp4(frame_dir, out_mp4, *, fps: float,
                  pattern: str = FRAME_PATTERN) -> str:
    """Кадры -> mp4. Шаблоном, а не списком, и всё абсолютными путями.

    Абсолютные пути и никакого `cwd` — тот же урок, что и в `chain._concat`:
    там имена внутри списка писались относительно cwd, а сам список
    передавался ffmpeg вместе с `cwd=каталог склейки`, и он падал ВСЕГДА —
    но падал после того, как все платные сегменты уже сгенерированы.

    `yuv420p` обязателен: без него плеер показывает первый кадр и молчит.
    Он требует чётных сторон — все три плана (384x576, 512x768, 640x960) чётные,
    и это не совпадение, а свойство кратных 64 разрешений диффузии.
    """
    import subprocess

    frame_dir = Path(frame_dir).resolve()
    out_mp4 = Path(out_mp4).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", f"{fps:g}", "-start_number", "0",
         "-i", str(frame_dir / pattern), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(out_mp4)],
        check=True, capture_output=True)
    return str(out_mp4)


def ab_report(runs: list) -> str:
    """Два прогона с одним сидом рядом, колонка к колонке.

    Печать одной таблицей — не оформление. Два отчёта, разнесённые по времени и
    по каталогам, сравнивают глазами, а глаз сравнивает то, что помнит; ровно
    так на этом проекте бар по позе стоял мягче собственного порога, и увидеть
    это можно было только сличив два файла глазами.
    Здесь обе колонки печатает одна функция из одних и тех же полей.

    `None` печатается словом «не измерено», а не пустым местом и не нулём:
    пустая клетка читается как «ноль», то есть как лучший возможный результат,
    и это ровно та подмена, против которой написан `_say`.
    """
    def cell(r):
        v = r.get("value")
        mark = "ПРОПУСК" if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")
        text = "не измерено" if v is None else (
            f"{v:.3f}" if isinstance(v, float) else str(v))
        return f"{text} {mark}"

    labels = []
    for run in runs:
        for r in run["rows"]:
            if r["label"] not in labels:
                labels.append(r["label"])
    width = max([len(x) for x in labels] + [14]) + 2
    head = "метрика".ljust(width) + "".join(
        f"{run['label']:<26}" for run in runs)
    loras = "прицеплено".ljust(width) + "".join(
        f"{(', '.join(run.get('loras') or []) or 'не ответил'):<26}"
        for run in runs)
    lines = [head, loras, "-" * len(head)]
    for label in labels:
        line = label.ljust(width)
        for run in runs:
            got = next((r for r in run["rows"] if r["label"] == label), None)
            line += f"{(cell(got) if got else 'нет строки'):<26}"
        lines.append(line)
    return "\n".join(lines)


def _stop(why: str) -> int:
    print(f"\nОСТАНОВЛЕНО: {why}")
    return 1


def build_parser():
    """Разбор аргументов отдельной функцией, чтобы умолчания сторожил тест.

    `--engine animatediff` по умолчанию — это продуктовое решение («внешний API
    из пути генерации уходит»), и оно обязано быть проверяемым без карты, а не
    держаться на том, что кто-то не переставил строку.
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="ball_reel.run_local",
        description="полный прогон на локальной карте, с измерением на каждом шаге")
    ap.add_argument("--engine", choices=ENGINES, default=ENGINES[0],
                    help="animatediff — локально, одним проходом, без сети; "
                         "chain — кейфреймы + сшивка через шлюз (проверен "
                         "живьём, остаётся запасным)")
    ap.add_argument("--face", required=True)
    ap.add_argument("--conditions", default="conditions")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative",
                    default="blurry, deformed, extra limbs, watermark, text")
    ap.add_argument("--out", default="run_out")
    ap.add_argument("--seed", type=int, default=0,
                    help="один и тот же сид для обеих половин замера «с LoRA / "
                         "без неё» — иначе разница объясняется сидом")
    ap.add_argument("--vram", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true",
                    help="chain: остановиться после одного кейфрейма; "
                         "animatediff: остановиться после сборки весов и "
                         "печати прицепленных LoRA — отдельного дешёвого кадра "
                         "там нет, кадры считаются совместно")

    ad = ap.add_argument_group("animatediff")
    ad.add_argument("--full-body", action="store_true",
                    help="полный рост вместо кадра по пояс: доля лица падает с "
                         "19%% до 9.4%% высоты, и гейт перестаёт судить лицо")
    ad.add_argument("--stride", type=int, default=1,
                    help="каждый N-й кадр условий в окно движения. 1 — истинный "
                         "темп и ~1.3 с; больше — шире охват движения, но mp4 "
                         "играется на 12/N fps и выглядит рвано")
    ad.add_argument("--from-frame", type=int, default=0,
                    help="с какого условия начать окно")
    ad.add_argument("--fps", type=float, default=0.0,
                    help="частота mp4; 0 — посчитать из --stride так, чтобы "
                         "темп движения остался настоящим")
    # Умолчания НЕ ДУБЛИРУЮТСЯ: None означает «не передавать», и число берётся
    # из `animate.animate`. Скопированная сюда цифра разошлась бы с модулем
    # ровно так же, как разошёлся бар по позе, — и точно так же молча.
    ad.add_argument("--ip-adapter-scale", type=float, default=None,
                    help="сила канала личности; подсказки гейта («поднять до "
                         "0.85») крутятся именно здесь")
    ad.add_argument("--controlnet-scale", type=float, default=None,
                    help="сила канала позы; подсказка гейта — «до 1.2»")
    ad.add_argument("--steps", type=int, default=None)
    ad.add_argument("--base", default=None,
                    help="базовый чекпоинт SD1.5. Умолчание — оригинал 2022 "
                         "года; фотореалистичные дообучения ТОЙ ЖЕ "
                         "архитектуры (например emilianJR/epiCRealism) "
                         "подставляются одной строкой и заметно лучше по коже "
                         "и свету. Проверять замером, а не верить на слово")
    lora = ad.add_mutually_exclusive_group()
    lora.add_argument("--motion-lora", default="",
                      help="движение КАМЕРЫ отдельно от движения субъекта: "
                           "zoom-in, pan-left и родня (см. animate.MOTION_LORAS)")
    lora.add_argument("--ab-motion-lora", default="",
                      help="ЗАМЕР: два прогона с одним сидом — без motion LoRA "
                           "и с ней — и числа гейта по обоим рядом")

    ch = ap.add_argument_group("chain (запасной путь через шлюз)")
    ch.add_argument("--every", type=int, default=DEFAULT_EVERY,
                    help="каждый N-й кадр условий -> узел цепочки")
    ch.add_argument("--video-model", default="wan-fast",
                    help="wan-fast дешевле seedance-2.0 в 18 раз; "
                         "переключаться, когда связка уже сошлась")
    ch.add_argument("--seconds", type=int, default=2, help="секунд на сегмент")
    ch.add_argument("--no-loop", action="store_true")
    ch.add_argument("--lora", default="",
                    help="LoRA реализма: repo_id или путь к файлу. ПО "
                         "УМОЛЧАНИЮ ВЫКЛЮЧЕНА — сначала прогон без неё, иначе "
                         "не с чем сравнивать её влияние")
    ch.add_argument("--lora-weight", default="",
                    help="имя файла внутри repo, если их там несколько")
    ch.add_argument("--lora-scale", type=float, default=0.7,
                    help="сила LoRA; ближе к 1.0 она начинает перебивать лицо")
    ap.add_argument("--garment-ref", default="",
                    help="НЕ РЕАЛИЗОВАНО на GPU-ветке: единственный адаптер "
                         "занят лицом. Флаг оставлен, чтобы прогон отказал "
                         "внятно, а не сделал вид, что учёл одежду")
    return ap


def wardrobe_rows(frames: list, points: list, driving_paths: list,
                  clock=None) -> list:
    """Одежда, прилегание и мимика — ОДИН И ТОТ ЖЕ блок для обоих движков.

    Три меры, и они меряют разное, поэтому нужны все три:

    * `garment_drift` ловит СМЕНУ одежды между кадрами по цвету: другая футболка;
    * `garment_fit` ловит, едет ли ТА ЖЕ ткань вместе с телом или скользит по
      нему, — по материальным точкам поверхности, а не по цвету. Клип с
      одинаковой одеждой во всех кадрах проходит первую и может провалить вторую;
    * `clip_expression` сверяет мимику с driving-кадрами.

    Ни одна не останавливает прогон: `garment_fit` и `clip_expression` ни разу
    не работали на сгенерированном клипе, только на живой съёмке. Ставить на
    непроверенном измерителе бар, который отбрасывает оплаченный результат, —
    это ровно то самое «выдать выбранное за измеренное».
    """
    from contextlib import nullcontext

    from .expression import clip_expression
    from .garment import garment_drift
    from .garment_fit import garment_fit as garment_surface_fit
    from .timing import PER_RUN

    def stage(name):
        return clock.stage(name, per=PER_RUN) if clock else nullcontext()

    with stage("garment"):
        g = garment_drift(frames, points)
    rows = [{"label": "одежда", "ok": bool(g["stable"]) if g.get("regions")
             else None, "value": _value(g.get("worst")), "note": g["note"]}]
    with stage("garment_fit"):
        fit = garment_surface_fit(frames, points)
    # `fits` уже трёхзначен: None, когда измеренных пар меньше `MIN_PAIRS`.
    rows.append({"label": "прилегание", "ok": fit.get("fits"),
                 "value": _value(fit.get("worst")), "note": fit["note"]})
    if driving_paths and all(driving_paths):
        with stage("expression"):
            exp = clip_expression(driving_paths, frames)
        # `verdict` None означает «нечем судить» (лица мелки, driving застыл) —
        # ровно тот третий исход, ради которого написан `_say`.
        rows.append({"label": "мимика", "ok": exp.get("verdict"),
                     "value": exp.get("distance_p50"), "note": exp["note"]})
    else:
        rows.append({"label": "мимика", "ok": None, "value": None,
                     "note": ("не для всех условий известен driving-кадр — "
                              "сравнивать не с чем")})
    return rows


def clip_gate(frames: list, face: str, points: list, clock=None, *,
              garment: bool = True) -> list:
    """Клиповые меры + вердикт. Одна функция на оба движка — см. `clip_verdict`.

    `garment=False` — не «не проверять одежду», а «её уже проверили ЭТИ ЖЕ
    кадры выше». На шлюзовом пути `wardrobe_rows` судит кейфреймы, а сюда
    приходят кадры готового клипа — два разных набора, обе строки осмысленны.
    На локальном пути набор ОДИН: кадры и есть узлы. Повторный `garment_drift`
    там дал бы то же число второй раз и вторую строку с тем же именем, а в
    таблице замера одинаковых имён быть не должно — колонка берёт первую.
    """
    from contextlib import nullcontext

    from .garment import garment_drift
    from .identity import arcface_drift
    from .motion import loop_seam, motion_quality
    from .pose import limb_consistency
    from .timing import PER_RUN

    def stage(name):
        return clock.stage(name, per=PER_RUN) if clock else nullcontext()

    with stage("arcface_clip"):
        drift = arcface_drift(frames, face)
    with stage("motion"):
        seam, quality = loop_seam(frames), motion_quality(frames)
    with stage("limbs"):
        limbs = limb_consistency(frames)
    gclip = {"regions": {}, "worst": None, "stable": False, "note": ""}
    if garment:
        with stage("garment_clip"):
            gclip = garment_drift(frames, points)
    rows = clip_verdict(drift, seam, quality, limbs, gclip)
    return [r for r in rows if garment or r["label"] != "одежда"]


def _animatediff_once(args, *, cfg, conditions, driving_paths, prompt, out,
                      clock, motion_lora: str, label: str, measure) -> dict:
    """Одна генерация целиком: сборка -> кадры -> mp4 -> гейт. Возвращает числа.

    Отдельной функцией именно ради замера «с LoRA / без неё»: две половины
    обязаны отличаться РОВНО ОДНИМ — наличием motion LoRA. Всё остальное (сид,
    окно условий, план, промт, эмбеддинг лица) приходит сюда снаружи одним и
    тем же объектом, и подменить что-то по дороге негде.

    Пайплайн освобождается в конце: две сборки живьём одновременно — это два
    пика памяти подряд на карте, где и один впритык.
    """
    from PIL import Image

    from . import animate
    from .device import empty_cache
    from .gpu_keyframes import face_embeds
    from .timing import PER_RUN

    print(f"\n--- генерация [{label}] ---")
    # Эмбеддинг ДО сборки весов: insightface весит десятки мегабайт против
    # нескольких гигабайт диффузии, и «на фото не нашлось лица» обязано стоить
    # секунду, а не полную загрузку. У FaceID своего энкодера картинок нет —
    # сюда идёт 512-мерный вектор ArcFace, тот же самый, которым лицо потом и
    # ПРОВЕРЯЕТСЯ: считай его другая модель, расхождение двух моделей читалось
    # бы как дрейф личности.
    with clock.stage("face_embeds", per=PER_RUN):
        embeds = face_embeds(args.face, dtype=cfg.dtype)

    # БАЗУ МОЖНО ПОДМЕНИТЬ, И ЭТО САМЫЙ ДЕШЁВЫЙ РЫЧАГ КАЧЕСТВА.
    #
    # Дообучения SD1.5 имеют ТУ ЖЕ архитектуру — размер весов совпадает байт в
    # байт (3438 МБ), поэтому ControlNet, FaceID со своей LoRA и модуль
    # движения продолжают работать без единой правки. Слабое место базовой
    # модели 2022 года — ровно кожа и свет, то есть то, что здесь и нужно.
    #
    # Оговорка, которую нельзя терять: модуль движения обучался на БАЗОВОЙ
    # SD1.5. С фотореалистичными дообучениями он работает, с сильно
    # стилизованными может конфликтовать. Поэтому подмена — параметр, а не
    # новое умолчание: менять умолчание без замера значит ровно то, против чего
    # написан весь измерительный слой.
    base = args.base or animate.BASE_MODEL
    if args.base:
        _say("база", None, f"подменена на {args.base} (умолчание "
                           f"{animate.BASE_MODEL}); сравнивать с ним замером")
    with clock.stage("build", per=PER_RUN):
        pipe = animate.build(cfg, motion_lora=motion_lora or None,
                             base=base, verbose=False)
    active = animate.active_loras(pipe)
    ok, note = lora_verdict(active, motion_lora=motion_lora or None)
    _say("LoRA", ok, note)
    if ok is False:
        return {"label": label, "loras": active, "rows": [],
                "stop": f"состояние модели не подтверждает заявленное: {note}"}

    if args.smoke:
        print("\n--smoke: веса собраны, LoRA перечислены выше. Генерация не "
              "запускалась — у AnimateDiff нет дешёвого одиночного кадра, "
              "кадры считаются совместно одним проходом.")
        return {"label": label, "loras": active, "rows": [], "smoke": True}

    frames_dir = out / "frames"
    kw = {k: v for k, v in (("steps", args.steps),
                            ("controlnet_scale", args.controlnet_scale),
                            ("ip_adapter_scale", args.ip_adapter_scale))
          if v is not None}
    images = [Image.open(c).convert("RGB").resize((cfg.width, cfg.height))
              for c in conditions]
    with clock.stage("animate", per=PER_RUN):
        made = animate.animate(pipe, cfg, face_embeds=embeds, conditions=images,
                               prompt=prompt, negative=args.negative,
                               seed=args.seed, **kw)
    paths = save_frames(made, frames_dir)
    _say("кадры", len(paths) == cfg.frames,
         f"{len(paths)} шт. в {frames_dir}, сид {args.seed}")

    # Пайплайн больше не нужен: дальше только измерители, а они на CPU.
    pipe = None
    import gc

    gc.collect()
    empty_cache()

    fps = args.fps or _source_fps() / max(1, args.stride)
    clip = ""
    try:
        with clock.stage("mp4", per=PER_RUN):
            clip = frames_to_mp4(frames_dir, out / "clip.mp4", fps=fps)
        _say("mp4", True, f"{clip} на {fps:.1f} fps")
    except Exception as e:  # noqa: BLE001
        # Сборка mp4 не должна ронять прогон: кадры уже на диске, гейт считается
        # по ним, а ролик пересобирается одной строкой ffmpeg когда угодно.
        _say("mp4", False, f"{type(e).__name__}: {str(e)[:80]} — кадры целы")

    # --- гейт: те же меры, что судят клип из шлюза ---
    print(f"\n--- гейт [{label}] ---")
    from . import dwpose
    from .pose import landmarks

    source = dwpose.pose_points if dwpose.available() else landmarks
    points = [source(p) for p in paths]

    # Узловые меры. У AnimateDiff кадр соответствует условию ОДИН К ОДНОМУ,
    # поэтому позу КАЖДОГО кадра есть с чем сверять — в отличие от шлюзового
    # пути, где между узлами лежит выдумка видеомодели. Значит и бар применим
    # к каждому кадру: `smoke_verdict` судит старт-кадр строго именно потому,
    # что поза там полностью задана условием, а здесь она задана везде.
    import statistics

    verdicts, idents, poses = [], [], []
    for frame, cond, drv in zip(paths, conditions, driving_paths):
        ident, delta = measure(frame, cond, still=False)
        verdicts.append(smoke_verdict(ident, delta, driving=drv))
        # Неизмеренное НЕ подставляется единицей. Раньше сюда шло
        # `ident if ident is not None else 1.0`, и кадр, на котором лицо просто
        # не нашли, входил в медиану как максимальный дрейф: отчёт показывал
        # число вместо признания, что мерить было нечего.
        if ident is not None:
            idents.append(ident)
        if delta:
            poses.append(delta["mean"])
    if verdicts:
        _say("лицо (кадр 0)", verdicts[0]["face_ok"], verdicts[0]["face_note"])
        _say("поза (кадр 0)", verdicts[0]["pose_ok"], verdicts[0]["pose_note"])

    n = len(verdicts)
    face_pass = sum(1 for v in verdicts if v["face_ok"])
    pose_pass = sum(1 for v in verdicts if v["pose_ok"])
    rows = [
        {"label": "лицо по кадрам",
         "ok": (face_pass == n) if idents else None,
         "value": statistics.median(idents) if idents else None,
         "note": (f"{face_pass}/{n} кадров в баре; медиана "
                  f"{statistics.median(idents):.3f} по {len(idents)} измеренным"
                  if idents else
                  "ArcFace не нашёл судимого лица ни на одном кадре — это "
                  "отсутствие измерения, а не вердикт")},
        {"label": "поза по кадрам",
         "ok": (pose_pass == n) if poses else None,
         "value": statistics.median(poses) if poses else None,
         "note": (f"{pose_pass}/{n} кадров в баре; медиана "
                  f"{statistics.median(poses):.3f} по {len(poses)} измеренным"
                  if poses else
                  "тело не найдено ни на одном кадре — сверять позу не с чем")},
    ]

    # `garment=False`: одежду по ЭТИМ кадрам уже посудил `wardrobe_rows` —
    # у AnimateDiff узлы и кадры клипа это одно и то же множество.
    rows += wardrobe_rows(paths, points, driving_paths, clock)
    rows += clip_gate(paths, args.face, points, clock, garment=False)
    for r in rows:
        _row(r)
    if next((r for r in rows if r["label"] == "луп" and r["ok"] is False), None):
        # Клип AnimateDiff не замкнут по построению (в отличие от цепочки, где
        # последний сегмент возвращается в первый кейфрейм). Это не оправдание
        # провала, а адрес починки: подрезать по лучшему стыку локально,
        # бесплатно и без второй генерации.
        print("      луп не сошёлся: у окна нет возврата в первый кадр. "
              "Резать по лучшему стыку — motion.best_loop_cut / trim_to_loop, "
              "это ffmpeg и секунды, а не вторая генерация.")
    return {"label": label, "loras": active, "rows": rows, "clip": clip,
            "frames": paths, "fps": fps, "seed": args.seed,
            "measured": {"face": len(idents), "pose": len(poses), "of": n}}


def _source_fps() -> float:
    """Частота съёмки условий берётся из `driving`, а не из головы."""
    from .driving import SPEC_FPS

    return float(SPEC_FPS)


def main(argv: list) -> int:
    args = build_parser().parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 0 ------------------------------------------------- проверки аргументов
    # До предполёта и до единой секунды на карте. Модель видео раньше
    # проверялась только внутри generate_chain, то есть на пятом шаге — после
    # предполёта, дыма и всех кейфреймов. Опечатка в имени стоила бы всего
    # прогона ради ValueError, который виден отсюда.
    import shutil

    from .timing import (PER_FRAME, Timings, latency_verdict,
                         render as render_timings)

    clock = Timings()
    animatediff = args.engine == "animatediff"

    if args.garment_ref:
        # Раньше этот флаг только дописывал в промт «одежда как на референсе»,
        # а сам файл не открывался и в рендерер не передавался: модель получала
        # ссылку на изображение, которого ей не дали. Это не помогало, а мешало
        # — и при этом шаг «одежда» честно докладывал «одежда плывёт» и советовал
        # задать флаг, который уже задан.
        _say("--garment-ref", False, "не подключён к рендереру на этой ветке")
        return _stop("одежда в GPU-ветке задаётся только текстом промта: в "
                     "пайплайне один адаптер, и он занят лицом (FaceID). "
                     "Второй референс требует второго IP-Adapter'а — это "
                     "отдельная задача, и она не проверена на железе. Пока: "
                     "держать формулировку одежды побуквенно одинаковой и "
                     "фиксировать сид, а дрейф ловить мерой «одежда».")

    if animatediff:
        from . import animate

        # Имя motion LoRA — здесь, а не в `animate.build`: там оно проверяется
        # первой строкой (это уже починено), но до `build` успевает пройти весь
        # предполёт и эмбеддинг лица. Опечатка обязана стоить миллисекунду.
        for name in (args.motion_lora, args.ab_motion_lora):
            if name and name not in animate.MOTION_LORAS:
                _say("motion LoRA", False, f"{name} — такой нет")
                return _stop(f"выбрать из {', '.join(animate.MOTION_LORAS)}; "
                             f"это движение КАМЕРЫ, отдельное от движения "
                             f"субъекта")
        if not shutil.which("ffmpeg"):
            # Проверяется до генерации, потому что иначе выясняется после неё —
            # ровно тот сценарий, на котором `chain._concat` терял оплаченные
            # сегменты.
            _say("ffmpeg", False, "не найден в PATH")
            return _stop("без ffmpeg кадры соберутся, а ролика не будет — а "
                         "узнать об этом после генерации значит потерять её "
                         "время. apt install ffmpeg")
    else:
        from .chain import END_FRAME_POLLEN, segment_seconds_ok

        if args.motion_lora or args.ab_motion_lora:
            # Молча проигнорировать флаг — худшее из возможного: оператор
            # получил бы отчёт «с motion LoRA» по прогону, где её физически не
            # было, и приписал бы разницу ей. Движение камеры живёт в модуле
            # движения AnimateDiff; в шлюзовом пути такого модуля нет вовсе.
            _say("motion LoRA", False, "запрошена на движке chain")
            return _stop("motion LoRA — свойство локального движка (модуль "
                         "движения AnimateDiff). На шлюзовом пути её прицепить "
                         "не к чему: кадры рисует SD1.5 по одному, а движение "
                         "между ними придумывает чужая видеомодель. Замер «с "
                         "LoRA / без неё» делается на --engine animatediff.")
        if args.video_model not in END_FRAME_POLLEN:
            _say("модель видео", False,
                 f"{args.video_model} не умеет end_frame — цепочку нечем сшивать")
            return _stop(f"выбрать из {', '.join(END_FRAME_POLLEN)} "
                         f"(pollen/с: {END_FRAME_POLLEN}). Именно end_frame "
                         f"держит каждый сегмент за оба конца, без него узлы "
                         f"перестают что-либо закреплять.")
        seconds_ok, seconds_note = segment_seconds_ok(args.video_model,
                                                      args.seconds)
        if not seconds_ok:
            _say("длительность", False, f"--seconds {args.seconds}")
            return _stop(seconds_note)
        if seconds_note:
            _say("длительность", True, seconds_note)

    # 1 -------------------------------------------------------------- предполёт
    from .preflight_gpu import (check_conditions, check_disk, check_driver_build,
                                check_driving, check_face, check_face_model,
                                check_ffmpeg, check_gateway, check_packages,
                                check_pose_model, check_smi, check_torch,
                                check_vram, check_weights)

    if animatediff:
        # Шлюз НЕ проверяется намеренно: продуктовое требование — внешний API
        # уходит из пути генерации целиком, и прогон, который спотыкается о
        # недоступный шлюз, этому требованию противоречит.
        # torch/diffusers/insightface щупает сам `animate.preflight` — он же
        # различает «карты нет» и «torch собран без CUDA вовсе», а это разные
        # починки.
        rep = animate.preflight(args.vram)
        for c in rep["checks"]:
            _say(c["name"], c["ok"], c["detail"][:96])
        for n in rep["notes"]:
            print("\n" + n)
        if not rep["ok"]:
            return _stop("предполёт не пройден — чинить и запускать заново")
        probes = [("диск", lambda: check_disk("."))]
    else:
        # `peft` в этой ветке не проверяет НИЧТО: у animatediff его ловит
        # `animate.preflight`, а здесь FaceID-LoRA прицепляется той же самой
        # библиотекой — и без неё diffusers не падает, а молча грузит модель
        # без адаптера. Поэтому `пакеты` идут первой строкой: 1 мс на
        # `find_spec`, и это дешевле всего остального предполёта.
        probes = [("пакеты", check_packages),
                  ("диск", lambda: check_disk(".")), ("torch", check_torch),
                  ("драйвер", check_smi),
                  ("драйвер/сборка", check_driver_build)]
    probes += [("vram", check_vram), ("веса", check_weights),
               ("ffmpeg", check_ffmpeg),
               ("модель позы", check_pose_model),
               ("модель лица", check_face_model),
               ("условия", lambda: check_conditions(args.conditions)),
               ("driving-кадры", lambda: check_driving(args.conditions)),
               ("лицо", lambda: check_face(args.face))]
    if not animatediff:
        probes.append(("шлюз", check_gateway))
    stopped, unknown = preflight_verdict(probes)
    if unknown:
        print(f"\nНЕПРОВЕРЕНО (это не «в порядке»): {', '.join(unknown)}. "
              f"Прогон продолжается, но эти строки ничего не подтвердили.")
    if stopped:
        return _stop("предполёт не пройден — чинить и запускать заново")

    import glob

    from .identity import arcface_drift
    from .identity_arcface import MIN_FACE_PX, START_MIN_FACE_PX
    from .pose import landmarks, pose_delta

    conditions = sorted(glob.glob(str(Path(args.conditions) / "*.png")))

    # Чем сверять позу сгенерированного кадра. НЕ условием: условие — это
    # цветные палки на чёрном фоне, и детектор поз на нём не находит ничего
    # (проверено: landmarks() на условии возвращает None во всех кадрах).
    # Пока сравнение шло с условием, `delta` был None всегда, блок с числами
    # не печатался, а `pose_ok` вычислялся как `delta is None or ...`, то есть
    # был истиной при любой позе. Единственная проверка, ради которой взят
    # ControlNet, не выполнялась ни разу и при этом рапортовала «в норме».
    #
    # Сверять надо с исходным driving-кадром — он настоящая фотография, на нём
    # детектор работает, и он же то, что условие кодирует. Соответствие
    # «условие -> driving-кадр» пишет render_sequence в manifest.json рядом
    # с условиями.
    manifest_path = Path(args.conditions) / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    raw_driving: dict = manifest.get("driving_frames") or {}
    if not raw_driving:
        _say("манифест", False,
             f"нет {manifest_path} с картой условие->driving-кадр — позу "
             f"сверять не с чем")
        return _stop("без манифеста проверка позы невозможна, а без неё прогон "
                     "не отличит воспроизведённое движение от выдуманного. "
                     "Перерендерить условия текущим skeleton.render_sequence — "
                     "он пишет манифест сам.")
    # Пути в манифесте относительны каталога, откуда рендерили условия, а не
    # текущего; см. `resolve_driving`. Разрешаются они ЗДЕСЬ, один раз и до
    # генерации, чтобы отсутствующий кадр стал числом в отчёте, а не
    # исключением из измерителя посреди гейта.
    driving_of = {k: resolve_driving(v, args.conditions)
                  for k, v in raw_driving.items()}
    found = sum(1 for v in driving_of.values() if v)
    if not found:
        _say("driving-кадры", False,
             f"ни один из {len(raw_driving)} путей манифеста не открывается "
             f"из этого каталога (первый: {next(iter(raw_driving.values()))})")
        return _stop("позу сверять не с чем: пути в манифесте записаны "
                     "относительно каталога, откуда рендерили условия. "
                     "Запускать прогон оттуда же (для тест-кита — из kit/) "
                     "или держать driving-кадры рядом с условиями.")
    _say("driving-кадры", found == len(raw_driving) or None,
         f"{found}/{len(raw_driving)} найдено рядом с условиями")

    def measure(keyframe: str, condition: str, *, still: bool) -> tuple:
        """Измерить один кадр. `still` выбирает ПОРОГ СУДИМОСТИ ЛИЦА.

        ОДНА ФУНКЦИЯ, ДВА КОНТЕКСТА — и в этом был дефект. `measure`
        обслуживает и дым (одиночный резкий старт-кадр), и покадровый обход
        клипа, а порог стоял один на оба: `START_MIN_FACE_PX` = 70.

        Пороги разные не по прихоти. Резкий стилл судится от 70 px: измерено,
        живой старт-кадр на 91 px дал дистанцию 0.139 и породил полностью
        прошедший клип. Кадр ВИДЕО судится от 100, потому что к мелкому лицу
        добавляется смаз, и в полосе 70-99 дистанции раздуты.

        Чем это грозило на демо: в одном отчёте строка «лицо по кадрам»
        печатала PASS по бару 70, а строка клипового вердикта — «судить нечем»
        по бару 100. Два утверждения об одних и тех же пикселях, оба наши.
        Воспроизведено на kit/face.jpg (лица 77-94 px): бар 70 даёт median
        0.0086 при покрытии 1.0, бар 100 — median None.

        Это ЧЕТВЁРТЫЙ случай одной формы за два дня: второй способ узнать то,
        что уже кто-то знает. Поэтому порог теперь не «умолчание в теле
        функции», а решение вызывающего, и вызывающий обязан сказать, что
        именно он мерит.
        """
        bar = START_MIN_FACE_PX if still else MIN_FACE_PX
        # Каждый измеритель тактируется отдельно: они и есть те этапы, которые
        # в продукте с низкой латентностью пришлось бы держать в бюджете, —
        # генерация туда не влезет никогда, а проверка может.
        with clock.stage("arcface", per=PER_FRAME):
            ident = arcface_drift([keyframe], args.face,
                                  min_face_px=bar)["median"]
        driving = driving_of.get(Path(condition).stem)
        with clock.stage("landmarks", per=PER_FRAME):
            a = landmarks(driving) if driving else None
            b = landmarks(keyframe)
        with clock.stage("pose_delta", per=PER_FRAME):
            delta = pose_delta(a, b) if (a and b) else None
        return ident, delta

    # Промт — до всего дорогого. SD1.5 режет хвост по 77 токенам МОЛЧА, и
    # реальный промт этого пайплайна выходит примерно на 89. Кадрирование сюда
    # не входит намеренно — позу и композицию уже держит ControlNet.
    from .gpu_keyframes import fit_prompt

    prompt, dropped = fit_prompt([args.prompt])
    if dropped:
        _say("промт", False,
             f"не поместилось {len(dropped)} фрагмент(ов) в 77 токенов CLIP — "
             f"сократить, иначе часть описания просто не применится")
        return _stop("промт длиннее текстового энкодера; обрезанная одежда "
                     "потом читается как дрейф ткани в клипе")

    if animatediff:
        return _run_animatediff(args, out, clock, conditions, driving_of,
                                prompt, measure, render_timings,
                                latency_verdict, manifest)
    return _run_chain(args, out, clock, conditions, driving_of, prompt, measure,
                      render_timings, latency_verdict, manifest)


def _run_animatediff(args, out, clock, conditions, driving_of, prompt, measure,
                     render_timings, latency_verdict, manifest) -> int:
    """Целевой путь: один совместный проход, ноль сетевых вызовов."""
    from . import animate

    # План строится по КАДРИРОВКЕ УСЛОВИЙ, а не по флагу: условия уже
    # отрендерены, и флаг их не перерисует. См. `framing_from_manifest`.
    waist_up, framing_row = framing_from_manifest(manifest,
                                                  full_body=args.full_body)
    # Без обрезки: у этих двух строк главное стоит в конце фразы.
    _row(framing_row, width=0)
    cfg = animate.plan(args.vram, waist_up=waist_up)
    print("\n" + cfg.render())
    forecast = identity_forecast(manifest, cfg.height, waist_up=waist_up,
                                 plan_px=getattr(cfg, "face_px", None))
    _row(forecast, width=0)

    chosen, fps, note = choose_conditions(
        conditions, cfg.frames, stride=args.stride, start=args.from_frame)
    _say("окно условий", bool(chosen), note)
    if not chosen:
        return _stop("окно модуля движения не набирается из этих условий")
    if args.fps and abs(args.fps - fps) > 1e-6:
        # Заданный вручную fps — это решение оператора, а не ошибка, но темп
        # движения от него меняется, и это должно быть сказано числом: клип,
        # снятый на 12 fps и сыгранный на 24, показывает упражнение вдвое
        # быстрее, чем человек его делал.
        _say("темп", None,
             f"--fps {args.fps:g} против настоящего {fps:.1f}: движение пойдёт "
             f"в {args.fps / fps:.2f}x скорости оригинала")
    driving_paths = [driving_of.get(Path(c).stem) for c in chosen]
    if not all(driving_paths):
        # Не остановка: без driving-кадров теряется мимика и поза, но клип
        # осмыслен. Зато это обязано быть видно ДО генерации, а не после.
        _say("driving для окна", None,
             f"известны для {sum(1 for d in driving_paths if d)}/{len(chosen)} "
             f"условий окна — на остальных поза и мимика не проверяются")

    runs = []
    if args.ab_motion_lora:
        # Порядок «сначала без, потом с» не случаен: базу надо получить до
        # того, как что-то прицеплено, — иначе останется сомнение, что LoRA
        # не отцепилась до конца.
        plans = [("", "без motion LoRA"),
                 (args.ab_motion_lora, f"с motion LoRA {args.ab_motion_lora}")]
    else:
        plans = [(args.motion_lora,
                  f"motion LoRA {args.motion_lora}" if args.motion_lora
                  else "без motion LoRA")]

    for lora_name, label in plans:
        sub = out / (f"lora_{lora_name}" if lora_name else "lora_off")
        sub.mkdir(parents=True, exist_ok=True)
        got = _animatediff_once(args, cfg=cfg, conditions=chosen,
                                driving_paths=driving_paths, prompt=prompt,
                                out=sub, clock=clock, motion_lora=lora_name,
                                label=label, measure=measure)
        if got.get("stop"):
            return _stop(got["stop"])
        if got.get("smoke"):
            return 0
        runs.append(got)

    if len(runs) > 1:
        print(f"\n--- замер «с LoRA / без неё», сид {args.seed}, окно одно ---")
        print(ab_report(runs))
        print("\n  Обе половины прогнаны одним сидом на одном окне условий, и "
              "отличаются они ровно наличием motion LoRA — строка «прицеплено» "
              "выше это подтверждает состоянием модели, а не намерением.")

    profile = clock.summary()
    latency = latency_verdict(profile)
    print("\n--- тайминги ---")
    print(render_timings(profile))
    print(f"\n  {latency['note']}")

    report = {"engine": "animatediff", "timing": profile, "latency": latency,
              "seed": args.seed, "vram_gb": args.vram, "plan": cfg.render(),
              # Отдельными строками, а не внутри текста плана: завтра на демо
              # никто не должен принять «НЕ ПРОВЕРЕНО» за «проверено и хорошо».
              "framing": framing_row, "identity_forecast": forecast,
              "conditions": chosen, "fps": runs[0]["fps"] if runs else fps,
              "runs": [{k: v for k, v in r.items() if k != "frames"}
                       for r in runs]}
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    for r in runs:
        print(f"\n[{r['label']}] клип: {r.get('clip') or '(не собрался)'}")
    print(f"отчёт: {out / 'report.json'}")
    # Код возврата — по ГЕЙТУ, а не по факту «дошли до конца». Судится
    # ПОСЛЕДНИЙ прогон: при замере «с LoRA / без неё» показываем именно его.
    code, why = run_verdict(runs[-1].get("rows", []) if runs else [])
    _say("итог", code == 0, why)
    return code


def _run_chain(args, out, clock, conditions, driving_of, prompt, measure,
               render_timings, latency_verdict, manifest) -> int:
    """Запасной путь: кейфреймы поодиночке + сшивка видеомоделью через шлюз.

    Проверен живьём и потому оставлен без изменений по существу: единственное,
    что здесь поменялось, — гейт вынесен в общие функции, чтобы числа этого
    пути и локального были получены одним прибором.
    """
    import statistics

    from . import dwpose, pollinations
    from .chain import END_FRAME_POLLEN, Keyframe, generate_chain
    from .gpu_keyframes import load_pipeline, plan, render_keyframes
    from .pose import landmarks
    from .timing import PER_KEYFRAME, PER_RUN

    nodes = conditions[::args.every]
    cfg = plan(vram_gb=args.vram, keyframes=len(nodes))
    cfg.realism_lora = args.lora
    cfg.realism_lora_weight = args.lora_weight
    cfg.realism_lora_scale = args.lora_scale
    print(f"\nусловий {len(conditions)}, узлов {len(nodes)}, "
          f"кадр {cfg.width}x{cfg.height}, шагов {cfg.steps}, "
          f"оценка VRAM {cfg.estimated_vram_gb} ГБ")
    for note in cfg.notes:
        print(f"      {note}")
    # Печатается всегда, обеими сторонами. Строка «LoRA: нет» в отчёте —
    # это и есть база, против которой потом читается прогон с ней; без неё
    # два отчёта через неделю уже не различить.
    print(f"      LoRA реализма: "
          + (f"{cfg.realism_lora} @ {cfg.realism_lora_scale}"
             if cfg.realism_lora else "нет (база для сравнения)"))
    # Тот же прогноз, что и на локальном пути: доля лица измерена по САМИМ
    # условиям, а не взята из таблицы, и умножается на высоту ЭТОГО плана.
    # У шлюзового пути своей кадрировки нет — она целиком в условиях.
    forecast = identity_forecast(
        manifest, cfg.height,
        waist_up=(manifest or {}).get("framing") == "waist_up")
    _row(forecast, width=0)

    # 2 ------------------------------------------------------------------- дым
    print("\n--- дым: один кейфрейм ---")
    # Пайплайн собирается ОДИН раз на прогон и передаётся дальше: дым и полный
    # прогон — два вызова render_keyframes, и загрузка весов заново означала бы
    # вторую полную загрузку UNet + ControlNet + адаптера и второй пик памяти
    # сразу после первого. На 4 ГБ с offload это и минуты, и риск.
    with clock.stage("load_pipeline", per=PER_RUN):
        pipe = load_pipeline(cfg)
    # То же правило, что и на локальном пути: что прицеплено — читается у
    # модели, а не у собственных аргументов. Здесь адаптер лица называется
    # `faceid` (в AnimateDiff — `faceid_0`), поэтому сверка идёт по префиксу.
    from .animate import active_loras

    _say("LoRA", *lora_verdict(active_loras(pipe)))
    with clock.stage("keyframe", per=PER_KEYFRAME):
        smoke = render_keyframes(nodes[:1], args.face, prompt, out / "smoke",
                                 cfg=cfg, negative=args.negative, pipe=pipe,
                                 seed=args.seed)
    if not smoke.get("keyframes"):
        return _stop("кейфрейм не отрисовался — смотреть ошибку выше")
    ident, delta = measure(smoke["keyframes"][0], nodes[0], still=True)
    smoke_check = smoke_verdict(
        ident, delta, driving=driving_of.get(Path(nodes[0]).stem))
    face_ok, pose_ok = smoke_check["face_ok"], smoke_check["pose_ok"]
    _say("лицо", face_ok, smoke_check["face_note"])
    _say("поза", pose_ok, smoke_check["pose_note"])
    if args.smoke:
        if face_ok and pose_ok:
            print("\n--smoke: числа в норме. Запускать без --smoke.")
            return 0
        return _stop("дым не сошёлся — крутить ip_adapter_scale / "
                     "controlnet_scale по подсказкам выше и повторить дым; "
                     "это секунды, полный прогон — минуты")
    if not face_ok:
        return _stop("лицо не сошлось на первом же кадре: полный прогон "
                     "потратит время на тот же промах")

    # 3 ------------------------------------------------------- все кейфреймы
    print(f"\n--- кейфреймы: {len(nodes)} ---")
    with clock.stage("keyframes_all", per=PER_RUN):
        made = render_keyframes(nodes, args.face, prompt, out / "kf",
                                cfg=cfg, negative=args.negative, pipe=pipe,
                                seed=args.seed)
    keyframes = made.get("keyframes") or []
    if len(keyframes) < 2:
        return _stop(f"отрисовано {len(keyframes)} кейфрейм(ов) — цепочку не из "
                     f"чего собирать")
    poses, idents = [], []
    for kf, cond in zip(keyframes, nodes):
        i, d = measure(kf, cond, still=False)
        # Неизмеренное не подставляется единицей: кадр, на котором лицо не
        # нашли, входил в медиану как максимальный дрейф, и отчёт показывал
        # число там, где мерить было нечего.
        if i is not None:
            idents.append(i)
        if d:
            poses.append(d["mean"])
    _say("кейфреймы", bool(idents or poses) or None,
         f"{len(keyframes)}/{len(nodes)} отрисовано; лицо медиана "
         + (f"{statistics.median(idents):.3f} по {len(idents)}"
            if idents else "НЕ ИЗМЕРЕНО")
         + "; поза медиана "
         + (f"{statistics.median(poses):.3f} по {len(poses)}"
            if poses else "НЕ ИЗМЕРЕНА"))

    # 4 ---------------------------------------- одежда, прилегание, мимика
    source = dwpose.pose_points if dwpose.available() else landmarks
    kf_points = [source(k) for k in keyframes]
    driving_paths = [driving_of.get(Path(c).stem)
                     for c in nodes[:len(keyframes)]]
    for r in wardrobe_rows(keyframes, kf_points, driving_paths, clock):
        _row(r)

    # 5 --------------------------------------------------------------- сшивка
    # Цена берётся из той же таблицы, что и допустимость модели, — иначе они
    # расходятся, и «оценка» показывает не то, за что придёт счёт.
    rate = END_FRAME_POLLEN[args.video_model]
    segs = len(keyframes) if not args.no_loop else len(keyframes) - 1
    print(f"\n--- сшивка: {segs} сегмент(ов) x {args.seconds} c на "
          f"{args.video_model} = {rate * segs * args.seconds:.2f} pollen ---")
    kfs = []
    for i, p in enumerate(keyframes):
        k = Keyframe(index=i, t=float(i), driving_frame=nodes[i], rendered=p)
        k.url, k.accepted = pollinations.upload(p), True
        kfs.append(k)
    res = generate_chain(kfs, prompt, out / "chain",
                         model=args.video_model,
                         seconds_per_segment=args.seconds, loop=not args.no_loop)
    _say("клип", bool(res.clip_path), res.note[:96])
    if not res.clip_path:
        return _stop("клип не собрался")

    # 6 ----------------------------------------------------------------- гейт
    print("\n--- гейт ---")
    with clock.stage("extract_frames", per=PER_RUN):
        frames = pollinations.extract_frames(res.clip_path, out / "frames", fps=6)
    rows = clip_gate(frames, args.face, [source(f) for f in frames], clock)
    for r in rows:
        _row(r)

    profile = clock.summary()
    latency = latency_verdict(profile)
    print("\n--- тайминги ---")
    print(render_timings(profile))
    print(f"\n  {latency['note']}")

    report = {"engine": "chain", "timing": profile, "latency": latency,
              "identity_forecast": forecast,
              "lora": cfg.realism_lora or None,
              "lora_scale": cfg.realism_lora_scale if cfg.realism_lora else None,
              "seed": args.seed,
              "clip": res.clip_path, "keyframes": len(keyframes),
              "rows": rows, "chain": res.note}
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nклип: {res.clip_path}\nотчёт: {out / 'report.json'}")
    code, why = run_verdict(rows)
    _say("итог", code == 0, why)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
