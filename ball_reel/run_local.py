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
  3. план по VRAM, окно условий, эмбеддинг лица (падает без весов диффузии);
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


def _row(r: dict) -> None:
    _say(r["label"], r["ok"], (r.get("note") or "")[:110])


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
    `START_MIN_FACE_PX` и потому не судится вовсе, — как FAIL. Гейт при этом
    выглядел работающим.

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
                      start: int = 0, source_fps: float = 12.0) -> tuple:
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
    1.3 с движения; шаг 4 покрывает впятеро больший кусок, но играется на 3 fps
    и выглядит рвано. Выбор за оператором, но обе цифры печатаются, чтобы он
    выбирал числами.
    """
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
    так на этом проекте бар по позе полгода стоял мягче собственного порога.
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

    with clock.stage("build", per=PER_RUN):
        pipe = animate.build(cfg, motion_lora=motion_lora or None, verbose=False)
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
        ident, delta = measure(frame, cond)
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
    from .preflight_gpu import (check_conditions, check_disk, check_face,
                                check_gateway, check_pose_model, check_torch,
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
        probes = [("диск", lambda: check_disk(".")), ("torch", check_torch)]
    probes += [("vram", check_vram), ("веса", check_weights),
               ("модель позы", check_pose_model),
               ("условия", lambda: check_conditions(args.conditions)),
               ("лицо", lambda: check_face(args.face))]
    if not animatediff:
        probes.append(("шлюз", check_gateway))
    for label, fn in probes:
        try:
            ok, _, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        _say(label, ok, detail[:96])
        if not ok:
            return _stop("предполёт не пройден — чинить и запускать заново")

    import glob

    from .identity import arcface_drift
    from .identity_arcface import START_MIN_FACE_PX
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
    raw_driving: dict = {}
    if manifest_path.exists():
        raw_driving = json.loads(
            manifest_path.read_text()).get("driving_frames") or {}
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

    def measure(keyframe: str, condition: str) -> tuple:
        # Каждый измеритель тактируется отдельно: они и есть те этапы, которые
        # в продукте с низкой латентностью пришлось бы держать в бюджете, —
        # генерация туда не влезет никогда, а проверка может.
        with clock.stage("arcface", per=PER_FRAME):
            ident = arcface_drift([keyframe], args.face,
                                  min_face_px=START_MIN_FACE_PX)["median"]
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
                                prompt, measure, render_timings, latency_verdict)
    return _run_chain(args, out, clock, conditions, driving_of, prompt, measure,
                      render_timings, latency_verdict)


def _run_animatediff(args, out, clock, conditions, driving_of, prompt, measure,
                     render_timings, latency_verdict) -> int:
    """Целевой путь: один совместный проход, ноль сетевых вызовов."""
    from . import animate

    cfg = animate.plan(args.vram, waist_up=not args.full_body)
    print("\n" + cfg.render())

    chosen, fps, note = choose_conditions(
        conditions, cfg.frames, stride=args.stride, start=args.from_frame,
        source_fps=_source_fps())
    _say("окно условий", bool(chosen), note)
    if not chosen:
        return _stop("окно модуля движения не набирается из этих условий")
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
              "conditions": chosen, "fps": runs[0]["fps"] if runs else fps,
              "runs": [{k: v for k, v in r.items() if k != "frames"}
                       for r in runs]}
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    for r in runs:
        print(f"\n[{r['label']}] клип: {r.get('clip') or '(не собрался)'}")
    print(f"отчёт: {out / 'report.json'}")
    return 0


def _run_chain(args, out, clock, conditions, driving_of, prompt, measure,
               render_timings, latency_verdict) -> int:
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
    ident, delta = measure(smoke["keyframes"][0], nodes[0])
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
        i, d = measure(kf, cond)
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
              "lora": cfg.realism_lora or None,
              "lora_scale": cfg.realism_lora_scale if cfg.realism_lora else None,
              "seed": args.seed,
              "clip": res.clip_path, "keyframes": len(keyframes),
              "rows": rows, "chain": res.note}
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nклип: {res.clip_path}\nотчёт: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
