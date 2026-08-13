"""Один прогон от начала до конца на локальной карте.

    python3 -m ball_reel.run_local --face face.jpg --conditions conditions \\
        --prompt "a woman exercising on a fitness ball in a bright studio, \\
                  black sports top and leggings, natural light, photographic"

    python3 -m ball_reel.run_local ... --smoke      # ОДИН кейфрейм и стоп

Собрано в одну команду не для красоты. Разложенный по десяти сниппетам прогон
на своей машине разваливается ровно там, где что-то пошло не так: половина
шагов сделана, половина нет, и непонятно, какой артефакт от какого запуска.
Здесь каждый этап печатает измерение и останавливает всё при провале — тот же
принцип, что и в предполёте, только теперь дорога не арендная минута, а твоё
время на разбор.

ЧТО ДЕЛАЕТ, по шагам:
  1. предполёт (карта, веса, условия, лицо) — не начинать на сломанной машине;
  2. ОДИН кейфрейм на дым, с проверкой лица и позы против driving-кадра;
  3. остальные кейфреймы, каждый — с проверкой, негодные перерисовываются;
  4. одежда: не поплыла ли она между узлами;
  5. сшивка сегментов через шлюз (start|end), склейка, луп;
  6. гейт по итоговому клипу и отчёт.

Шаг 2 отделён намеренно: один кадр стоит секунды, а разбираться, почему всё не
то, после полного прогона — десятки минут. `--smoke` останавливается на нём.

Платный тут ровно один шаг — пятый, и он считается заранее и печатается.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Каждый N-й кадр условий становится узлом цепочки. При 12 fps шестой кадр —
#: два узла в секунду: плотнее держит траекторию, но и сегментов вдвое больше.
DEFAULT_EVERY = 6


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


def _stop(why: str) -> int:
    print(f"\nОСТАНОВЛЕНО: {why}")
    return 1


def main(argv: list) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="ball_reel.run_local",
        description="полный прогон на локальной карте, с измерением на каждом шаге")
    ap.add_argument("--face", required=True)
    ap.add_argument("--conditions", default="conditions")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative",
                    default="blurry, deformed, extra limbs, watermark, text")
    ap.add_argument("--out", default="run_out")
    ap.add_argument("--every", type=int, default=DEFAULT_EVERY,
                    help="каждый N-й кадр условий -> узел цепочки")
    ap.add_argument("--video-model", default="wan-fast",
                    help="wan-fast дешевле seedance-2.0 в 18 раз; "
                         "переключаться, когда связка уже сошлась")
    ap.add_argument("--seconds", type=int, default=2, help="секунд на сегмент")
    ap.add_argument("--vram", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true",
                    help="остановиться после одного кейфрейма")
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--lora", default="",
                    help="LoRA реализма: repo_id или путь к файлу. ПО "
                         "УМОЛЧАНИЮ ВЫКЛЮЧЕНА — сначала прогон без неё, иначе "
                         "не с чем сравнивать её влияние")
    ap.add_argument("--lora-weight", default="",
                    help="имя файла внутри repo, если их там несколько")
    ap.add_argument("--lora-scale", type=float, default=0.7,
                    help="сила LoRA; ближе к 1.0 она начинает перебивать лицо")
    ap.add_argument("--garment-ref", default="",
                    help="НЕ РЕАЛИЗОВАНО на GPU-ветке: единственный адаптер "
                         "занят лицом. Флаг оставлен, чтобы прогон отказал "
                         "внятно, а не сделал вид, что учёл одежду")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 0 ------------------------------------------------- проверки аргументов
    # До предполёта и до единой секунды на карте. Модель видео раньше
    # проверялась только внутри generate_chain, то есть на пятом шаге — после
    # предполёта, дыма и всех кейфреймов. Опечатка в имени стоила бы всего
    # прогона ради ValueError, который виден отсюда.
    from .chain import END_FRAME_POLLEN, segment_seconds_ok
    from .timing import (PER_FRAME, PER_KEYFRAME, PER_RUN, Timings,
                         latency_verdict, render as render_timings)

    clock = Timings()

    if args.video_model not in END_FRAME_POLLEN:
        _say("модель видео", False,
             f"{args.video_model} не умеет end_frame — цепочку нечем сшивать")
        return _stop(f"выбрать из {', '.join(END_FRAME_POLLEN)} "
                     f"(pollen/с: {END_FRAME_POLLEN}). Именно end_frame "
                     f"держит каждый сегмент за оба конца, без него узлы "
                     f"перестают что-либо закреплять.")
    seconds_ok, seconds_note = segment_seconds_ok(args.video_model, args.seconds)
    if not seconds_ok:
        _say("длительность", False, f"--seconds {args.seconds}")
        return _stop(seconds_note)
    if seconds_note:
        _say("длительность", True, seconds_note)
    if args.garment_ref:
        # Раньше этот флаг только дописывал в промт «одежда как на референсе»,
        # а сам файл не открывался и в рендерер не передавался: модель получала
        # ссылку на изображение, которого ей не дали. Это не помогало, а мешало
        # — и при этом шаг 4 честно докладывал «одежда плывёт» и советовал
        # задать флаг, который уже задан.
        _say("--garment-ref", False,
             "не подключён к рендереру на этой ветке")
        return _stop("одежда в GPU-ветке задаётся только текстом промта: в "
                     "пайплайне один адаптер, и он занят лицом (FaceID). "
                     "Второй референс требует второго IP-Adapter'а — это "
                     "отдельная задача, и она не проверена на железе. Пока: "
                     "держать формулировку одежды побуквенно одинаковой и "
                     "фиксировать сид, а дрейф ловить шагом 4.")

    # 1 -------------------------------------------------------------- предполёт
    from .preflight_gpu import (check_conditions, check_disk, check_face,
                                check_gateway, check_pose_model, check_torch,
                                check_vram, check_weights)
    for label, fn in (("диск", lambda: check_disk(".")),
                      ("torch", check_torch), ("vram", check_vram),
                      ("веса", check_weights), ("модель позы", check_pose_model),
                      ("условия", lambda: check_conditions(args.conditions)),
                      ("лицо", lambda: check_face(args.face)),
                      ("шлюз", check_gateway)):
        try:
            ok, _, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        _say(label, ok, detail[:96])
        if not ok:
            return _stop("предполёт не пройден — чинить и запускать заново")

    import glob

    from .gpu_keyframes import load_pipeline, plan, render_keyframes
    from .identity import arcface_drift
    from .identity_arcface import START_MIN_FACE_PX
    from .pose import landmarks, pose_delta

    conditions = sorted(glob.glob(str(Path(args.conditions) / "*.png")))
    nodes = conditions[::args.every]

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
    driving_of: dict = {}
    if manifest_path.exists():
        driving_of = json.loads(manifest_path.read_text()).get("driving_frames") or {}
    if not driving_of:
        _say("манифест", False,
             f"нет {manifest_path} с картой условие->driving-кадр — позу "
             f"сверять не с чем")
        return _stop("без манифеста проверка позы невозможна, а без неё прогон "
                     "не отличит воспроизведённое движение от выдуманного. "
                     "Перерендерить условия текущим skeleton.render_sequence — "
                     "он пишет манифест сам.")

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

    # 2 ------------------------------------------------------------------- дым
    print("\n--- дым: один кейфрейм ---")
    from .gpu_keyframes import fit_prompt

    prompt = args.prompt
    # Порядок важности, а не вкуса: SD1.5 режет хвост по 77 токенам молча,
    # и реальный промт этого пайплайна выходит примерно на 89. Кадрирование
    # сюда не входит намеренно — позу и композицию уже держит ControlNet.
    prompt, dropped = fit_prompt([prompt])
    if dropped:
        _say("промт", False,
             f"не поместилось {len(dropped)} фрагмент(ов) в 77 токенов CLIP — "
             f"сократить, иначе часть описания просто не применится")
        return _stop("промт длиннее текстового энкодера; обрезанная одежда "
                     "потом читается как дрейф ткани в клипе")
    # Пайплайн собирается ОДИН раз на прогон и передаётся дальше: дым и полный
    # прогон — два вызова render_keyframes, и загрузка весов заново означала бы
    # вторую полную загрузку UNet + ControlNet + адаптера и второй пик памяти
    # сразу после первого. На 4 ГБ с offload это и минуты, и риск.
    with clock.stage("load_pipeline", per=PER_RUN):
        pipe = load_pipeline(cfg)
    with clock.stage("keyframe", per=PER_KEYFRAME):
        smoke = render_keyframes(nodes[:1], args.face, prompt,
                                 out / "smoke", cfg=cfg,
                                 negative=args.negative, pipe=pipe)
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
                                cfg=cfg, negative=args.negative, pipe=pipe)
    keyframes = made.get("keyframes") or []
    if len(keyframes) < 2:
        return _stop(f"отрисовано {len(keyframes)} кейфрейм(ов) — цепочку не из "
                     f"чего собирать")
    poses, idents = [], []
    for kf, cond in zip(keyframes, nodes):
        i, d = measure(kf, cond)
        idents.append(i if i is not None else 1.0)
        if d:
            poses.append(d["mean"])
    import statistics

    _say("кейфреймы", True,
         f"{len(keyframes)}/{len(nodes)}; лицо медиана "
         f"{statistics.median(idents):.3f}; поза медиана "
         f"{statistics.median(poses):.3f}" if poses else f"{len(keyframes)}")

    # 4 ------------------------------------------------------------- одежда
    from . import dwpose
    from .garment import garment_drift
    source = dwpose.pose_points if dwpose.available() else landmarks
    kf_points = [source(k) for k in keyframes]
    g = garment_drift(keyframes, kf_points)
    _say("одежда", g["stable"], g["note"][:96])
    if not g["stable"]:
        print("      одежда плывёт между узлами. Задать --garment-ref одной "
              "картинкой одежды или зафиксировать сид; см. garment.py.")

    # 4б -------------------------------------------- прилегание и мимика
    # ДВЕ МЕТРИКИ, КОТОРЫЕ БЫЛИ НАПИСАНЫ И НИКУДА НЕ ПОДАВАЛИСЬ. Счётчик без
    # ручки — задел; счётчик, к которому ручку так и не приделали, — долг.
    #
    # Они меряют РАЗНОЕ, и обе нужны. `garment_drift` выше ловит смену одежды
    # между узлами по цвету: другая футболка. `garment_fit` ловит, едет ли ТА
    # ЖЕ ткань вместе с телом или скользит по нему — по материальным точкам
    # поверхности, а не по цвету. Клип с одинаковой одеждой во всех кадрах
    # проходит первую проверку и может провалить вторую.
    #
    # Ни одна из них не останавливает прогон: обе ни разу не работали на
    # сгенерированном клипе, только на живой съёмке. Ставить на непроверенном
    # измерителе бар, который отбрасывает оплаченный результат, — это ровно то
    # самое «выдать выбранное за измеренное».
    from .expression import clip_expression
    from .garment_fit import garment_fit as garment_surface_fit

    with clock.stage("garment_fit", per=PER_RUN):
        fit = garment_surface_fit(keyframes, kf_points)
    _say("прилегание", fit["fits"], fit["note"][:110])

    driving_paths = [driving_of.get(Path(c).stem) for c in nodes[:len(keyframes)]]
    if all(driving_paths):
        with clock.stage("expression", per=PER_RUN):
            exp = clip_expression(driving_paths, keyframes)
        _say("мимика", exp.get("verdict"), exp["note"][:110])
    else:
        _say("мимика", None,
             "не для всех условий известен driving-кадр — сравнивать не с чем")

    # 5 --------------------------------------------------------------- сшивка
    from . import pollinations
    from .chain import Keyframe, generate_chain
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
    from .motion import loop_seam, motion_quality
    from .pose import limb_consistency

    with clock.stage("extract_frames", per=PER_RUN):
        frames = pollinations.extract_frames(res.clip_path, out / "frames", fps=6)
    drift = arcface_drift(frames, args.face)
    seam, quality = loop_seam(frames), motion_quality(frames)
    limbs = limb_consistency(frames)
    gclip = garment_drift(frames, [source(f) for f in frames])
    for label, data, ok in (("идентичность", drift["note"], drift["median"] is not None),
                            ("луп", seam["note"], bool(seam["seamless"])),
                            ("движение", quality["note"], bool(quality["smooth"])),
                            ("анатомия", limbs["note"], bool(limbs["anatomical"])),
                            ("одежда", gclip["note"], bool(gclip["stable"]))):
        _say(label, ok, data[:96])

    profile = clock.summary()
    latency = latency_verdict(profile)
    print("\n--- тайминги ---")
    print(render_timings(profile))
    print(f"\n  {latency['note']}")

    report = {"timing": profile, "latency": latency,
              "lora": cfg.realism_lora or None,
              "lora_scale": cfg.realism_lora_scale if cfg.realism_lora else None,
              "seed": 0,
              "clip": res.clip_path, "keyframes": len(keyframes),
              "identity": drift.get("note"), "loop": seam.get("ratio"),
              "motion": quality.get("worst_jump"),
              "garment": gclip.get("regions"), "chain": res.note}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nклип: {res.clip_path}\nотчёт: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
