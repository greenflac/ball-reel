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


def _say(step: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {step:<22} {detail}")


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
    print(f"\nусловий {len(conditions)}, узлов {len(nodes)}, "
          f"кадр {cfg.width}x{cfg.height}, шагов {cfg.steps}, "
          f"оценка VRAM {cfg.estimated_vram_gb} ГБ")
    for note in cfg.notes:
        print(f"      {note}")

    def measure(keyframe: str, condition: str) -> tuple:
        ident = arcface_drift([keyframe], args.face,
                              min_face_px=START_MIN_FACE_PX)["median"]
        driving = driving_of.get(Path(condition).stem)
        a = landmarks(driving) if driving else None
        b = landmarks(keyframe)
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
    pipe = load_pipeline(cfg)
    smoke = render_keyframes(nodes[:1], args.face, prompt,
                             out / "smoke", cfg=cfg, negative=args.negative,
                             pipe=pipe)
    if not smoke.get("keyframes"):
        return _stop("кейфрейм не отрисовался — смотреть ошибку выше")
    ident, delta = measure(smoke["keyframes"][0], nodes[0])
    _say("лицо", ident is not None and ident <= 0.35,
         f"дрифт {ident} (бар 0.35; выше — поднять ip_adapter_scale до 0.85)")
    face_ok = ident is not None and ident <= 0.35
    if delta:
        pose_ok = delta["mean"] <= 0.25
        _say("поза", pose_ok,
             f"среднее {delta['mean']}, худший сустав {delta['worst']} "
             f"(бары 0.25/0.40; выше — controlnet_scale до 1.2)")
    else:
        # «Не измерено» — это не «в норме». Раньше здесь стояло
        # `delta is None or ...`, и непроверенная поза шла как пройденная.
        pose_ok = False
        _say("поза", False,
             f"НЕ ИЗМЕРЕНА: на driving-кадре "
             f"{driving_of.get(Path(nodes[0]).stem)} или на кейфрейме тело не "
             f"найдено. Считать это провалом, а не пропуском.")
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
    g = garment_drift(keyframes, [source(k) for k in keyframes])
    _say("одежда", g["stable"], g["note"][:96])
    if not g["stable"]:
        print("      одежда плывёт между узлами. Задать --garment-ref одной "
              "картинкой одежды или зафиксировать сид; см. garment.py.")

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

    report = {"clip": res.clip_path, "keyframes": len(keyframes),
              "identity": drift.get("note"), "loop": seam.get("ratio"),
              "motion": quality.get("worst_jump"),
              "garment": gclip.get("regions"), "chain": res.note}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nклип: {res.clip_path}\nотчёт: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
