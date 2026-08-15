"""ОДИН кадр на карте, с числами. Цикл настройки в полминуты вместо пятнадцати.

ЗАЧЕМ ОТДЕЛЬНЫЙ ВХОД. Промт, базу, силу адаптера и масштаб LoRA подбирают
итерациями — это не расчёт, это глаз. А единственный доступный способ увидеть
результат стоил 15-31 минуту: у AnimateDiff нет дешёвого одиночного кадра,
модуль движения обучен на окне в 16 и считает их совместно. За вечер это
четыре попытки вместо сорока.

Здесь кадр рисует ДРУГОЙ пайплайн — `StableDiffusionControlNetPipeline`, без
модуля движения. Те же веса, тот же ControlNet, то же условие, тот же
эмбеддинг лица. Нет только временной связности, которая для одного кадра и не
нужна.

ПОБОЧНОЕ СЛЕДСТВИЕ, И ОНО ВАЖНЕЕ СКОРОСТИ. На этом пути FaceID-LoRA
ПРИЦЕПЛЯЕТСЯ. Тот `size mismatch`, из-за которого канал личности на
AnimateDiff работает наполовину, возникает только у `UNetMotionModel`: ключи
LoRA адресованы attention базовой SD1.5 (640), а после подключения модуля
движения размерности становятся другими (320/1280). Без модуля движения
конфликта нет.

Значит одиночный кадр показывает, на что канал личности способен ВООБЩЕ — то
есть верхнюю границу сходства, недостижимую в клипе. Читать его надо именно
так: если лицо не сошлось ЗДЕСЬ, в клипе оно не сойдётся тем более, и дело не
в модуле движения.

ЧТО ЗДЕСЬ НЕ ДЕЛАЕТСЯ. Не выносится вердикт. Печатаются числа — дистанция
лица, его размер в пикселях, расхождение позы с driving-кадром — и кладётся
файл. Судит гейт, и судит он клип, а не пробу.
"""

from __future__ import annotations

from pathlib import Path


def render(condition: str, face: str, prompt: str, out_path, *,
           base: str = "", negative: str = "", lora: str = "",
           lora_scale: float = 0.8, ip_adapter_scale: float = 0.7,
           faceid_lora_scale: float = 1.0,
           controlnet_scale: float = 1.0, steps: int = 24,
           guidance: float = 6.0, seed: int = 0, vram: float = 6.0) -> dict:
    """Одно условие + одно лицо -> один кадр. Возвращает отчёт с числами.

    Пайплайн собирается на каждый вызов: это цена ~30 с, и она приемлема, пока
    вызов один. Для серии есть `render_keyframes`, который принимает готовый
    `pipe`.
    """
    from PIL import Image

    from .gpu_keyframes import face_embeds, load_pipeline, plan

    cfg = plan(vram_gb=vram, keyframes=1)
    if base:
        cfg.base_model = base
    if lora:
        cfg.realism_lora, cfg.realism_lora_scale = lora, lora_scale
    cfg.ip_adapter_scale = ip_adapter_scale
    cfg.faceid_lora_scale = faceid_lora_scale
    cfg.controlnet_scale = controlnet_scale
    cfg.steps, cfg.guidance = steps, guidance

    pipe = load_pipeline(cfg)
    # ЧТО РЕАЛЬНО ПРИЦЕПИЛОСЬ — спрашивается у модели, а не у аргументов. На
    # этом пути FaceID-LoRA обязана быть в списке; если её нет, сходство лица
    # мерить бессмысленно, и знать об этом надо ДО того, как смотреть на кадр.
    from .animate import active_loras

    attached = active_loras(pipe)

    cond = Image.open(condition).convert("RGB").resize((cfg.width, cfg.height))
    embeds = face_embeds(face)

    import torch

    out = pipe(prompt=prompt, negative_prompt=negative or None,
               image=cond, controlnet_conditioning_scale=controlnet_scale,
               ip_adapter_image_embeds=[embeds],
               num_inference_steps=steps, guidance_scale=guidance,
               generator=torch.Generator("cpu").manual_seed(seed))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.images[0].save(out_path)
    return {"path": str(out_path), "loras": attached,
            "size": (cfg.width, cfg.height), "base": cfg.base_model}


def measure(frame: str, face: str, driving: str = "") -> dict:
    """Числа по одному кадру: лицо и поза. Каждое трёхзначно.

    `None` здесь означает «не смогли измерить», и это НЕ «плохо»: лицо мельче
    порога детектора и лицо непохожее — разные исходы, и путать их нельзя.
    """
    from .identity_arcface import START_MIN_FACE_PX, face_detail
    from .pose import landmarks, pose_delta

    got = {"face_px": None, "identity": None, "pose": None, "worst_joint": None}
    try:
        d = face_detail(frame)
    except Exception:  # noqa: BLE001 — кадр без лица не должен ронять пробу
        d = None
    if d:
        got["face_px"] = d["face_px"]
        if d["face_px"] >= START_MIN_FACE_PX:
            from .identity import arcface_drift

            drift = arcface_drift([frame], face,
                                  min_face_px=START_MIN_FACE_PX)
            got["identity"] = drift.get("median")
    if driving:
        a, b = landmarks(driving), landmarks(frame)
        if a and b:
            delta = pose_delta(a, b)
            if delta:
                got["pose"] = delta["mean"]
                got["worst_joint"] = delta.get("worst")
                got["compared"] = delta.get("compared")
                got["measurable"] = delta.get("measurable")
                got["coverage"] = delta.get("coverage")
    return got


def main(argv: list) -> int:
    import argparse
    import glob

    ap = argparse.ArgumentParser(
        prog="ball_reel.probe",
        description="один кадр на карте с числами — цикл настройки промта, "
                    "базы и сил адаптеров")
    ap.add_argument("--face", required=True)
    ap.add_argument("--conditions", default="demo/kit/conditions",
                    help="каталог условий; берётся ПЕРВОЕ")
    ap.add_argument("--condition", default="", help="конкретный файл условия")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative",
                    default="blurry, deformed, extra limbs, watermark, text")
    ap.add_argument("--base", default="",
                    help="фотореалистичное дообучение SD1.5; за реализм "
                         "отвечает ИМЕННО оно, а не LoRA личности")
    ap.add_argument("--lora", default="", help="обученная LoRA личности")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--ip-adapter-scale", type=float, default=0.7)
    ap.add_argument("--faceid-lora-scale", type=float, default=1.0,
                    help="вес FaceID-LoRA. Вместе с проекцией они давят В ОДНУ ТОЧКУ: 1.0 + 0.7 на чужой базе даёт радужные потёки вместо лица")
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vram", type=float, default=6.0)
    ap.add_argument("--out", default="probe_frame.png")
    args = ap.parse_args(argv)

    cond = args.condition
    if not cond:
        found = sorted(glob.glob(f"{args.conditions}/*.png"))
        if not found:
            print(f"нет условий в {args.conditions}")
            return 1
        cond = found[0]

    # driving-кадр к этому условию — из манифеста, чтобы позу было с чем
    # сверять. Нет манифеста — поза просто не меряется, это не отказ.
    driving = ""
    man = Path(args.conditions) / "manifest.json"
    if man.exists():
        import json

        from .run_local import resolve_driving

        raw = json.loads(man.read_text(encoding="utf-8")).get(
            "driving_frames") or {}
        driving = resolve_driving(raw.get(Path(cond).stem, ""),
                                  args.conditions) or ""

    print(f"условие: {cond}")
    got = render(cond, args.face, args.prompt, args.out, base=args.base,
                 negative=args.negative, lora=args.lora,
                 lora_scale=args.lora_scale,
                 ip_adapter_scale=args.ip_adapter_scale,
                 faceid_lora_scale=args.faceid_lora_scale,
                 controlnet_scale=args.controlnet_scale, steps=args.steps,
                 guidance=args.guidance, seed=args.seed, vram=args.vram)
    print(f"база: {got['base']}")
    attached = ", ".join(got["loras"]) or "НИЧЕГО — канал личности не собран"
    print(f"прицеплено: {attached}")

    m = measure(args.out, args.face, driving)
    px = m["face_px"]
    print(f"\nлицо: " + (f"{px} px" if px else "НЕ НАЙДЕНО"))
    if m["identity"] is not None:
        from .identity_arcface import SAME_PERSON_MAX

        mark = "тот же" if m["identity"] <= SAME_PERSON_MAX else "НЕ ТОТ"
        print(f"сходство: {m['identity']:.3f} при баре {SAME_PERSON_MAX} "
              f"-> {mark}")
    else:
        print("сходство: НЕ ИЗМЕРЕНО (лицо мельче порога или не найдено) — "
              "это не «прошло» и не «провалено»")
    if m["pose"] is not None:
        # ПО СКОЛЬКИМ СУСТАВАМ — рядом с числом, а не в подробностях. Измерено:
        # полноростовой кадр дал 0.239, поясной 0.081, и второе читалось как
        # «поза втрое точнее». На деле у поясного просто нет в кадре ног, и
        # среднее считалось по вчетверо меньшему набору. Без размера выборки
        # среднее — впечатление, а не измерение.
        cov = ""
        if m.get("compared"):
            cov = (f" по {m['compared']} суставам из {m['measurable']}"
                   f" (покрытие {m['coverage']:.2f})")
        print(f"поза против driving: {m['pose']:.3f}{cov}"
              + (f", худший сустав {m['worst_joint']}" if m["worst_joint"]
                 else ""))
    print(f"\nкадр: {args.out}  — смотреть ГЛАЗАМИ, числа не видят стиля")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
