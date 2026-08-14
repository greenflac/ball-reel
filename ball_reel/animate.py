"""AnimateDiff локально: личность из фото, движение из условий, одним проходом.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ `gpu_keyframes` + `chain`. Там кадры рисуются поодиночке
и сшиваются видеомоделью через шлюз — то есть связность между кадрами
обеспечивает чужой сервис. Здесь связность обеспечивает модуль движения внутри
той же UNet: кадры считаются СОВМЕСТНО, и «мигание» между ними лечится не
сшивкой, а тем, что их никогда и не было по отдельности.

Практическое следствие для продукта: внешний API из пути генерации уходит
целиком. Это было требование, а не пожелание — прод не может ходить наружу.

ТРИ КАНАЛА, И КАЖДЫЙ ОТВЕЧАЕТ ЗА СВОЁ

* **личность** — эмбеддинг ArcFace через IP-Adapter FaceID. Проверено по
  исходнику diffusers 0.39: у FaceID своя ветка загрузки, энкодер картинок ему
  не нужен (`image_encoder_folder=None`), а вместе с адаптером ШТАТНО
  подгружается его LoRA — `self.load_lora_weights(lora, adapter_name="faceid_0")`.
  То есть LoRA здесь не украшение, а несущая часть канала личности;
* **движение** — покадровые условия ControlNet OpenPose, снятые DWPose с
  driving-видео. Ровно то, что лежит в `kit/conditions`;
* **движение КАМЕРЫ** — отдельная Motion LoRA (`pan`, `zoom`, `tilt`). Она
  закрывает строку из `PRODUCT.md`, где сказано, что движение камеры у нас не
  отделено от движения субъекта.

ПАМЯТЬ — ГЛАВНОЕ ОГРАНИЧЕНИЕ, И ОНО РЕШАЕТСЯ НЕ НАСТРОЙКАМИ, А ПЛАНОМ

На одной UNet висят три довеска сразу: модуль движения, ControlNet и адаптер
лица. Поэтому режим считается ЗАРАНЕЕ (`plan`), до единой загрузки весов, и
считается по числу гигабайт, а не по надежде. Плохой план виден за секунду,
плохая загрузка — через минуту и с падением.

ПОТОЛОК РАЗРЕШЕНИЯ ПЕРЕХОДИТ В ТРЕБОВАНИЕ К КАДРИРОВКЕ. На 512x768 лицо в
полный рост занимает около 72 px, а ArcFace отказывается судить мельче ~110.
Значит либо кадр по пояс (доля лица удваивается), либо идентичность в этом
прогоне не проверяется — и тогда так и надо сказать, а не показывать зелёный
вердикт, полученный ни на чём.

Ничего не меряет: измерение — дело гейта. Здесь только генерация.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: База. SD1.5, потому что вся экосистема AnimateDiff (модуль движения,
#: ControlNet, FaceID) собрана под неё. Лицензия базы — creativeml-openrail-m:
#: коммерция разрешена, но с поведенческими ограничениями, и это тот пункт,
#: который надо показать юристу вместе с остальными.
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

#: Модуль движения. v1-5-3 — последняя ревизия; проверено живьём, что в
#: репозитории лежит `diffusion_pytorch_model.safetensors` и его fp16-вариант.
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-3"

#: Условия позы. Имена файлов сверены с HF API, а не взяты по памяти: один раз
#: на этом проекте имя весов уже оказалось выдуманным, и код писался против
#: несуществующего API.
CONTROLNET = "lllyasviel/control_v11p_sd15_openpose"

#: Личность. `ip-adapter-faceid_sd15.bin` + одноимённая LoRA, которую diffusers
#: подгружает сам. Файлы проверены живьём.
IP_ADAPTER_REPO = "h94/IP-Adapter-FaceID"
IP_ADAPTER_WEIGHT = "ip-adapter-faceid_sd15.bin"

#: Движение камеры, отдельно от движения субъекта. Апстрим Apache 2.0.
MOTION_LORA_REPO = "guoyww/animatediff-motion-lora-{name}"
MOTION_LORAS = ("zoom-in", "zoom-out", "pan-left", "pan-right",
                "tilt-up", "tilt-down")

#: Сколько кадров модуль движения держит в одном окне. 16 — то, на чём он
#: обучен; больше требует скользящего окна (`enable_free_noise`), и это
#: отдельная проверка, которой у нас не было.
CONTEXT_FRAMES = 16

#: Доля высоты кадра, которую занимает лицо. ИЗМЕРЕНО на живых прогонах:
#: 9.4% на полном росте, вдвое больше при кадрировке по пояс. Отсюда и
#: считается, увидит ли гейт лицо вообще.
FULL_BODY_FACE_SHARE = 0.094
WAIST_UP_FACE_SHARE = 0.19

#: Ниже этого числа пикселей ArcFace отказывается судить старт-кадр. Держится
#: здесь копией намеренно: `animate` не должен импортировать гейт ради одного
#: числа, а расхождение поймает тест.
MIN_JUDGED_FACE_PX = 70


@dataclass
class Plan:
    """Режим генерации, посчитанный ДО загрузки весов."""

    width: int
    height: int
    frames: int
    dtype: str
    offload: str
    attention_slicing: bool
    vae_slicing: bool
    face_px: float
    identity_verifiable: bool
    notes: list = field(default_factory=list)

    def render(self) -> str:
        lines = [f"{self.width}x{self.height}, {self.frames} кадров, "
                 f"{self.dtype}, выгрузка: {self.offload}",
                 f"лицо в кадре ~{self.face_px:.0f} px -> "
                 + ("идентичность проверяема"
                    if self.identity_verifiable else
                    f"ИДЕНТИЧНОСТЬ НЕ ПРОВЕРЯЕТСЯ (порог {MIN_JUDGED_FACE_PX} px)")]
        return "\n".join(lines + [f"  - {n}" for n in self.notes])


def plan(vram_gb: float, *, waist_up: bool = True,
         frames: int = CONTEXT_FRAMES) -> Plan:
    """Гигабайты видеопамяти -> режим генерации. Чистая функция, без весов.

    ПОЧЕМУ ЭТО ОТДЕЛЬНАЯ ФУНКЦИЯ. На одной UNet висят модуль движения,
    ControlNet и адаптер лица; ошибиться в режиме — значит узнать об этом
    минутой позже и падением по памяти, уже после загрузки нескольких
    гигабайт. Здесь ответ получается за миллисекунду и проверяется тестом.

    Границы ВЫБРАНЫ по размерам весов в fp16 (SD1.5 UNet ~1.7 ГБ, модуль
    движения ~1.8 ГБ, ControlNet ~0.7 ГБ, адаптер лица ~0.1 ГБ = ~4.3 ГБ
    только на веса) плюс активации на 16 кадров. На карте не проверялись —
    пометка обязательна, пока не проверены.
    """
    notes = []
    if vram_gb >= 10:
        w, h, offload = 640, 960, "none"
    elif vram_gb >= 7:
        w, h, offload = 512, 768, "model"
        notes.append("модельная выгрузка: медленнее, но 16 кадров помещаются")
    elif vram_gb >= 5:
        w, h, offload = 512, 768, "model"
        notes.append("6 ГБ — впритык: три довеска на одной UNet. Если упадёт по "
                     "памяти, первым снимать ControlNet, а не разрешение: без "
                     "него теряется движение, но кадр остаётся судимым")
    else:
        w, h, offload = 384, 576, "sequential"
        notes.append("последовательная выгрузка: работает, но в разы медленнее")

    share = WAIST_UP_FACE_SHARE if waist_up else FULL_BODY_FACE_SHARE
    face_px = h * share
    verifiable = face_px >= MIN_JUDGED_FACE_PX
    if not verifiable:
        notes.append(f"на этом разрешении лицо выходит ~{face_px:.0f} px, "
                     f"а судить можно от {MIN_JUDGED_FACE_PX}: либо кадрировать "
                     f"по пояс, либо честно сказать, что идентичность в этом "
                     f"прогоне не проверялась")
    if not waist_up:
        notes.append("полный рост: доля лица 9.4% высоты — измерено, не оценка")
    return Plan(width=w, height=h, frames=frames, dtype="float16",
                offload=offload, attention_slicing=True, vae_slicing=True,
                face_px=face_px, identity_verifiable=verifiable, notes=notes)


def preflight(vram_gb: float | None = None) -> dict:
    """Всё ли на месте ДО генерации. Отдельно от `plan`: тот считает, этот щупает.

    Порядок проверок — по стоимости отказа: сначала то, что делает генерацию
    невозможной (нет CUDA), потом то, что делает её бессмысленной (нет условий).
    """
    report: dict = {"ok": True, "checks": [], "notes": []}

    def add(name: str, ok, detail: str):
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        if ok is False:
            report["ok"] = False

    try:
        import torch

        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "-"
        got = vram_gb
        if cuda and got is None:
            got = torch.cuda.get_device_properties(0).total_memory / 1e9
        # CPU-сборка определяется по `torch.version.cuda`, а НЕ по суффиксу
        # `+cpu` в номере версии: колесо с PyPI суффикса не несёт, и проверка
        # по строке молча пропускала ровно тот случай, ради которого написана.
        # Поймано тестом; это второй за день дефект вида «два способа ответить
        # на один вопрос».
        built_with_cuda = getattr(torch.version, "cuda", None)
        add("torch", cuda,
            f"{torch.__version__}, cuda={cuda}, собран с CUDA "
            f"{built_with_cuda or 'НЕТ'}, карта: {name}"
            + (f", {got:.1f} ГБ" if got else ""))
        if not cuda and not built_with_cuda:
            report["notes"].append(
                "стоит CPU-сборка torch (собрана без CUDA вовсе). "
                "Переустановить с индекса CUDA:\n"
                "  pip uninstall -y torch torchvision\n"
                "  pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu124")
    except ImportError:
        add("torch", False, "не установлен")
        got = vram_gb

    try:
        import diffusers

        need = (diffusers.__version__ >= "0.30")
        add("diffusers", need,
            f"{diffusers.__version__}"
            + ("" if need else " — FaceID и AnimateDiff требуют 0.30+"))
    except ImportError:
        add("diffusers", False, "не установлен")

    for mod, why in (("insightface", "эмбеддинг лица для FaceID"),
                     ("onnxruntime", "исполняет insightface")):
        try:
            __import__(mod)
            add(mod, True, why)
        except ImportError:
            add(mod, False, f"не установлен — {why}")

    report["plan"] = plan(got or 0.0).render() if got else "VRAM неизвестна"
    return report


def build(cfg: Plan, *, motion_lora: str | None = None, base: str = BASE_MODEL,
          verbose: bool = True):
    """Собрать пайплайн под уже посчитанный план. Возвращает готовый pipe.

    ПОРЯДОК СБОРКИ ВАЖЕН, и он не произвольный:

    1. адаптер движения и ControlNet передаются В КОНСТРУКТОР — они часть
       архитектуры, а не довесок;
    2. адаптер лица грузится ПОСЛЕ, и обязательно с `image_encoder_folder=None`:
       у FaceID своего энкодера картинок нет, он берёт эмбеддинг ArcFace. Без
       этого аргумента загрузчик пойдёт искать несуществующую папку;
    3. выгрузка на CPU включается ПОСЛЕДНЕЙ. Включённая раньше, она уводит веса
       с карты до того, как адаптер успел к ним прицепиться, — на этом проекте
       такая ошибка уже была найдена разбором, до всякого железа.
    """
    # СНАЧАЛА ПРОВЕРКИ, КОТОРЫЕ НИЧЕГО НЕ СТОЯТ. Здесь имя motion LoRA
    # проверялось в середине функции, уже после загрузки адаптера движения и
    # ControlNet, — то есть опечатка в аргументе обходилась в полтора гигабайта
    # трафика и минуту ожидания вместо мгновенного отказа. Поймано тестом,
    # который на этом повис.
    if motion_lora is not None and motion_lora not in MOTION_LORAS:
        raise ValueError(
            f"неизвестная motion LoRA {motion_lora!r}; есть: "
            f"{', '.join(MOTION_LORAS)}")

    import torch
    from diffusers import (AnimateDiffControlNetPipeline, ControlNetModel,
                           MotionAdapter)

    dtype = torch.float16
    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER, torch_dtype=dtype)
    controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=dtype)
    pipe = AnimateDiffControlNetPipeline.from_pretrained(
        base, motion_adapter=adapter, controlnet=controlnet,
        torch_dtype=dtype, safety_checker=None)

    # Личность. Вместе с адаптером diffusers ШТАТНО подгружает его LoRA —
    # проверено по исходнику загрузчика, adapter_name будет `faceid_0`.
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=None,
                         weight_name=IP_ADAPTER_WEIGHT,
                         image_encoder_folder=None)

    if motion_lora:
        pipe.load_lora_weights(MOTION_LORA_REPO.format(name=motion_lora),
                               adapter_name=f"motion_{motion_lora}")

    if cfg.attention_slicing:
        pipe.enable_attention_slicing()
    if cfg.vae_slicing:
        pipe.enable_vae_slicing()
    if cfg.offload == "model":
        pipe.enable_model_cpu_offload()
    elif cfg.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    if verbose:
        print(f"собрано: {base} + {MOTION_ADAPTER.split('/')[-1]} + "
              f"{CONTROLNET.split('/')[-1]} + FaceID"
              + (f" + motion-lora:{motion_lora}" if motion_lora else ""))
        print("активные LoRA:", active_loras(pipe))
    return pipe


def active_loras(pipe) -> list:
    """Какие LoRA реально прицеплены. Не «мы её загрузили», а «она в модели».

    Существует потому, что «подключили LoRA» — утверждение, которое обязано
    быть проверяемым в рантайме, а не на слово. Ровно то же правило, по
    которому гейт не спрашивает генератор, получилось ли у него.
    """
    for attr in ("get_active_adapters", "get_list_adapters"):
        fn = getattr(pipe, attr, None)
        if fn is None:
            continue
        try:
            got = fn()
        except Exception:
            continue
        if isinstance(got, dict):
            names = sorted({n for v in got.values() for n in v})
        else:
            names = list(got)
        if names:
            return names
    return []


def animate(pipe, cfg: Plan, *, face_embeds, conditions: list, prompt: str,
            negative: str = "", steps: int = 20, guidance: float = 7.5,
            controlnet_scale: float = 1.0, ip_adapter_scale: float = 0.7,
            seed: int = 0):
    """Условия + эмбеддинг лица -> кадры. Возвращает список PIL-кадров.

    `conditions` — покадровые рисунки скелета из `skeleton.render_sequence`.
    Их число задаёт длину клипа и обязано совпасть с `cfg.frames`: модуль
    движения обучен на окне в 16 кадров, и молча взять другое число значит
    получить рассыпающееся движение вместо ошибки.

    `face_embeds` — эмбеддинг ArcFace, а НЕ картинка лица. У FaceID нет
    энкодера картинок; подать сюда изображение — распространённая ошибка,
    которая не падает, а тихо портит идентичность.
    """
    import torch

    if len(conditions) != cfg.frames:
        raise ValueError(
            f"условий {len(conditions)}, а план требует {cfg.frames}. Модуль "
            f"движения обучен на окне {CONTEXT_FRAMES}; другое число кадров "
            f"требует скользящего окна (enable_free_noise), и это отдельная "
            f"проверка, которой у нас не было.")
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    out = pipe(
        prompt=prompt, negative_prompt=negative or None,
        num_frames=cfg.frames, width=cfg.width, height=cfg.height,
        conditioning_frames=conditions,
        controlnet_conditioning_scale=controlnet_scale,
        ip_adapter_image_embeds=[face_embeds],
        num_inference_steps=steps, guidance_scale=guidance,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    return out.frames[0]


def main(argv: list) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="ball_reel.animate",
        description="локальная генерация видео: AnimateDiff + ControlNet + FaceID")
    ap.add_argument("--preflight", action="store_true",
                    help="только проверить окружение и показать план")
    ap.add_argument("--vram", type=float, default=None,
                    help="гигабайт видеопамяти (по умолчанию спросить у карты)")
    ap.add_argument("--full-body", action="store_true",
                    help="полный рост вместо кадра по пояс (лицо вдвое мельче)")
    args = ap.parse_args(argv)

    rep = preflight(args.vram)
    for c in rep["checks"]:
        mark = "ПРОПУСК" if c["ok"] is None else ("PASS" if c["ok"] else "FAIL")
        print(f"{mark:<7} {c['name']:<14} {c['detail']}")
    for n in rep["notes"]:
        print("\n" + n)
    if args.vram:
        print("\n" + plan(args.vram, waist_up=not args.full_body).render())
    if args.preflight:
        return 0 if rep["ok"] else 1
    if not rep["ok"]:
        print("\nОСТАНОВЛЕНО: чинить предполёт, генерация без него бессмысленна.")
        return 1
    print("\nГенерация запускается из run_local; этот модуль — сборка и план.")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
