"""Pose-exact keyframes on a 4 GB GPU: SD1.5 + ControlNet OpenPose + IP-Adapter.

This is the piece that makes motion a constraint. Measured on the public
gateway, a role-labelled pose reference did not transfer pose at all (distance
0.59-0.65 against a bar of 0.25) — the renderer was free to reinterpret. A
ControlNet does not reinterpret: the skeleton is an input channel, so the joints
land where the condition image puts them.

WHY 4 GB IS ENOUGH HERE, when no video model fits in it. A video model has to
hold every frame's latents at once, which is what pushes AnimateDiff to ~8-12 GB
and Wan VACE past 24. Generating keyframes is a sequence of SINGLE-image
diffusions, so peak memory is one frame's worth. The trajectory still gets
pinned — at the nodes rather than continuously — and the API's end-frame
interpolation fills the gaps at 0.01 pollen/second. Keyframe density is the
accuracy dial, and cost scales with it rather than with VRAM.

MEMORY BUDGET at 512x768, fp16, on a 4 GB card:
    UNet                    ~1.7 GB
    ControlNet (openpose)   ~0.7 GB
    VAE (tiled/sliced)      ~0.2 GB
    IP-Adapter + encoder    ~0.6 GB
    latents + workspace     ~0.4 GB
                            ~3.6 GB, which is why every saving below is ON by
default and not a tuning knob. Attention slicing, VAE slicing and tiling, and
model CPU offload are what keep the peak under the card rather than over it.

Nothing here runs without a GPU, and the generating half was written but NOT
executed — there is no GPU in the environment it was authored in. `plan()`
exists so the configuration can be reviewed and tested on CPU; anything that
actually loads weights says plainly that it is unverified until it runs on real
hardware.

Что из этого УЖЕ проверено без карты, потому что проверялось не исполнением:
имена весов и раскладка репозиториев сверены с живым HuggingFace API, и три
дефекта загрузки, найденных адверсарным ревью, исправлены — репозиторий
FaceID, отсутствующий `image_encoder`, и порядок offload относительно загрузки
адаптера. Все три роняли бы первый же прогон ПОСЛЕ скачивания ~7 ГБ весов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Base model. SD1.5 rather than SDXL: SDXL's UNet alone is ~5 GB in fp16 and
#: does not fit, and the ControlNet/IP-Adapter ecosystem for SD1.5 is the one
#: with mature OpenPose and FaceID adapters.
def _base_model() -> str:
    """Имя базы берётся из `animate`, а НЕ дублируется здесь.

    Тут стояло `runwayml/stable-diffusion-v1-5`, а в `animate` —
    `stable-diffusion-v1-5/stable-diffusion-v1-5`. Первое редиректит на второе,
    то есть веса ОДНИ И ТЕ ЖЕ, и расхождение выглядело безобидным.

    Безобидным оно не было: кэш HuggingFace ключуется СТРОКОЙ идентификатора, а
    не тем, куда она ведёт. Два имени — две записи кэша и 1.7 ГБ, скачанных
    дважды. На карте, где время считают минутами, это выяснилось бы посреди
    прогона.

    Пятый за два дня случай одной формы: второй способ узнать то, что уже
    кто-то знает.
    """
    from .animate import BASE_MODEL as _BASE

    return _BASE


BASE_MODEL = _base_model()
CONTROLNET_OPENPOSE = "lllyasviel/control_v11p_sd15_openpose"

#: Identity by ADAPTER, not fine-tuning. A LoRA per user does not scale and
#: creates an artefact indistinguishable from real identity at verification
#: time; an adapter conditions on the embedding and leaves the gate meaningful.
#:
#: Репозиторий именно этот, и это не мелочь: FaceID лежит ОТДЕЛЬНО от обычных
#: IP-Adapter'ов. Проверено по HF API — в `h94/IP-Adapter` файлов со словом
#: faceid нет вообще (там ip-adapter_sd15.bin и родня), а в
#: `h94/IP-Adapter-FaceID` они лежат в КОРНЕ, без подпапки `models`. Здесь
#: раньше стояла пара «первый репозиторий + subfolder=models», то есть первый
#: же вызов на арендованной карте упал бы EntryNotFoundError — после того, как
#: скачаны ~7 ГБ весов SD1.5 и ControlNet.
IP_ADAPTER_REPO = "h94/IP-Adapter-FaceID"
IP_ADAPTER_FACEID = "ip-adapter-faceid_sd15.bin"
#: FaceID — это адаптер ПЛЮС LoRA; без неё лицо обусловлено наполовину.
IP_ADAPTER_LORA = "ip-adapter-faceid_sd15_lora.safetensors"

#: 512x768 is the largest 2:3 frame that stays inside the budget above. The reel
#: is finished at 9:16 by the video step, and upscaling happens AFTER the gate —
#: never before, or the gate measures the upscaler's invention.
WIDTH, HEIGHT = 512, 768

#: Какую долю высоты кадра занимает лицо при ПОЛНОРОСТОВОЙ композиции.
#: Замерено на живом клипе 1080x1920: лицо 181 px, то есть 9.4% высоты
#: [проверено live]. Число геометрическое, а не модельное: оно одинаково для
#: любого генератора, включая тот опенсорс, который поедет в прод.
FULL_BODY_FACE_SHARE = 0.094

#: Поясная композиция даёт примерно вдвое большую долю — это и есть выход для
#: карты, на которой полный рост не помещается по разрешению.
WAIST_UP_FACE_SHARE = 0.19


def face_px_at(height: int, share: float = FULL_BODY_FACE_SHARE) -> int:
    """Сколько пикселей займёт лицо в кадре такой высоты."""
    return int(height * share)


def identity_verifiable(height: int, *, share: float = FULL_BODY_FACE_SHARE,
                        floor: int | None = None) -> tuple:
    """Хватит ли разрешения, чтобы гейт вообще МОГ судить лицо.

    Это не про качество картинки, а про то, останется ли работающим главный
    измеритель проекта. ArcFace отказывается судить лицо мельче своего порога,
    и на полноростовом вертикальном кадре доля лица фиксирована геометрией —
    значит порог превращается в требование к РАЗРЕШЕНИЮ, о котором надо знать
    до аренды карты, а не после.

    Найдено живым прогоном: на 464x832 (wan-fast) лицо вышло 78 px и клип стал
    непроверяемым, на 1080x1920 (veo) — 181 px и идентичность стала измеримой.
    Ни та, ни другая модель в прод не поедет, но геометрия останется.
    """
    from .identity_arcface import MIN_FACE_PX

    floor = MIN_FACE_PX if floor is None else floor
    got = face_px_at(height, share)
    if got >= floor:
        return True, f"лицо ~{got} px при пороге {floor} — судимо"
    need = int(floor / share)
    return False, (
        f"лицо ~{got} px при пороге {floor}: гейт НЕ СМОЖЕТ судить "
        f"идентичность. Нужен кадр от {need} px по высоте, либо более тесное "
        f"кадрирование (по пояс даёт вдвое большую долю), либо честно принять, "
        f"что на этой конфигурации идентичность не проверяется.")


@dataclass
class GPUPlan:
    """The configuration a run would use, inspectable without a GPU."""

    base_model: str = BASE_MODEL
    controlnet: str = CONTROLNET_OPENPOSE
    ip_adapter: str = IP_ADAPTER_FACEID
    width: int = WIDTH
    height: int = HEIGHT
    steps: int = 24
    guidance: float = 6.0
    controlnet_scale: float = 1.0
    ip_adapter_scale: float = 0.7
    dtype: str = "float16"
    #: LoRA реализма: путь к файлу или repo_id. ПУСТО ПО УМОЛЧАНИЮ, и это
    #: принципиально. Влияние ручки измеримо только против прогона без неё, а
    #: если она включена с самого начала, базы для сравнения не существует —
    #: остаётся «мне кажется, стало лучше».
    #: Вес FaceID-LoRA. ЕДИНИЦА БЫЛА ЗАШИТА, и это оказалось дефектом: LoRA
    #: обучена под ванильную SD1.5, а на фотореалистичном дообучении полный вес
    #: вместе с проекцией (0.7) переобусловливает — лицо расползается радужными
    #: потёками. Замерено на живом кадре: 1.0 + 0.7 даёт разложение, и это не
    #: «модель слабая», а мы давим вдвоём в одну точку.
    faceid_lora_scale: float = 1.0
    realism_lora: str = ""
    realism_lora_weight: str = ""
    #: Насколько сильно. 1.0 обычно перебивает и лицо тоже.
    realism_lora_scale: float = 0.7
    optimisations: list = field(default_factory=lambda: [
        "attention_slicing", "vae_slicing", "vae_tiling",
        "model_cpu_offload"])
    estimated_vram_gb: float = 3.6
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def plan(vram_gb: float = 4.0, keyframes: int = 5) -> GPUPlan:
    """The settings for a card of this size, with the reasoning attached.

    Separated from execution so the plan can be reviewed, diffed and tested
    before any hardware is rented — the same split that lets the whole
    extraction half be trusted before a GPU exists.
    """
    p = GPUPlan()
    if vram_gb < 3.5:
        p.notes.append(
            f"{vram_gb} GB is below the ~3.6 GB this configuration needs even "
            f"with every saving on. Drop to 448x640, or use a larger card.")
    if vram_gb >= 8:
        # 768x1152, а не 640x960. Разница не в красоте: 960 даёт лицо ~90 px
        # при пороге 100, то есть на карте вдвое большей гейт всё равно
        # остаётся слепым. 1152 даёт ~108 px — измеритель начинает работать.
        # Выбирать разрешение, не глядя на порог идентичности, значит купить
        # память и не купить проверяемость.
        p.width, p.height = 768, 1152
        p.optimisations = ["attention_slicing", "vae_slicing", "vae_tiling"]
        p.estimated_vram_gb = 7.0
        p.notes.append(
            "768x1152 выбрано по порогу идентичности, а не по памяти: это "
            "минимальная высота, на которой лицо в полноростовом кадре "
            "остаётся судимым. Памяти хватило бы и на больше.")
        p.notes.append(
            "at this size a video model (AnimateDiff + ControlNet) becomes "
            "possible and would give CONTINUOUS pose control instead of "
            "per-node — worth benchmarking before committing to keyframes.")
    ok, why = identity_verifiable(p.height)
    if not ok:
        p.notes.append(f"ИДЕНТИЧНОСТЬ НА ПОЛНОМ РОСТЕ: {why}")
    else:
        p.notes.append(f"идентичность: {why}")
    p.notes.append(
        f"{keyframes} keyframes pin the trajectory at {keyframes} points; "
        f"between them the interpolation is the video model's guess. Denser "
        f"keyframes tighten the bound and cost proportionally more.")
    p.notes.append(
        "UNVERIFIED: written against the documented diffusers API, never "
        "executed — there was no GPU in the authoring environment. Weight "
        "names and repo layout ARE verified against the live HF API; the "
        "memory numbers are not. Treat the first real run as the test.")
    return p


#: Лимит текстового энкодера SD1.5. Не рекомендация: всё сверх молча
#: отбрасывается, без предупреждения и без ошибки.
CLIP_TOKEN_LIMIT = 77


def count_tokens(text: str, tokenizer=None) -> int:
    """Длина промта в токенах CLIP; при отсутствии токенизатора — оценка.

    Точный счёт возможен только там, где стоит transformers (то есть на машине
    с картой). В среде разработки его нет, поэтому используется приближение
    ~1.3 токена на слово — грубое, но достаточное, чтобы поймать промт, который
    вылезает за лимит в полтора раза.
    """
    if tokenizer is not None:
        return len(tokenizer(text).input_ids)
    return int(len(text.split()) * 1.3)


def fit_prompt(parts: list, tokenizer=None,
               limit: int = CLIP_TOKEN_LIMIT) -> tuple:
    """Собрать промт по убыванию важности и сказать, что не поместилось.

    Порядок здесь не косметический. SD1.5 обрезает ХВОСТ, поэтому то, что
    стоит последним, исчезает первым — молча. Реальный промт этого пайплайна
    выходит примерно на 89 токенов при лимите 77, так что обрезка не гипотеза.

    Первой идёт одежда: её задаёт только текст, и если она обрежется, ткань
    начнёт плыть между кейфреймами — тот самый дефект, который ловит
    `garment.garment_drift`. Кадрирование, наоборот, в этом стеке лишнее:
    ControlNet уже держит позу и композицию скелетом, а FRAMING писался для
    пути, где никакого управления позой не было.
    """
    kept, dropped, text = [], [], ""
    for part in [p for p in parts if p]:
        candidate = " ".join(kept + [part])
        if count_tokens(candidate, tokenizer) <= limit:
            kept.append(part)
            text = candidate
        else:
            dropped.append(part)
    return text, dropped


def face_embeds(face_photo: str, dtype: str = "", device: str = ""):
    """Лицо как ЭМБЕДДИНГ, в той форме, которую ждёт FaceID.

    FaceID обусловливается не картинкой: у него на входе не CLIP-эмбеддинг
    изображения, а 512-мерный вектор ArcFace. Передать сюда `ip_adapter_image`
    (как здесь было) — не ошибка типов и не падение: адаптер получит не тот
    сигнал, кадр отрисуется, а лицо будет чужим. Именно такой промах гейт потом
    и поймает, но объяснит невнятно — «дрейф идентичности» вместо «адаптеру
    дали не то».

    Вектор берётся у того же анализатора, которым лицо потом ПРОВЕРЯЕТСЯ. Это
    сознательно: если бы обуславливающий эмбеддинг считался другой моделью,
    расхождение двух моделей читалось бы как дрейф личности.

    Форма — [2, 1, 512]: первый ряд нулевой (негативная ветка CFG), второй сам
    вектор. Тип обязан совпасть с типом пайплайна, иначе torch скажет
    "expected scalar type Half but found Float" уже внутри UNet.
    """
    import torch  # type: ignore

    from .device import detect, dtype_for
    from .identity_arcface import face_detail

    device = device or detect()
    dtype = dtype or dtype_for(device)
    d = face_detail(face_photo)
    if d is None:
        raise RuntimeError(
            f"в {face_photo} не найдено лицо: FaceID нечем обуславливать. "
            f"Это должен был поймать intake ещё дома.")
    ref = torch.from_numpy(d["embedding"]).unsqueeze(0)          # [1, 512]
    both = torch.cat([torch.zeros_like(ref), ref])               # [2, 512]
    return both.unsqueeze(1).to(dtype=getattr(torch, dtype), device=device)


def _loaded_adapters(pipe) -> list:
    """Какие адаптеры РЕАЛЬНО в модели. Спрашиваем её, а не свои аргументы.

    То же правило, по которому гейт не спрашивает генератор, получилось ли у
    него: список имён, собранный из собственных вызовов, разойдётся с
    действительностью при первой же правке загрузчика в апстриме.
    """
    cfgs = getattr(getattr(pipe, "unet", None), "peft_config", None)
    return list(cfgs) if cfgs else []


def load_pipeline(cfg: GPUPlan | None = None, *, device: str = ""):
    """Собрать пайплайн один раз.

    Отдельной функцией, потому что `render_keyframes` вызывается дважды за
    прогон (дым, потом все узлы), и собирать веса заново на каждый вызов — это
    вторая полная загрузка UNet + ControlNet + адаптера и второй пик памяти
    сразу после первого. На арендованной карте это прямые минуты.

    ПОРЯДОК ВЫЗОВОВ ЗДЕСЬ — ЧАСТЬ КОНТРАКТА diffusers, а не стиль:
    from_pretrained -> load_ip_adapter -> LoRA -> экономии -> offload ПОСЛЕДНИМ.
    Если offload включить раньше загрузки адаптера, хуки уже расставлены, и
    догруженная проекция остаётся на CPU, пока UNet считает на карте — первый
    же `pipe(...)` падает на "Expected all tensors to be on the same device".
    Раньше здесь было именно так, и `except` вокруг загрузки объяснял бы это
    как «IP-Adapter failed to load», то есть чинили бы не то.
    """
    import torch  # type: ignore
    from diffusers import (ControlNetModel,  # type: ignore
                           StableDiffusionControlNetPipeline)

    from .device import detect

    device = device or detect()
    cfg = cfg or plan()
    dtype = getattr(torch, cfg.dtype)

    # variant="fp16" — не микрооптимизация. Без него diffusers качает веса
    # полной точности, то есть вдвое больше и байтов, и места на диске, а
    # считать всё равно будет в fp16: карта на 4 ГБ другого не выдержит. На
    # арендованной машине это лишние минуты, на ноутбуке — лишние гигабайты.
    # Оба репозитория fp16-варианты публикуют (проверено по HF API), но если
    # у какого-то его не окажется, падать нельзя — откатываемся на полные.
    def _load(cls, repo, **kw):
        try:
            return cls.from_pretrained(repo, torch_dtype=dtype,
                                       variant="fp16", **kw)
        except Exception:  # noqa: BLE001 — нет fp16-варианта, берём полный
            return cls.from_pretrained(repo, torch_dtype=dtype, **kw)

    controlnet = _load(ControlNetModel, cfg.controlnet)
    pipe = _load(StableDiffusionControlNetPipeline, cfg.base_model,
                 controlnet=controlnet, safety_checker=None)

    try:
        # subfolder=None: файл лежит в корне репозитория FaceID.
        # image_encoder_folder=None: у FaceID энкодера изображений НЕТ, а
        # diffusers по умолчанию идёт искать папку `image_encoder` и падает.
        pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=None,
                             weight_name=cfg.ip_adapter,
                             image_encoder_folder=None)
        # FaceID-LoRA ЗДЕСЬ НЕ ГРУЗИТСЯ РУКАМИ: `load_ip_adapter` выше уже
        # загрузил её сам, под именем `faceid_0`. Так устроена ветка FaceID в
        # diffusers, и в докстринге `animate` это записано верно — а здесь был
        # ещё один вызов, с рассуждением «без имён второй вызов заменит
        # первый». Рассуждение верное, применённое не к тому: заменять было
        # нечего.
        #
        # ЦЕНА ДУБЛЯ ИЗМЕРЕНА: в UNet оказывались ДВА адаптера с одной и той же
        # LoRA — ['faceid_0', 'faceid'], — и вместе с проекцией эмбеддинга они
        # давили в одну точку втроём. Лицо и кожа расползались радужными
        # потёками; выглядело как «модель слабая», а было переобусловливание
        # нашей же сборкой. diffusers об этом предупреждал: «Already found a
        # `peft_config` attribute in the model. This will lead to having
        # multiple adapters».
        names = [n for n in _loaded_adapters(pipe) if n.startswith("faceid")]
        weights = [cfg.faceid_lora_scale] * len(names)
        if cfg.realism_lora:
            kw = ({"weight_name": cfg.realism_lora_weight}
                  if cfg.realism_lora_weight else {})
            pipe.load_lora_weights(cfg.realism_lora, adapter_name="realism",
                                   **kw)
            names.append("realism")
            weights.append(cfg.realism_lora_scale)
        pipe.set_adapters(names, adapter_weights=weights)
        pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)
    except Exception as e:  # noqa: BLE001
        # Identity is load-bearing: losing the adapter silently would produce a
        # stranger in the right pose, which the gate would then reject with a
        # confusing reason. Say it here instead.
        raise RuntimeError(
            f"IP-Adapter failed to load ({e}). Without it the face is not "
            f"conditioned at all and every keyframe will be a stranger.") from e

    # Экономии — те, что назвал план, а не все подряд. План печатается перед
    # прогоном и потому читается как описание того, что произойдёт; пока сюда
    # был зашит фиксированный набор, на карте >=8 ГБ план говорил одно,
    # а исполнялось другое.
    # Нарезка внимания идёт ЧЕРЕЗ ОБЁРТКУ С ОТКАТОМ, а не напрямую: на
    # `UNet2DConditionModel` она стирает все 16 процессоров IP-Adapter, то есть
    # канал личности целиком (измерено; на `UNetMotionModel` — не стирает).
    # Подробности и числа — в `animate.enable_slicing_without_losing_identity`.
    from .animate import enable_slicing_without_losing_identity

    savings = {
        "attention_slicing": lambda: enable_slicing_without_losing_identity(pipe),
        "vae_slicing": pipe.enable_vae_slicing,
        "vae_tiling": pipe.enable_vae_tiling,
    }
    for name in cfg.optimisations:
        if name in savings:
            savings[name]()
    if "model_cpu_offload" in cfg.optimisations:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def render_keyframes(condition_images: list, face_photo: str, prompt: str,
                     out_dir: str | Path, *, cfg: GPUPlan | None = None,
                     negative: str = "", seed: int = 0, pipe=None,
                     device: str = "") -> dict:
    """Generate one image per condition skeleton, holding the face fixed.

    `condition_images` come from `skeleton.render_sequence` — already retargeted
    to the target's proportions, so the skeleton describes the client's body
    performing the driving motion rather than the donor's.

    `pipe` — уже собранный пайплайн (см. `load_pipeline`); передавать его между
    вызовами дешевле, чем грузить веса заново.

    Returns the same manifest shape `chain.render_keyframes` returns, so the
    downstream chain and gate do not care which renderer produced the nodes.
    """
    import torch  # type: ignore
    from PIL import Image

    from .device import detect

    device = device or detect()
    cfg = cfg or plan()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = pipe or load_pipeline(cfg, device=device)
    embeds = face_embeds(face_photo, dtype=cfg.dtype, device=device)

    made = []
    for i, cond_path in enumerate(condition_images):
        cond = Image.open(cond_path).convert("RGB").resize(
            (cfg.width, cfg.height))
        image = pipe(
            prompt=prompt, negative_prompt=negative or None, image=cond,
            num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
            controlnet_conditioning_scale=cfg.controlnet_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed + i),
            width=cfg.width, height=cfg.height,
            ip_adapter_image_embeds=[embeds]).images[0]
        path = out_dir / f"kf_{i:04d}.png"
        image.save(path)
        made.append(str(path))
        from .device import empty_cache

        empty_cache(device)
    return {"keyframes": made, "config": cfg.to_dict(),
            "count": len(made), "source": "gpu"}


def requirements() -> str:
    """What to install on the VPS, pinned to the 4 GB path."""
    return "\n".join([
        "# 4 GB VRAM keyframe renderer",
        "torch --index-url https://download.pytorch.org/whl/cu121",
        "diffusers>=0.27",
        "transformers",
        "accelerate",
        "safetensors",
        "insightface        # IP-Adapter FaceID needs the same embeddings the gate uses",
        "onnxruntime-gpu",
    ])
