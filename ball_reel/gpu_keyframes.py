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

Nothing here runs without a GPU, and this module was written but NOT executed —
there is no GPU in the environment it was authored in. `plan()` exists so the
configuration can be reviewed and tested on CPU; anything that actually loads
weights says plainly that it is unverified until it runs on real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Base model. SD1.5 rather than SDXL: SDXL's UNet alone is ~5 GB in fp16 and
#: does not fit, and the ControlNet/IP-Adapter ecosystem for SD1.5 is the one
#: with mature OpenPose and FaceID adapters.
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_OPENPOSE = "lllyasviel/control_v11p_sd15_openpose"

#: Identity by ADAPTER, not fine-tuning. A LoRA per user does not scale and
#: creates an artefact indistinguishable from real identity at verification
#: time; an adapter conditions on the embedding and leaves the gate meaningful.
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_FACEID = "ip-adapter-faceid_sd15.bin"

#: 512x768 is the largest 2:3 frame that stays inside the budget above. The reel
#: is finished at 9:16 by the video step, and upscaling happens AFTER the gate —
#: never before, or the gate measures the upscaler's invention.
WIDTH, HEIGHT = 512, 768


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
        p.width, p.height = 640, 960
        p.optimisations = ["attention_slicing", "vae_slicing"]
        p.estimated_vram_gb = 6.5
        p.notes.append(
            "at this size a video model (AnimateDiff + ControlNet) becomes "
            "possible and would give CONTINUOUS pose control instead of "
            "per-node — worth benchmarking before committing to keyframes.")
    p.notes.append(
        f"{keyframes} keyframes pin the trajectory at {keyframes} points; "
        f"between them the interpolation is the video model's guess. Denser "
        f"keyframes tighten the bound and cost proportionally more.")
    p.notes.append(
        "UNVERIFIED: written against the documented diffusers API, never "
        "executed — there was no GPU in the authoring environment. Treat the "
        "first real run as the test, and expect the memory numbers to move.")
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


def render_keyframes(condition_images: list, face_photo: str, prompt: str,
                     out_dir: str | Path, *, cfg: GPUPlan | None = None,
                     negative: str = "", seed: int = 0) -> dict:
    """Generate one image per condition skeleton, holding the face fixed.

    `condition_images` come from `skeleton.render_sequence` — already retargeted
    to the target's proportions, so the skeleton describes the client's body
    performing the driving motion rather than the donor's.

    Returns the same manifest shape `chain.render_keyframes` returns, so the
    downstream chain and gate do not care which renderer produced the nodes.
    """
    import torch  # type: ignore
    from diffusers import (ControlNetModel,  # type: ignore
                           StableDiffusionControlNetPipeline)
    from PIL import Image

    cfg = cfg or plan()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(
        cfg.controlnet, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.base_model, controlnet=controlnet, torch_dtype=torch.float16,
        safety_checker=None)
    # Order matters: slicing and tiling must be enabled BEFORE offload, or the
    # pipeline is moved to the device first and the peak happens anyway.
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()

    try:
        pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="models",
                             weight_name=cfg.ip_adapter)
        pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)
        face = Image.open(face_photo).convert("RGB")
        ip_kwargs = {"ip_adapter_image": face}
    except Exception as e:  # noqa: BLE001
        # Identity is load-bearing: losing the adapter silently would produce a
        # stranger in the right pose, which the gate would then reject with a
        # confusing reason. Say it here instead.
        raise RuntimeError(
            f"IP-Adapter failed to load ({e}). Without it the face is not "
            f"conditioned at all and every keyframe will be a stranger.") from e

    made = []
    for i, cond_path in enumerate(condition_images):
        cond = Image.open(cond_path).convert("RGB").resize(
            (cfg.width, cfg.height))
        image = pipe(
            prompt=prompt, negative_prompt=negative or None, image=cond,
            num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
            controlnet_conditioning_scale=cfg.controlnet_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed + i),
            width=cfg.width, height=cfg.height, **ip_kwargs).images[0]
        path = out_dir / f"kf_{i:04d}.png"
        image.save(path)
        made.append(str(path))
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
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
