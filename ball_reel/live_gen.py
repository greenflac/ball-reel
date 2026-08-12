"""Live generation adapters — the real instruments behind the ``--live`` flag.

Each function is the boeuf of one gateway stage. All are written against the
tools' documented APIs and are GUARDED (imports and keys are read lazily), so the
offline package never needs any of them installed. NOT RUN in this repo: they
need GPU/weights or a provider key, exactly the "unexercised code written
against a published contract" the eval repo is honest about. Wire your
environment, then these stop raising.

Config is read from the environment so no key ever lives in the tree:
    HF / local weights for the image path (diffusers)
    VIDEO_API_URL, VIDEO_API_KEY          for image->video (Kling/Seedance host)
    TTS_API_URL, TTS_API_KEY              for Russian TTS
    LIPSYNC_API_URL, LIPSYNC_API_KEY      for audio-driven lipsync (LivePortrait/Sync)
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Start frame: a face-conditioned still (identity adapter + optional LoRA)
# ---------------------------------------------------------------------------
def start_frame_live(
    prompt: str,
    face_ref: str,
    seed: int,
    width: int,
    height: int,
    out_path: str | Path,
    *,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lora_path: str | None = None,
    lora_scale: float = 0.8,
    ip_adapter_scale: float = 0.6,
) -> str:
    """Generate a start frame that carries the reference face's identity.

    The real tools, wired the standard way:
      * diffusers SDXL as the base,
      * IP-Adapter FaceID for IDENTITY from ``face_ref`` (the face becomes a
        conditioning signal, not a sticker),
      * an optional LoRA (``lora_path``) for a trained character or house style,
        loaded with a scale so it tints rather than overrides.

    An alternative identity path is InstantID (a community diffusers pipeline);
    the IP-Adapter FaceID route is used here because it composes with an
    arbitrary LoRA cleanly. Swap the pipeline, keep the contract.

    Needs: torch, diffusers, insightface, a GPU for any real speed. Raises a
    clear error if those are absent — it does not fall back to a stand-in.
    """
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
        from insightface.app import FaceAnalysis
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - live only
        raise RuntimeError(
            "start_frame_live needs torch, diffusers, insightface, numpy, Pillow. "
            "pip install -r requirements-live.txt and provide weights/GPU."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # IP-Adapter FaceID: derive a face embedding from the reference and hand it
    # to the adapter as the identity signal.
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    with Image.open(face_ref) as im:
        bgr = np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
    faces = app.get(bgr)
    if not faces:
        raise RuntimeError("no face found in the reference photo for conditioning")
    face_emb = faces[0].normed_embedding

    # image_encoder_folder=None обязателен: у репозитория FaceID НЕТ папки
    # `image_encoder` (проверено по HF API), а diffusers по умолчанию идёт её
    # искать и падает ещё до первого кадра.
    pipe.load_ip_adapter(
        "h94/IP-Adapter-FaceID", subfolder=None,
        weight_name="ip-adapter-faceid_sdxl.bin", image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    if lora_path:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    # Форма [2, 1, 512] и ТИП пайплайна. normed_embedding — float32, а пайплайн
    # на карте создан в float16: без приведения torch скажет "expected scalar
    # type Half but found Float" уже внутри UNet. Первый ряд нулевой — это
    # негативная ветка CFG, без неё форма не та, что ждёт diffusers.
    dtype = torch.float16 if device == "cuda" else torch.float32
    ref = torch.from_numpy(face_emb).unsqueeze(0)
    embeds = torch.cat([torch.zeros_like(ref), ref]).unsqueeze(1).to(
        dtype=dtype, device=device)
    image = pipe(
        prompt=prompt,
        ip_adapter_image_embeds=[embeds],
        width=width, height=height, generator=generator,
        num_inference_steps=30, guidance_scale=5.0,
    ).images[0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# 2. Video: start frame -> motion (image-to-video host)
# ---------------------------------------------------------------------------
def video_live(start_frame: str, prompt: str, out_dir: str | Path,
               *, duration_s: int = 5, fps: int = 24) -> list[str]:
    """Drive an image-to-video model (Kling/Seedance) from the start frame.

    Provider-agnostic: posts the start frame + a motion prompt to
    ``VIDEO_API_URL`` with a bearer key, polls for the result, downloads the mp4,
    and extracts frames with Pillow-friendly ffmpeg. The exact JSON shape is the
    provider's; the contract here is start-frame-in, frame-sequence-out.

    Motion prompt should describe the JUMP (physics), not the product — the ball
    bounces, the person leaves and returns to it. Keep any on-frame text OUT of
    this call: text in motion melts; composite it as a still layer afterwards.
    """
    import requests

    url = _require_env("VIDEO_API_URL")
    key = _require_env("VIDEO_API_KEY")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(start_frame, "rb") as fh:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            data={"prompt": prompt, "duration": duration_s, "fps": fps},
            files={"image": fh},
            timeout=600,
        )
    resp.raise_for_status()
    video_url = resp.json()["video_url"]  # provider-specific field
    mp4 = out_dir / "clip.mp4"
    mp4.write_bytes(requests.get(video_url, timeout=600).content)
    _extract_frames(mp4, out_dir)  # ffmpeg -> NN.png
    return sorted(str(p) for p in out_dir.glob("*.png"))


def _extract_frames(mp4_path: Path, out_dir: Path) -> None:  # pragma: no cover - live
    import subprocess

    from .pollinations import FRAME_PATTERN

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4_path), "-vf", "fps=6",
         str(out_dir / FRAME_PATTERN)],
        check=True, capture_output=True,
    )


# ---------------------------------------------------------------------------
# 3. Voice: Russian TTS + audio-driven lipsync
# ---------------------------------------------------------------------------
def voice_live(start_frame: str, script_ru: str, out_path: str | Path) -> str:
    """Talking-head lipsync in Russian, on a talking start frame.

    Two calls, because lipsync is AUDIO-driven and language lives in the TTS:
      1. Russian TTS (``TTS_API_URL``) turns ``script_ru`` into a wav.
      2. A lipsync model (``LIPSYNC_API_URL``; LivePortrait/Sync-class) drives the
         face in ``start_frame`` from that wav.
    The lipsync model never sees "Russian" — it sees phonemes as audio. That is
    the whole reason the chain is split here.
    """
    import requests

    tts_url, tts_key = _require_env("TTS_API_URL"), _require_env("TTS_API_KEY")
    ls_url, ls_key = _require_env("LIPSYNC_API_URL"), _require_env("LIPSYNC_API_KEY")
    out_path = Path(out_path)

    wav = out_path.with_suffix(".wav")
    r = requests.post(tts_url, headers={"Authorization": f"Bearer {tts_key}"},
                      json={"text": script_ru, "lang": "ru"}, timeout=120)
    r.raise_for_status()
    wav.write_bytes(r.content)

    with open(start_frame, "rb") as face, open(wav, "rb") as audio:
        r2 = requests.post(ls_url, headers={"Authorization": f"Bearer {ls_key}"},
                           files={"image": face, "audio": audio}, timeout=600)
    r2.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r2.content)
    return str(out_path)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"{name} is not set. Live stages read provider config from the "
            f"environment so no key lives in the tree. Set it and retry."
        )
    return val
