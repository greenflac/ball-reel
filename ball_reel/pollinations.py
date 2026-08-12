"""Pollinations — ONE gateway (gen.pollinations.ai) for the whole pipeline.

Text, images, VIDEO, audio, vision — all through one OpenAI-compatible host,
addressed by model name. This is the client the vacancy's stack runs on: Flux /
Flux Kontext for stills, Seedance/Veo/Wan for image-to-video, a vision model for
the judge, ElevenLabs multilingual for Russian TTS.

Auth: an sk_ key in POLLINATIONS_API_KEY (Bearer). Base overridable via
POLLINATIONS_BASE (default https://gen.pollinations.ai) and POLLINATIONS_MEDIA
(default https://media.pollinations.ai).

Guarded: `requests` imported lazily; ffmpeg on PATH only for video frame
extraction. Written against the documented gen.pollinations.ai endpoints; run it
on the bench with a key.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.parse import quote


def _base() -> str:
    return os.environ.get("POLLINATIONS_BASE", "https://gen.pollinations.ai").rstrip("/")


def _media() -> str:
    return os.environ.get("POLLINATIONS_MEDIA", "https://media.pollinations.ai").rstrip("/")


def _key() -> str:
    k = os.environ.get("POLLINATIONS_API_KEY")
    if not k:
        raise RuntimeError("POLLINATIONS_API_KEY not set (sk_ key from enter.pollinations.ai).")
    return k


def _auth() -> dict:
    return {"Authorization": f"Bearer {_key()}"}


# ---------------------------------------------------------------------------
# media: local file -> a public URL the image/video endpoints can reference
# ---------------------------------------------------------------------------
def upload(path: str | Path) -> str:
    """Upload a local file and return its public media URL. [verified live]

    POST media.pollinations.ai/upload (multipart) -> 200 application/json
        {"id": "<uuid>", "url": "https://media.pollinations.ai/<uuid>",
         "contentType": "image/jpeg", "size": 122539}
    `GET {url}` then serves the bytes publicly (200, original content-type), which
    is what makes it usable as the video endpoint's server-side-fetched start
    frame. The media host is SEPARATE from gen.pollinations.ai and needs
    *.pollinations.ai allowed by the egress proxy."""
    import requests

    with open(path, "rb") as fh:
        r = requests.post(f"{_media()}/upload", headers=_auth(),
                          files={"file": fh}, timeout=180)
    r.raise_for_status()
    data = r.json()
    url = data.get("url") or (f"{_media()}/{data['id']}" if data.get("id") else None)
    if not url:
        raise RuntimeError(f"upload: no url/id in response: {list(data.keys())}")
    return url


# ---------------------------------------------------------------------------
# image: text (+ optional reference image) -> still
# ---------------------------------------------------------------------------
def image(prompt: str, out_path: str | Path, *, model: str = "flux", seed: int = 0,
          width: int = 1080, height: int = 1920, image_url: str | None = None) -> str:
    """Generate a still (text->image). Returns image bytes synchronously.

    [verified live] GET /image/{prompt}?model=flux&width=&height=&seed= -> image/*.

    ``image_url`` conditions on a reference (image-to-image, e.g. a face on
    kontext). NOTE: Pollinations fetches that URL server-side, so it must be a
    PUBLICLY fetchable URL (a media.pollinations.ai/{id} from `upload`), not a
    local path and not a self-referential gen URL. For a local face photo prefer
    `images_edit()` — it uploads the bytes to the allowed host directly.
    Params are model/width/height/seed/quality/image/transparent — there is NO
    `nologo` param (verified: not in the API)."""
    import requests

    url = f"{_base()}/image/" + quote(prompt, safe="")
    params = {"model": model, "seed": seed, "width": width, "height": height}
    if image_url:
        params["image"] = image_url
    r = requests.get(url, params=params, headers=_auth(), timeout=300)
    r.raise_for_status()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)
    return str(out_path)


def images_edit(prompt: str, ref_path: str | Path, out_path: str | Path, *,
                model: str = "kontext", width: int = 1080, height: int = 1920) -> str:
    """Image-to-image from a LOCAL reference, no media host needed. [verified live]

    POST /v1/images/edits (multipart) -> 200 JSON
        {created, data:[{b64_json, revised_prompt}], usage}
    The face bytes go straight to gen.pollinations.ai (the allowed host); the
    result comes back base64 in data[0].b64_json. This is the preferred
    start-frame path: it avoids the separate media.pollinations.ai upload."""
    import requests

    with open(ref_path, "rb") as fh:
        r = requests.post(
            f"{_base()}/v1/images/edits", headers=_auth(),
            data={"model": model, "prompt": prompt, "size": f"{width}x{height}"},
            files={"image": (Path(ref_path).name, fh, "image/jpeg")}, timeout=300)
    r.raise_for_status()
    item = r.json()["data"][0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if item.get("b64_json"):
        out_path.write_bytes(base64.b64decode(item["b64_json"]))
    elif item.get("url"):
        img = requests.get(item["url"], headers=_auth(), timeout=180)
        img.raise_for_status()
        out_path.write_bytes(img.content)
    else:
        raise RuntimeError(f"images_edit: no b64_json/url in response: {item.keys()}")
    return str(out_path)


def compose(prompt: str, image_urls: list[str], out_path: str | Path, *,
            model: str = "nanobanana", width: int = 768, height: int = 1024,
            seed: int = 0) -> str:
    """Generate from SEVERAL reference images at once. [verified live]

    GET /image/{prompt}?model=&image=<url1>|<url2> -> image bytes. This is the
    only way to specify a body: an editing model restyles clothing from text but
    keeps the body it was handed, so a build that differs from the face photo
    has to arrive as a picture. Refer to the images by position in the prompt
    ("the FIRST image for the face, the SECOND for build and clothing") — that
    phrasing was what made the roles stick.

    Reference capacity is per model (`max_reference_images` in /image/models):
    kontext 1 (cannot do this), nanobanana 3, klein/seedream5-pro 10,
    seedream5/nanobanana-pro 14, gptimage 16. Verified on the same inputs:
    nanobanana took build and clothing from the second image while holding the
    face at 0.176 drift; seedream5 lost the face entirely at 0.752. Capacity is
    not fidelity — re-measure before switching models.

    The URLs are fetched server-side, so they must be public (`upload()` them).
    """
    import requests

    if len(image_urls) < 2:
        raise ValueError("compose() is for 2+ references; use images_edit/image "
                         "for a single one.")
    url = f"{_base()}/image/" + quote(prompt, safe="")
    r = requests.get(url, params={"model": model, "image": "|".join(image_urls),
                                  "width": width, "height": height, "seed": seed},
                     headers=_auth(), timeout=600)
    if not r.ok:
        raise RuntimeError(f"compose: HTTP {r.status_code} {r.text[:300]}")
    if "image" not in r.headers.get("content-type", ""):
        raise RuntimeError(f"compose: expected image bytes, got "
                           f"{r.headers.get('content-type')}: {r.text[:200]!r}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)
    return str(out_path)


# ---------------------------------------------------------------------------
# video: text (+ reference image = the start frame) -> mp4 -> frames
# ---------------------------------------------------------------------------
def video(prompt: str, out_mp4: str | Path, *, model: str = "seedance-2.0",
          image_url: str | list[str] | None = None, duration: int = 4,
          aspect_ratio: str = "9:16", audio: bool = False,
          resolution: str | None = None, seed: int | None = None) -> str:
    """Image-to-video: the start frame drives the motion.

    GET /video/{prompt}?model=&image={URL}&duration=&aspectRatio=&audio=
    -> 200 video/mp4, binary body (per the published OpenAPI schema; the
    response declares only video/mp4, no async job envelope). The JSON guard
    below stays as a tripwire in case a model ever answers with a job.

    `image_url` must be PUBLICLY fetchable (`upload()` the start frame first —
    the server fetches it from its own side). Pass a list to use both keyframes:
    element 0 is the start frame, element 1 the end frame on models whose
    `video_capabilities` include `end_frame`; they go over the wire as one
    `image` param joined by `|`. [verified: /openapi.json + /video/models]

    `duration` is per-model and RANGE-CHECKED server-side (400 otherwise):
    seedance-2.0 4-15s, seedance-pro 2-10s, wan 2-15s, veo 4/6/8s. Billing is
    per video-second, so duration is the cost dial: seedance-2.0 0.18 pollen/s,
    wan-fast 0.01 pollen/s (use wan-fast to shake out plumbing cheaply).
    `resolution` (480p/720p/1080p) only applies to resolution-priced models
    (veo, wan-pro, p-video, seedance-pro)."""
    import requests

    url = f"{_base()}/video/" + quote(prompt, safe="")
    params = {"model": model, "duration": duration, "aspectRatio": aspect_ratio,
              "audio": str(audio).lower()}
    if image_url:
        params["image"] = image_url if isinstance(image_url, str) else "|".join(image_url)
    if resolution:
        params["resolution"] = resolution
    if seed is not None:
        params["seed"] = seed
    r = requests.get(url, params=params, headers=_auth(), timeout=900)
    if not r.ok:
        raise RuntimeError(f"video: HTTP {r.status_code} {r.text[:300]}")
    ct = r.headers.get("content-type", "")
    if "video" not in ct and "octet-stream" not in ct:
        raise RuntimeError(
            f"video: expected mp4 bytes, got {ct}: {r.text[:200]!r}. "
            f"If this is a job/URL JSON, add a poll+download branch here.")
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_mp4.write_bytes(r.content)
    LAST_VIDEO_USAGE.clear()
    LAST_VIDEO_USAGE.update(_usage_of(r))
    return str(out_mp4)


# Billing is reported in RESPONSE HEADERS, not the body (the body is raw mp4):
# x-usage-completion-video-seconds, x-model-used, x-request-id. [verified live]
LAST_VIDEO_USAGE: dict = {}


def _usage_of(r) -> dict:
    """Pull the metering headers off a generation response, for the run report."""
    keep = ("x-usage-completion-video-seconds", "x-model-used", "x-request-id",
            "x-usage-completion-audio-seconds", "x-cache")
    out = {k: v for k, v in ((k, r.headers.get(k)) for k in keep) if v}
    secs = out.get("x-usage-completion-video-seconds")
    if secs:
        try:  # cost = video-seconds x the model's per-second pollen rate
            out["video_seconds"] = float(secs)
        except ValueError:
            pass
    return out


def video_loop(prompt: str, out_mp4: str | Path, start_url: str, **kwargs) -> str:
    """A clip that ends where it began, so it loops without a visible cut.

    Uses the video endpoint's SECOND reference image as the end frame
    (`image=<start>|<end>`) with the same frame at both ends. The model then has
    to return the subject to its starting position, which is what makes the last
    frame cut back to the first cleanly — rather than fading or freezing, which
    is what "make it loop" in the prompt text alone tends to produce.

    Only for models whose `video_capabilities` include `end_frame`. Снято с
    GET /video/models [проверено live]: wan-fast, veo, wan-pro, seedance-2.0 —
    и только они, у всех четырёх max_reference_images=2. Здесь раньше был
    назван `wan`, у которого end_frame НЕТ (maxref=1), и не был назван
    `wan-fast`, на котором прогон считается по умолчанию.

    Канонический список — `chain.END_FRAME_POLLEN`; этот абзац его повторяет
    для чтения, а проверяет вызов `chain.generate_chain`.
    """
    return video(prompt, out_mp4, image_url=[start_url, start_url], **kwargs)


#: Шаблон имени кадра. Число знаков — не косметика, см. `extract_frames`:
#: кадры собираются обратно сортировкой ИМЁН, и ширины поля должно хватать на
#: весь клип, иначе порядок ломается молча.
FRAME_PATTERN = "%04d.png"


def frame_names_sort_correctly(count: int, pattern: str = FRAME_PATTERN) -> bool:
    """Совпадает ли лексикографический порядок имён с порядком кадров.

    Существует затем, чтобы это утверждение можно было проверить тестом, а не
    держать в голове: ровно его нарушение и было дефектом.
    """
    names = [pattern % i for i in range(1, count + 1)]
    return sorted(names) == names


def extract_frames(mp4_path: str | Path, out_dir: str | Path, *, fps: int = 6) -> list[str]:
    """mp4 -> NNNN.png sequence via ffmpeg, so identity/motion can be measured.

    Четыре знака, а не два, и это не запас на будущее. Кадры собираются обратно
    лексикографической сортировкой имён, и при `%02d` сотый кадр называется
    `100.png`, что встаёт между `10.png` и `11.png`. Клипы до сих пор были
    короткими (4-8 с при fps=6 — меньше сотни кадров), поэтому это не срабатывало
    ни разу; штатный прогон из GPU_RUNBOOK даёт 32 с, то есть 192 кадра.

    Последствие было бы тихим и хуже, чем падение: `loop_seam` мерил бы стык
    не с последним кадром клипа, а с серединой, а `motion_quality` считал бы
    шаги по перемешанной последовательности. На заведомо гладком клипе это
    воспроизводимо даёт «16 телепортов конечностей» — то есть гейт отправил бы
    чинить движение, с которым всё в порядке.
    """
    import subprocess

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4_path), "-vf", f"fps={fps}",
         str(out_dir / FRAME_PATTERN)],
        check=True, capture_output=True,
    )
    return sorted(str(p) for p in out_dir.glob("*.png"))


# ---------------------------------------------------------------------------
# chat / vision judge
# ---------------------------------------------------------------------------
def chat(messages: list[dict], *, model: str = "openai", temperature: float = 0.0) -> str:
    import requests

    r = requests.post(f"{_base()}/v1/chat/completions", headers=_auth(),
                      json={"model": model, "temperature": temperature, "messages": messages},
                      timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


JUDGE_SYSTEM = (
    "You score one vertical 9:16 ad frame. Return STRICT JSON only with keys: "
    '"first_frame_hook","trend_fit","composition","brand_safety" (0-1), '
    '"transcribed_text" (verbatim on-image words or empty), "opinion" (0-1).'
)


def judge_frame(frame_path: str | Path, *, model: str = "claude") -> dict:
    b64 = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
    content = chat([
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": "Score this frame. JSON only."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]},
    ], model=model)
    return _parse_json(content)


def opinion_of(frame_path: str | Path, *, model: str = "claude") -> float:
    return float(judge_frame(frame_path, model=model).get("opinion", 0.0))


# ---------------------------------------------------------------------------
# audio: Russian TTS (lipsync is a separate model, not in this catalog)
# ---------------------------------------------------------------------------
def tts(text: str, out_path: str | Path, *, voice: str = "nova",
        model: str = "eleven-multilingual-v2") -> str:
    """Russian TTS -> mp3. Language lives here; a lipsync model consumes the wav."""
    import requests

    url = f"{_base()}/audio/" + quote(text, safe="")
    r = requests.get(url, params={"voice": voice, "model": model},
                     headers=_auth(), timeout=300)
    r.raise_for_status()
    out_path = Path(out_path)
    out_path.write_bytes(r.content)
    return str(out_path)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"model returned no JSON: {text[:120]!r}")
    return json.loads(text[s:e + 1])
