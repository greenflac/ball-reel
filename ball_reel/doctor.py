"""Live preflight: prove every Pollinations endpoint the pipeline needs, cheap first.

    python3 -m ball_reel.doctor            # free + cheap checks, NO video spend
    python3 -m ball_reel.doctor --video    # + one short seedance clip (costs balance)

Order is by cost so a failure stops you before you spend: models list (free) ->
image (cheap) -> images_edit (cheap) -> upload (cheap) -> tts (cheap) -> video
(only with --video). Each line prints status, content-type, bytes, seconds.
This is the "живой смок обязателен" gate — run it before produce.py on a real face.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

OUT = Path(os.environ.get("BALL_REEL_DOCTOR_OUT", "doctor_out"))


def _t(fn):
    t = time.time()
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 — preflight reports, never crashes
        ok, detail = False, f"EXC {type(e).__name__}: {e}"
    return ok, detail, time.time() - t


def main(argv: list[str]) -> int:
    import requests
    from . import pollinations

    OUT.mkdir(parents=True, exist_ok=True)
    base = pollinations._base()
    try:
        pollinations._key()
    except RuntimeError as e:
        print(f"FAIL  key: {e}")
        return 2

    results = []

    def check(name, fn):
        ok, detail, dt = _t(fn)
        print(f"{'PASS' if ok else 'FAIL'}  {name:<16} {dt:5.1f}s  {detail}")
        results.append(ok)
        return ok

    # 1. models list (free, no auth) — is the stack present?
    def _models():
        r = requests.get(f"{base}/v1/models", timeout=60)
        ids = {m.get("id") for m in r.json().get("data", [])}
        need = ["kontext", "seedance-2.0", "eleven-multilingual-v2"]
        missing = [m for m in need if m not in ids]
        return (not missing), f"{len(ids)} models; missing={missing or 'none'}"
    check("models", _models)

    # 2. image text->image (cheap)
    def _image():
        p = pollinations.image("a plain photo portrait, neutral background",
                               OUT / "img.jpg", model="flux",
                               width=512, height=512, seed=7)
        return Path(p).stat().st_size > 1000, f"-> {p}"
    have_face = check("image", _image)

    # 3. images_edit local face -> start frame (cheap, no media host)
    def _edit():
        p = pollinations.images_edit(
            "Keep this exact face; put him mid-jump on a fitness ball, vertical.",
            OUT / "img.jpg", OUT / "start.jpg", model="kontext",
            width=720, height=1280)
        return Path(p).stat().st_size > 1000, f"-> {p}"
    have_start = check("images_edit", _edit) if have_face else False

    # 4. upload -> media URL (needs *.pollinations.ai allowlisted)
    media_url = {}
    def _upload():
        u = pollinations.upload(OUT / "start.jpg")
        media_url["url"] = u
        return u.startswith("http"), f"-> {u[:70]}"
    have_url = check("upload", _upload) if have_start else False

    # 5. TTS Russian (cheap)
    def _tts():
        p = pollinations.tts("Привет! Прыгай на мяче каждое утро.",
                             OUT / "voice.mp3", voice="nova",
                             model="eleven-multilingual-v2")
        return Path(p).stat().st_size > 1000, f"-> {p}"
    check("tts", _tts)

    # 6. video image->video (ONLY with --video; the one that spends)
    if "--video" in argv:
        def _video():
            if not media_url.get("url"):
                return False, "no start-frame URL (upload failed) — cannot ref"
            p = pollinations.video(
                "The person jumps on the fitness ball; it compresses and rebounds.",
                OUT / "clip.mp4", model=os.environ.get("BALL_REEL_VIDEO_MODEL", "seedance-2.0"),
                image_url=media_url["url"],
                duration=int(os.environ.get("BALL_REEL_VIDEO_DURATION", "4")),
                aspect_ratio="9:16")
            return Path(p).stat().st_size > 10000, f"-> {p}"
        check("video(seedance)", _video)
    else:
        print("SKIP  video            pass --video to spend one short seedance clip")

    ok = all(results)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} "
          f"({sum(results)}/{len(results)})")
    print(f"artifacts in {OUT}/")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
