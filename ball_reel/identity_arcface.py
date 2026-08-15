"""The REAL identity instrument: ArcFace face-embedding drift.

This is what `identity.identity_drift`'s perceptual proxy becomes at live time.
The proxy in `identity.py` is composition-blind; this is not — it crops the face,
embeds it with a face-recognition model, and measures cosine distance to a
reference embedding. Same shape of return as the proxy, so the pipeline swaps
one for the other without touching the caller.

DEPENDENCIES (live only, not in the default install):
    pip install insightface onnxruntime numpy
The first run downloads the `buffalo_l` model pack (~300 MB) from the InsightFace
release. No GPU required — onnxruntime-cpu is enough for a demo; add
onnxruntime-gpu for throughput.

NOT RUN in the offline package. The cosine arithmetic below is unit-tested on
synthetic vectors (no model needed); the model-backed path is written against
InsightFace's documented API and is exercised only where a real environment is
present — the same honesty the eval repo applies to its live gateway code.

THRESHOLD. Cosine DISTANCE (1 - cosine similarity), not the proxy's hash scale.
On ArcFace embeddings the same person across frames typically sits well under
~0.35 and different identities well over ~0.6, model- and crop-dependent. So the
live bar is re-derived on this scale — it is NOT the proxy's 0.20 carried over.
`SAME_PERSON_MAX` is a documented starting point, to be calibrated on real frames
the way every constant here is.
"""

from __future__ import annotations

from pathlib import Path

#: Cosine-distance bar for "still the same person", on JUDGEABLE frames.
#: Calibrated on live clips (2026-08-12): a true-identity clip sat at 0.18-0.22
#: on its steady frames, so 0.35 leaves real headroom above that band.
SAME_PERSON_MAX = 0.35

#: A single frame may exceed SAME_PERSON_MAX (blur, extreme pose) without the
#: clip being a different person — but nothing should ever get this far out.
HARD_DRIFT_MAX = 0.6

#: Minimum detected-face size (shorter bbox side, px) for an embedding to be
#: TRUSTED. Not a taste call: the ArcFace recognizer takes a 112x112 crop, so a
#: smaller detection is mostly upsampled pixels and its distances inflate.
#: Confirmed on two live clips — faces at 111-114 px produced a 0.18-0.22
#: same-person band, while faces at 64-86 px never dropped below 0.36 even
#: frame-to-frame within one continuous shot. Below this we report "cannot
#: verify", which is NOT the same claim as "different person".
#:
#: УТОЧНЕНО ЗАМЕРОМ 2026-08-14, и уточнение меняет ПРИЧИНУ, а не число.
#: Утверждение «мелкая детекция — это в основном растянутые пиксели, и её
#: дистанции раздуваются» проверено напрямую: одна и та же фотография
#: уменьшалась так, чтобы лицо стало 200/160/140/127/110/95/83/70 px, и
#: сравнивалась САМА С СОБОЙ.
#:
#:     лицо px   200    160    140    127    110     95     83     70
#:     ArcFace 0.002  0.002  0.003  0.004  0.005  0.005  0.007  0.007
#:
#: То есть РАЗМЕР САМ ПО СЕБЕ не стоит почти ничего: 0.007 при баре 0.35, в
#: пятьдесят раз меньше. Раздутие в полосе 64-86 px, замеренное на живых
#: клипах, приходит НЕ от размера — остаются смаз и поза, а это другие
#: величины и другие пороги.
#:
#: Порог не сдвинут: чтобы его двигать, нужен замер на смазанных кадрах, а
#: его нет. Но обоснование теперь честное, и следствие практическое: число
#: 0.55 на мелком лице — НАСТОЯЩЕЕ расстояние, а не шум прибора, и списывать
#: его на «лицо мелкое» нельзя.
MIN_FACE_PX = 100

#: Floor for a single SHARP STILL (the start frame), which is a different
#: measurement from a video frame and must not borrow MIN_FACE_PX. A still has
#: no motion blur, so a smaller crop still yields a usable number: measured
#: live, start frames at 72/78/91/98 px scored 0.253/0.179/0.139/0.130 — a tight
#: band, and the 91 px one is the still that went on to produce a fully passing
#: clip. Judging stills at MIN_FACE_PX would have rejected that known-good frame
#: (a bug this constant exists to prevent). Distances here are noisier than on a
#: big crop, so the start check is a cheap screen for gross failure, not the
#: verdict — the verdict is taken on video frames at MIN_FACE_PX.
START_MIN_FACE_PX = 70

#: Fraction of frames that must be judgeable before a verdict means anything.
MIN_COVERAGE = 0.5

_ANALYZER = None


def _analyzer():
    """Lazy singleton FaceAnalysis. Import guarded so the offline package that
    never calls this does not need insightface installed."""
    global _ANALYZER
    if _ANALYZER is None:
        from insightface.app import FaceAnalysis  # type: ignore

        # genderage rides along with the pack we already download, so apparent
        # sex/age cost nothing extra and stay LOCAL — the subject check does not
        # get outsourced to the model being judged.
        app = FaceAnalysis(name="buffalo_l",
                           allowed_modules=["detection", "recognition", "genderage"])
        # ctx_id под реальное устройство: указать номер несуществующего
        # ускорителя значит получить молчаливый откат внутри чужой библиотеки.
        from .device import detect, insightface_ctx

        app.prepare(ctx_id=insightface_ctx(detect()), det_size=(640, 640))
        _ANALYZER = app
    return _ANALYZER


def cosine_distance(a, b) -> float:
    """1 - cosine similarity of two embedding vectors. Pure numpy; unit-tested.

    Separated from the model path on purpose: this is the arithmetic a sceptic
    checks, and it must be recomputable without downloading 300 MB of weights.
    """
    import numpy as np

    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    return round(max(0.0, 1.0 - sim), 4)


def face_detail(path: str | Path) -> dict | None:
    """The largest face in an image: embedding + how big and confident it was.

    None means no face at all. The size comes back with the embedding on
    purpose — an embedding without its crop size is not interpretable, because
    distance inflates on small faces (see MIN_FACE_PX).
    """
    import numpy as np  # noqa: F401  (ensures numpy present alongside the model)

    faces = _analyzer().get(_read_bgr(path))
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    f = faces[-1]
    x0, y0, x1, y1 = (float(v) for v in f.bbox)
    out = {"embedding": f.normed_embedding,
           "face_px": round(min(x1 - x0, y1 - y0)),
           "det_score": round(float(f.det_score), 3),
           # Kept so a caller can go back to the pixels — the face mesh needs a
           # crop when the face is small inside a full-body frame.
           "bbox": (x0, y0, x1, y1)}
    sex, age = getattr(f, "sex", None), getattr(f, "age", None)
    if sex is not None:
        out["sex"] = sex
    if age is not None:
        out["age"] = int(age)
    return out


def face_embedding(path: str | Path):
    """The largest face's embedding in an image, or None if no face is found."""
    d = face_detail(path)
    return None if d is None else d["embedding"]


def face_attributes(path: str | Path) -> dict | None:
    """Apparent sex/age of the largest face, from the local estimator.

    Returns the estimator's labels for THIS IMAGE (``{"sex": "M"|"F", "age":
    int, "face_px": int}``), or None if no face is found. Coarse by nature, and
    not a statement about who anyone is — its use here is strictly comparative:
    run it on the reference and on a generated frame, and a disagreement means
    the generator changed the person, which is the only question being asked.
    Compare like with like; do not read either number on its own as truth.
    """
    d = face_detail(path)
    if d is None:
        return None
    return {k: d[k] for k in ("sex", "age", "face_px") if d.get(k) is not None}


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile. Local so the verdict needs no scipy."""
    if not sorted_vals:
        return 1.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _read_bgr(path: str | Path):
    """Load an image as BGR for InsightFace (which expects OpenCV order)."""
    from PIL import Image
    import numpy as np

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def arcface_drift(frame_paths, reference_path, *,
                  min_face_px: int = MIN_FACE_PX) -> dict:
    """Per-frame identity drift away from a reference face, by embedding distance.

    Keeps `identity.identity_drift`'s keys (``per_frame`` / ``worst`` /
    ``drifted`` / ``readable`` / ``note``) so it stays a drop-in, and adds the
    ones a verdict on MOVING footage actually needs: ``median``, ``p90``,
    ``coverage``, ``judgeable``, ``too_small``, ``no_face``, ``face_px``.

    Why a median and not the worst frame: measured live, a clip of the genuine
    person still spikes to 0.71 on the one frame where the face is blurred at
    the top of a jump. Judging by the worst frame therefore fails every honest
    jump clip — it measures "is the face legible right now", not "is this the
    same person". The steady mass of frames is what carries identity, so the
    verdict rests on the median of the JUDGEABLE frames, with ``p90`` left
    exposed so a clip that morphs partway through still gets caught.

    Frames whose face is smaller than ``min_face_px`` are excluded from the
    distances and counted in ``too_small`` instead of being scored as maximum
    drift. That is not leniency: on a too-small crop the number is not evidence
    of anything, and "cannot verify" is a different claim from "different
    person". ``coverage`` is what stops that from becoming a free pass — a clip
    nobody could verify has low coverage, and the caller fails it on that.
    """
    ref_detail = face_detail(reference_path)
    empty = {"per_frame": {}, "face_px": {}, "worst": (None, None),
             "drifted": [], "readable": 0, "judgeable": 0, "too_small": [],
             "no_face": [], "median": None, "p90": None, "coverage": 0.0}
    if ref_detail is None:
        return {**empty,
                "note": "no face in the reference photo: cannot measure identity."}
    if ref_detail["face_px"] < min_face_px:
        return {**empty,
                "note": (f"reference face is only {ref_detail['face_px']}px "
                         f"(< {min_face_px}px): too small to identify from. "
                         f"Supply a closer photo.")}
    ref = ref_detail["embedding"]

    per_frame: dict[str, float] = {}
    face_px: dict[str, int] = {}
    drifted: list[str] = []
    too_small: list[str] = []
    no_face: list[str] = []
    total = 0
    for p in frame_paths:
        total += 1
        name = Path(p).name
        d = face_detail(p)
        if d is None:
            no_face.append(name)
            continue
        face_px[name] = d["face_px"]
        if d["face_px"] < min_face_px:
            too_small.append(name)
            continue
        dist = cosine_distance(ref, d["embedding"])
        per_frame[name] = dist
        if dist > SAME_PERSON_MAX:
            drifted.append(name)

    if not per_frame:
        why = (f"{len(too_small)} frame(s) had a face under {min_face_px}px and "
               f"{len(no_face)} had none" if total else "no frames")
        return {**empty, "face_px": face_px, "readable": total,
                "too_small": too_small, "no_face": no_face,
                "note": (f"identity NOT VERIFIABLE: {why}. The face is too small "
                         f"in this clip to identify — frame it closer.")}

    vals = sorted(per_frame.values())
    worst = max(per_frame, key=lambda n: per_frame[n])
    median = round(_quantile(vals, 0.5), 4)
    p90 = round(_quantile(vals, 0.9), 4)
    coverage = round(len(per_frame) / total, 3) if total else 0.0
    note = (f"identity via ArcFace cosine distance: median {median}, p90 {p90}, "
            f"worst {per_frame[worst]} on {len(per_frame)}/{total} judgeable "
            f"frame(s) (coverage {coverage:.0%}; {len(too_small)} face(s) under "
            f"{min_face_px}px, {len(no_face)} with no face). "
            f"{len(drifted)} judgeable frame(s) past {SAME_PERSON_MAX}.")
    return {"per_frame": per_frame, "face_px": face_px,
            "worst": (worst, per_frame[worst]), "drifted": drifted,
            "readable": total, "judgeable": len(per_frame),
            "too_small": too_small, "no_face": no_face,
            "median": median, "p90": p90, "coverage": coverage, "note": note}
