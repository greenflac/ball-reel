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

#: Cosine-distance starting point for "still the same person". Chosen, to be
#: calibrated on real crops; see module docstring.
SAME_PERSON_MAX = 0.35

_ANALYZER = None


def _analyzer():
    """Lazy singleton FaceAnalysis. Import guarded so the offline package that
    never calls this does not need insightface installed."""
    global _ANALYZER
    if _ANALYZER is None:
        from insightface.app import FaceAnalysis  # type: ignore

        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=0, det_size=(640, 640))
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


def face_embedding(path: str | Path):
    """The largest face's embedding in an image, or None if no face is found.

    None is a real answer, not an error: a frame the detector cannot find a face
    in is a frame whose identity we cannot vouch for, and the caller scores that
    as maximum drift rather than laundering it into a pass.
    """
    import numpy as np  # noqa: F401  (ensures numpy present alongside the model)

    faces = _analyzer().get(_read_bgr(path))
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return faces[-1].normed_embedding


def _read_bgr(path: str | Path):
    """Load an image as BGR for InsightFace (which expects OpenCV order)."""
    from PIL import Image
    import numpy as np

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def arcface_drift(frame_paths, reference_path) -> dict:
    """Per-frame identity drift away from a reference face, by embedding distance.

    Same return shape as `identity.identity_drift` so it is a drop-in for the
    live path: ``per_frame`` / ``worst`` / ``drifted`` / ``readable`` / ``note``.
    A frame with no detectable face gets drift 1.0 and is listed as drifted — an
    unverifiable face is not a passing one.
    """
    ref = face_embedding(reference_path)
    if ref is None:
        return {"per_frame": {}, "worst": (None, None), "drifted": [],
                "readable": 0,
                "note": "no face in the reference photo: cannot measure identity."}
    per_frame: dict[str, float] = {}
    drifted: list[str] = []
    readable = 0
    for p in frame_paths:
        emb = face_embedding(p)
        name = Path(p).name
        readable += 1
        if emb is None:
            per_frame[name] = 1.0
            drifted.append(name)
            continue
        d = cosine_distance(ref, emb)
        per_frame[name] = d
        if d > SAME_PERSON_MAX:
            drifted.append(name)
    if not per_frame:
        return {"per_frame": {}, "worst": (None, None), "drifted": [],
                "readable": 0, "note": "no readable frames."}
    worst = max(per_frame, key=lambda n: per_frame[n])
    note = (f"identity via ArcFace cosine distance (real embedding): "
            f"{len(drifted)}/{readable} frame(s) past {SAME_PERSON_MAX}. "
            f"Bar re-derived on cosine scale, not the proxy's.")
    return {"per_frame": per_frame, "worst": (worst, per_frame[worst]),
            "drifted": drifted, "readable": readable, "note": note}
