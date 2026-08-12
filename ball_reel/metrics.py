"""The photo side: what one still of a person actually yields, as numbers.

The counterpart to `driving.py`. That module measures the MOVEMENT to copy;
this one measures the PERSON to apply it to. Together they are the two data
packages a conditioned generation needs, and both are produced on CPU, before
any generator is chosen.

Grouped by how much each number can be trusted, because they are not equal and
a flat dict would imply they are:

  identity   ArcFace embedding — a strong signal, and the one the output gate
             later re-measures. This is the anchor.
  geometry   Ratios from face and body landmarks. Objective measurements of the
             image: distances, divided by other distances. They describe the
             photo reliably.
  estimated  Apparent sex and age from a small classifier. Coarse, and labels
             for THIS IMAGE, not facts about a person. Kept separate so nobody
             reads them as identity.

Ratios rather than pixels throughout: a pixel distance changes with crop and
camera distance, so only ratios survive being compared across two photos.

What a face does NOT give: build, height, weight. Those need the body, and if
the photo is a head-and-shoulders portrait they are simply absent — which is
exactly why `subject.Subject` takes a body reference image instead of guessing.
"""

from __future__ import annotations

from pathlib import Path


def _dist(a, b) -> float:
    import numpy as np

    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


_MESH: dict = {}


def _face_mesh(model_path):
    """One landmarker per model file, kept alive for the process.

    Cached rather than created per call because MediaPipe's finaliser runs at
    interpreter shutdown and throws there — creating one per photo turns a
    normal loop into a wall of teardown tracebacks.
    """
    from mediapipe.tasks.python import BaseOptions, vision  # type: ignore

    key = str(model_path)
    if key not in _MESH:
        _MESH[key] = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=key),
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=True))
    return _MESH[key]


def face_metrics(photo: str | Path, *, face_model: str | Path | None = None) -> dict:
    """Everything measurable from a face, grouped by trustworthiness.

    Face geometry uses MediaPipe's 478-point mesh; the ratios are the classic
    proportion measures (how wide the face is for its height, how far apart the
    eyes sit relative to face width, and so on). They are stable under crop and
    scale, which pixel distances are not.
    """
    import os

    import numpy as np

    from .identity_arcface import face_detail

    out: dict = {"photo": str(photo), "identity": {}, "geometry": {},
                 "estimated": {}, "notes": []}

    det = face_detail(photo)
    if det is None:
        out["notes"].append("no face found: nothing measurable here.")
        return out
    out["identity"] = {
        "embedding_dim": int(np.asarray(det["embedding"]).shape[0]),
        "face_px": det["face_px"], "detector_score": det["det_score"],
    }
    if det["face_px"] < 100:
        out["notes"].append(
            f"face is only {det['face_px']}px: measurable, but both the "
            f"embedding and the ratios get noisy below ~100px.")
    for k in ("sex", "age"):
        if det.get(k) is not None:
            out["estimated"][k] = det[k]
    if out["estimated"]:
        out["notes"].append(
            "sex/age are a coarse classifier's labels for this image, not "
            "facts about a person; use them only to compare like with like.")

    model = Path(str(face_model or os.environ.get(
        "BALL_REEL_FACE_MODEL", "~/.mediapipe/face_landmarker.task"))).expanduser()
    if not model.exists():
        out["notes"].append(f"no face mesh model at {model}: geometry and "
                            f"expression skipped.")
        return out

    import mediapipe as mp  # type: ignore
    from PIL import Image

    landmarker = _face_mesh(model)
    with Image.open(photo) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        w, h = im.size
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        out["notes"].append("face mesh not found (too small or turned away).")
        return out

    lm = res.face_landmarks[0]
    # Aspect-corrected image coordinates, so ratios are not distorted by a
    # non-square frame.
    def p(i):
        return (lm[i].x * w, lm[i].y * h)

    # Canonical mesh indices: 10 forehead, 152 chin, 234/454 cheek sides,
    # 33/133 left eye corners, 362/263 right eye, 1 nose tip, 61/291 mouth.
    face_w = _dist(p(234), p(454))
    face_h = _dist(p(10), p(152))
    if face_w > 0 and face_h > 0:
        out["geometry"] = {
            "face_width_to_height": round(face_w / face_h, 4),
            "eye_spacing_to_face_width": round(_dist(p(133), p(362)) / face_w, 4),
            "left_eye_width_to_face_width": round(_dist(p(33), p(133)) / face_w, 4),
            "mouth_width_to_face_width": round(_dist(p(61), p(291)) / face_w, 4),
            "nose_to_chin_over_face_height": round(_dist(p(1), p(152)) / face_h, 4),
            "eye_to_mouth_over_face_height": round(
                _dist(((p(133)[0] + p(362)[0]) / 2, (p(133)[1] + p(362)[1]) / 2),
                      ((p(61)[0] + p(291)[0]) / 2, (p(61)[1] + p(291)[1]) / 2))
                / face_h, 4),
        }
    if res.face_blendshapes:
        shapes = {b.category_name: round(float(b.score), 4)
                  for b in res.face_blendshapes[0]}
        out["expression"] = {"count": len(shapes),
                             "top": dict(sorted(shapes.items(),
                                                key=lambda kv: -kv[1])[:6]),
                             "all": shapes}
    return out


def body_metrics(photo: str | Path) -> dict:
    """Build, as proportions — the part a face photo cannot give you.

    All lengths are divided by torso length, so they are comparable between a
    close crop and a wide shot, and between two different people. These are the
    numbers that say "broad-shouldered" or "long-legged" without an adjective.

    A portrait returns nothing here, and says so: that absence is precisely why
    build has to be supplied as a reference image rather than described.
    """
    from .pose import LIMBS, _normalise, landmarks

    out: dict = {"photo": str(photo), "proportions": {}, "notes": []}
    pts = landmarks(photo)
    if pts is None:
        out["notes"].append("no body found: a head-and-shoulders portrait "
                            "carries no build information at all.")
        return out
    norm = _normalise(pts)
    if norm is None:
        out["notes"].append("hips or shoulders not visible: build not measurable.")
        return out

    def seg(a, b):
        if norm[a][1] < 0.5 or norm[b][1] < 0.5:
            return None
        return round(_dist(norm[a][0], norm[b][0]), 4)

    prop = {f"{a}->{b}": seg(a, b) for a, b in LIMBS}
    shoulders = seg("l_shoulder", "r_shoulder")
    hips = seg("l_hip", "r_hip")
    prop["shoulder_width"] = shoulders
    prop["hip_width"] = hips
    if shoulders and hips and hips > 0:
        prop["shoulder_to_hip"] = round(shoulders / hips, 4)
    leg = [v for v in (seg("l_hip", "l_knee"), seg("l_knee", "l_ankle")) if v]
    if len(leg) == 2:
        prop["leg_length"] = round(sum(leg), 4)  # already in torso lengths
    out["proportions"] = {k: v for k, v in prop.items() if v is not None}
    missing = [k for k, v in prop.items() if v is None]
    if missing:
        out["notes"].append(f"{len(missing)} segment(s) occluded or out of "
                            f"frame: {', '.join(missing)}.")
    return out


def subject_package(photo: str | Path, **kw) -> dict:
    """Both halves of the photo-side data package, plus what is missing.

    `missing` is the useful field: it names what this photo cannot specify, so
    a caller knows what still has to be supplied (or will be invented by the
    generator, which is the failure this whole pipeline exists to prevent).
    """
    face = face_metrics(photo, **kw)
    body = body_metrics(photo)
    missing = []
    if not face.get("identity"):
        missing.append("identity (no face)")
    if not face.get("geometry"):
        missing.append("face geometry")
    if not body.get("proportions"):
        missing.append("build (no body in frame — supply a body reference)")
    return {"face": face, "body": body, "missing": missing}
