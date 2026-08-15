"""Body pose as a measurement: did the pose survive, and is the body anatomical.

ArcFace answers "is this the same face". It says nothing about the body, which
is why a run could hold identity while quietly changing the pose from the
reference — caught by eye in a live run, not by any gate. This module is the
missing instrument.

It reads 33 body landmarks per frame with MediaPipe Pose (CPU, local, no
network, no key). Local matters for the same reason it matters for identity:
the check must not be outsourced to the model being judged.

Two questions, deliberately separate:

    pose_drift      — does the generated frame stand the way the reference does?
                      Scale- and position-invariant, so "same pose, framed
                      closer" is not scored as a different pose.

    limb_consistency— is it a BODY, or generator rubber? A real person's upper
                      arm does not change length between frames; a hallucinated
                      one does. Measured as variation of limb length across the
                      clip, in torso units.

DEPENDENCY (live only): pip install mediapipe. Imported lazily, so the offline
package and its tests do not need it. The arithmetic — normalisation, distance,
variation — is unit-tested on synthetic skeletons without the model, because
that is the part a sceptic recomputes.
"""

from __future__ import annotations

from . import cure

from pathlib import Path

#: Landmark indices used for the pose comparison. Face landmarks are excluded on
#: purpose: identity already owns the face, and including it would let a good
#: face mask a wrong body (and vice versa).
BODY_POINTS = {
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}

#: Limbs whose length must stay put across a clip, as (start, end) point names.
LIMBS = (("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
         ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
         ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
         ("r_hip", "r_knee"), ("r_knee", "r_ankle"))

#: Below this MediaPipe visibility a point is a guess, not an observation.
MIN_VISIBILITY = 0.5

#: Calibrated live, in torso-lengths (mean joint displacement after
#: normalisation), which fixes the scale everything below is read against:
#:     0.00        a frame against itself
#:     0.04-0.25   the same configuration, limbs in different detail
#:     0.05-0.30   the spread WITHIN one clip — i.e. ordinary motion
#:     0.67        a genuinely different pose (standing vs seated on the ball)
#:
#: The awkward consequence: on a MOVING clip, "the model changed the pose" and
#: "the person moved" both live in 0.05-0.30 and cannot be told apart. So the
#: pose is judged where motion is not a confound — the START FRAME, before
#: anything moves — and the video is only checked for wandering off to a
#: different configuration entirely. Same two-threshold shape the identity check
#: needed, for the same reason: two different measurements, one scale.

#: Start frame against the pose reference. Strict: nothing has moved yet, so
#: anything past this is the generator reinterpreting the pose, not the subject.
SAME_POSE_MAX = 0.15

#: Video frames against the reference. Loose on purpose — it must clear the 0.30
#: of legitimate motion and still catch the 0.67 of a different pose.
POSE_WANDER_MAX = 0.45

#: Bar for the WORST single joint, since the mean dilutes a change confined to
#: one limb (a clearly bent arm: mean 0.11, wrist 0.97).
#:
#: This turned out to be the measure that actually works on clips. Live, all
#: four models reproduced the reference pose faithfully in the START FRAME
#: (mean 0.018-0.043) and then diverged over the clip — so the pose problem is
#: accumulated drift during generation, not a misread still, and a gate on the
#: still alone would have caught nothing. Over the clip the worst joint orders
#: the models cleanly where the mean does not:
#:     veo 0.13 | wan 0.33 | wan-fast 0.41 | happyhorse 0.49
#: The ORDERING is the trustworthy part. Exactly where to cut rests on one
#: reference and one prompt, and 0.40 splits that particular set — re-derive it
#: on your own footage before trusting it to reject anything expensive.
WORST_JOINT_MAX = 0.40

#: Limb length may vary this much across a clip (coefficient of variation)
#: before the body is rubber rather than moving. Some variation is real:
#: foreshortening changes projected length as a limb swings toward the camera.
LIMB_WOBBLE_MAX = 0.25

#: Where the landmarker weights live. mediapipe 1.x dropped the old
#: `mp.solutions` API for the Tasks API, which needs an explicit model file:
#:   curl -sSLO --output-dir ~/.mediapipe \
#:     https://storage.googleapis.com/mediapipe-models/pose_landmarker/\
#: pose_landmarker_lite/float16/1/pose_landmarker_lite.task
MODEL_ENV = "BALL_REEL_POSE_MODEL"
DEFAULT_MODEL = "~/.mediapipe/pose_landmarker_lite.task"

#: Откуда качается, если её нет. Отдельной константой, потому что
#: адрес попадает в ТЕКСТ ОТКАЗА, а склеенный там по кусочкам он
#: разъезжается с настоящим при первой же правке.
MODEL_ONLINE = ("https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/1/"
                "pose_landmarker_lite.task")

#: Разделитель пути ДЛЯ ОБОЛОЧКИ, а не для Python.
SEP = "\\" if cure.WINDOWS else "/"

_POSE = None


def _model_path() -> Path:
    import os

    p = Path(os.environ.get(MODEL_ENV, DEFAULT_MODEL)).expanduser()
    if not p.exists():
        raise RuntimeError(
            f"pose model not found at {p}. Download it once:\n"
            f"  {cure.mkdir(cure.home('.mediapipe'))}\n"
            f"  {cure.download(MODEL_ONLINE, cure.home('.mediapipe') + SEP + p.name)}\n"
            f"or point {MODEL_ENV} at it. Not bundled: it is 5.5 MB of weights, "
            f"and a missing model must fail loudly rather than skip the check.")
    return p


def _pose_model():
    """Lazy singleton landmarker in IMAGE mode.

    IMAGE rather than VIDEO mode on purpose: video mode tracks between frames
    and would smooth over exactly the frame-to-frame instability
    `limb_consistency` exists to detect. Each frame is judged on its own.
    """
    global _POSE
    if _POSE is None:
        from mediapipe.tasks.python import BaseOptions  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        _POSE = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_model_path())),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5))
    return _POSE


def landmarks(path: str | Path) -> dict | None:
    """Body landmarks for one image, or None if no body is found.

    Returns ``{name: (x, y, visibility)}`` in MediaPipe's normalised image
    coordinates (0..1), which is why comparisons must renormalise — see
    `_normalise`.
    """
    import mediapipe as mp  # type: ignore
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    result = _pose_model().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    return {name: (float(lm[i].x), float(lm[i].y), float(lm[i].visibility))
            for name, i in BODY_POINTS.items()}


def world_landmarks(path: str | Path) -> dict | None:
    """Body landmarks in METRIC 3D space, rooted at the hips — view-independent.

    The image landmarks `landmarks()` returns are a PROJECTION, and projection
    destroys proportions: turn a person 45 degrees and their shoulders
    foreshorten toward nothing while their hips, being rounder, hold up. Measured
    on two real photos, shoulder width in torso units came out 0.143 on a turned
    subject against 1.019 on a frontal one — a sevenfold disagreement that says
    nothing about either body.

    The same detector also estimates 3D world coordinates, and those are stable:
    the same two photos, of two DIFFERENT people, gave 0.638 and 0.643 — within
    one percent. That is what makes build measurable from the photos users
    actually send, which are rarely frontal.

    Returns ``{name: (x, y, z, visibility)}`` in metres relative to the hip
    midpoint. Use this for PROPORTIONS. Keep `landmarks()` for comparing a
    generated frame against a reference framing, where the projection is the
    thing being compared.
    """
    import mediapipe as mp  # type: ignore
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    result = _pose_model().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return None
    lm = result.pose_world_landmarks[0]
    vis = result.pose_landmarks[0]
    return {name: (float(lm[i].x), float(lm[i].y), float(lm[i].z),
                   float(vis[i].visibility))
            for name, i in BODY_POINTS.items()}


def world_proportions(path: str | Path) -> dict | None:
    """Build as view-independent ratios, in torso lengths.

    Every length divided by the 3D torso length, so the numbers are comparable
    between people, between photos, and across camera angles — which the 2D
    equivalent is not.
    """
    import numpy as np

    pts = world_landmarks(path)
    if pts is None:
        return None

    def v(name):
        return np.array(pts[name][:3])

    def seen(*names):
        return all(pts[n][3] >= MIN_VISIBILITY for n in names)

    if not seen("l_hip", "r_hip", "l_shoulder", "r_shoulder"):
        return None
    hip_c = (v("l_hip") + v("r_hip")) / 2
    sho_c = (v("l_shoulder") + v("r_shoulder")) / 2
    torso = float(np.linalg.norm(sho_c - hip_c))
    if torso < 1e-6:
        return None

    out: dict = {"torso_metres": round(torso, 4)}
    for a, b in LIMBS:
        if seen(a, b):
            out[f"{a}->{b}"] = round(float(np.linalg.norm(v(a) - v(b))) / torso, 4)
    if seen("l_shoulder", "r_shoulder"):
        out["shoulder_width"] = round(
            float(np.linalg.norm(v("l_shoulder") - v("r_shoulder"))) / torso, 4)
    if seen("l_hip", "r_hip"):
        out["hip_width"] = round(
            float(np.linalg.norm(v("l_hip") - v("r_hip"))) / torso, 4)
    if out.get("shoulder_width") and out.get("hip_width"):
        out["shoulder_to_hip"] = round(
            out["shoulder_width"] / out["hip_width"], 4)
    legs = [out.get("l_hip->l_knee"), out.get("l_knee->l_ankle")]
    if all(legs):
        out["leg_length"] = round(sum(legs), 4)
    return out


def _normalise(points: dict) -> dict | None:
    """Centre on the hips and scale by torso length.

    Without this, moving the camera closer or the subject sideways would read as
    a changed pose — which is precisely the confusion that made a "pose" check
    worth building rather than eyeballing. What survives normalisation is the
    body's configuration, which is what "same pose" means.
    """
    import numpy as np

    need = ("l_hip", "r_hip", "l_shoulder", "r_shoulder")
    if any(points.get(n) is None or points[n][2] < MIN_VISIBILITY for n in need):
        return None
    hip = np.array([(points["l_hip"][0] + points["r_hip"][0]) / 2,
                    (points["l_hip"][1] + points["r_hip"][1]) / 2])
    sho = np.array([(points["l_shoulder"][0] + points["r_shoulder"][0]) / 2,
                    (points["l_shoulder"][1] + points["r_shoulder"][1]) / 2])
    torso = float(np.linalg.norm(sho - hip))
    if torso < 1e-6:
        return None
    return {n: ((np.array([x, y]) - hip) / torso, v)
            for n, (x, y, v) in points.items()}


def pose_delta(a: dict, b: dict) -> dict | None:
    """Per-joint displacement between two poses, in torso lengths.

    Returns the ``mean`` and the ``worst`` single joint, and both are needed.
    The mean is an average over twelve joints, so a change confined to one limb
    is diluted by the joints that did not move: a clearly bent arm scores 0.11
    on the mean while the wrist itself has moved almost a whole torso length.
    Since "the arms are somewhere else" is exactly the kind of mismatch worth
    catching, the worst joint is reported alongside rather than averaged away.

    Only joints clearly visible in BOTH poses are compared — an occluded joint
    carries no information, and scoring it would invent disagreement.
    """
    import numpy as np

    # `landmarks()` по контракту возвращает None, когда тела в кадре нет, и
    # ровно этот None сюда и приезжает — из сниппетов рунбука, из ноутбуков,
    # отовсюду, где вызов не обёрнут проверкой. Раньше он превращался в
    # AttributeError из глубины _normalise ('NoneType' has no attribute 'get'),
    # то есть в сообщение, по которому не видно ни причины, ни что чинить.
    if a is None or b is None:
        which = " и ".join(n for n, v in (("первой", a), ("второй", b))
                           if v is None)
        raise ValueError(
            f"pose_delta: в {which} позе тела не нашли (landmarks вернул None). "
            f"Сравнивать нечего. Частая причина — сравнение с УСЛОВИЕМ "
            f"ControlNet: это рисунок скелета на чёрном фоне, детектор поз на "
            f"нём ничего не находит. Сверять надо с driving-кадром "
            f"(см. manifest.json рядом с условиями).")
    na, nb = _normalise(a), _normalise(b)
    if na is None or nb is None:
        return None
    shared = [n for n in na
              if na[n][1] >= MIN_VISIBILITY and nb[n][1] >= MIN_VISIBILITY]
    if len(shared) < 4:
        return None
    per = {n: float(np.linalg.norm(na[n][0] - nb[n][0])) for n in shared}
    worst = max(per, key=lambda n: per[n])
    # СКОЛЬКО СУСТАВОВ УЧАСТВОВАЛО — ЧАСТЬ ОТВЕТА, А НЕ ПОДРОБНОСТЬ.
    #
    # Замерено на живых кадрах: полноростовой кадр дал mean 0.239, поясной —
    # 0.081, и второе выглядело как «поза стала втрое точнее». На деле у
    # поясного кадра просто нет в кадре ног: среднее считалось по вчетверо
    # меньшему набору суставов. Два числа сравнили как однородные, а они
    # однородными не были.
    #
    # Это тот же класс, что `coverage` у ArcFace, где он давно есть: средняя
    # величина без размера выборки — не измерение, а впечатление. Пол в 4
    # сустава оставлен как был (двигать порог без калибровки нельзя), но
    # молчать о нём больше нельзя.
    return {"mean": round(float(np.mean(list(per.values()))), 4),
            "worst": round(per[worst], 4), "worst_joint": worst,
            "compared": len(per), "measurable": len(na),
            "coverage": round(len(per) / len(na), 3) if na else 0.0,
            "joints": {n: round(v, 4) for n, v in per.items()}}


def pose_distance(a: dict, b: dict) -> float | None:
    """Mean joint displacement between two poses, in torso lengths."""
    d = pose_delta(a, b)
    return None if d is None else d["mean"]


def pose_drift(frame_paths, reference_path, *,
               max_pose_distance: float = SAME_POSE_MAX,
               max_worst_joint: float = WORST_JOINT_MAX) -> dict:
    """How far the frames drift from the reference's pose.

    Same shape of answer as the identity check, and the same discipline: judged
    on the median of the frames it could actually measure, with ``coverage``
    reporting how much of the clip that was. A clip in which no body is
    detectable returns "not verifiable" rather than a pass or a failure.

    Note a clip of a MOVING person is expected to leave the reference pose — the
    reference sets the starting configuration, not every frame. Use it on the
    start frame with ``SAME_POSE_MAX`` for "did the still keep the pose", and
    across video frames with ``POSE_WANDER_MAX`` for "did the clip wander off
    into an unrelated posture". Passing the strict bar over video frames would
    fail honest motion; passing the loose one over a still would wave through a
    pose the generator rewrote.
    """
    import numpy as np

    ref = landmarks(reference_path)
    empty = {"per_frame": {}, "median": None, "worst": (None, None),
             "worst_joint": None, "coverage": 0.0, "measured": 0,
             "frames": 0, "held": False}
    if ref is None or _normalise(ref) is None:
        return {**empty, "note": "no usable body in the pose reference "
                                 "(hips and shoulders must be visible)."}
    per_frame: dict[str, float] = {}
    per_joint: dict[str, float] = {}
    total = 0
    for p in frame_paths:
        total += 1
        got = landmarks(p)
        if got is None:
            continue
        d = pose_delta(ref, got)
        if d is not None:
            per_frame[Path(p).name] = d["mean"]
            per_joint[Path(p).name] = d["worst"]
    if not per_frame:
        return {**empty, "frames": total,
                "note": f"pose NOT VERIFIABLE: no body measurable in "
                        f"{total} frame(s)."}
    vals = sorted(per_frame.values())
    median = round(float(np.median(vals)), 4)
    worst = max(per_frame, key=lambda n: per_frame[n])
    coverage = round(len(per_frame) / total, 3)
    worst_joint = round(float(np.median(sorted(per_joint.values()))), 4)
    # Both must hold: the mean catches a wholesale reinterpretation, the worst
    # joint catches one limb put somewhere else, which the mean would dilute.
    held = median <= max_pose_distance and worst_joint <= max_worst_joint
    return {"per_frame": per_frame, "median": median, "worst_joint": worst_joint,
            "worst": (worst, per_frame[worst]), "coverage": coverage,
            "measured": len(per_frame), "frames": total, "held": held,
            "note": (f"pose distance median {median} torso-lengths from the "
                     f"reference, worst single joint {worst_joint}, over "
                     f"{len(per_frame)}/{total} measurable frame(s); "
                     f"{'held' if held else 'DRIFTED'} against "
                     f"{max_pose_distance}/{max_worst_joint}.")}


def limb_consistency(frame_paths, *,
                     max_wobble: float = LIMB_WOBBLE_MAX) -> dict:
    """Do the limbs keep their length across the clip — is this a real body.

    A limb's projected length changes as it swings toward or away from the
    camera, so some variation is physical. What is not physical is a limb whose
    length swings wildly, which is what a generator does when it loses the body:
    arms stretch, legs grow a joint, the torso rubber-bands.

    Reported per limb as coefficient of variation (sd/mean) in torso units.
    """
    import numpy as np

    lengths: dict[str, list[float]] = {f"{a}->{b}": [] for a, b in LIMBS}
    measured = 0
    for p in frame_paths:
        got = landmarks(p)
        if got is None:
            continue
        norm = _normalise(got)
        if norm is None:
            continue
        measured += 1
        for a, b in LIMBS:
            if norm[a][1] < MIN_VISIBILITY or norm[b][1] < MIN_VISIBILITY:
                continue
            lengths[f"{a}->{b}"].append(
                float(np.linalg.norm(norm[a][0] - norm[b][0])))
    wobble = {}
    for name, vals in lengths.items():
        if len(vals) < 3:
            continue
        mean = float(np.mean(vals))
        if mean > 1e-6:
            wobble[name] = round(float(np.std(vals)) / mean, 4)
    if not wobble:
        return {"wobble": {}, "worst": (None, None), "anatomical": False,
                "measured": measured,
                "note": "limb consistency NOT VERIFIABLE: no limb tracked "
                        "across enough frames."}
    worst = max(wobble, key=lambda n: wobble[n])
    bad = [n for n, v in wobble.items() if v > max_wobble]
    verdict = ("anatomically stable" if not bad else
               f"{len(bad)} limb(s) past {max_wobble:.0%} — the body is stretching")
    return {"wobble": wobble, "worst": (worst, wobble[worst]),
            "anatomical": not bad, "measured": measured, "unstable": bad,
            "note": (f"limb length varies most on {worst} "
                     f"({wobble[worst]:.0%} of its own length) over {measured} "
                     f"frame(s); {verdict}.")}
