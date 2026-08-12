"""Pose track -> ControlNet condition images, retargeted to the target body.

This is what turns a measured trajectory into something a generator must obey.
ControlNet takes a drawn skeleton, not a description, so the motion stops being
a request the model may reinterpret: the joints are where they are.

Two jobs, and the second one is not optional.

RENDER. MediaPipe's 33 landmarks become the 18-point COCO skeleton OpenPose
ControlNet expects, drawn with its colour convention — the colours are the
channel encoding, not decoration, and a differently-coloured skeleton conditions
badly.

RETARGET. The driving skeleton carries the DRIVING person's proportions. Feed it
unchanged and you have asked for the donor's body wearing the client's face —
the exact failure the whole subject package exists to prevent. So each bone
keeps its direction from the driving frame and takes its LENGTH from the
target's measured 3D proportions, walking outward from the hips so the change
propagates down each limb instead of tearing it apart.

Runs on CPU, no GPU, no network. Which means the expensive half of the pipeline
can be prepared and checked before a single GPU-minute is rented.

A NOTE ON INDEPENDENCE. GPU_BRANCH.md says the extractor that CONDITIONS and the
one that VERIFIES must differ, or the gate ends up confirming the conditioner's
own errors. DWPose is the intended conditioning source. Rendering from MediaPipe
(the verifier) is supported because it works today and needs nothing installed,
but it weakens the gate, so `from_mediapipe=True` is explicit at the call site
and recorded in the output — never a silent default.
"""

from __future__ import annotations

from pathlib import Path

#: OpenPose COCO-18 joint order. ControlNet's openpose annotator emits exactly
#: this, and the model was trained on it.
COCO18 = ("nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder",
          "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee",
          "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear")

#: MediaPipe's 33-point indices for the joints COCO-18 needs. `neck` is absent
#: from MediaPipe and is synthesised as the shoulder midpoint.
MEDIAPIPE_INDEX = {
    "nose": 0, "l_eye": 2, "r_eye": 5, "l_ear": 7, "r_ear": 8,
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}

#: OpenPose limb connections, in its own drawing order.
LIMBS18 = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
           (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
           (0, 15), (15, 17))

#: The OpenPose palette, in the same order as LIMBS18. These exact colours are
#: how the model reads which limb is which.
LIMB_COLOURS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
                (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
                (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
                (255, 0, 170))

JOINT_COLOURS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
                 (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
                 (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                 (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
                 (255, 0, 170), (255, 0, 85))

#: Bones, as (parent, child), ordered outward from the hips. Retargeting walks
#: this order so that moving a shoulder carries its whole arm with it.
BONE_TREE = (("hip_c", "neck"), ("neck", "nose"),
             ("neck", "l_shoulder"), ("l_shoulder", "l_elbow"),
             ("l_elbow", "l_wrist"),
             ("neck", "r_shoulder"), ("r_shoulder", "r_elbow"),
             ("r_elbow", "r_wrist"),
             ("hip_c", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
             ("hip_c", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"))

#: Which measured proportion sets each bone's length. Bones with no entry keep
#: the driving length — the head and the hip half-widths, where the subject
#: package has nothing better to say.
BONE_TO_PROPORTION = {
    ("l_shoulder", "l_elbow"): "l_shoulder->l_elbow",
    ("l_elbow", "l_wrist"): "l_elbow->l_wrist",
    ("r_shoulder", "r_elbow"): "r_shoulder->r_elbow",
    ("r_elbow", "r_wrist"): "r_elbow->r_wrist",
    ("l_hip", "l_knee"): "l_hip->l_knee",
    ("l_knee", "l_ankle"): "l_knee->l_ankle",
    ("r_hip", "r_knee"): "r_hip->r_knee",
    ("r_knee", "r_ankle"): "r_knee->r_ankle",
}


def pose_points(path: str | Path) -> dict | None:
    """The COCO-18 joints for one image, in normalised image coordinates.

    Values are ``(x, y, visibility)`` in 0..1. Returns None when no body is
    found — an absent skeleton must not become a drawn guess.
    """
    import mediapipe as mp  # type: ignore
    import numpy as np
    from PIL import Image

    from .pose import _pose_model

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    res = _pose_model().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks[0]
    pts = {name: (float(lm[i].x), float(lm[i].y), float(lm[i].visibility))
           for name, i in MEDIAPIPE_INDEX.items()}
    ls, rs = pts["l_shoulder"], pts["r_shoulder"]
    pts["neck"] = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2,
                   min(ls[2], rs[2]))
    lh, rh = pts["l_hip"], pts["r_hip"]
    pts["hip_c"] = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, min(lh[2], rh[2]))
    # The SOURCE aspect ratio travels with the points. Normalised coordinates
    # are relative to their own frame, so drawing them straight onto a canvas of
    # a different shape rescales x and y by different amounts and silently
    # changes the body's proportions — caught live: a 1280x720 driving frame
    # drawn onto 512x768 stretched the figure vertically about 2.7x, which would
    # have taught the generator a body nobody has.
    pts["__size__"] = (float(rgb.shape[1]), float(rgb.shape[0]), 1.0)
    return pts


def retarget(points: dict, proportions: dict, *,
             min_visibility: float = 0.5) -> dict:
    """Rescale bone lengths to the target's proportions, keeping directions.

    `proportions` is `metrics.body_metrics()["proportions"]` — 3D ratios in
    torso lengths, which is why they had to be measured in 3D: a projected
    ratio would carry the reference photo's camera angle into every frame.

    Directions come from the driving pose (that is the motion), lengths from the
    target (that is the body). Walking outward from the hips means a rescaled
    upper arm carries the forearm and hand with it, instead of detaching them.
    """
    import numpy as np

    if not proportions:
        return dict(points)
    neck, hip_c = points.get("neck"), points.get("hip_c")
    if neck is None or hip_c is None:
        return dict(points)
    torso = float(np.hypot(neck[0] - hip_c[0], neck[1] - hip_c[1]))
    if torso < 1e-6:
        return dict(points)

    out = {k: tuple(v) for k, v in points.items()}
    for parent, child in BONE_TREE:
        key = BONE_TO_PROPORTION.get((parent, child))
        if key is None or key not in proportions:
            continue
        p, c = out.get(parent), out.get(child)
        if p is None or c is None or c[2] < min_visibility:
            continue
        vec = np.array([c[0] - p[0], c[1] - p[1]])
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue
        target = float(proportions[key]) * torso
        shift = vec / length * (target - length)
        # Move the child and everything hanging off it, so the limb stays whole.
        for name in _descendants(child):
            if name in out:
                x, y, v = out[name]
                out[name] = (x + shift[0], y + shift[1], v)
    return out


def _descendants(joint: str) -> list:
    """`joint` and everything below it in BONE_TREE."""
    found = [joint]
    changed = True
    while changed:
        changed = False
        for parent, child in BONE_TREE:
            if parent in found and child not in found:
                found.append(child)
                changed = True
    return found


def _window(points: dict, aspect: float, margin: float = 0.35) -> tuple:
    """A crop window around the subject, in normalised coords, at `aspect`.

    Cropping rather than letterboxing, because the driving footage is usually
    landscape and the reel is vertical: letterboxing would leave the figure
    tiny in a band of black, wasting most of the conditioning resolution on
    nothing. The window is grown to the target aspect around the person, so the
    skeleton fills the frame AND keeps its proportions.
    """
    vis = [(x, y) for k, (x, y, v) in points.items()
           if k != "__size__" and v >= 0.5]
    if not vis:
        return (0.0, 0.0, 1.0, 1.0)
    xs, ys = [p[0] for p in vis], [p[1] for p in vis]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0, 1e-3) * (1 + margin), max(y1 - y0, 1e-3) * (1 + margin)
    # Grow the short side until the window matches the target aspect.
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    return (cx - w / 2, cy - h / 2, w, h)


def draw(points: dict, out_path: str | Path, *, width: int = 512,
         height: int = 768, min_visibility: float = 0.5,
         line_width: int = 4) -> str:
    """Draw the OpenPose skeleton on black — the ControlNet condition image.

    Joints below `min_visibility` are omitted along with their limbs rather than
    drawn at a guessed position: a confidently drawn wrong limb conditions the
    generator into a wrong pose, which is worse than leaving it unconstrained.

    Proportions are preserved by mapping through a crop window of the SOURCE
    aspect (see `_window`), not by stretching normalised coordinates onto the
    canvas — that stretch is a real bug this function used to have.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (0, 0, 0))
    d = ImageDraw.Draw(img)

    src_w, src_h, _ = points.get("__size__", (float(width), float(height), 1.0))
    # Work in source PIXELS so x and y share one scale, then fit the window.
    win = _window({k: (v[0] * src_w, v[1] * src_h, v[2])
                   for k, v in points.items() if k != "__size__"},
                  aspect=width / height)
    wx, wy, ww, wh = win
    scale = width / ww if ww > 0 else 1.0

    def xy(name):
        p = points.get(name)
        if p is None or name == "__size__" or p[2] < min_visibility:
            return None
        return ((p[0] * src_w - wx) * scale, (p[1] * src_h - wy) * scale)

    for (a, b), colour in zip(LIMBS18, LIMB_COLOURS):
        pa, pb = xy(COCO18[a]), xy(COCO18[b])
        if pa and pb:
            d.line([pa, pb], fill=colour, width=line_width)
    for i, name in enumerate(COCO18):
        p = xy(name)
        if p:
            r = line_width
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                      fill=JOINT_COLOURS[i])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return str(out_path)


def render_sequence(frames: list, out_dir: str | Path, *,
                    proportions: dict | None = None, width: int = 512,
                    height: int = 768, from_mediapipe: bool = True) -> dict:
    """A whole driving segment -> a folder of condition images.

    Returns the manifest a GPU run consumes, including `coverage` — the share of
    frames that produced a skeleton. A gap in the middle of a sequence means the
    generator gets no constraint for those frames, so it is reported rather than
    silently skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made, missing = [], []
    drawn_joints = []
    for i, f in enumerate(frames):
        pts = pose_points(f)
        if pts is None:
            missing.append(i)
            continue
        if proportions:
            pts = retarget(pts, proportions)
        # Сколько суставов реально попало в условие. Кадр со скелетом ещё не
        # означает скелет ЦЕЛИКОМ: невидимый локоть выбрасывает вместе с собой
        # предплечье, и в этом кадре рука ничем не ограничена. Поймано глазами
        # на живом референсе, где `coverage` показывал 1.0, а у фигуры не было
        # одной руки. Кадр — единица слишком крупная, чтобы это заметить.
        drawn_joints.append(sum(1 for n in COCO18
                                if pts.get(n) and pts[n][2] >= 0.5))
        made.append(draw(pts, out_dir / f"{i:04d}.png",
                         width=width, height=height))
    coverage = round(len(made) / len(frames), 3) if frames else 0.0
    joint_cover = (round(sum(drawn_joints) / (len(drawn_joints) * len(COCO18)), 3)
                   if drawn_joints else 0.0)
    partial = sum(1 for n in drawn_joints if n < len(COCO18) - 2)
    manifest = {
        "conditions": made, "missing_frames": missing, "coverage": coverage,
        "joint_coverage": joint_cover, "partial_frames": partial,
        "size": [width, height], "retargeted": bool(proportions),
        "source": "mediapipe" if from_mediapipe else "dwpose",
        "warnings": [],
    }
    if partial:
        manifest["warnings"].append(
            f"{partial} of {len(made)} condition(s) are missing more than two "
            f"joints ({joint_cover:.0%} of joints drawn overall): those limbs "
            f"are unconstrained in those frames, and the generator will invent "
            f"them.")
    if from_mediapipe:
        manifest["warnings"].append(
            "conditions were rendered from MediaPipe, which is also the "
            "verifier: the gate can no longer catch this extractor's own "
            "errors. Switch to DWPose for conditioning before trusting the "
            "pose verdict.")
    if coverage < 1.0:
        manifest["warnings"].append(
            f"{len(missing)} frame(s) produced no skeleton: the generator is "
            f"unconstrained there.")
    if not proportions:
        manifest["warnings"].append(
            "not retargeted: these are the DRIVING person's proportions, so "
            "the output will carry their body, not the target's.")
    return manifest
