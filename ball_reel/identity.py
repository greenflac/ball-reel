"""Face-consistency measurement across a video's frames — arithmetic, no model.

The role's thinking task: a character must stay the SAME person across the
frames of a clip, and the useful half of the answer is not "how do you keep it
consistent" but "how do you MEASURE, automatically, that it drifted". This file
is that measurement, and it is deliberately dependency-light so a sceptic can
recompute it from the committed frames and Pillow alone.

HONEST INSTRUMENT NOTE. The real instrument for face identity is a
face-recognition embedding (ArcFace / InsightFace): crop the face, embed it,
take cosine distance to a reference embedding. That is what `identity_drift`
becomes at live time — `arcface_drift` below is the seam, and it raises until a
real encoder is wired in. What ships and runs offline here is a PERCEPTUAL
PROXY: a difference-hash agreement between each frame and the reference face.
It is blind to identity in the way an embedding is not — it tracks coarse
composition, so a frame that merely re-poses the same person can read as drift,
and a different person in the same pose can read as stable. It is a stand-in
that exercises the pipeline and the gate, not a claim about faces. The proxy is
labelled everywhere it is printed, exactly as the eval repo labels its
`style_match` Pillow proxy against a real CLIP distance.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

#: Side of the greyscale square each frame is reduced to before hashing. Same
#: construction as the eval repo's perceptual hash: 2 * side * side bits.
_HASH_SIDE = 16

#: Proxy-agreement at/above which two faces are called "the same look". Chosen,
#: not measured, and it is a REPORT threshold — a wrong call costs a sentence,
#: not a rank. It sits where the eval repo's duplicate bar sits, for the same
#: reason: far above real variation, far below identity.
_SAME_LOOK_AT = 0.90


def _dhash(path: str | Path, side: int = _HASH_SIDE) -> tuple[bool, ...] | None:
    """Difference hash of a frame, or None if it cannot be read.

    Both gradient directions are compared, not just the horizontal one: a face
    drifts down the frame as much as across it, and a hash blind to one axis has
    a free way to be fooled.
    """
    try:
        with Image.open(path) as image:
            reduced = image.convert("L").resize((side + 1, side + 1), Image.BILINEAR)
    except (OSError, ValueError):
        return None
    cells = reduced.load()
    bits: list[bool] = []
    for y in range(side):
        for x in range(side):
            bits.append(cells[x, y] > cells[x + 1, y])
    for y in range(side):
        for x in range(side):
            bits.append(cells[x, y] > cells[x, y + 1])
    return tuple(bits)


def _agreement(a: tuple[bool, ...] | None, b: tuple[bool, ...] | None) -> float:
    """Fraction of hash bits two frames agree on: 1.0 identical, ~0.5 unrelated."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def arcface_drift(frame_paths, reference_path, **kwargs):
    """The real measure, at live time: 1 - cosine(embed(frame), embed(reference)).

    Delegates to ``identity_arcface`` (guarded import), which needs insightface +
    onnxruntime + numpy and downloads the buffalo_l model on first use. Same
    return shape as ``identity_drift``, so it is a drop-in for the live path. If
    those deps are absent it raises a clear install message rather than silently
    falling back to the proxy — a proxy dressed as the instrument is the failure
    this seam exists to prevent.

    ``kwargs`` pass straight through (e.g. ``min_face_px``), so callers can pick
    the right measurability floor for what they are judging — a sharp still and
    a motion-blurred video frame are not the same measurement.
    """
    from . import identity_arcface

    return identity_arcface.arcface_drift(frame_paths, reference_path, **kwargs)


def identity_drift(frame_paths, reference_path) -> dict:
    """Per-frame drift of a face away from a reference. PERCEPTUAL PROXY.

    drift = 1 - (dhash agreement with the reference face). 0.0 == same coarse
    look as the reference; ~0.5 == unrelated. Returns a mapping a caller can act
    on and a reader can argue with:

    * ``per_frame`` -- {frame_name: drift}, rounded;
    * ``worst`` -- (frame_name, drift) of the most-drifted readable frame;
    * ``drifted`` -- names whose agreement fell below ``_SAME_LOOK_AT``, i.e. the
      frames a human should be sent to look at / the pipeline should regenerate;
    * ``note`` -- one line, never empty, and it names the proxy so the number is
      never read as an embedding distance;
    * ``readable`` -- how many frames could be opened.
    """
    ref = _dhash(reference_path)
    per_frame: dict[str, float] = {}
    drifted: list[str] = []
    readable = 0
    for path in frame_paths:
        h = _dhash(path)
        if h is None:
            continue
        readable += 1
        agreement = _agreement(ref, h) if ref is not None else 0.0
        drift = round(max(0.0, 1.0 - agreement), 4)
        name = Path(path).name
        per_frame[name] = drift
        if agreement < _SAME_LOOK_AT:
            drifted.append(name)
    if not per_frame:
        return {
            "per_frame": {},
            "worst": (None, None),
            "drifted": [],
            "readable": 0,
            "note": "no readable frames: nothing to measure identity on.",
        }
    worst_name = max(per_frame, key=lambda n: per_frame[n])
    note = (
        f"identity via PERCEPTUAL PROXY (dhash vs reference face), not an "
        f"embedding: {len(drifted)}/{readable} frame(s) drifted past "
        f"{1 - _SAME_LOOK_AT:.0%}. Swap arcface_drift() in at live time for a "
        f"real face-recognition distance."
    )
    return {
        "per_frame": per_frame,
        "worst": (worst_name, per_frame[worst_name]),
        "drifted": drifted,
        "readable": readable,
        "note": note,
    }


def motion_presence(frame_paths) -> dict:
    """Is there MOVEMENT across the clip's frames? Presence, not plausibility.

    Mean absolute luma difference between consecutive frames, normalised to
    0..1. A static "video" (six copies of the start frame) scores ~0.0 and is
    flagged: a jump on a ball that does not move is a generation failure, and
    this is the cheap arithmetic that catches it before a judge is paid.

    HONEST BOUND. This says the pixels changed, NOT that a person plausibly
    jumped on a ball. Whether the motion is coherent human/ball physics is a
    judgement for the multimodal critic (or a human), and this number must never
    be read as that. It is the floor that rejects a frozen clip, nothing more.
    """
    frames = []
    for path in frame_paths:
        try:
            with Image.open(path) as image:
                frames.append(image.convert("L").resize((64, 114), Image.BILINEAR).load())
        except (OSError, ValueError):
            continue
    if len(frames) < 2:
        return {"motion": 0.0, "moving": False,
                "note": "fewer than two readable frames: no motion to measure."}
    diffs = []
    for a, b in zip(frames, frames[1:]):
        total = sum(abs(a[x, y] - b[x, y]) for y in range(114) for x in range(64))
        diffs.append(total / (64 * 114 * 255))
    motion = round(sum(diffs) / len(diffs), 4)
    moving = motion >= 0.02
    note = (
        f"motion PRESENCE {motion:.3f} (mean inter-frame luma change): "
        + ("frames move." if moving else "frames are near-static — a jump that "
           "does not move is a generation failure.")
        + " Presence only; plausibility of ball physics is the critic's call."
    )
    return {"motion": motion, "moving": moving, "note": note}
