"""Input validation: is this photo usable, and for WHAT — before anything is spent.

Users send what they have: turned away, badly lit, cropped at the shoulders,
half a face. The pipeline cannot demand studio portraits, but it also must not
pretend a photo carries information it does not. So intake answers one question
per capability — can THIS photo support identity? build? expression? — and says
what to send instead when it cannot.

Two rules the answers follow, both learned the hard way in this pipeline:

  * A measurement that cannot be trusted is not reported as a value. It is
    reported as absent, with the reason. "Too small to tell" and "different"
    are different answers and must never collapse into one.
  * A blocker names the FIX, not the fault. "No face found" is a dead end for
    whoever sent the photo; "send a photo where the face is at least a third of
    the frame height" is actionable.

Everything here is local and free. It runs before the first paid call, which is
the whole point: the cheapest rejection is the one that happens before spending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Face size (px, shorter bbox side) below which identity cannot be verified in
#: the OUTPUT either — see identity_arcface.MIN_FACE_PX for the calibration.
#: A source below this generates fine and then cannot be checked, which is the
#: worst outcome: an unverifiable clip that looks finished.
MIN_SOURCE_FACE_PX = 100

#: Below this the face is too small for the 478-point mesh even after cropping,
#: so expression and face geometry are unavailable.
MIN_MESH_FACE_PX = 60

#: Detector confidence below which the "face" may not be one.
MIN_DETECTOR_SCORE = 0.6


@dataclass
class Intake:
    photo: str
    supports: list = field(default_factory=list)
    #: Capabilities present only because a default was substituted, never
    #: because they were measured. Kept apart so a caller cannot mistake one
    #: for the other.
    assumed: list = field(default_factory=list)
    blocked: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    measurements: dict = field(default_factory=dict)

    def can(self, capability: str) -> bool:
        return capability in self.supports

    @property
    def usable(self) -> bool:
        return "identity" in self.supports

    def render(self) -> str:
        shown = [s + (" (assumed)" if s in self.assumed else "")
                 for s in self.supports]
        lines = [f"{Path(self.photo).name}: "
                 + (", ".join(shown) if shown else "UNUSABLE")]
        for cap, why in self.blocked.items():
            lines.append(f"  cannot {cap}: {why}")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


def inspect(photo: str | Path) -> Intake:
    """What this photo can and cannot support, with the fix for each gap."""
    from .metrics import subject_package

    photo = str(photo)
    got = Intake(photo=photo)
    pkg = subject_package(photo)
    face, body = pkg["face"], pkg["body"]
    ident = face.get("identity") or {}

    if not ident:
        got.blocked["identity"] = (
            "no face detected. Send a photo where the face is clearly visible, "
            "unobstructed and roughly facing the camera.")
        got.blocked["expression"] = "no face detected."
        _check_build(got, body)
        return got

    px = ident.get("face_px") or 0
    score = ident.get("detector_score") or 0.0
    got.measurements["face_px"] = px
    got.measurements["detector_score"] = score

    if score < MIN_DETECTOR_SCORE:
        got.warnings.append(
            f"detector confidence {score} is low: this may not be a clear face. "
            f"A sharper, better-lit photo would be safer.")

    if px >= MIN_SOURCE_FACE_PX:
        got.supports.append("identity")
    elif px >= MIN_MESH_FACE_PX:
        got.supports.append("identity")
        got.warnings.append(
            f"face is {px}px. Identity can be carried, but the OUTPUT gate needs "
            f"~{MIN_SOURCE_FACE_PX}px to verify it, and a small source tends to "
            f"produce a small face — expect 'not verifiable' rather than a fail. "
            f"A closer crop of the same photo usually fixes this.")
    else:
        got.blocked["identity"] = (
            f"face is only {px}px — too small to identify from. Send a closer "
            f"photo, or crop this one to head and shoulders.")

    if face.get("geometry"):
        got.supports.append("face_geometry")
    else:
        got.blocked["face_geometry"] = (
            "the 478-point mesh could not read this face (too small, turned, or "
            "blurred). A front-on, sharper photo would give it.")

    expr = face.get("expression") or {}
    if expr.get("all"):
        got.supports.append("expression")
        got.measurements["expression_count"] = expr.get("count")
    else:
        got.blocked["expression"] = "no face mesh, so no expression coefficients."

    _check_build(got, body)
    return got


def _check_build(got: Intake, body: dict) -> None:
    """Build, measured if the photo allows and assumed if it does not.

    Deliberately NOT a blocker. The face is the part that has to be real —
    that is the person the viewer recognises — while a body can be typical
    without anyone noticing, and turning users away for sending a portrait
    would reject most of the photos people actually have. So a missing build
    falls back to ASSUMED_PROPORTIONS and is flagged `assumed`; only a missing
    face stops the job.

    The honesty requirement survives intact: an assumed body is never reported
    as a measured one.
    """
    from .metrics import ASSUMED_PROPORTIONS

    prop = body.get("proportions") or {}
    unusable = (not prop) or (not body.get("reliable", True))
    if unusable:
        got.supports.append("build")
        got.assumed.append("build")
        got.measurements["build_source"] = "assumed"
        got.warnings.append(
            "build could not be read from this photo, so typical proportions "
            "are assumed. Identity is unaffected — the face is real — but the "
            "body is a stand-in. Send a photo showing head to hips, or a "
            "separate body reference, if the build matters.")
        got.measurements["shoulder_to_hip"] = ASSUMED_PROPORTIONS["shoulder_to_hip"]
        return
    got.supports.append("build")
    got.measurements["build_source"] = "measured"
    got.measurements["shoulder_to_hip"] = prop.get("shoulder_to_hip")
    if body.get("turned"):
        got.warnings.append(
            "subject is turned away from camera. Build is taken from 3D "
            "landmarks and holds up, but matching identity across a large "
            "viewpoint change is harder — a front-on photo gives a better clip.")


def report(photos: list) -> str:
    """Inspect several photos and say which to use for what.

    Written for the real case, where a user sends a handful of pictures and the
    best one differs per capability: the sharpest face is often not the one that
    shows the body.
    """
    results = [inspect(p) for p in photos]
    lines = [r.render() for r in results]
    best: dict = {}
    for cap in ("identity", "build", "expression"):
        havers = [r for r in results if r.can(cap)]
        if cap == "identity":
            havers.sort(key=lambda r: -(r.measurements.get("face_px") or 0))
        else:
            # A photo that MEASURED the capability beats one that only fell
            # back to a default — otherwise the assumed build would win purely
            # by being listed first, and a real body in the set would go unused.
            havers.sort(key=lambda r: cap in r.assumed)
        if havers:
            best[cap] = Path(havers[0].photo).name
    lines.append("")
    lines.append("use: " + (", ".join(f"{c} <- {p}" for c, p in best.items())
                            if best else "nothing usable in this set"))
    return "\n".join(lines)
