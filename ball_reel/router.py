"""Which generation stack this job needs, decided from what was measured.

The one new fork in the architecture (see GPU_BRANCH.md). Everything upstream —
extraction, normalisation, validation — is stack-independent, and everything
downstream — the gate — is shared. So the choice of backend is not a standing
preference but a per-job consequence of two questions: what does this job
require, and what did the inputs actually give us.

Deciding it in code rather than by habit matters, because the expensive answer
is the tempting one. A job that only needs a start frame runs on the public API
for a fraction of a pollen; sending it to a GPU because "GPU is better" is pure
waste. The reverse error is worse and quieter: asking the public API for a
trajectory it cannot honour returns a confident clip of the wrong motion.

Grounded in measurement, not opinion — every rule below traces to a number in
POLLINATIONS_CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: What a job can ask for. Only `pose_trajectory` forces the expensive branch;
#: the rest are already covered by the public gateway.
CAPABILITIES = ("identity", "build", "pose_trajectory", "loop", "audio",
                "object_fidelity")

#: Measured: role-labelled multi-reference transfers face and build but NOT
#: pose (pose distance 0.59-0.65 against a bar of 0.25), and the gateway's whole
#: video capability vocabulary is start_frame/end_frame/audio_output. So a
#: continuous trajectory is not available there at any price.
API_CANNOT = ("pose_trajectory", "object_fidelity")


@dataclass
class Decision:
    stack: str
    reasons: list = field(default_factory=list)
    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    estimated_pollen: float | None = None

    @property
    def runnable(self) -> bool:
        return not self.blockers

    def render(self) -> str:
        lines = [f"stack: {self.stack}"
                 + ("" if self.runnable else "  [BLOCKED]")]
        for tag, items in (("blocker", self.blockers), ("reason", self.reasons),
                           ("warning", self.warnings)):
            lines += [f"  {tag}: {t}" for t in items]
        if self.estimated_pollen is not None:
            lines.append(f"  est. cost: {self.estimated_pollen:.2f} pollen")
        return "\n".join(lines)


def choose(required: tuple = ("identity",), *, subject: dict | None = None,
           spec=None, seconds: int = 4, video_model: str = "wan-fast",
           loop: bool = True) -> Decision:
    """Pick a stack, and say what would stop the job before anything is spent.

    `subject` is a `metrics.subject_package`, `spec` a `driving.DrivingSpec`.
    Both optional: the decision degrades to "what does the job require" when the
    inputs have not been measured yet, and sharpens when they have.

    The default `video_model` declares `end_frame`, because looping is on by
    default and a model without it silently drops the second keyframe. The two
    defaults have to agree, or the out-of-the-box call blocks itself.
    """
    unknown = [c for c in required if c not in CAPABILITIES]
    if unknown:
        raise ValueError(f"unknown capability {unknown}; known: {CAPABILITIES}")

    needs_gpu = [c for c in required if c in API_CANNOT]
    d = Decision(stack="gpu" if needs_gpu else "pollinations")
    for c in needs_gpu:
        d.reasons.append(
            f"'{c}' is not available on the public gateway at any price — "
            f"measured, not assumed (see POLLINATIONS_CONTRACT.md).")
    if not needs_gpu:
        d.reasons.append("everything required is covered by the public gateway; "
                         "a GPU would cost more and buy nothing.")
        d.estimated_pollen = _pollen(video_model, seconds)

    if loop and d.stack == "pollinations":
        from .chain import END_FRAME_MODELS

        if video_model not in END_FRAME_MODELS:
            d.blockers.append(
                f"a seamless loop needs an end keyframe, and '{video_model}' "
                f"ignores it. Use one of {END_FRAME_MODELS}.")

    _check_subject(d, subject, required)
    _check_spec(d, spec, required)
    return d


def _check_subject(d: Decision, subject: dict | None, required: tuple) -> None:
    """What the photo does or does not license this job to claim."""
    if subject is None:
        d.warnings.append("subject not measured: run metrics.subject_package "
                          "before spending, or the inputs are unverified.")
        return
    face = subject.get("face") or {}
    ident = face.get("identity") or {}
    px = ident.get("face_px")
    if not ident:
        d.blockers.append("no face found in the photo: identity cannot be "
                          "carried or later verified.")
    elif px and px < 100:
        d.warnings.append(
            f"source face is {px}px. It will generate, but the OUTPUT gate "
            f"needs ~100px to judge identity, and a small source tends to "
            f"produce a small face — expect 'not verifiable' rather than a fail.")
    body = subject.get("body") or {}
    if "build" in required:
        if not body.get("proportions"):
            d.blockers.append("build required but no body in the photo: supply "
                              "a body reference.")
        elif body.get("turned"):
            d.warnings.append(
                "subject is turned: build comes from 3D landmarks and holds, "
                "but identity matching across a large viewpoint change is "
                "harder (measured drift 0.55 on a turned source).")


def _check_spec(d: Decision, spec, required: tuple) -> None:
    """What the driving video does or does not specify."""
    if spec is None:
        if "pose_trajectory" in required:
            d.blockers.append("pose_trajectory required but no driving spec "
                              "supplied: there is no trajectory to follow.")
        return
    problems = spec.validate()
    for p in problems:
        # A missing expression is only a blocker if the job claimed to copy it.
        if "expression is NOT specified" in p:
            d.warnings.append(p)
        elif "motion track has holes" in p or "too short" in p:
            d.blockers.append(p)
        else:
            d.warnings.append(p)
    if "pose_trajectory" in required and spec.pose_coverage < 0.9:
        d.blockers.append(
            f"pose_trajectory required but the body was found in only "
            f"{spec.pose_coverage:.0%} of frames: the trajectory has gaps that "
            f"conditioning would have to invent.")


def _pollen(model: str, seconds: int) -> float | None:
    """Cost of the video call, from the live price list."""
    rates = {"wan-fast": 0.01, "seedance-pro": 0.025, "veo": 0.08,
             "wan": 0.1, "wan-pro": 0.1, "grok-imagine-video-1.5": 0.14,
             "seedance-2.0": 0.18}
    rate = rates.get(model)
    return None if rate is None else round(rate * seconds, 3)
